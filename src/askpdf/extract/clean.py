"""Text cleaning and header/footer detection (design §4, ADR-0002).

Cleaning is deliberately conservative: it only undoes artefacts of PDF layout
(compatibility characters, soft hyphens, words broken across lines). Anything that
changes the words themselves would make verbatim quote checking unreliable.

A hyphen at a line end is ambiguous: "algo-/rithm" is a word broken by layout, but
"multi-/factor" is a compound that happens to break at its hyphen. No rule on the line
alone can tell them apart, so the document decides (ADR-0002 Amendment 1): the hyphen
is removed only if the document spells the word solid more often than hyphenated.
"""

import itertools
import math
import re
import unicodedata
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field

SOFT_HYPHEN = "\u00ad"
# U+2010 HYPHEN; NFKC turns U+2011 NON-BREAKING HYPHEN into it.
HYPHENS = "-\u2010"

# Part of the index pins: changing what cleaning produces must rebuild old indexes.
CLEANING_VERSION = 2

# Header/footer detection (design §4): only the first and last few lines of a page are
# candidates, and a line must recur on at least half the pages of a long-enough document.
EDGE_LINES = 3
MIN_SHARE = 0.5
MIN_PAGES = 4

_WHITESPACE = re.compile("[ \\t\\f\\v\\u00a0\\u2000-\\u200a\\u202f\\u205f\\u3000]+")
# Lexicon tokens: runs of letters, joined by internal hyphens ("out-of-band").
_WORD = re.compile(r"[^\W\d_]+(?:[-\u2010][^\W\d_]+)*")
# The token that ends at a line-end hyphen, and the one that starts the next line.
_TAIL = re.compile(r"[\w\-\u2010]*\Z")
_HEAD = re.compile(r"\w+")
_DIGITS = re.compile(r"\d+")


def _drop_control(ch: str) -> str:
    if ch in "\n\t" or ch == SOFT_HYPHEN:  # soft hyphens are resolved below
        return ch
    category = unicodedata.category(ch)
    if category == "Cc":
        return " "
    if category == "Cf":  # zero-width and bidi marks
        return ""
    if category == "Cs":
        # Lone surrogates come from broken ToUnicode maps; SQLite cannot store them.
        return "\ufffd"
    return ch


def _normalise_lines(text: str) -> str:
    lines = (_WHITESPACE.sub(" ", line).strip() for line in text.split("\n"))
    return "\n".join(line for line in lines if line)


def clean_page(text: str) -> str:
    """Normalise one page's extracted text, keeping its line structure.

    Line-end hyphens are left alone; :func:`clean_document` resolves them, because
    that needs the whole document. Idempotent.
    """
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "".join(_drop_control(ch) for ch in text)
    # Whitespace and blank lines go first, so "wor- \n\n ld" is handled like "wor-\nld".
    text = _normalise_lines(text)
    # A soft hyphen marks a place where a word may break, so at a line end it joins the
    # word; anywhere else it is invisible and dropped.
    text = text.replace(SOFT_HYPHEN + "\n", "").replace(SOFT_HYPHEN, "")
    # Removing invisible characters can leave a base letter next to its combining mark;
    # compose them, or a second pass would give different text (and quotes would not
    # match). NFC only composes, so it cannot bring back anything removed above.
    return unicodedata.normalize("NFC", _normalise_lines(text))


def _is_letter_or_digit(ch: str) -> bool:
    category = unicodedata.category(ch)
    return category[0] == "L" or category == "Nd"


def _ends_with_break(line: str, following: str) -> bool:
    """A line-end hyphen event: letter or digit, hyphen, newline, letter or digit."""
    return (
        len(line) >= 2
        and following != ""
        and line[-1] in HYPHENS
        and _is_letter_or_digit(line[-2])
        and _is_letter_or_digit(following[0])
    )


def _fold(word: str) -> str:
    return word.casefold().replace("\u2010", "-")


@dataclass
class Lexicon:
    """How often a document spells each word, solid and hyphenated.

    Built from page-cleaned text. The two tokens around every line-end hyphen are left
    out, so a decision never counts as evidence for itself.
    """

    words: Counter[str] = field(default_factory=Counter)  # whole tokens
    pairs: Counter[str] = field(default_factory=Counter)  # "a-b", also inside "a-b-c"

    @classmethod
    def build(cls, pages: Sequence[str]) -> "Lexicon":
        lexicon = cls()
        for page in pages:
            lines = page.split("\n")
            for i, line in enumerate(lines):
                breaks_before = i > 0 and _ends_with_break(lines[i - 1], line)
                breaks_after = i + 1 < len(lines) and _ends_with_break(line, lines[i + 1])
                for match in _WORD.finditer(line):
                    if breaks_before and match.start() == 0:
                        continue
                    if breaks_after and match.end() == len(line) - 1:
                        continue
                    lexicon.add(match.group())
        return lexicon

    def add(self, token: str) -> None:
        token = _fold(token)
        self.words[token] += 1
        parts = token.split("-")
        self.pairs.update(f"{a}-{b}" for a, b in itertools.pairwise(parts))

    def prefers_solid(self, left: str, right: str) -> bool:
        """True if ``left+right`` occurs more often than ``left-right``; a tie keeps it."""
        left, right = _fold(left), _fold(right)
        return self.words[left + right] > self.pairs[f"{left}-{right}"]


def _join(line: str, following: str, lexicon: Lexicon) -> tuple[str, bool]:
    """Join a line ending in a hyphen to the next; say whether the hyphen was dropped."""
    # Both always match: _TAIL may match empty, and `following` starts with a letter or
    # digit. findall()[0] is the first match without an Optional to unwrap.
    left_word = _TAIL.findall(line[:-1])[0].lstrip(HYPHENS)
    right_word = _HEAD.findall(following)[0]
    drop = (
        line[-2].isalpha()
        and following[0].islower()
        and not any(h in left_word for h in HYPHENS)
        and lexicon.prefers_solid(left_word, right_word)
    )
    return (line[:-1] if drop else line) + following, drop


def resolve_hyphens(
    pages: Sequence[str],
    lexicon: Lexicon | None = None,
    protected: Sequence[set[int]] | None = None,
) -> list[str]:
    """Join every word broken at a line end, keeping the hyphen unless the document
    spells the word solid (ADR-0002 Amendment 1).

    ``pages`` must be page-cleaned. ``protected`` holds, per page, indices of lines that
    are never joined to a neighbour: headers and footers, so a page number below a
    hyphenated last body line is not glued onto it (Amendment 1a). A break across pages
    is left alone: chunks and citations never cross a page.
    """
    if lexicon is None:
        lexicon = Lexicon.build(pages)
    resolved = []
    for n, page in enumerate(pages):
        keep_apart = protected[n] if protected is not None else set()
        out: list[str] = []
        last_protected = False  # whether out[-1] holds a protected line
        for i, line in enumerate(page.split("\n")):
            joinable = out and not last_protected and i not in keep_apart
            if joinable and _ends_with_break(out[-1], line):
                out[-1], _ = _join(out[-1], line, lexicon)
            else:
                out.append(line)
                last_protected = i in keep_apart
        resolved.append(unicodedata.normalize("NFC", "\n".join(out)))
    return resolved


def clean_document(pages: Sequence[str]) -> list[str]:
    """Clean every page of one document. Idempotent."""
    cleaned = [clean_page(p) for p in pages]
    boilerplate = detect_boilerplate([p.split("\n") if p else [] for p in cleaned])
    return resolve_hyphens(cleaned, protected=boilerplate)


def clean_text(text: str) -> str:
    """Clean a standalone piece of text, such as a page label, as a one-page document."""
    return clean_document([text])[0]


def _signature(line: str) -> str:
    """Page numbers and dates vary from page to page; compare lines without them."""
    return _DIGITS.sub("#", line.casefold())


def _edge_indices(n_lines: int) -> range | list[int]:
    if n_lines <= 2 * EDGE_LINES:
        return range(n_lines)
    return [*range(EDGE_LINES), *range(n_lines - EDGE_LINES, n_lines)]


def detect_boilerplate(pages: Sequence[Sequence[str]]) -> list[set[int]]:
    """Return, for each page, the indices of lines that look like headers or footers.

    ``pages`` holds each page's cleaned lines. A line counts when its signature appears
    among the edge lines of at least ``MIN_SHARE`` of the document's pages, and the
    document has at least ``MIN_PAGES`` pages.
    """
    flagged: list[set[int]] = [set() for _ in pages]
    if len(pages) < MIN_PAGES:
        return flagged
    pages_with: Counter[str] = Counter()
    for lines in pages:
        pages_with.update({_signature(lines[i]) for i in _edge_indices(len(lines))})
    threshold = math.ceil(MIN_SHARE * len(pages))
    repeated = {sig for sig, count in pages_with.items() if count >= threshold}
    for page_flags, lines in zip(flagged, pages, strict=True):
        for i in _edge_indices(len(lines)):
            if _signature(lines[i]) in repeated:
                page_flags.add(i)
    return flagged
