"""Text cleaning and header/footer detection (design §4, ADR-0002).

Cleaning is deliberately conservative: it only undoes artefacts of PDF layout
(compatibility characters, soft hyphens, words broken across lines). Anything that
changes the words themselves would make verbatim quote checking unreliable.
"""

import math
import re
import unicodedata
from collections import Counter
from collections.abc import Sequence

SOFT_HYPHEN = "­"

# Header/footer detection (design §4): only the first and last few lines of a page are
# candidates, and a line must recur on at least half the pages of a long-enough document.
EDGE_LINES = 3
MIN_SHARE = 0.5
MIN_PAGES = 4

_WHITESPACE = re.compile("[ \\t\\f\\v\\u00a0\\u2000-\\u200a\\u202f\\u205f\\u3000]+")
# A letter, a hyphen at the end of a line, then the next line: the break is undone only
# when the continuation starts with a lowercase letter ("wor-\nld" -> "world"), so
# "Covid-\n19" and "Smith-\nJones" keep their hyphens.
_LINE_BREAK_HYPHEN = re.compile(r"(?<=[^\W\d_])-\n(?=[^\W\d_])")
_DIGITS = re.compile(r"\d+")


def _drop_control(ch: str) -> str:
    if ch in "\n\t":
        return ch
    category = unicodedata.category(ch)
    if category == "Cc":
        return " "
    if category == "Cf":  # zero-width and bidi marks
        return ""
    return ch


def clean_text(text: str) -> str:
    """Normalise one page's extracted text, keeping its line structure."""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace(SOFT_HYPHEN, "")
    text = "".join(_drop_control(ch) for ch in text)
    text = _LINE_BREAK_HYPHEN.sub(
        lambda m: "" if m.string[m.end()].islower() else m.group(0),
        text,
    )
    lines = (_WHITESPACE.sub(" ", line).strip() for line in text.split("\n"))
    return "\n".join(line for line in lines if line)


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
