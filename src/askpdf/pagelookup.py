"""Questions about a named page ("what is on page 5 of report.pdf?").

Search ranks pages by their words, but neither a file name nor a page number is a word on
the page, so such a question is answered from the page it names instead of being
searched. It is recognised only when it names a page number and, unless the folder holds
one document, a document. Anything else returns None and is searched as usual.

"Page 5" is ambiguous: the 5th page of the file, or the page printed "5". When those
differ both are given to the model, and the citation says which one it used. "PDF page 5"
means the 5th page of the file only.

A question about a line ("the second line of page 4") is answered without the model: the
line is read from the page's stored text, since a small model counts lines unreliably.
Lines are counted as the PDF's text layer gives them, blank lines skipped, so a header
counts and a layout in columns may not read in the order it looks.
"""

import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import PurePath

from askpdf.answer import ANSWER_K, Answer, Citation
from askpdf.quotes import QuoteCheck
from askpdf.retrieve import PageHit
from askpdf.snapshot import Chunk, Snapshot

# "page 5", "p. 5", "pg 12", "page iv", "PDF page 5"; the number must stand alone. Words
# such as "did" fit the Roman letters too; _page keeps only labels a document prints.
_PAGE = re.compile(
    r"(?<![\w.])(?P<pdf>pdf\s+)?(?:page|pg\.?|p\.)\s*(?P<n>\d{1,5}|[ivxlcdm]{1,8})(?![\w])",
    re.I,
)
_WORD = re.compile(r"[^\W_]+")
_LETTER = re.compile(r"[^\W\d_]")
SHOWN_NAMES = 5  # documents listed when asking which one was meant
_ORDINALS = (
    *("first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth"),
    *("ninth", "tenth", "eleventh", "twelfth", "thirteenth", "fourteenth", "fifteenth"),
    *("sixteenth", "seventeenth", "eighteenth", "nineteenth", "twentieth"),
)
_NTH = rf"(?:(?P<word>{'|'.join(_ORDINALS)}|last)|(?P<nth>\d{{1,4}})(?:st|nd|rd|th))"
# "the second pdf", "the 3rd document", "the last file", "the second one"
_NTH_DOCUMENT = re.compile(rf"\b{_NTH}\s+(?:pdf|document|doc|file|one)\b", re.I)
# "the second line", "the 3rd line", "line 3", "the last line"
_LINE = re.compile(
    rf"\b{_NTH}\s+line\b"
    r"|\bline\s+(?P<num>\d{1,4})\b",
    re.I,
)


@dataclass(frozen=True, slots=True)
class Lookup:
    hits: list[PageHit]  # empty if the page does not exist or the document is unclear
    notes: tuple[str, ...]
    which: bool = False  # True when asking which document was meant
    answer: Answer | None = None  # set when the page text answers it without the model
    # What to ask the model instead of the user's words, when those only point at the page.
    question: str | None = None


def _words(text: str) -> list[str]:
    return _WORD.findall(unicodedata.normalize("NFKC", text).casefold())


# "summarise X", "a summary of X", "an overview of X", "the gist", "tl;dr"
_SUMMARY = re.compile(r"\b(?:summari[sz]e|summary|overview|gist|tl;?dr)\b", re.I)
# "what is X about?": a summary request only when X names a document
_ABOUT = re.compile(r"\bwhat(?:'s|\s+is)\b.*\babout\s*\??\s*$", re.I)
_PART = re.compile(r"\b(?:chapters?|sections?|parts?|appendix|appendices)\b", re.I)


def _summary(question: str, snapshot: Snapshot) -> Lookup | None:
    """A plain reply to a request to summarise a whole document or chapter, which askpdf
    cannot do yet: it answers from a few searched pages, so a "summary" would cover those
    alone and look complete. A summary of a topic ("summarise what it says about
    passwords") is left to the search, which serves it well."""
    asks = _SUMMARY.search(question) or _ABOUT.search(question)
    if not asks or not snapshot.paths:
        return None
    sha, _ = _document(snapshot, [question])
    if sha is None and not _SUMMARY.search(question):
        return None  # "what is this about?" is a follow-up, not a summary request
    first = sha or min(snapshot.paths, key=lambda s: snapshot.paths[s])
    name = PurePath(snapshot.paths[first][0]).name
    example = f"\u201csummarise page 12 of {name}\u201d"
    if _PART.search(question):
        note = (
            "askpdf cannot find chapters or sections yet: it does not read a PDF's table of"
            f" contents. Name a page instead, such as {example}, or ask about the topic."
        )
        return Lookup([], (note,))
    paths = [p for ps in snapshot.paths.values() for p in ps]
    if _topic(question, paths):
        return None  # a topic to search for
    note = (
        f"askpdf cannot summarise a whole document yet. It answers from {ANSWER_K} pages"
        " at a time, so a summary would cover only those. Ask about a topic in it, or name"
        f" a page, such as {example}."
    )
    return Lookup([], (note,))


def _ordinal(match: re.Match[str]) -> int:
    """1-based from "second" or "2nd"; -1 for "last" (Python's index of the last item)."""
    if match.group("word"):
        word = match.group("word").casefold()
        return -1 if word == "last" else _ORDINALS.index(word) + 1
    return int(match.group("nth"))


def _matched_words(name: list[str], text: list[str]) -> int:
    """How many words of a file name, from its first word on, appear in a row in the
    text: "NIST SP 800-63B" matches 4 of the 5 words of nist-sp-800-63b-4.pdf."""
    best = 0
    for start in range(len(text)):
        n = 0
        while n < len(name) and start + n < len(text) and text[start + n] == name[n]:
            n += 1
        best = max(best, n)
    return best


def _document(snapshot: Snapshot, texts: Sequence[str]) -> tuple[str | None, list[str]]:
    """The one document the texts name, newest text first; or None, with the candidates
    when the best match is a tie."""
    names = {
        sha: [_words(PurePath(p).stem) for p in paths]
        for sha, paths in snapshot.paths.items()
        if paths
    }
    if len(names) == 1:  # one document in the folder: "page 5" can only mean it
        return next(iter(names)), []
    for text in texts:
        words = _words(text)
        scores: dict[str, int] = {}
        for sha, stems in names.items():
            for stem in stems:
                n = _matched_words(stem, words)
                # The whole name, or most of a longer one (at least 2 words).
                if n == len(stem) or (n >= 2 and 2 * n >= len(stem)):
                    scores[sha] = max(scores.get(sha, 0), n)
        if not scores:
            # One distinctive word of a name ("the naca report"), if only one name has it.
            loose = {
                sha
                for sha, stems in names.items()
                for stem in stems
                if any(len(w) >= 3 and _LETTER.search(w) and w in words for w in stem)
            }
            if len(loose) == 1:
                return loose.pop(), []
            if loose:
                return None, sorted(loose)
        if scores:
            top = max(scores.values())
            best = [sha for sha, n in scores.items() if n == top]
            if len(best) == 1:
                return best[0], []
            return None, best
        nth = _NTH_DOCUMENT.search(text)
        if nth is not None:  # "the second pdf": as the documents are listed, by path
            listed = sorted(names, key=lambda sha: snapshot.paths[sha])
            n = _ordinal(nth)
            if n == -1 or 1 <= n <= len(listed):
                return listed[n if n == -1 else n - 1], []
    return None, []


def _hit(snapshot: Snapshot, sha: str, page: int) -> PageHit:
    first = next(
        (c for c in snapshot.chunks if c.sha256 == sha and c.page == page),
        Chunk(0, sha, page, 0, snapshot.texts.get((sha, page), "")),
    )
    return PageHit(
        sha256=sha,
        page=page,
        label=snapshot.labels.get((sha, page), str(page)),
        paths=snapshot.paths.get(sha, ()),
        score=1.0,
        best_chunk=first,
    )


def _page(question: str, snapshot: Snapshot) -> re.Match[str] | None:
    """The first page number in the question. A Roman numeral counts only when some
    document prints it as a page label, so "on which page did NIST define MFA?" is
    searched rather than read as page "di"."""
    printed = {label.casefold() for label in snapshot.labels.values()}
    for match in _PAGE.finditer(question):
        wanted = match.group("n")
        if wanted.isdecimal() or wanted.casefold() in printed:
            return match
    return None


def find_page(question: str, snapshot: Snapshot, earlier: Sequence[str] = ()) -> Lookup | None:
    """The page a question names, or None if it names none. A follow-up such as "and
    page 6?" takes the document from the earlier questions. A request to summarise a
    whole document or chapter gets a plain reply, with no pages."""
    match = _page(question, snapshot)
    if match is None:
        return _summary(question, snapshot)
    sha, tied = _document(snapshot, [question, *reversed(earlier)])
    if sha is None:
        # A page number but no clear document: searching would not find the page either.
        names = sorted(snapshot.paths[s][0] for s in (tied or snapshot.paths))
        if not names:
            return None  # nothing indexed: let the search say so
        listed = ", ".join(names[:SHOWN_NAMES]) + (", …" if len(names) > SHOWN_NAMES else "")
        note = f"which document do you mean? Name it, e.g. {listed}"
        if not tied:  # positions count the whole list, so offer them only for it
            note += ", or say \u201cthe second pdf\u201d"
        return Lookup([], (note,), which=True)
    name = PurePath(snapshot.paths[sha][0]).name
    pages = sorted(page for (s, page) in snapshot.labels if s == sha)
    wanted = match.group("n").casefold()
    physical = int(wanted) if wanted.isdecimal() and int(wanted) in pages else None
    printed = (
        []
        if match.group("pdf")
        else [page for page in pages if snapshot.labels[(sha, page)].casefold() == wanted]
    )
    found = list(dict.fromkeys([*printed, *([physical] if physical else [])]))
    if not found:
        count = f"{len(pages)} page" + ("" if len(pages) == 1 else "s")
        return Lookup([], (f"{name} has no page {match.group('n')} ({count})",))
    if len(found) == 1:
        note = f"reading {name} page {found[0]} directly"
    else:
        where = " and ".join(f"page {p}" for p in found)
        note = f"page {match.group('n')} of {name} could mean {where}; reading both"
    hits = [_hit(snapshot, sha, p) for p in found]
    line = _LINE.search(question)
    if line is None:
        return Lookup(hits, (note,), question=_describe(question, hits, snapshot.paths[sha]))
    return _read_line(line, hits, snapshot, name)


# Words that only point at a page: "what is on page 5 of the third doc?"
_POINTING = frozenset(
    {
        "what",
        "whats",
        "is",
        "are",
        "was",
        "on",
        "in",
        "at",
        "of",
        "the",
        "a",
        "an",
        "does",
        "do",
        "say",
        "says",
        "said",
        "contain",
        "contains",
        "there",
        "show",
        "shows",
        "tell",
        "me",
        "can",
        "you",
        "please",
        "describe",
        "summarise",
        "summarize",
        "summary",
        "explain",
        "give",
        "overview",
        "this",
        "that",
        "it",
        "its",
        "and",
        "doc",
        "document",
        "file",
        "pdf",
        "page",
        "pages",
        "p",
        "pg",
        "here",
        "written",
        "about",
        "gist",
        "tldr",
        "whole",
        "entire",
        "short",
        "brief",
        "quick",
        "documents",
        "pdfs",
        "files",
        "report",
        "paper",
        "article",
        "book",
        "guide",
        "one",
    }
)


def _topic(question: str, paths: Sequence[str]) -> list[str]:
    """The words of a question that are about something, not just pointing at a page or
    a document: none in "what is on page 5 of the annual report?"."""
    names = {w for path in paths for w in _words(PurePath(path).stem)}
    return [
        w
        for w in _words(question)
        if len(w) > 2
        and w not in _POINTING
        and w not in names
        and w not in _ORDINALS
        and w != "last"
        and not w.isdecimal()
        and not re.fullmatch(r"[ivxlcdm]+|\d+(?:st|nd|rd|th)", w)
    ]


def _describe(question: str, hits: Sequence[PageHit], paths: Sequence[str]) -> str | None:
    """A neutral question for a page question with no topic of its own. Asked "what is on
    page 55", llama3.2:3b replied NOT_FOUND: no sentence on the page answers it. Told to
    describe the pages, it does, in the citation format. A question with a topic ("what
    does page 5 say about cookies?") keeps its own words."""
    if _topic(question, paths):
        return None
    marks = " or ".join(f"[{n}]" for n in range(1, len(hits) + 1))
    which = "these pages say" if len(hits) > 1 else "this page says"
    return (
        f"What {which}? Describe the main points in a few sentences, each ending with"
        f" the number of its page, {marks}. Then write the Quotes section, as rule 2 says."
    )


def _line_number(match: re.Match[str]) -> int:
    """1-based, or -1 for the last line."""
    return int(match.group("num")) if match.group("num") else _ordinal(match)


def _read_line(match: re.Match[str], hits: list[PageHit], snapshot: Snapshot, name: str) -> Lookup:
    wanted = _line_number(match)
    sentences: list[str] = []
    citations: list[Citation] = []
    for hit in hits:
        text = snapshot.texts.get((hit.sha256, hit.page), "")
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        where = (
            f"page {hit.page}"
            if hit.label == str(hit.page)
            else (f"the page printed {hit.label} (PDF page {hit.page})")
        )
        if not 1 <= abs(wanted) <= len(lines) or wanted == 0:
            sentences.append(f"{where[0].upper()}{where[1:]} has {len(lines)} lines of text.")
            continue
        line = lines[wanted - 1] if wanted > 0 else lines[-1]
        n = len(citations) + 1
        which = "The last line" if wanted < 0 else f"Line {wanted}"
        sentences.append(f"{which} of {where} reads: \u201c{line}\u201d [{n}].")
        # Read from the index, not written by a model: it is the page's own text.
        citations.append(Citation(n, hit, (QuoteCheck(line, verified=True),)))
    text = " ".join(sentences)
    answer = Answer(
        text=text,
        not_found=False,
        citations=tuple(citations),
        dropped=(),
        sentences=len(sentences),
        supported=len(citations),
        raw="",
    )
    return Lookup(hits, (f"read from {name} without the model",), answer=answer)
