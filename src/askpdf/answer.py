"""Answer a question from retrieved pages, with checked citations (design §9, ADR-0006).

The model sees one numbered block per page and must answer with ``[n]`` markers, then a
``Quotes:`` section with one ``[n] "…"`` line per cited number, or reply ``NOT_FOUND``.
Everything it says about sources is checked: a number must be one of the pages it was
given, and a quote must be on that page (``quotes.py``). File names and page numbers
shown to the user come from the index, never from the model's text.

``answer`` yields events in the order the web app streams them: ``Pages`` once retrieval
is done, ``Token`` while the model writes, and ``Final`` with the validated answer.
"""

import re
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import PurePath

from askpdf.providers.base import ChatModel
from askpdf.quotes import PageKeys, QuoteCheck, check
from askpdf.retrieve import PageHit

NOT_FOUND = "NOT_FOUND"
# Pages given to the model. The answer model reads about 33 prompt tokens/s on the target
# Intel Mac, so 3 pages take about a minute before the first word and 5 nearly two
# (eval/TUNING.md).
ANSWER_K = 3
MAX_PAGE_CHARS = 4000  # per context block; the quote is still checked on the whole page

SYSTEM = f"""You answer questions about a user's documents, using only the numbered pages \
you are given. Do not use outside knowledge.

Rules:
1. End every sentence of your answer with the number of the page it comes from, in \
square brackets: [1], [2] and so on.
2. After the answer, write a line "Quotes:". Under it, for each number you cited, write \
one line: the number in brackets, then words copied exactly from that page, in double \
quotes. Copy at least 5 words in a row, exactly as they appear on the page.
3. If the pages do not contain the answer, reply with exactly {NOT_FOUND} and nothing else.

Example of the format:
The tank holds 40 litres when full [2]. It is refilled every Monday [1].
Quotes:
[2] "the tank holds 40 litres of water when full"
[1] "the tank is refilled every Monday morning\""""


@dataclass(frozen=True, slots=True)
class Pages:
    hits: tuple[PageHit, ...]


@dataclass(frozen=True, slots=True)
class Token:
    text: str


@dataclass(frozen=True, slots=True)
class Citation:
    number: int  # as shown in the final answer: renumbered by first use
    hit: PageHit
    quotes: tuple[QuoteCheck, ...]

    @property
    def verified(self) -> bool:
        return any(q.verified for q in self.quotes)


@dataclass(frozen=True, slots=True)
class Answer:
    text: str  # the answer without its Quotes section, markers renumbered
    not_found: bool
    citations: tuple[Citation, ...]
    dropped: tuple[int, ...]  # cited numbers that were not in the context
    sentences: int  # sentences in the answer
    supported: int  # of those, sentences carrying a citation with a verified quote
    raw: str  # what the model wrote

    @property
    def grounded(self) -> bool:
        return any(c.verified for c in self.citations)


@dataclass(frozen=True, slots=True)
class Final:
    answer: Answer


type Event = Pages | Token | Final


def source_name(hit: PageHit) -> str:
    return PurePath(hit.paths[0]).name if hit.paths else hit.sha256[:12]


def build_prompt(
    question: str, hits: Sequence[PageHit], texts: Mapping[tuple[str, int], str]
) -> str:
    """The user message: one block per page, then the question."""
    blocks = []
    for n, hit in enumerate(hits, start=1):
        printed = "" if hit.label == str(hit.page) else f" (printed {hit.label})"
        text = texts.get((hit.sha256, hit.page), hit.best_chunk.text)
        if len(text) > MAX_PAGE_CHARS:
            text = text[:MAX_PAGE_CHARS] + " …"
        blocks.append(f"[{n}] {source_name(hit)}, page {hit.page}{printed}\n{text}")
    return "Pages:\n\n" + "\n\n".join(blocks) + f"\n\nQuestion: {question}"


def context_window(user: str, reply_tokens: int) -> int:
    """Ollama's ``num_ctx`` for this prompt: big enough not to cut the context, small
    enough not to waste memory on the target 8 GB machine. About 3 characters a token
    is a conservative estimate for English; the result is rounded up to 1024."""
    needed = (len(SYSTEM) + len(user)) // 3 + reply_tokens
    return min(32768, max(2048, -(-needed // 1024) * 1024))


def answer(
    question: str,
    hits: Sequence[PageHit],
    texts: Mapping[tuple[str, int], str],
    chat: ChatModel,
) -> Iterator[Event]:
    """Retrieved pages in, events out. With no pages the model is not asked at all."""
    hits = tuple(hits)
    yield Pages(hits)
    if not hits:
        yield Final(parse(NOT_FOUND, hits, texts))
        return
    pieces = []
    for piece in chat.stream(SYSTEM, build_prompt(question, hits, texts)):
        pieces.append(piece)
        yield Token(piece)
    yield Final(parse("".join(pieces), hits, texts))


# --- validation ---------------------------------------------------------------------------

# "Quotes:" on its own line (or starting one), tolerating Markdown such as "**Quotes:**".
_HEADING = re.compile(r"^[ \t*#_>]*quotes?[ \t*_]*(?::[ \t*_]*|$)", re.I | re.M)
_MARKER = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
_QUOTE_LINE = re.compile(r"^[\W_]*?\[(\d+)\]\s*[:\-\u2013\u2014]?\s*(.+?)\s*$")
_WRAPPING = "\"'" + "".join(map(chr, (0x201C, 0x201D, 0x2018, 0x2019, 0xAB, 0xBB)))
_SENTENCE = re.compile(r"(?<=[.!?])\s+|\n+")
_LETTER = re.compile(r"[^\W\d_]")


def _not_found(raw: str) -> bool:
    first = next((line for line in raw.splitlines() if line.strip()), "")
    return re.fullmatch(rf"[\W_]*{NOT_FOUND}[\W_]*", first.strip()) is not None


def _split(raw: str) -> tuple[str, str]:
    heading = _HEADING.search(raw)
    if heading is None:
        return raw.strip(), ""
    return raw[: heading.start()].strip(), raw[heading.end() :]


def _quotes(section: str) -> dict[int, list[str]]:
    found: dict[int, list[str]] = {}
    for line in section.splitlines():
        match = _QUOTE_LINE.match(line)
        if match is None:
            continue
        # Quote marks, and punctuation at either end, are not part of the quoted words.
        text = match.group(2).strip(_WRAPPING + ".,;:\u2026 \t")
        if text:
            found.setdefault(int(match.group(1)), []).append(text)
    return found


def parse(raw: str, hits: Sequence[PageHit], texts: Mapping[tuple[str, int], str]) -> Answer:
    """Validate what the model wrote against the pages it was given."""
    body, section = _split(raw)
    valid = range(1, len(hits) + 1)
    order: dict[int, int] = {}  # number in the context -> number in the answer
    dropped: list[int] = []

    def renumber(match: re.Match[str]) -> str:
        kept = []
        for part in match.group(1).split(","):
            n = int(part)
            if n not in valid:
                dropped.append(n)
                continue
            order.setdefault(n, len(order) + 1)
            if order[n] not in kept:
                kept.append(order[n])
        return "".join(f"[{n}]" for n in kept)

    text = _MARKER.sub(renumber, body)
    quotes = _quotes(section)
    if not order and _not_found(raw):
        return Answer(NOT_FOUND, True, (), tuple(dropped), 0, 0, raw)
    for n in quotes:  # quoted but never cited in the text: still a source
        if n in valid:
            order.setdefault(n, len(order) + 1)
    keys = PageKeys()
    citations = []
    for n, shown in sorted(order.items(), key=lambda item: item[1]):
        hit = hits[n - 1]
        page = texts.get((hit.sha256, hit.page), "")
        page_keys = keys.get((hit.sha256, hit.page), page)
        checks = tuple(check(q, page, page_keys) for q in quotes.get(n, []))
        citations.append(Citation(shown, hit, checks))
    verified = {c.number for c in citations if c.verified}
    sentences = [s for s in _SENTENCE.split(text) if _LETTER.search(s)]
    supported = sum(
        1
        for s in sentences
        if any(int(n) in verified for m in _MARKER.finditer(s) for n in m.group(1).split(","))
    )
    return Answer(
        text=text,
        not_found=False,
        citations=tuple(citations),
        dropped=tuple(dict.fromkeys(dropped)),
        sentences=len(sentences),
        supported=supported,
        raw=raw,
    )
