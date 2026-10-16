"""Split a page into overlapping word windows (design §5).

Chunks never cross a page boundary, so every citation points at exactly one page.
Each chunk is a contiguous slice of the page's words; the next chunk starts exactly
``overlap`` words before the previous one ended, so dropping each chunk's first
``overlap`` words (except the first chunk's) reproduces the page word for word.
"""

import re
from dataclasses import dataclass

# A word that ends a sentence: terminal punctuation, optionally followed by closing
# quotes or brackets ("end." / "end.)" / 'end?"').
_SENTENCE_END = re.compile("[.!?\u2026][\"'\u201d\u2019)\\]]*$")


@dataclass(frozen=True, slots=True)
class ChunkParams:
    """Starting values; all three are tuned on the dev split only (design §11)."""

    target: int = 220
    overlap: int = 40
    snap: int = 30

    def __post_init__(self) -> None:
        if self.target < 1 or self.overlap < 0 or self.snap < 0:
            raise ValueError("target must be positive; overlap and snap non-negative")
        # Even the earliest snapped end must leave room for the overlap, or the next
        # chunk would not start after the previous one and chunking would never finish.
        if self.target - self.snap - self.overlap < 1:
            raise ValueError("target must exceed snap + overlap")


@dataclass(frozen=True, slots=True)
class Chunk:
    start: int  # first word, inclusive
    end: int  # last word, exclusive
    text: str


def _snapped_end(words: list[str], nominal: int, snap: int) -> int:
    """The end nearest ``nominal`` that falls just after a sentence, within ``snap`` words.

    Ties go to the earlier end. Without a sentence end in range, ``nominal`` stands.
    """
    best: int | None = None
    for end in range(nominal - snap, nominal + snap + 1):
        if not 1 <= end <= len(words):
            continue
        if _SENTENCE_END.search(words[end - 1]) and (
            best is None or abs(end - nominal) < abs(best - nominal)
        ):
            best = end
    return nominal if best is None else best


def chunk_words(words: list[str], params: ChunkParams) -> list[Chunk]:
    chunks: list[Chunk] = []
    start = 0
    while start < len(words):
        nominal = start + params.target
        if nominal >= len(words):
            end = len(words)
        else:
            end = min(_snapped_end(words, nominal, params.snap), len(words))
        chunks.append(Chunk(start, end, " ".join(words[start:end])))
        if end == len(words):
            break
        start = end - params.overlap
    return chunks


def chunk_page(text: str, params: ChunkParams) -> list[Chunk]:
    return chunk_words(text.split(), params)
