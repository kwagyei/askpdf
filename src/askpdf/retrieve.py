"""Rank pages for a question (design §7, ADR-0004).

Retrieval scores chunks, but citations and the ground truth are per page, so the
ranking is collapsed to pages: a page scores as its best chunk. Ties are broken by
chunk id, so the same index and question always give the same order.
"""

from dataclasses import dataclass

import numpy as np

from askpdf.bm25 import tokenize
from askpdf.snapshot import Chunk, Snapshot

DEFAULT_K = 5  # provisional; k is chosen on the dev split in PR 6 (ADR-0004)


@dataclass(frozen=True, slots=True)
class PageHit:
    sha256: str
    page: int  # physical, 1-based
    label: str  # printed page label
    paths: tuple[str, ...]
    score: float
    best_chunk: Chunk  # the chunk that gave the page its score


def search_bm25(snapshot: Snapshot, question: str, k: int = DEFAULT_K) -> list[PageHit]:
    """The ``k`` best pages for ``question``. Pages matching no query term are left out."""
    if k < 1:
        raise ValueError("k must be at least 1")
    scores = snapshot.bm25.scores(tokenize(question))
    matching = np.flatnonzero(scores > 0)
    # Best score first; among equal scores the lower chunk id (lexsort: last key first).
    order = matching[np.lexsort((matching, -scores[matching]))]
    hits: list[PageHit] = []
    seen: set[tuple[str, int]] = set()
    for i in order:
        chunk = snapshot.chunks[i]
        key = (chunk.sha256, chunk.page)
        if key in seen:
            continue
        seen.add(key)
        hits.append(
            PageHit(
                sha256=chunk.sha256,
                page=chunk.page,
                label=snapshot.labels.get(key, str(chunk.page)),
                paths=snapshot.paths.get(chunk.sha256, ()),
                score=float(scores[i]),
                best_chunk=chunk,
            )
        )
        if len(hits) == k:
            break
    return hits
