"""Rank pages for a question (design §7, ADR-0004).

Retrieval scores chunks, but citations and the ground truth are per page, so the
ranking is collapsed to pages: a page scores as its best chunk. Ties are broken by
chunk id, so the same index and question always give the same order.
"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

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


RETRIEVERS = ("bm25", "dense", "hybrid")
RRF_DEPTH = 50  # chunks taken from each ranking before fusing (ADR-0004)
RRF_K = 60  # provisional; chosen on dev (eval/TUNING.md)
DEFAULT_RETRIEVER = "bm25"  # provisional; chosen on dev (eval/TUNING.md)


def _pages(
    snapshot: Snapshot,
    order: npt.NDArray[np.intp],
    scores: npt.NDArray[np.float64],
    k: int,
) -> list[PageHit]:
    """Collapse a chunk ranking to pages: each page takes its best-ranked chunk."""
    if k < 1:
        raise ValueError("k must be at least 1")
    hits: list[PageHit] = []
    seen: set[tuple[str, int]] = set()
    for i, score in zip(order, scores, strict=True):
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
                score=float(score),
                best_chunk=chunk,
            )
        )
        if len(hits) == k:
            break
    return hits


def _ranked(
    candidates: npt.NDArray[np.intp], scores: npt.NDArray[np.float64]
) -> npt.NDArray[np.intp]:
    """Candidates by score, best first; equal scores go to the lower chunk id."""
    return candidates[np.lexsort((candidates, -scores[candidates]))]


def bm25_chunks(
    snapshot: Snapshot, question: str
) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.float64]]:
    scores = snapshot.bm25.scores(tokenize(question))
    return _ranked(np.flatnonzero(scores > 0), scores), scores


def dense_chunks(
    snapshot: Snapshot, query: npt.NDArray[np.float32]
) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.float64]]:
    if snapshot.vectors.shape[1] != query.shape[0]:
        raise ValueError("the query vector does not match the index's embedder")
    scores = (snapshot.vectors @ query).astype(np.float64)
    return _ranked(np.flatnonzero(snapshot.embedded), scores), scores


def search_bm25(snapshot: Snapshot, question: str, k: int = DEFAULT_K) -> list[PageHit]:
    """The ``k`` best pages for ``question``. Pages matching no query term are left out."""
    order, scores = bm25_chunks(snapshot, question)
    return _pages(snapshot, order, scores[order], k)


def search_dense(
    snapshot: Snapshot, query: npt.NDArray[np.float32], k: int = DEFAULT_K
) -> list[PageHit]:
    """The ``k`` pages whose chunks are closest to the query vector (cosine)."""
    order, scores = dense_chunks(snapshot, query)
    return _pages(snapshot, order, scores[order], k)


def search_hybrid(
    snapshot: Snapshot,
    question: str,
    query: npt.NDArray[np.float32],
    k: int = DEFAULT_K,
    rrf_k: int = RRF_K,
) -> list[PageHit]:
    """Reciprocal rank fusion of the top ``RRF_DEPTH`` chunks of BM25 and dense."""
    fused: dict[int, float] = {}
    for order, _ in (bm25_chunks(snapshot, question), dense_chunks(snapshot, query)):
        for rank, i in enumerate(order[:RRF_DEPTH], start=1):
            fused[int(i)] = fused.get(int(i), 0.0) + 1.0 / (rrf_k + rank)
    candidates = np.fromiter(fused, dtype=np.intp, count=len(fused))
    scores = np.zeros(len(snapshot.chunks), dtype=np.float64)
    for i, value in fused.items():
        scores[i] = value
    order = _ranked(candidates, scores)
    return _pages(snapshot, order, scores[order], k)
