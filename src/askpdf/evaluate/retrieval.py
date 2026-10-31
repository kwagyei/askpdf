"""Stage (b): does retrieval find the right page? (design §11, ADR-0007)

The whole corpus is indexed once, as a user would index a folder, and every answerable
question is searched across all of it. Pages are identified as (document, physical
page), the document being the PDF's file name without ``.pdf``.
"""

import tempfile
from collections import defaultdict
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from askpdf.chunking import ChunkParams
from askpdf.evaluate.metrics import RETRIEVAL_METRICS, mean_scores, retrieval_scores
from askpdf.evaluate.questions import Question
from askpdf.evaluate.stats import bootstrap_ci
from askpdf.extract.pypdf_ext import PypdfExtractor
from askpdf.indexer import embed_missing, index_folder
from askpdf.providers.base import Embedder, Matrix
from askpdf.retrieve import RRF_K, PageHit, search_bm25, search_dense, search_hybrid
from askpdf.snapshot import Snapshot
from askpdf.store import Store

DEPTH = 10  # every metric looks at the top 10 pages at most

type Retriever = Callable[[str], list[tuple[str, int]]]


def _ids(hits: Sequence[PageHit]) -> list[tuple[str, int]]:
    return [(Path(h.paths[0]).stem if h.paths else h.sha256, h.page) for h in hits]


def bm25_retriever(snapshot: Snapshot) -> Retriever:
    def retrieve(question: str) -> list[tuple[str, int]]:
        return _ids(search_bm25(snapshot, question, k=DEPTH))

    return retrieve


class QueryCache:
    """Embed each question once, however many retrievers are compared on it."""

    def __init__(self, embedder: Embedder) -> None:
        self.embedder = embedder
        self._vectors: dict[str, Matrix] = {}

    def __call__(self, question: str) -> Matrix:
        if question not in self._vectors:
            self._vectors[question] = self.embedder.embed_query(question)
        return self._vectors[question]


def model_retriever(
    snapshot: Snapshot, queries: QueryCache, kind: str, rrf_k: int = RRF_K
) -> Retriever:
    """Dense (cosine) or hybrid (RRF of BM25 and dense) retrieval."""
    if not snapshot.fully_embedded:
        raise ValueError("the evaluation index is not fully embedded")

    def retrieve(question: str) -> list[tuple[str, int]]:
        vector = queries(question)
        if kind == "dense":
            return _ids(search_dense(snapshot, vector, k=DEPTH))
        return _ids(search_hybrid(snapshot, question, vector, k=DEPTH, rrf_k=rrf_k))

    return retrieve


def load_corpus_snapshot(corpus: Path) -> Snapshot:
    """Index the corpus into a throwaway index and return its snapshot (no vectors)."""
    with tempfile.TemporaryDirectory() as tmp, Store(Path(tmp) / "index.db") as store:
        index_folder(corpus, store, PypdfExtractor(), ChunkParams())
        return Snapshot.load(store)


def load_embedded_snapshot(corpus: Path, embedder: Embedder, index: Path) -> Snapshot:
    """Index and embed the corpus into a lasting index, so later runs skip the slow part."""
    with Store(index) as store:
        index_folder(corpus, store, PypdfExtractor(), ChunkParams())
        if embed_missing(store, embedder).refused:
            raise ValueError("the model refused to embed some corpus chunks")
        return Snapshot.load(store)


def evaluate(questions: Sequence[Question], retrieve: Retriever) -> dict[str, Any]:
    """Per-question scores, and their means overall, per kind and per document."""
    per_question: dict[str, dict[str, float]] = {}
    ranked: dict[str, list[list[object]]] = {}
    for q in questions:
        if not q.answerable:
            continue
        pages = retrieve(q.question)
        per_question[q.id] = {m: round(v, 6) for m, v in retrieval_scores(pages, q.gold).items()}
        ranked[q.id] = [[doc, page] for doc, page in pages]
    by: dict[str, dict[str, dict[str, float]]] = {}
    for field in ("kind", "doc"):
        groups: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
        for q in questions:
            if q.id in per_question:
                groups[getattr(q, field)][q.id] = per_question[q.id]
        by[field] = {name: _rounded(mean_scores(scores)) for name, scores in sorted(groups.items())}
    intervals = {}
    if per_question:
        for metric in RETRIEVAL_METRICS:
            ci = bootstrap_ci([scores[metric] for scores in per_question.values()])
            intervals[metric] = [round(ci.low, 6), round(ci.high, 6)]
    return {
        "metrics": list(RETRIEVAL_METRICS),
        "n_questions": len(per_question),
        "overall": _rounded(mean_scores(per_question)),
        "interval_95": intervals,  # percentile bootstrap, 10,000 resamples, seed 0
        "by_kind": by["kind"],
        "by_doc": by["doc"],
        "per_question": dict(sorted(per_question.items())),
        "ranked": dict(sorted(ranked.items())),
    }


def _rounded(scores: dict[str, float]) -> dict[str, float]:
    return {m: round(v, 6) for m, v in scores.items()}
