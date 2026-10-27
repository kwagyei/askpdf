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
from askpdf.indexer import index_folder
from askpdf.retrieve import search_bm25
from askpdf.snapshot import Snapshot
from askpdf.store import Store

DEPTH = 10  # every metric looks at the top 10 pages at most

type Retriever = Callable[[str], list[tuple[str, int]]]


def bm25_retriever(snapshot: Snapshot) -> Retriever:
    def retrieve(question: str) -> list[tuple[str, int]]:
        return [
            (Path(hit.paths[0]).stem if hit.paths else hit.sha256, hit.page)
            for hit in search_bm25(snapshot, question, k=DEPTH)
        ]

    return retrieve


def load_corpus_snapshot(corpus: Path) -> Snapshot:
    """Index the corpus into a throwaway index and return its snapshot."""
    with tempfile.TemporaryDirectory() as tmp, Store(Path(tmp) / "index.db") as store:
        index_folder(corpus, store, PypdfExtractor(), ChunkParams())
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
