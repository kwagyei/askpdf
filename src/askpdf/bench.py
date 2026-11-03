"""Latency on this machine (design §11, stage g): p50 and p95, always with n.

Indexing and search need no model. Answers need Ollama: the first one is timed after
the model is unloaded (cold, it includes loading the model from disk) and reported on
its own; the rest are warm.
"""

import math
import tempfile
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from askpdf.answer import Final, Pages, Token, answer
from askpdf.chunking import ChunkParams
from askpdf.extract.pypdf_ext import PypdfExtractor
from askpdf.indexer import index_folder
from askpdf.providers.base import ChatModel
from askpdf.retrieve import PageHit, search_bm25
from askpdf.snapshot import Snapshot
from askpdf.store import Store

Clock = Callable[[], float]


def percentile(values: Sequence[float], share: float) -> float:
    """Nearest-rank percentile: always one of the measured values, never interpolated."""
    if not values:
        raise ValueError("no values")
    ordered = sorted(values)
    return ordered[max(0, math.ceil(share * len(ordered)) - 1)]


def spread(values: Sequence[float], digits: int = 3) -> dict[str, Any]:
    return {
        "n": len(values),
        "p50": round(percentile(values, 0.5), digits),
        "p95": round(percentile(values, 0.95), digits),
    }


def time_indexing(
    pdfs: Sequence[Path], runs: int, clock: Clock = time.perf_counter
) -> dict[str, Any]:
    """Each PDF indexed alone into a fresh index, ``runs`` times: extract, clean, chunk,
    store. Embedding is not included; it is timed by ``askpdf index`` as it goes."""
    seconds: list[float] = []
    pages = 0
    for _ in range(runs):
        for pdf in pdfs:
            with tempfile.TemporaryDirectory() as tmp, Store(Path(tmp) / "i.db") as store:
                started = clock()
                index_folder(pdf, store, PypdfExtractor(), ChunkParams())
                seconds.append(clock() - started)
                pages += sum(doc.n_pages for doc in store.documents())
    total = sum(seconds)
    return {
        "documents": len(pdfs),
        "runs": runs,
        "seconds_per_document": spread(seconds),
        "pages_per_second": round(pages / total, 1) if total else None,
    }


def time_search(
    snapshot: Snapshot, queries: Sequence[str], runs: int, k: int, clock: Clock = time.perf_counter
) -> dict[str, Any]:
    """Keyword search over the whole snapshot, in milliseconds."""
    ms: list[float] = []
    for _ in range(runs):
        for query in queries:
            started = clock()
            search_bm25(snapshot, query, k)
            ms.append((clock() - started) * 1000)
    return spread(ms, digits=2)


def time_answer(
    snapshot: Snapshot,
    question: str,
    k: int,
    chat_for: Callable[[str, Sequence[PageHit]], ChatModel],
    clock: Clock = time.perf_counter,
) -> dict[str, float | None]:
    """Seconds from the question to the pages shown, the first token and the checked
    answer, as the web app sees them. ``first_token`` is None if the model was not asked."""
    started = clock()
    hits = search_bm25(snapshot, question, k)
    times: dict[str, float | None] = {"pages": None, "first_token": None, "total": None}
    for event in answer(question, hits, snapshot.texts, chat_for(question, hits)):
        now = clock() - started
        if isinstance(event, Pages):
            times["pages"] = now
        elif isinstance(event, Token) and times["first_token"] is None:
            times["first_token"] = now
        elif isinstance(event, Final):
            times["total"] = now
    return times


def time_answers(
    snapshot: Snapshot,
    questions: Sequence[str],
    k: int,
    chat_for: Callable[[str, Sequence[PageHit]], ChatModel],
    unload: Callable[[], None],
    on_answer: Callable[[int, int], None] | None = None,
    clock: Clock = time.perf_counter,
) -> dict[str, Any]:
    """The first answer cold (after ``unload``), the rest warm."""
    unload()
    timed = []
    for n, question in enumerate(questions, start=1):
        timed.append(time_answer(snapshot, question, k, chat_for, clock))
        if on_answer is not None:
            on_answer(n, len(questions))
    cold, warm = timed[0], timed[1:]

    def column(key: str) -> dict[str, Any] | None:
        values = [v for t in warm if (v := t[key]) is not None]
        return spread(values, digits=1) if values else None

    return {
        "cold": {key: None if v is None else round(v, 1) for key, v in cold.items()},
        "warm": {key: column(key) for key in ("pages", "first_token", "total")},
    }
