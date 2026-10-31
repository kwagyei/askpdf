"""Stages (c) grounding and (e) abstention: answers from a real model (design §11, ADR-0007).

Every question, answerable or not, is retrieved with the default retriever and answered
from the top ``ANSWER_K`` pages, exactly as ``askpdf ask`` would. Nothing here judges
whether an answer is *correct*; that needs people (stage d, at release).

Grounding:
- quote-verified rate: the share of citations with a quote found on the cited page
- folding-only share: of the verified citations, those verified only because hyphens
  between letters were folded (ADR-0006 Amendment 1)
- supported-sentence rate: the share of answer sentences carrying a verified citation
- grounded rate: the share of answers with at least one verified citation

Abstention, where an answer is anything but NOT_FOUND:
- false-answer rate: unanswerable questions that were answered
- NOT_FOUND precision and recall, with unanswerable as the positive class
- over-abstention: answerable questions answered NOT_FOUND
"""

import time
from collections.abc import Callable, Sequence
from pathlib import PurePath
from typing import Any

from askpdf.answer import ANSWER_K, NOT_FOUND, Answer, Final, answer, parse
from askpdf.evaluate.questions import Question
from askpdf.evaluate.stats import bootstrap_ci
from askpdf.providers.base import ChatModel
from askpdf.retrieve import PageHit
from askpdf.snapshot import Snapshot

type Retrieve = Callable[[str, int], list[PageHit]]


def run(
    questions: Sequence[Question],
    snapshot: Snapshot,
    retrieve: Retrieve,
    chat: Callable[[str, Sequence[PageHit]], ChatModel],
    on_question: Callable[[int, int, str], None] = lambda done, total, qid: None,
    k: int = ANSWER_K,
) -> dict[str, dict[str, Any]]:
    """Answer every question; one record per question id."""
    records: dict[str, dict[str, Any]] = {}
    for i, q in enumerate(questions, start=1):
        on_question(i, len(questions), q.id)
        hits = retrieve(q.question, k)
        started = time.monotonic()
        final = None
        for event in answer(q.question, hits, snapshot.texts, chat(q.question, hits)):
            if isinstance(event, Final):
                final = event.answer
        if final is None:  # answer() always ends with Final
            raise RuntimeError(f"{q.id}: the answer did not finish")
        records[q.id] = record(q, final, time.monotonic() - started)
    return records


def rescore(
    records: dict[str, dict[str, Any]],
    questions: Sequence[Question],
    snapshot: Snapshot,
    retrieve: Retrieve,
    k: int = ANSWER_K,
) -> dict[str, dict[str, Any]]:
    """Validate saved replies again, without the model: retrieval is deterministic, so
    the same pages come back, and only the checking of ``raw`` is redone."""
    by_id = {q.id: q for q in questions}
    rescored = {}
    for qid, old in records.items():
        q = by_id[qid]
        hits = retrieve(q.question, k)
        result = parse(old["raw"], hits, snapshot.texts) if hits else parse(NOT_FOUND, hits, {})
        rescored[qid] = record(q, result, old["seconds"])
    return rescored


def record(q: Question, result: Answer, seconds: float) -> dict[str, Any]:
    return {
        "answerable": q.answerable,
        "kind": q.kind,
        "not_found": result.not_found,
        "citations": [
            {
                "doc": PurePath(c.hit.paths[0]).stem if c.hit.paths else c.hit.sha256,
                "page": c.hit.page,
                "verified": c.verified,
                "needed_folding": c.verified
                and not any(x.verified and not x.needed_folding for x in c.quotes),
                "quotes": [x.quote for x in c.quotes],
                "reasons": [x.reason for x in c.quotes],
            }
            for c in result.citations
        ],
        "dropped": list(result.dropped),
        "sentences": result.sentences,
        "supported": result.supported,
        "seconds": round(seconds, 1),
        "answer": result.text,
        "raw": result.raw,
    }


def _rate(hits: int, total: int) -> float | None:
    return None if total == 0 else round(hits / total, 6)


def _interval(values: Sequence[float]) -> list[float] | None:
    if not values:
        return None
    ci = bootstrap_ci(values)
    return [round(ci.low, 6), round(ci.high, 6)]


def summarise(records: dict[str, dict[str, Any]]) -> dict[str, Any]:
    answerable = [r for r in records.values() if r["answerable"]]
    unanswerable = [r for r in records.values() if not r["answerable"]]
    answered = [r for r in answerable if not r["not_found"]]
    citations = [c for r in answered for c in r["citations"]]
    verified = [c for c in citations if c["verified"]]
    sentences = sum(r["sentences"] for r in answered)
    supported = sum(r["supported"] for r in answered)
    not_found = [r for r in records.values() if r["not_found"]]
    true_not_found = [r for r in not_found if not r["answerable"]]
    grounded = [1.0 if any(c["verified"] for c in r["citations"]) else 0.0 for r in answered]
    return {
        "n_questions": len(records),
        "n_answerable": len(answerable),
        "n_unanswerable": len(unanswerable),
        "grounding": {
            "n_answers": len(answered),
            "n_citations": len(citations),
            "quote_verified_rate": _rate(len(verified), len(citations)),
            "folding_only_share": _rate(sum(c["needed_folding"] for c in verified), len(verified)),
            "supported_sentence_rate": _rate(supported, sentences),
            "grounded_rate": _rate(int(sum(grounded)), len(grounded)),
            # Per answer, so citations of one answer are not counted as independent.
            "grounded_interval_95": _interval(grounded),
            "dropped_citations": sum(len(r["dropped"]) for r in answered),
            # Unverified citations by cause: no quote at all, or only quotes under 5 words.
            "unverified_no_quote": sum(not c["quotes"] for c in citations),
            "unverified_too_short": sum(
                bool(c["quotes"]) and all(x.startswith("shorter") for x in c["reasons"])
                for c in citations
            ),
        },
        "abstention": {
            "false_answer_rate": _rate(
                sum(not r["not_found"] for r in unanswerable), len(unanswerable)
            ),
            "not_found_precision": _rate(len(true_not_found), len(not_found)),
            "not_found_recall": _rate(len(true_not_found), len(unanswerable)),
            "over_abstention_rate": _rate(len(not_found) - len(true_not_found), len(answerable)),
            "over_abstention_interval_95": _interval(
                [1.0 if r["not_found"] else 0.0 for r in answerable]
            ),
        },
        "seconds_median": sorted(r["seconds"] for r in records.values())[len(records) // 2]
        if records
        else None,
    }
