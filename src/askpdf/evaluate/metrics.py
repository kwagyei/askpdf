"""Retrieval and detection metrics, hand-rolled and cross-checked against pytrec_eval.

Retrieval is judged at page level with binary relevance: a run is the ranked list of
pages returned for one question, and its relevant pages are the gold ones. Questions
without relevant pages (unanswerable ones) have no retrieval score; they are measured
by abstention instead (ADR-0007).
"""

import math
from collections.abc import Collection, Hashable, Mapping, Sequence
from dataclasses import dataclass

type Page = Hashable  # e.g. (document, physical page)

CUTOFF = 10  # MRR@10 and nDCG@10


def hit_at(ranked: Sequence[Page], relevant: Collection[Page], k: int) -> float:
    """1 if any relevant page is in the top ``k``, else 0."""
    return float(any(p in relevant for p in ranked[:k]))


def recall_at(ranked: Sequence[Page], relevant: Collection[Page], k: int) -> float:
    """Share of the relevant pages found in the top ``k``."""
    _require(relevant)
    return len(set(ranked[:k]) & set(relevant)) / len(set(relevant))


def reciprocal_rank(ranked: Sequence[Page], relevant: Collection[Page], k: int = CUTOFF) -> float:
    """1 / rank of the first relevant page within the top ``k``; 0 if there is none."""
    for rank, page in enumerate(ranked[:k], start=1):
        if page in relevant:
            return 1 / rank
    return 0.0


def ndcg_at(ranked: Sequence[Page], relevant: Collection[Page], k: int = CUTOFF) -> float:
    """Normalised DCG with binary gains and a log2(rank + 1) discount, as trec_eval."""
    _require(relevant)
    dcg = sum(1 / math.log2(rank + 1) for rank, p in enumerate(ranked[:k], 1) if p in relevant)
    ideal = sum(1 / math.log2(rank + 1) for rank in range(1, min(len(set(relevant)), k) + 1))
    return dcg / ideal


def _require(relevant: Collection[Page]) -> None:
    if not relevant:
        raise ValueError("retrieval metrics need at least one relevant page")


def _check_run(ranked: Sequence[Page]) -> None:
    if len(set(ranked)) != len(ranked):
        raise ValueError("a run lists a page twice")


RETRIEVAL_METRICS = ("hit@1", "hit@3", "hit@5", "recall@5", "recall@10", "mrr@10", "ndcg@10")


def retrieval_scores(ranked: Sequence[Page], relevant: Collection[Page]) -> dict[str, float]:
    """Every retrieval metric of ADR-0007 for one question."""
    _check_run(ranked)
    return {
        "hit@1": hit_at(ranked, relevant, 1),
        "hit@3": hit_at(ranked, relevant, 3),
        "hit@5": hit_at(ranked, relevant, 5),
        "recall@5": recall_at(ranked, relevant, 5),
        "recall@10": recall_at(ranked, relevant, 10),
        "mrr@10": reciprocal_rank(ranked, relevant),
        "ndcg@10": ndcg_at(ranked, relevant),
    }


def mean_scores(per_question: Mapping[str, Mapping[str, float]]) -> dict[str, float]:
    """The mean of each metric over questions."""
    if not per_question:
        return {}
    names = next(iter(per_question.values())).keys()
    return {m: sum(q[m] for q in per_question.values()) / len(per_question) for m in names}


@dataclass(frozen=True, slots=True)
class Detection:
    """Precision and recall of a yes/no detector (boilerplate lines, scanned pages)."""

    true_positives: int
    false_positives: int
    false_negatives: int

    @classmethod
    def of(cls, predicted: Collection[Hashable], gold: Collection[Hashable]) -> "Detection":
        p, g = set(predicted), set(gold)
        return cls(len(p & g), len(p - g), len(g - p))

    @property
    def precision(self) -> float | None:
        """None when nothing was predicted: undefined, not perfect."""
        flagged = self.true_positives + self.false_positives
        return self.true_positives / flagged if flagged else None

    @property
    def recall(self) -> float | None:
        actual = self.true_positives + self.false_negatives
        return self.true_positives / actual if actual else None
