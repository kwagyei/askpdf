"""Evaluation metrics and statistics (ADR-0007)."""

import math

import pytest
import pytrec_eval
from hypothesis import given
from hypothesis import strategies as st

from askpdf.evaluate.metrics import (
    Detection,
    hit_at,
    mean_scores,
    ndcg_at,
    recall_at,
    reciprocal_rank,
    retrieval_scores,
)
from askpdf.evaluate.stats import (
    bootstrap_ci,
    cohens_kappa,
    paired_bootstrap,
    sign_flip_exact,
    sign_flip_test,
)

# --- retrieval metrics by hand -----------------------------------------------------------

RUN = ["p3", "p1", "p7", "p2", "p9"]
GOLD = {"p1", "p2"}


def test_hit() -> None:
    assert hit_at(RUN, GOLD, 1) == 0.0
    assert hit_at(RUN, GOLD, 2) == 1.0


def test_recall() -> None:
    assert recall_at(RUN, GOLD, 3) == 0.5
    assert recall_at(RUN, GOLD, 5) == 1.0


def test_reciprocal_rank() -> None:
    assert reciprocal_rank(RUN, GOLD) == 0.5
    assert reciprocal_rank(RUN, {"p9"}, k=4) == 0.0  # beyond the cutoff
    assert reciprocal_rank([], GOLD) == 0.0


def test_ndcg() -> None:
    # Relevant at ranks 2 and 4: DCG = 1/log2(3) + 1/log2(5); ideal = 1 + 1/log2(3).
    expected = (1 / math.log2(3) + 1 / math.log2(5)) / (1 + 1 / math.log2(3))
    assert ndcg_at(RUN, GOLD) == pytest.approx(expected)
    assert ndcg_at(["p1", "p2"], GOLD) == 1.0


def test_metrics_need_a_relevant_page_and_a_run_without_repeats() -> None:
    with pytest.raises(ValueError, match="relevant"):
        ndcg_at(RUN, set())
    with pytest.raises(ValueError, match="twice"):
        retrieval_scores(["p1", "p1"], GOLD)


def test_mean_scores() -> None:
    per_q = {"q1": {"hit@1": 1.0}, "q2": {"hit@1": 0.0}}
    assert mean_scores(per_q) == {"hit@1": 0.5}
    assert mean_scores({}) == {}


# --- cross-check against pytrec_eval (ADR-0007: must match to 1e-9) ----------------------

_PAGES = [f"d{d}p{p}" for d in range(3) for p in range(6)]


@st.composite
def _case(draw: st.DrawFn) -> tuple[list[str], set[str]]:
    ranked = draw(st.permutations(_PAGES).map(lambda ps: ps[: draw(st.integers(0, 18))]))
    gold = draw(st.sets(st.sampled_from(_PAGES), min_size=1, max_size=5))
    return list(ranked), gold


@given(st.lists(_case(), min_size=1, max_size=6))
def test_metrics_match_pytrec_eval(cases: list[tuple[list[str], set[str]]]) -> None:
    qrels = {f"q{i}": dict.fromkeys(gold, 1) for i, (_, gold) in enumerate(cases)}
    # Truncated to 10, with strictly falling scores so trec_eval keeps our order.
    run = {
        f"q{i}": {p: float(len(ranked) - r) for r, p in enumerate(ranked[:10])}
        for i, (ranked, _) in enumerate(cases)
    }
    measures = {"success.1,3,5", "recall.5,10", "recip_rank", "ndcg_cut.10"}
    reference = pytrec_eval.RelevanceEvaluator(qrels, measures).evaluate(run)
    names = {
        "hit@1": "success_1",
        "hit@3": "success_3",
        "hit@5": "success_5",
        "recall@5": "recall_5",
        "recall@10": "recall_10",
        "mrr@10": "recip_rank",
        "ndcg@10": "ndcg_cut_10",
    }
    for i, (ranked, gold) in enumerate(cases):
        ours = retrieval_scores(ranked, gold)
        theirs = reference.get(f"q{i}")
        if theirs is None:  # trec_eval skips a query with an empty run
            assert not ranked and all(v == 0 for v in ours.values())
            continue
        for metric, trec_name in names.items():
            assert ours[metric] == pytest.approx(theirs[trec_name], abs=1e-9), metric


# --- detection ------------------------------------------------------------------------------


def test_detection_precision_and_recall() -> None:
    d = Detection.of(predicted={1, 2, 3}, gold={2, 3, 4, 5})
    assert (d.true_positives, d.false_positives, d.false_negatives) == (2, 1, 2)
    assert d.precision == pytest.approx(2 / 3)
    assert d.recall == 0.5
    nothing = Detection.of(predicted=set(), gold=set())
    assert nothing.precision is None and nothing.recall is None


# --- statistics -----------------------------------------------------------------------------


def test_cohens_kappa_matches_the_worked_example() -> None:
    # The textbook example (Wikipedia, "Cohen's kappa"): 50 items; both yes 20, A yes and
    # B no 5, A no and B yes 10, both no 15. p_o = 0.7, p_e = 0.5, so kappa = 0.4.
    a = ["yes"] * 25 + ["no"] * 25
    b = ["yes"] * 20 + ["no"] * 5 + ["yes"] * 10 + ["no"] * 15
    assert cohens_kappa(a, b) == pytest.approx(0.4)


def test_kappa_edge_cases() -> None:
    assert cohens_kappa(["x", "y"], ["x", "y"]) == 1.0
    with pytest.raises(ValueError, match="single label"):
        cohens_kappa(["x", "x"], ["x", "x"])
    with pytest.raises(ValueError, match="equally long"):
        cohens_kappa(["x"], [])


def test_bootstrap_interval_is_reproducible_and_contains_the_mean() -> None:
    values = [0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0]
    first, second = bootstrap_ci(values), bootstrap_ci(values)
    assert first == second
    assert first.low <= first.estimate == 0.7 <= first.high
    assert bootstrap_ci([0.5] * 8) == bootstrap_ci([0.5] * 8)
    assert bootstrap_ci([0.5] * 8).low == bootstrap_ci([0.5] * 8).high == 0.5
    with pytest.raises(ValueError, match="empty"):
        bootstrap_ci([])


def test_paired_bootstrap_of_a_clear_improvement_excludes_zero() -> None:
    better = [1.0] * 18 + [0.0] * 2
    worse = [0.0] * 14 + [1.0] * 6
    assert paired_bootstrap(better, worse).excludes_zero()
    assert not paired_bootstrap(better, better).excludes_zero()


def test_exact_sign_flip_by_hand() -> None:
    # Three equal differences: only the all-plus and all-minus patterns reach |mean| = 1.
    assert sign_flip_exact([1, 1, 1], [0, 0, 0]) == 2 / 8
    assert sign_flip_exact([0, 0], [0, 0]) == 1.0
    with pytest.raises(ValueError, match="1 to 20"):
        sign_flip_exact([], [])


@given(
    st.lists(st.sampled_from([0.0, 0.25, 0.5, 1.0]), min_size=2, max_size=10).flatmap(
        lambda a: st.tuples(
            st.just(a), st.lists(st.sampled_from([0.0, 0.5, 1.0]), min_size=len(a), max_size=len(a))
        )
    )
)
def test_random_sign_flips_agree_with_exact_enumeration(
    pair: tuple[list[float], list[float]],
) -> None:
    a, b = pair
    exact = sign_flip_exact(a, b)
    sampled = sign_flip_test(a, b)
    # 10,000 flips: the standard error is at most 0.005, so 0.025 is five of them.
    # The sampled p includes the observed statistic, which adds at most 1/10,001.
    assert abs(sampled - exact) < 0.025


def test_paired_samples_must_match_in_length() -> None:
    with pytest.raises(ValueError, match="same length"):
        sign_flip_test([1.0], [1.0, 0.0])
    with pytest.raises(ValueError, match="empty"):
        sign_flip_test([], [])
