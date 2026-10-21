"""Adversarial checks on PR 4's evaluation code (tester pass, 2026-10-21).

Tests marked ``xfail(strict=True)`` document a defect found in review: they assert the
behaviour the design asks for, and will start to fail (strictly) once the defect is fixed,
at which point the marker should be removed.
"""

import json
import random
import shutil
from pathlib import Path

import pytest

from askpdf.cli import main
from askpdf.evaluate.metrics import Detection, retrieval_scores
from askpdf.evaluate.questions import Question, evidence_problems, load
from askpdf.evaluate.stats import (
    bootstrap_ci,
    cohens_kappa,
    paired_bootstrap,
    sign_flip_exact,
    sign_flip_test,
)
from askpdf.store import PageRecord

EVAL = Path(__file__).resolve().parent.parent / "eval"
in_repo = pytest.mark.skipif(not (EVAL / "questions").is_dir(), reason="eval/ is not in the sdist")


def _row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "id": "q1",
        "question": "How long?",
        "lang": "en",
        "doc": "d",
        "pages": [1],
        "evidence": ["the timeout is 15 minutes"],
        "answer": "15 minutes",
        "kind": "lexical",
        "answerable": True,
    }
    row.update(overrides)
    return row


def _page(n: int, text: str, boilerplate: tuple[int, ...] = ()) -> PageRecord:
    return PageRecord(n, str(n), text, boilerplate, False)


# --- metrics --------------------------------------------------------------------------------


def test_empty_run_scores_zero_everywhere() -> None:
    assert set(retrieval_scores([], {"a"}).values()) == {0.0}


def test_run_shorter_than_k_and_relevant_pages_beyond_it() -> None:
    scores = retrieval_scores(["x", "g1"], {"g1", "g2", "g3"})
    assert scores["recall@5"] == pytest.approx(1 / 3)
    assert scores["recall@10"] == pytest.approx(1 / 3)
    assert scores["hit@1"] == 0.0
    assert scores["mrr@10"] == 0.5


def test_more_relevant_pages_than_the_cutoff_cap_the_ideal_ranking() -> None:
    """With 12 relevant pages and all of the top 10 relevant, nDCG@10 is 1, recall@10 10/12."""
    gold = {f"g{i}" for i in range(12)}
    run = [f"g{i}" for i in range(10)]
    scores = retrieval_scores(run, gold)
    assert scores["ndcg@10"] == pytest.approx(1.0)
    assert scores["recall@10"] == pytest.approx(10 / 12)
    assert scores["recall@5"] == pytest.approx(5 / 12)


def test_relevant_page_at_rank_eleven_is_invisible() -> None:
    run = [f"x{i}" for i in range(10)] + ["g"]
    assert retrieval_scores(run, {"g"})["recall@10"] == 0.0
    assert retrieval_scores(run, {"g"})["ndcg@10"] == 0.0


def test_metrics_match_pytrec_eval_with_many_relevant_pages() -> None:
    """The committed cross-check caps the gold set at 5; trec_eval's ideal DCG differs above 10."""
    pytrec_eval = pytest.importorskip("pytrec_eval")
    rng = random.Random(0)  # noqa: S311
    pages = [f"p{i}" for i in range(40)]
    for _ in range(300):
        ranked = rng.sample(pages, rng.randint(1, 10))
        gold = set(rng.sample(pages, rng.randint(1, 18)))
        run = {"q": {p: float(len(ranked) - r) for r, p in enumerate(ranked)}}
        qrels = {"q": dict.fromkeys(gold, 1)}
        ref = pytrec_eval.RelevanceEvaluator(
            qrels, {"recall.5,10", "recip_rank", "ndcg_cut.10"}
        ).evaluate(run)["q"]
        ours = retrieval_scores(ranked, gold)
        assert ours["recall@5"] == pytest.approx(ref["recall_5"], abs=1e-9)
        assert ours["recall@10"] == pytest.approx(ref["recall_10"], abs=1e-9)
        assert ours["mrr@10"] == pytest.approx(ref["recip_rank"], abs=1e-9)
        assert ours["ndcg@10"] == pytest.approx(ref["ndcg_cut_10"], abs=1e-9)


def test_detection_with_nothing_predicted_or_nothing_to_find() -> None:
    none = Detection.of(set(), {1})
    assert none.precision is None
    assert none.recall == 0.0
    nothing_gold = Detection.of({1}, set())
    assert nothing_gold.recall is None
    assert nothing_gold.precision == 0.0


# --- statistics -----------------------------------------------------------------------------


def test_bootstrap_of_one_value_and_of_a_constant() -> None:
    ci = bootstrap_ci([0.25])
    assert (ci.low, ci.estimate, ci.high) == (0.25, 0.25, 0.25)
    ci = bootstrap_ci([1.0] * 30)
    assert ci.low == ci.high == 1.0


def test_bootstrap_percentile_is_the_percentile_of_the_resampled_means() -> None:
    """Independent re-implementation: same generator, same draws, plain numpy."""
    import numpy as np

    data = [0.0, 1.0, 1.0, 0.5, 0.0, 1.0, 0.25]
    rng = np.random.default_rng(0)
    arr = np.asarray(data)
    means = np.array([arr[rng.integers(0, arr.size, arr.size)].mean() for _ in range(2000)])
    ci = bootstrap_ci(data, resamples=2000)
    # Row-wise draws from a (R, n) block equal R successive size-n draws.
    assert ci.low == pytest.approx(float(np.percentile(means, 2.5)))
    assert ci.high == pytest.approx(float(np.percentile(means, 97.5)))


def test_paired_bootstrap_of_identical_systems_is_exactly_zero() -> None:
    ci = paired_bootstrap([0.3, 0.9, 0.1], [0.3, 0.9, 0.1])
    assert (ci.low, ci.estimate, ci.high) == (0.0, 0.0, 0.0)
    assert not ci.excludes_zero()


def test_sign_flip_with_every_difference_zero_is_one() -> None:
    assert sign_flip_test([0.5] * 8, [0.5] * 8) == 1.0
    assert sign_flip_exact([0.5] * 8, [0.5] * 8) == 1.0


def test_sign_flip_never_reports_zero_and_floors_at_one_over_permutations_plus_one() -> None:
    p = sign_flip_test([1.0] * 40, [0.0] * 40, permutations=1000)
    assert p == pytest.approx(1 / 1001)


def test_sign_flip_exact_counts_ties_with_the_observed_statistic() -> None:
    """d = [1, 1]: |mean| is 1 for 2 of 4 sign patterns, so p = 0.5 exactly."""
    assert sign_flip_exact([1.0, 1.0], [0.0, 0.0]) == 0.5
    # d = [1, -1, 0]: the observed |mean| is 0, so every sign pattern ties or exceeds it.
    assert sign_flip_exact([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]) == 1.0


def test_sign_flip_random_agrees_with_exact_on_dev_like_differences() -> None:
    rng = random.Random(7)  # noqa: S311
    a = [float(rng.random() < 0.7) for _ in range(16)]
    b = [float(rng.random() < 0.5) for _ in range(16)]
    exact = sign_flip_exact(a, b)
    approx = sign_flip_test(a, b)
    assert approx == pytest.approx(exact, abs=0.02)


def test_sign_flip_is_two_sided() -> None:
    a, b = [1.0, 1.0, 1.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0, 0.0]
    assert sign_flip_exact(a, b) == sign_flip_exact(b, a)
    assert sign_flip_test(a, b) == sign_flip_test(b, a)


def test_kappa_edge_cases() -> None:
    assert cohens_kappa(["a", "b", "a", "b"], ["a", "b", "a", "b"]) == 1.0
    assert cohens_kappa(["a", "b"], ["b", "a"]) == -1.0
    assert cohens_kappa(["a", "a", "b", "b"], ["a", "b", "a", "b"]) == 0.0
    # labels that one labeller never uses
    assert cohens_kappa(["a", "a", "b", "b"], ["c", "c", "c", "c"]) == 0.0
    with pytest.raises(ValueError):
        cohens_kappa(["a", "a"], ["a", "a"])


# --- the evidence check ---------------------------------------------------------------------


def test_a_quote_split_across_a_page_boundary_is_refused() -> None:
    pages = {"d": [_page(1, "the timeout is"), _page(2, "15 minutes at most")]}
    q = Question.from_json(_row(pages=[1, 2], kind="multi-page"))
    assert evidence_problems(q, pages)  # neither page holds the whole quote


def test_a_quote_on_the_wrong_page_is_refused() -> None:
    pages = {"d": [_page(1, "nothing here"), _page(2, "the timeout is 15 minutes")]}
    q = Question.from_json(_row(pages=[1]))
    assert "quote not on its pages" in evidence_problems(q, pages)[0]


def test_a_hyphen_trick_does_not_match() -> None:
    pages = {"d": [_page(1, "the timeout is 15 minutes")]}
    q = Question.from_json(_row(evidence=["the time-out is 15 minutes"]))
    assert evidence_problems(q, pages)
    q = Question.from_json(_row(evidence=["the timeout is 15\u00a0minutes"]))
    assert evidence_problems(q, pages) == []  # NBSP counts as whitespace, as intended


def test_a_whitespace_only_quote_is_not_evidence() -> None:
    with pytest.raises(ValueError, match="shorter than"):
        Question.from_json(_row(evidence=[" " * 20]))
    # Even if one is built directly, the evidence check refuses it.
    q = Question("q1", "?", "en", "d", (1,), (" " * 20,), "a", "lexical", True)
    assert evidence_problems(q, {"d": [_page(1, "anything at all")]}) == [
        "q1: an empty quote proves nothing"
    ]


def test_padding_a_short_quote_does_not_defeat_the_length_guard() -> None:
    with pytest.raises(ValueError, match="shorter than"):
        Question.from_json(_row(evidence=["1.6" + " " * 20]))


def test_a_header_line_is_not_evidence() -> None:
    header = "NIST SP 800-63B-4 Digital Identity Guidelines"
    pages = {"d": [_page(n, f"{header}\nbody {n}", (0,)) for n in (1, 2, 3)]}
    q = Question.from_json(_row(pages=[1, 2, 3], kind="multi-page", evidence=[header]))
    assert evidence_problems(q, pages)


def test_a_multi_page_question_needs_two_distinct_pages() -> None:
    with pytest.raises(ValueError, match="at least two"):
        Question.from_json(_row(pages=[1, 1], kind="multi-page"))


@in_repo
@pytest.mark.parametrize("split", ["dev", "test"])
def test_committed_quotes_are_long_enough_after_whitespace_is_collapsed(split: str) -> None:
    for q in load(EVAL / "questions" / f"{split}.jsonl"):
        for quote in q.evidence:
            assert len(" ".join(quote.split())) >= 15, (q.id, quote)


# --- CLI error paths ------------------------------------------------------------------------


@in_repo
def test_malformed_question_file_is_an_error_not_a_traceback(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    shutil.copytree(EVAL / "corpus", tmp_path / "corpus")
    (tmp_path / "questions").mkdir()
    (tmp_path / "questions" / "dev.jsonl").write_text("{not json\n", encoding="utf-8")
    assert main(["eval", "retrieval", "--eval-dir", str(tmp_path)]) == 1  # raises JSONDecodeError
    assert "askpdf: error" in capsys.readouterr().err


@in_repo
def test_unwritable_json_path_is_an_error_not_a_traceback(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    target = tmp_path / "missing-dir" / "out.json"
    assert main(["eval", "conversation", "--eval-dir", str(EVAL), "--json", str(target)]) == 1
    assert "askpdf: error" in capsys.readouterr().err


@in_repo
def test_the_test_split_writes_its_release_file(tmp_path: Path) -> None:
    shutil.copytree(EVAL, tmp_path / "eval", ignore=shutil.ignore_patterns("results", "pages"))
    code = main(
        [
            "eval",
            "conversation",
            "--split",
            "test",
            "--release",
            "v9.9.9",
            "--eval-dir",
            str(tmp_path / "eval"),
        ]
    )
    assert code == 0
    assert (tmp_path / "eval" / "results" / "test-v9.9.9.json").is_file()


@in_repo
def test_release_pattern_rejects_near_misses(capsys: pytest.CaptureFixture[str]) -> None:
    for bad in (
        "1.0.0",
        "v1.0",
        "v1.0.0-rc1",
        "v1.0.0\n",
        " v1.0.0",
        "v1.0.0 ",
        "vX.Y.Z",
    ):
        assert (
            main(
                ["eval", "retrieval", "--split", "test", "--release", bad, "--eval-dir", str(EVAL)]
            )
            == 1
        ), bad
    assert "once per release" in capsys.readouterr().err


@in_repo
def test_release_pattern_is_ascii_only(tmp_path: Path) -> None:
    code = main(
        [
            "eval",
            "retrieval",
            "--split",
            "test",
            "--release",
            "v\uff11.0.0",
            "--eval-dir",
            str(EVAL),
        ]
    )
    assert code == 1


@in_repo
def test_results_do_not_depend_on_where_the_eval_directory_lives(tmp_path: Path) -> None:
    """Copy with files created in reverse order: BM25 ties must not follow directory order."""
    dest = tmp_path / "somewhere" / "else"
    (dest / "corpus").mkdir(parents=True)
    for pdf in sorted((EVAL / "corpus").iterdir(), reverse=True):
        shutil.copy(pdf, dest / "corpus" / pdf.name)
    shutil.copytree(EVAL / "questions", dest / "questions")
    shutil.copytree(EVAL / "conversations", dest / "conversations")
    out = tmp_path / "r.json"
    assert main(["eval", "retrieval", "--eval-dir", str(dest), "--json", str(out)]) == 0
    committed = json.loads((EVAL / "results" / "dev-retrieval-bm25.json").read_text("utf-8"))
    assert json.loads(out.read_text("utf-8")) == committed
