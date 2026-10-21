"""Stage (b): retrieval evaluated over the question sets."""

from pathlib import Path

import pytest

from askpdf.cli import main
from askpdf.evaluate.questions import Question
from askpdf.evaluate.retrieval import evaluate
from tests.eval_pins import require_pinned_extractor

EVAL = Path(__file__).resolve().parent.parent / "eval"


def _q(qid: str, doc: str, pages: list[int], kind: str = "lexical") -> Question:
    answerable = bool(pages)
    return Question(
        qid,
        f"question {qid}",
        "en",
        doc,
        tuple(pages),
        ("a long enough quote",) if answerable else (),
        "answer" if answerable else None,
        kind,
        answerable,
    )


def test_scores_are_grouped_by_kind_and_document() -> None:
    runs = {
        "question a": [("d1", 3), ("d1", 4)],
        "question b": [("d2", 9), ("d2", 1)],
        "question u": [("d1", 1)],
    }
    questions = [_q("a", "d1", [3]), _q("b", "d2", [1], "paraphrase"), _q("u", "d1", [])]
    result = evaluate(questions, lambda text: runs[text])
    assert result["n_questions"] == 2  # the unanswerable one is not scored here
    assert result["per_question"]["a"]["hit@1"] == 1.0
    assert result["per_question"]["b"]["mrr@10"] == 0.5
    assert result["overall"]["hit@1"] == 0.5
    assert result["by_kind"] == {
        "lexical": result["per_question"]["a"],
        "paraphrase": result["per_question"]["b"],
    }
    assert set(result["by_doc"]) == {"d1", "d2"}
    assert result["ranked"]["b"] == [["d2", 9], ["d2", 1]]


def test_the_test_split_needs_a_release(capsys: pytest.CaptureFixture[str]) -> None:
    for extra in ([], ["--release", "next"]):
        assert main(["eval", "retrieval", "--split", "test", "--eval-dir", str(EVAL), *extra]) == 1
    assert "once per release" in capsys.readouterr().err


def test_a_missing_question_set_is_reported(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "corpus").mkdir()
    assert main(["eval", "retrieval", "--eval-dir", str(tmp_path)]) == 1
    assert "no question set" in capsys.readouterr().err


@pytest.mark.skipif(not (EVAL / "questions").is_dir(), reason="eval/ is not in the sdist")
def test_bm25_on_dev_reproduces_the_committed_results(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    require_pinned_extractor()
    out = tmp_path / "retrieval.json"
    assert main(["eval", "retrieval", "--eval-dir", str(EVAL), "--json", str(out)]) == 0
    committed = EVAL / "results" / "dev-retrieval-bm25.json"
    assert out.read_text(encoding="utf-8") == committed.read_text(encoding="utf-8")
    assert "BM25 on dev" in capsys.readouterr().out
