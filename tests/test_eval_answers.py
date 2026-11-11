"""Stages (c) and (e): grounding and abstention numbers from answer records."""

import datetime
import shutil
from collections.abc import Iterator, Sequence
from pathlib import Path

import pytest

from askpdf.evaluate import answers
from askpdf.evaluate.questions import Question
from askpdf.retrieve import PageHit
from askpdf.snapshot import Chunk, Snapshot

SHA = "a" * 64
TEXT = "Passwords SHALL be at least fifteen characters long for single-factor use."


def _q(qid: str, answerable: bool) -> Question:
    return Question(
        id=qid,
        question=f"question {qid}",
        lang="en",
        doc="policy",
        pages=(1,) if answerable else (),
        evidence=("at least fifteen characters",) if answerable else (),
        answer="15" if answerable else None,
        kind="lexical",
        answerable=answerable,
    )


class Replies:
    """A chat model whose reply depends on the question."""

    def __init__(self, replies: dict[str, str]) -> None:
        self.replies = replies

    def __call__(self, question: str, hits: Sequence[PageHit]) -> "Replies.Chat":
        return Replies.Chat(self.replies[question])

    class Chat:
        def __init__(self, reply: str) -> None:
            self.reply = reply

        def stream(self, system: str, user: str) -> Iterator[str]:
            yield self.reply


def _hits(question: str, k: int) -> list[PageHit]:
    return [PageHit(SHA, 1, "1", ("docs/policy.pdf",), 1.0, Chunk(1, SHA, 1, 0, TEXT))]


@pytest.fixture
def snapshot(tmp_path: Path) -> Snapshot:
    from askpdf.bm25 import BM25

    return Snapshot(0, (), {}, {}, BM25.build([]), texts={(SHA, 1): TEXT})


GOOD = 'Fifteen characters [1].\nQuotes:\n[1] "be at least fifteen characters long"'
FOLDED = 'Fifteen [1].\nQuotes:\n[1] "characters long for singlefactor use"'
BAD = 'Twenty [1]. Always [1].\nQuotes:\n[1] "be at least twenty characters long"'
SHORT = 'Fifteen [1].\nQuotes:\n[1] "fifteen characters long"'


def test_grounding_and_abstention_rates(snapshot: Snapshot) -> None:
    questions = [
        _q("a1", True),
        _q("a2", True),
        _q("a3", True),
        _q("a4", True),
        _q("u1", False),
        _q("u2", False),
    ]
    replies = {
        "question a1": GOOD,
        "question a2": FOLDED,
        "question a3": BAD,
        "question a4": "NOT_FOUND",
        "question u1": "NOT_FOUND",
        "question u2": SHORT,
    }
    seen: list[str] = []
    records = answers.run(
        questions, snapshot, _hits, Replies(replies), lambda d, t, qid: seen.append(qid)
    )
    assert seen == [q.id for q in questions]
    assert records["a1"]["citations"][0]["doc"] == "policy"
    summary = answers.summarise(records)
    g, a = summary["grounding"], summary["abstention"]
    assert g["n_answers"] == 3 and g["n_citations"] == 3  # answerable and answered only
    assert g["quote_verified_rate"] == pytest.approx(2 / 3, abs=1e-6)
    assert g["folding_only_share"] == 0.5
    assert g["supported_sentence_rate"] == pytest.approx(2 / 4, abs=1e-6)
    assert g["grounded_rate"] == pytest.approx(2 / 3, abs=1e-6)
    assert g["unverified_too_short"] == 0 and g["unverified_no_quote"] == 0
    assert a["false_answer_rate"] == 0.5
    assert a["not_found_precision"] == 0.5
    assert a["not_found_recall"] == 0.5
    assert a["over_abstention_rate"] == 0.25
    low, high = a["over_abstention_interval_95"]
    assert 0 <= low <= 0.25 <= high <= 1


def test_empty_groups_give_no_rate_rather_than_a_crash(snapshot: Snapshot) -> None:
    records = answers.run([_q("u1", False)], snapshot, _hits, Replies({"question u1": SHORT}))
    summary = answers.summarise(records)
    assert summary["grounding"]["quote_verified_rate"] is None
    assert summary["abstention"]["over_abstention_rate"] is None
    assert summary["abstention"]["not_found_precision"] is None
    assert answers.summarise({})["seconds_median"] is None


def test_a_short_quote_is_counted_as_its_own_cause(snapshot: Snapshot) -> None:
    records = answers.run([_q("a1", True)], snapshot, _hits, Replies({"question a1": SHORT}))
    assert answers.summarise(records)["grounding"]["unverified_too_short"] == 1


ROOT = Path(__file__).resolve().parents[1]


def test_the_stage_needs_ollama(capsys: pytest.CaptureFixture[str]) -> None:
    from askpdf.cli import main

    assert main(["eval", "answers", "--eval-dir", str(ROOT / "eval")]) == 1
    assert "needs Ollama running" in capsys.readouterr().err


def test_the_stage_end_to_end_with_a_fake_model(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    import json

    from askpdf.cli import main
    from tests.fake_models import fake_model_server

    eval_dir = tmp_path / "eval"
    (eval_dir / "questions").mkdir(parents=True)
    shutil.copytree(ROOT / "eval" / "corpus", eval_dir / "corpus")  # no symlinks on Windows
    lines = (ROOT / "eval" / "questions" / "dev.jsonl").read_text(encoding="utf-8").splitlines()
    picked = [next(x for x in lines if '"answerable": true' in x)]
    picked.append(next(x for x in lines if '"answerable": false' in x))
    (eval_dir / "questions" / "dev.jsonl").write_text("\n".join(picked) + "\n", encoding="utf-8")
    with fake_model_server() as (url, models):
        monkeypatch.setenv("ASKPDF_OLLAMA_URL", url)
        models.chat_reply = ["NOT_FOUND"]
        out = tmp_path / "answers.json"
        args = ["eval", "answers", "--eval-dir", str(eval_dir), "--json", str(out)]
        assert main([*args, "--machine", "test box"]) == 0
    result = json.loads(out.read_text(encoding="utf-8"))
    assert result["stage"] == "answers" and result["k"] == 3 and result["model"] == "llama3.2:3b"
    # When, where and from which commit, so the README needs no dates of its own.
    assert result["machine"] == "test box" and "git_commit" in result
    assert result["date"] == datetime.date.today().isoformat()
    assert result["abstention"]["not_found_recall"] == 1.0
    assert result["abstention"]["over_abstention_rate"] == 1.0
    assert "answering 2/2" in capsys.readouterr().err


def test_rescoring_rechecks_saved_replies_without_the_model(snapshot: Snapshot) -> None:
    questions = [_q("a1", True), _q("u1", False)]
    records = answers.run(
        questions, snapshot, _hits, Replies({"question a1": GOOD, "question u1": "NOT_FOUND"})
    )
    records["a1"]["citations"] = []  # as if an older checker had found nothing
    rescored = answers.rescore(records, questions, snapshot, _hits)
    assert rescored["a1"]["citations"][0]["verified"]
    assert rescored["a1"]["seconds"] == records["a1"]["seconds"]
    assert rescored["u1"]["not_found"]
    empty = answers.rescore(records, questions, snapshot, lambda q, k: [])
    assert empty["a1"]["not_found"]  # no pages: answered NOT_FOUND without the model
