"""Stage (f): follow-up questions searched raw, concatenated and rewritten by hand."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from askpdf.chunking import ChunkParams
from askpdf.cli import main
from askpdf.evaluate.conversation import Conversation, evaluate, load
from askpdf.evaluate.questions import evidence_problems
from askpdf.extract.pypdf_ext import PypdfExtractor
from askpdf.indexer import build_records
from tests.eval_pins import require_pinned_extractor

EVAL = Path(__file__).resolve().parent.parent / "eval"
CONVERSATIONS = EVAL / "conversations"
in_repo = pytest.mark.skipif(not CONVERSATIONS.is_dir(), reason="eval/ is not in the sdist")


def _turn(
    question: str, rewrite: str | None = None, page: int = 1, switch: bool = False
) -> dict[str, object]:
    return {
        "question": question,
        "rewrite": rewrite or question,
        "doc": "d",
        "pages": [page],
        "evidence": ["a long enough quote"],
        "switch": switch,
    }


def _conv(*turns: dict[str, object]) -> Conversation:
    return Conversation.from_json({"id": "c", "lang": "en", "turns": list(turns)})


def test_the_three_queries_for_a_follow_up() -> None:
    conv = _conv(_turn("first question"), _turn("and then?", "second question standalone"))
    assert conv.queries(1) == {
        "raw": "and then?",
        "concat": "first question and then?",
        "gold": "second question standalone",
    }


@pytest.mark.parametrize(
    ("turns", "message"),
    [
        ([_turn("only one")], "needs a follow-up"),
        ([_turn("first", "rewritten"), _turn("next")], "already be standalone"),
        ([_turn("first", switch=True), _turn("next")], "already be standalone"),
    ],
)
def test_malformed_conversations_are_refused(turns: list[dict[str, object]], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _conv(*turns)


def test_follow_ups_are_scored_per_condition_and_for_topic_switches() -> None:
    conv = _conv(
        _turn("alpha"),
        _turn("beta?", "beta standalone", page=2),
        _turn("gamma?", "gamma standalone", page=3, switch=True),
    )
    runs = {
        "beta?": [("d", 9)],
        "alpha beta?": [("d", 2)],
        "beta standalone": [("d", 2)],
        "gamma?": [("d", 3)],
        "alpha beta? gamma?": [("d", 9), ("d", 3)],
        "gamma standalone": [("d", 3)],
    }
    result = evaluate([conv], lambda q: runs[q])
    assert result["n_follow_ups"] == 2 and result["n_topic_switches"] == 1
    assert result["overall"]["raw"]["hit@1"] == 0.5
    assert result["overall"]["concat"]["hit@1"] == 0.5
    assert result["overall"]["gold"]["hit@1"] == 1.0
    assert result["topic_switches"]["concat"]["mrr@10"] == 0.5
    assert set(result["per_turn"]["raw"]) == {"c#2", "c#3"}


@in_repo
def test_the_conversations_are_what_the_source_script_writes() -> None:
    spec = importlib.util.spec_from_file_location(
        "conversations_source", EVAL / "tools" / "conversations_source.py"
    )
    assert spec is not None and spec.loader is not None
    source = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(source)
    for split, rows in source.split().items():
        written = (CONVERSATIONS / f"{split}.jsonl").read_text(encoding="utf-8")
        assert written == "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n"


@in_repo
def test_every_turn_cites_evidence_on_its_pages() -> None:
    pages = {
        pdf.stem: build_records(PypdfExtractor().extract(pdf), ChunkParams())[0]
        for pdf in sorted((EVAL / "corpus").glob("*.pdf"))
    }
    convs = load(CONVERSATIONS / "dev.jsonl") + load(CONVERSATIONS / "test.jsonl")
    assert len(convs) >= 12
    assert sum(t.switch for c in convs for t in c.turns) >= 2
    problems = [
        p
        for c in convs
        for i, t in enumerate(c.turns)
        for p in evidence_problems(t.as_question(f"{c.id}#{i + 1}"), pages)
    ]
    assert problems == []


@in_repo
def test_the_test_conversations_are_frozen() -> None:
    digest = hashlib.sha256((CONVERSATIONS / "test.jsonl").read_bytes()).hexdigest()
    assert digest == (CONVERSATIONS / "test.sha256").read_text().strip()


@in_repo
def test_conversations_on_dev_reproduce_the_committed_results(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    require_pinned_extractor()
    out = tmp_path / "conversation.json"
    assert main(["eval", "conversation", "--eval-dir", str(EVAL), "--json", str(out)]) == 0
    committed = EVAL / "results" / "dev-conversation-bm25.json"
    assert out.read_text(encoding="utf-8") == committed.read_text(encoding="utf-8")
    assert "follow-ups" in capsys.readouterr().out


def test_missing_conversations_are_reported(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "corpus").mkdir()
    assert main(["eval", "conversation", "--eval-dir", str(tmp_path)]) == 1
    assert "no conversations" in capsys.readouterr().err
