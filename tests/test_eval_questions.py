"""The in-house question sets (ADR-0011)."""

import hashlib
import importlib.util
import json
from collections import Counter
from pathlib import Path

import pytest

from askpdf.chunking import ChunkParams
from askpdf.evaluate.questions import Question, evidence_problems, load
from askpdf.extract.pypdf_ext import PypdfExtractor
from askpdf.indexer import build_records
from askpdf.store import PageRecord

EVAL = Path(__file__).resolve().parent.parent / "eval"
QUESTIONS = EVAL / "questions"
in_repo = pytest.mark.skipif(not QUESTIONS.is_dir(), reason="eval/ is not in the sdist")


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


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"kind": "trivia"}, "unknown kind"),
        ({"pages": []}, "answerable questions need"),
        ({"answer": None}, "reference answer"),
        ({"evidence": ["1.6"]}, "shorter than 15"),
        ({"kind": "multi-page"}, "at least two pages"),
        ({"answerable": False}, "answerable questions need"),
    ],
)
def test_malformed_questions_are_refused(overrides: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        Question.from_json(_row(**overrides))


def test_an_unanswerable_question_has_no_pages_or_evidence() -> None:
    q = Question.from_json(_row(pages=[], evidence=[], answer=None, answerable=False))
    assert q.gold == frozenset()


def test_evidence_must_be_on_the_cited_pages() -> None:
    pages = {"d": [PageRecord(1, "1", "Here the timeout\nis 15 minutes.", (), False)]}
    q = Question.from_json(_row())
    assert evidence_problems(q, pages) == []  # line breaks do not matter
    moved = Question.from_json(_row(pages=[1], evidence=["something else entirely"]))
    assert "quote not on its pages" in evidence_problems(moved, pages)[0]
    two = Question.from_json(_row(pages=[1, 1], kind="multi-page"))
    assert evidence_problems(two, {"d": pages["d"]}) == []
    out_of_range = Question.from_json(_row(pages=[3]))
    assert "out of range" in evidence_problems(out_of_range, pages)[0]
    assert "no document" in evidence_problems(q, {})[0]


def test_every_cited_page_needs_a_quote() -> None:
    pages = {
        "d": [
            PageRecord(1, "1", "the timeout is 15 minutes", (), False),
            PageRecord(2, "2", "unrelated", (), False),
        ]
    }
    q = Question.from_json(_row(pages=[1, 2], kind="multi-page"))
    assert evidence_problems(q, pages) == ["q1: page 2 holds none of the quotes"]


@pytest.fixture(scope="module")
def corpus_pages() -> dict[str, list[PageRecord]]:
    return {
        pdf.stem: build_records(PypdfExtractor().extract(pdf), ChunkParams())[0]
        for pdf in sorted((EVAL / "corpus").glob("*.pdf"))
    }


@in_repo
@pytest.mark.parametrize("split", ["dev", "test"])
def test_every_quote_is_on_its_page_as_askpdf_reads_it(
    split: str, corpus_pages: dict[str, list[PageRecord]]
) -> None:
    problems = [
        p for q in load(QUESTIONS / f"{split}.jsonl") for p in evidence_problems(q, corpus_pages)
    ]
    assert problems == []


@in_repo
def test_the_test_split_is_frozen() -> None:
    # Changing a test question is allowed only as a visible, reviewed act: update the
    # hash in the same PR (ADR-0007).
    digest = hashlib.sha256((QUESTIONS / "test.jsonl").read_bytes()).hexdigest()
    assert digest == (QUESTIONS / "test.sha256").read_text().strip()


@in_repo
def test_the_files_are_what_the_source_script_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = importlib.util.spec_from_file_location(
        "questions_source", EVAL / "tools" / "questions_source.py"
    )
    assert spec is not None and spec.loader is not None
    source = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(source)
    for split, rows in source.split(source.records()).items():
        written = (QUESTIONS / f"{split}.jsonl").read_text(encoding="utf-8")
        assert written == "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n"


@in_repo
def test_the_mix_matches_the_design() -> None:
    dev, test = load(QUESTIONS / "dev.jsonl"), load(QUESTIONS / "test.jsonl")
    both = dev + test
    assert len({q.id for q in both}) == len(both)  # no question in both splits
    answerable = [q for q in both if q.answerable]
    assert Counter(q.kind for q in answerable)["paraphrase"] * 2 >= len(answerable)
    unanswerable = len(both) - len(answerable)
    assert 0.10 <= unanswerable / len(both) <= 0.20  # ADR-0011: about 16%
    # Stratified: every document appears in both splits, with similar sizes.
    assert {q.doc for q in dev} == {q.doc for q in test}
    assert abs(len(dev) - len(test)) <= len({(q.doc, q.kind, q.answerable) for q in both})
