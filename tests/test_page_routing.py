"""Which questions skip search and are read from a named page (ADR-0012).

The answers eval never takes this path, so these checks are its only measure: questions
that name a page must route there, and no ordinary dev question may. The snapshot holds
the eval corpus's file names with every Roman label up to xx printed, the case most
likely to capture a word by mistake. Only dev questions are read: the test split is
frozen.
"""

import json
from collections.abc import Iterator
from pathlib import Path

import pytest

from askpdf.pagelookup import find_page
from askpdf.snapshot import Snapshot
from askpdf.store import ChunkRecord, FileRow, PageRecord, Store

EVAL = Path(__file__).parent.parent / "eval"
CORPUS = sorted(p.name for p in (EVAL / "corpus").glob("*.pdf"))
ONES = ["", "i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix"]
ROMAN = [tens + one for tens in ("", "x", "xx") for one in ONES][1:21]  # i to xx


def _dev_questions() -> list[str]:
    questions = [
        json.loads(line)["question"]
        for line in (EVAL / "questions" / "dev.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    for line in (EVAL / "conversations" / "dev.jsonl").read_text(encoding="utf-8").splitlines():
        questions += [turn["question"] for turn in json.loads(line)["turns"]]
    return questions


@pytest.fixture(scope="module")
def snapshot(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Snapshot]:
    with Store(tmp_path_factory.mktemp("routing") / "i.db") as store:
        for number, name in enumerate(CORPUS):
            labels = [*ROMAN, *map(str, range(1, 41))]
            store.put_document(
                FileRow(name, f"{number:064x}", 1, 1, "indexed", None),
                [PageRecord(n, label, "text", (), False) for n, label in enumerate(labels, 1)],
                [ChunkRecord(n, 0, 0, 1, "text") for n in range(1, len(labels) + 1)],
            )
        yield Snapshot.load(store)


def test_the_corpus_and_dev_questions_are_there() -> None:
    assert len(CORPUS) >= 4 and len(_dev_questions()) >= 58


@pytest.mark.parametrize("question", _dev_questions())
def test_no_dev_question_skips_search(question: str, snapshot: Snapshot) -> None:
    assert find_page(question, snapshot) is None


@pytest.mark.parametrize(
    "question",
    [
        "what is on page 5 of nist-sp-800-63b-4.pdf?",
        "What does page iv of the naca memorandum say? naca-rm-l56l14a",
        "p. 12 of qasper-2105.03011v1",
        "PDF page 3 of nist-sp-800-63b-4",
        "what is the second line on page 2 of nist-sp-800-63b-4?",
        "summarise nist-sp-800-63b-4.pdf",
    ],
)
def test_questions_naming_a_page_skip_search(question: str, snapshot: Snapshot) -> None:
    assert find_page(question, snapshot) is not None
