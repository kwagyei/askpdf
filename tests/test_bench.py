"""The latency harness (design §11, stage g). A fake clock makes every timing exact."""

import json
from collections.abc import Iterator, Sequence
from pathlib import Path

import pytest

from askpdf.bench import percentile, spread, time_answers, time_indexing, time_search
from askpdf.cli import main
from askpdf.evaluate.retrieval import load_corpus_snapshot
from askpdf.retrieve import PageHit
from askpdf.snapshot import Snapshot
from tests.pdfs import make_pdf, text_page

# One real question: the bench reads the dev set's format.
DEV_LINE = (Path("eval") / "questions" / "dev.jsonl").read_text(encoding="utf-8").split("\n")[0]
DEV_LINE += "\n"


class Clock:
    """Advances by ``step`` seconds every time it is read."""

    def __init__(self, step: float = 1.0) -> None:
        self.now, self.step = 0.0, step

    def __call__(self) -> float:
        self.now += self.step
        return self.now


class SlowChat:
    """Streams its pieces, moving the clock on before each one."""

    def __init__(self, clock: Clock, *pieces: str, think: float = 10.0) -> None:
        self.clock, self.pieces, self.think = clock, pieces, think

    def stream(self, system: str, user: str) -> Iterator[str]:
        self.clock.now += self.think
        for piece in self.pieces:
            self.clock.now += 1
            yield piece


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    folder = tmp_path / "corpus"
    make_pdf(folder / "a.pdf", [text_page("Passwords are hashed with a salt.")] * 3)
    make_pdf(folder / "b.pdf", [text_page("The wing was tested in a wind tunnel.")])
    return folder


@pytest.fixture
def snapshot(corpus: Path) -> Snapshot:
    return load_corpus_snapshot(corpus)


@pytest.mark.parametrize(
    ("values", "share", "expected"),
    [
        ([5.0], 0.5, 5.0),
        ([5.0], 0.95, 5.0),
        ([1.0, 2.0, 3.0, 4.0], 0.5, 2.0),
        ([4.0, 3.0, 2.0, 1.0], 0.95, 4.0),
        ([float(n) for n in range(1, 21)], 0.95, 19.0),  # nearest rank: ceil(19.0) = 19th
    ],
)
def test_percentile_is_a_measured_value(values: list[float], share: float, expected: float) -> None:
    assert percentile(values, share) == expected


def test_percentile_of_nothing_is_an_error() -> None:
    with pytest.raises(ValueError, match="no values"):
        percentile([], 0.5)


def test_spread_states_n() -> None:
    assert spread([1.0, 2.0, 3.0]) == {"n": 3, "p50": 2.0, "p95": 3.0}


def test_indexing_is_timed_per_document_and_run(corpus: Path) -> None:
    result = time_indexing(sorted(corpus.glob("*.pdf")), runs=2, clock=Clock(0.5))
    # each document: two clock reads, half a second apart
    assert result["seconds_per_document"] == {"n": 4, "p50": 0.5, "p95": 0.5}
    assert result["pages_per_second"] == 4.0  # (3 + 1) pages, 2 runs, / 2 s


def test_search_is_timed_in_milliseconds(snapshot: Snapshot) -> None:
    result = time_search(snapshot, ["passwords", "wing"], runs=3, k=3, clock=Clock(0.002))
    assert result == {"n": 6, "p50": 2.0, "p95": 2.0}


def test_the_first_answer_is_cold_and_the_rest_warm(snapshot: Snapshot) -> None:
    clock = Clock(0.0)  # time moves only while the model works
    asked: list[str] = []

    def chat_for(question: str, hits: Sequence[PageHit]) -> SlowChat:
        asked.append(question)
        cold = len(asked) == 1  # the first answer also loads the model: 20 s more
        return SlowChat(clock, "Salted [1].\n", 'Quotes:\n[1] "x"', think=30.0 if cold else 10.0)

    unloads: list[int] = []
    result = time_answers(
        snapshot,
        ["passwords?", "salt?", "hashed?"],
        3,
        chat_for,
        lambda: unloads.append(len(asked)),
        clock=clock,
    )
    assert unloads == [0]  # unloaded once, before the first answer
    assert result["cold"] == {"pages": 0.0, "first_token": 31.0, "total": 32.0}
    assert result["warm"]["first_token"] == {"n": 2, "p50": 11.0, "p95": 11.0}
    assert result["warm"]["total"] == {"n": 2, "p50": 12.0, "p95": 12.0}


def test_an_answer_without_pages_has_no_first_token(snapshot: Snapshot) -> None:
    def chat_for(question: str, hits: Sequence[PageHit]) -> SlowChat:
        assert hits == []
        return SlowChat(Clock(), "never streamed")  # no pages: the model is not asked

    result = time_answers(snapshot, ["zzzz", "qqqq"], 3, chat_for, lambda: None)
    assert result["cold"]["first_token"] is None
    assert result["warm"]["first_token"] is None
    assert result["warm"]["total"] is not None and result["warm"]["total"]["n"] == 1


def test_bench_command_without_the_model(
    corpus: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    eval_dir = tmp_path
    (eval_dir / "questions").mkdir()
    (eval_dir / "questions" / "dev.jsonl").write_text(DEV_LINE, encoding="utf-8")
    target = tmp_path / "bench.json"
    code = main(
        [
            *("bench", "--eval-dir", str(eval_dir), "--answers", "0", "--runs", "1"),
            *("--machine", "test box", "--json", str(target)),
        ]
    )
    assert code == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed == json.loads(target.read_text(encoding="utf-8"))
    assert printed["machine"] == "test box" and "answers_seconds" not in printed
    assert printed["indexing"]["documents"] == 2 and printed["search_ms"]["n"] == 1


def test_bench_needs_the_evaluation_data(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["bench", "--eval-dir", str(tmp_path), "--answers", "0"]) == 1
    assert "no corpus or dev questions" in capsys.readouterr().err


def test_bench_says_when_ollama_is_not_running(
    corpus: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "questions").mkdir()
    (tmp_path / "questions" / "dev.jsonl").write_text(DEV_LINE, encoding="utf-8")
    # conftest points Ollama at a closed port
    assert main(["bench", "--eval-dir", str(tmp_path), "--runs", "1", "--answers", "2"]) == 1
    assert "answers need Ollama running" in capsys.readouterr().err
