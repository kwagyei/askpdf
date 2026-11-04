"""Tester additions for v0.1.1: page routing edge cases, quote-line parsing, banner."""

import time
from pathlib import Path

import pytest

from askpdf.answer import parse
from askpdf.pagelookup import find_page
from askpdf.retrieve import PageHit
from askpdf.snapshot import Chunk, Snapshot
from askpdf.store import ChunkRecord, FileRow, PageRecord, Store


def _snapshot(tmp_path: Path, labels: dict[str, list[str]]) -> Snapshot:
    with Store(tmp_path / "i.db") as store:
        for number, (name, printed) in enumerate(labels.items()):
            store.put_document(
                FileRow(name, f"{number:064x}", 1, 1, "indexed", None),
                [PageRecord(n, lab, f"text {n}", (), False) for n, lab in enumerate(printed, 1)],
                [ChunkRecord(n, 0, 0, 1, f"text {n}") for n in range(1, len(printed) + 1)],
            )
        return Snapshot.load(store)


@pytest.fixture
def snap(tmp_path: Path) -> Snapshot:
    return _snapshot(tmp_path, {"report.pdf": ["i", "ii", "iv", "1", "2", "3", "4", "5", "6", "7"]})


@pytest.mark.parametrize(
    "question",
    [
        "on which page did the report define MFA?",
        "page mid-way through, what does the report say?",
        "what happens on page ill-defined cases in the report?",
        "page mix of passwords in the report?",
        "pg dim lights",
    ],
)
def test_words_after_page_are_searched(question: str, snap: Snapshot) -> None:
    assert find_page(question, snap) is None


@pytest.mark.parametrize(
    ("question", "page"),
    [
        ("what is on page iv of the report?", 3),
        ("what is on PAGE IV of the report?", 3),
        ("on page did, page 5 of the report", 8),
        ("page did it say? p. 6 of the report", 9),
        ("and page 6?", 9),
        ("PDF page 3 of the report", 3),
        ("pg. ii report", 2),
    ],
)
def test_real_page_after_a_word_still_routes(question: str, page: int, snap: Snapshot) -> None:
    found = find_page(question, snap, earlier=["tell me about report"])
    assert found is not None
    assert page in [h.page for h in found.hits]


def test_roman_not_printed_is_not_a_page(snap: Snapshot) -> None:
    assert find_page("what is on page ix of the report?", snap) is None


def test_hostile_replies_parse_in_linear_time() -> None:
    hits = [
        PageHit("0" * 64, 1, "1", ("a.pdf",), 1.0, Chunk(0, "0" * 64, 1, 0, "x")),
    ]
    texts = {("0" * 64, 1): "alpha beta"}
    times = []
    for n in (5_000, 20_000):
        raw = "Quotes:\n" + "\n".join(["- _ _ _"] * n) + "\n[1] alpha\n"
        start = time.perf_counter()
        answer = parse(raw, hits, texts)
        times.append(time.perf_counter() - start)
        assert answer.citations[0].quotes[0].quote == "alpha"
    assert times[1] < max(times[0] * 12, 0.5)


def test_nonstandard_printed_roman_label_still_routes(tmp_path: Path) -> None:
    """Front matter numbered iiii (or xxxx) is a printed label, so it is a page."""
    snap = _snapshot(tmp_path, {"report.pdf": ["i", "ii", "iii", "iiii", "1"]})
    found = find_page("what is on page iiii of the report?", snap)
    assert found is not None
    assert [h.page for h in found.hits] == [4]


@pytest.mark.parametrize(
    ("extra", "notes"),
    [
        ([], ["gpu-box:11434"]),
        (
            ["--api-url", "https://user:pw@api.example.com/v1"],
            ["api.example.com"],
        ),
        (
            ["--api-url", "https://api.example.com:8443/v1", "--retriever", "hybrid"],
            ["api.example.com:8443", "gpu-box:11434"],
        ),
        (["--allow-remote"], ["gpu-box:11434"]),
    ],
)
def test_serve_prints_one_note_per_remote_host(
    extra: list[str],
    notes: list[str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import uvicorn

    from askpdf.cli import main

    monkeypatch.setenv("ASKPDF_OLLAMA_URL", "http://gpu-box:11434")
    monkeypatch.setenv("ASKPDF_API_KEY", "sk-secret")
    monkeypatch.setenv("ASKPDF_ALLOW_REMOTE", "1")
    monkeypatch.setattr(uvicorn.Server, "run", lambda self, *a, **k: None)
    (tmp_path / "g").mkdir()
    argv = ["serve", str(tmp_path / "g"), "--index", str(tmp_path / "j.db"), "--no-browser"]
    argv += ["--model", "m", *extra]
    main(argv)
    err = capsys.readouterr().err
    lines = [ln for ln in err.splitlines() if ln.startswith("note:")]
    assert [ln.rsplit(" ", 1)[1] for ln in lines] == notes
    assert "pw" not in err and "sk-secret" not in err
    assert "Traceback" not in err


def test_banner_script_uses_text_content_only() -> None:
    js = (Path(__file__).parent.parent / "src/askpdf/web/static/app.js").read_text()
    assert "innerHTML" not in js and "insertAdjacentHTML" not in js
