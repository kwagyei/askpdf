"""Stage (a): transcripts, CER and header/footer scoring."""

from collections import Counter
from pathlib import Path

import pytest

from askpdf.evaluate.extraction import (
    Transcript,
    boilerplate_words,
    evaluate_pages,
    load_transcripts,
    normalise,
    scanned_detection,
    summarise,
    word_detection,
)
from askpdf.store import PageRecord
from tests.eval_pins import require_pinned_extractor

REPO_TRANSCRIPTS = Path(__file__).resolve().parent.parent / "eval" / "transcripts"


def _write(folder: Path, name: str, text: str) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    path.write_text(text, encoding="utf-8")
    return path


def test_a_transcript_separates_labels_from_text(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "doc-a-p012.txt",
        "[header] Report title\n[margin] stamp\nBody line.\n[footer] 12\n",
    )
    t = Transcript.read(path)
    assert (t.doc, t.page) == ("doc-a", 12)
    assert t.text == "Report title\nstamp\nBody line.\n12"
    assert t.boilerplate == ("Report title", "12")  # margin text is not a header


def test_a_badly_named_transcript_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="pNNN"):
        Transcript.read(_write(tmp_path, "notes.txt", "x"))


def test_normalise_ignores_layout_whitespace() -> None:
    assert normalise(" a\tb\n\nc  ") == "a b c"


def test_boilerplate_is_scored_on_words() -> None:
    page = PageRecord(1, "1", "NIST SP 800-63B-4 July 2025\nBody text\n35", (0, 2), False)
    assert boilerplate_words(page) == Counter(["NIST", "SP", "800-63B-4", "July", "2025", "35"])
    gold = Counter(["NIST", "SP", "800-63B-4", "July", "2025", "Digital", "35"])
    d = word_detection(boilerplate_words(page), gold)
    assert (d.true_positives, d.false_positives, d.false_negatives) == (6, 0, 1)


def test_scanned_detection_over_all_pages() -> None:
    pages = {
        "scan": [PageRecord(1, "1", "", (), True), PageRecord(2, "2", "", (), True)],
        "text": [PageRecord(1, "1", "words", (), False)],
    }
    d = scanned_detection(pages, gold={("scan", 1), ("scan", 2)})
    assert d.precision == 1.0 and d.recall == 1.0


def test_cer_and_summary() -> None:
    pytest.importorskip("jiwer")  # the eval group
    from askpdf.evaluate.extraction import character_error_rate

    assert character_error_rate("abcd", "abcd") == 0.0
    assert character_error_rate("abcd", "abxd") == 0.25
    assert character_error_rate("a  b", "a b") == 0.0
    assert character_error_rate("", "") == 0.0 and character_error_rate("", "x") == 1.0
    transcripts = [Transcript("d", 1, "Head\nhello world", ("Head",))]
    pages = {"d": [PageRecord(1, "1", "Head\nhello wurld", (0,), False)]}
    summary = summarise(evaluate_pages(transcripts, pages), scanned_detection(pages, set()))
    assert summary["pages"] == [{"doc": "d", "page": 1, "cer": round(1 / 16, 6)}]
    assert summary["boilerplate"] == {"precision": 1.0, "recall": 1.0}
    assert summary["scanned"] == {"precision": None, "recall": None}


@pytest.mark.skipif(not REPO_TRANSCRIPTS.is_dir(), reason="eval/ is not in the sdist")
def test_the_committed_transcripts_parse_and_name_corpus_pages() -> None:
    corpus = REPO_TRANSCRIPTS.parent / "corpus"
    transcripts = load_transcripts(REPO_TRANSCRIPTS)
    assert len(transcripts) >= 8
    for t in transcripts:
        assert (corpus / f"{t.doc}.pdf").is_file(), t.doc
        assert t.text.strip()


def test_eval_without_data_says_where_to_run_it(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from askpdf.cli import main

    assert main(["eval", "extraction", "--eval-dir", str(tmp_path)]) == 1
    assert "no evaluation data" in capsys.readouterr().err


def test_eval_without_the_eval_group_says_how_to_install_it(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import sys

    from askpdf.cli import main

    monkeypatch.setitem(sys.modules, "jiwer", None)  # makes `import jiwer` fail
    assert main(["eval", "extraction", "--eval-dir", str(REPO_TRANSCRIPTS.parent)]) == 1
    assert "pip install --group eval" in capsys.readouterr().err


@pytest.mark.skipif(not REPO_TRANSCRIPTS.is_dir(), reason="eval/ is not in the sdist")
def test_eval_extraction_reproduces_the_committed_results(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # The CI gate runs the same comparison; this makes a local `pytest` catch it too.
    require_pinned_extractor()
    pytest.importorskip("jiwer")
    from askpdf.cli import main

    out = tmp_path / "extraction.json"
    assert (
        main(["eval", "extraction", "--eval-dir", str(REPO_TRANSCRIPTS.parent), "--json", str(out)])
        == 0
    )
    committed = REPO_TRANSCRIPTS.parent / "results" / "extraction.json"
    assert out.read_text(encoding="utf-8") == committed.read_text(encoding="utf-8")
    assert "mean CER" in capsys.readouterr().out
