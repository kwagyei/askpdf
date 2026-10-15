from pathlib import Path

import pytest

from askpdf.extract.base import ExtractionError, RawPage
from askpdf.extract.pypdf_ext import PypdfExtractor
from tests.pdfs import PageSpec, make_pdf, text_page


def test_pages_come_back_in_order_with_physical_numbers(tmp_path: Path) -> None:
    pdf = make_pdf(tmp_path / "a.pdf", [text_page("First page."), text_page("Second page.")])
    pages = PypdfExtractor().extract(pdf)
    assert [p.index for p in pages] == [1, 2]
    assert "First page." in pages[0].text
    assert "Second page." in pages[1].text


def test_printed_labels_are_kept_alongside_physical_numbers(tmp_path: Path) -> None:
    pdf = make_pdf(
        tmp_path / "a.pdf",
        [text_page("Preface"), text_page("Contents"), text_page("Chapter 1")],
        labels=(0, 1, "/r", 1),
    )
    pages = PypdfExtractor().extract(pdf)
    assert [(p.index, p.label) for p in pages] == [(1, "i"), (2, "ii"), (3, "1")]


def test_swedish_text_survives_extraction(tmp_path: Path) -> None:
    pdf = make_pdf(tmp_path / "sv.pdf", [text_page("Räksmörgås på Åland")])
    assert "Räksmörgås på Åland" in PypdfExtractor().extract(pdf)[0].text


def test_image_without_text_is_image_only(tmp_path: Path) -> None:
    pdf = make_pdf(tmp_path / "scan.pdf", [PageSpec(image=True), text_page("Real text here.")])
    pages = PypdfExtractor().extract(pdf)
    assert pages[0].has_image and pages[0].image_only
    assert not pages[1].has_image and not pages[1].image_only


def test_image_with_enough_text_is_not_image_only(tmp_path: Path) -> None:
    spec = PageSpec(lines=["A figure caption long enough to count."], image=True)
    page = PypdfExtractor().extract(make_pdf(tmp_path / "fig.pdf", [spec]))[0]
    assert page.has_image and not page.image_only


@pytest.mark.parametrize(
    ("text", "image_only"),
    [("", True), ("x" * 19, True), (" " * 50 + "x" * 19, True), ("x" * 20, False)],
)
def test_image_only_threshold(text: str, image_only: bool) -> None:
    assert RawPage(1, "1", text, has_image=True).image_only is image_only
    assert RawPage(1, "1", text, has_image=False).image_only is False


def test_blank_page_has_empty_text(tmp_path: Path) -> None:
    page = PypdfExtractor().extract(make_pdf(tmp_path / "blank.pdf", [PageSpec()]))[0]
    assert page.text.strip() == ""
    assert not page.image_only


def test_encrypted_with_empty_password_is_opened(tmp_path: Path) -> None:
    pdf = make_pdf(tmp_path / "e.pdf", [text_page("Open sesame.")], user_password="")
    assert "Open sesame." in PypdfExtractor().extract(pdf)[0].text


def test_encrypted_with_a_password_fails_with_a_reason(tmp_path: Path) -> None:
    pdf = make_pdf(tmp_path / "e.pdf", [text_page("Secret.")], user_password="hunter2")
    with pytest.raises(ExtractionError, match="password"):
        PypdfExtractor().extract(pdf)


@pytest.mark.parametrize(
    "content",
    [
        b"",
        b"not a pdf at all",
        b"%PDF-1.7\n1 0 obj << /Type /Catalog",
        b"%PDF-1.4\n" + b"\x00" * 64,
    ],
)
def test_corrupt_files_raise_extraction_error(tmp_path: Path, content: bytes) -> None:
    bad = tmp_path / "bad.pdf"
    bad.write_bytes(content)
    with pytest.raises(ExtractionError):
        PypdfExtractor().extract(bad)


def test_truncated_pdf_raises_extraction_error(tmp_path: Path) -> None:
    pdf = make_pdf(tmp_path / "t.pdf", [text_page("Some text.")] * 3)
    pdf.write_bytes(pdf.read_bytes()[: len(pdf.read_bytes()) // 3])
    with pytest.raises(ExtractionError):
        PypdfExtractor().extract(pdf)


def test_missing_file_raises_extraction_error(tmp_path: Path) -> None:
    with pytest.raises(ExtractionError, match="cannot open"):
        PypdfExtractor().extract(tmp_path / "nope.pdf")


def test_extractor_identifies_itself() -> None:
    import pypdf

    assert (PypdfExtractor.name, PypdfExtractor.version) == ("pypdf", pypdf.__version__)
