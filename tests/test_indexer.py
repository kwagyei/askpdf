import os
import shutil
from collections.abc import Iterator
from pathlib import Path

import pytest

from askpdf.chunking import ChunkParams
from askpdf.extract.base import ExtractionError, Extractor, RawPage
from askpdf.extract.pypdf_ext import PypdfExtractor
from askpdf.indexer import IndexReport, build_records, index_folder, pins, scan
from askpdf.store import Store
from tests.pdfs import PageSpec, make_pdf, text_page

PARAMS = ChunkParams(target=20, overlap=5, snap=3)


@pytest.fixture
def folder(tmp_path: Path) -> Path:
    root = tmp_path / "pdfs"
    make_pdf(root / "a.pdf", [text_page("Alpha page one."), text_page("Alpha page two.")])
    make_pdf(root / "sub" / "b.pdf", [text_page("Bravo only page.")])
    return root


@pytest.fixture
def store(tmp_path: Path) -> Iterator[Store]:
    with Store(tmp_path / "index" / "i.db") as s:
        yield s


def run(folder: Path, store: Store, extractor: Extractor | None = None) -> IndexReport:
    return index_folder(folder, store, extractor or PypdfExtractor(), PARAMS)


def test_first_run_indexes_every_pdf(folder: Path, store: Store) -> None:
    report = run(folder, store)
    assert sorted(report.added) == ["a.pdf", "sub/b.pdf"]
    docs = {d.paths: d for d in store.documents()}
    assert docs[("a.pdf",)].n_pages == 2
    assert docs[("sub/b.pdf",)].n_chunks == 1


def test_second_run_on_an_unchanged_folder_is_a_no_op(folder: Path, store: Store) -> None:
    run(folder, store)
    generation = store.generation
    report = run(folder, store)
    assert not report.changed
    assert report.unchanged == 2
    assert store.generation == generation


class CountingExtractor(PypdfExtractor):
    def __init__(self) -> None:
        self.calls: list[Path] = []

    def extract(self, path: Path) -> list[RawPage]:
        self.calls.append(path)
        return super().extract(path)


def test_rename_is_recorded_without_re_extracting(folder: Path, store: Store) -> None:
    run(folder, store)
    (folder / "a.pdf").rename(folder / "renamed.pdf")
    extractor = CountingExtractor()
    report = run(folder, store, extractor)
    assert report.reused == ["renamed.pdf"]
    assert report.removed == ["a.pdf"]
    assert extractor.calls == []
    assert len(store.documents()) == 2


def test_copy_shares_the_document(folder: Path, store: Store) -> None:
    run(folder, store)
    shutil.copy(folder / "a.pdf", folder / "copy.pdf")
    report = run(folder, store, CountingExtractor())
    assert report.reused == ["copy.pdf"]
    [doc] = [d for d in store.documents() if "a.pdf" in d.paths]
    assert doc.paths == ("a.pdf", "copy.pdf")


def test_deleted_file_leaves_the_index(folder: Path, store: Store) -> None:
    run(folder, store)
    (folder / "sub" / "b.pdf").unlink()
    report = run(folder, store)
    assert report.removed == ["sub/b.pdf"]
    assert [d.paths for d in store.documents()] == [("a.pdf",)]


def test_changed_file_is_re_extracted(folder: Path, store: Store) -> None:
    run(folder, store)
    make_pdf(folder / "a.pdf", [text_page("Completely new content.")])
    report = run(folder, store)
    assert report.added == ["a.pdf"]
    [doc] = [d for d in store.documents() if d.paths == ("a.pdf",)]
    assert store.pages(doc.sha256)[0].text == "Completely new content."


def test_touched_but_identical_file_is_not_re_extracted(folder: Path, store: Store) -> None:
    run(folder, store)
    st = (folder / "a.pdf").stat()
    os.utime(folder / "a.pdf", ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))
    extractor = CountingExtractor()
    report = run(folder, store, extractor)
    assert extractor.calls == [] and not report.added
    # The new timestamp is remembered, so the next run does not even hash the file.
    assert store.files()["a.pdf"].mtime_ns == st.st_mtime_ns + 10**9


def test_one_bad_file_does_not_stop_the_rest(folder: Path, store: Store) -> None:
    (folder / "broken.pdf").write_bytes(b"%PDF-1.4 garbage")
    report = run(folder, store)
    assert sorted(report.added) == ["a.pdf", "sub/b.pdf"]
    assert "broken.pdf" in report.failed
    row = store.files()["broken.pdf"]
    assert row.status == "failed" and row.error


def test_failed_file_is_not_retried_until_it_changes(folder: Path, store: Store) -> None:
    (folder / "broken.pdf").write_bytes(b"%PDF-1.4 garbage")
    run(folder, store)
    extractor = CountingExtractor()
    report = run(folder, store, extractor)
    assert extractor.calls == [] and not report.failed
    make_pdf(folder / "broken.pdf", [text_page("Fixed now.")])
    report = run(folder, store)
    assert report.added == ["broken.pdf"]


def test_password_protected_file_is_reported(folder: Path, store: Store) -> None:
    make_pdf(folder / "locked.pdf", [text_page("Secret")], user_password="pw")
    report = run(folder, store)
    assert "password" in report.failed["locked.pdf"]


def test_changed_settings_rebuild_the_index(folder: Path, store: Store) -> None:
    run(folder, store)
    report = index_folder(
        folder, store, PypdfExtractor(), ChunkParams(target=30, overlap=5, snap=3)
    )
    assert report.rebuilt
    assert sorted(report.added) == ["a.pdf", "sub/b.pdf"]


def test_pins_name_extractor_and_chunking() -> None:
    p = pins(PypdfExtractor(), PARAMS)
    assert p["extractor"]["name"] == "pypdf"
    assert p["chunking"] == {"target": 20, "overlap": 5, "snap": 3}


def test_progress_callback_sees_each_file_it_reads(folder: Path, store: Store) -> None:
    seen: list[str] = []
    index_folder(folder, store, PypdfExtractor(), PARAMS, on_file=seen.append)
    assert seen == ["a.pdf", "sub/b.pdf"]
    seen.clear()
    index_folder(folder, store, PypdfExtractor(), PARAMS, on_file=seen.append)
    assert seen == []


def test_scan_skips_hidden_files_folders_and_non_pdfs(folder: Path) -> None:
    make_pdf(folder / ".hidden.pdf", [text_page("x")])
    make_pdf(folder / ".git" / "c.pdf", [text_page("x")])
    (folder / ".a.pdf.icloud").write_bytes(b"placeholder")
    (folder / "notes.txt").write_text("x")
    make_pdf(folder / "UPPER.PDF", [text_page("x")])
    assert [rel for rel, _ in scan(folder)] == ["UPPER.PDF", "a.pdf", "sub/b.pdf"]


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="no symlinks")
def test_scan_ignores_symlinks_that_leave_the_folder(folder: Path, tmp_path: Path) -> None:
    outside = make_pdf(tmp_path / "outside" / "secret.pdf", [text_page("x")])
    try:
        (folder / "link.pdf").symlink_to(outside)
        (folder / "linkdir").symlink_to(outside.parent, target_is_directory=True)
        (folder / "inside-link.pdf").symlink_to(folder / "a.pdf")
    except OSError:
        pytest.skip("symlinks not permitted here")
    rels = [rel for rel, _ in scan(folder)]
    assert "link.pdf" not in rels
    assert not any(r.startswith("linkdir") for r in rels)
    assert "inside-link.pdf" in rels


def test_a_single_pdf_can_be_indexed(folder: Path, store: Store) -> None:
    report = run(folder / "a.pdf", store)
    assert report.added == ["a.pdf"]


def test_a_single_non_pdf_yields_nothing(tmp_path: Path) -> None:
    (tmp_path / "x.txt").write_text("x")
    assert list(scan(tmp_path / "x.txt")) == []


class FlakyExtractor(PypdfExtractor):
    def extract(self, path: Path) -> list[RawPage]:
        raise ExtractionError("boom")


def test_extractor_errors_are_recorded(folder: Path, store: Store) -> None:
    report = run(folder, store, FlakyExtractor())
    assert report.failed == {"a.pdf": "boom", "sub/b.pdf": "boom"}


def test_records_keep_labels_boilerplate_and_image_flags() -> None:
    raw = [
        RawPage(i, f"p{i}", f"Running header\nbody {w} words here\nFooter {i}", False)
        for i, w in enumerate(["alpha", "bravo", "charlie", "delta"], start=1)
    ] + [RawPage(5, "p5", "", True)]
    pages, chunks = build_records(raw, PARAMS)
    assert [p.label for p in pages] == ["p1", "p2", "p3", "p4", "p5"]
    assert list(pages[0].boilerplate) == [0, 2]
    assert pages[4].image_only and pages[4].text == ""
    # Header lines stay in the chunk text until dev-split results say otherwise.
    assert chunks[0].text.startswith("Running header")
    assert {c.page for c in chunks} == {1, 2, 3, 4}


def test_whole_pipeline_on_a_scanned_document(tmp_path: Path, store: Store) -> None:
    make_pdf(tmp_path / "f" / "scan.pdf", [PageSpec(image=True)] * 3 + [text_page("Index")])
    run(tmp_path / "f", store)
    [doc] = store.documents()
    assert doc.n_image_only == 3 and doc.looks_scanned
