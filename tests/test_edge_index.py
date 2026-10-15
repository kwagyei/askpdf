"""Edge cases for scanning, the store and the CLI (tester, PR 2)."""

import os
import sqlite3
import sys
import unicodedata
from collections.abc import Iterator
from pathlib import Path

import pytest

from askpdf.chunking import ChunkParams
from askpdf.cli import main
from askpdf.extract.base import RawPage
from askpdf.extract.pypdf_ext import PypdfExtractor
from askpdf.indexer import IndexReport, index_folder, scan
from askpdf.store import Store, default_index_path
from tests.pdfs import make_pdf, text_page
from tests.test_edge_extract_clean import make_cmap_pdf, make_zero_page_pdf

PARAMS = ChunkParams(target=20, overlap=5, snap=3)
IS_ROOT = hasattr(os, "geteuid") and os.geteuid() == 0


@pytest.fixture
def store(tmp_path: Path) -> Iterator[Store]:
    with Store(tmp_path / "index" / "i.db") as s:
        yield s


def run(folder: Path, store: Store) -> IndexReport:
    return index_folder(folder, store, PypdfExtractor(), PARAMS)


def _assert_consistent(path: Path) -> None:
    db = sqlite3.connect(path)
    try:
        assert db.execute("PRAGMA integrity_check").fetchone() == ("ok",)
        assert db.execute("PRAGMA foreign_key_check").fetchall() == []
        # Every indexed file has its document, and every document has a file.
        assert (
            db.execute(
                "SELECT count(*) FROM files f WHERE status = 'indexed'"
                " AND NOT EXISTS (SELECT 1 FROM documents d WHERE d.sha256 = f.sha256)"
            ).fetchone()[0]
            == 0
        )
        assert (
            db.execute(
                "SELECT count(*) FROM documents d WHERE NOT EXISTS"
                " (SELECT 1 FROM files f WHERE f.sha256 = d.sha256 AND status = 'indexed')"
            ).fetchone()[0]
            == 0
        )
    finally:
        db.close()


# --- filenames ---------------------------------------------------------------------------

WEIRD_NAMES = [
    "with spaces.pdf",
    "  leading and trailing  .pdf",
    "-starts-with-dash.pdf",
    "hash#and?query%20.pdf",
    "räksmörgås.pdf",
    unicodedata.normalize("NFD", "décomposé.pdf"),
    "日本語のファイル.pdf",
    "emoji 📄.pdf",
    "quote'and\"double.pdf",
    "UPPER.Pdf",
]
if sys.platform == "win32":  # ? and " are not allowed in Windows filenames
    WEIRD_NAMES = [n for n in WEIRD_NAMES if not set(n) & set('?"')]


def test_weird_filenames_index_round_trip_and_are_a_no_op_twice(
    tmp_path: Path, store: Store
) -> None:
    root = tmp_path / "f"
    for i, name in enumerate(WEIRD_NAMES):
        make_pdf(root / "sub dir" / name, [text_page(f"Document number {i}.")])
    report = run(root, store)
    assert not report.failed
    assert len(report.added) == len(WEIRD_NAMES)
    for rel in store.files():
        assert rel.startswith("sub dir/")
        assert (root / rel).is_file(), rel  # the stored path leads back to the file
    generation = store.generation
    report = run(root, store)
    assert not report.changed and store.generation == generation


def test_stored_paths_use_forward_slashes(tmp_path: Path, store: Store) -> None:
    make_pdf(tmp_path / "f" / "a" / "b" / "c.pdf", [text_page("deep")])
    run(tmp_path / "f", store)
    assert list(store.files()) == ["a/b/c.pdf"]


@pytest.mark.skipif(sys.platform == "win32", reason="backslash is a separator on Windows")
def test_backslash_in_a_posix_filename_is_kept_verbatim(tmp_path: Path, store: Store) -> None:
    make_pdf(tmp_path / "f" / "back\\slash.pdf", [text_page("x")])
    run(tmp_path / "f", store)
    assert list(store.files()) == ["back\\slash.pdf"]


def _case_insensitive(path: Path) -> bool:
    probe = path / "CaseProbe"
    probe.mkdir()
    try:
        return (path / "caseprobe").exists()
    finally:
        probe.rmdir()


def test_case_only_rename_is_not_re_extracted(tmp_path: Path, store: Store) -> None:
    root = tmp_path / "f"
    make_pdf(root / "report.pdf", [text_page("x")])
    run(root, store)
    (root / "report.pdf").rename(root / "tmp.pdf")
    (root / "tmp.pdf").rename(root / "Report.pdf")
    report = run(root, store)
    assert report.reused == ["Report.pdf"] and report.removed == ["report.pdf"]
    assert list(store.files()) == ["Report.pdf"]


def test_same_folder_spelled_with_other_case_shares_one_index(tmp_path: Path) -> None:
    # F11. resolve() does not canonicalise case, so "Docs" and "docs" get two indexes.
    root = tmp_path / "Docs"
    root.mkdir()
    if not _case_insensitive(tmp_path):
        pytest.skip("case-sensitive filesystem")
    assert default_index_path(tmp_path / "docs", tmp_path) == default_index_path(root, tmp_path)


def test_same_folder_spelled_nfc_or_nfd_shares_one_index(tmp_path: Path) -> None:
    # F11. On macOS both spellings open the same folder but hash to different indexes.
    nfd = tmp_path / unicodedata.normalize("NFD", "Résumé")
    nfd.mkdir()
    nfc = tmp_path / unicodedata.normalize("NFC", "Résumé")
    if not nfc.exists():
        pytest.skip("filesystem distinguishes NFC and NFD names")
    assert default_index_path(nfc, tmp_path) == default_index_path(nfd, tmp_path)


def test_undecodable_filename_fails_alone(
    tmp_path: Path, store: Store, monkeypatch: pytest.MonkeyPatch
) -> None:
    # F2 (path variant). On Linux, a filename that is not valid UTF-8 reaches Python as
    # a str with surrogate escapes, and SQLite refuses to store it. Simulated here,
    # because macOS and Windows will not create such a name.
    root = tmp_path / "f"
    good = make_pdf(root / "good.pdf", [text_page("fine")])
    bad = make_pdf(root / "bad.pdf", [text_page("fine too")])
    monkeypatch.setattr(
        "askpdf.indexer.scan", lambda *_args: iter([("bad\udcff.pdf", bad), ("good.pdf", good)])
    )
    report = run(root, store)
    assert "good.pdf" in report.added


@pytest.mark.skipif(sys.platform in ("darwin", "win32"), reason="needs a byte-path filesystem")
def test_real_undecodable_filename_fails_alone(tmp_path: Path, store: Store) -> None:
    root = tmp_path / "f"
    make_pdf(root / "good.pdf", [text_page("fine")])
    data = (root / "good.pdf").read_bytes()
    try:
        with open(os.fsencode(root) + b"/bad\xff.pdf", "wb") as f:
            f.write(data + b"\n%different")
    except OSError:
        pytest.skip("filesystem rejects non-UTF-8 names")
    report = run(root, store)
    assert "good.pdf" in report.added or "good.pdf" in report.reused


@pytest.mark.skipif(sys.platform == "win32", reason="newline and : not allowed in names")
def test_control_characters_in_filenames_cannot_forge_docs_lines(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # F13. A newline in a name lets one file print a second, fake line in `docs`
    # (and escape sequences reach the terminal raw).
    root = tmp_path / "f"
    make_pdf(root / "real.pdf", [text_page("x")])
    make_pdf(root / "evil\nfake.pdf  FAILED: x.pdf", [text_page("y")])
    idx = str(tmp_path / "i.db")
    main(["index", str(root), "--index", idx])
    capsys.readouterr()
    main(["docs", str(root), "--index", idx])
    out = capsys.readouterr().out
    assert len(out.splitlines()) == 2  # one line per document
    assert "evil\\nfake.pdf" in out  # the newline is shown escaped


# --- symlinks (security: files will be served over HTTP) ------------------------------------


def _symlink(link: Path, target: Path | str, directory: bool = False) -> None:
    try:
        link.symlink_to(target, target_is_directory=directory)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks not permitted here")


def test_symlink_loops_do_not_hang_or_crash_the_scan(tmp_path: Path) -> None:
    root = tmp_path / "f"
    make_pdf(root / "a.pdf", [text_page("x")])
    _symlink(root / "self.pdf", "self.pdf")
    _symlink(root / "l1.pdf", "l2.pdf")
    _symlink(root / "l2.pdf", "l1.pdf")
    (root / "d").mkdir()
    _symlink(root / "d" / "up", "..", directory=True)
    _symlink(root / "d" / "loop", root / "d", directory=True)
    assert [rel for rel, _ in scan(root)] == ["a.pdf"]


def test_symlink_chain_that_ends_outside_is_ignored(tmp_path: Path) -> None:
    root = tmp_path / "f"
    outside = make_pdf(tmp_path / "secret" / "s.pdf", [text_page("secret")])
    root.mkdir()
    _symlink(root / "hop2.pdf", outside)
    _symlink(root / "hop1.pdf", root / "hop2.pdf")
    _symlink(root / "dotdot.pdf", Path("..") / "secret" / "s.pdf")
    assert list(scan(root)) == []


def test_symlink_whose_name_is_inside_but_target_shares_a_prefix(tmp_path: Path) -> None:
    # "/x/f-evil" starts with "/x/f" as a string but is not inside "/x/f".
    root = tmp_path / "f"
    root.mkdir()
    evil = make_pdf(tmp_path / "f-evil" / "s.pdf", [text_page("secret")])
    _symlink(root / "s.pdf", evil)
    assert list(scan(root)) == []


def test_file_swapped_for_an_outside_symlink_leaves_the_index(tmp_path: Path, store: Store) -> None:
    root = tmp_path / "f"
    make_pdf(root / "a.pdf", [text_page("x")])
    outside = make_pdf(tmp_path / "secret.pdf", [text_page("secret")])
    run(root, store)
    (root / "a.pdf").unlink()
    _symlink(root / "a.pdf", outside)
    report = run(root, store)
    assert report.removed == ["a.pdf"] and store.documents() == []


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="no FIFOs")
def test_fifo_named_pdf_does_not_block(tmp_path: Path, store: Store) -> None:
    root = tmp_path / "f"
    make_pdf(root / "a.pdf", [text_page("x")])
    os.mkfifo(root / "pipe.pdf")
    report = run(root, store)  # would hang forever if the FIFO were opened
    assert report.added == ["a.pdf"]


# --- permissions --------------------------------------------------------------------------


@pytest.mark.skipif(IS_ROOT or sys.platform == "win32", reason="needs POSIX permissions")
def test_permission_denied_file_fails_alone_and_rerun_writes_nothing(
    tmp_path: Path, store: Store
) -> None:
    root = tmp_path / "f"
    make_pdf(root / "a.pdf", [text_page("x")])
    locked = make_pdf(root / "locked.pdf", [text_page("y")])
    locked.chmod(0)
    try:
        report = run(root, store)
        assert report.added == ["a.pdf"] and "locked.pdf" in report.failed
        generation = store.generation
        run(root, store)
        assert store.generation == generation
    finally:
        locked.chmod(0o644)


@pytest.mark.skipif(IS_ROOT or sys.platform == "win32", reason="needs POSIX permissions")
def test_permission_denied_file_is_listed_by_docs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # F10. The run prints "1 failed", but `docs` never mentions the file.
    root = tmp_path / "f"
    make_pdf(root / "a.pdf", [text_page("x")])
    locked = make_pdf(root / "locked.pdf", [text_page("y")])
    locked.chmod(0)
    try:
        idx = str(tmp_path / "i.db")
        main(["index", str(root), "--index", idx])
        capsys.readouterr()
        main(["docs", str(root), "--index", idx])
        assert "locked.pdf" in capsys.readouterr().out
    finally:
        locked.chmod(0o644)


@pytest.mark.skipif(IS_ROOT or sys.platform == "win32", reason="needs POSIX permissions")
def test_unreadable_subfolder_does_not_drop_its_documents(tmp_path: Path, store: Store) -> None:
    # F4. os.walk swallows the PermissionError, so the files look deleted.
    root = tmp_path / "f"
    make_pdf(root / "a.pdf", [text_page("x")])
    make_pdf(root / "sub" / "b.pdf", [text_page("y")])
    run(root, store)
    (root / "sub").chmod(0)
    try:
        report = run(root, store)
    finally:
        (root / "sub").chmod(0o755)
    assert report.removed == []
    assert "sub/b.pdf" in store.files()


@pytest.mark.skipif(IS_ROOT or sys.platform == "win32", reason="needs POSIX permissions")
def test_unreadable_root_does_not_wipe_the_index(tmp_path: Path, store: Store) -> None:
    # F4. A folder that is briefly unreadable (iCloud, a network share) empties the index.
    root = tmp_path / "f"
    make_pdf(root / "a.pdf", [text_page("x")])
    run(root, store)
    root.chmod(0)
    try:
        report = run(root, store)
    finally:
        root.chmod(0o755)
    assert report.removed == []
    assert len(store.documents()) == 1


# --- files changing mid-run -----------------------------------------------------------------


def test_file_deleted_after_listing_is_settled_next_run(tmp_path: Path, store: Store) -> None:
    root = tmp_path / "f"
    make_pdf(root / "a.pdf", [text_page("x")])
    make_pdf(root / "b.pdf", [text_page("y")])

    def delete_b(rel: str) -> None:
        if rel == "b.pdf":
            (root / "b.pdf").unlink()

    report = index_folder(root, store, PypdfExtractor(), PARAMS, on_file=delete_b)
    assert report.added == ["a.pdf"] and "b.pdf" not in store.files()
    report = run(root, store)
    assert not report.changed
    _assert_consistent(store.path)


class RewritingExtractor(PypdfExtractor):
    """Replaces the file after it was hashed and before it is read."""

    def __init__(self, new_text: str) -> None:
        self.new_text = new_text

    def extract(self, path: Path) -> list[RawPage]:
        make_pdf(path, [text_page(self.new_text)])
        return super().extract(path)


def test_file_rewritten_between_hash_and_read_heals_on_the_next_run(
    tmp_path: Path, store: Store
) -> None:
    root = tmp_path / "f"
    make_pdf(root / "a.pdf", [text_page("Old text.")])
    index_folder(root, store, RewritingExtractor("New text."), PARAMS)
    run(root, store)
    [doc] = store.documents()
    assert store.pages(doc.sha256)[0].text == "New text."
    _assert_consistent(store.path)


# --- odd PDFs inside a real run ------------------------------------------------------------


def test_bad_tounicode_map_fails_only_that_file(tmp_path: Path, store: Store) -> None:
    # F2. Today the UnicodeEncodeError from SQLite aborts the whole run; files after the
    # bad one are never indexed, and every later run aborts at the same file.
    root = tmp_path / "f"
    make_pdf(root / "a.pdf", [text_page("before")])
    make_cmap_pdf(root / "b.pdf", "D800")
    make_pdf(root / "c.pdf", [text_page("after")])
    report = run(root, store)
    assert "c.pdf" in report.added
    assert "b.pdf" in report.added or "b.pdf" in report.failed


def test_zero_page_pdf_is_indexed_without_crashing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "f"
    root.mkdir()
    make_zero_page_pdf(root / "zero.pdf")
    idx = str(tmp_path / "i.db")
    assert main(["index", str(root), "--index", idx]) == 0
    assert main(["docs", str(root), "--index", idx]) == 0
    assert "zero.pdf  0 pages, 0 chunks" in capsys.readouterr().out


def test_two_new_copies_in_one_run_are_extracted_once(tmp_path: Path, store: Store) -> None:
    root = tmp_path / "f"
    make_pdf(root / "a.pdf", [text_page("same")])
    (root / "b.pdf").write_bytes((root / "a.pdf").read_bytes())
    report = run(root, store)
    assert report.added == ["a.pdf"] and report.reused == ["b.pdf"]
    [doc] = store.documents()
    assert doc.paths == ("a.pdf", "b.pdf")


# --- the --index option -------------------------------------------------------------------


def test_index_file_inside_the_folder_is_never_indexed_itself(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "f"
    make_pdf(root / "a.pdf", [text_page("x")])
    for name in ["index.db", "idx.pdf"]:
        idx = root / name
        assert main(["index", str(root), "--index", str(idx)]) == 0
        capsys.readouterr()
        with Store(idx) as s:
            assert set(s.files()) == {"a.pdf"}, name


def test_index_file_inside_the_folder_still_allows_a_no_op(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "f"
    make_pdf(root / "a.pdf", [text_page("x")])
    idx = root / "index.db"
    main(["index", str(root), "--index", str(idx)])
    main(["index", str(root), "--index", str(idx)])
    out = capsys.readouterr().out.splitlines()[-1]
    assert out.startswith("0 added") and "1 unchanged" in out


def test_index_pointing_at_a_pdf_is_refused_and_untouched(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "f"
    pdf = make_pdf(root / "a.pdf", [text_page("x")])
    before = pdf.read_bytes()
    assert main(["index", str(root), "--index", str(pdf)]) == 1
    assert "not an askpdf index" in capsys.readouterr().err
    assert pdf.read_bytes() == before
    assert not Path(f"{pdf}-wal").exists() and not Path(f"{pdf}-shm").exists()


@pytest.mark.parametrize("content", [b"x", b"hello world\n", b"SQLite format 3"])
def test_small_non_index_files_are_refused(
    tmp_path: Path, content: bytes, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "f"
    make_pdf(root / "a.pdf", [text_page("x")])
    target = tmp_path / "notes.txt"
    target.write_bytes(content)
    assert main(["index", str(root), "--index", str(target)]) == 1
    assert target.read_bytes() == content


def test_index_pointing_at_a_directory_is_a_clean_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # F12. Today: an uncaught sqlite3.OperationalError traceback.
    root = tmp_path / "f"
    make_pdf(root / "a.pdf", [text_page("x")])
    (tmp_path / "adir").mkdir()
    assert main(["index", str(root), "--index", str(tmp_path / "adir")]) == 1
    assert "askpdf: error" in capsys.readouterr().err


def test_sqlite_database_from_another_app_with_a_meta_table_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "other.db"
    db = sqlite3.connect(path)
    db.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT)")
    db.execute("INSERT INTO meta VALUES ('app', '\"other\"')")
    db.execute("PRAGMA user_version = 1")
    db.commit()
    db.close()
    from askpdf.store import NotAnIndexError

    with pytest.raises(NotAnIndexError):
        Store(path)
    db = sqlite3.connect(path)
    assert db.execute("SELECT value FROM meta").fetchall() == [('"other"',)]
    db.close()


# --- no-op means no writes at all -----------------------------------------------------------


def test_unchanged_rerun_writes_no_bytes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "f"
    make_pdf(root / "a.pdf", [text_page("x")])
    make_pdf(root / "sub" / "b.pdf", [text_page("y")])
    idx = tmp_path / "idx" / "i.db"
    main(["index", str(root), "--index", str(idx)])
    # Fold the WAL into the main file so a later write would show in either.
    db = sqlite3.connect(idx)
    db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    db.close()

    def snapshot() -> dict[str, bytes]:
        return {p.name: p.read_bytes() for p in idx.parent.iterdir() if not p.name.endswith("-shm")}

    before = snapshot()
    main(["index", str(root), "--index", str(idx)])
    assert snapshot() == before


def test_unchanged_rerun_makes_no_database_changes(tmp_path: Path, store: Store) -> None:
    root = tmp_path / "f"
    make_pdf(root / "a.pdf", [text_page("x")])
    run(root, store)
    changes = store._db.total_changes
    run(root, store)
    assert store._db.total_changes == changes
