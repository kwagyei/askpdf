import errno
import sqlite3
import sys
import threading
import time
from pathlib import Path

import pytest

from askpdf import store as store_module
from askpdf.store import (
    SCHEMA_VERSION,
    ChunkRecord,
    FileRow,
    IndexBusyError,
    NotAnIndexError,
    PageRecord,
    Store,
    cache_dir,
    default_index_path,
)

HOME = Path("/home/u")


@pytest.mark.parametrize(
    ("platform", "env", "expected"),
    [
        ("darwin", {}, HOME / "Library" / "Caches"),
        ("linux", {}, HOME / ".cache"),
        ("linux", {"XDG_CACHE_HOME": "/xdg"}, Path("/xdg")),
        ("linux", {"XDG_CACHE_HOME": "relative/ignored"}, HOME / ".cache"),
        ("linux", {"XDG_CACHE_HOME": ""}, HOME / ".cache"),
        ("win32", {"LOCALAPPDATA": "C:/Users/u/AppData/Local"}, Path("C:/Users/u/AppData/Local")),
        ("win32", {}, HOME / "AppData" / "Local"),
    ],
)
def test_cache_dir_per_platform(platform: str, env: dict[str, str], expected: Path) -> None:
    assert cache_dir(platform, env, HOME) == expected


def test_index_path_is_outside_the_folder_and_stable(tmp_path: Path) -> None:
    folder = tmp_path / "pdfs"
    folder.mkdir()
    cache = tmp_path / "cache"
    path = default_index_path(folder, cache)
    assert not path.is_relative_to(folder)
    assert path.parent.parent == cache / "askpdf"
    assert path == default_index_path(folder / ".." / "pdfs", cache)  # same real folder
    other = tmp_path / "other"
    other.mkdir()
    assert default_index_path(other, cache) != path


def _doc(store: Store, path: str, sha: str, n_pages: int = 2) -> None:
    pages = [PageRecord(i, str(i), f"text {i}", [], False) for i in range(1, n_pages + 1)]
    chunks = [ChunkRecord(i, 0, 0, 2, f"text {i}") for i in range(1, n_pages + 1)]
    store.put_document(FileRow(path, sha, 1, 1, "indexed", None), pages, chunks)


def test_new_index_is_empty_at_generation_zero(tmp_path: Path) -> None:
    with Store(tmp_path / "i.db") as store:
        assert store.generation == 0
        assert store.documents() == [] and store.files() == {}


def test_every_write_bumps_the_generation(tmp_path: Path) -> None:
    with Store(tmp_path / "i.db") as store:
        _doc(store, "a.pdf", "aa")
        assert store.generation == 1
        store.put_file(FileRow("b.pdf", "aa", 1, 1, "indexed", None))
        assert store.generation == 2
        store.remove_files(["b.pdf"])
        assert store.generation == 3
        store.remove_files([])  # nothing to do: no new generation
        assert store.generation == 3


def test_document_round_trip(tmp_path: Path) -> None:
    with Store(tmp_path / "i.db") as store:
        pages = [
            PageRecord(1, "i", "Header\nBody", [0], False),
            PageRecord(2, "ii", "", [], True),
        ]
        chunks = [ChunkRecord(1, 0, 0, 2, "Header Body")]
        store.put_document(FileRow("a.pdf", "aa", 10, 20, "indexed", None), pages, chunks)
        assert store.pages("aa") == [
            PageRecord(1, "i", "Header\nBody", (0,), False),
            PageRecord(2, "ii", "", (), True),
        ]
        assert store.chunks("aa") == chunks
        [doc] = store.documents()
        assert (doc.paths, doc.n_pages, doc.n_image_only, doc.n_chunks) == (("a.pdf",), 2, 1, 1)


def test_copies_share_one_document(tmp_path: Path) -> None:
    with Store(tmp_path / "i.db") as store:
        _doc(store, "a.pdf", "aa")
        store.put_file(FileRow("copy/a.pdf", "aa", 1, 1, "indexed", None))
        [doc] = store.documents()
        assert doc.paths == ("a.pdf", "copy/a.pdf")


def test_document_is_dropped_with_its_last_file(tmp_path: Path) -> None:
    with Store(tmp_path / "i.db") as store:
        _doc(store, "a.pdf", "aa")
        store.put_file(FileRow("b.pdf", "aa", 1, 1, "indexed", None))
        assert store.remove_files(["a.pdf"]) == 0
        assert store.remove_files(["b.pdf"]) == 1
        assert store.documents() == []
        assert store.pages("aa") == [] and store.chunks("aa") == []  # cascaded


def test_changed_content_drops_the_old_document(tmp_path: Path) -> None:
    with Store(tmp_path / "i.db") as store:
        _doc(store, "a.pdf", "old")
        _doc(store, "a.pdf", "new")
        assert [d.sha256 for d in store.documents()] == ["new"]
        assert store.chunks("old") == []


def test_a_file_that_now_fails_releases_its_document(tmp_path: Path) -> None:
    with Store(tmp_path / "i.db") as store:
        _doc(store, "a.pdf", "aa")
        store.put_file(FileRow("a.pdf", "bb", 1, 2, "failed", "corrupt"))
        assert store.documents() == []
        assert store.files()["a.pdf"].error == "corrupt"


def test_failed_write_leaves_no_trace(tmp_path: Path) -> None:
    with Store(tmp_path / "i.db") as store:
        pages = [PageRecord(1, "1", "t", [], False)]
        bad_chunks = [ChunkRecord(99, 0, 0, 1, "chunk on a page that does not exist")]
        with pytest.raises(sqlite3.IntegrityError):
            store.put_document(FileRow("a.pdf", "aa", 1, 1, "indexed", None), pages, bad_chunks)
        assert store.documents() == [] and store.files() == {}
        assert store.generation == 0


@pytest.mark.parametrize(
    ("n_pages", "n_image_only", "scanned"), [(4, 2, False), (4, 3, True), (0, 0, False)]
)
def test_looks_scanned_means_more_than_half(n_pages: int, n_image_only: int, scanned: bool) -> None:
    from askpdf.store import DocumentSummary

    assert DocumentSummary("s", (), n_pages, n_image_only, 0).looks_scanned is scanned


def test_pins_wipe_the_index_when_settings_change(tmp_path: Path) -> None:
    with Store(tmp_path / "i.db") as store:
        assert store.ensure_pins({"chunking": 1}) is False  # first run: nothing to wipe
        _doc(store, "a.pdf", "aa")
        assert store.ensure_pins({"chunking": 1}) is False
        assert store.documents() != []
        assert store.ensure_pins({"chunking": 2}) is True
        assert store.documents() == [] and store.files() == {}


def test_index_survives_reopening(tmp_path: Path) -> None:
    with Store(tmp_path / "i.db") as store:
        _doc(store, "a.pdf", "aa")
    with Store(tmp_path / "i.db") as store:
        assert [d.paths for d in store.documents()] == [("a.pdf",)]
        assert store.generation == 1


def test_older_schema_is_rebuilt(tmp_path: Path) -> None:
    path = tmp_path / "i.db"
    with Store(path) as store:
        _doc(store, "a.pdf", "aa")
    db = sqlite3.connect(path)
    db.execute("PRAGMA user_version = 0")
    db.close()
    with Store(path) as store:
        assert store.documents() == [] and store.generation == 0


def test_refuses_to_overwrite_a_database_it_did_not_create(tmp_path: Path) -> None:
    path = tmp_path / "precious.db"
    db = sqlite3.connect(path)
    db.execute("CREATE TABLE notes (body TEXT)")
    db.execute("INSERT INTO notes VALUES ('keep me')")
    db.commit()
    db.close()
    with pytest.raises(NotAnIndexError):
        Store(path)
    db = sqlite3.connect(path)
    assert db.execute("SELECT body FROM notes").fetchall() == [("keep me",)]
    db.close()


def test_refuses_a_file_that_is_not_sqlite(tmp_path: Path) -> None:
    path = tmp_path / "notes.txt"
    path.write_text("not a database, and long enough to have a header" * 10)
    with pytest.raises(NotAnIndexError):
        Store(path)
    assert path.read_text().startswith("not a database")


def test_creates_missing_parent_directories(tmp_path: Path) -> None:
    with Store(tmp_path / "deep" / "er" / "i.db") as store:
        assert store.path.exists()


def test_foreign_meta_values_that_are_not_json_are_refused(tmp_path: Path) -> None:
    path = tmp_path / "photos.db"
    db = sqlite3.connect(path)
    db.execute("CREATE TABLE meta (key TEXT, value TEXT)")
    db.execute("INSERT INTO meta VALUES ('app', 'photos')")
    db.commit()
    db.close()
    with pytest.raises(NotAnIndexError):
        Store(path)


def test_creation_leaves_no_temporary_files(tmp_path: Path) -> None:
    with Store(tmp_path / "i.db"):
        pass
    assert sorted(p.name for p in tmp_path.iterdir() if p.name.endswith(".tmp")) == []


def test_interrupted_creation_does_not_lock_askpdf_out(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "i.db"

    def crash(*_args: object) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr("askpdf.store.os.link", crash)
    with pytest.raises(KeyboardInterrupt):
        Store(path)
    monkeypatch.undo()
    assert not path.exists()  # only a temporary file was written
    with Store(path) as store:  # the next run starts cleanly
        assert store.generation == 0


def test_claiming_twice_for_the_same_folder_is_fine(tmp_path: Path) -> None:
    from askpdf.store import WrongFolderError

    with Store(tmp_path / "i.db") as store:
        store.claim_folder("/a")
        store.claim_folder("/a")
        store.check_folder("/a")
        with pytest.raises(WrongFolderError):
            store.check_folder("/b")


def test_folder_key_folds_case_only_where_the_filesystem_does(tmp_path: Path) -> None:
    from askpdf.store import folder_key

    folder = tmp_path / "Docs"
    folder.mkdir()
    other = tmp_path / "docs"
    if other.exists():  # case-insensitive volume: one folder, one key
        assert folder_key(folder) == folder_key(other)
    else:  # case-sensitive: two folders, two keys
        other.mkdir()
        assert folder_key(folder) != folder_key(other)


def test_a_second_creator_never_replaces_an_index_already_in_use(tmp_path: Path) -> None:
    # Two processes racing to create the index: the loser must use the winner's file,
    # not swap in its own, or the winner would go on writing into an unlinked file.
    path = tmp_path / "i.db"
    with Store(path) as first:
        first.claim_folder("folder")
        Store._create(path)  # the loser of the race, arriving late
        with Store(path) as second:
            assert second._get_meta("folder") == "folder"
    assert [p.name for p in tmp_path.iterdir() if p.name.endswith(".tmp")] == []


def test_an_old_schema_is_rebuilt_in_place(tmp_path: Path) -> None:
    path = tmp_path / "i.db"
    with Store(path) as store:
        store.claim_folder("folder")
        store._db.execute("PRAGMA user_version = 1")
    inode = path.stat().st_ino
    with Store(path) as store:
        assert store._get_meta("folder") is None and store.generation == 0
        assert store._db.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
    assert path.stat().st_ino == inode  # the same file: other processes still see it


def test_only_one_writer_holds_an_index(tmp_path: Path) -> None:
    with Store(tmp_path / "i.db") as first, Store(tmp_path / "i.db") as second:
        with (
            first.writing(),
            pytest.raises(IndexBusyError, match="another askpdf"),
            second.writing(),
        ):
            pass
        with second.writing():  # released when the first is done
            pass


def test_a_second_creator_waits_while_one_creates(tmp_path: Path) -> None:
    """Clearing a deleted index's -wal and linking the new index happen under the lock,
    so no creator can clear the live WAL of one that linked first (seen in CI as "not an
    askpdf index")."""
    index = tmp_path / "i.db"
    done = threading.Event()

    def create() -> None:
        with Store(index) as store:
            store.put_file(FileRow("a.pdf", "ab", 1, 1, "failed", "e"))
        done.set()

    with store_module._creating(index) as alone:
        assert alone
        thread = threading.Thread(target=create, daemon=True)
        thread.start()
        assert not done.wait(0.3) and not index.exists()
    thread.join(10)
    with Store(index) as store:
        assert "a.pdf" in store.files()


def test_creating_an_index_does_not_make_a_server_think_it_is_busy(tmp_path: Path) -> None:
    """The flake behind this: a server's indexer took the writer lock while a reader was
    creating the index, was told "another askpdf is writing", and never indexed."""
    index = tmp_path / "i.db"
    Store(index).close()
    with store_module._creating(index), Store(index) as server, server.writing():
        pass  # no IndexBusyError


@pytest.mark.skipif(sys.platform == "win32", reason="Windows cannot delete an open index")
def test_a_server_whose_index_was_deleted_recreates_it_without_waiting(tmp_path: Path) -> None:
    index = tmp_path / "i.db"
    with Store(index) as server, server.writing():
        index.unlink()
        start = time.monotonic()
        with Store(index) as fresh:
            assert fresh.files() == {}
        assert time.monotonic() - start < 1.0


def test_a_creator_stops_waiting_once_the_index_exists(tmp_path: Path) -> None:
    index = tmp_path / "i.db"
    Store(index).close()
    lock = open(f"{index}.creating", "a+b")  # another creator, hung  # noqa: SIM115
    try:
        store_module._try_lock(lock)
        start = time.monotonic()
        with store_module._creating(index) as alone:
            assert not alone
        assert time.monotonic() - start < 1.0
    finally:
        lock.close()


def test_a_lock_file_that_cannot_be_opened_does_not_stop_creation(tmp_path: Path) -> None:
    index = tmp_path / "i.db"
    Path(f"{index}.creating").mkdir()
    with store_module._creating(index) as alone:
        assert alone
    with Store(index) as store:
        assert store.files() == {}


def test_a_volume_without_locks_does_not_stop_creation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def no_locks(file: object) -> None:
        raise OSError(errno.ENOLCK, "No locks available")

    monkeypatch.setattr(store_module, "_try_lock", no_locks)
    with store_module._creating(tmp_path / "i.db") as alone:
        assert alone
