import sqlite3
from pathlib import Path

import pytest

from askpdf.store import (
    ChunkRecord,
    FileRow,
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

    monkeypatch.setattr("askpdf.store.os.replace", crash)
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
