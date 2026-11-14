"""Edge cases for the review-lows branch: file ids, served PDF size, failed COMMIT."""

import io
import socket
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

import pytest
import uvicorn

from askpdf.cli import free_port, main
from askpdf.store import (
    SCHEMA_VERSION,
    ChunkRecord,
    FileRow,
    OutdatedIndexError,
    PageRecord,
    Store,
)
from askpdf.web.app import Library, cookie_name, create_app
from tests.pdfs import make_pdf, text_page


def _put(store: Store, path: str, n: int) -> None:
    store.put_document(
        FileRow(path, f"{n:064x}", 1, 1, "indexed", None),
        [PageRecord(1, "1", "text", (), False)],
        [ChunkRecord(1, 0, 0, 1, "text")],
    )


def _row(path: str, n: int) -> FileRow:
    return FileRow(path, f"{n:064x}", 1, 1, "indexed", None)


# --- criterion 1: ids -----------------------------------------------------------------


def test_schema_version_is_three() -> None:
    assert SCHEMA_VERSION == 3


def test_file_ids_survive_reopen_and_stay_unique_across_delete_all(tmp_path: Path) -> None:
    with Store(tmp_path / "i.db") as store:
        _put(store, "a.pdf", 1)
        _put(store, "b.pdf", 2)
        seen = set(store.file_ids().values())
        store.remove_files(["a.pdf", "b.pdf"])  # table now empty
    with Store(tmp_path / "i.db") as store:
        _put(store, "c.pdf", 3)
        assert store.file_ids()["c.pdf"] not in seen


def test_rename_gets_a_new_id_and_the_old_one_is_dead(tmp_path: Path) -> None:
    with Store(tmp_path / "i.db") as store:
        _put(store, "a.pdf", 1)
        old = store.file_ids()["a.pdf"]
        store.put_file(_row("b.pdf", 1))  # a copy: same document
        store.remove_files(["a.pdf"])  # then the original leaves: a rename
        new = store.file_ids()["b.pdf"]
        assert new != old
        assert store.file_path(old) is None
        assert store.file_path(new) == "b.pdf"
        assert store.has_document(f"{1:064x}")  # the document survives the rename


def test_copy_has_its_own_id_and_both_resolve(tmp_path: Path) -> None:
    with Store(tmp_path / "i.db") as store:
        _put(store, "a.pdf", 1)
        store.put_file(_row("copy.pdf", 1))
        ids = store.file_ids()
        assert len(set(ids.values())) == 2
        assert store.file_path(ids["copy.pdf"]) == "copy.pdf"


def test_put_file_on_existing_path_keeps_its_id(tmp_path: Path) -> None:
    with Store(tmp_path / "i.db") as store:
        _put(store, "a.pdf", 1)
        _put(store, "b.pdf", 2)
        before = store.file_ids()["a.pdf"]
        store.put_file(_row("a.pdf", 2))  # a.pdf now holds b's content
        assert store.file_ids()["a.pdf"] == before


def test_a_failed_then_fixed_file_keeps_its_id(tmp_path: Path) -> None:
    with Store(tmp_path / "i.db") as store:
        _put(store, "a.pdf", 1)
        before = store.file_ids()["a.pdf"]
        store.put_file(FileRow("a.pdf", "", 1, 1, "failed", "broken"))
        assert "a.pdf" not in store.file_ids()
        assert store.file_path(before) is None
        _put(store, "a.pdf", 1)
        assert store.file_ids()["a.pdf"] == before


def test_path_is_unique(tmp_path: Path) -> None:
    with Store(tmp_path / "i.db") as store:
        _put(store, "a.pdf", 1)
        with pytest.raises(sqlite3.IntegrityError):
            store._db.execute(
                "INSERT INTO files (path, sha256, size, mtime_ns, status) VALUES"
                " ('a.pdf', 'x', 1, 1, 'indexed')"
            )


def _make_schema2(index: Path) -> None:
    """Turn a fresh index into a schema 2 one: path is the primary key, no id column."""
    with Store(index) as store:
        _put(store, "a.pdf", 1)
        db = store._db
        db.execute("PRAGMA foreign_keys = OFF")
        db.execute("ALTER TABLE files RENAME TO files_new")
        db.execute(
            "CREATE TABLE files (path TEXT PRIMARY KEY, sha256 TEXT NOT NULL,"
            " size INTEGER NOT NULL, mtime_ns INTEGER NOT NULL, status TEXT NOT NULL,"
            " error TEXT)"
        )
        db.execute(
            "INSERT INTO files SELECT path, sha256, size, mtime_ns, status, error FROM files_new"
        )
        db.execute("DROP TABLE files_new")
        db.execute("PRAGMA user_version = 2")


def test_a_schema_2_index_is_rebuilt_by_store(tmp_path: Path) -> None:
    index = tmp_path / "i.db"
    _make_schema2(index)
    with pytest.raises(OutdatedIndexError):
        Store(index, rebuild=False)
    with Store(index, rebuild=True) as store:
        assert store.file_ids() == {}
        _put(store, "a.pdf", 1)
        assert store.file_path(store.file_ids()["a.pdf"]) == "a.pdf"
        cols = [r[1] for r in store._db.execute("PRAGMA table_info(files)")]
        assert cols[0] == "id"


def _folder(tmp_path: Path) -> Path:
    make_pdf(tmp_path / "f" / "p.pdf", [text_page("Passwords need fifteen characters.")])
    return tmp_path / "f"


def test_every_command_handles_a_schema_2_index(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    folder = _folder(tmp_path)
    index = tmp_path / "i.db"
    assert main(["index", str(folder), "--index", str(index)]) == 0
    _make_schema2(index)
    capsys.readouterr()
    # readers refuse with a pointer to `index`, they never misread the old table
    for cmd in (["docs", str(folder)], ["search", str(folder), "passwords"]):
        main([*cmd, "--index", str(index)])
        assert "askpdf index" in capsys.readouterr().err
        assert sqlite3.connect(index).execute("PRAGMA user_version").fetchone()[0] == 2
    capsys.readouterr()
    assert main(["index", str(folder), "--index", str(index)]) == 0
    assert sqlite3.connect(index).execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
    capsys.readouterr()
    assert main(["docs", str(folder), "--index", str(index)]) == 0
    assert "p.pdf" in capsys.readouterr().out
    assert main(["search", str(folder), "passwords", "--index", str(index)]) == 0


# --- criterion 2: /pdf size -----------------------------------------------------------


class _Server:
    def __init__(self, folder: Path, index: Path) -> None:
        self.port: int = free_port() or 0
        app = create_app(
            root=folder,
            index=index,
            port=self.port,
            token="tok",
            chat_for=lambda *a: None,  # type: ignore[arg-type]
        )
        self.server = uvicorn.Server(
            uvicorn.Config(app, host="127.0.0.1", port=self.port, log_level="error")
        )
        self.thread = threading.Thread(target=self.server.run, daemon=True)

    def __enter__(self) -> "_Server":
        self.thread.start()
        for _ in range(100):
            if self.server.started:
                return self
            time.sleep(0.05)
        raise RuntimeError("server did not start")

    def __exit__(self, *exc: object) -> None:
        self.server.should_exit = True
        self.thread.join(5)

    def get(self, path: str) -> bytes:
        s = socket.create_connection(("127.0.0.1", self.port), timeout=5)
        cookie = f"{cookie_name(self.port)}=tok"
        s.sendall(
            f"GET {path} HTTP/1.1\r\nHost: 127.0.0.1:{self.port}\r\nCookie: {cookie}\r\n"
            "Connection: close\r\n\r\n".encode()
        )
        data = b""
        try:
            while chunk := s.recv(65536):
                data += chunk
        finally:
            s.close()
        return data


@pytest.fixture
def served(tmp_path: Path) -> tuple[Path, Path]:
    folder = _folder(tmp_path)
    index = tmp_path / "i.db"
    assert main(["index", str(folder), "--index", str(index)]) == 0
    return folder, index


def test_real_server_sends_exactly_content_length_when_file_grows(
    served: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    size = 200_000
    monkeypatch.setattr(Library, "pdf", lambda self, i: (io.BytesIO(b"y" * (size * 3)), size))
    with _Server(*served) as srv:
        raw = srv.get("/pdf/1")
    head, _, body = raw.partition(b"\r\n\r\n")
    assert f"content-length: {size}".encode() in head.lower()
    assert len(body) == size


def test_real_server_survives_a_file_that_shrinks(
    served: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(Library, "pdf", lambda self, i: (io.BytesIO(b"z" * 10), 5000))
    with _Server(*served) as srv:
        t0 = time.monotonic()
        raw = srv.get("/pdf/1")  # must end (connection closed), not hang
        assert time.monotonic() - t0 < 4
        assert len(raw.partition(b"\r\n\r\n")[2]) <= 5000
        # and the server still answers afterwards
        again = srv.get("/pdf/1")
        assert again.startswith(b"HTTP/1.1 200")


def test_file_of_exactly_chunk_multiple_and_empty_file(
    served: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    from askpdf.web.app import CHUNK_BYTES

    for n in (0, CHUNK_BYTES, CHUNK_BYTES * 2):
        monkeypatch.setattr(Library, "pdf", lambda self, i, n=n: (io.BytesIO(b"q" * (n + 9)), n))
        with _Server(*served) as srv:
            body = srv.get("/pdf/1").partition(b"\r\n\r\n")[2]
        assert len(body) == n


# --- criterion 3: transaction ---------------------------------------------------------


class _CommitFails:
    def __init__(self, db: sqlite3.Connection, error: BaseException) -> None:
        self.db = db
        self.error = error

    def execute(self, sql: str, *args: Any) -> Any:
        if sql == "COMMIT":
            raise self.error
        return self.db.execute(sql, *args)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.db, name)


def test_commit_failing_with_a_non_sqlite_error_still_rolls_back(tmp_path: Path) -> None:
    with Store(tmp_path / "i.db") as store:
        real = store._db
        store._db = _CommitFails(real, KeyboardInterrupt())  # type: ignore[assignment]
        with pytest.raises(KeyboardInterrupt), store.transaction():
            real.execute("INSERT INTO meta (key, value) VALUES ('k', '1')")
        store._db = real
        assert not real.in_transaction
        assert real.execute("SELECT 1 FROM meta WHERE key = 'k'").fetchone() is None


def test_a_body_that_raises_rolls_back(tmp_path: Path) -> None:
    with Store(tmp_path / "i.db") as store:
        with pytest.raises(ValueError), store.transaction():
            store._db.execute("INSERT INTO meta (key, value) VALUES ('k', '1')")
            raise ValueError("boom")
        assert not store._db.in_transaction
        assert store._db.execute("SELECT 1 FROM meta WHERE key = 'k'").fetchone() is None


def test_commit_failure_after_a_failed_put_document_leaves_store_usable(tmp_path: Path) -> None:
    with Store(tmp_path / "i.db") as store:
        real = store._db
        store._db = _CommitFails(real, sqlite3.OperationalError("full"))  # type: ignore[assignment]
        with pytest.raises(sqlite3.OperationalError):
            _put(store, "a.pdf", 1)
        store._db = real
        assert store.file_ids() == {}
        assert not store.has_document(f"{1:064x}")
        _put(store, "a.pdf", 1)
        assert "a.pdf" in store.file_ids()


def test_writing_lock_with_a_failing_commit_releases_and_reacquires(tmp_path: Path) -> None:
    with Store(tmp_path / "i.db") as store:
        real = store._db
        with pytest.raises(sqlite3.OperationalError), store.writing():
            store._db = _CommitFails(real, sqlite3.OperationalError("full"))  # type: ignore[assignment]
            with store.transaction():
                pass
        store._db = real
        with store.writing(), store.transaction():
            pass


def test_a_rollback_that_finds_no_transaction_does_not_mask_the_error(tmp_path: Path) -> None:
    with (
        Store(tmp_path / "i.db") as store,
        pytest.raises(ValueError, match="boom"),
        store.transaction(),
    ):
        store._db.execute("ROLLBACK")  # the body ended the transaction itself
        raise ValueError("boom")
