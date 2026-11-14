"""File ids are never reused, and a failed COMMIT leaves no transaction open."""

import sqlite3
from pathlib import Path
from typing import Any

import pytest

from askpdf.store import ChunkRecord, FileRow, PageRecord, Store


def _put(store: Store, path: str, n: int) -> None:
    store.put_document(
        FileRow(path, f"{n:064x}", 1, 1, "indexed", None),
        [PageRecord(1, "1", "text", (), False)],
        [ChunkRecord(1, 0, 0, 1, "text")],
    )


def test_a_deleted_files_id_is_not_given_to_the_next_file(tmp_path: Path) -> None:
    with Store(tmp_path / "i.db") as store:
        _put(store, "a.pdf", 1)
        _put(store, "b.pdf", 2)
        old = store.file_ids()
        store.remove_files(["b.pdf"])  # the newest row, whose rowid SQLite would reuse
        _put(store, "c.pdf", 3)
        assert store.file_ids()["c.pdf"] not in old.values()
        assert store.file_path(old["b.pdf"]) is None  # an old link to b finds nothing


def test_reindexing_a_path_keeps_its_id(tmp_path: Path) -> None:
    with Store(tmp_path / "i.db") as store:
        _put(store, "a.pdf", 1)
        before = store.file_ids()["a.pdf"]
        _put(store, "a.pdf", 2)  # the file changed
        assert store.file_ids()["a.pdf"] == before


class _CommitFails:
    def __init__(self, db: sqlite3.Connection) -> None:
        self.db = db

    def execute(self, sql: str, *args: Any) -> Any:
        if sql == "COMMIT":
            raise sqlite3.OperationalError("disk I/O error")
        return self.db.execute(sql, *args)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.db, name)


def test_a_failed_commit_rolls_back_and_frees_the_connection(tmp_path: Path) -> None:
    with Store(tmp_path / "i.db") as store:
        real = store._db
        store._db = _CommitFails(real)  # type: ignore[assignment]
        with pytest.raises(sqlite3.OperationalError), store.transaction():
            real.execute("INSERT INTO meta (key, value) VALUES ('x', '1')")
        store._db = real
        assert not real.in_transaction
        assert real.execute("SELECT 1 FROM meta WHERE key = 'x'").fetchone() is None
        with store.transaction():  # the next one still starts
            real.execute("INSERT INTO meta (key, value) VALUES ('y', '1')")
        assert real.execute("SELECT 1 FROM meta WHERE key = 'y'").fetchone() is not None
