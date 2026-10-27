"""The SQLite index (ADR-0003).

The index is a cache of one folder. It lives outside that folder, keys documents by the
SHA-256 of their bytes, and pins the settings that produced it: if any of them change,
it is rebuilt rather than migrated.
"""

import hashlib
import json
import os
import sqlite3
import sys
import unicodedata
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

SCHEMA_VERSION = 2  # 2: chunk ids are never reused

_SCHEMA = """
CREATE TABLE meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
-- One row per file in the folder. Several files may share a document (copies).
CREATE TABLE files (
    path     TEXT PRIMARY KEY,           -- relative to the indexed root, '/'-separated
    sha256   TEXT NOT NULL,
    size     INTEGER NOT NULL,
    mtime_ns INTEGER NOT NULL,
    status   TEXT NOT NULL CHECK (status IN ('indexed', 'failed')),
    error    TEXT
);
CREATE INDEX files_sha256 ON files (sha256);
CREATE TABLE documents (
    sha256       TEXT PRIMARY KEY,
    n_pages      INTEGER NOT NULL,
    n_image_only INTEGER NOT NULL
);
-- Full cleaned page text: citations quote from the whole page, not just one chunk.
CREATE TABLE pages (
    sha256      TEXT NOT NULL REFERENCES documents ON DELETE CASCADE,
    page        INTEGER NOT NULL,         -- physical, 1-based
    label       TEXT NOT NULL,            -- printed page label
    text        TEXT NOT NULL,
    boilerplate TEXT NOT NULL,            -- JSON list of header/footer line indices
    image_only  INTEGER NOT NULL,
    PRIMARY KEY (sha256, page)
);
CREATE TABLE chunks (
    -- AUTOINCREMENT: an id is never reused, so a vector computed for a chunk that was
    -- deleted meanwhile can never land on a new chunk.
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    sha256     TEXT NOT NULL,
    page       INTEGER NOT NULL,
    ordinal    INTEGER NOT NULL,
    start_word INTEGER NOT NULL,
    end_word   INTEGER NOT NULL,
    text       TEXT NOT NULL,
    FOREIGN KEY (sha256, page) REFERENCES pages ON DELETE CASCADE
);
CREATE INDEX chunks_page ON chunks (sha256, page);
CREATE TABLE vectors (
    chunk_id INTEGER PRIMARY KEY REFERENCES chunks ON DELETE CASCADE,
    vector   BLOB NOT NULL                -- float32, L2-normalised
);
"""


def cache_dir(
    platform: str = sys.platform,
    env: Mapping[str, str] = os.environ,
    home: Path | None = None,
) -> Path:
    home = Path.home() if home is None else home
    if platform == "darwin":
        return home / "Library" / "Caches"
    if platform == "win32":
        local = env.get("LOCALAPPDATA")
        return Path(local) if local else home / "AppData" / "Local"
    xdg = env.get("XDG_CACHE_HOME")
    # The XDG spec says a relative value is invalid and must be ignored. Judged by POSIX
    # rules, so the check means the same whichever OS runs it.
    return Path(xdg) if xdg and PurePosixPath(xdg).is_absolute() else home / ".cache"


def _same_file(a: str, b: str) -> bool:
    try:
        return a != b and os.path.samefile(a, b)
    except OSError:
        return False


def folder_key(root: Path) -> str:
    """One spelling per folder, so ``~/Docs`` and ``~/docs`` share an index.

    ``resolve()`` canonicalises neither case nor Unicode normalisation. Rather than
    guess from the OS, ask the filesystem: case is folded only if the swapped-case
    spelling opens the same folder, so a case-sensitive volume keeps ``A`` and ``a``
    apart. The same goes for NFC and NFD spellings.
    """
    path = str(root.resolve())
    nfc = unicodedata.normalize("NFC", path)
    if nfc != path and not _same_file(nfc, path):
        nfc = path  # a normalisation-sensitive filesystem: keep the name as it is
    swapped = nfc.swapcase()
    return nfc.casefold() if _same_file(swapped, nfc) else nfc


def default_index_path(root: Path, cache: Path | None = None) -> Path:
    """``<cache>/askpdf/<sha256(folder key)[:16]>/index.db``: never inside the folder itself.

    Writing a live SQLite WAL into an iCloud- or Dropbox-synced folder invites conflicts.
    """
    key = hashlib.sha256(folder_key(root).encode("utf-8", "surrogateescape")).hexdigest()[:16]
    return (cache_dir() if cache is None else cache) / "askpdf" / key / "index.db"


@dataclass(frozen=True, slots=True)
class FileRow:
    path: str
    sha256: str
    size: int
    mtime_ns: int
    status: str
    error: str | None


@dataclass(frozen=True, slots=True)
class PageRecord:
    page: int
    label: str
    text: str
    boilerplate: Sequence[int]
    image_only: bool


@dataclass(frozen=True, slots=True)
class ChunkRecord:
    page: int
    ordinal: int
    start_word: int
    end_word: int
    text: str


@dataclass(frozen=True, slots=True)
class DocumentSummary:
    sha256: str
    paths: tuple[str, ...]
    n_pages: int
    n_image_only: int
    n_chunks: int

    @property
    def looks_scanned(self) -> bool:
        """More than half the pages are images without text: OCR would be needed."""
        return self.n_pages > 0 and self.n_image_only * 2 > self.n_pages


class NotAnIndexError(Exception):
    """The file exists but was not created by askpdf; it is never overwritten."""


class OutdatedIndexError(Exception):
    """The index was made by another askpdf version or with other settings.

    Raised only for readers: wiping it is the indexer's job, never a side effect of a
    search.
    """


class WrongFolderError(Exception):
    """The index belongs to another folder; reusing it would wipe that folder's index."""


def _connect(path: Path) -> sqlite3.Connection:
    db = sqlite3.connect(path, isolation_level=None, check_same_thread=False)
    db.execute("PRAGMA foreign_keys = ON")
    return db


def _is_askpdf_index(db: sqlite3.Connection) -> bool:
    try:
        row = db.execute("SELECT value FROM meta WHERE key = 'app'").fetchone()
        return row is not None and json.loads(row[0]) == "askpdf"
    except (sqlite3.DatabaseError, ValueError, TypeError):
        return False  # another app's table, or values that are not our JSON


class Store:
    def __init__(self, path: Path, *, rebuild: bool = True) -> None:
        """Open (or create) the index at ``path``.

        With ``rebuild=False``, for commands that only read, an index from another schema
        version raises OutdatedIndexError instead of being replaced by an empty one.
        """
        if path.is_dir():
            raise NotAnIndexError(f"{path} is a directory, not an index file")
        self.path = path
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            self._create(path)
        self._db = _connect(path)
        try:
            version = self._db.execute("PRAGMA user_version").fetchone()[0]
            ours = _is_askpdf_index(self._db)
        except sqlite3.DatabaseError as exc:
            self._db.close()
            raise NotAnIndexError(f"{path} is not an askpdf index") from exc
        if not ours:
            # Includes empty and 1-byte files, which SQLite would happily treat as a new
            # database: only a file askpdf creates itself is ever written.
            self._db.close()
            raise NotAnIndexError(f"{path} is not an askpdf index; refusing to overwrite it")
        if version != SCHEMA_VERSION and not rebuild:
            self._db.close()
            raise OutdatedIndexError(f"{path} was made by another version of askpdf")
        if version != SCHEMA_VERSION:
            # An index from another schema version. It is a cache of the folder, so it is
            # replaced, never migrated.
            self._db.close()
            self._create(path)
            self._db = _connect(path)

    @staticmethod
    def _create(path: Path) -> None:
        """Build the index in a temporary file and move it into place.

        A crash or Ctrl-C halfway leaves only the temporary file, never a half-made
        index that later runs would refuse as "not an askpdf index".
        """
        tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        tmp.unlink(missing_ok=True)
        db = sqlite3.connect(tmp, isolation_level=None)
        try:
            db.execute("BEGIN")
            # Statement by statement: executescript() would commit the open transaction.
            for statement in _SCHEMA.split(";"):
                if statement.strip():
                    db.execute(statement)
            for key, value in (("app", "askpdf"), ("generation", 0)):
                db.execute("INSERT INTO meta VALUES (?, ?)", (key, json.dumps(value)))
            db.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
            db.execute("COMMIT")
            db.execute("PRAGMA journal_mode = WAL")
        finally:
            db.close()
        for suffix in ("-wal", "-shm"):
            Path(f"{path}{suffix}").unlink(missing_ok=True)
        os.replace(tmp, path)

    def close(self) -> None:
        self._db.close()

    def __enter__(self) -> "Store":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @contextmanager
    def read_transaction(self) -> Iterator[None]:
        """Reads inside see one committed state of the index (WAL snapshot isolation),
        however many documents the indexer commits meanwhile."""
        self._db.execute("BEGIN DEFERRED")
        try:
            yield
        finally:
            self._db.execute("COMMIT")  # a read-only transaction: nothing to undo

    @contextmanager
    def transaction(self) -> Iterator[None]:
        self._db.execute("BEGIN IMMEDIATE")
        try:
            yield
        except BaseException:
            self._db.execute("ROLLBACK")
            raise
        self._db.execute("COMMIT")

    # --- meta ------------------------------------------------------------------

    def _set_meta(self, key: str, value: Any) -> None:
        self._db.execute(
            "INSERT INTO meta (key, value) VALUES (?, ?)"
            " ON CONFLICT (key) DO UPDATE SET value = excluded.value",
            (key, json.dumps(value, sort_keys=True)),
        )

    def _get_meta(self, key: str) -> Any:
        row = self._db.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return None if row is None else json.loads(row[0])

    @property
    def generation(self) -> int:
        """Bumped by every committed change, so readers know when to rebuild a snapshot."""
        return int(self._get_meta("generation"))

    def _bump_generation(self) -> None:
        self._set_meta("generation", self.generation + 1)

    def claim_folder(self, key: str) -> None:
        """Tie the index to one folder, so pointing it at another raises instead of wiping it."""
        with self.transaction():  # read and claim under one write lock: no race
            owner = self._get_meta("folder")
            if owner is None:
                self._set_meta("folder", key)
        if owner is not None and owner != key:
            raise WrongFolderError(f"{self.path} is the index of {owner}")

    def check_folder(self, key: str) -> None:
        """Like claim_folder, for readers: raise if the index belongs to another folder."""
        owner = self._get_meta("folder")
        if owner is not None and owner != key:
            raise WrongFolderError(f"{self.path} is the index of {owner}")

    def pins_match(self, pins: Mapping[str, Any]) -> bool:
        """True if the index was built with these settings; for readers, who never wipe."""
        return bool(self._get_meta("pins") == json.loads(json.dumps(pins, sort_keys=True)))

    def ensure_pins(self, pins: Mapping[str, Any]) -> bool:
        """Record the settings this index is built with; wipe it if they changed.

        Returns True if the index was wiped and must be rebuilt from scratch.
        """
        stored = self._get_meta("pins")
        wanted = json.loads(json.dumps(pins, sort_keys=True))
        if stored == wanted:
            return False
        with self.transaction():
            had_content = stored is not None
            for table in ("vectors", "chunks", "pages", "documents", "files"):
                self._db.execute(f"DELETE FROM {table}")  # noqa: S608 - fixed names
            self._set_meta("pins", wanted)
            self._bump_generation()
        return had_content

    # --- files -----------------------------------------------------------------

    def files(self) -> dict[str, FileRow]:
        rows = self._db.execute(
            "SELECT path, sha256, size, mtime_ns, status, error FROM files"
        ).fetchall()
        return {row[0]: FileRow(*row) for row in rows}

    def files_get(self, path: str) -> FileRow | None:
        row = self._db.execute(
            "SELECT path, sha256, size, mtime_ns, status, error FROM files WHERE path = ?",
            (path,),
        ).fetchone()
        return None if row is None else FileRow(*row)

    def has_document(self, sha256: str) -> bool:
        query = "SELECT 1 FROM documents WHERE sha256 = ?"
        return self._db.execute(query, (sha256,)).fetchone() is not None

    def put_file(self, row: FileRow) -> None:
        """Record a file whose document is already stored (a copy or a rename), or a failure.

        Writing a row identical to the stored one is skipped, so re-runs stay no-ops.
        """
        if self.files_get(row.path) == row:
            return
        with self.transaction():
            self._upsert_file(row)
            self._collect_garbage()
            self._bump_generation()

    def _collect_garbage(self) -> int:
        """Drop documents no indexed file refers to; pages, chunks and vectors cascade."""
        return self._db.execute(
            "DELETE FROM documents WHERE sha256 NOT IN"
            " (SELECT sha256 FROM files WHERE status = 'indexed')"
        ).rowcount

    def _upsert_file(self, row: FileRow) -> None:
        self._db.execute(
            "INSERT INTO files (path, sha256, size, mtime_ns, status, error)"
            " VALUES (?, ?, ?, ?, ?, ?)"
            " ON CONFLICT (path) DO UPDATE SET sha256 = excluded.sha256,"
            " size = excluded.size, mtime_ns = excluded.mtime_ns,"
            " status = excluded.status, error = excluded.error",
            (row.path, row.sha256, row.size, row.mtime_ns, row.status, row.error),
        )

    def remove_files(self, paths: Sequence[str]) -> int:
        """Forget files that left the folder, and documents no file refers to any more.

        Returns the number of documents removed.
        """
        if not paths:
            return 0
        with self.transaction():
            self._db.executemany("DELETE FROM files WHERE path = ?", [(p,) for p in paths])
            removed = self._collect_garbage()
            self._bump_generation()
        return removed

    # --- documents ---------------------------------------------------------------

    def put_document(
        self,
        file: FileRow,
        pages: Sequence[PageRecord],
        chunks: Sequence[ChunkRecord],
    ) -> None:
        """Store a document, its pages and chunks, and its file row, in one transaction.

        A reader sees either none of it or all of it.
        """
        with self.transaction():
            self._db.execute(
                "INSERT OR REPLACE INTO documents (sha256, n_pages, n_image_only) VALUES (?, ?, ?)",
                (file.sha256, len(pages), sum(p.image_only for p in pages)),
            )
            self._db.executemany(
                "INSERT INTO pages (sha256, page, label, text, boilerplate, image_only)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                [
                    (
                        file.sha256,
                        p.page,
                        p.label,
                        p.text,
                        json.dumps(sorted(p.boilerplate)),
                        int(p.image_only),
                    )
                    for p in pages
                ],
            )
            self._db.executemany(
                "INSERT INTO chunks (sha256, page, ordinal, start_word, end_word, text)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                [
                    (file.sha256, c.page, c.ordinal, c.start_word, c.end_word, c.text)
                    for c in chunks
                ],
            )
            self._upsert_file(file)
            self._collect_garbage()
            self._bump_generation()

    def documents(self) -> list[DocumentSummary]:
        rows = self._db.execute(
            "SELECT d.sha256, d.n_pages, d.n_image_only,"
            " (SELECT count(*) FROM chunks c WHERE c.sha256 = d.sha256)"
            " FROM documents d"
        ).fetchall()
        paths: dict[str, list[str]] = {}
        for path, sha256 in self._db.execute(
            "SELECT path, sha256 FROM files WHERE status = 'indexed' ORDER BY path"
        ):
            paths.setdefault(sha256, []).append(path)
        summaries = [
            DocumentSummary(sha, tuple(paths.get(sha, ())), n_pages, n_image_only, n_chunks)
            for sha, n_pages, n_image_only, n_chunks in rows
        ]
        return sorted(summaries, key=lambda d: d.paths)

    def pages(self, sha256: str) -> list[PageRecord]:
        rows = self._db.execute(
            "SELECT page, label, text, boilerplate, image_only FROM pages"
            " WHERE sha256 = ? ORDER BY page",
            (sha256,),
        ).fetchall()
        return [
            PageRecord(page, label, text, tuple(json.loads(bp)), bool(image_only))
            for page, label, text, bp, image_only in rows
        ]

    def chunks(self, sha256: str) -> list[ChunkRecord]:
        rows = self._db.execute(
            "SELECT page, ordinal, start_word, end_word, text FROM chunks"
            " WHERE sha256 = ? ORDER BY page, ordinal",
            (sha256,),
        ).fetchall()
        return [ChunkRecord(*row) for row in rows]

    # --- query snapshot ----------------------------------------------------------

    def all_chunks(self) -> list[tuple[int, str, int, int, str]]:
        """``(id, sha256, page, ordinal, text)`` for every chunk, in id order."""
        return self._db.execute(
            "SELECT id, sha256, page, ordinal, text FROM chunks ORDER BY id"
        ).fetchall()

    def page_labels(self) -> dict[tuple[str, int], str]:
        return {
            (sha, page): label
            for sha, page, label in self._db.execute("SELECT sha256, page, label FROM pages")
        }

    def indexed_paths(self) -> dict[str, tuple[str, ...]]:
        """The files of each document, sorted; a document has several if it was copied."""
        paths: dict[str, list[str]] = {}
        for path, sha256 in self._db.execute(
            "SELECT path, sha256 FROM files WHERE status = 'indexed' ORDER BY path"
        ):
            paths.setdefault(sha256, []).append(path)
        return {sha: tuple(p) for sha, p in paths.items()}

    # --- vectors -------------------------------------------------------------------

    def chunks_without_vectors(self) -> list[tuple[int, str]]:
        """``(id, text)`` of every chunk not embedded yet, in id order."""
        return self._db.execute(
            "SELECT c.id, c.text FROM chunks c LEFT JOIN vectors v ON v.chunk_id = c.id"
            " WHERE v.chunk_id IS NULL ORDER BY c.id"
        ).fetchall()

    def put_vectors(self, rows: Sequence[tuple[int, bytes]]) -> None:
        """Store float32 vectors for chunks, in one transaction; readers rebuild after it."""
        if not rows:
            return
        with self.transaction():
            # A chunk deleted since it was read has no row to refer to; skip it.
            self._db.executemany(
                "INSERT OR REPLACE INTO vectors (chunk_id, vector)"
                " SELECT ?, ? WHERE EXISTS (SELECT 1 FROM chunks WHERE id = ?)",
                [(cid, blob, cid) for cid, blob in rows],
            )
            self._bump_generation()

    def vector_dim(self) -> int | None:
        """The length of the stored vectors, or None if there are none yet."""
        row = self._db.execute("SELECT length(vector) FROM vectors LIMIT 1").fetchone()
        return None if row is None else int(row[0]) // 4

    def all_vectors(self) -> dict[int, bytes]:
        return dict(self._db.execute("SELECT chunk_id, vector FROM vectors").fetchall())
