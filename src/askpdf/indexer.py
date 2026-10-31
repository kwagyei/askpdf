"""Bring the index in line with a folder of PDFs.

Run it twice on an unchanged folder and the second run touches nothing: files whose
size and modification time match the index are skipped without being read.
Documents are keyed by content, so a renamed or copied file is recorded without
extracting it again. One unreadable file is recorded as failed and the rest carry on.

PR 10 runs this from a background thread driven by the folder watcher; here it is a
plain function.
"""

import hashlib
import io
import os
import stat
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from askpdf.chunking import ChunkParams, chunk_page
from askpdf.extract import clean
from askpdf.extract.base import ExtractionError, Extractor, RawPage
from askpdf.extract.clean import clean_document, clean_text, detect_boilerplate
from askpdf.providers.base import Embedder, Matrix
from askpdf.providers.http import ProviderError, UnreachableError
from askpdf.store import ChunkRecord, FileRow, PageRecord, Store, folder_key

PIN_SCHEMA = 1
EMBEDDER = "ollama/nomic-embed-text"
EMBED_BATCH = 64  # chunks per committed batch: an interrupted run keeps its progress


def pins(extractor: Extractor, params: ChunkParams, embedder: str = EMBEDDER) -> dict[str, Any]:
    """Everything that changes what the index holds. A change forces a rebuild."""
    return {
        "pipeline": PIN_SCHEMA,
        # The embedder is pinned even while Ollama is down: vectors from two models
        # cannot be compared, so switching models must rebuild the index.
        "embedder": embedder,
        "extractor": {"name": extractor.name, "version": extractor.version},
        "cleaning": clean.CLEANING_VERSION,
        "chunking": {"target": params.target, "overlap": params.overlap, "snap": params.snap},
    }


@dataclass
class IndexReport:
    added: list[str] = field(default_factory=list)
    reused: list[str] = field(default_factory=list)  # content already indexed (copy, rename)
    failed: dict[str, str] = field(default_factory=dict)
    removed: list[str] = field(default_factory=list)
    unreadable: list[str] = field(default_factory=list)  # folders that could not be listed
    out_of_reach: dict[str, str] = field(default_factory=dict)  # indexed, unreadable now
    unchanged: int = 0
    rebuilt: bool = False

    @property
    def changed(self) -> bool:
        return bool(self.added or self.reused or self.failed or self.removed or self.rebuilt)


# A file that could not be read at all is recorded with this modification time, which no
# real file has, so it is retried on every run instead of being skipped as unchanged.
UNREADABLE_MTIME = -1


def _is_hidden(name: str) -> bool:
    # Also skips iCloud placeholders (".name.pdf.icloud") and macOS "._" resource forks.
    return name.startswith(".")


def scan(
    root: Path, unreadable: list[str] | None = None, exclude: frozenset[Path] = frozenset()
) -> Iterator[tuple[str, Path]]:
    """Yield ``(relative posix path, real path)`` for every PDF under ``root``.

    Symlinks are followed only when their target stays inside ``root``: the app later
    serves these files over HTTP, so nothing outside the chosen folder may get in.
    Folders that cannot be listed are appended to ``unreadable`` ("." for the root),
    so the caller can tell "deleted" from "temporarily out of reach".
    """
    root = root.resolve()
    if root.is_file():
        if root.suffix.casefold() == ".pdf" and root not in exclude:
            yield root.name, root
        return

    def on_error(exc: OSError) -> None:
        if unreadable is not None and exc.filename is not None:
            path = Path(exc.filename)
            unreadable.append("." if path == root else path.relative_to(root).as_posix())

    for dirpath, dirnames, filenames in os.walk(root, onerror=on_error, followlinks=False):
        dirnames[:] = sorted(d for d in dirnames if not _is_hidden(d))
        for name in sorted(filenames):
            if _is_hidden(name) or not name.casefold().endswith(".pdf"):
                continue
            path = Path(dirpath) / name
            # os.path.realpath, unlike Path.resolve() on Python 3.12, never raises on a
            # symlink loop; it returns a path that then fails the checks below.
            real = Path(os.path.realpath(path))
            if not real.is_relative_to(root) or real in exclude:
                continue
            try:
                if not stat.S_ISREG(real.stat().st_mode):
                    continue
            except PermissionError:
                pass  # yield it, so the indexer reports it instead of it vanishing
            except OSError:
                continue  # dangling link, symlink loop, or deleted since it was listed
            yield path.relative_to(root).as_posix(), real


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def build_records(
    raw_pages: list[RawPage], params: ChunkParams
) -> tuple[list[PageRecord], list[ChunkRecord]]:
    cleaned = clean_document([p.text for p in raw_pages])
    boilerplate = detect_boilerplate([text.split("\n") if text else [] for text in cleaned])
    pages: list[PageRecord] = []
    chunks: list[ChunkRecord] = []
    for raw, text, flags in zip(raw_pages, cleaned, boilerplate, strict=True):
        label = clean_text(raw.label) or str(raw.index)  # labels come from the PDF too
        pages.append(PageRecord(raw.index, label, text, sorted(flags), raw.image_only))
        # Header and footer lines stay in the chunks: they are dropped only if that is
        # shown to help retrieval on the dev split (ADR-0002).
        for ordinal, chunk in enumerate(chunk_page(text, params)):
            chunks.append(ChunkRecord(raw.index, ordinal, chunk.start, chunk.end, chunk.text))
    return pages, chunks


def _storable(text: str) -> bool:
    """SQLite stores UTF-8; a non-UTF-8 filename arrives with surrogate escapes."""
    try:
        text.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True


def _under(path: str, folders: list[str]) -> bool:
    return any(f == "." or path == f or path.startswith(f + "/") for f in folders)


def index_folder(
    root: Path,
    store: Store,
    extractor: Extractor,
    params: ChunkParams,
    on_file: Callable[[str], None] | None = None,
) -> IndexReport:
    store.claim_folder(folder_key(root))
    report = IndexReport(rebuilt=store.ensure_pins(pins(extractor, params)))
    known = store.files()
    seen: set[str] = set()
    # The index may have been put inside the folder with --index; never index it.
    own_files = frozenset(
        Path(f"{store.path.resolve()}{suffix}") for suffix in ("", "-wal", "-shm", "-journal")
    )
    for rel, real in scan(root, report.unreadable, own_files):
        if not _storable(rel):
            shown = rel.encode("utf-8", "backslashreplace").decode("ascii", "backslashreplace")
            report.failed[shown] = "file name is not valid UTF-8; rename the file"
            continue
        seen.add(rel)
        previous = known.get(rel)
        try:
            info = real.stat()
            if (
                previous is not None
                and previous.size == info.st_size
                and previous.mtime_ns == info.st_mtime_ns
            ):
                report.unchanged += 1
                continue
            if on_file is not None:
                on_file(rel)
            # Read once: the hash and the extracted text must describe the same bytes,
            # even if the file is rewritten while we work.
            data = real.read_bytes()
        except FileNotFoundError:
            continue  # deleted since it was listed; the next run settles it
        except OSError as exc:
            reason = f"cannot read file: {exc.strerror or exc}"
            if previous is not None and previous.status == "indexed":
                # Out of reach (e.g. iCloud evicted it), not gone: keep what we have.
                report.out_of_reach[rel] = reason
                continue
            # Recorded so `docs` shows it, but with a modification time no real file has,
            # so it is retried next run (e.g. once iCloud has downloaded it).
            store.put_file(FileRow(rel, "", 0, UNREADABLE_MTIME, "failed", reason))
            report.failed[rel] = reason
            continue
        sha = sha256_bytes(data)
        row = FileRow(rel, sha, info.st_size, info.st_mtime_ns, "indexed", None)
        if previous is not None and previous.sha256 == sha and previous.status == "indexed":
            store.put_file(row)  # touched but identical: only the timestamp changed
            report.unchanged += 1
            continue
        if store.has_document(sha):
            store.put_file(row)
            report.reused.append(rel)
            continue
        try:
            raw_pages = extractor.extract(io.BytesIO(data))
        except ExtractionError as exc:
            mtime = UNREADABLE_MTIME if exc.retry else info.st_mtime_ns
            store.put_file(FileRow(rel, sha, info.st_size, mtime, "failed", str(exc)))
            report.failed[rel] = str(exc)
            continue
        pages, chunks = build_records(raw_pages, params)
        store.put_document(row, pages, chunks)
        report.added.append(rel)
    # Files in a folder that could not be listed are out of reach, not deleted.
    gone = sorted(p for p in set(known) - seen if not _under(p, report.unreadable))
    store.remove_files(gone)
    report.removed = gone
    return report


@dataclass
class EmbedReport:
    embedded: int = 0
    refused: list[int] = field(default_factory=list)  # chunk ids the model would not embed


def embed_missing(
    store: Store, embedder: Embedder, on_batch: Callable[[int, int], None] | None = None
) -> EmbedReport:
    """Embed every chunk that has no vector yet.

    Runs after indexing, and on every later run, so an index built while Ollama was not
    running catches up instead of being rebuilt. ``on_batch(done, total)`` reports
    progress. If Ollama cannot be reached the error propagates and the batches already
    stored are kept. A chunk the model refuses (too long, say) is skipped and reported,
    so it cannot block the others; it stays findable by keyword search.
    """
    todo = store.chunks_without_vectors()
    report = EmbedReport()
    dim = store.vector_dim()
    for start in range(0, len(todo), EMBED_BATCH):
        batch = todo[start : start + EMBED_BATCH]
        rows = _embed_batch(embedder, batch, report)
        for _, vector in rows:
            if dim is None:
                dim = len(vector)
            elif len(vector) != dim:
                raise ProviderError(
                    f"the embedding model now returns {len(vector)}-dimensional vectors but"
                    f" the index holds {dim}-dimensional ones; delete {store.path} and index again"
                )
        store.put_vectors([(cid, v.astype("<f4").tobytes()) for cid, v in rows])
        report.embedded += len(rows)
        if on_batch is not None:
            on_batch(start + len(batch), len(todo))
    return report


def _embed_batch(
    embedder: Embedder, batch: list[tuple[int, str]], report: EmbedReport
) -> list[tuple[int, Matrix]]:
    try:
        vectors = embedder.embed_documents([text for _, text in batch])
        return [(cid, v) for (cid, _), v in zip(batch, vectors, strict=True)]
    except UnreachableError:
        raise
    except ProviderError:
        if len(batch) == 1:
            report.refused.append(batch[0][0])
            return []
    # One chunk in the batch was refused: embed them one at a time to find it.
    rows: list[tuple[int, Matrix]] = []
    for item in batch:
        rows.extend(_embed_batch(embedder, [item], report))
    return rows
