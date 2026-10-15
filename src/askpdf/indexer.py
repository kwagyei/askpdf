"""Bring the index in line with a folder of PDFs.

Run it twice on an unchanged folder and the second run touches nothing: files whose
size and modification time match the index are skipped without being read.
Documents are keyed by content, so a renamed or copied file is recorded without
extracting it again. One unreadable file is recorded as failed and the rest carry on.

PR 10 runs this from a background thread driven by the folder watcher; here it is a
plain function.
"""

import hashlib
import os
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from askpdf.chunking import ChunkParams, chunk_page
from askpdf.extract.base import ExtractionError, Extractor, RawPage
from askpdf.extract.clean import clean_text, detect_boilerplate
from askpdf.store import ChunkRecord, FileRow, PageRecord, Store

PIN_SCHEMA = 1


def pins(extractor: Extractor, params: ChunkParams) -> dict[str, Any]:
    """Everything that changes what the index holds. A change forces a rebuild."""
    return {
        "pipeline": PIN_SCHEMA,
        "extractor": {"name": extractor.name, "version": extractor.version},
        "chunking": {"target": params.target, "overlap": params.overlap, "snap": params.snap},
    }


@dataclass
class IndexReport:
    added: list[str] = field(default_factory=list)
    reused: list[str] = field(default_factory=list)  # content already indexed (copy, rename)
    failed: dict[str, str] = field(default_factory=dict)
    removed: list[str] = field(default_factory=list)
    unchanged: int = 0
    rebuilt: bool = False

    @property
    def changed(self) -> bool:
        return bool(self.added or self.reused or self.failed or self.removed or self.rebuilt)


def _is_hidden(name: str) -> bool:
    # Also skips iCloud placeholders (".name.pdf.icloud") and macOS "._" resource forks.
    return name.startswith(".")


def scan(root: Path) -> Iterator[tuple[str, Path]]:
    """Yield ``(relative posix path, real path)`` for every PDF under ``root``.

    Symlinks are followed only when their target stays inside ``root``: the app later
    serves these files over HTTP, so nothing outside the chosen folder may get in.
    """
    root = root.resolve()
    if root.is_file():
        if root.suffix.casefold() == ".pdf":
            yield root.name, root
        return
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames[:] = sorted(d for d in dirnames if not _is_hidden(d))
        for name in sorted(filenames):
            if _is_hidden(name) or not name.casefold().endswith(".pdf"):
                continue
            path = Path(dirpath) / name
            real = path.resolve()
            if not real.is_relative_to(root) or not real.is_file():
                continue
            yield path.relative_to(root).as_posix(), real


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def build_records(
    raw_pages: list[RawPage], params: ChunkParams
) -> tuple[list[PageRecord], list[ChunkRecord]]:
    cleaned = [clean_text(p.text) for p in raw_pages]
    boilerplate = detect_boilerplate([text.split("\n") if text else [] for text in cleaned])
    pages: list[PageRecord] = []
    chunks: list[ChunkRecord] = []
    for raw, text, flags in zip(raw_pages, cleaned, boilerplate, strict=True):
        pages.append(PageRecord(raw.index, raw.label, text, sorted(flags), raw.image_only))
        # Header and footer lines stay in the chunks: they are dropped only if that is
        # shown to help retrieval on the dev split (ADR-0002).
        for ordinal, chunk in enumerate(chunk_page(text, params)):
            chunks.append(ChunkRecord(raw.index, ordinal, chunk.start, chunk.end, chunk.text))
    return pages, chunks


def index_folder(
    root: Path,
    store: Store,
    extractor: Extractor,
    params: ChunkParams,
    on_file: Callable[[str], None] | None = None,
) -> IndexReport:
    report = IndexReport(rebuilt=store.ensure_pins(pins(extractor, params)))
    known = store.files()
    seen: set[str] = set()
    for rel, real in scan(root):
        seen.add(rel)
        try:
            stat = real.stat()
        except OSError:
            continue  # vanished between listing and reading; the next run settles it
        previous = known.get(rel)
        if (
            previous is not None
            and previous.size == stat.st_size
            and previous.mtime_ns == stat.st_mtime_ns
        ):
            report.unchanged += 1
            continue
        if on_file is not None:
            on_file(rel)
        try:
            sha = sha256_file(real)
        except OSError as exc:
            # Not recorded: an unreadable file (e.g. not downloaded from iCloud yet) is
            # retried next time instead of being remembered as failed.
            report.failed[rel] = f"cannot read file: {exc}"
            continue
        row = FileRow(rel, sha, stat.st_size, stat.st_mtime_ns, "indexed", None)
        if previous is not None and previous.sha256 == sha and previous.status == "indexed":
            store.put_file(row)  # touched but identical: only the timestamp changed
            report.unchanged += 1
            continue
        if store.has_document(sha):
            store.put_file(row)
            report.reused.append(rel)
            continue
        try:
            raw_pages = extractor.extract(real)
        except ExtractionError as exc:
            store.put_file(FileRow(rel, sha, stat.st_size, stat.st_mtime_ns, "failed", str(exc)))
            report.failed[rel] = str(exc)
            continue
        pages, chunks = build_records(raw_pages, params)
        store.put_document(row, pages, chunks)
        report.added.append(rel)
    gone = sorted(set(known) - seen)
    store.remove_files(gone)
    report.removed = gone
    return report
