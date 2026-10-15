"""Concurrent readers, a second writer, and interrupted runs (tester, PR 2)."""

import os
import signal
import sqlite3
import subprocess
import sys
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any, BinaryIO

import pytest

from askpdf.chunking import ChunkParams
from askpdf.cli import main
from askpdf.extract.base import RawPage
from askpdf.extract.pypdf_ext import PypdfExtractor
from askpdf.indexer import index_folder
from askpdf.store import Store
from tests.pdfs import make_pdf, text_page
from tests.test_edge_index import _assert_consistent

PARAMS = ChunkParams(target=20, overlap=5, snap=3)


@pytest.fixture
def folder(tmp_path: Path) -> Path:
    root = tmp_path / "f"
    for name in ["a", "b", "c", "d"]:
        make_pdf(root / f"{name}.pdf", [text_page(f"Page of {name}."), text_page("Two.")])
    return root


@pytest.fixture
def store(tmp_path: Path) -> Iterator[Store]:
    with Store(tmp_path / "idx" / "i.db") as s:
        yield s


# --- a reader during a write ---------------------------------------------------------------


def test_reader_sees_all_or_nothing_while_a_document_is_written(
    folder: Path, store: Store, monkeypatch: pytest.MonkeyPatch
) -> None:
    index_folder(folder, store, PypdfExtractor(), PARAMS)
    make_pdf(folder / "e.pdf", [text_page("New document.")] * 3)
    inside = threading.Event()
    release = threading.Event()
    real_gc = Store._collect_garbage

    def paused_gc(self: Store) -> int:
        inside.set()  # pages and chunks are written but not committed
        assert release.wait(10)
        return real_gc(self)

    monkeypatch.setattr(Store, "_collect_garbage", paused_gc)
    errors: list[BaseException] = []

    def write() -> None:
        try:
            index_folder(folder, store, PypdfExtractor(), PARAMS)
        except BaseException as exc:  # pragma: no cover - reported below
            errors.append(exc)

    writer = threading.Thread(target=write)
    writer.start()
    try:
        assert inside.wait(10)
        with Store(store.path) as reader:  # opening must not block on the writer
            generation = reader.generation
            docs = reader.documents()
            assert len(docs) == 4
            assert "e.pdf" not in reader.files()
            assert all(reader.pages(d.sha256) for d in docs)
    finally:
        release.set()
        writer.join(10)
    assert not errors
    with Store(store.path) as reader:
        assert reader.generation > generation
        [new] = [d for d in reader.documents() if d.paths == ("e.pdf",)]
        assert len(reader.pages(new.sha256)) == 3


def test_reader_opened_before_the_first_index_exists(tmp_path: Path, folder: Path) -> None:
    # The web app may open the index before the indexer has created any table.
    path = tmp_path / "idx" / "i.db"
    with Store(path) as reader, Store(path) as writer:
        index_folder(folder, writer, PypdfExtractor(), PARAMS)
        assert len(reader.documents()) == 4


# --- a second writer -----------------------------------------------------------------------


def test_second_writer_gets_a_clean_error(
    tmp_path: Path, folder: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # F9. Two `askpdf index` runs (or later, the CLI and the web app's indexer) on one
    # index: the second waits 5 s and then dies with a sqlite3.OperationalError traceback.
    idx = tmp_path / "idx" / "i.db"
    assert main(["index", str(folder), "--index", str(idx)]) == 0
    make_pdf(folder / "new.pdf", [text_page("new")])
    other = sqlite3.connect(idx, isolation_level=None)
    other.execute("BEGIN IMMEDIATE")
    try:
        assert main(["index", str(folder), "--index", str(idx)]) == 1
        assert "askpdf: error" in capsys.readouterr().err
    finally:
        other.execute("ROLLBACK")
        other.close()


# --- Ctrl-C -------------------------------------------------------------------------------


class InterruptingExtractor(PypdfExtractor):
    def __init__(self, after: int) -> None:
        self.left = after

    def extract(self, source: Path | BinaryIO) -> list[RawPage]:
        if self.left == 0:
            raise KeyboardInterrupt
        self.left -= 1
        return super().extract(source)


def test_ctrl_c_during_extraction_keeps_committed_documents(folder: Path, store: Store) -> None:
    with pytest.raises(KeyboardInterrupt):
        index_folder(folder, store, InterruptingExtractor(after=2), PARAMS)
    assert sorted(store.files()) == ["a.pdf", "b.pdf"]
    _assert_consistent(store.path)
    report = index_folder(folder, store, PypdfExtractor(), PARAMS)
    assert report.added == ["c.pdf", "d.pdf"] and not report.removed


def test_ctrl_c_inside_the_document_transaction_rolls_it_back(
    folder: Path, store: Store, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Index a.pdf alone first. The index belongs to the folder, so move the others
    # out of the way rather than indexing a.pdf as a single file.
    aside = folder.parent / "aside"
    aside.mkdir()
    for name in ["b.pdf", "c.pdf", "d.pdf"]:
        (folder / name).rename(aside / name)
    index_folder(folder, store, PypdfExtractor(), PARAMS)
    for name in ["b.pdf", "c.pdf", "d.pdf"]:
        (aside / name).rename(folder / name)
    generation = store.generation
    real_upsert = Store._upsert_file

    def interrupt(self: Store, row: Any) -> None:
        real_upsert(self, row)
        raise KeyboardInterrupt  # after pages, chunks and the file row are written

    monkeypatch.setattr(Store, "_upsert_file", interrupt)
    with pytest.raises(KeyboardInterrupt):
        index_folder(folder, store, PypdfExtractor(), PARAMS)
    monkeypatch.undo()
    assert store.generation == generation
    db = sqlite3.connect(store.path)
    assert db.execute("SELECT count(*) FROM pages").fetchone()[0] == 2  # a.pdf only
    db.close()
    _assert_consistent(store.path)
    # The connection is usable afterwards: no transaction was left open.
    report = index_folder(folder, store, PypdfExtractor(), PARAMS)
    assert sorted(report.added) == ["b.pdf", "c.pdf", "d.pdf"] and report.unchanged == 1


def test_ctrl_c_during_a_settings_rebuild_leaves_a_usable_index(folder: Path, store: Store) -> None:
    index_folder(folder, store, PypdfExtractor(), PARAMS)
    other = ChunkParams(target=30, overlap=5, snap=3)
    with pytest.raises(KeyboardInterrupt):
        index_folder(folder, store, InterruptingExtractor(after=1), other)
    _assert_consistent(store.path)
    report = index_folder(folder, store, PypdfExtractor(), other)
    assert not report.rebuilt  # the pins already match; the rest is filled in
    assert len(store.documents()) == 4


@pytest.mark.skipif(sys.platform == "win32", reason="SIGINT to a child is POSIX-only here")
def test_real_sigint_mid_index_leaves_a_consistent_index(tmp_path: Path) -> None:
    root = tmp_path / "many"
    for i in range(60):
        lines = [f"Document {i} line {j} with some words." for j in range(40)]
        make_pdf(root / f"doc{i:03}.pdf", [text_page(*lines)] * 4)
    idx = tmp_path / "idx" / "i.db"
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    cmd = [sys.executable, "-m", "askpdf", "index", str(root), "--index", str(idx)]
    proc = subprocess.Popen(  # noqa: S603 - our own interpreter and module
        cmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE, env=env, text=True
    )
    assert proc.stderr is not None
    seen = 0
    for line in proc.stderr:
        seen += line.startswith("reading ")
        if seen == 5:
            proc.send_signal(signal.SIGINT)
            break
    proc.communicate(timeout=60)
    assert proc.returncode != 0
    _assert_consistent(idx)
    with Store(idx) as store:
        done = len(store.files())
        assert 0 < done < 60
        report = index_folder(root, store, PypdfExtractor(), ChunkParams())
        assert len(report.added) == 60 - done and not report.removed and not report.rebuilt
