"""Folder watching and background indexing (ADR-0009, design §13).

The watcher holds no clock: each test hands it listings scan by scan, so "two scans in a
row" is two calls, not four seconds of sleeping.
"""

import argparse
import os
import shutil
import sys
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

from askpdf import cli
from askpdf.chunking import ChunkParams
from askpdf.cli import build_parser, main
from askpdf.extract.pypdf_ext import PypdfExtractor
from askpdf.indexer import index_folder, own_files, read_bytes
from askpdf.store import Store
from askpdf.watch import Watcher, listing, watch
from askpdf.web.app import Progress
from tests.pdfs import make_pdf, text_page

PARAMS = ChunkParams(target=20, overlap=5, snap=3)


@pytest.fixture
def root(tmp_path: Path) -> Path:
    folder = tmp_path / "pdfs"
    make_pdf(folder / "a.pdf", [text_page("Alpha page one.")])
    return folder


@pytest.fixture
def store(tmp_path: Path) -> Iterator[Store]:
    with Store(tmp_path / "index" / "i.db") as s:
        yield s


class Folder:
    """A watcher and an index over one folder, stepped one scan at a time."""

    def __init__(self, root: Path, store: Store) -> None:
        self.root, self.store = root, store
        self.exclude = own_files(store.path)
        index_folder(root, store, PypdfExtractor(), PARAMS)  # the start-up run
        self.watcher = Watcher(listing(root, self.exclude))

    def scan(self) -> frozenset[str] | None:
        """One poll: the paths skipped if it indexed, None if it did nothing."""
        skip = self.watcher.tick(listing(self.root, self.exclude))
        if skip is not None:
            index_folder(self.root, self.store, PypdfExtractor(), PARAMS, skip=skip)
        return skip

    def indexed(self) -> dict[str, str]:
        return {path: row.status for path, row in self.store.files().items()}


# --- the watcher's decisions -----------------------------------------------------------------


def test_an_unchanged_folder_needs_no_indexing() -> None:
    watcher = Watcher({"a.pdf": (10, 1)})
    assert watcher.tick({"a.pdf": (10, 1)}) is None
    assert watcher.tick({"a.pdf": (10, 1)}) is None


def test_a_new_file_waits_one_scan_to_settle() -> None:
    watcher = Watcher({})
    assert watcher.tick({"a.pdf": (10, 1)}) == frozenset({"a.pdf"})  # seen once: leave it
    assert watcher.tick({"a.pdf": (10, 1)}) == frozenset()  # the same twice: index it
    assert watcher.tick({"a.pdf": (10, 1)}) is None  # and then nothing more


def test_a_file_still_growing_is_left_until_it_stops() -> None:
    watcher = Watcher({})
    for size in (10, 20, 30):
        assert watcher.tick({"a.pdf": (size, size)}) == frozenset({"a.pdf"})
    assert watcher.tick({"a.pdf": (30, 30)}) == frozenset()


def test_a_growing_file_does_not_hold_up_the_others() -> None:
    watcher = Watcher({})
    assert watcher.tick({"a.pdf": (1, 1), "big.pdf": (1, 1)}) == frozenset({"a.pdf", "big.pdf"})
    assert watcher.tick({"a.pdf": (1, 1), "big.pdf": (2, 2)}) == frozenset({"big.pdf"})


def test_a_deletion_is_indexed_on_the_next_scan() -> None:
    watcher = Watcher({"a.pdf": (10, 1)})
    assert watcher.tick({}) == frozenset()


def test_a_failed_run_is_retried_on_the_next_scan() -> None:
    watcher = Watcher({})
    watcher.tick({"a.pdf": (1, 1)})
    assert watcher.tick({"a.pdf": (1, 1)}) == frozenset()
    watcher.retry()
    assert watcher.tick({"a.pdf": (1, 1)}) == frozenset()
    assert watcher.tick({"a.pdf": (1, 1)}) is None


# --- the folder, scan by scan -----------------------------------------------------------------


def test_a_half_written_file_is_never_indexed(root: Path, store: Store) -> None:
    folder = Folder(root, store)
    make_pdf(root / "tmp.pdf", [text_page("Bravo arrives.")])
    data = (root / "tmp.pdf").read_bytes()
    (root / "tmp.pdf").unlink()
    (root / "b.pdf").write_bytes(data[: len(data) // 2])  # the copy is halfway
    assert folder.scan() == frozenset({"b.pdf"})
    assert "b.pdf" not in folder.indexed()
    (root / "b.pdf").write_bytes(data)  # the copy finishes
    assert folder.scan() == frozenset({"b.pdf"})
    assert folder.scan() == frozenset()
    assert folder.indexed()["b.pdf"] == "indexed"


def test_a_rename_is_recorded_without_reading_again(root: Path, store: Store) -> None:
    folder = Folder(root, store)
    sha = store.files()["a.pdf"].sha256
    (root / "a.pdf").rename(root / "renamed.pdf")
    folder.scan()
    folder.scan()
    assert folder.indexed() == {"renamed.pdf": "indexed"}
    assert store.files()["renamed.pdf"].sha256 == sha
    assert len(store.documents()) == 1


def test_a_duplicate_shares_the_document(root: Path, store: Store) -> None:
    folder = Folder(root, store)
    shutil.copy2(root / "a.pdf", root / "copy.pdf")
    folder.scan()
    folder.scan()
    assert folder.indexed() == {"a.pdf": "indexed", "copy.pdf": "indexed"}
    assert len(store.documents()) == 1


def test_delete_then_add_back(root: Path, store: Store) -> None:
    folder = Folder(root, store)
    data = (root / "a.pdf").read_bytes()
    (root / "a.pdf").unlink()
    folder.scan()
    assert folder.indexed() == {} and store.documents() == []
    (root / "a.pdf").write_bytes(data)
    folder.scan()
    folder.scan()
    assert folder.indexed() == {"a.pdf": "indexed"}


def test_a_file_changed_while_indexing_is_read_again(root: Path, store: Store) -> None:
    folder = Folder(root, store)
    make_pdf(root / "b.pdf", [text_page("First version.")])
    folder.scan()
    folder.scan()  # indexed; and just after it was read, the file changes
    first = store.files()["b.pdf"].sha256
    make_pdf(root / "b.pdf", [text_page("Second version, longer than the first.")])
    folder.scan()
    folder.scan()
    assert store.files()["b.pdf"].sha256 != first
    assert len(store.documents()) == 2  # a.pdf and the second version of b.pdf


@pytest.mark.skipif(sys.platform == "win32", reason="symlinks need privileges on Windows")
def test_a_symlink_out_of_the_folder_is_never_seen(root: Path, tmp_path: Path) -> None:
    make_pdf(tmp_path / "outside" / "secret.pdf", [text_page("Secret.")])
    (root / "secret.pdf").symlink_to(tmp_path / "outside" / "secret.pdf")
    (root / "inner.pdf").symlink_to(root / "a.pdf")
    assert sorted(listing(root)) == ["a.pdf", "inner.pdf"]


def test_the_index_inside_the_folder_is_not_listed(root: Path) -> None:
    with Store(root / "index.db") as store, store.writing():
        assert sorted(listing(root, own_files(store.path))) == ["a.pdf"]


# --- unreadable files --------------------------------------------------------------------------


def test_a_read_that_hangs_gives_up(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    release = threading.Event()
    monkeypatch.setattr(Path, "read_bytes", lambda self: release.wait(5) and b"")
    try:
        started = time.monotonic()
        with pytest.raises(OSError, match="not available locally"):
            read_bytes(tmp_path / "evicted.pdf", timeout=0.05)
        assert time.monotonic() - started < 2
    finally:
        release.set()  # let the abandoned reader end


def test_read_errors_pass_through(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read_bytes(tmp_path / "missing.pdf")


def test_a_file_that_cannot_be_read_in_time_fails_and_the_rest_carry_on(
    root: Path, store: Store, monkeypatch: pytest.MonkeyPatch
) -> None:
    make_pdf(root / "evicted.pdf", [text_page("Somewhere in iCloud.")])
    release = threading.Event()
    real = Path.read_bytes

    def slow(self: Path) -> bytes:
        if self.name == "evicted.pdf":
            release.wait(5)
        return real(self)

    monkeypatch.setattr(Path, "read_bytes", slow)
    try:
        report = index_folder(root, store, PypdfExtractor(), PARAMS, read_timeout=0.05)
    finally:
        release.set()
    assert report.added == ["a.pdf"]
    assert "not available locally" in report.failed["evicted.pdf"]


# --- the polling loop ------------------------------------------------------------------------


def test_watch_indexes_what_settles_and_stops(root: Path) -> None:
    runs: list[frozenset[str]] = []
    stop = threading.Event()

    def index(skip: frozenset[str]) -> bool:
        runs.append(skip)
        if skip == frozenset():
            stop.set()
        return True

    first = listing(root)
    make_pdf(root / "b.pdf", [text_page("New.")])
    watch(root, frozenset(), index, stop, first, interval=0.01)
    assert runs == [{"b.pdf"}, frozenset()]


def test_watch_tries_again_after_a_failed_run(root: Path) -> None:
    results = iter([False, True])
    runs = 0
    stop = threading.Event()

    def index(skip: frozenset[str]) -> bool:
        nonlocal runs
        runs += 1
        ok = next(results)
        if ok:
            stop.set()
        return ok

    (root / "a.pdf").unlink()
    watch(root, frozenset(), index, stop, {"a.pdf": (1, 1)}, interval=0.01)
    assert runs == 2


# --- the web app's indexer thread --------------------------------------------------------------


def _serve_args(root: Path, index: Path, *extra: str) -> argparse.Namespace:
    return build_parser().parse_args(["serve", str(root), "--index", str(index), *extra])


def test_startup_reconciles_the_index_with_the_folder(root: Path, tmp_path: Path) -> None:
    index = tmp_path / "i.db"
    make_pdf(root / "b.pdf", [text_page("Bravo.")])
    assert main(["index", str(root), "--index", str(index)]) == 0
    (root / "b.pdf").rename(root / "c.pdf")  # while askpdf was not running
    (root / "a.pdf").unlink()
    make_pdf(root / "d.pdf", [text_page("Delta.")])
    progress = Progress()
    with open(os.devnull, "w") as errors:
        cli._index_in_background(_serve_args(root, index), errors, None)(progress)
    assert progress.state == "ready"
    with Store(index) as store:
        assert sorted(store.files()) == ["c.pdf", "d.pdf"]


def test_the_thread_picks_up_a_new_file(root: Path, tmp_path: Path) -> None:
    progress = Progress()
    errors = open(os.devnull, "w")  # noqa: SIM115 - closed below
    thread = threading.Thread(
        target=cli._index_in_background(_serve_args(root, tmp_path / "i.db"), errors, 0.02),
        args=(progress,),
        daemon=True,
    )
    try:
        thread.start()
        make_pdf(root / "b.pdf", [text_page("Bravo.")])
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            with Store(tmp_path / "i.db") as store:
                if "b.pdf" in store.files():
                    break
            time.sleep(0.02)
        else:
            pytest.fail("the new file was never indexed")
    finally:
        errors.close()  # the daemon thread dies with the test process
    assert progress.state == "ready"


def test_a_second_server_leaves_the_index_to_the_first(root: Path, tmp_path: Path) -> None:
    index = tmp_path / "i.db"
    progress = Progress()
    with Store(index) as first, first.writing(), open(os.devnull, "w") as errors:
        cli._index_in_background(_serve_args(root, index), errors, 0.02)(progress)
    assert progress.state == "ready"
    assert "another askpdf" in progress.message


def test_index_refuses_while_another_askpdf_writes(
    root: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    index = tmp_path / "i.db"
    with Store(index) as first, first.writing():
        assert main(["index", str(root), "--index", str(index)]) == 1
    assert "another askpdf is indexing or serving" in capsys.readouterr().err


def test_indexing_waits_while_an_answer_is_written(root: Path, tmp_path: Path) -> None:
    make_pdf(root / "b.pdf", [text_page("Bravo.")])
    progress = Progress()
    progress.idle.clear()  # an answer is being written
    errors = open(os.devnull, "w")  # noqa: SIM115 - closed below
    thread = threading.Thread(
        target=cli._index_in_background(_serve_args(root, tmp_path / "i.db"), errors, None),
        args=(progress,),
    )
    try:
        thread.start()
        thread.join(0.3)
        assert thread.is_alive()  # waiting before its first file
        with Store(tmp_path / "i.db") as store:
            assert store.files() == {}
        progress.idle.set()  # the answer is done
        thread.join(10)
        assert not thread.is_alive()
    finally:
        progress.idle.set()
        errors.close()
    with Store(tmp_path / "i.db") as store:
        assert sorted(store.files()) == ["a.pdf", "b.pdf"]
