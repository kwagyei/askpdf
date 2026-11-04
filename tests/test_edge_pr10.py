"""Adversarial edge cases for folder watching and background indexing (PR 10).

Tests marked ``xfail(strict=True)`` are genuine bugs found by the tester (F<n> in the
report, ~/Vault/Projects/code-projects/tester/2026-11-02-askpdf-pr10.md); they flip to
failures when the bug is fixed so the marker gets removed.
"""

import argparse
import errno
import gc
import os
import sqlite3
import sys
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

from askpdf import cli
from askpdf import store as store_module
from askpdf.chunking import ChunkParams
from askpdf.cli import build_parser, main
from askpdf.extract.pypdf_ext import PypdfExtractor
from askpdf.indexer import index_folder, own_files, read_bytes
from askpdf.store import FileRow, IndexBusyError, Store
from askpdf.watch import Watcher, listing
from askpdf.web.app import Progress, create_app
from tests.pdfs import make_pdf, text_page
from tests.test_web import BASE, PORT, TOKEN, ScriptedChat, _app, _client, _conversation
from tests.test_web import folder as folder

PARAMS = ChunkParams(target=20, overlap=5, snap=3)
POSIX = sys.platform != "win32"


@pytest.fixture
def root(tmp_path: Path) -> Path:
    folder_ = tmp_path / "pdfs"
    make_pdf(folder_ / "a.pdf", [text_page("Alpha page one.")])
    return folder_


@pytest.fixture
def store(tmp_path: Path) -> Iterator[Store]:
    with Store(tmp_path / "index" / "i.db") as s:
        yield s


def _args(root: Path, index: Path, *extra: str) -> argparse.Namespace:
    return build_parser().parse_args(["serve", str(root), "--index", str(index), *extra])


def _run_thread(target: Callable[[Progress], None], progress: Progress) -> threading.Thread:
    """Run like ``create_app`` does, keeping the exception the thread dies of."""
    thread = threading.Thread(target=target, args=(progress,), daemon=True)
    thread.start()
    return thread


# --- the same size and mtime ----------------------------------------------------------------


def test_replaced_file_with_same_size_and_mtime_is_not_noticed(root: Path, store: Store) -> None:
    """F1 (by design, low): the change test is (size, mtime_ns), so a same-size rewrite that
    keeps the mtime (cp -p, rsync -t, a coarse-mtime filesystem) is never re-read."""
    index_folder(root, store, PypdfExtractor(), PARAMS)
    before = store.files()["a.pdf"]
    target = root / "a.pdf"
    info = target.stat()
    original = target.read_bytes()
    target.unlink()
    replacement = bytearray(original)
    replacement[-1] ^= 0x01  # same length, different bytes
    target.write_bytes(bytes(replacement))
    os.utime(target, ns=(info.st_atime_ns, info.st_mtime_ns))
    assert listing(root)["a.pdf"] == (info.st_size, info.st_mtime_ns)
    assert Watcher(listing(root)).tick(listing(root)) is None  # nothing to do
    report = index_folder(root, store, PypdfExtractor(), PARAMS)
    assert report.unchanged == 1 and not report.changed
    assert store.files()["a.pdf"].sha256 == before.sha256  # the index still holds the old bytes


# --- the folder itself ----------------------------------------------------------------------


def test_root_deleted_while_serving_keeps_the_index(root: Path, store: Store) -> None:
    index_folder(root, store, PypdfExtractor(), PARAMS)
    exclude = own_files(store.path)
    watcher = Watcher(listing(root, exclude))
    (root / "a.pdf").unlink()
    root.rmdir()
    assert listing(root, exclude) == {}
    skip = watcher.tick(listing(root, exclude))
    assert skip == frozenset()
    report = index_folder(root, store, PypdfExtractor(), PARAMS, skip=skip)
    assert report.unreadable == ["."] and report.removed == []
    assert list(store.files()) == ["a.pdf"]  # out of reach, not deleted


def test_root_renamed_and_back_is_picked_up_without_reindexing(root: Path, store: Store) -> None:
    index_folder(root, store, PypdfExtractor(), PARAMS)
    moved = root.with_name("moved")
    root.rename(moved)
    assert index_folder(root, store, PypdfExtractor(), PARAMS).removed == []
    moved.rename(root)
    report = index_folder(root, store, PypdfExtractor(), PARAMS)
    assert report.unchanged == 1 and not report.changed


@pytest.mark.skipif(not POSIX or os.geteuid() == 0, reason="needs POSIX permissions, not root")
def test_unreadable_subfolder_is_out_of_reach_not_deleted(root: Path, store: Store) -> None:
    make_pdf(root / "sub" / "b.pdf", [text_page("Bravo.")])
    index_folder(root, store, PypdfExtractor(), PARAMS)
    exclude = own_files(store.path)
    watcher = Watcher(listing(root, exclude))
    (root / "sub").chmod(0)
    try:
        assert "sub/b.pdf" not in listing(root, exclude)
        skip = watcher.tick(listing(root, exclude))
        assert skip == frozenset()
        report = index_folder(root, store, PypdfExtractor(), PARAMS, skip=skip)
        assert report.removed == [] and report.unreadable == ["sub"]
    finally:
        (root / "sub").chmod(0o755)
    # Readable again: it reappears, waits one scan to settle, and costs no re-read.
    assert watcher.tick(listing(root, exclude)) == frozenset({"sub/b.pdf"})
    assert watcher.tick(listing(root, exclude)) == frozenset()
    assert index_folder(root, store, PypdfExtractor(), PARAMS).unchanged == 2


def test_a_file_whose_stat_fails_is_listed_and_recorded_as_failed(
    root: Path, store: Store, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_stat = Path.stat

    def stat(self: Path, *, follow_symlinks: bool = True) -> os.stat_result:
        if self.name == "a.pdf" and sys._getframe(1).f_code.co_name in {"listing", "index_folder"}:
            raise PermissionError(errno.EACCES, "denied")
        return real_stat(self, follow_symlinks=follow_symlinks)

    monkeypatch.setattr(Path, "stat", stat)
    assert listing(root)["a.pdf"] == (-1, -1)
    watcher = Watcher(listing(root))
    assert watcher.tick(listing(root)) is None  # stably unreadable: no busy loop
    report = index_folder(root, store, PypdfExtractor(), PARAMS)
    assert "a.pdf" in report.failed
    # ... and it is retried on the next run, not skipped as unchanged.
    assert "a.pdf" in index_folder(root, store, PypdfExtractor(), PARAMS).failed


def test_a_file_that_appears_unreadable_then_readable_is_indexed(root: Path, store: Store) -> None:
    watcher = Watcher({"a.pdf": (-1, -1)})
    assert watcher.tick(listing(root)) == frozenset({"a.pdf"})  # changed: wait a scan
    assert watcher.tick(listing(root)) == frozenset()  # settled: index it


# --- --index inside the folder --------------------------------------------------------------


def test_index_inside_the_watched_folder_never_lists_its_own_files(root: Path) -> None:
    index = root / "idx.pdf"  # a name that would match the *.pdf filter
    with Store(index) as store:
        with store.writing():
            exclude = own_files(index)
            assert (root / "idx.pdf.lock").exists()
            index_folder(root, store, PypdfExtractor(), PARAMS)
            assert list(store.files()) == ["a.pdf"]
            assert list(listing(root, exclude)) == ["a.pdf"]
        assert (root / "idx.pdf.lock") in exclude  # a lock named *.pdf would be excluded too


def test_index_through_a_symlink_into_the_folder_is_excluded(root: Path, tmp_path: Path) -> None:
    (tmp_path / "elsewhere").mkdir()
    link = tmp_path / "elsewhere" / "link.db"
    real = root / "real.pdf"
    link.symlink_to(real)  # the index is a symlink to a *.pdf inside the folder
    assert real.resolve() in own_files(link)


# --- the writer lock ------------------------------------------------------------------------


@pytest.mark.skipif(not POSIX, reason="directory-as-lock-file is POSIX behaviour")
def test_lock_file_that_cannot_be_opened_is_not_reported_as_busy(tmp_path: Path) -> None:
    with Store(tmp_path / "i.db") as store:
        (tmp_path / "i.db.lock").unlink()  # left by creating the index
        (tmp_path / "i.db.lock").mkdir()
        with pytest.raises(OSError) as info, store.writing():
            pass
        assert not isinstance(info.value, IndexBusyError)


@pytest.mark.skipif(not POSIX, reason="directory-as-lock-file is POSIX behaviour")
def test_lock_file_that_cannot_be_opened_fails_the_progress(root: Path, tmp_path: Path) -> None:
    (tmp_path / "i.db.lock").mkdir()
    progress = Progress()
    with open(os.devnull, "w") as errors:
        thread = _run_thread(
            cli._index_in_background(_args(root, tmp_path / "i.db"), errors, None), progress
        )
        thread.join(10)
    assert progress.state in {"failed", "ready"}, progress.state


@pytest.mark.skipif(not POSIX, reason="directory-as-lock-file is POSIX behaviour")
def test_index_command_with_an_unopenable_lock_file_exits_cleanly(
    root: Path, tmp_path: Path
) -> None:
    index = tmp_path / "i.db"
    with Store(index):
        pass
    (tmp_path / "i.db.lock").unlink()  # left by creating the index
    (tmp_path / "i.db.lock").mkdir()
    assert main(["index", str(root), "--index", str(index)]) == 1


@pytest.mark.skipif(not POSIX, reason="flock is POSIX")
def test_a_failing_flock_that_is_not_contention_is_not_called_busy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(file: Any) -> None:
        raise OSError(errno.ENOLCK, "No locks available")

    with Store(tmp_path / "i.db") as store:
        monkeypatch.setattr(store_module, "_try_lock", broken)
        with pytest.raises(OSError) as info, store.writing():
            pass
        assert not isinstance(info.value, IndexBusyError)


def test_the_lock_is_released_when_the_body_raises(tmp_path: Path) -> None:
    with Store(tmp_path / "i.db") as first:
        with pytest.raises(RuntimeError), first.writing():
            raise RuntimeError("boom")
        with first.writing():  # free again
            pass


def test_the_lock_is_exclusive_between_two_stores_in_one_process(tmp_path: Path) -> None:
    index = tmp_path / "i.db"
    with Store(index) as one, Store(index) as two, one.writing():  # noqa: SIM117
        with pytest.raises(IndexBusyError), two.writing():
            pass


# --- schema rebuild and creation ------------------------------------------------------------


def _make_old(path: Path) -> None:
    with Store(path) as s:
        s._db.execute("PRAGMA user_version = 1")


def test_rebuild_while_a_reader_holds_a_transaction(tmp_path: Path) -> None:
    index = tmp_path / "i.db"
    with Store(index) as s:
        s.put_file(FileRow("x.pdf", "ab", 1, 1, "failed", "e"))
        s._db.execute("PRAGMA user_version = 1")
    reader = sqlite3.connect(index, isolation_level=None)
    reader.execute("BEGIN")
    assert reader.execute("SELECT count(*) FROM files").fetchone()[0] == 1
    try:
        with Store(index) as rebuilt:  # WAL: not blocked by the reader
            assert rebuilt.files() == {}
        assert reader.execute("SELECT count(*) FROM files").fetchone()[0] == 1  # its snapshot
    finally:
        reader.close()


def test_two_openers_of_an_old_index_do_not_wipe_each_other(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    index = tmp_path / "i.db"
    _make_old(index)
    original = Store._rebuild
    state = {"inner": False}

    def racing(self: Store) -> None:
        if not state["inner"]:
            state["inner"] = True
            with Store(index) as other:  # process A: sees v1, rebuilds, indexes a file
                other.put_file(FileRow("a.pdf", "ab", 1, 1, "failed", "e"))
        original(self)  # process B had already read v1 and now rebuilds too

    monkeypatch.setattr(Store, "_rebuild", racing)
    with Store(index) as b:
        assert "a.pdf" in b.files()


def test_stale_wal_beside_a_deleted_index_is_not_replayed(tmp_path: Path) -> None:
    """F5 probe: `_create` no longer deletes `-wal`/`-shm` left by a removed index."""
    index = tmp_path / "i.db"
    store = Store(index)
    store.put_file(FileRow("old.pdf", "ab", 1, 1, "failed", "e"))
    wal = Path(f"{index}-wal")
    assert wal.exists() and wal.stat().st_size > 0
    stale = wal.read_bytes()
    store.close()
    index.unlink()
    for suffix in ("-wal", "-shm"):
        Path(f"{index}{suffix}").unlink(missing_ok=True)
    wal.write_bytes(stale)  # the user deleted only the .db, or a backup restored the -wal
    with Store(index) as fresh:
        assert fresh.files() == {}


def test_create_keeps_the_winner_of_a_race(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    index = tmp_path / "i.db"
    real_link = os.link
    calls = {"n": 0}

    def link(src: Any, dst: Any) -> None:
        if calls["n"] == 0:
            calls["n"] += 1
            Store._create(index)  # another process wins first
            with Store(index) as winner:
                winner.put_file(FileRow("w.pdf", "ab", 1, 1, "failed", "e"))
        real_link(src, dst)

    monkeypatch.setattr(os, "link", link)
    with Store(index) as loser:
        assert "w.pdf" in loser.files()
    assert not list(tmp_path.glob(".*.tmp"))


# --- the indexer thread ---------------------------------------------------------------------


def test_unexpected_error_at_startup_marks_progress_failed(
    root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(*a: Any, **k: Any) -> None:
        raise RuntimeError("unexpected")

    monkeypatch.setattr("askpdf.cli.index_folder", boom)
    progress = Progress()
    with open(os.devnull, "w") as errors:
        thread = _run_thread(
            cli._index_in_background(_args(root, tmp_path / "i.db"), errors, None), progress
        )
        thread.join(10)
    assert not thread.is_alive()
    assert progress.state == "failed", progress.state


def test_unexpected_error_in_a_later_run_does_not_end_watching(
    root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = index_folder
    calls = {"n": 0}

    def flaky(*a: Any, **k: Any) -> Any:
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("unexpected")
        return real(*a, **k)

    monkeypatch.setattr("askpdf.cli.index_folder", flaky)
    progress = Progress()
    index = tmp_path / "i.db"
    with open(os.devnull, "w") as errors:
        thread = _run_thread(cli._index_in_background(_args(root, index), errors, 0.02), progress)
        time.sleep(0.2)
        make_pdf(root / "b.pdf", [text_page("Bravo.")])  # run 2 explodes
        time.sleep(0.3)
        make_pdf(root / "c.pdf", [text_page("Charlie.")])
        deadline = time.monotonic() + 5
        seen: set[str] = set()
        while time.monotonic() < deadline and "c.pdf" not in seen:
            with Store(index) as s:
                seen = set(s.files())
            time.sleep(0.05)
    assert thread.is_alive() and "c.pdf" in seen


def test_progress_recovers_after_a_transient_failure(
    root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = index_folder
    calls = {"n": 0}

    def flaky(*a: Any, **k: Any) -> Any:
        calls["n"] += 1
        if calls["n"] == 2:
            (root / "b.pdf").unlink()  # the new file goes away again: nothing left to index
            raise OSError(errno.EIO, "transient")
        return real(*a, **k)

    monkeypatch.setattr("askpdf.cli.index_folder", flaky)
    progress = Progress()
    with open(os.devnull, "w") as errors:
        _run_thread(
            cli._index_in_background(_args(root, tmp_path / "i.db"), errors, 0.02), progress
        )
        time.sleep(0.2)
        make_pdf(root / "b.pdf", [text_page("Bravo.")])  # run 2 fails; run 3 has nothing to do
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and calls["n"] < 3:
            time.sleep(0.02)
        time.sleep(0.3)
    assert calls["n"] >= 3
    assert progress.state == "ready", progress.state


def test_a_failed_startup_run_never_starts_watching(root: Path, tmp_path: Path) -> None:
    """F8 (design gap, low): a start-up failure (here the wrong folder) ends the thread for
    good; the watcher's own retry only covers later runs. Documented, not asserted as a bug."""
    index = tmp_path / "i.db"
    other = tmp_path / "other"
    make_pdf(other / "o.pdf", [text_page("Other.")])
    assert main(["index", str(other), "--index", str(index)]) == 0
    progress = Progress()
    with open(os.devnull, "w") as errors:
        thread = _run_thread(cli._index_in_background(_args(root, index), errors, 0.02), progress)
        thread.join(10)
    assert not thread.is_alive() and progress.state == "failed"


# --- answering and the pause ----------------------------------------------------------------


def test_an_abandoned_answer_stream_releases_the_indexer(folder: Path) -> None:
    holder: list[Progress] = []

    def indexer(progress: Progress) -> None:
        holder.append(progress)
        progress.update(state="ready")

    chat = ScriptedChat("Passwords need ", "fifteen characters [1].")
    app = create_app(
        root=folder,
        index=folder.parent / "i.db",
        port=PORT,
        token=TOKEN,
        chat_for=lambda q, hits, texts: chat,
        index_folder=indexer,
    )
    with _client(app) as client:
        cid = _conversation(client)
        deadline = time.monotonic() + 5
        while not holder and time.monotonic() < deadline:
            time.sleep(0.01)
        progress = holder[0]
        with client.stream(
            "POST",
            "/api/ask",
            json={"conversation_id": cid, "question": "How long?"},
            headers={"Origin": BASE},
        ) as response:
            next(response.iter_lines())  # the first line, then the client goes away
        del response
        gc.collect()
        assert progress.idle.wait(5), "indexing is paused for ever after an abandoned answer"


def test_a_second_question_while_answering_does_not_unpause_or_pause_twice(
    folder: Path,
) -> None:
    """The busy-rejection path returns before idle.clear(), so it cannot leave it cleared."""
    release = threading.Event()
    chat = ScriptedChat("x [1]", wait=release)
    app = _app(folder, chat)
    with _client(app) as client:
        first = _conversation(client)
        second = _conversation(client)
        out: list[str] = []

        def ask_first() -> None:
            out.append(
                client.post(
                    "/api/ask",
                    json={"conversation_id": first, "question": "passwords fifteen characters"},
                    headers={"Origin": BASE},
                ).text
            )

        thread = threading.Thread(target=ask_first)
        thread.start()
        assert chat.entered.wait(5)
        rejected = client.post(
            "/api/ask",
            json={"conversation_id": second, "question": "passwords fifteen characters"},
            headers={"Origin": BASE},
        )
        assert "another question" in rejected.text
        release.set()
        thread.join(10)


def test_pause_for_answers_has_no_timeout() -> None:
    """F9 (low): the indexer waits on `idle` with no timeout, so an answer that never
    finishes (a hung model) stops indexing for as long as it hangs."""
    progress = Progress()
    progress.idle.clear()
    waiter = threading.Thread(target=progress.pause_for_answers, daemon=True)
    waiter.start()
    waiter.join(0.3)
    assert waiter.is_alive()
    progress.idle.set()
    waiter.join(5)
    assert not waiter.is_alive()


# --- read_bytes -----------------------------------------------------------------------------


@pytest.mark.skipif(not POSIX, reason="needs a FIFO")
def test_read_bytes_times_out_and_leaks_a_daemon_thread_until_the_read_returns(
    tmp_path: Path,
) -> None:
    fifo = tmp_path / "blocked.pdf"
    os.mkfifo(fifo)
    before = {t for t in threading.enumerate() if t.name == "askpdf-read"}
    started = time.monotonic()
    with pytest.raises(OSError) as info:
        read_bytes(fifo, 0.1)
    assert info.value.errno == errno.ETIMEDOUT and time.monotonic() - started < 5
    leaked = [t for t in threading.enumerate() if t.name == "askpdf-read" and t not in before]
    try:
        assert len(leaked) == 1 and leaked[0].daemon and leaked[0].is_alive()
    finally:
        fd = os.open(fifo, os.O_WRONLY)  # unblock the reader so the test leaves nothing behind
        os.write(fd, b"x")
        os.close(fd)
        for thread in leaked:
            thread.join(5)
    assert not any(t.is_alive() for t in leaked)


def test_read_bytes_ignores_a_result_that_arrives_after_the_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gate = threading.Event()
    real = Path.read_bytes

    def slow(self: Path) -> bytes:
        gate.wait(5)
        return real(self)

    target = tmp_path / "f.pdf"
    target.write_bytes(b"data")
    monkeypatch.setattr(Path, "read_bytes", slow)
    with pytest.raises(OSError):
        read_bytes(target, 0.05)
    gate.set()  # the late result lands in a list nobody reads; nothing raises
    time.sleep(0.1)
