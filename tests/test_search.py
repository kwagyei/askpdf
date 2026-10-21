"""Snapshots, page ranking and the `search` command."""

import errno
import sqlite3
import subprocess
import sys
import threading
from collections.abc import Iterator
from pathlib import Path

import pytest

from askpdf.chunking import ChunkParams
from askpdf.cli import main, page_ref
from askpdf.extract import clean
from askpdf.extract.pypdf_ext import PypdfExtractor
from askpdf.indexer import index_folder
from askpdf.retrieve import search_bm25
from askpdf.snapshot import Chunk, Snapshot
from askpdf.store import ChunkRecord, FileRow, PageRecord, Store
from tests.pdfs import make_pdf, text_page


def _put(store: Store, path: str, sha: str, pages: dict[int, list[str]]) -> None:
    """Store a document whose pages hold the given chunk texts."""
    store.put_document(
        FileRow(path, sha, 1, 1, "indexed", None),
        [PageRecord(n, str(n), " ".join(texts), (), False) for n, texts in pages.items()],
        [
            ChunkRecord(n, i, 0, len(text.split()), text)
            for n, texts in pages.items()
            for i, text in enumerate(texts)
        ],
    )


@pytest.fixture
def store(tmp_path: Path) -> Iterator[Store]:
    with Store(tmp_path / "index.db") as store:
        yield store


# --- ranking ------------------------------------------------------------------------------


def test_a_page_scores_as_its_best_chunk_and_appears_once(store: Store) -> None:
    _put(store, "a.pdf", "a" * 64, {1: ["kiwi", "kiwi kiwi kiwi"], 2: ["kiwi and more words"]})
    hits = search_bm25(Snapshot.load(store), "kiwi")
    assert [(h.page, h.best_chunk.ordinal) for h in hits] == [(1, 1), (2, 0)]
    assert hits[0].score > hits[1].score


def test_equal_scores_are_ordered_by_chunk_id(store: Store) -> None:
    _put(store, "b.pdf", "b" * 64, {1: ["fig tree"]})
    _put(store, "a.pdf", "a" * 64, {1: ["fig tree"]})
    hits = search_bm25(Snapshot.load(store), "fig")
    assert hits[0].score == hits[1].score
    assert [h.paths for h in hits] == [("b.pdf",), ("a.pdf",)]  # stored first, lower id


def test_k_limits_pages_not_chunks(store: Store) -> None:
    _put(store, "a.pdf", "a" * 64, {n: ["lime", "lime lime"] for n in range(1, 6)})
    snapshot = Snapshot.load(store)
    assert [h.page for h in search_bm25(snapshot, "lime", k=3)] == [1, 2, 3]
    with pytest.raises(ValueError, match="at least 1"):
        search_bm25(snapshot, "lime", k=0)


def test_pages_without_any_query_word_are_left_out(store: Store) -> None:
    _put(store, "a.pdf", "a" * 64, {1: ["plum"], 2: ["pear"]})
    snapshot = Snapshot.load(store)
    assert [h.page for h in search_bm25(snapshot, "plum")] == [1]
    assert search_bm25(snapshot, "the and of") == []
    assert search_bm25(snapshot, "durian") == []


def test_a_hit_carries_its_label_and_every_path(tmp_path: Path) -> None:
    root = tmp_path / "f"
    make_pdf(
        root / "a.pdf",
        [text_page("Preface."), text_page("Quince jam recipe.")],
        labels=(0, 1, "/r", 1),
    )
    (root / "copy.pdf").write_bytes((root / "a.pdf").read_bytes())
    with Store(tmp_path / "i.db") as store:
        index_folder(root, store, PypdfExtractor(), ChunkParams())
        (hit,) = search_bm25(Snapshot.load(store), "quince")
    assert (hit.page, hit.label, hit.paths) == (2, "ii", ("a.pdf", "copy.pdf"))


def test_a_hyphenated_question_finds_the_spaced_and_hyphenated_forms(store: Store) -> None:
    _put(store, "a.pdf", "a" * 64, {1: ["use multi-factor login"], 2: ["a multi factor setup"]})
    hits = search_bm25(Snapshot.load(store), "multi\u2010factor")
    assert sorted(h.page for h in hits) == [1, 2]


# --- snapshot isolation -------------------------------------------------------------------


def test_an_old_snapshot_keeps_answering_from_its_generation(store: Store) -> None:
    _put(store, "a.pdf", "a" * 64, {1: ["mango"]})
    old = Snapshot.load(store)
    _put(store, "b.pdf", "b" * 64, {1: ["mango mango"]})
    store.remove_files(["a.pdf"])
    assert [h.paths for h in search_bm25(old, "mango")] == [("a.pdf",)]
    assert not old.is_current(store)
    new = Snapshot.load(store)
    assert new.is_current(store)
    assert [h.paths for h in search_bm25(new, "mango")] == [("b.pdf",)]


def test_a_commit_during_loading_is_not_seen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The snapshot reads several tables. A writer committing between those reads must
    # not leak into it: everything comes from one generation.
    with Store(tmp_path / "i.db") as reader, Store(tmp_path / "i.db") as writer:
        _put(writer, "a.pdf", "a" * 64, {1: ["apricot"]})
        real_all_chunks = reader.all_chunks

        def all_chunks_after_a_commit() -> list[tuple[int, str, int, int, str]]:
            _put(writer, "b.pdf", "b" * 64, {1: ["apricot"], 2: ["apricot"]})
            return real_all_chunks()

        monkeypatch.setattr(reader, "all_chunks", all_chunks_after_a_commit)
        snapshot = Snapshot.load(reader)
        monkeypatch.undo()
        assert snapshot.generation == 1
        assert {c.sha256 for c in snapshot.chunks} == {"a" * 64}
        assert set(snapshot.paths) == {"a" * 64}
        assert Snapshot.load(reader).generation == 2


def test_snapshots_taken_while_indexing_are_always_whole_documents(tmp_path: Path) -> None:
    pages_per_doc = 3
    with Store(tmp_path / "i.db") as writer, Store(tmp_path / "i.db") as reader:
        done = threading.Event()

        def write() -> None:
            for n in range(40):
                pages = {p: [f"doc{n} page{p}"] for p in range(1, pages_per_doc + 1)}
                _put(writer, f"{n}.pdf", f"{n:064d}", pages)
            done.set()

        thread = threading.Thread(target=write)
        thread.start()
        seen = 0
        while not done.is_set() or seen == 0:
            snapshot = Snapshot.load(reader)
            per_doc: dict[str, int] = {}
            for chunk in snapshot.chunks:
                per_doc[chunk.sha256] = per_doc.get(chunk.sha256, 0) + 1
            assert all(count == pages_per_doc for count in per_doc.values())
            assert set(per_doc) == set(snapshot.paths)
            assert len(per_doc) == snapshot.generation
            seen += 1
        thread.join()


def test_a_snapshot_cannot_be_modified(store: Store) -> None:
    _put(store, "a.pdf", "a" * 64, {1: ["date"]})
    snapshot = Snapshot.load(store)
    with pytest.raises(TypeError):
        snapshot.paths["x"] = ()  # type: ignore[index]
    with pytest.raises(AttributeError):
        snapshot.generation = 9  # type: ignore[misc]
    assert isinstance(snapshot.chunks[0], Chunk)


# --- the command --------------------------------------------------------------------------


@pytest.fixture
def folder(tmp_path: Path) -> Path:
    root = tmp_path / "pdfs"
    make_pdf(
        root / "guide.pdf",
        [text_page("Front matter."), text_page("Passkeys resist phishing.")],
        labels=(0, 1, "/r", 1),
    )
    make_pdf(root / "notes.pdf", [text_page("Passwords are weak.")])
    return root


def _search(folder: Path, *extra: str) -> list[str]:
    return ["search", str(folder), *extra, "--index", str(folder.parent / "i.db")]


def test_search_prints_ranked_pages(folder: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["index", str(folder), "--index", str(folder.parent / "i.db")]) == 0
    capsys.readouterr()
    assert main(_search(folder, "phishing-resistant passkeys")) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].startswith("1. guide.pdf  p. ii (PDF page 2)  score ")
    assert lines[1] == "   Passkeys resist phishing."
    assert len(lines) == 2


def test_search_with_no_match_says_so(folder: Path, capsys: pytest.CaptureFixture[str]) -> None:
    main(["index", str(folder), "--index", str(folder.parent / "i.db")])
    capsys.readouterr()
    assert main(_search(folder, "the of and")) == 0
    captured = capsys.readouterr()
    assert captured.out == "" and "no page matches" in captured.err


def test_search_before_indexing_points_to_index(
    folder: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(_search(folder, "anything")) == 1
    assert "run: askpdf index" in capsys.readouterr().err


def test_search_refuses_another_folders_index(
    folder: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    main(["index", str(folder), "--index", str(folder.parent / "i.db")])
    other = tmp_path / "other"
    other.mkdir()
    capsys.readouterr()
    assert main(_search(other, "x")) == 1
    assert "is the index of" in capsys.readouterr().err


@pytest.mark.parametrize("k", ["0", "-1", "two"])
def test_k_must_be_a_positive_number(folder: Path, k: str) -> None:
    with pytest.raises(SystemExit) as exc:
        main(_search(folder, "x", "-k", k))
    assert exc.value.code == 2


def test_long_chunks_are_shortened_to_one_line(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "pdfs"
    make_pdf(root / "long.pdf", [text_page(*(["walnut " * 12] * 4))])
    index = str(tmp_path / "i.db")
    main(["index", str(root), "--index", index])
    capsys.readouterr()
    main(["search", str(root), "walnut", "--index", index])
    snippet = capsys.readouterr().out.splitlines()[1]
    assert snippet.endswith("…") and len(snippet) == 3 + 160


def test_page_ref() -> None:
    assert page_ref(20, "9") == "p. 9 (PDF page 20)"
    assert page_ref(3, "3") == "page 3"


# --- reading never rebuilds (tester PR 3, F3) ----------------------------------------------


def _indexed(folder: Path) -> str:
    index = str(folder.parent / "i.db")
    assert main(["index", str(folder), "--index", index]) == 0
    return index


def test_search_refuses_an_index_from_another_schema_without_wiping_it(
    folder: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    index = _indexed(folder)
    with sqlite3.connect(index) as db:
        db.execute("PRAGMA user_version = 0")
    capsys.readouterr()
    assert main(_search(folder, "passkeys")) == 1
    assert "another version of askpdf; run: askpdf index" in capsys.readouterr().err
    assert main(["docs", str(folder), "--index", index]) == 1
    with sqlite3.connect(index) as db:
        assert db.execute("SELECT count(*) FROM chunks").fetchone()[0] > 0  # untouched


def test_search_refuses_an_index_built_with_other_settings(
    folder: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    index = _indexed(folder)
    monkeypatch.setattr(clean, "CLEANING_VERSION", clean.CLEANING_VERSION + 1)
    capsys.readouterr()
    assert main(_search(folder, "passkeys")) == 1
    assert "other settings" in capsys.readouterr().err
    assert main(["docs", str(folder), "--index", index]) == 0  # still lists, with a note
    captured = capsys.readouterr()
    assert "guide.pdf" in captured.out and "settings changed" in captured.err


def test_invisible_characters_in_a_question_do_not_split_its_words(store: Store) -> None:
    _put(store, "a.pdf", "a" * 64, {1: ["passkeys"]})
    snapshot = Snapshot.load(store)
    for question in ["pass­keys", "pass‍keys", "pass​keys"]:
        assert [h.page for h in search_bm25(snapshot, question)] == [1], repr(question)


def test_a_closed_pipe_exits_quietly(folder: Path) -> None:
    # Like `askpdf search ... | head -0`: the reader is gone before anything is written.
    index = _indexed(folder)
    argv = [sys.executable, "-m", "askpdf", "search", str(folder), "passkeys", "--index", index]
    proc = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE)  # noqa: S603
    assert proc.stdout is not None and proc.stderr is not None
    proc.stdout.close()  # Python takes far longer to start than this takes
    stderr = proc.stderr.read().decode()
    proc.stderr.close()
    assert proc.wait(timeout=60) == 1
    assert stderr == ""


@pytest.mark.parametrize(
    ("error", "platform", "closed"),
    [
        (BrokenPipeError(), "linux", True),
        (OSError(errno.EINVAL, "Invalid argument"), "win32", True),
        (OSError(errno.EINVAL, "Invalid argument"), "linux", False),  # a real error there
        (OSError(errno.ENOSPC, "No space left"), "win32", False),
    ],
)
def test_what_counts_as_a_closed_pipe(
    error: OSError, platform: str, closed: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    from askpdf.cli import _closed_pipe

    monkeypatch.setattr(sys, "platform", platform)
    assert _closed_pipe(error) is closed


def test_other_write_errors_are_not_swallowed(monkeypatch: pytest.MonkeyPatch) -> None:
    import askpdf.cli as cli

    def full(*args: object) -> int:
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setitem(cli.main.__globals__, "cmd_docs", full)
    with pytest.raises(OSError, match="No space"):
        cli.main(["docs", "."])
