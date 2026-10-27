"""Vectors in the index, dense and hybrid retrieval, and the CLI around them."""

import sqlite3
from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np
import pytest

from askpdf.cli import main
from askpdf.indexer import embed_missing
from askpdf.providers.base import Matrix, normalise
from askpdf.providers.ollama import OllamaEmbedder
from askpdf.retrieve import search_bm25, search_dense, search_hybrid
from askpdf.snapshot import Snapshot
from askpdf.store import ChunkRecord, FileRow, PageRecord, Store
from tests.fake_models import fake_model_server, fake_vector
from tests.pdfs import make_pdf, text_page


class WordEmbedder:
    name = "fake/words"

    def embed_documents(self, texts: Sequence[str]) -> Matrix:
        return normalise([fake_vector(t) for t in texts])

    def embed_query(self, text: str) -> Matrix:
        return normalise([fake_vector(text)]).reshape(-1)


def _put(store: Store, path: str, sha: str, pages: dict[int, list[str]]) -> None:
    store.put_document(
        FileRow(path, sha, 1, 1, "indexed", None),
        [PageRecord(n, str(n), " ".join(t), (), False) for n, t in pages.items()],
        [
            ChunkRecord(n, i, 0, len(x.split()), x)
            for n, t in pages.items()
            for i, x in enumerate(t)
        ],
    )


@pytest.fixture
def store(tmp_path: Path) -> Iterator[Store]:
    with Store(tmp_path / "i.db") as s:
        _put(s, "a.pdf", "a" * 64, {1: ["apple banana"], 2: ["cherry date"], 3: ["apple cherry"]})
        yield s


def test_missing_vectors_are_filled_in_and_a_second_run_does_nothing(store: Store) -> None:
    assert embed_missing(store, WordEmbedder()) == 3
    assert embed_missing(store, WordEmbedder()) == 0
    snapshot = Snapshot.load(store)
    assert snapshot.fully_embedded and snapshot.vectors.shape == (3, 32)
    assert not snapshot.vectors.flags.writeable


def test_a_partly_embedded_index_is_marked(store: Store) -> None:
    first = store.chunks_without_vectors()[0]
    store.put_vectors([(first[0], np.ones(32, np.float32).tobytes())])
    snapshot = Snapshot.load(store)
    assert snapshot.embedded.tolist() == [True, False, False] and not snapshot.fully_embedded


def test_vectors_for_deleted_chunks_are_skipped(store: Store) -> None:
    store.put_vectors([(999, np.ones(32, np.float32).tobytes())])
    assert store.all_vectors() == {}


def test_dense_ranks_by_cosine_and_ignores_unembedded_chunks(store: Store) -> None:
    embed_missing(store, WordEmbedder())
    snapshot = Snapshot.load(store)
    hits = search_dense(snapshot, WordEmbedder().embed_query("cherry date"), k=3)
    assert hits[0].page == 2 and hits[0].score == pytest.approx(1.0)
    with pytest.raises(ValueError, match="does not match"):
        search_dense(snapshot, np.ones(5, np.float32))


def test_hybrid_fuses_both_rankings(store: Store) -> None:
    embed_missing(store, WordEmbedder())
    snapshot = Snapshot.load(store)
    q = "apple"
    hybrid = search_hybrid(snapshot, q, WordEmbedder().embed_query(q), k=3, rrf_k=60)
    bm25_pages = [h.page for h in search_bm25(snapshot, q, k=3)]
    assert {h.page for h in hybrid} >= set(bm25_pages)
    # Each chunk's fused score is the sum of 1 / (60 + rank) over the two rankings.
    assert all(0 < h.score <= 2 / 61 for h in hybrid)


# --- the command line ---------------------------------------------------------------------


@pytest.fixture
def folder(tmp_path: Path) -> Path:
    make_pdf(
        tmp_path / "f" / "a.pdf",
        [
            text_page("Passwords need fifteen characters."),
            text_page("Cookies need the Secure flag."),
        ],
    )
    return tmp_path / "f"


def _index(folder: Path) -> list[str]:
    return ["index", str(folder), "--index", str(folder.parent / "i.db")]


def _search(folder: Path, *extra: str) -> list[str]:
    return ["search", str(folder), "cookie flag", "--index", str(folder.parent / "i.db"), *extra]


def test_index_without_ollama_falls_back_to_keywords(
    folder: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(_index(folder)) == 0
    captured = capsys.readouterr()
    assert "semantic search unavailable" in captured.err and "chunks embedded" not in captured.out
    assert main(_search(folder, "--retriever", "hybrid")) == 0
    captured = capsys.readouterr()
    assert "using keyword search" in captured.err and "a.pdf" in captured.out


def test_index_with_ollama_embeds_and_dense_search_uses_it(
    folder: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    with fake_model_server() as (url, _):
        monkeypatch.setenv("ASKPDF_OLLAMA_URL", url)
        assert main(_index(folder)) == 0
        assert "2 chunks embedded" in capsys.readouterr().out
        assert main(_index(folder)) == 0
        assert "0 chunks embedded" in capsys.readouterr().out  # nothing left to do
        for retriever in ("dense", "hybrid"):
            assert main(_search(folder, "--retriever", retriever, "-k", "1")) == 0
            captured = capsys.readouterr()
            assert "page 2" in captured.out and "keyword" not in captured.err
    with sqlite3.connect(folder.parent / "i.db") as db:
        assert db.execute("SELECT count(*) FROM vectors").fetchone()[0] == 2


def test_ollama_stopping_after_indexing_falls_back(
    folder: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    with fake_model_server() as (url, _):
        monkeypatch.setenv("ASKPDF_OLLAMA_URL", url)
        main(_index(folder))
    capsys.readouterr()
    assert main(_search(folder, "--retriever", "dense")) == 0  # the server is gone now
    assert "using keyword search" in capsys.readouterr().err


def test_a_remote_ollama_needs_permission(
    folder: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ASKPDF_OLLAMA_URL", "https://models.example.com")
    assert main(_index(folder)) == 0
    assert "--allow-remote" in capsys.readouterr().err


def test_the_embedder_is_pinned(folder: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from askpdf import indexer

    main(_index(folder))
    monkeypatch.setattr(indexer, "EMBEDDER", "ollama/another-model")
    monkeypatch.setattr(indexer.pins, "__defaults__", ("ollama/another-model",))
    assert main(_search(folder)) == 1  # built with other settings
    assert isinstance(OllamaEmbedder().name, str)
