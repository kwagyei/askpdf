# ruff: noqa: SIM117
"""Adversarial tests for PR 6: remote guard, key hygiene, streaming, embeddings."""

import socket
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, ClassVar

import numpy as np
import pytest

from askpdf.providers import http
from askpdf.providers.ollama import OllamaChat, OllamaEmbedder
from askpdf.providers.openai_compat import OpenAICompatChat


@contextmanager
def raw_server(reply: Callable[[socket.socket], None]) -> Iterator[str]:
    """A server that answers each connection with arbitrary bytes."""
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(5)
    srv.settimeout(5)

    def run() -> None:
        try:
            while True:
                conn, _ = srv.accept()
                with conn:
                    conn.settimeout(2)
                    try:
                        conn.recv(65536)
                        reply(conn)
                    except OSError:
                        pass
        except OSError:
            pass

    threading.Thread(target=run, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{srv.getsockname()[1]}"
    finally:
        srv.close()


def _send(data: bytes) -> Callable[[socket.socket], None]:
    return lambda conn: conn.sendall(data)


# --- remote guard -----------------------------------------------------------------


@pytest.mark.parametrize(
    "url",
    [
        "http://localhost.evil.com",
        "http://127.0.0.1.evil.com",
        "http://evil.com#@localhost",
        "http://evil.com?@localhost",
        "http://localhost@evil.com",
        "http://127.0.0.1@evil.com:80",
        "http://0.0.0.0:11434",
        "http://[::]:11434",
        "http://[::ffff:8.8.8.8]",
        "http://\u2113ocalhost",
        "http://evil.com/ http://localhost",
    ],
)
def test_guard_refuses_lookalikes(url: str) -> None:
    with pytest.raises(http.ProviderError):
        http.require_local_or_allowed(url, False)


@pytest.mark.parametrize(
    "url",
    ["http://LOCALHOST:1", "http://localhost.:1", "http://[::1]:1", "http://127.5.5.5:1"],
)
def test_guard_accepts_loopback_forms(url: str) -> None:
    http.require_local_or_allowed(url, False)


class _Redirect(BaseHTTPRequestHandler):
    seen: ClassVar[list[dict[str, str]]] = []
    target: ClassVar[str] = ""

    def do_POST(self) -> None:
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        self.send_response(302)
        self.send_header("Location", self.target)
        self.end_headers()

    def do_GET(self) -> None:
        _Redirect.seen.append(dict(self.headers))
        body = b'{"embeddings": []}'
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a: Any) -> None:
        pass


def test_redirect_to_another_host_does_not_carry_the_api_key() -> None:
    """A guard-approved endpoint can bounce the client elsewhere; the key must not follow."""
    final = ThreadingHTTPServer(("127.0.0.1", 0), _Redirect)
    first = ThreadingHTTPServer(("127.0.0.1", 0), _Redirect)
    _Redirect.seen = []
    # "localhost" vs "127.0.0.1" are different hosts to urllib, like remote vs local.
    _Redirect.target = f"http://localhost:{final.server_port}/steal"
    for s in (final, first):
        threading.Thread(target=s.serve_forever, daemon=True).start()
    try:
        http.post_json(
            f"http://127.0.0.1:{first.server_port}/x", {"a": 1}, {"Authorization": "Bearer SECRET"}
        )
    except http.ProviderError:
        pass
    finally:
        final.shutdown()
        first.shutdown()
    assert not any("SECRET" in str(h) for h in _Redirect.seen), _Redirect.seen


def test_redirects_are_not_followed_at_all() -> None:
    final = ThreadingHTTPServer(("127.0.0.1", 0), _Redirect)
    first = ThreadingHTTPServer(("127.0.0.1", 0), _Redirect)
    _Redirect.seen = []
    _Redirect.target = f"http://localhost:{final.server_port}/steal"
    for s in (final, first):
        threading.Thread(target=s.serve_forever, daemon=True).start()
    try:
        with pytest.raises(http.ProviderError):
            http.post_json(f"http://127.0.0.1:{first.server_port}/x", {"a": 1})
    finally:
        final.shutdown()
        first.shutdown()
    assert _Redirect.seen == []


def test_proxy_environment_is_not_used_for_a_loopback_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    got: list[bytes] = []

    def reply(conn: socket.socket) -> None:
        got.append(b"hit")
        conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\n{}")

    with raw_server(reply) as proxy:
        monkeypatch.setenv("http_proxy", proxy)
        monkeypatch.delenv("no_proxy", raising=False)
        monkeypatch.delenv("NO_PROXY", raising=False)
        # a loopback target on a port nobody listens on: only a proxy would answer
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        dead = s.getsockname()[1]
        s.close()
        with pytest.raises(http.ProviderError):
            http.post_json(f"http://127.0.0.1:{dead}/api/embed", {"input": ["secret text"]})
    assert got == []


# --- key hygiene ------------------------------------------------------------------


def test_error_body_that_echoes_the_key_is_not_shown() -> None:
    body = b'{"error": "bad key Bearer SECRET-KEY"}'
    resp = b"HTTP/1.1 401 Unauthorized\r\nContent-Length: %d\r\n\r\n" % len(body) + body
    with raw_server(_send(resp)) as url:
        chat = OpenAICompatChat(url, "m", "SECRET-KEY")
        with pytest.raises(http.ProviderError) as info:
            list(chat.stream("s", "u"))
    assert "SECRET-KEY" not in str(info.value)
    assert "SECRET-KEY" not in repr(chat)


# --- streaming robustness ---------------------------------------------------------


def _chunked_then_drop(conn: socket.socket) -> None:
    conn.sendall(
        b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n"
        b'20\r\n{"message":{"content":"hi"}}\n\r\n'  # then the connection just closes
    )


def test_connection_drop_mid_stream_is_a_provider_error() -> None:
    with raw_server(_chunked_then_drop) as url, pytest.raises(http.ProviderError):
        list(OllamaChat(url).stream("s", "u"))


def test_stream_that_ends_without_done_is_an_error() -> None:
    body = b'{"message":{"content":"half an ans"}}\n'
    resp = b"HTTP/1.1 200 OK\r\nContent-Length: %d\r\n\r\n" % len(body) + body
    with raw_server(_send(resp)) as url, pytest.raises(http.ProviderError):
        list(OllamaChat(url).stream("s", "u"))


def test_sse_that_ends_without_done_is_an_error() -> None:
    body = b'data: {"choices":[{"delta":{"content":"half"}}]}\n\n'
    resp = b"HTTP/1.1 200 OK\r\nContent-Length: %d\r\n\r\n" % len(body) + body
    with raw_server(_send(resp)) as url, pytest.raises(http.ProviderError):
        list(OpenAICompatChat(url, "m", "k").stream("s", "u"))


def test_non_utf8_stream_bytes_are_a_provider_error() -> None:
    body = b'{"message":{"content":"\xff\xfe"}}\n'
    resp = b"HTTP/1.1 200 OK\r\nContent-Length: %d\r\n\r\n" % len(body) + body
    with raw_server(_send(resp)) as url, pytest.raises(http.ProviderError):
        list(OllamaChat(url).stream("s", "u"))


@pytest.mark.parametrize("line", ['{"message": null}', '{"message": "x"}', "[1]", "null", "{"])
def test_odd_ndjson_lines_are_provider_errors(line: str) -> None:
    body = line.encode() + b"\n"
    resp = b"HTTP/1.1 200 OK\r\nContent-Length: %d\r\n\r\n" % len(body) + body
    with raw_server(_send(resp)) as url, pytest.raises(http.ProviderError):
        list(OllamaChat(url).stream("s", "u"))


@pytest.mark.parametrize("data", ["data: {", "data: []", 'data: {"choices": null}', "data: 5"])
def test_odd_sse_events_are_provider_errors(data: str) -> None:
    body = data.encode() + b"\n\n"
    resp = b"HTTP/1.1 200 OK\r\nContent-Length: %d\r\n\r\n" % len(body) + body
    with raw_server(_send(resp)) as url, pytest.raises(http.ProviderError):
        list(OpenAICompatChat(url, "m", "k").stream("s", "u"))


@pytest.mark.parametrize(
    "raw",
    [b"garbage\r\n\r\n", b"HTTP/1.1 200 OK\r\nContent-Length: 50\r\n\r\n{}", b""],
)
def test_malformed_http_is_a_provider_error(raw: bytes) -> None:
    with raw_server(_send(raw)) as url:
        with pytest.raises(http.ProviderError):
            http.post_json(f"{url}/x", {})
        with pytest.raises(http.ProviderError):
            list(http.stream_lines(f"{url}/x", {}))


# --- embedding edge cases ---------------------------------------------------------


def _json_server(payload: bytes) -> Any:
    resp = b"HTTP/1.1 200 OK\r\nContent-Length: %d\r\n\r\n" % len(payload) + payload
    return raw_server(_send(resp))


def test_wrong_dimension_rows_in_one_reply_are_a_provider_error() -> None:
    with _json_server(b'{"embeddings": [[1,0,0],[1,0]]}') as url:
        with pytest.raises(http.ProviderError):
            OllamaEmbedder(url).embed_documents(["a", "b"])


def test_nan_and_inf_vectors_are_rejected() -> None:
    with _json_server(b'{"embeddings": [[NaN, 1.0]]}') as url:
        with pytest.raises(http.ProviderError):
            OllamaEmbedder(url).embed_documents(["a"])


def test_empty_vector_is_rejected() -> None:
    with _json_server(b'{"embeddings": [[]]}') as url, pytest.raises(http.ProviderError):
        OllamaEmbedder(url).embed_documents(["a"])


def test_non_numeric_vector_is_rejected() -> None:
    with _json_server(b'{"embeddings": [["a", "b"]]}') as url:
        with pytest.raises(http.ProviderError):
            OllamaEmbedder(url).embed_documents(["a"])


def test_zero_vector_stays_zero_not_nan() -> None:
    with _json_server(b'{"embeddings": [[0, 0]]}') as url:
        v = OllamaEmbedder(url).embed_documents(["a"])
    assert not np.isnan(v).any()


# --- vectors, snapshot, retrieval and CLI -----------------------------------------

from pathlib import Path  # noqa: E402

from askpdf.cli import main  # noqa: E402
from askpdf.indexer import embed_missing  # noqa: E402
from askpdf.retrieve import RRF_DEPTH, search_dense, search_hybrid  # noqa: E402
from askpdf.snapshot import Snapshot  # noqa: E402
from askpdf.store import ChunkRecord, FileRow, PageRecord, Store  # noqa: E402
from tests import fake_models  # noqa: E402
from tests.fake_models import fake_model_server  # noqa: E402
from tests.pdfs import make_pdf, text_page  # noqa: E402


def _put(store: Store, path: str, sha: str, texts: list[str]) -> None:
    store.put_document(
        FileRow(path, sha, 1, 1, "indexed", None),
        [PageRecord(n, str(n), t, (), False) for n, t in enumerate(texts, 1)],
        [ChunkRecord(n, 0, 0, len(t.split()), t) for n, t in enumerate(texts, 1)],
    )


def _vec(store: Store, rows: list[list[float]]) -> None:
    ids = [i for i, _ in store.chunks_without_vectors()]
    store.put_vectors([(i, np.asarray(r, "<f4").tobytes()) for i, r in zip(ids, rows, strict=True)])


def test_rrf_scores_by_hand(tmp_path: Path) -> None:
    with Store(tmp_path / "i.db") as s:
        # bm25 for "alpha": chunk1 (short, 1 hit) > chunk2 (longer) ; chunk3 no match
        _put(s, "a.pdf", "a" * 64, ["alpha", "alpha pad pad pad pad", "zzz"])
        # dense order for q=[1,0]: chunk3 (1.0), chunk1 (.6), chunk2 (0)
        _vec(s, [[0.6, 0.8], [0.0, 1.0], [1.0, 0.0]])
        snap = Snapshot.load(s)
    q = np.array([1.0, 0.0], np.float32)
    hits = search_hybrid(snap, "alpha", q, k=3, rrf_k=60)
    got = {h.page: h.score for h in hits}
    assert got[1] == pytest.approx(1 / 61 + 1 / 62)  # bm25 rank1 + dense rank2
    assert got[2] == pytest.approx(1 / 62 + 1 / 63)  # bm25 rank2 + dense rank3
    assert got[3] == pytest.approx(1 / 61)  # dense rank1 only
    assert [h.page for h in hits] == [1, 3, 2] or [h.page for h in hits] == [1, 2, 3]
    assert got[1] > got[3] > got[2] or got[1] > got[2]


def test_rrf_depth_cuts_each_list_at_fifty(tmp_path: Path) -> None:
    n = RRF_DEPTH + 5
    with Store(tmp_path / "i.db") as s:
        _put(s, "a.pdf", "a" * 64, [f"word{i}" for i in range(n)])
        _vec(s, [[1.0 - i / 1000, i / 1000] for i in range(n)])
        snap = Snapshot.load(s)
    hits = search_hybrid(snap, "nomatch", np.array([1.0, 0.0], np.float32), k=n)
    assert len(hits) == RRF_DEPTH


def test_dense_ties_are_broken_by_chunk_id(tmp_path: Path) -> None:
    with Store(tmp_path / "i.db") as s:
        _put(s, "a.pdf", "a" * 64, ["x", "y", "z", "w"])
        _vec(s, [[1.0, 0.0]] * 4)
        snap = Snapshot.load(s)
    q = np.array([1.0, 0.0], np.float32)
    assert [h.page for h in search_dense(snap, q, 4)] == [1, 2, 3, 4]


def test_nan_stored_vector_does_not_break_or_win_the_ranking(tmp_path: Path) -> None:
    with Store(tmp_path / "i.db") as s:
        _put(s, "a.pdf", "a" * 64, ["x", "y"])
        _vec(s, [[float("nan"), 1.0], [1.0, 0.0]])
        snap = Snapshot.load(s)
    hits = search_dense(snap, np.array([1.0, 0.0], np.float32), 2)
    assert hits[0].page == 2


def test_blob_of_the_wrong_length_does_not_crash_snapshot(tmp_path: Path) -> None:
    with Store(tmp_path / "i.db") as s:
        _put(s, "a.pdf", "a" * 64, ["x", "y"])
        ids = [i for i, _ in s.chunks_without_vectors()]
        s.put_vectors([(ids[0], np.zeros(4, "<f4").tobytes()), (ids[1], b"\x00\x00\x00")])
        Snapshot.load(s)


def test_removed_and_changed_files_leave_no_orphan_vectors(tmp_path: Path) -> None:
    with Store(tmp_path / "i.db") as s:
        _put(s, "a.pdf", "a" * 64, ["x", "y"])
        _put(s, "b.pdf", "b" * 64, ["z"])
        _vec(s, [[1.0, 0.0]] * 3)
        s.remove_files(["a.pdf"])
        assert len(s.all_vectors()) == 1
        # same path, new content: old document is collected, its vectors with it
        _put(s, "b.pdf", "c" * 64, ["q"])
        assert len(s.all_vectors()) == 0
        assert [t for _, t in s.chunks_without_vectors()] == ["q"]
        ids = {c[0] for c in s.all_chunks()}
        assert set(s.all_vectors()) <= ids


def test_chunk_ids_are_not_reused_after_deletion(tmp_path: Path) -> None:
    """A stale put_vectors for a deleted chunk must not land on a new chunk."""
    with Store(tmp_path / "i.db") as s:
        _put(s, "a.pdf", "a" * 64, ["x"])
        ((old_id, _),) = s.chunks_without_vectors()
        s.remove_files(["a.pdf"])
        _put(s, "b.pdf", "b" * 64, ["y"])
        ((new_id, _),) = s.chunks_without_vectors()
        assert new_id != old_id


class _OneBadChunk:
    name = "fake/bad"

    def embed_documents(self, texts: Any) -> Any:
        raise http.ProviderError("input length exceeds the context length")

    def embed_query(self, text: str) -> Any:
        raise AssertionError


def test_one_unembeddable_chunk_does_not_block_the_rest(tmp_path: Path) -> None:
    class Picky:
        name = "fake/picky"

        def embed_documents(self, texts: Any) -> Any:
            if any("TOOLONG" in t for t in texts):
                raise http.ProviderError("too long")
            return np.ones((len(texts), 2), np.float32)

        def embed_query(self, text: str) -> Any:
            raise AssertionError

    with Store(tmp_path / "i.db") as s:
        _put(s, "a.pdf", "a" * 64, ["ok one", "TOOLONG", "ok two"])
        report = embed_missing(s, Picky())
        assert report.embedded == 2 and len(report.refused) == 1
        assert len(s.all_vectors()) == 2
        assert [t for _, t in s.chunks_without_vectors()] == ["TOOLONG"]


def test_an_unreachable_server_mid_run_keeps_what_was_stored(tmp_path: Path) -> None:
    class Flaky:
        name = "fake/flaky"
        calls = 0

        def embed_documents(self, texts: Any) -> Any:
            Flaky.calls += 1
            if Flaky.calls > 1:
                raise http.UnreachableError("cannot reach 127.0.0.1:11434")
            return np.ones((len(texts), 2), np.float32)

        def embed_query(self, text: str) -> Any:
            raise AssertionError

    from askpdf import indexer

    with Store(tmp_path / "i.db") as s:
        _put(s, "a.pdf", "a" * 64, ["one", "two", "three"])
        original = indexer.EMBED_BATCH
        indexer.EMBED_BATCH = 1
        try:
            with pytest.raises(http.UnreachableError):
                embed_missing(s, Flaky())
        finally:
            indexer.EMBED_BATCH = original
        assert len(s.all_vectors()) == 1


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


def _idx(folder: Path) -> list[str]:
    return ["index", str(folder), "--index", str(folder.parent / "i.db")]


def _srch(folder: Path, *extra: str) -> list[str]:
    return ["search", str(folder), "cookie flag", "--index", str(folder.parent / "i.db"), *extra]


def test_query_dimension_change_is_reported_not_a_traceback(
    folder: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ASKPDF_OLLAMA_URL", "http://127.0.0.1:1")
    with fake_model_server() as (url, _):
        monkeypatch.setenv("ASKPDF_OLLAMA_URL", url)
        assert main(_idx(folder)) == 0
        monkeypatch.setattr(fake_models, "DIM", 8)
        capsys.readouterr()
        for retriever in ("dense", "hybrid"):
            code = main(_srch(folder, "--retriever", retriever))
            assert code in (0, 1), code
            assert "Traceback" not in capsys.readouterr().err


def test_embedding_dimension_change_between_runs_is_not_silent(
    folder: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    with fake_model_server() as (url, _):
        monkeypatch.setenv("ASKPDF_OLLAMA_URL", url)
        main(_idx(folder))
        make_pdf(folder / "b.pdf", [text_page("Extra page about tokens.")])
        monkeypatch.setattr(fake_models, "DIM", 8)  # e.g. Ollama's model was swapped
        capsys.readouterr()
        main(_idx(folder))
        capsys.readouterr()
        main(_srch(folder, "--retriever", "dense"))
        err = capsys.readouterr().err
    # mixed 32- and 8-dim vectors: dense search silently becomes keyword search and
    # claims Ollama was not running
    assert "was not running" not in err


def test_embedding_with_remote_url_sends_nothing(
    folder: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sent: list[str] = []
    monkeypatch.setattr(http, "post_json", lambda *a, **k: sent.append("x"))
    for url in ("http://LOCALHOST.evil.com:11434", "http://[::ffff:1.2.3.4]:1", "http://0.0.0.0:1"):
        monkeypatch.setenv("ASKPDF_OLLAMA_URL", url)
        main(_idx(folder))
    assert sent == []
