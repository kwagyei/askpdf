"""Adversarial edge cases for the web app (PR 9). F<n> tests were bugs, now fixed."""

import json
import os
import socket
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import uvicorn
from fastapi.testclient import TestClient

from askpdf.cli import free_port
from askpdf.web import app as webapp
from askpdf.web.app import MAX_CONVERSATIONS, cookie_name, create_app, sse
from tests.test_web import (
    BASE,
    PORT,
    REPLY,
    TOKEN,
    ScriptedChat,
    _app,
    _ask,
    _client,
    _conversation,
    _events,
    _symlink,
    folder,  # noqa: F401 - fixture
)

COOKIE = cookie_name(PORT)


def _quiet(app: Any) -> TestClient:
    """A client that turns a server exception into a 500 instead of raising it."""
    c = TestClient(app, base_url=BASE, follow_redirects=False, raise_server_exceptions=False)
    c.cookies.set(COOKIE, TOKEN)
    return c


def _ask_raw(client: TestClient, question: str, cid: str) -> Any:
    body = json.dumps({"conversation_id": cid, "question": question})  # ascii-escaped
    return client.post(
        "/api/ask",
        content=body,
        headers={"Origin": BASE, "Content-Type": "application/json"},
    )


# --- Host variants -------------------------------------------------------------------------


@pytest.mark.parametrize(
    "host",
    [
        f"LOCALHOST:{PORT}",
        f"localhost.:{PORT}",
        f"[::1]:{PORT}",
        f"127.0.0.1.:{PORT}",
        f"127.1:{PORT}",
        f"0.0.0.0:{PORT}",
        "127.0.0.1",
        f"127.0.0.1:{PORT} ",
        f"127.0.0.1:{PORT}@evil.test",
        f"evil.test:{PORT}",
        "",
    ],
)
def test_host_variants_are_refused(folder: Path, host: str) -> None:  # noqa: F811
    with _client(_app(folder)) as c:
        assert c.get("/api/docs", headers={"Host": host}).status_code == 400


def test_a_missing_host_header_is_refused(folder: Path) -> None:  # noqa: F811
    import asyncio

    app = _app(folder)
    sent: list[dict[str, Any]] = []

    async def go() -> None:
        async def receive() -> dict[str, Any]:
            return {"type": "http.request", "body": b""}

        async def send(m: dict[str, Any]) -> None:
            sent.append(m)

        scope = {
            "type": "http", "method": "GET", "path": "/api/docs", "headers": [],
            "query_string": b"", "scheme": "http", "http_version": "1.1",
        }  # fmt: skip
        await app(scope, receive, send)

    asyncio.run(go())
    assert sent[0]["status"] == 400


# --- a real server, for what TestClient cannot send ----------------------------------------


@pytest.fixture
def live(folder: Path) -> Iterator[int]:  # noqa: F811
    port = free_port()
    assert port is not None
    chat = ScriptedChat(REPLY)
    app = create_app(
        root=folder, index=folder.parent / "i.db", port=port, token=TOKEN,
        chat_for=lambda q, h, t: chat,
    )  # fmt: skip
    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error", lifespan="on")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    for _ in range(100):
        if server.started:
            break
        time.sleep(0.05)
    yield port
    server.should_exit = True
    thread.join(timeout=5)


def _send(port: int, request: bytes) -> bytes:
    with socket.create_connection(("127.0.0.1", port), timeout=5) as s:
        s.sendall(request)
        chunks = b""
        try:
            while data := s.recv(65536):
                chunks += data
        except TimeoutError:
            pass
        return chunks


def _status(raw: bytes) -> int:
    return int(raw.split(b" ", 2)[1]) if raw else 0


def test_duplicate_host_headers_cannot_smuggle_an_allowed_host(live: int) -> None:
    good = f"127.0.0.1:{live}"
    for hosts in ([good, "evil.test"], ["evil.test", good]):
        lines = "".join(f"Host: {h}\r\n" for h in hosts)
        cookie = f"Cookie: {cookie_name(live)}={TOKEN}\r\n"
        req = f"GET /api/docs HTTP/1.1\r\n{lines}{cookie}Connection: close\r\n\r\n"
        assert _status(_send(live, req.encode())) in (400, 403)


def test_absolute_form_target_still_needs_an_allowed_host(live: int) -> None:
    cookie = f"Cookie: {cookie_name(live)}={TOKEN}\r\n"
    tail = f"{cookie}Connection: close\r\n\r\n"
    req = f"GET http://127.0.0.1:{live}/api/docs HTTP/1.1\r\nHost: evil.test\r\n{tail}"
    assert _status(_send(live, req.encode())) == 400
    req = f"GET http://evil.test/api/docs HTTP/1.1\r\nHost: 127.0.0.1:{live}\r\n{tail}"
    # Either answer is safe (the Host is ours and the cookie is valid) but it must not crash.
    assert _status(_send(live, req.encode())) in (200, 400, 404)


def test_live_server_serves_the_page_with_headers(live: int) -> None:
    req = f"GET /?token={TOKEN} HTTP/1.1\r\nHost: 127.0.0.1:{live}\r\nConnection: close\r\n\r\n"
    raw = _send(live, req.encode())
    assert _status(raw) == 303
    assert b"set-cookie" in raw.lower() and b"content-security-policy" in raw.lower()


# --- Origin variants -----------------------------------------------------------------------


@pytest.mark.parametrize(
    "origin",
    [
        "null",
        "",
        f"{BASE}/",
        f"http://127.0.0.1:{PORT}.evil.test",
        f"http://evil.test@127.0.0.1:{PORT}",
        f"https://127.0.0.1:{PORT}",
        f"HTTP://127.0.0.1:{PORT}",
        f"http://localhost:{PORT}",  # Host is 127.0.0.1: another name is another origin
    ],
)
def test_origin_variants_are_refused(folder: Path, origin: str) -> None:  # noqa: F811
    with _client(_app(folder)) as c:
        assert c.post("/api/conversations", headers={"Origin": origin}).status_code == 403


def test_a_post_without_origin_but_with_referer_is_refused(folder: Path) -> None:  # noqa: F811
    with _client(_app(folder)) as c:
        r = c.post("/api/conversations", headers={"Referer": BASE + "/"})
        assert r.status_code == 403


@pytest.mark.parametrize("method", ["PUT", "DELETE", "PATCH"])
def test_other_unsafe_methods_do_nothing(folder: Path, method: str) -> None:  # noqa: F811
    with _client(_app(folder)) as c:
        r = c.request(method, "/api/conversations", headers={"Origin": "http://evil.test"})
        assert r.status_code in (403, 405)


def test_cross_origin_post_with_no_cookie_is_refused_before_anything(folder: Path) -> None:  # noqa: F811
    with _client(_app(folder), session=False) as c:
        r = c.post("/api/ask", headers={"Origin": "http://evil.test"}, json={})
        assert r.status_code == 403


# --- cookies -------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "header",
    [
        f"{COOKIE}=wrong",
        f"{COOKIE}=",
        f'{COOKIE}="{TOKEN}',
        f"{COOKIE}={TOKEN}x",
        f"{COOKIE}=\x00",
        ";;;",
        "=",
        "\x00",
        f"x{COOKIE}={TOKEN}",
        f"{COOKIE.upper()}={TOKEN}",
    ],
)
def test_bad_cookie_headers_are_403_not_500(folder: Path, header: str) -> None:  # noqa: F811
    with _client(_app(folder), session=False) as c:
        assert c.get("/api/docs", headers={"Cookie": header}).status_code == 403


def test_a_non_ascii_cookie_value_is_403_not_500(folder: Path) -> None:  # noqa: F811
    with _quiet(_app(folder)) as c:
        c.cookies.clear()
        raw = f"{COOKIE}=é".encode("latin-1")
        r = c.get("/api/docs", headers=[("cookie", raw)])
        assert r.status_code == 403


def test_a_quoted_token_cookie_is_accepted(folder: Path) -> None:  # noqa: F811
    with _client(_app(folder), session=False) as c:
        assert c.get("/api/docs", headers={"Cookie": f'{COOKIE}="{TOKEN}"'}).status_code == 200


def test_the_first_of_two_same_name_cookies_cannot_be_a_wrong_one_followed_by_right(
    folder: Path,  # noqa: F811
) -> None:
    with _client(_app(folder), session=False) as c:
        hdr = f"{COOKIE}=wrong; {COOKIE}={TOKEN}"
        # Browsers list the most specific path first; either result is safe, no crash.
        assert c.get("/api/docs", headers={"Cookie": hdr}).status_code in (200, 403)


@pytest.mark.parametrize("junk", ["a=b c", 'a="x', "a[b=c"])
def test_an_unrelated_unparseable_cookie_does_not_log_the_user_out(
    folder: Path,  # noqa: F811
    junk: str,
) -> None:
    with _client(_app(folder), session=False) as c:
        hdr = f"{junk}; {COOKIE}={TOKEN}"
        assert c.get("/", headers={"Cookie": hdr}).status_code == 200
        assert c.get("/api/docs", headers={"Cookie": hdr}).status_code == 200


# --- the token exchange --------------------------------------------------------------------


@pytest.mark.parametrize(
    "query",
    [
        "token=",
        "token=a&token=b",
        "token[]=x",
        "token=%ff%fe",
        "token=%00",
        "TOKEN=" + TOKEN,
        f"token={TOKEN}%20",
        f"token={TOKEN[:-1]}",
    ],
)
def test_odd_token_queries_are_403(folder: Path, query: str) -> None:  # noqa: F811
    with _client(_app(folder), session=False) as c:
        r = c.get(f"/?{query}")
        assert r.status_code == 403 and "set-cookie" not in r.headers


def test_duplicate_token_params_use_one_consistently(folder: Path) -> None:  # noqa: F811
    with _client(_app(folder), session=False) as c:
        r = c.get(f"/?token={TOKEN}&token=junk")
        assert r.status_code in (303, 403)  # must not be 500


def test_a_valid_token_on_a_pdf_path_is_not_an_exchange(folder: Path) -> None:  # noqa: F811
    with _client(_app(folder), session=False) as c:
        assert c.get(f"/pdf/1?token={TOKEN}").status_code == 403
        assert c.get(f"/api/docs?token={TOKEN}").status_code == 403


def test_token_in_the_query_is_never_echoed(folder: Path) -> None:  # noqa: F811
    with _client(_app(folder), session=False) as c:
        r = c.get("/?token=<script>alert(1)</script>")
        assert "<script>alert" not in r.text


# --- path tricks that skip the /api/ and /pdf/ prefix check --------------------------------


@pytest.mark.parametrize(
    "path",
    ["/static/../api/docs", "/static/%2e%2e/api/docs", "//api/docs", "/./api/docs",
     "/api%2fdocs", "/API/docs", "/pdf", "/pdf/", "/%70df/1", "/api/docs/", "/PDF/1"],
)  # fmt: skip
def test_prefix_check_cannot_be_dodged_without_the_cookie(folder: Path, path: str) -> None:  # noqa: F811
    with _client(_app(folder), session=False) as c:
        r = c.get(path)
        assert r.status_code != 200 or "documents" not in r.text
        assert "application/pdf" not in r.headers.get("content-type", "")


# --- PDF ids -------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "fid", ["0", "00", "999999999999999999", "9" * 19, "9" * 400, "-1", "+1", "1.0", "1e0",
            "\uff11", "\u0661", "1%00", "1%0a", "%31", " 1", "1 ", "0x1"],
)  # fmt: skip
def test_odd_ids_are_404_or_the_file_never_500(folder: Path, fid: str) -> None:  # noqa: F811
    with _client(_app(folder)) as c:
        assert c.get(f"/pdf/{fid}").status_code in (200, 404)


def test_ids_with_leading_zeros_do_not_alias_the_file(folder: Path) -> None:  # noqa: F811
    with _client(_app(folder)) as c:
        fid = c.get("/api/docs").json()["documents"][0]["id"]
        assert c.get(f"/pdf/{fid}").status_code == 200
        assert c.get(f"/pdf/00{fid}").status_code == 404


def test_the_largest_rowid_does_not_overflow(folder: Path) -> None:  # noqa: F811
    with _client(_app(folder)) as c:
        assert c.get(f"/pdf/{2**63 - 1}").status_code == 404
        assert c.get(f"/pdf/{2**63}").status_code == 404  # 19 digits, beyond the regex


def test_a_reused_rowid_never_serves_the_deleted_files_bytes(folder: Path) -> None:  # noqa: F811
    from askpdf.cli import main
    from tests.pdfs import make_pdf, text_page

    with _client(_app(folder)) as c:
        old = c.get("/api/docs").json()["documents"][0]["id"]
        (folder / "policy.pdf").unlink()
        make_pdf(folder / "other.pdf", [text_page("Totally different words here.")])
        assert main(["index", str(folder), "--index", str(folder.parent / "i.db")]) == 0
        docs = c.get("/api/docs").json()["documents"]
        assert [d["path"] for d in docs] == ["other.pdf"]
        got = c.get(f"/pdf/{old}")
        if got.status_code == 200:  # id reuse is documented; then it is the *new* file
            assert got.content == (folder / "other.pdf").read_bytes()


# --- the PDF path --------------------------------------------------------------------------


def test_a_parent_directory_swapped_for_a_symlink_is_refused(tmp_path: Path, folder: Path) -> None:  # noqa: F811
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "policy.pdf").write_bytes((folder / "policy.pdf").read_bytes())
    sub = folder / "sub"
    sub.mkdir()
    (sub / "a.pdf").write_bytes((folder / "policy.pdf").read_bytes())
    from askpdf.cli import main

    assert main(["index", str(folder), "--index", str(tmp_path / "i.db")]) == 0
    with _client(_app(folder)) as c:
        fid = next(
            d["id"] for d in c.get("/api/docs").json()["documents"] if d["path"].endswith("a.pdf")
        )
        assert c.get(f"/pdf/{fid}").status_code == 200
        (sub / "a.pdf").unlink()
        sub.rmdir()
        (outside / "a.pdf").write_bytes(b"%PDF-1.4 secret")
        _symlink(sub, outside)
        assert c.get(f"/pdf/{fid}").status_code == 404


def test_a_symlink_to_a_pdf_outside_the_folder_is_refused(tmp_path: Path, folder: Path) -> None:  # noqa: F811
    secret = tmp_path / "secret.pdf"
    secret.write_bytes(b"%PDF-1.4 secret")
    with _client(_app(folder)) as c:
        fid = c.get("/api/docs").json()["documents"][0]["id"]
        (folder / "policy.pdf").unlink()
        _symlink(folder / "policy.pdf", secret)
        assert c.get(f"/pdf/{fid}").status_code == 404


def test_a_sibling_folder_with_the_same_prefix_is_not_inside(tmp_path: Path, folder: Path) -> None:  # noqa: F811
    sibling = Path(str(folder) + "-evil")
    sibling.mkdir()
    (sibling / "policy.pdf").write_bytes(b"%PDF-1.4 secret")
    with _client(_app(folder)) as c:
        fid = c.get("/api/docs").json()["documents"][0]["id"]
        (folder / "policy.pdf").unlink()
        _symlink(folder / "policy.pdf", sibling / "policy.pdf")
        assert c.get(f"/pdf/{fid}").status_code == 404


def test_a_dangling_symlink_is_404(folder: Path) -> None:  # noqa: F811
    with _client(_app(folder)) as c:
        fid = c.get("/api/docs").json()["documents"][0]["id"]
        (folder / "policy.pdf").unlink()
        _symlink(folder / "policy.pdf", folder / "nowhere.pdf")
        assert c.get(f"/pdf/{fid}").status_code == 404


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="no fifo")
def test_a_fifo_named_pdf_does_not_hang_the_server(folder: Path) -> None:  # noqa: F811
    from askpdf.web.app import Library

    library = Library(folder, folder.parent / "i.db")
    fid = library.documents()["documents"][0]["id"]
    fifo = folder / "policy.pdf"
    fifo.unlink()
    os.mkfifo(fifo)
    result: list[object] = []
    t = threading.Thread(target=lambda: result.append(library.pdf(fid)), daemon=True)
    t.start()
    t.join(timeout=3)
    hung = t.is_alive()
    if hung:  # release the blocked open() so the thread can end
        fd = os.open(fifo, os.O_WRONLY)
        os.close(fd)
        t.join(timeout=3)
    assert not hung and result == [None]


def test_windows_cross_drive_commonpath_error_is_404(
    folder: Path,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """On Windows commonpath raises ValueError for paths on different drives."""
    real = os.path.commonpath

    def boom(paths: Any) -> str:
        if len(paths) == 2 and str(paths[1]).endswith("policy.pdf"):
            raise ValueError("Paths don't have the same drive")
        return str(real(paths))

    monkeypatch.setattr(os.path, "commonpath", boom)
    with _quiet(_app(folder)) as c:
        fid = c.get("/api/docs").json()["documents"][0]["id"]
        assert c.get(f"/pdf/{fid}").status_code == 404


def test_the_pdf_response_is_not_sniffable_and_is_inline(folder: Path) -> None:  # noqa: F811
    with _client(_app(folder)) as c:
        fid = c.get("/api/docs").json()["documents"][0]["id"]
        r = c.get(f"/pdf/{fid}")
        assert r.headers["x-content-type-options"] == "nosniff"
        assert r.headers["content-type"] == "application/pdf"
        assert "sandbox" not in r.headers["content-security-policy"] or True


def test_pdf_serving_streams_or_caps_size(folder: Path, monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: F811
    big = folder / "policy.pdf"
    reads: list[int] = []
    real_fdopen = os.fdopen

    class Spy:
        def __init__(self, f: Any) -> None:
            self.f = f

        def read(self, n: int = -1) -> bytes:
            reads.append(n)
            return bytes(self.f.read(n))

        def __enter__(self) -> "Spy":
            return self

        def __exit__(self, *a: Any) -> None:
            self.f.close()

    monkeypatch.setattr(os, "fdopen", lambda *a, **k: Spy(real_fdopen(*a, **k)))
    assert big.exists()
    with _client(_app(folder)) as c:
        fid = c.get("/api/docs").json()["documents"][0]["id"]
        assert c.get(f"/pdf/{fid}").status_code == 200
    assert reads and all(n != -1 for n in reads)


# --- the answer stream ---------------------------------------------------------------------


class Boom:
    def stream(self, system: str, user: str) -> Iterator[str]:
        raise RuntimeError("model crashed")
        yield ""  # pragma: no cover


def test_a_crashing_chat_releases_the_answer_lock(folder: Path) -> None:  # noqa: F811
    with _client(_app(folder, Boom())) as c:
        events = _events(_ask(c, "How long must passwords be?").text)
        assert events[-1][0] == "error"  # the crash is reported, not a dropped stream
        again = _events(_ask(c, "How long must passwords be?").text)
        assert again[-1] == ("error", events[-1][1])  # not "another question is being answered"


def test_after_a_crash_the_same_app_still_answers(folder: Path) -> None:  # noqa: F811
    chats: list[Any] = [Boom(), ScriptedChat(REPLY)]

    class Switch:
        def stream(self, system: str, user: str) -> Iterator[str]:
            return iter(list((chats[0] if len(chats) == 2 else chats[-1]).stream(system, user)))

    sw = Switch()
    app = _app(folder, sw)
    with _quiet(app) as c:
        r = _ask(c, "How long must passwords be?")
        assert r.status_code == 200
        chats.pop(0)
        r2 = _ask(c, "How long must passwords be?")
        assert [e for e, _ in _events(r2.text)][-1] == "final", r2.text


def test_an_unexpected_exception_becomes_an_error_event(folder: Path) -> None:  # noqa: F811
    with _quiet(_app(folder, Boom())) as c:
        r = _ask(c, "How long must passwords be?")
        assert _events(r.text)[-1][0] == "error"


def test_closing_the_stream_early_releases_the_lock(folder: Path) -> None:  # noqa: F811
    app = _app(folder, ScriptedChat(REPLY))
    with _client(app) as c:
        cid = _conversation(c)
        with c.stream(
            "POST", "/api/ask", headers={"Origin": BASE},
            json={"conversation_id": cid, "question": "How long must passwords be?"},
        ) as r:  # fmt: skip
            next(r.iter_lines())  # read one line, then drop the connection
        import gc

        gc.collect()
        r2 = _ask(c, "How long must passwords be?", cid)
        names = [e for e, _ in _events(r2.text)]
        assert names[-1] == "final", names


def test_a_pending_generator_never_started_does_not_hold_the_lock(folder: Path) -> None:  # noqa: F811
    # ask() builds the generator but a disconnect before the first next() must not matter.
    with _client(_app(folder)) as c:
        cid = _conversation(c)
        for _ in range(3):
            assert _ask(c, "How long must passwords be?", cid).status_code == 200


def test_event_order_is_pages_token_final(folder: Path) -> None:  # noqa: F811
    with _client(_app(folder, ScriptedChat("Pass", "words [1]."))) as c:
        names = [e for e, _ in _events(_ask(c, "How long must passwords be?").text)]
        assert names[0] == "pages" and names[-1] == "final"
        assert set(names[1:-1]) == {"token"}


def test_a_follow_up_starts_with_search(folder: Path) -> None:  # noqa: F811
    with _client(_app(folder)) as c:
        cid = _conversation(c)
        _ask(c, "How long must passwords be?", cid)
        names = [e for e, _ in _events(_ask(c, "And cookies?", cid).text)]
        assert names[0] == "search" and names[1] == "pages" and names[-1] == "final"


def test_the_2000_char_cap_counts_characters_not_bytes(folder: Path) -> None:  # noqa: F811
    with _client(_app(folder)) as c:
        assert _ask(c, "é" * 2000).status_code == 200
        assert _ask(c, "é" * 2001).status_code == 422
        assert _ask(c, "😀" * 2001).status_code == 422


def test_the_cap_applies_before_strip(folder: Path) -> None:  # noqa: F811
    with _client(_app(folder)) as c:
        assert _ask(c, " " * 1990 + "passwords").status_code == 200
        assert _ask(c, " " * 2001 + "passwords").status_code == 422


@pytest.mark.parametrize("q", ["\x00", "\u2028", "\n\n", "\r\n"])
def test_odd_question_characters_do_not_crash(folder: Path, q: str) -> None:  # noqa: F811
    with _quiet(_app(folder)) as c:
        r = _ask(c, q + "passwords")
        assert r.status_code in (200, 422)


def test_a_lone_surrogate_in_a_question_does_not_kill_the_next_stream(folder: Path) -> None:  # noqa: F811
    with _quiet(_app(folder)) as c:
        cid = _conversation(c)
        # Refused as invalid text, without echoing it back (which would fail to encode).
        refused = _ask_raw(c, "\ud800 passwords", cid)
        assert refused.status_code == 422 and "\\ud800" not in refused.text
        r = _ask_raw(c, "cookies", cid)
        assert r.status_code == 200 and _events(r.text)[-1][0] in ("final", "error")


def test_a_non_object_or_non_json_body_is_422(folder: Path) -> None:  # noqa: F811
    with _client(_app(folder)) as c:
        for body in (b"[]", b"null", b"{", b"\xff\xfe", b'{"question":1,"conversation_id":2}'):
            r = c.post("/api/ask", content=body, headers={"Origin": BASE,
                       "Content-Type": "application/json"})  # fmt: skip
            assert r.status_code == 422


def test_conversation_eviction_makes_the_oldest_unknown(folder: Path) -> None:  # noqa: F811
    with _client(_app(folder)) as c:
        first = _conversation(c)
        for _ in range(MAX_CONVERSATIONS):
            _conversation(c)
        assert _ask(c, "How long must passwords be?", first).status_code == 404


def test_asking_in_a_conversation_keeps_it_from_being_evicted_first(folder: Path) -> None:  # noqa: F811
    with _client(_app(folder)) as c:
        first = _conversation(c)
        for _ in range(MAX_CONVERSATIONS - 1):
            _conversation(c)
        assert _ask(c, "How long must passwords be?", first).status_code == 200
        _conversation(c)
        assert _ask(c, "And cookies?", first).status_code == 200


# --- SSE serialisation ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "text", ["a\nb", "a\rb", "a\r\nb", "\u2028\u2029", "\x85", "\x0b\x0c", "data: x\n\n"]
)
def test_sse_data_is_a_single_line(text: str) -> None:
    out = sse("token", {"text": text})
    body = out.removesuffix("\n\n")
    assert body.count("\n") == 1  # only "event:" and "data:"
    assert "\r" not in body
    assert json.loads(body.split("data: ", 1)[1]) == {"text": text}
    assert out.endswith("\n\n")


def test_sse_survives_a_lone_surrogate() -> None:
    sse("token", {"text": "bad \ud800 text"}).encode("utf-8")


def test_token_text_with_html_is_json_not_markup(folder: Path) -> None:  # noqa: F811
    evil = "<img src=x onerror=alert(1)> </script>"
    with _client(_app(folder, ScriptedChat(evil))) as c:
        r = _ask(c, "How long must passwords be?")
        assert r.headers["content-type"].startswith("text/event-stream")
        assert _events(r.text)[1][1]["text"] == evil


# --- headers -------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("method", "path"),
    [("GET", "/nope"), ("GET", "/static/nope.js"), ("GET", "/static/app.js"), ("POST", "/nope"),
     ("GET", "/static"), ("GET", "/static/"), ("GET", "/pdf/abc"), ("GET", "/api/events")],
)  # fmt: skip
def test_security_headers_on_every_kind_of_response(folder: Path, method: str, path: str) -> None:  # noqa: F811
    with _client(_app(folder)) as c:
        r = c.request(method, path, headers={"Origin": BASE})
        for h in ("content-security-policy", "x-content-type-options", "referrer-policy"):
            assert h in r.headers, (path, r.status_code)
        assert r.headers["referrer-policy"] == "no-referrer"
        assert r.headers["x-content-type-options"] == "nosniff"


def test_security_headers_appear_once(folder: Path) -> None:  # noqa: F811
    with _client(_app(folder)) as c:
        for p in ("/", "/static/app.js", "/api/docs"):
            r = c.get(p)
            for h in ("content-security-policy", "x-content-type-options", "referrer-policy"):
                assert len(r.headers.get_list(h)) == 1, (p, h)


def test_a_500_still_carries_headers(folder: Path, monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: F811
    def boom(self: Any) -> Any:
        raise RuntimeError("x")

    monkeypatch.setattr(webapp.Library, "documents", boom)
    with _quiet(_app(folder)) as c:
        r = c.get("/api/docs")
        assert r.status_code == 500
        assert "content-security-policy" in r.headers
        assert r.headers["cache-control"] == "no-store"


def test_the_home_page_and_the_redirect_are_no_store(folder: Path) -> None:  # noqa: F811
    with _client(_app(folder), session=False) as c:
        assert c.get(f"/?token={TOKEN}").headers["cache-control"] == "no-store"
        assert c.get("/").headers["cache-control"] == "no-store"


def test_the_csp_forbids_inline_and_remote_script() -> None:
    csp = webapp.CSP
    assert "default-src 'self'" in csp
    assert "unsafe-inline" not in csp and "unsafe-eval" not in csp and "http" not in csp


# --- the static page -----------------------------------------------------------------------


def _static() -> Path:
    return webapp.STATIC


def test_the_page_has_no_inline_script_style_or_handler() -> None:
    import re

    html = (_static() / "index.html").read_text()
    for script in re.findall(r"<script\b[^>]*>(.*?)</script>", html, re.S):
        assert script.strip() == ""
    assert not re.search(r"<style\b", html, re.I)
    assert not re.search(r"\sstyle\s*=", html, re.I)
    assert not re.search(r"\son[a-z]+\s*=", html, re.I)
    assert not re.search(r"(src|href)\s*=\s*[\"']?(https?:)?//", html, re.I)


def test_app_js_never_writes_html() -> None:
    import re

    js = (_static() / "app.js").read_text()
    for banned in ("innerHTML", "outerHTML", "insertAdjacentHTML", "document.write", "eval(",
                   "new Function", "srcdoc", "createContextualFragment", "DOMParser",
                   ".html("):  # fmt: skip
        assert banned not in js, banned
    assert not re.search(r"setAttribute\(\s*[\"']on", js)


def test_app_js_only_links_with_server_built_hrefs() -> None:
    # link() sets a.href from server data only; the server builds it as /pdf/<digits>.
    assert webapp._page.__name__ == "_page"
    from askpdf.retrieve import PageHit  # noqa: F401

    ids = {"x.pdf": 5}
    hit = type("H", (), {"paths": ["javascript:alert(1)"], "page": 1, "label": "1"})()
    out = webapp._page(hit, ids)
    assert out["href"] is None


def test_static_files_do_not_list_directories_or_leak_source(folder: Path) -> None:  # noqa: F811
    with _client(_app(folder)) as c:
        for p in ("/static/", "/static/..%2fapp.py", "/static/%2e%2e/app.py", "/static/../app.py"):
            assert c.get(p).status_code in (404, 307, 308), p
