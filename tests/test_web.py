"""The web app: routes, the answer stream, and every security case in design §12."""

import json
import os
import socket
import sys
import threading
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, ClassVar

import pytest
from fastapi.testclient import TestClient

from askpdf import cli
from askpdf.cli import build_parser, free_port, main
from askpdf.providers import http
from askpdf.providers.base import ChatModel
from askpdf.retrieve import PageHit
from askpdf.store import FileRow, Store
from askpdf.web.app import (
    CSP,
    HISTORY_TURNS,
    MAX_CONVERSATIONS,
    MAX_QUESTION,
    Conversations,
    Progress,
    cookie_name,
    create_app,
)
from tests.pdfs import make_pdf, text_page

PORT = 8765
TOKEN = "launch-token-for-tests"  # noqa: S105 - not a real secret
BASE = f"http://127.0.0.1:{PORT}"
REPLY = (
    "Passwords need fifteen characters [1].\n"
    'Quotes:\n[1] "Passwords need fifteen characters or more"'
)


class ScriptedChat:
    def __init__(self, *pieces: str, wait: threading.Event | None = None) -> None:
        self.pieces = pieces
        self.wait = wait
        self.users: list[str] = []
        self.entered = threading.Event()

    def stream(self, system: str, user: str) -> Iterator[str]:
        self.users.append(user)
        self.entered.set()
        if self.wait is not None:
            self.wait.wait(timeout=10)
        yield from self.pieces


class Unreachable:
    def stream(self, system: str, user: str) -> Iterator[str]:
        raise http.UnreachableError("cannot reach Ollama at http://127.0.0.1:11434")


@pytest.fixture
def folder(tmp_path: Path) -> Path:
    make_pdf(
        tmp_path / "f" / "policy.pdf",
        [
            text_page("Passwords need fifteen characters or more in every case."),
            text_page("Cookies need the Secure flag on every response."),
        ],
    )
    assert main(["index", str(tmp_path / "f"), "--index", str(tmp_path / "i.db")]) == 0
    return tmp_path / "f"


def _app(folder: Path, chat: Any = None, **kwargs: Any) -> Any:
    chat = chat or ScriptedChat(REPLY)

    def chat_for(q: str, hits: Sequence[PageHit], texts: Mapping[Any, str]) -> ChatModel:
        return chat  # type: ignore[no-any-return]

    return create_app(
        root=folder,
        index=folder.parent / "i.db",
        port=PORT,
        token=TOKEN,
        chat_for=chat_for,
        **kwargs,
    )


def _client(app: Any, *, session: bool = True) -> TestClient:
    client = TestClient(app, base_url=BASE, follow_redirects=False)
    if session:
        client.cookies.set(cookie_name(PORT), TOKEN)
    return client


@pytest.fixture
def client(folder: Path) -> Iterator[TestClient]:
    with _client(_app(folder)) as c:
        yield c


def _events(text: str) -> list[tuple[str, dict[str, Any]]]:
    events = []
    for block in text.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.split("\n"))
        events.append((lines["event"], json.loads(lines["data"])))
    return events


def _conversation(client: TestClient) -> str:
    response = client.post("/api/conversations", headers={"Origin": BASE})
    assert response.status_code == 200
    return str(response.json()["id"])


def _ask(client: TestClient, question: str, cid: str | None = None) -> Any:
    cid = cid or _conversation(client)
    return client.post(
        "/api/ask",
        json={"conversation_id": cid, "question": question},
        headers={"Origin": BASE},
    )


# --- the launch token -----------------------------------------------------------------------


def test_the_launch_token_is_swapped_for_a_strict_httponly_cookie(folder: Path) -> None:
    with _client(_app(folder), session=False) as client:
        response = client.get(f"/?token={TOKEN}")
        assert response.status_code == 303 and response.headers["location"] == "/"
        cookie = response.headers["set-cookie"]
        assert f"{cookie_name(PORT)}={TOKEN}" in cookie
        assert "HttpOnly" in cookie and "SameSite=strict" in cookie
        page = client.get("/")  # the client kept the cookie
        assert page.status_code == 200 and "/static/app.js" in page.text


@pytest.mark.parametrize("query", ["", "?token=wrong", "?token=%C3%A9t%C3%A9", "?token="])
def test_without_the_token_the_page_says_to_start_from_the_terminal(
    folder: Path, query: str
) -> None:
    with _client(_app(folder), session=False) as client:
        response = client.get(f"/{query}")
        assert response.status_code == 403 and "terminal" in response.text
        assert "set-cookie" not in response.headers


# --- request checks -------------------------------------------------------------------------


@pytest.mark.parametrize(
    "host", ["evil.example", f"evil.example:{PORT}", "127.0.0.1:9999", "127.0.0.1", ""]
)
@pytest.mark.parametrize("path", ["/", "/api/docs", "/pdf/1", "/static/app.js"])
def test_a_wrong_host_is_refused(client: TestClient, host: str, path: str) -> None:
    response = client.get(path, headers={"Host": host})
    assert response.status_code == 400


def test_localhost_by_name_is_accepted(client: TestClient) -> None:
    assert client.get("/api/docs", headers={"Host": f"localhost:{PORT}"}).status_code == 200


@pytest.mark.parametrize("path", ["/api/docs", "/api/events", "/pdf/1"])
@pytest.mark.parametrize("cookie", [None, "wrong", "café"])
def test_api_and_pdf_need_the_session_cookie(folder: Path, path: str, cookie: str | None) -> None:
    # Sent as raw bytes: anyone can put non-ASCII in a cookie header.
    headers = {} if cookie is None else {"Cookie": f"{cookie_name(PORT)}={cookie}".encode()}
    with _client(_app(folder), session=False) as client:
        assert client.get(path, headers=headers).status_code == 403


def test_a_cookie_for_another_port_does_not_count(folder: Path) -> None:
    with _client(_app(folder), session=False) as client:
        client.cookies.set(cookie_name(PORT + 1), TOKEN)
        assert client.get("/api/docs").status_code == 403


@pytest.mark.parametrize(
    "origin", [None, "null", "http://evil.example", f"http://localhost:{PORT + 1}", BASE + "/"]
)
@pytest.mark.parametrize("path", ["/api/conversations", "/api/ask"])
def test_a_post_from_another_origin_is_refused(
    client: TestClient, origin: str | None, path: str
) -> None:
    headers = {} if origin is None else {"Origin": origin}
    response = client.post(path, json={}, headers=headers)
    assert response.status_code == 403


def test_the_origin_must_match_the_host_used(client: TestClient) -> None:
    response = client.post(
        "/api/conversations",
        headers={"Host": f"localhost:{PORT}", "Origin": BASE},  # 127.0.0.1 vs localhost
    )
    assert response.status_code == 403


@pytest.mark.parametrize("path", ["/", "/api/docs", "/pdf/1", "/pdf/999", "/static/app.css"])
def test_every_response_carries_the_security_headers(client: TestClient, path: str) -> None:
    headers = client.get(path).headers
    assert headers["content-security-policy"] == CSP
    assert headers["x-content-type-options"] == "nosniff"
    assert headers["referrer-policy"] == "no-referrer"


def test_refusals_carry_the_security_headers_too(client: TestClient) -> None:
    response = client.get("/", headers={"Host": "evil.example"})
    assert response.headers["content-security-policy"] == CSP


def test_api_responses_are_not_cached(client: TestClient) -> None:
    assert client.get("/api/docs").headers["cache-control"] == "no-store"
    assert client.get("/pdf/1").headers["cache-control"] == "no-store"


def test_no_api_schema_or_docs_pages_are_served(client: TestClient) -> None:
    for path in ("/docs", "/redoc", "/openapi.json"):
        assert client.get(path).status_code == 404


# --- documents and PDFs ---------------------------------------------------------------------


def _doc_id(client: TestClient, path: str = "policy.pdf") -> int:
    docs = client.get("/api/docs").json()
    return int(next(d["id"] for d in docs["documents"] if d["path"] == path))


def test_docs_lists_each_indexed_file_with_its_id(client: TestClient, folder: Path) -> None:
    data = client.get("/api/docs").json()
    assert data["folder"] == "f"
    assert data["documents"] == [
        {"id": _doc_id(client), "path": "policy.pdf", "pages": 2, "scanned": False}
    ]
    assert data["failed"] == []


def test_a_pdf_is_served_inline_by_its_id(client: TestClient, folder: Path) -> None:
    response = client.get(f"/pdf/{_doc_id(client)}")
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/pdf"
    assert response.headers["content-disposition"] == "inline"
    assert response.content == (folder / "policy.pdf").read_bytes()


@pytest.mark.parametrize(
    "path",
    [
        "/pdf/999",
        "/pdf/0",
        "/pdf/-1",
        "/pdf/1.0",
        "/pdf/1e0",
        "/pdf/%2e%2e/%2e%2e/etc/hosts",
        "/pdf/..%2f..%2fetc%2fhosts",
        "/pdf/%2Fetc%2Fhosts",
        "/pdf/policy.pdf",
        "/pdf/" + "9" * 30,
        "/pdf/" + chr(0x661),  # an Arabic-Indic digit: isdigit() but not a file id
        "/pdf/",
    ],
)
def test_only_a_known_numeric_id_serves_a_file(client: TestClient, path: str) -> None:
    assert client.get(path).status_code == 404


def test_dot_dot_paths_never_reach_a_file(client: TestClient) -> None:
    for raw in ("/pdf/../../etc/hosts", "/pdf/1/../../etc/passwd", "/pdf/1/../../../etc/hosts"):
        response = client.get(raw)
        assert response.status_code == 404
        assert b"root:" not in response.content and b"localhost" not in response.content


def _symlink(link: Path, target: Path) -> None:
    try:
        link.symlink_to(target)
    except OSError:  # Windows without the privilege
        pytest.skip("cannot create symlinks here")


def _add_row(folder: Path, rel: str) -> int:
    """Record a file as indexed, as if it had been safe when the folder was scanned."""
    with Store(folder.parent / "i.db") as store:
        sha = next(iter(store.indexed_paths()))
        store.put_file(FileRow(rel, sha, 1, 1, "indexed", None))
        return store.file_ids()[rel]


def test_a_file_swapped_for_a_symlink_out_of_the_folder_is_refused(
    client: TestClient, folder: Path, tmp_path: Path
) -> None:
    file_id = _doc_id(client)
    outside = make_pdf(tmp_path / "outside" / "secret.pdf", [text_page("secret words")])
    (folder / "policy.pdf").unlink()
    _symlink(folder / "policy.pdf", outside)
    response = client.get(f"/pdf/{file_id}")
    assert response.status_code == 404 and b"secret" not in response.content


@pytest.mark.skipif(sys.platform == "win32", reason="no /etc/hosts")
def test_a_symlink_to_etc_hosts_is_refused(client: TestClient, folder: Path) -> None:
    _symlink(folder / "hosts.pdf", Path("/etc/hosts"))
    assert client.get(f"/pdf/{_add_row(folder, 'hosts.pdf')}").status_code == 404


def test_a_symlinked_folder_leading_out_is_refused(
    client: TestClient, folder: Path, tmp_path: Path
) -> None:
    make_pdf(tmp_path / "outside" / "secret.pdf", [text_page("secret words")])
    _symlink(folder / "linked", tmp_path / "outside")
    assert client.get(f"/pdf/{_add_row(folder, 'linked/secret.pdf')}").status_code == 404


def test_an_absolute_path_in_the_index_is_refused(
    client: TestClient, folder: Path, tmp_path: Path
) -> None:
    outside = make_pdf(tmp_path / "outside" / "secret.pdf", [text_page("secret words")])
    assert client.get(f"/pdf/{_add_row(folder, str(outside))}").status_code == 404


def test_a_dot_dot_path_in_the_index_is_refused(
    client: TestClient, folder: Path, tmp_path: Path
) -> None:
    make_pdf(tmp_path / "outside" / "secret.pdf", [text_page("secret words")])
    assert client.get(f"/pdf/{_add_row(folder, '../outside/secret.pdf')}").status_code == 404


def test_a_symlink_inside_the_folder_is_served(client: TestClient, folder: Path) -> None:
    _symlink(folder / "alias.pdf", folder / "policy.pdf")
    response = client.get(f"/pdf/{_add_row(folder, 'alias.pdf')}")
    assert response.status_code == 200
    assert response.content == (folder / "policy.pdf").read_bytes()


def test_a_deleted_file_is_404(client: TestClient, folder: Path) -> None:
    file_id = _doc_id(client)
    (folder / "policy.pdf").unlink()
    assert client.get(f"/pdf/{file_id}").status_code == 404


def test_a_file_replaced_by_a_folder_is_404(client: TestClient, folder: Path) -> None:
    file_id = _doc_id(client)
    (folder / "policy.pdf").unlink()
    (folder / "policy.pdf").mkdir()
    assert client.get(f"/pdf/{file_id}").status_code == 404


def test_a_file_that_is_not_a_pdf_is_refused(client: TestClient, folder: Path) -> None:
    (folder / "notes.txt").write_text("not a pdf")
    assert client.get(f"/pdf/{_add_row(folder, 'notes.txt')}").status_code == 404


def test_a_failed_file_is_not_served(client: TestClient, folder: Path) -> None:
    (folder / "broken.pdf").write_bytes(b"%PDF-1.4 nonsense")
    with Store(folder.parent / "i.db") as store:
        store.put_file(FileRow("broken.pdf", "", 0, -1, "failed", "cannot read"))
        db = store._db
        (file_id,) = db.execute("SELECT rowid FROM files WHERE path = 'broken.pdf'").fetchone()
    assert client.get(f"/pdf/{file_id}").status_code == 404
    failed = client.get("/api/docs").json()["failed"]
    assert failed == [{"path": "broken.pdf", "reason": "cannot read"}]


# --- asking ---------------------------------------------------------------------------------


def test_the_answer_streams_pages_then_tokens_then_the_checked_answer(
    client: TestClient,
) -> None:
    response = _ask(client, "how many characters do passwords need")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = _events(response.text)
    names = [name for name, _ in events]
    assert names == ["pages", "token", "final"]  # a first question is searched as typed
    file_id = _doc_id(client)
    pages = events[0][1]["pages"]
    assert pages[0] == {
        "number": 1,
        "file": "policy.pdf",
        "path": "policy.pdf",
        "page": 1,
        "label": "1",
        "href": f"/pdf/{file_id}#page=1",
        "snippet": "Passwords need fifteen characters or more in every case.",
    }
    final = events[-1][1]
    assert final["grounded"] and not final["not_found"]
    assert final["text"] == "Passwords need fifteen characters [1]."
    (citation,) = final["citations"]
    assert citation["href"] == f"/pdf/{file_id}#page=1" and citation["verified"]
    assert citation["quotes"] == [
        {"quote": "Passwords need fifteen characters or more", "verified": True, "reason": ""}
    ]


def test_a_follow_up_is_searched_with_the_earlier_questions(folder: Path) -> None:
    chat = ScriptedChat(REPLY)
    with _client(_app(folder, chat)) as client:
        cid = _conversation(client)
        questions = ["passwords", "and cookies", "which flag", "on every response"]
        searches = []
        for q in questions:
            events = _events(_ask(client, q, cid).text)
            searches.append(next((d["query"] for n, d in events if n == "search"), None))
    assert searches == [
        None,
        "passwords and cookies",
        "passwords and cookies which flag",
        # Only the last HISTORY_TURNS questions are joined.
        "and cookies which flag on every response",
    ]
    assert HISTORY_TURNS == 3
    # The model is asked the question as typed; only the search uses the history.
    assert all(
        user.endswith(f"Question: {q}") for user, q in zip(chat.users, questions, strict=True)
    )


def test_new_conversations_do_not_share_history(client: TestClient) -> None:
    _ask(client, "passwords")
    events = _events(_ask(client, "cookies").text)  # a new conversation
    assert "search" not in [name for name, _ in events]


def test_no_matching_page_says_not_found_without_the_model(folder: Path) -> None:
    chat = ScriptedChat(REPLY)
    with _client(_app(folder, chat)) as client:
        events = _events(_ask(client, "zzzz qqqq").text)
    assert [name for name, _ in events] == ["pages", "final"]
    assert events[0][1]["pages"] == [] and events[1][1]["not_found"]
    assert chat.users == []


def test_an_unreachable_model_becomes_an_error_event(folder: Path) -> None:
    with _client(_app(folder, Unreachable())) as client:
        events = _events(_ask(client, "passwords").text)
    assert events[-1][0] == "error" and "ollama serve" in events[-1][1]["message"]


def test_a_second_question_while_one_is_answered_is_turned_away(folder: Path) -> None:
    release = threading.Event()
    chat = ScriptedChat(REPLY, wait=release)
    with _client(_app(folder, chat)) as client:
        first: dict[str, Any] = {}
        thread = threading.Thread(target=lambda: first.update(r=_ask(client, "passwords")))
        thread.start()
        try:
            assert chat.entered.wait(timeout=10)  # the first answer now holds the model
            busy = _events(_ask(client, "passwords").text)
        finally:
            release.set()
            thread.join(timeout=10)
        assert busy == [("error", {"message": "another question is being answered; try again"})]
        assert _events(first["r"].text)[-1][0] == "final"
        # Once it is done, the next question is answered.
        assert _events(_ask(client, "passwords").text)[-1][0] == "final"


def test_an_unknown_conversation_is_404(client: TestClient) -> None:
    assert _ask(client, "passwords", cid="never-started").status_code == 404


@pytest.mark.parametrize(
    "body",
    [
        {"conversation_id": "x", "question": "q" * (MAX_QUESTION + 1)},
        {"conversation_id": "x", "question": ""},
        {"conversation_id": "", "question": "q"},
        {"conversation_id": "x" * 65, "question": "q"},
        {"question": "q"},
        {"conversation_id": "x", "question": 5},
    ],
)
def test_bad_ask_bodies_are_refused(client: TestClient, body: dict[str, Any]) -> None:
    response = client.post("/api/ask", json=body, headers={"Origin": BASE})
    assert response.status_code == 422


def test_a_blank_question_is_refused(client: TestClient) -> None:
    assert _ask(client, "   \n ").status_code == 422


def test_the_longest_question_is_accepted(client: TestClient) -> None:
    assert _ask(client, "password " * (MAX_QUESTION // 9)).status_code == 200


def test_new_documents_are_found_after_reindexing(client: TestClient, folder: Path) -> None:
    assert _events(_ask(client, "transonic wing").text)[-1][1]["not_found"]
    make_pdf(folder / "wing.pdf", [text_page("The transonic wing was tested in 1956.")])
    assert main(["index", str(folder), "--index", str(folder.parent / "i.db")]) == 0
    pages = _events(_ask(client, "transonic wing").text)[0][1]["pages"]
    assert [p["file"] for p in pages] == ["wing.pdf"]


def test_conversations_are_capped_and_keep_the_last_questions() -> None:
    conversations = Conversations()
    first = conversations.start()
    for q in ("a", "b", "c", "d"):
        conversations.add(first, q)
    assert conversations.earlier(first) == ["c", "d"]
    for _ in range(MAX_CONVERSATIONS):
        conversations.start()
    assert conversations.earlier(first) is None  # the oldest was forgotten
    conversations.add(first, "e")  # adding to a forgotten one does nothing
    assert conversations.earlier(first) is None


# --- indexing at start-up -------------------------------------------------------------------


def test_indexing_progress_streams_until_ready(folder: Path) -> None:
    def index(progress: Progress) -> None:
        progress.update(current="policy.pdf", done=0)
        progress.update(state="ready", current="", done=1)

    with _client(_app(folder, index_folder=index)) as client:
        events = _events(client.get("/api/events").text)
    assert events[-1][1]["state"] == "ready" and events[-1][1]["done"] == 1


def test_without_an_indexer_the_index_is_ready(client: TestClient) -> None:
    (event,) = _events(client.get("/api/events").text)
    assert event[1]["state"] == "ready"


def test_serve_indexes_the_folder_in_the_background(tmp_path: Path) -> None:
    make_pdf(tmp_path / "g" / "a.pdf", [text_page("Langley tested the transonic wing.")])
    (tmp_path / "g" / "bad.pdf").write_bytes(b"not a pdf at all")
    args = build_parser().parse_args(
        ["serve", str(tmp_path / "g"), "--index", str(tmp_path / "j.db")]
    )
    Store(tmp_path / "j.db").close()
    errors = open(os.devnull, "w")  # noqa: SIM115 - closed below
    try:
        app = create_app(
            root=args.path,
            index=tmp_path / "j.db",
            port=PORT,
            token=TOKEN,
            chat_for=lambda q, h, t: ScriptedChat(REPLY),
            index_folder=cli._index_in_background(args, errors),
        )
        with _client(app) as client:
            events = _events(client.get("/api/events").text)
            docs = client.get("/api/docs").json()
    finally:
        errors.close()
    assert events[-1][1]["state"] == "ready"
    assert [d["path"] for d in docs["documents"]] == ["a.pdf"]
    assert [f["path"] for f in docs["failed"]] == ["bad.pdf"]


def test_indexing_failure_is_reported(tmp_path: Path) -> None:
    make_pdf(tmp_path / "g" / "a.pdf", [text_page("words")])
    make_pdf(tmp_path / "h" / "b.pdf", [text_page("words")])
    index = tmp_path / "j.db"
    assert main(["index", str(tmp_path / "h"), "--index", str(index)]) == 0  # another folder's
    args = build_parser().parse_args(["serve", str(tmp_path / "g"), "--index", str(index)])
    progress = Progress()
    with open(os.devnull, "w") as errors:
        cli._index_in_background(args, errors)(progress)
    assert progress.state == "failed" and "index of" in progress.message


# --- the serve command ----------------------------------------------------------------------


def test_askpdf_folder_means_serve(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    seen: list[Any] = []

    def serve(args: Any, out: Any, err: Any) -> int:
        seen.append(args)
        return 0

    monkeypatch.setattr(cli, "cmd_serve", serve)
    assert main([str(tmp_path), "--no-browser"]) == 0
    assert seen[0].command == "serve" and seen[0].path == tmp_path and seen[0].no_browser


def test_known_commands_are_not_taken_for_folders(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "cmd_serve", lambda args, out, err: pytest.fail("served"))
    with pytest.raises(SystemExit):
        main(["--version"])


class FakeServer:
    instances: ClassVar[list["FakeServer"]] = []

    def __init__(self, config: Any) -> None:
        self.config = config
        self.started = True
        FakeServer.instances.append(self)

    def run(self) -> None:
        # What uvicorn does on start-up: run the app's lifespan, which calls on_start.
        with TestClient(self.config.app, base_url=BASE):
            pass


def test_serve_binds_to_loopback_and_prints_a_link_with_the_token(
    folder: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import uvicorn

    opened: list[str] = []
    monkeypatch.setattr(uvicorn, "Server", FakeServer)
    monkeypatch.setattr("webbrowser.open", opened.append)
    index = str(folder.parent / "i.db")
    assert main(["serve", str(folder), "--index", index, "--port", str(PORT)]) == 0
    config = FakeServer.instances[-1].config
    assert config.host == "127.0.0.1" and config.port == PORT
    out = capsys.readouterr().out
    assert f"http://127.0.0.1:{PORT}/?token=" in out
    token = out.split("token=")[1].split()[0]
    assert len(token) >= 40 and opened == [f"http://127.0.0.1:{PORT}/?token={token}"]


def test_serve_with_no_browser_only_prints(
    folder: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import uvicorn

    monkeypatch.setattr(uvicorn, "Server", FakeServer)
    monkeypatch.setattr("webbrowser.open", lambda url: pytest.fail("opened"))
    index = str(folder.parent / "i.db")
    assert main(["serve", str(folder), "--index", index, "--no-browser"]) == 0
    assert "?token=" in capsys.readouterr().out


def test_serve_needs_a_folder(folder: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["serve", str(folder / "policy.pdf")]) == 1
    assert "not a folder" in capsys.readouterr().err


def test_serve_refuses_a_remote_model_without_permission(
    folder: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("ASKPDF_OLLAMA_URL", "http://203.0.113.5:11434")
    assert main(["serve", str(folder), "--index", str(folder.parent / "i.db")]) == 1
    assert "allow-remote" in capsys.readouterr().err


def test_serve_refuses_another_folders_index(
    folder: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    other = tmp_path / "other"
    other.mkdir()
    assert main(["serve", str(other), "--index", str(folder.parent / "i.db")]) == 1
    assert "index of" in capsys.readouterr().err


@pytest.mark.parametrize("port", ["0", "70000", "http"])
def test_serve_port_must_be_a_port(port: str) -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["serve", ".", "--port", port])


def test_free_port_skips_a_busy_one() -> None:
    with socket.socket() as busy:
        busy.bind(("127.0.0.1", 0))
        taken = busy.getsockname()[1]
        busy.listen()
        port = free_port(taken, tries=5)
    assert port is not None and port != taken


def test_the_package_ships_the_page() -> None:
    from askpdf.web import app

    for name in ("index.html", "app.css", "app.js"):
        assert (app.STATIC / name).is_file()
    html = (app.STATIC / "index.html").read_text()
    # The CSP allows no inline script or style, so the page must have none.
    assert "<script>" not in html and "style=" not in html and "<style" not in html
    js = (app.STATIC / "app.js").read_text()
    assert "innerHTML" not in js and "insertAdjacentHTML" not in js
