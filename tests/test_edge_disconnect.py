"""Browser disconnects through a real uvicorn server (tester, stop-on-disconnect)."""

import http.client
import json
import select
import socket
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest
import uvicorn

from askpdf.cli import main
from askpdf.providers.ollama import OllamaChat
from askpdf.web.app import Progress, cookie_name, create_app
from tests.pdfs import make_pdf, text_page

TOKEN = "disconnect-token"  # noqa: S105 - not a real secret
REPLY = (
    "Passwords need fifteen characters [1].\n"
    'Quotes:\n[1] "Passwords need fifteen characters or more"'
)


@dataclass
class Script:
    """How the fake model behaves for one request."""

    withhold_headers: bool = False  # no byte at all until `go` is set (prompt reading)
    pieces: list[str] = field(default_factory=lambda: [REPLY])
    delay: float = 0.0  # between pieces
    endless: bool = False  # keep sending until the client goes


class SlowModel:
    def __init__(self) -> None:
        self.scripts: list[Script] = []
        self.go = threading.Event()
        self.started = threading.Event()
        self.closed = threading.Event()
        self.finished = threading.Event()
        self.sent = 0
        self.lock = threading.Lock()

    def reset(self) -> None:
        self.go.clear()
        self.started.clear()
        self.closed.clear()
        self.finished.clear()
        self.sent = 0


def _peer_gone(conn: socket.socket) -> bool:
    r, _, _ = select.select([conn], [], [], 0)
    if not r:
        return False
    try:
        return conn.recv(1, socket.MSG_PEEK) == b""
    except OSError:
        return True


@contextmanager
def slow_model() -> Iterator[tuple[str, SlowModel]]:
    model = SlowModel()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            with model.lock:
                script = model.scripts.pop(0) if model.scripts else Script()
            model.started.set()
            conn = self.connection

            def wait(seconds: float) -> bool:
                """Sleep, True if the client went away meanwhile."""
                end = time.monotonic() + seconds
                while True:
                    if _peer_gone(conn):
                        model.closed.set()
                        return True
                    if time.monotonic() >= end:
                        return False
                    time.sleep(0.005)

            if script.withhold_headers:
                while not model.go.is_set():
                    if wait(0.01):
                        return
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.end_headers()
            self.wfile.flush()
            i = 0
            try:
                while script.endless or i < len(script.pieces):
                    piece = "x " * 2000 if script.endless else script.pieces[i]
                    line = json.dumps({"message": {"content": piece}, "done": False})
                    self.wfile.write(line.encode() + b"\n")
                    self.wfile.flush()
                    model.sent += 1
                    i += 1
                    if wait(script.delay):
                        return
                self.wfile.write(json.dumps({"done": True}).encode() + b"\n")
                self.wfile.flush()
                model.finished.set()
            except OSError:
                model.closed.set()

        def log_message(self, *args: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}", model
    finally:
        server.shutdown()
        server.server_close()


class Live:
    def __init__(self, port: int, progress: Progress) -> None:
        self.port = port
        self.progress = progress

    def conversation(self) -> str:
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        c.request("POST", "/api/conversations", headers=self._headers())
        body = json.loads(c.getresponse().read())
        c.close()
        return str(body["id"])

    def _headers(self) -> dict[str, str]:
        return {
            "Host": f"127.0.0.1:{self.port}",
            "Origin": f"http://127.0.0.1:{self.port}",
            "Cookie": f"{cookie_name(self.port)}={TOKEN}",
        }

    def ask_socket(
        self, question: str = "How long must passwords be?", rcvbuf: int = 0
    ) -> socket.socket:
        body = json.dumps({"conversation_id": self.conversation(), "question": question}).encode()
        s = socket.socket()
        if rcvbuf:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, rcvbuf)
        s.connect(("127.0.0.1", self.port))
        head = {
            **self._headers(),
            "Content-Type": "application/json",
            "Content-Length": str(len(body)),
        }
        lines = "".join(f"{k}: {v}\r\n" for k, v in head.items())
        s.sendall(f"POST /api/ask HTTP/1.1\r\n{lines}\r\n".encode() + body)
        return s

    def ask(self) -> str:
        s = self.ask_socket()
        s.settimeout(15)
        data = b""
        while chunk := s.recv(65536):  # the server ends the stream, so read to the end
            data += chunk
        s.close()
        return data.decode()


def answer_threads() -> list[threading.Thread]:
    return [t for t in threading.enumerate() if t.name.startswith("askpdf-answer")]


def wait_for(cond: Any, timeout: float = 3.0) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.01)
    return bool(cond())


@pytest.fixture
def live(tmp_path: Path) -> Iterator[tuple[Live, SlowModel]]:
    make_pdf(
        tmp_path / "f" / "policy.pdf",
        [text_page("Passwords need fifteen characters or more in every case.")],
    )
    assert main(["index", str(tmp_path / "f"), "--index", str(tmp_path / "i.db")]) == 0
    with slow_model() as (url, model):
        captured: list[Progress] = []
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        app = create_app(
            root=tmp_path / "f",
            index=tmp_path / "i.db",
            port=port,
            token=TOKEN,
            chat_for=lambda q, h, t: OllamaChat(url=url, timeout=30),
            index_folder=captured.append,
        )
        server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()
        assert wait_for(lambda: server.started and captured)
        try:
            yield Live(port, captured[0]), model
        finally:
            server.should_exit = True
            thread.join(10)


def _recv_until(s: socket.socket, needle: bytes, timeout: float = 10) -> bytes:
    s.settimeout(timeout)
    data = b""
    while needle not in data:
        chunk = s.recv(65536)
        if not chunk:
            break
        data += chunk
    return data


def _next_question_is_answered(lv: Live, model: SlowModel) -> None:
    model.reset()
    model.scripts.append(Script())
    text = lv.ask()
    assert "another question is being answered" not in text
    assert "event: final" in text


def test_baseline_answer_works(live: tuple[Live, SlowModel]) -> None:
    lv, _ = live
    text = lv.ask()
    assert "event: final" in text
    assert lv.progress.idle.is_set()


def test_disconnect_before_first_byte_of_reply(live: tuple[Live, SlowModel]) -> None:
    # The model is still reading the prompt: nothing has come back, not even headers.
    lv, model = live
    model.scripts.append(Script(withhold_headers=True))
    s = lv.ask_socket()
    assert model.started.wait(5)
    s.close()
    time.sleep(0.2)
    # Whatever the model does now, the lock should not stay held for its prompt time.
    assert wait_for(lv.progress.idle.is_set, 3), "idle not set after disconnect"
    _next_question_is_answered(lv, model)


def test_disconnect_while_waiting_for_the_next_token(live: tuple[Live, SlowModel]) -> None:
    lv, model = live
    model.scripts.append(Script(pieces=["a", "b"], delay=5.0))
    s = lv.ask_socket()
    assert model.started.wait(5)
    assert wait_for(lambda: model.sent >= 1)
    s.close()
    assert wait_for(model.closed.is_set, 3), "model connection was not closed"
    assert wait_for(lv.progress.idle.is_set, 3)
    _next_question_is_answered(lv, model)


def test_prompt_wait_is_cut_without_waiting_for_the_model(
    live: tuple[Live, SlowModel],
) -> None:
    # The model never answers on its own: the cut socket is what frees the lock.
    lv, model = live
    before = len(answer_threads())
    model.scripts.append(Script(withhold_headers=True, pieces=["a", "b", "c"], delay=0.05))
    s = lv.ask_socket()
    assert model.started.wait(5)
    s.close()
    assert wait_for(lv.progress.idle.is_set, 3)
    model.go.set()
    assert wait_for(lambda: len(answer_threads()) <= before, 3)
    _next_question_is_answered(lv, model)


def test_disconnect_mid_token_stream_closes_model_stream(live: tuple[Live, SlowModel]) -> None:
    lv, model = live
    model.scripts.append(Script(pieces=["tok "] * 200, delay=0.05))
    s = lv.ask_socket()
    assert _recv_until(s, b"event: token")
    s.close()
    assert wait_for(model.closed.is_set, 3), "fake model never saw its connection close"
    assert not model.finished.is_set()
    assert model.sent < 200
    assert wait_for(lv.progress.idle.is_set, 3)
    _next_question_is_answered(lv, model)


def test_disconnect_mid_send_with_unread_output(live: tuple[Live, SlowModel]) -> None:
    # The client stops reading, so uvicorn is stuck sending; then it goes away.
    lv, model = live
    model.scripts.append(Script(endless=True, delay=0.0))
    s = lv.ask_socket(rcvbuf=1024)
    assert model.started.wait(5)
    time.sleep(0.5)
    s.close()
    assert wait_for(model.closed.is_set, 5), "model connection not closed"
    assert wait_for(lv.progress.idle.is_set, 5)
    _next_question_is_answered(lv, model)


def test_disconnect_after_final(live: tuple[Live, SlowModel]) -> None:
    lv, model = live
    s = lv.ask_socket()
    _recv_until(s, b"event: final")
    s.close()
    assert wait_for(lv.progress.idle.is_set, 3)
    _next_question_is_answered(lv, model)


def test_second_question_is_refused_while_one_really_runs(live: tuple[Live, SlowModel]) -> None:
    lv, model = live
    model.scripts.append(Script(pieces=["tok "] * 100, delay=0.05))
    first = lv.ask_socket()
    assert _recv_until(first, b"event: token")
    second = lv.ask()
    assert "another question is being answered" in second
    assert not lv.progress.idle.is_set()
    first.close()
    assert wait_for(lv.progress.idle.is_set, 3)


def test_repeated_disconnects_leave_no_threads(live: tuple[Live, SlowModel]) -> None:
    lv, model = live
    before = len(answer_threads())
    for _ in range(15):
        model.reset()
        model.scripts.append(Script(pieces=["tok "] * 200, delay=0.05))
        s = lv.ask_socket()
        _recv_until(s, b"event: token")
        s.close()
        assert wait_for(lv.progress.idle.is_set, 3)
    assert wait_for(lambda: len(answer_threads()) <= before, 3), answer_threads()
    _next_question_is_answered(lv, model)
    assert wait_for(lambda: len(answer_threads()) <= before, 3)
