# ruff: noqa: S603, S607
"""Edge cases for `askpdf app` (ADR-0013, PR 1): the AppleScript boundary, refusals, the
process lifecycle against a real server, the Ollama check and pull, and the page's JS.

No dialog is ever shown: ``run`` is always a fake. The one real ``osascript`` call runs a
harmless script that only echoes its arguments.
"""

import http.client
import json
import os
import queue
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from askpdf import cli, desktop
from askpdf.cli import main
from askpdf.providers import http as providers_http
from askpdf.providers import ollama
from askpdf.web.app import EMPTY_FOLDER, Progress
from tests.pdfs import make_pdf, text_page
from tests.test_web import BASE, PORT, TOKEN, FakeServer, _app, _client, _events, folder  # noqa: F401

posix = pytest.mark.skipif(sys.platform == "win32", reason="POSIX signals")
mac = pytest.mark.skipif(sys.platform != "darwin", reason="macOS only")

HOSTILE = [
    'Papers "2024"',
    "back\\slash\\",
    "new\nline",
    'x" & (do shell script "touch /tmp/pwned") & "',
    '"; do shell script "id"; "',
    'end run\ndo shell script "id"\non run argv',
    "-e",
    "$(id) `id` ; rm -rf /",
    '\'; quit app "Finder"',
    "tab\there",
]

_ROOT = hasattr(os, "geteuid") and os.geteuid() == 0  # Windows has no geteuid


def _recorder(code: int = 0, stderr: str = "") -> Any:
    calls: list[list[str]] = []

    def run(argv: list[str]) -> subprocess.CompletedProcess[str]:
        calls.append(argv)
        return subprocess.CompletedProcess(argv, code, "/x/\n", stderr)

    run.calls = calls  # type: ignore[attr-defined]
    return run


# --- 2. no user-controlled text reaches the AppleScript source -------------------------------


def test_the_script_text_is_the_same_whatever_the_folder_or_message(tmp_path: Path) -> None:
    baseline_pick = _recorder()
    desktop.osascript_picker(tmp_path, run=baseline_pick)
    baseline_alert = _recorder()
    desktop.osascript_alert("plain", run=baseline_alert)
    for text in HOSTILE:
        pick = _recorder()
        desktop.osascript_picker(tmp_path / text, run=pick)
        alert = _recorder()
        desktop.osascript_alert(text, run=alert)
        assert pick.calls[0][:3] == baseline_pick.calls[0][:3]
        assert pick.calls[0][3:] == [desktop.PROMPT, str(tmp_path / text)]
        assert alert.calls[0][:3] == baseline_alert.calls[0][:3]
        assert alert.calls[0][3:] == [text]


def test_a_hostile_folder_name_reaches_the_alert_only_as_an_argument(tmp_path: Path) -> None:
    for text in HOSTILE:
        bad = tmp_path / text.replace("/", "_")
        bad.mkdir()
        bad.chmod(0)
        try:
            alerts: list[str] = []
            picks = iter([bad, None])
            run = _recorder()

            def alert(message: str, run: Any = run, alerts: list[str] = alerts) -> None:
                alerts.append(message)
                desktop.osascript_alert(message, run=run)

            def pick(home: Path, picks: Iterator[Path | None] = picks) -> Path | None:
                return next(picks)

            desktop.choose_folder(pick, alert, tmp_path)
        finally:
            bad.chmod(0o700)
        if _ROOT:
            continue
        (call,) = run.calls
        assert call[2] == desktop._ALERT
        assert call[3] == alerts[0]
        assert bad.name in call[3]


@mac
def test_osascript_hands_hostile_arguments_to_the_script_verbatim() -> None:
    """The real osascript, a script that only echoes ``argv``: no dialog, no side effect."""
    script = 'on run argv\nreturn (item 1 of argv) & "|" & (item 2 of argv)\nend run'
    for text in HOSTILE:
        if text == "-e":
            continue  # osascript reads it as an option; no message or path starts with "-"
        done = subprocess.run(
            ["/usr/bin/osascript", "-e", script, text, "/Users/me"],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        assert done.returncode == 0, (text, done.stderr)
        assert done.stdout == f"{text}|/Users/me\n"


# --- 1. refusals -----------------------------------------------------------------------------


def test_the_whole_disk_is_refused_however_it_is_spelled(tmp_path: Path) -> None:
    link = tmp_path / "root"
    link.symlink_to("/")
    for spelling in ("/", "//", "/.", "/private/..", str(link), str(link / "..")):
        assert desktop.refusal(Path(spelling), tmp_path) is not None, spelling


def test_the_home_folder_is_refused_through_symlinks_and_dots(tmp_path: Path) -> None:
    home = tmp_path / "home"
    (home / "Documents").mkdir(parents=True)
    hop1 = tmp_path / "hop1"
    hop1.symlink_to(home)
    hop2 = tmp_path / "hop2"
    hop2.symlink_to(hop1)
    for spelling in (home, hop1, hop2, home / ".", home / "Documents" / "..", Path(f"{home}/")):
        reason = desktop.refusal(spelling, hop2)  # the home itself may be a link as well
        assert reason is not None and "home folder" in reason, spelling
    assert desktop.refusal(home / "Documents", hop2) is None


@mac
def test_the_home_folder_is_refused_when_its_case_differs(tmp_path: Path) -> None:
    """APFS is case-insensitive by default: /users/NAME is the home folder."""
    probe = tmp_path / "CaseProbe"
    probe.mkdir()
    if not (tmp_path / "caseprobe").exists():
        pytest.skip("this volume is case-sensitive")
    assert desktop.refusal(tmp_path / "caseprobe", tmp_path / "CaseProbe") is not None


@pytest.mark.skipif(_ROOT, reason="root reads everything")
def test_an_unreadable_folder_from_the_command_line_is_refused_with_the_settings_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    locked = tmp_path / "Locked"
    locked.mkdir()
    locked.chmod(0)
    monkeypatch.setattr(sys, "platform", "darwin")
    try:
        assert main(["app", str(locked)]) == 1
    finally:
        locked.chmod(0o700)
    assert "Files and Folders" in capsys.readouterr().err


def test_a_file_given_as_the_folder_is_refused(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    (tmp_path / "a.pdf").write_bytes(b"x")
    monkeypatch.setattr(sys, "platform", "darwin")
    assert main(["app", str(tmp_path / "a.pdf")]) == 1
    assert "can't open" in capsys.readouterr().err


def test_the_dialog_starts_at_home_on_every_round(tmp_path: Path) -> None:
    starts: list[Path] = []
    answers = iter([Path("/"), tmp_path, tmp_path, None])

    def pick(start: Path) -> Path | None:
        starts.append(start)
        return next(answers)

    alerts: list[str] = []
    assert desktop.choose_folder(pick, alerts.append, tmp_path) is None
    assert starts == [tmp_path] * 4
    assert len(alerts) == 3  # "/", home and home again


def test_cancel_in_the_picker_is_none_even_with_extra_stderr(tmp_path: Path) -> None:
    run = _recorder(code=1, stderr="0:5: execution error: User canceled. (-128)\n")
    assert desktop.osascript_picker(tmp_path, run=run) is None


# --- 4. lifecycle against a real server ------------------------------------------------------


class _Ollama:
    """A loopback Ollama: /api/tags without the chat model, /api/pull streaming slowly."""

    def __init__(self) -> None:
        self.pull_bodies: list[dict[str, Any]] = []
        self.pull_ended = threading.Event()
        self.pull_cut = threading.Event()
        self.first_line = threading.Event()
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args: object) -> None:
                pass

            def do_GET(self) -> None:
                body = json.dumps({"models": [{"name": "nomic-embed-text:latest"}]}).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length", 0))
                outer.pull_bodies.append(json.loads(self.rfile.read(length) or b"{}"))
                self.send_response(200)
                self.send_header("Content-Type", "application/x-ndjson")
                self.end_headers()
                try:
                    for i in range(300):
                        line = json.dumps({"status": "downloading", "completed": i, "total": 300})
                        self.wfile.write(line.encode() + b"\n")
                        self.wfile.flush()
                        outer.first_line.set()
                        time.sleep(0.1)
                except OSError:
                    outer.pull_cut.set()
                outer.pull_ended.set()

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


class _AppProcess:
    def __init__(self, tmp_path: Path, ollama_url: str, *extra: str, watch: int | None = None):
        self.home = tmp_path / "home"
        docs = self.home / "docs"
        make_pdf(docs / "a.pdf", [text_page("Passwords need fifteen characters.")])
        self.beacon = tmp_path / "opened.log"
        browser = tmp_path / "fake-browser"
        browser.write_text(f"#!/bin/sh\necho \"$1\" >> '{self.beacon}'\n")
        browser.chmod(0o755)
        env = {
            **os.environ,
            "HOME": str(self.home),
            "ASKPDF_OLLAMA_URL": ollama_url,
            "BROWSER": f"{browser} %s",
            "PYTHONUNBUFFERED": "1",
        }
        env.pop("ASKPDF_API_KEY", None)
        self.port: int = cli.free_port(20000 + os.getpid() % 5000) or 0
        argv = [sys.executable, "-m", "askpdf", "app", str(docs), "--port", str(self.port)]
        if watch is not None:
            argv += ["--watch-pid", str(watch)]
        self.proc = subprocess.Popen(
            [*argv, *extra],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self.lines: queue.Queue[str] = queue.Queue()
        threading.Thread(target=self._pump, daemon=True).start()
        self.url = self._wait_for_url()

    def _pump(self) -> None:
        assert self.proc.stdout is not None
        for line in self.proc.stdout:
            self.lines.put(line.strip())

    def _wait_for_url(self) -> str:
        deadline = time.monotonic() + 40
        while time.monotonic() < deadline:
            try:
                line = self.lines.get(timeout=0.5)
            except queue.Empty:
                if self.proc.poll() is not None:
                    break
                continue
            if "running at" in line:
                # uvicorn runs on_start before it binds, so wait until it listens.
                for _ in range(100):
                    try:
                        socket.create_connection(("127.0.0.1", self.port), timeout=1).close()
                        break
                    except OSError:
                        time.sleep(0.1)
                return line.split("running at ", 1)[1]
        self.proc.kill()
        err = self.proc.stderr.read() if self.proc.stderr else ""
        raise AssertionError(f"the app never started: {err}")

    @property
    def token(self) -> str:
        return self.url.split("token=", 1)[1]

    @property
    def cookie(self) -> str:
        return f"{cli.cookie_name(self.port)}={self.token}" if hasattr(cli, "cookie_name") else ""

    def stop(self) -> None:
        if self.proc.poll() is None:
            self.proc.kill()
        self.proc.wait(10)
        for stream in (self.proc.stdout, self.proc.stderr):
            if stream:
                stream.close()


@pytest.fixture
def ollama_fake() -> Iterator[_Ollama]:
    fake = _Ollama()
    yield fake
    fake.close()


def _cookie(port: int, token: str) -> str:
    from askpdf.web.app import cookie_name

    return f"{cookie_name(port)}={token}"


@mac
@posix
def test_sigusr1_reopens_the_page_of_a_real_server(tmp_path: Path, ollama_fake: _Ollama) -> None:
    app = _AppProcess(tmp_path, ollama_fake.url)
    try:
        before = ""
        app.proc.send_signal(signal.SIGUSR1)
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and not (
            app.beacon.exists() and app.url in app.beacon.read_text()
        ):
            time.sleep(0.1)
        assert app.url in app.beacon.read_text(), before
        assert app.proc.poll() is None
    finally:
        app.stop()


@mac
@posix
def test_the_real_server_stops_soon_after_the_watched_pid_dies(
    tmp_path: Path, ollama_fake: _Ollama
) -> None:
    watched = subprocess.Popen(["/bin/sleep", "300"])
    app = _AppProcess(tmp_path, ollama_fake.url, "--no-browser", watch=watched.pid)
    try:
        assert app.proc.poll() is None
        time.sleep(3)
        assert app.proc.poll() is None, "stopped while the watched process was alive"
        watched.kill()
        watched.wait()
        started = time.monotonic()
        app.proc.wait(timeout=15)
        assert time.monotonic() - started < 6
        assert app.proc.returncode == 0
    finally:
        watched.kill()
        app.stop()


@mac
@posix
def test_sigterm_with_a_stream_open_stops_within_a_few_seconds_and_cuts_the_pull(
    tmp_path: Path, ollama_fake: _Ollama
) -> None:
    app = _AppProcess(tmp_path, ollama_fake.url, "--no-browser")
    conn = http.client.HTTPConnection("127.0.0.1", app.port, timeout=20)
    try:
        conn.request(
            "POST",
            "/api/model",
            body="{}",
            headers={
                "Cookie": _cookie(app.port, app.token),
                "Origin": f"http://127.0.0.1:{app.port}",
                "Content-Type": "application/json",
            },
        )
        response = conn.getresponse()
        assert response.status == 200
        assert b"progress" in response.fp.readline() + response.fp.readline()
        started = time.monotonic()
        app.proc.send_signal(signal.SIGTERM)
        app.proc.wait(timeout=15)
        took = time.monotonic() - started
        assert took < 5.5, f"shutdown took {took:.1f} s with a stream open"
        assert ollama_fake.pull_cut.wait(5), "the pull to Ollama was not cut"
    finally:
        conn.close()
        app.stop()


@mac
@posix
def test_closing_the_page_mid_download_stops_the_pull(tmp_path: Path, ollama_fake: _Ollama) -> None:
    app = _AppProcess(tmp_path, ollama_fake.url, "--no-browser")
    try:
        sock = socket.create_connection(("127.0.0.1", app.port), timeout=20)
        request = (
            "POST /api/model HTTP/1.1\r\n"
            f"Host: 127.0.0.1:{app.port}\r\n"
            f"Origin: http://127.0.0.1:{app.port}\r\n"
            f"Cookie: {_cookie(app.port, app.token)}\r\n"
            "Content-Type: application/json\r\nContent-Length: 2\r\n\r\n{}"
        )
        sock.sendall(request.encode())
        assert ollama_fake.first_line.wait(20)
        sock.recv(200)
        sock.close()  # the page went away
        assert ollama_fake.pull_cut.wait(10), "Ollama kept downloading after the page closed"
        # And the model asked for is the fixed one, whatever the body said.
        assert ollama_fake.pull_bodies == [{"model": ollama.CHAT_MODEL, "stream": True}]
    finally:
        app.stop()


# --- 5. the Ollama check ---------------------------------------------------------------------


def _capture_create_app(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    import uvicorn

    from askpdf.web import app as web_app

    seen: dict[str, Any] = {}
    real = web_app.create_app

    def spy(**kwargs: Any) -> Any:
        seen.update(kwargs)
        return real(**kwargs)

    monkeypatch.setattr(web_app, "create_app", spy)
    monkeypatch.setattr(uvicorn, "Server", FakeServer)
    monkeypatch.setattr("webbrowser.open", lambda url: None)
    return seen


def test_docs_model_is_null_for_a_remote_api_and_a_dict_for_local_ollama(
    folder: Path,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    seen = _capture_create_app(monkeypatch)
    args = ["serve", str(folder), "--index", str(folder.parent / "i.db"), "--no-browser"]
    monkeypatch.setenv("ASKPDF_API_KEY", "k")
    remote = ["--api-url", "http://127.0.0.1:9/v1", "--allow-remote", "--model", "m"]
    code = main([*args, *remote])
    assert code == 0, capsys.readouterr().err
    assert seen["model_status"] is None and seen["pull_model"] is None
    seen.clear()
    monkeypatch.delenv("ASKPDF_API_KEY")
    assert main(args) == 0
    assert callable(seen["model_status"]) and callable(seen["pull_model"])


def _silent_listener() -> tuple[socket.socket, str]:
    """Accepts the connection and never answers: a host that swallows the request."""
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(16)
    return listener, f"http://127.0.0.1:{listener.getsockname()[1]}"


def test_a_page_load_does_not_hang_when_ollama_swallows_the_request(
    folder: Path,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen = _capture_create_app(monkeypatch)
    listener, url = _silent_listener()
    monkeypatch.setenv("ASKPDF_OLLAMA_URL", url)
    try:
        index = str(folder.parent / "i.db")
        assert main(["serve", str(folder), "--index", index, "--no-browser"]) == 0
        app = _app(folder, model_status=seen["model_status"])
        with _client(app) as client:
            started = time.monotonic()
            first = client.get("/api/docs")
            took = time.monotonic() - started
            assert first.status_code == 200
            assert first.json()["model"]["running"] is False
            assert took < 4, f"the page waited {took:.1f} s for a silent Ollama"
            started = time.monotonic()
            client.get("/api/docs")
            assert time.monotonic() - started < 1, "a failed check was not cached"
    finally:
        listener.close()


def test_a_closed_port_answers_fast(
    folder: Path,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen = _capture_create_app(monkeypatch)
    assert main(["serve", str(folder), "--index", str(folder.parent / "i.db")]) == 0
    started = time.monotonic()
    status = seen["model_status"]()
    assert status["running"] is False
    assert time.monotonic() - started < 1


def test_a_tags_reply_with_null_models_does_not_break_the_page(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(providers_http, "get_json", lambda url, timeout=0: {"models": None})
    assert ollama.model_status("http://x", "m") == {"running": True, "model": "m", "pulled": False}


# --- POST /api/model --------------------------------------------------------------------------


def _model_app(folder: Path, calls: list[str]) -> Any:  # noqa: F811
    def pull() -> Iterator[dict[str, Any]]:
        calls.append("pull")
        yield {"status": "x"}

    return _app(folder, pull_model=pull)


def test_the_download_needs_a_session_and_the_same_origin(folder: Path) -> None:  # noqa: F811
    calls: list[str] = []
    app = _model_app(folder, calls)
    with _client(app, session=False) as client:
        assert client.post("/api/model", headers={"Origin": BASE}).status_code == 403
    with _client(app) as client:
        assert (
            client.post("/api/model", headers={"Origin": "http://evil.example"}).status_code == 403
        )
        assert client.post("/api/model").status_code == 403  # no Origin at all
        assert client.get("/api/model").status_code in (403, 404, 405)
    assert calls == []


def test_the_download_ignores_whatever_the_body_says(folder: Path) -> None:  # noqa: F811
    calls: list[str] = []
    with _client(_model_app(folder, calls)) as client:
        reply = client.post(
            "/api/model", json={"model": "../../evil", "url": "http://x"}, headers={"Origin": BASE}
        )
        assert [name for name, _ in _events(reply.text)] == ["progress", "done"]
    assert calls == ["pull"]


# --- 6. an empty folder -------------------------------------------------------------------------


def test_a_folder_still_being_indexed_is_not_called_empty(tmp_path: Path) -> None:
    """Before the first file is indexed, 'no PDFs in this folder' would be false."""
    docs = tmp_path / "docs"
    make_pdf(docs / "a.pdf", [text_page("Passwords need fifteen characters.")])
    release = threading.Event()

    def slow_index(progress: Progress) -> None:
        release.wait(10)

    app = _app(docs, index_folder=slow_index)
    with _client(app) as client:
        cid = client.post("/api/conversations", headers={"Origin": BASE}).json()["id"]
        reply = client.post(
            "/api/ask",
            json={"conversation_id": cid, "question": "passwords"},
            headers={"Origin": BASE},
        )
        release.set()
        final = _events(reply.text)[-1][1]
        assert final.get("message") != EMPTY_FOLDER, "told 'no PDFs' while the folder was indexing"


def test_a_folder_of_only_unreadable_pdfs_is_not_called_empty(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "broken.pdf").write_bytes(b"not a pdf")
    assert main(["index", str(docs), "--index", str(tmp_path / "i.db")]) in (0, 1)
    with _client(_app(docs)) as client:
        cid = client.post("/api/conversations", headers={"Origin": BASE}).json()["id"]
        reply = client.post(
            "/api/ask",
            json={"conversation_id": cid, "question": "anything"},
            headers={"Origin": BASE},
        )
        final = _events(reply.text)[-1][1]
        assert final.get("message") != EMPTY_FOLDER


# --- the page's JavaScript, run under node against a stub DOM --------------------------------

HARNESS = r"""
const fs = require("fs");
const vm = require("vm");
const source = fs.readFileSync(process.argv[2], "utf8");
const scenario = JSON.parse(process.argv[3]);

function mk(id) {
  const store = { id, hidden: false, disabled: false, textContent: "", listeners: {}, children: 0 };
  return new Proxy(store, {
    get(t, p) {
      if (typeof p === "symbol" || p === "then") return undefined;
      if (p in t) return t[p];
      if (p === "addEventListener") return (ev, f) => { (t.listeners[ev] ||= new Set()).add(f); };
      if (p === "classList") return { add() {} };
      if (p === "content") return { firstElementChild: { cloneNode: () => mk("turn") } };
      if (p === "append") return () => { t.children++; };
      if (p === "replaceChildren") return () => { t.children = 0; };
      return () => mk("anon");
    },
    set(t, p, v) { t[p] = v; return true; },
  });
}
const els = {};
const $ = (id) => (els[id] ||= mk(id));
const sse = (events) =>
  events.map((e) => `event: ${e.name}\ndata: ${JSON.stringify(e.data)}\n\n`).join("");
const streamOf = (text) => new Response(text, { headers: { "content-type": "text/event-stream" } });
let docs = scenario.docs;
const fetchLog = [];
const context = {
  document: { getElementById: $, createElement: () => mk("anon") },
  EventSource: class { addEventListener() {} },
  fetch: async (path, opts) => {
    fetchLog.push(path);
    if (path === "/api/docs") {
      const d = typeof docs === "function" ? docs() : docs;
      return { json: async () => d };
    }
    if (path === "/api/conversations") return { ok: true, json: async () => ({ id: "c1" }) };
    if (path === "/api/model") {
      if (scenario.pullStatus) return { ok: false, status: scenario.pullStatus };
      return streamOf(sse(scenario.pullEvents));
    }
    throw new Error("unexpected fetch " + path);
  },
  Response, TextDecoderStream, setInterval, clearInterval,
  Date, JSON, Math, Number, String, Map, Error, console,
};
context.window = context;
vm.createContext(context);
(async () => {
  vm.runInContext(source, context, { filename: "app.js" });
  await new Promise((r) => setTimeout(r, 20)); // the initial loadDocs
  const out = { fetchLog };
  const t = (id) => els[id] && els[id].textContent;
  out.afterLoad = {
    boxHidden: $("model").hidden, text: t("model-text"), buttonHidden: $("model-pull").hidden,
    clickHandlers: ($("model-pull").listeners.click || new Set()).size,
  };
  for (const step of scenario.steps) {
    if (step === "click-pull") {
      const handlers = [...($("model-pull").listeners.click || [])];
      out.handlersAtClick = handlers.length;
      for (const h of handlers) await h();
    } else if (step === "click-new-chat") {
      for (const h of [...($("new-chat").listeners.click || [])]) await h();
    } else if (step === "reload-docs") {
      vm.runInContext("loadDocs()", context);
      await new Promise((r) => setTimeout(r, 20));
    } else if (step.startsWith("set-docs:")) {
      docs = JSON.parse(step.slice(9));
    }
  }
  out.final = {
    boxHidden: $("model").hidden, text: t("model-text"),
    buttonHidden: $("model-pull").hidden, buttonDisabled: $("model-pull").disabled,
    docsChildren: $("docs").children,
  };
  out.pulling = vm.runInContext("pulling", context);
  console.log(JSON.stringify(out));
})().catch((e) => { console.log(JSON.stringify({ crash: String(e && e.stack || e) })); });
"""

APP_JS = Path(__file__).parent.parent / "src/askpdf/web/static/app.js"
node = pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")


def _docs(model: dict[str, Any] | None, documents: int = 1, failed: int = 0) -> dict[str, Any]:
    return {
        "folder": "f",
        "remote": [],
        "model": model,
        "documents": [
            {"path": f"{i}.pdf", "pages": 1, "id": i, "chunks": 1, "name": f"{i}.pdf"}
            for i in range(documents)
        ],
        "failed": [{"path": f"{i}.pdf", "reason": "bad"} for i in range(failed)],
    }


def _page(tmp_path: Path, scenario: dict[str, Any]) -> dict[str, Any]:
    harness = tmp_path / "harness.js"
    harness.write_text(HARNESS)
    done = subprocess.run(
        ["node", str(harness), str(APP_JS), json.dumps(scenario)],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert done.stdout.strip(), done.stderr
    out = json.loads(done.stdout.strip().splitlines()[-1])
    assert "crash" not in out, out["crash"]
    return out  # type: ignore[no-any-return]


MISSING = {"running": True, "model": "llama3.2:3b", "pulled": False}
DOWN = {"running": False, "model": "llama3.2:3b", "pulled": False}
READY = {"running": True, "model": "llama3.2:3b", "pulled": True}


@node
def test_js_banner_states(tmp_path: Path) -> None:
    for model, hidden, button_hidden, word in (
        (None, True, None, None),
        (READY, True, None, None),
        (DOWN, False, True, "Ollama"),
        (MISSING, False, False, "llama3.2:3b"),
    ):
        out = _page(tmp_path, {"docs": _docs(model), "steps": []})
        assert out["afterLoad"]["boxHidden"] is hidden, model
        if button_hidden is not None:
            assert out["afterLoad"]["buttonHidden"] is button_hidden, model
            assert word in out["afterLoad"]["text"]


@node
def test_js_the_download_button_works_as_soon_as_the_page_loads(tmp_path: Path) -> None:
    out = _page(
        tmp_path,
        {
            "docs": _docs(MISSING),
            "pullEvents": [
                {"name": "progress", "data": {"status": "x"}},
                {"name": "done", "data": {}},
            ],
            "steps": ["click-pull"],
        },
    )
    assert out["afterLoad"]["clickHandlers"] == 1, (
        "the button has no click handler until 'New chat'"
    )
    assert "/api/model" in out["fetchLog"]


@node
def test_js_a_download_error_stays_on_screen(tmp_path: Path) -> None:
    out = _page(
        tmp_path,
        {
            "docs": _docs(MISSING),
            "pullEvents": [{"name": "error", "data": {"message": "disk full"}}],
            "steps": ["click-new-chat", "click-pull"],  # 'New chat' first wires the button
        },
    )
    assert "disk full" in out["final"]["text"], out["final"]["text"]
    assert out["final"]["buttonDisabled"] is False and out["pulling"] is False


@node
def test_js_a_refused_download_shows_a_message_and_re_enables_the_button(tmp_path: Path) -> None:
    out = _page(
        tmp_path,
        {
            "docs": _docs(MISSING),
            "pullStatus": 403,
            "pullEvents": [],
            "steps": ["click-new-chat", "click-pull"],
        },
    )
    assert "403" in out["final"]["text"], out["final"]["text"]
    assert out["final"]["buttonDisabled"] is False


@node
def test_js_a_finished_download_hides_the_banner_once_ollama_has_the_model(tmp_path: Path) -> None:
    out = _page(
        tmp_path,
        {
            "docs": _docs(MISSING),
            "pullEvents": [{"name": "done", "data": {}}],
            "steps": ["click-new-chat", "set-docs:" + json.dumps(_docs(READY)), "click-pull"],
        },
    )
    assert out["final"]["boxHidden"] is True
    assert out["pulling"] is False


@node
def test_js_the_empty_folder_note(tmp_path: Path) -> None:
    out = _page(tmp_path, {"docs": _docs(None, documents=0), "steps": []})
    assert out["final"]["docsChildren"] == 1
    out = _page(tmp_path, {"docs": _docs(None, documents=0, failed=1), "steps": []})
    assert out["final"]["docsChildren"] == 0
