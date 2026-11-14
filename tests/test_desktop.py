"""The Mac app's command and the page's model banner (ADR-0013).

The dialog itself can't run in CI, so ``osascript`` is replaced by a fake ``run`` and the
picker by a function: what is tested is everything around them.
"""

import json
import os
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from askpdf import cli, desktop
from askpdf.cli import main
from askpdf.providers import http, ollama
from askpdf.web.app import EMPTY_FOLDER, SEE_LOG
from tests.test_web import BASE, FakeServer, _app, _client, _events, folder  # noqa: F401

# --- the dialog -------------------------------------------------------------------------------


def _ran(stdout: str = "", stderr: str = "", code: int = 0) -> Any:
    calls: list[list[str]] = []

    def run(argv: list[str]) -> subprocess.CompletedProcess[str]:
        calls.append(argv)
        return subprocess.CompletedProcess(argv, code, stdout, stderr)

    run.calls = calls  # type: ignore[attr-defined]
    return run


def test_the_picker_returns_the_chosen_folder(tmp_path: Path) -> None:
    run = _ran(stdout="/Users/me/Papers/\n")
    assert desktop.osascript_picker(tmp_path, run) == Path("/Users/me/Papers")
    argv = run.calls[0]
    # The prompt and start folder go in as arguments, never into the script's text.
    assert argv[:2] == ["/usr/bin/osascript", "-e"]
    assert argv[3:] == [desktop.PROMPT, str(tmp_path)]
    assert str(tmp_path) not in argv[2]


def test_only_the_scripts_own_newline_is_dropped(tmp_path: Path) -> None:
    run = _ran(stdout="/Users/me/odd name\n\n")  # a folder whose name ends in a newline
    assert str(desktop.osascript_picker(tmp_path, run)) == "/Users/me/odd name\n"


def test_cancel_is_none(tmp_path: Path) -> None:
    run = _ran(stderr="execution error: User canceled. (-128)", code=1)
    assert desktop.osascript_picker(tmp_path, run) is None


def test_another_failure_is_an_error(tmp_path: Path) -> None:
    run = _ran(stderr="execution error: something else (-1700)", code=1)
    with pytest.raises(desktop.DialogError, match="-1700"):
        desktop.osascript_picker(tmp_path, run)


def test_the_alert_passes_its_message_as_an_argument() -> None:
    run = _ran()
    desktop.osascript_alert('say "hi"; do shell script "rm"', run)
    assert run.calls[0][3] == 'say "hi"; do shell script "rm"'
    assert "rm" not in run.calls[0][2]


# --- which folders are refused ------------------------------------------------------------------


def test_the_whole_disk_and_the_home_folder_are_refused(tmp_path: Path) -> None:
    root = Path(Path.cwd().anchor)
    assert "whole disk" in (desktop.refusal(root, tmp_path) or "")
    assert "whole home folder" in (desktop.refusal(tmp_path, tmp_path) or "")


@pytest.mark.skipif(sys.platform == "win32", reason="symlinks need privileges on Windows")
def test_a_link_to_the_home_folder_is_refused_too(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    (tmp_path / "link").symlink_to(home)
    assert desktop.refusal(tmp_path / "link", home) is not None


def test_an_ordinary_folder_is_fine(tmp_path: Path) -> None:
    (tmp_path / "papers").mkdir()
    assert desktop.refusal(tmp_path / "papers", tmp_path) is None


@pytest.mark.skipif(
    sys.platform == "win32" or os.geteuid() == 0, reason="needs POSIX permissions, not root"
)
def test_an_unreadable_folder_points_to_privacy_settings(tmp_path: Path) -> None:
    locked = tmp_path / "locked"
    locked.mkdir()
    locked.chmod(0)
    try:
        assert "Privacy & Security" in (desktop.refusal(locked, tmp_path) or "")
    finally:
        locked.chmod(0o700)


def test_a_missing_folder_is_refused(tmp_path: Path) -> None:
    assert "can't open" in (desktop.refusal(tmp_path / "gone", tmp_path) or "")


def test_the_dialog_comes_back_until_a_usable_folder_is_chosen(tmp_path: Path) -> None:
    papers = tmp_path / "papers"
    papers.mkdir()
    answers = iter([Path(tmp_path.anchor), tmp_path, papers])
    alerts: list[str] = []
    starts: list[Path] = []

    def pick(start: Path) -> Path | None:
        starts.append(start)
        return next(answers)

    assert desktop.choose_folder(pick, alerts.append, tmp_path) == papers
    assert len(alerts) == 2
    assert starts == [tmp_path] * 3  # always from home: nothing is remembered


def test_cancelling_the_dialog_chooses_nothing(tmp_path: Path) -> None:
    assert desktop.choose_folder(lambda start: None, pytest.fail, tmp_path) is None


# --- askpdf app ---------------------------------------------------------------------------------


def test_app_is_for_macos_only(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    assert main(["app"]) == 2
    assert "askpdf FOLDER" in capsys.readouterr().err


def test_cancelling_the_app_dialog_exits_quietly(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(desktop, "choose_folder", lambda *a: None)
    assert main(["app"]) == 0
    assert "no folder chosen" in capsys.readouterr().err


def test_a_dialog_that_cannot_open_is_an_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def broken(*args: Any) -> None:
        raise desktop.DialogError("no window server")

    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(desktop, "choose_folder", broken)
    assert main(["app"]) == 1
    assert "no window server" in capsys.readouterr().err


def test_app_refuses_the_home_folder_given_on_the_command_line(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    assert main(["app", str(tmp_path)]) == 1
    assert "whole home folder" in capsys.readouterr().err


def test_app_serves_the_chosen_folder_worded_for_the_app(
    folder: Path,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import uvicorn

    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(desktop, "choose_folder", lambda *a: folder)
    monkeypatch.setattr(uvicorn, "Server", FakeServer)
    monkeypatch.setattr(cli, "_index_path", lambda args: folder.parent / "i.db")
    FakeServer.instances.clear()
    assert main(["app", "--no-browser"]) == 0
    printed = capsys.readouterr()
    assert "?token=" in printed.out
    assert "quit askpdf from the Dock" in printed.err
    config = FakeServer.instances[-1].config
    assert config.host == "127.0.0.1"
    assert config.timeout_graceful_shutdown == cli.SHUTDOWN_GRACE


def test_alive() -> None:
    assert cli._alive(os.getpid())
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    child.wait()
    assert not cli._alive(child.pid)


@pytest.mark.skipif(sys.platform == "win32", reason="the app is macOS; signals are POSIX")
def test_the_server_stops_when_the_app_it_watches_has_gone(
    folder: Path,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli, "WATCH_PID_EVERY", 0.05)
    monkeypatch.setattr("webbrowser.open", lambda url: None)
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    child.wait()  # an app that has already quit
    args = cli.build_parser().parse_args(
        ["serve", str(folder), "--index", str(folder.parent / "i.db"), "--no-browser"]
    )
    args.port = cli.free_port()
    result: list[int] = []
    runner = threading.Thread(
        target=lambda: result.append(
            cli._serve(args, sys.stdout, sys.stderr, app_mode=True, watch_pid=child.pid)
        )
    )
    started = time.monotonic()
    runner.start()
    runner.join(30)
    assert result == [0]
    assert time.monotonic() - started < 30


# --- the page: app wording, an empty folder, the answer model --------------------------------


def test_the_app_page_points_to_the_dock_not_a_terminal(
    folder: Path,  # noqa: F811
) -> None:
    with _client(_app(folder, app_mode=True), session=False) as client:
        page = client.get("/")
        assert page.status_code == 403
        assert "Dock" in page.text and "terminal" not in page.text
        assert "Dock" in client.get("/api/docs").text


def test_an_answer_that_fails_points_to_the_log_in_the_app(
    folder: Path,  # noqa: F811
) -> None:
    class Broken:
        def stream(self, system: str, user: str) -> Iterator[str]:
            raise RuntimeError("boom")

    with _client(_app(folder, chat=Broken(), app_mode=True)) as client:
        cid = client.post("/api/conversations", headers={"Origin": BASE}).json()["id"]
        reply = client.post(
            "/api/ask",
            json={"conversation_id": cid, "question": "passwords"},
            headers={"Origin": BASE},
        )
        assert SEE_LOG in reply.text


def test_a_question_in_an_empty_folder_is_answered_without_the_model(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()

    class NoModel:
        def stream(self, system: str, user: str) -> Iterator[str]:
            pytest.fail("the model was asked")
            yield ""

    with _client(_app(empty, chat=NoModel())) as client:
        cid = client.post("/api/conversations", headers={"Origin": BASE}).json()["id"]
        reply = client.post(
            "/api/ask",
            json={"conversation_id": cid, "question": "anything"},
            headers={"Origin": BASE},
        )
        events = _events(reply.text)
        assert events[-1][0] == "final"
        assert events[-1][1]["message"] == EMPTY_FOLDER


def test_docs_carries_the_model_status_when_given(folder: Path) -> None:  # noqa: F811
    with _client(_app(folder)) as client:
        assert client.get("/api/docs").json()["model"] is None
    status = {"running": True, "model": "llama3.2:3b", "pulled": False}
    with _client(_app(folder, model_status=lambda: status)) as client:
        assert client.get("/api/docs").json()["model"] == status


def test_the_model_download_streams_progress(folder: Path) -> None:  # noqa: F811
    def pull() -> Iterator[dict[str, Any]]:
        yield {"status": "pulling manifest"}
        yield {"status": "downloading", "completed": 5, "total": 10}

    with _client(_app(folder, pull_model=pull)) as client:
        reply = client.post("/api/model", headers={"Origin": BASE})
        events = _events(reply.text)
        assert [name for name, _ in events] == ["progress", "progress", "done"]
        assert events[1][1] == {"status": "downloading", "completed": 5, "total": 10}


def test_a_failed_download_says_why(folder: Path) -> None:  # noqa: F811
    def pull() -> Iterator[dict[str, Any]]:
        yield {"status": "pulling manifest"}
        raise http.ProviderError("Ollama could not download llama3.2:3b: no space left")

    with _client(_app(folder, pull_model=pull)) as client:
        events = _events(client.post("/api/model", headers={"Origin": BASE}).text)
        assert events[-1] == (
            "error",
            {"message": "Ollama could not download llama3.2:3b: no space left"},
        )


def test_the_download_is_only_offered_for_a_local_ollama(folder: Path) -> None:  # noqa: F811
    with _client(_app(folder)) as client:
        assert client.post("/api/model", headers={"Origin": BASE}).status_code == 404


def test_another_site_cannot_start_the_download(folder: Path) -> None:  # noqa: F811
    with _client(_app(folder, pull_model=lambda: pytest.fail("pulled"))) as client:
        reply = client.post("/api/model", headers={"Origin": "https://evil.example"})
        assert reply.status_code == 403


def test_the_download_takes_no_input(folder: Path) -> None:  # noqa: F811
    asked: list[int] = []

    def pull() -> Iterator[dict[str, Any]]:
        asked.append(1)
        yield {"status": "success"}

    with _client(_app(folder, pull_model=pull)) as client:
        reply = client.post(
            "/api/model?model=evil", json={"model": "evil"}, headers={"Origin": BASE}
        )
        assert reply.status_code == 200 and asked == [1]  # the same one model, whatever is sent


# --- Ollama: model status and pull ----------------------------------------------------------------


def test_model_status_when_ollama_is_not_running(monkeypatch: pytest.MonkeyPatch) -> None:
    def down(url: str, timeout: float = 10.0) -> Any:
        raise http.UnreachableError("nothing answered")

    monkeypatch.setattr(http, "get_json", down)
    assert ollama.model_status("http://127.0.0.1:1", "m") == {
        "running": False,
        "model": "m",
        "pulled": False,
    }


@pytest.mark.parametrize(("names", "pulled"), [(["m"], True), (["m:latest"], True), (["x"], False)])
def test_model_status_finds_the_model(
    names: list[str], pulled: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    tags = {"models": [{"name": n} for n in names]}
    monkeypatch.setattr(http, "get_json", lambda url, timeout=10.0: tags)
    assert ollama.model_status("http://127.0.0.1:1", "m")["pulled"] is pulled


def test_pull_yields_progress_and_raises_ollamas_error(monkeypatch: pytest.MonkeyPatch) -> None:
    lines = [
        json.dumps({"status": "pulling manifest"}),
        json.dumps({"status": "downloading", "digest": "sha", "completed": 1, "total": 2}),
        json.dumps({"error": "disk full"}),
    ]
    sent: list[dict[str, Any]] = []

    def stream(url: str, payload: dict[str, Any], *a: Any, **k: Any) -> Iterator[str]:
        sent.append(payload)
        yield from lines

    monkeypatch.setattr(http, "stream_lines", stream)
    got: list[dict[str, Any]] = []
    with pytest.raises(http.ProviderError, match="disk full"):
        for event in ollama.pull("http://127.0.0.1:1", "m"):
            got.append(event)
    assert got == [
        {"status": "pulling manifest"},
        {"status": "downloading", "completed": 1, "total": 2},
    ]
    assert sent == [{"model": "m", "stream": True}]


def test_the_page_loads_with_the_banner_hidden_by_default() -> None:
    html = (Path(__file__).parent.parent / "src/askpdf/web/static/index.html").read_text("utf-8")
    assert 'id="model"' in html and "hidden" in html.split('id="model"')[1].split(">")[0]


@pytest.mark.skipif(sys.platform == "win32", reason="SIGUSR1 is POSIX")
def test_opening_the_app_again_shows_the_page_again(
    folder: Path,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import signal

    import uvicorn

    opened: list[str] = []

    class Reopened(FakeServer):
        def run(self) -> None:
            super().run()  # the first page, from on_start
            os.kill(os.getpid(), signal.SIGUSR1)  # what the applet's "on reopen" sends

    monkeypatch.setattr(uvicorn, "Server", Reopened)
    monkeypatch.setattr("webbrowser.open", opened.append)
    previous = signal.getsignal(signal.SIGUSR1)
    try:
        args = cli.build_parser().parse_args(
            ["serve", str(folder), "--index", str(folder.parent / "i.db")]
        )
        assert cli._serve(args, sys.stdout, sys.stderr, app_mode=True) == 0
    finally:
        signal.signal(signal.SIGUSR1, previous)
    assert len(opened) == 2 and opened[0] == opened[1] and "?token=" in opened[0]
