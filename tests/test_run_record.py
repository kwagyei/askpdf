"""Model runs record when, where and from which commit they were made (ADR-0007 Amendment 1)."""

import importlib.util
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

from askpdf import cli

ROOT = Path(__file__).resolve().parent.parent
TOOL = ROOT / "eval" / "tools" / "readme_tables.py"
has_git = pytest.mark.skipif(shutil.which("git") is None, reason="needs git")


def _git(repo: Path, *args: str) -> str:
    done = subprocess.run(  # noqa: S603
        [str(shutil.which("git")), "-C", str(repo), *args],
        capture_output=True,
        text=True,
        check=True,
    )
    return done.stdout.strip()


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    _git(tmp_path, "init", "-q")
    (tmp_path / "a.txt").write_text("a", encoding="utf-8")
    _git(tmp_path, "add", "a.txt")
    _git(tmp_path, "-c", "user.name=t", "-c", "user.email=t@example.com", "commit", "-qm", "a")
    return tmp_path


@has_git
def test_a_clean_checkout_records_its_commit(repo: Path) -> None:
    assert cli._git_commit(repo) == _git(repo, "rev-parse", "HEAD")


@has_git
def test_changed_tracked_files_mark_the_commit_dirty(repo: Path) -> None:
    (repo / "a.txt").write_text("b", encoding="utf-8")
    assert cli._git_commit(repo) == _git(repo, "rev-parse", "HEAD") + "-dirty"


@has_git
def test_untracked_files_do_not_mark_it_dirty(repo: Path) -> None:
    (repo / "results.json").write_text("{}", encoding="utf-8")  # a run writes its results
    assert cli._git_commit(repo) == _git(repo, "rev-parse", "HEAD")


@has_git
def test_outside_a_checkout_there_is_no_commit(tmp_path: Path) -> None:
    (tmp_path / "eval").mkdir()
    # GIT_CEILING_DIRECTORIES stops git finding a repository above tmp_path.
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path))
        assert cli._git_commit(tmp_path / "eval") is None


def test_without_git_there_is_no_commit(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(shutil, "which", lambda _: None)
    assert cli._git_commit(tmp_path) is None


def test_the_machine_defaults_to_the_platform(tmp_path: Path) -> None:
    import platform

    assert cli._run_record(None, tmp_path)["machine"] == platform.platform()
    assert cli._run_record("my laptop", tmp_path)["machine"] == "my laptop"


@pytest.mark.skipif(not (ROOT / "eval" / "questions").is_dir(), reason="eval/ is not in sdist")
def test_check_ignores_when_where_and_which_commit(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    eval_dir = tmp_path / "eval"
    shutil.copytree(ROOT / "eval" / "questions", eval_dir / "questions")
    shutil.copytree(ROOT / "eval" / "corpus", eval_dir / "corpus")
    base = ["eval", "retrieval", "--split", "test", "--release", "v9.9.9"]
    base += ["--eval-dir", str(eval_dir)]
    assert cli.main(base) == 0
    path = eval_dir / "results" / "test-v9.9.9.json"
    saved = json.loads(path.read_text(encoding="utf-8"))
    saved["stages"]["retrieval"].update(date="2000-01-01", machine="old", git_commit="0" * 40)
    path.write_text(json.dumps(saved), encoding="utf-8")
    assert cli.main([*base, "--check"]) == 0
    saved["stages"]["retrieval"]["n_questions"] += 1  # a score still has to match
    path.write_text(json.dumps(saved), encoding="utf-8")
    assert cli.main([*base, "--check"]) == 1
    assert "n_questions" in capsys.readouterr().err


@pytest.mark.skipif(not TOOL.is_file(), reason="eval/ is not in the sdist")
def test_the_readme_uses_a_runs_own_date_and_machine() -> None:
    spec = importlib.util.spec_from_file_location("readme_tables", TOOL)
    assert spec is not None and spec.loader is not None
    tables: Any = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tables)
    run = {"date": "2027-01-02", "machine": "box"}
    assert tables._date(run, "fallback") == "2027-01-02"
    assert tables._machine(run) == "box"
    assert tables._date({}, "fallback") == "fallback"  # files from before v0.1.1
    assert tables._machine({}) == tables._load(f"bench-{tables.RELEASE}.json")["machine"]
