"""Edge cases for the run record (PR #13): where git runs, what it reports, how it fails."""

import platform
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

from askpdf import cli

has_git = pytest.mark.skipif(shutil.which("git") is None, reason="needs git")


def _git(repo: Path, *args: str) -> str:
    done = subprocess.run(  # noqa: S603
        [str(shutil.which("git")), "-C", str(repo), *args],
        capture_output=True,
        text=True,
        check=True,
    )
    return done.stdout.strip()


def _commit(repo: Path, msg: str = "c") -> None:
    _git(repo, "-c", "user.name=t", "-c", "user.email=t@example.com", "commit", "-qm", msg)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "my repo with spaces"
    root.mkdir()
    _git(root, "init", "-q")
    (root / "a.txt").write_text("a", encoding="utf-8")
    (root / "sub").mkdir()
    (root / "sub" / "b.txt").write_text("b", encoding="utf-8")
    _git(root, "add", ".")
    _commit(root)
    return root


@pytest.fixture(autouse=True)
def _clean_git_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE"):
        monkeypatch.delenv(name, raising=False)


@has_git
def test_a_subdirectory_of_a_repo_with_spaces_reports_the_repo_commit(repo: Path) -> None:
    assert cli._git_commit(repo / "sub") == _git(repo, "rev-parse", "HEAD")


@has_git
def test_dirt_elsewhere_in_the_repo_marks_a_subdirectory_dirty(repo: Path) -> None:
    (repo / "a.txt").write_text("changed", encoding="utf-8")
    assert str(cli._git_commit(repo / "sub")).endswith("-dirty")


@has_git
def test_a_staged_only_change_is_dirty(repo: Path) -> None:
    (repo / "a.txt").write_text("changed", encoding="utf-8")
    _git(repo, "add", "a.txt")
    assert str(cli._git_commit(repo)).endswith("-dirty")


@has_git
def test_a_deleted_tracked_file_is_dirty(repo: Path) -> None:
    (repo / "a.txt").unlink()
    assert str(cli._git_commit(repo)).endswith("-dirty")


@has_git
def test_touching_a_file_without_changing_it_is_not_dirty(repo: Path) -> None:
    (repo / "a.txt").write_text("a", encoding="utf-8")  # same bytes, new mtime
    assert cli._git_commit(repo) == _git(repo, "rev-parse", "HEAD")


@has_git
def test_a_repo_with_no_commits_yet_has_no_commit(tmp_path: Path) -> None:
    _git(tmp_path, "init", "-q")
    assert cli._git_commit(tmp_path) is None


@has_git
def test_a_detached_head_still_records_the_sha(repo: Path) -> None:
    sha = _git(repo, "rev-parse", "HEAD")
    _git(repo, "checkout", "-q", "--detach", sha)
    assert cli._git_commit(repo) == sha


@has_git
def test_a_missing_directory_has_no_commit(tmp_path: Path) -> None:
    assert cli._git_commit(tmp_path / "nope") is None


@has_git
def test_an_ignored_untracked_and_a_staged_new_file(repo: Path) -> None:
    (repo / ".gitignore").write_text("*.log\n", encoding="utf-8")
    (repo / "x.log").write_text("x", encoding="utf-8")
    assert cli._git_commit(repo) == _git(repo, "rev-parse", "HEAD")
    (repo / "new.txt").write_text("n", encoding="utf-8")
    _git(repo, "add", "new.txt")  # tracked now, so it counts
    assert str(cli._git_commit(repo)).endswith("-dirty")


def test_a_hanging_git_gives_none(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(shutil, "which", lambda _: "/usr/bin/git")

    def hang(*args: Any, **kwargs: Any) -> Any:
        assert kwargs["timeout"] > 0  # there is a timeout to expire
        raise subprocess.TimeoutExpired(args[0], kwargs["timeout"])

    monkeypatch.setattr(subprocess, "run", hang)
    assert cli._git_commit(tmp_path) is None


def test_a_git_that_cannot_be_started_gives_none(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(shutil, "which", lambda _: str(tmp_path / "git"))  # not executable
    assert cli._git_commit(tmp_path) is None


def test_an_empty_machine_label_falls_back_to_the_platform(tmp_path: Path) -> None:
    assert cli._run_record("", tmp_path)["machine"] == platform.platform()


def test_the_record_has_exactly_the_three_fields(tmp_path: Path) -> None:
    assert set(cli._run_record(None, tmp_path)) == set(cli.RUN_RECORD)


def test_without_run_leaves_scores_alone() -> None:
    summary = {"date": "x", "machine": "y", "git_commit": None, "recall": 0.5}
    assert cli._without_run(summary) == {"recall": 0.5}
