import os
import runpy
import sys
from importlib.metadata import version
from pathlib import Path

import pytest

from askpdf import __version__
from askpdf.cli import main, shown
from askpdf.store import Store
from tests.pdfs import PageSpec, make_pdf, text_page


def test_version_matches_installed_metadata() -> None:
    assert __version__ == version("askpdf")


def test_version_flag_prints_name_and_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert capsys.readouterr().out == f"askpdf {__version__}\n"


def test_unknown_option_is_a_usage_error(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--no-such-option"])
    assert exc.value.code == 2
    assert "unrecognized arguments" in capsys.readouterr().err


def test_no_command_prints_help_and_is_a_usage_error(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([]) == 2
    assert "usage: askpdf" in capsys.readouterr().err


def test_runs_as_a_module(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys, "argv", ["askpdf", "--version"])
    with pytest.raises(SystemExit) as exc:
        runpy.run_module("askpdf", run_name="__main__")
    assert exc.value.code == 0
    assert capsys.readouterr().out == f"askpdf {__version__}\n"


@pytest.fixture
def folder(tmp_path: Path) -> Path:
    make_pdf(tmp_path / "pdfs" / "a.pdf", [text_page("Alpha page one."), text_page("Two.")])
    make_pdf(tmp_path / "pdfs" / "scan.pdf", [PageSpec(image=True)] * 2)
    (tmp_path / "pdfs" / "broken.pdf").write_bytes(b"%PDF-1.4 nope")
    return tmp_path / "pdfs"


def _args(folder: Path, command: str) -> list[str]:
    return [command, str(folder), "--index", str(folder.parent / "idx" / "i.db")]


def test_index_then_docs(folder: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(_args(folder, "index")) == 0
    out, err = capsys.readouterr()
    assert out == "2 added, 0 already known (copies or renames), 0 removed, 0 unchanged, 1 failed\n"
    assert "reading a.pdf" in err and "failed: broken.pdf: cannot read PDF" in err

    assert main(_args(folder, "docs")) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == "a.pdf  2 pages, 2 chunks, 0 image-only"
    assert lines[1].startswith("scan.pdf  2 pages, 0 chunks, 2 image-only  [looks scanned")
    assert lines[2].startswith("broken.pdf  FAILED: cannot read PDF")


def test_second_index_run_reports_everything_unchanged(
    folder: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    main(_args(folder, "index"))
    capsys.readouterr()
    assert main(_args(folder, "index")) == 0
    out, err = capsys.readouterr()
    assert "0 added" in out and "3 unchanged" in out and "0 failed" in out
    assert "reading" not in err


def test_docs_lists_copies_on_one_line(folder: Path, capsys: pytest.CaptureFixture[str]) -> None:
    (folder / "b.pdf").write_bytes((folder / "a.pdf").read_bytes())
    main(_args(folder, "index"))
    main(_args(folder, "docs"))
    assert "a.pdf  2 pages, 2 chunks, 0 image-only  (also: b.pdf)" in capsys.readouterr().out


def test_docs_before_indexing_says_what_to_do(
    folder: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(_args(folder, "docs")) == 0
    assert "nothing indexed yet; run: askpdf index" in capsys.readouterr().err


def test_missing_folder_is_an_error(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["index", str(tmp_path / "nope")]) == 1
    assert "does not exist" in capsys.readouterr().err


def test_foreign_index_file_is_refused(folder: Path, capsys: pytest.CaptureFixture[str]) -> None:
    precious = folder.parent / "precious.txt"
    precious.write_text("my notes " * 100)
    assert main(["index", str(folder), "--index", str(precious)]) == 1
    assert "not an askpdf index" in capsys.readouterr().err
    assert precious.read_text().startswith("my notes")


def test_default_index_lives_in_the_cache_dir(
    folder: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr("askpdf.store.cache_dir", lambda: tmp_path / "cache")
    assert main(["index", str(folder)]) == 0
    assert f"index: {tmp_path / 'cache' / 'askpdf'}" in capsys.readouterr().err
    assert not any(p.suffix == ".db" for p in folder.rglob("*"))  # nothing written into the folder


def test_index_of_another_folder_is_not_reused(
    folder: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    idx = str(tmp_path / "idx" / "i.db")
    main(["index", str(folder), "--index", idx])
    other = tmp_path / "other"
    make_pdf(other / "z.pdf", [text_page("Zulu.")])
    capsys.readouterr()
    assert main(["index", str(other), "--index", idx]) == 1
    assert "is the index of" in capsys.readouterr().err
    with Store(Path(idx)) as store:
        assert "a.pdf" in store.files()  # untouched


def test_ctrl_c_exits_130_with_a_hint(
    folder: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def interrupt(*_args: object, **_kwargs: object) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr("askpdf.cli.index_folder", interrupt)
    assert main(_args(folder, "index")) == 130
    assert "run the same command again" in capsys.readouterr().err


@pytest.mark.skipif(sys.platform == "win32" or os.geteuid() == 0, reason="POSIX permissions")
def test_unreadable_subfolder_is_reported_not_dropped(
    folder: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    make_pdf(folder / "sub" / "b.pdf", [text_page("Bravo.")])
    main(_args(folder, "index"))
    (folder / "sub").chmod(0)
    try:
        assert main(_args(folder, "index")) == 0
    finally:
        (folder / "sub").chmod(0o755)
    out, err = capsys.readouterr()
    assert "skipped unreadable folder sub" in err
    assert out.splitlines()[-1].startswith("0 added") and "0 removed" in out


@pytest.mark.skipif(sys.platform == "win32" or os.geteuid() == 0, reason="POSIX permissions")
def test_unreadable_root_is_an_error(folder: Path, capsys: pytest.CaptureFixture[str]) -> None:
    main(_args(folder, "index"))
    folder.chmod(0)
    try:
        assert main(_args(folder, "index")) == 1
    finally:
        folder.chmod(0o755)
    assert "the index was left as it was" in capsys.readouterr().err


def test_index_in_an_unwritable_place_is_a_clean_error(
    folder: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    blocker = tmp_path / "file"
    blocker.write_text("x")
    assert main(["index", str(folder), "--index", str(blocker / "sub" / "i.db")]) == 1
    assert "cannot open the index" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("name", "expected"),
    [("plain.pdf", "plain.pdf"), ("a\nb.pdf", "a\\nb.pdf"), ("e\x1b[31m.pdf", "e\\x1b[31m.pdf")],
)
def test_shown_escapes_invisible_characters(name: str, expected: str) -> None:
    assert shown(name) == expected
