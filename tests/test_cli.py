import runpy
import sys
from importlib.metadata import version

import pytest

from askpdf import __version__
from askpdf.cli import main


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


def test_no_arguments_succeeds() -> None:
    assert main([]) == 0


def test_runs_as_a_module(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys, "argv", ["askpdf", "--version"])
    with pytest.raises(SystemExit) as exc:
        runpy.run_module("askpdf", run_name="__main__")
    assert exc.value.code == 0
    assert capsys.readouterr().out == f"askpdf {__version__}\n"
