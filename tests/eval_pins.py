"""Shared by the tests that reproduce committed evaluation results."""

import re
from pathlib import Path

import pypdf
import pytest

CONSTRAINTS = Path(__file__).resolve().parent.parent / "eval" / "constraints.txt"


def require_pinned_extractor() -> None:
    """Skip unless pypdf is the version the results were made with.

    The test jobs install the newest pypdf; the eval gate installs the pinned one and
    checks the results there, so a new pypdf release cannot fail unrelated PRs.
    """
    match = re.search(r"^pypdf==(\S+)", CONSTRAINTS.read_text(), re.MULTILINE)
    assert match is not None, "eval/constraints.txt must pin pypdf"
    if pypdf.__version__ != match[1]:
        pytest.skip(f"results are pinned to pypdf {match[1]}; this run has {pypdf.__version__}")


EVAL = CONSTRAINTS.parent


def synthetic_eval(dest: Path) -> Path:
    """An eval directory whose test split is a dev entry, never the real test split.

    Tests of the `--split test` guards use it: if a guard ever regresses, the frozen test
    questions are still not scored, printed or written anywhere (ADR-0007).
    """
    import shutil

    shutil.copytree(EVAL / "corpus", dest / "corpus")
    for folder in ("questions", "conversations"):
        (dest / folder).mkdir(parents=True)
        dev = (EVAL / folder / "dev.jsonl").read_text(encoding="utf-8")
        (dest / folder / "dev.jsonl").write_text(dev, encoding="utf-8")
        first = next(line for line in dev.splitlines() if line.strip())
        (dest / folder / "test.jsonl").write_text(first + "\n", encoding="utf-8")
    return dest
