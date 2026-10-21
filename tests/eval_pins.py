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
