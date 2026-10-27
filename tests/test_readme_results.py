"""The README's result tables are rendered from eval/results (ADR-0007, README sync)."""

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
TOOL = ROOT / "eval" / "tools" / "readme_tables.py"
pytestmark = pytest.mark.skipif(not TOOL.is_file(), reason="eval/ is not in the sdist")


def test_every_results_block_matches_the_results_files() -> None:
    spec = importlib.util.spec_from_file_location("readme_tables", TOOL)
    assert spec is not None and spec.loader is not None
    tables = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tables)
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert {m["name"] for m in tables.BLOCK.finditer(readme)} == set(tables.RENDER)
    # If this fails, run `python eval/tools/readme_tables.py` and commit the README.
    assert tables.render_readme(readme) == readme
