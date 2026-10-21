"""The evaluation corpus is pinned: a changed file is a visible, reviewed act (ADR-0011)."""

import hashlib
from pathlib import Path

import pytest

CORPUS = Path(__file__).resolve().parent.parent / "eval" / "corpus"
pytestmark = pytest.mark.skipif(not CORPUS.is_dir(), reason="eval/ is not in the sdist")


def _sums() -> dict[str, str]:
    lines = (CORPUS / "SHA256SUMS").read_text().splitlines()
    return {name: digest for digest, name in (line.split(maxsplit=1) for line in lines)}


def test_every_pdf_matches_its_checksum() -> None:
    sums = _sums()
    assert sorted(sums) == sorted(p.name for p in CORPUS.glob("*.pdf"))
    for name, digest in sums.items():
        assert hashlib.sha256((CORPUS / name).read_bytes()).hexdigest() == digest, name


def test_every_pdf_has_a_sources_entry() -> None:
    sources = (CORPUS / "SOURCES.md").read_text()
    for name in _sums():
        assert f"## {name}" in sources, name


def test_the_corpus_stays_small() -> None:
    assert sum(p.stat().st_size for p in CORPUS.glob("*.pdf")) <= 30 * 1024 * 1024
