"""Checks against real-world PDFs that are not committed (tester, PR 2).

Set ASKPDF_REAL_PDF_DIR to a folder holding nist-sp-800-63b-4.pdf (NIST SP 800-63B-4,
public domain) to run them; otherwise they are skipped.
"""

import os
import re
from collections import Counter
from pathlib import Path

import pytest

from askpdf.chunking import ChunkParams
from askpdf.extract.clean import clean_document
from askpdf.extract.pypdf_ext import PypdfExtractor
from askpdf.indexer import build_records, index_folder
from askpdf.store import Store

_DIR = os.environ.get("ASKPDF_REAL_PDF_DIR")
NIST = Path(_DIR) / "nist-sp-800-63b-4.pdf" if _DIR else None
pytestmark = pytest.mark.skipif(
    NIST is None or not NIST.is_file(), reason="set ASKPDF_REAL_PDF_DIR to run"
)


@pytest.fixture(scope="module")
def nist() -> Path:
    assert NIST is not None
    return NIST


def test_nist_pages_labels_and_chunks(nist: Path) -> None:
    raw = PypdfExtractor().extract(nist)
    assert [p.index for p in raw] == list(range(1, len(raw) + 1))
    assert len(raw) == 129
    assert raw[4].label == "i"  # front matter in roman numerals, physical page 5
    pages, chunks = build_records(raw, ChunkParams())
    assert not any(p.image_only for p in pages)
    by_page = {p.page: p.text.split() for p in pages}
    for c in chunks:
        assert c.text.split() == by_page[c.page][c.start_word : c.end_word]


def test_nist_running_header_is_flagged_on_body_pages(nist: Path) -> None:
    pages, _ = build_records(PypdfExtractor().extract(nist), ChunkParams())
    body = pages[10:120]
    flagged = sum(
        any("NIST SP 800-63B-4" in p.text.split("\n")[i] for i in p.boilerplate) for p in body
    )
    assert flagged / len(body) > 0.9


def test_nist_dehyphenation_does_not_invent_words(nist: Path) -> None:
    # F1. For every line-end hyphen the cleaner removes, the solid form should be a
    # word the document uses elsewhere, not a hyphenated compound glued together.
    raw = PypdfExtractor().extract(nist)
    text = "\n".join(p.text for p in raw)
    vocab = Counter(re.findall(r"[\w-]+", text.replace("-\n", " ")))
    cleaned = "\n".join(clean_document([p.text for p in raw]))
    invented = []
    for head, tail in re.findall(r"(\w+)-\n(\w+)", text):
        solid = head + tail
        if solid in cleaned and vocab[solid] == 0:
            invented.append(f"{head}-{tail} -> {solid}")
    assert invented == []


def test_nist_reindex_is_a_no_op(nist: Path, tmp_path: Path) -> None:
    root = tmp_path / "f"
    root.mkdir()
    (root / nist.name).write_bytes(nist.read_bytes())
    with Store(tmp_path / "i.db") as store:
        index_folder(root, store, PypdfExtractor(), ChunkParams())
        generation = store.generation
        report = index_folder(root, store, PypdfExtractor(), ChunkParams())
        assert not report.changed and store.generation == generation
