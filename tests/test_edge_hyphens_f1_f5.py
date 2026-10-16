"""Tester probes for ADR-0002 Amendment 1 (hyphens) and ADR-0001 Amendment 1 (crypto)."""

import subprocess
import sys
from pathlib import Path

import pytest

from askpdf.chunking import ChunkParams
from askpdf.extract.base import RawPage
from askpdf.extract.clean import clean_document
from askpdf.indexer import build_records

FIXTURES = Path(__file__).parent / "fixtures" / "encrypted"
SOLID = " algorithm algorithm"


@pytest.mark.parametrize(
    "hyphen", ["-", "\u2010", "\u2011", "\uff0d", "\ufe63"], ids=lambda h: f"U+{ord(h):04X}"
)
def test_every_hyphen_spelling_resolves_like_the_ascii_one(hyphen: str) -> None:
    assert clean_document([f"x algo{hyphen}\nrithm.{SOLID}"]) == ["x algorithm." + SOLID]


def test_casefolded_evidence_and_uppercase_right_hand_side() -> None:
    assert clean_document(["Algo-\nrithm. ALGORITHM Algorithm"])[0].startswith("Algorithm.")
    assert clean_document(["algo-\nRithm. algoRithm algorithm"])[0].startswith("algo-Rithm.")


def test_evidence_on_another_page_counts() -> None:
    assert clean_document(["x algo-\nrithm", "algorithm algorithm"])[0] == "x algorithm"


def test_page_number_footer_is_not_glued_to_a_hyphenated_last_body_line() -> None:
    texts = [
        "Intro line\nthe multi-\n3" if i == 2 else f"Intro line\nbody\n{i + 1}" for i in range(1, 9)
    ]
    raw = [
        RawPage(index=i + 1, label=str(i + 1), text=t, has_image=False) for i, t in enumerate(texts)
    ]
    pages, _ = build_records(raw, ChunkParams())
    assert pages[1].text.endswith("\n3")
    assert 2 in pages[1].boilerplate


@pytest.mark.parametrize("name", ["rc4-128.pdf", "aes-128.pdf", "aes-256.pdf"])
def test_no_backend_at_all_is_reported_without_a_traceback(name: str, tmp_path: Path) -> None:
    code = (
        "import sys\n"
        "sys.modules['Crypto'] = None\n"
        "sys.modules['cryptography'] = None\n"
        "from pathlib import Path\n"
        "from askpdf.extract.base import ExtractionError\n"
        "from askpdf.extract.pypdf_ext import PypdfExtractor\n"
        "try:\n"
        "    PypdfExtractor().extract(Path(sys.argv[1]))\n"
        "except ExtractionError as e:\n"
        "    print('retry' if e.retry else 'final', '|', e)\n"
        "else:\n"
        "    print('read')\n"
    )
    out = subprocess.run(  # noqa: S603
        [sys.executable, "-c", code, str(FIXTURES / name)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if name == "rc4-128.pdf":
        assert out == "read"  # pypdf's own RC4 needs no backend
    else:
        assert out.startswith("retry | AES-encrypted; askpdf's crypto backend")
