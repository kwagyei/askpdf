"""Line-end hyphens are resolved by the document's own spelling (ADR-0002 Amendment 1)."""

import re
import unicodedata
from pathlib import Path

import pytest
from hypothesis import example, given
from hypothesis import strategies as st

from askpdf.chunking import ChunkParams
from askpdf.extract import clean
from askpdf.extract.clean import Lexicon, clean_document, clean_page, resolve_hyphens
from askpdf.extract.pypdf_ext import PypdfExtractor
from askpdf.indexer import index_folder
from askpdf.store import Store
from tests.pdfs import PageSpec, make_pdf

# --- the rule, case by case ---------------------------------------------------------------


def test_broken_word_is_rejoined_when_the_document_spells_it_solid() -> None:
    pages = ["We describe the algo-\nrithm here.", "The algorithm is fast."]
    assert clean_document(pages)[0] == "We describe the algorithm here."


def test_broken_word_keeps_its_hyphen_when_the_document_never_spells_it_solid() -> None:
    assert clean_document(["We describe the algo-\nrithm here."]) == [
        "We describe the algo-rithm here."
    ]


def test_compound_keeps_its_hyphen_when_the_document_uses_it_hyphenated() -> None:
    pages = ["Use multi-\nfactor authentication.", "Multi-factor is required."]
    assert clean_document(pages)[0] == "Use multi-factor authentication."


def test_a_tie_keeps_the_hyphen() -> None:
    pages = ["Use multi-\nfactor here.", "multifactor and multi-factor both appear."]
    assert clean_document(pages)[0] == "Use multi-factor here."


def test_solid_form_must_outnumber_the_hyphenated_one() -> None:
    pages = ["Use multi-\nfactor here.", "multifactor, multifactor and multi-factor."]
    assert clean_document(pages)[0] == "Use multifactor here."


def test_the_lexicon_is_casefolded() -> None:
    pages = ["The Algo-\nrithm.", "ALGORITHM"]
    assert clean_document(pages)[0] == "The Algorithm."


def test_a_word_that_already_has_a_hyphen_keeps_the_break_hyphen() -> None:
    # TeX does not break a word that already holds an explicit hyphen.
    pages = ["An out-of-\nband channel.", "ofband"]
    assert clean_document(pages)[0] == "An out-of-band channel."


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("NIST SP 800-\n63B applies.", "NIST SP 800-63B applies."),
        ("Covid-\n19 cases", "Covid-19 cases"),
        ("Smith-\nJones", "Smith-Jones"),
        ("x2-\nb", "x2-b"),
    ],
)
def test_digits_and_capitals_join_the_lines_and_keep_the_hyphen(raw: str, expected: str) -> None:
    assert clean_document([raw, "smithjones jones covidcases"]) == [
        expected,
        "smithjones jones covidcases",
    ]


@pytest.mark.parametrize(
    "raw",
    [
        "Item one -\nitem two",  # a dash after a space
        "- first\n- second",  # list markers
        "word.-\nnext",  # punctuation before the hyphen
        "word-\n(next)",  # punctuation after the break
        "-\nalone",  # a lone hyphen
    ],
)
def test_dashes_and_list_markers_keep_the_line_break(raw: str) -> None:
    assert clean_document([raw]) == [raw]


def test_unicode_hyphens_are_line_end_hyphens_too() -> None:
    # U+2010 HYPHEN, and U+2011 NON-BREAKING HYPHEN, which NFKC maps to U+2010.
    assert unicodedata.normalize("NFKC", "\u2011") == "\u2010"
    assert clean_document(["multi\u2010\nfactor"]) == ["multi\u2010factor"]
    assert clean_document(["multi\u2011\nfactor"]) == ["multi\u2010factor"]
    assert clean_document(["algo\u2010\nrithm", "algorithm"])[0] == "algorithm"


def test_hyphenated_form_inside_a_longer_compound_counts() -> None:
    pages = ["two-\nfactor", "two-factor-based twofactor"]  # 1 solid vs 1 hyphenated: tie
    assert clean_document(pages)[0] == "two-factor"


def test_a_decision_never_counts_as_evidence_for_itself() -> None:
    lexicon = Lexicon.build(["the algo-\nrithm", "re-\ntrieval and re-trieval"])
    assert lexicon.words["algo"] == 0 and lexicon.words["rithm"] == 0
    assert lexicon.words["re"] == 0 and lexicon.words["trieval"] == 0
    assert lexicon.pairs["re-trieval"] == 1
    assert lexicon.words["the"] == 1


def test_chained_breaks_are_resolved_in_order() -> None:
    pages = ["a con-\ntin-\nuation", "continuation"]
    # "con-tin" is not attested, so it keeps its hyphen; then "con-tin" already has a
    # hyphen, so the second break keeps its hyphen too.
    assert clean_document(pages)[0] == "a con-tin-uation"


def test_a_break_across_pages_is_left_alone() -> None:
    assert clean_document(["the algo-", "rithm", "algorithm"])[:2] == ["the algo-", "rithm"]


def test_clean_page_keeps_line_end_hyphens_for_the_document_step() -> None:
    assert clean_page("multi-\nfactor") == "multi-\nfactor"


# --- properties ---------------------------------------------------------------------------

_LINE_ALPHABET = st.sampled_from(
    [*"abcAB12 -.\n", "\u2010", "\u2011", "\u00ad", "å", "ö", "\u0301", "\u1100", "\u1161"]
)
_PAGES = st.lists(st.lists(_LINE_ALPHABET, max_size=40).map("".join), max_size=4)
_BREAK = re.compile(r"[^\W_][-\u2010]\n[^\W_]")


@given(_PAGES)
@example(["ab-\nab", "ab"])
@example(["a-\nb-\nc", "ab abc"])
def test_document_cleaning_is_idempotent(pages: list[str]) -> None:
    once = clean_document(pages)
    assert clean_document(once) == once


@given(_PAGES)
def test_no_letter_or_digit_is_left_hyphenated_at_a_line_end(pages: list[str]) -> None:
    for page in clean_document(pages):
        assert not _BREAK.search(page), page


@given(_PAGES)
def test_cleaning_is_deterministic(pages: list[str]) -> None:
    assert clean_document(pages) == clean_document(list(pages))


@given(_PAGES)
def test_every_removed_hyphen_is_attested(pages: list[str]) -> None:
    cleaned = [clean_page(p) for p in pages]
    lexicon = Lexicon.build(cleaned)
    removed: list[tuple[str, str]] = []

    class Spy(Lexicon):
        def prefers_solid(self, left: str, right: str) -> bool:
            result = super().prefers_solid(left, right)
            if result:
                removed.append((left, right))
            return result

    spy = Spy(lexicon.words, lexicon.pairs)
    resolve_hyphens(cleaned, spy)
    for left, right in removed:
        solid = (left + right).casefold()
        assert lexicon.words[solid] > lexicon.pairs[f"{left.casefold()}-{right.casefold()}"]
        assert lexicon.words[solid] >= 1


# --- the index is rebuilt when cleaning changes -------------------------------------------


def test_a_new_cleaning_version_rebuilds_once_then_is_a_no_op(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "folder"
    root.mkdir()
    make_pdf(root / "a.pdf", [PageSpec(lines=["The algo-", "rithm and the algorithm."])])
    with Store(tmp_path / "index.db") as store:
        index_folder(root, store, PypdfExtractor(), ChunkParams())
        monkeypatch.setattr(clean, "CLEANING_VERSION", clean.CLEANING_VERSION + 1)
        rebuilt = index_folder(root, store, PypdfExtractor(), ChunkParams())
        again = index_folder(root, store, PypdfExtractor(), ChunkParams())
    assert rebuilt.rebuilt and rebuilt.added == ["a.pdf"]
    assert not again.changed


def test_an_indexed_pdf_has_the_broken_word_rejoined(tmp_path: Path) -> None:
    root = tmp_path / "folder"
    root.mkdir()
    make_pdf(root / "a.pdf", [PageSpec(lines=["The algo-", "rithm and the algorithm."])])
    with Store(tmp_path / "index.db") as store:
        index_folder(root, store, PypdfExtractor(), ChunkParams())
        text = store._db.execute("SELECT text FROM pages").fetchone()[0]
    assert text == "The algorithm and the algorithm."


# --- headers and footers are never joined (Amendment 1a) ----------------------------------

_DOC_LINES = st.sampled_from(["Header", "the multi-", "factor", "3", "4", "algo-", "rithm", "x"])
_DOCS = st.lists(
    st.lists(_DOC_LINES, min_size=1, max_size=7).map("\n".join), min_size=4, max_size=6
)


@given(_DOCS)
@example(["x\nthe multi-\n3", "x\nfactor\n4", "x\nalgo-\n3", "x\nrithm\n4"])
def test_documents_with_headers_and_footers_clean_idempotently(pages: list[str]) -> None:
    once = clean_document(pages)
    assert clean_document(once) == once


def test_a_footer_after_a_hyphenated_line_stays_on_its_own_line() -> None:
    pages = [f"Report title\nbody text-\n{n}" for n in range(1, 6)]
    assert clean_document(pages)[0] == "Report title\nbody text-\n1"
