"""Edge cases for extraction, cleaning, boilerplate and chunking (tester, PR 2).

Several tests here encode the design's requirement that cleaning "must never change
words" (quotes are later checked verbatim against page text). Where they fail, the
failure is a finding in the tester report, not a broken test.
"""

import unicodedata
from pathlib import Path

import pytest
from hypothesis import example, given
from hypothesis import strategies as st
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from askpdf.chunking import ChunkParams
from askpdf.extract.base import RawPage
from askpdf.extract.clean import clean_text, detect_boilerplate
from askpdf.extract.pypdf_ext import PypdfExtractor
from askpdf.indexer import build_records
from tests.pdfs import PageSpec, make_pdf, text_page

# --- PDF builders this file needs beyond tests/pdfs.py ---------------------------------


def _font(writer: PdfWriter, to_unicode: bytes | None = None) -> object:
    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    if to_unicode is not None:
        cmap = DecodedStreamObject()
        cmap.set_data(to_unicode)
        font[NameObject("/ToUnicode")] = writer._add_object(cmap)
    return writer._add_object(font)


def _page(writer: PdfWriter, ops: bytes, font: object) -> None:
    page = writer.add_blank_page(612, 792)
    page[NameObject("/Resources")] = DictionaryObject(
        {NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})}
    )
    contents = DecodedStreamObject()
    contents.set_data(ops)
    page[NameObject("/Contents")] = writer._add_object(contents)


def make_cmap_pdf(path: Path, maps_a_to: str) -> Path:
    """One page "ABAB some text" whose ToUnicode map sends glyph A to ``maps_a_to`` (hex)."""
    cmap = (
        "/CIDInit /ProcSet findresource begin 12 dict begin begincmap /CMapName /X def "
        "1 begincodespacerange <00> <FF> endcodespacerange "
        f"2 beginbfchar <41> <{maps_a_to}> <42> <0042> endbfchar "
        "endcmap CMapName currentdict /CMap defineresource pop end end"
    ).encode()
    writer = PdfWriter()
    _page(writer, b"BT /F1 11 Tf 72 700 Td (ABAB some text) Tj ET", _font(writer, cmap))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        writer.write(f)
    return path


def make_inline_image_pdf(path: Path) -> Path:
    """A page whose only content is an inline image (BI ... ID ... EI), as some scanners write."""
    writer = PdfWriter()
    ops = b"q 400 0 0 600 100 100 cm BI /W 1 /H 1 /CS /G /BPC 8 ID \x00 EI Q\n"
    _page(writer, ops, _font(writer))
    with path.open("wb") as f:
        writer.write(f)
    return path


def make_zero_page_pdf(path: Path) -> Path:
    with path.open("wb") as f:
        PdfWriter().write(f)
    return path


# --- cleaning: must never change words ----------------------------------------------------


@pytest.mark.parametrize(
    "compound",
    ["multi-factor", "out-of-band", "side-channel", "phishing-resistant", "non-exportable"],
)
def test_line_break_inside_a_hyphenated_compound_keeps_the_hyphen(compound: str) -> None:
    # F1. These are the exact compounds NIST SP 800-63B-4 breaks at a line end; the
    # current rule ("join before a lowercase letter") turns them into "multifactor" etc.
    head, tail = compound.split("-", 1)
    cleaned = clean_text(f"Use {head}-\n{tail} authentication.")
    assert compound in cleaned.replace("-\n", "-")


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        # "world" appears solid elsewhere, so the lexicon rule removes the hyphen.
        ("Hello wor­\nld, world.", "Hello world, world."),  # soft hyphen at the break
        ("Hello wor-\n ld, world.", "Hello world, world."),  # indented continuation line
        ("Hello wor- \nld, world.", "Hello world, world."),  # trailing space after the hyphen
        ("Hello wor-\n\nld, world.", "Hello world, world."),  # blank line in between
    ],
)
def test_line_end_hyphenation_variants_are_treated_alike(raw: str, expected: str) -> None:
    # F3. Whatever the cleaner decides for "wor-\nld" it should decide for these too.
    assert clean_text(raw) == expected


_HYPHEN_ALPHABET = st.sampled_from(
    # Letters, hyphens, whitespace, a soft hyphen, zero-width characters (ZWJ, ZWSP),
    # combining marks and Hangul jamo: removing an invisible character can leave a
    # sequence that only composes on a second pass.
    [
        *["a", "b", "e", "X", "-", "\n", " ", "\t"],
        *["\u00ad", "\u200d", "\u200b", "\u0301", "\u1100", "\u1161"],
    ]
)


@given(st.lists(_HYPHEN_ALPHABET, max_size=30).map("".join))
@example("a- \nb")  # the shrunk counterexample, so every profile fails alike
def test_clean_text_is_idempotent_around_hyphens_and_whitespace(raw: str) -> None:
    # F3. The existing idempotence property uses st.text(), which almost never
    # generates "-" next to "\n" and whitespace; this alphabet does.
    once = clean_text(raw)
    assert clean_text(once) == once


_PLAIN_WORD = st.text(
    alphabet=st.sampled_from(list("abcdefghijklmnopqrstuvwxyzåäöÅÄÖé.,;:!?'\"()0123456789")),
    min_size=1,
    max_size=12,
)


@given(st.lists(st.tuples(_PLAIN_WORD, st.sampled_from([" ", "\n", "  ", " \n "])), max_size=40))
def test_cleaning_never_changes_plain_words(pieces: list[tuple[str, str]]) -> None:
    raw = "".join(w + sep for w, sep in pieces)
    raw = unicodedata.normalize("NFC", raw)
    assert clean_text(raw).split() == raw.split()


def test_decomposed_swedish_letters_are_composed() -> None:
    nfd = unicodedata.normalize("NFD", "Räksmörgås på Åland")
    assert clean_text(nfd) == "Räksmörgås på Åland"


def test_unicode_line_separators_do_not_glue_words() -> None:
    for sep in ["\u2028", "\u2029", "\u0085", "\x0b", "\x0c", "\x1c"]:
        assert clean_text(f"alpha{sep}beta").split() == ["alpha", "beta"], repr(sep)


# --- image-only and scanned detection -----------------------------------------------------


def test_inline_image_scan_is_detected_as_image_only(tmp_path: Path) -> None:
    # F7. Only /XObject images are looked for; a page painted with an inline image is
    # not image-only, so a document of such pages never gets the "looks scanned" notice.
    page = PypdfExtractor().extract(make_inline_image_pdf(tmp_path / "inline.pdf"))[0]
    assert page.image_only


def test_control_characters_do_not_count_as_text_for_image_only() -> None:
    # F8. Cleaning turns these into nothing; they should not stop a page counting as a scan.
    assert RawPage(1, "1", "\x00" * 25, has_image=True).image_only
    assert RawPage(1, "1", "​" * 25, has_image=True).image_only


def test_exactly_half_image_only_is_not_scanned(tmp_path: Path) -> None:
    pages, _ = build_records(
        [RawPage(1, "1", "", True), RawPage(2, "2", "real text " * 5, False)], ChunkParams()
    )
    assert sum(p.image_only for p in pages) == 1  # 1 of 2: not "more than half"


# --- extraction edge cases ----------------------------------------------------------------


def test_zero_page_pdf_extracts_to_nothing(tmp_path: Path) -> None:
    assert PypdfExtractor().extract(make_zero_page_pdf(tmp_path / "zero.pdf")) == []


def test_lone_surrogate_from_a_bad_tounicode_map_is_cleaned(tmp_path: Path) -> None:
    # F2 (unit level). pypdf decodes ToUnicode maps with "surrogatepass", so a malformed
    # map yields lone surrogates; SQLite (UTF-8) then refuses the text.
    pdf = make_cmap_pdf(tmp_path / "bad.pdf", "D800")
    raw = PypdfExtractor().extract(pdf)[0].text
    assert "\ud800" in raw  # precondition: the extractor really produces one
    cleaned = clean_text(raw)
    cleaned.encode("utf-8")  # must not raise


def test_very_large_page_is_chunked_within_the_page(tmp_path: Path) -> None:
    lines = [f"line {i} has several words in it." for i in range(1500)]
    pdf = make_pdf(tmp_path / "big.pdf", [PageSpec(lines=lines), text_page("Second.")])
    raw = PypdfExtractor().extract(pdf)
    pages, chunks = build_records(raw, ChunkParams())
    words = pages[0].text.split()
    assert len(words) >= 1500 * 7
    page1 = [c for c in chunks if c.page == 1]
    assert len(page1) > 30
    assert all(c.text.split() == words[c.start_word : c.end_word] for c in page1)
    assert [c.page for c in chunks].count(2) == 1


def test_label_falls_back_to_physical_number_without_page_labels(tmp_path: Path) -> None:
    pdf = make_pdf(tmp_path / "a.pdf", [text_page("x")] * 3)
    assert [p.label for p in PypdfExtractor().extract(pdf)] == ["1", "2", "3"]


def test_labels_with_offset_start(tmp_path: Path) -> None:
    pdf = make_pdf(tmp_path / "a.pdf", [text_page("x")] * 3, labels=(0, 2, "/D", 14))
    pages = PypdfExtractor().extract(pdf)
    assert [(p.index, p.label) for p in pages] == [(1, "14"), (2, "15"), (3, "16")]


# --- boilerplate boundaries ------------------------------------------------------------------


def test_four_pages_with_a_header_on_exactly_two_is_flagged() -> None:
    pages = [["Header", f"body {w}"] for w in ["a", "b", "c", "d"]]
    pages[2][0] = "Other c"
    pages[3][0] = "Other d"
    flags = detect_boilerplate(pages)
    assert 0 in flags[0] and 0 in flags[1]  # 2 of 4 = 50%: recurs on >= 50%


def test_image_only_blank_pages_count_towards_the_page_total() -> None:
    # 6 pages, 3 blank: a header on the other 3 is exactly 50% and is flagged.
    pages: list[list[str]] = [[], [], [], ["Hdr", "a"], ["Hdr", "b"], ["Hdr", "c"]]
    assert all(0 in f for f in detect_boilerplate(pages)[3:])


def test_line_recurring_in_both_header_and_footer_counts_once_per_page() -> None:
    # On 2 of 5 pages the line appears twice; it must still need 3 distinct pages.
    pages = [["Rec", "x", "Rec"], ["Rec", "y", "Rec"], ["a", "b"], ["c", "d"], ["e", "f"]]
    assert all(f == set() for f in detect_boilerplate(pages))


# --- chunking over whole documents ---------------------------------------------------------

_page_text = st.lists(
    st.sampled_from(["alpha", "beta.", "gamma?", "δέλτα", "räk", "x", "\n", "end.\n"]),
    max_size=400,
).map(" ".join)


@given(st.lists(_page_text, min_size=1, max_size=6))
def test_document_chunks_stay_on_their_page_and_rebuild_it(texts: list[str]) -> None:
    params = ChunkParams(target=40, overlap=8, snap=6)
    raw = [RawPage(i + 1, str(i + 1), t, False) for i, t in enumerate(texts)]
    pages, chunks = build_records(raw, params)
    for page in pages:
        mine = [c for c in chunks if c.page == page.page]
        words = page.text.split()
        rebuilt: list[str] = []
        for i, c in enumerate(mine):
            assert c.text.split() == words[c.start_word : c.end_word]
            assert c.text.strip()
            rebuilt.extend(c.text.split()[0 if i == 0 else params.overlap :])
        assert rebuilt == words
        assert [c.ordinal for c in mine] == list(range(len(mine)))
    assert build_records(raw, params) == (pages, chunks)
