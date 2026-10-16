import unicodedata

import pytest
from hypothesis import given
from hypothesis import strategies as st

from askpdf.extract.clean import clean_text, detect_boilerplate


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Hello wor-\nld, world.", "Hello world, world."),  # solid form attested: rejoined
        ("Hello wor-\nld.", "Hello wor-ld."),  # not attested: lines joined, hyphen kept
        ("Covid-\n19 cases", "Covid-19 cases"),  # digit after the break: keep the hyphen
        ("Smith-\nJones", "Smith-Jones"),  # capital after the break: keep the hyphen
        ("räk-\nsmörgås och räksmörgås", "räksmörgås och räksmörgås"),  # non-ASCII letters
        ("multi­ple", "multiple"),  # soft hyphen
        ("ﬁnance", "finance"),  # ligature (NFKC)
        ("a\u00a0 b\t\tc", "a b c"),  # whitespace runs, incl. no-break space
        ("zero​width", "zerowidth"),  # format character
        ("bell\x07char", "bell char"),  # control character
        ("one\r\ntwo\rthree", "one\ntwo\nthree"),  # line endings
        ("  \n\nfirst\n \n second \n", "first\nsecond"),  # blank lines dropped
        ("", ""),
    ],
)
def test_clean_text(raw: str, expected: str) -> None:
    assert clean_text(raw) == expected


@given(st.text())
def test_clean_text_is_idempotent(raw: str) -> None:
    once = clean_text(raw)
    assert clean_text(once) == once


@given(st.text())
def test_clean_text_leaves_no_blank_lines_or_edge_spaces(raw: str) -> None:
    for line in clean_text(raw).split("\n"):
        if line:
            assert line == line.strip()
    assert "\n\n" not in clean_text(raw)


WORDS = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel", "india"]


def _body(i: int) -> list[str]:
    # Distinct words, not distinct numbers: digits are ignored when comparing lines.
    return [f"{WORDS[i]} {w} body line" for w in WORDS[:6]]


def _doc(n: int, header: str = "ACME Annual Report", footer: str = "Page {n}") -> list[list[str]]:
    return [[header, *_body(i), footer.format(n=i + 1)] for i in range(n)]


def test_repeated_header_and_numbered_footer_are_flagged() -> None:
    flags = detect_boilerplate(_doc(6))
    assert all(f == {0, 7} for f in flags)


def test_body_text_repeated_mid_page_is_not_flagged() -> None:
    pages = [
        ["Title", "intro", "x", "Repeated mid-page line", "y", "z", "w", "end"] for _ in range(6)
    ]
    flags = detect_boilerplate(pages)
    # Index 3 sits outside the first and last three lines of an 8-line page.
    assert all(3 not in f for f in flags)


def test_short_documents_are_never_flagged() -> None:
    assert detect_boilerplate(_doc(3)) == [set(), set(), set()]


def test_line_on_fewer_than_half_the_pages_is_not_flagged() -> None:
    pages = _doc(6, header="Header {n}")
    pages[0][0] = "Only once"
    pages[1][0] = "Only once"
    flags = detect_boilerplate(pages)
    assert 0 not in flags[0]


def test_threshold_is_half_rounded_up() -> None:
    # 7 pages: a line must appear on at least 4.
    pages = [[f"unique {WORDS[i]}", *_body(i)] for i in range(7)]
    for i in range(3):
        pages[i][0] = "Draft"
    assert all(0 not in f for f in detect_boilerplate(pages))
    pages[3][0] = "Draft"
    assert all(0 in f for f in detect_boilerplate(pages)[:4])


def test_case_and_digits_are_ignored_when_comparing() -> None:
    pages = [[f"CHAPTER {i} — Results", "body"] for i in range(5)]
    pages[0][0] = "chapter 99 — results"
    assert all(0 in f for f in detect_boilerplate(pages))


def test_empty_pages_are_handled() -> None:
    pages: list[list[str]] = [[], ["Header", "a"], ["Header", "b"], ["Header", "c"], []]
    flags = detect_boilerplate(pages)
    assert flags[0] == set() and flags[4] == set()
    assert all(0 in f for f in flags[1:4])


@pytest.mark.parametrize(
    "raw",
    ["e‍́", "e­́", "ᄀ​ᅡ"],  # reviewer's counterexamples
)
def test_removing_invisible_characters_leaves_composed_text(raw: str) -> None:
    once = clean_text(raw)
    assert clean_text(once) == once
    assert once == unicodedata.normalize("NFC", once)
