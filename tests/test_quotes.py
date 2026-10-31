"""The verbatim quote check and its matching key (ADR-0006 Amendment 1)."""

import pytest
from hypothesis import example, given
from hypothesis import strategies as st

from askpdf.quotes import PageKeys, check, match_key

PAGE = (
    "Verifiers SHALL require multi-factor authentication for\n"
    "all accounts covered by NIST SP 800-63B. The limit is 3-5 attempts,\n"
    "and the subscriber is not required to change passwords periodically."
)

# --- hyphen spellings ----------------------------------------------------------------------

SPELLINGS = ["multi-factor", "multi\u2010factor", "multi- factor", "multifactor"]


@pytest.mark.parametrize("on_page", SPELLINGS)
@pytest.mark.parametrize("in_quote", SPELLINGS)
def test_every_hyphen_spelling_verifies_against_every_other(on_page: str, in_quote: str) -> None:
    page = f"Verifiers SHALL require {on_page} authentication for all accounts."
    result = check(f"verifiers shall require {in_quote} authentication", page)
    assert result.verified
    # Folding decided it only when the two spellings differ in the hyphen itself.
    same = in_quote.replace("\u2010", "-") == on_page.replace("\u2010", "-")
    assert result.needed_folding is not same


def test_a_quote_without_folding_is_not_marked_as_needing_it() -> None:
    result = check("require multi-factor authentication for all", PAGE)
    assert result.verified and not result.needed_folding


# --- near-misses that must fail -------------------------------------------------------------


@pytest.mark.parametrize(
    "quote",
    [
        "all accounts covered by NIST SP 80063B",  # digits keep their hyphen
        "all accounts covered by NIST SP 800 63B",
        "The limit is 35 attempts",  # 3-5 is not 35
        "Verifiers SHALL require multi-factor authentification for",  # one changed letter
        "Verifiers SHALL require multi-factor authentication far",
        "the subscriber is required to change passwords periodically",  # "not" removed
        "Verifiers SHALL not require multi-factor authentication",  # "not" inserted
        "Verifiers SHALL require authentication for all accounts",  # a dropped word
        "Verifiers require multi-factor authentication for all",
        "verifiers shall require multi-factor authentication for all users",  # more words
    ],
)
def test_near_misses_are_not_verified(quote: str) -> None:
    result = check(quote, PAGE)
    assert not result.verified
    assert result.reason == "not found on the page"


def test_a_quote_spanning_two_pages_fails_on_each() -> None:
    first, second = "The policy is reviewed every", "year by the security team."
    quote = "The policy is reviewed every year by the security team"
    assert not check(quote, first).verified
    assert not check(quote, second).verified
    assert check(quote, f"{first}\n{second}").verified  # the same words on one page


def test_short_quotes_are_not_verified_even_when_present() -> None:
    result = check("SP 800-63B", PAGE)
    assert not result.verified and result.reason == "shorter than 5 words"
    assert check("covered by NIST SP 800-63B.", PAGE).verified  # five words


# --- differences of style that must pass ----------------------------------------------------


@pytest.mark.parametrize(
    "quote",
    [
        "VERIFIERS SHALL REQUIRE MULTI-FACTOR AUTHENTICATION",  # case
        "verifiers   shall\nrequire multi-factor\tauthentication",  # whitespace
        "limit is 3\u20135 attempts, and",  # en dash in a range
        "limit is 3\u22125 attempts, and",  # minus sign
        "the subscriber is not required to change passwords periodically.",
        "accounts covered by NIST SP 800\u201163B",  # non-breaking hyphen, NFKC'd away
    ],
)
def test_differences_of_style_still_verify(quote: str) -> None:
    assert check(quote, PAGE).verified


def test_quote_marks_are_unified() -> None:
    page = "He said \u201cthe system is ready\u201d and it\u2019s up to the team."
    assert check('said "the system is ready" and it\'s up', page).verified


def test_the_quote_gets_document_independent_cleaning() -> None:
    page = "The configuration file defines each setting."
    assert check(
        "The con\ufb01guration file de\u00adfines each", page
    ).verified  # \ufb01, soft hyphen
    assert check("The configuration\u200b file defines each setting", page).verified


def test_page_keys_are_computed_once_per_page() -> None:
    keys = PageKeys()
    first = keys.get(("a", 1), PAGE)
    assert keys.get(("a", 1), "different text is ignored") is first
    assert check("require multi-factor authentication for all", PAGE, first).verified


# --- properties -----------------------------------------------------------------------------

_TEXT = st.text(
    st.sampled_from([*"aAbB1 -.\n'\"", "\u2010", "\u2013", "\u201c", "\u2019", "\u00df", "\u0130"]),
    max_size=40,
)


@given(_TEXT)
@example("a- -b")
@example("a-- b")
@example("\u0130-a")
def test_the_matching_key_is_idempotent(text: str) -> None:
    key = match_key(text)
    assert match_key(key) == key
    plain = match_key(text, fold_hyphens=False)
    assert match_key(plain, fold_hyphens=False) == plain


_WORDS = st.lists(st.text("abcxyz", min_size=1, max_size=6), min_size=2, max_size=6)


@given(_WORDS, st.data())
def test_any_hyphen_spelling_of_a_page_has_the_same_key(
    words: list[str], data: st.DataObject
) -> None:
    joins = [data.draw(st.sampled_from(["", "-", "- ", "\u2010", "-\n"])) for _ in words[1:]]
    spelled = words[0] + "".join(j + w for j, w in zip(joins, words[1:], strict=True))
    assert match_key(spelled) == match_key("".join(words))
