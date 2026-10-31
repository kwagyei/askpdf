"""The verbatim quote check (ADR-0006 Amendment 1).

A citation's quote is verified if, after the matching key ``M``, it is a substring of
the whole stored text of the cited page. ``M`` only removes differences of style:

1. case (``str.casefold``)
2. quote marks: single and double curly quotes and primes become ``'`` and ``"``
3. dashes: U+2010 to U+2015, U+2212, U+FE58, U+FE63 and U+FF0D become ``-``
4. whitespace: every run, newlines included, becomes one space; the ends are trimmed
5. a ``-`` between two letters, and one space after it, is deleted, so
   ``multi-factor``, ``multi- factor`` and ``multifactor`` match; next to a digit the
   hyphen stays, so ``SP 800-63B`` never matches ``SP 80063B``

No fuzzy matching: a changed letter, a dropped word or an inserted "not" is unverified.
The page side is the stored, already-cleaned text and is never cleaned again; the quote
side gets only the cleaning steps that do not depend on the document.
"""

import re
from dataclasses import dataclass

from askpdf.extract.clean import clean_page

MIN_WORDS = 5  # a shorter quote proves little ("in the United States")


def _chars(*codes: int) -> str:
    return "".join(map(chr, codes))


_QUOTES = str.maketrans(
    dict.fromkeys(_chars(0x2018, 0x2019, 0x201A, 0x201B, 0x2032), "'")
    | dict.fromkeys(_chars(0x201C, 0x201D, 0x201E, 0x201F, 0x2033), '"')
    | dict.fromkeys(_chars(*range(0x2010, 0x2016), 0x2212, 0xFE58, 0xFE63, 0xFF0D), "-")
)
_SPACE = re.compile(r"\s+")
_LETTER_HYPHEN = re.compile(r"(?<=[^\W\d_])- ?(?=[^\W\d_])")


def _style(text: str) -> str:
    """Steps 1 to 4 of ``M``."""
    return _SPACE.sub(" ", text.casefold().translate(_QUOTES)).strip()


def match_key(text: str, *, fold_hyphens: bool = True) -> str:
    """``M``: the form both sides are compared in. Idempotent."""
    key = _style(text)
    if fold_hyphens:
        key = _LETTER_HYPHEN.sub("", key)
    return key


def clean_quote(quote: str) -> str:
    """The document-independent part of indexing's cleaning, applied to a model's quote."""
    return clean_page(quote)


@dataclass(frozen=True, slots=True)
class QuoteCheck:
    quote: str  # as the model wrote it
    verified: bool
    # Verified only because hyphens between letters were folded (step 5); reported next
    # to the verified rate, so a loose key cannot inflate grounding unseen.
    needed_folding: bool = False
    reason: str = ""  # why an unverified quote failed, for display


class PageKeys:
    """``M`` of each page's text, computed once per page."""

    def __init__(self) -> None:
        self._keys: dict[tuple[str, int], tuple[str, str]] = {}

    def get(self, page: tuple[str, int], text: str) -> tuple[str, str]:
        if page not in self._keys:
            self._keys[page] = (match_key(text), match_key(text, fold_hyphens=False))
        return self._keys[page]


def check(quote: str, page_text: str, keys: tuple[str, str] | None = None) -> QuoteCheck:
    """Is ``quote`` on the page whose stored text is ``page_text``?"""
    cleaned = clean_quote(quote)
    if len(cleaned.split()) < MIN_WORDS:
        return QuoteCheck(quote, False, reason=f"shorter than {MIN_WORDS} words")
    folded_page, plain_page = keys or (
        match_key(page_text),
        match_key(page_text, fold_hyphens=False),
    )
    if match_key(cleaned) not in folded_page:
        return QuoteCheck(quote, False, reason="not found on the page")
    plain = match_key(cleaned, fold_hyphens=False) in plain_page
    return QuoteCheck(quote, True, needed_folding=not plain)
