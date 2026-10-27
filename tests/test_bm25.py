"""BM25 and its tokenizer, checked against scores worked out by hand."""

import math

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from askpdf.bm25 import BM25, STOPWORDS, tokenize

# --- tokenizer ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "multi-factor",
        "multi\u2010factor",  # U+2010 HYPHEN
        "multi\u2011factor",  # U+2011 NON-BREAKING HYPHEN
        "multi factor",
        "Multi-Factor",
        "multi\u2013factor",  # en dash
    ],
)
def test_hyphenated_and_spaced_compounds_give_the_same_tokens(text: str) -> None:
    assert tokenize(text) == ["multi", "factor"]


def test_document_numbers_split_at_hyphens() -> None:
    assert tokenize("SP 800-63B") == ["sp", "800", "63b"]


def test_swedish_letters_survive() -> None:
    assert tokenize("Räksmörgås på Åland") == ["räksmörgås", "på", "åland"]
    # A decomposed å (as macOS file names have) is composed first.
    assert tokenize("Åland") == ["åland"]


def test_casefold_and_compatibility_characters() -> None:
    assert tokenize("STRASSE Straße") == ["strasse", "strasse"]
    assert tokenize("ﬁnance") == ["finance"]  # the "fi" ligature


def test_stopwords_and_punctuation_are_dropped() -> None:
    assert tokenize("The cat, and the hat!") == ["cat", "hat"]
    assert tokenize("of the and") == []
    assert tokenize("don't") == ["don", "t"]
    assert tokenize("snake_case") == ["snake", "case"]


def test_stopword_list_is_lucenes_english_set() -> None:
    assert len(STOPWORDS) == 33
    assert {"the", "not", "no", "with"} <= STOPWORDS


@given(st.text())
def test_tokenizing_tokens_changes_nothing(text: str) -> None:
    tokens = tokenize(text)
    assert tokenize(" ".join(tokens)) == tokens


@given(st.text())
def test_tokens_are_never_empty_or_stopwords(text: str) -> None:
    assert all(token and token not in STOPWORDS for token in tokenize(text))


# --- BM25, by hand ------------------------------------------------------------------------
# Three documents: lengths 3, 2 and 1, so the average length is 2. k1 = 1.2, b = 0.75.
DOCS = [["apple", "banana", "apple"], ["banana", "cherry"], ["cherry"]]


@pytest.fixture
def bm25() -> BM25:
    return BM25.build(DOCS)


def test_lucene_idf(bm25: BM25) -> None:
    # apple is in 1 of 3 documents: ln(1 + (3 - 1 + 0.5) / (1 + 0.5)) = ln(8/3)
    assert bm25.idf("apple") == pytest.approx(math.log(8 / 3))
    # banana is in 2 of 3: ln(1 + 1.5 / 2.5) = ln(1.6)
    assert bm25.idf("banana") == pytest.approx(math.log(1.6))
    assert bm25.idf("durian") == 0.0


def test_single_term_scores(bm25: BM25) -> None:
    # Document 0: tf 2, length 3. norm = 1.2 * (0.25 + 0.75 * 3/2) = 1.65
    # score = ln(8/3) * 2 * 2.2 / (2 + 1.65) = 1.18237
    assert bm25.scores(["apple"]) == pytest.approx([1.1823695, 0.0, 0.0])


def test_two_term_scores_add_up(bm25: BM25) -> None:
    # banana: d0 ln(1.6) * 2.2 / (1 + 1.65) = 0.39019; d1 ln(1.6) * 2.2 / (1 + 1.2) = 0.47000
    # cherry: d1 0.47000 as above; d2 length 1, norm = 1.2 * (0.25 + 0.375) = 0.75,
    #         ln(1.6) * 2.2 / 1.75 = 0.59086
    assert bm25.scores(["banana", "cherry"]) == pytest.approx([0.3901917, 0.9400073, 0.5908617])


def test_a_repeated_query_term_counts_once(bm25: BM25) -> None:
    assert list(bm25.scores(["apple", "apple"])) == list(bm25.scores(["apple"]))


def test_a_term_in_every_document_still_scores_above_zero() -> None:
    bm25 = BM25.build([["x"], ["x", "y"]])
    assert bm25.idf("x") == pytest.approx(math.log(1 + 0.5 / 2.5))
    assert all(bm25.scores(["x"]) > 0)


def test_empty_corpus_and_empty_documents() -> None:
    assert BM25.build([]).scores(["x"]).shape == (0,)
    assert list(BM25.build([[], []]).scores(["x"])) == [0.0, 0.0]


def test_the_index_cannot_be_changed_after_it_is_built(bm25: BM25) -> None:
    with pytest.raises(ValueError, match="read-only"):
        bm25.lengths[0] = 99
    with pytest.raises(ValueError, match="read-only"):
        bm25.postings["apple"][1][0] = 99


_WORDS = st.lists(st.sampled_from(["a1", "b2", "c3", "d4", "e5"]), max_size=8)


@given(
    st.lists(_WORDS, max_size=8).flatmap(
        lambda docs: st.tuples(st.just(docs), st.permutations(range(len(docs))))
    ),
    st.lists(st.sampled_from(["a1", "b2", "zz"]), max_size=3),
)
def test_scores_are_non_negative_and_ignore_document_order(
    docs_and_order: tuple[list[list[str]], list[int]], query: list[str]
) -> None:
    docs, order = docs_and_order
    scores = BM25.build(docs).scores(query)
    assert (scores >= 0).all()
    shuffled = BM25.build([docs[i] for i in order]).scores(query)
    assert np.allclose(shuffled, scores[order])
