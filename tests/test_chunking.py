import itertools

import pytest
from hypothesis import given
from hypothesis import strategies as st

from askpdf.chunking import ChunkParams, chunk_page, chunk_words

words_st = st.lists(
    st.sampled_from(["alpha", "beta", "end.", "why?", "wow!", 'said."', "(see", "it).", "x"]),
    max_size=600,
)


def _params(target: int, overlap: int, snap: int) -> ChunkParams | None:
    try:
        return ChunkParams(target, overlap, snap)
    except ValueError:
        return None


valid_params = st.builds(
    _params, st.integers(5, 120), st.integers(0, 30), st.integers(0, 30)
).filter(lambda p: p is not None)


@given(words_st, valid_params)
def test_chunks_reproduce_the_page_exactly(words: list[str], params: ChunkParams) -> None:
    chunks = chunk_words(words, params)
    rebuilt: list[str] = []
    for i, c in enumerate(chunks):
        piece = c.text.split()
        rebuilt.extend(piece if i == 0 else piece[params.overlap :])
    assert rebuilt == words


@given(words_st, valid_params)
def test_chunks_are_contiguous_slices_with_exact_overlap(
    words: list[str], params: ChunkParams
) -> None:
    chunks = chunk_words(words, params)
    for c in chunks:
        assert c.text == " ".join(words[c.start : c.end])
        assert c.end > c.start  # no empty chunks
    for prev, nxt in itertools.pairwise(chunks):
        assert nxt.start == prev.end - params.overlap
    if words:
        assert chunks[0].start == 0 and chunks[-1].end == len(words)
    else:
        assert chunks == []


@given(words_st, valid_params)
def test_chunk_length_stays_within_snap_of_target(words: list[str], params: ChunkParams) -> None:
    chunks = chunk_words(words, params)
    for c in chunks[:-1]:  # the last chunk takes whatever is left
        assert abs((c.end - c.start) - params.target) <= params.snap


@given(words_st, valid_params)
def test_chunking_is_deterministic(words: list[str], params: ChunkParams) -> None:
    assert chunk_words(words, params) == chunk_words(list(words), params)


def test_snaps_to_the_nearest_sentence_end() -> None:
    words = [f"w{i}" for i in range(30)]
    words[11] = "stop."  # chunk would end after word 12 (index 11): one before nominal
    words[14] = "later."
    chunks = chunk_words(words, ChunkParams(target=13, overlap=2, snap=3))
    assert chunks[0].end == 12


def test_tie_goes_to_the_earlier_sentence_end() -> None:
    words = [f"w{i}" for i in range(30)]
    words[8] = "a."  # end 9, nominal 10: distance 1
    words[10] = "b."  # end 11: distance 1
    assert chunk_words(words, ChunkParams(target=10, overlap=0, snap=2))[0].end == 9


def test_without_a_sentence_end_the_nominal_length_stands() -> None:
    words = [f"w{i}" for i in range(30)]
    assert chunk_words(words, ChunkParams(target=10, overlap=3, snap=2))[0].end == 10


@pytest.mark.parametrize("closing", ["end.", 'end."', "end.)", "end?”", "end!"])
def test_sentence_ends_with_closing_punctuation(closing: str) -> None:
    words = [f"w{i}" for i in range(30)]
    words[9] = closing
    assert chunk_words(words, ChunkParams(target=11, overlap=0, snap=2))[0].end == 10


def test_page_shorter_than_target_is_one_chunk() -> None:
    chunks = chunk_page("one two three.\nfour", ChunkParams())
    assert [(c.start, c.end, c.text) for c in chunks] == [(0, 4, "one two three. four")]


def test_empty_page_has_no_chunks() -> None:
    assert chunk_page(" \n ", ChunkParams()) == []


def test_smallest_valid_params_are_accepted() -> None:
    assert ChunkParams(10, 9, 0).target == 10


@pytest.mark.parametrize(
    ("target", "overlap", "snap"),
    [(0, 0, 0), (10, -1, 0), (10, 0, -1), (10, 5, 5), (10, 10, 0), (10, 0, 10)],
)
def test_invalid_params_are_rejected(target: int, overlap: int, snap: int) -> None:
    with pytest.raises(ValueError, match="target"):
        ChunkParams(target, overlap, snap)


def test_default_params_match_the_design() -> None:
    assert ChunkParams() == ChunkParams(target=220, overlap=40, snap=30)
