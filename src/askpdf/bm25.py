"""Okapi BM25 over chunks (ADR-0004), with its own tokenizer.

Own implementation rather than a library: the formula fits on one screen, and every
score can be checked by hand (the tests do). k1 = 1.2, b = 0.75 and Lucene's IDF,
which is never negative, so a term in every chunk still counts a little.
"""

import math
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from functools import cache
from typing import Any

import numpy as np
import numpy.typing as npt

K1 = 1.2
B = 0.75

# Lucene's default English stop set (EnglishAnalyzer). English only, as ADR-0004 says;
# per-language metrics in the eval show what that costs Swedish questions.
# fmt: off
STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "if", "in", "into", "is",
    "it", "no", "not", "of", "on", "or", "such", "that", "the", "their", "then", "there",
    "these", "they", "this", "to", "was", "will", "with",
})
# fmt: on

# Maximal runs of letters and digits, with their combining marks: without the marks,
# Devanagari "हिंदी" or Thai would fall apart at every vowel sign, and "İstanbul" (which
# casefolds to i + U+0307) would split after the i. Hyphens, apostrophes and all other
# punctuation separate tokens, so "multi-factor", the same with U+2010 HYPHEN, and
# "multi factor" index alike (ADR-0002 Amendment 1), and "SP 800-63B" gives sp, 800, 63b.


def _ranges(categories: set[str]) -> str:
    """A regex character class body for every code point in the given categories."""
    parts: list[str] = []
    start = previous = -2
    for cp in range(sys.maxunicode + 1):
        if unicodedata.category(chr(cp)) not in categories:
            continue
        if cp != previous + 1:
            if start >= 0:
                parts.append(f"{re.escape(chr(start))}-{re.escape(chr(previous))}")
            start = cp
        previous = cp
    parts.append(f"{re.escape(chr(start))}-{re.escape(chr(previous))}")
    return "".join(parts)


@cache
def _patterns() -> tuple[re.Pattern[str], re.Pattern[str]]:
    # Built on first use, not at import: scanning Unicode takes a few tenths of a second.
    token = re.compile(f"(?:[^\\W_]|[{_ranges({'Mn', 'Mc', 'Me'})}])+")
    # Soft hyphens, zero-width joiners and other format characters: cleaning drops them
    # from page text, so a question must lose them too or it would not match.
    invisible = re.compile(f"[{_ranges({'Cf'})}]")
    return token, invisible


def tokenize(text: str) -> list[str]:
    """Casefolded word tokens without stopwords, for both chunks and questions."""
    token, invisible = _patterns()
    text = unicodedata.normalize("NFKC", invisible.sub("", text)).casefold()
    return [t for t in token.findall(text) if t not in STOPWORDS]


FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


def _frozen[A: np.ndarray[Any, Any]](array: A) -> A:
    array.flags.writeable = False
    return array


@dataclass(frozen=True, slots=True)
class BM25:
    """An inverted index over tokenised documents (here: chunks), numbered 0..n-1."""

    postings: dict[str, tuple[IntArray, FloatArray]]  # term -> (documents, term frequencies)
    lengths: FloatArray  # tokens per document, stopwords excluded
    avg_length: float

    @classmethod
    def build(cls, documents: Iterable[Sequence[str]]) -> "BM25":
        docs: dict[str, list[int]] = defaultdict(list)
        freqs: dict[str, list[int]] = defaultdict(list)
        lengths: list[int] = []
        for doc_id, tokens in enumerate(documents):
            lengths.append(len(tokens))
            for term, tf in Counter(tokens).items():
                docs[term].append(doc_id)
                freqs[term].append(tf)
        postings = {
            term: (
                _frozen(np.array(docs[term], dtype=np.int64)),
                _frozen(np.array(freqs[term], dtype=np.float64)),
            )
            for term in docs
        }
        length_array = _frozen(np.array(lengths, dtype=np.float64))
        avg = float(length_array.mean()) if lengths else 0.0
        return cls(postings, length_array, avg)

    @property
    def n_documents(self) -> int:
        return len(self.lengths)

    def idf(self, term: str) -> float:
        """Lucene's IDF: ln(1 + (N - n + 0.5) / (n + 0.5)); 0 for an unknown term."""
        posting = self.postings.get(term)
        if posting is None:
            return 0.0
        n = len(posting[0])
        return math.log(1 + (self.n_documents - n + 0.5) / (n + 0.5))

    def scores(self, query: Sequence[str]) -> FloatArray:
        """The BM25 score of every document for a tokenised query.

        Each distinct query term counts once, however often the question repeats it.
        """
        scores = np.zeros(self.n_documents, dtype=np.float64)
        # All documents empty means there are no postings, so nothing below runs.
        avg = self.avg_length or 1.0
        for term in dict.fromkeys(query):  # distinct, in order: deterministic sums
            posting = self.postings.get(term)
            if posting is None:
                continue
            ids, tf = posting
            norm = K1 * (1 - B + B * self.lengths[ids] / avg)
            scores[ids] += self.idf(term) * tf * (K1 + 1) / (tf + norm)
        return scores
