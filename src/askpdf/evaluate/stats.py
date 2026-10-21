"""Intervals, paired comparisons and agreement (ADR-0007).

Bootstrap intervals use 10,000 resamples and seed 0, and the sign-flip test 10,000
random flips and seed 0, so every reported number is reproducible. numpy only: no
scipy, and the sign-flip test is checked against exact enumeration.
"""

import itertools
from collections import Counter
from collections.abc import Hashable, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

RESAMPLES = 10_000
SEED = 0
# Differences this small are float noise, not a difference (sums of the same numbers in
# another order); they count as ties with the observed statistic.
_TOLERANCE = 1e-12


@dataclass(frozen=True, slots=True)
class Interval:
    estimate: float
    low: float
    high: float

    def excludes_zero(self) -> bool:
        return self.low > 0 or self.high < 0


def bootstrap_ci(
    values: Sequence[float], *, resamples: int = RESAMPLES, seed: int = SEED, level: float = 0.95
) -> Interval:
    """Percentile bootstrap interval for the mean."""
    data = np.asarray(values, dtype=np.float64)
    if data.size == 0:
        raise ValueError("cannot bootstrap an empty sample")
    rng = np.random.default_rng(seed)
    means = data[rng.integers(0, data.size, size=(resamples, data.size))].mean(axis=1)
    tail = (1 - level) / 2 * 100
    low, high = np.percentile(means, [tail, 100 - tail])
    return Interval(float(data.mean()), float(low), float(high))


def _differences(a: Sequence[float], b: Sequence[float]) -> npt.NDArray[np.float64]:
    if len(a) != len(b):
        raise ValueError("paired samples must have the same length")
    return np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)


def paired_bootstrap(
    a: Sequence[float], b: Sequence[float], *, resamples: int = RESAMPLES, seed: int = SEED
) -> Interval:
    """Interval for the mean per-question difference a - b."""
    return bootstrap_ci(_differences(a, b).tolist(), resamples=resamples, seed=seed)


def sign_flip_test(
    a: Sequence[float], b: Sequence[float], *, permutations: int = RESAMPLES, seed: int = SEED
) -> float:
    """Two-sided p-value of a paired randomisation test on the mean difference.

    Under the null hypothesis each per-question difference is as likely to have either
    sign. The observed statistic is included among the permutations, so p is never 0.
    """
    d = _differences(a, b)
    if d.size == 0:
        raise ValueError("cannot test an empty sample")
    observed = abs(d.mean())
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(permutations, d.size))
    extreme = np.count_nonzero(np.abs((signs * d).mean(axis=1)) >= observed - _TOLERANCE)
    return float((extreme + 1) / (permutations + 1))


def sign_flip_exact(a: Sequence[float], b: Sequence[float]) -> float:
    """The same test by enumerating all 2**n sign patterns; the oracle for small n."""
    d = _differences(a, b)
    if not 0 < d.size <= 20:
        raise ValueError("exact enumeration needs 1 to 20 pairs")
    observed = abs(d.mean())
    extreme = sum(
        abs((np.array(signs) * d).mean()) >= observed - _TOLERANCE
        for signs in itertools.product((-1.0, 1.0), repeat=d.size)
    )
    return float(extreme / 2**d.size)


def cohens_kappa(first: Sequence[Hashable], second: Sequence[Hashable]) -> float:
    """Cohen's kappa for two labellers who labelled the same items."""
    if len(first) != len(second) or not first:
        raise ValueError("kappa needs two equally long, non-empty label lists")
    n = len(first)
    observed = sum(x == y for x, y in zip(first, second, strict=True)) / n
    counts_a, counts_b = Counter(first), Counter(second)
    expected = sum(counts_a[label] * counts_b[label] for label in counts_a) / n**2
    if expected == 1:
        raise ValueError("kappa is undefined when both labellers use a single label")
    return (observed - expected) / (1 - expected)
