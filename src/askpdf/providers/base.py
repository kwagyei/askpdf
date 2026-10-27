"""What the rest of askpdf needs from a model provider (ADR-0005)."""

from collections.abc import Iterator, Sequence
from typing import Protocol

import numpy as np
import numpy.typing as npt

type Matrix = npt.NDArray[np.float32]


class Embedder(Protocol):
    @property
    def name(self) -> str:
        """Pinned in the index: a different embedder means vectors that cannot be compared."""
        ...

    def embed_documents(self, texts: Sequence[str]) -> Matrix:
        """One L2-normalised float32 row per text."""
        ...

    def embed_query(self, text: str) -> Matrix:
        """A single L2-normalised float32 vector."""
        ...


class ChatModel(Protocol):
    def stream(self, system: str, user: str) -> Iterator[str]:
        """The answer's text, piece by piece as it is generated."""
        ...


def normalise(rows: npt.ArrayLike) -> Matrix:
    """L2-normalise each row, so a dot product is the cosine similarity."""
    m = np.asarray(rows, dtype=np.float32)
    norms = np.linalg.norm(m, axis=-1, keepdims=True)
    return (m / np.where(norms == 0, 1, norms)).astype(np.float32)
