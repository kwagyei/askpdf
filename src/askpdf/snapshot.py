"""An immutable in-memory view of the index for queries (ADR-0003).

The indexer keeps committing documents while questions are answered. A query works on
a ``Snapshot`` read in one read transaction, so it sees the index as of one generation:
never half a document, never chunks of a document whose file is gone. When the
generation changes, a new snapshot is built and swapped in by reference; the old one
stays valid for queries still using it.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

import numpy as np
import numpy.typing as npt

from askpdf.bm25 import BM25, tokenize
from askpdf.store import Store


@dataclass(frozen=True, slots=True)
class Chunk:
    id: int
    sha256: str
    page: int
    ordinal: int
    text: str


@dataclass(frozen=True, slots=True)
class Snapshot:
    generation: int
    chunks: tuple[Chunk, ...]  # in id order; position i is document i of ``bm25``
    labels: Mapping[tuple[str, int], str]  # (sha256, page) -> printed page label
    paths: Mapping[str, tuple[str, ...]]  # sha256 -> the files holding that document
    bm25: BM25
    # One L2-normalised float32 row per chunk, in the same order; rows of chunks not
    # embedded yet are zero and marked False in ``embedded``.
    vectors: npt.NDArray[np.float32] = field(default_factory=lambda: np.zeros((0, 0), np.float32))
    embedded: npt.NDArray[np.bool_] = field(default_factory=lambda: np.zeros(0, np.bool_))

    @classmethod
    def load(cls, store: Store) -> "Snapshot":
        with store.read_transaction():
            generation = store.generation
            rows = store.all_chunks()
            labels = store.page_labels()
            paths = store.indexed_paths()
            blobs = store.all_vectors()
        chunks = tuple(Chunk(*row) for row in rows)
        vectors, embedded = _matrix(chunks, blobs)
        return cls(
            generation=generation,
            chunks=chunks,
            labels=MappingProxyType(labels),
            paths=MappingProxyType(paths),
            bm25=BM25.build(tokenize(c.text) for c in chunks),
            vectors=vectors,
            embedded=embedded,
        )

    @property
    def fully_embedded(self) -> bool:
        return bool(self.embedded.size) and bool(self.embedded.all())

    def is_current(self, store: Store) -> bool:
        return store.generation == self.generation


def _matrix(
    chunks: tuple[Chunk, ...], blobs: Mapping[int, bytes]
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.bool_]]:
    dims = {len(b) // 4 for b in blobs.values()}
    if not blobs or len(dims) != 1:  # none yet, or a mix that cannot be compared
        return np.zeros((len(chunks), 0), np.float32), np.zeros(len(chunks), np.bool_)
    (dim,) = dims
    matrix = np.zeros((len(chunks), dim), np.float32)
    embedded = np.zeros(len(chunks), np.bool_)
    for i, chunk in enumerate(chunks):
        blob = blobs.get(chunk.id)
        if blob is not None:
            matrix[i] = np.frombuffer(blob, dtype="<f4")
            embedded[i] = True
    matrix.flags.writeable = False
    embedded.flags.writeable = False
    return matrix, embedded
