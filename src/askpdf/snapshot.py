"""An immutable in-memory view of the index for queries (ADR-0003).

The indexer keeps committing documents while questions are answered. A query works on
a ``Snapshot`` read in one read transaction, so it sees the index as of one generation:
never half a document, never chunks of a document whose file is gone. When the
generation changes, a new snapshot is built and swapped in by reference; the old one
stays valid for queries still using it.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

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

    @classmethod
    def load(cls, store: Store) -> "Snapshot":
        with store.read_transaction():
            generation = store.generation
            rows = store.all_chunks()
            labels = store.page_labels()
            paths = store.indexed_paths()
        chunks = tuple(Chunk(*row) for row in rows)
        return cls(
            generation=generation,
            chunks=chunks,
            labels=MappingProxyType(labels),
            paths=MappingProxyType(paths),
            bm25=BM25.build(tokenize(c.text) for c in chunks),
        )

    def is_current(self, store: Store) -> bool:
        return store.generation == self.generation
