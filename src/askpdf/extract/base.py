"""The page model every extractor produces (ADR-0002)."""

import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Protocol

# A page with an image and fewer non-whitespace characters than this is treated as a scan.
IMAGE_ONLY_MAX_CHARS = 20


@dataclass(frozen=True, slots=True)
class RawPage:
    """One page as the PDF library returned it, before cleaning.

    ``index`` is the physical 1-based page number: the canonical one, and the one that
    ``#page=N`` links use. ``label`` is the printed page label, which may differ.
    """

    index: int
    label: str
    text: str
    has_image: bool

    @property
    def image_only(self) -> bool:
        return self.has_image and _visible_chars(self.text) < IMAGE_ONLY_MAX_CHARS


def _visible_chars(text: str) -> int:
    """Characters that would show on the page: not whitespace, control or zero-width."""
    return sum(
        1 for ch in text if not ch.isspace() and unicodedata.category(ch) not in {"Cc", "Cf", "Cs"}
    )


class ExtractionError(Exception):
    """The file cannot be read: encrypted with a password, corrupt, or not a PDF."""


class Extractor(Protocol):
    name: str
    version: str

    def extract(self, source: Path | BinaryIO) -> list[RawPage]:
        """Return every page of a file or an in-memory PDF, or raise ExtractionError."""
        ...
