"""The page model every extractor produces (ADR-0002)."""

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

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
        visible = sum(1 for ch in self.text if not ch.isspace())
        return self.has_image and visible < IMAGE_ONLY_MAX_CHARS


class ExtractionError(Exception):
    """The file cannot be read: encrypted with a password, corrupt, or not a PDF."""


class Extractor(Protocol):
    name: str
    version: str

    def extract(self, path: Path) -> list[RawPage]:
        """Return every page in order, or raise ExtractionError."""
        ...
