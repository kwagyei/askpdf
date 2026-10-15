"""Text extraction with pypdf, the provisional default (ADR-0002).

PR 4 compares it with pdfplumber and pypdfium2 on hand-transcribed pages; until then
this is the only extractor the app uses.
"""

import logging
import re
from pathlib import Path
from typing import Any, BinaryIO

import pypdf
from pypdf.errors import DependencyError, PdfReadError
from pypdf.generic import DictionaryObject, IndirectObject

from askpdf.extract.base import ExtractionError, RawPage

# pypdf logs a warning for every malformed object it recovers from; a messy PDF
# would flood the terminal with them.
logging.getLogger("pypdf").setLevel(logging.ERROR)


def _resolve(obj: Any) -> Any:
    return obj.get_object() if isinstance(obj, IndirectObject) else obj


def _has_image(resources: Any, depth: int = 0) -> bool:
    """True if the page's resources hold an image, looking one level into forms.

    Reads the resource dictionary only; decoding the images themselves would be slow
    and can fail on exotic filters.
    """
    resources = _resolve(resources)
    if not isinstance(resources, DictionaryObject):
        return False
    xobjects = _resolve(resources.get("/XObject"))
    if not isinstance(xobjects, DictionaryObject):
        return False
    for ref in xobjects.values():
        xobject = _resolve(ref)
        if not isinstance(xobject, DictionaryObject):
            continue
        subtype = xobject.get("/Subtype")
        if subtype == "/Image":
            return True
        if subtype == "/Form" and depth < 1 and _has_image(xobject.get("/Resources"), depth + 1):
            return True
    return False


# An inline image (BI ... ID ... EI) painted straight into the content stream, as some
# scanners and fax tools write them.
_BEGIN_IMAGE = re.compile(rb"(?:^|\s)BI\s")
_IMAGE_DATA = re.compile(rb"\sID\s")


def contains_inline_image(data: bytes) -> bool:
    """True if a ``BI`` operator is followed somewhere by ``ID``.

    Two linear searches instead of one ``BI.*?ID`` pattern: that pattern rescans to the
    end of the stream from every ``BI`` without an ``ID``, which is quadratic, and a
    crafted page full of " BI " would hold the indexer (and the GIL) for minutes. If no
    ``ID`` follows the first ``BI``, none follows any later one either.
    """
    begin = _BEGIN_IMAGE.search(data)
    return begin is not None and _IMAGE_DATA.search(data, begin.end() - 1) is not None


def _has_inline_image(page: Any) -> bool:
    try:
        contents = page.get_contents()
        return contents is not None and contains_inline_image(contents.get_data())
    except Exception:
        return False  # an unreadable content stream also fails text extraction


class PypdfExtractor:
    name = "pypdf"
    version = pypdf.__version__

    def extract(self, source: Path | BinaryIO) -> list[RawPage]:
        try:
            reader = pypdf.PdfReader(source)
            if reader.is_encrypted and not reader.decrypt(""):
                raise ExtractionError("encrypted with a password")
            labels = reader.page_labels
            pages = []
            for i, page in enumerate(reader.pages):
                pages.append(
                    RawPage(
                        index=i + 1,
                        label=labels[i] if i < len(labels) else str(i + 1),
                        text=page.extract_text() or "",
                        has_image=_has_image(page.get("/Resources")) or _has_inline_image(page),
                    )
                )
        except ExtractionError:
            raise
        except DependencyError as exc:
            # AES-encrypted files need the optional `cryptography` package.
            raise ExtractionError(f"encrypted in a way pypdf cannot open here ({exc})") from exc
        except PdfReadError as exc:
            raise ExtractionError(f"cannot read PDF: {exc}") from exc
        except OSError as exc:
            raise ExtractionError(f"cannot open file: {exc}") from exc
        except Exception as exc:
            # On malformed input pypdf can raise almost anything (KeyError, IndexError,
            # RecursionError...). One bad file must fail alone, not stop the indexer.
            raise ExtractionError(f"cannot read PDF: {type(exc).__name__}: {exc}") from exc
        return pages
