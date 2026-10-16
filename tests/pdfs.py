"""Build small, real PDFs for tests, so no binary fixtures live in the repo."""

from dataclasses import dataclass, field
from pathlib import Path

from pypdf import PdfWriter
from pypdf.generic import (
    ArrayObject,
    DecodedStreamObject,
    DictionaryObject,
    NameObject,
    NumberObject,
)


@dataclass
class PageSpec:
    lines: list[str] = field(default_factory=list)
    image: bool = False


def _escape(line: str) -> bytes:
    raw = line.encode("cp1252")  # WinAnsiEncoding, so å ä ö work
    return raw.replace(b"\\", b"\\\\").replace(b"(", b"\\(").replace(b")", b"\\)")


def _stream(writer: PdfWriter, data: bytes, **entries: object) -> object:
    stream = DecodedStreamObject()
    stream.set_data(data)
    for key, value in entries.items():
        stream[NameObject(f"/{key}")] = value
    return writer._add_object(stream)


def make_pdf(
    path: Path,
    pages: list[PageSpec],
    *,
    labels: tuple[int, int, str, int] | None = None,
    user_password: str | None = None,
) -> Path:
    """Write ``pages`` to ``path``.

    ``labels`` is ``(first page index, last page index, style, start)``, e.g.
    ``(0, 1, "/r", 1)`` for roman numerals on the first two pages.
    """
    writer = PdfWriter()
    font = writer._add_object(
        DictionaryObject(
            {
                NameObject("/Type"): NameObject("/Font"),
                NameObject("/Subtype"): NameObject("/Type1"),
                NameObject("/BaseFont"): NameObject("/Helvetica"),
                NameObject("/Encoding"): NameObject("/WinAnsiEncoding"),
            }
        )
    )
    for spec in pages:
        page = writer.add_blank_page(612, 792)
        resources = DictionaryObject(
            {NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})}
        )
        ops = b""
        if spec.image:
            image = _stream(
                writer,
                b"\x00",
                Type=NameObject("/XObject"),
                Subtype=NameObject("/Image"),
                Width=NumberObject(1),
                Height=NumberObject(1),
                ColorSpace=NameObject("/DeviceGray"),
                BitsPerComponent=NumberObject(8),
            )
            resources[NameObject("/XObject")] = DictionaryObject({NameObject("/Im0"): image})
            ops += b"q 400 0 0 600 100 100 cm /Im0 Do Q\n"
        if spec.lines:
            ops += b"BT /F1 11 Tf 72 740 Td 13 TL\n"
            ops += b"".join(b"(" + _escape(line) + b") Tj T*\n" for line in spec.lines)
            ops += b"ET\n"
        page[NameObject("/Resources")] = resources
        page[NameObject("/Contents")] = _stream(writer, ops)
        page[NameObject("/MediaBox")] = ArrayObject(
            [NumberObject(0), NumberObject(0), NumberObject(612), NumberObject(792)]
        )
    if labels is not None:
        first, last, style, start = labels
        writer.set_page_label(first, last, style=style, start=start)  # type: ignore[arg-type]
    if user_password is not None:
        writer.encrypt(user_password=user_password, owner_password="owner", algorithm="RC4-128")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        writer.write(f)
    return path


def text_page(*lines: str) -> PageSpec:
    return PageSpec(lines=list(lines))
