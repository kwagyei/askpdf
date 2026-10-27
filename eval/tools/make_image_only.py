"""Build eval/corpus/nist-image-only.pdf: two NIST pages as pictures, no text layer.

It is the ground truth for scanned-page detection (ADR-0011, corpus file 5): unlike a
real scan, it is certain to have no OCR layer.

    python eval/tools/make_image_only.py
"""

import time
from pathlib import Path

import pypdfium2 as pdfium

SOURCE = Path("eval/corpus/nist-sp-800-63b-4.pdf")
TARGET = Path("eval/corpus/nist-image-only.pdf")
PAGES = (40, 46)  # physical pages of the source
DPI = 100


def main() -> None:
    doc = pdfium.PdfDocument(SOURCE)
    images = [doc[n - 1].render(scale=DPI / 72, grayscale=True).to_pil() for n in PAGES]
    # A fixed date (2026-10-21, as Pillow wants it) makes the file byte-for-byte reproducible.
    # Pillow also writes the output file name as the PDF /Title, so it is identical only
    # under this name.
    fixed = time.gmtime(1792540800)
    images[0].save(
        TARGET,
        save_all=True,
        append_images=images[1:],
        resolution=DPI,
        creationDate=fixed,
        modDate=fixed,
    )
    print(TARGET, TARGET.stat().st_size, "bytes")


if __name__ == "__main__":
    main()
