"""Render PDF pages to PNG for transcribing and checking evidence by eye (ADR-0011).

    python eval/tools/render.py eval/corpus/nist-sp-800-63b-4.pdf 5 46 --out eval/pages

Pages are physical and 1-based, as everywhere in askpdf. Needs the `eval` group.
"""

import argparse
from pathlib import Path

import pypdfium2 as pdfium

DPI = 110  # readable on screen, small enough to commit a dozen pages


def render(pdf: Path, pages: list[int], out: Path, dpi: int = DPI) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    doc = pdfium.PdfDocument(pdf)
    written = []
    for number in pages:
        image = doc[number - 1].render(scale=dpi / 72, grayscale=True).to_pil()
        target = out / f"{pdf.stem}-p{number:03d}.png"
        image.save(target, optimize=True)
        written.append(target)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("pdf", type=Path)
    parser.add_argument("pages", type=int, nargs="+")
    parser.add_argument("--out", type=Path, default=Path("eval/pages"))
    parser.add_argument("--dpi", type=int, default=DPI)
    args = parser.parse_args()
    for path in render(args.pdf, args.pages, args.out, args.dpi):
        print(path)


if __name__ == "__main__":
    main()
