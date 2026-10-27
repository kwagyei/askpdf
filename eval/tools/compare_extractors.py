"""Compare text extractors on the transcribed pages, by ADR-0002's pre-registered rule.

    python eval/tools/compare_extractors.py [--json eval/results/extractor-comparison.json]

Each extractor's raw page text goes through askpdf's own cleaning (clean_document and
header/footer detection), then is scored by character error rate against the hand
transcriptions, with header/footer lines included and excluded. Speed is pages per
second over the whole corpus, best of three runs. Needs the eval group.

Rule (ADR-0002): pick the lowest mean CER; keep pypdf if the leader is within 1 CER
point of it or more than 3x slower.
"""

import argparse
import json
import platform
import time
from collections.abc import Callable
from importlib.metadata import version
from pathlib import Path
from typing import Any

import pdfplumber
import pypdfium2 as pdfium

from askpdf.evaluate.extraction import character_error_rate, load_transcripts
from askpdf.extract.clean import clean_document, detect_boilerplate
from askpdf.extract.pypdf_ext import PypdfExtractor

EVAL = Path(__file__).resolve().parents[1]

type Reader = Callable[[Path], list[str]]


def read_pypdf(pdf: Path) -> list[str]:
    return [p.text for p in PypdfExtractor().extract(pdf)]


def read_pdfplumber(pdf: Path) -> list[str]:
    with pdfplumber.open(pdf) as doc:
        return [page.extract_text() or "" for page in doc.pages]


def read_pypdfium2(pdf: Path) -> list[str]:
    doc = pdfium.PdfDocument(pdf)
    try:
        return [doc[i].get_textpage().get_text_range() for i in range(len(doc))]
    finally:
        doc.close()


READERS: dict[str, tuple[Reader, str]] = {
    "pypdf": (read_pypdf, "pypdf"),
    "pdfplumber": (read_pdfplumber, "pdfplumber"),
    "pypdfium2": (read_pypdfium2, "pypdfium2"),
}


def cleaned(reader: Reader, pdf: Path) -> tuple[list[str], list[set[int]]]:
    pages = clean_document(reader(pdf))
    return pages, detect_boilerplate([p.split("\n") if p else [] for p in pages])


def without(text: str, lines: set[int]) -> str:
    return "\n".join(line for i, line in enumerate(text.split("\n")) if i not in lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    transcripts = load_transcripts(EVAL / "transcripts")
    corpus = sorted((EVAL / "corpus").glob("*.pdf"))
    total_pages = sum(len(read_pypdf(pdf)) for pdf in corpus)
    result: dict[str, Any] = {
        "machine": f"{platform.system()} {platform.machine()}, Python {platform.python_version()}",
        "pages_transcribed": len(transcripts),
        "extractors": {},
    }
    for name, (reader, package) in READERS.items():
        docs = {t.doc: cleaned(reader, EVAL / "corpus" / f"{t.doc}.pdf") for t in transcripts}
        per_page = []
        for t in transcripts:
            pages, flags = docs[t.doc]
            text, labelled = pages[t.page - 1], set(t.boilerplate)
            reference_body = "\n".join(line for line in t.text.split("\n") if line not in labelled)
            per_page.append(
                {
                    "doc": t.doc,
                    "page": t.page,
                    "cer_with_boilerplate": round(character_error_rate(t.text, text), 6),
                    "cer_without_boilerplate": round(
                        character_error_rate(reference_body, without(text, flags[t.page - 1])), 6
                    ),
                }
            )
        best = float("inf")
        for _ in range(3):
            start = time.perf_counter()
            for pdf in corpus:
                reader(pdf)
            best = min(best, time.perf_counter() - start)
        n = len(per_page)
        result["extractors"][name] = {
            "version": version(package),
            "mean_cer_with_boilerplate": round(
                sum(p["cer_with_boilerplate"] for p in per_page) / n, 6
            ),
            "mean_cer_without_boilerplate": round(
                sum(p["cer_without_boilerplate"] for p in per_page) / n, 6
            ),
            "pages_per_second": round(total_pages / best, 1),
            "pages": per_page,
        }
    for name, r in result["extractors"].items():
        print(
            f"{name:10} {r['version']:10} CER {r['mean_cer_with_boilerplate']:.3f}"
            f" (without boilerplate {r['mean_cer_without_boilerplate']:.3f})"
            f"  {r['pages_per_second']:.0f} pages/s"
        )
    if args.json:
        args.json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
