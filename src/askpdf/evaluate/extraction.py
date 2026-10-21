"""Stage (a): how well pages are read (design §11, ADR-0007).

- Character error rate of askpdf's cleaned page text against hand transcriptions.
- Header/footer detection, scored on words: the words of flagged lines against the
  words of the lines the transcriber labelled ``[header]`` or ``[footer]``. Words, not
  lines, because an extractor may merge a two-sided header into one line.
- Scanned-page detection against pages known to have no text layer.

CER needs jiwer from the ``eval`` group; nothing else here does.
"""

import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from askpdf.evaluate.metrics import Detection
from askpdf.store import PageRecord

_LABEL = re.compile(r"^\[(header|footer|margin)\] ")
_NAME = re.compile(r"^(?P<doc>.+)-p(?P<page>\d{3})$")


@dataclass(frozen=True, slots=True)
class Transcript:
    doc: str  # corpus file name without .pdf
    page: int  # physical, 1-based
    text: str  # every line, labels removed
    boilerplate: tuple[str, ...]  # the [header] and [footer] lines

    @classmethod
    def read(cls, path: Path) -> "Transcript":
        match = _NAME.match(path.stem)
        if match is None:
            raise ValueError(f"{path.name}: expected <document>-pNNN.txt")
        lines = path.read_text(encoding="utf-8").splitlines()
        boilerplate = tuple(
            _LABEL.sub("", line) for line in lines if line.startswith(("[header] ", "[footer] "))
        )
        text = "\n".join(_LABEL.sub("", line) for line in lines)
        return cls(match["doc"], int(match["page"]), text, boilerplate)


def load_transcripts(folder: Path) -> list[Transcript]:
    return [Transcript.read(p) for p in sorted(folder.glob("*.txt"))]


def normalise(text: str) -> str:
    """Single spaces: layout whitespace is not an extraction error."""
    return " ".join(text.split())


def character_error_rate(reference: str, hypothesis: str) -> float:
    import jiwer  # the eval group; imported here so the app never needs it

    reference, hypothesis = normalise(reference), normalise(hypothesis)
    if not reference:
        return 0.0 if not hypothesis else 1.0
    return float(jiwer.cer(reference, hypothesis))


def boilerplate_words(page: PageRecord) -> Counter[str]:
    lines = page.text.split("\n")
    return Counter(w for i in page.boilerplate for w in lines[i].split())


def word_detection(predicted: Counter[str], gold: Counter[str]) -> Detection:
    """Detection counts on words, as multisets (a header word on two lines counts twice)."""
    hits = sum((predicted & gold).values())
    return Detection(hits, sum(predicted.values()) - hits, sum(gold.values()) - hits)


@dataclass(frozen=True, slots=True)
class PageResult:
    doc: str
    page: int
    cer: float
    boilerplate: Detection


def evaluate_pages(
    transcripts: Sequence[Transcript], pages: Mapping[str, Sequence[PageRecord]]
) -> list[PageResult]:
    """Score each transcribed page against the extracted pages of its document."""
    results = []
    for t in transcripts:
        record = pages[t.doc][t.page - 1]
        gold = Counter(w for line in t.boilerplate for w in line.split())
        results.append(
            PageResult(
                t.doc,
                t.page,
                character_error_rate(t.text, record.text),
                word_detection(boilerplate_words(record), gold),
            )
        )
    return results


def scanned_detection(
    pages: Mapping[str, Sequence[PageRecord]], gold: set[tuple[str, int]]
) -> Detection:
    predicted = {(doc, p.page) for doc, records in pages.items() for p in records if p.image_only}
    return Detection.of(predicted, gold)


def summarise(results: Sequence[PageResult], scanned: Detection) -> dict[str, Any]:
    """The numbers the gate pins and the README reports, rounded so they compare exactly."""
    total = Detection(
        sum(r.boilerplate.true_positives for r in results),
        sum(r.boilerplate.false_positives for r in results),
        sum(r.boilerplate.false_negatives for r in results),
    )

    def r6(x: float | None) -> float | None:
        return None if x is None else round(x, 6)

    return {
        "pages": [
            {"doc": r.doc, "page": r.page, "cer": r6(r.cer)}
            for r in sorted(results, key=lambda r: (r.doc, r.page))
        ],
        "mean_cer": r6(sum(r.cer for r in results) / len(results)) if results else None,
        "boilerplate": {"precision": r6(total.precision), "recall": r6(total.recall)},
        "scanned": {"precision": r6(scanned.precision), "recall": r6(scanned.recall)},
    }
