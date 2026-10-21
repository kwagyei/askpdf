"""The question sets (ADR-0011): loading, and the checks every question must pass."""

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from askpdf.store import PageRecord

KINDS = frozenset({"paraphrase", "lexical", "table", "multi-page", "scanned"})
# A shorter quote proves little: "1.6" is also found inside "81.6%".
MIN_QUOTE_CHARS = 15


@dataclass(frozen=True, slots=True)
class Question:
    id: str
    question: str
    lang: str
    doc: str
    pages: tuple[int, ...]  # physical, 1-based; empty when unanswerable
    evidence: tuple[str, ...]  # verbatim quotes from those pages
    answer: str | None
    kind: str
    answerable: bool

    @classmethod
    def from_json(cls, row: Mapping[str, object]) -> "Question":
        q = cls(
            id=str(row["id"]),
            question=str(row["question"]),
            lang=str(row["lang"]),
            doc=str(row["doc"]),
            pages=tuple(int(p) for p in row["pages"]),  # type: ignore[attr-defined]
            evidence=tuple(str(e) for e in row["evidence"]),  # type: ignore[attr-defined]
            answer=None if row["answer"] is None else str(row["answer"]),
            kind=str(row["kind"]),
            answerable=bool(row["answerable"]),
        )
        q.validate()
        return q

    def validate(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"{self.id}: unknown kind {self.kind!r}")
        if self.answerable != bool(self.pages) or self.answerable != bool(self.evidence):
            raise ValueError(
                f"{self.id}: answerable questions need pages and evidence; others none"
            )
        if self.answerable and not self.answer:
            raise ValueError(f"{self.id}: an answerable question needs a reference answer")
        short = [e for e in self.evidence if len(e) < MIN_QUOTE_CHARS]
        if short:
            raise ValueError(
                f"{self.id}: quotes shorter than {MIN_QUOTE_CHARS} characters: {short}"
            )
        if self.kind == "multi-page" and len(self.pages) < 2:
            raise ValueError(f"{self.id}: a multi-page question cites at least two pages")

    @property
    def gold(self) -> frozenset[tuple[str, int]]:
        return frozenset((self.doc, p) for p in self.pages)


def load(path: Path) -> list[Question]:
    lines = path.read_text(encoding="utf-8").splitlines()
    questions = [Question.from_json(json.loads(line)) for line in lines if line.strip()]
    ids = [q.id for q in questions]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{path.name}: duplicate question ids")
    return questions


def _flat(text: str) -> str:
    return " ".join(text.split())


def evidence_problems(question: Question, pages: Mapping[str, Sequence[PageRecord]]) -> list[str]:
    """Why a question's evidence does not hold on its pages; empty when it does.

    Each quote must appear (whitespace aside) on at least one cited page, and each cited
    page must hold at least one quote, so a page is never gold without a reason.
    """
    records = pages.get(question.doc)
    if records is None:
        return [f"{question.id}: no document {question.doc!r} in the corpus"]
    problems = []
    texts = {}
    for p in question.pages:
        if not 1 <= p <= len(records):
            problems.append(f"{question.id}: page {p} is out of range")
            continue
        texts[p] = _flat(records[p - 1].text)
    for quote in question.evidence:
        if not any(_flat(quote) in text for text in texts.values()):
            problems.append(f"{question.id}: quote not on its pages: {quote!r}")
    for p, text in texts.items():
        if not any(_flat(quote) in text for quote in question.evidence):
            problems.append(f"{question.id}: page {p} holds none of the quotes")
    return problems
