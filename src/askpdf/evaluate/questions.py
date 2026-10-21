"""The question sets (ADR-0011): loading, and the checks every question must pass."""

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
        try:
            return cls._from_json(row)
        except (KeyError, TypeError) as exc:
            raise ValueError(f"{row.get('id', '?')}: missing or malformed field {exc}") from exc

    @classmethod
    def _from_json(cls, row: Mapping[str, object]) -> "Question":
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
        # Measured after collapsing whitespace, so padding cannot lengthen a quote.
        short = [e for e in self.evidence if len(_flat(e)) < MIN_QUOTE_CHARS]
        if short:
            raise ValueError(
                f"{self.id}: quotes shorter than {MIN_QUOTE_CHARS} characters: {short}"
            )
        if self.kind == "multi-page" and len(set(self.pages)) < 2:
            raise ValueError(f"{self.id}: a multi-page question cites at least two pages")
        if len(set(self.pages)) != len(self.pages):
            raise ValueError(f"{self.id}: a page is listed twice")

    @property
    def gold(self) -> frozenset[tuple[str, int]]:
        return frozenset((self.doc, p) for p in self.pages)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Rows of a JSON Lines file; a bad line is a ValueError naming the file and line."""
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path.name}:{number}: not valid JSON ({exc.msg})") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{path.name}:{number}: expected a JSON object")
        rows.append(row)
    if not rows:
        raise ValueError(f"{path.name}: no entries")
    return rows


def load(path: Path) -> list[Question]:
    questions = [Question.from_json(row) for row in read_jsonl(path)]
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
        # Header and footer lines repeat on every page, so they prove nothing.
        record = records[p - 1]
        lines = record.text.split("\n")
        texts[p] = _flat(" ".join(ln for i, ln in enumerate(lines) if i not in record.boilerplate))
    for quote in question.evidence:
        if not _flat(quote):  # "" is in every text
            problems.append(f"{question.id}: an empty quote proves nothing")
        elif not any(_flat(quote) in text for text in texts.values()):
            problems.append(f"{question.id}: quote not on its pages: {quote!r}")
    for p, text in texts.items():
        if not any(_flat(quote) in text for quote in question.evidence):
            problems.append(f"{question.id}: page {p} holds none of the quotes")
    return problems
