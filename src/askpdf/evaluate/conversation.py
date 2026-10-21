"""Stage (f): follow-up questions (design §11, ADR-0010).

A follow-up such as "What about at AAL3?" means little on its own. Each follow-up is
searched three ways, all model-free:

- ``raw``: the follow-up exactly as typed
- ``concat``: the conversation's questions so far, joined
- ``gold``: the hand-written standalone question

The model rewrite (PR 8) is scored as a fourth condition, and reported as the share of
the raw-to-gold gap it closes.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from askpdf.evaluate.metrics import mean_scores, retrieval_scores
from askpdf.evaluate.questions import Question, read_jsonl
from askpdf.evaluate.retrieval import Retriever

CONDITIONS = ("raw", "concat", "gold")


@dataclass(frozen=True, slots=True)
class Turn:
    question: str
    rewrite: str  # the gold standalone question; equals ``question`` on the first turn
    doc: str
    pages: tuple[int, ...]
    evidence: tuple[str, ...]
    switch: bool  # a change of topic

    def as_question(self, qid: str) -> Question:
        """The turn as a question, so it passes the same evidence checks."""
        q = Question(
            qid, self.rewrite, "en", self.doc, self.pages, self.evidence, "-", "lexical", True
        )
        q.validate()
        return q


@dataclass(frozen=True, slots=True)
class Conversation:
    id: str
    lang: str
    turns: tuple[Turn, ...]

    @classmethod
    def from_json(cls, row: Mapping[str, Any]) -> "Conversation":
        turns = tuple(
            Turn(
                str(t["question"]),
                str(t["rewrite"]),
                str(t["doc"]),
                tuple(int(p) for p in t["pages"]),
                tuple(str(e) for e in t["evidence"]),
                bool(t["switch"]),
            )
            for t in row["turns"]
        )
        conversation = cls(str(row["id"]), str(row["lang"]), turns)
        if len(turns) < 2:
            raise ValueError(f"{conversation.id}: a conversation needs a follow-up")
        if turns[0].rewrite != turns[0].question or turns[0].switch:
            raise ValueError(f"{conversation.id}: the first turn must already be standalone")
        for i, turn in enumerate(turns):
            turn.as_question(f"{conversation.id}#{i + 1}")
        return conversation

    def queries(self, i: int) -> dict[str, str]:
        """The three ways of searching for follow-up ``i`` (0-based, i >= 1)."""
        return {
            "raw": self.turns[i].question,
            "concat": " ".join(t.question for t in self.turns[: i + 1]),
            "gold": self.turns[i].rewrite,
        }


def load(path: Path) -> list[Conversation]:
    conversations = []
    for row in read_jsonl(path):
        try:
            conversations.append(Conversation.from_json(row))
        except (KeyError, TypeError) as exc:
            raise ValueError(f"{row.get('id', '?')}: missing or malformed field {exc}") from exc
    return conversations


def evaluate(conversations: Sequence[Conversation], retrieve: Retriever) -> dict[str, Any]:
    scores: dict[str, dict[str, dict[str, float]]] = {c: {} for c in CONDITIONS}
    switches: set[str] = set()
    for conv in conversations:
        for i in range(1, len(conv.turns)):
            turn = conv.turns[i]
            key = f"{conv.id}#{i + 1}"
            gold = frozenset((turn.doc, p) for p in turn.pages)
            if turn.switch:
                switches.add(key)
            for condition, query in conv.queries(i).items():
                per_metric = retrieval_scores(retrieve(query), gold)
                scores[condition][key] = {m: round(v, 6) for m, v in per_metric.items()}

    def means(keys: set[str] | None) -> dict[str, dict[str, float]]:
        out = {}
        for condition in CONDITIONS:
            chosen = {k: v for k, v in scores[condition].items() if keys is None or k in keys}
            out[condition] = {m: round(v, 6) for m, v in mean_scores(chosen).items()}
        return out

    return {
        "n_follow_ups": len(scores["raw"]),
        "n_topic_switches": len(switches),
        "overall": means(None),
        "topic_switches": means(switches),
        "per_turn": {c: dict(sorted(scores[c].items())) for c in CONDITIONS},
    }
