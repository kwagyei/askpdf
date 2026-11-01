"""Re-check saved answers with the current validation code, without running the model.

The answer stage takes over an hour on the target machine, but its records keep each
model reply (``raw``). When only the checking of replies changes (quote parsing, the
matching key), this recomputes every number from those replies. Retrieval is BM25 and
deterministic, so each reply is checked against the same pages it was written from.

    python eval/tools/rescore_answers.py eval/results/dev-answers.json
"""

import json
import sys
from pathlib import Path

from askpdf.evaluate import answers, questions, retrieval
from askpdf.retrieve import search_bm25

ROOT = Path(__file__).resolve().parents[2]


def main(path: Path) -> None:
    saved = json.loads(path.read_text(encoding="utf-8"))
    snapshot = retrieval.load_corpus_snapshot(ROOT / "eval" / "corpus")
    qs = questions.load(ROOT / "eval" / "questions" / f"{saved['split']}.jsonl")
    records = answers.rescore(
        saved["per_question"], qs, snapshot, lambda q, k: search_bm25(snapshot, q, k), saved["k"]
    )
    saved.update(answers.summarise(records))
    saved["per_question"] = dict(sorted(records.items()))
    path.write_text(json.dumps(saved, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        json.dumps({"grounding": saved["grounding"], "abstention": saved["abstention"]}, indent=2)
    )


if __name__ == "__main__":
    main(Path(sys.argv[1]))
