"""Render the README's result tables from eval/results/*.json (ADR-0007, README sync).

The README holds each table between ``<!-- results:NAME -->`` and ``<!-- /results -->``.
`tests/test_readme_results.py` fails if a block differs from what this renders, so a
number in the README cannot drift from the committed results.

    python eval/tools/readme_tables.py          # rewrite the blocks in README.md
"""

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "eval" / "results"
BLOCK = re.compile(r"(<!-- results:(?P<name>[a-z-]+) -->\n)(?P<body>.*?)(<!-- /results -->)", re.S)


def _f(x: float) -> str:
    return f"{x:.3f}"


def _load(name: str) -> dict[str, Any]:
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def extraction() -> str:
    ex = _load("extraction.json")
    scanned_pages = sum(
        len(pages)
        for doc, pages in json.loads(
            (ROOT / "eval" / "corpus" / "scanned-pages.json").read_text(encoding="utf-8")
        ).items()
        if not doc.startswith("_")
    )
    cers = [p["cer"] for p in ex["pages"]]
    return (
        f"Extraction with {ex['extractor']}, on {len(ex['pages'])} hand-transcribed pages from a"
        " security\nstandard, a two-column paper and a 1957 scanned report:\n\n"
        "| Measure | Result |\n| --- | --- |\n"
        f"| Mean character error rate | {_f(ex['mean_cer'])} (best page {_f(min(cers))},"
        f" worst {_f(max(cers))}) |\n"
        f"| Header/footer detection, precision / recall | {_f(ex['boilerplate']['precision'])}"
        f" / {_f(ex['boilerplate']['recall'])} |\n"
        f"| Scanned-page detection, precision / recall | {_f(ex['scanned']['precision'])}"
        f" / {_f(ex['scanned']['recall'])} ({scanned_pages} pages without text) |\n"
    )


def retrieval() -> str:
    rt = _load("dev-retrieval-bm25.json")
    n: Counter[str] = Counter()
    for line in (
        (ROOT / "eval" / "questions" / "dev.jsonl").read_text(encoding="utf-8").splitlines()
    ):
        q = json.loads(line)
        if q["answerable"]:
            n[q["kind"]] += 1
            n["all"] += 1
    rows = [("all", rt["overall"]), *rt["by_kind"].items()]
    table = "\n".join(
        f"| {name} | {n[name]} | {_f(s['hit@1'])} | {_f(s['hit@5'])} | {_f(s['mrr@10'])}"
        f" | {_f(s['ndcg@10'])} |"
        for name, s in rows
    )
    i = rt["interval_95"]
    return (
        f"Keyword retrieval (BM25) over the whole corpus, {rt['n_questions']} answerable dev"
        " questions:\n\n"
        "| Questions | n | hit@1 | hit@5 | MRR@10 | nDCG@10 |\n"
        "| --- | --- | --- | --- | --- | --- |\n"
        f"{table}\n\n"
        f"95% bootstrap intervals for the overall row: hit@1 {i['hit@1'][0]:.2f} to"
        f" {i['hit@1'][1]:.2f}, nDCG@10 {i['ndcg@10'][0]:.2f} to {i['ndcg@10'][1]:.2f}.\n"
    )


def conversation() -> str:
    cv = _load("dev-conversation-bm25.json")
    labels = {"raw": "as typed", "concat": "history joined", "gold": "hand-written rewrite"}
    rows = "\n".join(
        f"| {labels[c]} | {_f(s['hit@1'])} ({iv['hit@1'][0]:.2f} to {iv['hit@1'][1]:.2f})"
        f" | {_f(s['ndcg@10'])} |"
        for c, s in cv["overall"].items()
        for iv in [cv["interval_95"][c]]
    )
    return (
        f'Follow-up questions ({cv["n_follow_ups"]} on dev, such as "What about at AAL3?"),'
        " searched three ways:\n\n"
        "| Follow-up searched | hit@1 (95% interval) | nDCG@10 |\n| --- | --- | --- |\n"
        f"{rows}\n"
    )


def answers() -> str:
    an = _load("dev-answers.json")
    g, a = an["grounding"], an["abstention"]
    gi = g["grounded_interval_95"]
    return (
        f"Answers from {an['model']} (Ollama {an['ollama_version']}) on the"
        f" {an['n_questions']} dev questions, {an['k']} BM25 pages each:\n\n"
        "| Measure | Result |\n| --- | --- |\n"
        f"| Citations whose quote is on the cited page |"
        f" {round(g['quote_verified_rate'] * g['n_citations'])} of {g['n_citations']}"
        f" ({_f(g['quote_verified_rate'])}) |\n"
        f"| of those, verified only by folding hyphens | {_f(g['folding_only_share'])} |\n"
        f"| Answer sentences carrying a verified citation | {_f(g['supported_sentence_rate'])} |\n"
        f"| Answers with at least one verified citation | {_f(g['grounded_rate'])}"
        f" of {g['n_answers']} (95% interval {gi[0]:.2f} to {gi[1]:.2f}) |\n"
        f"| Unanswerable questions answered anyway |"
        f" {round(a['false_answer_rate'] * an['n_unanswerable'])} of {an['n_unanswerable']} |\n"
        f"| NOT_FOUND precision / recall | {_f(a['not_found_precision'])}"
        f" / {_f(a['not_found_recall'])} |\n"
        f"| Answerable questions answered NOT_FOUND |"
        f" {round(a['over_abstention_rate'] * an['n_answerable'])} of {an['n_answerable']} |\n"
        f"| Median time per answer | {an['seconds_median']:.0f} s |\n"
    )


RENDER = {
    "extraction": extraction,
    "retrieval": retrieval,
    "conversation": conversation,
    "answers": answers,
}


def render_readme(text: str) -> str:
    def replace(m: re.Match[str]) -> str:
        return m[1] + RENDER[m["name"]]() + m[4]

    return BLOCK.sub(replace, text)


def main() -> None:
    readme = ROOT / "README.md"
    readme.write_text(render_readme(readme.read_text(encoding="utf-8")), encoding="utf-8")


if __name__ == "__main__":
    main()
