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
        f" {an['n_questions']} dev questions, {an['k']} BM25 pages each; {_machine(an)},"
        f" {_date(an, DEV_ANSWERS_DATE)}:\n\n"
        "| Measure | Result |\n| --- | --- |\n"
        f"| Citations with a quote found on the cited page |"
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


RELEASE = "v0.1.0"
# Model runs from v0.1.1 on record their own date, machine and git commit. These are for
# the two files made before that (dev-answers.json, test-v0.1.0.json), all on the machine
# named in bench-v0.1.0.json; delete them once both have been re-run.
DEV_ANSWERS_DATE = "2026-10-31"
TEST_SPLIT_DATE = "2026-11-02"


def _machine(run: dict[str, Any] | None = None) -> str:
    if run and "machine" in run:
        return str(run["machine"])
    return str(_load(f"bench-{RELEASE}.json")["machine"])


def _date(run: dict[str, Any], before_dates: str) -> str:
    return str(run.get("date", before_dates))


def _interval(iv: list[float]) -> str:
    return f"({iv[0]:.2f} to {iv[1]:.2f})"


def test_split() -> str:
    """Dev next to the frozen test split, scored once for the release."""
    test = _load(f"test-{RELEASE}.json")["stages"]
    dev_rt, dev_cv = _load("dev-retrieval-bm25.json"), _load("dev-conversation-bm25.json")
    rt, cv = test["retrieval"], test["conversation"]
    rows = [
        (
            "Retrieval hit@1, all answerable",
            f"{_f(dev_rt['overall']['hit@1'])} {_interval(dev_rt['interval_95']['hit@1'])}",
            f"{_f(rt['overall']['hit@1'])} {_interval(rt['interval_95']['hit@1'])}",
        ),
        (
            "Retrieval nDCG@10",
            _f(dev_rt["overall"]["ndcg@10"]),
            _f(rt["overall"]["ndcg@10"]),
        ),
        (
            "hit@1, lexical / paraphrased questions",
            f"{_f(dev_rt['by_kind']['lexical']['hit@1'])} /"
            f" {_f(dev_rt['by_kind']['paraphrase']['hit@1'])}",
            f"{_f(rt['by_kind']['lexical']['hit@1'])} / {_f(rt['by_kind']['paraphrase']['hit@1'])}",
        ),
        (
            "Follow-ups, hit@1 with history joined",
            f"{_f(dev_cv['overall']['concat']['hit@1'])} (n = {dev_cv['n_follow_ups']})",
            f"{_f(cv['overall']['concat']['hit@1'])} (n = {cv['n_follow_ups']})",
        ),
    ]
    if "answers" in test:
        dev_an, an = _load("dev-answers.json"), test["answers"]

        def grounded(x: dict[str, Any]) -> str:
            g = x["grounding"]
            return f"{_f(g['grounded_rate'])} {_interval(g['grounded_interval_95'])}"

        def false_answers(x: dict[str, Any]) -> str:
            n = x["n_unanswerable"]
            return f"{round(x['abstention']['false_answer_rate'] * n)} of {n}"

        rows += [
            ("Answers with a verified citation", grounded(dev_an), grounded(an)),
            ("Unanswerable questions answered anyway", false_answers(dev_an), false_answers(an)),
            (
                "Median time per answer",
                f"{dev_an['seconds_median']:.0f} s",
                f"{an['seconds_median']:.0f} s",
            ),
        ]
    table = "\n".join(f"| {a} | {b} | {c} |" for a, b, c in rows)
    model = test.get("answers", {})
    run = (
        f" Answers from {model['model']} (Ollama {model['ollama_version']})"
        f" on {_machine(model)}; test split scored {_date(model, TEST_SPLIT_DATE)}."
        if model
        else ""
    )
    return (
        f"BM25, k = 3 pages, {rt['n_questions']} answerable test questions"
        f" ({dev_rt['n_questions']} on dev); 95% bootstrap intervals in brackets.{run}\n\n"
        f"| Measure | dev (tuned on) | test, {RELEASE} (scored once) |\n| --- | --- | --- |\n"
        f"{table}\n"
    )


def latency() -> str:
    b = _load(f"bench-{RELEASE}.json")
    ix, sm, an = b["indexing"], b["search_ms"], b["answers_seconds"]
    warm, cold = an["warm"], an["cold"]
    doc = ix["seconds_per_document"]
    slowest_pages = max(warm["pages"]["p95"], cold["pages"])  # seconds, rounded to 0.1
    shown = "under 0.05 s" if slowest_pages == 0 else f"at most {slowest_pages:.1f} s"
    return (
        f"`askpdf bench` on {b['machine']}; {b['model']} on Ollama {b['ollama_version']},"
        f" {b['date']}:\n\n"
        "| Step | n | p50 | p95 (the slowest, when n < 20) |\n| --- | --- | --- | --- |\n"
        f"| Index one document (extract, clean, chunk, store) | {doc['n']}"
        f" | {doc['p50']:.2f} s | {doc['p95']:.2f} s |\n"
        f"| Keyword search over the corpus | {sm['n']}"
        f" | {sm['p50']:.2f} ms | {sm['p95']:.2f} ms |\n"
        f"| Answer, warm: first token | {warm['first_token']['n']}"
        f" | {warm['first_token']['p50']:.0f} s | {warm['first_token']['p95']:.0f} s |\n"
        f"| Answer, warm: checked answer shown | {warm['total']['n']}"
        f" | {warm['total']['p50']:.0f} s | {warm['total']['p95']:.0f} s |\n"
        f"| Answer, cold (model loaded first): first token / shown | 1"
        f" | {cold['first_token']:.0f} s / {cold['total']:.0f} s | |\n\n"
        f"Indexing ran at {ix['pages_per_second']} pages a second over {ix['documents']}"
        f" documents ({ix['runs']} runs); embedding for search by meaning is extra. The pages"
        f" to be read are shown in {shown}; nearly all of the wait is the model reading"
        " them. Ollama is given a context window sized to each prompt, and may reload the"
        " model when that size changes, so some warm times may include a reload.\n"
    )


def correctness() -> str:
    from askpdf.evaluate.stats import bootstrap_ci  # the same intervals as everywhere else

    data = _load(f"test-{RELEASE}-labels.json")
    lines = (ROOT / "eval" / "questions" / "test.jsonl").read_text(encoding="utf-8").splitlines()
    questions = {q["id"]: q for q in map(json.loads, lines)}
    labels = data["labels"]
    answerable = [v for k, v in labels.items() if questions[k]["answerable"]]
    unanswerable = [v for k, v in labels.items() if not questions[k]["answerable"]]
    count = Counter(v["label"] for v in answerable)
    causes = Counter(v["cause"] for v in answerable if v["cause"])
    ci = bootstrap_ci([float(v["label"] == "correct") for v in answerable])
    n, declined = len(answerable), sum(v["label"] == "correct" for v in unanswerable)
    return (
        f"Each of the {len(labels)} test answers labelled by hand against its evidence, by"
        " the rules in `eval/LABELS.md`:\n\n"
        "| Answer | Questions |\n| --- | --- |\n"
        f"| Correct | {count['correct']} of {n} ({_f(count['correct'] / n)},"
        f" 95% interval {ci.low:.2f} to {ci.high:.2f}) |\n"
        f"| Partly correct | {count['partial']} of {n} |\n"
        f"| Wrong | {count['wrong']} of {n} |\n"
        f"| Said not found, though the answer is in the documents | {count['abstained']} of {n} |\n"
        f"| Unanswerable, correctly declined | {declined} of {len(unanswerable)} |\n\n"
        f"Of the {n - count['correct']} answerable questions not answered correctly,"
        f" {causes['retrieval']} never had the page with the answer among the 3 pages the model"
        f" read, and {causes['reading']} had it and misread it.\n"
    )


RENDER = {
    "extraction": extraction,
    "retrieval": retrieval,
    "conversation": conversation,
    "answers": answers,
    "test-split": test_split,
    "latency": latency,
    "correctness": correctness,
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
