"""Choose the retriever and the RRF constant on the dev split (ADR-0004).

    python eval/tools/tune_retrieval.py      # needs Ollama running locally

Writes eval/results/dev-retrieval-{dense,hybrid}.json and dev-retrieval-tuning.json.

Rule, fixed in ADR-0004 before any numbers were seen: the primary metric is nDCG@10 on
dev. The simpler option (BM25 < dense < hybrid; for the RRF constant, the usual 60) is
kept unless the other one is better by a paired sign-flip test with p < 0.05 AND the
95% paired-bootstrap interval of the per-question difference excludes 0.
"""

import json
from pathlib import Path
from typing import Any

from askpdf.evaluate import questions, retrieval
from askpdf.evaluate.stats import paired_bootstrap, sign_flip_test
from askpdf.providers import ollama
from askpdf.providers.ollama import OllamaEmbedder
from askpdf.store import default_index_path

EVAL = Path(__file__).resolve().parents[1]
METRIC = "ndcg@10"
RRF_CANDIDATES = (10, 30, 60)
RRF_BASELINE = 60


def compare(simpler: dict[str, Any], other: dict[str, Any]) -> dict[str, Any]:
    ids = sorted(simpler["per_question"])
    a = [other["per_question"][q][METRIC] for q in ids]
    b = [simpler["per_question"][q][METRIC] for q in ids]
    ci = paired_bootstrap(a, b)
    p = sign_flip_test(a, b)
    return {
        "difference": round(ci.estimate, 6),
        "interval_95": [round(ci.low, 6), round(ci.high, 6)],
        "p_value": round(p, 6),
        "adopt": p < 0.05 and ci.excludes_zero() and ci.estimate > 0,
    }


def main() -> None:
    dev = questions.load(EVAL / "questions" / "dev.jsonl")
    embedder = OllamaEmbedder(ollama.base_url())
    provenance = {"embedder": embedder.name, **ollama.server_info(embedder.url)}
    corpus = EVAL / "corpus"
    snapshot = retrieval.load_embedded_snapshot(corpus, embedder, default_index_path(corpus))
    queries = retrieval.QueryCache(embedder)
    runs: dict[str, dict[str, Any]] = {
        "bm25": retrieval.evaluate(dev, retrieval.bm25_retriever(snapshot)),
        "dense": retrieval.evaluate(dev, retrieval.model_retriever(snapshot, queries, "dense")),
    }
    for c in RRF_CANDIDATES:
        hybrid = retrieval.model_retriever(snapshot, queries, "hybrid", rrf_k=c)
        runs[f"hybrid-{c}"] = retrieval.evaluate(dev, hybrid)

    steps: dict[str, Any] = {}
    rrf = RRF_BASELINE
    for c in RRF_CANDIDATES:
        if c != RRF_BASELINE:
            steps[f"hybrid-{c} vs hybrid-{RRF_BASELINE}"] = verdict = compare(
                runs[f"hybrid-{RRF_BASELINE}"], runs[f"hybrid-{c}"]
            )
            if verdict["adopt"] and (rrf == RRF_BASELINE or verdict["difference"] > 0):
                rrf = c
    chosen = "bm25"
    for candidate in ("dense", f"hybrid-{rrf}"):
        steps[f"{candidate} vs {chosen}"] = verdict = compare(runs[chosen], runs[candidate])
        if verdict["adopt"]:
            chosen = candidate

    tuning = {
        "split": "dev",
        "metric": METRIC,
        "n_questions": runs["bm25"]["n_questions"],
        **provenance,
        "means": {name: run["overall"] for name, run in runs.items()},
        "intervals": {name: run["interval_95"][METRIC] for name, run in runs.items()},
        "comparisons": steps,
        "comparisons_made": len(steps),
        "chosen": {"retriever": chosen.split("-")[0], "rrf_k": rrf},
    }
    results = EVAL / "results"
    for name, retriever in (("dense", "dense"), (f"hybrid-{rrf}", "hybrid")):
        summary = {"stage": "retrieval", "split": "dev", "retriever": retriever, **provenance}
        if retriever == "hybrid":
            summary["rrf_k"] = rrf
        summary.update(runs[name])
        path = results / f"dev-retrieval-{retriever}.json"
        path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (results / "dev-retrieval-tuning.json").write_text(
        json.dumps(tuning, indent=2) + "\n", encoding="utf-8"
    )
    for name, means in tuning["means"].items():
        print(f"{name:10} nDCG@10 {means[METRIC]:.3f}  hit@1 {means['hit@1']:.3f}")
    for step, verdict in steps.items():
        low, high = verdict["interval_95"]
        print(
            f"{step}: difference {verdict['difference']:+.3f} [{low:.3f}, {high:.3f}],"
            f" p = {verdict['p_value']:.3f}, adopted: {verdict['adopt']}"
        )
    print("chosen:", tuning["chosen"])


if __name__ == "__main__":
    main()
