# Tuning log

Every setting chosen with evaluation data is recorded here: what was compared, on which
split, and the numbers. All tuning uses the dev split only (ADR-0004, ADR-0007).

## Chosen so far

| Setting | Choice | Evidence |
|---|---|---|
| Extractor | pypdf | Same mean CER as pypdfium2 (0.031) on 8 transcribed pages; the rule fixed in advance keeps pypdf within 1 CER point (`eval/results/extractor-comparison.json`, ADR-0002 Amendment 2). |
| Header/footer lines in chunks | kept | Not tuned: removing them was to be tested on dev retrieval, cut in design revision 3. |
| Chunk size and overlap | defaults (`ChunkParams`) | Not tuned yet. |
| Retriever | BM25 (default) | Dense and hybrid were not better on dev by the rule below; see "Retriever and RRF constant". |
| RRF constant | 60 | 10 and 30 made no detectable difference. |
| k (pages sent to the model) | 5 | Not tuned yet; chosen with the answering stage. |
| Follow-up questions | history joined (`concat`) | Dev: hit@1 0.500 vs 0.429 as typed (14 follow-ups; the intervals overlap). The model rewrite was cut in design revision 3. |

## Retriever and RRF constant (2026-10-27)

Rule, fixed in ADR-0004 before the numbers: the primary metric is nDCG@10 on dev; the
simpler option (BM25 < dense < hybrid; RRF constant 60) is kept unless the other is better
by a paired sign-flip test with p < 0.05 and a 95% paired-bootstrap interval of the
per-question difference that excludes 0. 4 comparisons were made.
`eval/tools/tune_retrieval.py` reproduces them; embeddings by ollama/nomic-embed-text
(digest `0a109f422b47`) on Ollama 0.40.5, Intel Mac, CPU only.

| Retriever on dev (49 questions) | nDCG@10 | hit@1 |
|---|---|---|
| bm25 | 0.768 | 0.592 |
| dense | 0.734 | 0.571 |
| hybrid-10 | 0.775 | 0.571 |
| hybrid-30 | 0.771 | 0.571 |
| hybrid-60 | 0.771 | 0.571 |

| Comparison (difference in nDCG@10) | Result | Adopted |
|---|---|---|
| hybrid-10 vs hybrid-60 | +0.003 [-0.019, 0.026], p = 1.00 | no |
| hybrid-30 vs hybrid-60 | +0.000 [0.000, 0.000], p = 1.00 | no |
| dense vs bm25 | -0.033 [-0.121, 0.053], p = 0.46 | no |
| hybrid-60 vs bm25 | +0.004 [-0.065, 0.066], p = 0.92 | no |

Dense retrieval does what it is for on paraphrased questions (nDCG@10 0.648 with BM25,
0.673 dense, 0.696 hybrid) but loses on lexical and table questions, and over 49
questions no difference is detectable. BM25 stays the default; `--retriever hybrid` is
available. A larger dev set is what would change this.

## Test-split exposure

On 2026-10-21, while checking the `--split test` guard, a test run printed aggregate
scores for the test questions once. Nothing was tuned, recorded or committed from them.
Later the same week some test questions gained gold pages (`01f0a1c`); each addition came
from reading page images, was applied to dev questions alike, and moved no question
between splits. Since then tests never touch the real test split.
