# Tuning log

Every setting chosen with evaluation data is recorded here: what was compared, on which
split, and the numbers. All tuning uses the dev split only (ADR-0004, ADR-0007).

## Chosen so far

| Setting | Choice | Evidence |
|---|---|---|
| Extractor | pypdf | Same mean CER as pypdfium2 (0.031) on 8 transcribed pages; the rule fixed in advance keeps pypdf within 1 CER point (`eval/results/extractor-comparison.json`, ADR-0002 Amendment 2). |
| Header/footer lines in chunks | kept | Not tuned: removing them was to be tested on dev retrieval, cut in design revision 3. |
| Chunk size and overlap | defaults (`ChunkParams`) | Not tuned yet. |
| Retriever, k, RRF constant | BM25, k = 5 | Not tuned yet; dense and hybrid arrive with the model providers and are adopted only with a paired test on dev. |
| Follow-up questions | history joined (`concat`) | Dev: hit@1 0.500 vs 0.429 as typed (14 follow-ups; the intervals overlap). The model rewrite was cut in design revision 3. |

## Test-split exposure

On 2026-10-21, while checking the `--split test` guard, a test run printed aggregate
scores for the test questions once. Nothing was tuned, recorded or committed from them.
Later the same week some test questions gained gold pages (`01f0a1c`); each addition came
from reading page images, was applied to dev questions alike, and moved no question
between splits. Since then tests never touch the real test split.
