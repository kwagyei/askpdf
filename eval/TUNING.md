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
| k (pages sent to the model) | 3 | Latency against recall on dev; see "Pages given to the answer model". |
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

## Pages given to the answer model (2026-10-31)

On the target Intel Mac, llama3.2:3b reads the prompt at about 33 tokens a second before
it writes anything (Ollama's `prompt_eval_duration`: 1,942 tokens in 59 s for 3 NIST pages,
then about 4.5 tokens a second while writing). Each page adds roughly 650 tokens, so
k = 5 adds about 40 s of waiting to every answer over k = 3.

What that costs in retrieval, BM25 on the 49 answerable dev questions
(`eval/results/dev-retrieval-bm25.json`, `ranked`):

| k | Gold page among the k pages |
|---|---|
| 1 | 29 (0.592) |
| 3 | 39 (0.796) |
| 5 | 43 (0.878) |

k = 3 was chosen: 4 of 49 questions lose their gold page, in exchange for answers about a
minute sooner. This is a judgement on speed, not a significance test; `ask -k 5` is
available. The answer stage below is measured at k = 3.

The prompt wording was changed once, after one ad-hoc question about NIST password length
(not from either split):
the first wording put `[n]` only in the Quotes section, never after sentences, so rule 1
now asks for a number at the end of every sentence and the example shows two.

## Answer prompt: block numbering (2026-10-31)

The first full dev run (58 questions) verified 12 of 52 citations (0.231). Of the 29
citations whose quote failed, 22 quoted words that are on the page word for word, but
under the wrong number: the words were on another of the 3 pages the model was given,
usually `[1]` cited as `[2]`. The checker was right to fail them (a citation names a
page), so the fix belongs in the prompt:

- the example cited `[2]` before `[1]`; it now cites in order. On 4 of the failing
  questions this alone changed nothing;
- each page block now opens with `=== Page [n]: file, page … ===` and closes with
  `=== End of page [n] ===`. On the same 4 questions, 1 became verified and 1 more cited
  the right page with a paraphrased quote.

The full rerun with both changes is the one recorded in `eval/results/dev-answers.json`:

| Dev, 58 questions | First prompt | Fenced blocks (kept) |
|---|---|---|
| Citations whose quote is on the cited page | 12 of 52 (0.231) | 11 of 59 (0.186) |
| Answers with a verified citation | 12 of 48 (0.250) | 11 of 49 (0.224) |
| Unverified: words on another given page | 22 | 13 |
| Unverified: no quote given | 11 | 20 |
| Unanswerable questions answered anyway | 5 of 9 | 3 of 9 |
| NOT_FOUND precision | 0.800 | 1.000 |

Misnumbering fell, but the model now more often leaves out the Quotes section and copies
the page's words into the answer sentence instead. Grounding did not change detectably
(the per-answer intervals overlap widely). The fenced prompt is kept for its abstention
and because a misnumbered citation is the worse failure; prompt tuning stops here, as
each full run takes about 75 minutes. Checking the cited sentence itself when no quote is
given would count many of the 20 copied sentences, but it changes ADR-0006, so it is left
to the architect.
A quote that is on another given page is now reported as such (`quote_on`), both in
`ask` and as `unverified_misnumbered` in the results, but it still does not verify.

## Test-split exposure

On 2026-10-21, while checking the `--split test` guard, a test run printed aggregate
scores for the test questions once. Nothing was tuned, recorded or committed from them.
Later the same week some test questions gained gold pages (`01f0a1c`); each addition came
from reading page images, was applied to dev questions alike, and moved no question
between splits. Since then tests never touch the real test split.
