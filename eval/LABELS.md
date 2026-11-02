# Answer-correctness labels

The automatic checks show that a quoted passage is on the cited page, not that the answer
is right. So each answer on the test split is labelled by hand, once per release, using
the rules below. They were written down before any test answer was read.

**Who labels.** One labeller: Claude, the AI model working as this project's lead
developer, which also wrote the questions. There is no second labeller and no agreement
score (the design's second labeller was cut to ship sooner). Read the labels as one
careful reader's judgement, checkable line by line in the labels file, not as an
independent measurement.

## Labels

For an **answerable** question, the reference is the question's evidence passages and
pages in `eval/questions/test.jsonl`, and the pages themselves:

| Label | When |
| --- | --- |
| `correct` | The answer states what the evidence says, and nothing it states contradicts the document. Extra true detail is fine. |
| `partial` | Part of what the question asks is right and part is missing or wrong, or the right fact comes with a wrong qualifier (a wrong number, level or condition). |
| `wrong` | The main claim is wrong, or the answer is about something else. |
| `abstained` | The answer is `NOT_FOUND`. |

For an **unanswerable** question: `correct` if the answer is `NOT_FOUND` or says plainly
that the pages do not say; `wrong` if it gives an answer.

Labels judge the answer text only. Citations and quotes are scored automatically and do
not change the label: a correct answer with a misnumbered quote is still `correct`.

## Why an answer was not correct

Every label other than `correct` gets one cause:

| Cause | When |
| --- | --- |
| `retrieval` | No page holding the evidence was among the 3 pages given to the model (from the retrieval results for the same question). |
| `reading` | The evidence was on a page given, and the model misread or misused it. |
| `abstention` | An unanswerable question was answered anyway. |

The labels and causes are in `eval/results/test-v0.1.0-labels.json`, one entry per
question with a one-line reason.
