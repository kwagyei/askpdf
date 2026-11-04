# askpdf

[![CI](https://github.com/kwagyei/askpdf/actions/workflows/ci.yml/badge.svg)](https://github.com/kwagyei/askpdf/actions/workflows/ci.yml)

Point askpdf at a folder of PDFs and ask questions about them in your browser. Answers
cite the pages they came from, and each quoted passage is checked word for word against
its page, so you can see which claims are backed and which are not (the results below say
how often a small local model gets there). It runs locally with
[Ollama](https://ollama.com); nothing leaves your machine unless you opt in to a remote
model.

![askpdf answering a question about a NIST standard, with a checked quote under each citation](docs/screenshots/answer.png)

## Install and run

askpdf needs Python 3.12 or newer and [Ollama](https://ollama.com) for answers. It is
installed from GitHub, not PyPI:

```sh
pipx install git+https://github.com/kwagyei/askpdf@v0.1.0
ollama pull llama3.2:3b          # answers (2 GB)
ollama pull nomic-embed-text     # optional: search by meaning (270 MB)
askpdf ~/Documents/papers        # indexes the folder, watches it, opens the browser
```

Without Ollama, indexing and keyword search still work, and askpdf says why answers are
not available. On a laptop without a GPU an answer takes one to three minutes (measured
below).

### From the command line

```sh
$ askpdf index ~/Documents/papers
reading nist-sp-800-63b-4.pdf
1 added, 0 already known (copies or renames), 0 removed, 0 unchanged, 0 failed
$ askpdf docs ~/Documents/papers
nist-sp-800-63b-4.pdf  129 pages, 279 chunks, 0 image-only
$ askpdf search ~/Documents/papers "phishing-resistant authenticators" -k 2
1. nist-sp-800-63b-4.pdf  p. 35 (PDF page 46)  score 8.33
   NIST SP 800-63B-4 July 2025 Digital Identity Guidelines Authentication and Authenticator Management 3.2.5. Phishing Resistance Phishing attacks, previously ref…
2. nist-sp-800-63b-4.pdf  p. 29 (PDF page 40)  score 7.16
   meet the requirements of Sec. 3.2.3, including limits on the number of consecutive authentication failures. The password or biometric sample used for activatio…
```

`search` ranks pages by BM25 over their chunks, needs no model, and treats `multi-factor`
and `multi factor` as the same words. With [Ollama](https://ollama.com) running and
`nomic-embed-text` pulled, `index` also embeds every chunk, and `search --retriever dense`
or `--retriever hybrid` (reciprocal rank fusion of both) finds pages by meaning. Keyword
search stays the default because on the development questions neither alternative was
measurably better (see `eval/TUNING.md`). Without Ollama everything still works by
keyword, and askpdf says so. Re-running `index` only reads new or changed files and only
embeds chunks that have no vector yet; renamed and copied files are recognised by their
content. The index lives in your cache folder, never in the folder you point it at.

### Answers with checked quotes

`askpdf ask FOLDER "question"` gives the 3 best pages to a local model (llama3.2:3b via
Ollama by default) and asks for an answer that cites them as `[1]`, `[2]`, followed by a
quote from each cited page, or `NOT_FOUND` if the pages do not say. askpdf then checks the
reply instead of trusting it:

- A citation must name one of the pages the model was given; any other number is removed.
  File names and page numbers are always taken from the index, never from the model's text.
- Each quote must appear on the cited page. "Appear" means a plain substring match after
  ignoring case, the style of quote marks and dashes, runs of whitespace, and a hyphen
  between two letters (so `multi-factor` matches `multifactor`). A hyphen next to a digit
  counts (`SP 800-63B` is not `SP 80063B`), and nothing else is forgiven: a changed letter,
  a dropped word or an added "not" fails. Quotes under 5 words are not accepted.
- Every source is printed with a ✓ or the reason its quote failed. An answer with no
  verified quote is flagged as ungrounded, and `ask` exits with status 3.

A verified quote proves that these words are on that page, not that they support the
sentence they are attached to.

A real run on the development Mac (Intel, CPU only); two long quotes are shortened where marked …:

```text
$ askpdf ask ~/Documents/papers "Which Unicode normalization form should be applied to passwords before hashing?"
answering from 3 pages…
The Unicode normalization form that should be applied to passwords before hashing is NFC normalization defined in Sec. 12.1 of Unicode Normalization Forms [UAX15].

Sources:
[1] nist-sp-800-63b-4.pdf  p. 15 (PDF page 26)
    ✓ “This process is applied before hashing the byte string that represents the password”
[2] nist-sp-800-63b-4.pdf  p. 16 (PDF page 27)
    the quoted words are on nist-sp-800-63b-4.pdf p. 15 (PDF page 26); the answer numbered its source wrongly
    ✗ not found on the page: “Subscribers choosing passwords that contain Unicode characters SHOULD be advised …”
[3] nist-sp-800-63b-4.pdf  p. 87 (PDF page 98)
    the quoted words are on nist-sp-800-63b-4.pdf p. 15 (PDF page 26); the answer numbered its source wrongly
    ✗ not found on the page: “The entire password SHALL be subject to comparison, not substrings or words …”
```

The answer is right (page 26 names NFC), but only one of its three quotes checks out: the
other two are real sentences from page 26, filed under the wrong number. That is typical
of the 3B model, and the results below count it.

### Questions about a page or a line

Search finds pages by their words, but "page 5 of report.pdf" is not a phrase on any page.
So a question that names a page is answered from that page directly, both in `ask` and in
the browser:

- The document is matched by its file name, in any case, with or without `.pdf`, and with
  spaces for dashes. Most of a long name ("NIST SP 800-63B") or one word that only one
  file has ("the naca report") is enough. A folder of one PDF needs no name. A follow-up
  ("and page 6?") keeps the document of the earlier question. If the name is missing or
  fits several files, askpdf asks which one.
- "Page 5" can mean the fifth page of the file or the page printed "5". When those differ,
  both are read and the citation shows which one the answer came from. "PDF page 5" means
  the fifth page of the file only.
- A page that does not exist gets a plain reply ("report.pdf has no page 99 (12 pages)").
- A question about a line ("what is the second line on page 4 of the naca report?") is
  answered without the model at all: the line is read from the page's stored text and
  cited like any quote. Lines are counted as the PDF's text layer gives them, blank lines
  skipped, so a running header counts as a line and columns may not read in the order
  they look.
- A request to summarise a whole document or chapter gets a plain reply saying askpdf
  cannot do that yet, with a suggestion that works ("summarise page 12 of …"), rather
  than a summary of the few pages a search happened to find. A summary of a topic
  ("summarise what it says about passwords") is searched as usual.

### In the browser

`askpdf FOLDER` (short for `askpdf serve FOLDER`) indexes the folder in the background,
then keeps watching it, and opens a page on `127.0.0.1`. The pages being read appear as soon as they are found, the
answer streams in while the model writes, and the checked version then replaces it: each
`[n]` links to its PDF, opened at the cited page, and each quote shows ✓ or why it failed.
A follow-up question is searched together with the conversation's earlier questions (up
to three in all), and the page shows what was searched for. Search is by keyword unless you
start it with `--retriever hybrid` (or `dense`), which also embeds the folder with Ollama
and falls back to keywords, saying why, whenever that is not possible.

The server is local, but other websites open in the same browser can still send it
requests, so it guards against that:

- It listens on 127.0.0.1 only, with no option to change it, and refuses any `Host` header
  but its own (against DNS rebinding).
- The link printed in the terminal carries a one-time key, which the page swaps for an
  HttpOnly, SameSite=Strict cookie. Without that cookie the API and the PDFs answer 403.
- A request that can change anything (any method but GET, HEAD and OPTIONS) must come
  from the page's own origin.
- A PDF is requested by the number of its row in the index, never by a path. Before it is
  sent, the file is resolved again and must still be a regular `.pdf` inside the folder,
  so a file swapped for a symlink after indexing is refused.
- Every response carries a Content Security Policy that allows no inline script, and model
  and PDF text is inserted as text, never as HTML.

Each of these has a test (`tests/test_web.py`).

#### Watching the folder

PDFs added to, changed in or deleted from the folder show up in the page without a
restart. The folder is polled every 2 seconds rather than watched with OS events: polling
needs no dependency and behaves the same on macOS, Linux, Windows, iCloud and network
folders, at a cost meant for hundreds of PDFs (not measured). A file is read only once its size
and modification time are the same in two scans in a row, so a file still being copied or
downloaded is never indexed half-written; it does not hold up the other files meanwhile.
Files are keyed by content, so a renamed or copied PDF is recorded without being read
again.

Measured by hand on the Intel Mac this was built on (10 runs each, keyword search, PDFs of
1.0 and 1.1 MB, with `askpdf serve` running), from the file landing in the folder to the
index holding it:

| Change | Median | Range |
| --- | --- | --- |
| Add a PDF | 5.5 s | 4.3–7.9 s |
| Change a PDF | 5.9 s | 4.3–7.4 s |
| Delete a PDF | 2.0 s | 2.0–2.0 s |

An add or change is two scans plus the time to read the file, so a larger PDF takes
longer. The deletes were each made just after a scan, so they show the slowest case; one
made at a random moment is picked up within 0–2 s. With `--retriever hybrid` the new
chunks are also embedded, which takes much longer (see the limitations).

Indexing gives way to answers: while one is being written the indexer waits before its
next file or batch of embeddings (a file or batch already started finishes first). Only one askpdf process writes an index at a
time (an OS file lock); a second `askpdf serve` on the same folder shows the index as it
is and says so, and `askpdf index` refuses with a clear message.

#### Opening a citation at its page

Whether the browser's own PDF viewer honours `#page=N` is checked by hand, on macOS 15,
by clicking a source in a running app (2026-11-03):

| Browser | Opens at the cited page? |
| --- | --- |
| Chrome | yes |
| Safari | yes |
| Firefox | not checked |

## What makes it different

The interesting part is not the chatbot but the measurement. Each stage is scored on its own,
so a bad answer can be traced to where it went wrong:

| Stage | Question it answers | Status |
| --- | --- | --- |
| Text extraction | Did we read the page correctly? (character error rate on hand-transcribed pages) | measured |
| Retrieval | Did we find the right page? (hit@k, recall@k, MRR, nDCG) | BM25, dense and hybrid measured |
| Grounding | Does the cited page actually contain the quote? | measured (local model) |
| Abstention | Does it say "not found" instead of guessing? | measured (local model) |
| Follow-ups | Does a follow-up question still find the right page? | measured |
| Latency | How long does it take on an 8 GB laptop without a GPU? | measured |

Tuning happens on a development split; the test split is frozen (hash-pinned) and scored
once per release. Questions from the Qasper dataset, written by people who had seen only
a paper's title and abstract, were planned as an outside check but were cut to ship
sooner; the limitations below say what that leaves. A CI job reruns every model-free number below (extraction, retrieval and follow-ups, on dev
and on the frozen test split), checks these tables against the results files, and fails if
a number changes without them. Numbers that need the model are run by hand and committed.

### First results (development split, model-free)

<!-- results:extraction -->
Extraction with pypdf 6.21.2, on 8 hand-transcribed pages from a security
standard, a two-column paper and a 1957 scanned report:

| Measure | Result |
| --- | --- |
| Mean character error rate | 0.031 (best page 0.003, worst 0.123) |
| Header/footer detection, precision / recall | 1.000 / 0.553 |
| Scanned-page detection, precision / recall | 1.000 / 1.000 (2 pages without text) |
<!-- /results -->

The worst pages are the scan, whose OCR layer reads "65A004" as "65AOOi-", and the
paper's title page, where a figure interrupts the reading order. Header recall is low
because the scan's OCR mangles each page's header differently, so it never repeats.

<!-- results:retrieval -->
Keyword retrieval (BM25) over the whole corpus, 49 answerable dev questions:

| Questions | n | hit@1 | hit@5 | MRR@10 | nDCG@10 |
| --- | --- | --- | --- | --- | --- |
| all | 49 | 0.592 | 0.878 | 0.712 | 0.768 |
| lexical | 13 | 0.923 | 1.000 | 0.949 | 0.955 |
| multi-page | 1 | 0.000 | 1.000 | 0.200 | 0.456 |
| paraphrase | 26 | 0.423 | 0.769 | 0.578 | 0.648 |
| scanned | 5 | 0.800 | 1.000 | 0.900 | 0.936 |
| table | 4 | 0.500 | 1.000 | 0.708 | 0.800 |

95% bootstrap intervals for the overall row: hit@1 0.45 to 0.73, nDCG@10 0.68 to 0.85.
<!-- /results -->

With so few questions per row, only the overall row and the lexical/paraphrase contrast
mean much. BM25 finds questions that reuse the document's words (lexical) far more
often than paraphrased ones. That gap is what the dense and hybrid retrievers have to
close, and each is adopted only if a paired test on dev says it helps. The follow-ups
below are even fewer, so their intervals are wide.

<!-- results:conversation -->
Follow-up questions (14 on dev, such as "What about at AAL3?"), searched three ways:

| Follow-up searched | hit@1 (95% interval) | nDCG@10 |
| --- | --- | --- |
| as typed | 0.429 (0.21 to 0.71) | 0.620 |
| history joined | 0.500 (0.21 to 0.79) | 0.754 |
| hand-written rewrite | 0.857 (0.64 to 1.00) | 0.933 |
<!-- /results -->

### Answers (development split, local model)

These need Ollama, so they are run by hand on the development Mac (Intel, no GPU) and
committed, not rerun in CI. Every reply is kept in `eval/results/dev-answers.json`.

<!-- results:answers -->
Answers from llama3.2:3b (Ollama 0.40.5) on the 58 dev questions, 3 BM25 pages each; MacBook Pro 13-inch 2019, Intel Core i5-8257U 1.4 GHz (4 cores), 8 GB, macOS 15.7, no GPU, 2026-10-31:

| Measure | Result |
| --- | --- |
| Citations with a quote found on the cited page | 11 of 59 (0.186) |
| of those, verified only by folding hyphens | 0.000 |
| Answer sentences carrying a verified citation | 0.159 |
| Answers with at least one verified citation | 0.224 of 49 (95% interval 0.12 to 0.35) |
| Unanswerable questions answered anyway | 3 of 9 |
| NOT_FOUND precision / recall | 1.000 / 0.667 |
| Answerable questions answered NOT_FOUND | 0 of 49 |
| Median time per answer | 63 s |
<!-- /results -->

The small local model rarely answers in the checkable form: of the citations that failed, 13 quoted the right words under the number of another page
it was given, and 20 had no quote because the model copied the page's wording into the
answer sentence instead. So the grounding rate here says more about a 3B model following
a format than about whether its answers are true; a larger model, or a remote one with
`--api-url`, is the obvious thing to try. Unanswerable questions are where it helps most:
it said "not found" for 6 of 9 and never for an answerable one.

### Release v0.1.0: the frozen test split

Every choice above was made on the development split (some settings were never tuned). The
test split was hash-pinned when the questions were written and scored once, for this
release, after all tuning was done; `eval/results/test-v0.1.0.json` holds every
per-question result. Two earlier contacts with it are logged in
[`eval/TUNING.md`](eval/TUNING.md#test-split-exposure): one accidental printout of test
aggregates, and gold pages added to some test questions, which changed the pin. Nothing
was tuned from either. The release workflow rescores its model-free stages from the tagged
commit and refuses to publish the release unless they match exactly.

<!-- results:test-split -->
BM25, k = 3 pages, 46 answerable test questions (49 on dev); 95% bootstrap intervals in brackets. Answers from llama3.2:3b (Ollama 0.40.5) on MacBook Pro 13-inch 2019, Intel Core i5-8257U 1.4 GHz (4 cores), 8 GB, macOS 15.7, no GPU; test split scored 2026-11-02.

| Measure | dev (tuned on) | test, v0.1.0 (scored once) |
| --- | --- | --- |
| Retrieval hit@1, all answerable | 0.592 (0.45 to 0.73) | 0.674 (0.54 to 0.80) |
| Retrieval nDCG@10 | 0.768 | 0.794 |
| hit@1, lexical / paraphrased questions | 0.923 / 0.423 | 0.818 / 0.615 |
| Follow-ups, hit@1 with history joined | 0.500 (n = 14) | 0.600 (n = 10) |
| Answers with a verified citation | 0.224 (0.12 to 0.35) | 0.395 (0.26 to 0.53) |
| Unanswerable questions answered anyway | 3 of 9 | 3 of 7 |
| Median time per answer | 63 s | 71 s |
<!-- /results -->

The test split came out a little better than dev, within the intervals; with under 50
questions on each, neither is a precise number, and the gap is not a sign of anything.
The share of answers with a verified citation is the measure that moved most, and it
depends heavily on whether the 3B model follows the quote format on a given question.

#### Are the answers right?

The automatic checks above show whether quoted words are on the cited page, not whether
the answer is right, so the test answers were also labelled by hand. One labeller did
this: Claude, the AI model that is this project's lead developer and wrote the questions.
There was no second labeller, so these are one careful reader's judgements, each with its
reason in `eval/results/test-v0.1.0-labels.json`, and the rules were written down before
any answer was read.

<!-- results:correctness -->
Each of the 53 test answers labelled by hand against its evidence, by the rules in `eval/LABELS.md`:

| Answer | Questions |
| --- | --- |
| Correct | 38 of 46 (0.826, 95% interval 0.72 to 0.93) |
| Partly correct | 1 of 46 |
| Wrong | 4 of 46 |
| Said not found, though the answer is in the documents | 3 of 46 |
| Unanswerable, correctly declined | 5 of 7 |

Of the 8 answerable questions not answered correctly, 5 never had the page with the answer among the 3 pages the model read, and 3 had it and misread it.
<!-- /results -->

Two of the 38 correct answers came from the model's own knowledge: none of the 3 pages it
read held the evidence (`nist-06`, `nist-22`). They count as correct because the labels
judge the answer, but askpdf could not have shown where they came from.

One difference from the automatic count above: it scores 3 unanswerable questions as
answered, but one of those replies says in words that the pages do not say, which counts
as declining here.

So the small model is right far more often than its verified citations suggest: most
misses are formatting (a quote filed under the wrong number, or no quote at all), not
wrong facts. When it is wrong, the cause is more often search than reading: the page
with the answer was not among the 3 it was given.

### Latency

<!-- results:latency -->
`askpdf bench` on MacBook Pro 13-inch 2019, Intel Core i5-8257U 1.4 GHz (4 cores), 8 GB, macOS 15.7, no GPU; llama3.2:3b on Ollama 0.40.5, 2026-11-02:

| Step | n | p50 | p95 (the slowest, when n < 20) |
| --- | --- | --- | --- |
| Index one document (extract, clean, chunk, store) | 12 | 0.70 s | 6.25 s |
| Keyword search over the corpus | 174 | 0.21 ms | 0.40 ms |
| Answer, warm: first token | 9 | 56 s | 120 s |
| Answer, warm: checked answer shown | 9 | 66 s | 185 s |
| Answer, cold (model loaded first): first token / shown | 1 | 89 s / 98 s | |

Indexing ran at 20.5 pages a second over 4 documents (3 runs); embedding for search by meaning is extra. The pages to be read are shown in under 0.05 s; nearly all of the wait is the model reading them. Ollama is given a context window sized to each prompt, and may reload the model when that size changes, so some warm times may include a reload.
<!-- /results -->

**Who wrote the evaluation data.** The questions, conversations and page transcriptions
were drafted by an AI model (Claude) working as the project's lead developer, and are
being checked by hand by the author. The questions were written from askpdf's extracted
text, so they lean towards passages the extractor reads well; self-written questions also
favour lexical retrieval. Questions written blind by outsiders would offset both; that
check is not part of this version. (The in-house questions with ids `qasper-*` are about
the paper that introduced Qasper, one of the corpus documents; they are not from the
dataset.) Every setting chosen on the data is logged in `eval/TUNING.md`. On the
scanned report, evidence quotes follow its OCR layer, misreadings included, because they are
checked against the text askpdf actually extracts. The corpus files and their licences are
listed in `eval/corpus/SOURCES.md`.

## Known limitations

- Answers are slow on a laptop without a GPU. On the Intel Mac this was built on, the 3B
  model reads the 3 pages for most of a minute before it writes; the median answer on the
  development questions took 63 seconds. Fewer pages are sent for that reason, so a few answers miss
  their page (see `eval/TUNING.md`).
- Answer correctness was labelled by one labeller, the AI model that built the project,
  on 53 test questions: 38 of 46 answerable ones correct. A second, human labeller and an
  agreement score were cut to ship sooner, so treat that number as a careful estimate.
- The answer model sees only the question and the retrieved pages, not earlier turns. A
  follow-up is searched with the earlier questions joined, which helps retrieval (hit@1
  0.500 against 0.429 as typed on 14 development follow-ups, though the intervals
  overlap), but a question such as "is it applied before hashing?" is still answered
  without knowing what "it" is.
- No summaries of whole documents or chapters: answers come from 3 pages, and askpdf
  does not read a PDF's table of contents. Such requests are declined plainly.
- The web app answers one question at a time. It searches by keyword unless started with
  `--retriever`.
- Files that iCloud has moved off the Mac ("Optimise Mac Storage") are downloaded when
  they are first read. askpdf gives up on a read after 30 seconds, marks the file "not
  available locally" and tries again the next time something in the folder changes. How
  current macOS behaves here has not been checked; the timeout is tested with a read that
  is made to hang. A read that hangs leaves one idle thread behind until it ends, and a
  file that stays unreadable costs those 30 seconds again on each later change.
- A change is noticed by size and modification time. A file rewritten with exactly the
  same size and modification time (a tool that restores the old timestamp) is not read
  again until something else about it changes.
- Indexing that takes under about 2 seconds may not show in the status line: the page
  hears about indexing every 2 seconds, and then shows the new document list.
- Polling scans the whole folder every 2 seconds, which is meant for hundreds of PDFs, not
  tens of thousands.

- Semantic search needs Ollama running locally, and embedding is slow on a CPU: about
  2 chunks a second on the Intel Mac this was built on (timed by hand, not in a results
  file), so a 130-page report takes about
  three minutes the first time. Later runs only embed what changed.
- No OCR: scanned pages without a text layer are detected and flagged, not read.
- Text is read with pypdf; tables and two-column layouts may come out in an odd order.
  On the 8 transcribed pages pypdfium2 ties with pypdf (character error rate 0.031) and
  pdfplumber reads across two-column pages (0.260), so pypdf stays.
- PDFs that need a password to open are skipped with the reason shown. PDFs that open
  with an empty password (RC4 or AES) are read, including ones marked "no copying" or "no
  printing": askpdf reads your own files locally and does not enforce those advisory
  permission flags.
- A word broken across lines at a hyphen is rejoined without the hyphen only if the same
  document spells it solid more often than hyphenated; otherwise the hyphen stays. So
  "multi-factor" stays a compound, but a word broken by layout that the document uses only
  once keeps a stray hyphen ("algo-rithm"). How often that happens is not measured.
  Header and footer lines are never joined to the text, but they are only recognised in
  documents of 4 pages or more; in a shorter one, a page number under a hyphenated last
  line can end up glued on ("multi-3").
- Keyword search splits words at spaces and punctuation, with no stemming and an English
  stopword list. Languages written without spaces between words (Thai, Chinese, Japanese)
  are matched only as whole runs of text, and "authenticate" does not find "authenticator".
- Zero-width joiners are removed while cleaning, which can change words in scripts such
  as Persian or Devanagari.
- The evaluation is small and self-written: 4 English documents, 111 questions and 12
  conversations, drafted by the lead developer (an AI model) and not checked against
  questions written blind by outsiders. Its intervals are wide, and it says more about
  relative differences (lexical versus paraphrased questions) than about absolute quality.

## Development

Requires Python 3.12 or newer. The runtime dependencies are deliberately few: pypdf, NumPy,
FastAPI, Uvicorn and pycryptodome. pycryptodome gives pypdf an AES backend, so encrypted PDFs
that open without a password can be read; it was chosen over `cryptography`, which no longer
publishes wheels for Intel Macs.

```sh
python3 -m venv .venv && .venv/bin/pip install --upgrade pip
.venv/bin/pip install --group dev && .venv/bin/pip install .
.venv/bin/pytest --cov
.venv/bin/ruff check && .venv/bin/ruff format --check && .venv/bin/mypy
```

## Licence

MIT. Evaluation documents keep their own licences, listed alongside them.
