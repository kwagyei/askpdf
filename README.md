# askpdf

[![CI](https://github.com/kwagyei/askpdf/actions/workflows/ci.yml/badge.svg)](https://github.com/kwagyei/askpdf/actions/workflows/ci.yml)

Point askpdf at a folder of PDFs and ask questions about them in your browser. Every answer
cites the page it came from, and each quoted passage is checked word for word against that
page. It runs locally with [Ollama](https://ollama.com); nothing leaves your machine unless
you opt in to a remote model.

> **Work in progress.** The design is settled and the code arrives through pull requests.
> What works today: reading a folder of PDFs into the local index and searching it by
> keyword (BM25). Answers and the web app come next.

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
and `multi factor` as the same words. Re-running `index` only reads new or changed files;
renamed and copied files are recognised by their content and not read again. The index lives in your cache folder, never in the
folder you point it at.

## What makes it different

The interesting part is not the chatbot but the measurement. Each stage is scored on its own,
so a bad answer can be traced to where it went wrong:

| Stage | Question it answers | Status |
| --- | --- | --- |
| Text extraction | Did we read the page correctly? (character error rate on hand-transcribed pages) | measured |
| Retrieval | Did we find the right page? (hit@k, recall@k, MRR, nDCG) | BM25 measured |
| Grounding | Does the cited page actually contain the quote? | planned |
| Abstention | Does it say "not found" instead of guessing? | planned |
| Follow-ups | Are follow-up questions rewritten correctly? | baselines measured |
| Latency | How long does it take on an 8 GB laptop without a GPU? | planned |

Tuning happens on a development split; the test split is frozen (hash-pinned) and scored
once per release. Questions written by people who had not seen the answer (Qasper)
complement the in-house set. A required CI job reruns every model-free number below and
fails if one changes without the results file changing with it.

### First results (development split, model-free)

Extraction with pypdf 6.21.2, on 8 hand-transcribed pages from a security
standard, a two-column paper and a 1957 scanned report:

| Measure | Result |
| --- | --- |
| Mean character error rate | 0.031 (best page 0.003, worst 0.123) |
| Header/footer detection, precision / recall | 1.000 / 0.553 |
| Scanned-page detection, precision / recall | 1.000 / 1.000 |

The worst pages are the scan, whose OCR layer reads "65A004" as "65AOOi-", and the
paper's title page, where a figure interrupts the reading order. Header recall is low
because the scan's OCR mangles each page's header differently, so it never repeats.

Keyword retrieval (BM25) over the whole corpus, 49 answerable dev questions:

| Questions | n | hit@1 | hit@5 | MRR@10 | nDCG@10 |
| --- | --- | --- | --- | --- | --- |
| all | 49 | 0.592 | 0.878 | 0.712 | 0.769 |
| lexical | 13 | 0.923 | 1.000 | 0.949 | 0.962 |
| multi-page | 2 | 0.000 | 1.000 | 0.267 | 0.513 |
| paraphrase | 26 | 0.423 | 0.769 | 0.578 | 0.648 |
| scanned | 5 | 0.800 | 1.000 | 0.900 | 0.936 |
| table | 3 | 0.667 | 1.000 | 0.833 | 0.877 |

With so few questions per row, only the overall row and the lexical/paraphrase contrast
mean much: the 95% bootstrap interval for overall hit@1 is 0.45 to 0.73, and for nDCG@10
0.68 to 0.85. BM25 finds questions that reuse the document's words (lexical) far more
often than paraphrased ones. That gap is what the dense and hybrid retrievers have to
close, and each is adopted only if a paired test on dev says it helps.

Follow-up questions (14 on dev, such as "What about at AAL3?"), searched three ways:

| Follow-up searched | hit@1 | nDCG@10 |
| --- | --- | --- |
| as typed | 0.429 | 0.620 |
| history joined | 0.500 | 0.754 |
| hand-written rewrite | 0.857 | 0.933 |

**Who wrote the evaluation data.** The questions, conversations and page transcriptions
were drafted by an AI model (Claude) working as the project's lead developer, and are
being checked by hand by the author. The questions were written from askpdf's extracted
text, so they lean towards passages the extractor reads well; self-written questions also
favour lexical retrieval. Both are reasons the blind-written Qasper questions are reported
next to them. The corpus files and their licences are listed in `eval/corpus/SOURCES.md`.

## Known limitations (so far)

- No OCR: scanned pages without a text layer are detected and flagged, not read.
- Text is read with pypdf; tables and two-column layouts may come out in an odd order.
  Extractors are compared on hand-transcribed pages before v0.1.
- PDFs that need a password to open are skipped with the reason shown. PDFs that open
  with an empty password (RC4 or AES) are read, including ones marked "no copying" or "no
  printing": askpdf reads your own files locally and does not enforce those advisory
  permission flags.
- A word broken across lines at a hyphen is rejoined without the hyphen only if the same
  document spells it solid more often than hyphenated; otherwise the hyphen stays. So
  "multi-factor" stays a compound, but a word broken by layout that the document uses only
  once keeps a stray hyphen ("algo-rithm"). How often that happens is measured before v0.1.
  Header and footer lines are never joined to the text, but they are only recognised in
  documents of 4 pages or more; in a shorter one, a page number under a hyphenated last
  line can end up glued on ("multi-3").
- Keyword search splits words at spaces and punctuation, with no stemming and an English
  stopword list. Languages written without spaces between words (Thai, Chinese, Japanese)
  are matched only as whole runs of text, and "authenticate" does not find "authenticator".
- Zero-width joiners are removed while cleaning, which can change words in scripts such
  as Persian or Devanagari. The test corpus is English and Swedish.

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
