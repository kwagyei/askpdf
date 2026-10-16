# askpdf

[![CI](https://github.com/kwagyei/askpdf/actions/workflows/ci.yml/badge.svg)](https://github.com/kwagyei/askpdf/actions/workflows/ci.yml)

Point askpdf at a folder of PDFs and ask questions about them in your browser. Every answer
cites the page it came from, and each quoted passage is checked word for word against that
page. It runs locally with [Ollama](https://ollama.com); nothing leaves your machine unless
you opt in to a remote model.

> **Work in progress.** The design is settled and the code arrives through pull requests.
> What works today: reading a folder of PDFs into the local index. Search, answers and the
> web app come next.

```sh
$ askpdf index ~/Documents/papers
reading nist-sp-800-63b-4.pdf
1 added, 0 already known (copies or renames), 0 removed, 0 unchanged, 0 failed
$ askpdf docs ~/Documents/papers
nist-sp-800-63b-4.pdf  129 pages, 279 chunks, 0 image-only
```

Re-running `index` only reads new or changed files; renamed and copied files are recognised
by their content and not read again. The index lives in your cache folder, never in the
folder you point it at.

## What makes it different

The interesting part is not the chatbot but the measurement. Each stage is scored on its own,
so a bad answer can be traced to where it went wrong:

| Stage | Question it answers | Status |
| --- | --- | --- |
| Text extraction | Did we read the page correctly? (character error rate on hand-transcribed pages) | planned |
| Retrieval | Did we find the right page? (recall@k, MRR, nDCG) | planned |
| Grounding | Does the cited page actually contain the quote? | planned |
| Abstention | Does it say "not found" instead of guessing? | planned |
| Follow-ups | Are follow-up questions rewritten correctly? | planned |
| Latency | How long does it take on an 8 GB laptop without a GPU? | planned |

Tuning happens on a development split; the test split is frozen and scored once per release.
Questions written by people who had not seen the answer (Qasper) complement the in-house set.

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
- Zero-width joiners are removed while cleaning, which can change words in scripts such
  as Persian or Devanagari. The test corpus is English and Swedish.

## Development

Requires Python 3.12 or newer.

```sh
python3 -m venv .venv && .venv/bin/pip install --upgrade pip
.venv/bin/pip install --group dev && .venv/bin/pip install .
.venv/bin/pytest --cov
.venv/bin/ruff check && .venv/bin/ruff format --check && .venv/bin/mypy
```

## Licence

MIT. Evaluation documents keep their own licences, listed alongside them.
