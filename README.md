# askpdf

[![CI](https://github.com/kwagyei/askpdf/actions/workflows/ci.yml/badge.svg)](https://github.com/kwagyei/askpdf/actions/workflows/ci.yml)

Point askpdf at a folder of PDFs and ask questions about them in your browser. Every answer
cites the page it came from, and each quoted passage is checked word for word against that
page. It runs locally with [Ollama](https://ollama.com); nothing leaves your machine unless
you opt in to a remote model.

> **Work in progress.** The design is settled and the code arrives through pull requests.
> Nothing below the scaffold works yet.

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
