# askpdf

Local web app for asking questions about a folder of PDFs, with page-linked, quote-verified
citations and a stage-wise evaluation. Public repo: github.com/kwagyei/askpdf.

Project notes: ~/Vault/Projects/code-projects/
Design and ADRs: ~/Vault/Projects/code-projects/architect/askpdf-design.md and askpdf-ADR-*.

## Stack

- Python `>=3.12` (numpy 2.5 needs it); CI on ubuntu × 3.12/3.13/3.14, macOS and Windows on 3.13
- src layout (`src/askpdf/`), hatchling, MIT; installed via `pipx install git+…`, not on PyPI
- Runtime deps only: pypdf, numpy, fastapi, uvicorn, pycryptodome (pypdf's AES backend;
  `cryptography` has no Intel-Mac wheels). Everything else is stdlib.
- Dependency groups (pip >= 25.1): `dev` (pytest, hypothesis, ruff, mypy, httpx,
  pytrec-eval-terrier) and `eval` (jiwer, pdfplumber, pypdfium2, rapidfuzz)
- Models via Ollama's native API (llama3.2:3b, nomic-embed-text); optional BYOK chat only

## Commands

```sh
python3.13 -m venv .venv && .venv/bin/pip install --upgrade pip
.venv/bin/pip install --group dev && .venv/bin/pip install .   # reinstall after src changes
.venv/bin/pytest --cov          # coverage floor 90%, branch coverage
.venv/bin/ruff check && .venv/bin/ruff format --check && .venv/bin/mypy
```

Tests run against the installed package, not the checkout. Reinstall with `pip install .`
after changing `src/`.

## Conventions

- Type hints everywhere; mypy strict covers `src` and `tests`.
- No test reaches the network: `tests/conftest.py` patches `connect`, `connect_ex`,
  `sendto`, `sendmsg` and `getaddrinfo` in `pytest_configure`, so imports, collection and
  every fixture scope are covered. Loopback stays open; `@pytest.mark.network` lifts it
  for one test. Models, arXiv and remote APIs are faked.
- Hypothesis loads the derandomised `ci` profile (no deadline) when `CI` is 1/true;
  `HYPOTHESIS_PROFILE` overrides it.
- `cli.py` prints; core modules return data or yield events.

## Settled decisions (see the ADRs for the reasoning)

- Physical page number (1-based) is canonical; chunks never cross a page.
- Line-end hyphens are resolved per document (`clean_document`): removed only when the
  document spells the word solid more often. Bump `CLEANING_VERSION` whenever cleaning
  output changes; it is pinned in the index, so old indexes rebuild.
- One SQLite index per watched folder, stored outside that folder; files keyed by SHA-256.
- Citations carry a quote that must appear verbatim on the cited page.
- All tuning on the dev split; the test split is frozen and scored once per release.
- Remote endpoints need `--allow-remote` or `ASKPDF_ALLOW_REMOTE=1`; the key comes only from
  `ASKPDF_API_KEY`.
- Actions are pinned by SHA (version in a comment); no Dependabot (its commits would land on
  `main`). Workflows never commit; result files change only in human PRs.
