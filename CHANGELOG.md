# Changelog

## Unreleased

- Closing the browser tab mid-answer no longer leaves askpdf refusing every later
  question ("another question is being answered") until it is restarted, and Ollama now
  stops generating within a token or two instead of finishing the answer for no one.
- `askpdf eval answers` and `askpdf bench` record the date, the machine (`--machine` to
  label it) and the git commit, marked `-dirty` when tracked files had changed. The README
  tables take the date and machine from the results file instead of constants.
  `eval --check` ignores these three fields and still compares every score.
- pypdf is capped below 7. pypdf 7.0 changes how some pages are extracted, which moves
  retrieval rankings away from the published results. The cap stays until the evaluation
  is re-run on it.
- A citation link can no longer open a different PDF: file ids are never reused after a
  delete (index schema 3; an older index is rebuilt on first use).
- A PDF that grows while being served is cut at the length announced in Content-Length.
- A transaction whose COMMIT fails is rolled back instead of left open.
- The release job installs pinned build tools (`.github/release-constraints.txt`).

## v0.1.1 (2026-11-04)

Fixes from the v0.1.0 code and design reviews. Search, answers and every test-split
number are unchanged; the model-free stages were scored again and match v0.1.0 exactly.

- Questions such as "On which page did NIST define MFA?" are searched again; "page did"
  was being read as a Roman page number. A capital "I" after "page" is the pronoun.
- Reading the model's reply takes linear time; a long run of punctuation lines took 31 s.
- Remote models: `serve` names every host that receives questions or document text, and
  the page keeps a banner naming them. Bring-your-own-key replies are capped at 768
  tokens, and error messages never show a password in a URL.
- Index creation no longer races: a creator could delete another's live write-ahead log,
  and a server starting mid-creation could think the index was busy.
- A CI check that ordinary questions never skip search for a named page.
- README claims trimmed to what the results show.

## v0.1.0 (2026-11-03)

First release: local web app and CLI over a folder of PDFs, with quote-checked citations,
folder watching, and a stage-wise evaluation scored once on a frozen test split.
