# Page transcriptions

Ground truth for text extraction (ADR-0011): each file is one page, typed from the
rendered page image (`eval/tools/render.py`), never from askpdf's output. A model drafted
them; Kwabena Sasu-Boakye is checking them side by side with the images in `eval/pages/`.
The main README says so too.

File names are `<document>-p<physical page, 3 digits>.txt`.

## Rules
- Reading order. Two columns are read left, then right. A figure's text comes where
  the figure sits in its column, followed by its caption.
- One paragraph, heading or list item per line. Line breaks inside a paragraph become
  spaces.
- A word broken across lines by layout is rejoined (`ques-` + `tions` → `questions`); a
  compound that happens to break at its own hyphen keeps it (`sting-` + `fuselage` →
  `sting-fuselage`). A hyphen at the very end of the page stays, since the word
  continues on the next page.
- Tables row by row, cells separated by one tab, a multi-line cell joined with spaces.
  Symbol lists are written the same way (symbol, tab, definition).
- Characters as printed: curly quotes, dashes, Greek letters, `°`. Small capitals are
  written in their normal case (`Qasper`). Sub- and superscripts follow their base
  without markup (`F1`, `lb-in.2`, `θm(s)`), and a stacked fraction is written
  `numerator/denominator`.
- Running headers and footers are prefixed `[header] ` or `[footer] `; text in the
  margin (an arXiv stamp) `[margin] `. A header with a left and a right side is read
  row by row, left before right.

## Scoring
Character error rate compares each transcription with askpdf's cleaned page text after
both are reduced to single spaces (labels removed, text kept). Header and footer
detection is scored against the `[header]`/`[footer]` lines.
