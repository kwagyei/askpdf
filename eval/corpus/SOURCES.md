# Evaluation corpus: sources and licences

Every file here is unmodified unless the entry says how it was derived. Checksums are in
`SHA256SUMS`; `tests/test_eval_corpus.py` fails if a file changes. Retrieved 2026-10-21.
None of the files is encrypted.

## nist-sp-800-63b-4.pdf
- **Document:** NIST Special Publication 800-63B-4, *Digital Identity Guidelines:
  Authentication and Authenticator Management* (July 2025). National Institute of
  Standards and Technology.
- **URL:** https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-63B-4.pdf
- **Licence:** the publication states (PDF page 3): "This publication may be used by
  nongovernmental organizations on a voluntary basis and is not subject to copyright in
  the United States. Attribution would, however, be appreciated by NIST." Its copyright
  status outside the United States may differ. Attribution: National Institute of
  Standards and Technology, NIST SP 800-63B-4.
- **Why it is here:** long (129 pages), roman-numbered front matter, a four-line running
  header, normative lists, and line breaks inside hyphenated compounds.

## qasper-2105.03011v1.pdf
- **Document:** Pradeep Dasigi, Kyle Lo, Iz Beltagy, Arman Cohan, Noah A. Smith, Matt
  Gardner. *A Dataset of Information-Seeking Questions and Answers Anchored in Research
  Papers.* NAACL 2021. arXiv:2105.03011v1.
- **URL:** https://arxiv.org/pdf/2105.03011v1 (abstract page: https://arxiv.org/abs/2105.03011)
- **Licence:** Creative Commons Attribution 4.0 International (CC BY 4.0), as shown on the
  arXiv abstract page (https://creativecommons.org/licenses/by/4.0/). © the authors.
- **Why it is here:** two columns, tables and footnotes.

## naca-rm-l56l14a.pdf
- **Document:** NACA Research Memorandum RM L56L14a, *Investigation of transonic flutter
  characteristics of a thin 10° sweptback wing having an aspect ratio of 4 and a taper
  ratio of 0.6* (1957). National Advisory Committee for Aeronautics.
- **URL:** https://ntrs.nasa.gov/api/citations/19930089681/downloads/19930089681.pdf
  (record: https://ntrs.nasa.gov/citations/19930089681)
- **Licence:** the NASA Technical Reports Server record
  (https://ntrs.nasa.gov/api/citations/19930089681) gives the copyright determination
  `GOV_PUBLIC_USE_PERMITTED`, no third-party material flagged, and distribution `PUBLIC`.
- **Why it is here:** a real scan with an OCR text layer of uneven quality (for example
  "NPCA ML56LI1fa" for "NACA RM L56L14a").

## nist-image-only.pdf
- **Derived from** `nist-sp-800-63b-4.pdf`: physical pages 40 and 46 rendered at 100 dpi
  in greyscale with pypdfium2 and wrapped as an image-only PDF with Pillow, by
  `eval/tools/make_image_only.py` (reproducible byte for byte).
- **Licence:** as the source (a US Government work).
- **Why it is here:** pages certain to have no text layer, the ground truth for
  scanned-page detection.

## Files derived from the corpus elsewhere in `eval/`
- `eval/pages/*.png` are renderings of corpus pages (pypdfium2, 110 dpi, greyscale), and
  `eval/transcripts/*.txt` are hand transcriptions of the same pages. Both keep the licence
  of the document they come from, as listed above. Those from `qasper-2105.03011v1.pdf` are
  adaptations of a CC BY 4.0 work: © the paper's authors, changed by rendering or
  transcribing single pages. They are not covered by askpdf's MIT licence.
- `eval/questions/` and `eval/conversations/` quote short passages from the corpus as
  evidence; the same attributions apply.

## How the licences were checked
- NIST: the notice on PDF page 3, quoted above (re-read offline by the reviewer).
- Qasper paper: the arXiv abstract page (https://arxiv.org/abs/2105.03011) shows the CC BY
  4.0 link; the PDF itself carries no licence. Read on 2026-10-21 and 2026-10-27.
- NACA: the NTRS API record (https://ntrs.nasa.gov/api/citations/19930089681) has
  `copyright.determinationType` `GOV_PUBLIC_USE_PERMITTED`; the scan has no notice. Read on
  2026-10-21 and 2026-10-27.

## Not yet included
- A Statistics Sweden (SCB) publication in Swedish, planned in ADR-0011. SCB's web
  firewall rejected an automated download; it will be added by hand.
