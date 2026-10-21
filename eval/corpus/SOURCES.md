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
  the United States." Its copyright status outside the United States may differ.
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

## Not yet included
- A Statistics Sweden (SCB) publication in Swedish, planned in ADR-0011. SCB's web
  firewall rejected an automated download; it will be added by hand.
