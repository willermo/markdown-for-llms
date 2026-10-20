# Benchmarks — conclusions and capability matrix

This page keeps the benchmark conclusions that support the current decisions ([ADR 0008](../decisions/0008-maintained-parser-and-provisional-benchmark.md), [ADR 0009](../decisions/0009-first-usable-product.md)) and the capability matrix of the converter. Raw outputs and run evidence stay in `temp/<run>/` and are not versioned; the full dated reports of runs a002 and a003 are archived at tag `pre-docs-2026-10-18` (see [Archived reports](#archived-reports)).

The results below are observations on a small synthetic sample, not certifications. Human judgment of the a003 outputs is still pending, and the a002/a003 measuring tools are recognized technical debt ([register](../development/technical-debt.md)).

## Conclusions

- **a002 — corpus and fidelity meter (closed 2026-10-12).** Closed by the user's acceptance of the reduced scope, without a retroactive technical GO: the experimental reader r014 stays as it is, the five debt items D-A002-01–05 remain open ([a002 debt register](a002-residual-debt.md)). The corpus, its gold files and the raw outputs with assets are usable for inspection and exploration; metrics and judgments of the unvalidated classes are provisional and do not support a definitive ranking or safety guarantees.
- **a003 — engine comparison (rounds r002, 2026-10-13, and r007, 2026-10-15).** All 17 current conversions had operational SUCCESS with isolation PASS, but **no PDF profile passes the complete gate**. On this sample Marker preserves native prose better than Docling, while both lose mathematical structure (fractions, sum limits, matrices) and alter characters or signs; Docling used less time and memory in the measured profile, with formula enrichment disabled. TXT and DOCX are preserved; EPUB loses a note and its backlink. The r007 re-qualification left these observations unchanged; its r007 review was arbitrated NO_GO, so the measures of the sample are only partially corroborated and the general qualifier is not validated.
- **Consequence for the product.** These observations motivated Marker 1.10.2 on CPU for the first increment I1 (ADR 0009), with quality reported as `unverified` and no promise of preserved mathematics. They are not a choice of a definitive engine and not a measure of maximum capability, other formats, other hosts or throughput.

## Method and limits

- Corpus: ten synthetic canonical cases with manual gold files, under `tests/fixtures/conversion-corpus/` (provenance and licences in its `NOTICE`); gold files stay PROVISIONAL until measured and humanly verified on the current hashes. The a003 sample uses seven of them: C01 (TXT, Italian), C02 (DOCX, Italian), C03 (EPUB, English), C06 (native PDF, Italian), C07 (two-column PDF, English), C08 (scanned PDF, Italian), C10 (mixed native and raster PDF, Italian and English).
- Engines and versions (a003 r002): Marker 1.10.2 with Surya 0.17.1 and torch 2.7.1+cpu (default checkpoints, `force_ocr` and `use_llm` off); Docling-slim 2.138.0 with torch 2.7.1+cpu, Heron and TableFormer, Tesseract 5.3.4 OCR (Italian and English), formula enrichment off; Pandoc 3.1.3 for DOCX and EPUB, direct and through the legacy wrapper. CPU only, in a confined runner without network.
- Measures: CER and WER summed over distances and denominators against the gold (no per-block averages), numbers, table cells, formulas and assets compared separately; no Unicode, sign, number or LaTeX normalization to obtain a PASS. Content observations of formulas and figures were made by the AI author; formal equivalence and the user's judgment were not measured.
- Resources for a whole PDF (consumer interval, including startup and models), Marker / Docling: C06 97.69 s, 7.817 GiB / 22.17 s, 1.271 GiB; C07 98.91 s, 7.819 GiB / 19.86 s, 1.199 GiB; C08 127.88 s, 8.145 GiB / 20.88 s, 1.212 GiB; C10 219.45 s, 8.242 GiB / 26.56 s, 1.438 GiB. RSS is sampled, not a hard upper bound; the operating system cache was not controlled and there are no statistics over repetitions.

## Capability matrix

Status values: *observed on sample* (measured on the cases cited, not certified) and *not verified* (no measurement). Update this matrix with what a run actually measures (skill `verify-conversion-fidelity`).

| Format and path | What holds | What is lost | Evidence (date, case) | Status |
| --- | --- | --- | --- | --- |
| TXT, literal path | Exact payload, CER/WER 0, numbers, Unicode, code and URIs intact | Nothing observed | a003 r002, 2026-10-13, C01 | observed on sample |
| DOCX, Pandoc 3.1.3 (direct and legacy wrapper) | Prose CER/WER 0, two formulas, nine cells, 13 numbers, PNG figure byte-identical, three references | Nothing observed | a003 r002, 2026-10-13, C02 | observed on sample |
| EPUB, Pandoc 3.1.3 (direct and legacy wrapper) | Two formulas, nine cells, figure byte-identical, remaining text | Note b11 and its backlink; CER 4.334%, WER 4.082%; numbers 11/13 | a003 r002, 2026-10-13, C03 | observed on sample |
| RTF, LaTeX bundle | — | — | none | not verified |
| Native PDF, Marker 1.10.2 CPU | Prose CER/WER 0 on both pages; table values kept | Structure of fractions, sum limits and matrix; tables gain a header row | a003 r002, 2026-10-13, C06, C07 | observed on sample |
| Native PDF, Docling 2.138.0 | Prose of p2; matrix and fraction readable as PNG; 8/9 cells exact | Minus sign and caption dash changed; formulas not decoded or limits lost; numbers 9/12 | a003 r002, 2026-10-13, C06, C07 | observed on sample |
| Scanned PDF, Marker 1.10.2 CPU | Most of the prose | CER 1.059% (over 1%), WER 9.231% (over 2%); numeric signs altered, fraction missing, sum deteriorated | a003 r002, 2026-10-13, C08 | observed on sample |
| Scanned PDF, Docling 2.138.0 | Most of the prose | CER 1.907%, WER 13.846%; URI and note marker altered, reference and note fragments interleaved; formulas unavailable; numbers 7/9 | a003 r002, 2026-10-13, C08 | observed on sample |
| Mixed PDF, Marker 1.10.2 CPU | Native Italian pages CER/WER 0; three charts present | Raster English page CER 0.333%, WER 2.198%; four prose units of the raster Italian page missing; matrix of p2 missing; numbers 25/29 | a003 r002, 2026-10-13, C10 | observed on sample |
| Mixed PDF, Docling 2.138.0 | Matrix of p2 readable as PNG | Errors on every page (raster English CER 5.167%, WER 7.692%); twelve cells missing; numbers 20/29; English caption reordered | a003 r002, 2026-10-13, C10 | observed on sample |
| Product I1: PDF with Marker 1.10.2 CPU | Complete local path (CLI, API, Web UI), source and original extraction kept, bundle states and ZIP download | As above for Marker; first use (P14) on two documents: reading order in two-column layouts, inline mathematics often left as plain text with subscripts turned into superscripts, about 40 to 140 s per page on CPU, the fixed timeout stopping longer documents, and a false missing reference that marked one bundle `incomplete` and blocked its ZIP | I1 delivery, 2026-10-17, C06, C07; first use, 2026-10-18, two PDFs of the user | observed on sample |
| GPU, Windows, macOS, remote providers | — | — | none | not verified |

## Archived reports

The dated reports are available at tag `pre-docs-2026-10-18`:

```bash
git ls-tree -r --name-only pre-docs-2026-10-18 -- documentation/benchmarks/
git show pre-docs-2026-10-18:<path>
```

The sources of this page are `documentation/benchmarks/a003-engine-comparison-r002.md` (archived, tag `pre-docs-2026-10-18`), section «Risultati per formato», and `documentation/benchmarks/a003-engine-comparison-r007.md` (archived, tag `pre-docs-2026-10-18`), together with the [a002 debt register](a002-residual-debt.md). New conclusions go here (method, versions, hardware, results, limits, date), with a dated file `documentation/benchmarks/YYYY-MM-DD-<topic>.md` only when they are too long for this page.
