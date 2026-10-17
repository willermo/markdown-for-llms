---
name: verify-conversion-fidelity
description: "Verifies content preservation when parsers, OCR, cleaning, formulas, images, reading order or the Markdown serializer of the converter change, and when engines are compared. Use for conversion regressions and engine benchmarks; not for editorial-only changes or performance comparisons without content checks."
---

# Verify conversion fidelity

1. Read [ADR 0001](../../../documentation/decisions/0001-document-fidelity.md), the relevant phase of the [roadmap](../../../documentation/roadmap.md) and the contracts actually implemented. Do not assume the planned architecture already exists.
2. Select the smallest sample that exposes the risk: for example LaTeX indices, a two-column page, a table or images with captions. Use synthetic or redistributable fixtures with recorded provenance; never add private documents to git.
3. Compare the source, the intermediate representation when available, and the real output. Check text and Unicode, omissions, numbers, block order, formulas, notes, cells and relevant assets. A present file, a validator score or renderable LaTeX does not prove content correctness.
4. Reproduce the defect before the change when possible and show the result after it. For state and storage include the relevant case: same name, re-run, incomplete output or interruption. Do not grow the suite indiscriminately when it does not concern the change.
5. Do not normalize significant symbols or spaces with regexes just to make results match. Report observed errors and limits, including unresolved ones.

## Engine comparisons and benchmarks

- Define the formats and phenomena to compare (columns, formulas, languages, scan quality, figures, tables) and the reference transcriptions.
- Record provenance, licence and hash of the samples; pin engine, weights, dependencies, configuration, hardware and the actual commands. Separate preparation or download from conversion, and cold from warm cache. Separate mocks from real runs.
- Measure text, order, formulas, tables, assets, time, memory and cost; keep failures and negative results, never only the successful examples. Distinguish typographic, syntactic and semantic equivalence, and state where a metric is not reliable.
- No paid remote benchmarks or heavy downloads unless explicitly authorized. Raw outputs stay in `temp/<run>/`; only conclusions that support a decision go into `documentation/benchmarks/` and the relevant ADR. Never create placeholder results that look like executed benchmarks.
- A major engine or model update needs a new comparison on the affected cases, not an assumption of equivalence.

Deliver before/after evidence and the relevant checks. When the reference source, the model or the hardware is missing, limit the conclusion to the checks actually performed.
