# ADR 0001 — Document fidelity and optional derivatives

- Date: 2026-10-02
- Decision status: accepted
- Implementation status: partly applied by the first increment I1 of the new pipeline ([ADR 0009](0009-first-usable-product.md)): source, original extraction and every attempt kept, quality `unverified`; derivatives not implemented
- Origin: goal stated by the user and accepted general approach

## Context and decision

The main result is a complete Markdown document, accompanied by the assets and by a conversion manifest. Preserve text, mathematics, tables, images, captions, notes and references, respecting the logical structure and the reading order. The original multi-column layout must not be reproduced as columns in the text.

Keep the source and represent blocks, hierarchies and provenance internally. The exact intermediate representation will be chosen after the engine comparison (at the time of the decision; I1 keeps the Marker Markdown and assets and checks their references with `markdown-it-py`, without settling this choice). OCR/AI corrections produce traceable revisions; uncertainties stay visible. Do not promise universal identity obtained automatically from ambiguous scans.

Chunking and RAG preparation are optional derivatives, disabled in the base path. The Markdown dialect and the export rules will be formalized with fixtures before declaring interoperability with other renderers or formats.

## Alternatives and consequences

A pipeline based only on Markdown strings is simpler, but it loses information needed for review, reading order and provenance. The current global cleaning is not a reusable model of conservative correction.

The checks must measure omissions, alterations, formulas, references and assets; syntactically valid Markdown or renderable formulas do not prove semantic fidelity. The reconstruction of charts or diagrams remains a derivative distinct from the original image.

References: draft (archived: `documentation/architecture/preliminary-draft.md`, tag `pre-docs-2026-10-18`), [roadmap](../roadmap.md).
