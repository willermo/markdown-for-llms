---
name: write-diataxis-docs
description: "Writes or reorganizes the README and the user and developer documentation in docs/ following Diátaxis. Use for tutorials, how-to guides, reference and explanation pages; for architecture choices use the ADR register in documentation/decisions/ instead."
---

# Write documentation with Diátaxis

1. Read the [docs index](../../../docs/README.md), the page concerned and the behaviour of the relevant code, schema and tests. Separate the legacy pipeline from the new product.
2. Identify the reader's main need and choose one form:
   - `docs/tutorials/`: guided learning, prerequisites, a sample and the expected result;
   - `docs/how-to/`: a concrete task, the essential steps and a final check;
   - `docs/reference/`: exact options, schemas, defaults, constraints and compatibility;
   - `docs/explanation/`: concepts and reasons, linking the relevant ADRs.
3. Split different needs into linked pages when one page would become confusing. State audience and version when relevant; do not create duplicate trees for users and developers with the same content.
4. Check commands and examples on the available product, or state that the check is missing. Do not invent APIs, `.env` variables or installation procedures for features that are only planned. Use fictitious credentials in examples and never private documents as tutorial material.
5. Update navigation and links. The README introduces the project and points to the guides; it must not become a second copy of the reference. For editorial-only changes do not start OCR benchmarks or model tests.
6. Check relative links, the section index and `git diff --check`, including new untracked files.

Write in English. Deliver pages in the right place and state which examples were verified. For the method see the [Diátaxis source](https://diataxis.fr/); the four types are reader needs, not a requirement of four empty folders.
