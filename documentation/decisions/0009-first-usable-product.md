# ADR 0009 — First local product: Marker CPU and inert interface

- Date: 2026-10-17
- Decision status: accepted for the first experimental increment
- Implementation status: I1 implemented (technical GO on round r004 after two independent reviews and an arbitration) and integrated into `dev` at `f8bf44c` from the feature commit `323f172` on 2026-10-17, push verified; not released on `main`. Quality `unverified`; three low residuals CLA-R4-I001…I003 in the [technical debt register](../development/technical-debt.md). First use with the user's feedback (P14) carried out on 2026-10-18 on two PDFs; its findings feed the next increments and do not certify fidelity.
- Origin: user priority of 16 October, a004 plan r001 and arbitration of the independent ChatGPT/Claude reviews, with verified local errata
- Supersedes / superseded by: none; refines ADR 0002/0003/0004/0008

## Context

A complete PDF→Markdown/assets path, with a usable CLI and browser, is needed before the consolidation of the validator. The observed a003 outputs show usefulness for prose, table and figure of C06, together with mathematical losses; they do not prove general fidelity, reliable scientific OCR or all the pages of mixed PDFs. The [recognized debt](../development/technical-debt.md) remains open and does not become a dependency of the first product.

## Decision

For I1, adopt Marker 1.10.2, Surya 0.17.1 and torch 2.7.1+cpu behind a direct adapter, in a confined child of the separate worker. First scope to prove: Linux x86_64, CPU, single user, loopback, no remote upload. Pins and stored models need a targeted check before use. Do not redistribute models automatically and do not promise other platforms.

FastAPI serves the API and an HTML/CSS/JavaScript UI without SPA framework or CDN. Application services shared with the CLI; SQLite with claim/lease and explicit recovery, local filesystem with publication and reconciliation. The browser shows Markdown as text, a verified raster gallery and downloads; scientific rendering and advanced visual comparison will come in later increments.

Use `markdown-it-py` for the operational check of the references of the profile, with version, configuration and needed plugins frozen in the implementation. The choice eases the Python integration without adding the Pandoc fork. It does not provide the observed model, the source map or the guarantees of the future validator of ADR 0008; no dialect equivalence is declared. Known HTML is read with the existing tokenizer, never executed or reserialized.

Distinguish execution, bundle state and `unverified` quality. `ready` checks inventory/hashes and the references in the profile. `unchecked` allows an ordinary download with label and limits when all the declared files are intact and there are no known faults; a known missing asset, tampering or an unsafe path keep blocking it. Internal anchors are not missing files; HTML without resources does not by itself make the result `unchecked`. These states do not certify the completeness of the assets with respect to the PDF.

Validate the images for the gallery in a separate process that does not import Marker, with explicit limits before decoding: the pin disables Pillow's global limit in its own process. Configuration, schema and storage identity are checked by API, worker and CLI. I1 does not include deletion from the archive: the limit must be documented and its management planned later.

## Alternatives considered

Docling remains an explicit future fallback: the samples do not show a universal advantage for formulas and signs. The legacy server does not satisfy the new worker/image contract. Pandoc does not convert PDF; the experimental trace is not approved as a dependency. React/TypeScript remains an option when comparison and review justify its cost; for I1 the text UI is enough. The advanced parser remains a separate project, with independent gold files.

## Consequences and verification

Implement the complete path and verify it from a wheel outside the checkout, with the real PDF C06, CLI, a real browser, external ZIP, C07, crash/restart, errors and the operational probes P01–P14. Add the normal forms of the pin: span/anchors, sup/sub, br, HTML tables and local/external HTML images. The reviews of the plan are static: no runtime evidence is transferred yet.

No promise of preserved mathematics, of unproven hardware support, of automatic validation or of a product already available (at the time of the decision; I1 has since been delivered on `dev`, still experimental and unreleased). Review the engine if the new path does not keep the observed usefulness; no destructive cleaning or cloud fallback to hide losses. Two independent reviews of the code precede the GO of the delivery; Git remains manual, by the user.

## Delivery history (2026-10-17)

The dated progress section is condensed here; the full text is at tag `pre-docs-2026-10-18` (`git show pre-docs-2026-10-18:documentation/decisions/0009-first-usable-product.md`).

I1 went through four delivery rounds on 2026-10-17, each followed by two independent reviews and an arbitration. Rounds r001, r002 and r003 were arbitrated NO_GO with local fixes (quota, temporaries and invalidation on transient errors; positive classification of corruption and durability restore in recovery; a complete result already publishing but not yet renamed), with the architectural choices unchanged; the real C06 conversion on the final monitor was proven in r003. Round r004 obtained the **technical GO** after two GO reviews and the arbitration: six r003 IDs closed and three low residuals accepted into the technical debt register without declaring them solved. The integration was verified on the same day: feature `323f172` merged into `dev` at `f8bf44c` with an identical tree, push confirmed on origin, 731 selected identities unchanged. Quality stays `unverified`; no retroactive GO to the validator and no certification of the assets with respect to the PDF.

## Links

[Roadmap](../roadmap.md), observed a003 results in [`documentation/benchmarks/README.md`](../benchmarks/README.md), [ADR 0008 in force](0008-maintained-parser-and-provisional-benchmark.md). Plan, errata and arbitration of a004 are operational context in `temp/`, not permanent bundles; the run summary is `documentation/runs/run-a004-fase2-prodotto-utilizzabile/manifest.md` (archived, tag `pre-docs-2026-10-18`). Primary sources of the pin, read as text and identified in the temporary evidence: `marker/renderers/markdown.py`, `html.py`, `schema/blocks/reference.py`, `schema/text/span.py` and `line.py` of 1.10.2. The capabilities are matched in the code and in the stored raw outputs; the execution of the new adapter was still to be proven at the time of the decision (the delivery then ran it on C06 and C07, see the history above).
