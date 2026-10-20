# ADR 0008 — Maintained parser and provisional benchmark

- Date: 2026-10-12
- Decision status: **accepted** as direction and as condition of the reduced delivery
- Implementation status: the maintained parser adapter is not implemented. Since the disposition of 2026-10-16 the usable product comes first: the first increment I1 was delivered according to [ADR 0009](0009-first-usable-product.md) and integrated into `dev`, without the advanced validator. Advanced validation, the trace r005 (delivered, not reviewed) and the a002 debt remain open; no retroactive technical GO.
- Origin: explicit acceptance by the user, run a002, after 14 author revisions
- Supersedes: no ADR; complements ADR 0001 and applies a local exception to ADR 0007

## Current disposition — product before advanced validation, 2026-10-16

The user approves moving towards the working converter and the roadmap features, Web UI included, accepting their immaturity and the absence of a complete automatic certification of contents and assets. The resolution of the parser/trace and of the evaluator debts is deferred and can be spread over time after the product is usable. No new automatic review/fix cycle of the trace r005 is required now.

This disposition **replaces the time constraints** of the mandatory resumption at the exit of phase 3 and of closing the parser before any experimental delivery or release. The original dispositions below remain history, not the current gate. It does not cancel D-A002-01–05, does not promote the previous NO_GO results and does not transfer the GO of the r002 core to the roles or to the adapters. R005 remains an author delivery, not reviewed; no experimental fork is adopted in the product.

The roadmap is carried out **by usable increments**, starting from a real PDF→Markdown path with assets, preservation of the source and of the original output, and CLI and Web UI on the same service. The first increment includes the foundations it needs, not the whole future architecture before a visible conversion. The later features (formats, review, distribution and export) remain authorized goals: dependencies, hardware and evidence determine their sequence. Optional derivatives stay optional; no paid service, remote upload or public deploy becomes implicitly authorized.

The absence of semantic certification does not remove the operational checks: visible engine errors, delivered files and assets identified, missing local references reported, confined writes, distinct executions and consistent downloads. A finished job does not attest fidelity. Unverified quality is kept apart from the integrity and completeness of the bundle; a known technical incompleteness is not hidden as a successful conversion. The preview of imported content is inert or confined; the engine output alone never authorizes the execution of untrusted HTML or assets.

Manual outputs and comparisons with declared limits are allowed, without new destructive normalizations, rewritten gold files or the COMPLETE/UNSAFE labels of the experimental meter as a product barrier. Value is verified on source, Markdown and assets that can really be consulted, not on test counts. The debt is reviewed when a feature needs advanced attestations, or at a roadmap check after the usable product; no artificial deadline at the end of phase 3. A promise of automatic certification or fidelity will still require the relevant evidence.

Expected benefits: concrete use, feedback on the documents and less dependence on the verifier. Accepted costs: manual verification, engine errors not always detected and possible future adaptations of the contracts. To limit them, keep raw outputs, assets, versions, configurations and revisions; keep the optional validator separate from the conversion services. New product preparation in `temp/run-a004-fase2-prodotto-utilizzabile`; plans and reviews of the application code remain necessary, Git stays manual. Decision status approved, product not yet built (status at the time of the disposition, 2026-10-16; the first increment I1 was then delivered, see ADR 0009).

### Recognized debt and authorized cleanup — user clarification, 2026-10-16

The inconclusive work on reader/comparator, parser/trace and validation is **recognized technical debt, to be consolidated as soon as we have the means, time and resources**. The priority given to the product postpones the work and makes its scheduling manageable; it is neither abandonment nor resolution of the IDs. The [cross-cutting register](../development/technical-debt.md) keeps status, responsibility and resumption criteria, together with the a002 register.

The user also authorizes the selective cleanup of `temp` material that is no longer needed. Rebuildable Haskell environments and builds were removed after checking the dependencies: 8,777,867,264 allocated bytes freed; sources, patches, reports, raw/gold, results, final binaries and engine models and environments kept. Old snapshots are not a promise that an environment is still present. Availability is described in the a003 summary `documentation/runs/run-a003-fase1-engine-comparison/manifest.md` (archived, tag `pre-docs-2026-10-18`); no change of results to obtain MATCH and no Git rewrite.

## Context

The benchmark reader progressively rebuilt Markdown syntax, metadata, code and HTML with decisions that were not always consistent. Local fixes closed real cases but left classes unvalidated. R014: Claude NO_GO (static), OpenAI review incomplete because of administrative cost: no final positive technical arbitration. The user accepts the experimental delivery and asks for the problem to be taken up again and solved, giving priority to a product that can be evaluated.

## Decision

Adopt a maintained parser for the syntax of the Markdown profile, with reproducible version, extensions and configuration; adapt the AST to the observed model instead of continuing to expand the project's own grammar. Pandoc, already local at 3.1.3 and available through the JSON AST in the a002 experiments, is the initial candidate chosen for the verification of the adapter. No new dependency or installation is made here; suitability for coordinates/provenance and contents has to be demonstrated. If the AST does not offer enough data, the contract must declare UNAVAILABLE for those measures or evaluate a maintained parser with an adequate source map; do not invent coordinates or reproduce the whole grammar to fill the gap.

Separate syntactic parsing, fidelity measures and confinement/resource checks. The converter's AST is not an independent oracle of the quality of its own conversion: gold files, sources and human evaluation remain necessary. No COMPLETE from the experimental reader authorizes active rendering or access.

Close a002 with **user acceptance of the reduced scope**, without a retroactive technical GO. Double reviews remain for future implementations. Reopen the debt obligatorily after the first application that can be evaluated (exit of phase 3), resolve it before attestation automations and before the release (timing replaced by the disposition of 2026-10-16 above). It does not block the exploratory phase 1 comparison carried out with limits and human review; it does not authorize skipping the roadmap dependencies.

## Alternatives considered

- Keep fixing the reader locally: partial results proven, cost of the rounds and shared assumptions not acceptable as the main strategy.
- Remove every verification: it would prevent reasoned comparisons and the diagnosis of losses.
- Use only the engine's AST as gold: it would confuse syntax with correctness.
- Maintained parser plus independent references: accepted direction; source map, profile and performance to be validated in a separate piece of work.

## Consequences and verification

Keep raw outputs, assets, references, revisions and errors. Classify separately reproduced defects, static traces and AST hypotheses. Close the debt only with evidence on originals and variants of lists, divs, fences, metadata, formulas, HTML and resources; source map verified where declared, no resource silently omitted, positive checks of inert code and values and negative checks of real HTML. Rerun on the corpus and on the invalidated tests, classifying every delta against the historical data; two new independent reviews and an arbitration on the adapter. No guarantee of perfection on arbitrary Markdown and no amnesty for the old FAIL results.

Experiment budgets must be sized to the expected work; quality is not judged by CPU consumption. Avoid repeated lookups of the inventories: index once by path/hash, distinguish new evidence from receipts, do not rerun the whole history for every probe. Real resource and isolation limits remain explicit, no quota added here.

## History of the debt work (2026-10-15/16)

The dated sections that recorded this work are condensed here; the full text is at tag `pre-docs-2026-10-18` (`git show pre-docs-2026-10-18:documentation/decisions/0008-maintained-parser-and-provisional-benchmark.md`).

- **2026-10-15, a003 review.** After seven deliveries and two NO_GO results on r007, the arbitration confirmed the mechanism described in the context (syntactic estimates promoted to evidence, roles dependent on the annotation, duplicated blocking rules); the 23 projections and 17 assessments of the sample stay corroborated, without attesting the general correctness of the qualifier. A feasibility check of the a003 comparator method was brought forward; Pandoc remained the initial candidate, its source-map capabilities not validated. The user then also brought forward the a002 debt work (D-A002-01–05), with distinct closure criteria; the debt stays open until implementation, new evidence, two independent reviews and an explicit arbitration per ID.
- **2026-10-15/16, feasibility r002.** The plan declared NOT_VIABLE for a003 and a002 on the three profiles and interfaces actually tried, which is not a general impossibility. The arbitration accepted the circumscribed negative diagnosis; no fork, dialect migration or runtime was chosen.
- **2026-10-16, provenance experiment and trace prototype.** Errata E01–E06 were accepted, and a local experiment instrumenting the Pandoc 3.1.3 reader was authorized in `temp`, explicitly **not** a permanent fork. The prototype (7 files of local delta, AST unchanged on the 34 core cases and on 381 extended raw outputs) received two NO_GO reviews on the r001 core; after the fixes the r002 core obtained two GO reviews and an arbitration, still without permanent adoption and with an explicit low limit (some phases remain attestations of the producer, not autonomous semantic evidence of the verifier without expected values), and an experimental continuation on the essential roles was authorized.
- **2026-10-16, r004 and r005.** The r004 completion also introduced two observational changes in the pinned TagSoup; the r004 reviews were arbitrated NO_GO, and the fix r005 was delivered by the author and **not reviewed**.

Outcome: the r002 core GO is circumscribed, the trace r005 is not reviewed, no fork is adopted in the product, and the a002 debt remains open (see the [a002 debt register](../benchmarks/a002-residual-debt.md) and the [technical debt register](../development/technical-debt.md)).

## Links

- [Fidelity](0001-document-fidelity.md), [supervised runs](0007-supervised-development-runs.md) (superseded by [ADR 0010](0010-multi-model-agent-governance.md)).
- a002 closure (archived: `documentation/runs/run-a002-fase1-corpus/manifest.md`, tag `pre-docs-2026-10-18`).
- [Debt register and criteria](../benchmarks/a002-residual-debt.md).
- [Roadmap](../roadmap.md), next increment of the time (archived: `documentation/development/next-run-a003-engine-comparison.md`, tag `pre-docs-2026-10-18`); benchmark conclusions in [`documentation/benchmarks/README.md`](../benchmarks/README.md).
- [Pandoc: reader→AST→writer and JSON](https://pandoc.org/filters.html), consulted 2026-10-12: it documents the capabilities of the interface, it does not prove the complete equivalence of r014.
