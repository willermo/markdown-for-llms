# Converter roadmap

Status as of 2026-10-18. The phases describe the path to the product; they are neither a time estimate nor an authorization to start every activity in advance. Dependencies, hardware and corpus will determine the size and duration of the increments.

The first increment I1 (local PDF to Markdown with assets, CLI, API and Web UI) is available on `dev` as an **experimental product**: integrated on 2026-10-17, not released on `main`, quality `unverified` ([ADR 0009](decisions/0009-first-usable-product.md), [known limits](../docs/explanation/experimental-product.md)). Everything after I1 below is planned, not available.

## Sequencing principle

The roadmap is carried out by usable increments, starting from real conversion with Markdown and assets that can be consulted, then extending formats, review, distribution and export. Parser/trace work and advanced validation are not prerequisites of phases 2–8 for this experimental use; the debt and the old NO_GO results stay open, with no retroactive GO and no adoption of the experimental fork. The principle and its limits are in the [disposition in force of ADR 0008](decisions/0008-maintained-parser-and-provisional-benchmark.md).

Error handling, presence and identity of the artifacts, preservation of the raw output and confinement remain obligations. Finished job, bundle integrity and unverified fidelity are kept apart; known missing assets are not hidden and no destructive cleaning is introduced. Free resources are authorized; no monetary cost and no remote upload without an explicit decision; no public deploy or automatic Git integration.

## Phase overview

| Phase | Verifiable result | Dependencies | Status |
| --- | --- | --- | --- |
| 0 | Governance, decisions and initial documentation | None | Completed: bootstrap integrated in `dev` at `66ba822`; governance replaced by [ADR 0010](decisions/0010-multi-model-agent-governance.md) |
| 0.1 | Python migration to uv and reproducible packaging | Governance bootstrap integrated in `dev`; run a001 | Completed: integrated in `dev` at `b90c6d4` on 2026-10-07 ([ADR 0006](decisions/0006-python-toolchain-uv.md)) |
| 1 | Corpus, benchmark and reasoned choice of the engines | Phase 0.1, samples and hardware profiles | Partial: observations on a synthetic sample in [benchmarks](benchmarks/README.md), [a002 debt](benchmarks/a002-residual-debt.md) open; does not block the experimental product |
| 2 | Application package, contracts, configuration, DB and worker | Operational evidence of the engines and local data | First increment I1 available on `dev` (experimental, not released) |
| 3 | First complete application with FastAPI and a minimal Web UI | Needed foundations and frontend | First increment I1 available on `dev` (experimental, not released, quality `unverified`); later features to come |
| 4 | Required formats, scientific PDFs, OCR and assets | Phases 1–3 | Not started |
| 5 | OCR/AI review and source/result comparison | Provenance and assets of phase 4 | Not started |
| 6 | Distribution and acceptance on CPU/GPU/remote on the intended systems | Phases 3–5; preliminary checks already in phase 1 | Not started |
| 7 | Exports and optional derivatives | Stable document and revisions | Not started |
| 8 | Migration, complete documentation and release | Criteria of the required phases met | Not started |

## Increments

Planned sequence of usable increments. Only I1 exists; the others are planned and not available, and their order follows use and the owner's decisions. Each delivery keeps the previous path working and updates the capability matrix in [benchmarks](benchmarks/README.md).

| Increment (phase) | Usable capability | Real proof required | Declared limits |
| --- | --- | --- | --- |
| I1 (2 and 3) — **available on `dev`** | Local PDF to Markdown with assets, CLI and browser, archive, errors and download | Probes P01–P14, especially browser on C06, restart and ZIP outside the application | Experimental product, Linux and CPU only; quality `unverified`; no review or advanced scientific rendering |
| I2a (4) — native formats | TXT, DOCX, RTF and EPUB with resources, preservation and the same UI path | Real conversion for every declared format; DOCX C02, EPUB C03 with a note/backlink check, pertinent synthetic RTF and TXT | The EPUB defect stays visible until recovered; unsupported variants not promised; TXT and DOCX do not replace PDF |
| I2b (4) — scientific documents | IT/EN OCR, column and mixed PDFs with explicit options, formulas and crops, source page that can be consulted | C07, C08, C10 and PDFs of real use; per page or region checks of formulas, tables, numbers, figures and order, including the raster region lost in C10 | No formula-to-LaTeX equivalence from rendering alone; fallback to a labelled crop, raw intact; partial outcomes declared, no universal ranking |
| I2c (4) — LaTeX and resources | Import of a declared LaTeX bundle, Markdown with assets and references | Redistributable bundle with two includes, formulas, figures, notes and references; negative traversal, external macros and resources | Explicit initial subset; compilation of arbitrary sources not implied |
| I3a (5) — comparison and manual review | Source and result side by side, reversible manual corrections, chosen revision in the download | Correct an OCR or number error on C08, reject and reopen, find the original again and download the selected revision | Unknown coordinates visible, no invented source map; human feedback traced |
| I3b (5) — optional AI review | Local review proposals with before, after and reason, accept or reject | Real local proof of OCR, negation, number and formula corrections, a harmful proposal rejected, provider unavailable | No local model declared ready today; mocks only for contracts; cloud excluded at zero cost; no initial auto-accept |
| I4 (6) — distribution | Local Docker start, backup and restore, repeatable offline use; then profiles and platforms one at a time | From a clean environment: installation and preparation, PDF conversion in the browser, restart and backup restored without network; GPU, WSL or Mac only on available hardware | An untested target is not verified; no universal GPU container; no automatic redistribution of weights or public deploy |
| I5 (7) — export | HTML and PDF, then DOCX from the chosen revision, with assets and formulas | Open the export outside the application and compare contents, notes, figures and formulas; a case with a render that cannot be represented, declared | Explicit layout differences; exported HTML never executed automatically in the application origin |
| I5 optional (7) — derivatives | Chunking/RAG or chart reconstruction only when requested | Recompose the chunk sequence, formulas and tables unsplit, a block over budget; compare a reconstruction with the original figure | Not needed for the release; no implicit chat or vector index; generated derivatives labelled |
| I6 (8) — migration and release | Selective migration of legacy outputs and experimental local release with complete guides | Import same-named legacy outputs with assets and explicit unknown provenance; reconvert a case; restore and documented commands proven | Do not fabricate missing versions or configurations; keep legacy readable; integration and promotion stay manual, by the owner, after the pertinent GO |
| Debt consolidation (track) | Reliable reader, comparator, parser and advanced validation in the promised profile | Planned work with adequate time and resources and independent evidence | Current limits are not removed by the plan; see the [technical debt register](development/technical-debt.md) |

Remote conversion and review providers remain roadmap extensions with distinct adapters: a first remote profile needs an explicit scope for data, endpoint and cost, submit/poll/recovery without duplicating jobs after a timeout, a real proof and the preservation of the outcomes. At zero cost they are not a hidden dependency of I2 or I3. Authentication and multi-user use must be designed before the product is exposed beyond the current local target.

## Phase 0 — Project base

Deliverables: branch `feature/document-converter-v2` from `dev`, initial draft (`documentation/architecture/preliminary-draft.md` (archived, tag `pre-docs-2026-10-18`)), ADRs, Web UI requirements, roadmap, `AGENTS.md`, local skills and workflows, Diátaxis structure of `docs/` and MCP evaluation. The application code remains legacy.

Completion: local links and skills validated, no technical decision still open presented as implemented. The README points to the new documents; its complete rewrite accompanies the development.

## Phase 0.1 — Reproducible Python toolchain

Decision [ADR 0006](decisions/0006-python-toolchain-uv.md), initial run `run-a001-fase0-uv`. Bootstrap integrated manually in `dev` at `66ba822`; branch `feature/run-a001-uv` verified on that base, as recorded in the development log (archived: `documentation/CHANGELOG.md`, tag `pre-docs-2026-10-18`). The run applied the supervised cycle of the time (archived: `documentation/development/run-lifecycle.md`, tag `pre-docs-2026-10-18`): plan, two reviews, arbitration, implementation, two reviews and final arbitration.

Introduce pyproject, lockfile, Python version and local environment; separate runtime and dev dependencies, fix package discovery and entry points, align instructions and reproducible builds. Exclude `.venv/` and `temp/` from the Docker contexts. Evaluate the scope of the Marker dependencies without prematurely choosing the final engine. Keep the application behaviour except for corrections needed by the packaging.

**Exit:** environment recreated from the lockfile, package importable and commands verified outside the source directory, pertinent legacy checks run, explicit limits, final GO on the snapshot and manual integration. This phase anticipates the packaging part of phase 2; the new application domain stays in phase 2.

## Phase 1 — Corpus and choice of the engines

Build a small but representative corpus, with references verified by hand: text PDFs with one and more columns, IT/EN scans, mixed pages, inline and multiline formulas, matrices, notes, bibliographies, tables and charts. Include TXT, DOCX, EPUB, RTF and LaTeX projects with inclusions and images. Specify rights and provenance; keep private documents out of the repository.

Compare native parsers/Pandoc, Docling and an up-to-date Marker. Evaluate a remote provider within the agreed data and cost scope. For each candidate record versions of code and weights, licences, hardware, configuration, time, memory, cost and failures. Run a CPU test and a preliminary check of the intended GPU paths before basing the architecture on a model.

Measure text accuracy against the transcription, omissions, block order, formulas and symbols, table cells, completeness of assets and references. Agree thresholds per document class on the observed corpus; no single average may hide a loss of formulas or pages. The engine's confidence scores are signals, not the correct reference.

**Exit:** reproducible report, matrix of formats, capabilities and limits (the [capability matrix](benchmarks/README.md#capability-matrix)), ADR on the engines and first choice of the document model. No predetermined winner. The points still uncertain must have a circumscribed follow-up experiment.

## Phase 2 — Reliable foundations

Create an installable Python package with reproducible dependencies, separating domain, services, adapters, API and worker. Define the structured document, assets, manifest, revisions and Markdown profile. Introduce persistent IDs and per-block provenance; avoid coupling all components to the internal schema of a single engine.

Implement typed configuration and a commented `.env.example`, with documented precedence and a snapshot without secrets. Implement SQLite with migrations, confined storage, source hash, distinct executions and invalidation of reuse based on configuration and versions. Design persistent jobs, attempts, cancellation and recovery after a crash; choose a queue mechanism proportionate to a single host.

Prepare CI and checks on the package, the contracts and the pertinent regression cases. The fast tests do not download models or use real credentials. Keep the legacy suite distinct during the transition, fixing its discovery and packaging when those components are touched.

**Exit:** installation from a clean environment; same-named documents and reruns do not collide; errors persisted; a restart does not lose the queue; DB and artifacts can be reconciled after an interrupted write; the effective options can be verified.

## Phase 3 — First usable path with Web UI

Since 2026-10-16 the validator debt stays open, but the exit of phase 3 does not require resuming the parser immediately. The user can use the experimental product and spread the validation work over time. No automatic attestation of fidelity or semantic completeness without evidence; operational checks, file safety and exposed limits remain necessary.

Expose a FastAPI API for upload, start, status, archive and download, with an OpenAPI schema. The worker runs a real conversion on a representative PDF, with TXT/DOCX as further checks when pertinent. The CLI calls the same services. Choose the frontend with an ADR (for I1: HTML, CSS and JavaScript without framework, [ADR 0009](decisions/0009-first-usable-product.md)) and provide a UI for upload, queue, archive, preview and download of the Markdown bundle.

Serve UI and API with a simple local configuration. Distinguish job state from document quality and review, show errors and prevent downloads that look complete when artifacts are missing. Keep the configuration of the conversions after a restart. Introduce right away the Diátaxis documentation of the features that are actually available.

**Exit:** proof in the browser from upload to download; reload and restart keep archive and state; API responsive during a conversion; a batch with a wrong file shows success or error per document. This is the first product increment.

## Phase 4 — Conversion of scientific documents and native formats

Complete the adapters for text PDFs, scans and mixed PDFs with IT/EN OCR, column reading order, LaTeX formulas, figures and captions. Verify TXT, DOCX, EPUB, RTF and LaTeX bundles with their resources. Document what cannot be represented faithfully in the Markdown profile and the explicit fallback.

Extract images or render regions when needed; keep charts with axes and legend. Check tables, notes, numbering and references. Integrate the first remote provider and separate submission, polling and retrieval: a timeout does not authorize duplicating a job automatically. Bind URLs and credentials to the intended provider.

Bring source pages, formulas, images and detected problems into the UI. Protect asset paths, uploads and archives; limit resources and confine any LaTeX process. Do not lose contents while rendering the result.

**Exit:** corpus measured against phase 1, agreed thresholds met or problems explicit; every referenced image exists; order and formulas checked; no implicit cloud fallback; the support matrix matches real evidence.

## Phase 5 — Assisted review and quality

Integrate optional AI review through a local or remote provider distinct from the conversion engine. Provide context and crops, collect proposals limited to identifiable blocks and keep original, proposal, reason, decision and revision. Start with manual acceptance; automation requires a later decision supported by measures of right, missed and harmful corrections.

Complete side-by-side comparison, navigation of problems, accept/reject, reopening of revisions and selection of the result to export. Separate OCR errors from typos already present in the source, and generated descriptions from original contents.

**Exit:** every change can be traced and reversed through revisions; mathematics, numbers and negations have specific test cases; the context sent to the provider matches the chosen profile; an AI outage does not destroy the base conversion.

## Phase 6 — Distribution and local operation

Produce reproducible Docker images and configurations for API, UI and worker, persistent volumes and a model cache separate from the data. Formalize CPU and NVIDIA profiles; test Linux, Windows/WSL2 and Mac. For Apple Silicon test the native inference service when Metal is needed, documenting start and connection from the container.

Manage downloads, version checks, required space and offline use after preparation. Publish measured RAM/VRAM and performance; verify the supported CPU architectures. Complete upload limits, readiness, logs without secrets, backup and restore. If use requires network access or several users, complete authentication and isolation consistent with that scope before declaring it supported.

**Exit:** start on the declared machines and profiles, real offline test, restart with persistence, restored backup and documented limits. Where hardware for a test is missing, state "not verified", without turning it into certified support.

## Phase 7 — Exports and derivatives

Verify HTML/PDF export and, by priority, DOCX from the Markdown with assets, formulas and notes. Verify the bundle also outside the application. Document the layout differences and the extensions required from the renderers.

If concrete use requires it, add semantic chunking/RAG from the reviewed document: budget of the target tokenizer, provenance metadata, formulas and tables not split, explicit handling of blocks over the limit. A vector index and a chat over the documents are not needed by the base converter.

Evaluate Mermaid or chart reconstruction as clearly labelled derivatives, always keeping the original figure and verifying any reconstructed data.

**Exit:** verified exports and declared limits. RAG and reconstructions remain optional extensions: not enabling them does not prevent the release of the converter.

## Phase 8 — Migration and release

By the disposition of 2026-10-16 an experimental local delivery or release is allowed, with advanced validation deferred and explicit limits, without a promise of certification or a public deploy. The reviews of the operational features remain.

Define what to import from the legacy outputs and what to reconvert because it has no provenance. Do not fabricate engine versions or parameters that are not available. Document the compatibility of the CLIs and the retirement of the old scripts.

Rewrite the README as a concise entry point to the new product. Complete `docs/`: initial tutorial, conversion, review and backup guides, configuration, API and format reference, explanations of fidelity and architecture, a path for developers. Verify commands, examples and links against the version to be distributed.

**Exit:** agreed criteria met, known limits published, CI and conversion evidence available, migration and restore proven. Integrate the feature branch into `dev` after review; promote the verified version to `main` according to the release decision. No automatic merge is part of the setup.

## Definition of the final result

A person can install and configure the product, convert the declared formats, consult source and problems, correct the OCR with traceability, recover a conversion from the archive and download complete Markdown with assets. The local CPU, GPU and remote API paths have documented capabilities and limits. Tests and benchmarks demonstrate fidelity on the reference corpus, without promising perfection on any document.
