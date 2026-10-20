# Technical debt of the legacy pipeline

This summary keeps the main outcomes of the survey that preceded the new architecture. Base examined: commit `8fa5be9`. It is not a new certification of the code nor a list of defects already solved. It serves to avoid carrying the same problems into the rewrite and to identify useful regression checks.

| Area | Problem observed | Planned response |
| --- | --- | --- |
| Integrity | Global cleaning can alter indexes, code, formulas and significant whitespace; some options have no effect | Typed blocks and content comparison, phase 2–4 |
| Configuration | Inconsistent precedence between CLI, JSON, presets and environment; errors can fall back silently to the defaults | Typed schema, explicit errors and effective snapshot, phase 2 |
| Identity | Collisions between files with the same stem; the presence of an output can make new inputs be skipped | Identity by content and distinct executions, phase 2 |
| Re-execution | Obsolete outputs and chunks remain mixed with the new ones | Manifest and atomic publication of the artifacts, phase 2 |
| State | Partly failed batches can look successful; some commands end successfully without results | States per document/job, consistent errors and exit codes, phase 2–3 |
| Validation | Several thresholds and configurations not applied; scores do not prove fidelity | Metrics on the corpus and problems with provenance, phase 1 and 4 |
| Chunking | Token limits, overlap and strategies not always respected; wrong offsets can produce empty bodies | Separate derivative and tests on the content, phase 7 |
| Unicode | Cuts and normalizations can lose or corrupt content | Round trip and multilingual fixtures, phase 2–4 |
| Formats | Declared support broader than the actual support of the Pandoc/Marker paths | Verified capability matrix, phase 1 and 4 |
| Provider | Parameters received by the server not always applied; images and provenance lost | Adapter contracts and integration tests, phase 2–4 |
| Remote jobs | Polling and retry can resubmit jobs; timeouts are not equivalent to a certain failure | Keep the remote ID, distinguish submission, wait and retrieval, phase 4 |
| Server | Synchronous processing blocks async endpoints; health can be positive with the engine not ready | Separate worker, distinct readiness and upload limits, phase 2–3 |
| Input and assets | Asset paths not confined and polling URL not bound to the provider | Safe resolution, validation of archives and URLs and confined credentials, phase 2–4 |
| Packaging | Package built with missing modules and inconsistent entry points; dependencies on the current directory | Installable package, verified entry points and lock of the dependencies, phase 2 |
| Container | Dependencies not pinned and configuration and mounts not always effective | Reproducible images and smoke tests of the profiles, phase 6 |
| Observability | Logs and monitor describe states and metrics that are not always consistent | Structured events and state per execution, phase 2–3 |
| Tests and guides | Tests do not cover real content loss; obsolete examples and scripts | Corpus, complete path tests and verified documentation |

In the survey the selected suite `tests/` reported **67 tests passed** in a Python 3.12 environment prepared for the audit. Discovery from the root showed a conflict between the two `test_pipeline.py` files. No benchmarks with real Marker models or cloud conversions had been run. These historical results do not replace the checks of future changes and do not attest the correctness of the conversions.

Reuse fixtures, logic and adapters selectively, only after comparing them with the new contracts. Before removing the legacy pipeline, document compatibility, migration of the outputs and CLI behaviour: see the [roadmap](roadmap.md).
