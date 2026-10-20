# ADR 0004 — Configuration and execution profiles

- Date: 2026-10-02
- Decision status: accepted as architectural direction
- Implementation status: I1 Linux CPU r004 GO, integrated into dev at f8bf44c and push verified on 2026-10-17; other profiles to be verified
- Origin: the user's requirements on local and remote processing, CPU/GPU, Docker and `.env`

## Decision

Support local processing and remote providers through replaceable adapters. Separate the conversion provider from the AI review provider. Local mode does not trigger a cloud call automatically. Keep the model, version and parameters actually used, without including keys or credentials.

The `.env` file will be the manual configuration point of the application, with a commented `.env.example` and typed validation at startup (for I1: a file of `MDF_` settings that is never looked up implicitly and is selected with `--env-file`, see the [local product reference](../../docs/reference/local-product.md)). Any per-conversion options in the UI/CLI/API will be validated and persisted in the DB. Precedence and default values must be unambiguous and documented; the exact policy will be defined in the foundation phase (for I1 the same reference defines it: defaults, then the file selected with `--env-file`, then the environment, then `--data-root` and `--model-root`). Do not add a second mandatory JSON file.

Provide CPU, NVIDIA GPU and native inference service profiles where needed, with explicit downloads, pinned versions, persistent cache and offline use after the models are prepared. A profile does not guarantee support for every model.

## Consequences and alternatives

Docker is the main proposed distribution for API, UI and worker. Access to the GPU requires a platform-specific path: verify Linux/NVIDIA and Windows/WSL2; on Mac evaluate native inference to use Metal. Do not assume that the same container exposes every GPU. The supported combinations will be published after real tests, together with RAM/VRAM requirements and performance.

Marker remains a candidate together with Docling and native parsers. No engine is selected on the basis of a demo alone. References and candidates are in the draft (archived: `documentation/architecture/preliminary-draft.md`, tag `pre-docs-2026-10-18`); criteria in the [roadmap](../roadmap.md).

## Refinement of 2026-10-17 — Initial profile

[ADR 0009](0009-first-usable-product.md) selects Marker 1.10.2 CPU for the first experimental local Linux product; adapter proven with C06/C07 and a new real C06 in r003, GO of I1 in r004. The general candidacy above remains history; it is not a promise of support for the other profiles or of content fidelity.
