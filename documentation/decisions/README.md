# Architecture decision register

Decision statuses: **proposed**, **accepted**, **superseded**, **withdrawn**. Each ADR records the implementation status and the origin of the decision separately. Initial date of the register: 2026-10-02.

| ADR | Decision | Status | Implementation |
| --- | --- | --- | --- |
| [0001](0001-document-fidelity.md) | Complete document, fidelity and optional derivatives | Accepted | Partly applied by I1 (source, original extraction and every attempt kept, quality `unverified`); derivatives not implemented |
| [0002](0002-fastapi-web-ui.md) | FastAPI backend, worker and Web UI | Accepted | I1 integrated in `dev` at `f8bf44c`; comparison and review in later increments |
| [0003](0003-persistence.md) | SQLite and filesystem storage | Accepted for the first local version | I1 foundations integrated in `dev` at `f8bf44c`; further features later |
| [0004](0004-execution-and-configuration.md) | Local/remote, hardware profiles, `.env` configuration | Accepted as direction | I1 Linux CPU integrated in `dev` at `f8bf44c`; other profiles to be verified |
| [0005](0005-documentation-and-governance.md) | Local governance and Diátaxis documentation | Partially superseded by 0010 | The documentation part (Diátaxis, `documentation/` and `docs/`) stays in force; the governance part is replaced by 0010 |
| [0006](0006-python-toolchain-uv.md) | uv Python toolchain before the benchmarks | Accepted | Final GO of the r003 implementation after two GO reviews, time-quota deviations kept as NON PASS; integrated in `dev` at `b90c6d4` on 2026-10-07, push verified at `80c31da` |
| [0007](0007-supervised-development-runs.md) | Supervised runs, double reviews and handover | Superseded by 0010 | Kept unchanged as history of the runs a001–a004 |
| [0008](0008-maintained-parser-and-provisional-benchmark.md) | Maintained parser and a002 closure with provisional benchmark | Accepted by the user's disposition of 2026-10-12 | Parser adapter not implemented; product priority since 2026-10-16; advanced validation deferred, trace r005 not reviewed, a002 debt open |
| [0009](0009-first-usable-product.md) | First local product: Marker CPU, inert UI and operational reference check | Accepted for the experimental I1 | I1 integrated in `dev` at `f8bf44c` (technical GO r004), not released on `main`; three low residuals tracked, quality `unverified` |
| [0010](0010-multi-model-agent-governance.md) | Multi-model agent governance (one source, N tools) | Accepted | Governance integrated in `dev` on 2026-10-17; documentation tranche in progress |

Use [template.md](template.md) for the next free number. Link alternatives, consequences and useful evidence; avoid an ADR for every implementation detail. A new explicit request can change a decision: document the change without introducing an unnecessary further authorization step.
