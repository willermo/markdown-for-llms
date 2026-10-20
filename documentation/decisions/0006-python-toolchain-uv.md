# ADR 0006 — Python project management with uv

- Date: 2026-10-02
- Decision status: accepted
- Implementation status: GO of plan and implementation r003 after double reviews; integrated manually into dev at `b90c6d4` on 2026-10-07, tree identical to the feature `8479f97`; archive verified, time deviations NON PASS kept, push verified at `80c31da`
- Origin: explicit approval of the user after the evaluation of the Python setup

## Decision

Adopt `uv` for Python environments, dependencies and commands, with `pyproject.toml`, a versioned `uv.lock`, `.python-version` and a recreatable local `.venv` environment. Separate runtime, development and engine extras. Fix the legacy packaging and the entry points during a circumscribed migration, before the benchmarks.

Manage the interpreter with uv too, progressively, without removing pyenv from the machine or changing its other projects. Pin a version compatible with the code and verify it; the final version for the engines depends on the benchmarks. The application configuration stays in `.env`.

Align Docker with the lockfile and with reproducible versions, distinguishing Python libraries, native dependencies and model weights. Configure the CPU/GPU variants explicitly; evaluate separate environments if the engines have incompatible requirements. Avoid turning the adoption of uv into an early rewrite of the pipeline.

## Context and verification

The survey found Python 3.12.3 from pyenv in the available shell, no version pinned in the repository and no local environment, dependencies without a lock, development tools installed as runtime, incomplete packaging, inline comments passed to `install_requires` and Marker installed from `master` in the Dockerfile.

Verify clean installation, import/entry point outside the code directory, the selected legacy suite and the documented commands. Do not declare an AI engine verified if only mocks were used. The application code is not migrated yet (at the time of the decision).

References: [uv projects](https://docs.astral.sh/uv/guides/projects/), [Python](https://docs.astral.sh/uv/concepts/python-versions/), [Docker](https://docs.astral.sh/uv/guides/integration/docker/), [roadmap](../roadmap.md), run cycle (archived: `documentation/development/run-lifecycle.md`, tag `pre-docs-2026-10-18`).

## Operational update of 2026-10-06 — EbookLib metadata for the resolver

Adopted, within the scope of the authorized migration, the official mechanism `[[tool.uv.dependency-metadata]]` of uv 0.10.10 for **ebooklib==0.18**: `requires-dist = ["lxml", "six"]`. The fields come from the audited setup and from two native offline wheels with identical METADATA; Requires-Python and extras absent. The declaration stays limited to the observed version and keeps PyPI source, sdist hash, pin, indexes and universal graph. It avoids running the backend during the lock while keeping `--no-build`. It is not an override of the requirements to work around a conflict, nor a cache or a hand-compiled lock.

Alternative evaluated: priming the registry with a networked backend and a cache to reuse; excluded because it adds an isolation exception and does not guarantee the reuse. The existing builds differ in the ZIP timestamps: byte reproducibility FAIL, identical contents. Incomplete pre-build bootstrap attestation declared; no complete PASS transferred to the product. Repeating the builds is not needed to use metadata consistent with the static source, with these explicit limits.

**Implementation status:** declaration frozen only in the new copy of the run; root pyproject unchanged and lock still to be run. After the lock is received, promoting the inputs into the product will require a new identity and snapshot and a comparison of the installed metadata with the declared ones. A change of source, version or requirements invalidates this declaration and the relevant evidence. It does not authorize runtime builds and installations or models; V7/V8 remain distinct.

Sources: [pinned documentation](https://github.com/astral-sh/uv/blob/0.10.10/docs/concepts/resolution.md#dependency-metadata), [pinned schema](https://github.com/astral-sh/uv/blob/0.10.10/uv.schema.json). Detailed provenance and receipts in the package-s010 of the run, to be archived in `documentation/runs` (archived, tag `pre-docs-2026-10-18`) at closure.

## Operational errata of 2026-10-06 — sdist eligibility in the lock

The attempt s010 is FAIL: uv 0.10.10 filters the sdists with no-build before consuming dependency-metadata. The previous choice of the metadata remains valid and sourced, but the claim that it was enough with that flag was incomplete. It is not fixed by building another wheel in the cache: the selection branch does not consult it.

Adopted, only in the copy, an **offline** R/D4 resumption: no global no-build filter on lock/check, build ban for every other name of the received index caches, observed metadata only for ebooklib 0.18. A closed name list and an empty source/build cache are gates verified before every start; zero network, backend or installations. It is not a general native allowlist and it does not authorize online executions of the command. P1 admits identified sdists and constrained backends; the global policy was an operational addition, now corrected explicitly without changing pin, graph, sources or product. The wheel installation commands keep their no-build and their distinct future costs.

**Status:** mandate 41 ready, inputs frozen; lock still to be obtained, no early validation. An incomplete cache, a new name or another source requires a concrete disposition; no automatic extension. Byte FAIL and bootstrap gap of s009 kept. Source: [uv 0.10.10 selection](https://github.com/astral-sh/uv/blob/0.10.10/crates/uv-resolver/src/version_map.rs#L521), [metadata consumption](https://github.com/astral-sh/uv/blob/0.10.10/crates/uv-distribution/src/distribution_database.rs#L543).

## Metadata recovery of 2026-10-06 — mandate 43

The lock of mandate 42 consumed the declared EbookLib metadata, but it is FAIL because of the data of narwhals/scikit-learn and other indexes missing from the universal cache. R/D4 PASS stays tied to that call; it does not prove that the lock succeeds.

Adopted a native diagnostic priming with uv pip compile in transitive resolution, no-build/only-binary and a separate configuration, followed in the same chat by the original offline lock/check. --no-deps does not acquire the metadata of the wheels: the pinned resolver requests metadata only in transitive mode and stops visiting the dependencies in direct mode. The diagnostic closure and its pins are not the graph or the pins of the product. The names acquired within the received limits are sealed and all banned from building in the following lock, except the EbookLib exception already observed. No forged cache and no further declared metadata.

The previous policy of static and offline names remains historical for its stage. R011 receives a delimited public acquisition and dependent names without a supervision step for every miss; the product lock stays offline/R/D4. A possible native fallback to the wheel to read metadata counts as an acquisition within the caps, never as an installation or an ABI proof. No weights, fonts, models or new backend.

Pinned sources: [transitive metadata request](https://github.com/astral-sh/uv/blob/0.10.10/crates/uv-resolver/src/resolver/mod.rs#L1669), [direct mode](https://github.com/astral-sh/uv/blob/0.10.10/crates/uv-resolver/src/resolver/mod.rs#L1808), [wheel fallback](https://github.com/astral-sh/uv/blob/0.10.10/crates/uv-distribution/src/distribution_database.rs#L520). Copies and hashes of the sources already received in the run; acquisition and successful lock still to be observed. Status and limits are not equivalent to the final GO of the migration.

## Product receipt of 2026-10-06 — mandate 46

The previous states of the copy remain historical. Universal lock and offline check of mandate 45 received PASS; mandate 46 promoted into the root the same bytes of the lock and only the EbookLib metadata table. Receipt 47 reconfirms hashes, TOML delta, nine base runtime wheels and the real inventory for product-s016. The metadata declaration is now in the product, without changes to the graph.

S/B/I/E base are ready for the resumption after the supervisor freeze: native uv offline build with the existing backend 84 and no-build-isolation, startup and import/get_requires before the build, then the same canonical build in the runtimes. It is an operational choice received within r013/r014, to be evaluated in the two reviews; attestations still future. A sync of the runtime alone does not prove the config-only/flag rebuild of V1. No complete PASS, multi-ABI installability or final GO; cumulative costs of 112 MiB and distinct heavy scopes unchanged. Details in the changelog (archived: `documentation/CHANGELOG.md`, tag `pre-docs-2026-10-18`).
