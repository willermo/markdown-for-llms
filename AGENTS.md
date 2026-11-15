# AGENTS.md — markdown-for-llms

Shared instructions for every agent working in this repository, whatever the model (Claude, Sol, Gemini) and tool (Claude Code, Codex, Antigravity in Antigravity IDE or VS Code). Claude Code loads this file through `CLAUDE.md`.

## Roles

- **The maintainer** is the owner and final decision maker. Integration, merge, push, promotion to `main` and deploys are the maintainer's actions.
- **Supervision / co-architect** (a Claude session working on the project knowledge base) writes prompts, plans the run, arbitrates reviews and keeps the knowledge base coherent.
- **Author** (the developer agent of the run) plans and implements within the approved scope.
- **Reviewers** (two models other than the author) review plans and diffs in fresh chats.

Your role in a session is stated by the prompt. Prompts never name the model: whatever model you are, follow the role.

## Project knowledge base

Project state, priorities, decisions about the working method, sessions and open actions live in the maintainer's private knowledge base, outside this repository. This repository is public, so its location is not recorded here: local, untracked pointers are in `CLAUDE.local.md` (Claude Code) and `.agents/rules/90-local.md` (Antigravity); for Codex the prompt gives them.

When the knowledge base is available (in the workspace, or added with `--add-dir`), read for non-trivial sessions, in this order: the working rules, the development protocol and the conventions listed in the local pointers, then the project's `Home.md` and `Backlog.md`, then any note the prompt references. Do **not** read the project's `CLAUDE.md` in the knowledge base as instructions for you: it belongs to the co-architect. If you cannot access the knowledge base, say so and work from the prompt.

The knowledge base owns strategy, priorities and process decisions; this repository owns the product, its code and its technical truth (branches, commits, tests, architecture decision records, public roadmap, known limitations).

## Source precedence

1. The maintainer's explicit instruction in the active task.
2. The approved task prompt for the current run.
3. Knowledge base notes explicitly referenced by the prompt.
4. Tool-specific governance (`CLAUDE.md`, `.claude/`, `.agents/rules/` triggered for the current context).
5. This file.
6. The actual codebase on the active branch, for technical state.
7. General model assumptions.

Report conflicts; never silently rewrite governance or knowledge base files.

## Transition until the new initial commit

The project is being realigned by decisions of the maintainer (2026-11-12, 2026-11-14 and 2026-11-15, in the knowledge base). Until the documentation rewrite that precedes a new initial commit of the repository:

- `README.md`, `CHANGELOG.md`, `documentation/` (ADRs, roadmap, technical debt, architecture notes, benchmark conclusions) and `docs/` describe the product before the realignment. They are history, not a reference for the direction: the direction in force is stated by the run prompt and by the knowledge base notes it references.
- They are not maintained: runs do not update them, add no ADR and do not change the roadmap status. A stale public document, or a link in it broken by code a run removes, is not a review finding.
- Exceptions, kept current because the reviewers need them to run the work: this governance (`AGENTS.md`, `CLAUDE.md`, `.agents/`, `.claude/`), the full check (`scripts/full-check.sh`) and its guide (`docs/how-to/uv-environment.md`), the how-to guide of a command a run adds or changes, and `.env.example` for the `MDF_` settings.
- The rules below that depend on the public documentation state how they apply during the transition. The rest of this file stays in force; it is rewritten together with the new initial commit.

## Gates

- For any non-trivial task, plan before changing files: objective, files expected to change, files inspected but not changed, risks and rollback, verification commands.
- A planning-only or audit task is read-only: no edits, no commits, no installs, no destructive commands. It ends with: `Questa NON è autorizzazione a implementare.`
- Implementation starts only with the exact formula `AUTORIZZATO A IMPLEMENTARE.` written by the maintainer. Approving a plan in a tool's interface (for example Antigravity's plan review) is **not** an authorization.
- Two stop classes, in planning and in execution. **Fix, record and continue**: an error whose fix touches neither security, caps, limits, scope nor decisions, and is the first one for that cause; record cause, fix and proof in the run's `stato.md` or in the report. **Stop for the maintainer**: a security or privacy risk; caps or limits at risk; a disproved assumption that changes the approved plan or the answers; a new or irreversible decision not covered; a dependency or a cost not authorized; a growing scope; a cause already seen in the run.
- Complete required checks within the run: never call a gate green without the command and its result, never leave checks "queued" or running in the background.
- Sub-agents, background tasks and worktree/branch modes only when declared in the prompt or approved plan.

## Branches, environments and promotion

This project has no `staging` branch: there is no remote environment yet (see "Environments and deployment"). When one exists, the default model with `staging` applies.

| Branch | Meaning | Who updates it |
| --- | --- | --- |
| `main` | releases | the maintainer only, by promotion from `dev` |
| `dev` | local development and integration | merges of feature branches (the maintainer, or a supervised executor) |
| `feature/…`, `fix/…`, `chore/…` | one run, created from `dev` | the author |

Promotion: feature branch -> merge into `dev` (squash; fast-forward for a single commit) -> local verification -> fast-forward to `main` when the maintainer decides a release. The roadmap is never an instruction to merge, release or deploy.

```bash
git switch dev
git switch -c <feature|fix|chore>/<short-name>
```

- Never work directly on `dev` or `main`.
- No push, pull request, merge, rebase, squash or force-push unless the maintainer explicitly asks. Prepare exact commands with the expected output instead.
- No refactors or reformatting outside the approved scope.

## Environments and deployment

- The tested target is a single user on Linux x86_64, CPU only, loopback access (`127.0.0.1`). Public, LAN or multi-user use, GPU profiles, Windows and macOS are not supported until verified.
- The product (`markdown_for_llms/`) runs locally as two processes (API and worker); the worker, and `mdf convert`, convert with the conversion engine `markdown_for_llms/engine/`. The legacy Docker and Compose files were removed with the legacy pipeline (decision of 2026-11-12), and the repository has no Docker distribution of the product. The only image is the engine toolchain, also used by the bench, built locally by `mdf prepare --toolchain` (alias `mdf bench prepare --toolchain`) from `scripts/bench-toolchain/`: the tools of the native and DjVu routes (pandoc, LibreOffice, DjVuLibre) run in it with no network, a read-only root, no capabilities and the caller's user. Marker 2.2.0 runs on the CPU in a Bubblewrap child without network, under a systemd user scope with a memory cap, with its 18 weight files prepared once by `mdf prepare --weights` and verified before every conversion; Datalab, only when the user chooses it, is called over HTTPS from the process that converts (`mdf convert` or the worker).
- Host ports: this project's block is `23460-23479`, the same on every host of the maintainer's infrastructure, assigned by the infrastructure port registry (see the development protocol, shared infrastructure resources). Today it applies only to the laptop, because the project has no remote environment; the API binds to `127.0.0.1` because the code accepts no other host. Every host port in compose files, `.env.example` and nginx configuration comes from `.env`, lies within the block and binds to `127.0.0.1` unless a reason is stated here; a compose fallback, if any, is a port of the block. A new or changed port outside the block, or missing from the list below, is a review finding.
  - `23465`: API and Web UI of `mdf` (`MDF_PORT`, default in `markdown_for_llms/settings.py` and `.env.example`), laptop only, `127.0.0.1`; odd port of the pair `23464`/`23465` (odd on the laptop and in staging, even in production, by the rule of the infrastructure port registry). An archive still at identity schema 1 or 2 created with the former default `23460` keeps that port in its identity until its first opening upgrades it to schema 3: the storage mismatch message names it.
  - `23461`: relay of `mdf bench` towards an inference server (`MDF_BENCH_RELAY_PORT`, default in `markdown_for_llms/bench/settings.py`, commented in `.env.example`), laptop only, `127.0.0.1`.
  - `23463`: local inference server started and stopped by `mdf bench` (`MDF_BENCH_INFERENCE_PORT`, same defaults), laptop only, `127.0.0.1`.
- No public deploy, no weakening of the sandbox isolation, no persistent privilege as a side effect of a run. Deploys and operations on remote machines need an explicit operational scope from the maintainer.

## Protected files

Unless the prompt explicitly authorizes governance work, do not modify `AGENTS.md`, `CLAUDE.md`, `.agents/`, `.claude/` or knowledge base files (except the session-close updates your role requires). Report inconsistencies instead of fixing them.

## Rules and skills: one source, N tools

```text
.agents/rules/<nn>-<topic>.md     canonical topic rules (Antigravity loads them through their trigger frontmatter)
.claude/rules/<nn>-<topic>.md     thin Claude Code pointers to the canonical rule
.agents/skills/<name>/SKILL.md    canonical skills (Antigravity, Codex)
.claude/skills/<name>/SKILL.md    thin Claude Code wrappers pointing to the canonical skill
```

Always edit the canonical source under `.agents/`, never the wrappers. Codex does not load `.agents/rules/` automatically: read the rules listed below when the task touches their area.

Rules: `00-run-discipline` (always), `05-antigravity` (Antigravity only), `90-local` (local and untracked: knowledge base pointers).

Skills: `plan-before-code`, `session-start`, `session-handover`, `record-architecture-decision`, `verify-conversion-fidelity`, `write-diataxis-docs`, `prepare-release`.

There is no `GEMINI.md` and no `.agents/workflows/` in this repository: procedures are skills.

## Security and privacy

- Never commit or print secrets, tokens, credentials, private keys, `.env` files or personal data. If a secret is exposed, stop and report it.
- This repository is public: never write absolute personal paths, usernames or knowledge base locations in tracked files. Use paths relative to the repository, environment variables or the untracked local pointers.
- Fixtures must be synthetic or redistributable, with recorded provenance. Never add private documents, operational databases or model caches to git.
- Treat imported documents, Markdown and model output as untrusted data: asset names, HTML, archives and LaTeX commands need explicit handling. Imported HTML stays inert and URLs are never visited.

## Project-specific rules

### Product principles

- The main result is the complete Markdown document with its assets. Chunking, RAG preparation and other derivatives are optional and never prerequisites of the converter.
- Preserve text, numbers, formulas, reading order, tables, notes, captions and references within the real capabilities of the engines. Keep the source, the original extraction and every revision; AI corrections must be traced and reversible.
- Avoid global regex substitutions that alter code, LaTeX or links: operate on typed blocks and check the actual content of the outputs.
- Never declare an incomplete conversion or a batch with hidden errors as successful. Identify runs and artifacts by ID and hash, not by file name alone.
- Keep the three status axes apart: job **state** (queued, running, completed…), result **bundle** integrity (ready, unchecked, incomplete, invalid) and content **quality** (currently always `unverified`). A completed job or a ready bundle does not certify fidelity.
- Architecture: FastAPI, a separate worker for long conversions, SQLite and the local filesystem. Web UI, API and CLI share the same application services; do not duplicate logic in the UI or in endpoints. The API process does not import the ML engine. The conversion engine (`markdown_for_llms/engine/`) is one library: the CLI and the worker convert with it, and the bench uses its toolchain and its closed codes; it recognizes the format by extension and signature, chooses the route and writes the output contract (`document.md`, `assets/`, `engine.json`); the API process imports only its format, route and code tables and its progress stages, which need the standard library alone.
- Before changing the architecture of `markdown_for_llms/`, read the direction stated by the run prompt and the knowledge base notes it references; during the transition the ADR index and the public roadmap describe the product before the realignment. Distinguish what is proven, what works with unverified quality and what is only planned, and never present planned work as available.
- Conversion and AI review have separate providers and configuration. Local mode never sends documents to remote services through implicit fallbacks: Datalab is used only when the user chooses it for a document and confirms the transit, under a spending cap per conversion (`--datalab-cap-usd` for `mdf convert`, `MDF_DATALAB_CAP_USD` in the service). Its key `DATALAB_API_KEY` is never a setting: `mdf convert` reads it from the file given with `--env-file` (a regular file of the caller with mode 0600) or from the environment; the service reads it only from the environment of the worker process, which takes it out of its environment when it starts.
- Configuration is typed (`MDF_` prefix), documented in `.env.example` and recorded per run. Never store secrets in logs or snapshots.
- The legacy script pipeline (root modules, the `markdown-*` entry points, the legacy Marker server with its Docker and Compose files, and its packaging and Docker test suites) was removed by the maintainer's decision of 2026-11-12, without compatibility. Do not reintroduce it: useful functions (for example chunking) come back later inside the product, by an explicit decision.

### Priorities and accepted debt

The maintainer's decision of 2026-10-16, whose direction the decisions of 2026-11-12 to 2026-11-15 supersede (see "Transition until the new initial commit"): move towards a usable converter and Web UI through the roadmap increments, with uncertified quality and explicit limits. The a002/a003 reader and comparator, the parser/trace work and advanced validation are recognized technical debt (`documentation/development/technical-debt.md`, `documentation/benchmarks/a002-residual-debt.md`, ADR 0008); they are not prerequisites of the experimental product, and the a002 benchmark code was removed in 2026-11.

- Do not restart the parser/trace cycle automatically and do not link the experimental Pandoc fork as an implicit dependency.
- Do not turn old `NO_GO` results into `GO`, and do not invent retroactive reviews.
- Keep the operational checks on files, paths, errors and storage.
- Check resource availability before reusing old environments or snapshots: the cleanup of 2026-10-17 removed part of the historical evidence and environments, and an old hash does not make deleted evidence available.

### Costs, resources and permissions

- Free resources needed by the task are authorized; no monetary cost without the maintainer's explicit approval. No paid remote benchmarks and no heavy model downloads as an implicit part of tests.
- A permission error during execution (`EPERM`, `EACCES`) is not a refusal by the approval system. Diagnose it and, when available, request the escalation the tools provide for an action already authorized, keeping the project's isolation. Stop at an actual refusal, or when recovery changes costs, privileges or boundaries; never work around a negative decision.

### Run conduct

- The author's mandate covers a complete result and its fixes within the authorized scope: ordinary errors, red tests and tool defects are fixed in the same chat and invalidated checks are repeated, without asking the supervision at every failure. Substantial deviations (method, scope, cost, boundaries) go back to the supervision.
- Technical choices that are reversible and within requirements, costs and boundaries are allowed even if the prompt does not list them: record choice, reason, risks and evidence for the reviewers.
- Reviewers work in fresh, independent chats, on the same object and with the same criteria, without reading the other review first. For each finding verify reachability on admitted inputs, impact, attribution to the baseline and the requirement violated; synthetic cases are valid for fidelity, boundaries and security.
- Prefer the decisive minimal proof and the diagnosis of a recurring cause to large test matrices or repeated evidence bundles. Do not duplicate archives at every round: raw evidence stays in `temp/<run>/` and is cleaned selectively, only after the run's validation and after checking that nothing still active depends on it.
- For UI changes, test in a browser including refresh, a failed job and persistence. When a change affects configuration, update `.env.example`; during the transition the reference docs and the roadmap status are not updated (see "Transition until the new initial commit").

### Verification

- Run checks proportionate to the change, keeping unit tests, integration tests and real benchmarks separate. For conversions, verify text, formulas, assets and order, not only the presence of files or metadata (skill `verify-conversion-fidelity`).
- Use the virtual environment linked to the managed local CPython 3.12, with the package installed in the same interpreter as the child processes. After code changes rebuild sdist and wheel, reinstall the canonical wheel and compare hashes before running with `--no-sync`: `uv sync` alone is not proof. Procedure in `docs/how-to/uv-environment.md`.
- Fast suite, the check of the development cycle, after the installation preflight of the full check (`RUN_SOURCE_MANIFEST`, `RUN_INSTALLATION_RECEIPT` and `RUN_TEST_MODE=standalone`, as the full check sets them):

  ```bash
  "$RUN_DEV_PY" -I -B -m pytest tests/ -q
  ```

  The selected suite is not a full acceptance test, and historical legacy results do not transfer to new code. Do not install or download anything to repair a failed collection.
- Repository hygiene guard: `tests/unit/test_repository_hygiene.py` (no personal home paths in tracked files, local pointer files untracked) runs in the fast suite; for documentation-only changes run it alone with `"$RUN_DEV_PY" -I -B -m pytest --noconftest -p no:cacheprovider tests/unit/test_repository_hygiene.py -q`.
- Full check: `scripts/full-check.sh`. It runs the installation preflight (O, P, S, B, reinstallation of the canonical wheel in the development venv with a RECORD check, I, F) and the preconditions X (the engine toolchain image, read-only) and W (the 18 weight files of the Marker route in `FULL_CHECK_MODEL_ROOT` against `markdown_for_llms/engine/data/weights.json`, read-only), then the fast suite with the bench JUnit guard (`scripts/bench_junit_guard.py`: at least one bench test and one engine test run, none skipped). No lint. It stops on a dirty tree, writes its evidence under `.cache/full-check/` (ignored by git) and exits nonzero when a step or a suite fails. Paths come from the `FULL_CHECK_*` variables described in `docs/how-to/uv-environment.md` ("Run the full check"); none of them lies under `temp/`.
- The full check runs on the commit handed to the diff review and again before the merge, with a clean tree; a selected run never stands for it in a report. The author saves its output and reports the file's path and SHA256:

  ```bash
  mkdir -p temp/<run>/checks
  f=temp/<run>/checks/full-check-$(git rev-parse --short HEAD).txt
  { git rev-parse HEAD; git status --short; scripts/full-check.sh; echo "exit $?"; } > "$f" 2>&1
  sha256sum "$f"
  ```

  One reviewer per review round, named by the supervision, reruns it on the same commit and reports the result next to the author's; the other reads the saved output, runs only the tests their findings need and never calls green a check they did not run. A different hash, a tracked file in `git status --short`, or a missing or incomplete output reopens the gate; a fix round repeats this on the new commit.
- One full check at a time: the reinstallation replaces the package in the shared development venv, so two full checks never run side by side, even from different worktrees (author and reviewer in sequence), and the script refuses to start while another one holds the venv. Stop API and worker processes that use the same venv before running it.
- Memory cap for long commands (development tool, Linux with a systemd user manager): `scripts/run-capped.sh NAME MAX_MEMORY -- COMMAND` runs the command in a transient user scope with `MemoryMax` and no swap, without expanding variables, and appends start, end, exit status and peak memory to `.cache/run-capped/NAME.mem`. Use it for the full check, long bench runs and diagnoses that start external tools; the containers of the engine toolchain and the scope of the Marker child are capped by the engine itself.
- For documentation-only changes: local links, consistency of statuses and `git diff --check`. Report what was verified and what remains unverified.

### Documentation

- During the transition the entries below describe the layout of the documentation, not content to maintain (see "Transition until the new initial commit").
- `documentation/`: public project documentation of the product (architecture decision records, roadmap, known limitations and technical debt, architecture notes, benchmark conclusions). No evidence trees, review bundles, test dumps, runtimes or recursive copies of runs. A hash records an identity; it does not make deleted evidence available.
- `docs/`: user and developer documentation of the available behaviour, organized by Diátaxis (skill `write-diataxis-docs`).
- `README.md`: entry point to the product and its guides; run outcomes do not belong there.
- Significant architecture choices: during the transition no ADR is added or changed (skill `record-architecture-decision` suspended); report them to the supervision, which records them in the knowledge base. Superseded ADRs are linked, not rewritten.
- Run plans, prompts, reviews, arbitrations, handovers and raw evidence live in `temp/<run>/`, ignored by git.
- Language: code, comments, docstrings, commit messages, branch names and all repository documentation in **English**. Declared exceptions: test fixtures and corpus data under `tests/fixtures/`; code whose behaviour depends on Italian text; Italian comments, docstrings and messages still present in existing code, including build and configuration files (`MANIFEST.in`, `pyproject.toml`), translated when that code next changes; ADR 0007, superseded and kept unchanged; the mandatory gate formulas `AUTORIZZATO A IMPLEMENTARE.` and `Questa NON è autorizzazione a implementare.`. Do not rewrite historical commits.
- Run documents under `temp/` (prompts, plans, review reports, arbitrations, `stato.md`, handovers) are written in **Italian**, the owner's language; quotations from the repository, identifiers, paths and commands stay verbatim.

## Final report

1. branch and commits;
2. files changed and inspected;
3. commands and tests executed, with results;
4. deviations from the approved plan;
5. risks, decisions needed, follow-ups;
6. suggested commit message if no commit was made.

Planning and audit reports also confirm that nothing was implemented.
