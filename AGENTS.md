# AGENTS.md — markdown-for-llms

Shared instructions for every agent working in this repository, whatever the model (Claude, Sol, Gemini) and tool (Claude Code, Codex, Antigravity in Antigravity IDE or VS Code). Claude Code loads this file through `CLAUDE.md`.

## Roles

- **Davide** is the owner and final decision maker. Integration, merge, push, promotion to `main` and deploys are his actions.
- **Supervision / co-architect** (a Claude session working on the project knowledge base) writes prompts, plans the run, arbitrates reviews and keeps the knowledge base coherent.
- **Author** (the developer agent of the run) plans and implements within the approved scope.
- **Reviewers** (two models other than the author) review plans and diffs in fresh chats.

Your role in a session is stated by the prompt. Prompts never name the model: whatever model you are, follow the role.

## Project knowledge base

Project state, priorities, decisions about the working method, sessions and open actions live in Davide's private knowledge base, outside this repository. This repository is public, so its location is not recorded here: local, untracked pointers are in `CLAUDE.local.md` (Claude Code) and `.agents/rules/90-local.md` (Antigravity); for Codex the prompt gives them.

When the knowledge base is available (in the workspace, or added with `--add-dir`), read for non-trivial sessions, in this order: the working rules, the development protocol and the conventions listed in the local pointers, then the project's `Home.md` and `Backlog.md`, then any note the prompt references. Do **not** read the project's `CLAUDE.md` in the knowledge base as instructions for you: it belongs to the co-architect. If you cannot access the knowledge base, say so and work from the prompt.

The knowledge base owns strategy, priorities and process decisions; this repository owns the product, its code and its technical truth (branches, commits, tests, architecture decision records, public roadmap, known limitations).

## Source precedence

1. Davide's explicit instruction in the active task.
2. The approved task prompt for the current run.
3. Knowledge base notes explicitly referenced by the prompt.
4. Tool-specific governance (`CLAUDE.md`, `.claude/`, `.agents/rules/` triggered for the current context).
5. This file.
6. The actual codebase on the active branch, for technical state.
7. General model assumptions.

Report conflicts; never silently rewrite governance or knowledge base files.

## Gates

- For any non-trivial task, plan before changing files: objective, files expected to change, files inspected but not changed, risks and rollback, verification commands.
- A planning-only or audit task is read-only: no edits, no commits, no installs, no destructive commands. It ends with: `Questa NON è autorizzazione a implementare.`
- Implementation starts only with the exact formula `AUTORIZZATO A IMPLEMENTARE.` written by Davide. Approving a plan in a tool's interface (for example Antigravity's plan review) is **not** an authorization.
- Stop and report when an assumption is disproved, an irreversible decision is not covered, a security or privacy risk appears, a dependency or a cost is not authorized, or the scope grows.
- Complete required checks within the run: never call a gate green without the command and its result, never leave checks "queued" or running in the background.
- Sub-agents, background tasks and worktree/branch modes only when declared in the prompt or approved plan.

## Branches, environments and promotion

This project has no `staging` branch: there is no remote environment yet (see "Environments and deployment"). When one exists, the default model with `staging` applies.

| Branch | Meaning | Who updates it |
| --- | --- | --- |
| `main` | releases | Davide only, by promotion from `dev` |
| `dev` | local development and integration | merges of feature branches (Davide, or a supervised executor) |
| `feature/…`, `fix/…`, `chore/…` | one run, created from `dev` | the author |

Promotion: feature branch -> merge into `dev` (squash; fast-forward for a single commit) -> local verification -> fast-forward to `main` when Davide decides a release. The roadmap is never an instruction to merge, release or deploy.

```bash
git switch dev
git switch -c <feature|fix|chore>/<short-name>
```

- Never work directly on `dev` or `main`.
- No push, pull request, merge, rebase, squash or force-push unless Davide explicitly asks. Prepare exact commands with the expected output instead.
- No refactors or reformatting outside the approved scope.

## Environments and deployment

- The tested target is a single user on Linux x86_64, CPU only, loopback access (`127.0.0.1`). Public, LAN or multi-user use, GPU profiles, Windows and macOS are not supported until verified.
- The new product (`markdown_for_llms/`) runs locally as two processes (API and worker). The Docker and Compose files in the repository belong to the legacy pipeline and do not prove that the new product can be distributed: Docker distribution of the new product is a roadmap increment.
- No public deploy, no weakening of the sandbox isolation, no persistent privilege as a side effect of a run. Deploys and operations on remote machines need an explicit operational scope from Davide.

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
- Architecture: FastAPI, a separate worker for long conversions, SQLite and the local filesystem. Web UI, API and CLI share the same application services; do not duplicate logic in the UI or in endpoints. The API process does not import the ML engine.
- Conversion and AI review have separate providers and configuration. Local mode never sends documents to remote services through implicit fallbacks.
- Configuration is typed (`MDF_` prefix), documented in `.env.example` and recorded per run. Never store secrets in logs or snapshots.
- The legacy script pipeline (root modules, `markdown-pipeline` and related entry points) coexists with the new product (`mdf`). Do not confuse the two paths and do not remove legacy compatibility without an explicit decision.

### Priorities and accepted debt

Davide's decision of 2026-10-16: move towards a usable converter and Web UI through the roadmap increments, with uncertified quality and explicit limits. The a002/a003 reader and comparator, the parser/trace work and advanced validation are recognized technical debt (`documentation/development/technical-debt.md`, `documentation/benchmarks/a002-residual-debt.md`, ADR 0008), to be consolidated when time and resources allow; they are not prerequisites of the experimental product.

- Do not restart the parser/trace cycle automatically and do not link the experimental Pandoc fork as an implicit dependency.
- Do not turn old `NO_GO` results into `GO`, and do not invent retroactive reviews.
- Keep the operational checks on files, paths, errors and storage.
- Check resource availability (`documentation/development/cleanup-2026-10-17.md`) before reusing old environments or snapshots.

### Costs, resources and permissions

- Free resources needed by the task are authorized; no monetary cost without Davide's explicit approval. No paid remote benchmarks and no heavy model downloads as an implicit part of tests.
- A permission error during execution (`EPERM`, `EACCES`) is not a refusal by the approval system. Diagnose it and, when available, request the escalation the tools provide for an action already authorized, keeping the project's isolation. Stop at an actual refusal, or when recovery changes costs, privileges or boundaries; never work around a negative decision.

### Run conduct

- The author's mandate covers a complete result and its fixes within the authorized scope: ordinary errors, red tests and tool defects are fixed in the same chat and invalidated checks are repeated, without asking the supervision at every failure. Substantial deviations (method, scope, cost, boundaries) go back to the supervision.
- Technical choices that are reversible and within requirements, costs and boundaries are allowed even if the prompt does not list them: record choice, reason, risks and evidence for the reviewers.
- Reviewers work in fresh, independent chats, on the same object and with the same criteria, without reading the other review first. For each finding verify reachability on admitted inputs, impact, attribution to the baseline and the requirement violated; synthetic cases are valid for fidelity, boundaries and security.
- Prefer the decisive minimal proof and the diagnosis of a recurring cause to large test matrices or repeated evidence bundles. Do not duplicate archives at every round: raw evidence stays in `temp/<run>/` and is cleaned selectively.
- For UI changes, test in a browser including refresh, a failed job and persistence. When a change affects configuration, update `.env.example` and the reference docs; update the roadmap status only when its criteria are met.

### Verification

- Run checks proportionate to the change. For conversions, verify text, formulas, assets and order, not only the presence of files or metadata (skill `verify-conversion-fidelity`).
- Use the virtual environment linked to the managed local CPython 3.12, with the package installed in the same interpreter as the child processes. After code changes rebuild sdist and wheel, reinstall the canonical wheel and compare hashes before running with `--no-sync`: `uv sync` alone is not proof. Procedure in `docs/how-to/ambiente-uv.md`.
- Fast suite, after the installation preflight (tokenizer cache, Pandoc and runner ready):

  ```bash
  "$RUN_DEV_PY" -I -B -m pytest tests/ --ignore=tests/api --ignore=tests/packaging --ignore=tests/docker -q
  ```

  The two root smoke scripts are excluded. The selected suite is not a full acceptance test, and historical legacy results do not transfer to new code. Do not install or download anything to repair a failed collection.
- For documentation-only changes: local links, consistency of statuses and `git diff --check`. Report what was verified and what remains unverified.

### Documentation

- `documentation/`: public project documentation of the product (architecture decision records, roadmap, known limitations and technical debt, architecture notes, benchmark conclusions). No evidence trees, review bundles, test dumps, runtimes or recursive copies of runs. A hash records an identity; it does not make deleted evidence available.
- `docs/`: user and developer documentation of the available behaviour, organized by Diátaxis (skill `write-diataxis-docs`).
- `README.md`: entry point to the product and its guides; run outcomes do not belong there.
- Significant architecture choices go in an ADR (skill `record-architecture-decision`), keeping the decision status separate from the implementation status. Superseded ADRs are linked, not rewritten.
- Run plans, prompts, reviews, arbitrations, handovers and raw evidence live in `temp/<run>/`, ignored by git.
- Language: code, comments, docstrings, commit messages, branch names and all repository documentation in **English**. Exception, until the documentation tranche translates them: the existing documents in `documentation/`, `docs/` and `README.md` are still in Italian; new or rewritten text is in English. Do not rewrite historical commits.

## Final report

1. branch and commits;
2. files changed and inspected;
3. commands and tests executed, with results;
4. deviations from the approved plan;
5. risks, decisions needed, follow-ups;
6. suggested commit message if no commit was made.

Planning and audit reports also confirm that nothing was implemented.
