# ADR 0010 — Multi-model agent governance

- Date: 2026-10-17
- Decision status: accepted
- Implementation status: governance integrated in `dev` on 2026-10-17; documentation tranche closed (development history archived, release changelog, ADR 0008 and ADR 0009, roadmap and indexes realigned, remaining documents translated into English; exceptions declared in `AGENTS.md`)
- Origin: explicit decision of the owner to adopt a common development protocol for all his projects, with several models and editors
- Supersedes: [ADR 0007](0007-supervised-development-runs.md); the agent governance part of [ADR 0005](0005-documentation-and-governance.md)

## Context

The project was developed with one tool (Codex in VS Code) and a project-specific supervised run cycle (run lifecycle (archived: `documentation/development/run-lifecycle.md`, tag `pre-docs-2026-10-18`), ADR 0007): two independent reviews per gate, arbitration, snapshots and fingerprints of every input, a long handover log in `temp/`, a development log in `documentation/CHANGELOG.md` (archived, tag `pre-docs-2026-10-18`).

The cycle produced real results (runs a001–a004, the first usable product I1), but also a growth of evidence and administrative verification that slowed the work, as recorded in the cleanup report (archived: `documentation/development/cleanup-2026-10-17.md`, tag `pre-docs-2026-10-18`). Meanwhile the owner adopted a common protocol for all his projects, where Claude, Sol and Gemini work interchangeably through Claude Code, Codex and Antigravity, in Antigravity IDE or VS Code.

Each tool reads different files: Claude Code reads `CLAUDE.md` and `.claude/`; Codex reads `AGENTS.md` and `.agents/skills/`; Antigravity reads `AGENTS.md`, `.agents/rules/` (only with a `trigger` frontmatter) and `.agents/skills/`, and no longer indexes `.agents/workflows/`. A rule that only one tool sees does not exist for the others.

## Decision

1. **One source, N tools.** `AGENTS.md` is the shared baseline; `CLAUDE.md` imports it; canonical rules live in `.agents/rules/` with a `trigger` frontmatter and thin pointers in `.claude/rules/`; canonical skills live in `.agents/skills/` with thin wrappers in `.claude/skills/`. No `GEMINI.md`, no `.agents/workflows/`: procedures are skills.
2. **Roles.** Supervision is a Claude session working on the owner's private knowledge base; the author (planner and implementer, chosen run by run) is one model; the two reviewers are the other two models, in fresh chats at every round, with the same prompt. Every gate goes through both reviewers.
3. **Gates.** Plans before code for non-trivial work; planning-only runs end with `Questa NON è autorizzazione a implementare.`; implementation starts only with `AUTORIZZATO A IMPLEMENTARE.` written by the owner. Integration, merge, push, promotion and deploys remain the owner's actions.
4. **Where things live.** The repository owns the product and its technical truth: code, tests, ADRs, public roadmap, known limitations, user documentation. The private knowledge base owns priorities, process decisions, sessions and open actions; its location is never written in tracked files (local untracked pointers: `CLAUDE.local.md`, `.agents/rules/90-local.md`). Run material (plans, prompts, reviews, arbitrations, handovers, raw evidence) lives in `temp/<run>/`, ignored by git.
5. **What is retired.** The skill `manage-implementation-run`, the six procedures in `.agents/workflows/` and the run lifecycle as the procedure for new runs. Snapshots, fingerprints and `scripts/run_context.py` are no longer required; the script stays available as an optional tool.
6. **What is kept.** The product principles, the priority decision of 2026-10-16 and the accepted debt (ADR 0008), the cost and permission rules, the author's mandate by result, the independence of reviews and the verification rules, moved into `AGENTS.md`; the skills `record-architecture-decision`, `verify-conversion-fidelity` (now also covering engine benchmarks) and `write-diataxis-docs`, translated; a new skill `prepare-release` replaces the release procedure. Narrower operational instructions of the old cycle are not carried over one by one but replaced by the general rules of `AGENTS.md`: authorized public sources and verified redirects, operational parameters and evidence of failed calls, placement of fixtures and reusable tests. The documentation tranche settled the two open items: benchmark conclusions and the capability matrix live in `documentation/benchmarks/README.md`, product open questions in the knowledge base.
7. **Branches.** `main` for releases and `dev` for integration; feature branches from `dev`. No `staging` branch until a remote environment exists.
8. **Language.** All repository documentation in English; the documentation tranche translated the existing Italian documents, reorganized `documentation/` (development history archived at tag `pre-docs-2026-10-18`) and introduced the release changelog. Remaining exceptions are declared in `AGENTS.md`.

## Alternatives considered

- **Keep the project-specific cycle and add the other tools to it.** Rejected: it duplicates the common protocol and keeps the evidence overhead that the owner asked to remove.
- **One instruction file per tool (`CLAUDE.md`, `GEMINI.md`, `AGENTS.md` with full content).** Rejected: copies drift; the common protocol requires one canonical source.
- **Move ADRs and roadmap to the private knowledge base.** Rejected: the repository is public, ADRs change together with the code they motivate and are reviewed in the same diff, and agents working from a clone must be able to read them.

## Consequences and verification

- Every model sees the same rules; reviews are comparable across rounds because reviewers stay the same for a whole run.
- The governance is verified with static checks (no placeholders, `trigger` frontmatter on every rule, a Claude wrapper for every skill, no `GEMINI.md` or `.agents/workflows/`, files under 24,000 bytes, no personal paths in governance files) and, for each new model, with an entry test: a planning-only run that must read the right files, stop at the gate and refuse to implement without the formula.
- Historical runs a001–a004 keep their records; their evidence availability is described in the cleanup report `documentation/development/cleanup-2026-10-17.md` (archived, tag `pre-docs-2026-10-18`).
- Reconsider this decision if a tool changes what it loads, or if the owner changes the common protocol.

## Links

- [AGENTS.md](../../AGENTS.md), [CLAUDE.md](../../CLAUDE.md), [`.agents/`](../../.agents/README.md)
- [ADR 0008](0008-maintained-parser-and-provisional-benchmark.md), [technical debt](../development/technical-debt.md)
