# Project documentation

This folder is the public project documentation of the product: architecture decisions, roadmap, known limitations and technical debt, architecture notes and benchmark conclusions. User and developer documentation of the available behaviour lives in [`docs/`](../docs/README.md), organized by Diátaxis; the entry point to the product is the [README](../README.md) and the release changes are in the [changelog](../CHANGELOG.md).

Current state: the first increment I1 is an experimental product integrated in `dev`, not released on `main`, with quality `unverified` ([roadmap](roadmap.md), [ADR 0009](decisions/0009-first-usable-product.md)). The inconclusive validation work is recognized technical debt ([ADR 0008](decisions/0008-maintained-parser-and-provisional-benchmark.md)).

## Contents

- [Architecture decision records](decisions/README.md): the source of the current choices, with decision status and implementation status kept apart.
- [Roadmap](roadmap.md): phases, increments and their completion criteria.
- [Technical debt](development/technical-debt.md): deferred validation and the I1 residuals.
- [a002 debt](benchmarks/a002-residual-debt.md): the experimental meter and the maintained parser.
- [Benchmarks](benchmarks/README.md): conclusions, method and limits, capability matrix.
- [Legacy findings](legacy-findings.md): known problems of the legacy pipeline.
- [Web UI](architecture/web-ui.md): goals and user path of the interface.

## Development history

The development history (the development log, the run archives a001–a004, the retired run lifecycle and its templates, the dated benchmark reports, the preliminary draft, the MCP note, the open questions and the cleanup report of 2026-10-17) is no longer in the tree: its last version is at tag `pre-docs-2026-10-18`. Documents that cite it use the form `` `path` (archived, tag `pre-docs-2026-10-18`) ``.

```bash
git ls-tree -r --name-only pre-docs-2026-10-18 -- documentation/
git show pre-docs-2026-10-18:<path>
```

Open product questions and the evidence inventory of the cleanup are kept in the project knowledge base, outside this repository. The older tag `pre-kit-2026-10-17` keeps the state before the agent governance of [ADR 0010](decisions/0010-multi-model-agent-governance.md).

## How to maintain this folder

- Record significant architecture choices as ADRs with the skill `record-architecture-decision`, using the [ADR template](decisions/template.md); update the register and link the roadmap phase concerned. Keep superseded ADRs, stating which ADR supersedes them. A proposal is not an accepted decision, and an accepted decision is not an implemented feature.
- Record user-visible changes in the root [changelog](../CHANGELOG.md) (skill `prepare-release`), not here.
- Put benchmark conclusions in [`benchmarks/README.md`](benchmarks/README.md) with versions, corpus, hardware and limits (skill `verify-conversion-fidelity`).
- No evidence trees, review bundles, test dumps, runtimes or copies of runs: run material stays in `temp/<run>/`, ignored by git.

Agents follow [`AGENTS.md`](../AGENTS.md) and the skills in [`.agents/skills/`](../.agents/README.md) ([ADR 0010](decisions/0010-multi-model-agent-governance.md)).
