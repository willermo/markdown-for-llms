# ADR 0005 — Local documentation and governance

> **Partially superseded by [ADR 0010](0010-multi-model-agent-governance.md) (2026-10-17):** the agent governance part (`.agents/workflows/`, instructions for a single tool) is replaced. The split between `documentation/`, `docs/` and `temp/` remains in force; the documentation tranche applied it: it archived the development history and translated the remaining documents into English.

- Date: 2026-10-02
- Decision status: accepted
- Implementation status: initial structure in place; the guides of the experimental product I1 are in `docs/` (tutorial, how-to guide, reference and explanation)
- Origin: explicit request of the user

## Decision

Keep proposals, ADR, roadmap, summaries and selected reports in `documentation/`; the operational evidence stays in `temp/` according to the refinement of 2026-10-15. Organize in `docs/` the operational and development documentation according to the four needs of Diátaxis: learning with tutorials, carrying out a task with how-to guides, consulting a reference, understanding with explanations. The user/developer audience is an attribute of the page, not a fifth type of document.

Use `AGENTS.md` at the root for the common instructions, `.agents/skills/` for reusable procedures discovered by Codex and `.agents/workflows/` for local work sequences invoked explicitly. This last folder is a convention of the repository, not an automation engine. The future CI will have separate executable workflows.

Keep the README as the entry point, migrating it along with the actual features. Planned documentation must say so and must not expose commands that do not exist. MCP servers are optional development tools, evaluated separately from the runtime.

## Consequences and references

The history of the motivations is preserved without mixing it with the usage paths. Maintenance requires links and updates made together with the changes.

### Refinement on retention — 2026-10-15

Explicit disposition of the user after the accumulation of the a003 bundles: the initial formula "evidence in documentation" does not authorize versioning all the evidence. The permanent memory must explain progress, decisions, results and limits; stdout/stderr, receipts, generated matrices, snapshots, raw/asset copies and temporary tools remain in their runs in `temp/`, excluded from Git. The evidence still needed is kept locally; after validation and verification of the dependencies it can be deleted without copying all of it into Git.

Implementation of the rule: protocol, templates, AGENTS and `.gitignore` guards updated in the working tree. After the r007 arbitration, the 8973 untracked a003 operational files (721,430,863 bytes) were moved to temp, with identical hashes and a map of the historical paths. The summary remains in `documentation/runs/run-a003-fase1-engine-comparison/manifest.md` (archived, tag `pre-docs-2026-10-18`). No evidence deleted, no removal of the archives already tracked and no Git rewrite; the move does not reduce the overall space of the workspace. The protocol (archived: `documentation/development/run-lifecycle.md#archivio-e-pulizia`, tag `pre-docs-2026-10-18`) defines selection, availability of the evidence and transition.

- [Diátaxis](https://diataxis.fr/).
- [AGENTS.md](https://learn.chatgpt.com/docs/agent-configuration/agents-md).
- [Codex local skills](https://learn.chatgpt.com/docs/build-skills).
- [Documentation index](../README.md), [operational structure](../../docs/README.md).
