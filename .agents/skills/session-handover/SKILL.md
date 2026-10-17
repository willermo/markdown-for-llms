---
name: session-handover
description: Use at the end of any working session, or when context is close to saturation, to leave a handover that a fresh chat (any model) or the co-architect can resume from without the conversation.
---

# Session handover

Write the handover in the report (and, when the run uses a run folder, as a file in `temp/<run>/`). The state lives in files, never in the chat.

## Format

- **Session metadata** — repositories, branches, commits, working-tree status, mode (planning-only / implementation / review / supervision), live systems touched (yes/no).
- **Summary** — goal, result, implementation performed (yes/no).
- **Verified facts** — separated from reported or assumed information.
- **Files inspected** and **files changed** (the latter only if implementation was authorized).
- **Commands and tests** — exact commands and results; if not run, why.
- **Risks and technical debt**.
- **Decisions required** from Davide (or others, named).
- **Knowledge base impact** — which notes should change (Backlog, Roadmap, Decisioni, documents).
- **Next step** — one actionable step.

## Knowledge base

If your role includes the session close and the knowledge base is in the workspace: update the touched documents and `Backlog.md`, write the session note in `Sessioni/`, and commit following the commit rule of the working-rules note (only your files, add by path, no push). Otherwise list the knowledge base updates in the handover for the co-architect.
