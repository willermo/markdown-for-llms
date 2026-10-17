---
name: plan-before-code
description: Use before changing files in any non-trivial task, and whenever the prompt asks for a plan or a planning-only run. Produces a compact plan grounded in the real codebase and stops at the authorization gate.
---

# Plan before code

1. **Objective** — restate the task and the expected outcome in one paragraph.
2. **Verified baseline** — repository, branch, HEAD, branch point, working-tree state.
3. **Sources read** — files of the repository and vault notes, with why they matter.
4. **Findings** — grounded in real paths and symbols; separate verified facts, hypotheses and stale or conflicting references.
5. **File map** — files expected to change; files inspected but not changed.
6. **Approach** — milestones in order; data model, endpoints, payloads or migrations when relevant.
7. **Verification** — exact commands and manual checks, with the expected result of each.
8. **Risks and rollback**.
9. **Decisions needed** from Davide before implementation.

Close the plan with this exact line:

```text
Questa NON è autorizzazione a implementare.
```

Do not modify files while planning. Implementation starts only after Davide writes `AUTORIZZATO A IMPLEMENTARE.`; approving the plan in a tool's interface is not enough.
