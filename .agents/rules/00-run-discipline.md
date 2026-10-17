---
trigger: always_on
---

# Run discipline

Applies to every non-trivial run in this repository, whatever the model and tool.

## Cold start

- Treat the active prompt and the files it lists as the only run-specific starting point. Do not rely on memory of earlier sessions.
- Before substantial analysis, verify repository, branch, HEAD, branch point and working-tree state; report any mismatch with the expected baseline.
- Build a concise file map before opening large files; read narrow ranges of the relevant symbols.

## Planning and authorization

- Planning-only and audit tasks are read-only. They end with `Questa NON è autorizzazione a implementare.`
- Implementation begins only when the active task contains the exact formula `AUTORIZZATO A IMPLEMENTARE.`
- Do not broaden scope because a nearby defect or refactor opportunity appears: record it as a finding.

## Synchronous completion

- Run required checks within the run; never leave tests, builds or agents running in the background.
- Call a gate green only when its command has completed and the result is known. If a check cannot complete, state the limitation and keep the gate open.

## Evidence

- Ground findings in real paths and symbols. Separate verified facts, hypotheses, proposals and decisions needed.
- Treat historical plans and notes as intent, not proof of the current implementation.

## Safety

- Never print or persist secrets or values from `.env.*`.
- No push, pull request, merge, rebase, squash, deploy or remote change without Davide's explicit authorization.
