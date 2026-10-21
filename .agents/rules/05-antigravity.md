---
trigger: always_on
---

# Antigravity-specific rules

These rules apply to the Antigravity agent (Antigravity IDE, VS Code extension, app, `agy`). Other tools ignore this file.

- `AGENTS.md` is the shared governance of this repository: read it as your primary instructions.
- Approving an implementation plan in the Antigravity interface is **not** authorization to implement. Only the exact formula `AUTORIZZATO A IMPLEMENTARE.`, written by Davide in the conversation, authorizes implementation.
- Do not spawn sub-agents, background tasks or `branch`/worktree workspace modes unless the prompt or the approved plan declares them.
- Procedures are skills in `.agents/skills/`. There are no workflows in this repository.
- If the session needs a file outside the workspace (for example a knowledge base note not in the workspace), ask Davide instead of working around the access policy.
- In a project with several folders, the conversation's default working directory can be a folder other than this repository (for example the knowledge base). The working repository is the folder that contains this `AGENTS.md`: run every command, git included, with that folder as the command's working directory, and before the first git command check that `git rev-parse --show-toplevel` returns it. If it returns another folder, stop and say so.
