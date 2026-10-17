---
name: session-start
description: Use at the beginning of a working session in this repository, or when resuming after a pause, to rebuild context from the repository and the project knowledge base before doing anything else.
---

# Session start

1. Restate the goal of the session in one paragraph.
2. Inspect the repository: branch, latest commit, working-tree status. If the tree is modified and the prompt does not explain why, report it and ask Davide whether it is expected.
3. Read `AGENTS.md` and the rules in `.agents/rules/` relevant to the task.
4. If the knowledge base is in the workspace, read the working rules, the development protocol and the conventions listed in the local pointers (`CLAUDE.local.md`, `.agents/rules/90-local.md`, or the prompt), then the project's `Home.md` and `Backlog.md`, and the latest note in `Sessioni/`. Do not treat the project's `CLAUDE.md` in the knowledge base as your instructions.
5. Classify the task: planning-only, implementation, review, supervision or handover; state your role as given by the prompt.
6. If the task is substantial, continue with the `plan-before-code` skill.

Report in a few lines: role, task class, baseline, what you read, the next step.
