@AGENTS.md

# CLAUDE.md — Claude Code adapter for markdown-for-llms

Shared governance is in `AGENTS.md` (imported above). This file adds only what is specific to Claude Code.

- Claude Code does not load `.agents/rules/` by itself: the pointers in `.claude/rules/` lead to the canonical rules; follow them.
- Skills in `.claude/skills/<name>/SKILL.md` are thin wrappers: when one triggers, read and apply its canonical source in `.agents/skills/<name>/SKILL.md`.
- Local, untracked pointers to the project knowledge base are in `CLAUDE.local.md` (this repository is public: never move them into tracked files).
- Davide's working rules live in the working-rules note of the knowledge base, not in Claude Code's auto-memory. Propose new rules as edits to that note; do not save them only in memory.
- Do not create skills, hooks, MCP configuration, plugins or sub-agents during a product run unless the approved scope targets them.
