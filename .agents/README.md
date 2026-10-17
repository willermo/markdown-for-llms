# Agent rules and skills

This folder is the canonical source of the agent governance of this repository ([ADR 0010](../documentation/decisions/0010-multi-model-agent-governance.md)). It installs no global skills, changes no personal configuration and starts no services.

- `rules/` — topic rules with a `trigger` frontmatter (Antigravity loads them; Claude Code reaches them through `.claude/rules/`; Codex reads them when `AGENTS.md` lists them). `90-local.md` is local and untracked.
- `skills/` — procedures, one folder per skill with a `SKILL.md` (`name`, `description`). Claude Code reaches them through the thin wrappers in `.claude/skills/`.

There is no `workflows/` folder: procedures are skills.

## Skills

| Skill | When to use it |
| --- | --- |
| [plan-before-code](skills/plan-before-code/SKILL.md) | Before changing files in a non-trivial task, or in a planning-only run |
| [session-start](skills/session-start/SKILL.md) | At the start of a session, or when resuming after a pause |
| [session-handover](skills/session-handover/SKILL.md) | At the end of a session, or when the context is close to saturation |
| [record-architecture-decision](skills/record-architecture-decision/SKILL.md) | To record or update an architecture decision, keeping ADRs and roadmap consistent |
| [verify-conversion-fidelity](skills/verify-conversion-fidelity/SKILL.md) | When parsers, OCR, cleaning, formulas, assets or the serializer change, and for engine benchmarks |
| [write-diataxis-docs](skills/write-diataxis-docs/SKILL.md) | To write user or developer documentation in the right Diátaxis form |
| [prepare-release](skills/prepare-release/SKILL.md) | To prepare the integration of a reviewed run or a release, with the commands for the owner |

## Maintenance

Edit only the canonical files here; the wrappers in `.claude/` repeat `name` and `description` and point to the source. After a change, check frontmatter and links, and test the skill on a concrete request: a valid format does not prove that a client loaded the skill or that the procedure works.
