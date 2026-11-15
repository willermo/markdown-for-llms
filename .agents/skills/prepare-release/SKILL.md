---
name: prepare-release
description: "Prepares the integration of a reviewed run into dev, or a release from dev to main: checks, changelog and documentation (not during the transition, AGENTS.md) and the exact git commands for the maintainer. Use after a GO on the final review or when the maintainer asks for a release; it never merges, pushes or deploys by itself."
---

# Prepare an integration or a release

Preconditions: a GO from both reviewers on the identified diff, or an explicit request from the maintainer for a release. The maintainer runs commit integration, merge, push and promotion; this skill prepares them.

1. Verify the real git state: branch, HEAD, merge base with `dev`, working tree. If the branch differs from the reviewed object, evaluate the effect and reopen the necessary checks or reviews before going on.
2. Before giving the integration commands, rerun the full check (`AGENTS.md`, "Verification") on the commit to integrate, with a clean tree, and save its output in `temp/<run>/checks/` in the form given there; a red or missing check stops the integration.
3. During the transition the roadmap and the known limitations are not release criteria: use the acceptance criteria of the run and the maintainer's decision. For a release, check migrations, packaging, model download, persistence, backup and restore for the declared profiles. Separate tested and unverified platforms.
4. During the transition do not update `CHANGELOG.md`, `README.md` or `docs/` (`AGENTS.md`, "Transition until the new initial commit"); report the user-visible changes in the run report. Afterwards: update the root `CHANGELOG.md` following [Keep a Changelog](https://keepachangelog.com/): user-visible changes only, under `Unreleased` until the maintainer tags a version. Update README and `docs/` only as needed to reach the new content, with migration and compatibility notes.
5. Give the maintainer numbered command blocks with the expected output of each step: integration into `dev` (fast-forward for a single commit, squash otherwise, always with `--ff-only` checks where applicable), then, for a release, fast-forward of `main` and the tag. Warnings on dangerous commands go right below the command.
6. After the maintainer reports the outcome, verify the resulting refs (local and, when the maintainer pushed, remote) and record the real commits or tags. Deploys happen only within an explicit operational scope.
