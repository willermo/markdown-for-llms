---
name: prepare-release
description: "Prepares the integration of a reviewed run into dev, or a release from dev to main: checks, changelog, documentation and the exact git commands for Davide. Use after a GO on the final review or when Davide asks for a release; it never merges, pushes or deploys by itself."
---

# Prepare an integration or a release

Preconditions: a GO from both reviewers on the identified diff, or an explicit request from Davide for a release. Davide runs commit integration, merge, push and promotion; this skill prepares them.

1. Verify the real git state: branch, HEAD, merge base with `dev`, working tree. If the branch differs from the reviewed object, evaluate the effect and reopen the necessary checks or reviews before going on.
2. For a release, check the roadmap criteria of the increment, the known limitations, migrations, packaging, model download, persistence, backup and restore for the declared profiles. Separate tested and unverified platforms.
3. Update the root `CHANGELOG.md` following [Keep a Changelog](https://keepachangelog.com/): user-visible changes only, under `Unreleased` until Davide tags a version. Update README and `docs/` only as needed to reach the new content, with migration and compatibility notes.
4. Give Davide numbered command blocks with the expected output of each step: integration into `dev` (fast-forward for a single commit, squash otherwise, always with `--ff-only` checks where applicable), then, for a release, fast-forward of `main` and the tag. Warnings on dangerous commands go right below the command.
5. After Davide reports the outcome, verify the resulting refs (local and, when he pushed, remote) and record the real commits or tags. Deploys happen only within an explicit operational scope.
