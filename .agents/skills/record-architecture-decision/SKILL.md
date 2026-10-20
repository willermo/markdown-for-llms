---
name: record-architecture-decision
description: "Records or updates architecture decisions of this converter in documentation/decisions/, keeping ADRs, the public roadmap and known limitations consistent. Use for choices about engines, document model, API, storage, frontend or distribution; not for small refactorings without architectural consequences."
---

# Record an architecture decision

1. Read the [ADR index](../../../documentation/decisions/README.md) and the relevant ADRs. Separate the current request, the choices already agreed and the hypotheses.
2. Describe the problem and compare realistic alternatives with proportionate evidence. For engines, models and APIs, separate declared capabilities from measured ones; cite up-to-date primary sources when the decision depends on capabilities, licences or tool versions.
3. Reuse the [template](../../../documentation/decisions/template.md) for a new significant decision (next free number); update an existing ADR when the topic is the same. Motivate the choice with the converter's constraints: fidelity, provenance, CPU/GPU, local or remote use, maintenance and verifiability.
4. Report the decision status (proposed, accepted, superseded, withdrawn) separately from the implementation status, and state the origin. Record as accepted a choice Davide has already authorized; do not ask for approval again. If a substantial decision is missing, record an explicit proposal and the open point, and continue the independent work.
5. Update the ADR index and the [roadmap](../../../documentation/roadmap.md) where relevant, without rewriting historical drafts. When an accepted choice is replaced, link the old and the new ADR. Priorities and process decisions belong to the project knowledge base, not to an ADR. Product open questions live in the project knowledge base, not in this repository: when a decision answers or raises one, report it to the supervision.
6. Check links and consistency between documents; report the choice and its practical effect.

Deliver the ADR with alternatives, consequences and a verification criterion. An ADR is not an implementation, a merge or a promise of an available feature.
