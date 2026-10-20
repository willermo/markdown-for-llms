# a002 debt — experimental meter and maintained parser

Availability on 2026-10-17: some evidence and historical copies were removed by the authorized cleanup; the later references describe the historical delivery. Corpus/gold and decisive cases remain; no new human signature or closure of the debt. Details (archived: `documentation/development/cleanup-2026-10-17.md`, tag `pre-docs-2026-10-18`).

2026-10-16. **OPEN, resolution deferred for priority to the usable product**. [ADR 0008](../decisions/0008-maintained-parser-and-provisional-benchmark.md), closure and evidence (archived: `documentation/runs/run-a002-fase1-corpus/manifest.md`, tag `pre-docs-2026-10-18`). Accepted decision distinct from the implementation of the adapter, which has not started; the experimental core is verified separately.

## Owner and current resumption

Owner of planning/verification: the project supervisor. Disposition of the user of 2026-10-16: the debt remains open, but it does not block the experimental product with conversion/Web UI and declared limits. Resolution can be spread over time after a usable product; no mandatory automatic resumption at the exit of phase 3. The previous trigger is superseded by the [updated ADR 0008](../decisions/0008-maintained-parser-and-provisional-benchmark.md). Before introducing features that automatically attest properties dependent on these IDs, implementation, evidence and relevant reviews are still needed. No old review turned into GO. Trace r005 delivered, not reviewed, not a dependency of the first product; operational checks and guards kept separate. The later dispositions below keep the chronology of the previous path.

### Recognized debt — refinement of 2026-10-16

The five IDs remain technical debt to be consolidated as soon as we have the means and time; the product priority does not authorize deletion or completion labels. The [cross-cutting register](../development/technical-debt.md) also includes the a003 comparator and the trace not reviewed. Cleanup of the rebuildable environments/builds only: saved evidence and criteria kept; to resume the trace, the Haskell environment must be rebuilt, without claiming a MATCH of the old incomplete snapshots.

### Authorized early resumption — 2026-10-15

By explicit disposition of the user, the work is brought forward into the review of the a003 method, without waiting for the exit of phase 3. Status of the debt: **OPEN, experimental resumption on the syntactic roles**. The r002 plan must cover the feasibility of the shared maintained core, the specific adaptation of the a002 observed model and the resolution criteria of the five IDs, together with the adapter of the a003 comparator. The common evidence is reused; the closure outcomes stay distinct.

The originally planned opening of a dedicated run is not a prerequisite for this authorized resumption in a003. Implementation and evidence remain future; closure per ID only after new reviews and arbitration. No retroactive change to the r014 outcomes or to the experimental limits outside the syntax.

Receipt of the r002 plan, 2026-10-15: preliminary NOT_VIABLE of the profiles tested also for a002. The planner reports new syntactic premises and counterproofs of the reader on the cases D01–03; they are deliveries to be verified in the two reviews, not closures or arbitrations already carried out. The historical classifications of the table below stay attributed to the r014 closure. a003 status (archived: `documentation/runs/run-a003-fase1-engine-comparison/manifest.md`, tag `pre-docs-2026-10-18`).

Arbitration of the plan, 2026-10-16: circumscribed diagnosis accepted, errata requested. Counterproven on P/C the omission from the AST of HTML/images in the excess cell of a table: mandatory case D03/D04. This probe alone does not already prove a new adapter or a false COMPLETE in the callers; resources and residues must have explicit evidence. All five IDs remain open.

Verification of the errata, 2026-10-16: accepted the proof that the AST wrapper is insufficient for residues/resources and that the values-only witness is insufficient. A provenance experiment is ordered at the original points of the reader, with accounting of the raw and AST equivalence. No new adapter/audit or COMPLETE state validated; D01–05 remain open, the reduced historical closure does not change.

Receipt of the trace, 2026-10-16: the observational visit of the original parser reports resources in the discarded cells in the 14 checks and keeps provenance in the indentations tested. The two following reviews led to a NO_GO of the core for trim and verifier; core r002 then approved with two GO and an arbitration. Roles prototype r003 then arbitrated NO_GO: false HTML origins can hide resources without diagnostics; D-A002-04 INSUFFICIENT until the fix r004 ordered. Two r004 reviews then arbitrated NO_GO: wrong HTML extensions and certificates rejected on admitted inputs; sufficiency for the prototype not confirmed. Mandate r005 on the native terms, incomplete derivations and exclusive origins, with the visit referred to P/TagSoup and guards separate from the browser model. Notes and non-invalidated closures preserved, no closure of D01–05. Complete caller/model still absent; the positive gate does not close D01–05 and is not an audit of the product.

## Open items and status of the evidence

| ID | Residue | Status of the evidence and impact | Closure criterion |
| --- | --- | --- | --- |
| D-A002-01 | List→heading/div/quote/note and paragraph state, CLA-R014-I001 | Deterministic static trace; Pandoc combination not executed. Possible false COMPLETE/asset omitted in the model and false UNSAFE; no fetch/execution proven. | Pinned AST of the decisive inputs, then proof of the adapter on originals/continuations/positives and negatives; if a premise is disproved, record it as rejected with evidence. |
| D-A002-02 | Unclosed div, token metadata `+`/`._c`, indented ATX, CLA-I002/O001/O002 | Conditional hypotheses, not confirmed runtime defects. Do not promote them automatically to blockers or solved bugs. | Confirm/reject the AST premises; properties of the opaque ranges and guard on real HTML verified. |
| D-A002-03 | Span permissiveness in pipe_cuts, ChatGPT r014 hypothesis | Review incomplete, no validated finding. | Reachable input, requirement and impact attributed; explicit decision, no fix without evidence. |
| D-A002-04 | Global reliability of parser/guard/measure and coordinates | The previous closures hold in the domains tested; no general certification of COMPLETE/UNSAFE. | Maintained parser adapter, fixed profile/version, resources/text/formulas/numbers/order/assets and source-map where declared; explicit out-of-profile, no silent success. |
| D-A002-05 | Technical review r014 incomplete and administrative cost | GPT 41.042049 own CPU-s, 35.510746 in the repeated lookup; Claude 6.44 measured. No GO/GO pair. | Do not fabricate a retroactive review: new reviews and arbitration of the future implementation; indexed inventories/proportionate evidence. |

The profile limits F36B1–B4/A8/backtick-run/rawhtml5-htmlspace not confirmed/comment timing/reader fallback ID remain declared: define their treatment in the future contract; do not claim a universal parser or change the severity to obtain GO. L01 PDF-OCR and L02 compile not measured, L04 output UNAVAILABLE, L05 lower/sampled/hardware, Chrome NON PASS/helper12gap/H15/E9 are limits of the experiments, not all defects solvable with the AST. Engine/hardware comparisons belong to the respective increments of the roadmap.

## What is usable now

Corpus/gold and raw outputs with kept assets, provenance and finished results are usable for inspection/exploration. Metrics and judgements of the non-validated classes are provisional; do not base a final ranking or safety guarantees on them. No change to the software in the closure: audit/CLI technically keep producing the current states, this label limits the user's conclusions, it does not correct those states.

## Verification of the future solution

1. Fix the parser profile and verify the AST of the residual premises; preserve the disproved cases and the CodeInline/CodeBlock checks that are really inert.
2. Adapt the AST and the provenance; file/process guards independent of the fidelity judgement. No loss of the raw Markdown or of its assets.
3. Compare against independent gold files, original cases of the reviewers and finite variants; invalidated tests and received corpus, classified deltas and human review of numbers/formulas/assets/order. Coordinates only if verified.
4. Two new independent reviews and an arbitration with explicit criteria; update this register with ID/evidence/bytes/version. Only then status RESOLVED.

Do not use the AST of the same engine as a certification of its own conversion. Do not transfer the historical suite 3156+310 or the 64 recalculations to the new implementation.
