# Recognized technical debt — deferred validation and product residuals

Updated 2026-10-17; status of P14 realigned on 2026-10-20. **OPEN, recognized, to be consolidated as soon as time and resources allow after a usable product.** Explicit disposition of the user: the deferral allows developing and using the converter and the Web UI with declared limits; it does not close the defects or turn old NO_GO into GO. [ADR 0008 in force](../decisions/0008-maintained-parser-and-provisional-benchmark.md).

| Area | Status and consequence | Consolidation required |
| --- | --- | --- |
| A002: D-A002-01–05 | Open; experimental reader/evaluator, no new COMPLETE/UNSAFE certification | Criteria and closure per ID in the [a002 register](../benchmarks/a002-residual-debt.md), relevant implementation/evidence and reviews |
| A003: comparator/projections | R007 NO_GO; measures of the sample partly corroborated, general qualifier not validated | Make the declared measures reliable, with independent sources/gold and explicit scope; no use as universal certification |
| Parser/trace and adapters | R005 delivered, not reviewed, core r002 GO in a circumscribed scope; no complete product adapter | Re-examine solution and cost when resumed, rebuild deleted environments where needed, new evidence and reviews on the actual bytes |
| Fidelity of contents/assets | No complete automatic certification; engine errors can escape the operational checks | Content checks and coverage of the properties actually promised; human evaluation and relevant regressions |

Phases 2/3 now have the first application increment I1 r004 **GO after two reviews and an arbitration**, integrated into dev at `f8bf44c`, push verified on 2026-10-17. They are not completed for the whole roadmap. The historical inconclusive attempts concern comparator, reader and validation of phase 1; their deferral has not been used to ignore the operational blocks of I1.

Owner: the project supervisor for planning and reviewing the debt, authors/reviewers for the execution. Re-examine the register at the first usable product and when a feature requires a property not yet proven. The user allows the work to be spread over time; this is not equivalent to deletion or abandonment. No automatism can attest a property that depends on the debt without adequate evidence. Finished job, bundle integrity and unverified quality remain distinct; paths, errors, files present and data retention remain necessary product checks.

The selective cleanup of 17 October (archived: `documentation/development/cleanup-2026-10-17.md`, tag `pre-docs-2026-10-18`) also removes superseded payloads and operational copies; it keeps register, outcomes, decisive cases and sources of the final prototype. Not all the old evidence remains available. The a003 summary (archived: `documentation/runs/run-a003-fase1-engine-comparison/manifest.md`, tag `pre-docs-2026-10-18`) declares availability and limits: historical hashes do not make deleted evidence present. The deferral must not become a promise of full reproducibility.

## I1 r004 operational residuals — low, open and not blocking

Review at the first product carried out on 17 October: the advanced debt above remains open and is not resumed automatically. The residuals below come from the Claude r004 review and are accepted by the GO arbitration; they are not resolved. Run summary (archived: `documentation/runs/run-a004-fase2-prodotto-utilizzabile/manifest.md`, tag `pre-docs-2026-10-18`).

| ID | Limit and reason for the deferral | Consolidation criterion |
| --- | --- | --- |
| CLA-R4-I001 | The broken UTF-8 test uses inconsistent inventories and goes through the rejection of the manifest; it no longer covers the intended branch. A probe with a consistent fixture confirms the correct behaviour of the product | Update both inventories and assert the UTF-8 cause and the invalid state persisted in the DB |
| CLA-R4-I002 | The GET diagnosis briefly takes worker.lock exclusively: a concurrent start can exit with Errno 11 even without another worker. Visible error, no data modified; repeating the start works. The frequency under stress is not a reliable estimate for real use | Diagnostics without ownership contention or a limited initial retry and a distinct message; concurrent test without spurious rejections |
| CLA-R4-I003 | An unsafe entry unrelated to the publication can make an integrity error be attributed to it and lose the recoverability of the same ID. Bytes kept. Reproduced with an unattributed symlink or in downloads; the normal flow tested reconciles the scratch of the next job first and does not produce that case. No recovery is promised from arbitrary alterations of the archive | The reserve/capacity errors on an unrelated entry defer publishing with a reason; once the entry is removed, the same ID is completed. Unsafe in the selected result stays blocking |

Owner of the review: the supervisor; implementer and two reviewers for the relevant fixes and verification. Resume at the next intervention on the areas involved and before widening the distribution or the management of the archive. If I003 appears in normal use, priority to the availability fix. Do not modify DB/storage by hand or add files to it to recover results or free quota.

Usability observations kept: diagnosis visible only in JSON/CLI, generic message for the manifest limit and a worker probe that only attests the presence of the owner, not the recovery of the single run. They do not become new gates or statements of correctness. No further automatic fix cycle; P14, the first use with the user's feedback, was carried out on 2026-10-18 on two PDFs ([ADR 0009](../decisions/0009-first-usable-product.md)): its findings feed the next increments and do not certify fidelity, and quality stays unverified.
