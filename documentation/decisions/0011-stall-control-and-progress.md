# ADR 0011 — Stall control and estimated progress instead of a total conversion deadline

- Date: 2026-10-20
- Decision status: accepted
- Implementation status: implemented on the branch `fix/conversion-blockers` after two independent reviews and integrated into `dev` at `d2e3bef` (squash) on 2026-10-20, push verified; not released on `main`; quality stays `unverified`
- Origin: first real use of I1, where an 11-page scientific article failed twice on the 450 s deadline and then converted in about 1540 s; decision of the owner of 2026-10-18 (no fixed total deadline by default, a stall guard, a progress bar in the UI and the CLI), with the run plan authorized on 2026-10-20
- Supersedes / superseded by: none; refines the worker limits and the configuration identity of [ADR 0009](0009-first-usable-product.md)

## Context

I1 stopped every conversion after `MDF_TIMEOUT_SECONDS` (450 s by default) and also set `RLIMIT_CPU` to that time multiplied by the threads. On a laptop CPU the engine needs about 140 s per page of a scientific article, so the limit allowed about ten pages against the 50 admitted by `MDF_MAX_PAGES`, and an expired deadline discarded work that was advancing. Changing the deadline required stopping every process and running `configure`, because every setting was part of the storage identity. The queue showed no progress at all.

The engine already reports progress: Marker 1.10.2 leaves the `tqdm` bars of Surya 0.17.1 enabled (`Recognizing Layout`, `Running OCR Error Detection`, `Detecting bboxes`, `Recognizing Text`, `Recognizing tables`) and `tqdm` 4.70.1 writes them to stderr, which the worker already reads from the isolated child together with stdout. The counts of a bar are known only when it starts, and `Recognizing Text` can appear up to three times (OCR, equations, tables).

## Decision

- **No total deadline by default.** `MDF_TIMEOUT_SECONDS` stays optional: 0, the default, means none; a positive value restores the deadline with reason `timeout` and an `RLIMIT_CPU` of deadline × threads. Without a deadline there is no CPU time limit, because a fixed one would be a hidden duration cap. Raster validation and the isolation probe keep their explicit 60 s and 10 s deadlines and have no stall guard.
- **Stall guard.** A conversion stops with reason `stalled` when the engine progress has not advanced for `MDF_STALL_SECONDS` (default 900, minimum 60), with the existing stop sequence (SIGTERM to the group, grace, SIGKILL). An advance is the start of a new bar or a higher count; redrawing the same line and log bytes are not. The clock is monotonic, starts with the child and is never persisted.
- **Progress source.** A pure module (`markdown_for_llms/progress.py`, no ML import) parses the default `tqdm` format in the worker, from the chunks of the existing pipe: no change to the child, its environment or the sandbox. A bar is read as soon as its closing bracket arrives, because `tqdm` writes `\r` before a bar and nothing after it; segments over 4096 decoded characters and unrecognized lines are counted and skipped. `tqdm` joins `runtime.PINS`, so a version drift makes the worker unready instead of producing false stalls.
- **Estimate.** Fixed stage bands (starting 0–5, layout 5–35, OCR error 35–37, detection 37–40, text 40–85, processors 85–95, finishing 95–99) change only when a bar starts, never go back, and the published estimate never decreases. The bands come from the single observed log; the phase and its count are exact and shown next to the estimate.
- **Persistence and exposure.** `runs.progress` (database schema 2) stores a bounded snapshot with wall-clock times, written in the lease update the worker already does (`COALESCE`, so a tick without a snapshot keeps the last one). `Service.get`, and so API, CLI and Web UI, exposes a `progress` field separate from `state`, `bundle` and `quality`: live for running and publishing runs, the last snapshot with `live: false` for stopped runs, `null` otherwise and for malformed data. The attempt receipt records the stall and deadline settings, the final progress, `progress_events`, `unparsed_segments` and `max_silence_seconds`, to recalibrate the defaults on real data.
- **Storage identity schema 2.** The identity hashes every setting except the time limits, which are recorded per run in the manifest: changing them needs only a restart. An archive with identity or database schema 1 is upgraded on its first opening, only by a process that takes the configuration lock exclusively (an opener refused while another process holds the archive asks to stop every process and restart), migrating the database first and rewriting `identity.json` last, with the previous identity kept in `history`. `configure` checks the expected identity before any upgrade and only changes the quota; `downgrade-storage` returns an upgraded archive to schema 1.

## Alternatives considered

- **A longer fixed deadline.** Simple, but it keeps an arbitrary cap that loses advanced work on larger documents and on slower machines; rejected by the owner's decision.
- **Liveness from log bytes or CPU time.** Bytes include redraws and warnings, and a busy loop consumes CPU without advancing: neither measures progress.
- **`TQDM_BAR_FORMAT` set in the child environment.** It gives a format chosen by the product but changes the diagnostic log and still depends on `tqdm`; kept as a fallback if the default format drifts.
- **Replacing `tqdm` in the child with a JSON emitter.** Invasive on third-party libraries; `PdfConverter` has no public progress callback.
- **Removing `MDF_TIMEOUT_SECONDS`.** Existing environment files would become invalid and the schema 1 identity could not be recomputed for the upgrade.
- **An explicit upgrade command.** Safer against a running old process but one more step for every user; the exclusive lock gives the same protection automatically.

## Consequences and verification

- A document can convert for as long as it needs; the user stops it with Annulla. Memory, output, process, file and log limits, isolation, lease and restart reconciliation are unchanged. Accepted risk: a child that keeps advancing can occupy the CPU indefinitely.
- 900 s is an initial default derived from one document on one laptop: the longest pause between two advances inside a bar was 82 s (the first `Recognizing Text` item, including warm-up). It is not a measure of every silence of the process, and steps without a bar (startup, pauses between phases, rendering) fall under the same threshold. It is not a guarantee on slower machines, with `MDF_THREADS=1` or on documents with long steps without a bar; the receipts record `max_silence_seconds` to recalibrate it.
- The parser depends on the `tqdm` format and on the Surya phase names of the pin; a contract test runs the real `tqdm` output through it and the pin blocks other versions.
- `downgrade-storage` drops the recorded progress and refuses archives born in schema 2, archives whose settings changed after the upgrade and archives with running runs.
- Verification: unit and contract tests in `tests/product/test_progress.py` and `tests/product/test_storage_identity.py`; a stall proof in the real Bubblewrap sandbox without ML; a real conversion of the 11-page article with the defaults (completed beyond 450 s, monotone estimate, no stall, `max_silence_seconds` recorded); a browser check of the bar, refresh, Annulla and restart; a real upgrade and downgrade of a schema 1 archive. Re-examine the bands and the threshold when receipts from other documents and machines are available, and the parser when the engine pin changes.

## Links

- [ADR 0009](0009-first-usable-product.md), [reference of the local product](../../docs/reference/local-product.md), [setup guide](../../docs/how-to/local-product-setup.md), [what «bundle disponibile» means](../../docs/explanation/experimental-product.md).
- Not a roadmap increment: I1 stays experimental ([roadmap](../roadmap.md)).
