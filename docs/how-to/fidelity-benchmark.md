# Run the local conversion benchmark

**Status 2026-10-12: experimental tool.** A002 is closed by user acceptance of the reduced scope, not by a technical GO for r014. The reader can assign wrong COMPLETE/UNSAFE and omit references in the observed model: results are provisional in the classes not validated, with no universal attestation of quality/safety. Code and commands are not corrected by this label. Use gold/raw and human evaluation; see [debt and mandatory trigger](../../documentation/benchmarks/a002-residual-debt.md) and [parser decision](../../documentation/decisions/0008-maintained-parser-and-provisional-benchmark.md). The earlier implementation sections describe historical evidence/domains, not the general closing of the debt. No new campaign is started here.

Guide for developers who have the verified local toolchain and an R target with isolated network/socket. It uses only the redistributable corpus and the Pandoc already available locally; it does not install engines or download models. The [reference](../reference/fidelity-benchmark.md) and the [protocol](../../scripts/benchmarks/PROTOCOL.md) define formats and limits.

## Prepare the test

Follow the preflight of the [uv guide](uv-environment.md): origin of the managed CPython 3.12.13, verified startup, sdist/wheel rebuilt, canonical wheel reinstalled last, ten hashes and S/B/I inputs compared. Have Pandoc 3.1.3 and the local tokenizer cache available. The external inventory covers helper/corpus/tests/tool, besides the modules of the wheel. Do not carry an a001 PASS over to the new code.

These names are working variables to set to **your own absolute paths**:

```bash
RUN_BENCH_REPO=/percorso/markdown-for-llms
RUN_BENCH_STAGE=/percorso/stage-verificato
RUN_BENCH_PY="$RUN_BENCH_STAGE/dev/bin/python"
RUN_BENCH_OUT="$RUN_BENCH_REPO/temp/prova-benchmark-001"
RUN_BENCH_LAUNCH="$RUN_BENCH_REPO/scripts/benchmarks/launch.py"
```

Do not create RUN_BENCH_OUT before freeze. In this repository prepare can create an offline stage using the a001 keepers that already exist; see its prerequisites in the reference. On other clones use the local S/B/I procedure, without copying receipts of a different interpreter.

First run the standalone checks of the reader/writer under R:

```bash
"$RUN_BENCH_PY" -I -B -m pytest \
  tests/unit/test_conversion_benchmark.py \
  tests/unit/test_conversion_comparison.py \
  tests/integration/test_conversion_writer.py -q
```

The command above is the pytest payload; for the official test wrap it in the R wrapper as in the next step. A FAIL of the test must be fixed and repeated with new input/snapshot, keeping the outcome. Do not change expected values or the writer to make a conversion output match the gold.

## Freeze and start the groups

Freeze is administrative: no engine is started.

```bash
"$RUN_BENCH_PY" -I -B "$RUN_BENCH_LAUNCH" campaign freeze \
  --corpus "$RUN_BENCH_REPO/tests/fixtures/conversion-corpus" \
  --stage "$RUN_BENCH_STAGE" --output "$RUN_BENCH_OUT" \
  --experiment-id esempio-native-001
```

Before the first attempt include the new operational JSON files in the hashed inputs of the R target and in the technical snapshot, without overwriting the previous ones. The preparation of the run keeps the S/B/I already unchanged: the new snapshot adds the config/manifest files and the target includes those descriptors. If a helper or a product input changes, rebuild/reinstall and repeat the invalidated tests.

Use one group per R invocation, a fixed expected-exit0, and new R output and workload output. Example of payload with direct-docx:

```bash
"$RUN_BENCH_PY" -I -B \
  "$RUN_BENCH_REPO/scripts/diagnostics/run-a001-fase0-uv/run_offline.py" \
  --runner firejail --python "$RUN_BENCH_PY" --repo "$RUN_BENCH_REPO" \
  --target /percorso/target-con-config.json \
  --snapshot /percorso/snapshot-tecnico.json \
  --preflight-config "$RUN_BENCH_STAGE/preflight.json" \
  --output /percorso/R-direct-docx-001 --cwd "$RUN_BENCH_REPO" \
  --timeout 600 --expected-exit 0 -- \
  "$RUN_BENCH_PY" -I -B "$RUN_BENCH_LAUNCH" campaign run-group \
  --manifest "$RUN_BENCH_OUT/manifest.json" --group direct-docx \
  --output "$RUN_BENCH_OUT/direct-docx"
```

Repeat sequentially for the eight group_id listed in the reference. For a fix of the inputs use `campaign freeze --groups` with only the dependent paths: each keeps four attempts and new identities; the summary requires all the declared groups. Keep the previous complete campaign, without presenting the subset as eight paths rerun. Verify every R receipt, group.json and the four final.json. A sandbox permission error must be diagnosed using the authorized escalation of the tools, keeping isolation and quotas; it is not equivalent to R PASS. Do not change expected-exit after seeing the outcome. An engine timeout can be valid data only with the children terminated/collected; an R or harness timeout invalidates the campaign.

## Summarize and read the losses

Pass the eight group.json in the same order as the manifest:

```bash
"$RUN_BENCH_PY" -I -B "$RUN_BENCH_LAUNCH" campaign summarize \
  --manifest "$RUN_BENCH_OUT/manifest.json" \
  --groups "$RUN_BENCH_OUT/direct-txt/group.json" \
  "$RUN_BENCH_OUT/direct-docx/group.json" "$RUN_BENCH_OUT/direct-epub/group.json" \
  "$RUN_BENCH_OUT/direct-rtf/group.json" "$RUN_BENCH_OUT/direct-tex/group.json" \
  "$RUN_BENCH_OUT/legacy-docx/group.json" "$RUN_BENCH_OUT/legacy-epub/group.json" \
  "$RUN_BENCH_OUT/legacy-rtf/group.json" --output "$RUN_BENCH_OUT/summary.json"
```

For an official test also run summarize inside a dedicated R target. Read execution_complete, conversion_batch and quality separately; open metrics.json for diffs and unavailable. Check A10: four real C02/C03 repetitions with Markdown and the expected figure referenced, with verified content. A PNG name or the TXT path alone do not satisfy that check. Compare medians and ranges of the three repetitions with the separate first start, without calling it cold OS.

## Collect V and keep the evidence

M and the gold packet are separate from the workload. The v2 documentary comparison shows the original on the left and the complete reference Markdown on the right. Open C02 first, then the other documents from the index. There are thirteen pairs, with visible MathML formulas, real cells, assets and downloadable Markdown files; no Pandoc output is promoted to gold. For TeX, uncompiled sources are compared.

The user answers in the chat, per document: «C02: corrisponde» ("matches"), «C02: manca …» ("… is missing") or «C02: non riesco a valutarlo» ("I cannot assess it"). Keep exactly the text really received in a new UTF-8 file. An explicit statement «Tutti i documenti corrispondono» ("All the documents match") refers to the pairs of the packet: the importer keeps a single collective statement and attributes its scope to the documents, without inventing original lines or per-property signatures. A vague approval, or one limited to the documents already verified, does not accept all the documents. The operator attributes author and reception time from the real response, without asking for per-property timestamps or a new designation of the verifier. If the platform time is missing, the importer records the import time and declares that limit. Import the text:

```bash
"$RUN_BENCH_PY" -I -B "$RUN_BENCH_LAUNCH" packet import-response \
  --packet /percorso/confronto-v2-immutabile --response /percorso/risposta.txt \
  --packet-sha256 DIGEST_DEL_PACKET --author "$RUN_V_AUTORE_REALE" \
  --received-at "$RUN_V_ORA_RICEZIONE_REALE" \
  --receipt-method "messaggio reale ricevuto in chat" \
  --output /percorso/risposta-importata-001
```

The `verification` command joins M and the imported response.json to produce separate receipts per document, without modifying the annotations. `--response` is repeatable to keep previous partial responses; a document absent from a new response does not erase the judgement already received. A later explicit response on the same document replaces the previous one. The original text and its hash stay kept. For the v2 packet no TSV is filled in; the old schema1 importer remains for historical evidence. `output_packet` prepares the passive L04 inventory from the manifest/summary of the actual campaign; it does not assign or accept human judgements.

Verify status and identity/hash, not only exit. An incomplete V gold stays PENDING; agent_visual and an agreement on the duration do not complete it. Output formula semantics/typography require separate judgements, not this gold signature. Keep manifest, raw/errors, R/S/B/I receipts, observations/metrics and ledger in `temp/<run>/`; after the workload, record only the conclusions that support a decision in `documentation/benchmarks/README.md` (a dated file only when they are too long), as the skill `verify-conversion-fidelity` requires. The two code reviews are independent fresh chats under the development protocol of ADR 0010, which retired the supervised run cycle. Following the roles in `AGENTS.md`, the author commits locally on the feature branch, while integration, merge, push and promotion to `main` are actions of the owner.

## Updated metadata and correction of historical measurements

Include `tests/unit/test_conversion_fix_r002.py` in the targeted tests. Keep the bytes of the consumed helpers/config/tests and the before/after manifests, besides the hashes. The receipts of the children must use the same interpreter as the verified wheel. The guard also waits for writers that follow the exit of the leader; a writer within cap and deadline must complete normally.

To add header metadata without changing the approved contents, first preserve the previous corpus and packet. Check the new headers on the source with M; produce the new comparison with delta and an independent checker. Use `derive_verification` with the old corpus/packet/response/verification and the new corpus/packet: the schema3 receipt inherits the real V only on the unchanged contents, without a new human confirmation. Do not copy VERIFIED onto new hashes. Do not regenerate unchanged PDF/canonical/render files. If a visible content changes or the invariance cannot be proven, request a targeted verification.

Pass `campaign freeze --verification /percorso/receipt-derivate` when the reference qualification must be frozen before the engines. The attempt_id of the new campaign include experiment_id: the previous ones stay immutable. Freeze expected values and input/helper bytes before the workload, check them afterwards; R expected-exit0 stays fixed, the operational exits stay data.

To correct observations/audit/metrics without engines, create byte-identical copies of the historical raw, new identities and new observe/audit/score/final. Publish before/after with links to the original evidence; engine_runs0 is not a rerun of the conversion. The historical helper provenance that cannot be recovered remains an explicit limit. Keep this correction separate from the new baseline.

The r003 revision of the meter keeps formulas, unknown and resources through nested inline HTML and passive fallbacks. Unbalanced markup or resources outside the profile make the bundle INCOMPLETE; no evasive target is opened. Complete CER/WER are UNAVAILABLE also for a textual unknown observed unmatched; the known segments remain counts conditional on the structural matching, with unchanged expected denominators; they are not guaranteed bounds (correction r004). Cells/captions/formulas have local domains distinct from the text of the whole blocks. `WER_decomposition` exposes aligned edits, structural omissions and insertions with ID/types/text and exact caption/title candidates, without changing the alignment. Do not add up WER errors and caption/block gates as independent losses. For the versioned rule and the limits see the [protocol](../../scripts/benchmarks/PROTOCOL.md).

## r004 measures and profiles

Reader `observations-v4-html-coverage`: unsupported inline block structures with leaves/raw kept; explicit table/figure residues, duplicate captions without an implicit choice, all resources audited. A literal `<` in prose stays text. An absent/empty caption is equivalent in the text domain only; a non-empty addition fails. The [protocol](../../scripts/benchmarks/PROTOCOL.md) lists the finite profiles and the coverage policy.

Scorer `metrics-v4-conditional-and-bounds`: complete CER/WER stay None where the text is not measurable. Read `conditional_text_counts` and `WER_decomposition` v2 as counts under the structural matching, also for the compatible `known_word_edits`; they are not guaranteed bounds. `text_lower_bound` is separate: exact global distance on a measurable stream, zero guaranteed on any completion of a non-measurable stream. It keeps the expected denominators, exposes rule/scope/correspondences/ID and does not change alignment or FAIL for missing/page/cells/numbers/assets. The zero guarantee derives from the nonnegativity of the distance, not from interpreting the unknown as a wildcard in the matching. Compare fields of the same type/version.

## Corrections r005: HTML diagnostics and text domains

Reader `observations-v5-html-diagnostics`: the structural diagnostics of the parser are kept independently of the projection of the wrappers. An unclosed container or an inconsistent closing keeps raw, offset and type of the diagnostic: audit INCOMPLETE, run/evaluate exit 2. Balanced wrappers keep the projected contents only once. HTML headings h1–h6 have HTML coordinates; only the Markdown headings marked `atx_heading` receive the mapping of the ATX prefix. HTML headings and HTML code are hosts of `html-inline-v1`: nested block structures are local unsupported, keeping known leaves, formulas and resources. Literal Markdown code stays inert.

`html-tokens-v1` keeps comments, declarations and processing instructions as out-of-profile tokens, with raw and source interval, at the start, in the middle and at the end. They produce no prose or references from the enclosed characters; an image cited in a comment is not an active resource. The diagnostic reaches audit/scorer and prevents COMPLETE/PASS. The cells, captions and residues keep the local position of the token. Literal `<`, entities and clean HTML keep the previous treatment. The profile is finite, without fetch or execution from the data.

**Domain of `text_lower_bound`:** the field bounds exclusively the edit distance of the declared global stream. It is NOT a lower bound of complete CER/WER or of `conditional_text_counts`, which depend on the structural matching; it can exceed them because of order and separators. For example, two swapped paragraphs can have a structural CER of 0 and a positive global distance; an omission can add an LF separator to the stream, excluded from the expected denominator. `EXACT_GLOBAL_DISTANCE` is an exact measure in the measurable global domain; `TRIVIAL_ZERO` is only the conservative nonnegative guarantee over the admitted completions, not a complete measure of 0 nor evidence of fidelity. No change to the r004 alignment, normalization, denominators or distance computation.

## Correction r006: lexical boundaries and inert contents

The current reader is `observations-v6-html-lexical-boundaries`. The r005 description of F25 was incomplete: CDATA with an inner `>` and comments across blank lines or Markdown markers could leave part of the body to the following producers. The INCOMPLETE states already detected did not make counts and resources correct. The case `</>` could instead result in an automatic COMPLETE/PASS without a diagnostic. The historical evidence stays in the r005 report; consult the new r006 data.

Every token of the finite branch keeps the complete closed construct or runs to EOF. CDATA ends at `]]>`; comments follow `-->`, `--!>` and the abrupt branches of the local managed tokenizer, including its actual precedence among closings. Comment/declaration/unknown_decl/PI stay unsupported in `html-tokens-v1`: audit INCOMPLETE, run/evaluate 2. The body of comments and CDATA is inert: zero prose/formulas/references/assets and no opening of the target from the body. A text present only in the comment does not satisfy the visible gold nor hide missing.

The boundaries of the Markdown producers respect open tokens, also in cells, captions, headings, lists, quotes, notes, wrappers and raw HTML fences. Raw, Unicode/byte offsets, physical lines and diagnostics reach audit/scorer once. Leaves with code, formulas, links and raw LaTeX keep their own delimiters; EOF trim, attributes, emphasis and labels cannot truncate inert tokens. Inline code and literal Markdown fences stay inert. The summary without leaves uses a neutral reason and keeps the specific reasons of the occurrences. No fetch or command from the data; the subset does not promise universal HTML.

F26 stays unchanged: global stream bound distinct from structural CER/WER and conditional counts, same algorithms/normalizations/denominators. The r006 rescore creates new derived identities of the 64 protected raw, zero engines and 0 new V.

Report and data of r006 (archived: `documentation/benchmarks/2026-10-09-a002-r006-confini-lessicali-html.md`, tag `pre-docs-2026-10-18`).

## Correction r007: Markdown contexts and producer boundaries

The current reader is `observations-v7-shared-markdown-contexts`. The local lexical closing of r006 did not guarantee all the global boundaries: the lookahead of code, math and tags could cross distinct producers or resume inside fences; autolinks installed a false HTML context; metadata and raw TeX commands consumed inline could instead open tokens in the scanner. These are F31–F33, with real before/after in the new r007 report; the r006 report/data and FAIL/E9 stay unchanged.

The domain is that of the actual producer before the merge: a paragraph up to a blank/marker of a block, heading/cell/caption/label in its own value, lists, quotes and notes with the continuations really admitted. Scanner and consumer share finite code/math/autolink/link/span/metadata/command atoms and pipes outside the atoms. Autolinks do not install an HTML stack; metadata are opaque only in the recognized syntactic host; unsupported raw TeX stays local, with separate ref/anchor.

The intervals of the inert tokens really recognized on the source reach the consumers also through merge, prefix stripping, cells/captions and labels, including the EOF label without destination. A token closed whole or running to EOF extends the Markdown unit; it does not extend the lookahead of the preceding atoms into its bytes. The consumers remap offsets without global normalization. Comments/CDATA have zero prose/formulas/references/assets and openings from the body, and do not satisfy the visible gold. Comment/declaration/unknown_decl/PI stay unsupported html-tokens-v1, audit INCOMPLETE and run/evaluate 2. Literal code and safe positive metadata remain discriminating; real HTML, active resources and unsafe URI keep the guards.

It is a finite subset, not universal HTML. F26/algorithms/normalizations/denominators unchanged: global bound distinct from structural CER/WER and conditional counts; TRIVIAL_ZERO is not a complete measure of 0. New derived measures of 64 with the original r001/r002 conversions preserved, 0 new V and zero campaign engines. CLI probes with a synthetic writer and PROVISIONAL gold: automatic PASS/INDETERMINATE/evaluate 5, without a new V.

Report and data of r007 (archived: `documentation/benchmarks/2026-10-10-a002-r007-contesti-markdown.md`, tag `pre-docs-2026-10-18`).

## Correction r008: progress and inline domains

The r008 reader is `observations-v8-progress-and-inline-domains`; the r007 section above documents the historical round and must be read with this correction. R007 did not close the domain of raw LaTeX nor all the interactions in the same block or the code labels on soft lines. The r007 reviews were static: the r008 before/after are new author evidence, not runtime attributed retroactively to the reviewers.

Raw LaTeX fence: the whole body up to the real physical closing or EOF is the domain, also after a blank/marker; inert tokens cross false closings in their own body. Every transition of the scanner advances strictly. EOF keeps the unknown fence and the inner tokens, without making the whole of TeX supported.

Incomplete code/math/tag delimiters do not skip real comments/CDATA in the same domain. The consumers receive the whole token, a body without prose, math, ref, assets or openings, and independent missing/page2. An incomplete opening tag before the token stays diagnosed `incomplete-start`; the recognized tags are validated with the same PassiveHTML before any projection. Quoted attributes, closed code/math, currency/literal< and active resources remain discriminating.

The code intervals accepted by the label walk are shared with the scanner, also on the soft lines of the quote/list/note producers. Destination/title stay separate; a closing is not taken from an extraneous block. Trim and prefix stripping use the source ledger: no reparsing of a partial line that starts inside a literal. Raw/char/byte/text/reference and outer blocks are verified with Unicode, positives and real EOF tokens outside the code.

Finite subset html-tokens-v1 unsupported, audit INCOMPLETE/run/evaluate 2; R PASS is isolation/custody, not quality. CLI with a synthetic writer and PROVISIONAL gold: automatic PASS, quality INDETERMINATE/evaluate 5, without a new V. Rescoring of 64 with new r008 ID values and the original r001/r002 conversions preserved: zero engines and no new V. Scorer/F26/normalizations/denominators unchanged; conditional counts separate from the global bound, TRIVIAL_ZERO not complete 0. L04 output UNAVAILABLE distinct from V13; L01/L02/L05, Chrome NON PASS and the 12-helper gap remain.

Report and data of r008 (archived: `documentation/benchmarks/2026-10-10-a002-r008-avanzamento-scanner-e-domini-inline.md`, tag `pre-docs-2026-10-18`).

## Check a result with the r009 reader

This section records the r009 checks. The current reader and the F43/F44 corrections are described in the following r010 section. The r009 correction fixes the HTML guard interrupted by tree errors, the false incomplete tag with a bare apostrophe and the loss of labels after stripping. The r008 evidence stays bound to its own bytes.

1. Read `audit.json`: `UNSAFE_REFERENCE/detail` precedes unsupported when there is a real unsafe opening, also after duplicates or an excessive depth.
2. Check text, references and real assets in `observations.json` and audit. A following comment can leave the bundle INCOMPLETE even if the preceding image is kept and verified; do not interpret COMPLETE as fidelity.
3. In the quote/list/note transports compare `raw`, the offsets of the block and `source_position.char/byte` with the UTF8 source. The markers inside the code of the label are content; a delimiter in another producer stays outside.
4. Read metrics and exits separately. A PROVISIONAL synthetic positive has automatic PASS, quality INDETERMINATE and score/evaluate5; a real loss stays FAIL. Do not assign V or full PASS from a TRIVIAL_ZERO bound.

For the detail of boundaries, limits and precedences consult the [r009 reference](../reference/fidelity-benchmark.md#reader-r009-historical-boundary-contracts) and the r009 report (archived: `documentation/benchmarks/2026-10-10-a002-r009-confini-reader.md`, tag `pre-docs-2026-10-18`). The CLI probes use a declared synthetic writer; the 64 derived measures reuse protected raw and do not run new engines or human judgements.

## Check the HTML and Markdown boundaries with the r010 reader

This section keeps the r010 checks, refuted in the F46–F48 interactions and corrected in the r011 section. It used the current installation, S/B/I and R gates. The r009 reviews identified tags hidden by the fallbacks and false unsafe in the code following balanced HTML; the historical v9 results do not certify these boundaries.

1. Compare the actual HTML opening with its domain in the raw. After an incomplete delimiter or a malformed link, real traversal/script/event must give `UNSAFE_REFERENCE` with detail before the unsupported. If the fallback loses a safe image, it must stay at least `UNSUPPORTED_RESOURCE_SYNTAX`, without COMPLETE with a vanished resource.
2. For `<span>words.</span>` followed by closed code, verify the text of the literal, the outer blocks and the absence of invented reference/asset. In PROVISIONAL synthetic positives the result is run0, automatic PASS and quality INDETERMINATE/score/evaluate5. A real unsafe tag outside code must still give run/evaluate2 and score3.
3. In the quote/list/note/cell transports verify raw, char/UTF8 coordinates and order. The closings of another producer do not close the current leaf; comment/CDATA keep the opaque body, also at EOF.
4. Read F36 as a finite limit: B1–B4 can be UNSUPPORTED while the pure `HTMLParser` recognizes openings. Do not extend that result to Markdown fallbacks nor declare universal parity with the pure parser.

The [r010 reference](../reference/fidelity-benchmark.md#reader-r010-inline-htmlmarkdown-boundaries) and the report with data (archived: `documentation/benchmarks/2026-10-10-a002-r010-confini-html-markdown.md`, tag `pre-docs-2026-10-18`) identify examples, oracles, CLI and limits. The writer of the CLI probes is a declared synthetic one; the recomputation of the 64 raw is a new derived measure, without engines or a new V. TRIVIAL_ZERO does not mean a complete measure at zero; L04 on the output stays distinct from V13/A03 on the reference.

## Verify ownership and metadata with the r011 reader

This section keeps the r011 checks; the reviews found F50–F52. Use the following r012 correction for the current reader.

1. Separate the attributes of the heading block from the text `# … {…}` inside the leaf of the producer. Verify H1/H6 run2/evaluate2/detail unsafe, HC2 metadata and H7 `.c` attributes. For H3 the safe fallback must give at least an explicit INCOMPLETE/UNSUPPORTED, without COMPLETE with a lost resource.
2. In the label really accepted, verify closed code on LF and an intact raw formula, link, three blocks, char/UTF8 coordinates and no reading of literal img. A free LF does not become a label. The math gold includes the independent formula; score5/INDETERMINATE does not certify the subset/L04. Opaque Comment/CDATA must not lend closings to the preceding literal.
3. Inside the HTML owner count all the actual openings, including the first inner unsafe, and verify the diagnostic before the outer unsafe. Invert the order of the payloads. After the closing of the span the outer code stays literal; a complex quote already guarded does not reread the raw HTML.
4. Check output/final and real OS exits; the CLI writer of these cases is a declared synthetic one, PROVISIONAL gold, without engines or a new V. Keep the expected H12 FAIL and the earlier tests invalidated. Recompute the 64 protected raw and verify the chain of original, previous derived and new measure without imposing counts. TRIVIAL_ZERO is not equivalent to full PASS.

[r011 reference](../reference/fidelity-benchmark.md#reader-r011-ownership-of-inline-leaves) and report/data (archived: `documentation/benchmarks/2026-10-10-a002-r011-ownership-inline.md`, tag `pre-docs-2026-10-18`) identify oracles and limits. F36 stays deliberately UNSUPPORTED; no universal HTML/CommonMark parity. The observations-v1 fallback is distinct from the actual reader: check bytes/R/snapshot. L04 output distinct from V13/A03, L01/L02/L05, the historical NON PASS and the non-blocking observations stay declared.

## Verify labels and raw in containers with the r012 reader

Historical procedure: for guards and fences also apply the [r013](#verify-the-guard-sources-and-the-fences-with-r013) checks.

After the current managed 3.12.13, S/B/I and R, verify the physical reader `observations-v12-shared-atoms` through bytes and receipts.

1. Try an accepted label with backslash+LF, code on LF and a combination of the two. For closed math compare raw formula, link, text, order of the three blocks and char/UTF8 coordinates. The synthetic result is run0/evaluate5 and quality INDETERMINATE. Remove the final delimiter: the real tags must receive the guard and give UNSAFE with detail. A free LF and an LF in the opaque token stay distinct checks, without borrowing closings from the token.
2. Insert the exact inline raw `{=html}` in the label and two violations in inverse orders. Check the detail of the first opening, besides the exit2/2. Ordinary code stays literal; an open HTML owner owns the content. The visits of the other domains continue after the first unsafe or a tree error.
3. In quote/list/note try a `{=html}` fence with script, evasive img and a safe asset. The first two give UNSAFE with detail; the third stays INCOMPLETE/UNSUPPORTED because the raw block of the container is not projected. Check the diagnostic and the position of the raw, without considering `segments` an evidence of coverage. The ordinary closed fence keeps the literal body.
4. Read output/final and OS exits, R receipts and frozen inputs. The CLI of the fixtures use declared synthetic writers, without new conversions or V. Keep the expected H12 FAIL separate from corrected failures. The recomputation of 64 is a derived measure: preserve the original conversion_final r001, the previous_derived_final and the new result; do not impose quality counts.

[r012 reference](../reference/fidelity-benchmark.md#reader-r012-labels-and-raw-in-containers) and report with data (archived: `documentation/benchmarks/2026-10-11-a002-r012-atomi-e-container.md`, tag `pre-docs-2026-10-18`) identify cases, decisions and results. The finite tests do not certify universal Markdown/HTML, Pandoc on the excluded raw formats or semantic fidelity. F36 and four r010 observations, L04 distinct from V13/A03, L01/L02/L05 and historical NON PASS stay declared.

## Verify the guard sources and the fences with r013

Historical procedure: complement it with the r014 checks below.

Identify `observations-v13-ordered-guards-fences` with bytes, origins and receipts. After every change rebuild sdist/wheel, reinstall the canonical wheel, compare ten modules and the actual helpers and verify S/B/I before importing the workload under R. An old STALE general snapshot does not carry PASS over to the new version.

1. Freeze raw and manual expected values before the runtime. Try unsafe HTML before and after active metadata of heading/div/fence/figure and inline code/link/span, also in cell/caption/quote/list/note. Record a single observe with snippet, char/UTF8, sequence and exact multiplicity; run the audit separately. Verify the tail also after a tree error and the reset on a following safe call. In metadata on LF try both a separator between attributes and a multiline quoted value: after stripping the prefixes, compare the position of each active token with the physical raw, not with the logical offset.
2. Vary preceding context, tilde/backtick, closing/EOF and container. Under R use the Pandoc 3.1.3 already installed (`-f markdown -t json`) to verify the owner of the discriminating body. Any Code elsewhere in the document is not enough. Keep input/AST/argv/version/hash and the corrections of the premises; do not derive the expected values from the reader under test.
3. Outside a real code owner verify script/URI with detail and the safe image as a preserved resource or an explicit unknown with position. Separate the received raw HTML/LaTeX profile from the ordinary syntax; preserve LF label, F36, F44 and opaque tokens. Do not add a second indiscriminate HTML feed to the Markdown residues.
4. On private copies try the two wrong assumptions: fail-fast attrs and a fence at every line. The expected values must detect errors of detail/visits and asset/unsafe loss. Do not alter the corpus or the sources for the test.
5. Compare the observations JSON and the 19 metric/gate fields separately. The private `_guard_coverage` must disappear only from the final serialization, keeping cell/location/reference. Also check the A8 limit: the reader precedes the audit check of the Markdown paths.

The r013 delivery (archived: `documentation/benchmarks/2026-10-11-a002-r013-guardia-e-fence.md`, tag `pre-docs-2026-10-18`) contains the source → property → test table and the outputs executed. The test CLI uses a declared synthetic writer and PROVISIONAL gold; the rescore of 64 does not start engines or new V. Keep H12 FAIL separate from the ordinary corrected failures. Fast suite and finite matrix do not replace new reviews, a complete acceptance test or the semantic verification of the L04 formulas.

## Verify acceptance and status with r014

After the preflight and R, compare the same raw with reader and V13 adapter archived together: an old reader imported with the new adapter does not represent the original baseline. Keep hashes and copies before the changes.

1. For malformed URI, alternate script and safe/evasive resources before and after the error. Verify the first diagnostic, every actual validation and the reset; audit must provide INCOMPLETE/UNSAFE_REFERENCE, without an internal error.
2. For the fences, vary valid/invalid info and the context outside the producer: attached/separated paragraph, lazy quote/list/note, figure. Verify the AST of the discriminating body, keeping also the real backtick inline Code.
3. For the metadata, try valid/invalid groups and literal HTML in the values in every producer. Compare intervals, raw and UTF8/LF/CRLF coordinates; valid active attributes must be rejected, invalid braces do not hide tags.
4. Use four private copies to refute escaped URI parsing, any fence info, a state reset in the caller and any metadata brace. Keep diff, base and results; do not mutate the product or the corpus.

The r014 outcomes (archived: `documentation/benchmarks/2026-10-11-a002-r014-contratti-sintassi.md`, tag `pre-docs-2026-10-18`) distinguish manual expected values, AST, synthetic writer and checks on the evidence. For the boundaries consult the [reference](../reference/fidelity-benchmark.md#reader-r014-acceptance-and-status). Do not carry PASS over from S/B/I or from evidence bound to different bytes.
<!-- r014: actual deltas and attribution -->

To compare the r014 results with r013, also consult the attributed deltas (archived: `documentation/benchmarks/2026-10-11-a002-r014-contratti-sintassi.md`, tag `pre-docs-2026-10-18`): 16 EPUB change failures/details because of the correct reading of adjacent ID values, without changes to the numeric values or the gates. Eight TeX only correct the classes of the observed divs. Do not modify the gold to cancel these differences.
