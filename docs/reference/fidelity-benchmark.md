# Local conversion benchmark reference

**Status 2026-10-12: experimental tool.** A002 is closed by user acceptance of the reduced scope, not by a technical GO for r014. The reader can assign wrong COMPLETE/UNSAFE and omit references in the observed model: results are provisional in the classes not validated, with no universal attestation of quality/safety. Code and commands are not corrected by this label. Use gold/raw and human evaluation; see [debt and mandatory trigger](../../documentation/benchmarks/a002-residual-debt.md) and [parser decision](../../documentation/decisions/0008-maintained-parser-and-provisional-benchmark.md). The earlier implementation sections describe historical evidence/domains, not the general closing of the debt. No new campaign is started here.

For developers: protocol `conversion-fidelity-v1`, schema1 for engines and metrics; the documentary v2 packets/responses/verification have schema2. The helpers in `scripts/benchmarks/` measure the light local paths of the script pipeline; they are not services of the new application. [Protocol](../../scripts/benchmarks/PROTOCOL.md) for metrics, normalizations, states, limits and M/V responsibilities. [How-to guide](../how-to/fidelity-benchmark.md) to run a campaign.

## Launcher and commands

Start with a verified interpreter, `-I -B` and the absolute path of the launcher. It adds to sys.path only the trusted directory of the helpers, not the product root.

| Script | Command | Main mandatory options |
| --- | --- | --- |
| benchmark | inventory | --corpus |
| benchmark | profile | --format, --engine, --output |
| benchmark | run | --corpus, --document, --engine, --config, --output, --experiment-id, --attempt-id, --environment, --installation |
| benchmark | observe | --bundle, --output |
| benchmark | score | --gold, --observations, --audit, --assets, --format, --output; --verification optional |
| benchmark | evaluate | run options plus --gold and --assets; --verification optional |
| benchmark | summarize | --records (one or more final.json), --output |
| campaign | freeze | --corpus, --stage, --output, --experiment-id; --groups optional for a dependent rerun |
| campaign | run-group | --manifest, --group, --output |
| campaign | summarize | --manifest, --groups (all the declared group.json, in order), --output |
| contracts | — | --case, --corpus, --output, --environment, --installation |
| inventory | capture / verify | capture: --repo, --output, --extra repeatable; verify: positional manifest |
| prepare | — | --repo, --output, --label, --target |
| generate_corpus | — | --spec, --output, --environment |
| corpus | — | --first, --second, --destination, --external |
| reference_check | — | --corpus, --manifest, --output, --environment |
| packet | build | --corpus, --mechanical, --control-m, --output, --environment; --epub-render-packet optional |
| comparison | — | --corpus, --mechanical, --control-m, --epub-packet, --output |
| comparison_check | — | --packet, --corpus, --output (new JSON file) |
| verification | — | --packet, --response (imported response.json, repeatable in v2), --corpus, --output |
| output_packet | — | --manifest, --summary, --output |
| packet | import-response | --packet, --response, --packet-sha256, --output; v2 requires --author; --received-at and --receipt-method optional |

Output directories must be new; overwriting or an implicit resume are not admitted. The environment file is a map of strings or a target with an environment field. The installation is the I receipt of the canonical wheel, checked on the same interpreter as the children. Prepare is the operational launcher **local to this run**: it consumes a001 keepers already present, offline; it is not a portable installer for a clone without those keepers.

## Registry, profile and corpus

| engine_id | Formats | Method |
| --- | --- | --- |
| txt-literal-v1 | UTF-8 txt with a final LF | fence longer than every backtick run, exact payload |
| pandoc-direct-v1 | docx, epub, rtf, tex | /usr/bin/pandoc absolute/hashed, no filter/template |
| pandoc-legacy-v1 | docx, epub, rtf | PandocConverter of the installed wheel, child -I -B |

The eight group_id are direct-txt/docx/epub/rtf/tex and legacy-docx/epub/rtf. Freeze creates four attempt_id per path before the engines and protects by hash all the canonical files, assets, includes, gold, profiles and installation. The manifest does not freeze the OS page cache; every attempt has a new process and workspace. The complete bitset profile is in config, with 22 enabled and the complement disabled.

Permanent corpus in `tests/fixtures/conversion-corpus`: ten original synthetic MIT documents, separate font notices, manual transcriptions/AST/cells, controls K01–K03. PDF/OCR not run in the baseline. TeX has two literal includes and is not compiled. Numbers annotated in text and cells; math numbers/operators have the separate formula denominator. No universal OCR threshold.

## Receipts and interpretation

`run.json` records identity, conversion/bundle, exit, actual argv, resources, audit and times. `observations.json` is gold-blind with raw SHA256 and offsets. `metrics.json` keeps denominators, alignment, totals, failure/unavailable, localized diffs, CER/WER, asset content verification. `final.json` links these artifacts by hash. `group.json` requires four valid final files and keeps the original exits; the summary requires all the declared groups. The default is eight; an explicit dependent rerun keeps four attempts for each selected path.

A benchmark evaluate exit 2/3/5 can be a legitimate outcome accounted for by the harness with exit 0. An internal/preflight/schema/hash error or a missing record does not fall into this category. R always has expected-exit0. H01–H08 use declared stubs; H09–H16 require NON PASS according to the cause. A summary with exit 0 does not make quality PASS or conversion_batch COMPLETE. A10 requires four real Pandoc C02/C03 with the expected figure present, referenced and with verified content.

Legacy: internal Pandoc argv reconstructed, engine_exit null if the API does not expose it, Markdown serialized from the text returned after removal of the original temp, API images distinct from the inventory of the real files. Do not attribute this extract-media configuration to the legacy default; child/API latency distinct from the direct Pandoc latency.

## Gold and human judgements

The v2 documentary packet contains thirteen complete pairs: actual original (or actual pages/chapters) and an engine-independent reference, plus .md with local assets. Native MathML formulas, real empty cells, figures/captions, notes and references are visible. Locator/AST/hash/M stay in `technical/`; no JavaScript/CDN. `comparison_check` reads the actual HTML and Markdown with a logic separate from the serializer and verifies content, order, assets and sources. Canonical M stays distinct: captions/cells/figures/targets on the containers, manual AST against the TeX source, PDF fields against independent locators and extractors.

One natural response per document, for example «C02: corrisponde» ("matches"), is enough. `response.json` schema2 declares a documentary granularity: original text in `received.txt` with SHA, author and reception time attributed by the caller with their limits, packet and pair SHA, revision, complete scope, decision/differences. It contains no per-block signatures nor `accepted_judgments`. The digest of the pair binds all the sources/includes/render/assets, the HTML/.md/map reference and the CSS. It excludes only the global administrative revision: evidence of pairs with identical bytes can stay attributed to the revision really seen.

«Tutti i documenti corrispondono» ("All the documents match") is a complete collective statement on the packet. `statements` keeps a single original text with `document_ids` and `attribution=explicit_all_documents_statement`; the documentary decisions declare `response_attribution=projection_of_single_explicit_all_documents_statement`. Spaces and an attached article («idocumenti») are tolerated in the interpretation, with the original bytes unchanged. Exceptions, doubts or approvals of subsets do not become collective acceptances; the reclassification also verifies the scope.

ACCEPTED requires an explicit complete sentence; «ok» stays PENDING, impossibility or doubt AMBIGUOUS, differences REJECTED, different identities STALE. Duplicates on the same document are AMBIGUOUS. Unknown ID values stay unattributed. A rejected document does not cancel ACCEPTED on the others. A repeatable `verification --response` keeps the history: later explicit responses replace those on the same document, absences do not erase previous judgements.

VERIFIED requires M required PASS, complete view PASS and an explicit documentary V ACCEPTED on the same hash/scope; all thirteen VERIFIED for A03. An import reclassified from the original bytes and the hash of the receipt prevent altering the decision in the JSON. It is not a cryptographic authentication of the author. The original annotations stay immutable and the separate receipt can be passed to score/evaluate. Packet/import schema1 TSV stay compatible for historical artifacts, without converting them artificially into v2 signatures.

L04 for typography/semantics of the **output formulas** stays UNAVAILABLE if those judgements are not assigned, even when the structural comparison passes. The agreed duration is not V.

## Measurement limits

Conversion/audit/score/end-to-end are distinct times. wait4 user/system and individual/reaped maxrss do not coincide with the sampled tree RSS 10ms: shared pages, short processes, missed PID and peaks between samples are reported. Sampled RSS stop 6GiB, stream 1MiB, file 32MiB; no certified aggregate RAM hard cap. Only the local Linux/toolchain observed; no certification for GPU/other OS.

## r002 profiles and derived receipts

Reader `observations-v2-typed-inline`, scorer `metrics-v2-local-typed`: typed contents with raw/offset in cells and captions, link/images with a separate title, unsupported inline raw distinct from code. The audit sees every resource; the scorer compares inline components and the expected asset without double counting. Gold/spec rev5 make the semantic header explicit from source/policy; the empty pipe placeholder for headerless tables is not body. Captions are included in the transcriptions too. The finite AST equivalences hold for the formula alignment; WER includes unmatched insertions with an unchanged expected denominator. Unknown is attributed only locally according to `exact-visible-unique-v1`, without hiding omissions or pages. See the finite rules in the protocol.

Additional options:

| Script | Options |
| --- | --- |
| campaign freeze | `--verification` directory of the receipts frozen before the engines |
| comparison | `--metadata-delta` verifiable delta against the previous gold |
| derive_verification | `--old-corpus`, `--corpus`, `--old-packet`, `--packet`, `--response`, `--old-verification`, `--output` |

`derive_verification` produces schema3 `derived_reference_content_qualification`. It keeps the original V response, the previous packet/pair/gold, a new M and an independent checker of the old/new views. `new_human_confirmation=false` makes explicit the inheritance of V on the unchanged content; the technical header is qualified by M. A derived VERIFIED receipt is not a new judgement of the user. L04 stays UNAVAILABLE. A substantial visible change requires a targeted human scope.

The effective Pandoc PATH admitted is `/usr/bin`: the legacy token and the direct path have consistent realpath/hash/version before/after. Passive payloads outside the HTML/CSS/SVG/ZIP subset are INVALID_INPUT before the tool. The resource supervisor keeps its own config/receipt and all its descendants until deadline, reap or writer closing, without signals to extraneous concurrent children.

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

## Reader r009: historical boundary contracts

The r009 contract is historical: the following reviews found F43/F44, corrected in the r010 section. `observations-v9-reader-boundaries` corrected the r008 claims on the complete guard and on the domain of the labels after stripping. The shared lexical contract and the finite matrix are in the [r009 protocol](../../scripts/benchmarks/PROTOCOL.md#correction-r009-reader-boundaries-and-guard-precedence).

| Condition | Observation and audit | Exit (operational, quality) |
| --- | --- | --- |
| Unsafe opening after duplicates/depth/node limit | First PassiveHTML diagnostic in source order, `UNSAFE_REFERENCE` with detail | run/evaluate2, score3/FAIL |
| Apostrophe or quotes inside a bare value | End of the tag according to tokenizer 3.12.13; following tag validated and observed | Depends on the resources and on the other real tokens |
| Token-looking bytes in an attribute value | Literals, without leaves/references from the value | They do not introduce a new resource |
| Real incomplete prefix before an inert token | Local diagnostic `incomplete-start` and whole token | INCOMPLETE/UNSUPPORTED, run/evaluate2 |
| Real image before comment/CDATA/PI/declaration | URI/hash/dimensions of the asset and known text kept; opaque token | Bundle still INCOMPLETE because of the token |
| Label with code and markers in the quote/list/note continuations | Producer domain carried, single reference and original coordinates | In PROVISIONAL positives automatic PASS, score/evaluate5 |
| Closing in a following producer | No borrowing of the delimiter; outer blocks kept | Unknown/losses stay explicit |

The guard does not build the tree and continues after the first violation. The projection keeps depth16 and at most 20000 nodes; file 32MiB. The internal map of the prefixes serves `source_position.char/byte` without modifying raw or UTF8. The finite tests do not certify a universal parser. No change to scorer, denominators, F26 or the V qualification of the references.

Report and data of r009 (archived: `documentation/benchmarks/2026-10-10-a002-r009-confini-reader.md`, tag `pre-docs-2026-10-18`).

## Reader r010: inline HTML/Markdown boundaries

The r010 contract is historical: the reviews demonstrated F46–F48, corrected in the r011 section. The [r010 protocol](../../scripts/benchmarks/PROTOCOL.md#correction-r010-html-and-markdown-domains-in-inline-leaves) defines the finite domains. The lexer provides openings and intervals of the HTML subtrees to the guard and to the projection. Markdown unknown does not suppress the guard; closed code/math stay literal outside the HTML owner. The tree parser receives only the HTML leaf, including the balanced subtree, without the Markdown tail. Producer and source mapping r009 keep their boundaries.

| Case | Required result |
| --- | --- |
| Unclosed code/math or broken link followed by real unsafe HTML | Audit INCOMPLETE/UNSAFE_REFERENCE and detail; score3, run/evaluate2 |
| Same fallback and a safe non-representable resource | INCOMPLETE/UNSUPPORTED_RESOURCE_SYNTAX with an explicit diagnostic |
| Balanced `<span>` and closed code with literal img/script/onclick, in both orders | Text and outer blocks preserved, no reference/asset; PROVISIONAL automatic PASS, quality INDETERMINATE/score5 |
| HTML duplicates/depth with an outer closed literal | Unsupported error; the literal does not introduce unsafe |
| Real unsafe HTML after a projection error | First violation in source order, without stopping the following visits |
| Real closed/EOF token or closing in another producer | Opaque body and original coordinates; no borrowing of closings |

F36 is an accepted divergence: B1–B4 with an opener in the syntactic position of the prefix can produce tags/attributes in the pure tokenizer, but stay opaque tokens and UNSUPPORTED in the finite profile. There is no second pure feed as a gate. This limit does not admit resources lost with COMPLETE in the Markdown fallbacks. The finite tests do not attest every HTML/CommonMark combination.

The r010 report (archived: `documentation/benchmarks/2026-10-10-a002-r010-confini-html-markdown.md`, tag `pre-docs-2026-10-18`) records matrix, original oracles, CLI with OS exit, regressions and the historical recomputation. No change to scorer/F26, denominators or V; conditional counts and global bound are not interchangeable. L04 stays UNAVAILABLE on the output, V13/A03 concerns the reference; the other historical limits remain.

## Reader r011: ownership of inline leaves

`observations-v11-inline-ownership` (historical contract, corrected in r012), [r011 contract](../../scripts/benchmarks/PROTOCOL.md#correction-r011-ownership-and-inline-metadata).

| Domain | Verified rule |
| --- | --- |
| Heading block | Owns the final attributes; HC2 keeps metadata |
| Leaf already produced | `# … {…}` after stripping stays text; H7 distinguishes `.c` of the block |
| Finite label | Closed code crosses LF; closed math keeps an opaque body, formula/link and char/UTF8. A free LF is not an accepted label |
| Opaque token | Does not lend code/math closings; the recursion keeps the source barrier |
| Open HTML owner | Inner backticks are HTML text, all starts validated and first unsafe stable |
| Closed owner | Outer code becomes inert again; no HTML feed of the tail or of the raw already projected |
| Incomplete fallback | Real unsafe before unsupported; safe H3 at least an explicit INCOMPLETE/UNSUPPORTED |

F47 has a manual lexical gold with formula, score5/INDETERMINATE for the unsupported math subset and L04. F36 stays a deliberate divergence/UNSUPPORTED. Backtick limits, raw outside the profile and comment timing are received as non-blocking, without renderer/optimization. The observations-v1 fallback is distinct from the ID of the executed reader; use bytes/R/gate to attribute it. The r011 report (archived: `documentation/benchmarks/2026-10-10-a002-r011-ownership-inline.md`, tag `pre-docs-2026-10-18`) compares the same 153 oracles, real CLI, regressions and 64 measures without engines/V. Conditional counts, global bound and full measures are distinct, TRIVIAL_ZERO is not full PASS; L04 distinct from V13/A03, L01/L02/L05 and historical NON PASS preserved.

## Reader r012: labels and raw in containers

Historical contract, corrected for the complete order of the guard and the fence ownership by the [r013 reference](#reader-r013-guards-fences-and-private-data).

The [r012 protocol](../../scripts/benchmarks/PROTOCOL.md#correction-r012-atom-decisions-and-raw-in-containers) defines `observations-v12-shared-atoms`. Shared internal descriptors contain type/interval/body/metadata/owner; the recursions receive a slice of the same decisions, with the source-map of the producer.

| Class | Contract | Operational result |
| --- | --- | --- |
| Label with LF in closed code or escape+LF, closed math | Opaque raw formula, link/text/blocks/char/UTF8 kept | In the PROVISIONAL manual golds run0/evaluate5, quality INDETERMINATE |
| LF of a comment/CDATA token, open delimiter | The opaque token does not lend closings; real tag outside the token visited | Unsupported for the token; real unsafe with detail |
| Exact raw inline `{=html}` in the label | Metadata belongs to the atom; guard of the body before the projection | First global unsafe, run/evaluate2 |
| Ordinary code or code after the closing of the HTML owner | Literal body, no invented resource | Audit consistent with the other leaves |
| `{=html}` fence in quote/list/note | Guarded HTML body; raw structure of the container not projected | Unsafe 2/2 or safe INCOMPLETE/UNSUPPORTED 2/2 |
| Normal closed fence in the container | Literal body owned by the block | No opening from the literal tags |
| Incomplete projection or tree error | Coverage of the domain distinct from segments; explicit unknown residue | No COMPLETE with a lost resource; the following guards continue |

The accepted LF are decided once by the label walk; a lookup by index replaces the repeated LF/code loop. It is not a proof of a global linear cost. The finite measures record input, observed cost, hardware/cache and limits; CPU lower/RSS sampled do not certify an upper bound. Source-map and raw stay intact. No new public schema or universal parser. F36 B1–B4, F40/F41/F46, scorer/F26, V13/A03 and denominators kept. Formats html5/html-space, backtick run length, historical cost of comments and the ID v1 fallback stay r010 limits. L04 output UNAVAILABLE distinct, L01/L02 NOT_MEASURED, L05 and historical NON PASS.

Report and data of r012 (archived: `documentation/benchmarks/2026-10-11-a002-r012-atomi-e-container.md`, tag `pre-docs-2026-10-18`) report the actual evidence and the recoveries; no author GO or retroactive PASS for the static r011 traces.

## Reader r013: guards, fences and private data

Historical contract, corrected by r014 below. `observations-v13-ordered-guards-fences`; [r013 contract](../../scripts/benchmarks/PROTOCOL.md#correction-r013-guard-sources-and-fence-context).

| Element | Behaviour |
| --- | --- |
| Active HTML and Markdown metadata | Deduplicated events with an absolute position; all visited in source order, first Unsafe of the reader as detail |
| Active attributes / tree error | Attributes excluded from the values; a handled error does not truncate the guard of the tail |
| Metadata on LF | LF/CRLF separate tokens outside quoted values; a per-character map keeps the physical positions after quote/list/note stripping |
| Reader/audit | Markdown paths verified by the audit after observe: limit A8, no global unsafe order between the two |
| Tilde fence | Requires a start or a separation from the preceding paragraph |
| Ordinary closed backtick fence | Can interrupt text; an initial inline Code in lists keeps the state relevant to the following block |
| Ordinary EOF fence | No code ownership; real HTML guarded, explicit non-represented residue |
| Exact raw HTML/LaTeX | Received contracts kept, including finite differences from Pandoc; non-projectable container raw unsupported |
| Safe asset outside code | Reference/asset preserved or unknown with raw/position/resource_syntax; never COMPLETE with a lost resource |
| `_guard_coverage` | Private until consumed by node/located/source-map, absent from the final JSON even nested |

The precedence concerns the guards of the reader and not the order in which the tree is built. Real code/math owners and opaque tokens keep inertness; F36 and F44 do not change. The first heading line of a note can be Plain according to Pandoc 3.1.3: the context does not derive from the marker alone after stripping.

Evidence and limits of r013 (archived: `documentation/benchmarks/2026-10-11-a002-r013-guardia-e-fence.md`, tag `pre-docs-2026-10-18`): 285 main cases and 16 supplementary LF, 199 main AST and 16 attrs AST, two counter-tests, regressions, synthetic CLI and rescore of 64. The mere removal of 640 nested private copies does not change the 19 metric/gate fields. L04 output UNAVAILABLE, V13/A03 on the reference and the received limits and NON PASS remain. No general equivalence with Pandoc or certified semantic quality; new reviews and arbitration still needed.

## Reader r014: acceptance and status

`observations-v14-syntax-contracts`; the [protocol](../../scripts/benchmarks/PROTOCOL.md#correction-r014-syntactic-acceptance-and-uri-rejections) defines the boundaries of the adapter and of the finite grammar.

| Decision | Input consumed | Outcome |
| --- | --- | --- |
| URI | String and allow_external, pinned parsing | Path/None or Unsafe for policy rejections and ValueError of the parser |
| Metadata | Interval, accepted tokens, bound and host | Only the wholly valid group is opaque; HTML in the values stays a value |
| Fence | Info, marker, closing, state before the line | Code only with a proven owner; the inline fallback keeps its own grammar |
| Root block | Physical record of the scanner | The producer consumes the same state, also after quote/note/figure |
| Container | Admitted opener, host and character map | No block start invented by stripping alone |

The reader/audit order A8, the raw/F36 subset, the four r010 observations and the semantic verification of formulas are not extended. L04 output UNAVAILABLE stays distinct from V13/A03; L01/L02/L05 and historical NON PASS are kept. The r014 report (archived: `documentation/benchmarks/2026-10-11-a002-r014-contratti-sintassi.md`, tag `pre-docs-2026-10-18`) contains the status of the tests, the correction of HC2 and the precise citation of the baseline 132/133. It certifies neither universal interoperability nor an author GO.
<!-- r014: verified semantics of adjacent tokens -->

In the r014 metadata contract the tokens can be adjacent: `a="x"b="y"` is two pairs and `#chapter1.xhtml#start` is two ID declarations, the last being `start`, according to the pinned AST. The URI targets stay distinct strings. The r014 results (archived: `documentation/benchmarks/2026-10-11-a002-r014-contratti-sintassi.md`, tag `pre-docs-2026-10-18`) distinguish 16 EPUB deltas of failures/details from zero numeric/gate deltas and from the eight changes of the TeX div classes only. They do not constitute a new conversion.
