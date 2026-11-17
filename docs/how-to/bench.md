# Prepare and run the reference bench (`mdf bench`)

`mdf bench` prepares the reference set of documents and the checks that score conversions (it builds the reference toolchain, downloads and verifies the documents, and generates and self-checks the checks), then measures the routes of the conversion engine on it: `mdf bench run` converts each document on each local route of its format through the engine and writes a report with the scores, times, memory and problems of every attempt; with `--route datalab` it also measures the remote, paid Datalab route on the public set, under two spending caps. `compare` and `crosscheck` read the reports, a personal set measures your own documents, and `secret-scan` checks that the Datalab key is in no file. The scores are provisional and do not certify fidelity: the quality of a conversion stays `unverified`.

For developers on Linux. The bench is a development tool, separate from the experimental product (`mdf convert`, API and Web UI) and from the a002 helpers of the [local conversion benchmark](fidelity-benchmark.md).

| Command | What it does today |
| --- | --- |
| `mdf bench prepare --toolchain` | builds the reference toolchain image locally, verifies it file by file and self-tests it (also `mdf prepare --toolchain`: the image is the toolchain of the conversion engine too) |
| `mdf bench list` | lists the documents of the manifest |
| `mdf bench fetch` | downloads and verifies the documents, their twins and derived inputs |
| `mdf bench checks` | generates the checks from the twins, from the cache only |
| `mdf bench checks --self-check` | also renders each twin with a third-party tool and judges the checks |
| `mdf bench run` | converts the documents on the local routes of the engine (on Datalab with `--route datalab`), scores the outputs and writes the report; `--resume` resumes an interrupted run |
| `mdf bench report RUN_DIR` | writes the report of a finished run again from its saved outputs, without converting or sending anything |
| `mdf bench compare A B` | compares two reports attempt by attempt; `--fail-on-regression` exits with 1 on a regression |
| `mdf bench crosscheck R...` | lists the checks that every route fails in the same way (suspects) |
| `mdf bench secret-scan` | searches the Datalab key in files, without printing it |

## Prerequisites

- **Linux x86_64**: tested. Other systems are not.
- **Docker**, with the `buildx` plugin, reachable through a local socket (`unix://`) by your user. Tested: Docker Engine 29.9.3 with buildx 0.39.0 on Ubuntu 24.04, user in the `docker` group (which is equivalent to root on that machine). The bench refuses a remote Docker endpoint, because it would receive the documents. Docker Desktop on macOS and Windows is **not tested**; note its licence: Docker Desktop is free for personal use, education, non-commercial open source projects and businesses with fewer than 250 employees and less than USD 10 million in annual revenue, while larger organizations need a paid subscription. Docker Engine on Linux has no such licence.
- **uv 0.13.7** (the `required-version` of `pyproject.toml`) and the uv-managed CPython 3.12.13 under `.venv-python/`, prepared as in [Prepare and update the uv environment](uv-environment.md).
- **No other system tool for preparing the set**: pandoc, poppler, qpdf, Ghostscript, DjVuLibre and LibreOffice run inside the toolchain image, never from the host `PATH`.
- **For the measured run (`mdf bench run`)**, what the conversion engine needs ([Convert documents](convert-documents.md)): Bubblewrap at `/usr/bin/bwrap` (`isolation_unavailable` without it), a systemd user manager (`systemd-run --user --scope`, which caps the memory of the Marker child) and the 18 weight files of the Marker route, prepared once with `mdf prepare --weights` (`weights_missing` without them).
- Disk space: about 1.5 GB for `.venv-bench`, 1.4 GB for the image, and the build cache of the BuildKit builder (it can grow to several GB over repeated builds).

## Create the bench environment

The bench has its own venv, `.venv-bench/`, with the `bench` extra (Marker 2.2.0, Surya 0.23.0, torch 2.15.0 and torchvision 0.29.1 for CPU, markdown-it-py, mdit-py-plugins). The development venv never receives this extra. At the root of the clone:

```bash
(
  RUN_REPO="$(pwd -P)"
  unset VIRTUAL_ENV
  export UV_PYTHON_INSTALL_DIR="$RUN_REPO/.venv-python" UV_PYTHON_DOWNLOADS=never
  export UV_CACHE_DIR="$RUN_REPO/.cache/uv-bench" UV_PROJECT_ENVIRONMENT="$RUN_REPO/.venv-bench"
  uv sync --locked --no-default-groups --extra bench --no-editable --python 3.12.13 --managed-python --no-python-downloads
  "$RUN_REPO/.venv-bench/bin/mdf" bench list
)
```

`--locked` stops when `uv.lock` is out of date. The first sync downloads the packages from PyPI and the PyTorch CPU index. No command of the bench downloads models today. After a change to the code, repeat the same `uv sync`: it rebuilds and reinstalls the package from the working tree. The examples of preparation, listing, download and checks below call `.venv-bench/bin/mdf` from the root of the clone.

The examples of the measured run and of the commands that read its reports (`run`, `report`, `compare`, `crosscheck`, `secret-scan`) call the development venv instead, `.venv-dev/bin/python -I -B -m markdown_for_llms.cli`: it is the venv of the product, built as in [Prepare and update the uv environment](uv-environment.md), where the full check installs the wheel of the commit, and it has the CPU engine with Marker (extra `engine`) and the Markdown reader of the bench. A run records the identity of the package it runs from, and it is that venv that the full check locks (`venv_busy`). `.venv-bench/bin/mdf` accepts the same commands.

## Build the reference toolchain

```bash
.venv-bench/bin/mdf bench prepare --toolchain
```

The command checks the build context `scripts/bench-toolchain/` of the checkout against the identity packaged in `markdown_for_llms/engine/data/toolchain.json` (it refuses a changed, added or missing file), creates the BuildKit builder `mdf-bench-toolchain` (memory cap 6 GiB) when it is absent, builds the image, verifies every file of the image against its content manifest and runs a self-test of the tools. Only then does it tag the image `mdf-bench-toolchain:<first 16 characters of the content SHA-256>`.

Everything happens on your machine: no account, no registry login, no push. The build needs the network: the BuildKit image and the Debian base image from Docker Hub, by digest, a Debian snapshot, conda-forge, the pixi release on GitHub, DjVuLibre on SourceForge and the LibreOffice archive, each input pinned by digest or SHA-256. A build from scratch took about 2.5 minutes on the tested laptop (161 s), and the whole command about 3 minutes. Expected output (abridged; the image ID of your build differs):

```text
toolchain context ok 4 files (Dockerfile, content-manifest.sh, pixi.lock, pixi.toml)
toolchain builder ok mdf-bench-toolchain memory 6g
toolchain build ok sha256:<image id> in 161 s
toolchain content ok dcc2395ac8922f90e926952475e4a32f43555afd5b0dcbd5b8204e46f42d2c84 (48792 manifest lines, every file verified)
toolchain reference build no (same content, another build)
selftest versions ok ddjvu 3.5.29 qpdf 12.4.2 pdftoppm 26.11.0 pdftotext 26.11.0 pdfinfo 26.11.0 pandoc 3.14 soffice 26.8.3.1 gs 10.08.0
selftest isolation ok host path not visible, the same file mounted is
...
selftest timeout ok container removed by name
toolchain ready mdf-bench-toolchain:dcc2395ac8922f90 sha256:<image id> docker 29.9.3
```

The identity of the toolchain is the content SHA-256 (`dcc2395a…`), not the image ID: every build has a new image ID, and `reference build no (same content, another build)` is the normal result of a local build. `--no-cache` builds without the builder cache; `--context DIR` reads the build context from another directory (default: `scripts/bench-toolchain` of the current directory). `mdf bench prepare` without `--toolchain` prepares nothing and exits with 2.

Each later command verifies the image at its first tool call and runs the tools with `--network none`, a read-only root, no capabilities, your user and a memory cap; only the input file of a call is mounted.

## List the documents

```bash
.venv-bench/bin/mdf bench list --level quick
```

```text
manifest sha256 b14035da7a3a68cc822dd020c61010099c75d2824b8988885daae8192c436067, 7 of 36 documents
D04  quick,standard   scientific pdf      12  jats            The potential carcinogenicity of orthopaedic implants - a scoping review
D08  quick,standard   office     pptx    ~12  pptx            Opening up your scientific contributions thanks to Creative Commons licenses
D15  quick,standard   scientific docx     ~5  docx            Comparison of ergonomic methods for inoculation of microorganisms (Ukrainian)
D18  quick,standard   scan       png       1  wikisource      La geometria non archimedea (Veronese), page 3 as an image
D19  quick,standard   legal      pdf       7  akoma-ntoso     D.lgs. 8 ottobre 2026, n. 194 (GU SG n. 248 del 24/10/2026, pp. 3-9)
D20  quick,standard   scan       djvu     12  wikisource      La geometria non archimedea (Veronese)
D35  quick,standard   office     txt      ~1  oracle          Le avventure di Pinocchio: Storia di un burattino (excerpt)
```

The columns are id, levels, profile, format, pages (`~` marks an estimate), the kind of twin (the independent source of the checks, `oracle` for a text judged on its expected decoding alone) and the title. The levels are `quick` (7 documents), `standard` (34) and `long` (D17 and D21, the same book of 272 pages as a PDF and as a DjVu). Without `--level` the command lists all 36; `--json` prints JSON, with the language (BCP 47) and the scripts (ISO 15924) of each document; `--manifest FILE` reads another manifest.

The set covers every format of the conversion engine: pdf, png, jpeg, tiff, djvu, docx, xlsx, pptx, odt, ods, epub, doc, rtf, html and txt. Several inputs are derived from another document of the set (D22-D25 are D10, D12, D07 and D11 saved by LibreOffice as doc, ods and rtf; D27 and D28 are page 3 of D14 as JPEG and TIFF), and the txt inputs test the encodings: D29 and D30 are the text of D15 in UTF-8 without a byte order mark and in UTF-16 with one, D31-D34 are Project Gutenberg texts recoded to Windows-1252 (Italian), 1250 (Czech), 1251 (Russian) and 1253 (Greek), D35 and D36 short excerpts of D31 and D33. A recode never replaces a character: a text that a code page cannot hold whole is `recode_lossy`. The manifest is `mdf-bench-manifest/2`; a manifest of schema 1 is refused with `manifest_schema_unsupported`.

## Download and verify the documents

```bash
.venv-bench/bin/mdf bench fetch --level standard
```

```text
D01 available
...
D36 available
34 available, 0 unavailable (manifest b14035da7a3a68cc)
```

`fetch` takes `--level` or `--docs D04,D19`. It downloads each source, its twin and the Wikisource revisions over HTTPS, checks size and SHA-256 against the manifest and derives the inputs that the manifest asks for (DjVu to PDF, page ranges, a page as PNG, JPEG or TIFF, an office file saved by LibreOffice, the text export of Writer, a recode, an excerpt) inside the toolchain image, so run `prepare --toolchain` first. A derived input that differs from the reference of the manifest is a warning on the line of the document: `derived_differs_from_reference`, or `derived_nondeterministic` for D23 and D24, which LibreOffice writes differently at every run; the document stays available. The Project Gutenberg texts come from the mirror of the Project Gutenberg Literary Archive Foundation (`gutenberg.pglaf.org`). A file that does not match goes to `quarantine/` and the document is reported `unavailable <code>`, for example `unavailable source_hash_mismatch`; the command then exits with 2. A source that changed upstream stays unavailable: the manifest is not updated silently. With a warm cache the standard level is verified in a few seconds. `--offline` verifies the cache without any network.

## Generate and self-check the checks

```bash
.venv-bench/bin/mdf bench checks --docs D18
.venv-bench/bin/mdf bench checks --level standard --self-check --json ~/bench-runs/self-check.json
```

`checks` reads the cache only (never the network) and generates, for each document, the checks of three classes (content, structure, form) from its twin. A document that was not fetched is reported `unavailable source_unreachable`. Each line gives the first characters of the SHA-256 of the check set, the counts per class and `nd`, the checks that cannot be determined from the twin:

```text
D18 checks 4e230d24e9d7 content 5 structure 4 form 17 nd 0
```

`--self-check` renders each twin with a third-party tool of the toolchain, chosen so that it shares no tool with the conversion paths in competition on the format of the document where it can (Pandoc for LaTeX, JATS, MediaWiki and Project Gutenberg HTML; LibreOffice Writer HTML for the Word and ODT twins, with the formulas read from the StarMath source of the Flat ODT; LibreOffice Calc HTML for the workbook; the LaTeXML HTML of arXiv for D03; poppler and Ghostscript for the PDF twins), and judges the checks on that rendering: the ratio of each class before and after the frozen exclusions (thresholds: content 0.98, structure 0.95, form 0.90, exclusions at most 5 %), the coverage of the generator against an independent inventory of the twin (`LOW` below 0.95), the omission proof (with one generated check in ten dropped, coverage must turn low) and a verdict `OK` or `FAIL`:

```text
D18 checks 4e230d24e9d7 content 5 structure 4 form 17 nd 0 | self-check cont 1.000/1.000 stru 1.000/1.000 form 1.000/1.000 | coverage ok | omission ok | OK
D16 checks a624d8caf79a content 2076 structure 655 form 1932 nd 493 | self-check cont 0.972/0.991 stru 0.977/0.977 form 0.987/0.987 | coverage ok | omission ok (not determinable: figure) | OK
```

A document without a twin (D35, D36) prints its oracle instead: `D35 oracle 77c689fa2f5b OK` when the derivation gives back the SHA-256 of the expected text. A type of check that the rendering cannot inventory makes the self-check fail unless it is declared for that kind of twin, and the JSON reports it (`no_inventory`). On the tested laptop the self-check passes on the 36 documents (34 with a twin, 2 with an oracle) in about 15 minutes with a warm cache. `--json FILE` writes the numbers and the toolchain identity, never document text. `--exclusions FILE` is a development option that replaces the packaged exclusions. A `FAIL` makes the command exit with 1.

## Run the measured conversions

```bash
.venv-dev/bin/python -I -B -m markdown_for_llms.cli bench run --level quick --runs-dir ~/bench-runs --out ~/bench-runs/quick-1
```

`run` takes `--level` or `--docs D04,D19`, and reads the cache only, like `checks`: run `fetch` first. Without `--route` it converts each document on every local route of its format in the matrix of the engine (for example pandoc, Marker and LibreOffice for a pptx, Marker and djvutxt for a DjVu); `--route marker,pandoc` keeps only those routes. `--ocr force` asks Marker to run OCR on every page (default `auto`); the other routes ignore it. `--repeat N` (at most 10) converts every attempt N times, to measure the spread of the times and whether the output is deterministic. Datalab runs only with `--route datalab` and its own options ([Measure the Datalab route](#measure-the-datalab-route)). The bench converts only through the engine, with the limits of the product (`mdf convert`) and the `limits` of the manifest (more pages for D13, D16, D17 and D21), and with the weights of the Marker route in `~/.cache/markdown-for-llms/models`.

Each attempt prints one line with its document, route, variant, repetition, generation, status and seconds, and the command ends with the number of documents unavailable (exit 0 means that every attempt has an outcome: some may have failed, as the report shows):

```text
D04 marker 6339a111 k1 g1 complete 24.8 s
D08 pandoc 44136fa3 k1 g1 complete 1.9 s
...
run 3c9d0e5a1b7f4c21: 13 keys, 0 documents unavailable
```

The run directory (`--out`, a new directory under the runs folder; by default `<runs folder>/<run id>`) holds `state.json`, `attempts/<attempt id>/out/` (the output of the engine: `document.md`, `assets/`, `engine.json`, never removed), `attempts/<attempt id>/results.json` (the outcome of every check, by id), `report.json` and `report.md`. The outputs contain document text: keep the directory local. The report holds numbers and closed codes only: for each attempt its status and error code, the arbiter of the document and whether the route shares a tool with it (`arbitrated: false`, then the route stays out of the averages), the three scores with their denominators, the ids, types and outcomes of the failed checks, times (total, per tool, the stages and pages of Marker, seconds per page), memory peaks, problem codes with their pages, the page methods of Marker and, for a txt input, the expected and detected encoding and whether `document.md` has the SHA-256 of the expected text. The summary averages the arbitrated documents per format and route, per profile and route and per script and route: each class has the mean of the documents, their minimum and number, against the provisional thresholds of the profiles (`threshold` and `mean_meets` when every document of the group has the same threshold, and `below_threshold`, the documents under the threshold of their own profile; the thresholds are in `summary.thresholds`). Format × route also has the median seconds per page and, for Datalab, the cost from the spend ledger. The summary then lists the routes not arbitrated and the cells of the matrix not measured. `run.status` says how the run ended (`complete` or `stopped`) and `run.stop` the code of a stop. An image is counted wherever the output has it: a Markdown image (in a table cell too), and every `img` of an HTML block (a `<figure>` with its caption, a table of Calc) or of an inline HTML tag; its file must be under `out/`. The output of a txt input is judged without the types a plain text cannot carry (`format_limit: "txt"`); D35 and D36 have only the encoding check, and a txt of a personal set, which has no oracle, shows `no oracle` in the encoding column of `report.md`. The output of djvutxt is judged as Markdown like every output: it is the text layer of the DjVu without structure, so its checks of emphasis and linked notes fail, and the report notes it (`notes: ["djvutxt_plain_text"]`).

A document whose cached files do not verify is `unavailable` (the line `D15 unavailable source_hash_mismatch`), the other documents are converted, and the command exits with 2. The identity of the package (its `RECORD`), of the toolchain image and of the weights is recorded at the start; the `RECORD` is verified again at the end, and a change makes the report not valid (`identity_changed`, exit 1).

### Interrupt and resume a run

Ctrl-C stops a run: the engine records the interrupted attempt (`interrupted`), the bench saves its state and exits with 1 (`error: interrupted`). A run started in the background by a non-interactive shell stops the same way on SIGINT, because the bench restores the default handler that such a shell turns off. To interrupt a run at a proven phase without a terminal, use `scripts/bench_interrupt.py`: it starts the command without a shell, waits for a path that proves the phase (the work directory of a Marker attempt exists only while it converts) while checking that the command is alive, then sends SIGINT and returns the exit status of the command (3 when the command ended first, 4 when the evidence did not come in time):

```bash
R=~/bench-runs
scripts/run-capped.sh bench-int 24G -- .venv-dev/bin/python -I -B scripts/bench_interrupt.py \
  --evidence "$R/standard-int/work/attempt-*-D04-marker-*" --timeout 1800 -- \
  .venv-dev/bin/python -I -B -m markdown_for_llms.cli bench run --level standard --runs-dir "$R" --out "$R/standard-int"
.venv-dev/bin/python -I -B -m markdown_for_llms.cli bench run --resume "$R/standard-int"
```

`--resume RUN_DIR` takes every option from the state of the run (selection, routes, repetitions, manifest, exclusions, cache, limits); any other option is `resume_option_not_allowed`. It refuses a run whose package, toolchain image, weights, manifest or exclusions changed (`resume_identity_changed`). An attempt that ended is never converted again; an attempt that was running is adopted when its `engine.json` belongs to it (same attempt, input and options, files as listed) and is complete or failed for a reason other than the interruption, otherwise it is kept and converted again in a new generation (`g2`). Each adopted attempt prints a line, for example `D04 marker 6339a111 k1 g1 adopted complete`. Two processes never work on one run: the second one exits with `run_busy`, and the lock goes away with the process, even after a SIGKILL.

The bench refuses to start while the full check holds the development venv (`venv_busy`), and while another process runs a program of the venv it runs from (`venv_in_use`: stop the API and the worker first; other bench runs are allowed).

## Measure the Datalab route

Datalab is remote and paid: the documents leave your machine for Datalab (https://www.datalab.to, US region). For the Datalab team of the key, training on the data and retention are on, and the API can neither read nor change them; the result of each document is deleted after the conversion (`DELETE`), but the copy kept for training may stay. `run` prints this warning when it is about to send, and the report repeats it. Only documents of the public set are sent, never a personal set.

```bash
P=".venv-dev/bin/python -I -B -m markdown_for_llms.cli"
scripts/run-capped.sh bench-datalab 4G -- $P bench run --docs D18 --route datalab --datalab-mode lite --confirm-transit \
  --max-cost-usd 0.02 --spend-cap-usd 4.00 --env-file .env --runs-dir ~/bench-runs --out ~/bench-runs/datalab-d18
```

```text
Datalab: the selected documents of the public set leave this machine for Datalab (https://www.datalab.to, US region). ...
D18 datalab:lite bd47c4e8 k1 g1 complete 9.2 s
run 5b0c8e1f2a3d4c67: 1 keys, 0 documents unavailable
```

`--route datalab` needs all of these, and a Datalab option without it is `invalid_setting`:

| Option | Meaning |
| --- | --- |
| `--datalab-mode M[,M...]` | the modes, among `lite`, `fast`, `balanced` and `accurate`; each mode is a variant of its own |
| `--confirm-transit` | your confirmation that the documents leave the machine (`transit_not_confirmed` without it) |
| `--max-cost-usd USD` | the cap of the run |
| `--spend-cap-usd USD` | the cap of the spend ledger of the runs folder, all runs together (default: `MDF_BENCH_CAP_DATALAB_USD`) |
| `--env-file FILE` | the file with `DATALAB_API_KEY`: a regular file of yours with mode 0600 (`datalab_key_unsafe` otherwise). `run` reads the key from this file only, never from the environment (`datalab_key_missing`), and reads no settings from it |

The refusals come before the key file is opened and before the run directory exists, in this order: a personal set is `datalab_refused_personal` (its key file is never opened), then the modes, the confirmation and the caps, then a ledger that does not add up (`spend_ledger_inconsistent`), then the key. The key is never printed or written: it is in no argument, state, log or report, and [`secret-scan`](#check-that-the-datalab-key-is-in-no-file) checks it.

**What is sent.** The PDF of each selected document whose format has the Datalab route (pdf, png, jpeg, tiff, djvu), as the engine prepares it, one document at a time, at least 5 seconds apart, the cheapest first: on the standard level D16 in `accurate` mode comes last. D20 is never sent, because it would send the PDF of D14; the report lists `djvu × datalab` as not measured, `datalab_excluded`. The standard level in the four modes is 44 sends (11 documents in 4 modes, 223 pages in each mode) and at most 3.9712 USD at the list price; with the discount for sharing the data, the expected cost is about 2.65 USD.

**The ledger.** `<runs folder>/spend.jsonl` holds the Datalab spend of every run of the runs folder (a new `--out` resets nothing), and two runs at the same time share it under a lock. Before each send, under that lock, the bench checks that the ledger plus the estimate of the document stays within `--spend-cap-usd`, and the run plus the estimate within `--max-cost-usd`, then appends a reservation with the estimate (the list price plus 1 cent, rounded up). After the conversion a completion replaces the estimate with the cost billed by Datalab when it reports one, keeps the estimate when the end is uncertain (no answer, an interruption, a result without a cost: `datalab_cost_unknown`) and counts 0 only for a request that Datalab refused before accepting it. The cap that the engine receives for each document is the smallest of what the two caps leave and of the estimate, so the engine never sends more than was reserved.

**Stops.** The run stops, exits with 1 and keeps its state (`status: "stopped"` and `stop`) and its report (`run.status`, `run.stop`) when a cap stops a send (`datalab_cap`: nothing was sent), when the ledger cannot be read or written (`spend_ledger_inconsistent`, `spend_ledger_unwritable`) and when Datalab bills more than the estimate (`datalab_billed_over_estimate`). The result of each send is settled in one place, after the call and after an adoption alike: the completion goes to the ledger first, then the attempt is done, then its billed cost is compared with its estimate, before any other send. A stop for a cost over the estimate stays: every later `--resume` of the run stops again with the same code and sends nothing, so that the maintainer looks at the cost first; a stop of a cap or of the ledger does not stay, and the resumption goes on when the cause is gone. The caps are not raised to let a run pass.

**Resume.** The transit confirmation and the key are never kept: `--resume RUN_DIR --confirm-transit --env-file FILE` when Datalab attempts remain; without remaining ones neither is read. A document whose reservation is in the ledger is never sent again: its result is adopted when it belongs to the attempt (its line ends with `adopted complete`), otherwise the attempt is `datalab_unresolved` (its line ends with `unresolved datalab_unresolved`) and its estimate counts. A run that ends with an attempt `datalab_unresolved` still exits with 0: every attempt has an outcome, and the report shows it as failed. An attempt stopped by a cap sent nothing and is converted again. `--retry-unaccepted` converts again only an attempt that Datalab refused before accepting it (`datalab_rejected` without a request id, for example HTTP 429), never one it accepted.

To interrupt a Datalab run once Datalab has accepted a request, wait for its `datalab-request.json` (in the work directory of the attempt from the acceptance until the result is deleted):

```bash
R=~/bench-runs
scripts/run-capped.sh bench-datalab-int 4G -- .venv-dev/bin/python -I -B scripts/bench_interrupt.py \
  --evidence "$R/datalab-int/work/attempt-*-D18-datalab-*/datalab-request.json" --timeout 600 -- \
  .venv-dev/bin/python -I -B -m markdown_for_llms.cli bench run --docs D18 --route datalab --datalab-mode lite \
  --confirm-transit --max-cost-usd 0.02 --spend-cap-usd 4.00 --env-file .env --runs-dir "$R" --out "$R/datalab-int"
.venv-dev/bin/python -I -B -m markdown_for_llms.cli bench run --resume "$R/datalab-int" --confirm-transit --env-file .env
```

The resumption reports `datalab_unresolved` and posts nothing. When Datalab completes before the signal, the harness exits with 3 and the run has its result.

In the report each Datalab attempt carries its pages, estimate, billed cost, cap and whether the result was deleted (`datalab`), `cost` sums the ledger lines of the run per mode (sends, pages, estimated, billed, billed unknown, counted), and each Datalab cell of the summary format × route has its `cost` from the same lines: the sends and their estimates, the billed cost known (`billed_cents`, `billed_known`), the uncertain cost, the estimates that stand for a cost never billed (`uncertain_cents`, `uncertain`), and what counts (`counted_cents`); local routes have `cost: null`. `report.md` has a Datalab table and a cost column.

## Write the report of a run again

```bash
.venv-dev/bin/python -I -B -m markdown_for_llms.cli bench report ~/bench-runs/standard
```

```text
report 3c9d0e5a1b7f4c21: 47 results verified, report.json and report.md written again (nothing converted)
```

`report` rewrites `report.json`, `report.md` and the `results.json` of each attempt of a finished run from the outputs it saved, with the scoring of the installed package, for example after a fix of the scoring. It converts nothing, starts no engine, sends nothing to Datalab and reads no key; it writes neither the state nor the outputs. It takes the lock of the run and refuses, with a closed code, whatever does not match: a run that did not end, interrupted or still running (`report_run_not_finished`: resume it instead); a manifest or an exclusions file whose SHA-256 is not the one of the run (`report_identity_changed`); a check set that the cache no longer has with the SHA-256 the run recorded (`report_checks_changed`: it is never generated again); a result whose `engine.json` is not the one the run recorded, or does not belong to its attempt, or whose listed files differ (`report_result_changed`), as an adoption checks it. The report then carries `rebuilt`, the time and the package whose scoring it follows, and `report.md` says so.

## Compare two runs

```bash
P=".venv-dev/bin/python -I -B -m markdown_for_llms.cli"
$P bench compare ~/bench-runs/quick-1/report.json ~/bench-runs/quick-2/report.json --fail-on-regression
```

`compare` aligns the attempts of the two reports by document, route, variant (the OCR mode of Marker or the mode of Datalab) and repetition, and prints for each pair the statuses, the difference of the three scores (B − A), the number of checks that passed in A and fail in B (only when the document has the same check set in both runs) and whether the Markdown output is the same. It names the differences of identity between the runs (package and its `RECORD`, toolchain image, weights, manifest, exclusions, versions of the bench) and the documents whose input differs (D23 and D24 are written differently by LibreOffice at every derivation). A regression is a complete attempt that fails in B, or a score that falls by more than `--tolerance` (default 0.005); with `--fail-on-regression` the command exits with 1 when there is one. `--json FILE` writes the comparison. Two runs of the same commit on the same cache give no regression.

## Find suspect checks

```bash
scripts/run-capped.sh bench-crosscheck 8G -- $P bench crosscheck ~/bench-runs/standard/report.json \
  ~/bench-runs/ocr-force/report.json --images --out ~/bench-runs/crosscheck
```

`crosscheck` takes the attempts of the given reports that share the check set of their document; each route and variant counts once, whatever the repetitions. When at least three of them completed and all fail a check with the same outcome, the check is `suspect`: the defect may be in the twin or in the generator rather than in the routes. The bench excludes nothing by itself; a suspect is a pointer for the maintainer. The variants of one route are correlated: the OCR modes of Marker share its models, the four modes of Datalab its service, and they fail together, so three variants of two routes are not three independent judgements of the twin. Each document carries its `paths` and `families` (its distinct routes), each suspect `variants` (the number of paths) and `families`, and `correlated: true` when two of its paths or more are variants of one route; `crosscheck.json` counts the suspects by their number of families (`suspects_by_families`). Each line of the output prints the paths and the families of a document, and the last line the suspects by families, in the form `<documents> documents, <suspects> suspects (by route families: 2 families <n>, 3 families <m>)`. The threshold stays three paths: read a suspect of two families as weak. djvutxt writes text without structure, and no local route recovers emphasis or linked notes from a scan, so the suspects of D20 on those checks are expected: they are a loss of the routes, not a defect of the checks. With `--images` and `--out`, the page of each suspect of a Wikisource twin (its id starts with `pg<N>/`) is rendered from the cached input into `OUT/suspects/` (pdftoppm or ddjvu in the toolchain image; a page image is copied); `crosscheck.json` in `--out` holds the numbers.

## Turn a noted defect into a check (feedback)

A defect the maintainer notes on an output becomes a feedback entry of the exclusions file, by position and type, never by text:

```json
{"document": "D25", "id": "fb:p12:paragraph", "class": "content", "type": "paragraph", "position": "p12",
 "origin": "feedback", "generator_version": "5"}
```

The generator then adds the check `fb:p12:paragraph` for the element of the twin at `p12`, the entries enter the key of the check set, and every later run judges it; the report counts the feedback checks next to the scores. A position that the twin no longer has is counted and reported as `feedback_position_missing`. During development, `--exclusions FILE` passes an exclusions file with such entries to `run`.

## Measure your own documents (personal set)

A personal set is a local manifest that never leaves your machine:

```json
{"schema": "mdf-bench-manifest/2", "set": "personal", "documents": [
 {"id": "P01", "format": "pdf", "pages": 12, "input": {"path": "/abs/path/report.pdf"}, "twin": null},
 {"id": "P02", "format": "docx", "input": {"path": "/abs/path/minutes.docx"},
  "twin": {"kind": "docx", "path": "/abs/path/minutes.docx"}}]}
```

Ids are `P01` to `P99`; paths are absolute; a twin is optional and of a kind read from its file alone (docx, odt, pptx, xlsx, epub, jats); `checks` is refused, because without a twin there is no oracle; `profile`, `pages`, `pages_estimate`, `language`, `scripts`, `limits` and `title` are optional. Select the documents with `--docs`:

```bash
$P bench run --manifest ~/personal.json --docs P01,P02 --runs-dir ~/bench-runs --out ~/bench-runs/personal-1
```

Each file is opened without following symbolic links, must be a regular file of yours, and is copied into `RUN/inputs/` (mode 0600) with its SHA-256 computed while copying; the engine converts the copy. The copy is written to a temporary file next to it and takes its name only once the whole file is read and on disk: a copy cut short (Ctrl-C, a full disk) stays as `RUN/inputs/.partial-<name>-*`, as evidence, and is never converted. A resumption reuses a copy only when the state registered its SHA-256 (`input_sha256`, `twin_sha256`), otherwise it copies again; a registered copy that changed is `resume_identity_changed`. A file that cannot be read so is `unavailable input_unreadable`. The check set of a document with a twin stays in `RUN/checks/`, never in the shared cache. Without a twin a document has only operational measures (status, codes, times, memory, pages) and `scores: null`. The report holds numbers only: no hash of the input or of the output, no ids of the failed checks (their number), no title, language or path; paths and hashes stay in `state.json`.

## Check that the Datalab key is in no file

```bash
scripts/run-capped.sh bench-secret-scan 2G -- $P bench secret-scan --env-file .env ~/bench-runs .cache/run-capped
```

`secret-scan` reads `DATALAB_API_KEY` only from the file given with `--env-file` (a regular file of yours with mode 0600; never from the environment) and searches it, as bytes, in every regular file under the paths, without following symbolic links. The search is literal: a key encoded (base64, URL or JSON escapes), compressed or split across files is not found. It prints the files where it occurs, the paths it could not read (`not read: PATH`) and a last line with the files read, the errors and the occurrences, never the key:

```text
secret-scan: 18607 files read, 0 errors, 0 occurrences
```

An error is a path that could not be scanned: a path given that is missing, is a symbolic link or is not a regular file or a directory, a directory that cannot be listed, a file that cannot be read. Symbolic links and special files inside a directory are skipped. The command exits with 0 only when everything was read and the key occurs nowhere; with 1 and `secret_found` when it occurs, and with 1 and `secret_scan_incomplete` when something could not be read.

## Work without network

After `prepare --toolchain` and `fetch`, the bench needs no network: `list`, `fetch --offline`, `checks` and `checks --self-check` read only the cache and the local image, and every tool container runs with `--network none`. Only `prepare --toolchain` and `fetch` without `--offline` go online. A cache copied from another machine works without Docker as long as it holds every derived input and check set: the JSON then reports `"engine": null`; when something is missing, the command needs the image and stops with `toolchain_missing` if it is not there.

## Folders, environment file and settings

The cache defaults to `$XDG_CACHE_HOME/markdown-for-llms/bench` (`~/.cache/markdown-for-llms/bench`); change it with `--cache DIR` or `MDF_BENCH_CACHE_DIR`. It holds `sources/` (files by SHA-256), `quarantine/`, `wikisource/`, `derived/`, `checks/`, `tool-work/` (the private working directory of each tool call, removed after it), `toolchain/` (the build logs of `prepare --toolchain`) and `work/`. It contains document text: keep it local. The runs folder defaults to `$XDG_DATA_HOME/markdown-for-llms/bench/runs` (`~/.local/share/markdown-for-llms/bench/runs`); `run` creates one directory per run there, with document text too.

Put `MDF_BENCH_*` settings in a file and pass it with `--env-file FILE`; do not export them, because [the full check](uv-environment.md#run-the-full-check) stops when any `MDF_` variable is in the environment. A variable in the environment wins over the file, and `--cache` wins over both. `run` and `secret-scan` read no settings file: their `--env-file` is the file of the Datalab key alone, so set their `MDF_BENCH_*` values on their command line only (`MDF_BENCH_CAP_DATALAB_USD=4.00 .venv-dev/bin/python ... bench run ...`) or use the options. An unknown `MDF_BENCH_` name stops the command with `unknown_setting`, a bad value with `invalid_setting`.

```bash
printf 'MDF_BENCH_CACHE_DIR=%s\nMDF_BENCH_TOOL_MEMORY_MB=4096\n' "$HOME/bench-cache" > ~/bench.env
.venv-bench/bin/mdf bench fetch --offline --level quick --env-file ~/bench.env
```

| Variable | Default | Used today |
| --- | --- | --- |
| `MDF_BENCH_CACHE_DIR` | `~/.cache/markdown-for-llms/bench` | yes, by every command that reads the cache |
| `MDF_BENCH_TOOL_MEMORY_MB` | `8192` | yes: memory cap in MiB of each tool container, 64 to 1048576 |
| `MDF_BENCH_RUNS_DIR` | `~/.local/share/markdown-for-llms/bench/runs` | yes, by `run` (or `--runs-dir`) |
| `MDF_BENCH_RELAY_PORT` | `23461` | no, reserved; must lie in 23460-23479 |
| `MDF_BENCH_INFERENCE_PORT` | `23463` | no, reserved; must lie in 23460-23479 and differ from the relay port |
| `MDF_BENCH_CAP_DATALAB_USD` | unset | yes, by `run`: the default of `--spend-cap-usd`, 0 to 1000 |

Ports 23461 and 23463 are only reserved for the measured conversion: no command opens them today. `.env.example` shows the two ports and the memory cap as comments.

## The image and the full check

The fast suite runs the real tests of the bench and of the conversion engine inside the toolchain image, so [`scripts/full-check.sh`](uv-environment.md#run-the-full-check) needs the image: before the suites, its precondition X runs `python -m markdown_for_llms.engine.toolchain verify` in the development venv and stops when the image is missing or differs from `toolchain.json`. It never builds or downloads anything. To check the image before a full check:

```bash
.venv-bench/bin/python -I -B -m markdown_for_llms.engine.toolchain verify
```

```text
toolchain ok content dcc2395ac8922f90 image sha256:<image id> reference build no docker 29.9.3
```

When it prints `toolchain toolchain_missing` or `toolchain toolchain_mismatch`, run `mdf bench prepare --toolchain` from a clean checkout.

## Exit codes and errors

| Exit | Meaning |
| --- | --- |
| 0 | the command did what was asked. For `run` the measures are complete: every attempt has an outcome, which may be a failed attempt (`converter_failed`, `page_limit`, ...) or `datalab_unresolved`; the report says which. It does not certify the quality of any output |
| 2 | some documents are unavailable (or `prepare` without `--toolchain`) |
| 1 | a failure: a self-check `FAIL`, an error of the toolchain, of the settings or of the manifest (`manifest_schema_unsupported` for a manifest of schema 1), a run that was interrupted, refused, stopped by a Datalab cap, by its spend ledger or by a cost over the estimate, or whose package changed, a report that cannot be written again (`report_*`), a secret scan that found the key or could not read everything |

Errors print a closed code, for example `error: toolchain_missing (build the reference toolchain image with ...)`, never document text, keys or local paths. A tool container that outlives its timeout and cannot be removed stops the whole command.

## Free the Docker resources

The image (`docker image ls mdf-bench-toolchain`) and the builder cache stay until you remove them. `docker buildx prune --builder mdf-bench-toolchain` frees the builder cache; the next `prepare --toolchain` then builds from scratch. `docker buildx rm mdf-bench-toolchain` removes the builder, which `prepare --toolchain` creates again. Removing the image makes the full check stop at precondition X until the next `prepare --toolchain`.

[How-to guides](README.md), [documentation index](../README.md).
