# Prepare the reference bench (`mdf bench`)

`mdf bench` prepares the reference set of documents and the checks that will score conversions: it builds the reference toolchain, downloads and verifies the documents, and generates and self-checks the checks. **It does not convert anything yet**: the measured conversion of each path, the scores of the conversions, `compare` and the reports are not available; they come with the conversion engine.

For developers on Linux. The bench is a development tool, separate from the experimental product (`mdf convert`, API and Web UI) and from the a002 helpers of the [local conversion benchmark](fidelity-benchmark.md).

| Command | What it does today |
| --- | --- |
| `mdf bench prepare --toolchain` | builds the reference toolchain image locally, verifies it file by file and self-tests it |
| `mdf bench list` | lists the documents of the manifest |
| `mdf bench fetch` | downloads and verifies the documents, their twins and derived inputs |
| `mdf bench checks` | generates the checks from the twins, from the cache only |
| `mdf bench checks --self-check` | also renders each twin with a third-party tool and judges the checks |

## Prerequisites

- **Linux x86_64**: tested. Other systems are not.
- **Docker**, with the `buildx` plugin, reachable through a local socket (`unix://`) by your user. Tested: Docker Engine 29.9.3 with buildx 0.39.0 on Ubuntu 24.04, user in the `docker` group (which is equivalent to root on that machine). The bench refuses a remote Docker endpoint, because it would receive the documents. Docker Desktop on macOS and Windows is **not tested**; note its licence: Docker Desktop is free for personal use, education, non-commercial open source projects and businesses with fewer than 250 employees and less than USD 10 million in annual revenue, while larger organizations need a paid subscription. Docker Engine on Linux has no such licence.
- **uv 0.13.7** (the `required-version` of `pyproject.toml`) and the uv-managed CPython 3.12.13 under `.venv-python/`, prepared as in [Prepare and update the uv environment](uv-environment.md).
- **No other system tool**: pandoc, poppler, qpdf, Ghostscript, DjVuLibre and LibreOffice run inside the toolchain image, never from the host `PATH`.
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

`--locked` stops when `uv.lock` is out of date. The first sync downloads the packages from PyPI and the PyTorch CPU index. No command of the bench downloads models today. After a change to the code, repeat the same `uv sync`: it rebuilds and reinstalls the package from the working tree. The examples below call `.venv-bench/bin/mdf` from the root of the clone.

## Build the reference toolchain

```bash
.venv-bench/bin/mdf bench prepare --toolchain
```

The command checks the build context `scripts/bench-toolchain/` of the checkout against the identity packaged in `markdown_for_llms/bench/data/toolchain.json` (it refuses a changed, added or missing file), creates the BuildKit builder `mdf-bench-toolchain` (memory cap 6 GiB) when it is absent, builds the image, verifies every file of the image against its content manifest and runs a self-test of the tools. Only then does it tag the image `mdf-bench-toolchain:<first 16 characters of the content SHA-256>`.

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
manifest sha256 9d52101f64bb1844d85b15e52328308d2c588e8bcdf556e0b947f21876dae804, 4 of 19 documents
D04  quick,standard   scientific pdf      12  jats            The potential carcinogenicity of orthopaedic implants - a scoping review
D15  quick,standard   scientific docx     ~5  docx            Comparison of ergonomic methods for inoculation of microorganisms (Ukrainian)
D18  quick,standard   scan       png       1  wikisource      La geometria non archimedea (Veronese), page 3 as an image
D19  quick,standard   legal      pdf       7  akoma-ntoso     D.lgs. 8 ottobre 2026, n. 194 (GU SG n. 248 del 24/10/2026, pp. 3-9)
```

The columns are id, levels, profile, format, pages (`~` marks an estimate), the kind of twin (the independent source of the checks) and the title. The levels are `quick` (4 documents), `standard` (18) and `long` (D17, 272 pages). Without `--level` the command lists all 19; `--json` prints JSON; `--manifest FILE` reads another manifest.

## Download and verify the documents

```bash
.venv-bench/bin/mdf bench fetch --level standard
```

```text
D01 available
...
D19 available
18 available, 0 unavailable (manifest 9d52101f64bb1844)
```

`fetch` takes `--level` or `--docs D04,D19`. It downloads each source, its twin and the Wikisource revisions over HTTPS, checks size and SHA-256 against the manifest and derives the inputs that the manifest asks for (DjVu to PDF, page images, page ranges) inside the toolchain image, so run `prepare --toolchain` first. A file that does not match goes to `quarantine/` and the document is reported `unavailable <code>`, for example `unavailable source_hash_mismatch`; the command then exits with 2. A source that changed upstream stays unavailable: the manifest is not updated silently. With a warm cache the standard level is verified in a few seconds. `--offline` verifies the cache without any network.

## Generate and self-check the checks

```bash
.venv-bench/bin/mdf bench checks --docs D18
.venv-bench/bin/mdf bench checks --level standard --self-check --json ~/bench-runs/self-check.json
```

`checks` reads the cache only (never the network) and generates, for each document, the checks of three classes (content, structure, form) from its twin. A document that was not fetched is reported `unavailable source_unreachable`. Each line gives the first characters of the SHA-256 of the check set, the counts per class and `nd`, the checks that cannot be determined from the twin:

```text
D18 checks 4e230d24e9d7 content 5 structure 4 form 17 nd 0
```

`--self-check` renders each twin with a third-party tool of the toolchain and judges the checks on that rendering: the ratio of each class before and after the frozen exclusions (thresholds: content 0.98, structure 0.95, form 0.90, exclusions at most 5 %), the coverage of the generator against an independent inventory of the twin (`LOW` below 0.95), the omission proof (with one generated check in ten dropped, coverage must turn low) and a verdict `OK` or `FAIL`:

```text
D18 checks 4e230d24e9d7 content 5 structure 4 form 17 nd 0 | self-check cont 1.000/1.000 stru 1.000/1.000 form 1.000/1.000 | coverage ok | omission ok | OK
D16 checks 9739d9db5b02 content 2046 structure 655 form 1932 nd 493 | self-check cont 0.973/0.992 stru 0.974/0.974 form 0.987/0.987 | coverage ok | omission ok (not determinable: figure) | OK
```

On the tested laptop the self-check of the standard level passes on 18 of 18 documents in about 2 minutes with a warm cache. `--json FILE` writes the numbers and the toolchain identity, never document text. `--exclusions FILE` is a development option that replaces the packaged exclusions. A `FAIL` makes the command exit with 1.

## Work without network

After `prepare --toolchain` and `fetch`, the bench needs no network: `list`, `fetch --offline`, `checks` and `checks --self-check` read only the cache and the local image, and every tool container runs with `--network none`. Only `prepare --toolchain` and `fetch` without `--offline` go online. A cache copied from another machine works without Docker as long as it holds every derived input and check set: the JSON then reports `"engine": null`; when something is missing, the command needs the image and stops with `toolchain_missing` if it is not there.

## Folders, environment file and settings

The cache defaults to `$XDG_CACHE_HOME/markdown-for-llms/bench` (`~/.cache/markdown-for-llms/bench`); change it with `--cache DIR` or `MDF_BENCH_CACHE_DIR`. It holds `sources/` (files by SHA-256), `quarantine/`, `wikisource/`, `derived/`, `checks/`, `tool-work/` (the private working directory of each tool call, removed after it), `toolchain/` (the build logs of `prepare --toolchain`) and `work/`. It contains document text: keep it local. The runs folder defaults to `$XDG_DATA_HOME/markdown-for-llms/bench/runs` (`~/.local/share/markdown-for-llms/bench/runs`); no command writes there yet.

Put `MDF_BENCH_*` settings in a file and pass it with `--env-file FILE`; do not export them, because [the full check](uv-environment.md#run-the-full-check) stops when any `MDF_` variable is in the environment. A variable in the environment wins over the file, and `--cache` wins over both. An unknown `MDF_BENCH_` name stops the command with `unknown_setting`, a bad value with `invalid_setting`.

```bash
printf 'MDF_BENCH_CACHE_DIR=%s\nMDF_BENCH_TOOL_MEMORY_MB=4096\n' "$HOME/bench-cache" > ~/bench.env
.venv-bench/bin/mdf bench fetch --offline --level quick --env-file ~/bench.env
```

| Variable | Default | Used today |
| --- | --- | --- |
| `MDF_BENCH_CACHE_DIR` | `~/.cache/markdown-for-llms/bench` | yes, by every command that reads the cache |
| `MDF_BENCH_TOOL_MEMORY_MB` | `8192` | yes: memory cap in MiB of each tool container, 64 to 1048576 |
| `MDF_BENCH_RUNS_DIR` | `~/.local/share/markdown-for-llms/bench/runs` | no, reserved for the runs of the measured conversion |
| `MDF_BENCH_RELAY_PORT` | `23461` | no, reserved; must lie in 23460-23479 |
| `MDF_BENCH_INFERENCE_PORT` | `23463` | no, reserved; must lie in 23460-23479 and differ from the relay port |
| `MDF_BENCH_CAP_DATALAB_USD` | unset | no, reserved; 0 to 1000 when set |

Ports 23461 and 23463 are only reserved for the measured conversion: no command opens them today. `.env.example` shows the two ports and the memory cap as comments.

## The image and the full check

The fast suite runs the real tests of the bench inside the toolchain image, so [`scripts/full-check.sh`](uv-environment.md#run-the-full-check) needs the image: before the suites, its precondition X runs `python -m markdown_for_llms.bench.toolchain verify` in the development venv and stops when the image is missing or differs from `toolchain.json`. It never builds or downloads anything. To check the image before a full check:

```bash
.venv-bench/bin/python -I -B -m markdown_for_llms.bench.toolchain verify
```

```text
toolchain ok content dcc2395ac8922f90 image sha256:<image id> reference build no docker 29.9.3
```

When it prints `toolchain toolchain_missing` or `toolchain toolchain_mismatch`, run `mdf bench prepare --toolchain` from a clean checkout.

## Exit codes and errors

| Exit | Meaning |
| --- | --- |
| 0 | everything requested was done |
| 2 | some documents are unavailable (or `prepare` without `--toolchain`) |
| 1 | a failure: a self-check `FAIL`, an error of the toolchain or the settings |

Errors print a closed code, for example `error: toolchain_missing (build the reference toolchain image with ...)`, never document text, keys or local paths. A tool container that outlives its timeout and cannot be removed stops the whole command.

## Free the Docker resources

The image (`docker image ls mdf-bench-toolchain`) and the builder cache stay until you remove them. `docker buildx prune --builder mdf-bench-toolchain` frees the builder cache; the next `prepare --toolchain` then builds from scratch. `docker buildx rm mdf-bench-toolchain` removes the builder, which `prepare --toolchain` creates again. Removing the image makes the full check stop at precondition X until the next `prepare --toolchain`.

[How-to guides](README.md), [documentation index](../README.md).
