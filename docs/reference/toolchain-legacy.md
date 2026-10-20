# Reference of the legacy toolchain and pipeline

Legacy pipeline: managed installation and S/B/I/E checks, separate suites and offline Docker contract. Linux x86_64 is the target of the documented diagnostics, not a restriction of the universal lock. The outcomes of the runs are in the development records (the development log is archived at tag `pre-docs-2026-10-18`). The new application (Web UI, DB and worker) is available on `dev` as the experimental first increment I1: see the [local product reference](local-product.md).

| Selection | Direct content | Limit |
| --- | --- | --- |
| base | requests, tiktoken, tqdm, python-dotenv | No server, ML or dev |
| dev group | pytest/cov/mock, black, flake8, mypy, httpx | Explicit selection |
| marker-server | FastAPI, uvicorn, python-multipart | Enough for the mock API, without ML |
| marker-cpu | marker-pdf full 1.10.2, Surya 0.17.1, torch 2.7.1, bs4 | Explicit CPU torch index; candidate (the profile of I1 according to ADR 0009) |
| marker-cu126 | same candidates, torch index cu126 | Incompatible with CPU in the same sync; GPU not verified |

uv 0.10.10, local managed CPython 3.12.13 and Setuptools 84.0.0 are pins of the run, not a promise of automatic updates. `pyproject.toml` declares the dependencies, `uv.lock` resolves them and is present in the worktree of the run. The backend constraints are separate from the runtime pins; Marker is wheel only (`no-build-package`). A transitive sdist with an unidentified or unconstrained backend blocks the preparation.

Ten distributed modules: `config`, `logging_config`, `exceptions`, `unified_converter`, `master_workflow`, `clean_markdown`, `validate_markdown`, `chunk_markdown`, `batch_monitor`, `marker_api_server`. The server requires the API extra; in the base it is only read and hashed. The wheel does not contain tests, diagnostics, fixtures, data, operational configurations or run helpers.

| Console script | Target | Options |
| --- | --- | --- |
| markdown-pipeline | master_workflow:main | --llm/--target-llm, --chunk-size, --overlap, --force, --step, --config-only, --dry-run |
| markdown-config | config:main | --create-default, --validate, --show |
| markdown-clean | clean_markdown:main | No parser/help: uses the JSON in the CWD |
| markdown-validate | validate_markdown:main | No parser/help: uses the JSON in the CWD |
| markdown-chunk | chunk_markdown:main | --input-dir, --output-dir, --target-llm, --chunk-size, --overlap |

Check the actual help before use. `--step` admits conversion, cleaning, validation, chunking. `--chunking-strategy` does not exist. `--config-only` creates the JSON and exits before the orchestrator and the prerequisites. Clean/validate do not answer `--help`: the invocation starts the processing. `config --show` shows the JSON and does not apply the overrides of the pipeline.

The pipeline captures the CWD as workspace at the moment of the instance. JSON, relative paths, validation_report and logs stay there; absolute paths stay absolute. To call it from Python, pass a `ConfigManager` to `UnifiedDocumentConverter`, not `config.conversion_settings`.

The pipeline and the standalone converter load only the `.env` of the workspace, without going up into the ancestors or searching in site-packages, with `override=False`. In the orchestrator: JSON → CLI → application overrides; the shell prevails over `.env`. The standalone converter keeps its own legacy overrides. `.env.example` documents only the `MDF_` settings of the new converter, not the legacy variables of the table below; it does not enable the cloud by itself. An empty `MARKER_DOCUMENT_FORMATS` does not clear a JSON list already configured.

| Variables | Legacy use |
| --- | --- |
| CHUNK_SIZE, OVERLAP_SIZE, TARGET_LLM | Override of the chunking parameters |
| MAX_WORKERS, LOG_LEVEL | Pipeline |
| CONVERSION_TIMEOUT, CONNECT_TIMEOUT | Conversion and connect timeout; timeout 0 waits without limit |
| MARKER_LOCAL_BASE_URL/ENDPOINT | Local HTTP endpoint |
| MARKER_CLOUD_BASE_URL/ENDPOINT, MARKER_BASE_URL | Cloud; BASE_URL is a legacy alias of the cloud |
| MARKER_API_KEY | Secret read by the cloud client; name selectable in the JSON |
| MARKER_DOCUMENT_FORMATS | Explicit list of extensions other than PDF for Marker |
| MARKER_USE_LLM, FORCE_OCR, PAGINATE, STRIP_EXISTING_OCR, DISABLE_IMAGE_EXTRACTION (prefix MARKER_) | Client options; booleans 1/true/yes/on |
| MAX_POLLS, POLL_INTERVAL | Cloud polling; 0 polls without limit |
| TIKTOKEN_CACHE_DIR | Tokenizer cache ready, identified before the checks |

The UV_* and RUN_* variables are toolchain and evidence settings, not application configuration. Do not include secrets in the snapshots or in the logs of the probes. The options of the Marker wrapper are reported in the metadata, without being applied to the converter; empty images and liveness health remain limits. `local_marker` and the host server are not verified after the migration. The new local and remote policy remains a project.

The official validation requires an identified execution snapshot, R PASS in the runner and the S/B/I chain verified before the collection, with tokenizer cache and Pandoc ready. `uv --offline` is not enough to confine requests and children. `run_offline.py` uses primary unshare, or Firejail noprofile/net none with blacklist D4; same netns for parent and children, positive local socketpair and daemon/IP denied. The preliminary probe of the run is not an official PASS. The wrapper does not prepare packages, does not create a freeze and is not a general sandbox for hostile code.

Every command below is **to be run after preparation and preflight**. `RUN_DEV_PY` is the installed dev venv; `RUN_API_PY` also contains marker-server. Set RUN_SOURCE_MANIFEST and RUN_INSTALLATION_RECEIPT to the current S/I files, verified in the same interpreter. For packaging: RUN_WHEEL_PY/RUN_WHEEL_RECEIPT, RUN_BASELINE/RUN_BASELINE_WORKSPACES. No missing prerequisite becomes a skip.

```bash
"$RUN_DEV_PY" -I -B -m pytest tests/ --ignore=tests/api --ignore=tests/packaging --ignore=tests/docker -q
"$RUN_API_PY" -I -B -m pytest tests/api/ -q
"$RUN_DEV_PY" -I -B -m pytest tests/packaging/ -q
"$RUN_DEV_PY" -I -B -m unittest discover -s tests/governance -v
"$RUN_DEV_PY" -I -B -m pytest tests/ --ignore=tests/api --ignore=tests/packaging --ignore=tests/docker --collect-only -q
"$RUN_DEV_PY" -I -B -m pytest --collect-only -q
"$RUN_DEV_PY" -I -B -m pytest . --ignore=tests/api --ignore=tests/packaging --ignore=tests/docker --collect-only -q
"$RUN_API_PY" -I -B -m pytest tests/ --collect-only -q
"$RUN_API_PY" -I -B -m pytest . --collect-only -q
```

The global collection uses the API environment, imports the harnesses without ML and does not build images. API/package/Docker remain distinct mandatory layers; the selected fast suite is not a complete acceptance test. The two root smoke scripts are manual and excluded: they are not safe inference checks. The historical results 67 pass and V0 62 pass/5 fail do not transfer to the new product.

For Docker, outside the runner that masks the daemon, RUN_IMAGE is a local ID, RUN_IMAGE_MANIFEST is the image-source.json produced by the context tool, RUN_IMAGE_INPUTS is the receipt of the supply tool, RUN_BACKEND_WHEEL is the hash-pinned wheel 84, RUN_IMAGE_CONTEXT_RECEIPT is the receipt of the context producer, RUN_TEST_OUTPUT is a new directory. C-docker requires a CPU V7 image already prepared: pull never/network none, venv Python -I -B on stdin, without mount or ordinary startup. The test invokes the versioned run_docker_contract.py: cap-drop ALL/no-new-privileges, healthcheck disabled, finite memory and pids, export of the filesystem of the stopped container before Python, full I and nine subcases. Current host preflight S/I, with explicit official or standalone mode; no reference to a fixed historical snapshot. For C-docker set `RUN_TEST_MODE=official|standalone` in the environment: the pytest flag `--test-mode` alone does not select the Docker driver; discordant choices are rejected.

```bash
"$RUN_DEV_PY" -I -B -m pytest tests/docker/test_marker_contract.py -q
```

Negatives D2/D3, cache probe, R, F1–F6, S/B/I/E and receipts with hash/argv/CWD/env are requirements of the run. The suite does not build or install to repair a mismatch. A change invalidates the dependent evidence according to the chain described in the uv guide. The current governance is [`AGENTS.md`](../../AGENTS.md) with [ADR 0010](../../documentation/decisions/0010-multi-model-agent-governance.md); the run cycle (archived: `documentation/development/run-lifecycle.md`, tag `pre-docs-2026-10-18`) is historical. For a clone without temp, the details of the chain are in the [uv guide](../how-to/uv-environment.md).

## Development tools

The `dev` group of the pyproject includes pytest and the tools Black, Flake8 and Mypy. Prepare the locked profile and reinstall the canonical wheel before the preflight. Running a formatter or a static check is an explicit choice: these tools do not replace the selected suites or prove the fidelity of the conversion. The suite does not install dependencies to repair the collection.
