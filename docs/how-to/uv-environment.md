# Prepare and update the uv environment

For Linux users and developers. The procedure serves both the legacy pipeline and the new product `markdown_for_llms`, whose [setup guide](local-product-setup.md) relies on it to build and reinstall the canonical wheel. Use the local managed interpreter and `uv.lock`, canonical installation and current preflight. Build, offline Docker contract and inference are distinct checks; the outcomes of the runs stay in the development records (the development log is archived at tag `pre-docs-2026-10-18`). These commands require preparation and preflight; the acquisition of ML is not implicit.

You need an identified uv **0.10.10**, host Pandoc and space for managed/cache/venv. The CPython patch **3.12.13** and Setuptools **84.0.0** are the pins of the run. The other Python minor versions are not in the declared support. The local pin can change the selection of the pyenv shim; it does not install Python in pyenv. No global change is needed. Windows testing remains future (path `Scripts` and `--no-registry` for the Windows acquisition, not a Linux protection).

In a subshell of the clone, set the absolute paths. The Python acquisition is explicit, with source/hash/build and consumption recorded before use in the run.

```bash
(
  RUN_REPO="$(pwd -P)"
  export UV_PYTHON_INSTALL_DIR="$RUN_REPO/.venv-python"
  export UV_CACHE_DIR="$RUN_REPO/.cache/uv-fidelity/cache"
  uv python install 3.12.13 --no-bin
)
```

After the explicit preparation, in the working subshell define the same paths, `UV_PYTHON_DOWNLOADS=never` and `UV_PROJECT_ENVIRONMENT` to the desired venv. Get the real path `RUN_MANAGED_PY` from the local catalogue, under `.venv-python`; before importing the product verify it with the stdlib. No global Python is accepted for its version number alone.

```bash
"$RUN_MANAGED_PY" -I -B "$RUN_REPO/scripts/diagnostics/run-a001-fase0-uv/check_python_origin.py" --expected-managed "$RUN_REPO/.venv-python"
uv lock --check --python 3.12.13 --managed-python --no-python-downloads
```

The persistent guard `python-downloads = "manual"` prevents implicit Python downloads; it does not impose the directory and does not protect from deliberate overrides. A shell without the uv variables can find an external managed interpreter: that path is rejected by the preflight. `uv run --no-sync` uses an environment already prepared. `--project` selects the project keeping the CWD of the data; `--directory` changes the CWD. Do not use `uv lock --locked`: the local check is `uv lock --check`.

For every change, even to a single `.py` without a version change, prepare a new evidence directory, rebuild and compare the sources. The persistent configuration `reinstall-package` and the flag force the rebuild of the local project; they do not update inactive environments. The normal flow without a run uses `.cache/uv-fidelity/`, excluded from Git, and S with scope `standalone`.

Before S, compile `package-inputs.json`: schema 1, scope `package-inputs`, `files` with absolute paths/bytes/sha256 of the artifacts actually acquired; `backend` with name `setuptools`, version `84.0.0`, `artifacts` (path of the Setuptools wheel), `config_settings: {}` and `environment: {}`. The inventory includes the lock, the interpreter/stdlib and the relevant input packages. The diagnostic rejects unknown backends and inputs. Do not invent digests for files still to be acquired.

`RUN_PROOF` is a new directory under `.cache/uv-fidelity/` and `RUN_ENV` the absolute venv. In the run replace `--standalone` with `--snapshot` and the path of the execution snapshot created by the implementer according to the protocol; the common snapshot of the reviews stays with the supervisor (procedure of the run lifecycle retired by [ADR 0010](../../documentation/decisions/0010-multi-model-agent-governance.md): snapshots are no longer required, and `scripts/run_context.py` stays available as an optional tool). S standalone is not official evidence. After S the build inputs, README included, stay stable until the outcome.

```bash
"$RUN_MANAGED_PY" -I -B "$RUN_REPO/scripts/diagnostics/run-a001-fase0-uv/make_source_manifest.py" --repo "$RUN_REPO" --standalone --input-inventory "$RUN_PROOF/package-inputs.json" --output "$RUN_PROOF/S.json"
uv build --python 3.12.13 --managed-python --no-python-downloads --build-constraints "$RUN_REPO/build-constraints.txt" --out-dir "$RUN_PROOF/dist"
```

`uv build` produces the sdist and the wheel from the sdist. Identify the two single artifacts as `RUN_SDIST` and `RUN_WHEEL`; do not pick an old wheel from `dist`. Check the archives before the installation, in the managed interpreter already proven.

```bash
"$RUN_MANAGED_PY" -I -B "$RUN_REPO/scripts/diagnostics/run-a001-fase0-uv/verify_distribution.py" --repo "$RUN_REPO" --source-manifest "$RUN_PROOF/S.json" --sdist "$RUN_SDIST" --wheel "$RUN_WHEEL" --profile base --expected-managed "$RUN_REPO/.venv-python" --archives-only --receipt "$RUN_PROOF/B-check.json"
UV_PROJECT_ENVIRONMENT="$RUN_ENV" uv sync --locked --no-default-groups --no-editable --reinstall-package markdown-for-llms --python 3.12.13 --managed-python --no-python-downloads
uv pip install --python "$RUN_ENV/bin/python" --no-deps --no-build --reinstall-package markdown-for-llms "$RUN_WHEEL"
"$RUN_ENV/bin/python" -I -B "$RUN_REPO/scripts/diagnostics/run-a001-fase0-uv/verify_distribution.py" --repo "$RUN_REPO" --source-manifest "$RUN_PROOF/S.json" --sdist "$RUN_SDIST" --wheel "$RUN_WHEEL" --profile base --expected-managed "$RUN_REPO/.venv-python" --receipt "$RUN_PROOF/I.json"
"$RUN_ENV/bin/python" -I -B "$RUN_REPO/scripts/diagnostics/run-a001-fase0-uv/check_preflight.py" --repo "$RUN_REPO" --source "$RUN_PROOF/S.json" --receipt "$RUN_PROOF/I.json" --standalone
```

The gate compares ten modules, build inputs, archives, RECORD and installation; a mismatch stops the start. After changes generate new S/B/I and repeat the dependent checks; keep the previous outcomes without attributing them to the new code. The RECORD extras created by uv, including `uv_cache.json`, are checked with hash/size and confined paths. For a local wheel uv may not record a hash in `direct_url.json`: I verifies the URI, the digest of B and all the bytes of the installed wheel, without declaring present a hash omitted by the installer. In this run the cache probe modifies only `exceptions.py` in a copy, verifies the negative before the sync and the positive before the canonical reinstallation, then restores the copy. Do not clean global caches to make it pass.

For dev add `--group dev` to the sync; for the mock API also add `--extra marker-server`, in a distinct venv. Every environment receives last the same canonical wheel and its preflight (profile `api` for the API). The extras `marker-cpu` and `marker-cu126` are incompatible and have a separate heavy preparation. The universal lock can require metadata of the extras even if the client does not install ML: account for network/cache. Every uv pip that can build uses `--build-constraints`; installing a canonical wheel uses `--no-build`.

For the external check prepare a new venv with the runtime exported from the hash-locked lock, outside the clone. The suite must not prepare it by itself:

```bash
uv export --locked --no-default-groups --no-emit-project --format requirements.txt --output-file "$RUN_PROOF/runtime.txt"
uv venv --python 3.12.13 --managed-python --no-python-downloads "$RUN_ENV"
uv pip sync --python "$RUN_ENV/bin/python" --build-constraints "$RUN_REPO/build-constraints.txt" --require-hashes "$RUN_PROOF/runtime.txt"
```

Then install the canonical wheel and verify it as above. In the IDE select the absolute `.venv/bin/python`; a local example is in [settings.uv.example.json](../../.vscode/settings.uv.example.json). Copy it only into your own workspace settings and complete the verified installation; no global setting changes. The children of the phases use the same `sys.executable`, with `-I -B -m`, and the data workspace.

## Explicit local tests without a run

In a normal clone without `temp/`, select **standalone** deliberately. The absence of temp does not change the mode automatically. Before S, create an inventory with `make_input_inventory.py --repo "$RUN_REPO" --managed "$RUN_REPO/.venv-python" --backend-wheel "$RUN_BACKEND_WHEEL" --uv "$RUN_UV" --tokenizer-cache "$TIKTOKEN_CACHE_DIR" --output "$RUN_PROOF/package-inputs.json"`, using the verified managed interpreter with `-I -B`. The paths of the tools are under `scripts/diagnostics/run-a001-fase0-uv/`; the historical name does not impose a run. `RUN_PROOF` is a new ignored directory, for example `.cache/uv-fidelity/`.

Generate S with `make_source_manifest.py --standalone --repo "$RUN_REPO" --input-inventory "$RUN_PROOF/package-inputs.json" --uv "$RUN_UV" --output "$RUN_PROOF/S.json"`. Rebuild sdist/wheel, audit them, install the canonical wheel and create I with `verify_distribution.py` and the same repo/S/archives: I derives the scope from S, without a `--standalone` flag of the verifier. The local Git of the clone is recorded even if no dev branch exists; a synthetic copy initialized without commits records HEAD/dev null.

For the tests, `RUN_TEST_MODE=standalone` or `--test-mode standalone` selects the local gate; invalid values or discordant choices are errors. `run_tests.py` accepts the same selection and requires a current R receipt, with preflight before pytest, cache/Pandoc ready and network guards before the collection. The fast selection is the one of the reference. The S/I/E results stay **standalone**, without becoming official evidence. The default **official** requires a run-stage S and a verified snapshot of the declared valid run_id. Stale/mismatched S or I stop the check before the application imports: rebuild and verify explicitly, without a sync hidden in the collection.

For **C-docker** instead select `RUN_TEST_MODE=official|standalone` in the environment: the pytest flag `--test-mode` alone does not select the mode of the Docker driver. Discordant choices remain errors. Prerequisites and procedure are in the [Docker CPU guide](marker-legacy-docker.md).

[Reference and tests](../reference/toolchain-legacy.md), [Docker](marker-legacy-docker.md), [index](../README.md).
