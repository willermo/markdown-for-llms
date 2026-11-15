# Prepare and update the uv environment

For Linux users and developers. The procedure serves the product `markdown_for_llms`, whose [setup guide](local-product-setup.md) relies on it to build and reinstall the canonical wheel. Use the local managed interpreter and `uv.lock`, canonical installation and current preflight. Build and inference are distinct checks; the outcomes of the runs stay in the development records (the development log is archived at tag `pre-docs-2026-10-18`). These commands require preparation and preflight; the acquisition of ML is not implicit.

You need an identified uv **0.13.7** and space for managed/cache/venv. The CPython patch **3.12.13** and Setuptools **84.0.0** are the pins of the run. The other Python minor versions are not in the declared support. The local pin can change the selection of the pyenv shim; it does not install Python in pyenv. No global change is needed. Windows testing remains future (path `Scripts` and `--no-registry` for the Windows acquisition, not a Linux protection).

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

The gate compares the build inputs (the package files included), archives, RECORD and installation; a mismatch stops the start. After changes generate new S/B/I and repeat the dependent checks; keep the previous outcomes without attributing them to the new code. The RECORD extras created by uv, including `uv_cache.json`, are checked with hash/size and confined paths. For a local wheel uv may not record a hash in `direct_url.json`: I verifies the URI, the digest of B and all the bytes of the installed wheel, without declaring present a hash omitted by the installer. Do not clean global caches to make it pass.

For dev add `--group dev` to the sync; the development venv of the full check is prepared as in [Recreate the development venv](#recreate-the-development-venv). Every environment receives last the same canonical wheel and its preflight. The extra `engine` (Marker 2.2.0 on the CPU, the conversion engine's pages route) and the extras of the bench resolve torch from the PyTorch CPU index; `bench-cu132` is incompatible with both and has a separate heavy preparation. The universal lock can require metadata of the extras even if the client does not install ML: account for network/cache. Every uv pip that can build uses `--build-constraints`; installing a canonical wheel uses `--no-build`.

For the external check prepare a new venv with the runtime exported from the hash-locked lock, outside the clone. The suite must not prepare it by itself:

```bash
uv export --locked --no-default-groups --no-emit-project --format requirements.txt --output-file "$RUN_PROOF/runtime.txt"
uv venv --python 3.12.13 --managed-python --no-python-downloads "$RUN_ENV"
uv pip sync --python "$RUN_ENV/bin/python" --build-constraints "$RUN_REPO/build-constraints.txt" --require-hashes "$RUN_PROOF/runtime.txt"
```

Then install the canonical wheel and verify it as above. In the IDE select the absolute `.venv-dev/bin/python` of the development venv; a local example is in [settings.uv.example.json](../../.vscode/settings.uv.example.json). Copy it only into your own workspace settings and complete the verified installation; no global setting changes. The conversion children of the product use the same `sys.executable`, with `-I -B -m`.

## Run the full check

`scripts/full-check.sh` is the project's full check (see `AGENTS.md`, "Verification"). At the root of a clean checkout it runs the chain of this guide on the current commit: origin of the managed interpreter (O), input inventory (P), standalone source manifest (S), offline build and audit of sdist and wheel (B), reinstallation of the wheel in the development venv with a RECORD check, installation receipt (I), preflight (F) and the toolchain image of the conversion engine (X: `python -m markdown_for_llms.engine.toolchain verify`, read-only, which stops when the image built by `mdf prepare --toolchain`, alias `mdf bench prepare --toolchain`, is missing or differs from `markdown_for_llms/engine/data/toolchain.json`; see [Prepare the reference bench](bench.md)) and the weights of the Marker route (W: `python -m markdown_for_llms.engine.weights verify "$FULL_CHECK_MODEL_ROOT"`, read-only, which stops when one of the 18 files of `markdown_for_llms/engine/data/weights.json` is missing, reached through a symbolic link or differs in bytes or SHA-256; prepare them once with `mdf prepare --weights --model-root DIR`, about 111 MB from Hugging Face and `models.datalab.to`); then the fast suite on `tests/` and the guard of the bench and engine tests. It never downloads or installs anything else: a missing cache or tool is a stop.

It reads six absolute paths, none of them under `temp/`. A variable already set in the environment wins; the missing ones are read from `.full-check.env` at the root of the checkout, or of the main checkout when it runs in a worktree. The file is ignored by git and holds only `NAME=value` lines and `#` comments, read without shell expansion:

| Variable | Path |
| --- | --- |
| `FULL_CHECK_PYTHON_INSTALL_DIR` | directory of the uv-managed interpreters, containing `cpython-3.12.13-linux-x86_64-gnu/`, for example `<checkout>/.venv-python` |
| `FULL_CHECK_DEV_PYTHON` | `bin/python` of the development venv, for example `<checkout>/.venv-dev/bin/python` |
| `FULL_CHECK_UV` | uv binary of the version in `required-version` of `pyproject.toml` (0.13.7), a regular file, not a symbolic link |
| `FULL_CHECK_UV_CACHE_DIR` | uv cache already prepared by that uv version (see below), for example `<checkout>/.cache/uv-fidelity/cache` |
| `FULL_CHECK_BACKEND_WHEEL` | `setuptools-84.0.0-py3-none-any.whl`, SHA-256 `51a52592b3b99e102b609654876bd65f19f999935166d1352678931132b0c670`, for example `<checkout>/.cache/uv-fidelity/backend/setuptools-84.0.0-py3-none-any.whl` |
| `FULL_CHECK_MODEL_ROOT` | model root with the weights of the Marker route, prepared by `mdf prepare --weights --model-root DIR`, for example `~/.cache/markdown-for-llms/models` written as an absolute path; the suite reads it for the real Marker tests (an `MDF_` variable would be refused) |

Each uv version reads its own cache buckets (uv 0.13.7 uses `wheels-v7`, `sdists-v10` and `simple-v27`), so the offline build of step B finds Setuptools only in a cache that the same uv version has filled. Prepare it once, online, outside the full check: build a copy of the checkout with `FULL_CHECK_UV_CACHE_DIR` as `UV_CACHE_DIR`, `--build-constraints` pointing to a file whose only line is `setuptools==84.0.0 --hash=sha256:51a52592b3b99e102b609654876bd65f19f999935166d1352678931132b0c670` and `--require-build-hashes`. Then compare, byte for byte, every file of the extracted Setuptools (`wheels-v7/pypi/setuptools/84.0.0-py3-none-any`) with the members of `FULL_CHECK_BACKEND_WHEEL`, extra and missing files included. The full check itself stays offline and never fills the cache. Buckets of older uv versions are not read and can be deleted.

The script stops when the tree is not clean, when a variable is missing or wrong, when an `MDF_` variable is set in the environment (the suites would read it), when a `PYTEST_` variable is set (for example `PYTEST_ADDOPTS` or `PYTEST_PLUGINS`, which change collection, selection or plugins) and when another full check is using the same development venv. The evidence of each step goes to a new directory `.cache/full-check/<commit>-<UTC time>/`, ignored by git; stdout carries one line per step with its duration, the pytest summary and counts of the suite, and `FULL_CHECK_PASS` or `FULL_CHECK_FAIL`. A suite passes only when pytest exits 0 and its JUnit report can be read, counts at least one test and has no failures and no errors. After the fast suite, `scripts/bench_junit_guard.py` reads its JUnit report: at least one test of `tests/unit/bench/` and one of `tests/unit/engine/` must run, counted apart, and no bench or engine test may be skipped; otherwise the check fails. The bench and engine tests that use real tools run them in the toolchain image, never from the host `PATH`, and fail when the image is missing; the engine tests also need Bubblewrap (`/usr/bin/bwrap`), and one of them converts a document with the engine as installed from the canonical wheel. The exit code is 0 only when every step, the suite and the guard pass.

The reinstallation replaces the package in the development venv. Two full checks never run at the same time, even from different worktrees, and API or worker processes that use the same venv should be stopped first.

### Recreate the development venv

The development venv is `<checkout>/.venv-dev/` (ignored by git), built from `uv.lock` with the product extras, the CPU engine and the dev group, without the project itself: the full check installs the canonical wheel into it. A venv records the absolute paths of its interpreter and scripts, so it is **recreated, never moved**: to change its place, build a new one and point `FULL_CHECK_DEV_PYTHON` to it. From the root of the main checkout:

```bash
R=$(pwd -P); test ! -e "$R/.venv-dev" || exit 1
scripts/run-capped.sh venv-dev 8G -- env UV_PROJECT_ENVIRONMENT="$R/.venv-dev" UV_CACHE_DIR="$R/.cache/uv-fidelity/cache" \
  UV_PYTHON_INSTALL_DIR="$R/.venv-python" UV_PYTHON_DOWNLOADS=never "$HOME/.local/bin/uv" sync --locked --no-default-groups \
  --extra app --extra engine --group dev --no-install-project --python 3.12.13 --managed-python --no-python-downloads
```

The cache is the one of the full check (`FULL_CHECK_UV_CACHE_DIR`): the first sync downloads the locked wheels from PyPI and from the PyTorch CPU index (about 1.6 GB installed) and builds `ebooklib` from its sdist; once it has run, later recreations need no network. Before the first full check, list the installed distributions (`"$R/.venv-dev/bin/python" -I -c 'import importlib.metadata as m; print(sorted(d.metadata["Name"]+"=="+d.version for d in m.distributions()))'`) and compare them, name and version, with the packages of `uv.lock` for these extras; the project itself is missing until the full check installs its wheel. Setuptools 84.0.0 for step B is kept next to the cache, in `<checkout>/.cache/uv-fidelity/backend/`.

## Explicit local tests without a run

In a normal clone without `temp/`, select **standalone** deliberately. The absence of temp does not change the mode automatically. Before S, create an inventory with `make_input_inventory.py --repo "$RUN_REPO" --managed "$RUN_REPO/.venv-python" --backend-wheel "$RUN_BACKEND_WHEEL" --uv "$RUN_UV" --output "$RUN_PROOF/package-inputs.json"`, using the verified managed interpreter with `-I -B`. The paths of the tools are under `scripts/diagnostics/run-a001-fase0-uv/`; the historical name does not impose a run. `RUN_PROOF` is a new ignored directory, for example `.cache/uv-fidelity/`.

Generate S with `make_source_manifest.py --standalone --repo "$RUN_REPO" --input-inventory "$RUN_PROOF/package-inputs.json" --uv "$RUN_UV" --output "$RUN_PROOF/S.json"`. Rebuild sdist/wheel, audit them, install the canonical wheel and create I with `verify_distribution.py` and the same repo/S/archives: I derives the scope from S, without a `--standalone` flag of the verifier. The local Git of the clone is recorded even if no dev branch exists; a synthetic copy initialized without commits records HEAD/dev null.

For the tests, `RUN_TEST_MODE=standalone` or `--test-mode standalone` selects the local gate; invalid values or discordant choices are errors. The pytest configuration (`tests/conftest.py`) installs the network guards and runs the preflight before the collection. The S/I/E results stay **standalone**, without becoming official evidence. The default **official** requires a run-stage S and a verified snapshot of the declared valid run_id. Stale/mismatched S or I stop the check before the application imports: rebuild and verify explicitly, without a sync hidden in the collection.

[Index](../README.md).
