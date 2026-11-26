# Prepare and update the uv environment

For Linux users and developers. The procedure describes how the canonical wheel of `markdown_for_llms` is built and reinstalled; since MFL051 the product itself runs in containers ([Run the product in containers](run-in-containers.md)) and this chain runs inside the full check (`make check`, below). Use the local managed interpreter and `uv.lock`, canonical installation and current preflight. Build and inference are distinct checks; the outcomes of the runs stay in the development records (the development log is archived at tag `pre-docs-2026-10-18`). These commands require preparation and preflight; the acquisition of ML is not implicit.

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

For dev add `--group dev` to the sync; the development venv of the full check is prepared as in [The development venv of the host](#the-development-venv-of-the-host). Every environment receives last the same canonical wheel and its preflight. The extra `engine` (Marker 2.2.0 on the CPU, the conversion engine's pages route) and the extras of the bench resolve torch from the PyTorch CPU index; `bench-cu132` is incompatible with both and has a separate heavy preparation. The universal lock can require metadata of the extras even if the client does not install ML: account for network/cache. Every uv pip that can build uses `--build-constraints`; installing a canonical wheel uses `--no-build`.

For the external check prepare a new venv with the runtime exported from the hash-locked lock, outside the clone. The suite must not prepare it by itself:

```bash
uv export --locked --no-default-groups --no-emit-project --format requirements.txt --output-file "$RUN_PROOF/runtime.txt"
uv venv --python 3.12.13 --managed-python --no-python-downloads "$RUN_ENV"
uv pip sync --python "$RUN_ENV/bin/python" --build-constraints "$RUN_REPO/build-constraints.txt" --require-hashes "$RUN_PROOF/runtime.txt"
```

Then install the canonical wheel and verify it as above. In the IDE select the absolute `.venv-dev/bin/python` of the development venv; a local example is in [settings.uv.example.json](../../.vscode/settings.uv.example.json). Copy it only into your own workspace settings and complete the verified installation; no global setting changes. That venv is a convenience of the host (the IDE, a test run by hand): it runs no child of the engine, each of which is a container started by the launcher from the image of the product.

## Run the full check

Since MFL051 the full check runs in containers: `make check`. It needs `git`, `make` and Docker (Engine 26 or newer, buildx and compose, Linux x86_64); it needs no Python, no uv and no virtual environment of the host. The procedure of the sections above still describes what the chain does inside the container, and how to work on the host for an IDE.

```bash
make check                                  # the gate: a clean tree, the commit of HEAD
make test                                   # the same chain on the working copy, no clean tree asked
make test TESTS="tests/unit/test_x.py"      # a selection in place of tests/
make test-components                        # tests/container against real containers
```

**`make check`** (`scripts/check.sh check`) stops when the tree is not clean. It clones `HEAD` into `.cache/full-check/<id>/src`, builds the images `engine` and `dev` from the clone with the capped builder of the toolchain, reserves the container of the check by creating it (its name is fixed, `mdf-full-check`: Docker admits one, so one full check runs at a time on a daemon), starts a launcher of its own and then runs `scripts/full-check.sh` on the clone, in a container without network, with a memory cap without swap (`MDF_COMPOSE_CHECK_MEMORY`, default 12g) and without the Docker socket:

- the installation preflight: origin of the managed interpreter (O), input inventory (P), source manifest (S), build of sdist and wheel (B), installation of the canonical wheel with a RECORD check, receipt (I), preflight (F);
- the payload gate (C): the launcher and a child run the code under test, the image was built from the inputs of the clone, and the environments agree;
- the preconditions X (the toolchain image, verified through the launcher) and W (the 18 weight files of the Marker route in the models volume against `markdown_for_llms/engine/data/weights.json`);
- the fast suite with the bench JUnit guard (at least one bench test and one engine test run, none skipped).

Then, as a step of the same gate, it runs `make test-components` from the same clone: the tests of `tests/container`, which need a driver with the Docker socket and so cannot run inside the check. The container of the check has ended by then and stays where it is, so its name is held until these tests end too; the output names the clone and the reservation before they start. `make check` ends with `CHECK_PASS` and exit 0 only when both steps are green, the clone is still at the commit with a clean tree, and the reservation was held to the end. The 65 tests of `tests/container` are skipped in the first step and run in the second.

What it needs before the first run: the models volume of the product with the weights (`make up` or `make prepare`; `MDF_COMPOSE_MODELS_VOLUME` names another volume) and the toolchain image. It takes about 20 minutes. The evidence is under `.cache/full-check/<id>/` (ignored by git): `proof/` holds the outputs of every step, the JUnit report and the configurations of compose, among them the one of the product (`compose-product.json`), which `tests/unit/test_compose_config.py` judges. The evidence of the component tests is in the clone, with a link to it in `.cache/test-components/`, where `make test-components` run by hand puts its own. What the check created (its project, its volumes, the two image tags of the invocation) is removed at the end, also when a step is red or the check is interrupted.

A second `make check` started while one runs, in the chain or in the tests of the components, stops at the reservation of the name and removes only its own project. If a check was interrupted and its container stayed, the message says how to look at it and remove it (`docker rm mdf-full-check`). `make test` has no fixed name: several run side by side.

Do not change files of the working copy while `make test` or `make test-components` runs: the working copy is mounted in the container. `make check` reads its clone alone, in both steps: a change of the working copy while it runs does not enter it. Stop the product first if the machine is short of memory; the check does not touch it.

The author saves the output of the check on the commit handed to a review, and reports the path and the SHA-256 of the file:

```bash
mkdir -p temp/<run>/checks
f=temp/<run>/checks/full-check-$(git rev-parse --short HEAD).txt
{ git rev-parse HEAD; git status --short; make check; echo "exit $?"; } > "$f" 2>&1
sha256sum "$f"
```

`scripts/full-check.sh` still reads its paths from the `FULL_CHECK_*` variables; in the containers `docker/dev/compose.yaml` sets them (the interpreter, uv and its cache, the Setuptools wheel and the development venv are in the image `dev`; `FULL_CHECK_MODEL_ROOT` is `/models/marker`). Started by hand on the host it no longer passes: the engine starts its children only through a launcher.

### The development venv of the host

A venv on the host (`<checkout>/.venv-dev/`, ignored by git) is no longer part of any gate. It stays useful for an IDE and for running quickly, without a container, the tests that start no child of the engine:

```bash
.venv-dev/bin/python -B -m pytest --noconftest -p no:cacheprovider -q tests/unit/test_routestate.py
```

Tests that convert, or that talk to a launcher, fail there by design: run them with `make test TESTS="..."`. To create the venv:

```bash
R=$(pwd -P); test ! -e "$R/.venv-dev" || exit 1
scripts/run-capped.sh venv-dev 8G -- env UV_PROJECT_ENVIRONMENT="$R/.venv-dev" UV_CACHE_DIR="$R/.cache/uv-fidelity/cache" \
  UV_PYTHON_INSTALL_DIR="$R/.venv-python" UV_PYTHON_DOWNLOADS=never "$HOME/.local/bin/uv" sync --locked --no-default-groups \
  --extra app --extra engine --group dev --no-install-project --python 3.12.13 --managed-python --no-python-downloads
```

## Explicit local tests without a run

In a normal clone without `temp/`, select **standalone** deliberately. The absence of temp does not change the mode automatically. Before S, create an inventory with `make_input_inventory.py --repo "$RUN_REPO" --managed "$RUN_REPO/.venv-python" --backend-wheel "$RUN_BACKEND_WHEEL" --uv "$RUN_UV" --output "$RUN_PROOF/package-inputs.json"`, using the verified managed interpreter with `-I -B`. The paths of the tools are under `scripts/diagnostics/run-a001-fase0-uv/`; the historical name does not impose a run. `RUN_PROOF` is a new ignored directory, for example `.cache/uv-fidelity/`.

Generate S with `make_source_manifest.py --standalone --repo "$RUN_REPO" --input-inventory "$RUN_PROOF/package-inputs.json" --uv "$RUN_UV" --output "$RUN_PROOF/S.json"`. Rebuild sdist/wheel, audit them, install the canonical wheel and create I with `verify_distribution.py` and the same repo/S/archives: I derives the scope from S, without a `--standalone` flag of the verifier. The local Git of the clone is recorded even if no dev branch exists; a synthetic copy initialized without commits records HEAD/dev null.

For the tests, `RUN_TEST_MODE=standalone` or `--test-mode standalone` selects the local gate; invalid values or discordant choices are errors. The pytest configuration (`tests/conftest.py`) installs the network guards and runs the preflight before the collection. The S/I/E results stay **standalone**, without becoming official evidence. The default **official** requires a run-stage S and a verified snapshot of the declared valid run_id. Stale/mismatched S or I stop the check before the application imports: rebuild and verify explicitly, without a sync hidden in the collection.

[Index](../README.md).
