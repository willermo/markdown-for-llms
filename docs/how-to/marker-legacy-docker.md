# Prepare and verify Marker in Docker

CPU build, offline contract and inference are distinct checks. You need Docker, Compose/buildx, the universal lock and explicit space for OCI/APT/wheel/venv. Do not start services to verify build or contract. The base client does not acquire ML implicitly. The outcomes of the runs are in the development records (the development log is archived at tag `pre-docs-2026-10-18`).

## Public inputs and provenance

Use **linux/amd64 pinned** manifests of the configured sources `registry-1.docker.io/library/python` (CPython 3.12.13-slim-bookworm) and `ghcr.io/astral-sh/uv` (0.10.10), not floating tags or illustrative digests. Maintaining the bases requires an explicit choice and the repetition of the checks. Choose an available Debian/bookworm and security snapshot, with active signatures. `RUN_PYTHON_IMAGE`, `RUN_UV_IMAGE` are `@sha256` references verified with imagetools; `RUN_APT_SNAPSHOT` is the chosen `YYYYMMDDTHHMMSSZ` timestamp.

The versioned tool `scripts/diagnostics/run-a001-fase0-uv/prepare_docker_inputs.py` has two operations: **supply** and **context**. It requires the verified dev managed interpreter 3.12.13 with packaging, `-I -B`, explicit repo/input/output and new public directories outside the operational data and Git. It does not depend on temp, a personal HOME or a run. The receipt is a sibling of the output, kept also on FAIL.

Supply derives the native locked uv export CPU/server without dev/cu126, selects wheels compatible with the ABI, verifies SHA/size of the lock, ZIP/payload/RECORD. The only admitted sdist is EbookLib 0.18 with an explicit hash-pinned backend 84; unknown requirements stop the preparation and the build. For APT it exports the keyring from the pinned stopped base, verifies gpgv InRelease and the SHA/size of the Packages, runs the native apt-get resolver offline in its own container and verifies the selected .deb files against those signed Packages. The container mounts only that public repository, readonly, network none, cap-drop ALL, no-new-privileges.

For a new clone add **--acquire** to the supply command only after the admission of the resources. It acquires only the URLs of the lock and of the configured sources and verified redirects, pinned OCI and the APT snapshot; it has finite deadline, storage and network. The OCI traffic is a declared upper bound, not a wire measure. For local reuse provide --wheel-dir/--apt-repository: it checks everything again before use, without trusting opaque copies or old receipts. Without --acquire a missing input is FAIL, not a download to repair a build. Remote weights and fonts are not supply.

Example of **reuse** of public data already available; for the first preparation omit the two reuse inputs and choose --acquire and quotas explicitly:

```bash
"$RUN_DEV_PY" -I -B "$RUN_REPO/scripts/diagnostics/run-a001-fase0-uv/prepare_docker_inputs.py" supply   --repo "$RUN_REPO" --output "$RUN_INPUTS" --uv "$RUN_UV"   --wheel-dir "$RUN_PUBLIC_WHEELS" --apt-repository "$RUN_PUBLIC_APT"   --python-image "$RUN_PYTHON_IMAGE" --uv-image "$RUN_UV_IMAGE"   --apt-snapshot "$RUN_APT_SNAPSHOT" --network-budget-bytes 0 --reuse-in-place
"$RUN_DEV_PY" -I -B "$RUN_REPO/scripts/diagnostics/run-a001-fase0-uv/prepare_docker_inputs.py" context   --repo "$RUN_REPO" --source "$RUN_SOURCE_MANIFEST" --mode standalone --output "$RUN_IMAGE_CONTEXT"
```

`--mode standalone` requires a standalone S generated with the [uv guide](uv-environment.md); for run checks select official and the S of the current snapshot. No future S/output in its own inputs. Context verifies the current S and copies only ten modules, metadata/lock/constraints/README/licence, Dockerfile/.dockerignore and admitted diagnostics. It generates image-source.json **in the throwaway copy**, never in the clone. New synthetic sentinels for .env/.git/venv/temp/config/data/cache verify the filter actually consumed by the builder. No recursive copy of the clone.

## Offline Compose build

Both Compose files require RUN_IMAGE_CONTEXT and RUN_VERIFIED_WHEELS and RUN_VERIFIED_APT (directories verified by the tool, supply output), pinned bases, APT timestamp and RUN_HOST_NETNS=`readlink /proc/self/ns/net`. RUN_IMAGE_TAG identifies a new test image. RUN_DATA_ROOT can point to an absolute external workspace; otherwise, the relative path of the source_documents resolves against the Compose file, **not** against the throwaway context.

```bash
docker compose -f "$RUN_REPO/docker-compose.yml" config
docker compose -f "$RUN_REPO/docker-compose-build.yml" config
docker compose -f "$RUN_REPO/docker-compose.yml" -f "$RUN_REPO/docker-compose.gpu.yml" config
docker compose --progress plain -f "$RUN_REPO/docker-compose.yml" build marker-api
```

build.network is none; additional_contexts provides the local verified-wheels and verified-apt. The second Compose has the same resolved build; the runtime can have different memory limits, which are not a throughput measure. The GPU override is only static config: do not start CUDA/device or inference to verify it.

The Dockerfile uses the signed local readonly APT only in the RUN, then uv export locked/pip sync hash-locked offline from the local supply, no-build-isolation and constraints 84. Backend/import/get_requires of EbookLib are attested before the build and the real native hooks traced. image_package.py builds sdist and wheel from the sdist, B/RECORD audit before installation, canonical and full I in the builder. The runtime contains venv and evidence, without src, UV, APT/wheel supply, build cache, .deb or compilers. Setuptools 84 is a Torch runtime dependency of the lock, not a dev group.

## Permanent C-docker

Prepare a local image by sha256 ID and set RUN_IMAGE, RUN_IMAGE_MANIFEST=image-source.json of the copy, RUN_IMAGE_INPUTS=supply receipt, RUN_BACKEND_WHEEL, RUN_IMAGE_CONTEXT_RECEIPT (receipt of the context producer), RUN_TEST_OUTPUT (new output), and the current RUN_SOURCE_MANIFEST and RUN_INSTALLATION_RECEIPT. In the normal clone select RUN_TEST_MODE=standalone; official requires a current snapshot. For C-docker the selection is `RUN_TEST_MODE=official|standalone` in the environment: the pytest flag `--test-mode` alone does not select the Docker driver. Discordant choices are rejected. The local daemon is an explicit scope distinct from the runner that masks it in the host tests.

```bash
"$RUN_DEV_PY" -I -B -m pytest "$RUN_REPO/tests/docker/test_marker_contract.py" -q
```

The test invokes the versioned run_docker_contract.py with a repo/input/output CLI. Container by ID, pull never, network none, no mount, cap-drop ALL, no-new-privileges, healthcheck disabled, pids 128, mem 2 GiB, cpu 2. Streaming export of the stopped filesystem: startup, .pth, customize, pyvenv and backend 84 checked **before** Python. Trusted stdin; full I of runtime, RECORD and canonical wheel before import, exact versions of the locked graph and native APT, nine offline subcases unchanged: CPU/import, signature/provider full, synthetic WeasyPrint, MarkdownOutput, wrapper, errors, constructor mock, cache-font. No real create_model_dict, download, inference, GPU options or normal CMD. Receipts keep their own inputs, argv, outcomes, processes and containers also on FAIL; cleanup only of their own diagnostic objects.

## Operational limits

Startup with weights, PDF/non-PDF inference and GPU require sources, hashes and licences, RAM, disk, cache, duration and hardware, and a distinct scope. They are not fast tests. /health measures liveness even after a failed model loading; the wrapper returns empty images and can report options without applying them. The mock contract and WeasyPrint do not certify Marker fidelity or the host service.

Model cache=/data/cache/models, HF=/data/cache/huggingface, fonts=/data/fonts, TMPDIR=/data/tmp; FONT_PATH and the Surya map are explicit. Do not reuse operational volumes in the checks. No prune/down with global removal of the volumes. Update and deploy require their own scope and new dependent checks.

[Environment](uv-environment.md), [reference](../reference/toolchain-legacy.md), [index](../README.md).
