# syntax=docker/dockerfile:1@sha256:4edf897a3ffa55b89f906fc8cc78afdb3f1834cc9c7083565e611a8a7d5fe99e
# Images of the product (MFL051): one file, three targets, built by `make images` with the capped builder
# of the toolchain (`mdf-bench-toolchain`, markdown_for_llms/engine/data/toolchain.json).
#
#   engine  the engine and the product: extras `app` and `engine` of uv.lock and the project, installed from
#           the wheel built here (not editable). The launcher, the worker, the CLI and the children that
#           run Python use this image; a child gets it by ID, from the launcher's own container.
#           It also holds the Modal CLI 1.6.3 in a virtual environment of its own, /opt/modal, from a lock with
#           hashes (docker/app/modal-requirements.txt): the worker runs it as a child process.
#   api     the API alone: extra `app` and the same wheel (used from tranche A2).
#   dev     what the checks need: the dependencies of `engine` and the `dev` group WITHOUT the project, uv
#           and what an offline build of the project reads (the Setuptools wheel and a uv cache that
#           holds it). The suite installs the wheel under test itself.
#
# Every input is pinned: the base image by digest (the index of amd64 and arm64, the same base as the
# toolchain image), uv 0.13.7 by the SHA-256 of its release archive, CPython 3.12.13 as uv installs it,
# the packages by uv.lock (`--frozen`), Setuptools 84.0.0 by SHA-256, and the Debian packages below from the
# snapshot of the toolchain image (snapshot.debian.org at a fixed date, `--no-install-recommends`, the apt
# lists removed in the same layer). tests/unit/test_dockerignore.py (`TestPins`) holds the two lists.
#
#   engine  curl and ca-certificates (`weights.prepare` and `cpuserver.prepare` download with curl); what
#           WeasyPrint 63.1 opens with dlopen for the Marker providers of docx, xlsx, pptx, epub and html
#           (GObject, Pango, PangoFT2, HarfBuzz, HarfBuzz subset, Fontconfig) with `fontconfig` for /etc/fonts
#           and its cache; and the system fonts of that rendering: Noto for `sans-serif` and `serif`, DejaVu
#           Sans Mono for `monospace`, the families fontconfig resolves on the development machine (the text
#           itself is set in the font of the weights).
#   dev     git, for the steps of the full check that read the checkout (O, S).
# arm64 is expected to build and is not proven; the checks of the interpreter below are amd64 only,
# like the guard they repeat (scripts/diagnostics/run-a001-fase0-uv/guard_product_inputs.py).
ARG BASE=debian:trixie-slim@sha256:a72b4433de1cf7d6bae0c4cee74d0547f58601a643b3375f070c51600d63b6b2
ARG SNAPSHOT=20261113T000000Z
ARG ENGINE_PACKAGES="ca-certificates curl fontconfig fonts-dejavu-mono fonts-noto-core libfontconfig1 libglib2.0-0t64 libharfbuzz-subset0 libharfbuzz0b libpango-1.0-0 libpangoft2-1.0-0"
ARG DEV_PACKAGES="git"

FROM ${BASE} AS uv-amd64
ADD --checksum=sha256:c5221ff41b7892bfc223543883600125f6a47fe845093e0f5bdff4f359196d83 \
    https://github.com/astral-sh/uv/releases/download/0.13.7/uv-x86_64-unknown-linux-gnu.tar.gz /tmp/uv.tar.gz

FROM ${BASE} AS uv-arm64
ADD --checksum=sha256:71279f8d3c67241e807b9d8db872d1bf541579fb9f08b7d1eb1682196d1e57dc \
    https://github.com/astral-sh/uv/releases/download/0.13.7/uv-aarch64-unknown-linux-gnu.tar.gz /tmp/uv.tar.gz

# uv and the managed interpreter. CPython 3.12.13 exists in several builds of python-build-standalone, and a
# given uv installs the newest one it knows (uv 0.13.7: 20260807). The guard of the full check
# (guard_product_inputs.py) pins the binary of the build 20260310, the one of the development machine, so that
# build is named here by URL and SHA-256 and uv installs it as it installs any managed interpreter. On amd64
# the binary must then have the SHA-256 the guard expects: another binary stops the build, the guard is not
# changed.
FROM uv-${TARGETARCH} AS toolbox
ARG TARGETARCH
ENV UV_PYTHON_INSTALL_DIR=/opt/mdf/python UV_PROJECT_ENVIRONMENT=/opt/mdf/venv UV_CACHE_DIR=/var/cache/uv \
    UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1 UV_NO_PROGRESS=1
RUN mkdir -p /opt/mdf \
 && tar -xzf /tmp/uv.tar.gz -C /opt/mdf --strip-components=1 --wildcards '*/uv' && rm /tmp/uv.tar.gz \
 && chmod 0755 /opt/mdf/uv && test "$(/opt/mdf/uv --version | cut -d' ' -f2)" = 0.13.7
COPY <<EOF /tmp/python-downloads.json
{
  "cpython-3.12.13-linux-x86_64-gnu": {
    "name": "cpython", "arch": {"family": "x86_64", "variant": null}, "os": "linux", "libc": "gnu",
    "major": 3, "minor": 12, "patch": 13, "prerelease": "", "variant": null, "build": "20260310",
    "url": "https://github.com/astral-sh/python-build-standalone/releases/download/20260310/cpython-3.12.13%2B20260310-x86_64-unknown-linux-gnu-install_only_stripped.tar.gz",
    "sha256": "7e08911171bd1ea2af43ae9d54917d9b2179d2cba4057057e0fd3c8ac1469368"
  },
  "cpython-3.12.13-linux-aarch64-gnu": {
    "name": "cpython", "arch": {"family": "aarch64", "variant": null}, "os": "linux", "libc": "gnu",
    "major": 3, "minor": 12, "patch": 13, "prerelease": "", "variant": null, "build": "20260310",
    "url": "https://github.com/astral-sh/python-build-standalone/releases/download/20260310/cpython-3.12.13%2B20260310-aarch64-unknown-linux-gnu-install_only_stripped.tar.gz",
    "sha256": "3fb2841e49c95c8eafa5354a9eb4c279af53e97859ac6b2b418a4fa291aa2700"
  }
}
EOF
RUN --mount=type=cache,target=/var/cache/uv \
    /opt/mdf/uv python install 3.12.13 --no-bin --python-downloads-json-url file:///tmp/python-downloads.json \
 && test "$(cat /opt/mdf/python/cpython-3.12.13-linux-*-gnu/BUILD)" = 20260310 \
 && if [ "$TARGETARCH" = amd64 ]; then \
      echo "8392ec78d090ab7e1b1c058cff454de64a0780e263e6a9e66cdb1256be322fb0  /opt/mdf/python/cpython-3.12.13-linux-x86_64-gnu/bin/python3.12" | sha256sum -c -; \
    fi
ENV UV_PYTHON_DOWNLOADS=never
WORKDIR /src
COPY pyproject.toml uv.lock ./

# The dependencies of the engine, without the project. EbookLib is the one package uv.lock has as a source
# archive only: its wheel is built here and must be pure Python.
FROM toolbox AS deps
RUN --mount=type=cache,target=/var/cache/uv \
    /opt/mdf/uv sync --frozen --no-default-groups --extra app --extra engine --no-install-project \
      --python 3.12.13 --managed-python \
 && grep -qx 'Tag: py3-none-any' /opt/mdf/venv/lib/python3.12/site-packages/[Ee]book[Ll]ib-0.18.dist-info/WHEEL

FROM toolbox AS deps-api
RUN --mount=type=cache,target=/var/cache/uv \
    /opt/mdf/uv sync --frozen --no-default-groups --extra app --no-install-project --python 3.12.13 --managed-python

# The Modal CLI in a virtual environment of its own (MFL051 A2): `modal==1.6.3` and its dependencies from a lock
# with the SHA-256 of every file (docker/app/modal-requirements.txt, `--require-hashes`). The worker starts
# /opt/modal/bin/modal as a child process; nothing of it is importable from the environment of the product.
FROM toolbox AS modal
COPY docker/app/modal-requirements.txt /tmp/modal-requirements.txt
RUN --mount=type=cache,target=/var/cache/uv \
    /opt/mdf/uv venv --python 3.12.13 --managed-python /opt/modal \
 && /opt/mdf/uv pip install --python /opt/modal/bin/python --require-hashes --no-deps -r /tmp/modal-requirements.txt \
 && test "$(/opt/modal/bin/modal --version | grep -o '[0-9][0-9.]*' | head -n 1)" = 1.6.3

# The wheel of the project, built as the full check builds it (source archive first, then the wheel).
FROM toolbox AS wheel
COPY MANIFEST.in README.md LICENSE .python-version build-constraints.txt ./
COPY markdown_for_llms ./markdown_for_llms
RUN --mount=type=cache,target=/var/cache/uv \
    /opt/mdf/uv build --python /opt/mdf/python/cpython-3.12.13-linux-*-gnu/bin/python3.12 \
      --build-constraints build-constraints.txt --out-dir /dist \
 && test -f /dist/markdown_for_llms-1.0.0-py3-none-any.whl

# What an offline build of the project reads (docs/how-to/uv-environment.md, "Run the full check"): the
# Setuptools wheel, by SHA-256, and a uv cache filled by this uv with that wheel only, required by hash and
# compared file by file with the wheel, extra and missing files included.
FROM wheel AS seed
ADD --checksum=sha256:51a52592b3b99e102b609654876bd65f19f999935166d1352678931132b0c670 \
    https://files.pythonhosted.org/packages/95/9c/c510029fc6ef33a6275cd2c5d3cecd6613dfd6aa401d57c54f1c18852ccf/setuptools-84.0.0-py3-none-any.whl \
    /opt/mdf/backend/setuptools-84.0.0-py3-none-any.whl
RUN chmod 0644 /opt/mdf/backend/setuptools-84.0.0-py3-none-any.whl \
 && printf '%s\n' 'setuptools==84.0.0 --hash=sha256:51a52592b3b99e102b609654876bd65f19f999935166d1352678931132b0c670' > /tmp/build-constraints.txt \
 && UV_CACHE_DIR=/opt/mdf/seed/uv /opt/mdf/uv build --python /opt/mdf/python/cpython-3.12.13-linux-*-gnu/bin/python3.12 \
      --build-constraints /tmp/build-constraints.txt --require-build-hashes --out-dir /tmp/seed-dist \
 && /opt/mdf/python/cpython-3.12.13-linux-*-gnu/bin/python3.12 -I -B -c 'import pathlib, sys, zipfile; \
wheel = zipfile.ZipFile("/opt/mdf/backend/setuptools-84.0.0-py3-none-any.whl"); \
root = pathlib.Path("/opt/mdf/seed/uv/wheels-v7/pypi/setuptools/84.0.0-py3-none-any").resolve(strict=True); \
members = {n: wheel.read(n) for n in wheel.namelist() if not n.endswith("/")}; \
found = {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*") if p.is_file()}; \
sys.exit(0 if found == members else "seed: Setuptools in the cache differs from the wheel: " + ", ".join(sorted(set(found) ^ set(members))[:5]))'

# Interpreter and dependencies, shared by `engine` and `dev` (the same layers).
FROM ${BASE} AS runtime
COPY --from=toolbox /opt/mdf/python /opt/mdf/python
COPY --from=deps /opt/mdf/venv /opt/mdf/venv
ENV PATH=/opt/mdf/venv/bin:/usr/local/bin:/usr/bin:/bin LANG=C.UTF-8 LC_ALL=C.UTF-8

FROM runtime AS engine
ARG SNAPSHOT
ARG ENGINE_PACKAGES
COPY --from=modal /opt/modal /opt/modal
RUN rm -f /etc/apt/sources.list.d/debian.sources \
 && echo "deb [check-valid-until=no] http://snapshot.debian.org/archive/debian/${SNAPSHOT} trixie main" > /etc/apt/sources.list \
 && echo "deb [check-valid-until=no] http://snapshot.debian.org/archive/debian-security/${SNAPSHOT} trixie-security main" >> /etc/apt/sources.list \
 && apt-get -o Acquire::Retries=3 update && apt-get install -y --no-install-recommends ${ENGINE_PACKAGES} \
 && rm -rf /var/lib/apt/lists/* && fc-cache -s > /dev/null
RUN --mount=type=bind,from=toolbox,source=/opt/mdf/uv,target=/usr/local/bin/uv \
    --mount=type=bind,from=wheel,source=/dist,target=/dist \
    UV_NO_CACHE=1 uv pip install --python /opt/mdf/venv/bin/python --offline --no-deps --compile-bytecode \
      /dist/markdown_for_llms-1.0.0-py3-none-any.whl
# The build inputs of this image, recorded for the payload gate (`containers.inputs_sha256`): a child and
# the launcher report what their image was built from.
RUN --mount=type=bind,source=.,target=/context \
    /opt/mdf/venv/bin/python -I -B -c 'import json; from markdown_for_llms.engine import containers; \
json.dump({"inputs_sha256": containers.inputs_sha256("/context"), "inputs": list(containers.INPUTS)}, open("/opt/mdf/inputs.json", "w"))' \
 && chmod 0644 /opt/mdf/inputs.json

FROM ${BASE} AS api
COPY --from=toolbox /opt/mdf/python /opt/mdf/python
COPY --from=deps-api /opt/mdf/venv /opt/mdf/venv
ENV PATH=/opt/mdf/venv/bin:/usr/local/bin:/usr/bin:/bin LANG=C.UTF-8 LC_ALL=C.UTF-8
RUN --mount=type=bind,from=toolbox,source=/opt/mdf/uv,target=/usr/local/bin/uv \
    --mount=type=bind,from=wheel,source=/dist,target=/dist \
    UV_NO_CACHE=1 uv pip install --python /opt/mdf/venv/bin/python --offline --no-deps --compile-bytecode \
      /dist/markdown_for_llms-1.0.0-py3-none-any.whl

# `dev`: the `dev` group on top of the dependencies of the engine, uv, the Setuptools wheel and the seed of
# the uv cache. The venv, its `.lock`, `bin` and `site-packages` are writable by every UID: the container
# of a check has a writable layer of its own, thrown away when it exits, and installs the wheel under
# test there as the user of the host. The interpreter is checked with the guard itself; a venv of uv
# 0.13.7 has no `_virtualenv.py` (the development venv of the host has none either): one that appeared
# would have to be the file the guard knows.
FROM runtime AS dev
ARG TARGETARCH
ARG SNAPSHOT
ARG DEV_PACKAGES
RUN rm -f /etc/apt/sources.list.d/debian.sources \
 && echo "deb [check-valid-until=no] http://snapshot.debian.org/archive/debian/${SNAPSHOT} trixie main" > /etc/apt/sources.list \
 && echo "deb [check-valid-until=no] http://snapshot.debian.org/archive/debian-security/${SNAPSHOT} trixie-security main" >> /etc/apt/sources.list \
 && apt-get -o Acquire::Retries=3 update && apt-get install -y --no-install-recommends ${DEV_PACKAGES} \
 && rm -rf /var/lib/apt/lists/*
COPY --from=toolbox /opt/mdf/uv /opt/mdf/uv
COPY --from=seed /opt/mdf/backend /opt/mdf/backend
COPY --from=seed /opt/mdf/seed /opt/mdf/seed
RUN --mount=type=cache,target=/var/cache/uv \
    --mount=type=bind,source=pyproject.toml,target=/src/pyproject.toml \
    --mount=type=bind,source=uv.lock,target=/src/uv.lock \
    cd /src && UV_PYTHON_INSTALL_DIR=/opt/mdf/python UV_PROJECT_ENVIRONMENT=/opt/mdf/venv UV_CACHE_DIR=/var/cache/uv \
      UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1 UV_NO_PROGRESS=1 UV_PYTHON_DOWNLOADS=never \
      /opt/mdf/uv sync --frozen --no-default-groups --extra app --extra engine --group dev --no-install-project \
        --python 3.12.13 --managed-python \
 && : > /opt/mdf/venv/.lock && chmod 0666 /opt/mdf/venv/.lock \
 && chmod a+rwx /opt/mdf/venv /opt/mdf/venv/bin /opt/mdf/venv/lib/python3.12/site-packages
RUN --mount=type=bind,source=scripts/diagnostics/run-a001-fase0-uv/guard_product_inputs.py,target=/tmp/guard_product_inputs.py \
    if [ "$TARGETARCH" = amd64 ]; then \
      /opt/mdf/python/cpython-3.12.13-linux-x86_64-gnu/bin/python3.12 -I -B /tmp/guard_product_inputs.py \
        --python /opt/mdf/python/cpython-3.12.13-linux-x86_64-gnu/bin/python3.12 --managed /opt/mdf/python > /dev/null; \
    fi \
 && if [ -e /opt/mdf/venv/lib/python3.12/site-packages/_virtualenv.py ]; then \
      echo "6cf30c56faf2a55228914dbbd17f8088ed371ebb08f5e7fa6fd931f913fcaf1d  /opt/mdf/venv/lib/python3.12/site-packages/_virtualenv.py" | sha256sum -c -; \
    fi
