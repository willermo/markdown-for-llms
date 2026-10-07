# Candidati della run: python:3.12.13-slim-bookworm e uv:0.10.10.
# Argomenti obbligatori: riferimenti @sha256 verificati prima della build V7.
ARG PYTHON_IMAGE
ARG UV_IMAGE
FROM scratch AS verified-wheels
# Il contesto omonimo deve essere fornito esplicitamente: nessun fallback registry.
COPY .python-version /unprepared
FROM scratch AS verified-apt
COPY .python-version /unprepared
FROM ${UV_IMAGE} AS uv
FROM ${PYTHON_IMAGE} AS native
ARG APT_SNAPSHOT
# APT locale conserva InRelease/Packages firmati e i payload selezionati dal resolver.
RUN --mount=type=bind,from=verified-apt,source=/,target=/verified-apt,readonly \
    test -n "$APT_SNAPSHOT" && test "$(cat /verified-apt/snapshot.txt)" = "$APT_SNAPSHOT" && \
    rm /etc/apt/sources.list.d/debian.sources && \
    printf 'deb [check-valid-until=no] file:/verified-apt/debian bookworm main\ndeb [check-valid-until=no] file:/verified-apt/debian-security bookworm-security main\n' > /etc/apt/sources.list && \
    apt-get -o Acquire::Languages=none -o Acquire::By-Hash=false update && \
    apt-get install -y --no-install-recommends ca-certificates curl libpango-1.0-0 \
      libpangoft2-1.0-0 libharfbuzz-subset0 fontconfig fonts-dejavu-core libgomp1 && \
    rm -rf /var/lib/apt/lists/*
FROM native AS builder
COPY --from=uv /uv /usr/local/bin/uv
COPY --from=verified-wheels / /verified-wheels
ENV UV_PYTHON_DOWNLOADS=never UV_PROJECT_ENVIRONMENT=/opt/venv UV_CACHE_DIR=/build-cache \
    UV_OFFLINE=1 UV_NO_INDEX=1 UV_FIND_LINKS=/verified-wheels PYTHONDONTWRITEBYTECODE=1
WORKDIR /src
COPY . /src/
ARG MARKER_EXTRA=marker-cpu
RUN test "$MARKER_EXTRA" = marker-cpu -o "$MARKER_EXTRA" = marker-cu126
RUN test -z "$(find /usr/local/lib/python3.12/site-packages -maxdepth 1 \( -name '*.pth' -o -name '*customize*' \))" && \
    echo '51a52592b3b99e102b609654876bd65f19f999935166d1352678931132b0c670  /verified-wheels/setuptools-84.0.0-py3-none-any.whl' | sha256sum -c - && \
    uv pip install --system --no-deps --no-build /verified-wheels/setuptools-84.0.0-py3-none-any.whl
ARG HOST_NETNS
ENV MARKER_EXTRA=$MARKER_EXTRA RUN_HOST_NETNS=$HOST_NETNS
RUN uv lock --check --python /usr/local/bin/python --no-managed-python --no-python-downloads && \
    /usr/local/bin/python -I -B scripts/diagnostics/run-a001-fase0-uv/image_prepare.py \
      --repo /src --supply /verified-wheels --output /opt/prepare-proof
RUN /usr/local/bin/python -I -B scripts/diagnostics/run-a001-fase0-uv/image_package.py \
      --repo /src --source /src/image-source.json --output /opt/build-proof && \
    /opt/venv/bin/python -I -B scripts/diagnostics/run-a001-fase0-uv/image_package.py \
      --repo /src --source /src/image-source.json --output /opt/build-proof --install
RUN mkdir /opt/build-proof/diagnostics && cp image-source.json /opt/build-proof/image-source.json && \
    cp scripts/diagnostics/run-a001-fase0-uv/verify_distribution.py scripts/diagnostics/run-a001-fase0-uv/make_source_manifest.py scripts/diagnostics/run-a001-fase0-uv/check_python_origin.py /opt/build-proof/diagnostics/
FROM native AS runtime
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /opt/build-proof /opt/build-proof
COPY --from=builder /opt/prepare-proof /opt/prepare-proof
ENV PYTHONDONTWRITEBYTECODE=1 TMPDIR=/data/tmp TORCH_DEVICE=cpu \
    MODEL_CACHE_DIR=/data/cache/models HF_HOME=/data/cache/huggingface \
    XDG_CACHE_HOME=/data/cache FONT_DIR=/data/fonts \
    FONT_PATH=/data/fonts/GoNotoCurrent-Regular.ttf \
    RECOGNITION_RENDER_FONTS='{"all":"/data/fonts/GoNotoCurrent-Regular.ttf","zh":"/data/fonts/GoNotoCJKCore.ttf","ja":"/data/fonts/GoNotoCJKCore.ttf","ko":"/data/fonts/GoNotoCJKCore.ttf"}'
WORKDIR /data
RUN mkdir -p /data/tmp /data/input /data/cache/models /data/fonts
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1
CMD ["/opt/venv/bin/python", "-I", "-B", "-m", "marker_api_server"]
