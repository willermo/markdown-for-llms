# syntax=docker/dockerfile:1@sha256:4edf897a3ffa55b89f906fc8cc78afdb3f1834cc9c7083565e611a8a7d5fe99e
# Reference toolchain of `mdf bench`: the tools of the derivations, of the D06 projection and of the
# self-check (pandoc, poppler, qpdf, Ghostscript, DjVuLibre, LibreOffice), never the measured
# conversion. Built only by `mdf bench prepare --toolchain`, which verifies this directory against
# markdown_for_llms/bench/data/toolchain.json first. Every input is pinned by digest or SHA-256:
# the base image, a Debian snapshot, the pixi release, the conda-forge lock (pixi.lock), the
# DjVuLibre 3.5.30 source tarball (it reports itself as 3.5.29) and the LibreOffice 26.8.3.1 debs
# of the permanent archive. The bench runs the tools with `--network none`, a read-only root, no
# capabilities and the caller's user (markdown_for_llms/bench/tools.py).
ARG BASE=debian:trixie-slim@sha256:a72b4433de1cf7d6bae0c4cee74d0547f58601a643b3375f070c51600d63b6b2
ARG SNAPSHOT=20261113T000000Z

FROM ${BASE} AS apt
ARG SNAPSHOT
RUN rm -f /etc/apt/sources.list.d/debian.sources \
 && echo "deb [check-valid-until=no] http://snapshot.debian.org/archive/debian/${SNAPSHOT} trixie main" > /etc/apt/sources.list \
 && echo "deb [check-valid-until=no] http://snapshot.debian.org/archive/debian-security/${SNAPSHOT} trixie-security main" >> /etc/apt/sources.list

FROM apt AS build
RUN apt-get -o Acquire::Retries=3 update && apt-get install -y --no-install-recommends ca-certificates g++ make pkg-config \
    libjpeg62-turbo-dev libtiff-dev && rm -rf /var/lib/apt/lists/*
ADD --checksum=sha256:6d2a576bde0a7b26533a068fec0aea6e4ee21c36668efc1dc73fce90110906ae \
    https://github.com/prefix-dev/pixi/releases/download/v0.84.2/pixi-x86_64-unknown-linux-musl.tar.gz /tmp/pixi.tar.gz
RUN tar -xzf /tmp/pixi.tar.gz -C /usr/local/bin pixi
WORKDIR /opt/toolchain
COPY pixi.toml pixi.lock ./
RUN PIXI_CACHE_DIR=/tmp/pixi-cache PIXI_NO_UPDATE_CHECK=1 pixi install --frozen && rm -rf /tmp/pixi-cache
ADD --checksum=sha256:ee5e457d4cfebe566f94b99e5e3d3cc7f5c79ddb741c2ac2ba2e456f00329644 \
    https://downloads.sourceforge.net/project/djvu/DjVuLibre/3.5.30/djvulibre-3.5.30.tar.gz /tmp/djvulibre.tar.gz
RUN mkdir /tmp/djvu && tar -xzf /tmp/djvulibre.tar.gz -C /tmp/djvu --strip-components=1 && cd /tmp/djvu \
 && ./configure --prefix=/opt/djvulibre --disable-desktopfiles --disable-static -q && make -j4 -s && make -s install
ADD --checksum=sha256:e7cd3538e85818ff9dd9bb65386769fde927011242aace5a144989f4000dcabf \
    https://downloadarchive.documentfoundation.org/libreoffice/old/26.8.3.1/deb/x86_64/LibreOffice_26.8.3.1_Linux_x86-64_deb.tar.gz /tmp/lo.tar.gz
RUN mkdir /tmp/lo && tar -xzf /tmp/lo.tar.gz -C /tmp/lo --strip-components=1

FROM apt
RUN apt-get -o Acquire::Retries=3 update \
 && apt-get install -y --no-install-recommends libjpeg62-turbo libtiff6 libxinerama1 libdbus-1-3 libcups2t64 libnss3 \
    libx11-xcb1 libxrandr2 libxext6 libxrender1 libsm6 libice6 libglib2.0-0t64 libcairo2 libxslt1.1 fontconfig \
    fonts-dejavu-core fonts-liberation2 fonts-crosextra-carlito fonts-crosextra-caladea \
 && rm -rf /var/lib/apt/lists/*
RUN --mount=type=bind,from=build,source=/tmp/lo/DEBS,target=/tmp/debs dpkg -i /tmp/debs/*.deb \
 && ln -s /opt/libreoffice26.8/program/soffice /usr/local/bin/soffice
COPY --from=build /opt/toolchain/.pixi/envs/default /opt/toolchain/.pixi/envs/default
COPY --from=build /opt/djvulibre /opt/djvulibre
COPY --chmod=0755 content-manifest.sh /usr/local/bin/mdf-bench-content-manifest
ENV PATH=/opt/djvulibre/bin:/opt/toolchain/.pixi/envs/default/bin:/usr/local/bin:/usr/bin:/bin LC_ALL=C.UTF-8 LANG=C.UTF-8 \
    LD_LIBRARY_PATH=/opt/djvulibre/lib
RUN fc-cache -s >/dev/null && mdf-bench-content-manifest > /etc/mdf-bench-toolchain.sha256
