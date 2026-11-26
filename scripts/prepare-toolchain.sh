#!/bin/sh
# The host side of the toolchain image of the engine (`make prepare`, MFL051): no container of the product has a
# Docker client, so the build is done here. POSIX sh; it needs Docker with buildx and compose, nothing else.
#
#   scripts/prepare-toolchain.sh ENV_FILE [--no-cache]
#
# When the image tagged `mdf-bench-toolchain:<content[:16]>` exists and the launcher verifies it (its config and
# its content manifest against markdown_for_llms/engine/data/toolchain.json), nothing is done. Otherwise, or with
# `--no-cache`:
#   1. the build context scripts/bench-toolchain/ is copied under .cache/prepare-toolchain/, every file with mode
#      0644 and one fixed time (BuildKit keys its cache on them), and the copy is verified against toolchain.json
#      in a container of the image `engine` without network;
#   2. the copy is built with the capped builder (the Makefile creates it) and tagged `mdf-bench-toolchain:candidate`;
#   3. `mdf prepare --toolchain` runs in the `cli` container: through the launcher, every file of the candidate is
#      verified against its content manifest and the self-test runs on it;
#   4. only then the image gets its tag, by ID, and the tag `candidate` is removed.
# The launcher of the project is started when it does not run (with `init` before it); nothing else is.
set -eu

[ "$#" -ge 1 ] || { echo "usage: scripts/prepare-toolchain.sh ENV_FILE [--no-cache]" >&2; exit 2; }
env_file=$1
no_cache=
[ "${2:-}" != "--no-cache" ] || no_cache=--no-cache
top=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd -P)
cd "$top"
. scripts/env-preflight.sh
compose_environment "$env_file"
BUILDER=${BUILDER:-mdf-bench-toolchain}
ENGINE_IMAGE=${MDF_COMPOSE_ENGINE_IMAGE:-mdf-app:engine}
compose="docker compose --env-file /dev/null --profile tools"
python="/opt/mdf/venv/bin/python -I -B"
closed="--rm --network none --read-only --cap-drop ALL --security-opt no-new-privileges --user $MDF_COMPOSE_UID:$MDF_COMPOSE_GID"

# shellcheck disable=SC2086
tag=$(docker run $closed "$ENGINE_IMAGE" $python -m markdown_for_llms.engine.toolchain tag)
case "$tag" in mdf-bench-toolchain:????????????????) ;; *) echo "prepare: the tag of the toolchain image cannot be read" >&2; exit 1 ;; esac

$compose up -d launcher > /dev/null
if [ -z "$no_cache" ] && docker image inspect "$tag" > /dev/null 2>&1; then
  # shellcheck disable=SC2086
  if $compose run --rm -T --no-deps cli $python -m markdown_for_llms.engine.toolchain verify; then exit 0; fi
  echo "prepare: the image $tag does not pass the verification: it is built again"
fi

id="$(date -u +%Y%m%dt%H%M%Sz)-$$"
out="$top/.cache/prepare-toolchain/$id"
mkdir -p "$out/context"
for file in scripts/bench-toolchain/*; do
  [ -f "$file" ] && [ ! -L "$file" ] || { echo "prepare: the build context holds something that is not a regular file: $file" >&2; exit 1; }
  cp -- "$file" "$out/context/"
done
chmod 0644 "$out/context"/*
# 2026-01-01T00:00:00Z: `toolchain.CONTEXT_MTIME`
TZ=UTC touch -t 202601010000.00 "$out/context"/*
# shellcheck disable=SC2086
docker run $closed -v "$out/context:/context:ro" "$ENGINE_IMAGE" $python -m markdown_for_llms.engine.toolchain context /context

echo "prepare: building the toolchain image (log: $out/build.log)"
# shellcheck disable=SC2086
docker buildx build --builder "$BUILDER" --load --provenance=false --sbom=false --progress=plain --iidfile "$out/iid" \
  $no_cache -f "$out/context/Dockerfile" "$out/context" > "$out/build.log" 2>&1 \
  || { tail -n 20 "$out/build.log" >&2; echo "prepare: the build of the toolchain image failed" >&2; exit 1; }
built=$(cat "$out/iid")
docker tag "$built" mdf-bench-toolchain:candidate

untag() { docker image rm mdf-bench-toolchain:candidate > /dev/null 2>&1 || true; }
trap untag EXIT
# shellcheck disable=SC2086
$compose run --rm -T --no-deps cli mdf prepare --toolchain > "$out/verify.log" 2>&1 \
  || { tail -n 20 "$out/verify.log" >&2; echo "prepare: the candidate did not pass the verification" >&2; exit 1; }
cat "$out/verify.log"
# the last line: `toolchain candidate ok <tag> <image ID> docker <version>`
verdict=$(tail -n 1 "$out/verify.log")
# shellcheck disable=SC2086
set -- $verdict
if [ "$1 $2 $3" != "toolchain candidate ok" ] || [ "$4" != "$tag" ] || [ "$5" != "$built" ]; then
  echo "prepare: the verification did not name the image that was built" >&2; exit 1
fi
docker tag "$built" "$tag"
echo "prepare: $tag is $built"
