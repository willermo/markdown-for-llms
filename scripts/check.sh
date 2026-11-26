#!/bin/sh
# The host side of `make check` and `make test` (MFL051): what has to happen outside the containers before
# scripts/full-check.sh can run in one. POSIX sh; it needs git and Docker with buildx and compose, nothing else.
#
#   scripts/check.sh check                    the full check: a clean tree, a clone of HEAD of its own, images built
#                                             from it, then the chain in the container `mdf-full-check`, then the
#                                             tests of the components (`make test-components`, run from the same
#                                             clone): green only when both are
#   scripts/check.sh test [SELECTION...]      the same chain on the working copy, without asking for a clean tree;
#                                             SELECTION replaces `tests/`
#
# One full check at a time on a Docker daemon: the container of the check has the fixed name `mdf-full-check`, and
# Docker admits one container of a name, decided when it is CREATED. So the container is created (not started)
# before anything else of the check exists that could be shared, in a compose project named after this invocation;
# an invocation that does not get the name stops there and removes only its own project. It never touches
# `mdf-full-check`, which is of the other check. The name is held for the WHOLE gate: when the chain ends its
# container stays where it is (a container that exists keeps its name, running or not) until the tests of the
# components have ended too, and goes with the project of the check at the very end, also after a signal or an
# error. `make test` has no fixed name: its project, its volumes and the tags of its images are of the
# invocation, so several run side by side.
#
# The gate judges one commit: the chain and the tests of the components both run on the clone of HEAD made at the
# start, never on the working copy, which may change meanwhile. Before the verdict the clone must still be at
# that commit, with a clean tree, and the reservation must still be held.
#
# The images are built with the capped builder of the toolchain (the Makefile creates it). The weights of the
# Marker route are read from the models volume of the product, mounted read-only: MDF_COMPOSE_MODELS_VOLUME, by
# default the volume `mdf-models` of the compose project of this checkout. Evidence: .cache/full-check/<id>/.
set -eu

mode=${1:-}
case "$mode" in check|test) shift ;; *) echo "usage: scripts/check.sh check | test [PYTEST SELECTION...]" >&2; exit 2 ;; esac
top=$(git rev-parse --show-toplevel)
cd "$top"
[ "$mode" = test ] || [ "$#" -eq 0 ] || { echo "check: no arguments" >&2; exit 2; }
if [ "$mode" = check ] && [ -n "$(git status --porcelain)" ]; then echo "STOP tree not clean" >&2; exit 1; fi
head=$(git rev-parse HEAD)
uid=$(id -u); gid=$(id -g)
[ "$uid" != 0 ] || { echo "STOP run as an ordinary user, not as root" >&2; exit 1; }
arch=$(docker version --format '{{.Server.Arch}}')
[ "$arch" = amd64 ] || { echo "STOP the checks run on linux/amd64 only (this daemon: $arch)" >&2; exit 1; }
endpoint=$(docker context inspect --format '{{.Endpoints.docker.Host}}')
case "$endpoint" in unix://*) socket=${endpoint#unix://} ;; *) echo "STOP the Docker endpoint is not a local socket: $endpoint" >&2; exit 1 ;; esac
socket_gid=$(ls -lnL "$socket" | awk '{ print $4 }')
# the compose project of the product of this checkout: its name is the one compose derives from the directory
product=$(basename "$top" | tr '[:upper:]' '[:lower:]' | tr -cd 'a-z0-9_-')
weights=${MDF_COMPOSE_MODELS_VOLUME:-${product}_mdf-models}
docker volume inspect "$weights" > /dev/null 2>&1 \
  || { echo "STOP no models volume $weights: prepare the weights of the Marker route first (make prepare)" >&2; exit 1; }

# lowercase: it is also part of the name of a compose project
id="$(git rev-parse --short HEAD)-$(date -u +%Y%m%dt%H%M%Sz)-$$"
project="mdf-$mode-$id"
out="$top/.cache/full-check/$id"
mkdir -p "$out/proof"
BUILDER=${BUILDER:-mdf-bench-toolchain}
IMAGE=${IMAGE:-mdf-app}
build="docker buildx build --builder $BUILDER --load --provenance=false --sbom=false"

if [ "$mode" = check ]; then
  # a clone of HEAD of its own: a real `.git` directory also from a worktree, and nothing of the working copy
  git clone --quiet --no-hardlinks --no-tags "$top" "$out/src"
  [ "$(git -C "$out/src" rev-parse HEAD)" = "$head" ] || { echo "STOP the clone is not at $head" >&2; exit 1; }
  if ! git -C "$out/src" show-ref --verify --quiet refs/heads/dev \
     && git -C "$out/src" show-ref --verify --quiet refs/remotes/origin/dev; then
    git -C "$out/src" branch --quiet dev refs/remotes/origin/dev
  fi
  src="$out/src"; target=/check/src; service=check
  # the mount point of the evidence, inside the clone and made by the user: Docker would make it as root
  mkdir -p "$src/.cache/full-check/proof"
  proof_host="$out/proof"; proof="$target/.cache/full-check/proof"
  common=
else
  src="$top"; target="$top"; service=test
  mkdir -p "$top/.cache/full-check/$id/proof"
  proof_host="$out/proof"; proof="$top/.cache/full-check/$id/proof"
  # the git directory of the checkout, mounted read-only at its own path over the working copy: the suite reads
  # it and never writes it. In a worktree it is the common directory, elsewhere, and git finds it at that path.
  common=$(git rev-parse --path-format=absolute --git-common-dir)
fi

export MDF_COMPOSE_UID="$uid" MDF_COMPOSE_GID="$gid" MDF_COMPOSE_DOCKER_SOCKET="$socket" MDF_COMPOSE_DOCKER_GID="$socket_gid" \
  MDF_COMPOSE_ENGINE_IMAGE="$IMAGE:engine-$id" MDF_COMPOSE_DEV_IMAGE="$IMAGE:dev-$id" MDF_COMPOSE_SOURCE="$src" \
  MDF_COMPOSE_TARGET="$target" MDF_COMPOSE_OUT="$out" MDF_COMPOSE_PROOF="$proof_host" MDF_COMPOSE_PROOF_TARGET="$proof" \
  MDF_COMPOSE_MODELS_NAME="$weights" MDF_COMPOSE_WEIGHTS_VOLUME="$weights" MDF_COMPOSE_COMMON="${common:-/nonexistent}" \
  MDF_COMPOSE_HEAD="$head" MDF_COMPOSE_TEST_MEMORY="${MDF_COMPOSE_CHECK_MEMORY:-12g}" \
  MDF_COMPOSE_CHECK_MEMORY="${MDF_COMPOSE_CHECK_MEMORY:-12g}"
compose="docker compose -p $project -f $src/docker/dev/compose.yaml --profile $service"

settle() {
  # the launcher of this project and the children it started; the containers of the project stay where they are
  $compose logs --no-color launcher > "$out/launcher.log" 2>&1 || true
  $compose stop -t 30 launcher > /dev/null 2>&1 || true
  docker ps -a --filter "label=mdf.project=$project" --format '{{.ID}}' | while read -r child; do
    docker container rm -f "$child" > /dev/null 2>&1 || true; done
}
cleanup() {
  # the children of this project only, then the project itself: its containers (with the one that holds the name
  # of the check), its network, its volumes, and the two tags of the invocation
  settle
  $compose down -v --remove-orphans > "$out/down.log" 2>&1 || true
  docker image rm "$IMAGE:engine-$id" "$IMAGE:dev-$id" > /dev/null 2>&1 || true
}
trap cleanup EXIT
trap 'exit 130' INT TERM

echo "== $mode at $head ($project)"
echo "evidence: $out"
# the configuration of compose of every profile, among the evidence
for profile in check test suite faults; do
  docker compose -p "$project" -f "$src/docker/dev/compose.yaml" --profile "$profile" config --format json \
    > "$out/proof/compose-dev-$profile.json" 2> "$out/compose-config.err" \
    || { cat "$out/compose-config.err" >&2; echo "STOP compose config ($profile)" >&2; exit 1; }
done
# the configuration of the product itself, as compose reads the files of the tree under test, with the defaults of
# an env file that is not there (the env file of the user is never looked at by a check): what
# tests/unit/test_compose_config.py judges
( unset MDF_COMPOSE_ENGINE_IMAGE MDF_COMPOSE_DEV_IMAGE
  cd "$src" && . scripts/env-preflight.sh && compose_environment "$out/absent/.env" \
  && docker compose --env-file /dev/null --profile tools config --format json > "$out/proof/compose-product.json" \
  && docker compose --env-file /dev/null -f compose.yaml -f compose.public.yaml --profile tools config --format json \
       > "$out/proof/compose-public.json" ) 2> "$out/compose-config.err" \
  || { cat "$out/compose-config.err" >&2; echo "STOP compose config (product)" >&2; exit 1; }
# building touches no container
$build -f "$src/docker/app/Dockerfile" --target engine -t "$IMAGE:engine-$id" "$src" > "$out/build-engine.log" 2>&1 \
  || { tail -n 20 "$out/build-engine.log" >&2; echo "STOP build engine" >&2; exit 1; }
$build -f "$src/docker/app/Dockerfile" --target dev -t "$IMAGE:dev-$id" "$src" > "$out/build-dev.log" 2>&1 \
  || { tail -n 20 "$out/build-dev.log" >&2; echo "STOP build dev" >&2; exit 1; }
# the tags of this invocation, each with the ID it names (an image can carry the tags of other invocations too)
{ docker version --format 'docker {{.Server.Version}} api {{.Server.APIVersion}} {{.Server.Os}}/{{.Server.Arch}}'
  for image in "$IMAGE:engine-$id" "$IMAGE:dev-$id"; do
    printf '%s %s\n' "$image" "$(docker image inspect --format '{{.Id}}' "$image")"
  done; } > "$out/proof/images.txt"
cat "$out/proof/images.txt"

# the reservation: the container of the check is created, not started, before the launcher and its volumes are used
if ! $compose up --no-start --no-deps "$service" > "$out/reserve.log" 2>&1; then
  if [ "$mode" = check ] && grep -q 'mdf-full-check' "$out/reserve.log"; then
    echo "STOP another full check is running on this Docker daemon (the container mdf-full-check exists)." >&2
    echo "     If no check is running, one was interrupted: look at it with" >&2
    echo "       docker ps -a --filter name=mdf-full-check" >&2
    echo "     and remove it with \`docker rm mdf-full-check\`." >&2
  else
    cat "$out/reserve.log" >&2
    echo "STOP the container of the $mode was not created" >&2
  fi
  exit 1
fi
$compose up -d launcher > "$out/up.log" 2>&1 || { cat "$out/up.log" >&2; echo "STOP launcher" >&2; exit 1; }

rc=0
container=$($compose ps -a -q "$service")
if [ "$mode" = check ]; then
  docker start --attach "$container" || rc=$?
else
  # the selection of `make test`, one argument per line, read by the command of the container
  : > "$out/selection"
  for argument in "$@"; do printf '%s\n' "$argument" >> "$out/selection"; done
  docker start --attach "$container" || rc=$?
fi
[ "$rc" -eq 0 ] || echo "$mode: exit $rc (evidence in $out)" >&2

if [ "$mode" = check ]; then
  # E-6: the tests of the components are a step of the same gate. They need a driver with the Docker socket, which
  # the container of the check never has: they run in their own project, after the chain. Two things make them
  # part of THIS gate (SOL-A2-B1):
  #   - they run from the clone of HEAD, as the chain did: `make test-components` works on the directory it is
  #     run in, and here that is the clone, never the working copy;
  #   - they run under the reservation: the container of the check has ended and is still there, so its name is
  #     taken until `cleanup`, which the trap runs at the end, whatever the outcome (a red step, a signal). Only
  #     the launcher of the chain and its children go now.
  settle
  reserved() {    # the container of this invocation still exists under the name of the check
    held=$(docker container inspect --format '{{.Name}} {{.State.Status}}' "$container" 2> /dev/null) || return 1
    case "$held" in "/mdf-full-check "*) ;; *) return 1 ;; esac
  }
  frozen() {      # the clone is still the commit under test, and nothing in it changed
    [ "$(git -C "$src" rev-parse HEAD)" = "$head" ] && [ -z "$(git -C "$src" status --porcelain)" ]
  }
  components="not run"
  lost=
  if ! reserved; then lost="the reservation was lost: the container of the check is not there any more"
  elif ! frozen; then lost="the clone is not at $head with a clean tree any more"
  else
    echo "== component tests from the clone of HEAD: $src"
    echo "reservation held until they end: ${held#/} (container $container)"
    components=0
    make -C "$src" --no-print-directory test-components || components=$?
    # the evidence of the components is in the clone; a link leaves it where `make test-components` puts it
    mkdir -p "$top/.cache/test-components"
    for found in "$src"/.cache/test-components/*/; do
      [ -d "$found" ] || continue
      name=$(basename "$found")
      [ -e "$top/.cache/test-components/$name" ] || [ -L "$top/.cache/test-components/$name" ] \
        || ln -s "../full-check/$id/src/.cache/test-components/$name" "$top/.cache/test-components/$name"
      echo "evidence of the component tests: $top/.cache/test-components/$name"
    done
    if ! reserved; then lost="the reservation was lost during the component tests"
    elif ! frozen; then lost="the clone changed during the component tests: it is not at $head with a clean tree"; fi
  fi
  [ -z "$lost" ] || echo "STOP $lost" >&2
  if [ "$rc" -eq 0 ] && [ "$components" = 0 ] && [ -z "$lost" ]; then
    echo "CHECK_PASS $head (full check and component tests)"
  else
    case "$components" in "not run") said="component tests not run" ;; *) said="component tests exit $components" ;; esac
    echo "CHECK_FAIL $head (full check exit $rc, $said${lost:+, $lost})"
    exit 1
  fi
fi
exit "$rc"
