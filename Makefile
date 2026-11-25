# markdown-for-llms in containers (MFL051). Tranche A1: the images of the product and the tests of the
# components against a real Docker daemon. The product itself still runs as before (API and worker on the
# host); `make up` and the other commands of the product come with tranche A2.
#
# Requirements: git, make and Docker Engine 26 or newer with the buildx and compose plugins, reachable by your
# user through a local socket. Nothing here needs a Python of the host.
.DEFAULT_GOAL := help
SHELL := /bin/sh

# The images are built by the capped builder of the toolchain image, with the options recorded in
# markdown_for_llms/engine/data/toolchain.json (`builder`); tests/unit/test_dockerignore.py keeps them equal.
BUILDER := mdf-bench-toolchain
BUILDKIT := moby/buildkit@sha256:c5f8862f52499773af85bc07695b2379ef08e8002352b0df2c4864c91d78dabe
BUILDER_MEMORY := 6g
IMAGE := mdf-app
DOCKERFILE := docker/app/Dockerfile
BUILD := docker buildx build --builder $(BUILDER) --load --provenance=false --sbom=false -f $(DOCKERFILE)
# One memory cap for the container that runs the component tests (the children have the caps of their requests).
TEST_MEMORY ?= 4g
TESTS ?= tests/container

.PHONY: help builder images test-components

help: ## list the commands
	@awk -F':.*## ' '/^[a-z][a-z-]*:.*## / { printf "  make %-16s %s\n", $$1, $$2 }' $(MAKEFILE_LIST)

builder:
	@docker buildx inspect $(BUILDER) > /dev/null 2>&1 || docker buildx create --name $(BUILDER) \
	  --driver docker-container --driver-opt image=$(BUILDKIT) --driver-opt memory=$(BUILDER_MEMORY) \
	  --driver-opt memory-swap=$(BUILDER_MEMORY)

images: builder ## build the images mdf-app:engine and mdf-app:dev from this working copy (the first build downloads 2-3 GB)
	$(BUILD) --target engine -t $(IMAGE):engine$(SUFFIX) .
	$(BUILD) --target dev -t $(IMAGE):dev$(SUFFIX) .

# The component tests need a real daemon: the images are built from this working copy with tags of this
# invocation, a compose project of this invocation starts `init` and a dedicated launcher
# (docker/dev/compose.yaml), and pytest runs tests/container in a container of the image `dev`. What the
# invocation created (containers, volumes, the two tags) is removed at the end, whatever the outcome; nothing
# else is touched. Evidence: .cache/test-components/<id>/. A run is green only when pytest exits 0 and its
# JUnit report, read with the function of scripts/bench_junit_guard.py, counts at least one test of
# tests/container and none skipped.
test-components: builder ## run tests/container against real containers (TESTS="..." selects; needs the toolchain image)
	@set -eu; \
	head=$$(git rev-parse HEAD); id="mdf-test-$$(git rev-parse --short HEAD)-$$$$"; \
	uid=$$(id -u); gid=$$(id -g); \
	if [ "$$uid" = 0 ]; then echo "test-components: run as an ordinary user, not as root" >&2; exit 1; fi; \
	endpoint=$$(docker context inspect --format '{{.Endpoints.docker.Host}}'); \
	case "$$endpoint" in unix://*) socket=$${endpoint#unix://} ;; \
	  *) echo "test-components: the Docker endpoint is not a local socket: $$endpoint" >&2; exit 1 ;; esac; \
	socket_gid=$$(ls -lnL "$$socket" | awk '{ print $$4 }'); \
	out="$(CURDIR)/.cache/test-components/$$id"; mkdir -p "$$out/fixtures"; \
	export MDF_COMPOSE_UID="$$uid" MDF_COMPOSE_GID="$$gid" MDF_COMPOSE_DOCKER_SOCKET="$$socket" \
	  MDF_COMPOSE_DOCKER_GID="$$socket_gid" MDF_COMPOSE_ENGINE_IMAGE="$(IMAGE):engine-$$id" \
	  MDF_COMPOSE_DEV_IMAGE="$(IMAGE):dev-$$id" MDF_COMPOSE_SOURCE="$(CURDIR)" MDF_COMPOSE_OUT="$$out" \
	  MDF_COMPOSE_TEST_MEMORY="$(TEST_MEMORY)"; \
	compose="docker compose -p $$id -f docker/dev/compose.yaml --profile suite --profile faults"; \
	cleanup() { \
	  $$compose logs --no-color launcher > "$$out/launcher.log" 2>&1 || true; \
	  $$compose stop -t 30 launcher > /dev/null 2>&1 || true; \
	  docker ps -a --filter "label=mdf.project" --format '{{.ID}} {{.Label "mdf.project"}}' | while read -r child project; do \
	    case "$$project" in "$$id"|"$$id"-*) docker container rm -f "$$child" > /dev/null 2>&1 || true ;; esac; done; \
	  $$compose down -v --remove-orphans > "$$out/down.log" 2>&1 || true; \
	  docker image ls --format '{{.Repository}}:{{.Tag}}' "$(IMAGE)" | while read -r image; do \
	    case "$$image" in *"-$$id") docker image rm "$$image" > /dev/null 2>&1 || true ;; esac; done; \
	}; \
	trap cleanup EXIT; trap 'exit 130' INT TERM; \
	echo "== component tests at $$head ($$id)"; echo "evidence: $$out"; \
	$(BUILD) --target engine -t "$(IMAGE):engine-$$id" . > "$$out/build-engine.log" 2>&1 \
	  || { tail -n 20 "$$out/build-engine.log" >&2; echo "TEST_COMPONENTS_FAIL build engine"; exit 1; }; \
	$(BUILD) --target dev -t "$(IMAGE):dev-$$id" . > "$$out/build-dev.log" 2>&1 \
	  || { tail -n 20 "$$out/build-dev.log" >&2; echo "TEST_COMPONENTS_FAIL build dev"; exit 1; }; \
	{ docker version --format 'docker {{.Server.Version}} api {{.Server.APIVersion}} {{.Server.Os}}/{{.Server.Arch}}'; \
	  docker image inspect --format '{{index .RepoTags 0}} {{.Id}}' "$(IMAGE):engine-$$id" "$(IMAGE):dev-$$id"; } > "$$out/images.txt"; \
	cat "$$out/images.txt"; \
	$$compose config --format json > "$$out/compose-config.json"; \
	$$compose up --no-start > "$$out/up.log" 2>&1 || { cat "$$out/up.log" >&2; echo "TEST_COMPONENTS_FAIL compose"; exit 1; }; \
	$$compose up -d launcher >> "$$out/up.log" 2>&1 || { cat "$$out/up.log" >&2; echo "TEST_COMPONENTS_FAIL launcher"; exit 1; }; \
	rc=0; \
	$$compose run --rm --no-deps -T suite /bin/sh -c ' \
	  rc=0; python -B -m pytest --noconftest -p no:cacheprovider -q -rfEs --junitxml /out/junit.xml "$$@" > /out/pytest.out 2>&1 || rc=$$?; \
	  tail -n 1 /out/pytest.out; echo "pytest exit $$rc"; \
	  python -I -B -c "import sys; sys.path.insert(0, \"scripts\"); import bench_junit_guard as g; ran, problems = g.check(\"/out/junit.xml\", \"tests.container.\", \"container\"); [print(\"problem: \" + p) for p in problems]; print(\"container guard: %d run, %d problems\" % (ran, len(problems))); sys.exit(1 if problems else 0)" || rc=1; \
	  exit $$rc' sh $(TESTS) || rc=$$?; \
	if [ "$$rc" -eq 0 ]; then echo "TEST_COMPONENTS_PASS $$head"; else tail -n 60 "$$out/pytest.out" >&2 || true; echo "TEST_COMPONENTS_FAIL $$head (exit $$rc)"; fi; \
	exit $$rc
