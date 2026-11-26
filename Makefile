# markdown-for-llms in containers (MFL051): the product (`make up`), its images, the suite and the full check in
# containers, and the tests of the components against a real Docker daemon.
#
# Requirements: git, make and Docker Engine 26 or newer (cgroup v2, not rootless) with the buildx and compose
# plugins, reachable by your user through a local socket. Nothing here needs a Python of the host.
#
# The product: `cp .env.example .env`, edit it, `make up`. The archive is ./data and the state ./state
# (MDF_COMPOSE_DATA, MDF_COMPOSE_STATE in `.env`); `make down` stops everything and keeps both, with the weights.
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
# `make test TESTS="..."` and `make test-components TESTS="..."`: a pytest selection in place of the default
TESTS ?=

# The env file of the user. Its keys are read only by the worker and the tools, in their containers; `make` and
# compose read from it the few names of scripts/env-preflight.sh, and compose never opens it (`--env-file /dev/null`).
ENV ?= .env
COMPOSE := docker compose --env-file /dev/null
TOOLS := $(COMPOSE) --profile tools
# every recipe of the product starts here: the functions of the preflight and what compose.yaml interpolates
PRODUCT = set -eu; . scripts/env-preflight.sh; compose_environment "$(ENV)"
# The `config` step as a gate, before any service is started or created again: it validates the env file and
# compares its settings with the identity of the archive. A service refused by its archive would leave with 1 and
# compose would start it again for ever; here the command stops instead, with the reason and the way out. When it
# passes it says nothing: its answer (the names of the settings it wrote) is shown only with a refusal.
GATE = $(COMPOSE) run --rm -T config
# the volumes `make clean` removes: what the services rebuild by themselves. Never the weights, the archive or the state.
EPHEMERAL := mdf-work mdf-run mdf-launcher mdf-api mdf-settings
# `make prepare PREPARE=--no-cache` builds the toolchain image again; `CPU_SERVER=1` also prepares the local CPU server
PREPARE ?=
CPU_SERVER ?=
# `make logs SERVICE=worker FOLLOW=1`
SERVICE ?=
FOLLOW ?=
# the arguments of `mdf bench` for `make bench`
BENCH ?= run --level quick
# where `make install-cli` writes `mdf`
BIN ?= $(HOME)/.local/bin
# seconds `make up` waits for the first heartbeat of the worker
UP_WAIT ?= 180

.PHONY: help builder images test check test-components up down restart rebuild status logs prepare install-cli bench clean clean-archive

help: ## list the commands
	@awk -F':.*## ' '/^[a-z][a-z-]*:.*## / { printf "  make %-16s %s\n", $$1, $$2 }' $(MAKEFILE_LIST)

builder:
	@docker buildx inspect $(BUILDER) > /dev/null 2>&1 || docker buildx create --name $(BUILDER) \
	  --driver docker-container --driver-opt image=$(BUILDKIT) --driver-opt memory=$(BUILDER_MEMORY) \
	  --driver-opt memory-swap=$(BUILDER_MEMORY)

images: builder ## build the images mdf-app:engine and mdf-app:dev from this working copy (the first build downloads 2-3 GB)
	$(BUILD) --target engine -t $(IMAGE):engine$(SUFFIX) .
	$(BUILD) --target dev -t $(IMAGE):dev$(SUFFIX) .

# --- the product ------------------------------------------------------------------------------------------
# `make up`, in order, each step skipping what is already done and verified:
#   1. preflight: the host (an ordinary user, Docker Engine 26+, cgroup v2, a local socket), the env file on its
#      original path (a link, another type or another owner stop here; another mode becomes 0600, and it is said),
#      the folders of the archive and of the state, made by the user;
#   2. the images `engine` and `api`, built from this working copy with the capped builder (layers from the cache);
#   3. the access to the Docker socket with the identity the launcher will have (the user, the group of the
#      socket): refused here, with the reason, in place of a launcher that restarts forever;
#   4. `init` and `config`: an env file the services or the archive would refuse stops here, with the reason,
#      before anything is started;
#   5. `make prepare`: the toolchain image and the weights of the Marker route;
#   6. the services; the first heartbeat of the worker is awaited through nginx, the only door, and the URL is
#      printed. A service that Docker had to start again in the meantime (it leaves with an error: the restart
#      policy starts it again for ever) is named, with the last error line of its log, also when the wait ends well.
up: builder ## start the product: preflight, images, toolchain and weights, then the services on 127.0.0.1 (the first run downloads 2-3 GB)
	@set -eu; . scripts/env-preflight.sh; \
	docker_host_check || { echo "up: stopped at the preflight of the host" >&2; exit 1; }; \
	case "$$(env_state "$(ENV)")" in \
	  ok) ;; \
	  mode) chmod 600 "$(ENV)"; echo "up: $(ENV) was readable by others or had another mode: it is 0600 now" ;; \
	  missing) echo "up: no $(ENV): cp .env.example .env, then edit it" >&2; exit 1 ;; \
	  link) echo "up: $(ENV) is a link: the env file is a regular file (its keys are read where it is)" >&2; exit 1 ;; \
	  type) echo "up: $(ENV) is not a regular file" >&2; exit 1 ;; \
	  owner) echo "up: $(ENV) belongs to another user" >&2; exit 1 ;; \
	esac; \
	compose_environment "$(ENV)"; \
	for dir in "$$MDF_COMPOSE_DATA" "$$MDF_COMPOSE_STATE" "$$MDF_COMPOSE_STATE/config"; do \
	  [ -e "$$dir" ] || (umask 077; mkdir -p "$$dir"); \
	  if [ -L "$$dir" ] || [ ! -d "$$dir" ] || [ ! -O "$$dir" ]; then \
	    echo "up: $$dir is not a directory of yours (a link, a file, or of another user)" >&2; exit 1; fi; \
	done; \
	echo "== images"; \
	$(BUILD) --target engine -t $(IMAGE):engine .; \
	$(BUILD) --target api -t $(IMAGE):api .; \
	echo "== access to Docker as the launcher"; \
	$(COMPOSE) run --rm --no-deps -T launcher /opt/mdf/venv/bin/python -I -B -m markdown_for_llms.launcher check \
	  --docker=/run/docker.sock \
	  || { echo "up: the launcher would not reach the Docker socket $$MDF_COMPOSE_DOCKER_SOCKET (group $$MDF_COMPOSE_DOCKER_GID) as your user; nothing was started" >&2; exit 1; }; \
	echo "== settings"; \
	refused=$$($(GATE) 2> /dev/null) || { echo "$$refused" >&2; echo "up: the env file is refused (above): nothing was started" >&2; exit 1; }; \
	$(MAKE) --no-print-directory prepare ENV="$(ENV)" PREPARE="$(PREPARE)" CPU_SERVER="$(CPU_SERVER)"; \
	echo "== services"; \
	$(COMPOSE) up -d --no-build --remove-orphans; \
	url="http://127.0.0.1:$$MDF_PORT"; waited=0; \
	until $(COMPOSE) exec -T nginx wget -q -O - --header "Host: 127.0.0.1:$$MDF_PORT" http://127.0.0.1:8080/api/v1/readiness 2> /dev/null \
	    | grep -q '"ready": *true'; do \
	  if [ "$$waited" -ge "$(UP_WAIT)" ]; then \
	    restart_report >&2 || true; \
	    echo "up: the worker is not ready after $(UP_WAIT) s: look at \`make status\` and \`make logs\`" >&2; exit 1; fi; \
	  sleep 2; waited=$$((waited + 2)); \
	done; \
	restart_report || echo "up: a service above left and Docker started it again (it does so for ever when the service leaves with an error): \`make logs SERVICE=...\`"; \
	echo "up: ready at $$url (archive $$MDF_COMPOSE_DATA)"; \
	$(MAKE) --no-print-directory status ENV="$(ENV)"

prepare: builder ## the toolchain image and the weights of the Marker route (PREPARE=--no-cache builds again; CPU_SERVER=1 adds the local CPU server, 1.4 GB)
	@$(PRODUCT); \
	echo "== toolchain image"; \
	scripts/prepare-toolchain.sh "$(ENV)" $(PREPARE); \
	echo "== weights of the Marker route"; \
	$(TOOLS) run --rm -T --no-deps cli mdf prepare --weights; \
	if [ -n "$(CPU_SERVER)" ]; then echo "== local CPU server"; $(TOOLS) run --rm -T --no-deps cli mdf prepare --cpu-server; fi

# `docker compose restart` would not read the env file again: the services are created again. Nothing is built
# and nothing is downloaded; the env file is looked at again, and one that is not safe stops only the routes
# with a key (the worker says which), not the start.
restart: ## create the services again (after a change of .env); no build, no download
	@$(PRODUCT); \
	case "$$(env_state "$(ENV)")" in \
	  ok) ;; \
	  mode) echo "restart: $(ENV) is not 0600: the routes with a key will not work (\`make up\` gives it its mode)" ;; \
	  missing) echo "restart: no $(ENV)" >&2; exit 1 ;; \
	  *) echo "restart: $(ENV) is a link, not a regular file, or of another user" >&2; exit 1 ;; \
	esac; \
	refused=$$($(GATE) 2> /dev/null) || { echo "$$refused" >&2; echo "restart: the env file is refused (above): the services were not touched" >&2; exit 1; }; \
	$(COMPOSE) up -d --force-recreate --no-build --remove-orphans

down: ## stop the product; the archive, the state and the weights stay
	@$(PRODUCT); \
	$(TOOLS) down --remove-orphans; \
	docker ps -a -q --filter "label=mdf.project=$$COMPOSE_PROJECT_NAME" | while read -r child; do \
	  docker container rm -f "$$child" > /dev/null 2>&1 || true; done

rebuild: builder ## build the images again without the cache of the layers, then `make up`
	@set -eu; \
	$(BUILD) --no-cache --target engine -t $(IMAGE):engine .; \
	$(BUILD) --target api -t $(IMAGE):api .; \
	$(MAKE) --no-print-directory up ENV="$(ENV)"

status: ## the services, the state of every route (what is missing, and where to write it) and a service that keeps restarting
	@$(PRODUCT); \
	$(COMPOSE) ps --format 'table {{.Service}}\t{{.State}}\t{{.Status}}'; \
	if $(COMPOSE) exec -T nginx wget -q -O /dev/null --header "Host: 127.0.0.1:$$MDF_PORT" http://127.0.0.1:8080/health 2> /dev/null; then \
	  echo "http://127.0.0.1:$$MDF_PORT answers"; \
	else echo "status: the API does not answer through nginx (is the product up? \`make up\`)"; fi; \
	$(COMPOSE) exec -T worker mdf --env-file=/run/mdf/settings/settings.env routes 2> /dev/null \
	  || echo "status: the routes above are not fresh, or the worker is not running (\`make logs SERVICE=worker\`)"; \
	restart_report || echo "status: a service above left and Docker started it again (it does so for ever when the service leaves with an error): \`make logs SERVICE=...\`"; \
	if [ -f "$(ENV)" ] && ! refused=$$(MDF_COMPOSE_CONFIG_MODE=--check $(COMPOSE) run --rm -T --no-deps config 2> /dev/null); then \
	  echo "status: the env file as it is now is refused; the services keep the settings of their start:"; \
	  echo "$$refused"; fi

logs: ## the logs of the services (SERVICE=worker for one, FOLLOW=1 to follow)
	@$(PRODUCT); $(COMPOSE) logs --no-color --tail=200 $(if $(FOLLOW),-f) $(SERVICE)

# The bench in the `bench` container, through the launcher of the product (`make up` first): its cache and its
# runs are in the state folder, under bench/. Nothing is downloaded by `run`: the sources come from
# `make bench BENCH="fetch --level quick"`, once.
bench: ## run the bench of the local routes in a container (BENCH="..." gives other arguments to `mdf bench`)
	@$(PRODUCT); \
	[ -n "$$($(TOOLS) ps --status running -q launcher 2> /dev/null)" ] \
	  || { echo "bench: the product is not running: run make up" >&2; exit 1; }; \
	$(TOOLS) run --rm -T --no-deps bench mdf bench $(BENCH)

# `mdf` on the PATH: a three-line file that runs bin/mdf of THIS checkout by its absolute path. When the checkout
# is moved or removed the file says so (exit 127): run `make install-cli` again from the new place.
install-cli: ## put `mdf` on your PATH: a small file in BIN (default ~/.local/bin) that runs bin/mdf of this checkout
	@set -eu; . scripts/env-preflight.sh; install_cli "$(CURDIR)" "$(BIN)"

# What the services rebuild by themselves: the containers, the networks and the volumes of work. The archive,
# the state and the weights are never named here.
clean: ## remove the containers, the networks and the work volumes of the product; never the archive, the state or the weights
	@$(PRODUCT); \
	$(TOOLS) down --remove-orphans; \
	docker ps -a -q --filter "label=mdf.project=$$COMPOSE_PROJECT_NAME" | while read -r child; do \
	  docker container rm -f "$$child" > /dev/null 2>&1 || true; done; \
	for volume in $(EPHEMERAL); do \
	  docker volume rm "$${COMPOSE_PROJECT_NAME}_$$volume" > /dev/null 2>&1 || true; done; \
	echo "clean: kept the archive, the state and the volumes of the weights"

# The archive is the documents of the user: it goes only by its own identifier, read from the archive itself,
# and only with the product stopped.
clean-archive: ## delete the archive (CONFIRM=<id of the archive>, with the product stopped); the state and the weights stay
	@$(PRODUCT); \
	identity="$$MDF_COMPOSE_DATA/identity.json"; \
	[ -f "$$identity" ] && [ ! -L "$$MDF_COMPOSE_DATA" ] || { echo "clean-archive: no archive at $$MDF_COMPOSE_DATA" >&2; exit 1; }; \
	id=$$(sed -n 's/.*"id"[[:space:]]*:[[:space:]]*"\([0-9a-f]\{32\}\)".*/\1/p' "$$identity" | head -n 1); \
	[ -n "$$id" ] || { echo "clean-archive: the identifier of the archive cannot be read" >&2; exit 1; }; \
	if [ "$(CONFIRM)" != "$$id" ]; then \
	  echo "clean-archive: this deletes the archive $$MDF_COMPOSE_DATA with every document in it." >&2; \
	  echo "               To confirm: make clean-archive CONFIRM=$$id" >&2; exit 1; fi; \
	if [ -n "$$($(TOOLS) ps -q)" ]; then echo "clean-archive: the product is running: \`make down\` first" >&2; exit 1; fi; \
	rm -rf -- "$$MDF_COMPOSE_DATA"; \
	echo "clean-archive: removed $$MDF_COMPOSE_DATA"

# --- the checks -------------------------------------------------------------------------------------------
# The suite and the full check run in a container of the image `dev`, next to a launcher of their own
# (scripts/check.sh, docker/dev/compose.yaml): the engine starts its children only through a launcher, and a
# check (the payload gate) makes sure they run the code under test. Linux x86_64 only.
test: builder ## run the suite of tests/ on this working copy, in a container (TESTS="..." selects)
	@scripts/check.sh test $(TESTS)

check: builder ## the full check on a clone of HEAD (a clean tree), then the tests of the components from the same clone; one at a time on a daemon
	@scripts/check.sh check

# The component tests need a real daemon: the images are built from this working copy (from the clone of HEAD when
# `make check` runs this target there, as the last step of its gate) with tags of this invocation, a compose project of this invocation starts `init` and a dedicated launcher
# (docker/dev/compose.yaml), and pytest runs tests/container in a container of the image `dev`. What the
# invocation created (containers, volumes, the two tags) is removed at the end, whatever the outcome; nothing
# else is touched. The cleanup removes the driver first: an interrupted `compose run` leaves its container
# running, and while it runs it still asks for children, after the children were listed. Then the launcher is
# stopped, the children go by their label, and the project with its volumes. Evidence:
# .cache/test-components/<id>/. A run is green only when pytest exits 0 and its
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
	  MDF_COMPOSE_TEST_MEMORY="$(TEST_MEMORY)" MDF_COMPOSE_MODELS_NAME="$${id}_mdf-models"; \
	compose="docker compose -p $$id -f docker/dev/compose.yaml --profile suite --profile faults"; \
	cleanup() { \
	  $$compose logs --no-color launcher > "$$out/launcher.log" 2>&1 || true; \
	  docker ps -a -q --filter "label=com.docker.compose.project=$$id" --filter "label=com.docker.compose.service=suite" | while read -r driver; do \
	    docker container rm -f "$$driver" > /dev/null 2>&1 || true; done; \
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
	  for image in "$(IMAGE):engine-$$id" "$(IMAGE):dev-$$id"; do \
	    printf '%s %s\n' "$$image" "$$(docker image inspect --format '{{.Id}}' "$$image")"; done; } > "$$out/images.txt"; \
	cat "$$out/images.txt"; \
	$$compose config --format json > "$$out/compose-config.json"; \
	$$compose up --no-start > "$$out/up.log" 2>&1 || { cat "$$out/up.log" >&2; echo "TEST_COMPONENTS_FAIL compose"; exit 1; }; \
	$$compose up -d launcher >> "$$out/up.log" 2>&1 || { cat "$$out/up.log" >&2; echo "TEST_COMPONENTS_FAIL launcher"; exit 1; }; \
	rc=0; \
	$$compose run --rm --no-deps -T suite /bin/sh -c ' \
	  rc=0; python -B -m pytest --noconftest -p no:cacheprovider -q -rfEs --junitxml /out/junit.xml "$$@" > /out/pytest.out 2>&1 || rc=$$?; \
	  tail -n 1 /out/pytest.out; echo "pytest exit $$rc"; \
	  python -I -B -c "import sys; sys.path.insert(0, \"scripts\"); import bench_junit_guard as g; ran, problems = g.check(\"/out/junit.xml\", \"tests.container.\", \"container\"); [print(\"problem: \" + p) for p in problems]; print(\"container guard: %d run, %d problems\" % (ran, len(problems))); sys.exit(1 if problems else 0)" || rc=1; \
	  exit $$rc' sh $(or $(TESTS),tests/container) || rc=$$?; \
	if [ "$$rc" -eq 0 ]; then echo "TEST_COMPONENTS_PASS $$head"; else tail -n 60 "$$out/pytest.out" >&2 || true; echo "TEST_COMPONENTS_FAIL $$head (exit $$rc)"; fi; \
	exit $$rc
