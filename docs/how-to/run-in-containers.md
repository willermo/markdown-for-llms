# Run the product in containers

For someone who wants to convert documents on their own computer, from the browser or from a terminal. The whole product runs in containers started by `make`: nothing of it needs a Python, a virtual environment or `systemd` on the host.

**Tested**: one user on Linux x86_64 with Docker Engine 29, the CPU only, the loopback address. **Expected to work, not proven**: Linux arm64, macOS and Windows through Docker Desktop (on Windows keep the checkout in the filesystem of WSL). On arm64 the tools of the toolchain image run emulated and the local CPU server of the `marker-server` route is not available. **Designed and tested with fake servers only**: your own GPU, on this computer or on another machine ([below](#your-own-gpu-on-this-computer-or-on-another-machine)). Public, LAN and multi-user use are not supported.

## What you need

- `git`, `make` and Docker Engine 26 or newer with the `buildx` and `compose` plugins, with cgroup v2, not rootless, reachable by your user through a local socket;
- about 12 GB of memory for a conversion with Marker (or a lower `MDF_MEMORY_MIB`), and 6-7 GB of disk for the images, the toolchain image and the weights.

## Start it

```bash
git clone <repository> markdown-for-llms && cd markdown-for-llms
cp .env.example .env        # every line is a default: edit only what you need
make up
```

`make up` does, in order, skipping what is already done: the checks of the host and of `.env`; the images of the product (the first build downloads 2-3 GB); the check that the launcher will reach the Docker socket as your user; the settings (`.env` validated against the archive); the toolchain image and the 18 weight files of the Marker route (about 111 MB); then the services. It waits for the worker and prints the address, `http://127.0.0.1:23465` by default, with the state of every route. A second `make up` takes some tens of seconds.

The archive (your documents and their conversions) is `./data`, the state (the spend ledger of the rented GPUs, `config/servers.json`, the bench) is `./state`; `MDF_COMPOSE_DATA` and `MDF_COMPOSE_STATE` in `.env` move them. Two checkouts in two directories share nothing but the images.

That sharing is a limit, for whoever keeps two working copies on one Docker daemon. All of them build and use the same two tags, `mdf-app:engine` and `mdf-app:api`: a `make up` or `make rebuild` in one copy changes the images that `make restart` and `mdf` (`bin/mdf`) use in the other, so the services created again there, and the one-off container of `mdf`, run the code of the other copy. The worker compares its code with that of the launcher at every check of the routes and takes no job while they differ (`isolation_unavailable`, `stale engine image`), but that covers a difference between those two alone: it does not see a `make restart` that created both again from the tag of the other copy, nor an `mdf convert` whose container comes from it. With two copies on one daemon, run `make up` again in the copy you are about to use. `make check`, `make test` and `make test-components` have tags of their own for each invocation and are not concerned.

## The commands

| Command | What it does |
| --- | --- |
| `make up` | checks, builds, prepares and starts; safe to run again |
| `make status` | the services, the state of every route with what is missing, and a service that Docker had to start again |
| `make restart` | creates the services again, after a change of `.env`; builds and downloads nothing |
| `make down` | stops everything; the archive, the state and the weights stay |
| `make logs` | the logs (`SERVICE=worker` for one, `FOLLOW=1` to follow) |
| `make prepare` | the toolchain image and the weights alone (`PREPARE=--no-cache` builds the image again; `CPU_SERVER=1` adds the local CPU server, 1.4 GB, x86_64 only) |
| `make rebuild` | builds the images without the cache of the layers, then `make up` |
| `make install-cli` | puts `mdf` on your PATH (`BIN=DIR`, default `~/.local/bin`) |
| `make bench` | the bench of the local routes (`BENCH="..."` for other arguments; see [the bench guide](bench.md)) |
| `make clean` | removes the containers, the networks and the work volumes; never the archive, the state or the weights |
| `make clean-archive CONFIRM=<id>` | deletes the archive, only with its own identifier and the product stopped |
| `make gpu-prepare`, `make gpu-up`, `make gpu-down` | the local GPU: image and weights once, start, stop ([your own GPU](#your-own-gpu-on-this-computer-or-on-another-machine)) |
| `make server-prepare`, `make server-up`, `make server-down` | on another machine: the remote GPU server, from a checkout there |
| `make check`, `make test`, `make test-components` | the checks of a developer ([the full check](uv-environment.md#run-the-full-check)) |

## The env file and the keys

`.env` is the one file of settings and keys. `make up` gives it mode 0600 and refuses a link, something that is not a regular file, or a file of another user.

- The settings (the `MDF_` lines) are validated by the `config` step and given to the services. A path of a container (`MDF_DATA_ROOT`, `MDF_MODEL_ROOT`, `MDF_SERVERS`, `MDF_GPU_LEDGER_DIR`) is fixed by compose: leave those lines commented, a value stops `make up` with its name.
- The keys are read by name, by the worker and by `mdf` only, each time one is used. No key is in the environment of a container, in `docker inspect`, in a log or in the archive. The API never sees the file.
- **After any change run `make restart`.** A running container keeps the file it was started with. When the file is replaced on the host (the atomic save of an editor, `sed -i`) or removed, the file the container holds has no name left and is refused: the routes with a key become `broken` and `make status` says to run `make restart`. A replacement always needs `make restart`; and if the old file is still named by a backup (a hard link, or a copy made by renaming it), it can still be read until the restart.
- A file with another mode does not stop the worker: the local routes go on and the routes with a key are `broken` (`datalab_key_unsafe`, `server_key_unsafe`) until `make up` gives the file its mode.

### Settings that cannot change

Most settings of the archive and of the engine are part of the **identity** of an archive (`.env.example` lists them). Once an archive exists, `make up` and `make restart` stop when one of them differs, name the ones that changed since the last start, and touch no running service. Put the value back, or start a new archive in another folder (`MDF_COMPOSE_DATA`).

Only the quota `MDF_STORAGE_MIB` can change, with the product stopped. In this order:

1. `make down`. `mdf configure` changes the archive itself, and the archive refuses it while a service holds it open (`Stop API, worker and CLI before changing configuration`).
2. Read the identity of the archive: the value of `"config"` at the top of `identity.json`, in the folder of the archive (`./data`, or the folder of `MDF_COMPOSE_DATA`).

   ```bash
   sed -n 's/^  "config": "\(.*\)",*$/\1/p' data/identity.json
   ```

3. `mdf configure --expected-config <that value> --storage-mib N`, with the new quota in MiB. It answers with the identity of before, the new one and the settings. `mdf configure` and `mdf downgrade-storage` are the two commands of `mdf` that run without the launcher: they convert nothing. They open the archive with the settings of the last start (`settings.env`, which `make down` keeps), so `.env` may still have the old quota or already the new one.
4. Write the same `MDF_STORAGE_MIB=N` in `.env`.
5. `make up`. The archive is accepted with its documents; with another value than N it is refused as above.

Two limits. `make clean` removes the settings of the last start: `mdf configure` then refuses and says what to do (put `.env` back as it was, `make up`, `make down`, then the command again). And one change at a time: after `mdf configure`, run `make up` before another change, because until then the settings of the last start are those of the quota of before, which the archive now refuses.

## The state of the routes

`make status`, `mdf routes` and `GET /api/v1/routes` show the same thing, computed by the worker every minute (`mdf routes --refresh` asks for it now):

| State | Meaning |
| --- | --- |
| `available` | the route can convert |
| `disabled` | everything is there and a cap is 0 (for example `MDF_DATALAB_CAP_USD=0`) |
| `not_configured` | something is missing: its name, where to write it and its kind (a key, a setting, an entry of `servers.json`, a command such as `make prepare`), never a value |
| `broken` | a check fails, with its closed code and what to do: the keys file is not safe, the launcher does not answer, the weights differ |

For Datalab, the rented GPUs and the remote GPU server, `transit` says whether a document or its page images would leave the machine. It is a property of the route: the confirmation is asked for each job. The local GPU has no transit and no spend; the remote server has a transit and no spend. `mdf routes --refresh` also reads, through the worker, the state of each rented provider (its workers or containers, from the management API alone, never its service URL) and the balance of RunPod with the key of `.env` (none when that key does not read it; Modal has no balance), and asks `/v1/models` of the GPUs of the user: `live` holds those readings with their time, until the next refresh.

## The command line

```bash
make install-cli
cd ~/Documents/papers
mdf convert paper.pdf --out paper-md
mdf convert --preset local-fast letter.docx --out letter-md
mdf submit --wait paper.pdf        # through the archive, like the browser
mdf routes
```

`mdf` (the file `bin/mdf` of the checkout) runs the CLI in a one-off container, as you, with the current directory mounted at its own path: paths under the current directory work as they are, and the files written there are yours. A path outside it is not visible to the command.

- The product must be running: the engine starts its children only through the launcher. Without it `mdf` answers `{"error": "the product is not running: run make up", "code": "isolation_unavailable"}` with exit 1; nothing is started for you and nothing runs on the host. The two commands that change the archive itself, `mdf configure` and `mdf downgrade-storage`, are the exception: they need the product stopped ([Settings that cannot change](#settings-that-cannot-change)) and convert nothing.
- `mdf --env-file FILE convert ...` reads the keys from another file of yours; the settings stay those of the product.
- `--preset NAME` takes the route and the options of a conversion preset (`mdf presets list`), with only `--lang` and `--confirm-transit` beside it. A preset on a rented GPU needs the session and the ledger of the worker: `mdf convert` refuses it with `preset_needs_archive`, and the command is `mdf submit --preset NAME --wait FILE`.
- The weights of the Marker route are those of the models volume (`/models/marker`) and the local CPU server is the one of its volume (`/cpu-server`): the children of the engine read them from the volumes alone, so `--model-root` with another folder is refused with `invalid_setting`, and another root of the CPU server with `server_not_configured`.
- Exit codes: 0, 1, 2 and 130 as before; 127 when Docker or the checkout is missing. If the checkout is moved or removed, the installed `mdf` says `checkout not found: run make install-cli again`.

## Datalab and the rented GPUs

- **Datalab**: put `DATALAB_API_KEY` in `.env` and a cap above 0 in `MDF_DATALAB_CAP_USD`, then `make restart`. A document leaves the machine only for a job that chose Datalab and confirmed the transit, under that cap.
- **Rented GPUs** (`marker-server` route): enable the presets in `MDF_GPU_PRESETS`, set the caps, copy `servers.example.json` to `state/config/servers.json` and put your endpoint in it, and put the keys in `.env`. For Modal the token of the SDK is `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET`: the worker gives it to each process of the Modal CLI in a private file of its own, and no `~/.modal.toml` is read. An endpoint of kind `local` (a server on the loopback of the host) is refused from a container with `server_not_configured`.

## Your own GPU: on this computer, or on another machine

**Designed and tested with fake servers only.** Nobody has run these two on a real GPU yet: the front, the relay, the presets, the compose files and the commands are tested with a fake vLLM and synthetic TLS, never with the vLLM image or the weights. Expect to be the first.

Both run Surya on vLLM with the image and the model of the rented presets (`vllm/vllm-openai` pinned by digest, `datalab-to/surya-ocr-2.5` at its pinned revision), behind a small front of the project (`markdown_for_llms/gpu/local_front.py`) that forwards only what Surya asks: the list of the models, the version, the metrics and the completions. They cost nothing: no session, no spend ledger, no cap.

What they need: an NVIDIA card with 24 GB (the arguments of vLLM are those of a 24 GB card), a driver with CUDA 13 or newer, the NVIDIA Container Toolkit (Linux) or the CUDA of WSL2, an amd64 machine; never a Mac. The vLLM image is 10-20 GB, the weights 1.4 GB.

### The local GPU (`gpu-local`)

The `gpu` profile of `compose.yaml`, next to the product. vLLM runs **without a network**; the front, in the same container, listens on a Unix socket of the volume `mdf-run`, which the worker reaches as it reaches the local CPU server. No port, and the page images never leave this computer.

1. `make up`, then `make gpu-prepare` once (the image and the weights; it stops with the reason when the NVIDIA runtime is not configured for Docker).
2. In `.env`: `MDF_GPU_PRESETS=gpu-local` (with your other presets, comma-separated). No cap is needed.
3. In `state/config/servers.json`, the endpoint of the preset:

   ```json
   "gpu": {"kind": "gpu", "preset": "gpu-local", "socket": "/run/mdf/run/gpu/front.sock"}
   ```

4. `make restart`, then `make gpu-up`. vLLM takes some minutes to start the first time; `make status` shows `marker-server:gpu-local` as `broken` with `make gpu-up` while the front is not there, and `available` once it is. `make gpu-down` stops it.

From the command line: `mdf submit --preset local-gpu --wait FILE`, **without** `--confirm-transit`: nothing leaves the machine, and a confirmation is refused (`route_option_unsupported`). The Web UI of today asks for the confirmation of every GPU preset and sends it: from the browser the preset `local-gpu` is refused with that code until the new UI. Use the command line.

### The remote server (`gpu-remote`)

`compose.server.yaml`, on the machine of the GPU, from a checkout of this repository: vLLM and the front on an internal network with no way out, and nginx with TLS in front, on port 23467 of an address of that machine on your LAN or VPS. Every request needs the key of the server, which the front compares in constant time, except `GET` and `HEAD /health`, which answer without it with the readiness alone (`{"ready": true}` or `{"ready": false}`).

On the machine of the GPU:

1. A private key file and a certificate, for example a self-signed one for the name the product will use (`gpu.lan` here; an IP address goes in `subjectAltName` as `IP:192.168.1.20`):

   ```bash
   mkdir -p state/server/tls && chmod 700 state/server
   (umask 077; openssl rand -hex 32 > state/server/key)
   (umask 077; openssl req -x509 -newkey ec -pkeyopt ec_paramgen_curve:prime256v1 -days 365 -nodes \
      -keyout state/server/tls/server.key -out state/server/tls/server.crt \
      -subj /CN=gpu.lan -addext subjectAltName=DNS:gpu.lan)
   ```

2. In its `.env`: `MDF_COMPOSE_SERVER_ADDRESS=` the address of that machine, an IPv6 one between brackets (`[2001:db8::10]`); never the loopback nor every address, in IPv4 or in IPv6 (`127.0.0.1`, `0.0.0.0`, `[::1]`, `[::]` and their other forms are refused); `MDF_COMPOSE_SERVER_PORT=23467`, `MDF_COMPOSE_SERVER_TLS=./state/server/tls` and `MDF_COMPOSE_SERVER_KEY=./state/server/key` are the defaults.
3. `make server-prepare` once, then `make server-up`; `make server-down` stops it.

On the machine of the product:

1. Copy `server.crt` (the certificate, never `server.key`) to `state/config/gpu-server-ca.pem`, and the content of `state/server/key` to `.env` as `GPU_SERVER_KEY=...`.
2. `MDF_GPU_PRESETS=gpu-remote` in `.env`, and the endpoint in `state/config/servers.json`:

   ```json
   "remote": {"kind": "user", "preset": "gpu-remote", "url": "https://gpu.lan:23467",
              "ca_file": "/state/config/gpu-server-ca.pem"}
   ```

   `ca_file` pins the authorities the relay trusts for that server, on both of its clients, in place of those of the system; the name in the URL must be a name of the certificate, and a redirect is never followed.
3. `make restart`. The page images leave this computer for your server: the transit is declared and confirmed for every job (`mdf submit --preset remote-gpu --confirm-transit --wait FILE`, or the box of the Web UI). The client does not verify the weights on the remote machine; it checks the version of vLLM and the model at the readiness of every attempt.

## How a conversion is isolated

Every child of the engine (a tool of the toolchain image, the Marker process, the local CPU server) is a container without network, with a read-only root, no capability, your user, a memory cap without swap and only the folders of that one call. They are started by the **launcher**, the only process with the Docker socket: it accepts a closed request and builds the container itself. Three things stop a child, each independent of the next: the limits of the kernel; a small supervisor that is PID 1 of the container and ends it at a stop, at a lease that the caller no longer renews, or at its term; and the launcher, which removes from outside what outlived its lease.

Limits you should know:

- The supervisor contains a child that fails, hangs or uses too much; it is not a security boundary against arbitrary code running inside the child. For that the kernel limits and the launcher remain.
- When the launcher is stopped while it is killing a child (between the kill and the removal, which can take long when the removal is slow or fails), the reason of the kill can be lost: the conversion still ends as failed or interrupted, but its code may be less precise.
- With a long term, the last metrics of a child may be missing: an outcome by timeout never certifies a final sample of memory and CPU.
- When the launcher does not answer, the worker stops taking jobs until Docker confirms that the children of the previous conversion are gone (`make status` says so; `make restart` if it lasts).

## When something is wrong

| What you see | What to do |
| --- | --- |
| `make up` stops at "access to Docker as the launcher" | your user does not reach the Docker socket of the context in use with the group of that socket; the message names the socket and the group |
| `make up` or `make restart` stops at "settings" | the env file is refused: the message names the setting, or says that the archive was created with other settings |
| `make status`: `stale engine image: run make up` | the images were rebuilt while a worker of the old image went on: `make up` |
| `make status`: a service "started again by Docker" | the service leaves with an error and Docker starts it again (it does so for ever): the line shows the last error of its log, `make logs SERVICE=...` the rest |
| a route with a key is `broken` after an edit of `.env` | `make restart` |
| `mdf`: `the product is not running` | `make up` |

## Remove it

`make down` stops the product, the local GPU with it. `make clean` also removes the work volumes. What stays, and how to remove it by hand: the images (`docker image rm mdf-app:engine mdf-app:api mdf-app:dev`, the toolchain image `mdf-bench-toolchain:<tag>`, and after `make gpu-prepare` the vLLM image `vllm/vllm-openai`), the volumes of the weights and of the CPU server (`docker volume rm <project>_mdf-models <project>_mdf-cpu-server <project>_mdf-vllm-models`, where `<project>` is the name of the checkout directory), and the folders `./data` and `./state`.

[Index](../README.md).
