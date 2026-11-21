#!/usr/bin/env python3
"""The proxy in front of vLLM on a rented worker (MFL033; plan r4 App. C.2, review r4 prescriptions).

Runs inside the container: on RunPod as the entrypoint (this file travels base64 in the endpoint
env, its SHA-256 checked before it runs), on Modal as threads of the web function. Standard
library only. It listens on PORT (8000), starts vLLM on 127.0.0.1:8001 in its own process group
and forwards every request but the health checks of the platform. vLLM's answers carry
`X-MDF-Proxy: forwarded`, its own `error`. `/health` and `/ping` (RunPod's load balancer polls
`/ping`, seen on 19/11 although HEALTH_CHECK_PATH is /health) are served here and never forwarded,
with `health`: 204 until vLLM is ready, then 200 for ever, inert mode included, so the provider sees
a healthy idle worker and stops it (MFL033 c7.7).

Every 5 s it reads vLLM's /health and /metrics (5 s each) and trips on W1 (never healthy within
900 s of the start of the proxy, 1500 s with MDF_ALLOW_DOWNLOAD=1: the populate), W2 (unhealthy for
60 s after readiness), W3 (work in flight and no token for 120 s), W4 (no inference and no token for
600 s), W5 (life over 2 h on RunPod, 4 h on Modal). The loop runs from the start, through the check
and the download of the weights. A trip kills vLLM's group and the download, answers 503
`mdf_tripped` to the pending requests and leaves the proxy inert (503
`mdf_inert`, /health and /ping 200): it never exits, so the provider stops the idle worker. Each request
has an absolute term `req_max_s` (290 s RunPod, 130 s Modal) from its acceptance, enforced by a
controller every second: the socket to vLLM is closed (504 `mdf_request_timeout` when nothing was
sent yet) and the client's 2 s later. Limits: 40 forwarding handlers (503 `mdf_busy`), bodies of
30,000,000 B (413; 502 for an answer), a buffer budget of 256 MiB for chunked answers; a body
needs one Content-Length (411), a GET or HEAD without one has length 0, ambiguous framing is 400.
Hop-by-hop headers, and those named by Connection, are dropped both ways.

Weights: the pinned files under HF_HOME, the size and SHA-256 of model.safetensors checked at
every start; missing or different (with MDF_ALLOW_DOWNLOAD=0) give the inert mode with 503
`mdf_weights_missing`. MDF_FAULT injects `hang-after-first` (SIGSTOP to vLLM after the first
successful inference) or `crash-after-first` (the process exits after it).

Arguments of vLLM: those of Surya's spawn for a 24 GB card, and the MTP speculation only when the
config.json of the verified snapshot declares an MTP head, by Surya's rule (MFL033 c7.5; the pinned
revision has none, and vLLM 0.25.1 then fails with ZeroDivisionError, seen on 19/11). A config.json
absent, unreadable, not JSON or not an object gives the inert mode `mdf_config_invalid`, never a
default. MDF_LEVERS (MFL033 N5b) names levers of the start from the closed set LEVERS
(`skip-mm-profiling`: `--skip-mm-profiling` with `--gpu-memory-utilization 0.80`; c8.4,
`compile-cache-populate` and `compile-cache:<sha256>`: the cache of torch.compile on the volume, see
LEVERS, with its `mdf_compile_cache` lines); another name gives the inert mode `mdf_levers_invalid`.
`mdf_start` and `mdf_vllm_args` name the levers.

Log (stderr, one JSON object a line, with `t` wall clock and `mono` monotonic seconds; events and
times only, never a document): `mdf_start`, `mdf_gpu` (the names of the GPUs of the container, from
`nvidia-smi --query-gpu=name --format=csv,noheader` or else /proc/driver/nvidia/gpus, MFL033 c7.14:
Modal may place the function on an L4 or an A10G), `mdf_weights_check`, `mdf_download_start`, a
`mdf_download_progress` every 10 s (bytes under HF_HOME, bytes received by the container),
`mdf_download_end` (exit, bytes, seconds), `mdf_vllm_args` (the MTP choice, the config fields read,
the names of the arguments, and the runtime of vLLM: the entries of PYTHONPATH by name, the `vllm`
on the PATH, its interpreter and the proxy's Python, MFL033 c7.11), `mdf_vllm_start`, the first answer of vLLM's /health and
/v1/models, the first 200 of /v1/models (`mdf_vllm_models_200`) and of /health (`mdf_ready`),
`mdf_vllm_exit`, every `mdf_trip` with its rule, `mdf_inert`, `mdf_error`; one `mdf_request` for
every answer of the proxy but the health checks (MFL033 c7.6): `kind` `forwarded` or `error`, the
method, the status, the code of an error, and the path without its query, named only when it is one
of `LOGGED_PATHS` (else `other`); `probe-negative` passes only when no such line comes (prompt N4e
§4.1). The health checks are counted by path and status, never one line each: an
`mdf_health_checks` line with the counts at the first answer of each path and status, then at most
every 30 s when they changed (c7.7, prompt N4f §3.2).
"""
import hashlib
import http.client
import http.server
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import stat
import subprocess
import sys
import tarfile
import tempfile
import threading
import time

MODEL = "datalab-to/surya-ocr-2.5"
REVISION = "00c843e9cae45611dca0a6dc2c9bbbf2dadaaed3"
WEIGHT = ("model.safetensors", 1378772208, "6940baca2d16a20b15b7302cac7253246c78c95469920fcdb064963f48b5cb17")
FILES = ("config.json", "generation_config.json", "preprocessor_config.json", "processor_config.json",
         "tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "video_preprocessor_config.json")
BODY_LIMIT = 30_000_000
CHUNK = 65536
BUFFER_BUDGET = 256 << 20
MAX_HANDLERS = 40
HOP = {"connection", "keep-alive", "transfer-encoding", "te", "trailer", "trailers", "upgrade", "content-length",
       "host"}
INFERENCE = ("/v1/chat/completions", "/v1/completions")
LOGGED_PATHS = ("/v1/models", "/version", "/metrics") + INFERENCE      # the paths an `mdf_request` names
# The health checks of the platform, served here and never forwarded (MFL033 c7.7). RunPod's load
# balancer polls `/ping` by default, or the HEALTH_CHECK_PATH of the env (/health here); on 19/11 it
# polled /ping although the env says /health (docs.runpod.io/serverless/load-balancing/overview,
# read on 19/11/2026: 200 healthy, 204 initializing, any other code unhealthy).
HEALTH_PATHS = ("/health", "/ping")
HEALTH_SUMMARY_S = 30           # a summary of the checks at most this often, when they changed
DEFAULTS = {"runpod": {"req_max_s": 290, "w5_s": 7200}, "modal": {"req_max_s": 130, "w5_s": 14400}}
RULES = {"w1_s": 900, "w2_s": 60, "w3_s": 120, "w4_s": 600, "tick_s": 5}
POPULATE_W1_S = 1500            # W1 of the populate (MDF_ALLOW_DOWNLOAD=1), from the start (MFL033 c7.4)
DOWNLOAD_TICK_S = 10
# Surya's MTP detection (surya/inference/backends/vllm.py `_checkpoint_has_mtp_head`, settings
# VLLM_MTP_TOKENS): the keys read in the config and in its `text_config`, the speculative tokens.
MTP_KEYS = ("num_nextn_predict_layers", "mtp_num_hidden_layers")
MTP_TOKENS = 2
# The levers of the start (MFL033 N5b, n5/leve-avvio.md), a closed set named in MDF_LEVERS, comma-separated.
# `skip-mm-profiling` (lever 7): vLLM's --skip-mm-profiling, the encoder's peak no longer profiled with an
# image at max_pixels, and --gpu-memory-utilization 0.80 instead of 0.85 to leave that peak a margin (the
# cache KV of 17.9 GiB holds 1.36 M tokens, 32 x 24,576 are needed at most).
# `compile-cache-populate` and `compile-cache:<sha256>` (lever 1, c8.4; Davide, 21/11 10:36: an exception
# to "only the weights on the volumes", with three conditions: our own populate writes it, its SHA-256 is
# checked before every use and a cache whose SHA differs is not used, nothing of the documents is in it).
# The populate packs VLLM_CACHE_ROOT/torch_compile_cache into CACHE_DIR on the volume once vLLM is ready,
# and refuses every inference meanwhile (503 `mdf_populating`): no document reaches the container. A use
# finds `torch-compile-<sha256>.tar`, checks its SHA-256 and its GPU before vLLM starts, and extracts it
# with the `data` filter of tarfile; anything else, and vLLM compiles as without the lever.
LEVERS = ("skip-mm-profiling", "compile-cache", "compile-cache-populate")
VALUED = ("compile-cache",)                      # these take `:<sha256>`, the others no value
GPU_MEMORY = "0.85"
GPU_MEMORY_SKIP_MM = "0.80"
CACHE_DIR = "mdf-compile-cache"                  # under the root of the volume, the parent of HF_HOME
CACHE_SUB = "torch_compile_cache"                # the part of VLLM_CACHE_ROOT archived
CACHE_MANIFEST = "mdf-compile-cache.json"        # in the archive: the GPU, the levers, the counts
CACHE_MAX_BYTES = 4 << 30
# A use reads an archive within the limits of its populate (MFL033 S-NB4): the archive at most the bytes of
# its files plus the headers of tar, the manifest small, the members regular files or directories under
# CACHE_SUB, their bytes at most CACHE_MAX_BYTES; extracted into a temporary directory, published whole.
CACHE_MAX_ARCHIVE = CACHE_MAX_BYTES + (256 << 20)
CACHE_MAX_MANIFEST = 64 << 10
CACHE_MAX_MEMBERS = 200_000
CACHE_WRITTEN = "/tmp/mdf-compile-cache-written"  # the path of the archive, for the commit of Modal's Volume
SHA256_HEX = set("0123456789abcdef")


class LeversInvalid(Exception):
    """MDF_LEVERS names a lever that is not in LEVERS, or gives one a value it does not take."""


def levers(text):
    """The levers named in `text` (MDF_LEVERS), in the order of LEVERS (the values of one lever sorted);
    LeversInvalid on another name, a missing or wrong value, or populate and use together."""
    items = sorted(set(x.strip() for x in (text or "").split(",") if x.strip()))
    bad = []
    for item in items:
        name, _, value = item.partition(":")
        if name not in LEVERS:
            bad.append(item)
        elif name in VALUED and not (len(value) == 64 and set(value) <= SHA256_HEX):
            bad.append(item)
        elif name not in VALUED and value:
            bad.append(item)
    if bad:
        raise LeversInvalid("unknown levers: %s" % ", ".join(bad))
    names = {x.partition(":")[0] for x in items}
    if {"compile-cache", "compile-cache-populate"} <= names:
        raise LeversInvalid("unknown levers: compile-cache with compile-cache-populate")
    return tuple(x for n in LEVERS for x in items if x.partition(":")[0] == n)


def cache_shas(lev):
    """The SHA-256 of the archives the lever `compile-cache` names."""
    return [x.partition(":")[2] for x in lev or () if x.partition(":")[0] == "compile-cache"]


def config(env=None):
    env = os.environ if env is None else env
    provider = env.get("MDF_PROVIDER", "runpod")
    allow = env.get("MDF_ALLOW_DOWNLOAD", "0") == "1"
    cfg = dict(RULES, **DEFAULTS.get(provider, DEFAULTS["runpod"]))
    if allow:
        cfg["w1_s"] = POPULATE_W1_S
    for key in list(cfg):
        if env.get("MDF_" + key.upper()):
            cfg[key] = float(env["MDF_" + key.upper()])
    cfg.update(provider=provider, port=int(env.get("PORT", "8000")), upstream=int(env.get("MDF_UPSTREAM_PORT", "8001")),
               fault=env.get("MDF_FAULT", ""), allow_download=allow, hf_home=env.get("HF_HOME", "/runpod-volume/hf"))
    try:                                                  # never an exit at the start: `prepare` goes inert instead
        cfg.update(levers=levers(env.get("MDF_LEVERS", "")), levers_error=None)
    except LeversInvalid as exc:
        cfg.update(levers=(), levers_error=str(exc))
    return cfg


def vllm_argv(cfg, mtp):
    """Surya's spawn for 24 GB (8192 batched tokens, 32 sequences); `--speculative-config` only with
    an MTP head (`mtp_head`); the levers of `cfg` (LEVERS)."""
    lev = cfg.get("levers") or ()
    memory = GPU_MEMORY_SKIP_MM if "skip-mm-profiling" in lev else GPU_MEMORY
    argv = ["vllm", "serve", MODEL, "--revision", REVISION, "--host", "127.0.0.1", "--port", str(cfg["upstream"]),
            "--served-model-name", MODEL, "--no-enable-log-requests", "--no-enforce-eager", "--max-num-seqs", "32",
            "--max-num-batched-tokens", "8192", "--max-model-len", "24576", "--gpu-memory-utilization", memory,
            "--dtype", "bfloat16", "--enable-prefix-caching", "--mm-processor-kwargs",
            json.dumps({"min_pixels": 3136, "max_pixels": 6291456})]
    if "skip-mm-profiling" in lev:
        argv.append("--skip-mm-profiling")
    if mtp:
        argv += ["--speculative-config", json.dumps({"method": "mtp", "num_speculative_tokens": MTP_TOKENS})]
    return argv


class ConfigInvalid(Exception):
    """The config.json of the pinned snapshot is absent, unreadable, not JSON or not an object."""


def mtp_head(hf_home):
    """Surya's rule: an MTP head when `num_nextn_predict_layers` or `mtp_num_hidden_layers` is an
    integer (not a boolean) over 0, in the config or in its `text_config`. Returns (head, the fields
    found, by `config.KEY` or `text_config.KEY`); ConfigInvalid with the reason otherwise."""
    path = snapshot(hf_home) / "config.json"
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise ConfigInvalid("absent") from None
    except (OSError, UnicodeDecodeError):
        raise ConfigInvalid("unreadable") from None
    try:
        cfg = json.loads(text)
    except ValueError:
        raise ConfigInvalid("not_json") from None
    if not isinstance(cfg, dict):
        raise ConfigInvalid("not_an_object")
    sub = cfg.get("text_config") if isinstance(cfg.get("text_config"), dict) else {}
    head, fields = False, {}
    for where, c in (("config", cfg), ("text_config", sub)):
        for key in MTP_KEYS:
            if key not in c:
                continue
            v = c[key]
            fields[where + "." + key] = v if isinstance(v, (bool, int, float)) or v is None else type(v).__name__
            if isinstance(v, int) and not isinstance(v, bool) and v > 0:
                head = True
    return head, fields


_LOG_LOCK = threading.Lock()


def log(event, **fields):
    line = dict(event=event, t=round(time.time(), 3), mono=round(time.monotonic(), 3), **fields)
    text = json.dumps(line, sort_keys=True) + "\n"
    with _LOG_LOCK:                                   # the handlers log too: one whole line at a time
        sys.stderr.write(text)
        sys.stderr.flush()


def request_path(path):
    """The path an `mdf_request` line names: without the query, and only a known one."""
    p = (path or "").split("?", 1)[0]
    return p if p in LOGGED_PATHS else "other"


def health_path(path):
    """The health path of the platform that `path` asks, without the query; None for any other."""
    p = (path or "").split("?", 1)[0]
    return p if p in HEALTH_PATHS else None


# --- the runtime of vLLM ----------------------------------------------------------------------------

RUNTIME_ENTRIES = 32            # at most this many entries of PYTHONPATH in the log line
RUNTIME_TEXT = 200              # and at most this many characters for each, for the `#!` line too


def runtime(exe, env):
    """What vLLM runs with (MFL033 c7.11, prompt N4i §3.2), for the `mdf_vllm_args` line: on Modal the
    runtime mounts its client's dependencies server side, and a PYTHONPATH inherited by `vllm serve`
    could hand vLLM other versions of its packages. `pythonpath`: the entries of PYTHONPATH in `env`
    by name, never what they hold (None when it is not set); `vllm_bin`: `exe` found on the PATH of
    `env`; `interpreter`: the `#!` line of `vllm_bin` and `interpreter_real` the real path of its
    interpreter (`/usr/bin/env NAME` resolved on that PATH), or `interpreter_error` (`not_found`,
    `unreadable`, `no_shebang`); `proxy_python`: the Python of this proxy. Never raises."""
    out = {"pythonpath": None, "vllm_bin": None, "interpreter": None, "interpreter_real": None,
           "interpreter_error": None, "proxy_python": sys.executable}
    if "PYTHONPATH" in env:
        out["pythonpath"] = [e[:RUNTIME_TEXT] for e in env["PYTHONPATH"].split(os.pathsep)[:RUNTIME_ENTRIES]]
    path = env.get("PATH", os.defpath)
    try:
        found = shutil.which(exe, path=path)
    except (OSError, ValueError):
        found = None
    if not found:
        out["interpreter_error"] = "not_found"
        return out
    out["vllm_bin"] = found[:RUNTIME_TEXT]
    try:
        with open(found, "rb") as fh:
            head = fh.read(RUNTIME_TEXT + 2)
    except OSError:
        out["interpreter_error"] = "unreadable"
        return out
    if not head.startswith(b"#!"):
        out["interpreter_error"] = "no_shebang"
        return out
    line = head[2:].split(b"\n", 1)[0].decode("utf-8", "replace").strip()
    out["interpreter"] = line[:RUNTIME_TEXT]
    words = line.split()
    if words:
        target = words[0]
        if os.path.basename(target) == "env" and len(words) > 1:
            try:
                target = shutil.which(words[1], path=path) or words[1]
            except (OSError, ValueError):
                target = words[1]
        out["interpreter_real"] = os.path.realpath(target)[:RUNTIME_TEXT]
    return out


# --- the GPU of the container (c7.14) ----------------------------------------------------------------

GPU_QUERY = ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"]
GPU_QUERY_S = 10                # nvidia-smi may wait for the driver: never longer than this
GPU_PROC = "/proc/driver/nvidia/gpus"
GPU_NAMES = 8


def gpu_names(run=subprocess.run, proc=GPU_PROC):
    """The GPUs of the container for the `mdf_gpu` line (MFL033 c7.14, prompt N4m §3.3): the names given
    by `nvidia-smi --query-gpu=name --format=csv,noheader`, else the `Model:` lines of the driver's
    /proc/driver/nvidia/gpus/*/information (standard library). {"names", "source" (`nvidia-smi`, `proc`
    or None), "error" (why nvidia-smi gave none)}. Never raises: the start goes on without a name."""
    out = {"names": [], "source": None, "error": None}
    try:
        r = run(GPU_QUERY, capture_output=True, text=True, timeout=GPU_QUERY_S, stdin=subprocess.DEVNULL)
        names = [n.strip()[:RUNTIME_TEXT] for n in (r.stdout or "").splitlines() if n.strip()]
        if r.returncode == 0 and names:
            out.update(names=names[:GPU_NAMES], source="nvidia-smi")
            return out
        out["error"] = "nvidia-smi: exit %s" % r.returncode
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        out["error"] = "nvidia-smi: " + type(exc).__name__
    names = []
    try:
        for info in sorted(Path(proc).glob("*/information")):
            for line in info.read_text(encoding="utf-8", errors="replace").splitlines():
                if line.startswith("Model:"):
                    names.append(line.split(":", 1)[1].strip()[:RUNTIME_TEXT])
    except OSError:
        pass
    if names:
        out.update(names=names[:GPU_NAMES], source="proc")
    return out


# --- the cache of torch.compile on the volume (MFL033 c8.4) ------------------------------------------

def cache_root(cfg):
    return Path(cfg["hf_home"]).parent / CACHE_DIR


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def compile_root(env=None):
    env = os.environ if env is None else env
    return Path(env.get("VLLM_CACHE_ROOT") or "/tmp/vllm")


def populate_cache(cfg, env=None, written=CACHE_WRITTEN, clock=time.monotonic):
    """`compile-cache-populate`, once vLLM is ready: VLLM_CACHE_ROOT/torch_compile_cache (regular files and
    directories only) and a manifest into a tar on the volume, written under a temporary name, then
    renamed `torch-compile-<sha256>.tar`; its path goes to `written` (Modal commits its Volume then).
    One `mdf_compile_cache` line, `phase` populate; never raises. Returns the SHA-256 or None."""
    t0, tmp = clock(), None
    try:
        src = compile_root(env) / CACHE_SUB
        files = [x for x in sorted(src.rglob("*")) if x.is_file() and not x.is_symlink()] if src.is_dir() else []
        size = sum(x.stat().st_size for x in files)
        if not files:
            raise ValueError("no compiled graph in %s" % CACHE_SUB)
        if size > CACHE_MAX_BYTES:
            raise ValueError("%d bytes over %d" % (size, CACHE_MAX_BYTES))
        root = cache_root(cfg)
        root.mkdir(parents=True, exist_ok=True)
        tmp = root / (".torch-compile-%d.tmp" % os.getpid())
        manifest = json.dumps({"gpu": cfg.get("gpu_names"), "levers": [x for x in cfg.get("levers") or ()
                                                                        if x != "compile-cache-populate"],
                               "files": len(files), "bytes": size}, sort_keys=True).encode("utf-8")

        def keep(info):
            return info if info.isfile() or info.isdir() else None
        with tarfile.open(tmp, "w") as tar:
            info = tarfile.TarInfo(CACHE_MANIFEST)
            info.size, info.mtime, info.mode = len(manifest), int(time.time()), 0o644
            tar.addfile(info, __import__("io").BytesIO(manifest))
            tar.add(str(src), arcname=CACHE_SUB, recursive=True, filter=keep)
        sha = sha256_file(tmp)
        final = root / ("torch-compile-%s.tar" % sha)
        os.replace(tmp, final)
        tmp = None
        Path(written).write_text(str(final))
        log("mdf_compile_cache", phase="populate", written=True, sha256=sha, bytes=final.stat().st_size,
            files=len(files), source_bytes=size, gpu=cfg.get("gpu_names"), seconds=round(clock() - t0, 1))
        return sha
    except Exception as exc:                              # the populate failed: said, nothing left behind
        log("mdf_compile_cache", phase="populate", written=False, error=type(exc).__name__, detail=str(exc)[:200],
            seconds=round(clock() - t0, 1))
        if tmp is not None:
            try:
                tmp.unlink()
            except OSError:
                pass
        return None


class CacheRefused(Exception):
    """An archive of the cache outside the limits of its populate: its `reason` goes to the log."""

    def __init__(self, reason):
        super().__init__(reason)
        self.reason = reason


def cache_members(tar):
    """(manifest, members) of an archive within the limits of its populate (S-NB4), else CacheRefused: one
    regular manifest of at most CACHE_MAX_MANIFEST bytes; every other member a regular file or a directory
    (no link, no device), named under CACHE_SUB without an absolute part, `..` or NUL; at most
    CACHE_MAX_MEMBERS, their bytes at most CACHE_MAX_BYTES."""
    members, manifest, size = [], None, 0
    for m in tar:
        if len(members) >= CACHE_MAX_MEMBERS:
            raise CacheRefused("too_many_members")
        if m.name == CACHE_MANIFEST:
            if manifest is not None or not m.isfile() or m.size > CACHE_MAX_MANIFEST:
                raise CacheRefused("manifest")
            manifest = m
            continue
        parts = m.name.split("/")
        if not (m.isfile() or m.isdir()):
            raise CacheRefused("member_type")
        if (parts[0] != CACHE_SUB or "\0" in m.name or m.name.startswith("/")
                or any(p in ("", ".", "..") for p in parts[1:])):
            raise CacheRefused("outside_" + CACHE_SUB)
        size += m.size if m.isfile() else 0
        if size > CACHE_MAX_BYTES:
            raise CacheRefused("over_the_limit")
        members.append(m)
    if manifest is None:
        raise CacheRefused("manifest")
    try:
        data = json.loads(tar.extractfile(manifest).read(CACHE_MAX_MANIFEST + 1).decode("utf-8"))
    except (ValueError, UnicodeDecodeError, AttributeError):
        raise CacheRefused("manifest") from None
    if not isinstance(data, dict):
        raise CacheRefused("manifest")
    return data, members


def use_cache(cfg, env=None, clock=time.monotonic):
    """`compile-cache:<sha256>`, before vLLM starts: the first archive named whose SHA-256 is right and whose
    manifest names the GPU of this container, extracted into VLLM_CACHE_ROOT with the limits of its populate
    (S-NB4): the archive a regular file of at most CACHE_MAX_ARCHIVE bytes before its SHA-256 is computed,
    its members checked by `cache_members` before anything is extracted, then extracted with the `data`
    filter of tarfile into a temporary directory next to the destination, which becomes CACHE_SUB only
    whole (a CACHE_SUB already there is not replaced). One `mdf_compile_cache` line, `phase` use, with
    `used` and for each archive not used the reason; never raises, never writes the volume."""
    shas = cache_shas(cfg.get("levers"))
    if not shas:
        return None
    t0, tried, dest = clock(), [], compile_root(env)
    for sha in shas:
        path = cache_root(cfg) / ("torch-compile-%s.tar" % sha)
        why = {"sha256": sha[:12]}
        tmp = None
        try:
            try:
                info = os.lstat(path)
            except FileNotFoundError:
                tried.append(dict(why, reason="absent"))
                continue
            if not stat.S_ISREG(info.st_mode):
                tried.append(dict(why, reason="not_a_file"))
                continue
            if info.st_size > CACHE_MAX_ARCHIVE:
                tried.append(dict(why, reason="over_the_limit", bytes=info.st_size))
                continue
            if sha256_file(path) != sha:
                tried.append(dict(why, reason="sha256_differs"))
                continue
            with tarfile.open(path) as tar:
                try:
                    manifest, members = cache_members(tar)
                except CacheRefused as exc:
                    tried.append(dict(why, reason=exc.reason))
                    continue
                if manifest.get("gpu") != cfg.get("gpu_names"):
                    tried.append(dict(why, reason="another_gpu", gpu=manifest.get("gpu")))
                    continue
                if (dest / CACHE_SUB).exists():
                    tried.append(dict(why, reason="present"))
                    continue
                dest.mkdir(parents=True, exist_ok=True)
                tmp = Path(tempfile.mkdtemp(prefix=".mdf-compile-cache-", dir=dest))
                tar.extractall(tmp, members=members, filter="data")
            os.replace(tmp / CACHE_SUB, dest / CACHE_SUB)
            shutil.rmtree(tmp, ignore_errors=True)
            tmp = None
            log("mdf_compile_cache", phase="use", used=True, sha256=sha, bytes=path.stat().st_size,
                files=manifest.get("files"), gpu=manifest.get("gpu"), tried=tried, seconds=round(clock() - t0, 1))
            return sha
        except Exception as exc:                          # an archive that cannot be read is not used
            tried.append(dict(why, reason=type(exc).__name__, detail=str(exc)[:120]))
        finally:
            if tmp is not None:
                shutil.rmtree(tmp, ignore_errors=True)
    log("mdf_compile_cache", phase="use", used=False, tried=tried, seconds=round(clock() - t0, 1))
    return None


# --- weights ----------------------------------------------------------------------------------------

def snapshot(hf_home):
    return Path(hf_home) / "hub" / ("models--" + MODEL.replace("/", "--")) / "snapshots" / REVISION


def weights_ok(hf_home):
    snap = snapshot(hf_home)
    if not all((snap / name).is_file() for name in FILES):
        return False
    path = snap / WEIGHT[0]
    try:
        if path.stat().st_size != WEIGHT[1]:
            return False
        digest = hashlib.sha256()
        with open(path, "rb") as fh:
            for block in iter(lambda: fh.read(1 << 22), b""):
                digest.update(block)
    except OSError:
        return False
    return digest.hexdigest() == WEIGHT[2]


def check_weights(wd, hf_home, phase):
    t0 = time.monotonic()
    ok = weights_ok(hf_home)
    log("mdf_weights_check", phase=phase, ok=ok, seconds=round(time.monotonic() - t0, 1), since_start=wd.since_start())
    return ok


def disk_bytes(root):
    """The bytes of the files under `root` (the partial downloads and the cache included)."""
    total = 0
    for parent, _, names in os.walk(root):
        for name in names:
            try:
                total += os.lstat(os.path.join(parent, name)).st_size
            except OSError:
                pass
    return total


def rx_bytes(path="/proc/net/dev"):
    """The bytes received by the container on every interface but lo; None when unreadable."""
    try:
        lines = Path(path).read_text().splitlines()[2:]
        return sum(int(data.split()[0]) for name, _, data in (x.partition(":") for x in lines)
                   if name.strip() != "lo" and data.split())
    except (OSError, ValueError, IndexError):
        return None


class Standalone:
    """What `download` needs of a watchdog, for a download outside the proxy (the Modal populate)."""

    def __init__(self, clock=time.monotonic):
        self.lock, self.inert, self.download_proc = threading.Lock(), None, None
        self.clock, self.t_start = clock, clock()

    def since_start(self):
        return round(self.clock() - self.t_start, 1)


def download(wd, hf_home, limit_s, tick_s=None, argv=None):
    """The pinned files into HF_HOME, a progress line every `tick_s`; the exit status (negative when
    killed: a trip kills the download, `limit_s` too; None when the proxy is already inert). `wd` is
    the watchdog, or a `Standalone` outside the proxy."""
    tick_s = DOWNLOAD_TICK_S if tick_s is None else tick_s
    code = ("from huggingface_hub import snapshot_download as s; s(repo_id=%r, revision=%r, allow_patterns=%r)"
            % (MODEL, REVISION, list(FILES) + [WEIGHT[0]]))
    env = dict(os.environ, HF_HOME=hf_home, HF_HUB_OFFLINE="0", HF_HUB_DISABLE_PROGRESS_BARS="1")
    t0, d0, r0 = time.monotonic(), disk_bytes(hf_home), rx_bytes()
    with wd.lock:
        if wd.inert is not None:
            return None
        proc = subprocess.Popen(argv or [sys.executable, "-c", code], env=env, stdin=subprocess.DEVNULL,
                                start_new_session=True)
        wd.download_proc = proc
    log("mdf_download_start", pid=proc.pid, disk=d0, rx=r0, since_start=wd.since_start())
    timeout = False
    try:
        while True:
            try:
                proc.wait(timeout=tick_s)
                break
            except subprocess.TimeoutExpired:
                pass
            log("mdf_download_progress", disk=disk_bytes(hf_home), rx=rx_bytes(), seconds=round(time.monotonic() - t0, 1),
                since_start=wd.since_start())
            if time.monotonic() - t0 >= limit_s:
                timeout = True
                kill_group(proc)
                proc.wait()
                break
    finally:
        with wd.lock:
            wd.download_proc = None
    d1, r1 = disk_bytes(hf_home), rx_bytes()
    log("mdf_download_end", exit=proc.returncode, timeout=timeout, seconds=round(time.monotonic() - t0, 1),
        bytes=d1 - d0, disk=d1, rx=r1, rx_bytes=None if r0 is None or r1 is None else r1 - r0,
        since_start=wd.since_start())
    return proc.returncode


def kill_group(proc):
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except OSError:
        pass


# --- vLLM metrics and rules -------------------------------------------------------------------------

def metrics(text):
    """{running, waiting, tokens} summed over the label sets of vLLM's Prometheus text."""
    names = {"vllm:num_requests_running": "running", "vllm:num_requests_waiting": "waiting",
             "vllm:generation_tokens_total": "tokens", "vllm:generation_tokens": "tokens"}
    out = {}
    for line in text.splitlines():
        if not line or line[0] == "#":
            continue
        name = line.split("{", 1)[0].split(" ", 1)[0]
        if name in names:
            try:
                value = float(line.rsplit(" ", 1)[1])
            except (IndexError, ValueError):
                continue
            out[names[name]] = out.get(names[name], 0.0) + value
    return out


class Rules:
    """W1-W5 on a clock; `step` returns the rule that trips, or None."""

    def __init__(self, cfg, t0):
        self.cfg, self.t0 = cfg, t0
        self.ready_at = self.bad_since = self.tokens = None
        self.progress_at = self.activity_at = t0

    def step(self, now, health_ok, m, inflight):
        c = self.cfg
        if now - self.t0 >= c["w5_s"]:
            return "W5"
        if self.ready_at is None:
            if not health_ok:
                return "W1" if now - self.t0 >= c["w1_s"] else None
            self.ready_at = self.progress_at = self.activity_at = now
        if health_ok:
            self.bad_since = None
        else:
            self.bad_since = now if self.bad_since is None else self.bad_since
            if now - self.bad_since >= c["w2_s"]:
                return "W2"
        m = m or {}
        if m.get("tokens") is not None and m["tokens"] != self.tokens:
            if self.tokens is not None:
                self.activity_at = now
            self.tokens, self.progress_at = m["tokens"], now
        busy = inflight > 0 or m.get("running", 0) > 0 or m.get("waiting", 0) > 0
        if busy:
            self.activity_at = now
            if now - self.progress_at >= c["w3_s"]:
                return "W3"
        else:
            self.progress_at = now
            if now - self.activity_at >= c["w4_s"]:
                return "W4"
        return None


# --- the proxy --------------------------------------------------------------------------------------

class Pending:
    def __init__(self, deadline):
        self.deadline, self.reason, self.started = deadline, None, False
        self.upstream = self.client = None
        self.client_closed = False


class Watchdog:
    def __init__(self, cfg, clock=time.monotonic, process=None, on_exit=None):
        self.cfg, self.clock = cfg, clock
        self.lock = threading.Lock()
        self.pending, self.handlers, self.buffered = set(), 0, 0
        self.inflight = 0                      # forwarded inference requests
        self.ready = False
        self.inert = None                      # the code of the inert mode
        self.first_success = False
        self.process = process
        self.download_proc = None
        self.on_exit = on_exit or (lambda code: os._exit(code))
        self.t_start = clock()
        self.rules = Rules(cfg, self.t_start)       # W1 counts from the start of the proxy (r2:181)
        self.seen = {"health": False, "models": False, "models_200": False, "exit": False}
        self.server = None
        self.checks = {}                       # the health checks answered: {path: {status: count}}
        self.checks_logged, self.checks_at = {}, self.t_start
        self.populating = "compile-cache-populate" in (cfg.get("levers") or ())   # c8.4: no inference

    def since_start(self):
        return round(self.clock() - self.t_start, 1)

    # the health checks of the platform (c7.7): counted, summarized, never one line each
    def count_check(self, path, status):
        with self.lock:
            by = self.checks.setdefault(path, {})
            first = status not in by
            by[status] = by.get(status, 0) + 1
        if first:                              # the first answer of each path and status: when it began
            self.summarize_checks(first=[path, status])

    def summarize_checks(self, first=None):
        """An `mdf_health_checks` line with the counts so far: at the first answer of a path and status,
        else at most every HEALTH_SUMMARY_S and only when they changed (the inert mode included)."""
        now = self.clock()
        with self.lock:
            if first is None and (now - self.checks_at < HEALTH_SUMMARY_S or self.checks == self.checks_logged):
                return
            counts = {p: {str(s): n for s, n in sorted(by.items())} for p, by in sorted(self.checks.items())}
            self.checks_logged = {p: dict(by) for p, by in self.checks.items()}
            self.checks_at = now
            ready, inert = self.ready, self.inert
        log("mdf_health_checks", counts=counts, first=first, ready=ready, inert=inert, since_start=self.since_start())

    # mode
    def trip(self, rule):
        with self.lock:
            if self.inert is not None:
                return
            self.inert = "mdf_inert"
            pending = list(self.pending)
            process, download = self.process, self.download_proc
        if process is not None:
            process.signal(signal.SIGKILL)
        if download is not None:
            kill_group(download)
        log("mdf_trip", rule=rule, since_start=self.since_start(), pending=len(pending), download=download is not None)
        for p in pending:
            p.reason = p.reason or "tripped"
            self._close_upstream(p)
            p.deadline = min(p.deadline, self.clock())

    def go_inert(self, code):
        with self.lock:
            self.inert, self.ready = code, True
        log("mdf_inert", code=code, since_start=self.since_start())

    # controller, every second: the absolute term of each request, the summary of the health checks
    def control(self):
        now = self.clock()
        with self.lock:
            pending = list(self.pending)
        for p in pending:
            if now >= p.deadline:
                p.reason = p.reason or "timeout"
                self._close_upstream(p)
            if now >= p.deadline + 2 and not p.client_closed:
                p.client_closed = True
                try:
                    p.client.shutdown(socket.SHUT_RDWR)
                except (OSError, AttributeError):
                    pass
        self.summarize_checks()

    @staticmethod
    def _close_upstream(p):
        conn = p.upstream
        if conn is not None and conn.sock is not None:
            try:
                conn.sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass

    # supervisor: one tick of the 5 s loop
    def probe(self, path):
        conn = http.client.HTTPConnection("127.0.0.1", self.cfg["upstream"], timeout=5)
        try:
            conn.request("GET", path)
            r = conn.getresponse()
            return r.status, r.read(4 << 20).decode("utf-8", "replace")
        except (OSError, http.client.HTTPException):
            return None, ""
        finally:
            conn.close()

    def tick(self):
        if self.inert is not None:
            return None
        status, _ = self.probe("/health")
        self.first_answers(status)
        m = None
        if status == 200:
            ms, text = self.probe("/metrics")
            m = metrics(text) if ms == 200 else None
        if status == 200 and not self.ready:
            self.ready = True
            log("mdf_ready", since_start=self.since_start())
            if self.populating:                    # c8.4: the compiled graphs, before any inference
                threading.Thread(target=populate_cache, args=(self.cfg,), name="mdf-cache", daemon=True).start()
        rule = self.rules.step(self.clock(), status == 200, m, self.inflight)
        if rule:
            self.trip(rule)
        return rule

    def first_answers(self, status):
        """Log lines only: the first answer of vLLM's /health and /v1/models, the first 200 of
        /v1/models (that of /health is `mdf_ready`), and vLLM's exit."""
        if status is not None and not self.seen["health"]:
            self.seen["health"] = True
            log("mdf_vllm_first_health", status=status, since_start=self.since_start())
        if self.seen["health"] and not self.seen["models_200"]:
            ms, text = self.probe("/v1/models")
            if ms is not None:
                try:
                    ids = [x.get("id") for x in json.loads(text).get("data", [])]
                except (ValueError, AttributeError):
                    ids = None
                if not self.seen["models"]:
                    self.seen["models"] = True
                    log("mdf_vllm_first_models", status=ms, models=ids, since_start=self.since_start())
                if ms == 200:
                    self.seen["models_200"] = True
                    log("mdf_vllm_models_200", models=ids, since_start=self.since_start())
        poll = getattr(self.process, "poll", None)
        code = poll() if poll is not None and not self.seen["exit"] else None
        if code is not None:
            self.seen["exit"] = True
            log("mdf_vllm_exit", code=code, since_start=self.since_start())

    def refusal(self, path):
        """The code of a request refused before vLLM (c8.4): every inference while the cache is populated."""
        if self.populating and request_path(path) in INFERENCE:
            return "mdf_populating"
        return None

    def after_success(self):
        fault = self.cfg.get("fault")
        with self.lock:
            first = not self.first_success
            self.first_success = True
        if not first or not fault:
            return
        log("mdf_fault", fault=fault)
        if fault == "hang-after-first" and self.process is not None:
            self.process.signal(signal.SIGSTOP)
        elif fault == "crash-after-first":
            self.on_exit(1)

    # server
    def serve(self, port=None, host="0.0.0.0"):
        wd = self

        class Server(http.server.ThreadingHTTPServer):
            daemon_threads = True
            request_queue_size = 64

        class Handler(ProxyHandler):
            watchdog = wd

        self.server = Server((host, self.cfg["port"] if port is None else port), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        return self.server.server_address[1]

    def loops(self):
        def controller():
            while True:
                self.control()
                time.sleep(1)

        def supervisor():
            while True:
                self.tick()
                time.sleep(self.cfg["tick_s"])

        threading.Thread(target=controller, daemon=True).start()
        threading.Thread(target=supervisor, daemon=True).start()


class ProxyHandler(http.server.BaseHTTPRequestHandler):
    watchdog = None
    protocol_version = "HTTP/1.1"
    server_version = "mdf-proxy"
    sys_version = ""

    def log_message(self, *args):
        pass

    def setup(self):
        super().setup()
        self.connection.settimeout(30)

    def _send(self, status, kind, body=b"", ctype="application/json", headers=()):
        self.send_response(status)
        self.send_header("X-MDF-Proxy", kind)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        for k, v in headers:
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD" and body:
            self.wfile.write(body)
        self.close_connection = True

    def _error(self, status, code):
        log("mdf_request", kind="error", method=self.command, path=request_path(self.path), status=status, code=code)
        self._send(status, "error", json.dumps({"error": code}).encode())

    def _length(self):
        lengths = self.headers.get_all("Content-Length") or []
        chunked = self.headers.get_all("Transfer-Encoding") or []
        if (lengths and chunked) or len(lengths) > 1:
            return None, 400
        if lengths:
            text = lengths[0].strip()
            if not text.isdigit():
                return None, 400
            n = int(text)
        elif chunked:
            return None, 411
        elif self.command == "POST":
            return None, 411
        else:
            n = 0                                         # a GET or HEAD without a body
        if self.command in ("GET", "HEAD") and n:
            return None, 400
        return (None, 413) if n > BODY_LIMIT else (n, None)

    def do_GET(self):
        self._handle()

    do_POST = do_HEAD = do_PUT = do_DELETE = do_OPTIONS = do_PATCH = do_GET

    def _handle(self):
        wd = self.watchdog
        health = health_path(self.path)
        if health is not None:                    # /health and /ping: never vLLM's (c7.7)
            status = 200 if wd.ready or wd.inert is not None else 204
            wd.count_check(health, status)
            return self._send(status, "health")
        if wd.inert is not None:
            return self._error(503, wd.inert)
        refused = wd.refusal(self.path)
        if refused is not None:
            return self._error(503, refused)
        with wd.lock:
            if wd.handlers >= MAX_HANDLERS:
                busy = True
            else:
                busy, wd.handlers = False, wd.handlers + 1
        if busy:
            return self._error(503, "mdf_busy")
        try:
            self._forward(wd)
        finally:
            with wd.lock:
                wd.handlers -= 1

    def _forward(self, wd):
        n, refusal = self._length()
        if refusal:
            return self._error(refusal, {400: "mdf_framing", 411: "mdf_length_required", 413: "mdf_too_large"}[refusal])
        inference = self.command == "POST" and self.path.split("?", 1)[0] in INFERENCE
        p = Pending(wd.clock() + wd.cfg["req_max_s"])
        p.client = self.connection
        with wd.lock:
            wd.pending.add(p)
            wd.inflight += 1 if inference else 0
        try:
            self._exchange(wd, p, n, inference)
        finally:
            with wd.lock:
                wd.pending.discard(p)
                wd.inflight -= 1 if inference else 0

    def _exchange(self, wd, p, n, inference):
        drop = HOP | {h.strip().lower() for v in self.headers.get_all("Connection") or [] for h in v.split(",")}
        conn = http.client.HTTPConnection("127.0.0.1", wd.cfg["upstream"], timeout=wd.cfg["req_max_s"] + 5)
        p.upstream = conn
        try:
            conn.putrequest(self.command, self.path, skip_host=True, skip_accept_encoding=True)
            conn.putheader("Host", "127.0.0.1:%d" % wd.cfg["upstream"])
            for k, v in self.headers.items():
                if k.lower() not in drop and not k.lower().startswith("proxy-"):
                    conn.putheader(k, v)
            conn.putheader("Content-Length", str(n))
            conn.putheader("Connection", "close")
            conn.endheaders()
            left = n
            while left:
                block = self.rfile.read(min(CHUNK, left))
                if not block:
                    raise ConnectionError("client body cut short")
                conn.send(block)
                left -= len(block)
            r = conn.getresponse()
            self._answer(wd, p, r, inference)
        except (OSError, http.client.HTTPException, ValueError):
            if not p.started:
                if p.reason == "tripped":
                    self._error(503, "mdf_tripped")
                elif p.reason == "timeout":
                    self._error(504, "mdf_request_timeout")
                else:
                    self._error(502, "mdf_upstream")
            self.close_connection = True
        finally:
            conn.close()

    def _answer(self, wd, p, r, inference):
        drop = HOP | {h.strip().lower() for v in r.headers.get_all("Connection") or [] for h in v.split(",")}
        headers = [(k, v) for k, v in r.getheaders() if k.lower() not in drop and not k.lower().startswith("proxy-")
                   and k.lower() != "x-mdf-proxy"]
        length = r.getheader("Content-Length")
        if self.command == "HEAD":
            self._head(r.status, headers, int(length) if length and length.strip().isdigit() else 0)
            p.started = True
        elif length is not None and length.strip().isdigit() and r.getheader("Transfer-Encoding") is None:
            n = int(length)
            if n > BODY_LIMIT:
                return self._error(502, "mdf_answer_too_large")
            self._head(r.status, headers, n)
            p.started = True
            left = n
            while left:
                block = r.read(min(CHUNK, left))
                if not block:
                    raise ConnectionError("upstream body cut short")
                self.wfile.write(block)
                left -= len(block)
        else:
            body = bytearray()
            while True:
                block = r.read(CHUNK)
                if not block:
                    break
                with wd.lock:
                    if wd.buffered + len(block) > BUFFER_BUDGET:
                        busy = True
                    else:
                        busy, wd.buffered = False, wd.buffered + len(block)
                if busy:
                    self._release(wd, len(body))
                    return self._error(503, "mdf_busy")
                body += block
                if len(body) > BODY_LIMIT:
                    self._release(wd, len(body))
                    return self._error(502, "mdf_answer_too_large")
            try:
                self._head(r.status, headers, len(body))
                p.started = True
                if self.command != "HEAD":
                    self.wfile.write(bytes(body))
            finally:
                self._release(wd, len(body))
        self.close_connection = True
        if inference and 200 <= r.status < 300:
            wd.after_success()

    @staticmethod
    def _release(wd, n):
        with wd.lock:
            wd.buffered -= n

    def _head(self, status, headers, n):
        log("mdf_request", kind="forwarded", method=self.command, path=request_path(self.path), status=status)
        self.send_response(status)
        for k, v in headers:
            self.send_header(k, v)
        self.send_header("X-MDF-Proxy", "forwarded")
        self.send_header("Content-Length", str(n))
        self.send_header("Connection", "close")
        self.end_headers()


class VLLM:
    """vLLM in its own process group."""

    def __init__(self, argv, env=None):
        self.proc = subprocess.Popen(argv, start_new_session=True, stdin=subprocess.DEVNULL, env=env)

    def signal(self, sig):
        try:
            os.killpg(self.proc.pid, sig)
        except OSError:
            pass

    def poll(self):
        return self.proc.poll()


def start(cfg, argv=None, download_argv=None):
    """Proxy, controller and supervisor (W1 counts from here), then weights, download and vLLM;
    returns the watchdog. Nothing here exits: an error gives the inert mode."""
    wd = Watchdog(cfg)
    wd.serve()
    log("mdf_start", provider=cfg["provider"], req_max_s=cfg["req_max_s"], fault=cfg["fault"],
        allow_download=cfg["allow_download"], rules={k: cfg[k] for k in RULES}, levers=list(cfg.get("levers") or ()))
    wd.loops()
    gpu = gpu_names()
    cfg["gpu_names"] = gpu["names"]                                  # the GPU of the cache (c8.4)
    log("mdf_gpu", **gpu, since_start=wd.since_start())              # the GPU Modal or RunPod gave (c7.14)
    boot(wd, cfg, argv, download_argv)
    return wd


def boot(wd, cfg, argv=None, download_argv=None):
    try:
        prepare(wd, cfg, argv, download_argv)
    except Exception as exc:                              # an exit would be a crash for the provider
        log("mdf_error", where="start", error=type(exc).__name__, detail=str(exc)[:300])
        wd.go_inert("mdf_start_error")


def prepare(wd, cfg, argv=None, download_argv=None):
    ok = check_weights(wd, cfg["hf_home"], "start")
    if not ok and cfg["allow_download"]:
        download(wd, cfg["hf_home"], cfg["w1_s"] + 30, argv=download_argv)
        ok = check_weights(wd, cfg["hf_home"], "after_download")
    if wd.inert is not None:                              # tripped meanwhile (W1)
        return
    if not ok:
        wd.go_inert("mdf_weights_missing")
        return
    if cfg.get("levers_error"):                           # MFL033 N5b: an unknown lever, never a default
        log("mdf_vllm_args", ok=False, error=cfg["levers_error"], since_start=wd.since_start())
        wd.go_inert("mdf_levers_invalid")
        return
    use_cache(cfg)                                        # c8.4: before vLLM, or nothing
    try:
        mtp, fields = mtp_head(cfg["hf_home"])           # after the weights: the config of the verified snapshot
    except ConfigInvalid as exc:
        log("mdf_vllm_args", ok=False, error=str(exc), since_start=wd.since_start())
        wd.go_inert("mdf_config_invalid")
        return
    argv = argv or vllm_argv(cfg, mtp)
    env = dict(os.environ, HF_HOME=cfg["hf_home"], HF_HUB_OFFLINE="1")
    log("mdf_vllm_args", ok=True, mtp=mtp, mtp_tokens=MTP_TOKENS if mtp else None, config=fields,
        args=[a for a in argv if a.startswith("--")], levers=list(cfg.get("levers") or ()),
        gpu_memory_utilization=argv[argv.index("--gpu-memory-utilization") + 1]
        if "--gpu-memory-utilization" in argv else None, **runtime(argv[0], env), since_start=wd.since_start())
    with wd.lock:
        if wd.inert is None:
            wd.process = VLLM(argv, env=env)
            pid = wd.process.proc.pid
        else:
            return
    log("mdf_vllm_start", pid=pid, since_start=wd.since_start())


def start_proxy(argv=None, env=None):
    """Modal: start everything and return (the container lives on)."""
    return start(config(env), argv)


def main():
    start(config())
    while True:                                           # never exits by itself
        time.sleep(3600)


if __name__ == "__main__":
    main()
