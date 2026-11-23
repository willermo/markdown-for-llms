"""The rented providers seen from the laptop (MFL033): RunPod REST v2 and Modal's CLI.

Shared by `session.py`, `runpod_endpoint.py` and `modal_tasks.py`; never imported by the engine
(the management credentials and the Modal SDK stay out of it, plan r4 §5.4). Keys come only from
`--env-file` (`envfile.read`: a regular file of the caller, mode 0600) and only go into headers;
an error never shows them. RunPod: REST v2 at https://api.runpod.io/v2 (REST v1 is forbidden) and
GraphQL only to read the balance. Modal: the SDK `modal==1.6.3`, outside the product, run as
`uv tool run --from modal==1.6.3 modal ...` with the token of `~/.modal.toml`.

Each provider has the same face for the session: `state()` (armed, busy and what was read),
`arm(fault, allow_download)`, `disarm()`, `zero()` (one reading: True when nothing runs), `url`.
"""
from datetime import datetime, timezone
from decimal import Decimal
import base64
import hashlib
import hmac
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

import httpx

from markdown_for_llms.engine import envfile

HERE = Path(__file__).resolve().parent
REST = "https://api.runpod.io/v2"
GRAPHQL = "https://api.runpod.io/graphql"
IMAGE = "vllm/vllm-openai@sha256:f0b9a0dc75a9fca3b6811e3279367b2d6a448055a000bfd13859587d74cef268"
VOLUME = "mdf-surya-weights"
VOLUME_GB = 10
POOL = "AMPERE_24"
MODAL_SDK = ["uv", "tool", "run", "--from", "modal==1.6.3", "modal"]
# The GPUs of the Modal app in their order of preference (`gpu=` of modal_app.py): Modal places the
# function on an L4, else on an A10G (MFL033 c7.14; on 20/11 no L4 was free for 13 minutes).
MODAL_GPUS = ("L4", "A10G")
# Modal's list prices in USD per second, read on 20/11/2026 (https://modal.com/pricing: "Nvidia L4",
# "Nvidia A10", a physical core, a GiB of memory; preemptible functions, no region chosen).
MODAL_GPU_USD_PER_S = {"L4": Decimal("0.000222"), "A10G": Decimal("0.000306")}
MODAL_CORE_USD_PER_S, MODAL_GIB_USD_PER_S = Decimal("0.0000131"), Decimal("0.00000222")


def modal_rate(gpus=MODAL_GPUS, cores=2, gib=16):
    """The rate of a Modal window: the dearest GPU the function may get, plus its cores and memory."""
    return max(MODAL_GPU_USD_PER_S[g] for g in gpus) + cores * MODAL_CORE_USD_PER_S + gib * MODAL_GIB_USD_PER_S


# Rates in USD per second (plan r2 C.3 line 588, r4 C.3): RunPod 24 GB with 20 GB of disk; Modal
# with 2 cores and 16 GiB at the dearest GPU of MODAL_GPUS (0.00036772 with the A10G; 0.00028372 was
# the L4 alone, until c7.13); Modal CPU with 2 cores and 4 GiB (the populate app).
RATES = {"runpod": Decimal("0.0002090948"), "modal": modal_rate(), "modal-cpu": Decimal("0.00003508")}
RUNPOD_MAX_USD_PER_H = Decimal("0.75")
# The CPU activity of Modal (MFL033 S-B3): the populate of the Volume, `modal_populate.py::populate`, a
# Function with a `timeout` and no retries; Modal stops it at that limit, so its cost is at most that timeout
# at the CPU rate, and the reserve of its activity must cover it before it starts.
MODAL_POPULATE = HERE / "modal_populate.py"


def modal_populate_timeout(path=MODAL_POPULATE):
    """The `timeout` of the Function `populate` of `path`, read from its source (never imported: it imports
    the Modal SDK), only with `retries=0`; ProviderError otherwise."""
    import ast
    tree = ast.parse(Path(path).read_text())
    consts = {t.id: n.value.value for n in tree.body if isinstance(n, ast.Assign) and isinstance(n.value, ast.Constant)
              for t in n.targets if isinstance(t, ast.Name)}

    def value(node):
        if isinstance(node, ast.Name):
            return consts.get(node.id)
        return node.value if isinstance(node, ast.Constant) else None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "populate":
            for deco in node.decorator_list:
                kw = {k.arg: value(k.value) for k in getattr(deco, "keywords", [])}
                if type(kw.get("timeout")) is int and kw["timeout"] > 0 and kw.get("retries") == 0:
                    return kw["timeout"]
    raise ProviderError("%s: no Function populate with a timeout and retries=0" % Path(path).name)
# Z, the latency from a disarm to the zero: 120 s until N4 measures it (r4 §5.4, prescription 4).
ZERO_LATENCY_S = {"runpod": 120, "modal": 120}
# The tail of the bound of a window after the stop of its container (MFL033 c7.19, prompt N4s §4.2):
# RunPod bills "until it fully stops"; on 19/11 the billing of the populate ended 25.6 s after the
# zero read by the session. The window is W = A / rate - 60 - Z - tail, its bound runs to the later
# of the zero and the stop of the container plus the tail. Modal stays as it was (N4 on Modal closed).
BILLING_TAIL_S = {"runpod": 30, "modal": 0}
ZERO_STOP_S = 600
ZERO_POLL_S = 10
KEYS = ("RUNPOD_API_KEY", "MODAL_PROXY_TOKEN_ID", "MODAL_PROXY_TOKEN_SECRET", "MODAL_PROXY_BEARER", "DATALAB_API_KEY")


class ProviderError(Exception):
    """A provider call that failed; the message never holds a key."""


class TermReached(ProviderError):
    """The term `T` of the session came during a management call of an arm (MFL033 S-B4): the outcome
    of the arm is uncertain, and the session verifies and disarms."""


def running_count(state):
    """The workers RUNNING in a reading of `state()`: the larger of the summary's `running` and the
    workers listed RUNNING (MFL033 c7.19: on RunPod only RUNNING is billed)."""
    workers = state.get("workers") or []
    listed = sum(1 for w in workers if str(w.get("status") or "").upper() == "RUNNING")
    try:
        summary = int((state.get("summary") or {}).get("running") or 0)
    except (TypeError, ValueError):
        summary = 0
    return max(listed, summary)


def read_key(env_file, name):
    try:
        found = envfile.read(env_file, [name])
    except envfile.Unsafe as exc:
        raise ProviderError(str(exc)) from None
    if not found.get(name) or any(c in found[name] for c in "\r\n\0"):
        raise ProviderError("%s missing in the env file" % name)
    return envfile.Secret(found[name])


def child_env(environ):
    """The environment of a wrapped command: no key ever travels in it."""
    return {k: v for k, v in environ.items() if k not in KEYS}


# --- RunPod ---------------------------------------------------------------------------------------

def watchdog_bytes():
    return (HERE / "watchdog.py").read_bytes()


# The levers of every arm when the session names none (MFL033 N5b step 8, the levers adopted): the cache of
# torch.compile of each GPU populated (c8.4), written on 21/11 by the populate windows 20261121T115250Z-bd85ef7e
# (RunPod, RTX A5000), 20261121T115251Z-ee0d32bf (Modal, NVIDIA A10) and, in N6, 20261121T155237Z-3107a62b
# (Modal, NVIDIA A10G: Modal serves its type A10 with either card), with the image and the arguments of vLLM
# of watchdog.py. The watchdog tries them in the order of its `levers` (the values sorted): on Modal an A10G
# finds the A10 archive of another GPU (`another_gpu`) and uses its own, as the cold start of 21/11
# 20261121T155812Z-fe8c2ca4 did; the archives of the other provider are `absent`; a GPU without an archive
# (an L4) compiles as without the lever. A change of the image, of vLLM or of its arguments needs a new
# populate and these SHA-256 changed. `--skip-mm-profiling` (lever 7) gave no gain and is not among them.
DEFAULT_LEVERS = ("compile-cache:29012f5b05f56b51698c99f6d5f573d3511ae06191946be983554a8b35a55b2b,"   # Modal, A10
                  "compile-cache:cd31ce211155dae8d34230a9988f7fd82e855c54b94dc4e79607a7bc93608224,"   # Modal, A10G
                  "compile-cache:d38180544831bf3c0d1a45712e25f125ff4d18a9929012376f844a2224e0ff73")   # RunPod, A5000
NO_LEVERS = "none"                               # session.py run --levers none: an arm without levers
# MFL045 c2 (plan r2 §4): the levers of an app of the snapshot (`*-snap`, `*-snap-fault`), exactly: `snapshot` and
# the archives of the cache, which only the fallback uses (vLLM in sleep mode compiles to another hash). A deploy
# changed in anything makes its snapshots obsolete, so the session refuses any other set there.
SNAPSHOT_LEVERS = "snapshot," + DEFAULT_LEVERS
# MFL045: the levers of one provider only (`snapshot`: Modal's memory snapshot; `flashboot`: RunPod's FlashBoot).
PROVIDER_LEVERS = {"snapshot": "modal", "flashboot": "runpod"}
# MFL045 F5 (Davide, 23/11/2026): the two levers measured in MFL045 and not adopted stay as experimental levers, each
# with the section of the guide that gives its measures and its risks: never among DEFAULT_LEVERS, armed only when
# `--levers` names them; a session that arms one declares it (session.py), informative only.
EXPERIMENTAL_LEVERS = {"snapshot": "docs/how-to/convert-documents.md#the-memory-snapshot-of-modal-experimental",
                       "flashboot": "docs/how-to/convert-documents.md#flashboot-of-runpod-experimental"}


def experimental_levers(text):
    """The experimental levers among the levers of `text`, in their order, each stripped as `watchdog.levers` strips
    it (MFL045 c8: the session passes the text of `levers_text`, which has no spaces)."""
    levers = [x.strip() for x in (text or "").split(",")]
    return [x for x in levers if x.partition(":")[0] in EXPERIMENTAL_LEVERS]


def levers_text(text, provider=None):
    """MDF_LEVERS of an arm (MFL033 N5b): the levers of `text` checked against the watchdog's closed set
    and written in its order, or "" for none; ProviderError on another name, or (with `provider`) on a lever
    of another provider (MFL045)."""
    sys.path.insert(0, str(HERE))
    import watchdog
    try:
        lev = watchdog.levers(text)
    except watchdog.LeversInvalid as exc:
        raise ProviderError(str(exc)) from None
    if provider is not None:
        for x in lev:
            owner = PROVIDER_LEVERS.get(x.partition(":")[0])
            if owner is not None and owner != provider:
                raise ProviderError("the lever %s is only for %s, not %s" % (x, owner, provider))
    return ",".join(lev)


def base_env(fault="", allow_download=False, levers=""):
    """The complete env map of the endpoint (r2 C.1, r3:245-249, r4 C.1), the watchdog included.

    No value is ever empty: RunPod stores the env without the empty ones (seen on 19/11, MFL033
    c7.3), so `MDF_FAULT` is there only with a fault (the watchdog reads its absence as none), and
    `MDF_LEVERS` only with levers (MFL033 N5b).
    With `allow_download` (the populate) the init timeout follows the W1 of the populate (c7.4)."""
    data = watchdog_bytes()
    env = {"PORT": "8000", "PORT_HEALTH": "8000", "HEALTH_CHECK_PATH": "/health", "HF_HOME": "/runpod-volume/hf",
           "HF_HUB_OFFLINE": "1", "TMPDIR": "/tmp", "VLLM_CACHE_ROOT": "/tmp/vllm", "XDG_CACHE_HOME": "/tmp/cache",
           "RUNPOD_INIT_TIMEOUT": str(init_timeout(allow_download)), "MDF_PROVIDER": "runpod",
           "MDF_ALLOW_DOWNLOAD": "1" if allow_download else "0",
           "MDF_WATCHDOG_B64": base64.b64encode(data).decode("ascii"),
           "MDF_WATCHDOG_SHA256": hashlib.sha256(data).hexdigest()}
    if fault:
        env["MDF_FAULT"] = fault
    lev = levers_text(levers, "runpod")
    if lev:
        env["MDF_LEVERS"] = lev
    return env


# RunPod refuses a request whose expanded body exceeds 102,400 bytes (HTTP 413, "Expanded upstream request exceeds the
# 102400 byte limit. Reduce environment, command, or template values", F0 of MFL045, 22/11): the env carries the
# watchdog base64, so the largest PATCH of an arm stays under the limit less a margin for the rest of the template
# (the code of the snapshot of Modal is not in it: `snapshot.py`, only in the image of Modal).
RUNPOD_REQUEST_MAX = 102_400
RUNPOD_REQUEST_MARGIN = 8_192
ENTRYPOINT = ["/bin/sh", "-c", "printf %s \"$MDF_WATCHDOG_B64\" | base64 -d > /tmp/mdf-watchdog.py && "
              "echo \"$MDF_WATCHDOG_SHA256  /tmp/mdf-watchdog.py\" | sha256sum -c - && exec python3 /tmp/mdf-watchdog.py"]
# The time a worker may take to become ready before RunPod replaces it: W1 (900 s) plus a margin.
# Set both ways: the env `RUNPOD_INIT_TIMEOUT` of the docs, and the field `workers.initTimeout` that
# the REST v2 added after 17/11 (null = the platform default; schema read on 19/11, MFL033 c7.2).
# The populate (MDF_ALLOW_DOWNLOAD=1) has W1 at 1500 s, so 1560 s by the same rule (c7.4).
INIT_TIMEOUT_S = 960
POPULATE_INIT_TIMEOUT_S = 1560
# The image asks for CUDA >= 13.0 (NVIDIA_REQUIRE_CUDA of its config, read on 19/11): the floor of
# the worker placement, in the create, in every arm and in verify-pins (c7.4).
CUDA_MIN = "13.0"
# The keys of the env map that the arm sets: verify-pins compares the others with the base.
ARM_ENV_KEYS = ("MDF_FAULT", "MDF_ALLOW_DOWNLOAD", "RUNPOD_INIT_TIMEOUT", "MDF_LEVERS")


def init_timeout(allow_download=False):
    return POPULATE_INIT_TIMEOUT_S if allow_download else INIT_TIMEOUT_S


def workers_block(max_, allow_download=False):
    """The `workers` block of every create and PATCH: always complete, `initTimeout` included."""
    return {"min": 0, "max": max_, "idleTimeout": 60, "initTimeout": init_timeout(allow_download)}


def gpu_ok(gpu):
    """The `gpu` block read back: the pool alone, no exclusion, one GPU, the CUDA floor."""
    gpu = gpu or {}
    return (gpu.get("pools") == [POOL] and not gpu.get("excludedTypes") and gpu.get("count", 1) == 1
            and gpu.get("minCudaVersion") == CUDA_MIN and not gpu.get("allowedCudaVersions"))


# The pins that change in place (`runpod_endpoint.py align`; the arm sets them too): the env map with the
# watchdog, the `workers` block and the CUDA floor of `gpu`; since MFL045 c3 (plan r2 §6, A7) `flashboot`, OFF at
# rest, FLASHBOOT only during a session with the lever. Every other pin is invariant (MFL033 S-B2).
PIN_ALIGNABLE = ("watchdog_sha256", "watchdog_b64", "env", "workers", "gpu", "flashboot")


def pin_checks(ep, volumes, gpus, data_centers):
    """The pins of a RunPod endpoint, {name: bool}, shared by `verify-pins` and every arm (MFL033 S-B2):
    the image and the entrypoint with the watchdog, the type, scaling, FlashBoot and ports of
    `create_body`; the one volume `VOLUME` mounted, the endpoint limited to its data center (Secure
    Cloud: the REST v2 has no cloud field, the data center of the volume stands for it, 17/11), that
    data center in the catalog with network volumes, every GPU of the pool `secure` there, and the
    serverless price of the pool at most `RUNPOD_MAX_USD_PER_H` (the rate of the ledger); the pool
    itself (`pool`); then those of PIN_ALIGNABLE, with the base env and the disarmed `workers`."""
    base, env, gpu = base_env(), ep.get("env") or {}, ep.get("gpu") or {}
    named = [v for v in volumes if v.get("name") == VOLUME]
    mounted = [m.get("volumeId") for m in (ep.get("mounts") or {}).get("network") or []]
    vol_dc = named[0].get("dataCenter") if len(named) == 1 else None
    pool = [g for g in gpus if g.get("pool") == POOL]
    try:
        prices = [Decimal(str((g.get("price") or {})["serverless"])) for g in pool]
    except (KeyError, TypeError, ArithmeticError):
        prices = []
    return {
        "image": ep.get("image") == IMAGE,
        "entrypoint": ep.get("entrypoint") == ENTRYPOINT,
        "type": ep.get("type") == "LOAD_BALANCER",
        "scaling": ep.get("scaling") == {"type": "REQUEST_COUNT", "requestCount": 32},
        "flashboot": ep.get("flashboot") == "OFF",
        "ports": ep.get("ports") == ["8000/http"],
        "volume": len(named) == 1 and mounted == [named[0].get("id")],
        "data_centers": vol_dc is not None and (ep.get("dataCenterIds") or []) == [vol_dc],
        "secure": bool(pool) and all(g.get("secure") is True for g in pool) and any(
            d.get("id") == vol_dc and d.get("networkVolumeTypes") for d in data_centers),
        "rate": bool(prices) and len(prices) == len(pool) and max(prices) <= RUNPOD_MAX_USD_PER_H,
        "pool": (gpu.get("pools") == [POOL] and not gpu.get("excludedTypes") and gpu.get("count", 1) == 1
                 and not gpu.get("allowedCudaVersions")),
        "watchdog_sha256": env.get("MDF_WATCHDOG_SHA256") == base["MDF_WATCHDOG_SHA256"],
        "watchdog_b64": env.get("MDF_WATCHDOG_B64") == base["MDF_WATCHDOG_B64"],
        "env": ({k: v for k, v in env.items() if k not in ARM_ENV_KEYS}
                == {k: v for k, v in base.items() if k not in ARM_ENV_KEYS}),
        "workers": ep.get("workers") == workers_block(0),          # initTimeout included (c7.2)
        "gpu": gpu_ok(gpu),                                         # the CUDA floor (c7.4)
    }


def invariant_failures(checks):
    """The pins of `checks` that fail and that neither `align` nor an arm can change."""
    return sorted(k for k, ok in checks.items() if not ok and k not in PIN_ALIGNABLE)


def create_body(name, volume_id, data_center):
    """`POST /v2/serverless`: born disarmed (workers.max 0), the volume's data center only."""
    return {"name": name, "type": "LOAD_BALANCER", "image": IMAGE, "entrypoint": ENTRYPOINT,
            "scaling": {"type": "REQUEST_COUNT", "requestCount": 32}, "workers": workers_block(0),
            "gpu": {"pools": [POOL], "minCudaVersion": CUDA_MIN}, "flashboot": "OFF", "disk": 20,
            "ports": ["8000/http"], "dataCenterIds": [data_center], "mounts": {"network": [{"volumeId": volume_id}]},
            "env": base_env()}


class RunPod:
    """REST v2 with the key of `--env-file`; `transport` and `clock` for the tests."""

    provider = "runpod"

    def __init__(self, key, name=None, transport=None, clock=time.time, sleep=time.sleep):
        self._key = key
        self.name = name
        self.clock, self.sleep = clock, sleep
        self._transport = transport
        self.http = self._client(httpx.Timeout(30.0, connect=10.0))
        self._endpoint = None
        self._deadline = None                  # the term T of the session during an arm (S-B4)

    def _client(self, timeout):
        return httpx.Client(transport=self._transport, timeout=timeout,
                            headers={"Authorization": "Bearer " + self._key.reveal()}, follow_redirects=False)

    def log_follower(self, path, t_arm, clock=time.time):
        """The follower of the worker logs of this endpoint (`worker_logs.py`, c7.4), not started."""
        import worker_logs
        return worker_logs.Follower(worker_logs.RunPodLogs(REST, self.endpoint()["id"], self._client), path, t_arm,
                                    clock=clock)

    def close(self):
        self.http.close()

    def _past(self, what):
        if self._deadline is not None and self.clock() >= self._deadline:
            raise TermReached("RunPod %s: the term of the session came during the arm" % what)

    def call(self, method, path, body=None, params=None, ok=(200, 201, 204)):
        """One call of the REST v2; during an arm each one gets at most the time left to the term T of
        the session, and one that ends after T is `TermReached` (S-B4)."""
        extra = {}
        if self._deadline is not None:
            self._past("%s %s" % (method, path))
            left = self._deadline - self.clock()
            extra["timeout"] = httpx.Timeout(min(30.0, left), connect=min(10.0, left))
        try:
            r = self.http.request(method, REST + path, json=body, params=params, **extra)
        except httpx.HTTPError as exc:
            self._past("%s %s" % (method, path))
            raise ProviderError("RunPod %s %s: %s" % (method, path, type(exc).__name__)) from None
        self._past("%s %s" % (method, path))
        if r.status_code not in ok:
            raise ProviderError("RunPod %s %s: HTTP %d %s" % (method, path, r.status_code, r.text[:300]))
        return r.json() if r.content else {}

    def balance(self):
        try:
            r = self.http.post(GRAPHQL, json={"query": "query { myself { clientBalance } }"})
            return r.json()["data"]["myself"]["clientBalance"]
        except Exception:
            return None

    # endpoints and volumes, by name
    def endpoints(self):
        out, cursor = [], None
        for _ in range(50):
            page = self.call("GET", "/serverless", params={"cursor": cursor} if cursor else None)
            out += page.get("endpoints") or []
            pag = page.get("pagination") or {}
            if not pag.get("hasNextPage"):
                return out
            cursor = pag.get("nextCursor")
        raise ProviderError("RunPod: too many endpoints")

    def endpoint(self, name=None, refresh=False):
        name = name or self.name
        if refresh or self._endpoint is None or self._endpoint.get("name") != name:
            found = [e for e in self.endpoints() if e.get("name") == name]
            if len(found) != 1:
                raise ProviderError("RunPod: %d endpoints named %s" % (len(found), name))
            self._endpoint = self.call("GET", "/serverless/" + found[0]["id"])
        return self._endpoint

    def exists(self, name):
        return any(e.get("name") == name for e in self.endpoints())

    @property
    def url(self):
        return "https://%s.api.runpod.ai" % self.endpoint()["id"]

    def volumes(self):
        return self.call("GET", "/network-volumes").get("networkVolumes") or []

    def workers(self):
        return self.call("GET", "/serverless/%s/workers" % self.endpoint()["id"])

    def pin_facts(self):
        """(volumes, GPUs, data centers) of the catalog that `pin_checks` reads."""
        return (self.volumes(), self.call("GET", "/catalog/gpus").get("gpus") or [],
                self.call("GET", "/catalog/datacenters").get("dataCenters") or [])

    def pins(self, ep=None):
        """`pin_checks` of the endpoint (read again unless given) and the facts it read."""
        ep = self.endpoint(refresh=True) if ep is None else ep
        facts = self.pin_facts()
        return pin_checks(ep, *facts), facts

    def pin_refusal(self):
        """The invariant pins that fail, read now (the session asks before its arm row), or []."""
        return invariant_failures(self.pins()[0])

    def patch(self, body):
        return self.call("PATCH", "/serverless/" + self.endpoint()["id"], body=body)

    # the face of the session
    def state(self):
        """The reading of the session and the drivers; since MFL045 c3 also FlashBoot and the configuration version
        of the endpoint and of each worker (`version`, `isStale`: a fact, never a criterion)."""
        ep = self.endpoint(refresh=True)
        w = self.workers()
        summary, workers = w.get("summary") or {}, w.get("workers") or []
        armed = (ep.get("workers") or {}).get("max", 0) > 0
        busy = summary.get("total", 0) > 0 or len(workers) > 0
        return {"armed": armed, "busy": busy, "workers_max": (ep.get("workers") or {}).get("max"),
                "flashboot": ep.get("flashboot"), "version": ep.get("version"),
                "summary": summary, "workers": [{"id": x.get("id"), "status": x.get("status"),
                                                 "uptime": x.get("uptimeSeconds"), "gpu": x.get("gpuTypeId"),
                                                 "data_center": x.get("dataCenterId"), "version": x.get("version"),
                                                 "stale": x.get("isStale")} for x in workers]}

    def arm(self, fault="", allow_download=False, levers="", deadline=None):
        """`workers.max = 1` with the env map when it changes (MFL033 S-B2): only on an endpoint whose
        invariant pins hold, checked on the same reading before the PATCH and again on the read-back.
        `deadline`: the term T of the session (epoch seconds), for every call of the arm (S-B4)."""
        self._deadline = deadline
        try:
            return self._arm(fault, allow_download, levers)
        finally:
            self._deadline = None

    def _arm(self, fault, allow_download, levers):
        ep = self.endpoint(refresh=True)
        facts = self.pin_facts()
        failed = invariant_failures(pin_checks(ep, *facts))
        if failed:
            raise ProviderError("RunPod: the endpoint is outside its pins (%s): no arm; align cannot change them, "
                                "see verify-pins" % ", ".join(failed))
        workers = workers_block(1, allow_download)
        body = {"workers": workers}
        env = base_env(fault, allow_download, levers)
        if ep.get("env") != env:
            body["env"] = env                       # the complete map, only when it changes (r4 §5.2)
        if not gpu_ok(ep.get("gpu")):
            body["gpu"] = {"minCudaVersion": CUDA_MIN}   # no `pools`: pool and exclusions stay (schema 19/11)
        # MFL045 c3 (plan r2 §6): FlashBoot on only with the lever, OFF otherwise; in the PATCH when it differs
        flash = "FLASHBOOT" if "flashboot" in (env.get("MDF_LEVERS") or "").split(",") else "OFF"
        if ep.get("flashboot") != flash:
            body["flashboot"] = flash
        self.patch(body)
        ep = self.endpoint(refresh=True)
        if (ep.get("workers") != workers or ep.get("env") != env or not gpu_ok(ep.get("gpu"))
                or ep.get("flashboot") != flash or invariant_failures(pin_checks(ep, *facts))):
            raise ProviderError("RunPod: the endpoint read back differs from the arm")
        return {"env_changed": "env" in body, "gpu_changed": "gpu" in body, "flashboot": flash,
                "flashboot_changed": "flashboot" in body}

    def restore_flashboot(self):
        """MFL045 c3 (plan r2 §6, A7): FlashBoot back to OFF, its value at rest, with a PATCH of its own (the disarm
        of N4 stays a PATCH of the workers only), read back; ProviderError when it is not OFF."""
        self.patch({"flashboot": "OFF"})
        if self.endpoint(refresh=True).get("flashboot") != "OFF":
            raise ProviderError("RunPod: flashboot is not OFF after its PATCH")

    def disarm(self):
        self.patch({"workers": workers_block(0)})
        if (self.endpoint(refresh=True).get("workers") or {}).get("max") != 0:
            raise ProviderError("RunPod: workers.max is not 0 after the disarm")

    def zero(self):
        s = self.state()
        return not s["busy"], s


# --- Modal ----------------------------------------------------------------------------------------

MODAL_COLUMNS = "1000"          # off a terminal rich wraps at 80 columns: the URL of the deploy must stay whole
SYSTEM_LOG_TAIL = 1000          # the system lines of a window fetched at most (the SDK allows 20,000)


def run_modal(args, env=None, timeout=900, runner=subprocess.run):
    """`modal ARGS` of the pinned SDK; (exit status, stdout). The SDK token stays in ~/.modal.toml."""
    if shutil.which("uv") is None and runner is subprocess.run:
        raise ProviderError("uv not found")
    env = dict(child_env(__import__("os").environ) if env is None else env, COLUMNS=MODAL_COLUMNS)
    try:
        r = runner(MODAL_SDK + list(args), capture_output=True, text=True, timeout=timeout, env=env,
                   stdin=subprocess.DEVNULL)
    except subprocess.TimeoutExpired:
        raise ProviderError("modal %s: timeout" % args[0]) from None
    return r.returncode, r.stdout, r.stderr


def cli_error(text, limit=300):
    """The end of the CLI's stderr as text: rich draws its error in a box as wide as COLUMNS, so the box
    and the runs of blanks go before the cut (seen on 19/11 with "Token missing": MFL033 c7.9)."""
    plain = re.sub(r"[\u2500-\u257f]+", " ", text or "")
    return " ".join(plain.split())[-limit:]


def modal_container(row):
    """A row of `modal container list --json` as a worker of the readings: the container id, PENDING
    until it starts (`start_time` "Pending"), then RUNNING with the start time (MFL033 c7.8)."""
    start = _field(row, "Start Time", "start_time")
    pending = start in (None, "", "Pending")
    return {"id": _field(row, "Container ID", "container_id"), "status": "PENDING" if pending else "RUNNING",
            "start": None if pending else start}


class Modal:
    """The app `name` of `modal_app.py`: deployed = armed, stopped = disarmed.

    The JSON keys of the CLI are those of `modal==1.6.3` (`app_id`, `description`, `state`;
    `container_id`, `app_id`, `app_name`, `start_time`; `name` of a volume), the display names of
    older versions still accepted. Only an app in the state `stopped` is disarmed (App. G step 5:
    "app ferma"); `stopping...` is still armed. The readings carry the containers as `workers`
    (`modal_container`), so `drivers.Watch` sees a container replaced (c7.8). The env file gives the
    proxy token that the deploy checks against the workspace of the SDK token (c7.13)."""

    provider = "modal"

    def __init__(self, name, runner=subprocess.run, clock=time.time, sleep=time.sleep, environ=None, env_file=None,
                 snapshot=False):
        self.name, self.runner, self.clock, self.sleep = name, runner, clock, sleep
        self.environ = child_env(environ if environ is not None else __import__("os").environ)
        self.env_file = env_file
        self._url = None
        # MFL045 c2: the function of the inputs, `serve` of modal_app.py or `Serve.*` of the class of modal_snap.py
        self.function = "Serve.*" if snapshot else "serve"
        self._deadline = None                  # the term T of the session during an arm (S-B4)

    def _left(self, default, what):
        """`default` seconds, at most the time left to the term T during an arm; `TermReached` after it."""
        if self._deadline is None:
            return default
        left = self._deadline - self.clock()
        if left <= 0:
            raise TermReached("modal %s: the term of the session came during the arm" % what)
        return min(default, left)

    def _json(self, *args):
        code, out, err = run_modal(list(args) + ["--json"], env=self.environ, timeout=self._left(120, args[0]),
                                   runner=self.runner)
        if code:
            raise ProviderError("modal %s: exit %d %s" % (" ".join(args), code, cli_error(err)))
        try:
            return json.loads(out or "[]")
        except ValueError:
            raise ProviderError("modal %s: not JSON" % " ".join(args)) from None

    def apps(self):
        return [a for a in self._json("app", "list") if _field(a, "Description", "description", "name") == self.name]

    def containers(self, app_ids):
        rows = self._json("container", "list")
        return [c for c in rows if _field(c, "App ID", "app_id") in app_ids]

    def app_containers(self):
        """The containers of the apps named `name`, from one `container list` (its `app_name`)."""
        return [modal_container(c) for c in self._json("container", "list")
                if _field(c, "App Name", "app_name") == self.name]

    @property
    def url(self):
        if self._url is None:
            raise ProviderError("modal: the URL is known after a deploy")
        return self._url

    def state(self):
        apps = self.apps()
        live = [a for a in apps if str(_field(a, "State", "state")).lower().rstrip(".") != "stopped"]
        conts = self.containers({_field(a, "App ID", "app_id") for a in apps})
        return {"armed": bool(live), "busy": bool(conts), "apps": [{"id": _field(a, "App ID", "app_id"),
                                                                    "state": _field(a, "State", "state")} for a in apps],
                "containers": [_field(c, "Container ID", "container_id") for c in conts],
                "workers": [modal_container(c) for c in conts]}

    def volume_ok(self):
        return any(_field(v, "Name", "name") == VOLUME for v in self._json("volume", "list"))

    def proxy_token(self):
        """The proxy token of the env file among the proxy tokens of the workspace (`modal workspace
        proxy-tokens list --json` of 1.6.3: `name`, `token_id`, `created_at`, `created_by`, `scoped`).
        The id is compared in memory and never printed: the answer holds only counts and `scoped`
        (MFL033 c7.13; on 19/11 the env file held a token that the workspace did not have, and Modal's
        edge refused every request of `probe-valid`)."""
        if self.env_file is None:
            raise ProviderError("modal: no env file, the proxy token cannot be checked")
        token = read_key(self.env_file, "MODAL_PROXY_TOKEN_ID").reveal().encode("utf-8")
        rows = self._json("workspace", "proxy-tokens", "list")
        if not isinstance(rows, list):
            raise ProviderError("modal workspace proxy-tokens list: not a list")
        found = [r for r in rows if hmac.compare_digest(str(_field(r, "Token ID", "token_id") or "").encode("utf-8"), token)]
        return {"proxy_tokens": len(rows), "in_workspace": bool(found),
                "scoped": _field(found[0], "Scoped", "scoped") if found else None}

    def arm(self, fault="", allow_download=False, levers="", deadline=None):
        """`modal deploy` after the Volume is found (r4 §5.3; the app never creates it: c7.8) and the
        proxy token of the env file is found in the workspace (c7.13); otherwise no deploy. MDF_LEVERS of
        the deploy names the levers (MFL033 N5b), absent without them. `deadline`: the term T of the
        session; every call of the CLI gets at most the time left to it, and a deploy still running or
        ended after it is `TermReached` (S-B4), after which the session stops the app."""
        self._deadline = deadline
        try:
            return self._arm(fault, allow_download, levers)
        finally:
            self._deadline = None

    def _arm(self, fault, allow_download, levers):
        lev = levers_text(levers, "modal")
        snapshot = "snapshot" in lev.split(",")             # MFL045 c2: the app of the snapshot, modal_snap.py
        if not self.volume_ok():
            raise ProviderError("modal: the Volume %s is missing: run the populate first" % VOLUME)
        token = self.proxy_token()
        if not token["in_workspace"]:
            raise ProviderError("modal: proxy_token_not_in_workspace: MODAL_PROXY_TOKEN_ID of the env file is not "
                                "among the %d proxy tokens of the workspace: no deploy" % token["proxy_tokens"])
        env = dict(self.environ, MDF_MODAL_APP=self.name)
        env.pop("MDF_FAULT", None)
        env.pop("MDF_LEVERS", None)
        if fault:
            env["MDF_FAULT"] = fault
        if lev:
            env["MDF_LEVERS"] = lev
        try:
            app_file = HERE / ("modal_snap.py" if snapshot else "modal_app.py")
            code, out, err = run_modal(["deploy", str(app_file)], env=env, timeout=self._left(1200, "deploy"),
                                       runner=self.runner)
        finally:
            if self._deadline is not None and self.clock() >= self._deadline:
                raise TermReached("modal deploy: the term of the session came during the deploy") from None
        if code:
            raise ProviderError("modal deploy: exit %d %s" % (code, cli_error(err)))
        self.function = "Serve.*" if snapshot else "serve"
        if snapshot:
            urls = self.class_urls()
        else:
            urls = [w for w in (out + err).split() if w.startswith("https://") and w.endswith(".modal.run")]
        if len(set(urls)) != 1:
            raise ProviderError("modal deploy: the URL of the web function not found")
        self._url = urls[0]
        return {"url": self._url, "volume": VOLUME, "proxy_token": token}

    def class_urls(self):
        """The web URLs of the methods of `Serve` of modal_snap.py, from `modal function info NAME/Serve.* --json` of
        1.6.3 (`cli/function.py` info: `method_details`), read while the app is deployed and within the term of the
        session. MFL045 F1 (S0, 23/11): the deploy of a class prints no URL, because the response of the creation
        does not set `method_definitions` (`modal/_output/status.py:78-93`); a failed reading gives no URL."""
        try:
            info = self._json("function", "info", "%s/%s" % (self.name, self.function))
        except TermReached:
            raise
        except ProviderError:
            return []
        details = info.get("method_details") if isinstance(info, dict) else None
        if not isinstance(details, dict):
            return []
        urls = [str((d or {}).get("web_url") or "") for d in details.values() if isinstance(d, dict)]
        return [u for u in urls if u.startswith("https://") and u.endswith(".modal.run")] \
            if len([u for u in urls if u]) == 1 else []

    def system_logs(self, since):
        """The system lines of the app since `since` (epoch seconds, whole seconds in UTC), fetched once,
        without `-f` (MFL033 c7.14): `modal app logs NAME --since ISO --tail N --timestamps
        --show-function-id --show-container-id -s system` of 1.6.3 (`cli/app.py` logs, `cli/_logs.py`).
        On 20/11 the stream `-f -s system` of the follower never delivered the scheduler's "waiting to
        be scheduled", which the SDK shows only in its progress bar (`_output/pty.py`), while this fetch
        had it. Read-only and free; an error raises."""
        stamp = datetime.fromtimestamp(int(since), timezone.utc).isoformat()
        code, out, err = run_modal(["app", "logs", self.name, "--since", stamp, "--tail", str(SYSTEM_LOG_TAIL),
                                    "--timestamps", "--show-function-id", "--show-container-id", "-s", "system"],
                                   env=self.environ, timeout=60, runner=self.runner)
        if code:
            raise ProviderError("modal app logs: exit %d %s" % (code, cli_error(err)))
        return [line for line in (out or "").splitlines() if line.strip()]

    def inputs(self, since):
        """The inputs of the web function enqueued since `since` (epoch seconds, whole seconds in UTC),
        oldest first (MFL033 c7.16): `modal function calls NAME/serve --since ISO --json` of 1.6.3 (MFL045 c2:
        `NAME/Serve.*` for the class of the snapshot)
        (`cli/function.py` calls; every HTTP request of a `web_server` is an input), each with
        `enqueued_at`, `started_at`, `container_id`, `execution_time_seconds` and `status`. Read-only
        and free; an error raises."""
        stamp = datetime.fromtimestamp(int(since), timezone.utc).isoformat()
        rows = self._json("function", "calls", "%s/%s" % (self.name, self.function), "--since", stamp)
        return [r for r in rows if isinstance(r, dict)] if isinstance(rows, list) else []

    def log_follower(self, path, t_arm, clock=time.time):
        """The follower of the logs and containers of this app (`worker_logs.ModalFollower`, c7.8)."""
        import worker_logs
        return worker_logs.ModalFollower(worker_logs.ModalLogs(self.name, self.environ, self.app_containers), path,
                                         t_arm, clock=clock)

    def disarm(self):
        """`modal app stop`. Its exit 1 for nothing to stop (modal 1.6.3 `cli/app.py`: an app never
        deployed is a NotFoundError, an app already stopped a SystemExit) leaves nothing armed: it is a
        disarm, and the zero is read from the state as always (MFL033 c7.12; on 19/11 a deploy refused
        by the SDK left no app, and 57 refused stops made the window orphan)."""
        code, _, err = run_modal(["app", "stop", self.name, "--yes"], env=self.environ, timeout=300, runner=self.runner)
        if code and not self.nothing_to_stop(cli_error(err, 2000)):
            raise ProviderError("modal app stop: exit %d %s" % (code, cli_error(err)))

    def nothing_to_stop(self, text):
        return "No App with name '%s' found" % self.name in text or "App is already stopped." in text

    def zero(self):
        s = self.state()
        return not s["armed"] and not s["busy"], s


def _field(row, *names):
    for n in names:
        if isinstance(row, dict) and n in row:
            return row[n]
    return None


def make(provider, name, env_file, runner=subprocess.run, transport=None):
    if provider == "runpod":
        return RunPod(read_key(env_file, "RUNPOD_API_KEY"), name, transport=transport)
    if provider == "modal":
        return Modal(name, runner=runner, env_file=env_file)
    raise ProviderError("provider runpod or modal")


def emit(obj, stream=None):
    (stream or sys.stdout).write(json.dumps(obj, sort_keys=True, default=str) + "\n")
    (stream or sys.stdout).flush()
