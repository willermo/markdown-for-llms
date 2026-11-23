#!/usr/bin/env python3
"""A series of starts of a rented server, inside `session.py run` (MFL033 N5b; plan r4 App. D and the
levers of the start, `n5/leve-avvio.md` §5).

    start_series.py --env-file F --server S --servers SV --input FILE [--input FILE ...] --cold N [--warm M] [--arm-ahead S]
                    [--via engine|page|populate] [--gpu NAME [--gpu-tries K]] [--cap-usd X]
                    [--resume N --idle S [--idle-last S] [--resume-via V1,V2,...]]
                    --spend-ledger L --spend-cap-usd C [--out DIR]

The session armed the provider: `MDF_SESSION_ARM` names its `arm` row in L, `SESSION_WORKER_LOG` the log
of its workers. `--input` once for every start, or once per cold start in order (MFL045 c2: D04, D16, D18 of
M2), a warm start with the input of its cold one. The series sends N cold starts; after the k-th, `M // N` warm ones (one more after each
of the first `M % N`). Every cold start after the first waits for the quiet of the provider: no worker
RUNNING on RunPod and, with the log of the session, a "stop container" after the last RUNNING of every
worker that ran (RUNNING absent for STOP_LINE_WAIT_S without that line is quiet too, and a fact); no
container on Modal. Both come after 60 s without requests (`idleTimeout`, `scaledown_window`); the
provider is read every QUIET_EVERY_S, at most QUIET_MAX_S, then QUIET_MARGIN_S more. With `--arm-ahead S`
the first start waits until S seconds after the arm, reading the provider every READ_EVERY_S (S2 of the
levers: on RunPod the arm alone loads the image, and nothing is billed before a worker is RUNNING; a
worker RUNNING before the first request is a fact, `running_before_request`).

Each start is a line as it ends:
- `--via engine` (the default): `mdf --env-file F convert FILE` on the route marker-server in balanced,
  with `--server-cap-usd X` and the ledger (the engine reserves inside the window, as for any document);
  from its `engine.json` the relay's wake, readiness, last answer and close, the stages of Marker, the
  pages, and the seconds per page after readiness;
- `--via populate` (c8.4): readiness from the first request, then the `mdf_compile_cache` lines of the
  populate of the cache of torch.compile in the log of the session (and of the commit of the Volume on
  Modal), with the SHA-256 of the archive; no inference;
- `--via page`: readiness from the first request (`GET /v1/models` every 5 s, GETs of 30 s), then one
  page straight to vLLM (`drivers.page_body`); no attempt of the engine. An attempt needs `T - now - 60 s`
  above its floor, `ready_within` (900 s on RunPod), which 420 s of arm ahead in a window of 0.30 $
  (1224 s) do not leave: S2 sends its page this way.

With `--gpu NAME` (MFL033 N6) each cold start first wakes the container (a GET of /v1/models every
GPU_WAKE_EVERY_S) until the log of the session gives its GPU (`mdf_gpu`); on another GPU the series sends
it nothing more, so that it leaves before vLLM is ready (no archive of the cache written, no document
sent), waits for the quiet of the provider and wakes the next one, at most `--gpu-tries` (1 by default);
none on NAME: the start fails with `gpu_not_assigned`. Modal serves its type `A10` (the `A10G` of `gpu=`)
with an NVIDIA A10 or an NVIDIA A10G as it likes (21/11), and an archive of the cache serves only the GPU
of its manifest. The start is the try taken (`send_s`); the tries are in `gpu_tries`. With `--via
populate` the populate must name NAME too.

With `--resume N` (MFL045 c3, plan r2 §6: FlashBoot of RunPod) N requests follow the cold and warm starts, each
after `--idle` seconds without requests from the end of the one before (`--idle-last` for the last one;
prescription P-4: 120 s, so that the worker is paused well before the request), each by the way of
`--resume-via` (`engine` or `page`, one per resume; `--via` by default). A resume is told from the log: the same
`boot_id` as the container before and an `mdf_pause` of it since the request before (`start_kind` `resume`); a new
`mdf_start` is a missed resume (`missed`); the same boot without a pause a warm start (`warm`).

After the series the log of the session gives the phases of every container start (`phases`), each
from the arm: the image (RunPod: from the first listing of the worker to "Loaded image ID" and "create
container", from the host's cache or not), RUNNING, the proxy's start (`mdf_start`), the check of the
weights, the import of vLLM up to its engine, the load of the weights, `torch.compile` (and whether its
graphs came from the cache, and the `mdf_compile_cache` lines of its use or populate), the profiling and
warm-up, the graphs and the cache KV, the API up to the proxy's readiness (`mdf_ready`), and the first useful
answer (`first_useful_s`: the first `mdf_request` forwarded POST /v1/chat/completions 200 of the container, MFL045
c2). With a snapshot (MFL045 c2, plan r2 E.7) each start has its `snapshot`: the kind (`create`, `restore`,
`boot`) and generation of its `mdf_restore`, the creation of the snapshot when the container took it (its
seconds, and the stages of vLLM before the snapshot in `snap_boot`), the wake or the declared fallback; the SDK's
"GPU memory snapshot restore failed" (exit 222) is a start of its own, `restore_failed`. The cold starts carry
the container start that followed them (and the failed restores before it) and its kind, `start_kind`. The last
line is the result. Exit 0: every start done; 6: a start failed, the provider not quiet, or a guarantee
broken (G1: two workers RUNNING at once in a reading); 2: outside a session; 4: a provider error.
"""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import drivers as dv  # noqa: E402
import providers as pv  # noqa: E402
import worker_logs as wl  # noqa: E402

from markdown_for_llms.bench import serverspend, spend  # noqa: E402
from markdown_for_llms.engine import servers  # noqa: E402
from markdown_for_llms.engine.codes import EngineError  # noqa: E402

READ_EVERY_S = 30               # --arm-ahead: a reading of the provider so often
QUIET_EVERY_S = 10
QUIET_MAX_S = 600
QUIET_MARGIN_S = 10
STOP_LINE_WAIT_S = 90           # RunPod: nothing RUNNING so long without a "stop container" line is quiet too
READY_WITHIN_S = {"runpod": 960, "modal": 1260}      # as the readiness of the drivers (W1 plus the margin)
READY_GET_S = 30
PAGE_TIMEOUT_S = 150.0
CONVERT_TIMEOUT_S = 3600        # one `mdf convert`; its own term `D` ends it first
POPULATE_WAIT_S = 600           # --via populate: the lines of the populate of the cache after readiness (c8.4)
SEND_SLACK_S = 5.0              # a container start belongs to the first cold start sent before it
GPU_WAKE_EVERY_S = 5.0          # --gpu: a GET of /v1/models so often, until the log gives the GPU (MFL033 N6)
GPU_WAIT_S = 180                # --gpu: the container start with its GPU in the log within so long of a try

# The lines of vLLM 0.25.1 at its start (the logs of N4 and N5a, RunPod and Modal)
VLLM = {
    "engine": re.compile(r"Initializing a V1 LLM engine"),
    "load_s": re.compile(r"Model loading took [\d.]+ GiB memory and ([\d.]+) seconds"),
    "compile_s": re.compile(r"torch\.compile took ([\d.]+) s in total"),
    "compile_cached": re.compile(r"Directly load the compiled graph"),
    "profiling_s": re.compile(r"Initial profiling/warmup run took ([\d.]+) s"),
    "init_engine_s": re.compile(r"init engine \(profile, create kv cache, warmup model\) took ([\d.]+) s"),
    "capture_s": re.compile(r"Graph capturing finished in ([\d.]+) secs"),
    "api": re.compile(r"Application startup complete"),
}
# The system lines of a RunPod worker about its image (the logs of N4 and N5a)
IMAGE_CACHE = re.compile(r"loading container image from cache")
CHAT_PATH = "/v1/chat/completions"
IMAGE_LOADED = re.compile(r"^Loaded image ID")
CREATE = re.compile(r"^create container\b")


def emit(obj):
    pv.emit(obj)


def _rel(t, t_arm):
    return None if t is None or t_arm is None else round(float(t) - t_arm, 1)


def _num(m):
    try:
        return float(m.group(1))
    except (IndexError, TypeError, ValueError):
        return None


def phases(path, t_arm=None):
    """The container starts in the log of a session, in order (module docstring): one dict per
    `mdf_start`, its times in seconds from the arm (`t_arm`, else the follower's first line)."""
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    workers, starts, current = {}, [], {}
    pre = {}                    # MFL045 c2: a container's snapshot before its proxy starts (snap_boot, mdf_restore)
    for raw in text.splitlines():
        e = wl.parse(raw)
        if e is None:
            continue
        ev, wid, src = e["event"], e["worker"], e["source"]
        if src == "session" and ev is not None and ev.get("event") == "follow":
            if t_arm is None:
                t_arm = wl.epoch(ev.get("t_arm"))
            continue
        if wid == "-":
            continue
        w = workers.setdefault(wid, {"listed_s": e["since_arm"], "status": None, "running": [], "image": None,
                                     "starts": 0})
        if src == "session":
            if ev is not None and ev.get("event") == "worker":
                status = str(ev.get("status") or "").upper()
                if status == "RUNNING" and w["status"] != "RUNNING":
                    w["running"].append(e["since_arm"])
                w["status"] = status
            continue
        if src == "system":
            line = e["line"]
            img = w["image"] if w["image"] is not None else {"listed_s": w["listed_s"], "from_cache": None,
                                                              "loaded_s": None, "create_s": None}
            if IMAGE_CACHE.search(line):
                img["from_cache"] = True
            elif IMAGE_LOADED.search(line) and img["loaded_s"] is None:
                img["loaded_s"] = e["since_arm"]
            elif CREATE.search(line) and img["create_s"] is None:
                img["create_s"] = e["since_arm"]
            else:
                continue
            if w["starts"] == 0:                          # the image of the first start of the worker only
                w["image"] = img
            continue
        if src != "container":
            continue
        s = current.get(wid)
        if ev is not None:
            kind, t = ev.get("event"), ev.get("t")
            if kind == "mdf_start":
                s = {"worker": wl.short(wid), "start_s": _rel(t, t_arm), "_wid": wid, "gpu": None,
                     "image": w["image"] if w["starts"] == 0 else None, "running_s": None,
                     "boot_id": ev.get("boot_id")}
                if s["image"] is not None and s["image"]["loaded_s"] is not None:
                    s["image"] = dict(s["image"], seconds=round(s["image"]["loaded_s"] - s["image"]["listed_s"], 1))
                w["starts"] += 1
                current[wid] = s
                starts.append(s)
                p = pre.pop(wid, {})
                r = p.get("restore")
                if r is not None:                                    # MFL045 c2: a start of the snapshot
                    s["snapshot"] = {"kind": r.get("kind"), "gen": r.get("gen"), "age_s": r.get("age_s"),
                                     "restore_s": _rel(r.get("t"), t_arm), "created": p.get("created"),
                                     "fallback": None, "wake": None}
                    if p.get("boot") is not None:
                        s["snap_boot"] = p["boot"]
            elif kind in ("mdf_snap_boot", "mdf_snapshot", "mdf_restore") or ev.get("snap_boot"):
                p = pre.setdefault(wid, {})
                if kind == "mdf_snap_boot":
                    p.setdefault("created", {})["ready_s"] = ev.get("ready_s")
                elif kind == "mdf_snapshot":
                    p.setdefault("created", {}).update({k: ev.get(k) for k in ("ok", "gen", "seconds", "reason")})
                elif kind == "mdf_restore":
                    p["restore"] = ev
                elif kind == "mdf_vllm_start":
                    p["boot"] = {"_vllm_t": t}
            elif s is not None and kind == "mdf_pause":           # MFL045 c3: FlashBoot froze the proxy
                s.setdefault("pauses", []).append({"since_arm": _rel(t, t_arm), "seconds": ev.get("seconds"),
                                                   "during": ev.get("during"), "applied": ev.get("applied"),
                                                   "boot_id": ev.get("boot_id")})
            elif s is not None and kind == "mdf_wake" and s.get("snapshot") is not None:
                s["snapshot"]["wake"] = {"ok": ev.get("ok"), "seconds": ev.get("seconds")}
            elif s is not None and kind == "mdf_fallback" and s.get("snapshot") is not None:
                s["snapshot"]["fallback"] = ev.get("reason")
            elif s is not None and kind == "mdf_request" and ev.get("kind") == "forwarded" and ev.get("method") == "POST" \
                    and ev.get("path") == CHAT_PATH and ev.get("status") == 200:
                s.setdefault("first_useful_s", _rel(t, t_arm))
            elif s is not None and kind == "mdf_gpu":
                s["gpu"] = ev.get("names")
            elif s is not None and kind == "mdf_weights_check" and ev.get("phase") == "start":
                s["weights_s"] = ev.get("seconds")
            elif s is not None and kind == "mdf_vllm_start":
                s["_vllm_t"] = t
            elif s is not None and kind == "mdf_ready":
                s["ready_s"] = _rel(t, t_arm)
                s["start_to_ready_s"] = ev.get("since_start")
                s["_ready_t"] = t
            elif s is not None and kind == "mdf_compile_cache":      # c8.4: its use, or its populate
                key = {"use": "compile_cache", "populate": "cache_populate", "commit": "cache_commit"}.get(
                    ev.get("phase"), "compile_cache_other")
                s[key] = {k: v for k, v in ev.items() if k not in ("event", "mono", "t", "phase")}
            continue
        if wl.RESTORE_FAILED.search(e["line"]):                     # MFL045 c2: exit 222 of the SDK
            starts.append({"worker": wl.short(wid), "start_s": e["since_arm"], "_wid": wid, "gpu": None, "image": None,
                           "running_s": None, "snapshot": {"kind": "restore_failed"}})
            continue
        target = s if s is not None else (pre.get(wid) or {}).get("boot")    # the stages of snap_boot too
        if target is None:
            continue
        t = wl.epoch(e["ts"])
        for key, rx in VLLM.items():
            m = rx.search(e["line"])
            if not m:
                continue
            if key == "engine":
                target.setdefault("_engine_t", t)
            elif key == "compile_cached":
                target["compile_cached"] = True
            elif key == "api":
                target.setdefault("_api_t", t)
            elif key == "init_engine_s":
                target.setdefault(key, _num(m))
                target.setdefault("_init_t", t)
            else:
                target.setdefault(key, _num(m))
    for s in starts:
        # RUNNING: the last listing RUNNING of the worker by 15 s after the start (on Modal it comes after it)
        runs = [r for r in workers[s["_wid"]]["running"] if r is not None and s["start_s"] is not None
                and r <= s["start_s"] + 15]
        s["running_s"] = runs[-1] if runs else None
        _stages(s)
        if s.get("snap_boot") is not None:
            _stages(s["snap_boot"])
    return starts


def _stages(s):
    """The stages of a start of vLLM from the times its lines gave; the private times dropped."""
    if s.get("_engine_t") is not None and s.get("_vllm_t") is not None:
        s["import_s"] = round(s["_engine_t"] - s["_vllm_t"], 1)
    if None not in (s.get("init_engine_s"), s.get("compile_s"), s.get("profiling_s")):
        s["graphs_s"] = round(s["init_engine_s"] - s["compile_s"] - s["profiling_s"], 1)
    if s.get("_ready_t") is not None and s.get("_init_t") is not None:
        s["api_s"] = round(s["_ready_t"] - s["_init_t"], 1)
    s.setdefault("compile_cached", False)
    for k in [k for k in s if k.startswith("_")]:
        del s[k]


def summary_of(stdout):
    """The summary `mdf convert` prints: one JSON document, indented over several lines (`cli.emit`); else
    the last line that is one (MFL033 c8.2.1: on 21/11 the series read only single lines, and lost it)."""
    text = (stdout or "").strip()
    try:
        return json.loads(text)
    except ValueError:
        pass
    for line in reversed(text.splitlines()):
        try:
            return json.loads(line)
        except ValueError:
            continue
    return None


class Refused(Exception):
    """Outside a session, or an endpoint that is not RunPod's or Modal's: exit 2."""


class StartFailed(Exception):
    """A start that failed, or a provider never quiet: the series stops, exit 6."""


class Series:
    def __init__(self, a, prov=None, transport=None, runner=subprocess.run, clock=time.time, sleep=time.sleep,
                 environ=None):
        self.a, self.clock, self.sleep, self.runner, self.transport = a, clock, sleep, runner, transport
        self.environ = os.environ if environ is None else environ
        self.ep = servers.load(a.servers, a.server)
        if self.ep.provider not in ("runpod", "modal"):
            raise Refused("the endpoint %s is not RunPod's or Modal's" % a.server)
        self.provider = self.ep.provider
        arm_id = self.environ.get(serverspend.SESSION_ENV)
        rows = [r for r in spend.read_only(a.spend_ledger) if r.get("kind") == "arm" and r.get("arm_id") == arm_id] \
            if arm_id else []
        if not rows:
            raise Refused("start_series.py runs inside session.py run (no arm row)")
        self.arm = rows[-1]
        if self.arm.get("provider") != self.provider:
            raise Refused("the session armed %s, the endpoint %s is %s" % (self.arm.get("provider"), a.server,
                                                                           self.provider))
        if a.via == "engine" and not a.cap_usd:
            raise Refused("--via engine needs --cap-usd (the --server-cap-usd of each conversion)")
        self.t_arm = float(self.arm["t_arm"])
        self.prov = prov or pv.make(self.provider, self.arm["endpoint"], a.env_file)
        self.headers = servers.headers(self.ep, servers.read_keys(self.ep, a.env_file))
        self.log_path = self.environ.get(wl.LOG_ENV)
        if a.gpu and not self.log_path:
            raise Refused("--gpu reads the GPU in the log of the session (%s)" % wl.LOG_ENV)
        self.watch = dv.Watch(self.prov, clock, sleep, extra_ok=True, tolerate=(pv.ProviderError,))
        self.inputs = list(a.input)                 # MFL045 c2: one for every start, or one per cold start
        self.out_dir = Path(a.out) if a.out else Path(a.spend_ledger).parent / "starts" / self.arm["arm_id"]
        self.live = dv.Live()

    def since(self, t=None):
        return round((self.clock() if t is None else t) - self.t_arm, 1)

    def plan(self):
        a = self.a
        kinds = []
        for i in range(a.cold):
            kinds.append("cold")
            kinds += ["warm"] * (a.warm // a.cold + (1 if i < a.warm % a.cold else 0))
        return kinds + ["resume"] * a.resume

    def read(self):
        """One reading of the provider; a failed one is recorded and gives None. G1 raises."""
        try:
            s, _ = self.watch.read()
        except pv.ProviderError as exc:
            self.watch.read_errors.append(str(exc)[:200])
            return None
        return s

    @staticmethod
    def brief(s):
        return [(w.get("id"), w.get("status")) for w in s.get("workers") or []]

    def ahead(self, res):
        """--arm-ahead: until S seconds after the arm, the provider read every READ_EVERY_S."""
        end = self.t_arm + self.a.arm_ahead
        while True:
            s = self.read()
            if s is not None:
                n = pv.running_count(s)
                res["before_first"].append({"s": self.since(), "running": n, "workers": self.brief(s)})
                if n:
                    self.watch.fact("running_before_request", None, s=self.since(), workers=self.brief(s))
            left = end - self.clock()
            if left <= 0:
                return
            self.sleep(min(READ_EVERY_S, left))

    def stopped(self):
        """RunPod with the log: every worker that ran has a "stop container" after its last RUNNING, or
        is GONE."""
        if not self.log_path:
            return True
        lv = wl.Workers().feed(self.log_path)
        for x in lv.order:
            life = lv.by[x]
            runs = [t for t, st in life["states"] if st == "RUNNING" and t is not None]
            if runs and life["gone"] is None and (life["stop"] is None or life["stop"] < runs[-1]):
                return False
        return True

    def quiet(self):
        """The provider quiet before a cold start (module docstring): {"quiet_after_s", ...}."""
        t0, idle_since, info = self.clock(), None, {}
        while True:
            s = self.read()
            if s is not None:
                if self.provider == "modal":
                    if not (s.get("workers") or s.get("containers")):
                        break
                elif pv.running_count(s) == 0:
                    idle_since = self.clock() if idle_since is None else idle_since
                    if self.stopped():
                        break
                    if self.clock() - idle_since >= STOP_LINE_WAIT_S:
                        info["no_stop_line"] = True
                        self.watch.fact("no_stop_line", self.since(), s=self.since(), workers=self.brief(s))
                        break
                else:
                    idle_since = None
            if self.clock() - t0 >= QUIET_MAX_S:
                raise StartFailed("the provider not quiet within %d s: %s" % (QUIET_MAX_S,
                                                                             self.brief(s) if s else "no reading"))
            self.sleep(QUIET_EVERY_S)
        info["quiet_after_s"] = round(self.clock() - t0, 1)
        self.sleep(QUIET_MARGIN_S)
        return info

    def convert(self, k, kind, path):
        a = argparse.Namespace(**vars(self.a))
        a.input = path
        out = self.out_dir / ("%02d-%s" % (k, kind))
        out.parent.mkdir(parents=True, exist_ok=True)
        cmd = dv.convert_argv(Path(sys.executable).with_name("mdf"), a, out, a.server, a.servers, a.cap_usd)
        t0 = self.clock()
        try:
            r = self.runner(cmd, capture_output=True, text=True, timeout=CONVERT_TIMEOUT_S,
                            env=pv.child_env(self.environ))
        except subprocess.TimeoutExpired:
            return {"ok": False, "error": "timeout", "wall_s": round(self.clock() - t0, 1), "out": str(out)}
        row = {"exit": r.returncode, "wall_s": round(self.clock() - t0, 1), "out": str(out)}
        doc = summary_of(r.stdout)
        if isinstance(doc, dict):
            row.update(status=doc.get("status"), error=doc.get("error"))
        try:
            eng = json.loads((out / "engine.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            eng = {}
        if row.get("status") is None and eng.get("status"):
            row["status"] = eng["status"]                 # the contract of the output, when stdout said nothing
        ms, timings = eng.get("marker_server") or {}, eng.get("timings") or {}
        wait, t_wake = (ms.get("readiness") or {}).get("wait_seconds"), ms.get("t_wake")
        pages = (eng.get("pages") or {}).get("count")
        row.update(pages=pages, wake_s=_rel(t_wake, self.t_arm), ready_wait_s=wait, clean=ms.get("clean"),
                   requests=ms.get("requests"), by_status=ms.get("by_status"), total_s=timings.get("total_seconds"),
                   stages=timings.get("stages"))
        if t_wake is not None and wait is not None:
            ready_t = t_wake + wait
            row["ready_s"] = _rel(ready_t, self.t_arm)
            if ms.get("t_close") is not None:
                row["after_ready_s"] = round(ms["t_close"] - ready_t, 1)
                if pages:
                    row["s_per_page"] = round(row["after_ready_s"] / pages, 2)
        row["ok"] = r.returncode == 0 and row.get("status") == "complete"
        return row

    def page(self, path):
        inf = dv.Inference(self.ep.url, self.headers, transport=self.transport, clock=self.clock, sleep=self.sleep)
        try:
            r = inf.ready(READY_WITHIN_S[self.provider], timeout=READY_GET_S)
            row = {"ready_wait_s": r["seconds"], "tries": r["tries"], "ready_s": self.since(),
                   "proxy": r.get("proxy")}
            if not r["ready"]:
                row.update(ok=False, stopped=r.get("stopped") or "no readiness", edge=r.get("edge"))
                return row
            p = inf.page(dv.page_body(path), PAGE_TIMEOUT_S)
            row.update(page=p, pages=1, s_per_page=p.get("s"),
                       ok=p.get("status") == 200 and p.get("proxy") == "forwarded")
            return row
        finally:
            inf.close()

    def populate(self, t_send):
        """`--via populate` (c8.4): readiness from the first request, then the `mdf_compile_cache` lines of
        the populate (and, on Modal, of the commit of the Volume) in the log of the session, from the
        start (`t_send`), at most POPULATE_WAIT_S; no inference is sent (the proxy refuses them while it
        populates). With `--gpu` the populate must name that GPU (N6)."""
        inf = dv.Inference(self.ep.url, self.headers, transport=self.transport, clock=self.clock, sleep=self.sleep)
        try:
            r = inf.ready(READY_WITHIN_S[self.provider], timeout=READY_GET_S)
        finally:
            inf.close()
        row = {"ready_wait_s": r["seconds"], "tries": r["tries"], "ready_s": self.since(), "pages": 0}
        if not r["ready"]:
            row.update(ok=False, stopped=r.get("stopped") or "no readiness", edge=r.get("edge"))
            return row
        need = ("populate", "commit") if self.provider == "modal" else ("populate",)
        t0, found = self.clock(), {}
        while True:
            for e in self.cache_lines(t_send - SEND_SLACK_S):
                found.setdefault(e.get("phase"), e)
            done = "populate" in found and (not found["populate"].get("written") or all(n in found for n in need))
            if done or self.clock() - t0 >= POPULATE_WAIT_S:
                break
            self.sleep(QUIET_EVERY_S)
        pop, com = found.get("populate") or {}, found.get("commit")
        row.update(populate={k: v for k, v in pop.items() if k not in ("event", "mono")}, commit=com,
                   sha256=pop.get("sha256"), waited_s=round(self.clock() - t0, 1))
        row["ok"] = bool(pop.get("written")) and (self.provider != "modal" or bool((com or {}).get("ok")))
        if self.a.gpu and self.a.gpu not in (pop.get("gpu") or []):
            row.update(ok=False, stopped="populate on another GPU")
        return row

    def cache_lines(self, since_t=None):
        """The `mdf_compile_cache` lines of the log of the session (with `since_t`, those of the clock
        `since_t` or later)."""
        if not self.log_path:
            return []
        out = []
        try:
            text = Path(self.log_path).read_text(encoding="utf-8", errors="replace")
        except OSError:
            return out
        for raw in text.splitlines():
            e = wl.parse(raw)
            if e is not None and e["event"] is not None and e["event"].get("event") == "mdf_compile_cache":
                if since_t is None or float(e["event"].get("t") or 0) >= since_t:
                    out.append(e["event"])
        return out

    def wake_on_gpu(self):
        """`--gpu NAME` before a cold start (module docstring): the tries, each {"try", "send_s",
        "worker", "start_s", "gpu", "taken"} (and the quiet before it, from the second); with the try
        taken, `send_t` (its clock); without, `ok` false and `stopped` gpu_not_assigned."""
        a, tries, seen = self.a, [], set()
        for i in range(1, a.gpu_tries + 1):
            t = {"try": i}
            if i > 1:
                t.update(self.quiet())                   # the container on another GPU leaves first
            t_send = self.clock()
            t["send_s"] = self.since(t_send)
            found = self.woken(t_send, seen)
            if found is None:
                raise StartFailed("try %d of --gpu: no container start with its GPU in the log within %d s"
                                  % (i, GPU_WAIT_S))
            seen.add((found["worker"], found["start_s"]))
            t.update(worker=found["worker"], start_s=found["start_s"], gpu=found["gpu"],
                     taken=a.gpu in (found["gpu"] or []))
            tries.append(t)
            emit({"gpu_try": t})
            if t["taken"]:
                return {"gpu_tries": tries, "send_t": t_send}
        return {"gpu_tries": tries, "ok": False, "stopped": "gpu_not_assigned"}

    def woken(self, t_send, seen):
        """GETs of /v1/models from `t_send`, every GPU_WAKE_EVERY_S, until the log of the session has a
        container start (not in `seen`) with its GPU; None after GPU_WAIT_S."""
        inf = dv.Inference(self.ep.url, self.headers, transport=self.transport, clock=self.clock, sleep=self.sleep)
        try:
            while True:
                inf.get("/v1/models", timeout=GPU_WAKE_EVERY_S)
                for p in phases(self.log_path, self.t_arm):
                    if (p.get("gpu") and p.get("start_s") is not None and (p["worker"], p["start_s"]) not in seen
                            and p["start_s"] >= self.since(t_send) - SEND_SLACK_S):
                        return p
                if self.clock() - t_send >= GPU_WAIT_S:
                    return None
                self.sleep(GPU_WAKE_EVERY_S)
        finally:
            inf.close()

    def run(self):
        a = self.a
        res = self.live.res = {"start_series": self.arm["endpoint"], "provider": self.provider,
                               "arm_id": self.arm["arm_id"], "via": a.via, "cold": a.cold, "warm": a.warm,
                               "gpu": a.gpu, "gpu_tries": a.gpu_tries if a.gpu else None,
                               "arm_ahead_s": a.arm_ahead, "starts": [], "before_first": []}
        code, cold, resumes, last_end = 0, -1, 0, None
        try:
            if a.arm_ahead:
                self.ahead(res)
            for k, kind in enumerate(self.plan(), 1):
                cold += 1 if kind == "cold" else 0
                path = self.inputs[cold % len(self.inputs)]
                via, idle = a.via, None
                if kind == "resume":                              # MFL045 c3: after an idle of its own
                    resumes += 1
                    idle = a.idle_last if resumes == a.resume and a.idle_last is not None else a.idle
                    if last_end is not None and idle - (self.clock() - last_end) > 0:
                        self.sleep(idle - (self.clock() - last_end))
                    via = a.resume_via[resumes - 1]
                quiet = self.quiet() if kind == "cold" and k > 1 else {}
                gpu = self.wake_on_gpu() if a.gpu and kind == "cold" else {}
                t_send = gpu.pop("send_t", None) or self.clock()
                row = {"start": k, "kind": kind, "input": Path(path).name, "via": via, "send_s": self.since(t_send),
                       "left_s": round(float(self.arm["until"]) - t_send, 1)}
                if idle is not None:
                    row["idle_s"] = idle
                row.update(quiet)
                row.update(gpu)
                if gpu.get("stopped"):
                    row["pages"] = 0
                else:
                    row.update(self.convert(k, kind, path) if via == "engine" else
                               self.populate(t_send) if via == "populate" else self.page(path))
                row["end_s"] = self.since()
                last_end = self.clock()
                res["starts"].append(row)
                emit({"start_row": row})
                if not row["ok"]:
                    raise StartFailed("start %d (%s) failed: %s" % (k, kind, {x: row.get(x) for x in
                                                                              ("exit", "status", "error", "stopped")}))
        except dv.Violation as exc:
            res["violation"], res["guarantee"], code = str(exc), exc.guarantee, 6
        except StartFailed as exc:
            res["failed"], code = str(exc), 6
        res["phases"] = self.phases_of(res["starts"])
        res["running_before_request"] = any(r["running"] for r in res["before_first"])
        res.update(readings=len(self.watch.readings), read_errors=self.watch.read_errors, facts=self.watch.facts,
                   workers=list(self.watch.ids), **{"pass": code == 0})
        emit(res)
        return code

    def phases_of(self, rows):
        """The container starts of the log; each cold start carries the first one after it (a warm start
        that met a new container says so: `new_container`)."""
        if not self.log_path:
            return None
        try:
            found = phases(self.log_path, self.t_arm)
        except OSError as exc:
            return {"error": type(exc).__name__}
        used, last, prev_end = set(), None, None
        for row in rows:
            end = row.get("end_s")
            got = [i for i, p in enumerate(found) if i not in used and p.get("start_s") is not None
                   and p["start_s"] >= row["send_s"] - SEND_SLACK_S and (end is None or p["start_s"] <= end)]
            failed = [i for i in got if (found[i].get("snapshot") or {}).get("kind") == "restore_failed"]
            real = [i for i in got if i not in failed]
            if failed:                                                  # MFL045 c2: the restores that failed first
                used.update(failed)
                row["restore_failed"] = [found[i] for i in failed]
            if real:
                used.add(real[0])
                row["container" if row["kind"] == "cold" else "new_container"] = found[real[0]]
            container = row.get("container") or row.get("new_container")
            snap = (container or {}).get("snapshot")
            row["start_kind"] = snap["kind"] if snap else ("cold" if container else None)
            if snap and snap.get("fallback"):
                row["fallback"] = snap["fallback"]
            if row["kind"] == "resume":                       # MFL045 c3: the same boot with a pause, or a new start
                if container is not None:
                    row["start_kind"] = "missed"
                else:
                    pauses = [x for x in (last or {}).get("pauses") or []
                              if (prev_end is None or x["since_arm"] >= prev_end)
                              and (end is None or x["since_arm"] <= end)]
                    row.update(boot_id=(last or {}).get("boot_id"), pauses=pauses,
                               start_kind="resume" if pauses else "warm")
            last, prev_end = container or last, end
        return found


def parser():
    ap = argparse.ArgumentParser(prog="start_series.py", description=__doc__.split("\n\n")[0])
    ap.add_argument("--env-file", required=True)
    ap.add_argument("--server", required=True)
    ap.add_argument("--servers", required=True)
    ap.add_argument("--input", required=True, action="append")      # MFL045 c2: repeatable
    ap.add_argument("--cold", type=int, required=True)
    ap.add_argument("--warm", type=int, default=0)
    ap.add_argument("--arm-ahead", type=int, default=0)
    ap.add_argument("--via", choices=("engine", "page", "populate"), default="engine")
    ap.add_argument("--gpu", help="each cold start on a container of this GPU (MFL033 N6), as mdf_gpu names it")
    ap.add_argument("--gpu-tries", type=int)
    ap.add_argument("--cap-usd")
    ap.add_argument("--resume", type=int, default=0)                  # MFL045 c3: requests after an idle
    ap.add_argument("--idle", type=float)
    ap.add_argument("--idle-last", type=float)
    ap.add_argument("--resume-via")
    ap.add_argument("--spend-ledger", required=True)
    ap.add_argument("--spend-cap-usd", required=True)
    ap.add_argument("--out")
    return ap


def main(argv=None, **inject):
    a = parser().parse_args(argv)
    if a.cold < 1 or a.warm < 0 or a.arm_ahead < 0:
        emit({"code": "invalid_setting", "detail": "--cold at least 1, --warm and --arm-ahead not below 0"})
        return 2
    if a.gpu is not None and not a.gpu.strip() or a.gpu_tries is not None and (a.gpu is None or a.gpu_tries < 1):
        emit({"code": "invalid_setting", "detail": "--gpu names a GPU; --gpu-tries goes with it, at least 1"})
        return 2
    a.gpu_tries = a.gpu_tries or 1
    if len(a.input) not in (1, a.cold):
        emit({"code": "invalid_setting", "detail": "--input once, or once per cold start (%d)" % a.cold})
        return 2
    vias = a.resume_via.split(",") if a.resume_via else [a.via] * max(a.resume, 0)
    if (a.resume < 0 or (a.resume and (a.idle is None or a.idle < 0))
            or (not a.resume and (a.idle is not None or a.idle_last is not None or a.resume_via))
            or len(vias) != a.resume or any(v not in ("engine", "page") for v in vias)):
        emit({"code": "invalid_setting", "detail": "--resume N with --idle S, and N ways in --resume-via (engine, page)"})
        return 2
    a.resume_via = vias
    try:
        s = Series(a, **inject)
    except Refused as exc:
        emit({"code": "invalid_setting", "detail": str(exc)})
        return 2
    except EngineError as exc:
        emit({"code": exc.code.value, "detail": exc.detail})
        return 2
    except pv.ProviderError as exc:
        emit({"code": "provider_error", "detail": str(exc)})
        return 4
    try:
        return dv.run_driver(s.run, s.live, emit)          # SIGINT and SIGTERM: the result so far
    except pv.ProviderError as exc:
        emit({"code": "provider_error", "detail": str(exc)})
        return 4


if __name__ == "__main__":
    sys.exit(main())
