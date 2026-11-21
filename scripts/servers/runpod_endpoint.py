#!/usr/bin/env python3
"""RunPod endpoints of the marker-server route (MFL033; plan r2 §5.2, r3 §5.2, r4 §5.4, App. D and G).

    runpod_endpoint.py --env-file F catalog
    runpod_endpoint.py --env-file F create --name N --data-center DC [--dry-run]
    runpod_endpoint.py --env-file F align --name N
    runpod_endpoint.py --env-file F verify-pins --name N
    runpod_endpoint.py --env-file F {populate,probe-negative,probe-valid} --name N      (inside session.py)
    runpod_endpoint.py --env-file F fault-test --name N --phase {A,B,D} [--observe-seconds S]  (inside session.py)
    runpod_endpoint.py --env-file F concurrency --name N --input PNG [--n 32] [--warmup 1] [--abandon 1]
                       [--probe-while-busy]                                             (inside session.py)
    runpod_endpoint.py --env-file F observe --name N --closed-utc-hours H (--read-only | --spend-ledger L
                       --spend-cap-usd C --cap-usd X)
    runpod_endpoint.py --env-file F fault-e --name N --input FILE --spend-ledger L --spend-cap-usd C --cap-usd X
    runpod_endpoint.py --env-file F read-history --name N --since ISO
    runpod_endpoint.py --env-file F delete --name N

REST v2 only (https://api.runpod.io/v2), GraphQL only for the balance; the key (`RUNPOD_API_KEY`)
comes from the env file and goes only into the Authorization header. Secure Cloud: the REST v2 has
no cloud field for serverless endpoints (plan r1 point 2); the endpoint is limited to the data
center of its network volume (`dataCenterIds`), as approved on 17/11. Every reading saves the
whole summary. Exit 0: done or passed; 4: a provider error; 6: a criterion violated (stop for
Davide: the caller disarms; since c7.20 a guarantee broken, and a worker on a disarmed endpoint is
disarmed again, never deleted); 8 (c7.20): inconclusive, the fault of the phase not exercised.

`align` brings a disarmed, idle endpoint to the pins of `create` that can change in place: the CUDA
floor, the `workers` block at max 0, the base env with the current watchdog (c7.4). `populate`
waits for readiness up to 1800 s from the arm when the session armed with `--allow-download`
(else 960 s), reads the endpoint every 30 s (`total > 1`, a worker outside the data center) and the
worker log of the session (`SESSION_WORKER_LOG`) for the fast stop of `worker_logs.StartRules`.
Every other readiness wait (`probe-valid`, phases A and B, `hold`) reads the same log until
readiness and gives the readiness from the arm too (c7.5, prompt N4d §5.3 and §5.5); since c7.7
(prompt N4f §3.4) it also reads the workers every 30 s, with GETs of 30 s, and stops at once on
`total > 1`, a worker unhealthy, replaced or outside the data center; `probe-valid` keeps reading
them for 30 s after the chat.
`probe-negative` (c7.6, prompt N4e §4.1): the arm alone starts a worker on a load-balanced
endpoint (seen on 19/11), so it passes with 401 of the edge for both probes, no request answered
by the proxy in the worker log, and at most one worker in the data center; the session checks the
zero after the disarm. Every reading of workers in `populate`, `probe-negative` and phases A and B
holds the data center of the endpoint (N4b §2).
Since c7.13 every readiness wait, the populate's included, stops at the first refusal of the edge
(401, 403 or 404 without `X-MDF-Proxy`), each attempt is a line (`ready_try`) as soon as it
happens, and SIGINT or SIGTERM print the result so far (`stopped_by`, `partial`) before the exit 130.

Since c7.19 (prompt N4s §4.1, road S1 of Davide, 20/11) the drivers that reach readiness
(`probe-valid`, phases A and B, `hold`) admit RunPod's extra workers: on 19/11 a second worker was
listed next to the one that ran, `INITIALIZING`, never `RUNNING`, never billed. Only the workers
RUNNING count (`drivers.Watch` with `extra_ok`), and the workers of the log of the session
(`worker_logs.Workers`, listed every 10 s) are judged at every check, as on Modal since c7.17:
- two workers RUNNING at once: always a violation;
- a worker never RUNNING besides the one that runs: an extra worker, declared at once
  (`extra_worker`) and in the result, admitted only `INITIALIZING`, `IDLE` or `THROTTLED`;
- a worker RUNNING after another one, judged once by the moment it was first RUNNING: before
  readiness rule (a) of Davide (the one before had no answer of vLLM and no inference line:
  `replacement`, a fact); after the crash of phase B a restart that the phase observes
  (`restart_after_fault`); after the hang of phase A a restart after the hang; otherwise a
  replacement after readiness. Both violations.
`populate` and `probe-negative` keep their criteria (`total > 1`); phase D and `observe` work on a
disarmed endpoint, where every worker is a violation.

Since c7.20 (prompt of the closure of N4 on RunPod: Davide, 20/11 19:44 and 19:54) phases A, B, D and E
fail only on a guarantee (`drivers` module docstring), exit 6 with `guarantee`:
- G1: two workers RUNNING at once; in A, no zero Z1 within A_ZERO_S of the hang (no worker RUNNING,
  and the container of each worker that ran stopped: a "stop container" line in the log of the
  session, a GONE, or no longer listed; the workers left IDLE, THROTTLED or INITIALIZING without a
  container are no obstacle); in B, no zero Z1 by the end of the observation; a worker RUNNING after
  the zero (in the readings or in the log); a worker on a disarmed endpoint (D, `observe`); no zero
  after the disarm of E;
- G2: a request sent without an outcome (the open and the abandoned requests of A, the three of B, the
  request of D); in E, `session.py run` or `mdf convert` that does not refuse with `session_orphaned`.
Everything else is a fact, a `fact` line at once and in the result (`facts`): an extra worker in any
state, a worker unhealthy or outside the data center, more workers listed than 1 + 2, a replacement
before readiness (rule (a), or outside it) or after it, a restart after the fault before the zero, an
open request that is not 503 `mdf_tripped`, the 500 after the crash of B, a failed reading, the wrapped
command of E alive after the SIGKILL (then killed). A phase whose fault could not be exercised (no
readiness, the first request failed, the session of E ended before its request) is inconclusive,
exit 8. A worker on a disarmed endpoint no longer deletes it (deleting is irreversible: Davide's
decision); the driver disarms again and stops.

`concurrency` (MFL033 N5b, prompt §4.4) is the driver of Modal (`modal_tasks.Tasks.concurrency`, c7.16)
on RunPod: readiness with the workers and the log as `probe-valid`, `--warmup` pages one after the other,
the `--abandon` pages first (their client gives up after ABANDON_AFTER_S while vLLM runs them), then `--n`
pages at once with `/metrics` read every half second in a thread of its own (`drivers.Sampler`, N6: the
loop that waited for each probe first missed the peak) and `/v1/models` with `--probe-while-busy`, then
the workers and `/metrics` for AFTER_LOAD_WATCH_S. The criteria of App. D (r3 §5.3: at least 24 running,
every page 200, the load within 4 times the warm page, every probe 200, the abandoned pages abandoned)
are a measure, in `criteria` and `measure_pass`; it fails only on a guarantee (G1 in the readings and
the log, G2: a page without an outcome), exit 6, and is inconclusive without readiness, exit 8.
"""
import argparse
from datetime import datetime, timezone
from decimal import Decimal
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import drivers as dv  # noqa: E402
import providers as pv  # noqa: E402
import worker_logs as wl  # noqa: E402

from markdown_for_llms.bench import serverspend, spend  # noqa: E402

READY_WITHIN_S = 960            # W1 900 s plus the margin of RUNPOD_INIT_TIMEOUT
READY_POPULATE_S = 1800         # the populate with the download: readiness from the arm (prompt N4c §3)
POPULATE_GET_S = 30             # each readiness GET of the populate: the log checked at least that often
POPULATE_READ_EVERY_S = 30
READY_GET_S = 30                # each readiness GET of the other waits: the log and the workers as often (c7.7)
READY_READ_EVERY_S = 30         # a reading of the workers in every readiness wait (prompt N4f §3.4)
AFTER_READY_WATCH_S = 30        # probe-valid reads the workers so long after the chat (19/11: +1.3 s)
A_ZERO_S = 215                  # phase A: zero within 215 s of the hang, plus a reading (r4 §5.4)
D_SECONDS = 120
UNKNOWN_S = 15.0                # c7.19: a worker RUNNING in the readings, not in the log of the session after so long
REPLACEMENTS = ("a", "outside_a", "after_readiness")   # c7.20: the kinds of replacement, all facts
HOLD_STEP_S = 5.0               # c7.19: phase B judges the log so often while its requests are held
# concurrency (MFL033 N5b), the values of modal_tasks.py (c7.16)
PAGE_TIMEOUT_S = 150.0          # one page: the `read_timeout` of RunPod's endpoints is 300 s, vLLM answers in seconds
ABANDON_AFTER_S = 5.0
ABANDON_RUNNING_S = 4.0
ABANDON_POLL_S = 0.5
AFTER_LOAD_WATCH_S = 30
AFTER_LOAD_EVERY_S = 10
RUNNING_MIN = 24                # App. D, r3 §5.3: vllm:num_requests_running at least so many during the load


def out(obj):
    pv.emit(obj)


def iso(t):
    return datetime.fromtimestamp(t, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class Driver:
    def __init__(self, a, rp=None, clock=time.time, sleep=time.sleep, transport=None, environ=None):
        self.a, self.clock, self.sleep = a, clock, sleep
        self.environ = os.environ if environ is None else environ
        self.key = pv.read_key(a.env_file, "RUNPOD_API_KEY")
        self.rp = rp or pv.RunPod(self.key, getattr(a, "name", None), transport=transport, clock=clock, sleep=sleep)
        self.transport = transport
        self.live = dv.Live()                   # the result so far, printed on a signal (c7.13)
        # c7.19: the workers of the log of the session, rule (a) and the extra workers
        self.log_path = self.environ.get(wl.LOG_ENV)
        self.workers_log = wl.Workers() if self.log_path else None
        self._t_arm = None
        self.t_ready = None                     # when the driver saw readiness (clock)
        self.fault = None                       # ("A" | "B", clock): from then on the phase judges new workers
        self.judged = {}                        # (old, new) -> "a" (rule (a)) or "restart" (after the crash of B)
        self.outside = {}                       # c7.20: (old, new) -> why a replacement is outside rule (a)
        self.extras = []                        # the extra workers declared from the log
        self.facts, self._fact_keys = [], set()  # c7.20: what stops nothing, in the order it came
        if self.workers_log is not None:
            self.live.extra["workers_log"] = self.workers_out

    def inference(self):
        self.live.inf = dv.Inference(self.rp.url, {"Authorization": "Bearer " + self.key.reveal()},
                                     transport=self.transport, clock=self.clock, sleep=self.sleep, log=out)
        return self.live.inf

    # --- read-only ----------------------------------------------------------------------------
    def catalog(self):
        dcs = self.rp.call("GET", "/catalog/datacenters").get("dataCenters") or []
        gpus = self.rp.call("GET", "/catalog/gpus").get("gpus") or []
        pool = [g for g in gpus if g.get("pool") == pv.POOL]
        out({"pool": pv.POOL, "gpus": [{k: g.get(k) for k in ("id", "memory", "secure", "price", "dataCenters")}
                                       for g in pool],
             "data_centers": [{k: d.get(k) for k in ("id", "region", "networkVolumeTypes", "compliance")}
                              for d in dcs if d.get("networkVolumeTypes")],
             "balance_usd": self.rp.balance()})
        return 0

    def pool_price(self):
        gpus = [g for g in self.rp.call("GET", "/catalog/gpus").get("gpus") or [] if g.get("pool") == pv.POOL]
        prices = [Decimal(str((g.get("price") or {}).get("serverless"))) for g in gpus
                  if (g.get("price") or {}).get("serverless") is not None]
        return gpus, (max(prices) if prices else None)

    def verify(self):
        """The pins of `providers.pin_checks`, the same function as every arm (MFL033 S-B2); `outside_align`
        names those that only a new endpoint can fix."""
        ep = self.rp.endpoint(refresh=True)
        checks, facts = self.rp.pins(ep)
        gpus, price = self.pool_price()
        report = {"endpoint": ep.get("id"), "url": self.rp.url, "checks": checks,
                  "outside_align": pv.invariant_failures(checks), "gpu": ep.get("gpu"),
                  "data_centers": ep.get("dataCenterIds"), "pool_price_raw": str(price),
                  "env_bytes": len(json.dumps(ep.get("env") or {})), "balance_usd": self.rp.balance()}
        out(report)
        return 0 if all(checks.values()) else 6

    # --- create and delete ----------------------------------------------------------------------
    def create(self):
        a = self.a
        if self.rp.exists(a.name):
            out({"create": a.name, "exists": True})
            return 0
        vols = [v for v in self.rp.volumes() if v.get("name") == pv.VOLUME]
        if len(vols) > 1:
            raise pv.ProviderError("two volumes named " + pv.VOLUME)
        if vols and vols[0].get("dataCenter") != a.data_center:
            raise pv.ProviderError("the volume %s is in %s, not %s" % (pv.VOLUME, vols[0].get("dataCenter"), a.data_center))
        vol_body = {"name": pv.VOLUME, "size": pv.VOLUME_GB, "dataCenter": a.data_center}
        if a.dry_run:
            out({"dry_run": True, "volume": vols[0] if vols else vol_body,
                 "endpoint": dict(pv.create_body(a.name, "<volume>", a.data_center), env="<base env>")})
            return 0
        vol = vols[0] if vols else self.rp.call("POST", "/network-volumes", body=vol_body)
        ep = self.rp.call("POST", "/serverless", body=pv.create_body(a.name, vol["id"], a.data_center))
        out({"create": a.name, "endpoint": ep.get("id"), "volume": vol.get("id"), "data_center": a.data_center,
             "workers": ep.get("workers")})
        return 0

    def align(self):
        """A disarmed, idle endpoint to the pins that change in place (no worker can start: max 0)."""
        s = self.rp.state()
        if s["armed"] or s["busy"]:
            raise pv.ProviderError("align: %s is armed or busy" % self.a.name)
        ep = self.rp.endpoint(refresh=True)
        body = {}
        if not pv.gpu_ok(ep.get("gpu")):
            body["gpu"] = {"minCudaVersion": pv.CUDA_MIN}
        if ep.get("workers") != pv.workers_block(0):
            body["workers"] = pv.workers_block(0)
        if ep.get("env") != pv.base_env():
            body["env"] = pv.base_env()
        if body:
            self.rp.patch(body)
        ep = self.rp.endpoint(refresh=True)
        ok = pv.gpu_ok(ep.get("gpu")) and ep.get("workers") == pv.workers_block(0) and ep.get("env") == pv.base_env()
        out({"align": self.a.name, "patched": sorted(body), "gpu": ep.get("gpu"), "workers": ep.get("workers"),
             "env_bytes": len(json.dumps(ep.get("env") or {})), "ok": ok})
        return 0 if ok else 6

    def delete(self):
        s = self.rp.state()
        if s["armed"]:
            self.rp.disarm()
        self.rp.call("DELETE", "/serverless/" + self.rp.endpoint()["id"])
        out({"delete": self.a.name, "state_before": s})
        return 0

    # --- inside a session -----------------------------------------------------------------------
    def arm_row(self):
        """The `arm` row of the session that runs this driver (MDF_SESSION_ARM in its ledger)."""
        arm_id = self.environ.get(serverspend.SESSION_ENV)
        ledger = getattr(self.a, "spend_ledger", None)    # `hold` has none (fault-e passes its own argv)
        if not arm_id or not ledger:
            return None
        rows = [r for r in spend.read_only(ledger) if r.get("kind") == "arm" and r.get("arm_id") == arm_id]
        return rows[-1] if rows else None

    def ready(self, inf, w=None):
        """Readiness within READY_WITHIN_S of the driver, with GETs of READY_GET_S. Before each GET,
        until readiness, the fast stop: the worker log of the session (an error of vLLM, a trip, the
        inert mode, an error of the proxy: prompt N4d §5.5) and, every READY_READ_EVERY_S, the workers
        of the endpoint on `w` (the caller's, so its later readings see the same workers): `total > 1`,
        a worker unhealthy, replaced or outside the data center (prompt N4f §3.4, c7.7); a failed
        reading is not a criterion. The readiness from the arm too (N4d §5.3)."""
        path = self.environ.get(wl.LOG_ENV)
        rules = wl.StartRules() if path else None
        w = self.watch(s1=True) if w is None else w
        last, read_errors, broken = [None], [], {}

        def check():
            if rules is not None:
                reason = rules.feed(path)
                if reason:
                    return reason
            if last[0] is None or self.clock() - last[0] >= READY_READ_EVERY_S:
                last[0] = self.clock()
                try:
                    w.read(no_unhealthy=True, no_restart=True)
                except pv.ProviderError as exc:
                    read_errors.append(str(exc)[:200])
                except dv.Violation as exc:
                    broken["guarantee"] = exc.guarantee
                    return str(exc)
            if w.extra_ok:
                try:
                    self.log_check()                     # every check, not only with the readings (c7.19)
                except dv.Violation as exc:
                    broken["guarantee"] = exc.guarantee
                    return str(exc)
            return None

        r = inf.ready(READY_WITHIN_S, check=check, timeout=READY_GET_S)
        if not r["ready"] and broken.get("guarantee"):
            r["guarantee"] = broken["guarantee"]         # c7.20: the stop broke a guarantee
        if r["ready"]:
            self.t_ready = self.clock()
            if w.extra_ok:
                try:
                    self.log_check()                     # the workers up to readiness, judged once more
                except dv.Violation as exc:
                    r.update(ready=False, stopped=str(exc), guarantee=exc.guarantee)
        arm = self.arm_row()
        if arm is not None:
            r["since_arm_s"] = round(self.clock() - float(arm["t_arm"]), 1)
        if rules is not None:
            r["proxy_events"] = rules.events
        r.update(readings=len(w.readings), workers=list(w.ids), read_errors=read_errors)
        if w.extra_ok:
            r["served"] = self.served()
            r.update(self.workers_out())
        return r

    def populate(self):
        arm = self.arm_row()
        if arm is None:
            out({"code": "invalid_setting", "detail": "populate runs inside session.py run (no arm row)"})
            return 2
        download = bool(arm.get("allow_download"))
        start = self.clock()
        deadline = float(arm["t_arm"]) + READY_POPULATE_S if download else start + READY_WITHIN_S
        rules = wl.StartRules()
        log_path = self.environ.get(wl.LOG_ENV)
        inf = self.inference()
        w = self.watch()
        res = self.live.res = {"populate": self.a.name, "allow_download": download,
                               "deadline_from_arm_s": round(deadline - float(arm["t_arm"]), 1), "worker_log": bool(log_path)}
        tries, last_read, ok, read_errors = inf.start_wait(), None, False, []
        try:
            while True:
                if rules is not None and log_path:
                    reason = rules.feed(log_path)
                    if reason:
                        raise dv.Violation(reason)
                if last_read is None or self.clock() - last_read >= POPULATE_READ_EVERY_S:
                    last_read = self.clock()
                    try:
                        w.read()
                    except pv.ProviderError as exc:        # a failed reading is not a criterion
                        read_errors.append(str(exc)[:200])
                left = deadline - self.clock()
                if left <= 0:
                    raise dv.Violation("not ready within %d s of the arm" % round(deadline - float(arm["t_arm"])))
                r = inf.attempt(timeout=min(POPULATE_GET_S, left) if download else min(150.0, left))
                if r.get("edge"):                          # a refusal of the edge: no retry passes it (c7.13)
                    res["edge"] = r["edge"]
                    raise dv.Violation("refused at the edge: HTTP %d" % r["status"])
                body = r.get("body")
                ids = [m.get("id") for m in body.get("data", [])] if isinstance(body, dict) else []
                if r.get("status") == 200 and dv.MODEL in ids:
                    res["ready"] = {"ready": True, "since_arm_s": round(self.clock() - float(arm["t_arm"]), 1),
                                    "since_driver_s": round(self.clock() - start, 1), "proxy": r.get("proxy")}
                    ok = True
                    break
                self.sleep(5)
        except dv.Violation as exc:
            res["violation"] = str(exc)
            res["ready"] = {"ready": False, "since_arm_s": round(self.clock() - float(arm["t_arm"]), 1)}
        res.update(tries=len(tries), last=tries[-3:], readings=len(w.readings), last_reading=w.readings[-1:],
                   read_errors=read_errors)
        if rules is not None:
            res["proxy_events"] = rules.events
        out(res)
        return 0 if ok else 6

    def watch(self, s1=False):
        """The readings of a driver; a worker outside the data center of the endpoint is a criterion
        violated (N4b §2). With `s1` (c7.19) the extra workers are admitted and the log judged; since
        c7.20 the criteria that break no guarantee are facts, and a failed reading is one too."""
        dcs = self.rp.endpoint().get("dataCenterIds")
        if s1:
            self.live.watch = dv.Watch(self.rp, self.clock, self.sleep, data_centers=dcs, extra_ok=True,
                                       judge=self.judge_reading, after_read=self.log_check, on_fact=self.fact_row,
                                       tolerate=(pv.ProviderError,))
        else:
            self.live.watch = dv.Watch(self.rp, self.clock, self.sleep, data_centers=dcs)
        return self.live.watch

    # --- facts (c7.20) ----------------------------------------------------------------------------
    def fact(self, kind, key=None, **detail):
        """Something that breaks no guarantee: a `fact` line at once and in the result, once per key."""
        k = (kind, key)
        if k in self._fact_keys:
            return
        self._fact_keys.add(k)
        self.fact_row(dict(detail, fact=kind))

    def fact_row(self, row):
        self.facts.append(row)
        out(row)

    @staticmethod
    def outcomes(*named):
        """G2: every request sent has a declared outcome (a status, a timeout, a refusal)."""
        for name, r in named:
            if not isinstance(r, dict) or not (r.get("status") is not None or r.get("error")):
                raise dv.Violation("%s has no outcome: %s" % (name, r), "G2")

    @staticmethod
    def not_ready(r):
        """A phase that did not reach readiness: a guarantee broken on the way (G1), else inconclusive."""
        if r["ready"]:
            return
        if r.get("guarantee"):
            raise dv.Violation(r.get("stopped"), r["guarantee"])
        raise dv.Inconclusive("not ready: %s" % (r.get("stopped") or "no readiness within %d s" % READY_WITHIN_S))

    def read_quiet(self, w, **criteria):
        """A reading of an armed phase; a failed one is recorded (c7.20: a fact, not a stop)."""
        try:
            return w.read(**criteria)
        except pv.ProviderError as exc:
            w.read_errors.append(str(exc)[:200])
            self.fact("read_error", len(w.read_errors), detail=str(exc)[:200], t=round(self.clock(), 3))
            return None

    def settled(self, listed):
        """The zero Z1 of an armed RunPod endpoint (Davide, 20/11 19:44), the part of the log: no worker
        RUNNING in the log of the session, and each worker that ran (in the log or in the readings) has
        its container stopped: a "stop container" line after it was first RUNNING, a GONE, or it is no
        longer listed (`listed`, the ids of the reading). None when settled, else the reason."""
        w = self.live.watch
        ran = set(w.running_ids) if w is not None else set()
        if self.workers_log is None:
            left = sorted(x for x in ran if x in listed)
            return None if not left else "the worker that ran %s is still listed, and no log of the session" % left
        lv = self.workers_log.feed(self.log_path)
        now = sorted(x for x, st in lv.status.items() if st == "RUNNING")
        if now:
            return "RUNNING in the log of the session: %s" % now
        ran |= {x for x in lv.order if lv.by[x]["running"] is not None}
        open_ = []
        for x in sorted(ran):
            life = lv.by.get(x)
            if x not in listed or (life is not None and life["gone"] is not None):
                continue
            if life is not None and life["stop"] is not None and (life["running"] is None
                                                                  or life["stop"] >= life["running"]):
                continue
            open_.append(x)
        if open_:
            return "no \"stop container\" in the log for %s, still listed" % open_
        return None

    # --- the workers of the log of the session (c7.19) ------------------------------------------
    def t_arm(self):
        """The arm of the window: its row in the ledger, else the follower's first line; or None."""
        if self._t_arm is None:
            arm = self.arm_row()
            if arm is not None:
                self._t_arm = float(arm["t_arm"])
            elif self.workers_log is not None and self.workers_log.t_arm is not None:
                self._t_arm = self.workers_log.t_arm
        return self._t_arm

    def judge_reading(self, old, new, t):
        """A replacement in the readings (`Watch.judge`): the log judges it (`log_check`); without the
        log rule (a) cannot judge it: a fact since c7.20 (two RUNNING at once, and a RUNNING after the
        zero, are judged by the readings alone)."""
        if self.workers_log is None:
            self.fact("replacement_not_judged", (old, new), old=old, new=new, t=t, why="no log of the session")
            return
        self.log_check()

    def first_running(self):
        """{worker: seconds since the arm when it was first RUNNING}, from the log and the readings."""
        lv, base, w = self.workers_log, self.t_arm(), self.live.watch
        first = {x: lv.by[x]["running"] for x in lv.order if lv.by[x]["running"] is not None}
        if w is not None and base is not None:
            for x, seen in w.seen.items():
                if seen.get("running_t") is not None:
                    r = round(seen["running_t"] - base, 1)
                    first[x] = min(first.get(x, r), r)
        return first

    def log_check(self):
        """The workers of the log of the session, read again and judged (c7.19, prompt N4s §4.1; the
        module docstring). Since c7.20 it raises `dv.Violation` only on G1: two workers RUNNING in one
        listing, or a worker listed RUNNING after the zero of the phase. A replacement is a fact,
        declared once by its kind: `rule` "a" before readiness without an answer of vLLM nor an
        inference on the worker replaced (Davide's rule (a)), "outside_a" before readiness otherwise,
        "after_readiness" before the fault; after the fault a restart (`restart_after_fault`); a
        worker RUNNING in the readings that the log does not know after UNKNOWN_S, a fact too."""
        if self.workers_log is None:
            return
        lv = self.workers_log.feed(self.log_path)
        if lv.overlaps:
            raise dv.Violation("two workers RUNNING at once: %s" % "; ".join(
                "%s in the listing at +%.1f s" % (" and ".join(o["workers"]), o["since_arm"]) for o in lv.overlaps[:3]),
                "G1")
        base = self.t_arm()
        w = self.live.watch
        if w is not None and w.zero_t is not None and base is not None:
            zero_s = w.zero_t - base
            for x in lv.order:
                late = [t for t, st in lv.by[x]["states"] if st == "RUNNING" and t is not None and t > zero_s]
                if late:
                    raise dv.Violation("a worker RUNNING after the zero of the phase, without requests: %s listed "
                                       "RUNNING at +%.1f s, the zero at +%.1f s" % (x, late[0], zero_s), "G1")
        first = self.first_running()
        order = sorted(first, key=lambda x: first[x])
        now = None if base is None else self.clock() - base
        ready_s = None if self.t_ready is None or base is None else self.t_ready - base
        fault_s = None if self.fault is None or base is None else self.fault[1] - base
        for old, new in zip(order, order[1:]):
            if (old, new) in self.judged:
                continue
            if base is None:
                self.judged[(old, new)] = "not_judged"
                self.fact("replacement_not_judged", (old, new), old=old, new=new, why="no time of the arm")
                continue
            if old not in lv.by or new not in lv.by:
                if now - first[new] < UNKNOWN_S:
                    continue                           # the next listing of the follower names it
                self.judged[(old, new)] = "not_judged"
                self.fact("replacement_not_judged", (old, new), old=old, new=new, running_s=first[new],
                          why="RUNNING in the readings, not in the log of the session after %d s" % UNKNOWN_S)
                continue
            row = self.pair_row(old, new, first)
            if ready_s is None or first[new] < ready_s:          # before readiness: rule (a)
                o = lv.by[old]
                outside = []
                if o["vllm"] is not None:
                    outside.append("after the first answer of vLLM (+%.1f s, %s)" % (o["vllm"], o["vllm_by"]))
                if o["inferences"]:
                    outside.append("with an inference on it (%d lines)" % o["inferences"])
                kind = "outside_a" if outside else "a"
                self.judged[(old, new)] = kind
                self.outside[(old, new)] = outside
                out(dict({"replacement": sum(1 for k in self.judged.values() if k in REPLACEMENTS),
                          "rule": kind, "outside_rule_a": outside}, **row))
            elif fault_s is not None and first[new] >= fault_s:
                self.judged[(old, new)] = "restart"
                out(dict({"restart_after_fault": sum(1 for k in self.judged.values() if k == "restart"),
                          "fault": self.fault[0]}, **row))
            else:
                self.judged[(old, new)] = "after_readiness"
                out(dict({"replacement": sum(1 for k in self.judged.values() if k in REPLACEMENTS),
                          "rule": "after_readiness"}, **row))
        primary = order[-1] if order else (lv.order[0] if lv.order else None)
        for x in lv.order:
            if x == primary or x in first:
                continue
            bad = [(t, st) for t, st in lv.by[x]["states"] if st not in wl.EXTRA_STATES + ("GONE",)]
            if bad:
                self.fact("extra_worker_state", (x, bad[0][1]), worker=x, status=bad[0][1], since_arm=bad[0][0],
                          admitted=list(wl.EXTRA_STATES))
            if x not in self.extras:
                self.extras.append(x)
                out(dict({"extra_worker": len(self.extras)}, **lv.brief(x)))

    def pair_row(self, old, new, first):
        """A worker RUNNING after another one: each with its times since the arm, GPU and data center."""
        lv = self.workers_log

        def side(x):
            b = lv.brief(x)
            return {"id": x, "seen_s": b["seen_s"], "running_s": first.get(x), "gone_s": b["gone_s"],
                    "last_s": b["last_s"], "stop_s": b["stop_s"], "gpu": b["gpu"], "data_center": b["data_center"]}
        o, n = side(old), side(new)
        return {"old": o, "new": n, "old_vllm_s": lv.by[old]["vllm"], "old_inference_lines": lv.by[old]["inferences"]}

    def served(self):
        """The worker RUNNING at readiness that answered through its proxy, else the last RUNNING."""
        if self.workers_log is None:
            w = self.live.watch
            return (w.running_ids or [None])[-1] if w is not None else None
        first = self.first_running()
        order = sorted(first, key=lambda x: first[x])
        answered = [x for x in order if x in self.workers_log.by and self.workers_log.by[x]["vllm"] is not None]
        return (answered or order or [None])[-1]

    def workers_out(self):
        """The workers for a result: the log's, the extra workers (log and readings), the replacements
        of rule (a) and the restarts after the crash of phase B."""
        w = self.live.watch
        res = {"extra_workers": [], "extra_workers_readings": w.extra_workers() if w is not None and w.extra_ok else [],
               "replacements": [], "restarts_after_fault": [], "facts": list(self.facts)}
        if self.workers_log is None:
            return res
        lv = self.workers_log
        first = self.first_running()
        # c7.21: a worker declared extra while none ran, that ran afterwards, is not one (20/11, phase B)
        res["extra_workers"] = [lv.brief(x) for x in self.extras if x in lv.by and x not in first]
        for (old, new), kind in self.judged.items():
            if kind in REPLACEMENTS:
                res["replacements"].append(dict(self.pair_row(old, new, first), rule=kind,
                                                outside_rule_a=self.outside.get((old, new), [])))
            elif kind == "restart":
                res["restarts_after_fault"].append(self.pair_row(old, new, first))
        res["workers_log"] = lv.summary()
        return res

    def join_watching(self, thread, timeout):
        """Join `thread` within `timeout` s, judging the log every HOLD_STEP_S (c7.19, phase B)."""
        end = time.monotonic() + timeout
        while thread.is_alive() and time.monotonic() < end:
            thread.join(min(HOLD_STEP_S, max(0.0, end - time.monotonic())))
            self.log_check()

    def probe_negative(self):
        """Two probes without a valid key on the armed endpoint (prompt N4e §4.1, c7.6). Passes when:
        (1) both get 401 from the edge, without `X-MDF-Proxy`; (2) the worker log of the session shows
        no request answered by the proxy (`mdf_request` `forwarded` or `error`) up to the end of the
        observation; (3) at most one worker for 120 s, in the data center of the endpoint, never
        unhealthy nor replaced. The arm alone starts a worker on this load-balanced endpoint (seen on
        19/11, window 20261119T110021Z-0b5aa77b, no request at all): its return to zero after the
        disarm is the session's (no zero within 600 s: exit 4, the window stays orphan). The first
        violation stops at once."""
        inf = self.inference()
        res = {}
        with dv.httpx.Client(timeout=dv.httpx.Timeout(30.0, connect=10.0), transport=self.transport) as anon:
            res["no_key"] = inf.get("/v1/models", client=anon, timeout=30.0)
            res["wrong_key"] = inf.get("/v1/models", client=anon, timeout=30.0,
                                       headers={"Authorization": "Bearer mdf-wrong-key"})
        probes = {k: {x: v.get(x) for x in ("status", "proxy", "error")} for k, v in res.items()}
        self.live.res = {"probe_negative": probes}
        path = self.environ.get(wl.LOG_ENV)
        requests = wl.RequestLines()
        w = self.watch()
        violation = None
        try:
            not_edge = {k: v for k, v in probes.items() if not (v["status"] == 401 and v["proxy"] is None)}
            if not_edge:
                raise dv.Violation("not a 401 of the edge: %s" % dv.dump(not_edge))
            if not path:
                raise dv.Violation("no worker log of the session (%s): the proxy cannot be checked" % wl.LOG_ENV)
            t0 = self.clock()
            while True:
                w.read(no_unhealthy=True, no_restart=True)
                self.no_request(requests, path)
                if self.clock() - t0 >= D_SECONDS:
                    break
                self.sleep(10)
        except dv.Violation as exc:
            violation = str(exc)
        counts = [max((r["state"].get("summary") or {}).get("total", 0), len(r["state"].get("workers") or []))
                  for r in w.readings]
        out({"probe_negative": probes, "violation": violation, "workers": w.ids, "max_total": max(counts, default=0),
             "proxy_requests": requests.found, "readings": w.readings})
        return 0 if violation is None else 6

    @staticmethod
    def no_request(requests, path):
        """Condition (2) of `probe-negative`: no request answered by the proxy in the worker log."""
        try:
            found = requests.feed(path)
        except OSError as exc:
            raise dv.Violation("the worker log cannot be read: %s" % type(exc).__name__) from None
        if found:
            raise dv.Violation("a request answered by the proxy: %s" % dv.dump(found[0]))

    def probe_valid(self):
        """Readiness (with the workers, c7.7), the version, a chat answered by vLLM; then the workers
        for AFTER_READY_WATCH_S more with the same criteria. `/ping` through the edge is recorded, not
        required (the balancer may answer it itself)."""
        inf, w = self.inference(), self.watch(s1=True)
        res = self.live.res = {}
        r = res["ready"] = self.ready(inf, w)
        violation, read_errors = r.get("stopped"), []
        if r["ready"]:
            res["version"] = inf.get("/version")
            res["metrics"] = {k: v for k, v in inf.get("/metrics").items() if k != "body"}
            res["health"] = {k: v for k, v in inf.get("/health").items() if k != "body"}
            res["ping"] = {k: v for k, v in inf.get("/ping", timeout=30.0).items() if k != "body"}
            res["chat"] = inf.chat()
            t0 = self.clock()
            try:
                while self.clock() - t0 < AFTER_READY_WATCH_S:
                    try:
                        w.read(no_unhealthy=True, no_restart=True)
                    except pv.ProviderError as exc:          # a failed reading is not a criterion
                        read_errors.append(str(exc)[:200])
                    self.sleep(10)
            except dv.Violation as exc:
                violation = str(exc)
        ok = (r["ready"] and violation is None and (res["version"].get("body") or {}).get("version") == dv.VLLM_VERSION
              and res["chat"].get("status") == 200 and res["chat"].get("proxy") == "forwarded")
        counts = [max((x["state"].get("summary") or {}).get("total", 0), len(x["state"].get("workers") or []))
                  for x in w.readings]
        out(dict({"probe_valid": res, "violation": violation, "workers": w.ids, "max_total": max(counts, default=0),
                  "max_running": max((pv.running_count(x["state"]) for x in w.readings), default=0),
                  "readings": len(w.readings), "read_errors_after": read_errors, "skew_s": inf.skew, "pass": bool(ok)},
                 **self.workers_out()))
        return 0 if ok else 6

    # --- concurrency (MFL033 N5b) -----------------------------------------------------------------
    def abandon_page(self, inf, body):
        """A page whose client gives up after ABANDON_AFTER_S without an answer, in a thread: (thread,
        box), the outcome and `t_send` in the box (as `modal_tasks.Tasks.abandon_page`, c7.16)."""
        box = {}

        def go():
            t = self.clock()
            box.update(inf.page(body, ABANDON_AFTER_S), t_send=round(t, 3))
        th = threading.Thread(target=go, daemon=True)
        th.start()
        return th, box

    def concurrency(self):
        """App. D on RunPod (module docstring): the criteria of r3 §5.3 are a measure; G1 (two workers
        RUNNING at once, in the readings or in the log) and G2 (a page without an outcome) are the
        failures, exit 6; no readiness is inconclusive, exit 8."""
        a = self.a
        inf, w = self.inference(), self.watch(s1=True)
        body = dv.page_body(a.input)
        res = self.live.res = {"concurrency": a.name, "n": a.n, "warmup": a.warmup, "abandon": a.abandon}
        results, probes, reads, after, peak = [], [], [], [], {"running": 0}
        ab = {"n": a.abandon, "after_s": ABANDON_AFTER_S, "before_load": [], "requests": []}
        lock, code = threading.Lock(), 0

        def one():
            r = inf.page(body, PAGE_TIMEOUT_S)
            with lock:
                results.append(r)

        def metric(m):
            return {k: m.get(k) for k in ("status", "error", "running", "waiting", "t")}
        try:
            res["ready"] = self.ready(inf, w)
            self.not_ready(res["ready"])
            res["warmup_pages"] = [inf.page(body, PAGE_TIMEOUT_S) for _ in range(a.warmup)]
            warm = max((r.get("s") or 0 for r in res["warmup_pages"]), default=None)
            res.update(results=results, peak=peak, metrics=reads, abandon=ab)     # so far, on a signal
            abandoned = [self.abandon_page(inf, body) for _ in range(a.abandon)]
            if abandoned:                                    # vLLM runs them before the load (c7.16)
                t_wait = self.clock()
                while True:
                    m = inf.metrics(timeout=5.0)
                    ab["before_load"].append(metric(m))
                    if (m.get("running") or 0) >= a.abandon or self.clock() - t_wait >= ABANDON_RUNNING_S:
                        break
                    self.sleep(ABANDON_POLL_S)
                ab["running_before_load"] = max((r.get("running") or 0 for r in ab["before_load"]), default=0)
            threads = [threading.Thread(target=one, daemon=True) for _ in range(a.n)]
            # /metrics in a thread of its own, apart from the probes (N6: the loop missed the peak);
            # the whole text, not the start of the body (c7.15); at least one reading, even of a load
            # already done
            sampler = dv.Sampler(inf, reads=reads, peak=peak).start()
            t0 = last_read = self.clock()
            for t in threads:
                t.start()
            try:
                while True:
                    if a.probe_while_busy:
                        p = inf.get("/v1/models", timeout=5.0)
                        probes.append({k: p.get(k) for k in ("status", "error", "t")})
                    self.log_check()                         # G1 in the log during the load
                    if self.clock() - last_read >= AFTER_LOAD_EVERY_S:
                        last_read = self.clock()
                        self.read_quiet(w)                   # and in the readings
                    if not any(t.is_alive() for t in threads):
                        break
                    dv.join_for(threads, 1.0)
            finally:
                sampler.stop()
            total = round(self.clock() - t0, 2)
            res.update(total_s=total, warm_s=warm)
            t1 = self.clock()
            while self.clock() - t1 < AFTER_LOAD_WATCH_S:
                self.read_quiet(w)
                after.append(metric(inf.metrics(timeout=5.0)))
                self.sleep(AFTER_LOAD_EVERY_S)
            for th, _ in abandoned:
                th.join(ABANDON_AFTER_S + 15.0)
            ab["requests"] = [dict(box) for _, box in abandoned]
            ab["exercised"] = all(r.get("error") == "ReadTimeout" for r in ab["requests"]) if abandoned else None
            with lock:
                done = list(results)
            if len(done) != a.n:
                raise dv.Violation("%d pages of the load without an outcome" % (a.n - len(done)), "G2")
            self.outcomes(*[("warm-up page %d" % (i + 1), r) for i, r in enumerate(res["warmup_pages"])],
                          *[("page %d of the load" % (i + 1), r) for i, r in enumerate(done)],
                          *[("abandoned page %d" % (i + 1), r) for i, r in enumerate(ab["requests"])])
            res["criteria"] = {"running_at_least_%d" % RUNNING_MIN: peak["running"] >= RUNNING_MIN,
                               "pages_200": all(r.get("status") == 200 for r in done),
                               "within_4_warm": warm is not None and warm > 0 and total < 4 * warm,
                               "probes_200": all(p.get("status") == 200 for p in probes),
                               "abandoned": ab["exercised"] is not False}
            res["measure_pass"] = all(res["criteria"].values())
            secs = sorted(r["s"] for r in done if r.get("status") == 200 and r.get("s") is not None)
            res["page_s"] = {"min": secs[0], "median": secs[len(secs) // 2], "max": secs[-1]} if secs else None
        except dv.Violation as exc:
            res["violation"], res["guarantee"], code = str(exc), exc.guarantee, 6
        except dv.Inconclusive as exc:
            res["inconclusive"], code = str(exc), dv.INCONCLUSIVE_EXIT
        with lock:
            statuses = [r.get("status") or r.get("error") for r in results]
        res.pop("results", None)
        res.pop("metrics", None)
        out(dict(res, statuses=statuses, probes=probes[-20:], after_load=after,
                 metrics={"reads": len(reads), "with_running": sum(1 for m in reads if m.get("running") is not None),
                          "last": reads[-20:]},
                 readings=len(w.readings), restarts=w.restarts, read_errors=w.read_errors, skew_s=inf.skew,
                 **{"pass": code == 0}, **self.workers_out()))
        return code

    def fault_test(self):
        phase = self.a.phase
        return {"A": self.phase_a, "B": self.phase_b, "D": self.phase_d}[phase]()

    def phase_a(self):
        """hang-after-first (c7.20, G1 and G2): the open request and the abandoned one have an outcome
        (G2; 503 `mdf_tripped` from the proxy is the one expected, another is a fact); within A_ZERO_S
        of the hang the zero Z1 (G1: no worker RUNNING, the container of the worker that ran stopped);
        then 60 s without a worker RUNNING (G1). Unhealthy, extra workers, a restart before the zero:
        facts. No readiness or a first request that failed: inconclusive (exit 8)."""
        inf, w = self.inference(), self.watch(s1=True)
        res = self.live.res = {"phase": "A"}
        ok, code, restarts_ready = False, 6, 0
        try:
            res["ready"] = self.ready(inf, w)
            self.not_ready(res["ready"])
            restarts_ready = w.restarts
            res["first"] = inf.chat()
            t_hang = self.clock()
            if res["first"].get("status") != 200:
                raise dv.Inconclusive("the first request failed: %s" % {k: res["first"].get(k)
                                                                         for k in ("status", "proxy", "error")})
            self.fault = ("A", t_hang)                  # vLLM stopped after its first success (c7.19)
            held, box = inf.hold(timeout=400.0)
            abandoned = inf.abandon(5.0)
            while held.is_alive():
                self.read_quiet(w, no_unhealthy=True, no_restart=True)
                held.join(10.0)
            abandoned.join(20.0)
            res["held"] = box.get("result")
            res["abandoned"] = (getattr(abandoned, "box", None) or {}).get("result")
            self.outcomes(("the open request", res["held"]), ("the abandoned request", res["abandoned"]))
            body = res["held"].get("body") or {}
            if not (res["held"].get("status") == 503 and res["held"].get("proxy") == "error"
                    and isinstance(body, dict) and body.get("error") == "mdf_tripped"):
                self.fact("open_request_not_tripped", None, held={k: res["held"].get(k) for k in ("status", "proxy",
                                                                                                 "error")})
            if res["held"].get("t") is not None:
                res["t_trip_s"] = round(res["held"]["t"] - t_hang, 1)
            left = A_ZERO_S + 10 - (self.clock() - t_hang)
            w.until_zero(left, settled=self.settled, no_unhealthy=True, no_restart=True)
            res["zero_since_hang_s"] = round(self.clock() - t_hang, 1)
            self.zero_since_arm(res)
            w.for_seconds(60, no_unhealthy=True, no_restart=True)
            ok, code = True, 0
        except dv.Violation as exc:
            res["violation"], res["guarantee"] = str(exc), exc.guarantee
        except dv.Inconclusive as exc:
            res["inconclusive"], code = str(exc), dv.INCONCLUSIVE_EXIT
        out(dict(res, readings=w.readings, restarts=w.restarts, restarts_at_readiness=restarts_ready,
                 read_errors=w.read_errors, **{"pass": ok}, **self.workers_out()))
        return code

    def zero_since_arm(self, res):
        base = self.t_arm()
        if base is not None:
            res["zero_since_arm_s"] = round(self.clock() - base, 1)

    def phase_b(self):
        """crash-after-first with two pending, the session open (c7.20, G1 and G2): the three requests
        have an outcome (G2; the 500 of a crash is a fact); by the end of the observation the zero Z1
        (G1), then no worker RUNNING up to its end (G1); two RUNNING at once (G1). The restarts after
        the crash before the zero are observed (facts). No readiness: inconclusive (exit 8)."""
        inf, w = self.inference(), self.watch(s1=True)
        res = self.live.res = {"phase": "B"}
        ok, code = False, 6
        try:
            res["ready"] = self.ready(inf, w)
            self.not_ready(res["ready"])
            self.fault = ("B", self.clock())            # the requests inject the crash at the first success (c7.19)
            boxes = [inf.hold(timeout=300.0) for _ in range(3)]
            for t, _ in boxes:
                self.join_watching(t, 310.0)
            res["requests"] = [b.get("result") for _, b in boxes]
            self.outcomes(*[("request %d of B" % (i + 1), r) for i, r in enumerate(res["requests"])])
            for i, r in enumerate(res["requests"]):
                if r.get("status") != 200:
                    self.fact("request_after_crash", i, request=i + 1,
                              outcome={k: r.get(k) for k in ("status", "proxy", "error")})
            t_obs = self.clock()
            w.until_zero(self.a.observe_seconds, settled=self.settled)
            res["zero_since_fault_s"] = round(self.clock() - self.fault[1], 1)
            res["zero_since_requests_s"] = round(self.clock() - t_obs, 1)
            self.zero_since_arm(res)
            res["observed_readings"] = w.for_seconds(max(0.0, self.a.observe_seconds - (self.clock() - t_obs)))
            ok, code = True, 0
        except dv.Violation as exc:
            res["violation"], res["guarantee"] = str(exc), exc.guarantee
        except dv.Inconclusive as exc:
            res["inconclusive"], code = str(exc), dv.INCONCLUSIVE_EXIT
        out(dict(res, readings=w.readings, restarts=w.restarts, worker_ids=w.ids, read_errors=w.read_errors,
                 **{"pass": ok}, **self.workers_out()))
        return code

    def phase_d(self):
        """Disarmed: a valid request allocates no worker for 120 s (G1, the full zero after the disarm
        holds), and it has an outcome (G2). A worker: disarmed again, never deleted (c7.20)."""
        w = self.live.watch = dv.Watch(self.rp, self.clock, self.sleep)
        res = self.live.res = {"phase": "D"}
        try:
            s = self.rp.state()
            if s["armed"] or s["busy"]:
                raise dv.Violation("the endpoint is not disarmed and idle", "G1")
            inf = self.inference()
            t, box = inf.hold(timeout=150.0)
            w.for_seconds(D_SECONDS, allow_workers=False)
            t.join(40.0)
            r = box.get("result")
            res["request"] = {k: (r or {}).get(k) for k in ("status", "proxy", "error")}
            self.outcomes(("the request of D", r))
            ok = True
        except dv.Violation as exc:
            res["violation"], res["guarantee"], ok = str(exc), exc.guarantee, False
            if exc.guarantee == "G1":
                self.stop_on_worker()
        out(dict(res, readings=w.readings, **{"pass": ok}))
        return 0 if ok else 6

    def stop_on_worker(self):
        """A worker on a disarmed endpoint (G1). Before c7.20 the endpoint was deleted here; deleting
        cannot be undone, and `mdf-surya-fault` is deleted only by the cleaning at the end of the run
        (prompt of the closure of N4, §4): disarm again, idempotent, and stop for Davide."""
        try:
            self.rp.disarm()
            out({"delete_not_done": self.a.name, "reason": "a worker on a disarmed endpoint (G1): disarmed again, "
                 "deleting the endpoint is Davide's decision"})
        except pv.ProviderError as exc:
            out({"disarm_failed": str(exc)})

    def hold_command(self):
        """The command wrapped by the session of fault-e: one valid request, then wait."""
        inf = self.inference()
        r = self.ready(inf)
        c = inf.chat() if r["ready"] else {}
        Path(self.a.marker).write_text(json.dumps({"pid": os.getpid(), "ready": r["ready"], "status": c.get("status")}))
        while True:
            time.sleep(3600)

    # --- outside a session (prescription 2: try ... finally with the signals) -------------------
    def zero_finally(self):
        """Disarm (idempotent) and check the zero; no zero within 600 s is a stop."""
        self.rp.disarm()
        w = dv.Watch(self.rp, self.clock, self.sleep, max_total=10)
        return w.until_zero(pv.ZERO_STOP_S)

    def observe(self):
        a = self.a
        if not a.read_only:
            with spend.Ledger.file(a.spend_ledger).locked() as locked:
                row = serverspend.activity(locked, "runpod", "observe:" + a.name, Decimal(a.cap_usd) * 100,
                                           Decimal(a.spend_cap_usd) * 100, rate=pv.RATES["runpod"])
            out({"activity": row["activity_id"], "reserve_cents": row["reserve_cents"]})
        start = self.clock()
        end = (int(start) // 3600 + 1 + a.closed_utc_hours) * 3600
        w = self.live.watch = dv.Watch(self.rp, self.clock, self.sleep)
        res = self.live.res = {"observe": a.name, "from": iso(start), "to": iso(end)}

        def body():
            try:
                while self.clock() < end:
                    try:
                        s, _ = w.read(allow_workers=False)
                    except pv.ProviderError as exc:          # c7.20: a failed reading is a fact
                        w.read_errors.append(str(exc)[:200])
                        self.fact("read_error", len(w.read_errors), detail=str(exc)[:200], t=round(self.clock(), 3))
                        self.sleep(min(60, max(1, end - self.clock())))
                        continue
                    if s.get("workers_max") != 0:
                        raise dv.Violation("workers.max is %s" % s.get("workers_max"), "G1")
                    self.sleep(min(60, max(1, end - self.clock())))
                res["pass"] = True
            except dv.Violation as exc:
                res.update(violation=str(exc), guarantee=exc.guarantee)
                self.stop_on_worker()

        def cleanup():
            res["zero_s"] = self.safe_zero()
            out(dict(res, readings=len(w.readings), last=w.readings[-3:], read_errors=w.read_errors,
                     facts=list(self.facts)))

        dv.guarded(body, cleanup)
        return 0 if res.get("pass") and res.get("zero_s") is not None else 6

    def safe_zero(self):
        try:
            if not self.rp.exists(self.a.name):
                return 0
            return self.zero_finally()
        except (dv.Violation, pv.ProviderError) as exc:
            out({"stop": "no zero after the driver", "detail": str(exc)})
            return None

    def fault_e(self):
        """r4 App. G: the session killed with SIGKILL after a valid request; the refusals; disarm.

        c7.20: G2, `session.py run` and `mdf convert` refuse with `session_orphaned` before any arm;
        G1, `session.py disarm` reaches the full zero (and the zero of the cleaning). A valid request
        that failed, or the wrapped command alive 10 s after the SIGKILL (then killed, before the
        refusals: it could hold the lock of the window), are facts. A session that ended before its
        request, or no answer within 1500 s: inconclusive (exit 8)."""
        a = self.a
        res, procs = {"fault_e": a.name}, []
        self.live.res = res
        tmp = tempfile.mkdtemp(prefix="mdf-fault-e-")
        marker = Path(tmp) / "held.json"
        here = Path(__file__).resolve().parent
        py = [sys.executable, "-I", "-B"]
        base = ["--provider", "runpod", "--name", a.name, "--env-file", a.env_file, "--spend-ledger", a.spend_ledger]
        run = py + [str(here / "session.py"), "run"] + base + ["--spend-cap-usd", a.spend_cap_usd]

        def body():
            try:
                steps()
            except dv.Violation as exc:                          # in the result the cleanup prints
                res.update(violation=str(exc), guarantee=exc.guarantee)
            except dv.Inconclusive as exc:
                res["inconclusive"] = str(exc)

        def steps():
            ses = subprocess.Popen(run + ["--cap-usd", a.cap_usd, "--", *py, str(here / "runpod_endpoint.py"),
                                          "--env-file", a.env_file, "hold", "--name", a.name, "--marker", str(marker)],
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            procs.append(ses)
            t0 = self.clock()
            while not marker.exists():
                if ses.poll() is not None:
                    res["session_output"] = ses.stdout.read().decode("utf-8", "replace")[-2000:]
                    raise dv.Inconclusive("the session ended before the request: exit %s" % ses.returncode)
                if self.clock() - t0 > 1500:
                    raise dv.Inconclusive("no answer within 1500 s")
                self.sleep(2)
            held = json.loads(marker.read_text())
            res["held"] = held
            if held.get("status") != 200:
                self.fact("valid_request_failed", None, held=held)
            ses.kill()                                            # SIGKILL of the session
            ses.wait()
            res["session_output"] = ses.stdout.read().decode("utf-8", "replace")[-2000:]
            t_kill = self.clock()
            while _alive(held["pid"]):
                if self.clock() - t_kill > 10:
                    self.fact("command_alive_after_sigkill", None, pid=held["pid"], seconds=10)
                    os.kill(held["pid"], 9)
                    break
                self.sleep(0.5)
            res["command_end_s"] = round(self.clock() - t_kill, 1)
            again = subprocess.run(run + ["--cap-usd", "0.05", "--", "true"], capture_output=True, text=True, timeout=300)
            res["session_again"] = {"exit": again.returncode, "out": again.stdout[-600:]}
            if again.returncode != 3 or '"session_orphaned"' not in again.stdout:
                raise dv.Violation("session.py run did not refuse with session_orphaned", "G2")
            conv = self.convert_refused(tmp)
            res["convert"] = conv
            if not conv["refused"]:
                raise dv.Violation("mdf convert did not refuse with session_orphaned", "G2")
            st = subprocess.run(py + [str(here / "session.py"), "status"] + base, capture_output=True, text=True,
                                timeout=300)
            res["status"] = {"exit": st.returncode, "out": st.stdout[-1500:]}
            dis = subprocess.run(py + [str(here / "session.py"), "disarm"] + base, capture_output=True, text=True,
                                 timeout=900)
            res["disarm"] = {"exit": dis.returncode, "out": dis.stdout[-1500:]}
            if dis.returncode != 0:
                raise dv.Violation("session.py disarm: exit %d" % dis.returncode, "G1")
            res["pass"] = True

        def cleanup():
            for p in procs:
                if p.poll() is None:
                    p.kill()
                    p.wait()
            if marker.exists():
                pid = json.loads(marker.read_text()).get("pid")
                if pid and _alive(pid):
                    os.kill(pid, 9)
            res["zero_s"] = self.safe_zero()
            if res["zero_s"] is None and "violation" not in res:
                res.update(violation="no zero after the driver", guarantee="G1")
            res["facts"] = list(self.facts)
            out(res)

        dv.guarded(body, cleanup)
        if res.get("pass") and res.get("zero_s") is not None:
            return 0
        if res.get("inconclusive") and not res.get("violation"):
            return dv.INCONCLUSIVE_EXIT
        return 6

    def convert_refused(self, tmp):
        a = self.a
        sv = Path(tmp) / "servers.json"
        sv.write_text(json.dumps({"default": "fault", "servers": {"fault": {
            "kind": "runpod", "url": self.rp.url, "rate_usd_per_s": str(pv.RATES["runpod"])}}}))
        cmd = dv.convert_argv(Path(sys.executable).with_name("mdf"), a, Path(tmp) / "out", "fault", sv, "0.30")
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300, env=pv.child_env(os.environ))
        return {"exit": r.returncode, "refused": dv.orphan_refused(r.returncode, r.stdout),
                "out": (r.stdout + r.stderr)[-1500:]}

    def read_history(self):
        a = self.a
        ep = self.rp.endpoint()
        start, end = a.since, iso(self.clock())
        records, cursor = [], None
        for _ in range(100):
            params = {"startTime": start, "endTime": end, "interval": "1m", "limit": 1000}
            if cursor:
                params["cursor"] = cursor
            page = self.rp.call("GET", "/serverless/%s/metrics/workers" % ep["id"], params=params)
            records += page.get("records") or []
            pag = page.get("pagination") or {}
            if not pag.get("hasNextPage"):
                break
            cursor = pag.get("nextCursor")
        billing = self.rp.call("GET", "/billing/serverless", params={"startTime": start, "endTime": end,
                                                                    "bucketSize": "hour", "serverlessId": ep["id"]})
        busy = [r for r in records if any(r.get(k) for k in ("running", "idle", "initializing", "throttled", "unhealthy"))]
        out({"history": a.name, "from": start, "to": end, "minutes": len(records), "busy_minutes": busy,
             "billing": billing})
        return 0


def _alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:
        with open("/proc/%d/stat" % pid) as fh:
            return fh.read().split(")")[-1].split()[0] != "Z"
    except OSError:
        return False


def parser():
    ap = argparse.ArgumentParser(prog="runpod_endpoint.py", description=__doc__.split("\n\n")[0])
    ap.add_argument("--env-file", required=True)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("catalog")
    for name in ("create", "align", "verify-pins", "populate", "probe-negative", "probe-valid", "fault-test", "observe",
                 "fault-e", "read-history", "delete", "hold", "concurrency"):
        q = sub.add_parser(name)
        q.add_argument("--name", required=True)
        if name in ("populate", "probe-negative", "probe-valid", "fault-test", "concurrency"):
            q.add_argument("--spend-ledger")                      # passed by the session: the arm row
        if name == "concurrency":
            q.add_argument("--input", required=True)
            q.add_argument("--n", type=int, default=32)
            q.add_argument("--warmup", type=int, default=1)
            q.add_argument("--abandon", type=int, default=0)
            q.add_argument("--probe-while-busy", action="store_true")
        if name == "create":
            q.add_argument("--data-center", required=True)
            q.add_argument("--dry-run", action="store_true")
        if name == "fault-test":
            q.add_argument("--phase", required=True, choices=("A", "B", "D"))
            q.add_argument("--observe-seconds", type=int, default=1200)
        if name in ("observe", "fault-e"):
            q.add_argument("--spend-ledger")
            q.add_argument("--spend-cap-usd")
            q.add_argument("--cap-usd")
        if name == "observe":
            q.add_argument("--closed-utc-hours", type=int, required=True)
            q.add_argument("--read-only", action="store_true")
        if name == "fault-e":
            q.add_argument("--input", required=True)
        if name == "read-history":
            q.add_argument("--since", required=True)
        if name == "hold":
            q.add_argument("--marker", required=True)
    return ap


def main(argv=None, **inject):
    a = parser().parse_args(argv)
    if a.cmd == "observe" and not a.read_only and not (a.spend_ledger and a.spend_cap_usd and a.cap_usd):
        out({"code": "invalid_setting", "detail": "observe: --read-only, or the ledger and the caps"})
        return 2
    if a.cmd == "fault-e" and not (a.spend_ledger and a.spend_cap_usd and a.cap_usd):
        out({"code": "invalid_setting", "detail": "fault-e: the ledger and the caps"})
        return 2
    try:
        d = Driver(a, **inject)
        fn = {"catalog": d.catalog, "create": d.create, "align": d.align, "verify-pins": d.verify, "populate": d.populate,
              "probe-negative": d.probe_negative, "probe-valid": d.probe_valid, "fault-test": d.fault_test,
              "observe": d.observe, "fault-e": d.fault_e, "read-history": d.read_history, "delete": d.delete,
              "hold": d.hold_command, "concurrency": d.concurrency}[a.cmd]
        return dv.run_driver(fn, d.live, out)                 # SIGINT and SIGTERM: the result so far (c7.13)
    except pv.ProviderError as exc:
        out({"code": "provider_error", "detail": str(exc)})
        return 4


if __name__ == "__main__":
    sys.exit(main())
