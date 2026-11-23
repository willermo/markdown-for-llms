#!/usr/bin/env python3
"""The N4 checks on Modal (MFL033; plan r3 §5.3, r4 §5.3, §5.4, App. D and G).

    modal_tasks.py --env-file F --server S --servers SV probe-negative --app A [--spend-ledger L]  (in session.py)
    modal_tasks.py --env-file F --server S --servers SV probe-valid --app A [--spend-ledger L]     (in session.py)
    modal_tasks.py --env-file F --server S --servers SV concurrency --app A --input PNG [--n 32] [--warmup 1]
                   [--abandon 1] [--probe-while-busy] [--spend-ledger L]    (in session.py; --abandon: pages, c7.16)
    modal_tasks.py --env-file F --server S --servers SV fault-test --app A --phase {A,B,D}
                   [--observe-seconds S] [--after-restore] [--spend-ledger L]                     (in session.py)
    modal_tasks.py --env-file F --server S --servers SV observe --app A --hours H (--read-only | --spend-ledger L
                   --spend-cap-usd C --cap-usd X)
    modal_tasks.py --env-file F --server S --servers SV fault-e --app A --input FILE --spend-ledger L
                   --spend-cap-usd C --cap-usd X
    modal_tasks.py --env-file F --server S --servers SV convert-refused --app A --input FILE --spend-ledger L
                   --spend-cap-usd C     (only `mdf convert` of phase E against an orphan, c7.18; no session)

The URL comes from the endpoint `S` of `servers.json`; the proxy token (`Modal-Key`,
`Modal-Secret`) from the env file; the state of the app from the pinned SDK (`modal app list`,
`modal container list`). Exit 0: passed; 4: a provider error; 6: a criterion violated; 7: no
capacity (`modal_no_capacity`: Modal's scheduler placed no container, c7.14); 8: inconclusive (MFL045 c2).

MFL045 c2 (plan r2 §7): on an app of the snapshot (the lever `snapshot` in the arm row) the readiness reads the
log with `worker_logs.StartRules(snapshot=True)` (prescription P-7: an error of the adopted vLLM during the wake
held until the declared fallback or readiness) and names the snapshot of the container that served it (kind,
generation, fallback, wake). `fault-test --after-restore` (phases A and B) first wakes a container (try 1),
waits for the quiet of the app (no container, at most AFTER_RESTORE_QUIET_S), then wakes another (try 2): the
phase runs on try 2 only when it is a `restore`, else it is inconclusive (exit 8). The container of try 1 is
not a replacement of rule (a): a declared quiet lies between the two (`next_try`, a fact).

The lessons of RunPod (MFL033 c7.8, prompt N4g §4.1): every readiness wait reads the log of the
session (`SESSION_WORKER_LOG`, `worker_logs.ModalFollower`) before each GET of 30 s and the
containers of the app every 30 s, and stops at once on an error of vLLM, a trip, the inert mode, an
error of the proxy, more than one container or a container replaced; it gives the readiness and the
first send from the arm (`--spend-ledger`). `probe-negative` passes with 401 of Modal's edge
(no `X-MDF-Proxy`) for both probes, no container for 120 s (the deploy starts none:
`min_containers=0`) and no request answered by the proxy. `probe-valid` is RunPod's: the version,
`/metrics`, `/health` and `/ping` of the proxy, a chat answered by vLLM, then the containers for 30 s
more; the concurrency of App. D follows it.

Since c7.13 (19/11: the edge refused `probe-valid` for 765 s without a trace) a readiness wait
stops at the first refusal of Modal's edge (401, 403 or 404 without `X-MDF-Proxy`), each attempt
is a line (`ready_try`) as soon as it happens, and SIGINT or SIGTERM print the result so far
(`stopped_by`, `partial`) before the exit 130.

Since c7.14 (20/11: no L4 free on Modal, nothing placed for 13 minutes) every readiness wait fetches
the system lines of the app without `-f` at most every 30 s (each a `modal_system` line) and stops
with `modal_no_capacity`, exit 7, when "waiting to be scheduled" is followed by no container for
300 s; the result names the GPU of the line and the GPUs requested (L4, then A10G), and the GPU the
proxy found (`mdf_gpu`).

Since c7.16 (20/11: the request `concurrency` abandoned was the short chat, answered in 0.14 s, before
the client's 5 s) the abandoned request is a page of the input, sent before the load: the driver
waits until vLLM runs it, then sends the pages, and its client gives up after 5 s while it runs. The
criterion of App. D on it (r3 §5.3: `num_requests_running`, the life of the container, the other
requests) is checked: the containers are read for 30 s after the load (not during it, so that the
time of the load stays that of c7.15; a container lost during it fails the pages, and one replaced is
seen after it), and one container, the same, never none, is required; a page answered before the
client gave up is not an abandonment, and fails the check. The result keeps what the session saw of the requests: the chats
in the log of the session by source (the proxy's `mdf_request` and trips, vLLM's access lines,
Modal's system lines) and Modal's inputs since the abandoned page (`modal function calls`).

Since c7.17 (20/11, `concurrency-2`: a container lived 11 s during the readiness, between two readings
of the driver, and was replaced unseen) every readiness wait and every phase reads the containers of
the log of the session too (`worker_logs.Lives`: listed every 5 s by the follower, and their own
lines), at each check of the wait, at each reading of the containers, during the load of
`concurrency` and while phase B holds its requests. Rule (a) of Davide (20/11 11:31) judges them:
a container replaced before the first answer of vLLM through its proxy, without an overlap and
without an inference on it, is a fact, declared at once (`replacement`, rule `a`: the two ids
abbreviated, their times and GPUs) and in the result, not a stop; two containers at once, or a
container replaced after readiness, are a criterion violated, and the fast stop. After the fault
of phase A a new container is "a container restarted after the hang" (App. G); after the requests
of phase B inject the crash, new containers are the restarts it observes (`restarts_after_fault`),
an overlap still a violation. Without a log of the session (`--disarmed`, `observe`) the readings
judge alone, as before.
"""
import argparse
import base64
from decimal import Decimal
import json
import os
from pathlib import Path
import re
import shutil
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
from markdown_for_llms.engine import servers  # noqa: E402

READY_WITHIN_S = 1260
READY_GET_S = 30                # each readiness GET: the log and the containers at least that often (c7.8)
READY_READ_EVERY_S = 30         # the containers and, since c7.14, the system lines of the app
AFTER_READY_WATCH_S = 30        # probe-valid reads the containers so long after the chat
NO_CAPACITY_EXIT = 7            # the scheduler of Modal placed no container (c7.14)
A_ZERO_S = 215
D_SECONDS = 120
# concurrency, the abandoned page (c7.16)
ABANDON_AFTER_S = 5.0           # the client of the abandoned page gives up after so long without an answer
ABANDON_RUNNING_S = 4.0         # the wait for vLLM to run it before the load, /metrics every ABANDON_POLL_S
ABANDON_POLL_S = 0.5
AFTER_LOAD_WATCH_S = 30         # after the load, the containers and /metrics every AFTER_LOAD_EVERY_S, so long
AFTER_LOAD_EVERY_S = 10
CHAT_PATH = "/v1/chat/completions"
VLLM_CHAT = re.compile(r'"POST /v1/chat/completions HTTP/1\.1" (\d{3})')        # vLLM's access line
MODAL_CHAT = re.compile(r"POST /v1/chat/completions -> (.*?)\s*(?:\((.*)\))?\s*$")   # Modal's system line
LOG_LINES_KEPT = 40
# the containers of the log of the session (c7.17)
RESOLVE_S = 15.0                # a container came while the one before is still listed: three listings, then judged
UNKNOWN_S = 15.0                # a container of the readings not yet in the log: as long, then a restart unjudged
HOLD_STEP_S = 5.0               # phase B: the log judged so often while its requests are held
AFTER_RESTORE_QUIET_S = 300     # MFL045 c2: the quiet of the app between the two tries of --after-restore


def out(obj):
    pv.emit(obj)


def overlap_text(o):
    """An overlap of `worker_logs.Lives.overlaps` in a few words."""
    if o["by"] == "listing":
        return "%s and %s in the same listing at +%.1f s" % (wl.short(o["a"]), wl.short(o["b"]), o["since_arm"])
    return "%s wrote %.1f s after %s wrote its first line" % (wl.short(o["a"]), o["a_last_t"] - o["b_first_t"],
                                                              wl.short(o["b"]))


def pair_text(row):
    """A row of `Tasks.pair_row` in a few words."""
    o, n = row["old"], row["new"]
    end = o["gone_s"] if o["gone_s"] is not None else o["last_s"]
    return "%s (+%s to +%s s, %s) then %s (from +%s s, %s)" % (o["id"], o["seen_s"], end, o["gpu"], n["id"], n["seen_s"],
                                                               n["gpu"])


class Tasks:
    def __init__(self, a, modal=None, transport=None, clock=time.time, sleep=time.sleep, environ=None):
        self.a, self.clock, self.sleep, self.transport = a, clock, sleep, transport
        self.environ = os.environ if environ is None else environ
        self.ep = servers.load(a.servers, a.server)
        if self.ep.provider != "modal":
            raise pv.ProviderError("the endpoint %s is not a Modal endpoint" % a.server)
        keys = servers.read_keys(self.ep, a.env_file)
        self.headers = servers.headers(self.ep, keys)
        arm = self.arm_row()                    # MFL045 c2: the app of the snapshot, from the levers of the arm
        self.snapshot = arm is not None and "snapshot" in str(arm.get("levers") or "").split(",")
        self.modal = modal or (pv.Modal(a.app, snapshot=self.snapshot) if getattr(a, "app", None) else None)
        self.live = dv.Live()                   # the result so far, printed on a signal (c7.13)
        self.no_capacity = None                 # the scheduler's summary when it stopped a wait (c7.14)
        # c7.17: the containers of the log of the session and rule (a)
        self.log_path = self.environ.get(wl.LOG_ENV)
        self.lives = wl.Lives() if self.log_path else None
        self._t_arm = None
        self.t_ready = None                     # when the driver saw readiness (clock)
        self.fault = None                       # ("A" | "B", clock): from then on the phase judges new containers
        self.served = None                      # the container alive at readiness
        self.require_alive = None               # concurrency: that container lives on (c7.16, from the log too)
        self.judged = {}                        # (old, new) -> "a" (rule (a)) or "restart" (after the fault of B)
        self.boundary = None                    # MFL045 c2: the declared quiet before try 2 of --after-restore
        if self.lives is not None:
            self.live.extra["containers_log"] = self.lives_out

    def inference(self, live=True):
        inf = dv.Inference(self.ep.url, self.headers, transport=self.transport, clock=self.clock, sleep=self.sleep,
                           log=out)
        if live:
            self.live.inf = inf
        return inf

    def watch(self):
        self.live.watch = dv.Watch(self.modal, self.clock, self.sleep, after_read=self.lives_check)
        return self.live.watch

    def arm_row(self):
        """The `arm` row of the session that runs this task (MDF_SESSION_ARM in `--spend-ledger`)."""
        arm_id = self.environ.get(serverspend.SESSION_ENV)
        ledger = getattr(self.a, "spend_ledger", None)
        if not arm_id or not ledger:
            return None
        rows = [r for r in spend.read_only(ledger) if r.get("kind") == "arm" and r.get("arm_id") == arm_id]
        return rows[-1] if rows else None

    # --- the containers of the log of the session (c7.17) --------------------------------------
    def t_arm(self):
        """The arm of the window: its row in the ledger, else the follower's first line; or None."""
        if self._t_arm is None:
            arm = self.arm_row()
            if arm is not None:
                self._t_arm = float(arm["t_arm"])
            elif self.lives is not None and self.lives.t_arm is not None:
                self._t_arm = self.lives.t_arm
        return self._t_arm

    def lives_check(self):
        """The containers of the log of the session, read again and judged (c7.17, prompt N4p §3.2);
        raises `dv.Violation`. Always a violation: two containers at once (`Lives.overlaps`). Each
        container that came after another one is judged once, by the moment it was first seen:
        before readiness, rule (a): the replaced one had no answer of vLLM through its proxy and no
        inference line, and the replacement is declared at once (`replacement`); after the fault of
        phase B, a restart that the phase observes (`restart_after_fault`); after the fault of phase
        A, a restart after the hang; otherwise, a replacement after readiness. A container that came
        while the one before is still listed waits RESOLVE_S for the listing that says GONE (or that
        holds both); one that was never listed ended at its last line. With `require_alive`, the
        container of readiness listed GONE is "no container" (c7.16). A container that the readings
        saw replace another one and that the log does not know after UNKNOWN_S is a restart rule (a)
        cannot judge: a violation, as before c7.17."""
        if self.lives is None:
            return
        lv = self.lives.feed(self.log_path)
        overlaps = lv.overlaps()
        if overlaps:
            raise dv.Violation("two containers at once: %s" % "; ".join(overlap_text(o) for o in overlaps[:3]))
        base = self.t_arm()
        now = None if base is None else self.clock() - base
        ready_s = None if self.t_ready is None or base is None else self.t_ready - base
        fault_s = None if self.fault is None or base is None else self.fault[1] - base
        for old, new in zip(lv.order, lv.order[1:]):
            if (old, new) in self.judged:
                continue
            if base is None:
                raise dv.Violation("%s after %s, and no time of the arm: rule (a) cannot judge it" % (new, old))
            o, n = lv.by[old], lv.by[new]
            b = self.boundary
            if b is not None and old in b["ids"] and new not in b["ids"] and o["gone"] is not None \
                    and n["seen"] >= b["since_arm"]:
                self.judged[(old, new)] = "next_try"            # MFL045 c2: a declared quiet lies between them
                out(dict({"next_try": sum(1 for k in self.judged.values() if k == "next_try")},
                         **self.pair_row(old, new)))
                continue
            if o["gone"] is None:
                if now - n["seen"] < RESOLVE_S:
                    continue                           # the next listing says GONE, or holds both
                if o["listed"] is not None:
                    raise dv.Violation("container %s came at +%.1f s while %s was still listed %.0f s later: no "
                                       "proof that they did not overlap" % (wl.short(new), n["seen"], wl.short(old),
                                                                            now - n["seen"]))
            text = pair_text(self.pair_row(old, new))
            if ready_s is None or n["seen"] < ready_s:         # before readiness: rule (a)
                if o["vllm"] is not None:
                    raise dv.Violation("a container replaced after the first answer of vLLM (+%.1f s, %s): %s"
                                       % (o["vllm"], o["vllm_by"], text))
                if o["inferences"]:
                    raise dv.Violation("a container replaced with an inference on it (%d lines): %s"
                                       % (o["inferences"], text))
                self.judged[(old, new)] = "a"
                out(dict({"replacement": sum(1 for k in self.judged.values() if k == "a"), "rule": "a"},
                         **self.pair_row(old, new)))
            elif fault_s is not None and n["seen"] >= fault_s:
                if self.fault[0] == "A":
                    raise dv.Violation("a container restarted after the hang: %s" % text)
                self.judged[(old, new)] = "restart"
                out(dict({"restart_after_fault": sum(1 for k in self.judged.values() if k == "restart")},
                         **self.pair_row(old, new)))
            else:
                raise dv.Violation("a container replaced after readiness: %s" % text)
        if self.require_alive is not None:
            life = lv.by.get(self.require_alive)
            if life is not None and life["gone"] is not None:
                raise dv.Violation("no container: %s listed GONE at +%.1f s, after readiness"
                                   % (wl.short(self.require_alive), life["gone"]))
        w = self.live.watch
        if w is not None and len(w.ids) > 1:
            for wid in w.ids:
                if wid in lv.by:
                    continue
                first = next((r["t"] for r in w.readings
                              if wid in [x.get("id") for x in r["state"].get("workers") or []]), self.clock())
                if self.clock() - first >= UNKNOWN_S:
                    raise dv.Violation("a restart: container %s after %s in the readings, not in the log of the "
                                       "session after %d s: rule (a) cannot judge it" % (wid, w.ids[0], UNKNOWN_S))

    def pair_row(self, old, new):
        """A container that came after another one: both ids abbreviated, the times since the arm (and
        UTC) each was first seen and listed GONE (or wrote its last line, when never listed), each
        GPU; the gap; the first answer of vLLM and the inference lines of the old one."""
        lv = self.lives

        def side(cid):
            life = lv.by[cid]
            return {"id": wl.short(cid), "seen_s": life["seen"], "seen_utc": life["seen_utc"], "seen_by": life["seen_by"],
                    "gone_s": life["gone"], "gone_utc": life["gone_utc"], "last_s": life["last"], "gpu": life["gpu"]}
        o, n = side(old), side(new)
        end = o["gone_s"] if o["gone_s"] is not None else o["last_s"]
        return {"old": o, "new": n, "gap_s": None if end is None or n["seen_s"] is None else round(n["seen_s"] - end, 1),
                "old_ended_by": "listing" if o["gone_s"] is not None else "last line",
                "old_vllm_s": lv.by[old]["vllm"], "old_inference_lines": lv.by[old]["inferences"]}

    def lives_out(self):
        """The containers of the log for a result: each one, the overlaps, the replacements of rule (a)
        and the restarts after the fault of phase B, rebuilt from the log as it is now (a GPU named
        after the declaration is in)."""
        if self.lives is None:
            return {}
        rows = {"a": [], "restart": [], "next_try": []}
        for (old, new), kind in self.judged.items():
            rows[kind].append(self.pair_row(old, new))
        res = {"containers_log": self.lives.summary(), "replacements": rows["a"], "restarts_after_fault": rows["restart"]}
        if rows["next_try"]:
            res["next_tries"] = rows["next_try"]                        # MFL045 c2, --after-restore
        return res

    def join_watching(self, thread, timeout):
        """Join `thread` within `timeout` s, judging the log every HOLD_STEP_S (c7.17, phase B)."""
        end = time.monotonic() + timeout
        while thread.is_alive() and time.monotonic() < end:
            thread.join(min(HOLD_STEP_S, max(0.0, end - time.monotonic())))
            self.lives_check()

    def ready(self, inf, w=None):
        """Readiness within READY_WITHIN_S with GETs of READY_GET_S; before each GET, until readiness,
        the fast stop from the log of the session (`worker_logs.StartRules`) and, every
        READY_READ_EVERY_S, the containers on `w`: more than one, or one replaced (c7.8, the lessons of
        RunPod c7.5 and c7.7); a failed reading is not a criterion. The first send and the readiness
        from the arm too (prompt N4g §4.3).

        Since c7.14 (20/11: Modal had no L4 free, its scheduler said so at +110 s and the wait went on
        for 800 s with no container) the system lines of the app are fetched without `-f` at most every
        READY_READ_EVERY_S (`worker_logs.Scheduler`), each new one a line at once (`modal_system`). A
        "waiting to be scheduled" followed by no container for NO_CAPACITY_S stops the wait with the
        code `modal_no_capacity` (exit 7), its text and the GPU it names; no GET runs past that
        deadline, and the containers are read again once it has passed. The GPU the proxy names at its
        start (`mdf_gpu`, c7.14) is in the result as `gpu`.

        Since c7.17 the containers of the log of the session are judged at every check (`lives_check`,
        rule (a)) and once more at readiness; a replacement in the readings is judged there too, and
        without a log of the session it stops the wait, as before. The result names the container
        that served readiness (`served`), the replacements of rule (a) and the containers of the log."""
        path = self.environ.get(wl.LOG_ENV)
        # MFL045 c8 (S-C1): with the snapshot, the clock and the arm of the window, so that its holds expire without lines
        timing = {"clock": self.clock, "t_arm": self.t_arm()} if self.snapshot else {}
        rules = wl.StartRules(snapshot=self.snapshot, **timing) if path else None
        w = self.watch() if w is None else w
        no_restart = self.lives is None
        last, fresh, read_errors = [None], [None], []
        arm = self.arm_row()
        since = float(arm["t_arm"]) if arm is not None else self.clock() - 60
        sched = wl.Scheduler(self.modal.system_logs, since, clock=self.clock, emit=out, every=READY_READ_EVERY_S,
                             requested=pv.MODAL_GPUS)
        self.live.extra["scheduler"] = sched.summary

        def check():
            if rules is not None:
                reason = rules.feed(path)
                if reason:
                    return reason
            if sched.due():
                sched.read()
            now, deadline = self.clock(), sched.deadline()
            past = deadline is not None and now >= deadline and (fresh[0] is None or fresh[0] < deadline)
            if last[0] is None or now - last[0] >= READY_READ_EVERY_S or past:
                last[0] = now
                try:
                    w.read(no_restart=no_restart)
                    fresh[0] = now
                except pv.ProviderError as exc:
                    read_errors.append(str(exc)[:200])
                except dv.Violation as exc:
                    return str(exc)
            try:
                self.lives_check()                       # every check, not only with the readings (c7.17)
            except dv.Violation as exc:
                return str(exc)
            reason = sched.verdict(w.ids, fresh[0])
            if reason:
                self.no_capacity = sched.summary()
            return reason

        def get_timeout():
            deadline = sched.deadline()
            if deadline is None or w.ids:
                return READY_GET_S
            return min(READY_GET_S, deadline - self.clock())

        t_send = self.clock()
        r = inf.ready(READY_WITHIN_S, check=check, timeout=get_timeout)
        if r["ready"] and rules is not None and self.snapshot:   # MFL045 c2, P-7: the outcome of the wake
            reason = rules.feed(path) or rules.at_readiness()
            if reason:
                r.update(ready=False, stopped=reason)
        if r["ready"]:
            self.t_ready = self.clock()
            try:
                self.lives_check()                       # the containers up to readiness, judged once more
            except dv.Violation as exc:
                r.update(ready=False, stopped=str(exc))
        if self.lives is not None:
            if r["ready"]:
                alive = self.lives.alive()
                answered = [c for c in alive if self.lives.by[c]["vllm"] is not None]
                self.served = (answered or alive or [None])[-1]
            r["served"] = None if self.served is None else wl.short(self.served)
            self.lives.feed(self.log_path)
            snap = None if self.served is None else self.lives.snapshot_of(self.served)
            if snap is not None or self.snapshot:                       # MFL045 c2: the start of a snapshot
                r["snapshot"] = snap
            r["replacements"] = self.lives_out()["replacements"]
            r["containers_log"] = self.lives.summary()["containers"]
        r["t_first_send"] = round(t_send, 3)
        if arm is not None:
            r["first_send_since_arm_s"] = round(t_send - float(arm["t_arm"]), 1)
            r["since_arm_s"] = round(self.clock() - float(arm["t_arm"]), 1)
        if rules is not None:
            r["proxy_events"] = rules.events
            r["gpu"] = (rules.events.get("mdf_gpu") or {}).get("names")
            if self.snapshot:
                r["exempt"] = rules.exempt
        r["scheduler"] = sched.summary()
        if self.no_capacity is not None:
            r["code"] = wl.NO_CAPACITY
        r.update(readings=len(w.readings), workers=list(w.ids), read_errors=read_errors)
        return r

    def after_restore(self, inf, w):
        """`--after-restore` (MFL045 c2, module docstring): (the readiness to run the phase on, the tries); raises
        `dv.Inconclusive` when neither try is a restore, or when the app is not quiet before try 2."""
        tries = self.live.res.setdefault("after_restore", [])
        r = None
        for i in (1, 2):
            row = {"try": i}
            if i == 2:
                t0 = self.clock()
                try:
                    w.until_zero(AFTER_RESTORE_QUIET_S)
                except dv.Violation as exc:
                    if not str(exc).startswith("no zero within"):
                        raise                               # two containers at once, or the log: a violation
                    raise dv.Inconclusive("no quiet of the app before try 2: %s" % exc) from None
                row["quiet_s"] = round(self.clock() - t0, 1)
                self.next_try()
            r = self.ready(inf, w)
            row.update(ready=r["ready"], kind=(r.get("snapshot") or {}).get("kind"), served=r.get("served"),
                       since_arm_s=r.get("since_arm_s"), stopped=r.get("stopped"))
            tries.append(row)
            out({"after_restore_try": row})
            if not r["ready"] or row["kind"] == "restore":
                return r, tries
        raise dv.Inconclusive("no restore in 2 tries: %s" % [x["kind"] for x in tries])

    def next_try(self):
        """The declared quiet before try 2: the containers seen so far, the moment; readiness judged anew."""
        lv = self.lives.feed(self.log_path) if self.lives is not None else None
        base = self.t_arm()
        self.boundary = {"ids": set(lv.order) if lv is not None else set(),
                         "since_arm": None if base is None else round(self.clock() - base, 1)}
        if self.boundary["since_arm"] is None:
            self.boundary = None
        self.t_ready = self.served = None

    def code(self, ok):
        """The exit of a driver: 0 passed, 7 no capacity on Modal (c7.14), 6 a criterion violated."""
        if ok:
            return 0
        return NO_CAPACITY_EXIT if self.no_capacity is not None else 6

    def no_request(self, requests, path):
        """No request answered by the proxy in the log of the session (c7.6)."""
        try:
            found = requests.feed(path)
        except OSError as exc:
            raise dv.Violation("the worker log cannot be read: %s" % type(exc).__name__) from None
        if found:
            raise dv.Violation("a request answered by the proxy: %s" % dv.dump(found[0]))

    def page(self, timeout=140.0):
        data = base64.b64encode(Path(self.a.input).read_bytes()).decode("ascii")
        body = {"model": dv.MODEL, "max_tokens": 1024, "temperature": 0, "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64," + data}},
            {"type": "text", "text": "OCR this image to HTML."}]}]}
        inf = self.inference(live=False)
        t0 = self.clock()
        try:
            r = inf.http.post(inf.url + "/v1/chat/completions", json=body, timeout=timeout)
            return {"status": r.status_code, "proxy": r.headers.get("x-mdf-proxy"), "s": round(self.clock() - t0, 2)}
        except dv.httpx.HTTPError as exc:
            return {"error": type(exc).__name__, "s": round(self.clock() - t0, 2)}
        finally:
            inf.close()

    def abandon_page(self, after_s=ABANDON_AFTER_S):
        """A page whose client gives up after `after_s` s without an answer, in a thread: (thread, box).
        The box gets the outcome (`error` ReadTimeout when the client gave up, `status` when the page
        came back first) and `t_send` (c7.16: the chat of `Inference.abandon` came back in 0.14 s)."""
        box = {}

        def go():
            t = self.clock()
            box.update(self.page(timeout=dv.httpx.Timeout(after_s, connect=10.0)), t_send=round(t, 3))
        th = threading.Thread(target=go, daemon=True)
        th.start()
        return th, box

    def alive(self, w, read_errors):
        """One reading of the containers after readiness (c7.16; App. D, r3 §5.3: the container lives
        through the abandoned request): one container, the same as before, never none. The violation,
        or None; a failed reading is not a criterion."""
        try:
            _, count = w.read(no_restart=True)
        except pv.ProviderError as exc:
            read_errors.append(str(exc)[:200])
            return None
        except dv.Violation as exc:
            return str(exc)
        if count == 0:
            return "no container at %s after readiness" % w.readings[-1]["t"]
        return None

    def chat_lines(self, sent):
        """The chats in the log of the session by source (c7.16, prompt N4o §3.1): the proxy's
        `mdf_request` lines by kind, status and code, and its trips; vLLM's access lines by status;
        Modal's system lines by result, each with its details (`duration`, `execution`). `sent`: the
        chats the driver sent. None without a log; the lines, never a body."""
        path = self.environ.get(wl.LOG_ENV)
        if not path:
            return None
        try:
            lines = wl.Tail().lines(path)
        except OSError as exc:
            return {"error": type(exc).__name__}
        proxy, vllm, modal, modal_lines, trips = {}, {}, {}, [], []
        for text in lines:
            e = wl.parse(text)
            if e is None or e["source"] == "session":
                continue
            ev = e["event"]
            if ev is not None:
                if ev.get("event") == "mdf_request" and ev.get("method") == "POST" and ev.get("path") == CHAT_PATH:
                    key = " ".join(str(x) for x in (ev.get("kind"), ev.get("status"), ev.get("code") or "")).strip()
                    proxy[key] = proxy.get(key, 0) + 1
                elif ev.get("event") == "mdf_trip":
                    trips.append({"rule": ev.get("rule"), "since_arm": e["since_arm"]})
                continue
            m = MODAL_CHAT.search(e["line"]) if e["source"] == "system" else None
            if m:
                modal[m.group(1)] = modal.get(m.group(1), 0) + 1
                modal_lines.append([e["since_arm"], m.group(1), (m.group(2) or "")[:80]])
                continue
            m = VLLM_CHAT.search(e["line"]) if e["source"] == "container" else None
            if m:
                vllm[m.group(1)] = vllm.get(m.group(1), 0) + 1
        return {"sent": sent, "proxy": proxy, "proxy_total": sum(proxy.values()), "vllm": vllm,
                "modal": modal, "modal_total": sum(modal.values()), "modal_lines": modal_lines[-LOG_LINES_KEPT:],
                "trips": trips}

    def inputs(self, since):
        """Modal's inputs of the function since `since` (c7.16, `providers.Modal.inputs`): how many, by
        status, the containers that ran them, those not `success` and the five longest. Every HTTP
        request is an input, the GETs of /metrics and /v1/models too. A failed reading is said."""
        try:
            rows = self.modal.inputs(since)
        except pv.ProviderError as exc:
            return {"since": round(since, 3), "error": str(exc)[:200]}

        def brief(r):
            return {k: r.get(k) for k in ("enqueued_at", "started_at", "execution_time_seconds", "status")}
        by = {}
        for r in rows:
            by[str(r.get("status"))] = by.get(str(r.get("status")), 0) + 1
        longest = sorted(rows, key=lambda r: r.get("execution_time_seconds") or 0, reverse=True)[:5]
        return {"since": round(since, 3), "n": len(rows), "status": by,
                "containers": sorted({r["container_id"] for r in rows if r.get("container_id")}),
                "not_success": [brief(r) for r in rows if r.get("status") != "success"][:10],
                "longest": [brief(r) for r in longest]}

    def probe_negative(self):
        """Two probes without a valid token on the deployed app (c7.8, as RunPod's c7.6). Passes when:
        (1) both get 401 from Modal's edge, without `X-MDF-Proxy`; (2) no container of the app for
        D_SECONDS (the deploy starts none, and a refused request must not either); (3) the log of the
        session shows no request answered by the proxy. The first violation stops at once."""
        inf = self.inference()
        with dv.httpx.Client(timeout=dv.httpx.Timeout(30.0, connect=10.0), transport=self.transport) as anon:
            res = {"no_key": inf.get("/v1/models", client=anon, timeout=30.0),
                   "wrong_key": inf.get("/v1/models", client=anon, timeout=30.0,
                                        headers={"Modal-Key": "wk-mdf-wrong", "Modal-Secret": "ws-mdf-wrong"})}
        probes = {k: {x: v.get(x) for x in ("status", "proxy", "error")} for k, v in res.items()}
        self.live.res = {"probe_negative": probes}
        path = self.environ.get(wl.LOG_ENV)
        requests, w, violation = wl.RequestLines(), self.watch(), None
        try:
            not_edge = {k: v for k, v in probes.items() if not (v["status"] == 401 and v["proxy"] is None)}
            if not_edge:
                raise dv.Violation("not a 401 of the edge: %s" % dv.dump(not_edge))
            if not path:
                raise dv.Violation("no log of the session (%s): the proxy cannot be checked" % wl.LOG_ENV)
            t0 = self.clock()
            while True:
                w.read(allow_workers=False)
                self.no_request(requests, path)
                if self.clock() - t0 >= D_SECONDS:
                    break
                self.sleep(10)
        except dv.Violation as exc:
            violation = str(exc)
        out({"probe_negative": probes, "violation": violation, "containers": w.ids, "proxy_requests": requests.found,
             "readings": w.readings, "pass": violation is None})
        return 0 if violation is None else 6

    def probe_valid(self):
        """Readiness (with the log and the containers), the version, `/metrics`, the health checks the
        proxy serves, a chat answered by vLLM; then the containers for AFTER_READY_WATCH_S more."""
        inf, w = self.inference(), self.watch()
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
                        w.read(no_restart=True)
                    except pv.ProviderError as exc:
                        read_errors.append(str(exc)[:200])
                    self.sleep(10)
            except dv.Violation as exc:
                violation = str(exc)
        version = res.get("version", {}).get("body")
        ok = (r["ready"] and violation is None and isinstance(version, dict)
              and version.get("version") == dv.VLLM_VERSION and res["chat"].get("status") == 200
              and res["chat"].get("proxy") == "forwarded")
        out(dict({"probe_valid": res, "violation": violation, "containers": w.ids,
                  "max_total": max((len(x["state"].get("containers") or []) for x in w.readings), default=0),
                  "readings": len(w.readings), "read_errors_after": read_errors, "skew_s": inf.skew, "pass": bool(ok)},
                 **self.lives_out()))
        return self.code(ok)

    def concurrency(self):
        """App. D on Modal: a warm-up page, then `n` pages at once with `/metrics` read every half
        second in a thread of its own, apart from the probes (`drivers.Sampler`, MFL033 N6) (r3 §5.3: `vllm:num_requests_running` at least 24, every page 200, all within 4 times the
        warm page; `--probe-while-busy`: every GET of /v1/models 200 too). With `--abandon` (c7.16) the
        abandoned pages go first, as long as `ABANDON_RUNNING_S` until vLLM runs them, and their client
        gives up after `ABANDON_AFTER_S` while the load runs; App. D on them: the other pages 200, and
        one container, the same as at readiness, in every reading of the AFTER_LOAD_WATCH_S after the
        load (none during it: an SDK reading takes about 2 s and would lengthen the measured load). A
        page answered before its client gave up was not abandoned: the check fails."""
        a, inf, w = self.a, self.inference(), self.watch()
        res = self.live.res = {}
        res["ready"] = self.ready(inf, w)
        if not res["ready"]["ready"]:
            out(dict(res, **{"pass": False}, **self.lives_out()))
            return self.code(False)
        self.require_alive = self.served                # in the log too, never none after readiness (c7.17)
        res["warmup"] = [self.page() for _ in range(a.warmup)]
        warm = max(r.get("s", 0) for r in res["warmup"])
        results, probes, reads, peak = [], [], [], {"running": 0}
        violation, read_errors = None, []
        ab = {"n": a.abandon, "after_s": ABANDON_AFTER_S, "before_load": []}
        self.live.res = dict(res, results=results, peak=peak, metrics=reads, abandon=ab)   # so far, on a signal
        t_first = self.clock()
        abandoned = [self.abandon_page() for _ in range(a.abandon)]
        if abandoned:                                    # vLLM runs them before the load (c7.16)
            t_wait = self.clock()
            while True:
                m = inf.metrics(timeout=5.0)
                ab["before_load"].append({k: m.get(k) for k in ("status", "error", "running", "waiting", "t")})
                if (m.get("running") or 0) >= a.abandon or self.clock() - t_wait >= ABANDON_RUNNING_S:
                    break
                self.sleep(ABANDON_POLL_S)
            ab["running_before_load"] = max((r.get("running") or 0 for r in ab["before_load"]), default=0)
        threads = [threading.Thread(target=lambda: results.append(self.page())) for _ in range(a.n)]
        # /metrics in a thread of its own, apart from the probes (N6: the loop missed the peak); the
        # whole text, not the start of the body (c7.15)
        sampler = dv.Sampler(inf, reads=reads, peak=peak).start()
        t0 = self.clock()
        for t in threads:
            t.start()
        try:
            while any(t.is_alive() for t in threads):
                if a.probe_while_busy:
                    p = inf.get("/v1/models", timeout=5.0)
                    probes.append({k: p.get(k) for k in ("status", "error", "t")})
                violation = violation or self.lives_violation()   # the log during the load, no SDK reading (c7.17)
                dv.join_for(threads, 1.0)
        finally:
            sampler.stop()
        total = round(self.clock() - t0, 2)
        after, t1 = [], self.clock()                     # the container after the load, and vLLM's queue (c7.16)
        while self.clock() - t1 < AFTER_LOAD_WATCH_S:
            if violation is None:
                violation = self.alive(w, read_errors) or self.lives_violation()
            m = inf.metrics(timeout=5.0)
            after.append({k: m.get(k) for k in ("status", "error", "running", "waiting", "t")})
            self.sleep(AFTER_LOAD_EVERY_S)
        for th, _ in abandoned:
            th.join(ABANDON_AFTER_S + 15.0)
        ab["requests"] = [dict(box) for _, box in abandoned]
        ab["exercised"] = all(r.get("error") == "ReadTimeout" for r in ab["requests"]) if abandoned else None
        if abandoned and not ab["exercised"] and violation is None:
            violation = "an abandoned page was not abandoned: %s" % dv.dump(ab["requests"])
        ab["log"] = self.chat_lines(a.warmup + a.abandon + a.n)
        ab["inputs"] = self.inputs(t_first - 1)
        ok = (peak["running"] >= 24 and all(r.get("status") == 200 for r in results) and total < 4 * warm
              and all(p.get("status") == 200 for p in probes) and violation is None)
        seen = {"reads": len(reads), "with_running": sum(1 for m in reads if m.get("running") is not None),
                "last": reads[-20:]}
        out(dict(res, total_s=total, warm_s=warm, peak=peak, statuses=[r.get("status") for r in results],
                 probes=probes[-20:], metrics=seen, after_load=after, abandon=ab, violation=violation,
                 containers=w.ids, readings=len(w.readings), read_errors=read_errors, **{"pass": ok},
                 **self.lives_out()))
        return self.code(ok)

    def lives_violation(self):
        """`lives_check` as a value: the violation, or None (c7.17)."""
        try:
            self.lives_check()
        except dv.Violation as exc:
            return str(exc)
        return None

    def fault_test(self):
        inf, w = self.inference(), self.watch()
        res = self.live.res = {"phase": self.a.phase}
        try:
            if self.a.phase == "D":
                s = self.modal.state()
                if s["armed"] or s["busy"]:
                    raise dv.Violation("the app is not stopped and idle")
                t, box = inf.hold(timeout=150.0)
                w.for_seconds(120, allow_workers=False)
                t.join(40.0)
                res["request"] = {k: (box.get("result") or {}).get(k) for k in ("status", "proxy", "error")}
            else:
                if getattr(self.a, "after_restore", False):         # MFL045 c2
                    res["ready"], res["after_restore"] = self.after_restore(inf, w)
                else:
                    res["ready"] = self.ready(inf, w)
                if not res["ready"]["ready"]:
                    raise dv.Violation(res["ready"].get("stopped") or "not ready")
                restarts_ready = w.restarts             # a replacement of rule (a) during readiness is not one (c7.17)
                if self.a.phase == "A":
                    res["first"] = inf.chat()
                    t_hang = self.clock()
                    if res["first"].get("status") != 200:
                        raise dv.Violation("the first request failed")
                    self.fault = ("A", t_hang)          # vLLM stopped after its first success (c7.17)
                    held, box = inf.hold(timeout=400.0)
                    inf.abandon(5.0)
                    while held.is_alive():
                        w.read(no_restart=True)
                        held.join(10.0)
                    h = box.get("result") or {}
                    res["held"] = h
                    body = h.get("body")
                    if not (h.get("status") == 503 and h.get("proxy") == "error" and isinstance(body, dict)
                            and body.get("error") == "mdf_tripped"):
                        raise dv.Violation("the open request did not get 503 mdf_tripped from the proxy")
                    res["t_trip_s"] = round(h["t"] - t_hang, 1) if h.get("t") else None
                    w.until_zero(A_ZERO_S + 10 - (self.clock() - t_hang), no_restart=True)
                    res["zero_since_hang_s"] = round(self.clock() - t_hang, 1)
                    w.for_seconds(60, no_restart=True)
                    if w.restarts > restarts_ready:
                        raise dv.Violation("a container restarted after the hang")
                else:
                    self.fault = ("B", self.clock())    # the requests inject the crash at the first success (c7.17)
                    boxes = [inf.hold(timeout=140.0) for _ in range(3)]
                    for t, _ in boxes:
                        self.join_watching(t, 150.0)
                    res["requests"] = [b.get("result") for _, b in boxes]
                    w.for_seconds(self.a.observe_seconds)
            ok = True
        except dv.Violation as exc:
            res["violation"], ok = str(exc), False
        except dv.Inconclusive as exc:                              # MFL045 c2: --after-restore without a restore
            res["inconclusive"] = str(exc)
            out(dict(res, readings=w.readings, restarts=w.restarts, **{"pass": False}, **self.lives_out()))
            return dv.INCONCLUSIVE_EXIT
        out(dict(res, readings=w.readings, restarts=w.restarts, **{"pass": ok}, **self.lives_out()))
        return self.code(ok)

    def zero_finally(self):
        try:
            self.modal.disarm()
            return dv.Watch(self.modal, self.clock, self.sleep, max_total=10).until_zero(pv.ZERO_STOP_S)
        except (dv.Violation, pv.ProviderError) as exc:
            out({"stop": "no zero after the driver", "detail": str(exc)})
            return None

    def observe(self):
        a = self.a
        if not a.read_only:
            with spend.Ledger.file(a.spend_ledger).locked() as locked:
                row = serverspend.activity(locked, "modal", "observe:" + a.app, Decimal(a.cap_usd) * 100,
                                           Decimal(a.spend_cap_usd) * 100, rate=pv.RATES["modal"])
            out({"activity": row["activity_id"]})
        end = self.clock() + a.hours * 3600
        w = self.watch()
        res = self.live.res = {"observe": a.app}

        def body():
            try:
                while self.clock() < end:
                    w.read(allow_workers=False)
                    self.sleep(60)
                res["pass"] = True
            except dv.Violation as exc:
                res["violation"] = str(exc)

        def cleanup():
            res["zero_s"] = self.zero_finally()
            out(dict(res, readings=len(w.readings), last=w.readings[-3:]))

        dv.guarded(body, cleanup)
        return 0 if res.get("pass") and res.get("zero_s") is not None else 6

    def fault_e(self):
        a, here, py = self.a, Path(__file__).resolve().parent, [sys.executable, "-I", "-B"]
        tmp = tempfile.mkdtemp(prefix="mdf-fault-e-")
        marker, res, procs = Path(tmp) / "held.json", {"fault_e": a.app}, []
        self.live.res = res
        base = ["--provider", "modal", "--name", a.app, "--env-file", a.env_file, "--spend-ledger", a.spend_ledger]
        run = py + [str(here / "session.py"), "run"] + base + ["--spend-cap-usd", a.spend_cap_usd, "--url", self.ep.url]

        def body():
            ses = subprocess.Popen(run + ["--cap-usd", a.cap_usd, "--", *py, str(here / "modal_tasks.py"), "--env-file",
                                          a.env_file, "--server", a.server, "--servers", a.servers, "hold",
                                          "--app", a.app, "--marker", str(marker)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            procs.append(ses)
            t0 = self.clock()
            while not marker.exists():
                if ses.poll() is not None or self.clock() - t0 > 1800:
                    raise dv.Violation("no answer before the session ended or 1800 s")
                self.sleep(2)
            held = json.loads(marker.read_text())
            if held.get("status") != 200:
                raise dv.Violation("the valid request failed")
            ses.kill()
            ses.wait()
            t_kill = self.clock()
            while _alive(held["pid"]):
                if self.clock() - t_kill > 10:
                    raise dv.Violation("the wrapped command is alive 10 s after the SIGKILL")
                self.sleep(0.5)
            again = subprocess.run(run + ["--cap-usd", "0.05", "--", "true"], capture_output=True, text=True, timeout=300)
            res["session_again"] = {"exit": again.returncode, "out": again.stdout[-600:]}
            if again.returncode != 3 or '"session_orphaned"' not in again.stdout:
                raise dv.Violation("session.py run did not refuse with session_orphaned")
            res["convert"] = conv = self.convert_refused(tmp)
            if not conv["refused"]:
                raise dv.Violation("mdf convert did not refuse with session_orphaned")
            dis = subprocess.run(py + [str(here / "session.py"), "disarm"] + base, capture_output=True, text=True,
                                 timeout=900)
            res["disarm"] = {"exit": dis.returncode, "out": dis.stdout[-1000:]}
            if dis.returncode != 0:
                raise dv.Violation("session.py disarm: exit %d" % dis.returncode)
            res["pass"] = True

        def cleanup():
            for p in procs:
                if p.poll() is None:
                    p.kill()
                    p.wait()
            res["zero_s"] = self.zero_finally()
            out(res)

        try:
            dv.guarded(body, cleanup)
        except dv.Violation as exc:
            out({"violation": str(exc)})
        return 0 if res.get("pass") and res.get("zero_s") is not None else 6

    def convert_refused(self, tmp):
        """`mdf --env-file F convert …` against the endpoint of this driver (c7.18)."""
        a = self.a
        cmd = dv.convert_argv(Path(sys.executable).with_name("mdf"), a, Path(tmp) / "out", a.server, a.servers, "0.50")
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300, env=pv.child_env(os.environ))
        return {"exit": r.returncode, "refused": dv.orphan_refused(r.returncode, r.stdout),
                "out": (r.stdout + r.stderr)[-1000:]}

    def convert_refused_command(self):
        """Only the refusal of phase E against an orphan already in the ledger (N4q §4.2): no session,
        no arm, no call to Modal. Exit 0: refused with `session_orphaned`; 6: not refused."""
        tmp = tempfile.mkdtemp(prefix="mdf-convert-refused-")
        try:
            conv = self.convert_refused(tmp)
            out({"convert_refused": self.a.app, "convert": conv, "pass": conv["refused"]})
            return 0 if conv["refused"] else 6
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def hold_command(self):
        inf = self.inference()
        r = self.ready(inf)
        c = inf.chat() if r["ready"] else {}
        Path(self.a.marker).write_text(json.dumps({"pid": os.getpid(), "ready": r["ready"], "status": c.get("status")}))
        while True:
            time.sleep(3600)


def _alive(pid):
    try:
        with open("/proc/%d/stat" % pid) as fh:
            return fh.read().split(")")[-1].split()[0] != "Z"
    except OSError:
        return False


def parser():
    ap = argparse.ArgumentParser(prog="modal_tasks.py", description=__doc__.split("\n\n")[0])
    ap.add_argument("--env-file", required=True)
    ap.add_argument("--server", required=True)
    ap.add_argument("--servers", required=True)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("probe-negative", "probe-valid", "concurrency", "fault-test", "observe", "fault-e", "hold",
                 "convert-refused"):
        q = sub.add_parser(name)
        q.add_argument("--app", required=True)
        q.add_argument("--spend-ledger")                 # the arm row of the session (since the arm), or the reserve
        if name == "concurrency":
            q.add_argument("--input", required=True)
            q.add_argument("--n", type=int, default=32)
            q.add_argument("--warmup", type=int, default=1)
            q.add_argument("--abandon", type=int, default=0)
            q.add_argument("--probe-while-busy", action="store_true")
        if name == "fault-test":
            q.add_argument("--phase", required=True, choices=("A", "B", "D"))
            q.add_argument("--observe-seconds", type=int, default=1300)
            q.add_argument("--after-restore", action="store_true")        # MFL045 c2
        if name in ("observe", "fault-e", "convert-refused"):
            q.add_argument("--spend-cap-usd")
        if name in ("observe", "fault-e"):
            q.add_argument("--cap-usd")
        if name == "observe":
            q.add_argument("--hours", type=int, required=True)
            q.add_argument("--read-only", action="store_true")
        if name in ("fault-e", "convert-refused"):
            q.add_argument("--input", required=True)
        if name == "hold":
            q.add_argument("--marker", required=True)
    return ap


def main(argv=None, **inject):
    a = parser().parse_args(argv)
    if a.cmd in ("observe", "fault-e") and not getattr(a, "read_only", False) and not (
            a.spend_ledger and a.spend_cap_usd and a.cap_usd):
        out({"code": "invalid_setting", "detail": "the ledger and the caps, or --read-only"})
        return 2
    if a.cmd == "convert-refused" and not (a.spend_ledger and a.spend_cap_usd):
        out({"code": "invalid_setting", "detail": "convert-refused: the ledger and its cap"})
        return 2
    try:
        t = Tasks(a, **inject)
        fn = {"probe-negative": t.probe_negative, "probe-valid": t.probe_valid, "concurrency": t.concurrency,
              "fault-test": t.fault_test, "observe": t.observe, "fault-e": t.fault_e, "hold": t.hold_command,
              "convert-refused": t.convert_refused_command}[a.cmd]
        return dv.run_driver(fn, t.live, out)                 # SIGINT and SIGTERM: the result so far (c7.13)
    except (pv.ProviderError, servers.EngineError) as exc:
        out({"code": "provider_error", "detail": str(exc)})
        return 4


if __name__ == "__main__":
    sys.exit(main())
