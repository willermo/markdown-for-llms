"""The drivers of the N4 checks (MFL033, plan r4 §5.4 and App. G), shared by RunPod and Modal.

`Inference` talks to the endpoint through the provider's edge with the inference key; `Watch`
reads the provider every few seconds and checks the criteria of App. G at every reading (the first
violated one stops the check: the caller disarms, or deletes an endpoint that shows a worker while
disarmed, and stops for Davide). `guarded` runs a driver outside `session.py run` inside a
`try ... finally` that turns SIGINT and SIGTERM into an orderly stop (prescription 2).

MFL033 c7.13 (19/11: Modal's edge refused every readiness GET of `probe-valid` for 765 s, and
nothing of it reached a file). A readiness wait stops at the first refusal of the edge, a 401, 403
or 404 without `X-MDF-Proxy`, with its status, the names of the response headers, the content type
and the start of the body (never a value of the request's keys); every attempt is a line as soon as
it happens (`log`); `run_driver` prints the result so far when a driver gets SIGINT or SIGTERM.

MFL033 c7.20 (prompt of the closure of N4 on RunPod, Davide 20/11 19:44 and 19:54). A RunPod phase
fails only on a guarantee: G1, the spend stops on the provider's side (no worker RUNNING and the
container of the worker that ran stopped within the limit of the phase, the zero "Z1" of an armed
endpoint; no worker RUNNING after it without requests; the full zero after the disarm); G2, no request
lost in silence (every request sent has a declared outcome); G3, the bill within the bound of the
window (read at closed hours, outside the drivers). `Violation` carries the guarantee it breaks;
everything else that used to stop is a fact, recorded once (`Watch.fact`). `Inconclusive` is a phase
whose fault could not be exercised (no readiness, a first request that failed): no guarantee broken.
"""
import json
import signal
import threading
import time

import httpx

MODEL = "datalab-to/surya-ocr-2.5"
VLLM_VERSION = "0.25.1"
CHAT = {"model": MODEL, "messages": [{"role": "user", "content": "Reply with OK."}], "max_tokens": 8,
        "temperature": 0}
EDGE_REFUSED = (401, 403, 404)      # without X-MDF-Proxy: the provider's edge, not the proxy (c7.13)
BODY_START = 200
SCRUBBED = "[key]"


def page_body(path):
    """The chat of one page sent straight to vLLM, as `modal_tasks.Tasks.page` (c7.16): the PNG of
    `path` and "OCR this image to HTML.", at most 1024 tokens (MFL033 N5b: RunPod's `concurrency` and
    the page of `start_series.py`)."""
    from base64 import b64encode
    from pathlib import Path
    data = b64encode(Path(path).read_bytes()).decode("ascii")
    return {"model": MODEL, "max_tokens": 1024, "temperature": 0, "messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64," + data}},
        {"type": "text", "text": "OCR this image to HTML."}]}]}


class Violation(Exception):
    """A criterion of the check violated: stop for Davide. `guarantee` names the one broken (c7.20:
    "G1" or "G2" in the RunPod phases), or None for the criteria of the drivers before c7.20."""

    def __init__(self, text, guarantee=None):
        super().__init__(text)
        self.guarantee = guarantee


class Inconclusive(Exception):
    """c7.20: the fault of a phase could not be exercised (no readiness, the first request failed, the
    session of E ended before its request); no guarantee is broken. Exit 8."""


INCONCLUSIVE_EXIT = 8


def secrets_of(headers):
    """The values of the request's keys, as they could come back in a body: every header value,
    the token of `Bearer`, the parts of a `id.secret` token; the longest first."""
    found = set()
    for value in (headers or {}).values():
        value = str(value)
        found.add(value)
        if value.startswith("Bearer "):
            value = value[len("Bearer "):]
            found.add(value)
        found.update(p for p in value.split(".") if len(p) >= 8)
    return sorted((s for s in found if len(s) >= 8), key=len, reverse=True)


VLLM_METRICS = {"vllm:num_requests_running": "running", "vllm:num_requests_waiting": "waiting"}


def vllm_metrics(text):
    """{running, waiting} summed over the label sets of vLLM's Prometheus text, as `watchdog.metrics`;
    a name absent from the text is absent from the result."""
    out = {}
    for line in text.splitlines():
        if not line or line[0] == "#":
            continue
        name = line.split("{", 1)[0].split(" ", 1)[0]
        if name in VLLM_METRICS:
            try:
                value = float(line.rsplit(" ", 1)[1])
            except (IndexError, ValueError):
                continue
            out[VLLM_METRICS[name]] = out.get(VLLM_METRICS[name], 0.0) + value
    return out


METRICS_EVERY_S = 0.5           # Sampler: /metrics read so often during a load (MFL033 N6)
METRICS_TIMEOUT_S = 5.0


class Sampler:
    """`/metrics` read in a thread of its own during the load of `concurrency` (MFL033 N6): every
    `every_s`, from `start()` to `stop()`, at least once, apart from the probes of /v1/models. On
    21/11 (RunPod, window aa09739b) vLLM logged «Running: 29 reqs» while the driver read /metrics 3
    times in 14 s and saw 6: each round of its loop waited first for the probe, up to 5 s. Each
    reading goes into `reads` (status, error, running, waiting, t) and its `running` into `peak`;
    both may be the objects of the live result. The waits are of the real clock."""

    def __init__(self, inf, reads=None, peak=None, every_s=METRICS_EVERY_S, timeout=METRICS_TIMEOUT_S):
        self.inf, self.every_s, self.timeout = inf, every_s, timeout
        self.reads = [] if reads is None else reads
        self.peak = {"running": 0} if peak is None else peak
        self._stop, self._lock, self._thread = threading.Event(), threading.Lock(), None

    def read(self):
        m = self.inf.metrics(timeout=self.timeout)
        r = {k: m.get(k) for k in ("status", "error", "running", "waiting", "t")}
        with self._lock:
            self.reads.append(r)
            if m.get("running") is not None:
                self.peak["running"] = max(self.peak.get("running") or 0, m["running"])
        return r

    def _loop(self):
        while True:
            self.read()
            if self._stop.wait(self.every_s):
                return

    def start(self):
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(self.timeout + 5.0)


def join_for(threads, seconds):
    """Wait at most `seconds` of the real clock for `threads` to end (the pages of a load, MFL033 N6):
    True when all have ended. The loops of `concurrency` wait so between their rounds instead of the
    injected sleep, like `Sampler`: the threads are real."""
    end = time.monotonic() + seconds
    for t in threads:
        left = end - time.monotonic()
        if left <= 0:
            break
        t.join(left)
    return not any(t.is_alive() for t in threads)


class Inference:
    def __init__(self, url, headers, transport=None, clock=time.time, sleep=time.sleep, log=None):
        self.url, self.clock, self.sleep = url.rstrip("/"), clock, sleep
        self.transport = transport       # the abandoned request's client too (c7.19: in tests never the network)
        self.http = httpx.Client(transport=transport, headers=headers, follow_redirects=False,
                                 timeout=httpx.Timeout(150.0, connect=10.0))
        self.skew = None
        self.log = log                   # a line per readiness attempt, as it happens (c7.13)
        self.wait = None                 # the readiness wait in progress, for a partial result
        self._secrets = secrets_of(headers)

    def close(self):
        self.http.close()

    def get(self, path, timeout=150.0, headers=None, client=None):
        try:
            r = (client or self.http).get(self.url + path, timeout=timeout, headers=headers)
        except httpx.HTTPError as exc:
            return {"path": path, "error": type(exc).__name__}
        return self._record(path, r)

    def metrics(self, timeout=5.0):
        """GET /metrics, its whole text read in memory: the status and vLLM's values (`vllm_metrics`),
        never the body. `get` keeps only the first BODY_START characters of a text body, and vLLM's
        lines come after thousands of characters of process metrics (c7.15, 20/11)."""
        try:
            r = self.http.get(self.url + "/metrics", timeout=timeout)
        except httpx.HTTPError as exc:
            return {"path": "/metrics", "error": type(exc).__name__, "t": round(self.clock(), 3)}
        out = {"path": "/metrics", "status": r.status_code, "proxy": r.headers.get("x-mdf-proxy"),
               "t": round(self.clock(), 3)}
        if r.status_code == 200:
            out.update(vllm_metrics(r.text))
        return out

    def chat(self, timeout=150.0):
        try:
            r = self.http.post(self.url + "/v1/chat/completions", json=CHAT, timeout=timeout)
        except httpx.HTTPError as exc:
            return {"path": "chat", "error": type(exc).__name__, "t": round(self.clock(), 3)}
        return self._record("chat", r)

    def page(self, body, timeout=150.0):
        """One page (`page_body`) on a client of its own, so that pages can run in threads at once
        (MFL033 N5b): the status, the proxy's header and the seconds, never the body; an error of
        transport by its name."""
        t0 = self.clock()
        try:
            with httpx.Client(transport=self.transport, headers=dict(self.http.headers), follow_redirects=False,
                              timeout=httpx.Timeout(timeout, connect=10.0)) as c:
                r = c.post(self.url + "/v1/chat/completions", json=body)
            return {"status": r.status_code, "proxy": r.headers.get("x-mdf-proxy"), "s": round(self.clock() - t0, 2),
                    "t": round(self.clock(), 3)}
        except httpx.HTTPError as exc:
            return {"error": type(exc).__name__, "s": round(self.clock() - t0, 2), "t": round(self.clock(), 3)}

    def _record(self, path, r):
        out = {"path": path, "status": r.status_code, "proxy": r.headers.get("x-mdf-proxy"), "t": round(self.clock(), 3)}
        try:
            out["body"] = r.json()
        except ValueError:
            out["body"] = self.scrub(r.text[:4096])[:BODY_START]
        date = r.headers.get("date")
        if date:
            from email.utils import parsedate_to_datetime
            try:
                out["skew_s"] = round(self.clock() - parsedate_to_datetime(date).timestamp(), 1)
                self.skew = out["skew_s"]
            except (TypeError, ValueError):
                pass
        if out["status"] in EDGE_REFUSED and out["proxy"] is None:
            out["edge"] = self.edge(r)
        return out

    def edge(self, r):
        """A refusal of the provider's edge (c7.13): the status, the names of the response headers
        (never their values), the content type and the start of the body, without the request's keys."""
        return {"status": r.status_code, "header_names": sorted({k.lower() for k in r.headers.keys()}),
                "content_type": r.headers.get("content-type"), "body_start": self.scrub(r.text[:4096])[:BODY_START]}

    def scrub(self, text):
        for secret in self._secrets:
            text = text.replace(secret, SCRUBBED)
        return text

    def start_wait(self):
        """A new readiness wait: its attempts, for `attempt`, `progress` and a partial result."""
        self.wait = {"t0": self.clock(), "tries": []}
        return self.wait["tries"]

    def attempt(self, timeout):
        """One readiness GET of `/v1/models` in the current wait; its line goes out at once (c7.13)."""
        r = self.get("/v1/models", timeout=timeout)
        tries = self.wait["tries"]
        tries.append({k: r.get(k) for k in ("status", "proxy", "error", "t")})
        if self.log is not None:
            line = {"ready_try": len(tries), "s": round(self.clock() - self.wait["t0"], 1), "t": round(self.clock(), 3),
                    "status": r.get("status"), "proxy": r.get("proxy"), "error": r.get("error")}
            if r.get("edge"):
                line["edge"] = r["edge"]
            self.log(line)
        return r

    def progress(self):
        """The readiness wait so far, or None."""
        if self.wait is None:
            return None
        tries = self.wait["tries"]
        return {"seconds": round(self.clock() - self.wait["t0"], 1), "tries": len(tries), "last": tries[-3:]}

    def ready(self, within, every=5.0, check=None, timeout=150.0):
        """Wait for `/v1/models` 200 with the model, with GETs of at most `timeout` s; the header of the
        proxy is recorded, not required (`probe-valid` requires `forwarded`). `check()`, called before
        each GET, names a reason for the fast stop (the worker log and the workers of the endpoint,
        MFL033 c7.5 and c7.7): the wait ends there, not ready. A refusal of the edge (401, 403 or 404
        without `X-MDF-Proxy`) ends it too, at the first one (c7.13): no retry would pass it.
        `timeout` may be a callable, asked before each GET (c7.14: a GET never runs past the deadline
        of the scheduler of Modal), at least 1 s."""
        tries = self.start_wait()
        t0 = self.wait["t0"]
        while self.clock() - t0 < within:
            reason = check() if check is not None else None
            if reason:
                return {"ready": False, "stopped": reason, "seconds": round(self.clock() - t0, 1), "tries": len(tries),
                        "last": tries[-3:]}
            r = self.attempt(max(1.0, float(timeout())) if callable(timeout) else timeout)
            if r.get("edge"):
                return {"ready": False, "stopped": "refused at the edge: HTTP %d" % r["status"], "edge": r["edge"],
                        "seconds": round(self.clock() - t0, 1), "tries": len(tries), "last": tries[-3:]}
            ids = [m.get("id") for m in (r.get("body") or {}).get("data", [])] if isinstance(r.get("body"), dict) else []
            if r.get("status") == 200 and MODEL in ids:
                return {"ready": True, "seconds": round(self.clock() - t0, 1), "tries": len(tries), "last": tries[-3:],
                        "proxy": r.get("proxy")}
            self.sleep(every)
        return {"ready": False, "seconds": round(self.clock() - t0, 1), "tries": len(tries), "last": tries[-3:]}

    def hold(self, timeout):
        """A chat request in a thread (it hangs when vLLM is stopped); returns (thread, result box)."""
        box = {}

        def go():
            box["result"] = self.chat(timeout=timeout)

        t = threading.Thread(target=go, daemon=True)
        t.start()
        return t, box

    def abandon(self, after_s):
        """A chat request whose client gives up after `after_s` seconds (a short read timeout). Its
        outcome goes to `thread.box` (c7.20, G2: an abandoned request has one too: the client's timeout,
        or the answer when it came first)."""
        box = {}

        def go():
            try:
                with httpx.Client(transport=self.transport, headers=dict(self.http.headers),
                                  timeout=httpx.Timeout(after_s, connect=10.0)) as c:
                    r = c.post(self.url + "/v1/chat/completions", json=CHAT)
                box["result"] = {"path": "chat", "status": r.status_code, "proxy": r.headers.get("x-mdf-proxy"),
                                 "t": round(self.clock(), 3), "abandoned_after_s": after_s}
            except httpx.HTTPError as exc:
                box["result"] = {"path": "chat", "error": type(exc).__name__, "t": round(self.clock(), 3),
                                 "abandoned_after_s": after_s}
        t = threading.Thread(target=go, daemon=True)
        t.box = box
        t.start()
        return t


EXTRA_STATES = ("INITIALIZING", "IDLE", "THROTTLED")   # RunPod: not billed (workers/overview, 20/11)
EXTRA_MAX = 2                                           # RunPod's extra workers, "default: 2" (20/11)


def running_of(state):
    """The ids of the workers RUNNING in a reading, and the count of the summary when it says more."""
    ids = sorted(str(w.get("id")) for w in state.get("workers") or [] if str(w.get("status") or "").upper() == "RUNNING")
    return ids, max(len(ids), _int((state.get("summary") or {}).get("running")))


class Watch:
    """Readings of a provider (`prov.state()`), with the criteria of App. G. `after_read`, when given,
    runs after each reading that passed them and may raise `Violation` too (MFL033 c7.17: the drivers
    of Modal judge there the containers of the log of the session, which the follower lists every 5 s,
    between readings that come every 10 or 30 s).

    `extra_ok` (MFL033 c7.19, prompt N4s §4.1, the RunPod drivers that reach readiness): RunPod lists
    "extra workers" beyond `max` (19/11, never RUNNING, never billed). Then only the workers RUNNING
    count. A worker never RUNNING besides the one that runs (or the first, while none runs) is an
    extra worker, recorded. A worker RUNNING after another one was RUNNING is a replacement: recorded,
    counted in `restarts`, and judged by `judge(old, new, t)` (the driver); without `judge` it is a
    violation with `no_restart`, as before. A new worker id is no longer a restart by itself.

    Since c7.20 (the guarantees G1-G3, module docstring) `extra_ok` fails a reading only on G1: two
    workers RUNNING at once, or a worker RUNNING after the zero of the phase (`zero_t`). A worker
    unhealthy, outside the data center, in a state an extra worker is not seen in, more than 1 +
    EXTRA_MAX listed, an uptime that goes back: facts (`fact`, once each, `on_fact` told). `until_zero`
    waits for the zero Z1 of an armed endpoint (Davide, 20/11 19:44): no worker RUNNING, and each
    worker that ran no longer listed, or settled by `settled(listed)` (the driver: the log of the
    session, its "stop container"). A reading that fails with one of `tolerate` is recorded
    (`read_errors`) and the wait goes on; its deadline stays."""

    def __init__(self, prov, clock=time.time, sleep=time.sleep, max_total=1, data_centers=None, after_read=None,
                 extra_ok=False, judge=None, on_fact=None, tolerate=()):
        self.prov, self.clock, self.sleep, self.max_total = prov, clock, sleep, max_total
        self.data_centers = data_centers           # the data centers allowed to a worker (N4b §2), or None
        self.after_read = after_read
        self.readings, self.ids, self.uptime = [], [], {}
        self.restarts = 0
        self.extra_ok, self.judge = extra_ok, judge
        self.running_ids, self.seen, self.replacements = [], {}, []
        self.facts, self._fact_keys, self.on_fact = [], set(), on_fact
        self.zero_t = None                         # c7.20: the zero Z1 of the phase (clock)
        self.tolerate, self.read_errors = tuple(tolerate), []
        self.why_not_zero = None

    def fact(self, kind, key=None, **detail):
        """c7.20: something that stops nothing, recorded once per (kind, key)."""
        k = (kind, key)
        if k in self._fact_keys:
            return None
        self._fact_keys.add(k)
        row = dict(detail, fact=kind)
        if self.on_fact is not None:
            self.on_fact(row)
        else:
            self.facts.append(row)
        return row

    def read(self, allow_workers=True, no_unhealthy=False, no_restart=False):
        s = self.prov.state()
        t = round(self.clock(), 3)
        self.readings.append({"t": t, "state": s})
        summary, workers = s.get("summary") or {}, s.get("workers") or []
        count = max(summary.get("total", 0), len(workers), len(s.get("containers") or []))
        if not self.extra_ok and count > self.max_total:
            raise Violation("total %d > %d at %s" % (count, self.max_total, t))
        if not allow_workers and (count or s.get("busy")):
            raise Violation("a worker or container while disarmed at %s" % t, "G1")
        if self.extra_ok:
            self._extra(s, workers, count, t, no_restart)
        for w in workers:
            wid = w.get("id")
            dc = w.get("data_center")
            if self.data_centers is not None and dc is not None and dc not in self.data_centers:
                if not self.extra_ok:
                    raise Violation("worker %s in %s, not in %s" % (wid, dc, self.data_centers))
                self.fact("outside_data_center", wid, worker=wid, data_center=dc, allowed=self.data_centers, t=t)
            if no_unhealthy and str(w.get("status", "")).upper() == "UNHEALTHY":
                if not self.extra_ok:
                    raise Violation("worker %s unhealthy" % wid)
                self.fact("unhealthy", wid, worker=wid, t=t)
            if wid not in self.ids:
                if self.ids and not self.extra_ok:
                    self.restarts += 1
                    if no_restart:
                        raise Violation("a restart: worker %s after %s" % (wid, self.ids))
                self.ids.append(wid)
            up = w.get("uptime")
            if up is not None and wid in self.uptime and up < self.uptime[wid]:
                self.restarts += 1
                if self.extra_ok:
                    self.fact("uptime_back", (wid, up), worker=wid, uptime=up, before=self.uptime[wid], t=t)
                elif no_restart:
                    raise Violation("worker %s restarted (uptime %s < %s)" % (wid, up, self.uptime[wid]))
            if up is not None:
                self.uptime[wid] = up
        if no_unhealthy and summary.get("unhealthy", 0):
            if not self.extra_ok:
                raise Violation("summary.unhealthy %s" % summary.get("unhealthy"))
            self.fact("unhealthy_summary", None, unhealthy=summary.get("unhealthy"), t=t)
        if self.after_read is not None:
            self.after_read()
        return s, count

    def _extra(self, s, workers, count, t, no_restart):
        """The criteria of `extra_ok` on one reading (class docstring)."""
        running, n_running = running_of(s)
        if n_running > 1:
            raise Violation("two workers RUNNING at once at %s: %s" % (t, running or n_running), "G1")
        if self.zero_t is not None and n_running:
            raise Violation("a worker RUNNING after the zero of the phase (%.1f s after it), without requests: %s"
                            % (t - self.zero_t, running or n_running), "G1")
        if count > 1 + EXTRA_MAX:
            self.fact("more_workers", count, total=count, admitted=1 + EXTRA_MAX, t=t)
        for w in workers:
            wid, status = w.get("id"), str(w.get("status") or "").upper()
            seen = self.seen.setdefault(wid, {"first_t": t, "states": [], "gpu": w.get("gpu"),
                                              "data_center": w.get("data_center")})
            seen["last_t"] = t
            if not seen["states"] or seen["states"][-1] != status:
                seen["states"].append(status)
            if status == "RUNNING" and seen.get("running_t") is None:
                seen["running_t"] = t
            if wid not in self.ids:
                self.ids.append(wid)
            if status == "RUNNING" and wid not in self.running_ids:
                old = self.running_ids[-1] if self.running_ids else None
                self.running_ids.append(wid)
                if old is not None:
                    self.restarts += 1
                    self.replacements.append({"old": old, "new": wid, "t": t})
                    if self.judge is not None:
                        self.judge(old, wid, t)
                    elif no_restart:
                        raise Violation("a restart: worker %s RUNNING after %s" % (wid, old))
        primary = self.running_ids[-1] if self.running_ids else (self.ids[0] if self.ids else None)
        for w in workers:
            wid, status = w.get("id"), str(w.get("status") or "").upper()
            if wid != primary and wid not in self.running_ids and status not in EXTRA_STATES:
                self.fact("extra_worker_state", (wid, status), worker=wid, status=status, admitted=list(EXTRA_STATES),
                          t=t)

    def extra_workers(self):
        """The workers never seen RUNNING besides the one that ran (or the first, when none ran)."""
        primary = self.running_ids[-1] if self.running_ids else (self.ids[0] if self.ids else None)
        return [dict(self.seen[w], id=w) for w in self.ids
                if w in self.seen and w not in self.running_ids and w != primary]

    def not_zero(self, s, settled=None):
        """`extra_ok`: why a reading is not the zero Z1 of an armed endpoint, or None (c7.20, Davide
        20/11 19:44: no worker RUNNING, and the container of each worker that ran stopped). Without
        `settled` a worker that ran must be gone from the listing; with it the driver decides from the
        log of the session (a "stop container" line, a GONE). The extra workers, and the worker that
        ran when it is listed IDLE, THROTTLED or INITIALIZING without a container, are no obstacle."""
        running, n_running = running_of(s)
        if n_running:
            return "RUNNING in the reading: %s" % (running or n_running)
        listed = {w.get("id") for w in s.get("workers") or []}
        if settled is not None:
            return settled(listed)
        left = [x for x in self.running_ids if x in listed]
        if left:
            return "the worker that ran %s is still listed, and no log of its container" % left
        return None

    def until_zero(self, within, every=10.0, settled=None, **criteria):
        t0 = self.clock()
        while True:
            try:
                s, count = self.read(**criteria)
            except self.tolerate as exc:
                self.read_errors.append(str(exc)[:200])
                s = count = None
            if s is not None:
                if self.extra_ok:
                    self.why_not_zero = self.not_zero(s, settled)
                    if self.why_not_zero is None:
                        self.zero_t = self.clock()
                        return round(self.clock() - t0, 1)
                elif count == 0 and not s.get("busy"):
                    return round(self.clock() - t0, 1)
            if self.clock() - t0 > within:
                why = (": " + self.why_not_zero) if self.extra_ok and self.why_not_zero else ""
                if s is None:
                    why += " (the last reading failed; %d failed)" % len(self.read_errors)
                raise Violation("no zero within %s s%s" % (round(within, 1), why), "G1" if self.extra_ok else None)
            self.sleep(every)

    def for_seconds(self, seconds, every=10.0, **criteria):
        t0 = self.clock()
        while self.clock() - t0 < seconds:
            try:
                self.read(**criteria)
            except self.tolerate as exc:
                self.read_errors.append(str(exc)[:200])
            self.sleep(every)
        return len(self.readings)


def _int(value):
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


class Stop(Exception):
    """SIGINT or SIGTERM during a driver."""


class Live:
    """What a driver has so far (c7.13): its result dict, its readings and its readiness wait, for
    the partial result that `run_driver` prints on a signal."""

    def __init__(self):
        self.res, self.watch, self.inf = {}, None, None
        self.extra = {}                  # name -> callable: more of the state so far (c7.14: the scheduler)

    def __call__(self):
        out = {"result": self.res}
        for name, part in self.extra.items():
            out[name] = part()
        if self.inf is not None:
            out["ready_wait"] = self.inf.progress()
        if self.watch is not None:
            out.update(workers=list(self.watch.ids), readings=len(self.watch.readings),
                       last_reading=self.watch.readings[-1:])
        return out


def run_driver(fn, partial, emit):
    """Run a driver (c7.13): SIGINT and SIGTERM raise `Stop` in it (a `guarded` inside still runs its
    cleanup first), then the result so far, `partial()`, is printed and the exit is 130. On 19/11 the
    SIGTERM of the session closed `probe-valid` without a line."""
    def handler(signum, frame):
        raise Stop(signal.Signals(signum).name)
    old = {s: signal.signal(s, handler) for s in (signal.SIGINT, signal.SIGTERM)}
    try:
        return fn()
    except Stop as exc:
        for s in old:
            signal.signal(s, signal.SIG_IGN)               # one partial result, not a second Stop in it
        emit({"stopped_by": str(exc), "partial": partial()})
        return 130
    finally:
        for s, h in old.items():
            signal.signal(s, h)


def guarded(body, cleanup):
    """Run `body()`; SIGINT and SIGTERM raise `Stop` in it; `cleanup()` always runs (prescription 2)."""
    def handler(signum, frame):
        raise Stop(signal.Signals(signum).name)
    old = {s: signal.signal(s, handler) for s in (signal.SIGINT, signal.SIGTERM)}
    try:
        return body()
    finally:
        for s in old:
            signal.signal(s, signal.SIG_IGN)
        try:
            cleanup()
        finally:
            for s, h in old.items():
                signal.signal(s, h)


def dump(obj):
    return json.dumps(obj, sort_keys=True, default=str)


def convert_argv(mdf, a, out_dir, server, servers_file, server_cap_usd):
    """The `mdf convert` of phase E (App. D: refused with `session_orphaned` before any arm).

    `--env-file` is an option of the parser of `mdf`, not of `convert`: it comes before the
    subcommand (c7.18; after it `mdf` stopped with exit 2, unrecognized arguments, 20/11)."""
    return [str(mdf), "--env-file", a.env_file, "convert", a.input, "--out", str(out_dir), "--route",
            "marker-server", "--server-mode", "balanced", "--server", server, "--servers", str(servers_file),
            "--confirm-transit", "--server-cap-usd", server_cap_usd, "--spend-ledger", a.spend_ledger,
            "--spend-cap-usd", a.spend_cap_usd]


def orphan_refused(exit_code, stdout):
    """True when `mdf convert` refused with `session_orphaned`: exit 1 and the JSON of the refusal
    (`cli.prepare_server`), not the word anywhere in the output."""
    try:
        doc = json.loads(stdout)
    except ValueError:
        return False
    return exit_code == 1 and isinstance(doc, dict) and doc.get("status") == "refused" \
        and doc.get("error") == "session_orphaned"
