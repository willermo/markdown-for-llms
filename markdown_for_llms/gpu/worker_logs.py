"""The logs of the RunPod workers of a window, read while they live (MFL033 c7.4).

`Follower` lists the workers of the endpoint every 10 s and, for each one, reads its log stream
(`GET /v2/serverless/{id}/workers/{workerId}/logs`, Server-Sent Events whose `data` is
`{"source", "line", "ts"}`; REST v2 schema read on 19/11, scope `serverless:read`), from 60 s
before the arm, resuming with `Last-Event-ID` after a quiet stream or a cut. A 404 ends the
reading of a worker only when the worker is no longer listed (while it starts, it is tried again).
Each event becomes a line of a file of the session (a private directory under /tmp until the
secret scan):

    <received, UTC>  +<seconds since the arm>  <worker>  <source>  <ts>  <line>

separated by tabs; the source `session` holds the worker list as read (assignment, status, GPU,
data center). The lines of the proxy are its JSON events (`watchdog.py`): events and times only.

`ModalFollower` (c7.8) writes the same file for a Modal app: one `modal app logs APP -f --timestamps
--show-container-id -s SOURCE` of the pinned SDK per source (off a terminal the CLI tells stdout,
stderr and system apart only by a colour, which it drops), the container as the worker, `stdout` and
`stderr` as the source `container`; and `modal container list --json` every 5 s for the containers of
the app (assignment as PENDING, the start with its `start_time`, GONE). A stream that ends while the
follower runs is started again, the lines already written skipped. The streams end when the app
stops; if the session dies first they end at their next write, on the closed pipe.

`StartRules` reads that file while a driver waits for readiness (the populate, and since c7.5 every
readiness wait of the RunPod drivers, since c7.8 of the Modal ones) and names the first reason for
the fast stop: a download that fails or does not move for 120 s, an error of vLLM (an ERROR or
CRITICAL line of its logger, or its exit), a trip, the inert mode, an error of the proxy.
With a snapshot (MFL045 c2, prescription P-7; `StartRules(snapshot=True)`): an error of vLLM from a pid of the
group adopted by `snap_wake` (the pids of its `mdf_restore`) during the wake of its container is held until the
explicit outcome: a declared fallback (`mdf_fallback`) lets it go (`exempt`), the wake or readiness without a
fallback makes it the stop (`at_readiness` too), and no outcome within WAKE_HOLD_S of the restore makes it the
stop; an error of the same pid after the fallback, of the vLLM of the fallback or of another container is the
stop as before. A line of a container whose `mdf_restore` is not in the file yet (stdout and stderr come in
separate streams) waits PENDING_S for it. A failed `snap_boot` (`mdf_snapshot` not ok) is a stop. The two holds run
from the arm on the latest of the lines read and, when the driver gives its clock and the arm of its window (MFL045 c8,
`modal_tasks.Tasks.ready`), of that clock: `ModalFollower` writes a container only when its listing changes, so a
container listed the same while its vLLM is silent adds no line, and the hold still expires.
`RequestLines` collects the requests the proxy answered (`mdf_request`, `forwarded` or `error`;
c7.6), for `probe-negative`.

`Scheduler` (c7.14) reads the system lines of a Modal app without `-f` while a driver waits for
readiness: on 20/11 the stream `-f` never delivered Modal's "waiting to be scheduled on a GPU_L4
worker" (the SDK shows it only in its progress bar), so the wait went on with no container at all.
Such a line followed by no container for NO_CAPACITY_S stops the wait with `modal_no_capacity`.
`gpus()` gives the GPUs the proxies named in the log of a window (`mdf_gpu`).

`Lives` (c7.17) reads the containers of a Modal window from the same file, for the drivers: on 20/11
(`concurrency-2`) a container lived 11 s between two of their readings, which come every 30 s,
while the follower listed it every 5 s and its proxy wrote its lines. Each container has the moment
it was first seen and the one it was listed GONE, its GPU, the first answer of vLLM through its
proxy and its inference lines; two containers alive at once are an overlap, with its evidence. With a snapshot
(MFL045 c2) the access lines of vLLM written by the warm-up of `snap_boot`, before the snapshot, are `internal`:
no answer through the proxy (which does not exist yet) and no inference; the kind of the start, the fallback and
the wake of each container are kept.

`Workers` (c7.19) reads the workers of a RunPod window from the same file, for the drivers and the
session: their states as listed every 10 s, when each was first RUNNING, the first answer of vLLM
and the inference lines of each, the "stop container" lines, the listings with two workers RUNNING
(on 19/11 RunPod listed an "extra worker" next to the one that ran, never RUNNING: it is recorded,
not a stop). `bound_facts()` gives the session the stop of the containers and the first moment two
workers were RUNNING at once, for the bound of the window.
"""
import json
import os
from pathlib import Path
import re
import subprocess
import threading
import time
from datetime import datetime, timezone

LOG_ENV = "SESSION_WORKER_LOG"
LIST_EVERY_S = 10
READ_TIMEOUT_S = 20
RECONNECT_S = 1                        # after a quiet stream or its end
RETRY_S = 5                            # after an error, or a 404 for a worker still listed
BACKFILL_S = 60
STALL_S = 120
STALL_RX = 1 << 20                     # received bytes that count as progress of the download
VLLM_ERROR = re.compile(r"(?:^|\)\s+)(?:ERROR|CRITICAL)\s+\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}")
VLLM_PID = re.compile(r"\bpid=(\d+)\)")
# MFL045 c2, P-7: an error of the adopted vLLM held during the wake (`watchdog.WAKE_S` plus a margin), and a line
# of a container whose restore is not in the file yet held so long for it
WAKE_HOLD_S = 180.0
PENDING_S = 60.0
# The SDK of Modal when the restore of a GPU snapshot fails (`task_lifecycle_manager.py`, exit 222)
RESTORE_FAILED = re.compile(r"GPU memory snapshot restore failed")


def iso(t, ms=False):
    d = datetime.fromtimestamp(t, timezone.utc)
    return d.strftime("%Y-%m-%dT%H:%M:%S") + (".%03dZ" % (d.microsecond // 1000) if ms else "Z")


class Gone(Exception):
    """The worker is not there any more (404)."""


class RunPodLogs:
    """The REST v2 seen by the follower; `client(timeout)` makes an authorized httpx client (each
    thread has its own)."""

    def __init__(self, rest, endpoint_id, client):
        self.rest, self.endpoint_id, self.client = rest, endpoint_id, client
        self._list = None

    def workers(self):
        import httpx
        if self._list is None:
            self._list = self.client(httpx.Timeout(30.0, connect=10.0))
        r = self._list.get("%s/serverless/%s/workers" % (self.rest, self.endpoint_id))
        if r.status_code != 200:
            raise RuntimeError("workers: HTTP %d" % r.status_code)
        return [{"id": w.get("id"), "status": w.get("status"), "gpu": w.get("gpuTypeId"),
                 "data_center": w.get("dataCenterId"), "uptime": w.get("uptimeSeconds")}
                for w in (r.json() or {}).get("workers") or []]

    def stream(self, wid, since, last_id):
        """(event id, data) of the log of `wid`; returns when the server ends the stream."""
        import httpx
        headers = {"Accept": "text/event-stream"}
        params = None
        if last_id:
            headers["Last-Event-ID"] = last_id
        else:
            params = {"since": since}
        url = "%s/serverless/%s/workers/%s/logs" % (self.rest, self.endpoint_id, wid)
        with self.client(httpx.Timeout(READ_TIMEOUT_S, connect=10.0)) as c:
            with c.stream("GET", url, params=params, headers=headers) as r:
                if r.status_code == 404:
                    raise Gone(wid)
                if r.status_code != 200:
                    raise RuntimeError("logs: HTTP %d" % r.status_code)
                eid, data = None, []
                for line in r.iter_lines():
                    if line == "":
                        if data:
                            yield eid, _payload("\n".join(data))
                        eid, data = None, []
                    elif line.startswith("id:"):
                        eid = line[3:].strip()
                    elif line.startswith("data:"):
                        data.append(line[5:].lstrip())
                if data:
                    yield eid, _payload("\n".join(data))


def _payload(text):
    try:
        d = json.loads(text)
        if isinstance(d, dict):
            return d
    except ValueError:
        pass
    return {"source": "?", "line": text, "ts": ""}


class Follower:
    def __init__(self, source, path, t_arm, clock=time.time, list_every=LIST_EVERY_S):
        self.source, self.path, self.t_arm, self.clock = source, Path(path), t_arm, clock
        self.list_every = list_every
        fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
        self.fh = os.fdopen(fd, "a", encoding="utf-8")
        self.lock = threading.Lock()
        self.stopping = threading.Event()
        self.closed = False
        self.readers, self.status, self.listed = {}, {}, set()
        self.lines, self.reconnects, self.errors = 0, 0, {}
        self.lister = None
        self.reconnect_s, self.retry_s = RECONNECT_S, RETRY_S

    def write(self, wid, source, ts, line):
        now = self.clock()
        text = str(line if line is not None else "").replace("\r", "\\r").replace("\n", "\\n").replace("\t", " ")
        with self.lock:
            if self.closed:
                return
            self.fh.write("%s\t+%.1f\t%s\t%s\t%s\t%s\n" % (iso(now, ms=True), now - self.t_arm, wid, source or "?",
                                                          ts or "", text))
            self.fh.flush()
            self.lines += 1

    def note(self, wid, **event):
        self.write(wid, "session", "", json.dumps(event, sort_keys=True))

    def error(self, where, exc):
        key = "%s: %s" % (where, type(exc).__name__)
        with self.lock:
            first = key not in self.errors
            self.errors[key] = self.errors.get(key, 0) + 1
        if first:
            # MFL023 B1 (SOL3-N1): the closed fields of `providers.closed`, never str() of the error: a failed
            # `container list` holds the stderr of the CLI of Modal in its exception (`remote`), in memory only
            fields = _providers().closed(exc)
            self.note("-", event="follower_error", where=where, error=fields.pop("class"), **fields)

    def start(self):
        self.note("-", event="follow", t_arm=iso(self.t_arm, ms=True))
        self.lister = threading.Thread(target=self._list, daemon=True)
        self.lister.start()
        return self

    def _list(self):
        while not self.stopping.is_set():
            self.list_once()
            self.stopping.wait(self.list_every)

    def list_once(self):
        try:
            workers = self.source.workers()
        except Exception as exc:                          # a reading fails: the next one comes
            self.error("workers", exc)
            return
        self.listed = {w.get("id") for w in workers}
        # c7.19: every line of a listing holds the workers RUNNING in it, so two RUNNING at once are
        # read from one line, never from the order in which the changes of a listing are written
        running = sorted(str(w.get("id")) for w in workers if str(w.get("status") or "").upper() == "RUNNING")
        for wid in [x for x in self.status if x not in self.listed and self.status[x] != ("GONE",)]:
            self.status[wid] = ("GONE",)
            self.note(wid, event="worker", status="GONE", running=running)       # no longer listed
        for w in workers:
            wid = w.get("id")
            key = (w.get("status"), w.get("gpu"), w.get("data_center"))
            if self.status.get(wid) != key:
                self.status[wid] = key
                self.note(wid, event="worker", status=w.get("status"), gpu=w.get("gpu"),
                          data_center=w.get("data_center"), uptime=w.get("uptime"), running=running)
            if wid not in self.readers and not self.stopping.is_set():
                t = threading.Thread(target=self._read, args=(wid,), daemon=True)
                self.readers[wid] = t
                t.start()

    def _read(self, wid):
        last, seen, missing = None, set(), False
        while not self.stopping.is_set():
            pause = self.reconnect_s
            try:
                for eid, data in self.source.stream(wid, iso(self.t_arm - BACKFILL_S), last):
                    if eid is not None:
                        if eid in seen:
                            continue
                        seen.add(eid)
                        last = eid
                    self.write(wid, data.get("source"), data.get("ts"), data.get("line"))
                    if self.stopping.is_set():
                        return
            except Gone:
                if wid not in self.listed:                # gone from the list too: its log is over
                    self.note(wid, event="logs_gone")
                    return
                if not missing:                           # listed, no log yet (starting): try again
                    missing = True
                    self.note(wid, event="logs_not_found_yet")
                pause = self.retry_s
            except Exception as exc:                      # a quiet stream times out: resume from `last`
                if type(exc).__name__ != "ReadTimeout":
                    self.error("logs", exc)
                    pause = self.retry_s
            with self.lock:
                self.reconnects += 1
            self.stopping.wait(pause)

    def stop(self, grace=READ_TIMEOUT_S + 5):
        """Stop and close the file; the readers end within the read timeout."""
        self.stopping.set()
        deadline = time.monotonic() + grace
        for t in [self.lister] + list(self.readers.values()):
            if t is not None:
                t.join(max(0.0, deadline - time.monotonic()))
        with self.lock:
            self.closed = True
            self.fh.close()
        return {"lines": self.lines, "workers": sorted(self.readers), "reconnects": self.reconnects,
                "errors": dict(self.errors)}


# --- Modal (c7.8) ----------------------------------------------------------------------------------

MODAL_SOURCES = (("stdout", "container"), ("stderr", "container"), ("system", "system"))
MODAL_LIST_EVERY_S = 5
MODAL_STOP_GRACE_S = 5
# A line of `modal app logs --timestamps --show-container-id`: the ISO time with its offset, the
# container, the text (`_output/rich.py` LineBufferedOutput of modal 1.6.3).
MODAL_LINE = re.compile(r"^(\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:[+-]\d{2}:\d{2}|Z)?) (ta-\S+) ?(.*)$")


def _providers():
    """`providers`, the sibling of this module: from the package, else from sys.path (MFL023 A2)."""
    if __package__:
        from . import providers
    else:
        import providers
    return providers


class ModalLogs:
    """The CLI of the pinned SDK seen by `ModalFollower`: `stream(source)` starts one `modal app logs
    -f`, its stdout a text pipe; `containers()` the containers of the app. Every line of the containers, of
    each source (`stdout`, `stderr`, `system`), comes on the stdout of the CLI (modal 1.6.3,
    `_output/rich.py`: `put_streaming_log`, `Console.out`); its stderr holds only the errors and warnings of
    the CLI, which may repeat text of the service, and goes to /dev/null (MFL023 B1, SOL3-N1): the outcome of
    a stream is its exit, in `logs_stream_end`."""

    def __init__(self, app, environ, containers, popen=subprocess.Popen, auth=None, auth_dir=None):
        if __package__:                 # MFL023 A2
            from . import providers
        else:
            import providers
        self.providers = providers
        self.app, self.containers, self.popen = app, containers, popen
        self.argv = providers.MODAL_SDK + ["app", "logs", app, "-f", "--timestamps", "--show-container-id", "-s"]
        self.env = dict(environ, COLUMNS=providers.MODAL_COLUMNS)
        # MFL051: the keys file with the token of the SDK and where its private files go (`providers.modal_auth`)
        self.auth, self.auth_dir = auth, auth_dir or providers.MODAL_AUTH_DIR

    def stream(self, source):
        """One process of the CLI that follows `source`. With a keys file it gets a credentials file of its own,
        which goes when the process has ended (`AuthStream.wait`): the streams of a follower, the operations of a
        session and a capacity watch never share one. The real CLI never starts without it."""
        if self.auth is None:
            if self.popen is subprocess.Popen:
                raise self.providers.ProviderError("%s missing: no keys file" % ", ".join(self.providers.MODAL_TOKEN))
            return self.popen(self.argv + [source], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
                              env=self.env, stdin=subprocess.DEVNULL, errors="replace")
        import contextlib
        stack = contextlib.ExitStack()
        try:
            extra = stack.enter_context(self.providers.modal_auth(self.auth, self.auth_dir))
            proc = self.popen(self.argv + [source], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
                              env=dict(self.env, **extra), stdin=subprocess.DEVNULL, errors="replace")
        except BaseException:
            stack.close()
            raise
        return AuthStream(proc, stack.close)


class AuthStream:
    """A process of the Modal CLI with the private file of its credentials. Everything is the process's own; the
    file is removed as soon as a `wait` or a `poll` sees that the process has ended."""

    def __init__(self, proc, release):
        self._proc, self._release = proc, release

    def __getattr__(self, name):
        return getattr(self._proc, name)

    def wait(self, timeout=None):
        code = self._proc.wait(timeout=timeout)
        self._release()
        return code

    def poll(self):
        code = self._proc.poll()
        if code is not None:
            self._release()
        return code


class ModalFollower(Follower):
    """The logs and the containers of a Modal app into the file of the session (module docstring)."""

    def __init__(self, source, path, t_arm, clock=time.time, list_every=MODAL_LIST_EVERY_S):
        super().__init__(source, path, t_arm, clock=clock, list_every=list_every)
        self.procs = {}

    def start(self):
        self.note("-", event="follow", t_arm=iso(self.t_arm, ms=True), provider="modal")
        for src, label in MODAL_SOURCES:
            t = threading.Thread(target=self._stream, args=(src, label), daemon=True)
            self.readers[src] = t
            t.start()
        self.lister = threading.Thread(target=self._list, daemon=True)
        self.lister.start()
        return self

    def list_once(self):
        try:
            containers = self.source.containers()
        except Exception as exc:                          # a reading fails: the next one comes
            self.error("containers", exc)
            return
        self.listed = {c.get("id") for c in containers}
        for cid in [x for x in self.status if x not in self.listed and self.status[x] != ("GONE",)]:
            self.status[cid] = ("GONE",)
            self.note(cid, event="worker", status="GONE")
        for c in containers:
            key = (c.get("status"), c.get("start"))
            if self.status.get(c.get("id")) != key:
                self.status[c.get("id")] = key
                self.note(c.get("id"), event="worker", status=c.get("status"), start=c.get("start"))

    def _stream(self, src, label):
        seen, runs = set(), 0
        while not self.stopping.is_set():
            try:
                proc = self.source.stream(src)
            except Exception as exc:
                self.error("logs " + src, exc)
                self.stopping.wait(self.retry_s)
                continue
            with self.lock:
                self.procs[src] = proc
            if self.stopping.is_set():                    # stop() may have missed it
                proc.terminate()
                proc.wait()
                break
            runs += 1
            for raw in proc.stdout:
                text = raw.rstrip("\n")
                m = MODAL_LINE.match(text)
                cid, ts, line = (m.group(2), m.group(1), m.group(3)) if m else ("-", "", text)
                key = (cid, ts, line)
                if runs > 1 and key in seen:              # a stream started again repeats what it had
                    continue
                seen.add(key)
                self.write(cid, label, ts, line)
            code = proc.wait()
            if self.stopping.is_set():
                break
            self.note("-", event="logs_stream_end", source=src, exit=code)
            with self.lock:
                self.reconnects += 1
            self.stopping.wait(self.retry_s)

    def stop(self, grace=MODAL_STOP_GRACE_S):
        """Stop the streams (SIGTERM, then SIGKILL) and the lister, then close the file."""
        self.stopping.set()
        with self.lock:
            procs = list(self.procs.values())
        for p in procs:
            if p.poll() is None:
                p.terminate()
        for p in procs:
            try:
                p.wait(timeout=grace)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait()
        info = super().stop(grace=grace + 15)
        info["workers"] = sorted(c for c in self.status)
        info["streams"] = sorted(self.readers)
        return info


# --- reading the file ------------------------------------------------------------------------------

def parse(text):
    """One line of the file: {recv, since_arm, worker, source, ts, line, event (the proxy's JSON or None)}."""
    parts = text.rstrip("\n").split("\t", 5)
    if len(parts) != 6:
        return None
    recv, since, worker, source, ts, line = parts
    try:
        since_arm = float(since.lstrip("+"))
    except ValueError:
        since_arm = None
    event = None
    if line.startswith("{"):
        try:
            d = json.loads(line)
            if isinstance(d, dict) and isinstance(d.get("event"), str):
                event = d
        except ValueError:
            pass
    return {"recv": recv, "since_arm": since_arm, "worker": worker, "source": source, "ts": ts, "line": line,
            "event": event}


class Tail:
    """The complete lines added to the file since the previous reading."""

    def __init__(self):
        self.offset, self.rest = 0, ""

    def lines(self, path):
        """The new complete lines of `path`; an OSError goes to the caller."""
        with open(path, encoding="utf-8", errors="replace") as fh:
            fh.seek(self.offset)
            data = fh.read()
            self.offset = fh.tell()
        lines = (self.rest + data).split("\n")
        self.rest = lines.pop()
        return lines


REQUEST_KINDS = ("forwarded", "error")


class RequestLines(Tail):
    """The requests the proxy answered, `forwarded` or `error` (its `mdf_request` lines, c7.6), as
    they come in the file: `probe-negative` passes only without any (prompt N4e §4.1)."""

    def __init__(self):
        super().__init__()
        self.found = []

    def feed(self, path):
        """Read what is new in `path`; all the requests found so far. An OSError goes to the caller."""
        for text in self.lines(path):
            e = parse(text)
            ev = e["event"] if e is not None and e["source"] != "session" else None
            if ev is not None and ev.get("event") == "mdf_request" and ev.get("kind") in REQUEST_KINDS:
                self.found.append(dict(ev, since_arm=e["since_arm"], worker=e["worker"]))
        return self.found


class StartRules(Tail):
    """The fast stop of a readiness wait from the lines of the file (prompt N4c §4, N4d §5.5). Read
    only until readiness: the trips that phases A and B provoke come after it."""

    def __init__(self, stall_s=STALL_S, stall_rx=STALL_RX, snapshot=False, clock=None, t_arm=None):
        super().__init__()
        self.stall_s, self.stall_rx = stall_s, stall_rx
        self.mark = None                   # (mono, disk, rx) of the last progress of the download
        self.downloading = False
        self.events = {}                   # the first occurrence of each proxy event, by name
        self.snapshot = snapshot           # MFL045 c2, P-7 (module docstring)
        self.gens = {}                     # container -> the generation of its vLLM since its `mdf_restore`
        self.pending, self.exempt = [], []
        self.latest = None                 # the latest second since the arm in the file
        self.clock, self.t_arm = clock, t_arm      # MFL045 c8: the holds on the clock too (`since_arm`)

    def since_arm(self):
        """The seconds since the arm for the holds of the snapshot: the latest of the lines read and, with a clock and
        the arm of the window (MFL045 c8, S-C1 of the review of the diff), of the clock, so that a hold expires without
        new lines (the follower of Modal writes a container only when its listing changes); None without either."""
        if self.clock is None or self.t_arm is None:
            return self.latest
        now = self.clock() - self.t_arm
        return now if self.latest is None else max(self.latest, now)

    def feed(self, path):
        """Read what is new in `path`; the first reason for a stop, or None. With the snapshot the holds are judged
        also when nothing is new or the file cannot be read (MFL045 c8)."""
        try:
            lines = self.lines(path)
        except OSError:
            if not self.snapshot:
                return None
            lines = []
        for text in lines:
            e = parse(text)
            if e is not None and e["since_arm"] is not None:
                self.latest = e["since_arm"] if self.latest is None else max(self.latest, e["since_arm"])
            reason = self.check(e)
            if reason:
                return reason
        return self.expired() if self.snapshot else None

    def check(self, e):
        if e is None or e["source"] == "session":
            return None
        ev = e["event"]
        if ev is None:
            if e["source"] == "container" and VLLM_ERROR.search(e["line"]):
                if self.snapshot:
                    return self.vllm_error(e)
                return "an error of vLLM: " + e["line"][:300]
            return None
        name = ev["event"]
        self.events.setdefault(name, dict(ev, since_arm=e["since_arm"]))
        if self.snapshot:
            reason = self.snap_event(e, ev)
            if reason:
                return reason
        if name == "mdf_download_start":
            self.downloading, self.mark = True, (ev.get("mono"), ev.get("disk") or 0, ev.get("rx"))
        elif name == "mdf_download_progress" and self.downloading:
            return self.progress(ev)
        elif name == "mdf_download_end":
            self.downloading = False
            if ev.get("exit") != 0:
                return "the download failed: exit %s, timeout %s" % (ev.get("exit"), ev.get("timeout"))
        elif name == "mdf_weights_check" and ev.get("phase") == "after_download" and not ev.get("ok"):
            return "the weights are not verified after the download"
        elif name in ("mdf_trip", "mdf_inert", "mdf_error", "mdf_vllm_exit"):
            return "%s %s" % (name, json.dumps({k: v for k, v in ev.items() if k not in ("t", "mono", "event")},
                                              sort_keys=True))
        return None

    # --- MFL045 c2, P-7: the wake of the vLLM of a snapshot --------------------------------------------------
    def snap_event(self, e, ev):
        name, cid = ev["event"], e["worker"]
        if name == "mdf_snapshot" and not ev.get("ok"):
            return "the snapshot failed in %s: %s" % (short(cid), ev.get("reason"))
        g = self.gens.get(cid)
        if name == "mdf_restore":
            self.gens[cid] = {"pids": {int(x) for x in ev.get("pids") or [] if isinstance(x, int)},
                              "since": e["since_arm"], "t": ev.get("t") or 0.0, "held": [], "state": "waking",
                              "t_fallback": None}
            waiting = [x for x in self.pending if x["container"] == cid]
            self.pending = [x for x in self.pending if x["container"] != cid]
            for item in waiting:
                reason = self.judge(item)
                if reason:
                    return reason
        elif g is not None and g["state"] == "waking" and name == "mdf_fallback":
            g.update(state="fallen", t_fallback=ev.get("t"))
            self.exempt += [dict(x, why="the fallback declared") for x in g["held"]]
            g["held"] = []
        elif g is not None and g["state"] == "waking" and (name == "mdf_ready" or (name == "mdf_wake" and ev.get("ok"))):
            if g["held"]:
                return "an error of vLLM during the wake of %s, then %s without a fallback: %s" % (
                    short(cid), name, g["held"][0]["line"])
            g["state"] = "awake"
        return None

    def vllm_error(self, e):
        pid = VLLM_PID.search(e["line"])
        item = {"container": e["worker"], "since": e["since_arm"], "line": e["line"][:300],
                "pid": int(pid.group(1)) if pid else None, "t": epoch(e["ts"])}
        if e["worker"] not in self.gens:
            self.pending.append(item)                 # its restore may be behind, in the other stream
            return None
        return self.judge(item)

    def judge(self, item):
        """An error of vLLM in a container whose `mdf_restore` is known: held, exempt or the stop."""
        g = self.gens[item["container"]]
        old, t = item["pid"] is not None and item["pid"] in g["pids"], item["t"]
        if old and t is not None and t < int(g["t"]):
            return "an error of vLLM: " + item["line"]                 # before the restore: not of the wake
        if old and g["state"] == "waking":
            g["held"].append(item)
            return None
        if old and g["state"] == "fallen" and t is not None and g["t_fallback"] is not None and t <= g["t_fallback"]:
            self.exempt.append(dict(item, why="the fallback declared"))
            return None
        return "an error of vLLM: " + item["line"]

    def expired(self):
        now = self.since_arm()
        if now is None:
            return None
        for cid, g in self.gens.items():
            if g["state"] == "waking" and g["held"] and now - g["since"] >= WAKE_HOLD_S:
                return "no outcome of the wake of %s within %d s of its restore, after an error of vLLM: %s" % (
                    short(cid), WAKE_HOLD_S, g["held"][0]["line"])
        for item in self.pending:
            if now - item["since"] >= PENDING_S:
                return "an error of vLLM: " + item["line"]
        return None

    def at_readiness(self):
        """At readiness: an error held without a declared fallback, or one still waiting for its restore, is the
        stop (MFL045 c2, P-7); None otherwise."""
        for cid, g in self.gens.items():
            if g["state"] == "waking" and g["held"]:
                return "an error of vLLM during the wake of %s, and no fallback: %s" % (short(cid), g["held"][0]["line"])
        if self.pending:
            return "an error of vLLM: " + self.pending[0]["line"]
        return None

    def progress(self, ev):
        mono, disk, rx = ev.get("mono"), ev.get("disk") or 0, ev.get("rx")
        if mono is None or self.mark is None or self.mark[0] is None:
            self.mark = (mono, disk, rx)
            return None
        m_mono, m_disk, m_rx = self.mark
        moved = disk > m_disk or (rx is not None and m_rx is not None and rx - m_rx >= self.stall_rx)
        if moved:
            self.mark = (mono, disk, rx)
            return None
        if mono - m_mono >= self.stall_s:
            return "the download does not move for %.0f s (disk %s, rx %s)" % (mono - m_mono, disk, rx)
        return None


PopulateRules = StartRules                 # the name of c7.4


def gpus(path):
    """The GPUs the proxies named in the log of a window (their `mdf_gpu` lines, c7.14), in order:
    [{"worker", "names", "source", "since_arm"}]. An unreadable file gives []."""
    found = []
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            for text in fh:
                e = parse(text)
                ev = e["event"] if e is not None and e["source"] != "session" else None
                if ev is not None and ev.get("event") == "mdf_gpu":
                    found.append({"worker": e["worker"], "names": ev.get("names"), "source": ev.get("source"),
                                  "since_arm": e["since_arm"]})
    except OSError:
        return []
    return found


# --- the scheduler of Modal (c7.14) ----------------------------------------------------------------

NO_CAPACITY = "modal_no_capacity"
NO_CAPACITY_S = 300                    # waiting to be scheduled this long without a container: stop
SCHEDULER_EVERY_S = 30                 # a fetch of the system lines at most this often
SCHEDULER_TEXT = 300
# A line of `modal app logs APP --timestamps --show-function-id --show-container-id` fetched without
# `-f` (modal 1.6.3): the time with its offset, then the function and the container when the entry has
# them, then the text. On 20/11: "2026-11-20 09:09:07+01:00 fu-sF4W... Function 'serve' (fu-sF4W...)
# is waiting to be scheduled on a GPU_L4 worker. We are actively working on acquiring more capacity
# for your workload."
MODAL_FETCHED = re.compile(r"^(\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:[+-]\d{2}:?\d{2}|Z)?) "
                           r"(?:(fu-\S+) )?(?:(ta-\S+)(?: |$))?(.*)$")
WAITING = re.compile(r"waiting to be scheduled(?: on (?:an? )?(.+?) worker)?")


def epoch(ts):
    """The epoch seconds of a time of the CLI ("2026-11-20 09:09:07+01:00"); None when it is not one.
    A time without an offset is the local time, as the SDK prints it."""
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()
    except (AttributeError, ValueError):
        return None


class Scheduler:
    """The system lines of a Modal app while a driver waits for readiness (MFL033 c7.14, prompt N4m
    §3.1). `fetch(since)` gives the lines since the arm (`providers.Modal.system_logs`), read at most
    every SCHEDULER_EVERY_S; each new line goes out at once through `emit` (`modal_system`). The first
    "waiting to be scheduled" line sets a deadline NO_CAPACITY_S after its own time (or after the
    moment it was read, when that comes first: the clocks may differ); `verdict()` names the stop when
    the deadline has passed and no container of the app was ever seen. A fetch that fails is recorded
    and is not a criterion. The text of the line and the GPU it names are kept."""

    def __init__(self, fetch, since, clock=time.time, emit=None, limit_s=NO_CAPACITY_S, every=SCHEDULER_EVERY_S,
                 requested=None):
        self.fetch, self.since, self.clock, self.emit = fetch, since, clock, emit
        self.limit_s, self.every, self.requested = limit_s, every, list(requested or [])
        self.seen, self.lines, self.errors = set(), [], []
        self.waiting, self.last, self.reads = None, None, 0

    def due(self):
        return self.last is None or self.clock() - self.last >= self.every

    def read(self):
        """Fetch the system lines once; the new ones go out and the first waiting line is kept."""
        self.last = self.clock()
        self.reads += 1
        try:
            rows = self.fetch(self.since)
        except Exception as exc:                           # a fetch fails: the next one comes
            self.errors.append("%s: %s" % (type(exc).__name__, str(exc)[:200]))
            return
        for raw in rows:
            if raw in self.seen:
                continue
            self.seen.add(raw)
            m = MODAL_FETCHED.match(raw)
            ts, function, container, text = (m.group(1), m.group(2), m.group(3), m.group(4)) if m else (None, None, None, raw)
            line = {"modal_system": len(self.lines) + 1, "t": round(self.clock(), 3), "ts": ts, "function": function,
                    "container": container, "text": text[:SCHEDULER_TEXT]}
            self.lines.append(line)
            if self.emit is not None:
                self.emit(line)
            w = WAITING.search(text)
            if w and self.waiting is None:
                at, seen = epoch(ts), self.clock()
                self.waiting = {"text": text[:SCHEDULER_TEXT], "ts": ts, "gpu": w.group(1), "t_seen": round(seen, 3),
                                "t_from": round(min(at, seen) if at is not None else seen, 3)}

    def deadline(self):
        return None if self.waiting is None else self.waiting["t_from"] + self.limit_s

    def verdict(self, containers, read_at):
        """The reason for the stop, or None: past the deadline, with a reading of the containers made at
        or after it (`read_at`) and no container ever seen (`containers`)."""
        d = self.deadline()
        if d is None or containers or read_at is None or read_at < d or self.clock() < d:
            return None
        return "%s: no container %d s after \"%s\" (requested %s)" % (
            NO_CAPACITY, self.limit_s, self.waiting["text"], "/".join(self.requested) or "?")

    def summary(self):
        return {"reads": self.reads, "lines": len(self.lines), "waiting": self.waiting, "requested": self.requested,
                "deadline": None if self.deadline() is None else round(self.deadline(), 3), "errors": self.errors[-5:]}


# --- the containers of a Modal window (c7.17) ------------------------------------------------------

# The first answer of vLLM through the proxy of a container (`watchdog.py`): its /health, its
# /v1/models, their first 200, or a request forwarded to it; and vLLM's own access line.
VLLM_ANSWERS = ("mdf_vllm_first_health", "mdf_vllm_first_models", "mdf_vllm_models_200", "mdf_ready")
VLLM_ACCESS = re.compile(r'"[A-Z]+ (\S+) HTTP/1\.[01]" (\d{3})')
CHAT_PATH = "/v1/chat/completions"
SYSTEM_CHAT = re.compile(r"POST /v1/chat/completions -> ")
LINE_OVERLAP_S = 2.0                   # the lines of a container this much after another one started: both alive
SHORT = 12


def short(cid):
    """A container id abbreviated as in the diary: `ta-01M7W785K…`."""
    cid = str(cid)
    return cid if len(cid) <= SHORT else cid[:SHORT] + "…"


def _utc(text):
    try:
        return datetime.strptime(text, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=timezone.utc).timestamp()
    except (TypeError, ValueError):
        return None


class Lives(Tail):
    """The containers of a Modal window in the file of the session (MFL033 c7.17, prompt N4p §3.1).

    A container is seen at its first `worker` line of the follower (listed every 5 s) or at its first
    own line (stdout, stderr or system), whichever comes first: on 20/11 a proxy wrote `mdf_start`
    1.4 s before its container was listed. It ends at its `worker` line GONE; a line that comes after
    it (a stream may lag) does not bring it back, only a new listing does. All times are the seconds
    since the arm of the file (`since_arm`) and the UTC time of reception.

    Two containers alive at once (an overlap) are seen in two ways, never inferred from the 5 s of a
    listing alone: a listing that holds both (the follower writes the GONE of a listing before its new
    containers, so a new container listed while another is still listed was listed with it), or a line
    of a container's own output written more than LINE_OVERLAP_S after another container wrote its
    first one (times of the containers: the proxy's `t`, else the time of the CLI, whole seconds).
    A failed reading of the file is kept in `error` and judges nothing."""

    def __init__(self):
        super().__init__()
        self.t_arm, self.error = None, None
        self.order, self.by, self.listed = [], {}, {}
        self.listing_overlaps, self.faults = [], []

    def feed(self, path):
        try:
            lines = self.lines(path)
        except OSError as exc:
            self.error = type(exc).__name__
            return self
        for text in lines:
            self.add(parse(text))
        for life in self.by.values():
            self.refresh(life)
        return self

    def life(self, cid, e, by):
        life = self.by.get(cid)
        if life is None:
            life = self.by[cid] = {"id": cid, "seen": e["since_arm"], "seen_utc": e["recv"], "seen_by": by,
                                   "listed": None, "running": None, "start": None, "gone": None, "gone_utc": None,
                                   "last": e["since_arm"], "gpu": None, "vllm": None, "vllm_by": None, "ready": None,
                                   "inferences": 0, "first_t": None, "last_t": None, "again": 0,
                                   # MFL045 c2: the snapshot; the answers and inferences, judged in `refresh`
                                   "snap": False, "snapshot": None, "snapshot_t": None, "restore": None,
                                   "restore_t": None, "fallback": None, "wake": None, "restore_failed": None,
                                   "internal": 0, "access": [], "proxy_answer": None, "inf_lines": 0}
            self.order.append(cid)
        return life

    @staticmethod
    def refresh(life):
        """MFL045 c2: the first answer of vLLM and the inferences of a container. An access line of vLLM written
        by the warm-up of `snap_boot`, before the snapshot (or before the restore when the snapshot line is
        missing; all of them while neither is known), is internal: not an answer through the proxy."""
        bound = life["snapshot_t"] if life["snapshot_t"] is not None else life["restore_t"]
        internal, external = 0, []
        for t, since, path in life["access"]:
            if life["snap"] and (bound is None or (t is not None and t <= bound)):
                internal += 1
            else:
                external.append((since, path))
        found = [life["proxy_answer"]] if life["proxy_answer"] is not None else []
        if external:
            found.append((external[0][0], "access " + external[0][1]))
        first = min(found, key=lambda x: x[0]) if found else (None, None)
        life["vllm"], life["vllm_by"] = first
        life["internal"] = internal
        life["inferences"] = life["inf_lines"] + sum(1 for _, path in external if path == CHAT_PATH)

    def add(self, e):
        if e is None:
            return
        cid, ev = e["worker"], e["event"]
        if e["source"] == "session":
            if ev is not None and ev.get("event") == "follow" and self.t_arm is None:
                self.t_arm = _utc(ev.get("t_arm"))
            if ev is not None and ev.get("event") == "worker" and cid.startswith("ta-"):
                self.listing(cid, e, ev)
            return
        if not cid.startswith("ta-"):
            return
        life = self.life(cid, e, "line")
        if life["gone"] is None:
            life["last"] = e["since_arm"]
        if e["source"] == "container":
            t = ev.get("t") if ev is not None and isinstance(ev.get("t"), (int, float)) else epoch(e["ts"])
            if t is not None:
                life["first_t"] = t if life["first_t"] is None else min(life["first_t"], t)
                life["last_t"] = t if life["last_t"] is None else max(life["last_t"], t)
        if ev is not None:
            name = ev["event"]
            if name == "mdf_gpu" and life["gpu"] is None:
                life["gpu"] = ev.get("names")
            if name == "mdf_ready" and life["ready"] is None:
                life["ready"] = e["since_arm"]
            if name == "mdf_fault":
                self.faults.append({"container": cid, "since_arm": e["since_arm"], "fault": ev.get("fault")})
            if name == "mdf_request" and ev.get("method") == "POST" and ev.get("path") == CHAT_PATH:
                life["inf_lines"] += 1
            if (name in VLLM_ANSWERS or (name == "mdf_request" and ev.get("kind") == "forwarded")) \
                    and life["proxy_answer"] is None:
                life["proxy_answer"] = (e["since_arm"], name)
            if ev.get("snap_boot") or name in ("mdf_snap_boot", "mdf_snapshot"):     # MFL045 c2
                life["snap"] = True
            if name == "mdf_snapshot" and life["snapshot"] is None:
                life["snapshot"] = {k: ev.get(k) for k in ("ok", "gen", "seconds", "ready_s", "reason")}
                life["snapshot_t"] = ev.get("t")
            elif name == "mdf_restore" and life["restore"] is None:
                life["restore"] = {k: ev.get(k) for k in ("kind", "gen", "age_s", "t_snap")}
                life["restore_t"] = ev.get("t")
            elif name == "mdf_fallback" and life["fallback"] is None:
                life["fallback"] = {"reason": ev.get("reason"), "since_arm": e["since_arm"]}
            elif name == "mdf_wake" and life["wake"] is None:
                life["wake"] = {"ok": ev.get("ok"), "seconds": ev.get("seconds"), "since_arm": e["since_arm"]}
            return
        if e["source"] == "container":
            m = VLLM_ACCESS.search(e["line"])
            if m:
                life["access"].append((epoch(e["ts"]), e["since_arm"], m.group(1)))
            elif RESTORE_FAILED.search(e["line"]) and life["restore_failed"] is None:
                life["restore_failed"] = e["since_arm"]
        elif e["source"] == "system" and SYSTEM_CHAT.search(e["line"]):
            life["inf_lines"] += 1

    @staticmethod
    def answered(life, e, by):
        if life["vllm"] is None:
            life["vllm"], life["vllm_by"] = e["since_arm"], by

    def listing(self, cid, e, ev):
        status = ev.get("status")
        if status == "GONE":
            life = self.by.get(cid)
            if life is not None and life["gone"] is None:
                life["gone"], life["gone_utc"], life["last"] = e["since_arm"], e["recv"], e["since_arm"]
            self.listed[cid] = "GONE"
            return
        new = self.listed.get(cid) in (None, "GONE")
        if new:
            for other, st in self.listed.items():
                if other != cid and st != "GONE":
                    self.listing_overlaps.append({"a": other, "b": cid, "by": "listing", "since_arm": e["since_arm"]})
        self.listed[cid] = status
        life = self.life(cid, e, "listing")
        if new and life["gone"] is not None:                   # listed again after its GONE
            life["again"] += 1
            life["gone"] = life["gone_utc"] = None
        life["last"] = e["since_arm"]
        if life["listed"] is None:
            life["listed"] = e["since_arm"]
        if status == "RUNNING" and life["running"] is None:
            life["running"] = e["since_arm"]
        if ev.get("start") and life["start"] is None:
            life["start"] = ev["start"]

    def overlaps(self):
        """Every overlap with its evidence: the listings, then the lines of the containers (the older
        one, by the time of its first line, still writing after the newer one wrote its first)."""
        found = list(self.listing_overlaps)
        for i, a in enumerate(self.order):
            for b in self.order[i + 1:]:
                la, lb = self.by[a], self.by[b]
                if None in (la["first_t"], lb["first_t"]):
                    continue
                old, new = (la, lb) if la["first_t"] <= lb["first_t"] else (lb, la)
                if old["last_t"] > new["first_t"] + LINE_OVERLAP_S:
                    found.append({"a": old["id"], "b": new["id"], "by": "lines", "a_last_t": round(old["last_t"], 3),
                                  "b_first_t": round(new["first_t"], 3)})
        return found

    def alive(self):
        """The containers not listed GONE, in order."""
        return [c for c in self.order if self.by[c]["gone"] is None]

    def brief(self, cid):
        life = self.by[cid]
        return {"id": short(cid), "container": cid, "seen_s": life["seen"], "seen_by": life["seen_by"],
                "seen_utc": life["seen_utc"], "listed_s": life["listed"], "running_s": life["running"],
                "start": life["start"], "gone_s": life["gone"], "gone_utc": life["gone_utc"], "last_s": life["last"],
                "gpu": life["gpu"], "vllm_s": life["vllm"], "vllm_by": life["vllm_by"], "ready_s": life["ready"],
                "inference_lines": life["inferences"], "listed_again": life["again"],
                "internal_lines": life["internal"], "snapshot": self.snapshot_of(cid)}

    def snapshot_of(self, cid):
        """MFL045 c2: the snapshot of a container: the kind of its start, its generation and age, the fallback, the
        wake, the creation of a snapshot, a failed restore; None for a container without any."""
        life = self.by.get(cid)
        if life is None or not (life["restore"] or life["snapshot"] or life["restore_failed"] is not None):
            return None
        out = dict(life["restore"] or {})
        out.update(fallback=(life["fallback"] or {}).get("reason"), wake=life["wake"], created=life["snapshot"],
                   restore_failed_s=life["restore_failed"])
        return out

    def summary(self):
        return {"containers": [self.brief(c) for c in self.order], "overlaps": self.overlaps(), "faults": self.faults,
                "error": self.error}


# --- the workers of a RunPod window (c7.19) ---------------------------------------------------------

# RunPod's worker states that are not billed ("Billed: No", workers/overview read on 20/11): a worker
# other than the one that runs is admitted only in these (prompt N4s §4.1).
EXTRA_STATES = ("INITIALIZING", "IDLE", "THROTTLED")
STOP_CONTAINER = re.compile(r"^stop container\b")
SYSTEM_KEPT = 8                        # the first system lines of each worker kept in its summary


class Workers(Tail):
    """The workers of a RunPod window in the file of the session (MFL033 c7.19, prompt N4s §4).

    The follower lists the workers every 10 s and writes a `worker` line for each change, with the
    workers RUNNING in that listing (`running`, since c7.19); a line without it (a log of 19/11) is
    judged on the states listed so far. For each worker: when it was first seen (its first line of
    any kind), its states with their times, when it was first RUNNING and listed GONE, its GPU and
    data center, its first system lines ("worker is ready" for the extra workers of 19/11), the first
    answer of vLLM through its proxy (as `Lives`) and its inference lines. Two workers RUNNING in one
    listing are an overlap. The "stop container" lines are kept with their time of reception (UTC
    epoch): the billing runs until the container stops. A failed reading of the file is kept in
    `error` and judges nothing."""

    def __init__(self):
        super().__init__()
        self.t_arm, self.error = None, None
        self.order, self.by, self.status = [], {}, {}
        self.overlaps, self.stops = [], []

    def feed(self, path):
        try:
            lines = self.lines(path)
        except OSError as exc:
            self.error = type(exc).__name__
            return self
        for text in lines:
            self.add(parse(text))
        return self

    def life(self, wid, e, by):
        life = self.by.get(wid)
        if life is None:
            life = self.by[wid] = {"id": wid, "seen": e["since_arm"], "seen_utc": e["recv"], "seen_by": by,
                                   "states": [], "running": None, "gone": None, "last": e["since_arm"], "gpu": None,
                                   "gpu_proxy": None, "data_center": None, "system": [], "vllm": None, "vllm_by": None,
                                   "ready": None,
                                   "inferences": 0, "stop": None, "again": 0}
            self.order.append(wid)
        return life

    def add(self, e):
        if e is None:
            return
        wid, ev = e["worker"], e["event"]
        if e["source"] == "session":
            if ev is not None and ev.get("event") == "follow" and self.t_arm is None:
                self.t_arm = _utc(ev.get("t_arm"))
            if ev is not None and ev.get("event") == "worker" and wid != "-":
                self.listing(wid, e, ev)
            return
        if wid == "-":
            return
        life = self.life(wid, e, "line")
        life["last"] = e["since_arm"]
        if e["source"] == "system":
            if len(life["system"]) < SYSTEM_KEPT:
                life["system"].append({"since_arm": e["since_arm"], "line": e["line"][:120]})
            if STOP_CONTAINER.match(e["line"]):
                life["stop"] = e["since_arm"]
                self.stops.append({"worker": wid, "since_arm": e["since_arm"], "recv": e["recv"],
                                   "t": _utc(e["recv"])})
            return
        if ev is not None:
            name = ev["event"]
            if name == "mdf_gpu" and life["gpu_proxy"] is None:
                life["gpu_proxy"] = ev.get("names")
            if name == "mdf_ready" and life["ready"] is None:
                life["ready"] = e["since_arm"]
            if name == "mdf_request" and ev.get("method") == "POST" and ev.get("path") == CHAT_PATH:
                life["inferences"] += 1
            if name in VLLM_ANSWERS or (name == "mdf_request" and ev.get("kind") == "forwarded"):
                Lives.answered(life, e, name)
            return
        if e["source"] == "container":
            m = VLLM_ACCESS.search(e["line"])
            if m:
                Lives.answered(life, e, "access " + m.group(1))
                if m.group(1) == CHAT_PATH:
                    life["inferences"] += 1

    def listing(self, wid, e, ev):
        status = str(ev.get("status") or "").upper()
        life = self.life(wid, e, "listing")
        life["states"].append([e["since_arm"], status])
        life["last"] = e["since_arm"]
        if status == "GONE":
            life["gone"] = e["since_arm"]
            self.status.pop(wid, None)
        else:
            if life["gone"] is not None:                       # listed again after its GONE
                life["again"] += 1
                life["gone"] = None
            self.status[wid] = status
            if status == "RUNNING" and life["running"] is None:
                life["running"] = e["since_arm"]
            life["gpu"] = ev.get("gpu") or life["gpu"]
            life["data_center"] = ev.get("data_center") or life["data_center"]
        running = ev.get("running")
        by = "listing"
        if not isinstance(running, list):                      # a log of 19/11: the states listed so far
            running, by = sorted(w for w, s in self.status.items() if s == "RUNNING"), "states"
        if len(running) > 1:
            self.overlaps.append({"workers": sorted(str(x) for x in running), "since_arm": e["since_arm"],
                                  "utc": e["recv"], "by": by})

    def running_order(self):
        """The workers that were RUNNING, by the moment they were first listed so."""
        return sorted((w for w in self.order if self.by[w]["running"] is not None), key=lambda w: self.by[w]["running"])

    def states(self, wid):
        return [s for _, s in self.by[wid]["states"]]

    def brief(self, wid):
        life = self.by[wid]
        return {"id": wid, "seen_s": life["seen"], "seen_utc": life["seen_utc"], "seen_by": life["seen_by"],
                "states": life["states"][:20], "running_s": life["running"], "gone_s": life["gone"],
                "last_s": life["last"], "gpu": life["gpu"], "gpu_proxy": life["gpu_proxy"],
                "data_center": life["data_center"],
                "system": life["system"], "vllm_s": life["vllm"], "vllm_by": life["vllm_by"], "ready_s": life["ready"],
                "inference_lines": life["inferences"], "stop_s": life["stop"], "listed_again": life["again"]}

    def summary(self):
        return {"workers": [self.brief(w) for w in self.order], "overlaps": self.overlaps, "stops": self.stops,
                "error": self.error}

    def facts(self):
        """For the bound of the window: the last "stop container" (epoch of its reception) and the
        first moment two workers were RUNNING in one listing (seconds since the arm), or None."""
        stops = [s["t"] for s in self.stops if s["t"] is not None]
        return {"t_stop": max(stops) if stops else None,
                "overlap_since_arm": min((o["since_arm"] for o in self.overlaps if o["since_arm"] is not None),
                                         default=None),
                "stops": len(self.stops), "error": self.error}


def bound_facts(path):
    """`Workers.facts()` of a file; never raises (an unreadable file gives no stop and no overlap)."""
    try:
        return Workers().feed(path).facts()
    except Exception as exc:                                    # the disarm row comes after it
        return {"t_stop": None, "overlap_since_arm": None, "stops": 0, "error": type(exc).__name__}
