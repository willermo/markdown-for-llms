"""The GPU sessions of the product in the worker (MFL023 B4; plan r2 §4.1-§4.5, r3 §4.2-§4.4, §4.6).

One session is one window of the spend ledger of the instance on one GPU preset `P`: it opens at the first job of `P`
(or at a request to arm ahead), serves the jobs of `P` while their queue has any, lingers, then closes. At most one
session at a time; a job of another preset closes the open one first.

    IDLE --(a job of P, or a request to arm P ahead)--> OPENING, in the main thread of the worker:
      1 P enabled and usable (`services.gpu_availability`); the endpoint of P in servers.json; its keys by name from
        the env file of the worker
      2 the ledger, read only: orphan windows (uncertain ones included) -> session_orphaned; live ones -> provider_not_idle
      3 A = min(session cap, provider cap - total); W = floor(A / rate - 60 - Z - tail)
      4 the floor: `servers.term(ep, pages, budget=A, until=t0 + W, now=t0)`, pages = MDF_MAX_PAGES for a job, 1 to
        arm ahead -> server_cap, without a lock, a row or a call
      5 a reading in a child (60 s of budget): the provider at rest; RunPod: the pins of the preset and the URL
      6 the ledger file exists; the durable row `opening`; the lock of the window (fd O_CLOEXEC); under the lock of
        the ledger, read again, the row `arm` and the anchor of the clocks; at once the clock thread, which takes the
        fd, T and the anchor
      7 the arm in a child of the main thread, killed at T, on the stop of the worker or on a request to close
      the main thread being busy in the children of steps 5 and 7, their stop also reads the requests to disarm P made
      since the job (or the request) that opens the session (`DisarmWatch`, MFL023 B10): the opening stops, and from
      step 6 on the session closes (`user_disarm`)
    OPEN --(the queue of P empty)--> LINGER --(a job of P)--> OPEN
      the requests to close: T; the end of LINGER (idle_s after the last job, or, armed ahead, the window that can no
      longer host one page); a disarm; the stop; Modal without capacity; an interval not observed (drift > 5 s)
    CLOSING, by the clock thread alone: the arm child reaped (<= 10 s), the disarm and the zero with a budget of their
      own (600 s of the monotonic clock from the start of CLOSING, also after T), the worker log scanned and saved, then
      `serverspend.closing`: certifiable -> the ordinary `disarm` row (`closed`); not certifiable -> a `disarm` row with
      `orphan: true` and `uncertain` (`uncertain`, the provider at its cap until a reconciliation with the invoice);
      no zero within the budget -> STOPPED: no row, the window orphan.

Discipline between the threads (plan r3 §4.2.1): `SessionState` holds the phase and the reason of the close; anyone
asks for the close, only the clock thread runs CLOSING and writes the final states. The rows of `gpu_sessions` change
by compare-and-set on the state (`transition`), so a late write never undoes a final state. Each child process has one
owner, which starts, waits for, kills and reaps it. No lock of the ledger or of the database is held during a call to a
provider, a wait or a join; SQLite connections are opened in the thread that uses them. Signals are handled by the main
thread only (the CLI); this module installs none.

Recovery (plan r3 §4.4, SOL3-N2 of the review of r3), before the first claim of the worker: the rows of `gpu_sessions`
that are not `closed` are closed from their own references (ledger, provider, resource, env file), never from the
settings of today; a ledger recorded but missing, a key missing, or no zero is `gpu_unrecoverable`: the worker stops
before any claim, with the commands that close the window by hand.

Every call to a provider goes through `opchild` (a child with its own group and limits) and comes back in closed
fields; the diary of a session (`<ledger dir>/worker-logs/<arm_id>/session.json`) holds events, times and closed fields
only. The keys stay in the worker as `envfile.Secret`, for the relay and the follower of the worker logs. The follower
writes into the private folder `<ledger dir>/worker-logs/<arm_id>/partial/` (0700, never a link); `keep_log` scans the
log, saves it and removes the folder; after a SIGKILL the recovery of that session removes the folder unread and keeps
nothing of the partial log (`Manager.recover_log`, MFL023 B10 and B12).
"""
from decimal import Decimal, ROUND_FLOOR
import json
import os
from pathlib import Path
import shutil
import stat
import threading
import time

from ..bench import secretscan, serverspend, spend
from ..bench.codes import BenchError
from ..engine import servers
from ..engine.codes import Code, EngineError
from . import codes, errors, opchild, providers

PHASES = ("opening", "open", "linger", "closing", "closed", "uncertain", "orphan", "stopped")
FINAL = frozenset({"closed", "uncertain", "orphan", "stopped"})
SERVING = frozenset({"open", "linger"})
LIVE = frozenset({"opening", "open", "linger"})
STEP_S = 0.25                       # the clock thread reads its clocks at least this often
REAP_WAIT_S = 10                    # CLOSING waits so long for the arm child to be reaped
STALE_REQUEST_S = 600               # a request to arm ahead older than this is refused (`stale`)
CLEANUP_CALL_S = {"runpod": {"disarm": 60, "zero": 60, "restore_flashboot": 60},
                  "modal": {"disarm": 300, "zero": 120, "restore_flashboot": 60}}
WAITING_EVERY_S = 30                # Modal: the system lines read at most this often (worker_logs.SCHEDULER_EVERY_S)
NO_CAPACITY_S = 300                 # Modal: waiting to be scheduled this long without a container (c7.14)
# The reasons of an ordinary close: the window can be clean.
ORDINARY = frozenset({"idle", "arm_ahead_expired", "local_job", "another_preset", "user_disarm", "term", "stop"})
# The closed code a job gets when the session it ran on closed under it.
JOB_CODES = {"term": Code.SERVER_DEADLINE.value, "unobserved_interval": Code.SERVER_DEADLINE.value,
             "modal_no_capacity": codes.MODAL_NO_CAPACITY, "user_disarm": "cancelled", "stop": "interrupted"}
NO_CAPACITY_TEXT = "Modal had no GPU: retry later or choose the preset Batch (RunPod)"
DISARM_READ_S = 1.0                 # OPENING: the requests to disarm read at most this often by the stop of its children
DISARM_READ_TIMEOUT_S = 0.5         # ... on a connection that waits at most this long for a lock (MFL023 B10)
PARTIAL = "partial"                 # the private folder of the worker log while the follower writes it (MFL023 B10)


class Refusal(Exception):
    """A session that does not open (or no longer serves) for a job: a closed code and a local detail."""

    def __init__(self, code, detail=None):
        super().__init__(code if detail is None else "%s: %s" % (code, detail))
        self.code, self.detail = code, detail


class Unrecoverable(Exception):
    """A window left by a dead worker that this start cannot close (`gpu_unrecoverable`), with the commands."""

    def __init__(self, detail, commands):
        super().__init__(detail)
        self.detail, self.commands = detail, commands


class Clocks:
    """The clocks of the sessions: the wall clock (epoch) and the monotonic one; `sleep`, a pause in which time passes
    (between two readings of the zero), and `idle`, the wait of the clock thread for something else to happen. The
    tests give their own, the wall clock and the monotonic one apart (a suspend moves the first alone)."""

    def __init__(self, wall=time.time, mono=time.monotonic, sleep=time.sleep, idle=time.sleep):
        self.wall, self.mono, self.sleep, self.idle = wall, mono, sleep, idle


class SessionState:
    """The phase of one session and the reason of its close (plan r3 §4.2.1): the first request fixes the reason."""

    def __init__(self):
        self.lock = threading.Lock()
        self.phase = "opening"
        self.reason = None
        self.closing = threading.Event()

    def request_close(self, reason):
        with self.lock:
            if self.reason is None and self.phase in LIVE:
                self.reason = reason
            self.closing.set()

    def move(self, new, allowed):
        """`phase` from one of `allowed` to `new`, never while a close is asked for (except into CLOSING and the
        final states, which only the clock thread writes)."""
        with self.lock:
            if self.phase not in allowed:
                return False
            if self.closing.is_set() and new in LIVE:
                return False
            self.phase = new
            return True

    def serving(self):
        with self.lock:
            return self.phase in SERVING and not self.closing.is_set()

    def snapshot(self):
        with self.lock:
            return self.phase, self.reason


def transition(storage, arm_id, state, allowed, **fields):
    """The row of `gpu_sessions` to `state` from one of `allowed` (compare-and-set), with `fields`; True when it
    changed. The names of `fields` are those of this module."""
    sets = "".join(", %s=?" % k for k in fields)
    with storage.connect() as conn:
        n = conn.execute("UPDATE gpu_sessions SET state=?, updated=?%s WHERE arm_id=? AND state IN (%s)"
                         % (sets, ",".join("?" * len(allowed))),
                         (state, time.time(), *fields.values(), arm_id, *allowed)).rowcount
    return n == 1


def cents(value):
    return Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_FLOOR)


def window_seconds(a_cents, rate, z, tail):
    """W = floor(A / rate - 60 s - Z - tail), as `session.py` (`session.window_seconds`)."""
    w = (Decimal(a_cents) / 100 / Decimal(str(rate)) - servers.CLOCK_MARGIN_S - z - Decimal(tail))
    return int(w.to_integral_value(rounding=ROUND_FLOOR))


def levers_of(preset):
    """The levers of the arm of a preset: its compile caches, in the order of the watchdog; no other lever ever."""
    text = ",".join("compile-cache:" + sha for sha in preset["levers"]["compile_cache"])
    return providers.levers_text(text, preset["provider"])


def endpoint_of(servers_path, name, table):
    """The endpoint of servers.json that names the GPU preset `name` (one at most: `servers.load` refuses two), expanded
    from the preset; Refusal `server_not_configured`."""
    try:
        doc = json.loads(Path(servers_path).read_bytes())
        entries = doc["servers"]
        found = [k for k, v in entries.items() if isinstance(v, dict) and v.get("preset") == name]
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        raise Refusal(Code.SERVER_NOT_CONFIGURED.value, "the servers file of the instance") from None
    if len(found) != 1:
        raise Refusal(Code.SERVER_NOT_CONFIGURED.value, "no endpoint of servers.json names the GPU preset")
    try:
        return servers.load(servers_path, found[0], presets=table)
    except EngineError as exc:
        raise Refusal(exc.code.value, "the endpoint of the GPU preset") from None


def minimum_reserve(ep, pages):
    """The smallest A in cents with which a session opens for `pages` (the floor of OPENING, step 4): the reserve shown
    as the minimum of a session (`minimum_reserve_cents`), never a charge. None when no A under 100 USD opens one."""
    z, tail = providers.ZERO_LATENCY_S[ep.provider], providers.BILLING_TAIL_S.get(ep.provider, 0)

    def opens(hundredths):                       # A in hundredths of a cent: the cents of the ledger have two decimals
        a = Decimal(hundredths) / 100
        w = window_seconds(a, ep.rate, z, tail)
        if w <= 0:
            return False
        try:
            servers.term(ep, pages, budget_cents=str(a), until=1_000_000_000 + w, now=1_000_000_000)
        except EngineError:
            return False
        return True
    low, high = 1, 1_000_000                     # up to 100 USD
    if not opens(high):
        return None
    while low < high:
        mid = (low + high) // 2
        if opens(mid):
            high = mid
        else:
            low = mid + 1
    return str(Decimal(low) / 100)


def commands(provider, resource, ledger_path, arm_id=None, stopped=False):
    """The commands that close a window by hand (`services.gpu_commands`, the same for the API)."""
    from ..services import gpu_commands
    return gpu_commands(provider, resource, ledger_path, arm_id, stopped)


def private_dir(path):
    """`path` as a private folder of this user (0700), created when missing; OSError for a link, another type or
    another owner (MFL023 B10)."""
    path = Path(path)
    try:
        path.mkdir(mode=0o700)
    except FileExistsError:
        pass
    info = os.lstat(path)
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid():
        raise OSError("not a private folder: %s" % path.name)
    if stat.S_IMODE(info.st_mode) != 0o700:
        os.chmod(path, 0o700)
    return path


class DisarmWatch:
    """The requests to disarm the preset of a session OPENING, for the stop of its children (MFL023 B10, SOL-B-B1 of the
    review of the diff B): the main thread is busy in them, so `Manager.requests()` cannot read the requests then.

    Only the requests made since `since` (the creation of the job, or of the request to arm ahead, that opens the
    session): an older one cancelled only the jobs that existed when it was made, and `requests()` handles it as
    before. Read at most every DISARM_READ_S (`force` reads now), on a connection that waits at most
    DISARM_READ_TIMEOUT_S for a lock: a busy database is read at the next step, the term T of the child is never held.
    The request found becomes `done` once (compare-and-set) and `seen` stays True; a read that fails is no request."""

    def __init__(self, storage, name, since=0.0):
        self.storage, self.name, self.since = storage, name, since
        self.seen, self._read = False, None

    def __call__(self, force=False):
        if self.seen:
            return True
        now = time.monotonic()
        if not force and self._read is not None and now - self._read < DISARM_READ_S:
            return False
        self._read = now
        try:
            with self.storage.connect(timeout=DISARM_READ_TIMEOUT_S) as conn:
                row = conn.execute("SELECT id FROM gpu_requests WHERE state='pending' AND kind='disarm' AND gpu_preset=? "
                                   "AND created >= ? ORDER BY created LIMIT 1", (self.name, self.since)).fetchone()
                if row is not None:
                    conn.execute("UPDATE gpu_requests SET state='done', updated=? WHERE id=? AND state='pending'",
                                 (time.time(), row["id"]))
        except Exception:                                 # noqa: BLE001 - the stop of a child never raises
            return False
        self.seen = row is not None
        return self.seen


class Session:
    """One GPU session (module docstring). Built and opened by `Manager.open` in the main thread."""

    def __init__(self, manager, name, preset, ep, keys, origin):
        self.m, self.name, self.preset, self.ep, self.keys, self.origin = manager, name, preset, ep, keys, origin
        self.provider, self.resource = ep.provider, ep.resource
        self.url = servers.canonical_url(ep.url)
        self.ledger = spend.Ledger.file(manager.ledger_path)
        self.arm_id = serverspend.new_arm_id()
        self.state = SessionState()
        self.row = self.fd = self.until = self.t_arm = self.mono_arm = None
        self.thread = None
        self.arm_reaped = threading.Event()
        self.armed_ok = False
        self.endpoint_id = None
        self.max_drift = 0.0
        self.diary = []
        self.jobs = 0
        self.last_job = None
        self.seen_container = False
        self.waiting = {"last": None, "at": None, "first_seen": None}
        self.follower = None
        self.a_cents = None
        self.w = None
        self.t_zero = None

    # --- the diary -------------------------------------------------------------------------------------

    def note(self, event, **fields):
        self.diary.append(dict(event=event, t=round(self.m.clocks.wall(), 3), **fields))

    def save_diary(self):
        try:
            d = Path(self.m.ledger_dir) / "worker-logs" / self.arm_id
            d.mkdir(parents=True, exist_ok=True, mode=0o700)
            phase, reason = self.state.snapshot()
            (d / "session.json").write_text(json.dumps({"arm_id": self.arm_id, "gpu_preset": self.name,
                                                        "provider": self.provider, "phase": phase, "reason": reason,
                                                        "events": self.diary}, sort_keys=True, indent=1))
        except OSError:
            pass

    # --- the clocks -----------------------------------------------------------------------------------

    def sample(self):
        """One reading of the clocks: the largest drift observed (by its size, never the last difference: a drift
        that goes back does not undo an interval not observed), and the requests to close at T and on a drift over
        the limit. False, so that it serves as the `stop` of a child that must not be stopped."""
        if self.t_arm is None:
            return False
        wall, mono = self.m.clocks.wall(), self.m.clocks.mono()
        drift = (wall - self.t_arm) - (mono - self.mono_arm)
        if abs(drift) > abs(self.max_drift):
            self.max_drift = drift
        if abs(drift) > servers.DRIFT_LIMIT_S:
            self.state.request_close("unobserved_interval")
        elif wall >= self.until:
            self.state.request_close("term")
        return False

    def pause(self, seconds):
        """A pause of `seconds` of the monotonic clock, the clocks read at every step."""
        end = self.m.clocks.mono() + seconds
        while self.m.clocks.mono() < end:
            self.m.clocks.sleep(STEP_S)
            self.sample()

    # --- children -------------------------------------------------------------------------------------

    def child(self, op, args=None, **limits):
        r = opchild.call(op, self.provider, self.resource, self.m.env_file, args or {}, argv=self.m.argv,
                         wall=self.m.clocks.wall, mono=self.m.clocks.mono, also=self.ep.key_env, **limits)
        if not r["ok"]:
            self.note(op + "-error", **(r["error"] or {}))
        return r

    # --- the clock thread -----------------------------------------------------------------------------

    def start_clock(self):
        self.thread = threading.Thread(target=self._clock, name="gpu-session-clock", daemon=True)
        self.thread.start()

    def _clock(self):
        try:
            while not self.state.closing.is_set():
                self.sample()
                self.watch_capacity()
                if self.state.closing.is_set():
                    break
                self.m.clocks.idle(STEP_S)
            self.close()
        except BaseException as exc:                     # noqa: BLE001 - recorded; the recovery closes the window
            self.note("clock-error", **providers.closed(exc))
            self.save_diary()

    def watch_capacity(self):
        """Modal (c7.14): a "waiting to be scheduled" line and no container for NO_CAPACITY_S closes the session
        (`modal_no_capacity`); no automatic change of provider."""
        if self.provider != "modal" or self.seen_container or not self.armed_ok:
            return
        now = self.m.clocks.wall()
        if self.waiting["last"] is not None and now - self.waiting["last"] < WAITING_EVERY_S:
            return
        self.waiting["last"] = now
        r = self.child("waiting", {"since": int(self.t_arm)}, budget_s=120,
                       stop=lambda: self.sample() or self.state.closing.is_set())
        if not r["ok"]:
            return
        if r["result"].get("containers"):
            self.seen_container = True
            return
        at = r["result"].get("waiting_at")
        if at is not None and self.waiting["at"] is None:
            self.waiting.update(at=float(at), first_seen=now)
            self.note("modal-waiting", waiting_at=at)
        if self.waiting["at"] is not None and now >= min(self.waiting["at"], self.waiting["first_seen"]) + NO_CAPACITY_S:
            self.note("modal-no-capacity", seconds=NO_CAPACITY_S)
            self.state.request_close("modal_no_capacity")

    def close(self):
        """CLOSING (module docstring), in the clock thread only."""
        m, st = self.m, self.state
        st.move("closing", LIVE)
        phase, reason = st.snapshot()
        transition(m.storage, self.arm_id, "closing", tuple(LIVE), reason=reason)
        self.note("closing", reason=reason)
        self.arm_reaped.wait(REAP_WAIT_S)
        started = m.clocks.mono()
        t_disarm = m.clocks.wall()
        zero, readings = self.disarm_and_zero(started)
        log_info = self.keep_log()
        if zero is not None and (readings[-1]["state"] or {}).get("flashboot") not in (None, "OFF"):
            left = providers.ZERO_STOP_S - (m.clocks.mono() - started)
            if left > 0:                                  # MFL049 (a): only on a reading explicitly not OFF
                self.child("restore_flashboot", budget_s=min(CLEANUP_CALL_S[self.provider]["restore_flashboot"], left),
                           stop=self.sample)
        self.sample()
        if zero is None:
            self.note("stopped", detail="no zero within %d s: the window stays orphan" % providers.ZERO_STOP_S)
            st.move("stopped", {"closing"})
            transition(m.storage, self.arm_id, "stopped", ("closing",), reason="zero_missing",
                       drift_s=round(self.max_drift, 3))
            os.close(self.fd)
            self.save_diary()
            return
        self.t_zero = round(zero, 3)
        latency = round(zero - t_disarm, 3)
        extra = {"t_disarm": round(t_disarm, 3), "zero_latency_s": latency, "readings": readings[-60:],
                 "reason": reason, "product": True}
        t_stop = double = None
        if self.provider == "runpod":
            from .session import Session as CliSession
            bound = CliSession.bound_extra(self.row, zero, readings, log_info)
            extra.update(bound)
            t_stop, double = bound["t_stop"], bound["double_s"]
        if log_info is not None:
            extra["worker_log"] = {k: log_info.get(k) for k in ("saved", "sha256", "lines", "workers", "gpus")}
        uncertain = serverspend.closing(self.row, self.t_zero, t_stop, double or 0, self.max_drift)
        clean = (self.armed_ok and reason in ORDINARY and not uncertain
                 and latency <= providers.ZERO_LATENCY_S[self.provider])
        if uncertain:
            extra.update(uncertain=uncertain, drift_s=round(self.max_drift, 3))
            self.note("uncertain", reason=uncertain, drift_s=round(self.max_drift, 3))
        with self.ledger.locked() as locked:
            serverspend.disarm(locked, self.provider, self.arm_id, self.t_zero, clean, orphan=bool(uncertain),
                               extra=extra)
        try:
            serverspend.lock_path(self.ledger.path, self.arm_id).unlink()
        except OSError:
            pass
        os.close(self.fd)
        final = "uncertain" if uncertain else "closed"
        limit_at = float(self.row["until"]) + float(servers.CLOCK_MARGIN_S) + providers.ZERO_LATENCY_S[self.provider]
        st.move(final, {"closing"})
        transition(m.storage, self.arm_id, final, ("closing",), reason=uncertain or reason, t_zero=self.t_zero,
                   limit_at=round(limit_at, 3), drift_s=round(self.max_drift, 3))
        self.note(final, clean=clean, zero_latency_s=latency)
        self.save_diary()

    def disarm_and_zero(self, started):
        """(t_zero or None, readings): the disarm, again while a reading shows the provider armed, and the zero read
        every ZERO_POLL_S, within ZERO_STOP_S of the monotonic clock from `started`; each child with its own budget.
        The children of the cleanup are never stopped by the stop of the worker."""
        m = self.m
        readings, disarmed = [], False
        while True:
            left = providers.ZERO_STOP_S - (m.clocks.mono() - started)
            if left <= 0:
                return None, readings
            calls = CLEANUP_CALL_S[self.provider]
            if not disarmed:
                disarmed = self.child("disarm", budget_s=min(calls["disarm"], left), stop=self.sample)["ok"]
            left = providers.ZERO_STOP_S - (m.clocks.mono() - started)
            if disarmed and left > 0:
                r = self.child("zero", budget_s=min(calls["zero"], left), stop=self.sample)
                if r["ok"]:
                    state = r["result"].get("state") or {}
                    readings.append({"t": round(m.clocks.wall(), 3), "state": state})
                    if r["result"].get("zero"):
                        return m.clocks.wall(), readings
                    if state.get("armed"):
                        disarmed = False                          # armed again: disarm again
            if providers.ZERO_STOP_S - (m.clocks.mono() - started) <= 0:
                return None, readings
            self.pause(providers.ZERO_POLL_S)

    # --- the follower of the worker logs -------------------------------------------------------------

    def start_follower(self):
        """The follower writes into `<ledger dir>/worker-logs/<arm_id>/partial/` (MFL023 B10, SOL-B-R1 in part): a
        private folder of the session, never the temporary directory of the system, so that a SIGKILL leaves nothing
        outside the ledger and the recovery of this session finds it and removes it (`Manager.recover_log`)."""
        make = self.m.follower
        if make is None:
            return
        tmp = None
        try:
            folder = private_dir(private_dir(Path(self.m.ledger_dir) / "worker-logs") / self.arm_id)
            tmp = folder / PARTIAL
            tmp.mkdir(mode=0o700)                         # a new folder: an existing one (or a link) is refused
            tmp = private_dir(tmp)
            self.follower = make(self, tmp / "worker-log.partial")
            if self.follower is not None:
                self.follower.start()
                self.note("worker-log")
        except Exception as exc:                          # the session goes on without its log
            self.note("worker-log-error", **providers.closed(exc))
            self.follower = None
        if self.follower is None and tmp is not None and tmp.is_dir() and not tmp.is_symlink():
            shutil.rmtree(tmp, ignore_errors=True)

    def keep_log(self):
        """Stop the follower, scan the log for the keys of the env file, then save it under the ledger (or drop it).
        Never raises: the `disarm` row comes after it."""
        f = self.follower
        if f is None:
            return None
        src = Path(f.path)
        try:
            info = f.stop()
            if self.provider == "runpod":
                from . import worker_logs
                info["bound"] = worker_logs.bound_facts(src)
            try:
                keys = secretscan.keys_bytes(self.m.env_file)
                read, errs, found = secretscan.scan_all([str(src)], keys)
            except BenchError as exc:
                read, errs, found = 0, [exc.code.value], []
            if found or errs or read != 1:
                info.update(saved=None, secret_found=sum(n for _, n in found), scan_errors=len(errs))
                self.note("worker-log-withheld", secret_found=info["secret_found"], scan_errors=len(errs))
                return info
            dest = Path(self.m.ledger_dir) / "worker-logs" / self.arm_id / "worker-log.txt"
            dest.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            shutil.copyfile(src, dest)
            from . import worker_logs
            import hashlib
            info.update(saved=str(dest), sha256=hashlib.sha256(dest.read_bytes()).hexdigest(), gpus=worker_logs.gpus(dest))
            self.note("worker-log-saved", lines=info.get("lines"))
            return info
        except Exception as exc:
            self.note("worker-log-error", **providers.closed(exc))
            return {"saved": None, "error": type(exc).__name__}
        finally:
            shutil.rmtree(src.parent, ignore_errors=True)


def real_follower(session, path):
    """The follower of the worker logs of the session (`worker_logs`), in the worker: RunPod by the endpoint id of the
    reading at rest, with its key in the headers of its own clients; Modal with the CLI of the SDK."""
    from . import worker_logs
    import httpx
    if session.provider == "runpod":
        if not session.endpoint_id:
            return None
        key = providers.read_key(session.m.env_file, "RUNPOD_API_KEY")

        def client(timeout):
            return httpx.Client(timeout=timeout, headers={"Authorization": "Bearer " + key.reveal()},
                                follow_redirects=False)
        return worker_logs.Follower(worker_logs.RunPodLogs(providers.REST, session.endpoint_id, client), path,
                                    session.t_arm)
    environ = providers.child_env(os.environ, also=session.ep.key_env)
    modal = providers.Modal(session.resource, environ=environ)
    return worker_logs.ModalFollower(worker_logs.ModalLogs(session.resource, environ, modal.app_containers), path,
                                     session.t_arm)


class Manager:
    """The GPU sessions of one worker (module docstring). `argv`: the command of the children (`opchild`, a driver in
    the tests); `clocks`; `follower(session, path)`: the follower of the worker logs, or None."""

    def __init__(self, worker, argv=None, clocks=None, follower=real_follower):
        self.worker, self.storage, self.settings = worker, worker.storage, worker.settings
        self.env_file = worker.env_file
        self.argv, self.clocks, self.follower = argv, clocks or Clocks(), follower
        self.ledger_dir = str(Path(self.settings.gpu_ledger_dir))
        self.ledger_path = str(Path(self.ledger_dir) / spend.LEDGER)
        self.session = None
        self.closed_reason = {}                         # arm_id -> reason, for the jobs it served
        self._checked = 0.0

    # --- opening ---------------------------------------------------------------------------------------

    def stopping(self):
        return bool(self.worker.stop)

    def open(self, name, origin, since=0.0):
        """OPENING (module docstring); returns the session OPEN, or raises Refusal with a closed code. `since`: the
        creation of the job or of the request that opens it; a request to disarm `name` made since then stops the
        opening (`DisarmWatch`)."""
        from ..services import gpu_availability
        from . import presets as gpu_presets
        from .session import take_lock
        avail = gpu_availability(name, self.settings)
        if not avail["available"]:
            raise Refusal(avail["code"])
        table = gpu_presets.load()
        preset = table[name]
        ep = endpoint_of(self.settings.servers, name, gpu_presets.endpoint_table(table))
        try:
            keys = servers.read_keys(ep, self.env_file)
        except EngineError as exc:
            raise Refusal(exc.code.value, "the keys of the GPU preset in the env file of the worker") from None
        s = Session(self, name, preset, ep, keys, origin)
        # step 2: the ledger, read only
        Path(self.ledger_dir).mkdir(parents=True, exist_ok=True, mode=0o700)
        try:
            with s.ledger.locked():                      # the ledger exists before the durable row (note of Sol)
                pass
            summary = serverspend.summary(spend.read_only(self.ledger_path), s.provider, self.ledger_path)
        except BenchError as exc:
            raise Refusal(exc.code.value, "the spend ledger of the instance") from None
        if summary["orphans"]:
            raise Refusal(Code.SESSION_ORPHANED.value, "orphan windows of %s" % s.provider)
        if summary["over_bound"]:
            raise Refusal(Code.SERVER_BILLED_OVER_BOUND.value)
        if any(w["status"] == "open" for w in summary["windows"]):
            raise Refusal(codes.PROVIDER_NOT_IDLE, "a live window of %s" % s.provider)
        # step 3: A and W
        cap = cents(Decimal(getattr(self.settings, "gpu_cap_usd_" + s.provider)) * 100)
        a = min(cents(Decimal(self.settings.gpu_session_cap_usd) * 100), cap - Decimal(summary["total_cents"]))
        a = cents(a)
        z, tail = providers.ZERO_LATENCY_S[s.provider], providers.BILLING_TAIL_S.get(s.provider, 0)
        w = window_seconds(a, ep.rate, z, tail) if a > 0 else 0
        if a <= 0 or w <= 0:
            raise Refusal(Code.SERVER_CAP.value, "nothing left under the caps of %s" % s.provider)
        # step 4: the floor, before any lock, row or call
        pages = self.settings.max_pages if origin == "job" else 1
        t0 = self.clocks.wall()
        try:
            servers.term(ep, pages, budget_cents=str(a), until=t0 + w, now=t0)
        except EngineError:
            raise Refusal(Code.SERVER_CAP.value, "A = %s cents is under the minimum reserve of the session" % a) from None
        s.a_cents, s.w = a, w
        asked = DisarmWatch(self.storage, name, since)
        # step 5: the reading at rest
        r = s.child("rest", {"preset": preset} if s.provider == "runpod" else {}, budget_s=opchild.READ_BUDGET_S,
                    stop=lambda: self.stopping() or asked())
        if not r["ok"]:
            if asked.seen:
                raise Refusal(JOB_CODES["user_disarm"], "disarmed by the user while the session opened")
            raise Refusal(Code.SERVER_UNAVAILABLE.value, "the reading of %s" % s.provider)
        state = r["result"].get("state") or {}
        if state.get("armed") or state.get("busy"):
            raise Refusal(codes.PROVIDER_NOT_IDLE, "%s %s is armed or busy" % (s.provider, s.resource))
        if s.provider == "runpod":
            if r["result"].get("pins_failed"):
                raise Refusal(codes.ENDPOINT_OUT_OF_PINS, ", ".join(r["result"]["pins_failed"]))
            if r["result"].get("url") != s.url:
                raise Refusal(Code.SERVER_NOT_CONFIGURED.value, "the URL of the endpoint is not the URL of servers.json")
            s.endpoint_id = r["result"].get("endpoint_id")
        if asked(force=True):                             # disarmed before any row: nothing to close
            raise Refusal(JOB_CODES["user_disarm"], "disarmed by the user while the session opened")
        # step 6: the durable row, the lock, the arm row, the clock thread
        now = time.time()
        with self.storage.connect() as conn:
            conn.execute("INSERT INTO gpu_sessions (arm_id, owner, gpu_preset, provider, resource, url, ledger_path, "
                         "env_file, key_env, origin, a_cents, w_s, state, created, updated) VALUES "
                         "(?,?,?,?,?,?,?,?,?,?,?,?,'opening',?,?)",
                         (s.arm_id, "local", name, s.provider, s.resource, s.url, self.ledger_path, self.env_file,
                          json.dumps(list(ep.key_env)), origin, str(a), w, now, now))
        self.session = s
        try:
            s.fd, lock = take_lock(self.ledger_path, s.arm_id)
        except OSError:
            transition(self.storage, s.arm_id, "closed", ("opening",), reason="never_armed")
            self.session = None
            raise Refusal(Code.SERVER_UNAVAILABLE.value, "the lock of the window") from None
        try:
            with s.ledger.locked() as locked:
                again = serverspend.parse(locked.entries()).get(s.provider) or serverspend.Provider(s.provider)
                s2 = again.summary(lambda x: serverspend.alive(self.ledger_path, x))
                if s2["orphans"]:
                    raise Refusal(Code.SESSION_ORPHANED.value)
                if any(x["status"] == "open" and x["arm_id"] != s.arm_id for x in s2["windows"]):
                    raise Refusal(codes.PROVIDER_NOT_IDLE, "another session armed %s" % s.provider)
                s.t_arm, s.mono_arm = round(self.clocks.wall(), 3), self.clocks.mono()
                s.until = s.t_arm + w
                s.row = serverspend.arm(locked, self.ledger_path, s.provider, s.resource, s.url, a, s.until, cap,
                                        ep.rate_usd_per_s, s.t_arm, arm_id=s.arm_id,
                                        extra={"window_s": w, "z_s": z, "tail_s": tail, "levers": levers_of(preset),
                                               "gpu_preset": name, "origin": origin, "product": True})
        except (Refusal, BenchError) as exc:
            try:
                lock.unlink()
            except OSError:
                pass
            os.close(s.fd)
            transition(self.storage, s.arm_id, "closed", ("opening",), reason="never_armed")
            self.session = None
            if isinstance(exc, BenchError):
                raise Refusal(exc.code.value) from None
            raise
        transition(self.storage, s.arm_id, "opening", ("opening",), t_arm=s.t_arm, until=s.until)
        s.note("arm-row", a_cents=str(a), window_s=w, until=s.until, pages=pages)
        s.start_clock()
        # step 7: the arm, killed at T, on the stop, on a request to close or on a request to disarm (`DisarmWatch`)
        args = {"levers": levers_of(preset)}
        if s.provider == "runpod":
            args["preset"] = preset

        def disarmed():
            if not asked():
                return False
            s.state.request_close("user_disarm")          # the clock thread alone runs CLOSING, once
            s.note("disarm-request", during="arm")
            return True
        try:
            r = s.child("arm", args, until=s.until,
                        stop=lambda: self.stopping() or s.state.closing.is_set() or disarmed())
        finally:
            s.arm_reaped.set()
        if not r["ok"]:
            cls = (r["error"] or {}).get("class")
            reason = {"TermReached": "term", "Stopped": "stop"}.get(cls, "arm_failed")
            s.state.request_close(reason)
            fixed = s.state.snapshot()[1]                 # the first request fixed the reason (a disarm, the term...)
            raise Refusal(JOB_CODES.get(fixed) or JOB_CODES.get(reason, Code.SERVER_UNAVAILABLE.value),
                          "the arm of %s" % s.provider)
        if r["result"].get("url") != s.url:
            s.state.request_close("server_not_configured")
            raise Refusal(Code.SERVER_NOT_CONFIGURED.value, "the URL after the arm differs from servers.json")
        s.armed_ok = True
        s.note("armed", **{k: v for k, v in (r["result"].get("info") or {}).items() if not isinstance(v, dict)})
        s.start_follower()
        if not (s.state.move("open", {"opening"}) and transition(self.storage, s.arm_id, "open", ("opening",))):
            raise Refusal(self.job_code(s), "the session closed while it opened")
        return s

    # --- jobs ------------------------------------------------------------------------------------------

    def job_code(self, s):
        _, reason = s.state.snapshot()
        return JOB_CODES.get(reason, codes.SESSION_STOPPED)

    def prepare(self, row, opts, request):
        """The session of the GPU preset of the job, opened when needed, and what the engine needs for its attempt:
        (endpoint, keys, the server options with the budget and the window, the reservation hook). Refusal."""
        name = opts["gpu_preset"]
        s = self.session
        if s is not None and (s.name != name or not s.state.serving()):
            if s.name != name:
                s.state.request_close("another_preset")
            self.finish()
            s = None
        if s is None:
            s = self.open(name, "job", since=row.get("created") or 0.0)
        s.origin = "job"
        if s.state.move("open", {"linger"}):
            transition(self.storage, s.arm_id, "open", ("linger",))
        try:
            info = serverspend.session(self.ledger_path, s.provider, s.url, {serverspend.SESSION_ENV: s.arm_id},
                                       rate=s.ep.rate_usd_per_s)
        except BenchError as exc:
            raise Refusal(exc.code.value, "the window of the session") from None
        srv = dict(request.get("server") or {}, name=s.ep.name, budget_cents=info["window_left_cents"],
                   session={"arm_id": s.arm_id, "armed_until": info["armed_until"]})
        hook = serverspend.Reservation(spend.Ledger.file(self.ledger_path), s.provider, s.arm_id, row["id"],
                                       row["attempt_id"], Decimal(s.a_cents), clock=self.clocks.wall)
        s.jobs += 1
        s.note("job", run_id=row["id"], budget_cents=info["window_left_cents"])
        return s.ep, s.keys, srv, hook

    def done(self, row, hook, result):
        """After the attempt of a job on the session: its completion in the ledger (never twice), the facts of the run
        (`runs.gpu`) and LINGER when the queue of the preset is empty."""
        s = self.session
        try:
            completed = hook.complete(result) if hook is not None else None
        except (BenchError, OSError) as exc:
            completed = None
            if s is not None:
                s.note("complete-error", **providers.closed(exc))
        if s is None:
            return
        s.jobs = max(0, s.jobs - 1)
        s.last_job = self.clocks.mono()
        cost = (result or {}).get("cost") if isinstance((result or {}).get("cost"), dict) else {}
        self.write_gpu(row, s, reserve=(hook.row or {}).get("reserve_cents") if hook is not None else None,
                       bound=cost.get("bound_cents"), completed=completed is not None)
        if not self.queued(s.name) and s.state.move("linger", {"open"}):
            transition(self.storage, s.arm_id, "linger", ("open",))
            s.note("linger")

    def write_gpu(self, row, s, reserve=None, bound=None, completed=False, error=None):
        """`runs.gpu`: the closed facts of the attempt on its session (preset, provider, window, reserve and bound of
        the attempt), in every outcome after the session opened and before the publication."""
        value = {"preset": s.name if s is not None else None, "provider": s.provider if s is not None else None,
                 "arm_id": s.arm_id if s is not None else None, "a_cents": str(s.a_cents) if s and s.a_cents else None,
                 "until": s.until if s is not None else None, "reserve_cents": reserve, "bound_cents": bound,
                 "completed": completed, "error": errors.public(error) if error else None}
        with self.storage.connect() as conn:
            conn.execute("UPDATE runs SET gpu=? WHERE id=?", (json.dumps(value, sort_keys=True), row["id"]))

    def refused(self, row, name, exc):
        """A job refused by its session: its facts, with the closed code."""
        value = {"preset": name, "provider": None, "arm_id": None, "a_cents": None, "until": None,
                 "reserve_cents": None, "bound_cents": None, "completed": False, "error": {"class": "Refusal",
                                                                                           "op": exc.code, "http": None}}
        s = self.session
        if s is not None and s.name == name:
            value.update(provider=s.provider, arm_id=s.arm_id, a_cents=str(s.a_cents) if s.a_cents else None,
                         until=s.until)
        with self.storage.connect() as conn:
            conn.execute("UPDATE runs SET gpu=? WHERE id=?", (json.dumps(value, sort_keys=True), row["id"]))

    def serving(self):
        s = self.session
        return s is not None and s.state.serving()

    def queued(self, name):
        with self.storage.connect() as conn:
            return conn.execute("SELECT 1 FROM runs WHERE state='queued' AND json_extract(options,'$.gpu_preset')=? "
                                "LIMIT 1", (name,)).fetchone() is not None

    def claim_preset(self):
        s = self.session
        return s.name if s is not None and s.state.serving() else None

    # --- the loop of the worker -------------------------------------------------------------------------

    def poll(self):
        """Between two jobs: the end of LINGER, the requests to arm ahead and to disarm, the sessions that are over,
        the orphan and uncertain sessions reconciled with the invoice."""
        s = self.session
        if s is not None:
            phase, _ = s.state.snapshot()
            if phase in FINAL or (s.thread is not None and not s.thread.is_alive()):
                self.finish()
            elif phase == "linger":
                if s.origin == "job" and s.last_job is not None:
                    if self.clocks.mono() - s.last_job >= s.preset["seconds"]["idle_s"]:
                        s.state.request_close("idle")
                    elif self.local_queued():
                        s.state.request_close("local_job")
                elif s.origin == "ahead":
                    try:
                        servers.term(s.ep, 1, budget_cents=str(s.a_cents), until=s.until, now=self.clocks.wall())
                    except EngineError:
                        s.state.request_close("arm_ahead_expired")
        self.requests()
        now = time.monotonic()
        if now - self._checked >= 10:
            self._checked = now
            self.reconciled()

    def local_queued(self):
        with self.storage.connect() as conn:
            return conn.execute("SELECT 1 FROM runs WHERE state='queued' AND (json_extract(options,'$.gpu_preset') "
                                "IS NULL) LIMIT 1").fetchone() is not None

    def requests(self):
        with self.storage.connect() as conn:
            rows = [dict(r) for r in conn.execute("SELECT * FROM gpu_requests WHERE state='pending' ORDER BY created")]
        for req in rows:
            code = None
            if req["kind"] == "disarm":
                s = self.session
                if s is not None and s.name == req["gpu_preset"]:
                    s.state.request_close("user_disarm")
                    s.note("disarm-request")
            elif time.time() - req["created"] > STALE_REQUEST_S:
                code = "stale"
            else:
                code = self.arm_ahead(req["gpu_preset"], since=req["created"])
            with self.storage.connect() as conn:
                conn.execute("UPDATE gpu_requests SET state=?, code=?, updated=? WHERE id=? AND state='pending'",
                             ("refused" if code else "done", code, time.time(), req["id"]))

    def arm_ahead(self, name, since=0.0):
        """A request to arm `name` ahead: a session of one page that lingers while the window can host one; None when
        armed (or already open on it), else the closed code of the refusal. `since`: the creation of the request."""
        from . import presets as gpu_presets
        table = gpu_presets.load()
        if name not in table or table[name]["levers"]["arm_ahead_s"] is None:
            return Code.ROUTE_OPTION_UNSUPPORTED.value
        s = self.session
        if s is not None and s.name == name and s.state.serving():
            return None
        if s is not None:
            s.state.request_close("another_preset")
            self.finish()
        try:
            s = self.open(name, "ahead", since=since)
        except Refusal as exc:
            return exc.code
        if s.state.move("linger", {"open"}):
            transition(self.storage, s.arm_id, "linger", ("open",))
        return None

    def finish(self):
        """Wait for the session to reach a final state (its clock thread runs CLOSING), with no lock held."""
        s = self.session
        if s is None:
            return
        s.state.request_close(s.state.snapshot()[1] or "stop")
        if s.thread is not None:
            s.thread.join()
        phase, reason = s.state.snapshot()
        self.closed_reason[s.arm_id] = reason
        self.session = None

    def shutdown(self):
        s = self.session
        if s is not None:
            s.state.request_close("stop")
            self.finish()

    def reconciled(self):
        """Orphan and uncertain sessions whose window the ledger shows reconciled (or closed) become `closed`."""
        with self.storage.connect() as conn:
            rows = [dict(r) for r in conn.execute("SELECT arm_id, provider, ledger_path FROM gpu_sessions "
                                                  "WHERE state IN ('orphan','uncertain')")]
        for r in rows:
            try:
                status = window_view(r["ledger_path"], r["provider"], r["arm_id"])
            except (BenchError, OSError):
                continue
            if status == "closed":
                transition(self.storage, r["arm_id"], "closed", ("orphan", "uncertain"), reason="reconciled")

    # --- readiness ---------------------------------------------------------------------------------------

    def serves(self, name):
        """Whether the open session of `name` can still host a job (MFL023 B10, SOL-B-N1 of the review of the diff B):
        serving (OPEN or LINGER, no close asked for) and the floor of the engine holds on what is left of its window and
        on its term T, with MDF_MAX_PAGES pages (the checks of the engine are those of `prepare` and of the attempt)."""
        s = self.session
        if s is None or s.name != name or not s.state.serving():
            return False
        try:
            info = serverspend.session(self.ledger_path, s.provider, s.url, {serverspend.SESSION_ENV: s.arm_id},
                                       rate=s.ep.rate_usd_per_s)
            servers.term(s.ep, self.settings.max_pages, budget_cents=info["window_left_cents"],
                         until=info["armed_until"], now=self.clocks.wall())
        except (BenchError, EngineError):
            return False
        return True

    def readiness(self):
        """What the API shows of the GPU presets and of the ledger (closed values, read by the worker alone): each
        enabled preset with its availability and its minimum reserve, the total and the cap of each provider.

        Two things apart (MFL023 B10, SOL-B-N1): `a_cents` and `w_s` are those of a new session opened now, and
        `open_session` says whether the open session of the preset can still host a job (`serves`). A reserve of a
        new opening under the minimum is `server_cap` only when no open session of the preset hosts the job."""
        from ..services import gpu_availability
        from . import presets as gpu_presets
        out = {"presets": {}, "ledgers": {}}
        table = gpu_presets.load()
        endpoints = gpu_presets.endpoint_table(table)
        for provider in ("modal", "runpod"):
            try:
                s = serverspend.summary(spend.read_only(self.ledger_path), provider, self.ledger_path)
                out["ledgers"][provider] = {"total_cents": s["total_cents"], "orphans": len(s["orphans"]),
                                            "cap_cents": str(cents(Decimal(getattr(self.settings, "gpu_cap_usd_" + provider))
                                                                   * 100))}
            except BenchError as exc:
                out["ledgers"][provider] = {"error": exc.code.value}
        for name in self.settings.gpu_presets_enabled():
            item = dict(gpu_availability(name, self.settings), minimum_reserve_cents=None, a_cents=None, w_s=None,
                        open_session=self.serves(name))
            try:
                ep = endpoint_of(self.settings.servers, name, endpoints)
                item["minimum_reserve_cents"] = minimum_reserve(ep, self.settings.max_pages)
                led = out["ledgers"].get(ep.provider) or {}
                if "total_cents" in led:                 # A and W of a session opened now (OPENING, step 3)
                    a = cents(min(cents(Decimal(self.settings.gpu_session_cap_usd) * 100),
                                  Decimal(led["cap_cents"]) - Decimal(led["total_cents"])))
                    w = window_seconds(a, ep.rate, providers.ZERO_LATENCY_S[ep.provider],
                                       providers.BILLING_TAIL_S.get(ep.provider, 0)) if a > 0 else 0
                    item.update(a_cents=str(max(a, Decimal(0))), w_s=max(w, 0))
                    if (item["available"] and Decimal(item["minimum_reserve_cents"] or "inf") > a
                            and not item["open_session"]):
                        item.update(available=False, code=Code.SERVER_CAP.value)
                servers.read_keys(ep, self.env_file)
            except Refusal as exc:
                item.update(available=False, code=exc.code)
            except EngineError as exc:
                item.update(available=False, code=item["code"] or exc.code.value)
            out["presets"][name] = item
        for name, item in out["presets"].items():
            led = out["ledgers"].get(table[name]["provider"]) or {}
            if item["available"] and led.get("orphans"):
                item.update(available=False, code=Code.SESSION_ORPHANED.value)
        s = self.session
        if s is not None:
            phase, reason = s.state.snapshot()
            out["session"] = {"arm_id": s.arm_id, "gpu_preset": s.name, "phase": phase, "reason": reason}
        return out

    # --- recovery ----------------------------------------------------------------------------------------

    def recover(self):
        """Before the first claim (module docstring): the windows left by a dead worker, then the orphan windows of the
        enabled presets in the ledger of today. Unrecoverable when a window cannot be closed."""
        with self.storage.connect() as conn:
            rows = [dict(r) for r in conn.execute("SELECT * FROM gpu_sessions WHERE state != 'closed' ORDER BY created")]
        for r in rows:
            self.recover_row(r)
        self.recover_cli()

    def recover_log(self, r):
        """The partial worker log of the session `r` left by a dead worker (MFL023 B10, SOL-B-R1 in part): the folder
        `<ledger dir>/worker-logs/<arm_id>/partial/` of that session only goes, unread, and nothing of it is kept
        (MFL023 B12, SOL-B11-B1). After a death the keys that the session used cannot be attested (an env file gone, no
        longer private, or rotated in place), so no scan can clear its log; the complete log of a session closed in
        order is the one `keep_log` scans and saves. A link or a file in place of the folder is removed, never
        followed; the outcome (`partial_discarded`) goes into the diary of the session, with no line of the log and no
        key. Never raises: the recovery of the window comes after it."""
        folder = Path(r["ledger_path"]).parent / "worker-logs" / r["arm_id"]
        tmp = folder / PARTIAL
        try:
            if any(not stat.S_ISDIR(os.lstat(d).st_mode) for d in (folder.parent, folder)):
                return None                               # a link or a file in the way: nothing is followed or touched
            info = os.lstat(tmp)
        except OSError:
            return None
        outcome = {"partial_discarded": False}
        try:
            if stat.S_ISDIR(info.st_mode):
                shutil.rmtree(tmp, ignore_errors=True)   # a link inside is removed, never followed
            else:
                os.unlink(tmp)
                outcome["not_a_folder"] = True
        except OSError as exc:
            outcome["error"] = type(exc).__name__
        outcome["partial_discarded"] = not os.path.lexists(tmp)
        try:
            path = folder / "session.json"
            doc = (json.loads(path.read_text()) if path.is_file() and not path.is_symlink() else
                   {"arm_id": r["arm_id"], "gpu_preset": r["gpu_preset"], "provider": r["provider"], "events": []})
            doc.setdefault("events", []).append(dict(event="worker-log-partial", t=round(self.clocks.wall(), 3),
                                                     **outcome))
            path.write_text(json.dumps(doc, sort_keys=True, indent=1))
        except (OSError, ValueError, AttributeError):
            pass
        return outcome

    def recover_row(self, r):
        arm_id, provider, ledger_path = r["arm_id"], r["provider"], r["ledger_path"]
        self.recover_log(r)                              # before the window, whatever its outcome (MFL023 B10)
        cmds = commands(provider, r["resource"], ledger_path, arm_id, stopped=True)
        try:
            info = os.lstat(ledger_path)
        except OSError:
            info = None
        if info is None or not __import__("stat").S_ISREG(info.st_mode):
            raise Unrecoverable("the ledger of the session %s is missing or not a regular file" % arm_id, cmds)
        try:
            entries = spend.read_only(ledger_path)
            p = serverspend.parse(entries).get(provider) or serverspend.Provider(provider)
        except BenchError:
            raise Unrecoverable("the ledger of the session %s cannot be read" % arm_id, cmds) from None
        w = p.windows.get(arm_id)
        lock = serverspend.lock_path(ledger_path, arm_id)
        if w is None:
            if r["state"] == "opening":                  # died before its arm row: the window never existed
                if not serverspend.alive(ledger_path, arm_id):
                    try:
                        lock.unlink()
                    except OSError:
                        pass
                transition(self.storage, arm_id, "closed", ("opening",), reason="never_armed")
                return
            raise Unrecoverable("the session %s has no arm row in its ledger" % arm_id, cmds)
        if w["disarm"] is not None or r["state"] in ("uncertain", "orphan"):
            # the row closed already (a death after the `disarm` row, or a session already uncertain or orphan): no
            # call; the state from the parser and the reconciliation (SOL3-N2: an orphan row stays `orphan`)
            if not serverspend.alive(ledger_path, arm_id):
                try:
                    lock.unlink()
                except OSError:
                    pass
            status = window_view(ledger_path, provider, arm_id, entries)
            if status == "closed":
                reason = "reconciled" if r["state"] in ("uncertain", "orphan") or w["disarm"] is None else r["reason"]
            else:
                died = r["state"] in ("opening", "open", "linger", "closing")
                reason = (w["disarm"] or {}).get("uncertain") or ("worker_died" if died else r["reason"])
            transition(self.storage, arm_id, status, tuple(x for x in PHASES if x != "closed"), reason=reason)
            return
        # armed and never disarmed: disarm by the resource recorded, wait for the zero, close as orphans
        env = self.key_file(provider, r["env_file"])
        if env is None:
            raise Unrecoverable("no key of %s for the session %s" % (provider, arm_id), cmds)
        zero, readings, flash = self.disarm_resource(provider, r["resource"], env)
        if zero is None:
            raise Unrecoverable("no zero of %s %s" % (provider, r["resource"]), cmds)
        self.close_orphans(provider, r["resource"], ledger_path, zero, readings, flash, env)
        transition(self.storage, arm_id, "orphan", tuple(s for s in PHASES if s != "closed"),
                   reason="zero_missing" if r["state"] == "stopped" else "worker_died", t_zero=round(zero, 3))

    def key_file(self, provider, registered):
        """The env file that holds the key of `provider` for the recovery: the file of the worker, then the one recorded
        by the session; None when neither has it. Modal reads its CLI token elsewhere: any readable file."""
        from ..engine import envfile
        names = ["RUNPOD_API_KEY"] if provider == "runpod" else []
        for path in (self.env_file, registered):
            if not path:
                continue
            try:
                found = envfile.read(path, names)
            except (envfile.Unsafe, OSError):
                continue
            if all(found.get(n) for n in names):
                return path
        return None

    def disarm_resource(self, provider, resource, env_file):
        """The disarm of a resource and its zero, within ZERO_STOP_S, by children: (t_zero or None, readings, whether
        FlashBoot must go back to OFF)."""
        started, readings, disarmed = self.clocks.mono(), [], False
        calls = CLEANUP_CALL_S[provider]
        while self.clocks.mono() - started < providers.ZERO_STOP_S:
            left = providers.ZERO_STOP_S - (self.clocks.mono() - started)
            if not disarmed:
                disarmed = opchild.call("disarm", provider, resource, env_file, {}, argv=self.argv,
                                        budget_s=min(calls["disarm"], left), wall=self.clocks.wall,
                                        mono=self.clocks.mono)["ok"]
            if disarmed:
                r = opchild.call("zero", provider, resource, env_file, {}, argv=self.argv,
                                 budget_s=min(calls["zero"], max(1.0, left)), wall=self.clocks.wall,
                                 mono=self.clocks.mono)
                if r["ok"]:
                    state = r["result"].get("state") or {}
                    readings.append({"t": round(self.clocks.wall(), 3), "state": state})
                    if r["result"].get("zero"):
                        flash = state.get("flashboot") not in (None, "OFF")
                        if flash:
                            opchild.call("restore_flashboot", provider, resource, env_file, {}, argv=self.argv,
                                         budget_s=60, wall=self.clocks.wall, mono=self.clocks.mono)
                        return self.clocks.wall(), readings, flash
                    if state.get("armed"):
                        disarmed = False
            self.clocks.sleep(providers.ZERO_POLL_S)
        return None, readings, False

    def close_orphans(self, provider, resource, ledger_path, zero, readings, flash, env_file):
        """Under the lock of the ledger, the `disarm` rows with `orphan: true` of the windows of the resource without a
        disarm and with a free lock (as `session.py disarm`)."""
        ledger = spend.Ledger.file(ledger_path)
        closed = []
        with ledger.locked() as locked:
            p = serverspend.parse(locked.entries()).get(provider) or serverspend.Provider(provider)
            for arm_id, w in p.windows.items():
                if (w["disarm"] is None and w["arm"].get("endpoint") == resource
                        and not serverspend.alive(ledger_path, arm_id)):
                    serverspend.disarm(locked, provider, arm_id, round(zero, 3), False, orphan=True,
                                       extra={"readings": readings[-60:], "recovered_by": "worker",
                                              **({"flashboot_restored": True} if flash else {})})
                    closed.append(arm_id)
        for arm_id in closed:
            try:
                serverspend.lock_path(ledger_path, arm_id).unlink()
            except OSError:
                pass
        return closed

    def recover_cli(self):
        """The orphan windows (an arm without a disarm and with a free lock) of the endpoints of the enabled presets in
        the ledger of today, left by a session of the command line: disarmed and closed as orphans."""
        from . import presets as gpu_presets
        if not Path(self.ledger_path).exists():
            return
        table = gpu_presets.load()
        endpoints = gpu_presets.endpoint_table(table)
        try:
            entries = spend.read_only(self.ledger_path)
            parsed = serverspend.parse(entries)
        except BenchError:
            raise Unrecoverable("the ledger of the instance cannot be read",
                                ["check %s" % self.ledger_path]) from None
        for name in self.settings.gpu_presets_enabled():
            try:
                ep = endpoint_of(self.settings.servers, name, endpoints)
            except Refusal:
                continue
            p = parsed.get(ep.provider)
            if p is None:
                continue
            left = [a for a, w in p.windows.items() if w["disarm"] is None and w["arm"].get("endpoint") == ep.resource
                    and not serverspend.alive(self.ledger_path, a)]
            if not left:
                continue
            env = self.key_file(ep.provider, None)
            cmds = commands(ep.provider, ep.resource, self.ledger_path, None, stopped=True)
            if env is None:
                raise Unrecoverable("no key of %s for the orphan windows %s" % (ep.provider, ", ".join(left)), cmds)
            zero, readings, flash = self.disarm_resource(ep.provider, ep.resource, env)
            if zero is None:
                raise Unrecoverable("no zero of %s %s" % (ep.provider, ep.resource), cmds)
            self.close_orphans(ep.provider, ep.resource, self.ledger_path, zero, readings, flash, env)


def window_view(ledger_path, provider, arm_id, entries=None):
    """The state of a window of the ledger for `gpu_sessions` (SOL3-N2): `closed` when reconciled or closed in the
    ordinary way; else `uncertain` with `uncertain`, `orphan` with `orphan: true` or without a disarm."""
    entries = spend.read_only(ledger_path) if entries is None else entries
    p = serverspend.parse(entries).get(provider) or serverspend.Provider(provider)
    w = p.windows.get(arm_id)
    if w is None:
        return "closed"
    s = p.summary(lambda a: serverspend.alive(ledger_path, a))
    view = next((x for x in s["windows"] if x["arm_id"] == arm_id), None)
    if view is not None and view["reconciled"]:
        return "closed"
    d = w["disarm"]
    if d is None:
        return "orphan"
    if d.get("uncertain"):
        return "uncertain"
    if d.get("orphan"):
        return "orphan"
    return "closed"
