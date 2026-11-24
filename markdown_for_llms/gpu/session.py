#!/usr/bin/env python3
"""The session around every paid command on a rented provider (MFL033, plan r4 §5.4 and App. G).

    session.py run --provider P --name N --cap-usd X --spend-ledger L --spend-cap-usd C --env-file F
                   [--fault MODE] [--allow-download] [--disarmed | --cpu-activity] [--window-seconds S]
                   [--url URL] [--worker-log-dir DIR] [--levers L1,L2 | none] -- COMMAND ...
    session.py status --provider P --name N --spend-ledger L --env-file F
    session.py disarm --provider P --name N --spend-ledger L --env-file F

`run`:

1. checks at the start, refused with exit 3: an orphan window of the provider in the ledger
   (`session_orphaned`, which comes first), then a provider not idle (`provider_not_idle`: RunPod
   `workers.max > 0`, `summary.total > 0` or workers listed; Modal an app deployed or containers;
   a window open by another live session); before an arm of RunPod, a pin of `verify-pins` that
   `align` cannot change (`endpoint_out_of_pins`: image, entrypoint, volume and its data center,
   Secure Cloud, pool, rate; `providers.pin_checks`, MFL033 S-B2), and the arm checks them again on
   its own reading before its PATCH;
2. `A = min(--cap-usd, residual of the provider)`, `W = floor(A / rate - 60 s - Z - tail)` (the tail
   of the bound, 30 s on RunPod since c7.19, `providers.BILLING_TAIL_S`; shortened by
   `--window-seconds`), refused when `W <= 0`; `T = t_arm + W`;
3. the lock of the window (`<ledger>.arm-<id>.lock`, `O_CLOEXEC`, held by this process only and
   never inherited), then under the lock of the ledger, read again, the `arm` row; then the arm
   (`workers.max = 1` with the env map when it changes; `modal deploy`) and its reading back;
4. the command, with `MDF_SESSION_ARM=<id>` and `PR_SET_PDEATHSIG` = SIGTERM (a child that finds
   its parent already gone exits before running anything); at `T`, on SIGINT or SIGTERM it is
   stopped (SIGTERM, then SIGKILL after 20 s). The term holds during the arm too (MFL033 S-B4): the
   provider gets `T` and gives each management call at most the time left (`providers.TermReached`
   after it); SIGINT, SIGTERM and, on the real clock, an alarm at `T` interrupt an arm in progress;
   the command starts only before `T` (exit 5 otherwise); an arm interrupted or ended after `T` has an
   uncertain outcome, and the disarm and the zero below verify it;
5. always, in `finally`: the disarm, the zero read every 10 s, then the `disarm` row with `t_zero`,
   the readings and `clean` (no fault, exit 0, a command of the engine, zero within `Z`); a zero
   later than `Z` makes the window unclean; no zero within 600 s is a stop (exit 4): the disarm is
   tried again and no `disarm` row is written, so the window stays orphan at the cap. On RunPod the
   `disarm` row also holds `t_stop`, the last "stop container" of the worker log, and `double_s`, the
   seconds counted twice if two workers were ever RUNNING at once (c7.19): the bound of an unclean
   window runs to the later of the zero and the stop, plus the tail (`serverspend`). Since MFL023 B7
   the close follows `serverspend.closing`, with the largest drift of the wall clock on the monotonic
   one read in every wait (on the real clock only): an interval not observed (a suspend, a step of the
   clock, over 5 s) or a zero after the bound of the window (`T + 60 s + Z` for a window not shortened)
   writes the `disarm` row with `orphan: true` and `uncertain`, the event `uncertain` and exit 4 (after
   130 and 5, which come first: the row holds the block); the provider stays at its cap until
   `mdf bench spend reconcile --orphan --arm ID --manual-cents N`, and no new window opens on it.

Worker logs (MFL033 c7.4): after a RunPod arm the session follows the logs of the workers
(`worker_logs.Follower`; after a Modal deploy, since c7.8, the logs and containers of the app:
`worker_logs.ModalFollower`) into a private directory under /tmp and gives the command its path in
`SESSION_WORKER_LOG`; after the zero it stops, runs the secret scan of `mdf bench secret-scan` with the
keys of the env file and only then saves `DIR/<arm_id>/worker-log.txt` (DIR: `--worker-log-dir`,
else `worker-logs` next to the ledger). A key found: nothing is saved and the exit is 4. The report of
the window and its `disarm` row name the GPUs the proxies found at their start (`gpus`, from their
`mdf_gpu` lines, c7.14: on Modal an L4 or an A10G).

`--levers` (MFL033 N5b) names levers of the start from the watchdog's closed set (`watchdog.LEVERS`):
they go into the arm, `MDF_LEVERS` of the RunPod env or of the Modal deploy, and into the `arm` row; an
unknown one is refused before the row (`invalid_setting`), and a disarmed activity takes none. Without
`--levers` an arm takes the levers adopted (`providers.DEFAULT_LEVERS`, step 8 of N5b: the cache of
torch.compile); `--levers none` arms without levers.

The lever `snapshot` (MFL045 c2, plan r2 §4: the memory and GPU snapshot of Modal, `modal_snap.py`) is only for
Modal and only for an app named `*-snap` or `*-snap-fault`, and such an app takes exactly
`providers.SNAPSHOT_LEVERS` (a deploy changed in anything makes its snapshots obsolete); a fault only on its
`-fault` app; no `--allow-download`, `--disarmed` or `--cpu-activity` there; the fault `wake-fail` (a declared
fallback after a restore) only with the lever. Refused before any reading of the provider (`invalid_setting`).

The lever `flashboot` (MFL045 c3, plan r2 §6: FlashBoot of RunPod) is only for RunPod: the arm puts `flashboot`
FLASHBOOT in its PATCH and reads it back; after the zero, in the `finally` of the session (also after a failed or
interrupted command), a PATCH of its own puts it back to OFF and reads it back (prescription P-6): the outcome is
`flashboot_restored` in the `disarm` row, written anyway; a failure is exit 4, which comes before the 130 of a
signal. The disarm itself is unchanged (the PATCH of the workers of N4). `disarm`, the recovery of a session that died
before its `finally`, does the same after its zero (MFL045 c8, S-B1 of the review of the diff) when a window it closes
armed the lever or the endpoint read at the zero is explicitly not OFF (a reading without `flashboot` asks for nothing:
MFL049 (a)): outside the lock of the ledger, within RESTORE_LIMIT_S for all its calls (the `_deadline` of each call
and an alarm, MFL023 B10), then the windows read again under the lock (MFL049 (b)); `flashboot_restored` in each
`disarm` row it writes and in its output, a failure exit 4 after the rows; otherwise the calls are those of a disarm
without the lever. The zero
itself does not depend on the levers (`providers.RunPod.zero`: no worker listed), in the session as in the recovery.

The levers `snapshot` and `flashboot` are experimental (MFL045 F5, Davide's decision of 23/11/2026: measured, not
adopted; `providers.EXPERIMENTAL_LEVERS`): never among the levers adopted, armed only when `--levers` names them. A
session that arms one declares it, before the arm row, in a line of its own (`experimental_levers`, `adopted` false,
`guide`: the section of `docs/how-to/convert-documents.md` on each), and again as `experimental_levers` in the `arm`
row and in the report of the window. Informative only: no refusal and no other behaviour comes with it.

Errors of a provider (MFL023 B1, SOL2-B2) reach the events, the report and the output of `main` only as the closed
fields of `providers.closed` (`class`, `op`, `http`, `detail`: the text of this code): the stderr of the CLI of Modal
stays in the exception in memory, the body of a RunPod answer is never kept.

After `--`, `populate`, `probe-negative`, `probe-valid`, `fault-test` and `concurrency` (MFL033 N5b) run
that subcommand of `runpod_endpoint.py` on the endpoint of the session. `--disarmed` arms nothing: the provider must
be disarmed and idle, and the cap is reserved as an activity; `--cpu-activity` is the same at the
CPU rate, for the populate of the Modal Volume only (`modal run .../modal_populate.py::populate`) and
only with a reserve that covers the timeout of that Function at the CPU rate (MFL033 S-B3: a reserve
is not a cap that stops a remote job, the timeout of the Function is). `status` reads, writes nothing, and prints the commands to disarm and reconcile; exit 3
when the provider is not idle or a window is open or orphan. `disarm` disarms, waits for the zero
and closes the orphan windows of the endpoint with `orphan: true` (only a manual reconciliation
releases them); on RunPod it also puts FlashBoot back to OFF when the lever or the endpoint asks for it (above).
"""
import argparse
import ctypes
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR
import fcntl
import hashlib
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time

if __package__:                     # markdown_for_llms.gpu.session: the siblings of the package (MFL023 A2)
    from . import providers as pv
    from . import worker_logs
else:                               # scripts/servers/session.py and its tests: the path of MFL033
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import providers as pv  # noqa: E402
    import worker_logs  # noqa: E402

from markdown_for_llms.bench import secretscan, serverspend, spend  # noqa: E402
from markdown_for_llms.bench.codes import BenchError  # noqa: E402

DRIVERS = ("populate", "probe-negative", "probe-valid", "fault-test", "concurrency")   # concurrency: MFL033 N5b
PR_SET_PDEATHSIG = 1
STOP_GRACE_S = 20
CLOCK_MARGIN_S = 60
RESTORE_LIMIT_S = 90         # MFL049 (b): the absolute limit of the restore of FlashBoot by `disarm`, as the alarm of an arm
PREEXEC_DELAY = 0.0          # tests only: widens the window between fork and prctl


class Interrupted(BaseException):
    """A signal or the term T during the arm (MFL033 S-B4): raised by the handler, so that a management
    call in progress ends at once; `BaseException`, never caught by the `except Exception` of a provider."""


class Refusal(Exception):
    def __init__(self, code, detail, exit_code=3):
        super().__init__(detail)
        self.code, self.detail, self.exit_code = code, detail, exit_code


def commands(provider, name, ledger):
    runs = str(Path(ledger).parent)
    return ["python -I -B scripts/servers/session.py disarm --provider %s --name %s --env-file .env --spend-ledger %s"
            % (provider, name, ledger),
            "mdf bench spend reconcile --runs-dir %s --provider %s --orphan --manual-cents N" % (runs, provider)]


def child_setup(parent):
    """preexec_fn of the command: SIGTERM when this process dies; exit if it is already gone."""
    def setup():
        if PREEXEC_DELAY:
            time.sleep(PREEXEC_DELAY)
        libc = ctypes.CDLL(None, use_errno=True)
        if libc.prctl(PR_SET_PDEATHSIG, signal.SIGTERM, 0, 0, 0) != 0:
            os._exit(126)
        if os.getppid() != parent:
            os._exit(125)
    return setup


def windows_of(entries, provider, ledger_path):
    p = serverspend.parse(entries).get(provider) or serverspend.Provider(provider)
    return p, p.summary(lambda a: serverspend.alive(ledger_path, a))


def check_start(entries, provider, name, ledger_path, prov, need_idle=True):
    """Prescription 2: an orphan window comes before a provider not idle."""
    p, s = windows_of(entries, provider, ledger_path)
    if s["orphans"]:
        raise Refusal("session_orphaned", "orphan windows of %s: %s" % (provider, ", ".join(s["orphans"])))
    if s["over_bound"]:
        raise Refusal("server_billed_over_bound", "a reconciliation of %s over the ledger" % provider)
    live = [w["arm_id"] for w in s["windows"] if w["status"] == "open"]
    if live:
        raise Refusal("provider_not_idle", "windows open by live sessions: " + ", ".join(live))
    state = prov.state()
    if need_idle and (state["armed"] or state["busy"]):
        raise Refusal("provider_not_idle", "%s %s is armed or busy" % (provider, name))
    return p, s, state


def window_seconds(a_cents, rate, z, limit=None, tail=0):
    """W = floor(A / rate - 60 s - Z - tail), the tail of the bound after the stop of the container
    (`providers.BILLING_TAIL_S`, MFL033 c7.19: 30 s on RunPod)."""
    w = (Decimal(a_cents) / 100 / rate - CLOCK_MARGIN_S - z - Decimal(tail)).to_integral_value(rounding=ROUND_FLOOR)
    if limit is not None:
        w = min(w, Decimal(limit))
    return int(w)


def take_lock(ledger_path, arm_id):
    path = serverspend.lock_path(ledger_path, arm_id)
    fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        os.close(fd)
        raise
    return fd, path


class Session:
    def __init__(self, args, prov, clock=time.time, sleep=time.sleep, environ=None, out=None):
        self.a, self.prov, self.clock, self.sleep = args, prov, clock, sleep
        self.environ = os.environ if environ is None else environ
        self.out = out
        self.ledger = spend.Ledger.file(args.spend_ledger)
        self.stop_signal = None
        self.events = []
        # MFL023 B7 (point 1 of the plan r3): the monotonic clock beside the wall clock, for the drift of a window; on the
        # real clock only, as `alarm()`: with another clock (the tests) the drift is nil unless the test gives `mono`
        self.mono = time.monotonic if clock is time.time else clock
        self.mono_arm = None
        self.max_drift = 0.0

    def note(self, event, **fields):
        self.events.append(dict(event=event, t=round(self.clock(), 3), **fields))

    def rate(self):
        return pv.RATES["modal-cpu" if self.a.cpu_activity else self.a.provider]

    def command(self):
        cmd = list(self.a.command)
        if cmd and cmd[0] in DRIVERS:
            if self.a.provider != "runpod":
                raise Refusal("invalid_setting", "the drivers of runpod_endpoint.py are for RunPod")
            cmd = [sys.executable, "-I", "-B", str(pv.HERE / "runpod_endpoint.py"), "--env-file", self.a.env_file,
                   cmd[0], "--name", self.a.name, "--spend-ledger", str(self.ledger.path)] + cmd[1:]
        if not cmd:
            raise Refusal("invalid_setting", "a command after --")
        return cmd

    MDF_OPTIONS = ("--env-file", "--data-root", "--model-root")   # the options of `mdf` before its subcommand
    PYTHON = re.compile(r"python(3(\.\d+)?)?")
    PYTHON_FLAGS = ("-I", "-B", "-IB", "-BI")

    def engine_command(self, cmd):
        """`mdf convert` or `mdf bench`, after the options of `mdf` itself (MFL033 c8.6: since c7.18 the
        form is `mdf --env-file F convert`, which the session took for another command: an unclean window),
        also as `<python> [-I] [-B] -m markdown_for_llms.cli ...`, the form of the guides (S-NB1)."""
        if not cmd:
            return False
        if Path(cmd[0]).name == "mdf":
            i = 1
        elif self.PYTHON.fullmatch(Path(cmd[0]).name):
            i = 1
            while i < len(cmd) and cmd[i] in self.PYTHON_FLAGS:
                i += 1
            if cmd[i:i + 2] != ["-m", "markdown_for_llms.cli"]:
                return False
            i += 2
        else:
            return False
        while i < len(cmd):
            if cmd[i] in self.MDF_OPTIONS:
                i += 2
            elif "=" in cmd[i] and cmd[i].split("=", 1)[0] in self.MDF_OPTIONS:
                i += 1
            else:
                break
        return i < len(cmd) and cmd[i] in ("convert", "bench")

    # --- run ---------------------------------------------------------------------------------
    def run(self):
        a = self.a
        cmd = self.command()
        entries = spend.read_only(self.ledger.path)
        disarmed = a.disarmed or a.cpu_activity
        self.arming = False
        named = getattr(a, "levers", None)
        if named is None:                                 # step 8 of N5b: the levers adopted, on an arm only
            named = "" if disarmed else pv.DEFAULT_LEVERS
        elif named.strip() == pv.NO_LEVERS:
            named = ""
        elif disarmed:
            raise Refusal("invalid_setting", "--levers needs an arm")
        try:
            self.levers = pv.levers_text(named, a.provider)
        except pv.ProviderError as exc:
            raise Refusal("invalid_setting", str(exc)) from None
        self.snapshot_rules(disarmed)
        self.experimental = pv.experimental_levers(self.levers)       # MFL045 F5: declared, informative only
        p, s, state = check_start(entries, a.provider, a.name, self.ledger.path, self.prov)
        self.note("start", state=state)
        refusal = None if disarmed else getattr(self.prov, "pin_refusal", None)
        if refusal is not None:                           # S-B2: the pins of verify-pins, before the arm row
            failed = refusal()
            if failed:
                raise Refusal("endpoint_out_of_pins", "%s %s outside its pins (%s): align cannot change them, "
                              "see verify-pins" % (a.provider, a.name, ", ".join(failed)))
        cap = spend._cents(str(Decimal(a.spend_cap_usd) * 100), "cap")
        residual = cap - Decimal(s["total_cents"])
        a_cents = min(Decimal(a.cap_usd) * 100, residual).quantize(Decimal("0.01"), rounding=ROUND_FLOOR)
        if a_cents <= 0:
            raise Refusal("server_cap", "%s: nothing left under the cap" % a.provider)
        if disarmed:
            return self.run_activity(cmd, a_cents, cap)
        z = pv.ZERO_LATENCY_S[a.provider]
        tail = pv.BILLING_TAIL_S.get(a.provider, 0)
        w = window_seconds(a_cents, self.rate(), z, a.window_seconds, tail=tail)
        if w <= 0:
            raise Refusal("server_cap", "the window W = %d s: nothing to arm" % w)
        url = self.url()
        if self.experimental:                  # MFL045 F5: before the arm row, so before anything is armed
            self.emit({"experimental_levers": self.experimental, "adopted": False,
                       "guide": {x.partition(":")[0]: pv.EXPERIMENTAL_LEVERS[x.partition(":")[0]]
                                 for x in self.experimental}})
        self.signals()                         # before the arm row: a signal now only stops the command
        arm_id = serverspend.new_arm_id()
        fd, lock = take_lock(self.ledger.path, arm_id)
        code, row = 4, None
        try:
            with self.ledger.locked() as locked:
                p, s2 = windows_of(locked.entries(), a.provider, self.ledger.path)   # read again under the lock
                if s2["orphans"]:
                    raise Refusal("session_orphaned", "orphan windows of " + a.provider)
                if any(x["status"] == "open" and x["arm_id"] != arm_id for x in s2["windows"]):
                    raise Refusal("provider_not_idle", "another session armed %s" % a.provider)
                t_arm = round(self.clock(), 3)
                self.t_arm_anchor, self.mono_arm = t_arm, self.mono()
                row = serverspend.arm(locked, self.ledger.path, a.provider, a.name, url, a_cents, t_arm + w, cap,
                                      self.rate(), t_arm, fault=a.fault or None, arm_id=arm_id,
                                      extra=dict({"window_s": w, "z_s": z, "tail_s": tail,
                                                  "allow_download": bool(a.allow_download)},
                                                 **({"levers": self.levers} if self.levers else {}),
                                                 **({"experimental_levers": self.experimental}
                                                    if self.experimental else {})))
            self.note("arm-row", arm_id=arm_id, a_cents=str(a_cents), window_s=w, until=row["until"])
            code = self.armed(cmd, row)
        except BenchError as exc:
            raise Refusal(exc.code.value, exc.detail) from None
        finally:
            if row is None:                       # refused before the arm row: no window, no lock file
                try:
                    lock.unlink()
                except OSError:
                    pass
            os.close(fd)
        return code

    SNAP_APP = re.compile(r".+-snap(-fault)?")

    def snapshot_rules(self, disarmed):
        """MFL045 c2 (module docstring): the app of the snapshot and its lever, before any reading."""
        a = self.a
        snap = "snapshot" in (self.levers or "").split(",")
        if a.fault == "wake-fail" and not snap:
            raise Refusal("invalid_setting", "--fault wake-fail needs the lever snapshot")
        snap_app = bool(self.SNAP_APP.fullmatch(a.name))
        if not (snap or snap_app):
            return
        if a.provider != "modal" or not snap_app:
            raise Refusal("invalid_setting", "the lever snapshot only on a Modal app named *-snap or *-snap-fault")
        if self.levers != pv.levers_text(pv.SNAPSHOT_LEVERS, "modal"):
            raise Refusal("invalid_setting", "an app of the snapshot takes exactly the levers %s" % pv.SNAPSHOT_LEVERS)
        if disarmed or a.allow_download:
            raise Refusal("invalid_setting", "an app of the snapshot: no --disarmed, --cpu-activity or --allow-download")
        if a.fault and not a.name.endswith("-fault"):
            raise Refusal("invalid_setting", "a fault only on the -fault app of the snapshot")

    def url(self):
        if self.a.provider == "runpod":
            return serverspend_url(self.prov.url)
        if self.a.url:
            return serverspend_url(self.a.url)
        return "modal:" + self.a.name

    def armed(self, cmd, row):
        a = self.a
        child, rc, t_disarm, armed_ok, failed, follower = None, None, None, False, False, None
        until = float(row["until"])
        try:
            if self.stop_signal is None:
                levers = getattr(self, "levers", "")
                self.arming = True
                self.alarm(until)
                try:
                    info = self.prov.arm(a.fault or "", a.allow_download, deadline=until,
                                         **({"levers": levers} if levers else {}))
                    self.note("armed", **(info or {}))
                    if a.provider == "modal" and a.url and serverspend_url(self.prov.url) != serverspend_url(a.url):
                        raise pv.ProviderError("modal: the URL after the deploy differs from --url")
                    armed_ok = True
                    follower = self.follow(row)
                finally:
                    self.arming = False
                    self.alarm(None)
            if self.stop_signal is None and self.clock() >= until:
                self.timed_out = True                 # S-B4: no command after T, the window closes
                self.note("stop-command", reason="term T before the command")
            if self.stop_signal is None and not self.timed_out:
                env = pv.child_env(self.environ)
                env[serverspend.SESSION_ENV] = row["arm_id"]
                if follower is not None:
                    env[worker_logs.LOG_ENV] = str(follower.path)
                child = subprocess.Popen(cmd, env=env, close_fds=True, preexec_fn=child_setup(os.getpid()))
                self.note("command", pid=child.pid)
                rc = self.wait(child, float(row["until"]))
        except pv.TermReached as exc:
            failed = self.timed_out = True
            self.note("provider-error", **pv.closed(exc))      # MFL023 B1: closed fields only
        except pv.ProviderError as exc:
            failed = True
            self.note("provider-error", **pv.closed(exc))
        except Interrupted as exc:
            failed = True
            self.timed_out = self.stop_signal is None
            self.note("arm-interrupted", reason=str(exc))
        finally:
            if child is not None and child.poll() is None:
                self.stop_child(child)
                rc = child.returncode
            t_disarm = self.clock()
            zero, readings = self.disarm_and_zero()
            flashboot_restored = self.restore_flashboot()       # MFL045 c3, P-6: after the zero, always
        log_info = self.keep_log(follower, row)
        ok = zero is not None
        latency = None if not ok else round(zero - t_disarm, 3)
        clean = (ok and armed_ok and not a.fault and rc == 0 and self.engine_command(cmd)
                 and latency <= pv.ZERO_LATENCY_S[a.provider])
        if not ok:
            self.note("stop", detail="no zero within %d s: the window stays orphan" % pv.ZERO_STOP_S)
            self.report(row, None, rc, readings, log_info)
            return 4
        extra = {"t_disarm": round(t_disarm, 3), "zero_latency_s": latency, "exit": rc, "readings": readings[-60:],
                 "signal": self.stop_signal}
        if a.provider == "runpod":
            extra.update(self.bound_extra(row, zero, readings, log_info))
        if log_info is not None:
            extra["worker_log"] = {k: log_info.get(k) for k in ("saved", "sha256", "lines", "workers", "gpus")}
        if flashboot_restored is not None:
            extra["flashboot_restored"] = flashboot_restored
        # MFL023 B7 (point 1 of the plan r3, SOL2-B1): the rule of the close of `serverspend`, with the largest drift
        # observed (never the last difference): a window not observed or a zero after its bound is an orphan with
        # `uncertain`, the provider at its cap until a reconciliation with the invoice; never a window cut at A
        self.sample_drift()
        uncertain = serverspend.closing(row, round(zero, 3), extra.get("t_stop"), extra.get("double_s") or 0,
                                        self.max_drift)
        if uncertain:
            extra.update(uncertain=uncertain, drift_s=round(self.max_drift, 3))
            self.note("uncertain", reason=uncertain, drift_s=round(self.max_drift, 3))
        with self.ledger.locked() as locked:
            serverspend.disarm(locked, a.provider, row["arm_id"], round(zero, 3), clean, orphan=bool(uncertain),
                               extra=extra)
        try:
            serverspend.lock_path(self.ledger.path, row["arm_id"]).unlink()
        except OSError:
            pass
        self.report(row, latency, rc, readings, log_info)
        if log_info is not None and not log_info.get("saved"):
            return 4                              # a key in the worker log, or a scan not complete: stop
        if flashboot_restored is False:
            return 4                              # MFL045 P-6: FlashBoot not back to OFF, before the 130 of a signal
        if self.stop_signal is not None:
            return 130
        if self.timed_out:
            return 5                              # the term T, also during the arm (S-B4)
        if uncertain:
            return 4                              # MFL023 B7: the close is not certified (after 130 and 5: the row holds it)
        if not armed_ok or (failed and child is None):
            return 4
        return 128 - rc if rc is not None and rc < 0 else rc

    timed_out = False
    experimental = ()                          # MFL045 F5: set by run(); a disarm alone has none

    def restore_flashboot(self, wanted=None, limit=None):
        """MFL045 c3 (P-6): with the lever `flashboot` on RunPod, FlashBoot back to OFF after the zero and read back
        (`providers.RunPod.restore_flashboot`): True, or False on a ProviderError (caught: the `disarm` row comes
        after it); None without the lever. `wanted`: the decision of the recovery (`disarm`, MFL045 c8), in place of
        the levers of the session.

        `limit` (MFL049 (b), the recovery `disarm`): an absolute limit on the wall clock for all the calls of the
        restore. Each call gets at most the time left (the `_deadline` of the provider, `TermReached` after it), and,
        on the real clock in the main thread, an alarm at `limit` interrupts a call still in progress, as for an arm
        (MFL023 B10, SOL-B-B2 of the review of the diff B: the timeout of HTTPX bounds the wait for each piece of an
        answer, not the whole answer). The interrupted restore is False, with `TermReached` in its event."""
        if wanted is None:
            wanted = "flashboot" in (getattr(self, "levers", "") or "").split(",")
        if self.a.provider != "runpod" or not wanted:
            return None
        restore = getattr(self.prov, "restore_flashboot", None)
        if restore is None:
            return None
        if limit is not None and hasattr(self.prov, "_deadline"):
            self.prov._deadline = limit            # MFL049 (b): every call of the restore within the absolute limit
        alarmed = (limit is not None and self.clock is time.time
                   and threading.current_thread() is threading.main_thread())
        previous = None
        if alarmed:
            def on_alarm(signum, frame):
                raise Interrupted("the limit of the restore")
            previous = signal.signal(signal.SIGALRM, on_alarm)
            signal.setitimer(signal.ITIMER_REAL, max(0.001, limit - time.time()))
        try:
            try:
                restore()
            finally:
                if alarmed:
                    signal.setitimer(signal.ITIMER_REAL, 0)
        except pv.ProviderError as exc:
            self.note("flashboot-error", **pv.closed(exc))
            return False
        except Interrupted:
            self.note("flashboot-error", **pv.closed(pv.TermReached(
                "RunPod: the absolute limit of the restore (%s s) came during its calls" % RESTORE_LIMIT_S)))
            return False
        finally:
            if alarmed:
                signal.setitimer(signal.ITIMER_REAL, 0)
                signal.signal(signal.SIGALRM, signal.SIG_DFL if previous is None else previous)
            if limit is not None and hasattr(self.prov, "_deadline"):
                self.prov._deadline = None
        self.note("flashboot", restored="OFF")
        return True

    @staticmethod
    def bound_extra(row, zero, readings, log_info):
        """The facts of the bound of a RunPod window for its `disarm` row (MFL033 c7.19, prompt N4s
        §4.2): `t_stop`, the last "stop container" line of the log (the bound runs to the later of it
        and the zero, plus the `tail_s` of the arm row); `double_s`, the seconds counted twice when two
        workers were RUNNING at once, in a listing of the log or in a reading of the zero: from one
        listing interval before the first such moment to the end of the bound."""
        facts = (log_info or {}).get("bound") or {}
        t_arm, t_stop = float(row["t_arm"]), facts.get("t_stop")
        end = max(zero, zero if t_stop is None else float(t_stop)) + float(row.get("tail_s") or 0)
        firsts = [t_arm + float(facts["overlap_since_arm"])] if facts.get("overlap_since_arm") is not None else []
        firsts += [float(r["t"]) for r in readings if pv.running_count(r.get("state") or {}) > 1][:1]
        double = 0.0
        if firsts:
            double = max(0.0, end - max(t_arm, min(firsts) - worker_logs.LIST_EVERY_S))
        return {"t_stop": t_stop, "double_s": round(double, 3)}

    # --- worker logs (c7.4) -------------------------------------------------------------------------
    def follow(self, row):
        """Start the follower of the worker logs when the provider has one (RunPod)."""
        make = getattr(self.prov, "log_follower", None)
        if make is None:
            return None
        tmp = Path(tempfile.mkdtemp(prefix="mdf-wlog-"))
        try:
            follower = make(tmp / "worker-log.partial", float(row["t_arm"]), clock=self.clock).start()
        except Exception as exc:
            shutil.rmtree(tmp, ignore_errors=True)
            raise pv.ProviderError("the worker log cannot be followed: %s" % type(exc).__name__) from None
        self.note("worker-log", path=str(follower.path))
        return follower

    def log_dir(self):
        return Path(self.a.worker_log_dir) if self.a.worker_log_dir else self.ledger.path.parent / "worker-logs"

    def keep_log(self, follower, row):
        """Stop the follower, scan the log for the keys of the env file, then save it (or drop it).
        Never raises: the `disarm` row comes after it."""
        if follower is None:
            return None
        try:
            return self._keep_log(follower, row)
        except Exception as exc:
            self.note("worker-log-error", error=type(exc).__name__)
            shutil.rmtree(Path(follower.path).parent, ignore_errors=True)
            return {"saved": None, "error": type(exc).__name__}

    def _keep_log(self, follower, row):
        info = follower.stop()
        src = Path(follower.path)
        if self.a.provider == "runpod":
            info["bound"] = worker_logs.bound_facts(src)      # times only, before the scan (c7.19)
        try:
            try:
                keys = secretscan.keys_bytes(self.a.env_file)
                read, errors, found = secretscan.scan_all([str(src)], keys)
            except BenchError as exc:
                read, errors, found = 0, [exc.code.value], []
            if found or errors or read != 1:
                info.update(saved=None, secret_found=sum(n for _, n in found), scan_errors=len(errors))
                self.note("worker-log-withheld", secret_found=info["secret_found"], scan_errors=len(errors))
                return info
            dest = self.log_dir() / row["arm_id"] / "worker-log.txt"
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dest)
            info.update(saved=str(dest), sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),
                        gpus=worker_logs.gpus(dest))      # what each proxy found at its start (c7.14)
            self.note("worker-log-saved", path=str(dest), lines=info["lines"])
            return info
        finally:
            shutil.rmtree(src.parent, ignore_errors=True)

    def wait(self, child, until):
        while child.poll() is None:
            if self.stop_signal is not None or self.clock() >= until:
                self.timed_out = self.stop_signal is None
                self.note("stop-command", reason=self.stop_signal or "term T")
                self.stop_child(child)
                break
            self.sleep(1)
            self.sample_drift()
        return child.returncode

    def sample_drift(self):
        """MFL023 B7: the largest drift of the wall clock on the monotonic clock since the arm row (by its size; a drift
        that goes back does not undo an interval not observed), read in every wait of the session."""
        if self.mono_arm is None:
            return
        drift = (self.clock() - self.t_arm_anchor) - (self.mono() - self.mono_arm)
        if abs(drift) > abs(self.max_drift):
            self.max_drift = drift

    def stop_child(self, child):
        try:
            child.send_signal(signal.SIGTERM)
            child.wait(timeout=STOP_GRACE_S)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait()

    def signals(self):
        """SIGINT and SIGTERM stop the command; during the arm they interrupt it at once (S-B4)."""
        def handler(signum, frame):
            self.stop_signal = signal.Signals(signum).name
            if getattr(self, "arming", False):
                raise Interrupted(self.stop_signal)

        def term(signum, frame):
            if getattr(self, "arming", False):
                raise Interrupted("term T")
        signal.signal(signal.SIGINT, handler)
        signal.signal(signal.SIGTERM, handler)
        signal.signal(signal.SIGALRM, term)

    def alarm(self, until):
        """An alarm at `until` on the real clock, the term of the arm (S-B4); None cancels it. With
        another clock (the tests) the providers' deadline and the check before the command hold it."""
        if self.clock is not time.time:
            return
        if until is None:
            signal.setitimer(signal.ITIMER_REAL, 0)
        else:
            signal.setitimer(signal.ITIMER_REAL, max(0.001, until - time.time()))

    def disarm_and_zero(self):
        """(t_zero or None, readings): the disarm (tried again on error), then the zero."""
        readings, t0, tries = [], self.clock(), 0
        disarmed = False
        while True:
            if not disarmed:
                try:
                    self.prov.disarm()
                    disarmed = True
                except pv.ProviderError as exc:
                    tries += 1
                    self.note("disarm-error", tries=tries, **pv.closed(exc))
            if disarmed:
                try:
                    zero, state = self.prov.zero()
                    readings.append({"t": round(self.clock(), 3), "state": state})
                    if zero:
                        return self.clock(), readings
                except pv.ProviderError as exc:
                    self.note("zero-error", **pv.closed(exc))
            if self.clock() - t0 >= pv.ZERO_STOP_S:
                try:
                    self.prov.disarm()
                except pv.ProviderError:
                    pass
                return None, readings
            self.sleep(pv.ZERO_POLL_S)
            self.sample_drift()

    @staticmethod
    def populate_command(cmd):
        """True for `... modal run [OPTIONS] PATH::populate` with PATH the `modal_populate.py` of this
        checkout (the CPU activity, MFL033 S-B3)."""
        names = [Path(c).name for c in cmd]
        for i, c in enumerate(cmd):
            path, sep, fn = c.partition("::")
            if sep and fn == "populate" and Path(path).resolve() == pv.MODAL_POPULATE.resolve():
                head = names[:i]
                return "modal" in head and "run" in head[head.index("modal"):]
        return False

    def activity_limit(self, cmd, a_cents):
        """S-B3: a reserve is a figure of the ledger, not a cap that stops a remote job. `--cpu-activity`
        runs only the populate of the Volume, whose Function Modal stops at its timeout, and only with a
        reserve that covers that timeout at the CPU rate; anything else is refused before the row."""
        if not self.populate_command(cmd):
            raise Refusal("invalid_setting", "--cpu-activity runs only `modal run scripts/servers/"
                          "modal_populate.py::populate`: the timeout of that Function bounds the remote job, a "
                          "reserve does not")
        try:
            timeout = pv.modal_populate_timeout()
        except pv.ProviderError as exc:
            raise Refusal("invalid_setting", str(exc)) from None
        need = (Decimal(timeout) * self.rate() * 100).quantize(Decimal("0.01"), ROUND_CEILING)
        if a_cents < need:
            raise Refusal("server_cap", "the reserve of %s cents is under the limit of the populate: %d s at the CPU "
                          "rate, %s cents" % (a_cents, timeout, need))
        return timeout, need

    def run_activity(self, cmd, a_cents, cap):
        a = self.a
        if a.fault or a.allow_download:
            raise Refusal("invalid_setting", "--fault and --allow-download need an arm")
        if a.cpu_activity:
            timeout, need = self.activity_limit(cmd, a_cents)
            self.note("activity-limit", timeout_s=timeout, limit_cents=str(need))
        try:
            with self.ledger.locked() as locked:
                row = serverspend.activity(locked, a.provider, "%s:%s" % (a.name, cmd[0] if cmd else ""), a_cents, cap,
                                           rate=self.rate())
        except BenchError as exc:
            raise Refusal(exc.code.value, exc.detail) from None
        self.note("activity", activity_id=row["activity_id"], reserve_cents=row["reserve_cents"])
        self.signals()
        child = subprocess.Popen(cmd, env=pv.child_env(self.environ), close_fds=True, preexec_fn=child_setup(os.getpid()))
        rc = self.wait(child, float("inf"))
        self.emit({"session": "activity", "activity_id": row["activity_id"], "exit": rc, "events": self.events})
        return 130 if self.stop_signal else rc

    def report(self, row, latency, rc, readings, log_info=None):
        self.emit({"session": "window", "arm_id": row["arm_id"], "provider": self.a.provider, "endpoint": self.a.name,
                   "exit": rc, "zero_latency_s": latency, "readings": len(readings), "worker_log": log_info,
                   "events": self.events,
                   **({"experimental_levers": self.experimental} if self.experimental else {})})

    def emit(self, obj):
        pv.emit(obj, self.out)


def serverspend_url(url):
    from markdown_for_llms.engine.servers import canonical_url
    return canonical_url(url)


# --- status and disarm -------------------------------------------------------------------------------

def status(a, prov, out=None):
    entries = spend.read_only(a.spend_ledger)
    _, s = windows_of(entries, a.provider, a.spend_ledger)
    state = prov.state()
    open_ = [w for w in s["windows"] if w["status"] == "open"]
    orphan = [w for w in s["windows"] if w["status"] == "orphan" and not w["reconciled"]]
    report = {"provider": a.provider, "name": a.name, "state": state, "total_cents": s["total_cents"],
              "open": open_, "orphan": orphan, "commands": commands(a.provider, a.name, a.spend_ledger)}
    if a.provider == "runpod":
        report["balance_usd"] = prov.balance()
    pv.emit(report, out)
    return 3 if (state["armed"] or state["busy"] or open_ or orphan) else 0


def disarm(a, prov, clock=time.time, sleep=time.sleep, out=None):
    ledger = spend.Ledger.file(a.spend_ledger)
    p, s = windows_of(spend.read_only(ledger.path), a.provider, ledger.path)
    live = [w["arm_id"] for w in s["windows"] if w["status"] == "open"]
    if live:
        raise Refusal("provider_not_idle", "a live session owns the windows %s: stop it instead" % ", ".join(live))
    ses = Session(a, prov, clock=clock, sleep=sleep, out=out)
    t_disarm = clock()
    zero, readings = ses.disarm_and_zero()
    if zero is None:
        pv.emit({"disarm": a.name, "zero": False, "readings": readings}, out)
        return 4
    closed = []
    # MFL045 c8 (S-B1 of the review of the diff): FlashBoot back to OFF, as the `finally` of the session does, when a
    # window to close armed the lever or the endpoint read at the zero is explicitly not OFF (MFL049 (a), MFL023 B7: a
    # reading without `flashboot` asks for nothing). MFL049 (b), MFL023 B7: the PATCH and its reading back run outside
    # the lock of the ledger, under an absolute limit of RESTORE_LIMIT_S for all their calls (the term of the provider
    # and, MFL023 B10, an alarm, as for an arm); then, under the lock, the windows are read again and only those still
    # orphan get their row.
    p, _ = windows_of(spend.read_only(ledger.path), a.provider, ledger.path)
    judged = [arm_id for arm_id, w in p.windows.items() if w["disarm"] is None and w["arm"].get("endpoint") == a.name
              and not serverspend.alive(ledger.path, arm_id)]
    wanted = (any("flashboot" in str(p.windows[x]["arm"].get("levers") or "").split(",") for x in judged)
              or (readings[-1]["state"] or {}).get("flashboot") not in (None, "OFF"))
    restored = ses.restore_flashboot(wanted=wanted, limit=clock() + RESTORE_LIMIT_S)
    with ledger.locked() as locked:
        p, _ = windows_of(locked.entries(), a.provider, ledger.path)
        orphans = [arm_id for arm_id, w in p.windows.items() if w["disarm"] is None and w["arm"].get("endpoint") == a.name
                   and not serverspend.alive(ledger.path, arm_id)]
        for arm_id in orphans:
            extra = {"t_disarm": round(t_disarm, 3), "zero_latency_s": round(zero - t_disarm, 3),
                     "readings": readings[-60:]}
            if restored is not None:
                extra["flashboot_restored"] = restored
            serverspend.disarm(locked, a.provider, arm_id, round(zero, 3), False, orphan=True, extra=extra)
            closed.append(arm_id)
    report = {"disarm": a.name, "zero": True, "zero_latency_s": round(zero - t_disarm, 3), "orphan_closed": closed,
              "commands": commands(a.provider, a.name, a.spend_ledger)[1:]}
    if restored is not None:
        report["flashboot_restored"] = restored
        if restored is False:
            report["flashboot_error"] = [e["detail"] for e in ses.events if e["event"] == "flashboot-error"][-1]
    pv.emit(report, out)
    return 4 if restored is False else 0                  # MFL045 c8: FlashBoot not back to OFF, after the rows


# --- command line ------------------------------------------------------------------------------------

def parser():
    ap = argparse.ArgumentParser(prog="session.py", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "status", "disarm"):
        q = sub.add_parser(name)
        q.add_argument("--provider", required=True, choices=("runpod", "modal"))
        q.add_argument("--name", required=True)
        q.add_argument("--spend-ledger", required=True)
        q.add_argument("--env-file", required=True)
        if name == "run":
            q.add_argument("--cap-usd", required=True)
            q.add_argument("--spend-cap-usd", required=True)
            q.add_argument("--fault", choices=("hang-after-first", "crash-after-first", "wake-fail"))
            q.add_argument("--allow-download", action="store_true")
            mode = q.add_mutually_exclusive_group()
            mode.add_argument("--disarmed", action="store_true")
            mode.add_argument("--cpu-activity", action="store_true")
            q.add_argument("--window-seconds", type=int)
            q.add_argument("--url")
            q.add_argument("--worker-log-dir")
            q.add_argument("--levers")                 # absent: providers.DEFAULT_LEVERS; "none": no levers
            q.add_argument("command", nargs=argparse.REMAINDER)
    return ap


def main(argv=None, factory=None, out=None):
    a = parser().parse_args(argv)
    if a.cmd == "run":
        if a.command and a.command[0] == "--":
            a.command = a.command[1:]
        for value in (a.cap_usd, a.spend_cap_usd):
            if not _usd(value):
                pv.emit({"code": "invalid_setting", "detail": "an amount in USD"}, out)
                return 2
    factory = factory or (lambda: pv.make(a.provider, a.name, a.env_file))
    try:
        prov = factory()
        if a.cmd == "status":
            return status(a, prov, out)
        if a.cmd == "disarm":
            return disarm(a, prov, out=out)
        return Session(a, prov, out=out).run()
    except Refusal as exc:
        pv.emit({"code": exc.code, "detail": exc.detail, "commands": commands(a.provider, a.name, a.spend_ledger)}, out)
        return exc.exit_code
    except pv.ProviderError as exc:
        pv.emit(dict({"code": "provider_error"}, **pv.closed(exc)), out)     # MFL023 B1: never str() of the error
        return 4
    except BenchError as exc:
        pv.emit({"code": exc.code.value, "detail": exc.detail}, out)
        return 3


def _usd(value):
    try:
        d = Decimal(value)
    except Exception:
        return False
    return d.is_finite() and Decimal(0) < d <= Decimal(100)


if __name__ == "__main__":
    sys.exit(main())
