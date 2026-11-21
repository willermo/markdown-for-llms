"""The light child of the engine: a Python module of the engine run in Bubblewrap.

Used for the steps that read untrusted bytes in Python (the package audit, `pkgaudit`), so that
a ZIP bomb, a deep XML tree or a parser defect stays inside a process without network, with a
read-only system and only its scratch directory writable:

    bwrap --unshare-all --die-with-parent --new-session --cap-drop ALL --clearenv
          --ro-bind /usr /lib /lib64 <base prefix> <prefix>   (the interpreter, read-only)
          --ro-bind <markdown_for_llms> /mdf/markdown_for_llms (the engine code, the same as the caller's)
          --ro-bind <input> /in/<name> ...                     (inputs, each alone, read-only)
          --bind <scratch> /scratch                            (the only writable path)
          --remount-ro /

`child.py` sets the resource limits (address space, file size, CPU time, processes, files)
before it imports the module. The caller bounds the wall time, the size of the scratch
directory and the output it reads; a missing Bubblewrap is `isolation_unavailable`, never a
fallback to an unisolated run.

The Marker child (`marker_child.py`) is the same child with more: the model root read-only on
`/models`, the fonts of the system, its own environment (offline, CPU) and, around Bubblewrap, a
transient scope of the user's systemd manager (`scope`): `systemd-run --user --scope
--unit <name> -p MemoryMax=<memory>M -p MemorySwapMax=0`, a true cap on the whole tree, whose
peak, CPU time and OOM kills are read from its cgroup while it runs. Without a user manager
the Marker route is `isolation_unavailable`, never a run without the cap. A child writes its
progress as JSON lines to `/scratch/events.jsonl`; `run` reads the new lines at every poll
and hands them to `on_events`.
"""
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time

from .codes import Code, EngineError

BWRAP = "/usr/bin/bwrap"
SYSTEMD_RUN = "/usr/bin/systemd-run"
SYSTEMCTL = "/usr/bin/systemctl"
UNIT = re.compile(r"mdf-eng-[A-Za-z0-9_.-]{1,80}")
EVENTS = "events.jsonl"
EVENTS_LIMIT = 4 << 20
PACKAGE = Path(__file__).resolve().parents[1]          # the markdown_for_llms directory of this code
BOOT = ("import sys; sys.path.insert(0, '/mdf'); from markdown_for_llms.engine.child import main; "
        "raise SystemExit(main())")
LOG_LIMIT = 1 << 20


def available():
    return os.path.isfile(BWRAP) and os.access(BWRAP, os.X_OK)


def command(scratch, module, args=(), inputs=(), ro_binds=(), env=None, fonts=False):
    """The bwrap argv of a child running `module.main(args)` with `inputs` [(host path, name)]
    under /in, `ro_binds` [(host path, absolute path in the child)] and `env` added to the
    fixed environment; `fonts` makes /etc/fonts visible (Marker renders with fontconfig)."""
    argv = [BWRAP, "--unshare-all", "--die-with-parent", "--new-session", "--cap-drop", "ALL", "--clearenv"]
    for path in ("/usr", "/lib", "/lib64", sys.base_prefix, sys.prefix):
        if Path(path).exists():
            argv += ["--ro-bind", path, path]
    argv += ["--proc", "/proc", "--dev", "/dev", "--dir", "/in", "--dir", "/mdf",
             "--ro-bind", str(PACKAGE), "/mdf/markdown_for_llms"]
    if fonts:
        argv += ["--dir", "/etc", "--ro-bind-try", "/etc/fonts", "/etc/fonts"]
    for host, name in inputs:
        if "/" in name or name in ("", ".", ".."):
            raise ValueError("input name: " + name)
        argv += ["--ro-bind", os.path.realpath(host), "/in/" + name]
    for host, dest in ro_binds:
        if not dest.startswith("/") or dest in ("/", "/scratch", "/in", "/mdf") or ".." in dest.split("/"):
            raise ValueError("bind target: " + dest)
        argv += ["--ro-bind", os.path.realpath(host), dest]
    argv += ["--bind", str(Path(scratch).absolute()), "/scratch", "--chdir", "/scratch", "--remount-ro", "/"]
    for key, value in {"PATH": "/usr/bin", "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8", "HOME": "/scratch",
                       "TMPDIR": "/scratch", **(env or {})}.items():
        argv += ["--setenv", key, value]
    return argv + ["--", sys.executable, "-I", "-B", "-c", BOOT, "limits.json", module, *map(str, args)]


def scope_available():
    """True when the user's systemd manager can run a transient scope with a memory cap."""
    if not (os.access(SYSTEMD_RUN, os.X_OK) and available()):
        return False
    try:
        done = subprocess.run([SYSTEMD_RUN, "--user", "--scope", "--quiet", "--collect", "-p", "MemoryMax=64M",
                               "-p", "MemorySwapMax=0", "--", "/bin/true"], stdin=subprocess.DEVNULL,
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=20, env=_user_env())
    except (OSError, subprocess.TimeoutExpired):
        return False
    return done.returncode == 0


# The variables systemd-run and systemctl need to reach the user's manager, nothing else.
USER_ENV = ("XDG_RUNTIME_DIR", "DBUS_SESSION_BUS_ADDRESS", "HOME", "PATH", "LANG")


def _user_env():
    """The variables of USER_ENV that are set (the manager of the user), nothing else."""
    env = {k: os.environ[k] for k in USER_ENV if k in os.environ}
    env.setdefault("XDG_RUNTIME_DIR", "/run/user/%d" % os.getuid())
    env.setdefault("PATH", "/usr/bin:/bin")
    return env


def scope_command(unit, memory_mib, argv):
    if not UNIT.fullmatch(unit):
        raise ValueError("unit name: " + unit)
    return [SYSTEMD_RUN, "--user", "--scope", "--quiet", "--collect", "--unit", unit,
            "-p", "MemoryMax=%dM" % memory_mib, "-p", "MemorySwapMax=0", "--", *argv]


class ScopeUnreachable(Exception):
    """The user's systemd manager does not answer: a stop cannot be attested."""


def scope_state(unit):
    """`absent` when the unit does not exist, else its ActiveState; ScopeUnreachable when the
    manager cannot be asked (K7, PB8: an unreachable bus is not an absent unit)."""
    if not UNIT.fullmatch(unit.removesuffix(".scope")):
        raise ValueError("unit name: " + unit)
    name = unit if unit.endswith(".scope") else unit + ".scope"
    try:
        done = subprocess.run([SYSTEMCTL, "--user", "show", "-p", "LoadState", "-p", "ActiveState", "--", name],
                              stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=20, env=_user_env())
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ScopeUnreachable(type(exc).__name__) from None
    if done.returncode != 0:
        raise ScopeUnreachable("systemctl exit %d" % done.returncode)
    fields = dict(line.split("=", 1) for line in done.stdout.splitlines() if "=" in line)
    if "LoadState" not in fields:
        raise ScopeUnreachable("no LoadState")
    if fields["LoadState"] == "not-found" or fields.get("ActiveState") in ("inactive", "failed", None):
        return "absent"
    return fields["ActiveState"]


def stop_scope(unit):
    """Kill every process of the scope; returns the state after the stop (see `scope_state`)."""
    if not UNIT.fullmatch(unit.removesuffix(".scope")):
        raise ValueError("unit name: " + unit)
    name = unit if unit.endswith(".scope") else unit + ".scope"
    try:
        subprocess.run([SYSTEMCTL, "--user", "kill", "--signal=SIGKILL", "--", name], stdin=subprocess.DEVNULL,
                       capture_output=True, timeout=20, env=_user_env())
        subprocess.run([SYSTEMCTL, "--user", "stop", "--", name], stdin=subprocess.DEVNULL,
                       capture_output=True, timeout=20, env=_user_env())
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ScopeUnreachable(type(exc).__name__) from None
    return scope_state(unit)


def list_scopes(prefix):
    """The scope units of the user's manager whose name starts with `prefix` (for example
    `mdf-eng-<attempt[:16]>-`), loaded or not; ScopeUnreachable when the manager cannot be
    asked."""
    if not re.fullmatch(r"mdf-eng-[A-Za-z0-9_.-]{1,64}-", prefix):
        raise ValueError("scope prefix: " + prefix)
    try:
        done = subprocess.run([SYSTEMCTL, "--user", "list-units", "--all", "--plain", "--no-legend", "--type=scope",
                               "--", prefix + "*"], stdin=subprocess.DEVNULL, capture_output=True, text=True,
                              timeout=20, env=_user_env())
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ScopeUnreachable(type(exc).__name__) from None
    if done.returncode != 0:
        raise ScopeUnreachable("systemctl exit %d" % done.returncode)
    names = [line.split()[0] for line in done.stdout.splitlines() if line.strip()]
    return [n for n in names if n.startswith(prefix) and n.endswith(".scope") and UNIT.fullmatch(n.removesuffix(".scope"))]


class Cgroup:
    """The cgroup of a scope, found from the child's pid; values read while it exists."""

    def __init__(self, pid, unit):
        self.pid, self.unit, self.path = pid, unit, None
        self.peak = self.cpu_usec = self.oom_kill = 0

    def find(self):
        if self.path is None:
            try:
                line = Path("/proc/%d/cgroup" % self.pid).read_text().strip().splitlines()[-1]
            except OSError:
                return None
            rel = line.split(":", 2)[2]
            if rel.endswith("/" + self.unit + ".scope"):
                self.path = Path("/sys/fs/cgroup" + rel)
        return self.path

    def sample(self):
        path = self.find()
        if path is None:
            return
        try:
            self.peak = max(self.peak, int((path / "memory.peak").read_text()))
            for line in (path / "cpu.stat").read_text().splitlines():
                if line.startswith("usage_usec "):
                    self.cpu_usec = int(line.split()[1])
            for line in (path / "memory.events").read_text().splitlines():
                if line.startswith("oom_kill "):
                    self.oom_kill = max(self.oom_kill, int(line.split()[1]))
        except (OSError, ValueError):
            pass

    def kill(self):
        path = self.find()
        if path is not None:
            try:
                (path / "cgroup.kill").write_text("1")
            except OSError:
                pass


def scratch_bytes(path):
    total = 0
    for parent, _, files in os.walk(path, followlinks=False):
        for name in files:
            try:
                info = os.lstat(os.path.join(parent, name))
            except OSError:
                continue
            total += max(info.st_size, info.st_blocks * 512)
    return total


def run(scratch, module, args=(), inputs=(), memory_mib=1024, output_mib=512, timeout=120, cpu_seconds=None,
        steps=None, observer=None, poll_seconds=0.1, ro_binds=(), env=None, fonts=False, scope=None,
        process_limit=16, files_limit=64, address_space_mib=None, stall_seconds=None, on_events=None, check=None,
        activity=None):
    """Run the child; returns its exit status and log (stdout and stderr, at most `LOG_LIMIT`
    bytes). `output_limit` when the scratch grows over `output_mib`, `timeout` past `timeout`
    seconds, `memory_limit` when the child reports it (or, in a scope, when the kernel killed a
    process of it for memory), `stalled` when a child in a scope writes no event and uses no CPU
    for `stall_seconds`, `cancelled`/`interrupted` from the observer. `scope` is the unit name
    of the systemd scope with cap `memory_mib` (the Marker child). `check()`, called at every poll,
    returns an `EngineError` that stops the child (the relay of the `marker-server` route: its
    server unavailable, its term reached) or None; `activity()` returns the monotonic time of an
    activity of the child outside its events and CPU time (its requests through the relay), which
    keeps it from `stalled`. The child writes its results into the scratch
    directory."""
    if not available():
        raise EngineError(Code.ISOLATION_UNAVAILABLE, "bubblewrap not found")
    scratch = Path(scratch)
    scratch.mkdir(parents=True, exist_ok=True, mode=0o700)
    limits = {"memory_mib": address_space_mib or memory_mib, "output_mib": output_mib,
              "cpu_seconds": cpu_seconds or max(1, int(timeout)), "process_limit": process_limit,
              "files_limit": files_limit}
    (scratch / "limits.json").write_text(json.dumps(limits))
    argv = command(scratch, module, args, inputs, ro_binds, env, fonts)
    if scope is not None:
        if not os.access(SYSTEMD_RUN, os.X_OK):
            raise EngineError(Code.ISOLATION_UNAVAILABLE, "systemd-run not found")
        argv = scope_command(scope, memory_mib, argv)
    started = time.monotonic()
    reason = None
    events_path, events_at = scratch / EVENTS, 0
    log_path = scratch.parent / (scratch.name + ".log")
    group = None
    last_activity, last_cpu = started, -1

    def read_events():
        nonlocal events_at, last_activity
        try:
            with open(events_path, "rb") as stream:
                stream.seek(events_at)
                data = stream.read(EVENTS_LIMIT)
        except FileNotFoundError:
            return
        end = data.rfind(b"\n") + 1
        if not end:
            return
        events_at += end
        items = []
        for line in data[:end].splitlines()[:1000]:
            try:
                item = json.loads(line)
            except ValueError:
                continue
            if isinstance(item, dict):
                items.append(item)
        if items:
            last_activity = time.monotonic()
            if on_events is not None:
                on_events(items)

    if scope is not None and observer is not None:
        # K7: the deterministic name of the scope is recorded before the child exists, so a worker
        # that dies while it starts leaves a name to stop (HC1 of the review of B of MFL032)
        if stop := observer({"event": "child", "module": module, "pid": None, "scope": scope + ".scope"}):
            raise EngineError(Code.INTERRUPTED if stop == "interrupted" else Code.CANCELLED, module)
    with open(log_path, "wb") as log:
        proc = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                env=_user_env() if scope is not None else {}, close_fds=True, start_new_session=True)
        if scope is not None:
            group = Cgroup(proc.pid, scope)
        try:
            first = {"event": "child", "module": module, "pid": proc.pid}
            if scope is not None:
                first["scope"] = scope + ".scope"
            if observer is not None and (stop := observer(first)):
                reason = EngineError(Code.INTERRUPTED if stop == "interrupted" else Code.CANCELLED, module)

            def over():
                if scratch_bytes(scratch) > output_mib * 1048576:
                    return EngineError(Code.OUTPUT_LIMIT, f"{module}: scratch over {output_mib} MiB")
                if os.fstat(log.fileno()).st_size > LOG_LIMIT:
                    return EngineError(Code.OUTPUT_LIMIT, f"{module}: log over {LOG_LIMIT} bytes")
                return None
            while reason is None:
                try:
                    proc.wait(timeout=poll_seconds)
                    if group is not None:
                        group.sample()
                    read_events()
                    reason = over()             # a child may fill its scratch and exit between two polls
                    break
                except subprocess.TimeoutExpired:
                    pass
                now = time.monotonic()
                if group is not None:
                    group.sample()
                    if group.cpu_usec != last_cpu:
                        last_cpu, last_activity = group.cpu_usec, now
                if activity is not None:
                    seen = activity()
                    if seen is not None and seen > last_activity:
                        last_activity = seen
                read_events()
                reason = over()
                if reason is None and check is not None:
                    reason = check()
                if reason is None and now - started > timeout:
                    reason = EngineError(Code.TIMEOUT, f"{module}: timeout")
                if reason is None and stall_seconds and now - last_activity > stall_seconds:
                    reason = EngineError(Code.STALLED, f"{module}: no event and no CPU time for {stall_seconds} s")
                if reason is None and observer is not None and (
                        stop := observer({"event": "tick", "module": module, "pid": proc.pid})):
                    reason = EngineError(Code.INTERRUPTED if stop == "interrupted" else Code.CANCELLED, module)
        finally:
            if proc.poll() is None:
                if group is not None:
                    group.kill()
                os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
    data = log_path.read_bytes()[-LOG_LIMIT:]
    log_path.unlink()
    step = {"tool": "bwrap:" + module.rsplit(".", 1)[-1], "args": [str(a) for a in args],
            "seconds": round(time.monotonic() - started, 3), "exit": proc.returncode}
    if group is not None:
        step.update(scope=scope + ".scope", memory_peak_bytes=group.peak or None, cpu_seconds=round(group.cpu_usec / 1e6, 1),
                    oom_kill=group.oom_kill)
    if steps is not None:
        steps.append(step)
    # In a scope nothing but the memory cap kills the child (our own stops set `reason` first): the
    # kernel's SIGKILL, which bwrap reports as 128 + 9, or the SIGTERM of systemd stopping the unit
    # after an OOM kill (OOMPolicy=stop), when the cgroup was gone before a sample.
    killed = (-signal.SIGKILL, 128 + signal.SIGKILL, -signal.SIGTERM, 128 + signal.SIGTERM)
    if reason is None and group is not None and (group.oom_kill or proc.returncode in killed):
        reason = EngineError(Code.MEMORY_LIMIT, f"{module}: killed in its scope (peak {group.peak} bytes)")
    if reason is not None:
        reason.log = data
        raise reason
    return proc.returncode, data


def clean(scratch):
    shutil.rmtree(scratch, ignore_errors=True)


def program_command(executable, args=(), ro_binds=(), binds=(), env=None):
    """The bwrap argv of a program of the user's machine, not a Python module of the engine (the
    local CPU server of the `marker-server` route, MFL033): the same sandbox as the light child
    (no network, a read-only system, no capability, a new session, dies with its parent), a
    private /tmp, `ro_binds` and `binds` [(host path, absolute path in the sandbox)]."""
    argv = [BWRAP, "--unshare-all", "--die-with-parent", "--new-session", "--cap-drop", "ALL", "--clearenv"]
    for path in ("/usr", "/lib", "/lib64"):
        if Path(path).exists():
            argv += ["--ro-bind", path, path]
    argv += ["--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp"]
    for flag, pairs in (("--ro-bind", ro_binds), ("--bind", binds)):
        for host, dest in pairs:
            if not dest.startswith("/") or dest in ("/", "/usr", "/tmp", "/proc", "/dev") or ".." in dest.split("/"):
                raise ValueError("bind target: " + dest)
            argv += [flag, os.path.realpath(host), dest]
    argv += ["--chdir", "/tmp", "--remount-ro", "/"]
    for key, value in {"PATH": "/usr/bin", "LANG": "C.UTF-8", "HOME": "/tmp", **(env or {})}.items():
        argv += ["--setenv", key, value]
    return argv + ["--", executable, *map(str, args)]


class Started:
    """A program started by `start` in its scope: its process, scope, cgroup and log."""

    def __init__(self, proc, scope, group, log, log_path, started):
        self.proc, self.scope, self.group, self.log, self.log_path, self.started = proc, scope, group, log, log_path, started


def start(argv, scope, memory_mib, log_path):
    """Start `argv` (from `program_command`) in a transient scope with a memory cap and return it
    running; `stop` ends it. `isolation_unavailable` without Bubblewrap or the user's manager."""
    if not available() or not os.access(SYSTEMD_RUN, os.X_OK):
        raise EngineError(Code.ISOLATION_UNAVAILABLE, "bubblewrap or systemd-run not found")
    log = open(log_path, "wb")
    try:
        proc = subprocess.Popen(scope_command(scope, memory_mib, argv), stdin=subprocess.DEVNULL, stdout=log,
                                stderr=subprocess.STDOUT, env=_user_env(), close_fds=True, start_new_session=True)
    except BaseException:
        log.close()
        raise
    return Started(proc, scope, Cgroup(proc.pid, scope), log, Path(log_path), time.monotonic())


def stop(started):
    """Kill the program and its scope; returns the step record (exit, seconds, peak, CPU, OOM, the
    state of the scope after the stop: `absent`, or `unreachable` when the manager does not answer)."""
    started.group.sample()
    if started.proc.poll() is None:
        started.group.kill()
        try:
            os.killpg(started.proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    try:
        started.proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        pass
    try:
        state = stop_scope(started.scope)
    except ScopeUnreachable:
        state = "unreachable"
    started.log.close()
    return {"scope": started.scope + ".scope", "exit": started.proc.returncode, "scope_after": state,
            "seconds": round(time.monotonic() - started.started, 3), "memory_peak_bytes": started.group.peak or None,
            "cpu_seconds": round(started.group.cpu_usec / 1e6, 1), "oom_kill": started.group.oom_kill}
