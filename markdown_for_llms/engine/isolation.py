"""The children of the engine that run Python, and the local CPU server: sibling containers (MFL051).

`run` runs a module of the engine as a `light` or `marker` child of the launcher (`containers.py`,
`markdown_for_llms/launcher.py`): a container of the image of the application without network, with a read-only
root, no capability, the user of the caller, a memory cap without swap on the whole tree, a process limit, and
only the directories of its call mounted. It is used for the steps that read untrusted bytes in Python (the
package audit, the PDF preparation, the raster check, the probe) and for Marker, so that a ZIP bomb, a deep XML
tree or a parser defect stays inside it. `child.py` sets the limits of the process (address space, file size,
CPU time, files) before it imports the module. Without the launcher there is no child: `isolation_unavailable`,
never a run without the sandbox.

    in/<name>     the inputs, copied by the caller, read-only for the child
    scratch/      the only directory the child writes: what the caller prepared in its own `scratch` is copied in
                  before the start, and the results are copied back after the attested end
    /models       the weights of Marker (a `marker` child), read-only
    /relay        the socket directory of the attempt (a `marker` child: the relay of the `marker-server` route)

The caller's `scratch` is a directory the child never sees. Its results come back with the rooted copy of
`containers.collect` and `containers.collect_tree`: every component opened from the one before, without following
a link, regular files and directories of this user only. A link that a command left in place of a result (to the
key file or to the secrets of the process that started it) is not opened and not copied, so no caller ever reads
it: the callers read from their own `scratch`, never from the directory the child wrote.

The caller bounds the wall time, the size of the scratch directory and the output it reads. A child writes its
progress as JSON lines to `/scratch/events.jsonl`; `run` reads the new lines at every poll, without following a
link, and hands them to `on_events`. `stalled`: no event, no activity of the caller's own (`activity`) and less
CPU time than `STALL_CPU_USEC_PER_SECOND` for `stall_seconds`.

`start` and `stop` run a program of the CPU server volume (`llama-server`) as a `cpuserver` child that lives
while its caller works: its lease is renewed by a thread of the caller, and `stop` ends it and attests its end.
"""
import json
import math
import os
from pathlib import Path
import shutil
import time

from . import containers
from .codes import Code, EngineError

EVENTS = "events.jsonl"
EVENTS_LIMIT = 4 << 20
PACKAGE = Path(__file__).resolve().parents[1]          # the markdown_for_llms directory of this code
LOG_LIMIT = 1 << 20
VARIABLES = {"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8", "HOME": "/scratch", "TMPDIR": "/scratch"}
PROGRAM_VARIABLES = {"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8", "HOME": "/tmp"}
TMPFS_MIB = 64
MAX_SECONDS = 86400                 # the longest term the launcher admits
# E-7 (MFL051 A2). The supervisor of a child runs in the cgroup it samples: a command that is blocked still shows
# about 2 ms of CPU time a second. Measured on real children: 2008 microseconds in a second at rest against 1006356
# at work (tranche A1); 2086 a second over 20.7 s of a command asleep, 2336 in its worst two seconds
# (`tests/container/test_child_stall.py`, which writes the measure of each run to its observations). So "the
# counter did not change" never holds, and the CPU time counts as activity only at this rate or more over a
# window of the stall period: about ten times that noise, a fiftieth of one busy core.
STALL_CPU_USEC_PER_SECOND = 20000
STALL_WINDOWS = 10                  # the stall period is judged in this many windows, each at least a second


def available():
    """True when the launcher answers: without it no child can run."""
    try:
        containers.launcher().hello()
    except (containers.Unreachable, containers.Refused):
        return False
    return True


class Stall:
    """The activity of a child for `stalled`: its events and its caller's own signs of life (`seen`), and its CPU
    time, which counts only at `STALL_CPU_USEC_PER_SECOND` or more over a window (`cpu`)."""

    def __init__(self, seconds, now):
        self.seconds, self.last = seconds, now
        self.window = max(1.0, (seconds or 0) / STALL_WINDOWS)
        self.mark = None                # (instant, CPU microseconds) at the start of the window

    def seen(self, at):
        if at is not None and at > self.last:
            self.last = at

    def cpu(self, now, usec):
        if usec is None:
            return
        if self.mark is None or usec < self.mark[1]:
            self.mark = (now, usec)
        elif now - self.mark[0] >= self.window:
            if usec - self.mark[1] >= STALL_CPU_USEC_PER_SECOND * (now - self.mark[0]):
                self.last = now
            self.mark = (now, usec)

    def stalled(self, now):
        return bool(self.seconds) and now - self.last > self.seconds


def _stopped(stop, what):
    return EngineError(Code.INTERRUPTED if stop == "interrupted" else Code.CANCELLED, what)


def _term(timeout):
    """The term of the supervisor for a call the caller stops after `timeout` seconds: a little later, so that the
    caller's own reason comes first; never over what the launcher admits."""
    return max(1, min(MAX_SECONDS, int(math.ceil(timeout)) + containers.TERM_MARGIN))


def scratch_bytes(path):
    return containers.tree_bytes(path)


def run(scratch, module, args=(), inputs=(), memory_mib=1024, output_mib=512, timeout=120, cpu_seconds=None,
        steps=None, observer=None, poll_seconds=0.1, env=None, kind="light", attempt=None, process_limit=16,
        files_limit=64, address_space_mib=None, stall_seconds=None, on_events=None, check=None, activity=None,
        results=None):
    """Run the child; returns its exit status and log (stdout, then stderr, at most `LOG_LIMIT` bytes of each).

    `scratch` is a directory of the caller: what is in it goes to the child's `/scratch` before the start, and
    `results` (relative paths, files or directories; None: everything) come back into it after the attested end,
    whatever the exit status. `inputs` [(path, name)] are copied to `/in/<name>`. `kind` is `light` or `marker`
    (with the weights on `/models` and the socket directory of the attempt on `/relay`); `attempt` names the
    container `mdf-eng-<attempt[:16]>-<kind>-<n>` (without one: a name of this process).

    `output_limit` when the scratch grows over `output_mib` or the streams over `LOG_LIMIT`, `timeout` past
    `timeout` seconds, `memory_limit` when the kernel killed a process of the child for memory (the module's own
    `MemoryError` is its exit status 3), `stalled` (see the module docstring), `cancelled`/`interrupted` from
    the observer, `interrupted` when its caller stopped renewing its lease or the launcher itself stopped it,
    `isolation_unavailable` when the launcher does not answer or the end is not attested. `check()`, called at
    every poll, returns an `EngineError` that stops the child (the relay of the `marker-server` route) or None;
    `activity()` returns the monotonic time of an activity of the child outside its events and CPU time (its
    requests through the relay). The observer gets `{"event": "child", "module", "container"}` before the
    container is created and `{"event": "tick", ...}` at every poll."""
    scratch = Path(scratch)
    scratch.mkdir(parents=True, exist_ok=True, mode=0o700)
    limits = {"memory_mib": address_space_mib or memory_mib, "output_mib": output_mib,
              "cpu_seconds": cpu_seconds or max(1, int(timeout)), "files_limit": files_limit}
    request = {"memory_mib": memory_mib, "pids": process_limit + containers.SUPERVISOR_PIDS, "fsize_mib": output_mib,
               "nofile": files_limit, "tmpfs_mib": TMPFS_MIB, "max_seconds": _term(timeout),
               "lease_seconds": containers.LEASE_SECONDS}
    child = containers.open_child(attempt, kind)
    started = time.monotonic()
    stall = Stall(stall_seconds, started)
    events_at = 0

    def read_events():
        nonlocal events_at
        data = containers.read_regular(child.call.root, "scratch/" + EVENTS, EVENTS_LIMIT, offset=events_at) or b""
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
            stall.seen(time.monotonic())
            if on_events is not None:
                on_events(items)

    def before(name):
        # K7: the name of the container is recorded before it exists, so a worker that dies while it starts
        # leaves a name to stop (HC1 of the review of B of MFL032)
        if observer is not None and (stop := observer({"event": "child", "module": module, "container": name})):
            raise _stopped(stop, module)

    def on_poll(child, metrics):
        now = time.monotonic()
        stall.cpu(now, (metrics or {}).get("cpu_usec"))
        if activity is not None:
            stall.seen(activity())
        read_events()
        if check is not None and (reason := check()) is not None:
            return reason
        if stall.stalled(now):
            return EngineError(Code.STALLED, f"{module}: no event and no CPU time for {stall_seconds} s")
        if observer is not None and (stop := observer({"event": "tick", "module": module, "container": child.name})):
            return _stopped(stop, module)
        return None

    with child:
        call = child.call
        if kind == "marker":
            containers.socket_dirs(containers.RUN_ROOT, call.prefix)
        containers.seed(scratch, call.scratch)
        (call.scratch / containers.LIMITS_FILE).write_text(json.dumps(limits))
        for host, name in inputs:
            call.add_input(os.path.realpath(host), name)
        outcome = child.run("app", containers.python_command(module, args), {**VARIABLES, **(env or {})}, request,
                            timeout=timeout, poll_seconds=poll_seconds, scratch_mib=output_mib, log_limit=LOG_LIMIT,
                            on_poll=on_poll, before=before)
        read_events()
        step = {"tool": "child:" + module.rsplit(".", 1)[-1], "args": [str(a) for a in args], **outcome.step()}
        if steps is not None:
            steps.append(step)
        reason = outcome.reason
        if reason == "timeout":
            reason = EngineError(Code.TIMEOUT, f"{module}: timeout")
        elif reason == "output_limit":
            reason = EngineError(Code.OUTPUT_LIMIT, f"{module}: scratch over {output_mib} MiB or streams over {LOG_LIMIT} bytes")
        if reason is None:
            reason = containers.error_of(outcome, module)
        if reason is None and containers.tree_bytes(call.scratch) > output_mib * 1048576:
            # a child may fill its scratch and exit between two polls
            reason = EngineError(Code.OUTPUT_LIMIT, f"{module}: scratch over {output_mib} MiB")
        log = outcome.stdout[-LOG_LIMIT:] + outcome.stderr[-LOG_LIMIT:]
        if reason is not None:
            reason.log = log
            raise reason
        try:
            collect_results(call.root, scratch, results, output_mib * 1048576)
        except containers.Uncollected as exc:
            raise EngineError(Code.OUTPUT_LIMIT, f"{module}: {exc}") from None
    return outcome.exit, log


def collect_results(root, dest, results, limit):
    """The results of a call from its `scratch` (under the trusted folder `root` of the call) into the directory
    `dest` of the caller, with the rooted copy of `containers`: `results` names files or directories, None the
    whole scratch. What is not a regular file or a directory of this user, at any level, is left where it is."""
    if results is None:
        containers.collect_tree(root, "scratch", dest, limit)
        return
    for name in results:
        if containers.collect(root, "scratch/" + name, Path(dest) / name, limit) is None:
            containers.collect_tree(root, "scratch/" + name, Path(dest) / name, limit)


def clean(scratch):
    shutil.rmtree(scratch, ignore_errors=True)


class Started:
    """A program started by `start`: its child, the thread that renews its lease, where its log goes."""

    def __init__(self, child, renewer, log_path):
        self.child, self.renewer, self.log_path, self.name = child, renewer, Path(log_path), child.name


def start(argv, attempt, memory_mib, log_path, env=None, timeout=MAX_SECONDS, process_limit=1024, files_limit=4096,
          observer=None):
    """Start `argv` (a program under the tree of the CPU server volume) as a `cpuserver` child and return it
    running; `stop` ends it. Its socket directory is `sock/<prefix of the attempt>/llama` of the run volume, on
    `/relay`. `isolation_unavailable` without the launcher."""
    request = {"memory_mib": memory_mib, "pids": process_limit + containers.SUPERVISOR_PIDS, "fsize_mib": 64,
               "nofile": files_limit, "tmpfs_mib": TMPFS_MIB, "max_seconds": _term(timeout),
               "lease_seconds": containers.LEASE_SECONDS}

    def before(name):
        if observer is not None and (stop := observer({"event": "child", "module": "llama-server", "container": name})):
            raise _stopped(stop, "llama-server")
    child = containers.open_child(attempt, "cpuserver")
    try:
        containers.socket_dirs(containers.RUN_ROOT, child.call.prefix)
        child.begin("app", [str(a) for a in argv], {**PROGRAM_VARIABLES, **(env or {})}, request, before=before)
    except BaseException as exc:
        child.__exit__(type(exc), exc, None)
        raise
    renewer = containers.Renewer(child.call)
    renewer.start()
    return Started(child, renewer, log_path)


def stop(started):
    """End the program and attest its end; returns the step record (container, exit, seconds, peak, CPU, OOM) and
    writes its log (stdout, then stderr) to its log path."""
    started.renewer.stop()
    try:
        outcome = started.child.end(reason="stop")
    except BaseException as exc:
        started.child.__exit__(type(exc), exc, None)
        raise
    started.child.__exit__(None, None, None)
    try:
        started.log_path.write_bytes(outcome.stdout[-LOG_LIMIT:] + outcome.stderr[-LOG_LIMIT:])
    except OSError:
        pass
    return dict(outcome.step(), cause=outcome.cause)


def run_program(argv, attempt=None, memory_mib=512, timeout=60, env=None):
    """Run a program of the CPU server volume to its end as a `cpuserver` child (its `--help`); returns its exit
    status and its output."""
    request = {"memory_mib": memory_mib, "pids": 64 + containers.SUPERVISOR_PIDS, "fsize_mib": 64, "nofile": 256,
               "tmpfs_mib": TMPFS_MIB, "max_seconds": _term(timeout), "lease_seconds": containers.LEASE_SECONDS}
    with containers.open_child(attempt, "cpuserver") as child:
        containers.socket_dirs(containers.RUN_ROOT, child.call.prefix)
        outcome = child.run("app", [str(a) for a in argv], {**PROGRAM_VARIABLES, **(env or {})}, request, timeout=timeout)
    reason = EngineError(Code.TIMEOUT, "program: timeout") if outcome.reason == "timeout" else containers.error_of(outcome, "program")
    if reason is not None:
        raise reason
    return outcome.exit, outcome.stdout + outcome.stderr
