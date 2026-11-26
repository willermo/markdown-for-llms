"""External tools of the engine and of the bench, run only in the toolchain image (`toolchain.py`).

`run` is the one place where an external tool starts: one `tool` child of the launcher per call
(`containers.py`), a container of the verified image, by ID, removed when it ends.
- Isolation: no network, no capabilities, no new privileges, a read-only root, the caller's user
  and a private tmpfs /tmp.
- Mounts (E3): the folder of the call on the work volume. Its `scratch` directory is the only
  writable path, mounted as /work (working directory, HOME and TMPDIR are there); each input of the
  call is copied to its `in` directory, mounted read-only, at /in/<name> (a file, or a directory with
  `In(..., directory=True)`). The cache, the runs, the home, personal sets and the Docker socket are
  never mounted. Arguments name inputs and outputs with `In` and `Out`; outputs are copied out of
  /work after a successful call, then the folder goes away.
- Outputs leave /work with the rooted copy of `containers` (`collect`, `collect_tree`): each
  component opened from the one before, without following a link, regular files and directories
  of this user only. A link in place of an output or of a directory above it is `tool_failed`
  ("no output"), a link or a special file inside an output directory too; none is ever opened.
- Environment: LC_ALL and LANG C.UTF-8, HOME and TMPDIR under /work; nothing of the caller. The
  stdin of the tool is a file of the call the caller wrote (`ctl/stdin`), or /dev/null.
- Memory (M4b: an uncapped `pandoc -f html` grew to 53 GB and the kernel killed it together with
  the editor): `MDF_BENCH_TOOL_MEMORY_MB` per call for the bench, the engine's own cap for a
  conversion, default `DEFAULT_MEMORY_MB`. Pandoc gets the cap of its own runtime,
  `+RTS -M<cap>m -RTS` ("Heap exhausted", exit 251); every other tool `RLIMIT_DATA` through the
  `data` ulimit of the container (heap and private writable mappings; poppler prints "Out of
  memory" and exits 0 with an incomplete page); the container as a whole the cap plus
  `GROUP_EXTRA_MB`, without swap (killed: exit 137, pandoc included). Each gives
  `tool_memory_limit`, also with exit status 0: the document is not determinable for that step.
- Pandoc always runs with `--sandbox`: its readers and writers read only the files of the
  command line (no `\\input`, no images or includes fetched); a Lua filter reads and writes
  only what the sandbox allows. The launcher refuses a pandoc without it.
- Timeout: the supervisor of the container sees the stop of the caller within a tick; a container
  that the launcher cannot attest as removed is a fatal `tool_failed`, which stops the command.

A call of the engine passes `watch` (a `Watch`): the runner then, every `poll_seconds`, measures
/work against its cap (`output_limit`), calls the observer (which can stop the call: `cancelled`
or `interrupted`) and checks the deadline (`tool_timeout`). The container also gets its process
limit and its file size limit. A call of the bench passes no `watch`. The memory peak and the CPU
time of every call come from the supervisor of its container.
"""
from contextlib import contextmanager
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

from . import containers, toolchain
from .codes import Code, EngineError

DEFAULT_MEMORY_MB = 8192
MEMORY_MB_RANGE = range(64, 1048577)
GROUP_EXTRA_MB = 512
MIB = 1024 * 1024
TMPFS = "/tmp:rw,nosuid,nodev,size=256m"
TMPFS_MIB = 256
VARIABLES = {"LC_ALL": "C.UTF-8", "LANG": "C.UTF-8", "HOME": "/work/home", "TMPDIR": "/work/tmp"}
# a call without a `Watch` (the bench) had no limit of its own on processes and file size: the widest the launcher admits
PIDS, FSIZE_MIB, NOFILE = 4096, 1048576, 4096
POLL_SECONDS = 0.25
_memory = {"mb": DEFAULT_MEMORY_MB}
# allocation failures as the tools print them (lower case): GHC runtime of pandoc ("Heap
# exhausted", "out of memory"), poppler gmalloc ("Out of memory", "Bogus memory allocation size"),
# C++ tools (std::bad_alloc), Ghostscript (VMerror), libc messages
MEMORY_MARKERS = (b"heap exhausted", b"out of memory", b"bogus memory allocation size", b"bad_alloc",
                  b"vmerror", b"cannot allocate memory")
TOOLS = ("ddjvu", "qpdf", "pdftoppm", "pdftotext", "pdfinfo", "pandoc", "soffice", "gs", "djvutxt", "djvused")
# tools without a version of their own in toolchain.json: the package that provides them
VERSION_OF = {"djvutxt": "ddjvu", "djvused": "ddjvu"}
ENGINE_EXITS = (126, 127)           # the supervisor could not start the command: not executable, not found
KILLED = 137                        # SIGKILL: the memory cgroup of the container
XFSZ = 128 + 25                     # SIGXFSZ: a file over `--ulimit fsize`
LOG_LIMIT = 64 * MIB                # stdout and stderr of a watched call
WORK_PATH = re.compile(r"[A-Za-z0-9_-][A-Za-z0-9._-]*(/[A-Za-z0-9_-][A-Za-z0-9._-]*)*")


class In:
    """An input of a call, mounted alone and read-only at /in/<name> (`name` defaults to the file
    name, made safe); the argument is `prefix` followed by that path. `directory=True` mounts a
    directory (its regular files and subdirectories only); `argument=False` mounts it without
    adding an argument (a file a Lua filter opens by its fixed path)."""

    def __init__(self, path, name=None, prefix="", directory=False, argument=True):
        self.path, self.name, self.prefix = Path(path), name, prefix
        self.directory, self.argument = directory, argument


class Out:
    """An output of a call: the tool gets `prefix` + /work/<arg> (`arg` defaults to the name of
    `dest`); after a successful call /work/<produced> (default: `arg`) must exist, non-empty unless
    `allow_empty`, and is moved to `dest`. `directory=True` expects a directory (regular files and
    directories only, no links); `optional=True` accepts its absence; `argument=False` names an
    output the tool writes by itself (for example the media directory of pandoc)."""

    def __init__(self, dest, arg=None, produced=None, prefix="", allow_empty=False, directory=False,
                 optional=False, argument=True):
        self.dest = Path(dest)
        self.arg = arg or self.dest.name
        self.produced = produced or self.arg
        self.prefix, self.allow_empty = prefix, allow_empty
        self.directory, self.optional, self.argument = directory, optional, argument
        for p in (self.arg, self.produced):
            if not WORK_PATH.fullmatch(p):
                raise ValueError("not a path inside /work: " + p)


class Watch:
    """The caps and the observer of a call of the engine (see the module docstring).

    `work_mib`: the cap of /work as a whole; `fsize_mib`: the cap of one file (`--ulimit fsize`);
    `pids`: the process limit of the container; `observer(event)` receives
    {"event": "tool", "tool", "container"} before the start and {"event": "tick", ...} at least
    every `poll_seconds`, and returns None, "cancelled" or "interrupted"; `attempt` names the
    container `mdf-eng-<attempt[:16]>-tool-<n>`; `deadline` is a `time.monotonic()` value after which the
    call stops with `tool_timeout`. `steps`, when a list, receives one record per call (tool,
    arguments without host paths, container, seconds, exit status, memory peak and CPU time)."""

    def __init__(self, work_mib, fsize_mib=None, pids=256, observer=None, attempt=None, deadline=None,
                 poll_seconds=0.25, steps=None):
        self.work_mib, self.fsize_mib, self.pids = work_mib, fsize_mib or work_mib, pids
        self.observer, self.attempt, self.deadline = observer, attempt, deadline
        self.poll_seconds, self.steps = poll_seconds, steps


def _check_name(name):
    if name not in TOOLS:
        raise ValueError("not a tool of the toolchain: " + name)


def version(name):
    """The version of a tool of the toolchain, from `toolchain.json` (no container)."""
    _check_name(name)
    return toolchain.versions()[VERSION_OF.get(name, name)]


def set_memory_limit(mb):
    """The cap of every tool call from now on, in MiB (`MDF_BENCH_TOOL_MEMORY_MB`)."""
    _memory["mb"] = _checked_mb(mb)


def memory_limit():
    return _memory["mb"]


def _checked_mb(mb):
    if isinstance(mb, bool) or not isinstance(mb, int) or mb not in MEMORY_MB_RANGE:
        raise EngineError(Code.INVALID_SETTING, "tool memory limit")
    return mb


def out_of_memory(name, returncode, stderr):
    if returncode == KILLED:            # the memory cap of the container's group
        return True
    err = stderr.lower()
    if name == "pandoc":    # the runtime's own messages, with its exit status
        return returncode == 251 and (b"heap exhausted" in err or b"out of memory" in err)
    return any(m in err for m in MEMORY_MARKERS)


def _host_file(path, directory=False):
    real = os.path.realpath(path)
    if "\n" in real or "\r" in real or not (os.path.isdir(real) if directory else os.path.isfile(real)):
        raise EngineError(Code.TOOL_FAILED, "input is not a regular " + ("directory" if directory else "file"))
    return real


class _Call:
    def __init__(self, proc, call, outputs, mb, memory):
        self.proc, self.call, self.outputs, self.mb, self.memory = proc, call, outputs, mb, memory


def tree_bytes(path):
    """Bytes of the regular files under `path` (links not followed), as allocated on disk or by size."""
    return containers.tree_bytes(path)


def _stopped(stop, what):
    return EngineError(Code.INTERRUPTED if stop == "interrupted" else Code.CANCELLED, what)


def _wait(child, name, tool, limits, timeout, watch):
    """The child of one call to its attested end; returns its `containers.Outcome` when the tool ended by itself
    or was killed for memory, and raises the closed code of every other end."""
    observer = None if watch is None else watch.observer
    seconds = timeout
    if watch is not None and watch.deadline is not None:
        seconds = min(seconds, watch.deadline - time.monotonic())

    def before(cname):
        if observer is not None and (stop := observer({"event": "tool", "tool": name, "container": cname})):
            raise _stopped(stop, f"{name}: before start")

    def on_poll(child, metrics):
        if observer is not None and (stop := observer({"event": "tick", "tool": name, "container": child.name})):
            return _stopped(stop, name)
        return None
    limits = dict(limits, max_seconds=max(1, min(86400, int(max(0, seconds)) + 1 + containers.TERM_MARGIN)))
    try:
        outcome = child.run(toolchain.role(), tool, VARIABLES, limits, timeout=seconds,
                            poll_seconds=POLL_SECONDS if watch is None else watch.poll_seconds,
                            scratch_mib=None if watch is None else watch.work_mib, log_limit=LOG_LIMIT,
                            on_poll=on_poll, before=before)
    except EngineError as exc:
        if exc.code is Code.ISOLATION_UNAVAILABLE and child.unconfirmed:
            # the container was to go and Docker did not confirm it: fatal for a whole command of the bench
            raise EngineError(Code.TOOL_FAILED, f"{name}: stopped, container {child.name} still exists", fatal=True) from None
        raise
    if watch is not None and watch.steps is not None:
        watch.steps.append({"tool": name, "args": list(tool), **outcome.step()})
    reason = outcome.reason
    if reason == "output_limit":
        raise EngineError(Code.OUTPUT_LIMIT, f"{name}: /work over its cap or output streams over {LOG_LIMIT} bytes")
    if reason == "timeout" or outcome.cause == "term":
        raise EngineError(Code.TOOL_TIMEOUT if watch is not None else Code.TOOL_FAILED, f"{name}: timeout")
    if reason is not None:
        raise reason
    if outcome.cause not in ("exited", "memory"):
        raise containers.error_of(outcome, name, tool=True)
    if watch is not None and containers.tree_bytes(child.call.scratch) > watch.work_mib * MIB:
        raise EngineError(Code.OUTPUT_LIMIT, f"{name}: /work over {watch.work_mib} MiB")     # filled between two polls
    return outcome


@contextmanager
def _call(name, args, work_root, timeout, stdin, memory_mb, group_mb, watch=None):
    _check_name(name)
    mb = memory_limit() if memory_mb is None else _checked_mb(memory_mb)
    group = mb + GROUP_EXTRA_MB if group_mb is None else _checked_mb(group_mb)
    toolchain.image()                   # verified by the launcher, once per process
    with containers.open_child(None if watch is None else watch.attempt, "tool") as child:
        call = child.call
        work = call.scratch
        (work / "home").mkdir()
        (work / "tmp").mkdir()
        argv, outputs, names = [], [], set()
        for a in args:
            if isinstance(a, In):
                safe = re.sub(r"[^A-Za-z0-9._-]", "_", a.name or a.path.name).lstrip(".") or "input"
                n, k = safe, 1
                while n in names:
                    k += 1
                    n = f"{k}-{safe}"
                names.add(n)
                try:
                    call.add_input(_host_file(a.path, a.directory), n)
                except (OSError, ValueError):
                    raise EngineError(Code.TOOL_FAILED, "input is not a regular " + ("directory" if a.directory else "file")) from None
                if a.argument:
                    argv.append(a.prefix + "/in/" + n)
            elif isinstance(a, Out):
                if a.argument:
                    argv.append(a.prefix + "/work/" + a.arg)
                outputs.append(a)
            elif isinstance(a, str):
                argv.append(a)
            else:
                raise TypeError("tool argument: " + type(a).__name__)
        limits = {"memory_mib": group, "pids": PIDS if watch is None else watch.pids,
                  "fsize_mib": FSIZE_MIB if watch is None else watch.fsize_mib, "nofile": NOFILE, "tmpfs_mib": TMPFS_MIB,
                  "lease_seconds": containers.LEASE_SECONDS}
        if name == "pandoc":
            tool = ["pandoc", "+RTS", f"-M{mb}m", "-RTS", "--sandbox", *argv]
        else:
            tool, limits["data_mib"] = [name, *argv], mb
        if stdin:
            call.stdin(stdin)
        outcome = _wait(child, name, tool, limits, timeout, watch)
        status = KILLED if outcome.exit is None else outcome.exit
        if status in ENGINE_EXITS:
            raise EngineError(Code.ENGINE_FAILED, f"{name}: engine exit {status}")
        proc = subprocess.CompletedProcess(tool, status, outcome.stdout, outcome.stderr)
        yield _Call(proc, call, outputs, mb, outcome.cause == "memory")


def _move_output(name, call, o):
    """One output of a call from its /work to its destination, with the rooted copy of `containers`."""
    produced = "scratch/" + o.produced
    kind = containers.entry_kind(call.root, produced)
    if o.directory:
        if kind == "absent":
            if o.optional:
                return
            raise EngineError(Code.TOOL_FAILED, f"{name}: no output {o.produced}")
        if kind != "directory":
            raise EngineError(Code.TOOL_FAILED, f"{name}: output {o.produced} is not a directory")
        if o.dest.exists() or o.dest.is_symlink():
            raise EngineError(Code.TOOL_FAILED, f"{name}: destination of {o.produced} exists")
        o.dest.parent.mkdir(parents=True, exist_ok=True)
        copied = containers.collect_tree(call.root, produced, o.dest)
        if copied is None or copied[1]:
            shutil.rmtree(o.dest, ignore_errors=True)
            raise EngineError(Code.TOOL_FAILED, f"{name}: output {o.produced} holds a link or a special file")
        return
    if kind == "absent" and o.optional:
        return
    if kind != "file" or (not o.allow_empty and containers.regular_size(call.root, produced) == 0):
        raise EngineError(Code.TOOL_FAILED, f"{name}: no output {o.produced}")
    if containers.collect(call.root, produced, o.dest) is None:
        raise EngineError(Code.TOOL_FAILED, f"{name}: no output {o.produced}")


def run(name, args, work_root, timeout=600, stdin=None, memory_mb=None, group_mb=None, watch=None):
    """Run a tool under the memory cap; returns stdout bytes after copying the outputs (`Out`) to their
    destinations. Raises `tool_memory_limit` when the tool reports an allocation failure or the
    container is killed, `engine_failed` for 126-127 and `tool_failed` for any other exit status, a
    missing output or a timeout (`tool_timeout` with `watch`). `group_mb` (self-test only) sets the
    container cap apart; `watch` (a `Watch`) is given by the engine. `work_root` is kept for the
    callers and no longer used: the call has its folder on the work volume."""
    with _call(name, args, work_root, timeout, stdin, memory_mb, group_mb, watch) as c:
        proc = c.proc
        if c.memory or out_of_memory(name, proc.returncode, proc.stderr):
            raise EngineError(Code.TOOL_MEMORY_LIMIT, f"{name}: cap {c.mb} MiB, exit {proc.returncode}")
        if watch is not None and proc.returncode == XFSZ:
            raise EngineError(Code.OUTPUT_LIMIT, f"{name}: file over {watch.fsize_mib} MiB")
        if proc.returncode != 0:
            raise EngineError(Code.TOOL_FAILED, f"{name}: exit {proc.returncode}: {proc.stderr[-300:]!r}")
        for o in c.outputs:
            _move_output(name, c.call, o)
        return proc.stdout


def probe(name, args, work_root, timeout=120):
    """(exit status, stdout, stderr) of a tool without judging the status (self-test of versions)."""
    with _call(name, args, work_root, timeout, None, None, None) as c:
        return c.proc.returncode, c.proc.stdout, c.proc.stderr


def soffice_convert(src, name, fmt, dest_dir, work_root, timeout=900):
    """Convert `src` (read as the file `name`, for example `twin.pptx`) with LibreOffice to `fmt`
    (`pdf`, `fodp`, `html`); the result is moved to `dest_dir` and its path returned.

    LibreOffice runs in the private /work of its call: profile, home, TMPDIR, working directory
    and output live there and go away with it (NSS writes `pkcs11.txt` in the working directory
    when it has no usable profile); the input is read-only at /in/<name>. soffice exits 0 when an
    export fails, so the converted file must exist and be non-empty.
    """
    ext = fmt.split(":")[0]
    stem = os.path.splitext(os.path.basename(name))[0]
    dest = Path(dest_dir) / f"{stem}.{ext}"
    run("soffice", ["-env:UserInstallation=file:///work/profile", "--headless", "--norestore", "--convert-to", fmt,
                    "--outdir", Out(dest, arg="out", produced=f"out/{stem}.{ext}"), In(src, name=os.path.basename(name))],
        work_root, timeout=timeout)
    return dest
