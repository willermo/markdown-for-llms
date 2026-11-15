"""External tools of the engine and of the bench, run only in the toolchain image (`toolchain.py`).

`run` is the one place where an external tool starts: one `docker run` per call of the verified
image, by ID, removed when it exits.
- Isolation: no network, no capabilities, no new privileges, a read-only root, the caller's user
  and a private tmpfs /tmp.
- Mounts (E3): a private 0700 directory created for the call under `work_root`, the only
  writable path, mounted as /work (working directory, HOME and TMPDIR are there), and each input
  of the call, alone and read-only, at /in/<name> (a file, or a directory with `In(..., directory=True)`).
  The cache, the runs, the home, personal sets and the Docker socket are never mounted. Arguments
  name inputs and outputs with `In` and `Out`; outputs are moved out of /work after a successful
  call, then /work goes away.
- Environment: LC_ALL and LANG C.UTF-8, HOME and TMPDIR under /work; nothing of the caller.
- Memory (M4b: an uncapped `pandoc -f html` grew to 53 GB and the kernel killed it together with
  the editor): `MDF_BENCH_TOOL_MEMORY_MB` per call for the bench, the engine's own cap for a
  conversion, default `DEFAULT_MEMORY_MB`. Pandoc gets the cap of its own runtime,
  `+RTS -M<cap>m -RTS` ("Heap exhausted", exit 251); every other tool `RLIMIT_DATA` through
  `--ulimit data` (heap and private writable mappings; poppler prints "Out of memory" and exits
  0 with an incomplete page); the container as a whole `--memory` of the cap plus
  `GROUP_EXTRA_MB`, without swap (killed: exit 137, pandoc included). Each gives
  `tool_memory_limit`, also with exit status 0: the document is not determinable for that step.
- Pandoc always runs with `--sandbox`: its readers and writers read only the files of the
  command line (no `\\input`, no images or includes fetched); a Lua filter reads and writes
  only what the sandbox allows.
- Timeout: the client is killed and the container removed by name (`docker rm -f`); a container
  that still exists afterwards is a fatal `tool_failed`, which stops the command.
- Exit 125-127 are failures of the engine, not of the tool: `engine_failed`.

A call of the engine passes `watch` (a `Watch`): the client then runs in the background and the
runner, every `poll_seconds`, measures /work against its cap (a bind mount has no size limit of
its own: `output_limit`), calls the observer (which can stop the call: `cancelled` or
`interrupted`) and checks the deadline (`tool_timeout`). The container also gets a process
limit and `--ulimit fsize`, and its memory peak is read from its own cgroup when the tool exits.
A call of the bench passes no `watch` and runs exactly as before.
"""
from contextlib import contextmanager
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import tempfile
import time

from . import toolchain
from .codes import Code, EngineError

DEFAULT_MEMORY_MB = 8192
MEMORY_MB_RANGE = range(64, 1048577)
GROUP_EXTRA_MB = 512
MIB = 1024 * 1024
TMPFS = "/tmp:rw,nosuid,nodev,size=256m"
_memory = {"mb": DEFAULT_MEMORY_MB}
# allocation failures as the tools print them (lower case): GHC runtime of pandoc ("Heap
# exhausted", "out of memory"), poppler gmalloc ("Out of memory", "Bogus memory allocation size"),
# C++ tools (std::bad_alloc), Ghostscript (VMerror), libc messages
MEMORY_MARKERS = (b"heap exhausted", b"out of memory", b"bogus memory allocation size", b"bad_alloc",
                  b"vmerror", b"cannot allocate memory")
TOOLS = ("ddjvu", "qpdf", "pdftoppm", "pdftotext", "pdfinfo", "pandoc", "soffice", "gs", "djvutxt", "djvused")
# tools without a version of their own in toolchain.json: the package that provides them
VERSION_OF = {"djvutxt": "ddjvu", "djvused": "ddjvu"}
ENGINE_EXITS = (125, 126, 127)      # docker run: daemon error, command not executable, command not found
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
    container `mdf-eng-<attempt>-<n>`; `deadline` is a `time.monotonic()` value after which the
    call stops with `tool_timeout`. `steps`, when a list, receives one record per call (tool,
    arguments without host paths, seconds, exit status, memory peak of the container)."""

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


def _field(text):
    """A field of `--mount`, which Docker reads as CSV: quoted when it holds a comma or a quote."""
    return '"' + text.replace('"', '""') + '"' if ("," in text or '"' in text) else text


def mount(src, dst, readonly=False):
    return ",".join(_field(f) for f in ["type=bind", f"src={src}", f"dst={dst}", *(["readonly"] if readonly else [])])


def _host_file(path, directory=False):
    real = os.path.realpath(path)
    if "\n" in real or "\r" in real or not (os.path.isdir(real) if directory else os.path.isfile(real)):
        raise EngineError(Code.TOOL_FAILED, "input is not a regular " + ("directory" if directory else "file"))
    return real


class _Call:
    def __init__(self, proc, work, outputs, mb):
        self.proc, self.work, self.outputs, self.mb = proc, work, outputs, mb


# The tool runs under `sh -c`: when it exits, the memory peak of the container's own cgroup goes
# to stderr after a marker, and the exit status of the tool is kept.
PEAK_MARKER = b"\nMDF-ENGINE-MEMORY-PEAK "
PEAK_SCRIPT = ('"$@"; rc=$?; printf "\\nMDF-ENGINE-MEMORY-PEAK %s\\n" "$(cat /sys/fs/cgroup/memory.peak 2>/dev/null)" >&2; '
               'exit $rc')


def tree_bytes(path):
    """Bytes of the regular files under `path` (links not followed), as allocated on disk or by size."""
    total = 0
    for parent, _, files in os.walk(path, followlinks=False):
        for name in files:
            try:
                info = os.lstat(os.path.join(parent, name))
            except OSError:
                continue
            total += max(info.st_size, info.st_blocks * 512)
    return total


def _split_peak(stderr):
    at = stderr.rfind(PEAK_MARKER)
    if at < 0:
        return stderr, None
    value = stderr[at + len(PEAK_MARKER):].strip()
    return stderr[:at], int(value) if value.isdigit() else None


def _watched(command, cname, name, work, stdin, timeout, watch):
    """The client in the background under the caps of `watch`; a CompletedProcess-like result."""
    observer = watch.observer
    if observer is not None and (stop := observer({"event": "tool", "tool": name, "container": cname})):
        raise EngineError(Code.INTERRUPTED if stop == "interrupted" else Code.CANCELLED, f"{name}: before start")
    started = time.monotonic()
    limit = started + timeout
    if watch.deadline is not None:
        limit = min(limit, watch.deadline)
    reason = None
    with tempfile.TemporaryFile() as out, tempfile.TemporaryFile() as err:
        proc = toolchain.engine_popen(command, out, err)
        try:
            try:
                proc.stdin.write(stdin)
                proc.stdin.close()
            except BrokenPipeError:
                pass
            def over():
                if tree_bytes(work) > watch.work_mib * MIB:
                    return EngineError(Code.OUTPUT_LIMIT, f"{name}: /work over {watch.work_mib} MiB")
                if os.fstat(out.fileno()).st_size + os.fstat(err.fileno()).st_size > LOG_LIMIT:
                    return EngineError(Code.OUTPUT_LIMIT, f"{name}: output streams over {LOG_LIMIT} bytes")
                return None
            while True:
                try:
                    proc.wait(timeout=watch.poll_seconds)
                    reason = over()             # a tool may fill /work and exit between two polls
                    break
                except subprocess.TimeoutExpired:
                    pass
                reason = over()
                if reason is None and time.monotonic() > limit:
                    reason = EngineError(Code.TOOL_TIMEOUT, f"{name}: timeout")
                if reason is None and observer is not None and (
                        stop := observer({"event": "tick", "tool": name, "container": cname})):
                    reason = EngineError(Code.INTERRUPTED if stop == "interrupted" else Code.CANCELLED, name)
                if reason is not None:
                    break
        except BaseException:           # an interrupted client (Ctrl-C) leaves its container running
            proc.kill()
            proc.wait()
            toolchain.remove_container(cname)
            raise
        if reason is not None:
            proc.kill()
            proc.wait()
            if not toolchain.remove_container(cname):
                raise EngineError(Code.TOOL_FAILED, f"{name}: stopped, container {cname} still exists", fatal=True)
            raise reason
        out.seek(0)
        err.seek(0)
        stdout, stderr = out.read(), err.read()
    stderr, peak = _split_peak(stderr)
    if watch.steps is not None:
        tail = command[command.index(toolchain.image()) + 1:]
        if tail[:2] == ["sh", "-c"]:
            tail = tail[4:]
        watch.steps.append({"tool": name, "args": tail, "seconds": round(time.monotonic() - started, 3),
                            "exit": proc.returncode, "memory_peak_bytes": peak})
    return subprocess.CompletedProcess(command, proc.returncode, stdout, stderr)


@contextmanager
def _call(name, args, work_root, timeout, stdin, memory_mb, group_mb, watch=None):
    _check_name(name)
    mb = memory_limit() if memory_mb is None else _checked_mb(memory_mb)
    group = mb + GROUP_EXTRA_MB if group_mb is None else _checked_mb(group_mb)
    image = toolchain.image()
    root = Path(work_root).absolute()
    root.mkdir(parents=True, exist_ok=True)
    work = Path(os.path.realpath(tempfile.mkdtemp(prefix="call-", dir=root)))      # mode 0700
    try:
        if "\n" in str(work) or "\r" in str(work):
            raise EngineError(Code.TOOL_FAILED, "work directory path")
        (work / "home").mkdir()
        (work / "tmp").mkdir()
        mounts, argv, outputs, names = [mount(work, "/work")], [], [], set()
        for a in args:
            if isinstance(a, In):
                safe = re.sub(r"[^A-Za-z0-9._-]", "_", a.name or a.path.name).lstrip(".") or "input"
                n, k = safe, 1
                while n in names:
                    k += 1
                    n = f"{k}-{safe}"
                names.add(n)
                mounts.append(mount(_host_file(a.path, a.directory), "/in/" + n, readonly=True))
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
        if name == "pandoc":
            tool, limits = ["pandoc", "+RTS", f"-M{mb}m", "-RTS", "--sandbox", *argv], []
        else:
            tool, limits = [name, *argv], ["--ulimit", f"data={mb * MIB}:{mb * MIB}"]
        cname = toolchain.container_name(None if watch is None else watch.attempt)
        if watch is not None:
            limits += ["--pids-limit", str(watch.pids), "--ulimit", f"fsize={watch.fsize_mib * MIB}:{watch.fsize_mib * MIB}"]
            tool = ["sh", "-c", PEAK_SCRIPT, "sh", *tool]
        command = ["run", "--rm", "-i", "--name", cname, *toolchain.ISOLATION, "--user", f"{os.getuid()}:{os.getgid()}",
                   "--tmpfs", TMPFS, "--memory", f"{group}m", "--memory-swap", f"{group}m", *limits,
                   *[x for m in mounts for x in ("--mount", m)], "-w", "/work",
                   "-e", "LC_ALL=C.UTF-8", "-e", "LANG=C.UTF-8", "-e", "HOME=/work/home", "-e", "TMPDIR=/work/tmp",
                   image, *tool]
        if watch is not None:
            proc = _watched(command, cname, name, work, b"" if stdin is None else stdin, timeout, watch)
        else:
            try:
                proc = toolchain.engine_run(command, timeout, b"" if stdin is None else stdin)
            except subprocess.TimeoutExpired:
                if not toolchain.remove_container(cname):
                    raise EngineError(Code.TOOL_FAILED, f"{name}: timeout, container {cname} still exists", fatal=True) from None
                raise EngineError(Code.TOOL_FAILED, f"{name}: timeout") from None
            except BaseException:           # an interrupted client (Ctrl-C) leaves its container running
                toolchain.remove_container(cname)
                raise
        if proc.returncode in ENGINE_EXITS:
            raise EngineError(Code.ENGINE_FAILED, f"{name}: engine exit {proc.returncode}")
        yield _Call(proc, work, outputs, mb)
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _move_output(name, work, o):
    produced = work / o.produced
    if o.directory:
        if produced.is_symlink() or (produced.exists() and not produced.is_dir()):
            raise EngineError(Code.TOOL_FAILED, f"{name}: output {o.produced} is not a directory")
        if not produced.exists():
            if o.optional:
                return
            raise EngineError(Code.TOOL_FAILED, f"{name}: no output {o.produced}")
        for parent, dirs, files in os.walk(produced, followlinks=False):
            for entry in dirs + files:
                info = os.lstat(os.path.join(parent, entry))
                if not (stat.S_ISDIR(info.st_mode) or stat.S_ISREG(info.st_mode)):
                    raise EngineError(Code.TOOL_FAILED, f"{name}: output {o.produced} holds a link or a special file")
        if o.dest.exists():
            raise EngineError(Code.TOOL_FAILED, f"{name}: destination of {o.produced} exists")
        o.dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(produced), str(o.dest))
        return
    if not produced.exists() and not produced.is_symlink() and o.optional:
        return
    if produced.is_symlink() or not produced.is_file() or (not o.allow_empty and produced.stat().st_size == 0):
        raise EngineError(Code.TOOL_FAILED, f"{name}: no output {o.produced}")
    o.dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(produced), str(o.dest))


def run(name, args, work_root, timeout=600, stdin=None, memory_mb=None, group_mb=None, watch=None):
    """Run a tool under the memory cap; returns stdout bytes after moving the outputs (`Out`) to their
    destinations. Raises `tool_memory_limit` when the tool reports an allocation failure or the
    container is killed, `engine_failed` for 125-127 and `tool_failed` for any other exit status, a
    missing output or a timeout (`tool_timeout` with `watch`). `group_mb` (self-test only) sets the
    container cap apart; `watch` (a `Watch`) is given by the engine."""
    with _call(name, args, work_root, timeout, stdin, memory_mb, group_mb, watch) as c:
        proc = c.proc
        if out_of_memory(name, proc.returncode, proc.stderr):
            raise EngineError(Code.TOOL_MEMORY_LIMIT, f"{name}: cap {c.mb} MiB, exit {proc.returncode}")
        if watch is not None and proc.returncode == XFSZ:
            raise EngineError(Code.OUTPUT_LIMIT, f"{name}: file over {watch.fsize_mib} MiB")
        if proc.returncode != 0:
            raise EngineError(Code.TOOL_FAILED, f"{name}: exit {proc.returncode}: {proc.stderr[-300:]!r}")
        for o in c.outputs:
            _move_output(name, c.work, o)
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
