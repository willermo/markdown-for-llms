"""External tools of the bench, run only in the reference toolchain image (`toolchain.py`).

`run` is the one place where the bench starts an external tool: one `docker run` per call of the
verified image, by ID, removed when it exits.
- Isolation: no network, no capabilities, no new privileges, a read-only root, the caller's user
  and a private tmpfs /tmp.
- Mounts (E3): a private 0700 directory created for the call under `<cache>/tool-work/`, the only
  writable path, mounted as /work (working directory, HOME and TMPDIR are there), and each input
  file of the call, alone and read-only, at /in/<name>. The cache, the runs, the home, personal
  sets and the Docker socket are never mounted. Arguments name inputs and outputs with `In` and
  `Out`; outputs are moved out of /work after a successful call, then /work goes away.
- Environment: LC_ALL and LANG C.UTF-8, HOME and TMPDIR under /work; nothing of the caller.
- Memory (M4b: an uncapped `pandoc -f html` grew to 53 GB and the kernel killed it together with
  the editor): `MDF_BENCH_TOOL_MEMORY_MB` per call, default `DEFAULT_MEMORY_MB`. Pandoc gets the
  cap of its own runtime, `+RTS -M<cap>m -RTS` ("Heap exhausted", exit 251); every other tool
  `RLIMIT_DATA` through `--ulimit data` (heap and private writable mappings; poppler prints "Out
  of memory" and exits 0 with an incomplete page); the container as a whole `--memory` of the cap
  plus `GROUP_EXTRA_MB`, without swap (killed: exit 137, pandoc included). Each gives
  `tool_memory_limit`, also with exit status 0: the document is not determinable for that step.
- Pandoc always runs with `--sandbox`: its readers and writers read only the files of the
  command line (no `\\input`, no images or includes fetched).
- Timeout: the client is killed and the container removed by name (`docker rm -f`); a container
  that still exists afterwards is a fatal `tool_failed`, which stops the command.
- Exit 125-127 are failures of the engine, not of the tool: `engine_failed`.
"""
from contextlib import contextmanager
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

from . import toolchain
from .codes import BenchError, Code

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
TOOLS = ("ddjvu", "qpdf", "pdftoppm", "pdftotext", "pdfinfo", "pandoc", "soffice", "gs")
ENGINE_EXITS = (125, 126, 127)      # docker run: daemon error, command not executable, command not found
KILLED = 137                        # SIGKILL: the memory cgroup of the container
WORK_PATH = re.compile(r"[A-Za-z0-9_-][A-Za-z0-9._-]*(/[A-Za-z0-9_-][A-Za-z0-9._-]*)*")


class In:
    """An input file of a call, mounted alone and read-only at /in/<name> (`name` defaults to the
    file name, made safe); the argument is `prefix` followed by that path."""

    def __init__(self, path, name=None, prefix=""):
        self.path, self.name, self.prefix = Path(path), name, prefix


class Out:
    """An output of a call: the tool gets `prefix` + /work/<arg> (`arg` defaults to the name of
    `dest`); after a successful call /work/<produced> (default: `arg`) must exist, non-empty unless
    `allow_empty`, and is moved to `dest`."""

    def __init__(self, dest, arg=None, produced=None, prefix="", allow_empty=False):
        self.dest = Path(dest)
        self.arg = arg or self.dest.name
        self.produced = produced or self.arg
        self.prefix, self.allow_empty = prefix, allow_empty
        for p in (self.arg, self.produced):
            if not WORK_PATH.fullmatch(p):
                raise ValueError("not a path inside /work: " + p)


def _check_name(name):
    if name not in TOOLS:
        raise ValueError("not a bench tool: " + name)


def version(name):
    """The version of a tool of the reference toolchain, from `toolchain.json` (no container)."""
    _check_name(name)
    return toolchain.versions()[name]


def set_memory_limit(mb):
    """The cap of every tool call from now on, in MiB (`MDF_BENCH_TOOL_MEMORY_MB`)."""
    _memory["mb"] = _checked_mb(mb)


def memory_limit():
    return _memory["mb"]


def _checked_mb(mb):
    if isinstance(mb, bool) or not isinstance(mb, int) or mb not in MEMORY_MB_RANGE:
        raise BenchError(Code.INVALID_SETTING, "tool memory limit")
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


def _host_file(path):
    real = os.path.realpath(path)
    if "\n" in real or "\r" in real or not os.path.isfile(real):
        raise BenchError(Code.TOOL_FAILED, "input is not a regular file")
    return real


class _Call:
    def __init__(self, proc, work, outputs, mb):
        self.proc, self.work, self.outputs, self.mb = proc, work, outputs, mb


@contextmanager
def _call(name, args, work_root, timeout, stdin, memory_mb, group_mb):
    _check_name(name)
    mb = memory_limit() if memory_mb is None else _checked_mb(memory_mb)
    group = mb + GROUP_EXTRA_MB if group_mb is None else _checked_mb(group_mb)
    image = toolchain.image()
    root = Path(work_root).absolute()
    root.mkdir(parents=True, exist_ok=True)
    work = Path(os.path.realpath(tempfile.mkdtemp(prefix="call-", dir=root)))      # mode 0700
    try:
        if "\n" in str(work) or "\r" in str(work):
            raise BenchError(Code.TOOL_FAILED, "work directory path")
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
                mounts.append(mount(_host_file(a.path), "/in/" + n, readonly=True))
                argv.append(a.prefix + "/in/" + n)
            elif isinstance(a, Out):
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
        cname = toolchain.container_name()
        command = ["run", "--rm", "-i", "--name", cname, *toolchain.ISOLATION, "--user", f"{os.getuid()}:{os.getgid()}",
                   "--tmpfs", TMPFS, "--memory", f"{group}m", "--memory-swap", f"{group}m", *limits,
                   *[x for m in mounts for x in ("--mount", m)], "-w", "/work",
                   "-e", "LC_ALL=C.UTF-8", "-e", "LANG=C.UTF-8", "-e", "HOME=/work/home", "-e", "TMPDIR=/work/tmp",
                   image, *tool]
        try:
            proc = toolchain.engine_run(command, timeout, b"" if stdin is None else stdin)
        except subprocess.TimeoutExpired:
            if not toolchain.remove_container(cname):
                raise BenchError(Code.TOOL_FAILED, f"{name}: timeout, container {cname} still exists", fatal=True) from None
            raise BenchError(Code.TOOL_FAILED, f"{name}: timeout") from None
        except BaseException:           # an interrupted client (Ctrl-C) leaves its container running
            toolchain.remove_container(cname)
            raise
        if proc.returncode in ENGINE_EXITS:
            raise BenchError(Code.ENGINE_FAILED, f"{name}: engine exit {proc.returncode}")
        yield _Call(proc, work, outputs, mb)
    finally:
        shutil.rmtree(work, ignore_errors=True)


def run(name, args, work_root, timeout=600, stdin=None, memory_mb=None, group_mb=None):
    """Run a tool under the memory cap; returns stdout bytes after moving the outputs (`Out`) to their
    destinations. Raises `tool_memory_limit` when the tool reports an allocation failure or the
    container is killed, `engine_failed` for 125-127 and `tool_failed` for any other exit status, a
    missing output or a timeout. `group_mb` (self-test only) sets the container cap apart."""
    with _call(name, args, work_root, timeout, stdin, memory_mb, group_mb) as c:
        proc = c.proc
        if out_of_memory(name, proc.returncode, proc.stderr):
            raise BenchError(Code.TOOL_MEMORY_LIMIT, f"{name}: cap {c.mb} MiB, exit {proc.returncode}")
        if proc.returncode != 0:
            raise BenchError(Code.TOOL_FAILED, f"{name}: exit {proc.returncode}: {proc.stderr[-300:]!r}")
        for o in c.outputs:
            produced = c.work / o.produced
            if produced.is_symlink() or not produced.is_file() or (not o.allow_empty and produced.stat().st_size == 0):
                raise BenchError(Code.TOOL_FAILED, f"{name}: no output {o.produced}")
            o.dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(produced), str(o.dest))
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
