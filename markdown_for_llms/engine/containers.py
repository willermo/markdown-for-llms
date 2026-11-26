"""Children of the engine as sibling containers: the client of the launcher (MFL051).

Every child of the engine starts here: `isolation.run` and `isolation.start` (the Python children and the local
CPU server), `tools.run` (the tools of the toolchain image) and `toolchain` (the content manifest of that image).

A child is a container that the launcher (`markdown_for_llms/launcher.py`, the one process with the Docker
socket) creates from a closed request: a kind, a name, the role of an image, a command the kind admits, variables
and limits. The request names no mount. Each call owns a folder on the work volume, `call/<name>/`, made of four
sister directories that the launcher mounts one by one, never their parent:

    scratch   read-write  /work (tool) or /scratch     the only place the command writes its results
    in        read-only   /in                          the inputs, copied here by the caller
    ctl       read-only   /ctl                         `lease`, `stop` and `stdin`, written by the caller
    stat      read-write  /stat                        `stdout`, `stderr`, `metrics`, `exit`, written by the supervisor

and a file `owner`, held with an exclusive `flock` for the life of the call (`Call`).

The caller (`Child`):
1. makes the folder, puts the inputs in `in/`, the counter in `ctl/lease` and the stdin in `ctl/stdin`;
2. asks the launcher to run the child; the supervisor (`supervisor.py`) is PID 1 of the container;
3. at every poll renews `ctl/lease` (an atomic rename, at most once a second), reads the last line of
   `stat/metrics` and measures `scratch` alone against its cap; `stat/exit` is a hint that the command ended;
4. the end is ATTESTED only by Docker through the launcher: `inspect` says the container exited (exit status,
   `OOMKilled`) and `remove` confirms it no longer exists. Then the streams are read, the caller takes its
   results from `scratch` and the folder goes away;
5. the end is NOT ATTESTED when the launcher does not answer for `UNREACHABLE_SECONDS`, or a removal is not
   confirmed: the caller writes `ctl/stop` (the supervisor sees it within a tick and ends the container without
   Docker), LEAVES THE FOLDER WHERE IT IS, releases `owner` and raises `isolation_unavailable`. The folder is
   collected by `sweep` once its `owner` is free and the launcher attests that the container does not exist.

Every file read while the child may still write it (`stat/*`, the events in `scratch`) goes through
`read_regular`, which is ROOTED: it starts from a directory the child cannot replace (the folder of the call,
which is never mounted in it) and walks the relative path one component at a time, following no link at any
level; the file itself is opened without blocking, a regular file of this user only, at most a cap. Anything else
reads as absent: a FIFO, a link or a directory in its place, and a link in place of any directory above it.

The RESULTS of a child leave `scratch` the same way and no other: `collect` (one file) and `collect_tree` (a
directory) copy them to a place of the caller, each component opened from the descriptor of the one before,
without following a link, regular files and directories of this user only. What a command left in place of a
result (a link to a file of the caller's container, such as its key file; a FIFO; a file of another user) is
never opened: it is not copied, and `collect_tree` names it. The callers read their results from the copy, never
from `scratch`.

`stop_attempt_children(prefix)` is the attestation that no child of an attempt is left: list by prefix, kill,
remove, confirmed absence, the list empty again. `Unreachable` (the launcher does not answer, or answers
`isolation_unavailable`) and a removal not confirmed are `Unattested`: whoever holds a reservation for the
attempt keeps it (the worker, in tranche A2).

`identity` and `gate` are the payload gate: the children must run the code under test, not an old image
(`identity` is what a process reports of itself, `gate` the three comparisons: code, build inputs, environment).

Standard library only. The socket path and the roots of the volumes are constants of the containers, not settings.
"""
import fcntl
import hashlib
import itertools
import json
import os
from pathlib import Path
import re
import shutil
import socket
import stat
import sys
import threading
import time

from . import supervisor
from .codes import Code, EngineError

LAUNCHER_SOCKET = "/run/mdf/launcher/launcher.sock"
WORK_ROOT = "/work"                     # the work volume, as the processes that start children see it
RUN_ROOT = "/run/mdf/run"               # the volume of the socket directories of the attempts
MODEL_ROOT = "/models/marker"           # the weights of Marker in the models volume: what a `marker` child gets on /models
CPU_SERVER_ROOT = "/cpu-server"         # the volume of the local CPU server: `llama/` and `gguf/`
REQUEST_LIMIT = 64 * 1024               # one JSON request per connection
ANSWER_LIMIT = 1 << 20                  # the JSON line of an answer
MANIFEST_LIMIT = 64 << 20               # stdout of a `manifest` child (`logs`)
KINDS = ("tool", "light", "marker", "cpuserver", "probe", "manifest")
IMAGES = ("app", "toolchain", "candidate")
# <prefix of the attempt>-<kind>-<n>: `mdf-eng-<attempt[:16]>` (an attempt as `core.ATTEMPT` admits it) or the
# prefix of the bench, `mdf-bench-<pid>-<8 hex>`; at most 63 characters; also the name of the folder of the call
PREFIX_PATTERN = r"mdf-eng-[A-Za-z0-9_.-]{1,16}|mdf-bench-[0-9]{1,10}-[0-9a-f]{8}"
PREFIX = re.compile(PREFIX_PATTERN)
NAME = re.compile(r"(" + PREFIX_PATTERN + r")-(" + "|".join(KINDS) + r")-([0-9]{1,6})")
NAME_LIMIT = 63
CALL_ROLES = {"tool": ("scratch", "in", "ctl", "stat"), "light": ("scratch", "in", "ctl", "stat"),
              "marker": ("scratch", "in", "ctl", "stat"), "cpuserver": ("ctl", "stat"),
              "probe": ("scratch", "ctl", "stat"), "manifest": ("ctl",)}
SOCKET_ROLES = ("relay", "llama")       # under `sock/<prefix>/` of the run volume

# The commands the kinds admit (the launcher compares; see `launcher.check_command`)
PYTHON = "/opt/mdf/venv/bin/python"
BOOT = "from markdown_for_llms.engine.child import main; raise SystemExit(main())"
PROBE_MODULE = "markdown_for_llms.engine.selfprobe"
LIGHT_MODULES = ("markdown_for_llms.engine.pkgaudit", "markdown_for_llms.engine.pdfprep",
                 "markdown_for_llms.engines.raster", PROBE_MODULE)
MARKER_MODULES = ("markdown_for_llms.engine.marker_child", PROBE_MODULE)
LIMITS_FILE = "limits.json"
MANIFEST_PATH = "/etc/mdf-bench-toolchain.sha256"               # `toolchain.MANIFEST_PATH`
CONTENT_SCRIPT = "/usr/local/bin/mdf-bench-content-manifest"    # `toolchain.CONTENT_SCRIPT`
MANIFEST_READ = ["cat", MANIFEST_PATH]
MANIFEST_REGENERATE = [CONTENT_SCRIPT]
CPU_SERVER_TREE = "/opt/llama"                                  # `cpuserver.TREE`
# The probe of the toolchain image, which has no Python: the entries a shell can read, one `key value` per line.
SHELL_PROBE = r'''w() { ( : > "$1" ) 2>/dev/null && echo "write $1 yes" || echo "write $1 no"; }
echo "uid $(id -u)"; echo "gid $(id -g)"
while read -r k v; do case $k in CapEff:|NoNewPrivs:|Seccomp:) echo "${k%:} $v";; esac; done < /proc/self/status
n=0; while read -r a _; do n=$((n + 1)); [ "$n" -gt 2 ] && echo "interface ${a%:}"; done < /proc/net/dev
for f in memory.max memory.swap.max pids.max; do echo "cgroup $f $(cat /sys/fs/cgroup/$f 2>/dev/null)"; done
echo "ptrace_scope $(cat /proc/sys/kernel/yama/ptrace_scope 2>/dev/null)"
echo "ulimit_fsize $(ulimit -f)"; echo "ulimit_nofile $(ulimit -n)"; echo "ulimit_core $(ulimit -c)"
echo "pid1 $(tr '\0' ' ' < /proc/1/cmdline | cut -c1-40)"
echo "bash ${BASH_VERSION:-$(/bin/bash -c 'echo $BASH_VERSION')}"
env | while IFS='=' read -r k _; do echo "variable $k"; done
w /usr/mdf-probe; w /ctl/lease; w /ctl/mdf-probe; w /work/mdf-probe; w /stat/mdf-probe; w /tmp/mdf-probe
w /proc/self/root/ctl/lease; w /proc/1/root/ctl/lease; w /relay/mdf-probe; w /opt/llama/mdf-probe; w /models/mdf-probe
[ -e /var/run/docker.sock ] || [ -e /run/docker.sock ] && echo "docker_socket yes" || echo "docker_socket no"
[ -t 0 ] && echo "tty yes" || echo "tty no"; echo "stdin $(readlink /proc/self/fd/0)"
while read -r _ _ _ root target options _; do echo "mount $target $root $options"; done < /proc/self/mountinfo
'''
PROBE_SHELL = ["sh", "-c", SHELL_PROBE]


def python_command(module, args=()):
    """The command of a `light` or `marker` child: the entry of the light child (`child.py`), limits first."""
    return [PYTHON, "-I", "-B", "-c", BOOT, LIMITS_FILE, module, *map(str, args)]


PROBE_PYTHON = python_command(PROBE_MODULE, ["container"])

LEASE_SECONDS = 30
WAKE_SECONDS = 0.1                      # the wait of a call wakes this often: the lease, the hint of the end
TERM_MARGIN = 60                        # the term of the supervisor comes this long after the caller's own timeout
SUPERVISOR_PIDS = 4                     # the processes of the supervisor, on top of those of the command
COLLECT_DEPTH = 64                      # directories below a collected one; deeper ones are not copied
UNREACHABLE_SECONDS = 20.0              # the launcher does not answer: after this, the end is not attested
STOP_WAIT_SECONDS = 10.0                # after `ctl/stop`, before `kill` through the launcher
INSPECT_SECONDS = 5.0                   # an end without a hint is seen within this
HINT_SECONDS = 1.0                      # a hint that Docker did not confirm is asked again after this
REMOVE_TIMEOUT = 60.0
# why the launcher itself killed or removed a child (`reaped` of `inspect`): its lease or its term from outside
# (the supervisor did nothing), an exited child nobody removed, the stop of the launcher
REAPED = ("lease", "term", "ended", "stop")
# why `sweep` keeps a folder: its call is alive; it has no owner yet and is young; its owner cannot be opened; the
# launcher did not attest (unreachable, or a refusal); the container exists; the folder is still there after its removal
SWEEP_KEPT = ("held", "young", "owner", "unattested", "container", "not_removed")
STAT_LIMIT = 65536                      # `stat/metrics` (its tail) and `stat/exit`
LOG_LIMIT = 64 << 20                    # stdout and stderr of a call (`tools.LOG_LIMIT`)
ORPHAN_AGE = 60.0                       # a folder without `owner` younger than this may be being made
INPUTS = ("uv.lock", "pyproject.toml", "build-constraints.txt", "docker/app/Dockerfile",
          "docker/app/modal-requirements.txt")
INPUTS_FILE = "/opt/mdf/inputs.json"    # written when the image `engine` is built


class Unreachable(Exception):
    """The launcher does not answer, or answers `isolation_unavailable`."""


class Unattested(Exception):
    """The end of a child, or the absence of the children of an attempt, is not attested by Docker."""


class Refused(Exception):
    """The launcher refused a request with a closed code."""

    def __init__(self, code, detail=""):
        super().__init__("%s: %s" % (code, detail))
        self.code, self.detail = code, detail


# --- the launcher's API ------------------------------------------------------------------------

def _connect(path, timeout):
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM | socket.SOCK_CLOEXEC)
    try:
        sock.settimeout(timeout)
        sock.connect(path)
    except BaseException:
        sock.close()
        raise
    return sock


class Launcher:
    """The launcher behind its socket: one JSON request per connection (H.1). `connect(path, timeout)` returns a
    connected stream socket (the tests of the suite, which may not `connect`, pass a `socketpair`)."""

    def __init__(self, path=LAUNCHER_SOCKET, timeout=30.0, connect=_connect):
        self.path, self.timeout, self._connect = str(path), timeout, connect

    def call(self, op, timeout=None, **fields):
        """The answer of the launcher as a dict (for `logs`: with the bytes under `data`). `Unreachable` when it
        does not answer or answers `isolation_unavailable`; `Refused` for any other closed code."""
        request = json.dumps(dict(fields, op=op), separators=(",", ":")).encode("utf-8")
        if len(request) > REQUEST_LIMIT:
            raise Refused(Code.INVALID_SETTING.value, "request over %d bytes" % REQUEST_LIMIT)
        try:
            sock = self._connect(self.path, timeout or self.timeout)
        except OSError as exc:
            raise Unreachable("connect: " + _errname(exc)) from None
        try:
            sock.sendall(request)
            sock.shutdown(socket.SHUT_WR)
            head, rest = _read_line(sock)
            answer = json.loads(head)
            if not isinstance(answer, dict) or not isinstance(answer.get("ok"), bool):
                raise ValueError("answer")
            if answer["ok"] and "bytes" in answer:
                answer["data"] = _read_exact(sock, rest, answer["bytes"])
        except (OSError, ValueError) as exc:
            raise Unreachable("%s: %s" % (op, _errname(exc))) from None
        finally:
            sock.close()
        if answer["ok"]:
            return answer
        code, detail = str(answer.get("code")), str(answer.get("detail", ""))
        if code == Code.ISOLATION_UNAVAILABLE.value:
            raise Unreachable(detail)
        raise Refused(code, detail)

    def hello(self):
        return self.call("hello")

    def run(self, kind, name, image, argv, variables, limits):
        return self.call("run", kind=kind, name=name, image=image, argv=list(argv), env=dict(variables), limits=dict(limits))

    def inspect(self, name):
        return self.call("inspect", name=name)

    def kill(self, name):
        return self.call("kill", name=name)

    def remove(self, name):
        return self.call("remove", name=name, timeout=REMOVE_TIMEOUT)

    def list(self, prefix):
        return self.call("list", prefix=prefix)["children"]

    def logs(self, name):
        return self.call("logs", name=name, timeout=REMOVE_TIMEOUT)["data"]

    def toolchain(self, name=None, candidate=False):
        fields = {"candidate": True} if candidate else {}
        if name is not None:
            fields["name"] = name
        return self.call("toolchain", timeout=REMOVE_TIMEOUT, **fields)


def _errname(exc):
    import errno
    if isinstance(exc, socket.timeout):
        return "no answer"
    if isinstance(exc, OSError) and exc.errno:
        return errno.errorcode.get(exc.errno, "errno %d" % exc.errno)
    return type(exc).__name__


def _read_line(sock):
    data = b""
    while b"\n" not in data:
        if len(data) > ANSWER_LIMIT:
            raise ValueError("answer too long")
        chunk = sock.recv(65536)
        if not chunk:
            raise ValueError("answer cut")
        data += chunk
    head, _, rest = data.partition(b"\n")
    return head, rest


def _read_exact(sock, rest, count):
    if isinstance(count, bool) or not isinstance(count, int) or not 0 <= count <= MANIFEST_LIMIT:
        raise ValueError("bytes")
    parts, have = [rest[:count]], min(len(rest), count)
    while have < count:
        chunk = sock.recv(min(1 << 20, count - have))
        if not chunk:
            raise ValueError("answer cut")
        parts.append(chunk)
        have += len(chunk)
    return b"".join(parts)


# --- reading what a child may still be writing ------------------------------------------------------

def _open_below(root, name):
    """A descriptor of the file `name` below the directory `root`, opened for reading without blocking, or None.

    `root` is trusted: a path, opened as it is given (links in it are followed), or a descriptor of a directory,
    which is left open. `name` is not: it must be a relative path of plain components, and each of them is opened
    from the descriptor of the one before, the directories with `O_DIRECTORY | O_NOFOLLOW` and the last component
    with `O_NOFOLLOW | O_NONBLOCK`. So an absolute path, an empty component, `.` or `..`, a link in place of a
    directory at any level and a link in place of the file all give None, and nothing outside `root` is opened
    through a link."""
    try:
        parts = os.fspath(name).split("/")
    except TypeError:
        return None
    if not isinstance(parts[0], str) or any(part in ("", ".", "..") for part in parts):
        return None
    ours = not isinstance(root, int)
    try:
        fd = os.open(root, os.O_PATH | os.O_DIRECTORY | os.O_CLOEXEC) if ours else root
    except OSError:
        return None
    try:
        for part in parts[:-1]:
            below = os.open(part, os.O_PATH | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=fd)
            if ours:
                os.close(fd)
            fd, ours = below, True
        return os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC, dir_fd=fd)
    except OSError:
        return None
    finally:
        if ours:
            os.close(fd)


def read_regular(root, name, limit, tail=False, offset=0):
    """The bytes of the file `name` below the trusted directory `root` (at most `limit`: from `offset`, or its end
    with `tail`), or None.

    What it guarantees: it never blocks; it follows no link in `name`, neither the file nor any directory above
    it (`_open_below`), so it reads nothing outside `root` that a link points to; what it returns comes from the
    descriptor it checked, a regular file owned by this user. A FIFO, a link, a directory, a file of another user,
    a missing file, a link in place of a parent, `..` and an absolute path all read as absent.

    What it does not: `root` is opened as given at every call, so it must be a directory that whoever writes below
    it cannot replace; a hard link to a file of this user is a regular file like any other; and a file that is
    being written is read as it is at that moment."""
    fd = _open_below(root, name)
    if fd is None:
        return None
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid():
            return None
        if tail and info.st_size > limit:
            os.lseek(fd, info.st_size - limit, os.SEEK_SET)
        elif offset:
            os.lseek(fd, offset, os.SEEK_SET)
        parts, left = [], limit
        while left > 0:
            chunk = os.read(fd, min(left, 1 << 20))
            if not chunk:
                break
            parts.append(chunk)
            left -= len(chunk)
        return b"".join(parts)
    except OSError:
        return None
    finally:
        os.close(fd)


def regular_size(root, name):
    """The size of the file `name` below `root` when `read_regular` would read it, else 0."""
    fd = _open_below(root, name)
    if fd is None:
        return 0
    try:
        info = os.fstat(fd)
        return info.st_size if stat.S_ISREG(info.st_mode) and info.st_uid == os.getuid() else 0
    except OSError:
        return 0
    finally:
        os.close(fd)


class Uncollected(Exception):
    """A result is over the cap of its collection."""


def _ours(info, kind):
    return kind(info.st_mode) and info.st_uid == os.getuid()


def _copy_out(fd, dest, limit):
    """The bytes of the open regular file `fd` to `dest` (written beside it, then moved into place); returns
    their number. `Uncollected` over `limit`."""
    dest = Path(dest)
    partial = dest.with_name(dest.name + ".collecting")
    out = os.open(partial, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
    copied = 0
    try:
        while True:
            chunk = os.read(fd, 1 << 20)
            if not chunk:
                break
            copied += len(chunk)
            if limit is not None and copied > limit:
                raise Uncollected("%s: over %d bytes" % (dest.name, limit))
            view = memoryview(chunk)
            while view:
                view = view[os.write(out, view):]
    except BaseException:
        os.close(out)
        os.unlink(partial)
        raise
    os.close(out)
    os.replace(partial, dest)
    return copied


def collect(root, name, dest, limit=None):
    """Copy the result `name` below the trusted directory `root` to the file `dest` of the caller. Returns the
    bytes copied, or None when `read_regular` would read nothing there (absent, a link at any level, a FIFO, a
    directory, a file of another user): then nothing is written. `Uncollected` when it is over `limit` bytes."""
    fd = _open_below(root, name)
    if fd is None:
        return None
    try:
        if not _ours(os.fstat(fd), stat.S_ISREG):
            return None
        Path(dest).parent.mkdir(parents=True, exist_ok=True)
        return _copy_out(fd, dest, limit)
    except OSError:
        return None
    finally:
        os.close(fd)


def copy_tree(fd, dest, limit=None):
    """Copy the directory open as the descriptor `fd` into the directory `dest` (made when absent): directories
    and regular files of this user, each opened from the descriptor of its parent without following a link.
    Returns (bytes copied, skipped): `skipped` are the relative paths of what is anything else (a link, a FIFO, a
    socket, an entry of another user, a directory more than `COLLECT_DEPTH` levels down, an entry that cannot be
    opened), which is never opened as a file and not copied. `Uncollected` when the copy goes over `limit` bytes."""
    state = {"bytes": 0}
    skipped = []

    def walk(here, target, relative, depth):
        os.makedirs(target, mode=0o700, exist_ok=True)
        for entry in sorted(os.listdir(here)):
            path = relative + entry
            try:
                child = os.open(entry, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC, dir_fd=here)
            except OSError:                         # a link (ELOOP), a socket, a directory that cannot be read
                skipped.append(path)
                continue
            try:
                info = os.fstat(child)
                if _ours(info, stat.S_ISDIR) and depth < COLLECT_DEPTH:
                    walk(child, os.path.join(target, entry), path + "/", depth + 1)
                elif _ours(info, stat.S_ISREG):
                    left = None if limit is None else limit - state["bytes"]
                    state["bytes"] += _copy_out(child, os.path.join(target, entry), left)
                else:
                    skipped.append(path)
            finally:
                os.close(child)

    walk(fd, os.fspath(dest), "", 0)
    return state["bytes"], skipped


def collect_tree(root, name, dest, limit=None):
    """`copy_tree` of the directory `name` below the trusted directory `root`, reached without following a link;
    None when it is not a directory of this user reachable that way."""
    try:
        parts = os.fspath(name).split("/")
    except TypeError:
        return None
    if any(part in ("", ".", "..") for part in parts):
        return None
    try:
        fd = os.open(root, os.O_PATH | os.O_DIRECTORY | os.O_CLOEXEC) if not isinstance(root, int) else os.dup(root)
    except OSError:
        return None
    try:
        for i, part in enumerate(parts):
            flags = os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC | (os.O_RDONLY if i == len(parts) - 1 else os.O_PATH)
            below = os.open(part, flags, dir_fd=fd)
            os.close(fd)
            fd = below
        if not _ours(os.fstat(fd), stat.S_ISDIR):
            return None
        return copy_tree(fd, dest, limit)
    except OSError:
        return None
    finally:
        os.close(fd)


def seed(source, dest):
    """Copy what the caller prepared in its own directory `source` (trusted) into `dest`, a directory of a call."""
    fd = os.open(source, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        return copy_tree(fd, dest)
    finally:
        os.close(fd)


def entry_kind(root, name):
    """What `name` below the trusted directory `root` is, reached without following a link: `file` or `directory`
    (of this user), `absent` (nothing there, or nothing where a directory above it should be), or `other` (a link
    at any level, a special file, an entry of another user, a path that is not a relative path of plain
    components)."""
    try:
        parts = os.fspath(name).split("/")
    except TypeError:
        return "other"
    if any(part in ("", ".", "..") for part in parts):
        return "other"
    try:
        fd = os.open(root, os.O_PATH | os.O_DIRECTORY | os.O_CLOEXEC) if not isinstance(root, int) else os.dup(root)
    except OSError:
        return "absent"
    try:
        for i, part in enumerate(parts):
            last = i == len(parts) - 1
            try:
                below = os.open(part, os.O_PATH | os.O_NOFOLLOW | os.O_CLOEXEC | (0 if last else os.O_DIRECTORY), dir_fd=fd)
            except FileNotFoundError:
                return "absent"
            except OSError:
                return "other"
            os.close(fd)
            fd = below
        info = os.fstat(fd)
        if _ours(info, stat.S_ISREG):
            return "file"
        return "directory" if _ours(info, stat.S_ISDIR) else "other"
    except OSError:
        return "other"
    finally:
        os.close(fd)


def tree_bytes(path):
    """Bytes of the regular files under `path` (links not followed), as allocated on disk or by size (`tools.tree_bytes`)."""
    total = 0
    for parent, _, files in os.walk(path, followlinks=False):
        for name in files:
            try:
                info = os.lstat(os.path.join(parent, name))
            except OSError:
                continue
            total += max(info.st_size, info.st_blocks * 512)
    return total


# --- the folder of a call ---------------------------------------------------------------------------

def split_name(name):
    """(prefix of the attempt, kind, n) of the name of a child; ValueError when it is out of the schema."""
    found = NAME.fullmatch(name) if isinstance(name, str) and len(name) <= NAME_LIMIT else None
    if not found:
        raise ValueError("child name: %r" % (name,))
    return found.group(1), found.group(2), int(found.group(3))


def child_name(attempt, kind, n, bench=False):
    """`mdf-eng-<attempt[:16]>-<kind>-<n>`; with `bench`, `attempt` is already the prefix `mdf-bench-<pid>-<8 hex>`."""
    name = "%s-%s-%d" % (attempt if bench else "mdf-eng-" + attempt[:16], kind, n)
    split_name(name)
    return name


_names = itertools.count(1)
_process = {"prefix": None, "launcher": None, "swept": set()}


def launcher():
    """The launcher of this process, behind its constant socket."""
    if _process["launcher"] is None:
        _process["launcher"] = Launcher()
    return _process["launcher"]


def process_prefix():
    """`mdf-bench-<pid>-<8 hex>`: the prefix of the children of a process outside an attempt (the bench, a probe,
    the check of an image); the same for the life of the process, so its children are listed by it."""
    mine = "mdf-bench-%d-" % os.getpid()
    if _process["prefix"] is None or not _process["prefix"].startswith(mine):       # also after a fork
        # imported here: a child imports this module too (its probe), and `random` keeps /dev/urandom open
        import secrets
        _process["prefix"] = mine + secrets.token_hex(4)
    return _process["prefix"]


def prefix_of(attempt):
    """The prefix of the names of the children of `attempt`; of this process when there is no attempt."""
    prefix = process_prefix() if attempt is None else "mdf-eng-" + str(attempt)[:16]
    if not PREFIX.fullmatch(prefix):
        raise ValueError("attempt: %r" % (attempt,))
    return prefix


def open_child(attempt, kind, launcher_=None, work_root=None, **options):
    """A `Child` with a name of its own, `<prefix of the attempt>-<kind>-<n>`: its folder is made exclusively, so a
    name that a call already has is skipped. `isolation_unavailable` when the work volume is not there or not
    writable: without the volume and the launcher there is no other way to run a child."""
    prefix = prefix_of(attempt)
    root = str(work_root or WORK_ROOT)
    if root not in _process["swept"]:
        # at the first child of a process: the folders that calls of dead processes left, whose containers Docker
        # attests absent, go away (H.4). Best effort: with the launcher away nothing is removed.
        _process["swept"].add(root)
        try:
            sweep(launcher_ or launcher(), root)
        except (OSError, Unreachable, Refused):
            pass
    for _ in range(1000):
        name = "%s-%s-%d" % (prefix, kind, next(_names))
        try:
            return Child(launcher_ or launcher(), root, name, **options)
        except FileExistsError:
            continue
        except OSError as exc:
            raise EngineError(Code.ISOLATION_UNAVAILABLE, "work volume %s: %s" % (root, _errname(exc))) from None
    raise EngineError(Code.ISOLATION_UNAVAILABLE, "no free name for a %s child of %s" % (kind, prefix))


class Call:
    """The folder of one call on the work volume, held by its `owner` lock until `close`."""

    def __init__(self, work_root, name):
        self.prefix, self.kind, _ = split_name(name)
        self.name, self.lease, self._renewed = name, 0, 0.0
        calls = Path(work_root) / "call"
        calls.mkdir(mode=0o700, exist_ok=True)
        self.root = calls / name
        self.owner_fd = None
        os.mkdir(self.root, 0o700)                  # exclusive: a folder already there belongs to another call
        try:
            self.owner_fd = os.open(self.root / "owner", os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
            fcntl.flock(self.owner_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            # the supervisor of a `manifest` child may be root without capabilities: it reads `lease` and `stop`
            # only when others may enter `ctl` (a counter and an empty file; `call/<name>` stays 0700)
            shared = self.kind == "manifest"
            for role in CALL_ROLES[self.kind]:
                os.mkdir(self.root / role, 0o755 if shared and role == "ctl" else 0o700)
                os.chmod(self.root / role, 0o755 if shared and role == "ctl" else 0o700)
            self._mode = 0o644 if shared else 0o600
            self._write_ctl("lease", b"0\n")
        except BaseException:
            self.close(remove=True)
            raise

    def path(self, role):
        if role not in CALL_ROLES[self.kind]:
            raise ValueError("no %s in a %s call" % (role, self.kind))
        return self.root / role

    @property
    def scratch(self):
        return self.path("scratch")

    @property
    def stat(self):
        return self.path("stat")

    def _write_ctl(self, name, data):
        ctl = self.root / "ctl"
        fd = os.open(ctl / (name + ".new"), os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW | os.O_CLOEXEC, self._mode)
        try:
            os.write(fd, data)
            os.fchmod(fd, self._mode)
        finally:
            os.close(fd)
        os.replace(ctl / (name + ".new"), ctl / name)

    def stdin(self, data):
        """The stdin of the command, read by the supervisor from `ctl/stdin`."""
        self._write_ctl("stdin", data)

    def add_input(self, source, name):
        """Copy a regular file, or a directory of regular files and directories, to `in/<name>`."""
        if "/" in name or name in ("", ".", ".."):
            raise ValueError("input name: " + name)
        source, dest = Path(source), self.path("in") / name
        if source.is_symlink():
            raise ValueError("input is a link: " + name)
        if source.is_dir():
            for parent, dirs, files in os.walk(source, followlinks=False):
                for entry in dirs + files:
                    info = os.lstat(os.path.join(parent, entry))
                    if not (stat.S_ISDIR(info.st_mode) or stat.S_ISREG(info.st_mode)):
                        raise ValueError("input holds a link or a special file: " + name)
            shutil.copytree(source, dest, symlinks=True)
        elif source.is_file():
            shutil.copyfile(source, dest, follow_symlinks=False)
        else:
            raise ValueError("input is not a regular file or a directory: " + name)
        return "/in/" + name

    def renew(self, now=None):
        """Renew the lease, at most once a second: a new counter, moved into place atomically."""
        now = time.monotonic() if now is None else now
        if now - self._renewed >= 1.0 or self.lease == 0:
            self.lease += 1
            self._write_ctl("lease", b"%d\n" % self.lease)
            self._renewed = now

    def stop(self):
        """Ask the supervisor to end the container: it sees `ctl/stop` within a tick, without Docker."""
        self._write_ctl("stop", b"")

    def metrics(self):
        return supervisor.last_metrics(read_regular(self.root, "stat/metrics", STAT_LIMIT, tail=True))

    def exit_hint(self):
        """True when the supervisor wrote `stat/exit`: a hint to ask Docker, never the end itself."""
        return bool((read_regular(self.root, "stat/exit", 64) or b"").strip())

    def close(self, remove):
        """Release `owner`; with `remove`, take the folder away first (an attested end). Returns False when the
        folder was to go and is still there (something in it cannot be removed: `sweep` will find it, and say
        the same), True otherwise."""
        gone = True
        if remove:
            shutil.rmtree(self.root, ignore_errors=True)
            gone = not os.path.lexists(self.root)
        if self.owner_fd is not None:
            os.close(self.owner_fd)
            self.owner_fd = None
        return gone


def socket_dirs(run_root, prefix):
    """Make `sock/<prefix>/relay` and `sock/<prefix>/llama` (0700) on the run volume; returns their parent."""
    if not PREFIX.fullmatch(prefix):
        raise ValueError("attempt prefix: " + prefix)
    base = Path(run_root) / "sock"
    base.mkdir(mode=0o700, exist_ok=True)
    root = base / prefix
    root.mkdir(mode=0o700, exist_ok=True)
    for role in SOCKET_ROLES:
        (root / role).mkdir(mode=0o700, exist_ok=True)
    return root


# --- one child --------------------------------------------------------------------------------------

class Outcome:
    """How a child ended, as Docker attested it.

    `cause`: `exited` (the command's own status in `exit`), `memory` (`OOMKilled`, 137 without a stop of ours or of
    the launcher, or an OOM kill in the metrics), `lease` (200, or killed by the launcher at its lease: the caller
    was stuck or dead), `term` (201, or killed by the launcher at its term), `stopped` (202 or a kill after
    `reason`, the caller's own reason), `stat` (203: `stat/` not usable), `interrupted` (killed by the stop of
    the launcher itself: the service is going down or restarting), `vanished` (gone before the caller saw its
    exit).

    A child the launcher killed has no exit status of its own (137 while it is still there, none once removed):
    its cause is the launcher's reason (`reaped` of `inspect`), never `memory`."""

    def __init__(self, name, cause, exit_status, oom_killed, metrics, stdout, stderr, seconds, reason):
        self.name, self.cause, self.exit, self.oom_killed = name, cause, exit_status, oom_killed
        self.metrics, self.stdout, self.stderr, self.seconds, self.reason = metrics or {}, stdout, stderr, seconds, reason

    def step(self):
        """The fields of the recorded step (H.4)."""
        cpu = self.metrics.get("cpu_usec")
        return {"container": self.name, "exit": self.exit, "seconds": self.seconds,
                "memory_peak_bytes": self.metrics.get("memory_peak_bytes"),
                "cpu_seconds": None if cpu is None else round(cpu / 1e6, 1), "oom_kill": self.metrics.get("oom_kill"),
                "oom_killed": self.oom_killed}


def error_of(outcome, what, tool=False):
    """The closed code of an outcome that is not the command's own exit (H.5), or None. A stop of the caller is
    the caller's own reason, never mapped here."""
    C = Code
    codes = {"memory": C.TOOL_MEMORY_LIMIT if tool else C.MEMORY_LIMIT, "lease": C.INTERRUPTED,
             "term": C.TOOL_TIMEOUT if tool else C.TIMEOUT, "stat": C.ENGINE_FAILED, "vanished": C.ENGINE_FAILED,
             "interrupted": C.INTERRUPTED}
    if outcome.cause in codes:
        return EngineError(codes[outcome.cause], "%s: %s (container %s, exit %s)" % (what, outcome.cause, outcome.name, outcome.exit))
    return None


def _cause(exit_status, oom_killed, metrics, reason, reaped=None, absent=False):
    if reason is not None:
        return "stopped"
    if reaped in ("lease", "term"):             # the launcher's own watch: its reason, whatever the status reads
        return reaped
    if reaped == "stop":                        # the stop of the launcher: an interruption, not a failure of the engine
        return "interrupted"
    if absent:
        return "vanished"
    if oom_killed or exit_status == 137 or (metrics or {}).get("oom_kill"):
        return "memory"
    return {supervisor.EXIT_LEASE: "lease", supervisor.EXIT_TERM: "term", supervisor.EXIT_STOP: "stopped",
            supervisor.EXIT_STAT: "stat"}.get(exit_status, "exited")


class Child:
    """One call: the folder, the request to the launcher, the wait, the attested end.

        with Child(launcher, work_root, name) as child:
            child.call.add_input(...); (child.call.scratch / "limits.json").write_text(...)
            outcome = child.run(image, argv, variables, limits, timeout=..., scratch_mib=..., on_poll=...)
            ... take the results from child.call.scratch ...

    Leaving the block removes the folder after an attested end; after an end that is not attested (the launcher
    unreachable, a removal not confirmed: `isolation_unavailable` was raised) `ctl/stop` is written and the folder
    stays where it is. A caller that leaves the block in the middle of the wait for another reason (an interrupt)
    has its container killed and removed on the way out. `removed` says afterwards whether the folder is gone: False also when its removal failed.
    `clock` and `sleep` are the monotonic clock and the pause of the wait (tests inject them)."""

    def __init__(self, launcher, work_root, name, clock=time.monotonic, sleep=time.sleep):
        self.launcher, self.name, self.clock, self.sleep = launcher, name, clock, sleep
        self.call = Call(work_root, name)
        self.attested = True                # nothing was started yet: nothing to attest
        self.removed = False                # the folder of the call is gone (known when the block is left)
        self._down_since = None
        self.started = self._checked = clock()
        self._hinted = False
        self.unconfirmed = False            # the end came and the removal of the container was not confirmed

    def __enter__(self):
        return self

    def __exit__(self, kind, exc, trace):
        if not self.attested:
            try:
                self.call.stop()
            except OSError:
                pass
            if exc is not None and not (isinstance(exc, EngineError) and exc.code is Code.ISOLATION_UNAVAILABLE):
                # the caller leaves in the middle of the wait (Ctrl-C, a defect of its own): its container is
                # killed and removed now, when the launcher answers; when it does not, the stop above ends it
                try:
                    self.launcher.kill(self.name)
                    self.attested = bool(self.launcher.remove(self.name).get("absent"))
                except (Unreachable, Refused, OSError):
                    pass
        self.removed = self.call.close(remove=self.attested) and self.attested

    # -- the launcher, with the 20 seconds of patience ------------------------------------------------
    def _ask(self, op, *args):
        """An operation on the child; None while the launcher does not answer, `isolation_unavailable` (the end is
        not attested) once it has not answered for `UNREACHABLE_SECONDS`."""
        try:
            answer = getattr(self.launcher, op)(*args)
        except Unreachable as exc:
            now = self.clock()
            if self._down_since is None:
                self._down_since = now
            if now - self._down_since >= UNREACHABLE_SECONDS:
                raise self._unattested("launcher unreachable for %d s (%s)" % (UNREACHABLE_SECONDS, exc)) from None
            return None
        self._down_since = None
        return answer

    def _unattested(self, detail):
        self.attested = False
        try:
            self.call.stop()
        except OSError:
            pass
        return EngineError(Code.ISOLATION_UNAVAILABLE, "%s: end not attested, call folder kept: %s" % (self.name, detail))

    def _insist(self, op, *args):
        """Like `_ask`, waiting through the unreachable seconds (the stop path and the end)."""
        while True:
            answer = self._ask(op, *args)
            if answer is not None:
                return answer
            self.sleep(0.5)

    # -- the call ---------------------------------------------------------------------------------------
    def begin(self, image, argv, variables, limits, before=None):
        """Ask the launcher for the child. `before(name)` is called before the creation (the observer that records
        the name of the container: K7) and may raise. `isolation_unavailable` when the launcher does not answer;
        its closed refusals (`invalid_setting`, `toolchain_missing`, `toolchain_mismatch`) as `EngineError`."""
        if before is not None:
            before(self.name)
        self.started = self._checked = self.clock()
        self._hinted = False
        self.attested = False               # from here on a container may exist
        try:
            self.launcher.run(self.call.kind, self.name, image, argv, variables, limits)
        except Refused as exc:              # a closed refusal: the launcher created nothing, or removed what it created
            self.attested = True
            code = next((c for c in Code if c.value == exc.code), Code.ENGINE_FAILED)
            raise EngineError(code, "%s: %s" % (self.name, exc.detail)) from None
        except Unreachable as exc:
            raise self._unattested("run: %s" % exc) from None

    def ended(self, now=None):
        """One step of the wait: renew the lease and look for the end. Returns what Docker says of the container,
        through the launcher, once it no longer runs; None while it runs or the launcher does not answer."""
        call, has_stat = self.call, "stat" in CALL_ROLES[self.call.kind]
        now = self.clock() if now is None else now
        call.renew(now)
        every = INSPECT_SECONDS if has_stat else HINT_SECONDS    # without `stat/` there is no hint: ask more often
        # the hint makes the caller ask at once; a hint Docker did not confirm is asked again a second later
        if now - self._checked >= every or (has_stat and now - self._checked >= HINT_SECONDS * self._hinted and call.exit_hint()):
            self._hinted = bool(has_stat and call.exit_hint())
            self._checked = now
            state = self._ask("inspect", self.name)
            if state is not None and state["state"] in ("exited", "dead", "absent"):
                return state
        return None

    def run(self, image, argv, variables, limits, timeout=None, poll_seconds=0.1, scratch_mib=None,
            log_limit=LOG_LIMIT, on_poll=None, before=None):
        """Start the child and wait for its attested end; returns its `Outcome`.

        The wait wakes every `WAKE_SECONDS` at most (the lease, the end, `timeout`); every `poll_seconds` it
        measures `scratch` against `scratch_mib` (the cap of `scratch` as a whole) and the two streams against
        `log_limit`, which stop the child with the reason `"output_limit"`, and calls `on_poll(child, metrics)`,
        which returns None or the caller's reason to stop it (any object: an `EngineError` of the caller).
        `timeout` seconds stop it with the reason `"timeout"`. See `begin` for `before` and the refusals."""
        call, kind = self.call, self.call.kind
        self.begin(image, argv, variables, limits, before)
        reason, polled = None, None
        has_stat = "stat" in CALL_ROLES[kind]
        while True:
            now = self.clock()
            state = self.ended(now)
            if state is not None:
                break
            if timeout is not None and now - self.started > timeout:
                reason = "timeout"
            elif polled is None or now - polled >= poll_seconds:
                polled = now
                if scratch_mib is not None and "scratch" in CALL_ROLES[kind] and tree_bytes(call.scratch) > scratch_mib * 1048576:
                    reason = "output_limit"
                elif has_stat and regular_size(call.root, "stat/stdout") + regular_size(call.root, "stat/stderr") > log_limit:
                    reason = "output_limit"
                elif on_poll is not None:
                    reason = on_poll(self, call.metrics() if has_stat else None)
            if reason is not None:
                state = self._stop()
                break
            self.sleep(min(poll_seconds, WAKE_SECONDS))
        return self._end(state, reason, log_limit)

    def end(self, reason=None, log_limit=LOG_LIMIT):
        """End a child started with `begin` (a program that runs until its caller stops it): our own stop with
        `reason`, unless it is already over, then its attested end."""
        state = self.ended()
        if state is None:
            state = self._stop()
        else:
            reason = None
        return self._end(state, reason, log_limit)

    def _stop(self):
        """Our own stop: `ctl/stop`, up to `STOP_WAIT_SECONDS` for the supervisor, then `kill` through the launcher."""
        self.call.stop()
        asked = self.clock()
        while True:
            state = self._insist("inspect", self.name)
            if state["state"] in ("exited", "dead", "absent"):
                return state
            if self.clock() - asked >= STOP_WAIT_SECONDS:
                self._insist("kill", self.name)
                asked = self.clock()
            self.sleep(0.1)

    def _end(self, state, reason, log_limit):
        call, started = self.call, self.started
        logged = b""
        if "stat" not in CALL_ROLES[call.kind] and state["state"] in ("exited", "dead"):
            try:                            # a `manifest` child: its stdout is the log of the container, read before the removal
                logged = self._insist("logs", self.name)
            except Refused:
                logged = b""
        try:
            removed = self._insist("remove", self.name)
            if not removed.get("absent"):
                raise self._unattested("removal not confirmed")
        except EngineError:
            self.unconfirmed = True
            raise
        self.attested = True
        has_stat = "stat" in CALL_ROLES[call.kind]
        metrics = call.metrics() if has_stat else None
        stdout = (read_regular(call.root, "stat/stdout", log_limit) or b"") if has_stat else logged
        stderr = (read_regular(call.root, "stat/stderr", log_limit) or b"") if has_stat else b""
        absent = state["state"] == "absent"
        exit_status = None if absent else state.get("exit")
        oom_killed = bool(state.get("oom_killed"))
        cause = _cause(exit_status, oom_killed, metrics, reason, state.get("reaped"), absent)
        return Outcome(self.name, cause, exit_status, oom_killed, metrics, stdout, stderr,
                       round(self.clock() - started, 3), reason)


# --- attempts: attestation and collection -----------------------------------------------------------

def stop_attempt_children(launcher, prefix):
    """Attest that no child of the attempt `prefix` is left: list, kill, remove, confirmed absence, list again
    empty. Returns the names removed; `Unattested` when the launcher is unreachable or a removal is not confirmed."""
    if not PREFIX.fullmatch(prefix):
        raise ValueError("attempt prefix: " + prefix)
    try:
        names = [c["name"] for c in launcher.list(prefix)]
        for name in names:
            launcher.kill(name)
            if not launcher.remove(name).get("absent"):
                raise Unattested("%s: removal not confirmed" % name)
        left = [c["name"] for c in launcher.list(prefix)]
    except Unreachable as exc:
        raise Unattested("launcher: %s" % exc) from None
    if left:
        raise Unattested("children still listed: " + ", ".join(left))
    return names


def sweep(launcher, work_root, prefix=None, min_age=ORPHAN_AGE):
    """Remove the folders of `call/` that have neither an owner nor a container: `owner` free (the caller ended or
    died) and the name attested absent by the launcher. With the launcher unreachable nothing is removed.

    Returns (removed, kept): the names of the folders that are NO LONGER THERE, and `{name: why}` for the others,
    with `why` of `SWEEP_KEPT`. A folder whose removal failed (something in it cannot be removed) is kept, as
    `not_removed`: a name is never reported as removed while its folder is on the volume. `prefix` keeps the sweep
    to one attempt."""
    removed, kept = [], {}
    calls = Path(work_root) / "call"
    try:
        entries = sorted(os.listdir(calls))
    except OSError:
        return removed, kept
    for name in entries:
        try:
            found = split_name(name)
        except ValueError:
            continue
        if prefix is not None and found[0] != prefix:
            continue
        root = calls / name
        try:
            info = os.lstat(root)
            if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid():
                continue
            fd = os.open(root / "owner", os.O_RDWR | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK)
        except FileNotFoundError:               # killed between the folder and its lock: no owner
            if time.time() - info.st_mtime < min_age:
                kept[name] = "young"
                continue
            fd = None
        except OSError:
            kept[name] = "owner"
            continue
        try:
            if fd is not None:
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:         # a live call
                    kept[name] = "held"
                    continue
            try:
                absent = launcher.inspect(name)["state"] == "absent"
            except (Unreachable, Refused):
                kept[name] = "unattested"
                continue
            if not absent:
                kept[name] = "container"
                continue
            shutil.rmtree(root, ignore_errors=True)
            if os.path.lexists(root):       # something in it cannot be removed: its permissions back, once, and again
                restore_permissions(root)
                shutil.rmtree(root, ignore_errors=True)
            if os.path.lexists(root):
                kept[name] = "not_removed"
            else:
                removed.append(name)
        finally:
            if fd is not None:
                os.close(fd)
    return removed, kept


def restore_permissions(root, depth=COLLECT_DEPTH * 4):
    """Give every directory of this user below `root` (itself included) the mode 0700 again, so that what a
    command left with its permissions taken away can be removed. Only for the folder of a call whose owner is free
    and whose container is attested absent (`sweep`). Each directory is reached from the descriptor of its parent,
    without following a link, and changed through that descriptor: nothing outside `root` is touched, whatever
    the names in it point to. Best effort: what it cannot open stays as it is."""
    def walk(fd, left):
        try:
            if not _ours(os.fstat(fd), stat.S_ISDIR):
                return
            os.chmod("/proc/self/fd/%d" % fd, 0o700)        # the inode of the descriptor: a path opened with O_PATH has no fchmod
            names = os.listdir("/proc/self/fd/%d" % fd)
        except OSError:
            return
        if left <= 0:
            return
        for name in names:
            try:
                info = os.stat(name, dir_fd=fd, follow_symlinks=False)
                if not stat.S_ISDIR(info.st_mode):
                    continue
                below = os.open(name, os.O_PATH | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=fd)
            except OSError:
                continue
            try:
                walk(below, left - 1)
            finally:
                os.close(below)
    try:
        top = os.open(root, os.O_PATH | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
    except OSError:
        return
    try:
        walk(top, depth)
    finally:
        os.close(top)


class Renewer(threading.Thread):
    """Renews the lease of a child that runs while its caller does other work (`begin` without `run`: the local
    CPU server): once a second, for as long as this process lives and until `stop`."""

    def __init__(self, call):
        super().__init__(name="lease-" + call.name, daemon=True)
        self.call, self.done = call, threading.Event()

    def run(self):
        while not self.done.wait(1.0):
            try:
                self.call.renew()
            except OSError:
                pass

    def stop(self):
        self.done.set()
        self.join(timeout=5)


# --- the payload gate (J.8.2) -----------------------------------------------------------------------

def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _listing_sha256(pairs):
    return hashlib.sha256("".join("%s  %s\n" % (sha, name) for name, sha in sorted(pairs)).encode("utf-8")).hexdigest()


def payload_sha256(package=None):
    """SHA-256 of the sorted list (path, SHA-256) of the files of an installed `markdown_for_llms/`, without
    `__pycache__` and `.pyc`: the code a process runs, read from the files it runs."""
    package = Path(package) if package is not None else Path(__file__).resolve().parents[1]
    pairs = []
    for parent, dirs, files in os.walk(package, followlinks=False):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        for name in files:
            if not name.endswith(".pyc"):
                path = Path(parent) / name
                pairs.append((package.name + "/" + path.relative_to(package).as_posix(), _sha256_file(path)))
    return _listing_sha256(pairs)


def inputs_sha256(root, names=INPUTS):
    """SHA-256 of the sorted list (path, SHA-256) of the build inputs of the image `engine` under `root`."""
    return _listing_sha256([(name, _sha256_file(Path(root) / name)) for name in names])


def environment():
    """The interpreter that runs this process and the distributions installed in its environment, project excluded."""
    from importlib import metadata
    names = sorted({"%s==%s" % (d.metadata["Name"].lower().replace("_", "-"), d.version) for d in metadata.distributions()
                    if d.metadata["Name"] and d.metadata["Name"].lower().replace("_", "-") != "markdown-for-llms"})
    return {"python_sha256": _sha256_file(os.path.realpath(sys.executable)), "distributions": names}


def in_product_image(inputs_file=INPUTS_FILE):
    """Whether this process runs from the image `engine` of the product (it recorded its build inputs): a
    container, which does not reach the loopback of the host. The suite and a process of the host do not."""
    return Path(inputs_file).is_file()


def identity(inputs_file=INPUTS_FILE):
    """What a process of an image reports of itself for the gate: the code it runs, the build inputs its image
    recorded (None outside an image) and its environment."""
    try:
        recorded = json.loads(Path(inputs_file).read_text(encoding="utf-8"))["inputs_sha256"]
    except (OSError, ValueError, KeyError, TypeError):
        recorded = None
    return {"payload_sha256": payload_sha256(), "inputs_sha256": recorded, "environment": environment()}


def gate(reference, launcher, child):
    """The three comparisons between the copy under test (`reference`: its `payload_sha256`, the `inputs_sha256`
    computed on its files, its `environment`), the launcher (`hello`) and a child (the probe). Returns the list of
    the comparisons that failed, of `code`, `inputs`, `environment`: empty when the children run what is under test.

    environment: the child runs the interpreter of the reference, every distribution of the child is in the
    reference at the same version (the reference has the `dev` group on top), and launcher and child agree."""
    failed = []
    if not (reference["payload_sha256"] == launcher["payload_sha256"] == child["payload_sha256"]):
        failed.append("code")
    if not (reference["inputs_sha256"] is not None
            and reference["inputs_sha256"] == launcher["inputs_sha256"] == child["inputs_sha256"]):
        failed.append("inputs")
    ref, lau, chi = reference["environment"], launcher["environment"], child["environment"]
    if not (chi["python_sha256"] == ref["python_sha256"] and set(chi["distributions"]) <= set(ref["distributions"])
            and lau == chi):
        failed.append("environment")
    return failed


def wheel_payload_sha256(wheel):
    """`payload_sha256` of the files of `markdown_for_llms/` in a wheel: what an installation of it runs."""
    import zipfile
    pairs = []
    with zipfile.ZipFile(wheel) as archive:
        for name in archive.namelist():
            parts = name.split("/")
            if parts[0] == "markdown_for_llms" and not name.endswith("/") and "__pycache__" not in parts and not name.endswith(".pyc"):
                pairs.append((name, hashlib.sha256(archive.read(name)).hexdigest()))
    return _listing_sha256(pairs)


def main(argv=None):
    """`python -m markdown_for_llms.engine.containers gate --repo DIR --wheel FILE --output FILE`: step C of the
    full check. The reference is the copy under test: the code of this installation, which must be the code of
    the canonical wheel; the build inputs computed on the files of `DIR`; this environment. The launcher answers
    `hello`, a child of it runs the probe. Exit 1 with `stale engine image (...)` when a comparison fails."""
    import argparse
    import tempfile
    parser = argparse.ArgumentParser(prog="markdown_for_llms.engine.containers")
    parser.add_argument("command", choices=["gate"])
    parser.add_argument("--repo", required=True)
    parser.add_argument("--wheel", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    from . import isolation
    mine = identity()
    wheel = wheel_payload_sha256(args.wheel)
    reference = {"payload_sha256": mine["payload_sha256"], "inputs_sha256": inputs_sha256(args.repo),
                 "environment": mine["environment"]}
    record = {"reference": reference, "wheel_payload_sha256": wheel, "launcher": None, "child": None, "failed": None}
    try:
        hello = launcher().hello()
        record["launcher"] = {k: hello.get(k) for k in ("payload_sha256", "inputs_sha256", "environment", "images", "docker", "project")}
        with tempfile.TemporaryDirectory(prefix="mdf-gate-") as scratch:
            status, log = isolation.run(Path(scratch) / "probe", PROBE_MODULE, ["container"], timeout=120, results=("probe.json",))
            report = json.loads((Path(scratch) / "probe" / "probe.json").read_bytes())
        record["child"] = report["identity"]
        failed = gate(reference, hello, report["identity"])
        if wheel != mine["payload_sha256"] and "code" not in failed:
            failed.insert(0, "code")
    except (Unreachable, Refused, EngineError, OSError, ValueError, KeyError) as exc:
        record["error"] = "%s: %s" % (type(exc).__name__, exc)
        Path(args.output).write_text(json.dumps(record, indent=1, sort_keys=True))
        print("the launcher or its child did not answer: %s" % record["error"])
        return 1
    record["failed"] = failed
    Path(args.output).write_text(json.dumps(record, indent=1, sort_keys=True))
    if failed:
        print("stale engine image (%s)" % " | ".join(failed))
        return 1
    print("ok code %s inputs %s environment %d distributions, image %s" % (
        reference["payload_sha256"][:16], reference["inputs_sha256"][:16], len(reference["environment"]["distributions"]),
        ((hello.get("images") or {}).get("app") or {}).get("id")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
