"""Isolation probe of the light child, run inside it by `availability` and the tests.

It tries what the child must not be able to do and writes the outcomes to /scratch/probe.json:
a TCP connection to the loopback and a DNS lookup (no network namespace but its own empty
loopback), reading a host path given by the caller (not mounted), writing outside /scratch (a
read-only root), and the Docker socket (absent). Each entry is true when the attempt failed.

`container` (MFL051, tranche A1) is the probe of a child that runs as a container started by the
launcher: the six entries above and what the container must show from inside (user, capabilities,
no-new-privileges, seccomp, interfaces, the exact names of its environment, the limits of its
cgroup, its mounts, the supervisor as PID 1, where its stdin comes from, the identity of the code
it runs for the payload gate) and what it must not be able to do (write under `/in` and `/ctl`,
directly and through the aliases of its root in /proc, reach them from a writable mount). When the
caller put a `steps.json` in the scratch, its steps run after the probe, each recorded in the
report: the behaviours the component tests need from a child (memory, processes, files, a FIFO in
`/stat`, signals to PID 1, a socket under `/relay`). A step does nothing a child could not do.
"""
import json
import os
from pathlib import Path
import socket


def _fails(action):
    try:
        action()
    except Exception:
        return True
    return False


def main(args):
    if args[:1] == ["container"]:
        return container(args[1:])
    if args[:1] == ["allocate"]:          # a process that grows past a memory cap (tests of the scope)
        held = []
        for _ in range(int(args[1])):
            held.append(bytearray(1 << 20))
            held[-1][::4096] = b"x" * len(held[-1][::4096])
        return 0
    host_path = args[0] if args else "/home"

    def connect():
        s = socket.create_connection(("127.0.0.1", 9), timeout=1)
        s.close()

    def write_root():
        Path("/usr/mdf-probe").write_text("x")
    result = {
        "loopback_refused": _fails(connect),
        "dns_failed": _fails(lambda: socket.getaddrinfo("example.com", 443)),
        "host_path_hidden": not os.path.exists(host_path),
        "root_read_only": _fails(write_root),
        "docker_socket_absent": not os.path.exists("/var/run/docker.sock") and not os.path.exists("/run/docker.sock"),
        "scratch_writable": not _fails(lambda: Path("/scratch/probe-write").write_text("x")),
    }
    Path("/scratch/probe.json").write_text(json.dumps(result))
    return 0


# --- `container`: the probe of a child started by the launcher (MFL051) --------------------------

REPORT = "/scratch/probe.json"
STEPS = "/scratch/steps.json"
ALIASES = ("", "/proc/self/root", "/proc/1/root")
WALK_LIMIT = 5000


def _error(action):
    """`ok` or the name of the errno of what `action` raised."""
    import errno
    try:
        action()
    except OSError as exc:
        return errno.errorcode.get(exc.errno, "errno %s" % exc.errno)
    except Exception as exc:
        return type(exc).__name__
    return "ok"


def _read(path):
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def _status():
    fields = {}
    for line in (_read("/proc/self/status") or "").splitlines():
        key, _, value = line.partition(":")
        fields[key] = value.strip()
    return fields


def _mounts():
    out = []
    for line in (_read("/proc/self/mountinfo") or "").splitlines():
        left, _, right = line.partition(" - ")
        parts, kind = left.split(), right.split()
        out.append({"target": parts[4], "root": parts[3], "read_only": "ro" in parts[5].split(","),
                    "type": kind[0] if kind else None, "source": kind[1] if len(kind) > 1 else None})
    return out


def _inode(path, follow=True):
    try:
        info = os.stat(path) if follow else os.lstat(path)
    except OSError:
        return None
    return [info.st_dev, info.st_ino]


def _try_write(path):
    def action():
        fd = os.open(path, os.O_WRONLY | os.O_CREAT, 0o600)
        os.close(fd)
    return _error(action)


def _reach(root, wanted):
    """The paths under the writable mount `root` whose inode is one of `wanted` (never expected), within a bound."""
    found, seen = [], 0
    for parent, dirs, _ in os.walk(root, followlinks=False):
        for name in [""] + dirs:
            path = os.path.join(parent, name) if name else parent
            seen += 1
            if _inode(path, follow=False) in wanted:
                found.append(path)
        if seen > WALK_LIMIT:
            break
    return found


def _report():
    status = _status()
    lines = (_read("/proc/net/dev") or "").splitlines()[2:]

    def connect(address):
        def action():
            with socket.socket() as sock:
                sock.settimeout(0.5)
                sock.connect(address)
        return _error(action)
    mounts = _mounts()
    protected = {name: _inode(name) for name in ("/in", "/ctl") if os.path.isdir(name)}
    writable = sorted(m["target"] for m in mounts if not m["read_only"] and os.access(m["target"], os.W_OK)
                      and os.path.isdir(m["target"]) and not m["target"].startswith(("/proc", "/sys", "/dev")))
    writes = {}
    for alias in ALIASES:
        for name in ("/in/mdf-probe", "/ctl/mdf-probe", "/ctl/lease", "/ctl/stop", "/usr/mdf-probe"):
            writes[alias + name] = _try_write(alias + name)
    for name in ("/in", "/ctl"):
        for inside in sorted(os.listdir(name)) if os.path.isdir(name) else []:
            writes[name + "/" + inside + " (remove)"] = _error(lambda: os.unlink(name + "/" + inside))
            writes[name + "/" + inside + " (link)"] = _error(lambda: os.symlink("/scratch", name + "/" + inside + ".lnk"))
    import resource
    limits = {name: list(resource.getrlimit(getattr(resource, "RLIMIT_" + name)))
              for name in ("AS", "FSIZE", "NOFILE", "NPROC", "CORE", "CPU", "DATA")}
    from . import containers
    return {
        "loopback_refused": connect(("127.0.0.1", 9)) != "ok",
        "dns_failed": _fails(lambda: socket.getaddrinfo("example.com", 443)),
        "host_path_hidden": not os.path.exists("/home") or not os.listdir("/home"),
        "root_read_only": _try_write("/usr/mdf-probe") != "ok",
        "docker_socket_absent": not os.path.exists("/var/run/docker.sock") and not os.path.exists("/run/docker.sock"),
        "scratch_writable": _try_write("/scratch/probe-write") == "ok",
        "uid": os.getuid(), "gid": os.getgid(), "groups": sorted(os.getgroups()),
        "cap_eff": status.get("CapEff"), "cap_prm": status.get("CapPrm"), "cap_bnd": status.get("CapBnd"),
        "no_new_privs": status.get("NoNewPrivs"), "seccomp": status.get("Seccomp"),
        "interfaces": [line.split(":", 1)[0].strip() for line in lines],
        "outside": connect(("198.51.100.1", 9)),
        "environment": sorted(os.environ),
        "cgroup": {name: _read("/sys/fs/cgroup/" + name) for name in ("memory.max", "memory.swap.max", "pids.max",
                                                                     "memory.peak", "cgroup.controllers")},
        "mounts": mounts,
        "pid": os.getpid(), "ppid": os.getppid(),
        "pid1": [arg for i, arg in enumerate((_read("/proc/1/cmdline") or "").split("\0")[:4]) if i in (0, 3)],
        "ptrace_scope": _read("/proc/sys/kernel/yama/ptrace_scope"),
        "stdin": _readlink("/proc/self/fd/0"),
        "tty": os.isatty(0) or os.isatty(1) or os.isatty(2),
        "controlling_tty": _error(lambda: os.close(os.open("/dev/tty", os.O_RDWR))),
        "descriptors": sorted(int(n) for n in os.listdir("/proc/self/fd") if n.isdigit())[:-1],
        "limits": limits,
        "writes": writes,
        "inodes": {name: _inode(name) for name in ("/", "/scratch", "/scratch/..", "/stat", "/stat/..", "/in", "/ctl",
                                                   "/proc/self/root/ctl", "/proc/1/root/ctl", "/proc/self/root/in")},
        "writable_mounts": writable,
        "reach": {root: _reach(root, [i for i in protected.values() if i]) for root in writable},
        "identity": containers.identity(),
        "cwd": os.getcwd(),
    }


def _readlink(path):
    try:
        return os.readlink(path)
    except OSError:
        return None


def _save(report):
    tmp = REPORT + ".new"
    Path(tmp).write_text(json.dumps(report))
    os.replace(tmp, REPORT)


def _step(step):
    """One step of `steps.json`; returns what the report records for it."""
    import hashlib
    import signal
    import sys
    import time
    do = step.get("do")
    if do == "allocate":                    # grow by `mib`, touched; kept unless `release`
        held = []
        for _ in range(int(step["mib"])):
            held.append(bytearray(1 << 20))
            held[-1][::4096] = b"x" * len(held[-1][::4096])
        if step.get("release"):
            del held
            return "released"
        _step.held = getattr(_step, "held", []) + [held]
        return "held"
    if do == "sleep":
        time.sleep(float(step["seconds"]))
        return "slept"
    if do == "spin":                        # use the CPU for `seconds`
        until = time.monotonic() + float(step["seconds"])
        while time.monotonic() < until:
            pass
        return "spun"
    if do == "write":                       # `mib` MiB into `path`
        def action():
            with open(step["path"], "wb") as stream:
                for _ in range(int(step.get("mib", 0))):
                    stream.write(b"\0" * (1 << 20))
                stream.write(step.get("text", "").encode())
        return _error(action)
    if do == "append":
        def action():
            with open(step["path"], "ab") as stream:
                stream.write(step["text"].encode())
        return _error(action)
    if do == "fork":                        # `count` children that wait; how many started and the first error
        started, error, pids = 0, None, []
        for _ in range(int(step["count"])):
            try:
                pid = os.fork()
            except OSError as exc:
                import errno
                error = errno.errorcode.get(exc.errno, str(exc.errno))
                break
            if pid == 0:
                time.sleep(float(step.get("seconds", 2)))
                os._exit(0)
            pids.append(pid)
            started += 1
        for pid in pids:
            os.kill(pid, signal.SIGKILL)
            os.waitpid(pid, 0)
        return {"started": started, "error": error}
    if do == "threads":
        import threading
        stop, started, error = threading.Event(), 0, None
        for _ in range(int(step["count"])):
            try:
                threading.Thread(target=stop.wait, daemon=True).start()
                started += 1
            except RuntimeError as exc:
                error = str(exc)[:60]
                break
        stop.set()
        return {"started": started, "error": error}
    if do == "child_allocate":              # a subprocess grows by `mib`: the kernel kills it, this process stays
        pid = os.fork()
        if pid == 0:
            held = []
            for _ in range(int(step["mib"])):
                held.append(bytearray(1 << 20))
                held[-1][::4096] = b"x" * len(held[-1][::4096])
            os._exit(0)
        _, status = os.waitpid(pid, 0)
        return {"signal": os.WTERMSIG(status) if os.WIFSIGNALED(status) else None,
                "exit": os.WEXITSTATUS(status) if os.WIFEXITED(status) else None}
    if do == "mkfifo":
        def action():
            try:
                os.unlink(step["path"])
            except FileNotFoundError:
                pass
            os.mkfifo(step["path"])
        return _error(action)
    if do == "remove":
        def action():
            import shutil
            for name in os.listdir(step["path"]):
                target = os.path.join(step["path"], name)
                shutil.rmtree(target) if os.path.isdir(target) and not os.path.islink(target) else os.unlink(target)
        return _error(action)
    if do == "kill":                        # a signal to a process: `self`, or a PID (1 is the supervisor)
        target = os.getpid() if step["pid"] == "self" else int(step["pid"])
        return _error(lambda: os.kill(target, getattr(signal, "SIG" + step["signal"])))
    if do == "flood":                       # write to a path (the pipe of PID 1) for `seconds`
        def action():
            until = time.monotonic() + float(step["seconds"])
            with open(step["path"], "wb", buffering=0) as stream:
                os.set_blocking(stream.fileno(), False)
                while time.monotonic() < until:
                    try:
                        stream.write(b"x\n" if step.get("newline", True) else b"x")
                    except BlockingIOError:
                        time.sleep(0.001)
        return _error(action)
    if do == "connect":                     # `repeat` requests on a Unix socket, each answered by one line
        replies = []
        for _ in range(int(step.get("repeat", 1))):
            def action():
                with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
                    sock.settimeout(5)
                    sock.connect(step["path"])
                    sock.sendall(step.get("send", "ping").encode() + b"\n")
                    replies.append(sock.recv(4096).decode("utf-8", "replace").strip())
            error = _error(action)
            if error != "ok":
                replies.append("error " + error)
            time.sleep(float(step.get("interval", 0)))
        return replies
    if do == "stdin":
        data = sys.stdin.buffer.read()
        return {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    if do == "print":
        getattr(sys, step.get("stream", "stdout")).write(step["text"])
        getattr(sys, step.get("stream", "stdout")).flush()
        return "printed"
    if do == "stat":
        return {path: _inode(path) for path in step["paths"]}
    if do == "walk":                        # every inode reachable under a directory, within a bound
        seen = []
        for parent, dirs, files in os.walk(step["path"], followlinks=False):
            seen += [_inode(os.path.join(parent, name), follow=False) for name in [""] + dirs + files]
            if len(seen) > WALK_LIMIT:
                break
        return [i for i in seen if i]
    if do == "symlink":
        return _error(lambda: os.symlink(step["target"], step["path"]))
    if do == "detach":                      # `count` processes in sessions of their own that outlive this one
        for _ in range(int(step["count"])):
            if os.fork() == 0:
                os.setsid()
                time.sleep(float(step.get("seconds", 600)))
                os._exit(0)
        return "detached"
    if do == "try_write":
        return {path: _try_write(path) for path in step["paths"]}
    if do == "exit":
        os._exit(int(step["status"]))
    return "unknown step"


def container(args):
    report = _report()
    report["steps"] = []
    _save(report)
    try:
        steps = json.loads(Path(STEPS).read_text())
    except (OSError, ValueError):
        steps = []
    for step in steps:
        report["steps"].append({"do": step.get("do"), "result": "started"})
        _save(report)
        report["steps"][-1]["result"] = _step(step)
        _save(report)
    return 0
