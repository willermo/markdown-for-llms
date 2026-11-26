"""The launcher (MFL051, tranche A1): the one process that holds the Docker socket.

A proxy that filters the Docker API by section and method does not read the body of `POST /containers/create`:
whoever reaches it can ask for a privileged container with a mount of the host. The launcher forwards nothing. It
exposes an API of its own with closed requests and builds the body of every child itself.

API (`engine/containers.py` is the client): the socket `launcher.sock`, mode 0600 in a 0700 directory; one JSON
request per connection, at most 64 KiB; the peer is verified with `SO_PEERCRED` (the configured UID). The answer is
one JSON line, `{"ok": true, ...}` or `{"ok": false, "code", "detail"}` with a closed code: `isolation_unavailable`,
`toolchain_missing`, `toolchain_mismatch`, `invalid_setting`. Operations:

    hello      what the launcher is: payload, build inputs and environment (the payload gate), the images of the
               roles, Docker's version and runtime, the maxima, the project
    run        validates the request, derives the mounts, creates, verifies, starts; answers name and id
    inspect    {state, exit, oom_killed, started, finished} of a child of the project; `absent` when it does not exist;
               with `reaped` (`lease`, `term`, `ended`, `stop`) when this launcher itself killed or removed it
    kill       a child of the project
    remove     a child of the project; answers `absent: true` only when Docker confirms that it is gone
    list       the children of the project of one attempt (the prefix of their names)
    logs       stdout of an exited `manifest` child, at most 64 MiB
    toolchain  the state of the toolchain image; with the name of an exited `manifest` child that printed its
               content manifest, the verification of that image (config and manifest, as `toolchain.verify`)

A `run` request is `{kind, name, image, argv, env, limits}` and nothing else: a field the launcher does not know,
`mounts` or `subpath` included, is a refusal. `image` is a ROLE, never a reference: `app` is the image of the
launcher itself, by ID, read from the inspect of its own container; `toolchain` is the local image tagged
`mdf-bench-toolchain:<content[:16]>`, usable by `tool` and `probe` only after its verification; `candidate` the
image tagged `mdf-bench-toolchain:candidate` (what `prepare` verifies before it tags it). The name is
`<prefix of the attempt>-<kind>-<n>`, and is also the name of the folder of the call.

Mounts are DERIVED from kind and name (`mounts_of`), never named by the request: the four sister directories of
the call in the work volume, each with its own target and mode, the socket directory of the attempt in the run
volume, and the read-only volumes of the kind. The finished body is checked again before the creation (`check_body`:
no root of a call, no ancestor, no other call, nothing below a role, no overlap, no target twice, no `..`, UID 0
only for the regeneration of the manifest, no writable mount for `manifest`), and every component of every source in
the work and run volumes with `lstat` (`check_folders`: real directories of the configured user, no link, distinct
inodes). Docker then resolves each subpath inside its volume: the second defence, which guards the boundary of the
volume, not that of the call.

Body, always (`body_of`): no network, read-only root, no capability, no new privileges, not privileged, the
supervisor as entrypoint and PID 1 (`Init` false), no TTY and no stdin, memory equal to memory-plus-swap, a process
limit, an explicit `/dev/shm`, ulimits, a tmpfs `/tmp`, volume mounts with `Subpath` and `NoCopy` only (never
`Binds`, devices, ports, GPUs), private IPC and cgroup namespaces, no restart, no auto-remove, no log (`local` with
a cap for `manifest`, whose output is read with `logs`), and the labels `mdf.project`, `mdf.kind`, `mdf.attempt`,
`mdf.max`, `mdf.lease`, `mdf.t0`.

After the creation and before the start the launcher reads the container back (`verify`) and compares every field
of `Config` and `HostConfig`: those it set, by equality; those the daemon generates, by their form; the defaults
of the daemon, by their written value; and EVERY OTHER field must be empty, zero, false or null. A difference
removes the container and answers `isolation_unavailable` with the name of the field.

From outside the children (`Watch`), every second: a child whose `ctl/lease` has not changed for its lease, or
past its term, is killed and removed. The supervisor of the child enforces the same lease and term from inside and
normally ends it first, with an exit status that says why (200, 201); the watch is what remains when the supervisor
does nothing, so it acts a few seconds later: `GRACE_SECONDS`, and for the term the share of it that slow ticks may
add (`TERM_DRIFT`), never more than `TERM_GRACE_MAX`. The launcher is the outer limit of the term: on a long term
it may well act before the supervisor does. A child it removes has no status left to read, and one it killed
reads as 137: so the launcher keeps WHY it did it and says so in `inspect` (`reaped`), for the caller to report
the term or the lease and never a memory limit. That memory is of the process: after a restart it is empty. Both
are measured with `time.monotonic()`, which does not count the time a machine is suspended; never with the civil
time, the mtime of a file or `State.StartedAt`. The term starts at the label `mdf.t0`, the monotonic instant of the
creation; the lease of a child found after a restart of the launcher starts at its first observation; a label in
the future (the clock restarted with the host) is an expired term. Children that exited are removed after ten
minutes.

Stop and creation are coordinated. Every `POST /containers/create`, the launcher's own manifest child included,
happens under one lock (`creating`), and the stop is checked again there, immediately before the request. On
SIGTERM the launcher sets the stop, waits for that lock (so a creation in flight is finished, and listed), then
kills and removes the children: after the stop began nothing is created, and what was being created is either
removed by its own creation (which checks the stop again before and after the start) or by the cleanup. The
cleanup has a budget (`TERM_SECONDS`): every call to Docker made for it gets what is left of the budget as its
timeout, never more, so a daemon that is slow or silent cannot make the stop last longer.

At its start (`main`) it verifies the EFFECTIVE access to the Docker endpoint it will use, with a real request
from its own identity (the user's UID and the GID of the socket): `version`, API 1.45 or newer. Without access it
exits with a closed diagnosis that names the endpoint, the error (ENOENT, EACCES, no answer, API too old) and the
effective UID and GIDs. There is no fallback and no capability. Then it finds its own container: exactly one, running,
with the compose labels of its project and service and an ID that starts with its hostname.

Who reaches the socket (the same user) can start only the admitted commands of the two images, without network or
capabilities, on the volumes of the project only, and stop only the children of the project. Who compromises the
launcher has the daemon: that is the declared boundary. Standard library only; its configuration comes from its
command line, never from the environment.
"""
import argparse
import errno
import hashlib
import http.client
import json
import os
from pathlib import Path
import posixpath
import re
import secrets
import signal
import socket
import stat
import struct
import sys
import threading
import time
from urllib.parse import quote, urlencode

from .engine import containers, supervisor, toolchain
from .engine.tools import TOOLS

API = "/v1.45"
API_MIN = (1, 45)
MIB = 1048576
RO, RW = True, False
CODES = ("isolation_unavailable", "toolchain_missing", "toolchain_mismatch", "invalid_setting")
REQUEST_KEYS = {"run": {"op", "kind", "name", "image", "argv", "env", "limits"}, "hello": {"op"},
                "inspect": {"op", "name"}, "kill": {"op", "name"}, "remove": {"op", "name"}, "list": {"op", "prefix"},
                "logs": {"op", "name"}, "toolchain": {"op", "name", "candidate"}}
LIMIT_KEYS = ("memory_mib", "pids", "fsize_mib", "nofile", "tmpfs_mib", "max_seconds", "lease_seconds")
MAXIMA = {"memory_mib": 1048576 + 512, "pids": 4096, "fsize_mib": 1048576, "nofile": 65536, "data_mib": 1048576,
          "tmpfs_mib": 1024, "max_seconds": 86400, "lease_seconds": 300}
MAX_CHILDREN = 8
SHM_BYTES = 64 * MIB
ARGV_LIMIT = 4096                       # arguments of a command
VALUE_LIMIT = 4096                      # bytes of one variable
CANDIDATE_TAG = toolchain.REPOSITORY + ":candidate"
EXITED_SECONDS = 600.0                  # a child that exited is removed after this
REMOVE_SECONDS = 20.0                   # how long `remove` waits for Docker to confirm
WATCH_SECONDS = 1.0
CONNECTIONS = 32                        # requests served at the same time; the others wait in the backlog
GRACE_SECONDS = 5.0                     # the watch acts this long after the supervisor of a child should have
TERM_DRIFT = 0.02                       # and, for the term, after the share of it that slow ticks may add,
TERM_GRACE_MAX = 30.0                   # which is never more than this: the launcher is the outer limit of the term
TERM_SECONDS = 25.0                     # the cleanup at SIGTERM, inside the 30 s of the service
DOCKER_SECONDS = 30.0                   # how long a call waits for the daemon, unless a budget leaves less
REAPED_SECONDS = EXITED_SECONDS         # how long `inspect` remembers why the launcher killed or removed a child

# The variables a kind admits (the image's own ENV comes with the image).
TOOL_VARIABLES = ("LC_ALL", "LANG", "HOME", "TMPDIR")
LIGHT_VARIABLES = ("PATH", "LANG", "LC_ALL", "HOME", "TMPDIR")
MARKER_VARIABLES = LIGHT_VARIABLES + ("XDG_CACHE_HOME", "TORCH_DEVICE", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE",
                                      "SURYA_INFERENCE_AUTOSTART", "TOKENIZERS_PARALLELISM", "FONT_PATH",
                                      "INDRA_OCR_MODEL_DIR", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
CPU_VARIABLES = ("PATH", "LANG", "HOME", "LD_LIBRARY_PATH")
MANIFEST_VARIABLES = ("LC_ALL", "LANG")

# kind: images it admits, {image role: target of the scratch}, working directory, variables, mounts outside the
# call as (volume role, subpath, target, read-only); the mounts of the call are `containers.CALL_ROLES`
KINDS = {
    "tool": (("toolchain", "candidate"), "/work", "/work", TOOL_VARIABLES, ()),
    "light": (("app",), "/scratch", "/scratch", LIGHT_VARIABLES, ()),
    "marker": (("app",), "/scratch", "/scratch", MARKER_VARIABLES,
               (("models", "marker", "/models", RO), ("run", "sock/{prefix}/relay", "/relay", RO))),
    "cpuserver": (("app",), None, "/tmp", CPU_VARIABLES,
                  (("cpuserver", "llama", containers.CPU_SERVER_TREE, RO), ("cpuserver", "gguf", "/models", RO),
                   ("run", "sock/{prefix}/llama", "/relay", RW))),
    "probe": (("app", "toolchain", "candidate"), {"app": "/scratch", "toolchain": "/work", "candidate": "/work"}, None,
              None, ()),
    "manifest": (("toolchain", "candidate"), None, "/", MANIFEST_VARIABLES, ()),
}
CALL_TARGETS = {"in": ("/in", RO), "ctl": ("/ctl", RO), "stat": ("/stat", RW)}
TARGET = re.compile(r"(/[A-Za-z0-9_][A-Za-z0-9._-]*)+")       # an absolute path of plain components
VOLUME_ROLES = ("work", "run", "models", "cpuserver")


class Refusal(Exception):
    """A request, or what Docker made of it, is refused with a closed code."""

    def __init__(self, code, detail):
        super().__init__("%s: %s" % (code, detail))
        self.code, self.detail = code, detail


def invalid(detail):
    return Refusal("invalid_setting", detail)


def unavailable(detail):
    return Refusal("isolation_unavailable", detail)


# --- Docker -----------------------------------------------------------------------------------------

class DockerUnreachable(Exception):
    """The daemon cannot be reached or does not answer: `kind` is ENOENT, EACCES, another errno name, or `no answer`."""

    def __init__(self, kind, detail=""):
        super().__init__(kind + (": " + detail if detail else ""))
        self.kind = kind


class _Deadline(socket.socket):
    """A socket whose every wait ends at one instant of the monotonic clock: before each operation its timeout is
    what is left until then, so the connection, the request, the head of the answer and its body, however many
    pieces the peer sends them in, never last longer together than the budget of the call (SOL-FIX-N1)."""
    until = None

    def _left(self):
        left = self.until - time.monotonic()
        if left <= 0:
            raise socket.timeout("the deadline of the call")
        self.settimeout(left)

    def connect(self, address):
        self._left()
        return super().connect(address)

    def sendall(self, *args):
        self._left()
        return super().sendall(*args)

    def send(self, *args):
        self._left()
        return super().send(*args)

    def recv(self, *args):
        self._left()
        return super().recv(*args)

    def recv_into(self, *args):
        self._left()
        return super().recv_into(*args)


class _UnixConnection(http.client.HTTPConnection):
    def __init__(self, path, timeout):
        super().__init__("localhost", timeout=timeout)
        self._path, self._until = path, time.monotonic() + timeout

    def connect(self):
        sock = _Deadline(socket.AF_UNIX, socket.SOCK_STREAM | socket.SOCK_CLOEXEC)
        sock.until = self._until
        try:
            sock.connect(self._path)
        except BaseException:
            sock.close()
            raise
        self.sock = sock


class Docker:
    """The Docker Engine API on a local socket, at the fixed version of `API`: `call` is the only door."""

    def __init__(self, path, timeout=DOCKER_SECONDS):
        self.path, self.timeout = str(path), timeout

    def call(self, method, path, query=None, body=None, timeout=None, versioned=True, limit=16 * MIB):
        """(status, bytes) of one request; `DockerUnreachable` when the daemon cannot be reached or does not answer.
        `timeout` is the budget of the WHOLE call on the monotonic clock: the connection, the request and the
        answer, its body included, whatever the number of pieces the daemon sends it in (`_Deadline`)."""
        url = (API if versioned else "") + path + ("?" + urlencode(query) if query else "")
        payload = None if body is None else json.dumps(body).encode("utf-8")
        headers = {"Host": "docker", "Accept": "application/json"}
        if payload is not None:
            headers["Content-Type"] = "application/json"
        conn = _UnixConnection(self.path, timeout or self.timeout)
        try:
            conn.request(method, url, body=payload, headers=headers)
            answer = conn.getresponse()
            data = answer.read(limit + 1)
            return answer.status, data
        except socket.timeout:
            raise DockerUnreachable("no answer") from None
        except OSError as exc:
            raise DockerUnreachable(errno.errorcode.get(exc.errno, "error") if exc.errno else "no answer",
                                    type(exc).__name__) from None
        except http.client.HTTPException as exc:
            raise DockerUnreachable("no answer", type(exc).__name__) from None
        finally:
            conn.close()


def _json(data):
    try:
        return json.loads(data)
    except ValueError:
        return None


def _message(data):
    parsed = _json(data)
    text = parsed.get("message", "") if isinstance(parsed, dict) else ""
    return str(text)[:300]


class Access:
    """The outcome of the access check (E-3): the version of the daemon, or the closed diagnosis."""

    def __init__(self, ok, error, detail, version=None):
        self.ok, self.error, self.detail, self.version = ok, error, detail, version


def _api(text):
    found = re.fullmatch(r"(\d{1,4})\.(\d{1,4})", text if isinstance(text, str) else "")
    return (int(found.group(1)), int(found.group(2))) if found else None


def check_access(docker, identity=None, timeout=10.0):
    """Effective access to the endpoint from the identity of this process: a connection and a real request,
    `GET /version`, with an API that includes 1.45. Nothing is deduced from the modes of the socket or from a GID
    read on the host."""
    uid, gids = identity or (os.geteuid(), sorted({os.getegid(), *os.getgroups()}))
    who = "effective uid %d, gids %s" % (uid, ",".join(map(str, gids)))
    where = "docker endpoint unix://" + docker.path

    def no(error, detail=""):
        return Access(False, error, "%s: %s%s; %s; no fallback" % (where, error, " (" + detail + ")" if detail else "", who))
    try:
        status, data = docker.call("GET", "/version", versioned=False, timeout=timeout)
    except DockerUnreachable as exc:
        return no(exc.kind)
    version = _json(data)
    if status != 200 or not isinstance(version, dict):
        return no("no answer", "HTTP %d" % status)
    newest, oldest = _api(version.get("ApiVersion")), _api(version.get("MinAPIVersion"))
    if newest is None or newest < API_MIN:
        return no("API too old", "daemon API %s, the launcher needs %d.%d" % (version.get("ApiVersion"), *API_MIN))
    if oldest is not None and oldest > API_MIN:
        return no("API too new", "daemon minimum API %s, the launcher speaks %d.%d" % (version.get("MinAPIVersion"), *API_MIN))
    return Access(True, None, "%s: API %s; %s" % (where, version.get("ApiVersion"), who), version)


# --- the request ------------------------------------------------------------------------------------

def check_request(request, maxima=MAXIMA):
    """The closed form of a `run` request; returns (kind, name, prefix, image role, argv, variables, limits)."""
    if set(request) - REQUEST_KEYS["run"]:
        raise invalid("unknown field: " + ", ".join(sorted(set(request) - REQUEST_KEYS["run"])))
    if set(request) != REQUEST_KEYS["run"]:
        raise invalid("missing field: " + ", ".join(sorted(REQUEST_KEYS["run"] - set(request))))
    kind, name, image = request["kind"], request["name"], request["image"]
    if kind not in KINDS:
        raise invalid("kind")
    try:
        prefix, named, _ = containers.split_name(name)
    except ValueError:
        raise invalid("name out of the schema") from None
    if named != kind:
        raise invalid("the name is of another kind")
    if not isinstance(image, str) or image not in KINDS[kind][0]:
        raise invalid("image role %r is not admitted for %s" % (image, kind))
    argv = request["argv"]
    if (not isinstance(argv, list) or not argv or len(argv) > ARGV_LIMIT
            or not all(isinstance(a, str) and "\0" not in a for a in argv)):
        raise invalid("argv")
    variables = request["env"]
    allowed = KINDS[kind][3] if kind != "probe" else (LIGHT_VARIABLES if image == "app" else TOOL_VARIABLES)
    if not isinstance(variables, dict):
        raise invalid("env")
    for key, value in variables.items():
        if key not in allowed:
            raise invalid("variable %r is not admitted for %s" % (key, kind))
        if not isinstance(value, str) or "\0" in value or "\n" in value or len(value.encode("utf-8")) > VALUE_LIMIT:
            raise invalid("value of " + key)
    limits = request["limits"]
    known = set(LIMIT_KEYS) | ({"data_mib"} if kind == "tool" else set())
    if not isinstance(limits, dict) or set(limits) - known or not set(LIMIT_KEYS) <= set(limits):
        raise invalid("limits: exactly %s%s" % (", ".join(LIMIT_KEYS), " and optionally data_mib" if kind == "tool" else ""))
    for key, value in limits.items():
        if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= maxima[key]:
            raise invalid("limit %s outside 1..%d" % (key, maxima[key]))
    return kind, name, prefix, image, argv, variables, limits


def check_command(kind, image, argv):
    """The command a kind admits (closed lists, never a free argv); returns the user of the child, `user` or `root`."""
    if kind == "tool":
        if argv[0] not in TOOLS:
            raise invalid("not a tool of the toolchain: " + argv[0][:40])
        if argv[0] == "pandoc":
            # the cap of its runtime and `--sandbox` first, as `tools.run` writes them, and nothing that turns them off
            if (len(argv) < 5 or argv[1] != "+RTS" or not re.fullmatch(r"-M[0-9]{1,9}m", argv[2]) or argv[3] != "-RTS"
                    or argv[4] != "--sandbox" or any(a in ("+RTS", "-RTS") or a.startswith("--sandbox") for a in argv[5:])):
                raise invalid("pandoc without `+RTS -M<cap>m -RTS --sandbox` first, or with another of them later")
        return "user"
    if kind in ("light", "marker"):
        modules = containers.LIGHT_MODULES if kind == "light" else containers.MARKER_MODULES
        head = [containers.PYTHON, "-I", "-B", "-c", containers.BOOT, containers.LIMITS_FILE]
        if argv[:6] != head or len(argv) < 7 or argv[6] not in modules:
            raise invalid("not a module of the %s child" % kind)
        return "user"
    if kind == "cpuserver":
        exe = argv[0]
        if not exe.startswith(containers.CPU_SERVER_TREE + "/") or posixpath.normpath(exe) != exe:
            raise invalid("the program of the CPU server is not under " + containers.CPU_SERVER_TREE)
        return "user"
    if kind == "probe":
        if argv != (containers.PROBE_PYTHON if image == "app" else containers.PROBE_SHELL):
            raise invalid("not the fixed probe")
        return "user"
    if argv == containers.MANIFEST_READ:
        return "user"
    if argv == containers.MANIFEST_REGENERATE:
        return "root"
    raise invalid("not one of the two commands of the manifest")


def mounts_of(kind, name, prefix, image, kinds=None):
    """The mounts of a call, derived from its kind and name: [(volume role, subpath, target, read-only)]."""
    _, scratch, _, _, extra = (kinds or KINDS)[kind]
    target = scratch.get(image) if isinstance(scratch, dict) else scratch
    mounts = []
    for role in containers.CALL_ROLES[kind]:
        where, readonly = (target, RW) if role == "scratch" else CALL_TARGETS[role]
        mounts.append(("work", "call/%s/%s" % (name, role), where, readonly))
    for volume, subpath, where, readonly in extra:
        mounts.append((volume, subpath.format(prefix=prefix), where, readonly))
    return mounts


def check_mounts(mounts, name, prefix):
    """The finished list against a wrong table: no source is the root of the call, an ancestor of it, another
    call, a level below a role or outside the socket directory of the attempt; no source inside another; no
    target twice; no `..`."""
    root = "call/" + name
    for i, (volume, subpath, target, readonly) in enumerate(mounts):
        parts = subpath.split("/") if isinstance(subpath, str) else [""]
        if volume not in VOLUME_ROLES or not subpath or ".." in parts or "." in parts or "" in parts:
            raise invalid("mount source: %s:%s" % (volume, subpath))
        if not isinstance(target, str) or not TARGET.fullmatch(target):
            raise invalid("mount target: %s" % (target,))
        if volume == "work":
            if subpath == root or root.startswith(subpath + "/"):
                raise invalid("the root of the call or an ancestor: " + subpath)
            if not subpath.startswith(root + "/") or len(parts) != 3 or parts[2] not in containers.CALL_ROLES[name_kind(name)]:
                raise invalid("outside the call: " + subpath)
            if parts[2] in ("in", "ctl") and readonly is not RO:
                raise invalid("writable: " + subpath)
        elif volume == "run":
            if parts[:2] != ["sock", prefix] or len(parts) != 3 or parts[2] not in containers.SOCKET_ROLES:
                raise invalid("outside the socket directory of the attempt: " + subpath)
            if parts[2] == "relay" and readonly is not RO:      # only the CPU server writes its own socket directory
                raise invalid("writable: " + subpath)
        elif readonly is not RO:
            raise invalid("writable: %s:%s" % (volume, subpath))
        for other_volume, other, _, _ in mounts[i + 1:]:
            if volume == other_volume and (subpath == other or subpath.startswith(other + "/") or other.startswith(subpath + "/")):
                raise invalid("overlapping sources: %s, %s" % (subpath, other))
    targets = [m[2] for m in mounts]
    if len(set(targets)) != len(targets):
        raise invalid("the same target twice")
    for a in targets:
        for b in targets:
            if a != b and b.startswith(a + "/"):
                raise invalid("a target inside another: %s, %s" % (a, b))


def name_kind(name):
    return containers.split_name(name)[1]


def check_folders(roots, mounts, uid):
    """`lstat` of every component of every source in the volumes the launcher sees (`roots`: {volume role: its
    path here}): real directories of the configured user, no link, and distinct inodes."""
    seen = []
    for volume, subpath, _, _ in mounts:
        if volume not in roots:
            continue
        path = Path(roots[volume])
        for part in subpath.split("/"):
            path = path / part
            try:
                info = os.lstat(path)
            except OSError as exc:
                raise invalid("%s:%s: %s" % (volume, subpath, errno.errorcode.get(exc.errno, "error"))) from None
            if not stat.S_ISDIR(info.st_mode) or info.st_uid != uid:
                raise invalid("not a directory of the user: %s:%s" % (volume, path.relative_to(roots[volume])))
        seen.append((info.st_dev, info.st_ino))
    if len(set(seen)) != len(seen):
        raise invalid("two sources on the same directory")


def body_of(config, kind, name, prefix, image, argv, variables, limits, user, mounts, t0):
    """The body of `POST /containers/create` of a child: everything is written here, nothing comes from the request
    but the validated command, variables and limits. `config` is the launcher's (`Config`)."""
    role = config.images[image]
    labels = {"mdf.project": config.project, "mdf.kind": kind, "mdf.attempt": prefix,
              "mdf.max": str(limits["max_seconds"]), "mdf.lease": str(limits["lease_seconds"]), "mdf.t0": repr(float(t0))}
    ulimits = [{"Name": "core", "Soft": 0, "Hard": 0},
               {"Name": "fsize", "Soft": limits["fsize_mib"] * MIB, "Hard": limits["fsize_mib"] * MIB},
               {"Name": "nofile", "Soft": limits["nofile"], "Hard": limits["nofile"]}]
    if "data_mib" in limits:
        ulimits.append({"Name": "data", "Soft": limits["data_mib"] * MIB, "Hard": limits["data_mib"] * MIB})
    has_stat = "stat" in containers.CALL_ROLES[kind]
    workdir = KINDS[kind][2] or next(t for v, s, t, r in mounts if s.endswith("/scratch"))
    return {
        "Image": role["id"],
        "Entrypoint": supervisor.entrypoint(limits["max_seconds"], limits["lease_seconds"], stat=has_stat),
        "Cmd": list(argv),
        "User": "0:0" if user == "root" else "%d:%d" % (config.uid, config.gid),
        "Env": ["%s=%s" % (k, v) for k, v in sorted(variables.items())],
        "WorkingDir": workdir,
        "Labels": labels,
        "Tty": False, "OpenStdin": False, "StdinOnce": False,
        "AttachStdin": False, "AttachStdout": False, "AttachStderr": False,
        "NetworkDisabled": True,
        "HostConfig": {
            "NetworkMode": "none", "ReadonlyRootfs": True, "CapDrop": ["ALL"], "SecurityOpt": ["no-new-privileges"],
            "Privileged": False, "Init": False,
            "Memory": limits["memory_mib"] * MIB, "MemorySwap": limits["memory_mib"] * MIB,
            "PidsLimit": limits["pids"], "ShmSize": SHM_BYTES, "Ulimits": ulimits,
            "Tmpfs": {"/tmp": "rw,nosuid,nodev,size=%dm" % limits["tmpfs_mib"]},
            "Mounts": [{"Type": "volume", "Source": config.volumes[volume], "Target": target, "ReadOnly": readonly,
                        "VolumeOptions": {"NoCopy": True, "Subpath": subpath}} for volume, subpath, target, readonly in mounts],
            "IpcMode": "private", "CgroupnsMode": "private", "RestartPolicy": {"Name": "no", "MaximumRetryCount": 0},
            "AutoRemove": False,
            "LogConfig": ({"Type": "local", "Config": {"max-size": "64m", "max-file": "1", "compress": "false"}}
                          if kind == "manifest" else {"Type": "none", "Config": {}}),
        },
    }


def check_body(body, kind, name, prefix, user, mounts, config):
    """The invariants of the finished body, whatever the tables say (a second defence, before anything is created)."""
    check_mounts(mounts, name, prefix)
    host = body["HostConfig"]
    root = body["User"] == "0:0" or body["User"].split(":")[0] in ("0", "root", "")
    if root and not (kind == "manifest" and body["Cmd"] == containers.MANIFEST_REGENERATE and user == "root"):
        raise invalid("UID 0 outside the regeneration of the manifest")
    if kind == "manifest" and any(not m["ReadOnly"] for m in host["Mounts"]):
        raise invalid("a writable volume for manifest")
    if kind == "cpuserver" and config.images["app"]["architecture"] != "amd64":
        raise unavailable("the local CPU server is amd64 only; the image of this launcher is %s"
                          % config.images["app"]["architecture"])
    if body["Entrypoint"][:2] != [supervisor.BASH, "-c"] or body["Entrypoint"][2] != supervisor.SUPERVISOR:
        raise invalid("entrypoint")
    wanted = {"NetworkMode": "none", "ReadonlyRootfs": True, "CapDrop": ["ALL"], "SecurityOpt": ["no-new-privileges"],
              "Privileged": False, "Init": False, "IpcMode": "private", "CgroupnsMode": "private", "AutoRemove": False}
    for key, value in wanted.items():
        if host.get(key) != value:
            raise invalid("body: " + key)
    if host["Memory"] != host["MemorySwap"] or host["Memory"] <= 0 or host["PidsLimit"] <= 0:
        raise invalid("body: memory or processes")
    for forbidden in ("Binds", "Devices", "DeviceRequests", "PortBindings", "CapAdd", "VolumesFrom", "Links"):
        if host.get(forbidden):
            raise invalid("body: " + forbidden)
    if any(m["Type"] != "volume" or m["Source"] not in config.volumes.values() for m in host["Mounts"]):
        raise invalid("body: a mount that is not a volume of the project")


# --- verification after the creation -----------------------------------------------------------------

def _empty(value):
    return value is None or value == "" or value == 0 or value is False or value == [] or value == {}


def _same(found, wanted):
    return found == wanted or (_empty(found) and _empty(wanted))


def _mount_key(mount):
    options = mount.get("VolumeOptions") or {}
    extra = {k: v for k, v in mount.items() if k not in ("Type", "Source", "Target", "ReadOnly", "VolumeOptions") and not _empty(v)}
    more = {k: v for k, v in options.items() if k not in ("NoCopy", "Subpath") and not _empty(v)}
    return (mount.get("Type"), mount.get("Source"), mount.get("Target"), bool(mount.get("ReadOnly")),
            bool(options.get("NoCopy")), options.get("Subpath"), json.dumps([extra, more], sort_keys=True))


def verify(found, body, image, daemon):
    """The fields of `found` (`GET /containers/{id}/json`, before the start) that are not what the launcher asked,
    as names like `HostConfig.Privileged`; empty when the container is the one of `body`.

    `image` is the inspect of the image (its own `Env` joins the request's), `daemon` what `hello` reports of
    Docker (its runtime and the AppArmor profile it gives). Four classes (H.2): set by the launcher, equal to what
    it asked; generated by the daemon, of the expected form; defaults of the daemon, with their written value;
    every other field of `Config` and `HostConfig` empty, zero, false or null."""
    bad = []
    config, host, wanted = found.get("Config") or {}, found.get("HostConfig") or {}, body["HostConfig"]
    ident = found.get("Id") or ""
    image_env = list((image.get("Config") or {}).get("Env") or [])
    names = {e.split("=", 1)[0] for e in body["Env"]}
    env = [e for e in image_env if e.split("=", 1)[0] not in names] + body["Env"]

    def check(where, key, ok):
        if not ok:
            bad.append(where + "." + key)

    top = {"Image": found.get("Image") == body["Image"],
           "Path": found.get("Path") == body["Entrypoint"][0],
           "Args": found.get("Args") == body["Entrypoint"][1:] + body["Cmd"],
           "Name": isinstance(found.get("Name"), str) and found["Name"].startswith("/"),
           "AppArmorProfile": found.get("AppArmorProfile") in ("", daemon.get("apparmor") or ""),
           "Id": bool(re.fullmatch(r"[0-9a-f]{64}", ident))}
    for key, ok in top.items():
        if not ok:
            bad.append(key)
    set_config = {
        "Image": body["Image"], "Entrypoint": body["Entrypoint"], "Cmd": body["Cmd"], "User": body["User"],
        "Tty": False, "OpenStdin": False, "StdinOnce": False, "AttachStdin": False, "AttachStdout": False,
        "AttachStderr": False, "WorkingDir": body["WorkingDir"], "Labels": body["Labels"], "Volumes": None,
        "NetworkDisabled": True, "Domainname": "", "ExposedPorts": None, "OnBuild": None, "Healthcheck": None,
        "MacAddress": "", "Shell": None, "StopSignal": None, "StopTimeout": None, "ArgsEscaped": None,
    }
    for key, value in set_config.items():
        check("Config", key, _same(config.get(key), value))
    check("Config", "Env", sorted(config.get("Env") or []) == sorted(env))
    check("Config", "Hostname", config.get("Hostname") == ident[:12] and bool(ident))
    for key in config:
        if key not in set_config and key not in ("Env", "Hostname") and not _empty(config[key]):
            bad.append("Config." + key)
    set_host = {
        "NetworkMode": "none", "ReadonlyRootfs": True, "CapAdd": None, "CapDrop": ["ALL"],
        "SecurityOpt": ["no-new-privileges"], "Privileged": False, "Init": False, "Memory": wanted["Memory"],
        "MemorySwap": wanted["MemorySwap"], "PidsLimit": wanted["PidsLimit"], "ShmSize": wanted["ShmSize"],
        "Tmpfs": wanted["Tmpfs"], "Binds": None, "Devices": None, "DeviceRequests": None, "DeviceCgroupRules": None,
        "PortBindings": None, "PublishAllPorts": False, "RestartPolicy": wanted["RestartPolicy"], "AutoRemove": False,
        "IpcMode": "private", "PidMode": "", "UTSMode": "", "UsernsMode": "", "CgroupnsMode": "private",
        "CgroupParent": "", "GroupAdd": None, "ExtraHosts": None, "Dns": None, "DnsOptions": None, "DnsSearch": None,
        "Links": None, "VolumesFrom": None, "OomKillDisable": None, "Sysctls": None, "LogConfig": wanted["LogConfig"],
        "ConsoleSize": [0, 0], "Isolation": "", "VolumeDriver": "", "ContainerIDFile": "", "Cgroup": "",
        "OomScoreAdj": 0, "StorageOpt": None, "Annotations": None,
    }
    for key, value in set_host.items():
        found_value = host.get(key)
        if key == "RestartPolicy":
            ok = (found_value or {}).get("Name") in ("no", "") and not (found_value or {}).get("MaximumRetryCount")
        elif key == "ConsoleSize":
            ok = found_value in ([0, 0], None)
        elif key == "LogConfig":
            ok = (found_value or {}).get("Type") == value["Type"] and _same((found_value or {}).get("Config"), value["Config"])
        else:
            ok = _same(found_value, value)
        check("HostConfig", key, ok)
    ulimits = sorted((u.get("Name"), u.get("Soft"), u.get("Hard")) for u in host.get("Ulimits") or [])
    check("HostConfig", "Ulimits", ulimits == sorted((u["Name"], u["Soft"], u["Hard"]) for u in wanted["Ulimits"]))
    check("HostConfig", "Mounts", sorted(map(_mount_key, host.get("Mounts") or []))
          == sorted(map(_mount_key, wanted["Mounts"])))
    check("HostConfig", "Runtime", host.get("Runtime") == daemon.get("runtime"))
    for key in ("MaskedPaths", "ReadonlyPaths"):
        paths = host.get(key)
        check("HostConfig", key, isinstance(paths, list) and bool(paths)
              and all(isinstance(p, str) and (p.startswith("/proc/") or p.startswith("/sys/")) for p in paths))
    known = set(set_host) | {"Ulimits", "Mounts", "Runtime", "MaskedPaths", "ReadonlyPaths"}
    for key in host:
        if key not in known and not _empty(host[key]):
            bad.append("HostConfig." + key)
    # the effective mounts: exactly the volumes asked, no implicit volume of the image
    effective = sorted((m.get("Type"), m.get("Name"), m.get("Destination"), bool(m.get("RW"))) for m in found.get("Mounts") or [])
    asked = sorted(("volume", m["Source"], m["Target"], not m["ReadOnly"]) for m in wanted["Mounts"])
    if effective != asked:
        bad.append("Mounts")
    return bad


# --- the launcher ------------------------------------------------------------------------------------

class Config:
    """What the launcher is told at its birth (its command line) and what it reads of itself and of the daemon."""

    def __init__(self, project, uid, gid, volumes, roots, images=None, daemon=None, maxima=None, max_children=MAX_CHILDREN,
                 instance=None):
        self.project, self.uid, self.gid = project, uid, gid
        self.instance = instance                # the ID of the container of this launcher (`hello`)
        self.volumes = dict(volumes)            # {volume role: the exact name of the volume of the project}
        self.roots = dict(roots)                # {volume role: where this process sees it}: work and run, read-only
        self.images = dict(images or {})        # {image role: {"id", "architecture", "inspect"}}
        self.daemon = dict(daemon or {})        # version, api, runtime, apparmor
        self.maxima, self.max_children = dict(maxima or MAXIMA), max_children


def demux(data, stream=1):
    """The bytes of one stream (1 stdout, 2 stderr) of a multiplexed log of Docker."""
    out, at = [], 0
    while at + 8 <= len(data):
        kind, size = data[at], struct.unpack(">I", data[at + 4:at + 8])[0]
        if kind == stream:
            out.append(data[at + 8:at + 8 + size])
        at += 8 + size
    return b"".join(out)


class Launcher:
    def __init__(self, docker, config, clock=time.monotonic, sleep=time.sleep, identity=None):
        self.docker, self.config, self.clock, self.sleep = docker, config, clock, sleep
        self.stopping = threading.Event()
        # one creation at a time, the launcher's own manifest child included: the count of the children holds, and
        # the stop waits here for the creation in flight (`_create`, `stop`). Taken after `verifying`, never before.
        self.creating = threading.Lock()
        self.verifying = threading.Lock()
        self.verified = set()                   # IDs of toolchain images whose content manifest was verified here
        self.internal = 0
        self.token = secrets.token_hex(4)       # of this process: the names of its own manifest children never collide
        self.reaped = {}                        # name: (why this launcher killed or removed the child, the instant)
        self.noting = threading.Lock()
        self._identity = identity

    # -- Docker ---------------------------------------------------------------------------------------
    def _left(self, until):
        """The timeout of a call to Docker made inside a budget that ends at the instant `until` of the clock (None:
        no budget): what is left of the budget, never more; `DockerUnreachable` when nothing is left."""
        if until is None:
            return None
        left = until - self.clock()
        if left <= 0:
            raise DockerUnreachable("no time left")
        return min(left, DOCKER_SECONDS)

    def _call(self, method, path, until=None, **kw):
        try:
            if until is not None:
                kw["timeout"] = self._left(until)
            return self.docker.call(method, path, **kw)
        except DockerUnreachable as exc:
            raise unavailable("docker unreachable: " + exc.kind) from None

    def _inspect(self, ref):
        status, data = self._call("GET", "/containers/%s/json" % quote(ref, safe=""))
        found = _json(data)
        if status == 404:
            return None
        if status != 200 or not isinstance(found, dict):
            raise unavailable("docker inspect: HTTP %d" % status)
        return found

    def _child(self, name):
        """The inspect of a child of this project, by name; None when no container has the name. A container
        that is not a child of this project is a refusal: the launcher touches nothing else."""
        try:
            containers.split_name(name)
        except ValueError:
            raise invalid("name out of the schema") from None
        found = self._inspect(name)
        if found is None:
            return None
        labels = (found.get("Config") or {}).get("Labels") or {}
        if found.get("Name") != "/" + name or labels.get("mdf.project") != self.config.project or labels.get("mdf.kind") not in KINDS:
            raise invalid("not a child of this project")
        return found

    def children(self, prefix=None, until=None):
        """The children of the project (of one attempt with `prefix`): [{name, id, state, labels}]."""
        labels = ["mdf.project=" + self.config.project] + ([] if prefix is None else ["mdf.attempt=" + prefix])
        status, data = self._call("GET", "/containers/json", until=until,
                                  query={"all": "1", "filters": json.dumps({"label": labels})})
        listed = _json(data)
        if status != 200 or not isinstance(listed, list):
            raise unavailable("docker list: HTTP %d" % status)
        out = []
        for item in listed:
            labels = item.get("Labels") or {}
            names = item.get("Names") or []
            if labels.get("mdf.project") != self.config.project or labels.get("mdf.kind") not in KINDS or not names:
                continue
            if prefix is not None and labels.get("mdf.attempt") != prefix:
                continue
            out.append({"name": names[0].lstrip("/"), "id": item.get("Id"), "state": item.get("State"), "labels": labels})
        return out

    def _force_remove(self, ref, until=None):
        try:
            self.docker.call("DELETE", "/containers/%s" % quote(ref, safe=""), query={"force": "1", "v": "1"},
                             timeout=self._left(until))
        except DockerUnreachable:
            pass

    def note(self, name, why):
        """Remember why this launcher kills or removes the child `name` (`containers.REAPED`), for `inspect`: the
        first reason given for it, kept while the launcher goes on acting on that child."""
        now = self.clock()
        with self.noting:
            # SOL-FIX-R1: the first reason noted for a child stays for as long as the launcher acts on it, that is
            # while its container exists. A removal that keeps failing brings the watch back on a child it already
            # killed for its term or its lease, ten minutes later: `ended`, or the stop of the launcher, said then
            # would make its 137 read as something else. Each act keeps the note fresh, whatever its age; only a
            # new child of the same name (`op_run`) starts again, and a note nobody refreshes is forgotten.
            first = self.reaped.get(name)
            for old in [n for n, (_, at) in self.reaped.items() if n != name and now - at > REAPED_SECONDS]:
                del self.reaped[old]
            self.reaped[name] = (first[0] if first is not None else why, now)

    def reap(self, ref, name=None, why=None, until=None):
        """Kill and remove a child (the watch and the stop of the launcher): best effort, repeated by the watch.
        With `name` and `why` the reason is noted BEFORE the kill, so no `inspect` ever shows the 137 of this
        kill without it. `until` is the end of the budget of the caller (`_left`)."""
        if name is not None and why is not None:
            self.note(name, why)
        try:
            self.docker.call("POST", "/containers/%s/kill" % quote(ref, safe=""), query={"signal": "KILL"},
                             timeout=self._left(until))
        except DockerUnreachable:
            return
        self._force_remove(ref, until)

    def _create(self, query, body):
        """`POST /containers/create` at its one serialized point: the caller holds `creating`, and the stop of the
        launcher is checked again here, immediately before the request. Once the stop began nothing is created."""
        if self.stopping.is_set():
            raise unavailable("the launcher is stopping")
        return self._call("POST", "/containers/create", query=query, body=body)

    def _start(self, ident, what):
        """Start a container created here, unless the stop of the launcher began meanwhile: checked before the
        request and again after it. Whoever calls removes the container when this raises."""
        if self.stopping.is_set():
            raise unavailable("the launcher is stopping")
        status, data = self._call("POST", "/containers/%s/start" % ident)
        if status not in (204, 304):
            raise unavailable("%s: HTTP %d %s" % (what, status, _message(data)))
        if self.stopping.is_set():
            raise unavailable("the launcher is stopping")

    def internal_prefix(self):
        """The prefix of the names of the manifest children the launchers of this project start for themselves."""
        return "mdf-launcher-%s-" % hashlib.sha256(self.config.project.encode()).hexdigest()[:8]

    def forget_internal(self):
        """At the start: remove the manifest children that a launcher of this project started for itself and that
        are not running (an instance that died in the middle of a verification leaves one, which nobody will read).
        Returns their names. One still running ends by its own supervisor within a minute and goes with the watch."""
        removed = []
        for child in self.children():
            if (child["name"].startswith(self.internal_prefix()) and child["labels"].get("mdf.kind") == "manifest"
                    and child["state"] in ("exited", "dead", "created")):
                self._force_remove(child["id"])
                removed.append(child["name"])
        return removed

    def _image_inspect(self, ref):
        status, data = self._call("GET", "/images/%s/json" % quote(ref, safe=":@"))
        found = _json(data)
        if status == 404:
            return None
        if status != 200 or not isinstance(found, dict):
            raise unavailable("docker image inspect: HTTP %d" % status)
        return found

    # -- images -----------------------------------------------------------------------------------------
    def image(self, role, kind):
        """{"id", "architecture", "inspect"} of an image role. `toolchain` is verified here before a `tool` or a
        `probe` may use it (config and content manifest, as `toolchain.verify`); a `manifest` child, which is how
        an image is verified, may use it as it is."""
        if role == "app":
            return self.config.images["app"]
        ref = toolchain.tag() if role == "toolchain" else CANDIDATE_TAG
        found = self._image_inspect(ref)
        if found is None:
            raise Refusal("toolchain_missing", "no image " + ref)
        ident = found.get("Id") or ""
        if not toolchain.IMAGE_ID.fullmatch(ident):
            raise Refusal("toolchain_mismatch", "image id")
        if toolchain.config_of(found) != toolchain.spec()["config"]:
            raise Refusal("toolchain_mismatch", "image config")
        entry = {"id": ident, "architecture": found.get("Architecture"), "inspect": found}
        if role == "toolchain" and kind != "manifest":
            self._verify_toolchain(entry)
        return entry

    def _verify_toolchain(self, entry):
        """The stored content manifest of the image against `toolchain.json`, read by a child of the image that
        the launcher starts itself: `cat` of the manifest as the user, with the body of every child and no
        mount at all (so no lease either: its supervisor ends it after `lease` seconds on its own)."""
        with self.verifying:
            if entry["id"] in self.verified:
                return
            self.internal += 1
            # unique for this process (`token`): a child left by an instance that died never takes the name
            name = "%s%s-manifest-%d" % (self.internal_prefix(), self.token, self.internal)
            limits = {"memory_mib": 512, "pids": 64, "fsize_mib": 64, "nofile": 256, "tmpfs_mib": 64, "max_seconds": 60,
                      "lease_seconds": 60}
            scoped = Config(self.config.project, self.config.uid, self.config.gid, self.config.volumes, self.config.roots,
                            {"toolchain": entry}, self.config.daemon)
            body = body_of(scoped, "manifest", name, "-", "toolchain", containers.MANIFEST_READ, {}, limits, "user", [],
                           self.clock())
            with self.creating:
                status, data = self._create({"name": name, "platform": "linux/amd64"}, body)
            created = _json(data)
            if status != 201 or not isinstance(created, dict) or not created.get("Id"):
                raise unavailable("docker create (toolchain manifest): HTTP %d %s" % (status, _message(data)))
            ident = created["Id"]
            try:
                bad = self._differences(ident, body, entry)
                if bad:
                    raise unavailable("the container Docker created differs from the request: " + ", ".join(bad[:8]))
                self._start(ident, "docker start (toolchain manifest)")
                until = self.clock() + 90
                while True:
                    state = ((self._inspect(ident) or {}).get("State") or {})
                    if state.get("Status") in ("exited", "dead"):
                        break
                    if self.stopping.is_set():
                        raise unavailable("the launcher is stopping")
                    if self.clock() > until:
                        raise Refusal("toolchain_mismatch", "content manifest: timeout")
                    self.sleep(0.1)
                if state.get("ExitCode") != 0:
                    raise Refusal("toolchain_mismatch", "content manifest: exit %s" % state.get("ExitCode"))
                status, data = self._call("GET", "/containers/%s/logs" % ident, query={"stdout": "1", "stderr": "0"},
                                          limit=containers.MANIFEST_LIMIT + (1 << 20), timeout=120)
                if status != 200:
                    raise unavailable("docker logs: HTTP %d" % status)
                if hashlib.sha256(demux(data)).hexdigest() != toolchain.identity():
                    raise Refusal("toolchain_mismatch", "content manifest")
            finally:
                self._force_remove(ident)
            self.verified.add(entry["id"])

    # -- operations -------------------------------------------------------------------------------------
    def identity(self):
        if self._identity is None:
            self._identity = containers.identity()
        return self._identity

    def op_hello(self, request):
        images = {role: {"id": entry["id"], "architecture": entry["architecture"]} for role, entry in self.config.images.items()}
        # `instance`: the ID of the container of this launcher. A client that sees another one knows the launcher
        # was recreated (another image, perhaps), and repeats what it checked of the first (the probe of the worker)
        return dict(self.identity(), images=images, docker=self.config.daemon, project=self.config.project,
                    uid=self.config.uid, max=dict(self.config.maxima, children=self.config.max_children),
                    instance=self.config.instance)

    def op_run(self, request):
        if self.stopping.is_set():
            raise unavailable("the launcher is stopping")
        kind, name, prefix, image, argv, variables, limits = check_request(request, self.config.maxima)
        user = check_command(kind, image, argv)
        mounts = mounts_of(kind, name, prefix, image)
        check_mounts(mounts, name, prefix)
        missing = sorted({m[0] for m in mounts} - set(self.config.volumes))
        if missing:
            raise unavailable("no volume for the role " + ", ".join(missing))
        entry = self.image(image, kind)
        check_folders(self.config.roots, mounts, self.config.uid)
        with self.creating:
            alive = [c for c in self.children() if c["state"] not in ("exited", "dead")]
            if len(alive) >= self.config.max_children:
                raise unavailable("%d children already run, at most %d" % (len(alive), self.config.max_children))
            scoped = Config(self.config.project, self.config.uid, self.config.gid, self.config.volumes, self.config.roots,
                            dict(self.config.images, **{image: entry}), self.config.daemon)
            body = body_of(scoped, kind, name, prefix, image, argv, variables, limits, user, mounts, self.clock())
            check_body(body, kind, name, prefix, user, mounts, scoped)
            query = {"name": name}
            if image != "app":
                query["platform"] = "linux/amd64"       # the toolchain image is amd64
            status, data = self._create(query, body)
            created = _json(data)
            if status == 409:
                raise invalid("a container named %s exists" % name)
            if status != 201 or not isinstance(created, dict) or not created.get("Id"):
                raise unavailable("docker create: HTTP %d %s" % (status, _message(data)))
            ident = created["Id"]
            with self.noting:                   # a new child of this name: what was noted of an old one is not its
                self.reaped.pop(name, None)
            try:
                bad = self._differences(ident, body, entry)
                if bad:
                    raise unavailable("the container Docker created differs from the request: " + ", ".join(bad[:8]))
                self._start(ident, "docker start")
            except BaseException:               # whatever went wrong, nothing that was created is left to start later
                self._force_remove(ident)
                raise
        return {"name": name, "id": ident}

    def _differences(self, ident, body, entry):
        """`verify` on what Docker says of the container just created; a container that cannot be read, or an
        answer that is not an inspect, is a difference like any other."""
        found = self._inspect(ident)
        if found is None:
            return ["absent"]
        try:
            return verify(found, body, entry["inspect"], self.config.daemon)
        except (AttributeError, KeyError, TypeError, ValueError, IndexError):
            return ["unreadable"]

    def op_inspect(self, request):
        found = self._child(request.get("name"))
        with self.noting:
            noted = self.reaped.get(request["name"])
        why = {"reaped": noted[0]} if noted else {}     # this launcher killed or removed it: its 137 is not a memory limit
        if found is None:
            return dict({"state": "absent"}, **why)
        state = found.get("State") or {}
        return dict({"state": state.get("Status"), "exit": state.get("ExitCode"), "oom_killed": bool(state.get("OOMKilled")),
                     "started": state.get("StartedAt"), "finished": state.get("FinishedAt")}, **why)

    def op_kill(self, request):
        found = self._child(request.get("name"))
        if found is None:
            return {"state": "absent"}
        status, data = self._call("POST", "/containers/%s/kill" % found["Id"], query={"signal": "KILL"})
        if status not in (204, 404, 409):               # 409: it is not running
            raise unavailable("docker kill: HTTP %d %s" % (status, _message(data)))
        return {"state": "killed"}

    def op_remove(self, request):
        found = self._child(request.get("name"))
        if found is None:
            return {"absent": True}
        status, data = self._call("DELETE", "/containers/%s" % found["Id"], query={"force": "1", "v": "1"})
        if status not in (204, 404, 409):               # 409: a removal is already in progress
            raise unavailable("docker remove: HTTP %d %s" % (status, _message(data)))
        until = self.clock() + REMOVE_SECONDS
        while self._inspect(found["Id"]) is not None:
            if self.clock() > until:
                raise unavailable("removal of %s not confirmed" % request["name"])
            self.sleep(0.05)
        return {"absent": True}

    def op_list(self, request):
        prefix = request.get("prefix")
        if not isinstance(prefix, str) or not containers.PREFIX.fullmatch(prefix):
            raise invalid("prefix of an attempt")
        return {"children": [{"name": c["name"], "state": c["state"]} for c in self.children(prefix)]}

    def op_logs(self, request):
        found = self._child(request.get("name"))
        if found is None:
            raise invalid("no such child")
        if ((found.get("Config") or {}).get("Labels") or {}).get("mdf.kind") != "manifest":
            raise invalid("logs are of a manifest child only")
        if ((found.get("State") or {}).get("Status")) not in ("exited", "dead"):
            raise invalid("the child has not exited")
        status, data = self._call("GET", "/containers/%s/logs" % found["Id"], query={"stdout": "1", "stderr": "0"},
                                  limit=containers.MANIFEST_LIMIT + (1 << 20), timeout=120)
        if status != 200:
            raise unavailable("docker logs: HTTP %d" % status)
        out = demux(data)
        if len(out) > containers.MANIFEST_LIMIT:
            raise invalid("stdout over %d bytes" % containers.MANIFEST_LIMIT)
        return {"bytes": len(out)}, out

    def op_toolchain(self, request):
        candidate = request.get("candidate", False)
        if not isinstance(candidate, bool) or request.get("name") is not None:
            raise invalid("toolchain: only `candidate`, true or false")
        entry = self.image("candidate" if candidate else "toolchain", "manifest" if candidate else "tool")
        spec = toolchain.spec()
        return {"image": entry["id"], "architecture": entry["architecture"], "verified": entry["id"] in self.verified,
                "content_sha256": spec["content_sha256"], "versions": dict(spec["versions"]),
                "reference_image": entry["id"] == spec["image_id"], "docker": self.config.daemon.get("version")}

    def handle(self, request):
        """(answer, payload) of one parsed request; never raises."""
        try:
            if not isinstance(request, dict) or request.get("op") not in REQUEST_KEYS:
                raise invalid("operation")
            op = request["op"]
            if op != "run" and set(request) - REQUEST_KEYS[op]:
                raise invalid("unknown field: " + ", ".join(sorted(set(request) - REQUEST_KEYS[op])))
            result = getattr(self, "op_" + op)(request)
            answer, payload = result if isinstance(result, tuple) else (result, None)
            return dict(answer, ok=True), payload
        except Refusal as exc:
            return {"ok": False, "code": exc.code, "detail": exc.detail}, None
        except Exception as exc:        # a defect of the launcher is never a crash of the service
            return {"ok": False, "code": "isolation_unavailable", "detail": "launcher error: " + type(exc).__name__}, None

    def answer(self, data, peer_uid):
        """(answer, payload) for the bytes of one connection from a peer with `peer_uid`; never raises."""
        if peer_uid != self.config.uid:
            return {"ok": False, "code": "invalid_setting", "detail": "peer of another user"}, None
        if len(data) > containers.REQUEST_LIMIT:
            return {"ok": False, "code": "invalid_setting", "detail": "request over %d bytes" % containers.REQUEST_LIMIT}, None
        try:
            request = json.loads(data.decode("utf-8"))
        except (ValueError, UnicodeDecodeError, RecursionError):     # the last: JSON nested too deep, within the 64 KiB
            return {"ok": False, "code": "invalid_setting", "detail": "request is not JSON"}, None
        return self.handle(request)

    def stop(self, seconds=TERM_SECONDS):
        """SIGTERM: no new child, then the children killed and removed, within `seconds`; True when Docker lists
        none any more.

        The stop is set first; then the lock of the creations is taken, so a creation in flight has finished and
        is listed, and none starts while the children are removed (`_create` refuses under the same lock). Every
        call to Docker gets what is left of `seconds` as its timeout, never more: with a daemon that is slow or
        silent the stop gives up at the end of its budget and says so (False); the children stay under their
        supervisors."""
        self.stopping.set()
        until = self.clock() + seconds
        if not self.creating.acquire(timeout=max(0.0, seconds)):
            return False
        try:
            while True:
                try:
                    left = self.children(until=until)
                except Refusal:                 # Docker unreachable, or no time left
                    return False
                if not left:
                    return True
                for child in left:
                    self.reap(child["id"], child["name"], "stop", until)
                pause = min(0.2, until - self.clock())
                if pause <= 0:
                    return False
                self.sleep(pause)
        finally:
            self.creating.release()


class Watch:
    """The launcher's own watch of the children, from outside them (H.6): lease and term on the monotonic clock."""

    def __init__(self, launcher, clock=time.monotonic):
        self.launcher, self.clock = launcher, clock
        self.leases = {}                # name: (the counter last read, the monotonic instant it changed)
        self.ended = {}                 # name: the monotonic instant it was first seen not running

    def lease(self, name):
        root = self.launcher.config.roots.get("work")
        if root is None:
            return None
        return containers.read_regular(root, "call/%s/ctl/lease" % name, 64)

    def tick(self):
        """One pass; returns {name: why} of the children it killed or removed."""
        now, done = self.clock(), {}
        try:
            children = self.launcher.children()
        except Refusal:
            return done                 # Docker unreachable: the children stay under their supervisors
        names = set()
        for child in children:
            name, labels = child["name"], child["labels"]
            names.add(name)
            try:
                t0, term, lease = float(labels["mdf.t0"]), float(labels["mdf.max"]), float(labels["mdf.lease"])
            except (KeyError, ValueError):
                t0, term, lease = now + 1, 0.0, 0.0     # a child without its labels is past its term
            # the supervisor of the child ends it at its term; the watch waits a little longer, and never long
            grace = min(max(GRACE_SECONDS, term * TERM_DRIFT), TERM_GRACE_MAX)
            past = t0 > now or now - t0 > term + grace      # a label in the future: the clock restarted with the host
            if child["state"] in ("exited", "dead", "created"):
                first = self.ended.setdefault(name, now)
                if now - first > EXITED_SECONDS or (past and (t0 > now or now - t0 > term + EXITED_SECONDS)):
                    self.launcher.reap(child["id"], name, "ended")
                    done[name] = "ended"
                continue
            counter = self.lease(name)
            last = self.leases.get(name)
            if last is None or last[0] != counter:      # the first observation (also after a restart), or a renewal
                self.leases[name] = last = (counter, now)
            if past:
                done[name] = "term"
            elif now - last[1] > lease + GRACE_SECONDS:
                done[name] = "lease"
            if name in done:
                self.launcher.reap(child["id"], name, done[name])
        for table in (self.leases, self.ended):
            for name in [n for n in table if n not in names]:
                del table[name]
        return done


# --- the service --------------------------------------------------------------------------------------

EXIT_USAGE, EXIT_ACCESS, EXIT_SELF, EXIT_HOST = 2, 3, 4, 5


class Startup(Exception):
    def __init__(self, status, detail):
        super().__init__(detail)
        self.status, self.detail = status, detail


def find_self(docker, project, service, hostname):
    """The launcher's own container (H.6): exactly one container with the compose labels of project and service,
    running, whose ID starts with this hostname. Returns its inspect."""
    labels = ["com.docker.compose.project=" + project, "com.docker.compose.service=" + service]
    try:
        status, data = docker.call("GET", "/containers/json", query={"all": "1", "filters": json.dumps({"label": labels})})
    except DockerUnreachable as exc:
        raise Startup(EXIT_ACCESS, "docker unreachable: " + exc.kind) from None
    listed = _json(data)
    if status != 200 or not isinstance(listed, list):
        raise Startup(EXIT_SELF, "docker list: HTTP %d" % status)
    mine = [c for c in listed if (c.get("Labels") or {}).get("com.docker.compose.project") == project
            and (c.get("Labels") or {}).get("com.docker.compose.service") == service]
    if len(mine) != 1:
        raise Startup(EXIT_SELF, "%d containers of %s/%s, exactly one expected" % (len(mine), project, service))
    ident = mine[0].get("Id") or ""
    if mine[0].get("State") != "running" or not hostname or len(hostname) < 12 or not ident.startswith(hostname):
        raise Startup(EXIT_SELF, "the container of %s/%s is not this process (state %s)" % (project, service, mine[0].get("State")))
    try:
        status, data = docker.call("GET", "/containers/%s/json" % ident)
    except DockerUnreachable as exc:
        raise Startup(EXIT_ACCESS, "docker unreachable: " + exc.kind) from None
    found = _json(data)
    if status != 200 or not isinstance(found, dict) or not toolchain.IMAGE_ID.fullmatch(found.get("Image") or ""):
        raise Startup(EXIT_SELF, "own container unreadable")
    return found


def daemon_of(docker, version):
    """What the launcher needs of the daemon, with the requirements of the host: cgroup v2, not rootless."""
    try:
        status, data = docker.call("GET", "/info")
    except DockerUnreachable as exc:
        raise Startup(EXIT_ACCESS, "docker unreachable: " + exc.kind) from None
    info = _json(data)
    if status != 200 or not isinstance(info, dict):
        raise Startup(EXIT_HOST, "docker info: HTTP %d" % status)
    options = [str(o) for o in info.get("SecurityOptions") or []]
    if str(info.get("CgroupVersion")) != "2":
        raise Startup(EXIT_HOST, "cgroup v%s: the children need cgroup v2" % info.get("CgroupVersion"))
    if any("rootless" in o for o in options):
        raise Startup(EXIT_HOST, "rootless Docker is not supported")
    return {"version": version.get("Version"), "api": version.get("ApiVersion"), "runtime": info.get("DefaultRuntime"),
            "apparmor": "docker-default" if any(o.startswith("name=apparmor") for o in options) else "",
            "architecture": info.get("Architecture"), "cgroup_driver": info.get("CgroupDriver")}


def bind(path, uid):
    """The listening socket, mode 0600, in a directory of the user with mode 0700 that is not a link."""
    directory = os.path.dirname(path)
    try:
        info = os.lstat(directory)
    except OSError:
        info = None
    if info is None or not stat.S_ISDIR(info.st_mode) or info.st_uid != uid or stat.S_IMODE(info.st_mode) != 0o700:
        raise Startup(EXIT_HOST, "the directory of the socket is not a 0700 directory of uid %d" % uid)
    try:
        if stat.S_ISSOCK(os.lstat(path).st_mode):
            os.unlink(path)                 # left by a launcher that died: one launcher per project (`find_self`)
    except FileNotFoundError:
        pass
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM | socket.SOCK_CLOEXEC)
    old = os.umask(0o177)
    try:
        server.bind(path)
    finally:
        os.umask(old)
    os.chmod(path, 0o600)
    server.listen(32)
    return server


def serve_connection(launcher, conn, done=None):
    try:
        conn.settimeout(15)
        _, peer_uid, _ = struct.unpack("3i", conn.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, struct.calcsize("3i")))
        data = b""
        while len(data) <= containers.REQUEST_LIMIT:
            chunk = conn.recv(65536)
            if not chunk:
                break
            data += chunk
        answer, payload = launcher.answer(data, peer_uid)
        conn.settimeout(120)
        conn.sendall(json.dumps(answer, separators=(",", ":")).encode("utf-8") + b"\n")
        if payload:
            conn.sendall(payload)
    except OSError:
        pass
    finally:
        conn.close()
        if done is not None:
            done.release()


def serve(args, out=sys.stderr):
    uid, gid = os.geteuid(), os.getegid()
    if uid == 0:
        raise Startup(EXIT_HOST, "the launcher never runs as UID 0")
    docker = Docker(args.docker)
    access = check_access(docker)
    if not access.ok:
        raise Startup(EXIT_ACCESS, access.detail)
    daemon = daemon_of(docker, access.version)
    own = find_self(docker, args.project, args.service, socket.gethostname())
    launcher = Launcher(docker, Config(args.project, uid, gid, {}, {}))
    image = launcher._image_inspect(own["Image"])
    if image is None:
        raise Startup(EXIT_SELF, "own image unreadable")
    volumes = dict(v.split("=", 1) for v in args.volume if "=" in v)
    if len(volumes) != len(args.volume) or set(volumes) - set(VOLUME_ROLES) or not {"work", "run"} <= set(volumes):
        raise Startup(EXIT_USAGE, "--volume ROLE=NAME for work and run, optionally models and cpuserver")
    maxima = dict(MAXIMA)
    for item in args.max:
        key, _, value = item.partition("=")
        if key not in MAXIMA or not value.isdigit() or int(value) < 1:
            raise Startup(EXIT_USAGE, "--max KEY=N with KEY of " + ", ".join(MAXIMA))
        maxima[key] = int(value)
    launcher.config = Config(args.project, uid, gid, volumes, {"work": args.work, "run": args.run},
                             {"app": {"id": own["Image"], "architecture": image.get("Architecture"), "inspect": image}},
                             daemon, maxima, args.max_children, instance=own.get("Id"))
    launcher.identity()
    try:
        launcher.forget_internal()              # what an instance that died in the middle of a verification left
    except Refusal:
        pass                                    # best effort: the names of this instance are its own anyway
    server = bind(args.socket, uid)
    watch = Watch(launcher)
    print("launcher ready: project %s, image %s (%s), docker %s API %s, uid %d" % (
        args.project, own["Image"][:19], image.get("Architecture"), daemon["version"], daemon["api"], uid), file=out, flush=True)

    def on_term(signum, frame):
        launcher.stopping.set()
        try:
            server.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
    signal.signal(signal.SIGTERM, on_term)
    signal.signal(signal.SIGINT, on_term)

    def pass_once():
        try:
            watch.tick()
        except Exception:               # the watch never dies of one bad pass
            pass

    def watching():
        while not launcher.stopping.wait(WATCH_SECONDS):
            pass_once()
    pass_once()                         # H.6: the cleanup of the start, before the first request is served (A2-e)
    threading.Thread(target=watching, name="watch", daemon=True).start()
    slots = threading.BoundedSemaphore(CONNECTIONS)
    while not launcher.stopping.is_set():
        try:
            conn, _ = server.accept()
        except OSError:
            if launcher.stopping.is_set():
                break
            continue
        slots.acquire()
        threading.Thread(target=serve_connection, args=(launcher, conn, slots), daemon=True).start()
    server.close()
    try:
        os.unlink(args.socket)
    except OSError:
        pass
    clean = launcher.stop()
    print("launcher stopped: children %s" % ("removed" if clean else "not all removed"), file=out, flush=True)
    return 0


def main(argv=None):
    """`python -m markdown_for_llms.launcher serve ...` (the service) or `check --docker SOCKET` (the access check alone)."""
    parser = argparse.ArgumentParser(prog="markdown_for_llms.launcher", description="The launcher of the children of the engine.")
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("check", help="verify the effective access to the Docker endpoint and exit")
    check.add_argument("--docker", required=True, help="path of the Docker socket, as this process sees it")
    check.add_argument("--timeout", type=float, default=10.0, help="seconds to wait for the answer of the daemon")
    run = sub.add_parser("serve", help="serve the API on a socket")
    run.add_argument("--docker", required=True, help="path of the Docker socket, as this process sees it")
    run.add_argument("--socket", required=True, help="path of the socket of the API")
    run.add_argument("--project", required=True, help="the compose project: the children carry it as a label")
    run.add_argument("--service", required=True, help="the compose service of this launcher")
    run.add_argument("--work", required=True, help="where this process sees the work volume (read-only)")
    run.add_argument("--run", required=True, help="where this process sees the run volume (read-only)")
    run.add_argument("--volume", action="append", default=[], metavar="ROLE=NAME",
                     help="the exact name of the volume of a role: work, run, models, cpuserver")
    run.add_argument("--max", action="append", default=[], metavar="KEY=N", help="a maximum of the limits of a request")
    run.add_argument("--max-children", type=int, default=MAX_CHILDREN, help="children at the same time")
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return EXIT_USAGE if exc.code else 0
    try:
        if args.command == "check":
            access = check_access(Docker(args.docker), timeout=args.timeout)
            print(access.detail, file=sys.stdout if access.ok else sys.stderr, flush=True)
            return 0 if access.ok else EXIT_ACCESS
        return serve(args)
    except Startup as exc:
        print("launcher: " + exc.detail, file=sys.stderr, flush=True)
        return exc.status


if __name__ == "__main__":
    sys.exit(main())
