"""Support of the unit tests of the launcher (MFL051, tranche A1): a Docker that is not there.

`FakeDocker` answers `call(method, path, ...)` as the Engine API answers the launcher, from a dictionary of
containers: the launcher under test is the real one, only its door to the daemon is replaced, so these tests run
in the suite without a socket and without Docker. What a container looks like when it is read back is either
synthesized from its body as Docker 29 does (`synth_inspect`) or a recorded inspect of a real container
(`tests/fixtures/launcher/`, see its `README.txt`).
"""
import copy
import hashlib
import json
from pathlib import Path
import struct

from markdown_for_llms import launcher
from markdown_for_llms.engine import containers, supervisor, toolchain

ROOT = Path(__file__).resolve().parents[3]
FIXTURES = ROOT / "tests" / "fixtures" / "launcher"
PLACEHOLDER = "<supervisor>"                     # stands for `supervisor.SUPERVISOR` in the recorded files
PLACEHOLDERS = {PLACEHOLDER: supervisor.SUPERVISOR, "<shell-probe>": containers.SHELL_PROBE}
PROJECT, UID, GID = "mdf-unit", 1000, 1000
APP_ID = "sha256:" + "a" * 64
TOOLCHAIN_ENV = ["PATH=/opt/djvulibre/bin:/opt/toolchain/.pixi/envs/default/bin:/usr/local/bin:/usr/bin:/bin",
                 "LC_ALL=C.UTF-8", "LANG=C.UTF-8", "LD_LIBRARY_PATH=/opt/djvulibre/lib"]
APP_ENV = ["PATH=/opt/mdf/venv/bin:/usr/local/bin:/usr/bin:/bin", "LANG=C.UTF-8", "LC_ALL=C.UTF-8"]
VOLUMES = {"work": PROJECT + "_mdf-work", "run": PROJECT + "_mdf-run", "models": PROJECT + "_mdf-models",
           "cpuserver": PROJECT + "_mdf-cpu-server"}
DAEMON = {"version": "29.10.0", "api": "1.57", "runtime": "runc", "apparmor": "docker-default", "architecture": "x86_64",
          "cgroup_driver": "systemd"}
LIMITS = {"memory_mib": 256, "pids": 64, "fsize_mib": 64, "nofile": 256, "tmpfs_mib": 16, "max_seconds": 120,
          "lease_seconds": 30}
ATTEMPT = "mdf-eng-0123456789abcdef"
VERSION = {"Version": "29.10.0", "ApiVersion": "1.57", "MinAPIVersion": "1.40", "Arch": "amd64", "Os": "linux"}
INFO = {"CgroupVersion": "2", "CgroupDriver": "systemd", "DefaultRuntime": "runc", "Architecture": "x86_64",
        "SecurityOptions": ["name=apparmor", "name=seccomp,profile=builtin", "name=cgroupns"]}
MASKED = ["/proc/acpi", "/proc/kcore", "/proc/keys", "/sys/firmware", "/sys/devices/virtual/powercap"]
READONLY = ["/proc/bus", "/proc/fs", "/proc/irq", "/proc/sys", "/proc/sysrq-trigger"]


def image_inspect(ident, env, architecture="amd64"):
    return {"Id": ident, "Architecture": architecture, "Os": "linux", "Config": {"Env": list(env), "Entrypoint": None, "User": ""}}


def toolchain_id():
    return toolchain.spec()["image_id"]


def images(architecture="amd64"):
    return {APP_ID: image_inspect(APP_ID, APP_ENV, architecture),
            toolchain.tag(): image_inspect(toolchain_id(), TOOLCHAIN_ENV)}


def config(architecture="amd64", **over):
    entry = {"id": APP_ID, "architecture": architecture, "inspect": image_inspect(APP_ID, APP_ENV, architecture)}
    values = dict(project=PROJECT, uid=UID, gid=GID, volumes=VOLUMES, roots={}, images={"app": entry}, daemon=DAEMON)
    values.update(over)
    return launcher.Config(**values)


def synth_inspect(body, ident, name, image_env):
    """The inspect of a container just created from `body`, as Docker 29.10 answers it (the fields seen in the
    recorded inspects; nothing a daemon would not write)."""
    host = body.get("HostConfig", {})
    names = {e.split("=", 1)[0] for e in body.get("Env") or []}
    env = [e for e in image_env if e.split("=", 1)[0] not in names] + list(body.get("Env") or [])
    config_part = {
        "AttachStderr": False, "AttachStdin": False, "AttachStdout": False, "Cmd": body.get("Cmd"), "Domainname": "",
        "Entrypoint": body.get("Entrypoint"), "Env": env, "Hostname": ident[:12], "Image": body.get("Image"),
        "Labels": body.get("Labels") or {}, "NetworkDisabled": bool(body.get("NetworkDisabled")), "OpenStdin": False,
        "StdinOnce": False, "Tty": False, "User": body.get("User", ""), "Volumes": None, "WorkingDir": body.get("WorkingDir", ""),
    }
    host_part = {key: None for key in (
        "Binds", "BlkioDeviceReadBps", "BlkioDeviceReadIOps", "BlkioDeviceWriteBps", "BlkioDeviceWriteIOps",
        "BlkioWeightDevice", "CapAdd", "DeviceCgroupRules", "DeviceRequests", "Dns", "DnsOptions", "DnsSearch",
        "ExtraHosts", "GroupAdd", "Links", "MemorySwappiness", "OomKillDisable", "VolumesFrom")}
    host_part.update({key: 0 for key in ("BlkioWeight", "CpuCount", "CpuPercent", "CpuPeriod", "CpuQuota", "CpuRealtimePeriod",
                                         "CpuRealtimeRuntime", "CpuShares", "IOMaximumBandwidth", "IOMaximumIOps",
                                         "MemoryReservation", "NanoCpus", "OomScoreAdj")})
    host_part.update({key: "" for key in ("Cgroup", "CgroupParent", "ContainerIDFile", "CpusetCpus", "CpusetMems", "Isolation",
                                          "PidMode", "UTSMode", "UsernsMode", "VolumeDriver")})
    host_part.update({
        "AutoRemove": False, "CapDrop": host.get("CapDrop"), "CgroupnsMode": host.get("CgroupnsMode", ""),
        "ConsoleSize": [0, 0], "Devices": [], "Init": host.get("Init"), "IpcMode": host.get("IpcMode", ""),
        "LogConfig": host.get("LogConfig"), "MaskedPaths": list(MASKED), "Memory": host.get("Memory", 0),
        "MemorySwap": host.get("MemorySwap", 0), "Mounts": copy.deepcopy(host.get("Mounts")),
        "NetworkMode": host.get("NetworkMode", "bridge"), "PidsLimit": host.get("PidsLimit"), "PortBindings": {},
        "Privileged": bool(host.get("Privileged")), "PublishAllPorts": False, "ReadonlyPaths": list(READONLY),
        "ReadonlyRootfs": bool(host.get("ReadonlyRootfs")), "RestartPolicy": host.get("RestartPolicy") or {"Name": "no", "MaximumRetryCount": 0},
        "Runtime": "runc", "SecurityOpt": host.get("SecurityOpt"), "ShmSize": host.get("ShmSize", 67108864),
        "Tmpfs": host.get("Tmpfs"), "Ulimits": copy.deepcopy(host.get("Ulimits")),
    })
    mounts = [{"Type": "volume", "Name": m["Source"], "Source": "/var/lib/docker/volumes/%s/_data" % m["Source"],
               "Destination": m["Target"], "Driver": "local", "Mode": "z", "RW": not m.get("ReadOnly"), "Propagation": ""}
              for m in host.get("Mounts") or []]
    entry = body.get("Entrypoint") or []
    return {"Id": ident, "Name": "/" + name, "Image": body.get("Image"), "Path": entry[0] if entry else "",
            "Args": list(entry[1:]) + list(body.get("Cmd") or []), "AppArmorProfile": "", "Config": config_part,
            "HostConfig": host_part, "Mounts": mounts, "Driver": "overlay2", "Platform": "linux",
            "State": {"Status": "created", "Running": False, "ExitCode": 0, "OOMKilled": False, "StartedAt": "", "FinishedAt": ""}}


class FakeDocker:
    def __init__(self, image_table=None):
        self.path = "/fake/docker.sock"
        self.images = dict(image_table if image_table is not None else images())
        self.containers = {}            # id: {"name", "body", "inspect", "logs"}
        self.calls = []                 # (method, path, query)
        self.fail = None                # a `DockerUnreachable` kind: the daemon cannot be reached
        self.alter = None               # callable(inspect) -> inspect, applied to what a created container reads as
        self.on_start = None            # callable(container): what happens when a container starts
        self.version, self.info = dict(VERSION), dict(INFO)

    # -- helpers for the tests ---------------------------------------------------------------------------
    def did(self, method, fragment):
        return [c for c in self.calls if c[0] == method and fragment in c[1]]

    def add(self, name, labels, state="running", **state_fields):
        ident = hashlib.sha256(name.encode()).hexdigest()
        inspect = {"Id": ident, "Name": "/" + name, "Config": {"Labels": dict(labels)},
                   "State": dict({"Status": state, "Running": state == "running", "ExitCode": 0, "OOMKilled": False,
                                  "StartedAt": "2026-01-01T00:00:00Z", "FinishedAt": "0001-01-01T00:00:00Z"}, **state_fields)}
        self.containers[ident] = {"name": name, "body": None, "inspect": inspect, "logs": b""}
        return ident

    def state(self, name):
        found = self._find(name)
        return None if found is None else self.containers[found]["inspect"]["State"]["Status"]

    def _find(self, ref):
        for ident, container in self.containers.items():
            if ref in (ident, container["name"]) or (len(ref) >= 12 and ident.startswith(ref)):
                return ident
        return None

    # -- the Engine API, as far as the launcher uses it ------------------------------------------------------
    def call(self, method, path, query=None, body=None, timeout=None, versioned=True, limit=None):
        query = dict(query or {})
        self.calls.append((method, path, query))
        if self.fail:
            raise launcher.DockerUnreachable(self.fail)
        status, answer = self._route(method, path, query, body)
        if isinstance(answer, bytes):
            return status, answer
        return status, b"" if answer is None else json.dumps(answer).encode()

    def _route(self, method, path, query, body):
        parts = path.strip("/").split("/")
        if (method, path) == ("GET", "/version"):
            return 200, self.version
        if (method, path) == ("GET", "/info"):
            return 200, self.info
        if (method, path) == ("GET", "/containers/json"):
            wanted = dict(item.split("=", 1) for item in json.loads(query.get("filters", "{}")).get("label", []))
            listed = []
            for ident, container in self.containers.items():
                labels = container["inspect"]["Config"].get("Labels") or {}
                if all(labels.get(k) == v for k, v in wanted.items()):
                    listed.append({"Id": ident, "Names": ["/" + container["name"]], "Labels": labels,
                                   "State": container["inspect"]["State"]["Status"], "Image": container["inspect"].get("Image")})
            return 200, listed
        if (method, path) == ("POST", "/containers/create"):
            name = query["name"]
            if self._find(name) is not None:
                return 409, {"message": "Conflict. The container name is already in use"}
            ident = hashlib.sha256(("id:" + name).encode()).hexdigest()
            image = next((i for i in self.images.values() if i["Id"] == body.get("Image")), None)
            if image is None:
                return 404, {"message": "No such image"}
            inspect = synth_inspect(body, ident, name, image["Config"]["Env"])
            if self.alter is not None:
                inspect = self.alter(inspect)
            self.containers[ident] = {"name": name, "body": body, "inspect": inspect, "logs": b""}
            return 201, {"Id": ident, "Warnings": []}
        if parts[0] == "images" and method == "GET":
            found = self.images.get("/".join(parts[1:-1]))
            return (200, found) if found is not None else (404, {"message": "No such image"})
        if parts[0] == "containers" and len(parts) >= 2:
            ident = self._find(parts[1])
            if ident is None:
                return 404, {"message": "No such container"}
            container = self.containers[ident]
            state = container["inspect"].setdefault("State", {"Status": "created"}) if isinstance(container["inspect"], dict) else {}
            action = parts[2] if len(parts) > 2 else ""
            if method == "GET" and action == "json":
                return 200, container["inspect"]
            if method == "POST" and action == "start":
                state.update(Status="running", Running=True)
                if self.on_start is not None:
                    self.on_start(container)
                return 204, None
            if method == "POST" and action == "kill":
                if state["Status"] != "running":
                    return 409, {"message": "is not running"}
                state.update(Status="exited", Running=False, ExitCode=137)
                return 204, None
            if method == "DELETE":
                del self.containers[ident]
                return 204, None
            if method == "GET" and action == "logs":
                return 200, container["logs"]
        return 500, {"message": "unexpected request %s %s" % (method, path)}


def frames(stdout, stderr=b""):
    """A multiplexed log of Docker with one frame per stream."""
    out = b""
    for kind, data in ((2, stderr), (1, stdout)):
        if data:
            out += bytes([kind, 0, 0, 0]) + struct.pack(">I", len(data)) + data
    return out


def manifest_exits(fake, data, exit_code=0):
    """Make the fake run the launcher's own manifest child: it exits at once and its log is `data`."""
    def on_start(container):
        if (container["body"].get("Labels") or {}).get("mdf.kind") == "manifest":
            container["inspect"]["State"].update(Status="exited", Running=False, ExitCode=exit_code)
            container["logs"] = frames(data)
    fake.on_start = on_start


class Clock:
    """A monotonic clock a test moves by hand; `sleep` advances it."""

    def __init__(self, now=1000.0):
        self.now = now

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


def make(fake=None, clock=None, **over):
    """(launcher, fake Docker, clock) with the configuration of the tests."""
    fake = fake or FakeDocker()
    clock = clock or Clock()
    identity = {"payload_sha256": "p" * 64, "inputs_sha256": "i" * 64,
                "environment": {"python_sha256": "y" * 64, "distributions": ["a==1"]}}
    return launcher.Launcher(fake, config(**over), clock=clock, sleep=clock.sleep, identity=identity), fake, clock


def request(kind="light", image="app", argv=None, n=1, **over):
    argv = argv if argv is not None else {
        "tool": ["qpdf", "--version"], "light": containers.python_command(containers.LIGHT_MODULES[0], ["docx", "/in/source.docx"]),
        "marker": containers.python_command(containers.MARKER_MODULES[0], ["convert"]), "cpuserver": ["/opt/llama/llama-server", "--help"],
        "probe": containers.PROBE_PYTHON if image == "app" else containers.PROBE_SHELL, "manifest": containers.MANIFEST_READ}[kind]
    variables = {"tool": {"LC_ALL": "C.UTF-8", "LANG": "C.UTF-8", "HOME": "/work/home", "TMPDIR": "/work/tmp"},
                 "manifest": {"LC_ALL": "C.UTF-8", "LANG": "C.UTF-8"},
                 "cpuserver": {"PATH": "/usr/bin", "LANG": "C.UTF-8", "HOME": "/tmp", "LD_LIBRARY_PATH": "/opt/llama"}}.get(
        kind, {"PATH": "/usr/bin", "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8", "HOME": "/scratch", "TMPDIR": "/scratch"})
    if kind == "probe" and image != "app":
        variables = {"LC_ALL": "C.UTF-8", "LANG": "C.UTF-8", "HOME": "/work", "TMPDIR": "/work"}
    base = {"op": "run", "kind": kind, "name": "%s-%s-%d" % (ATTEMPT, kind, n), "image": image, "argv": argv, "env": variables,
            "limits": dict(LIMITS)}
    base.update(over)
    return base


def with_supervisor(value):
    """A recorded structure with the placeholders replaced by the scripts of today (the supervisor, the shell probe)."""
    if isinstance(value, dict):
        return {k: with_supervisor(v) for k, v in value.items()}
    if isinstance(value, list):
        return [with_supervisor(v) for v in value]
    return PLACEHOLDERS.get(value, value) if isinstance(value, str) else value


def without_supervisor(value):
    if isinstance(value, dict):
        return {k: without_supervisor(v) for k, v in value.items()}
    if isinstance(value, list):
        return [without_supervisor(v) for v in value]
    for placeholder, script in PLACEHOLDERS.items():
        if isinstance(value, str) and value == script:
            return placeholder
    return value


def load_fixture(name):
    return with_supervisor(json.loads((FIXTURES / name).read_text(encoding="utf-8")))


def fixture_names():
    return sorted(p.name for p in FIXTURES.glob("inspect-*.json"))


def golden_cases():
    """(label, request) of every kind and image: the bodies of `tests/fixtures/launcher/body-<label>.json`."""
    cases = []
    for kind, (image_roles, _, _, _, _) in launcher.KINDS.items():
        for image in image_roles:
            if image == "candidate":
                continue
            cases.append(("%s-%s" % (kind, image), request(kind, image, **({"limits": dict(LIMITS, data_mib=512)} if kind == "tool" else {}))))
    cases.append(("manifest-toolchain-root", request("manifest", "toolchain", argv=containers.MANIFEST_REGENERATE)))
    cases.append(("tool-toolchain-pandoc", request("tool", "toolchain", argv=["pandoc", "+RTS", "-M512m", "-RTS", "--sandbox", "-f",
                                                                             "docx", "-t", "json", "/in/source.docx"])))
    return cases


def body_for(case_request, launcher_config=None, t0=1000.0):
    """The body the launcher builds for a request, with the toolchain image as the fake knows it."""
    cfg = launcher_config or config()
    entry = {"id": toolchain_id(), "architecture": "amd64", "inspect": image_inspect(toolchain_id(), TOOLCHAIN_ENV)}
    cfg = launcher.Config(cfg.project, cfg.uid, cfg.gid, cfg.volumes, cfg.roots, dict(cfg.images, toolchain=entry), cfg.daemon)
    kind, name, prefix, image, argv, variables, limits = launcher.check_request(case_request)
    user = launcher.check_command(kind, image, argv)
    mounts = launcher.mounts_of(kind, name, prefix, image)
    body = launcher.body_of(cfg, kind, name, prefix, image, argv, variables, limits, user, mounts, t0)
    launcher.check_body(body, kind, name, prefix, user, mounts, cfg)
    return body
