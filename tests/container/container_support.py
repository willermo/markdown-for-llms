"""Support of the component tests (MFL051, tranche A1): children as real containers, through a real launcher.

These tests run only inside the `suite` container of `docker/dev/compose.yaml`, started by `make test-components`:
there the launcher's socket, the work and run volumes and the volumes of the weights and of the CPU server are at
their fixed paths. Anywhere else (the fast suite of the full check on the host) every class skips, saying why.

The suite is a client of the launcher like any other (`containers.Launcher`, `containers.Child`). It also has the
Docker socket, for what a client of the launcher must not be able to do and a test needs: stopping and starting
the launcher, and reading what Docker really created (`raw_inspect`) to compare it with what the launcher says.
It only ever touches containers of its own compose project and children that carry its label.
"""
import json
import os
from pathlib import Path
import secrets
import stat
import time
import unittest

from markdown_for_llms import launcher as launcher_module
from markdown_for_llms.engine import containers

DOCKER_SOCKET = "/run/docker.sock"
OUT = Path("/out")
MODELS, CPU_SERVER = Path("/models"), Path("/cpu-server")
REASON = ("no launcher at %s or no Docker socket for the driver at %s: the component tests need a real Docker daemon, "
          "run them with `make test-components`" % (containers.LAUNCHER_SOCKET, DOCKER_SOCKET))
LIMITS = {"memory_mib": 256, "pids": 64, "fsize_mib": 64, "nofile": 256, "tmpfs_mib": 16, "max_seconds": 120,
          "lease_seconds": 30}
VARIABLES = {"PATH": "/opt/mdf/venv/bin:/usr/bin:/bin", "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8", "HOME": "/scratch",
             "TMPDIR": "/scratch"}
TOOL_VARIABLES = {"LC_ALL": "C.UTF-8", "LANG": "C.UTF-8", "HOME": "/work/home", "TMPDIR": "/work/tmp"}
# `limits.json` of the light child (`child.py`); `process_limit` is no longer read by it: the processes of a child are
# limited by its container (see test_child_isolation)
CHILD_LIMITS = {"memory_mib": 4096, "output_mib": 64, "cpu_seconds": 120, "process_limit": 65536, "files_limit": 256}
SHELL_SCRIPT = "#!/bin/sh\n" + containers.SHELL_PROBE


def available():
    """A launcher AND the Docker socket of the driver: in the container of the full check there is a launcher (the
    suite converts through it) and, by design, no Docker socket; these tests run with `make test-components`."""
    try:
        return stat.S_ISSOCK(os.stat(containers.LAUNCHER_SOCKET).st_mode) and stat.S_ISSOCK(os.stat(DOCKER_SOCKET).st_mode)
    except OSError:
        return False


def parse_shell_probe(text):
    """The lines of `containers.SHELL_PROBE` as a dict: single values, and lists for the repeated keys."""
    report = {"interface": [], "variable": [], "write": {}, "mount": [], "cgroup": {}}
    for line in text.splitlines():
        key, _, rest = line.partition(" ")
        if key in ("interface", "variable"):
            report[key].append(rest)
        elif key == "write":
            path, _, answer = rest.rpartition(" ")
            report["write"][path] = answer
        elif key == "cgroup":
            name, _, value = rest.partition(" ")
            report["cgroup"][name] = value
        elif key == "mount":
            target, root, options = rest.split(" ", 2)
            report["mount"].append({"target": target, "root": root, "read_only": "ro" in options.split(",")})
        else:
            report[key] = rest
    return report


def prepare_volumes():
    """What the read-only volumes of `marker` and `cpuserver` must hold for a child to start: a `marker` directory
    in the volume of the weights, and in the one of the CPU server a `gguf` directory and a `llama` tree with a
    program (here the shell probe, run by the image of the application). Every test file that starts such a
    child prepares them itself, so each file passes when it is run alone."""
    (MODELS / "marker").mkdir(mode=0o700, exist_ok=True)
    for name in ("llama", "gguf"):
        (CPU_SERVER / name).mkdir(mode=0o700, exist_ok=True)
    script = CPU_SERVER / "llama" / "probe.sh"
    script.write_text(SHELL_SCRIPT)
    script.chmod(0o755)


class ComponentCase(unittest.TestCase):
    """A test with a launcher, a fresh attempt (so a prefix of its own) and nothing left behind."""

    @classmethod
    def setUpClass(cls):
        cls.launcher = containers.Launcher()
        cls.docker = launcher_module.Docker(DOCKER_SOCKET)
        cls.hello = cls.wait_launcher()
        cls.project = cls.hello["project"]

    @classmethod
    def wait_launcher(cls, seconds=60):
        until = time.monotonic() + seconds
        while True:
            try:
                return cls.launcher.hello()
            except containers.Unreachable:
                if time.monotonic() > until:
                    raise
                time.sleep(0.2)

    def setUp(self):
        self.attempt = secrets.token_hex(8)
        self.prefix = "mdf-eng-" + self.attempt
        self._n = 0
        self.addCleanup(self._clean)

    def _clean(self):
        self.ensure_launcher()
        containers.stop_attempt_children(self.launcher, self.prefix)
        containers.sweep(self.launcher, containers.WORK_ROOT, prefix=self.prefix, min_age=0)
        calls = Path(containers.WORK_ROOT) / "call"
        left = [n for n in (os.listdir(calls) if calls.is_dir() else []) if n.startswith(self.prefix + "-")]
        self.assertEqual(left, [], "call folders left by the test")

    def name(self, kind):
        self._n += 1
        return containers.child_name(self.attempt, kind, self._n)

    # -- Docker, for the driver only -------------------------------------------------------------------
    def raw(self, method, path, **kw):
        status, data = self.docker.call(method, path, **kw)
        try:
            return status, json.loads(data) if data else None
        except ValueError:
            return status, data

    def raw_inspect(self, name):
        status, found = self.raw("GET", "/containers/%s/json" % name)
        return found if status == 200 else None

    def service(self, service):
        """The ID of the container of a compose service of this project."""
        labels = ["com.docker.compose.project=" + self.project, "com.docker.compose.service=" + service]
        status, listed = self.raw("GET", "/containers/json", query={"all": "1", "filters": json.dumps({"label": labels})})
        self.assertEqual((status, len(listed)), (200, 1), service)
        return listed[0]["Id"]

    def kill_launcher(self):
        ident = self.service("launcher")
        status, _ = self.raw("POST", "/containers/%s/kill" % ident, query={"signal": "KILL"})
        self.assertIn(status, (204, 409))
        until = time.monotonic() + 20
        while self.raw_inspect(ident)["State"]["Running"]:
            self.assertLess(time.monotonic(), until, "the launcher did not die")
            time.sleep(0.05)
        return ident

    def start_launcher(self):
        status, _ = self.raw("POST", "/containers/%s/start" % self.service("launcher"))
        self.assertIn(status, (204, 304))
        return self.wait_launcher()

    def ensure_launcher(self):
        if not self.raw_inspect(self.service("launcher"))["State"]["Running"]:
            self.start_launcher()

    # -- children ----------------------------------------------------------------------------------------
    def child(self, kind):
        child = containers.Child(self.launcher, containers.WORK_ROOT, self.name(kind))
        if kind == "marker" or kind == "cpuserver":
            containers.socket_dirs(containers.RUN_ROOT, self.prefix)
            self.addCleanup(self._remove_sockets)
        return child

    def _remove_sockets(self):
        import shutil
        shutil.rmtree(Path(containers.RUN_ROOT) / "sock" / self.prefix, ignore_errors=True)

    def probe(self, kind="light", steps=None, limits=None, child_limits=None, stdin=None, inputs=(), variables=None,
              prepare=None, **run):
        """Run the probe (`selfprobe container`) in a `light` or `marker` child, with `steps`; returns
        (outcome, report or None, the raw inspect taken while it ran or None)."""
        with self.child(kind) as child:
            scratch = child.call.scratch
            (scratch / containers.LIMITS_FILE).write_text(json.dumps(dict(CHILD_LIMITS, **(child_limits or {}))))
            if steps is not None:
                (scratch / "steps.json").write_text(json.dumps(steps))
            if stdin is not None:
                child.call.stdin(stdin)
            for source, name in inputs:
                child.call.add_input(source, name)
            if prepare is not None:
                prepare(child)
            run.setdefault("timeout", 60)
            outcome = child.run("app", containers.python_command(containers.PROBE_MODULE, ["container"]),
                                dict(VARIABLES, **(variables or {})), dict(LIMITS, **(limits or {})), **run)
            try:
                report = json.loads((scratch / "probe.json").read_text())
            except (OSError, ValueError):
                report = None
        return outcome, report

    def step_result(self, report, do):
        return next(s["result"] for s in report["steps"] if s["do"] == do)
