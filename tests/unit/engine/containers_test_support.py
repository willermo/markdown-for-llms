"""Support of the unit tests of the client of the launcher (MFL051, tranche A1): a launcher that is not there.

`FakeLauncher` answers the calls of `containers.Launcher` from a dictionary of children, and can be made
unreachable or unable to confirm a removal; `Clock` is a monotonic clock a test moves, whose `sleep` advances it and
runs what the test scheduled (what a child, or its supervisor, would do meanwhile).
"""
from markdown_for_llms.engine import containers

ATTEMPT = "0123456789abcdef"
PREFIX = "mdf-eng-" + ATTEMPT
LIMITS = {"memory_mib": 256, "pids": 64, "fsize_mib": 64, "nofile": 256, "tmpfs_mib": 16, "max_seconds": 120,
          "lease_seconds": 30}


class Clock:
    def __init__(self, now=1000.0):
        self.now, self.events = now, []

    def __call__(self):
        return self.now

    def at(self, seconds, action):
        """Run `action()` once the clock has advanced by `seconds` from now."""
        self.events.append((self.now + seconds, action))

    def sleep(self, seconds):
        self.now += seconds
        due = [e for e in self.events if e[0] <= self.now]
        self.events = [e for e in self.events if e[0] > self.now]
        for _, action in sorted(due, key=lambda e: e[0]):
            action()


_IDENTITY = None                        # `containers.identity()` of this process, computed once for every fake


class FakeLauncher:
    def __init__(self):
        self.children = {}              # name: {"state", "exit", "oom_killed"}
        self.down = False               # the launcher does not answer
        self.refuse = None              # (code, detail) for `run`
        self.unconfirmed = set()        # names whose removal Docker does not confirm
        self.reaped = {}                # name: why the launcher itself killed or removed the child (`reaped` of `inspect`)
        self.instance = "fixture-launcher-1"    # the ID of the container of the launcher (`hello`)
        self.code = {}                  # what `hello` reports in place of the code of this process
        self.calls = []

    def _answering(self, op, name=None):
        self.calls.append((op, name))
        if self.down:
            raise containers.Unreachable("connect: ECONNREFUSED")

    def ops(self, op):
        return [c for c in self.calls if c[0] == op]

    def run(self, kind, name, image, argv, variables, limits):
        self._answering("run", name)
        if self.refuse:
            raise containers.Refused(*self.refuse)
        self.children[name] = {"state": "running", "exit": 0, "oom_killed": False}
        self.request = {"kind": kind, "name": name, "image": image, "argv": list(argv), "env": dict(variables), "limits": dict(limits)}
        return {"ok": True, "name": name, "id": "0" * 64}

    def exits(self, name, status=0, oom_killed=False):
        self.children[name].update(state="exited", exit=status, oom_killed=oom_killed)

    def inspect(self, name):
        self._answering("inspect", name)
        child = self.children.get(name)
        why = {"reaped": self.reaped[name]} if name in self.reaped else {}
        return dict({"ok": True, "state": "absent"} if child is None else dict(child, ok=True), **why)

    def kill(self, name):
        self._answering("kill", name)
        if name in self.children and self.children[name]["state"] == "running":
            self.children[name].update(state="exited", exit=137)
        return {"ok": True, "state": "killed"}

    def remove(self, name):
        self._answering("remove", name)
        if name in self.unconfirmed:
            raise containers.Unreachable("removal of %s not confirmed" % name)
        self.children.pop(name, None)
        return {"ok": True, "absent": True}

    def logs(self, name):
        self._answering("logs", name)
        return self.children[name].get("log", b"")

    def hello(self):
        """As a launcher of the image under test: the code of this process, so a worker finds its own code in it.
        `code` replaces what it reports (a launcher of another build), `instance` is the ID of its container."""
        self._answering("hello")
        global _IDENTITY
        if _IDENTITY is None:
            _IDENTITY = containers.identity()
        return dict(_IDENTITY, ok=True, project="mdf-fixture", instance=self.instance, **self.code)

    def list(self, prefix):
        self._answering("list", prefix)
        return [{"name": name, "state": child["state"]} for name, child in sorted(self.children.items())
                if containers.split_name(name)[0] == prefix]


def name(kind="light", n=1, prefix=PREFIX):
    return "%s-%s-%d" % (prefix, kind, n)


FAKE_IMAGE = "sha256:" + "f" * 64


class ScriptedLauncher(FakeLauncher):
    """A launcher whose children run to their end as soon as they are asked for, without any container: the
    callers of the engine (`tools.run`, `isolation.run`, `toolchain`) against it, in a work directory of the test.

    `script(request, call)` stands for the command: it gets the request of `run` and the folder of the call (a
    `Path`: it writes its results under `call / "scratch"`, as the command would) and returns the exit status,
    or (status, stdout, stderr), or a dict with `exit`, `stdout`, `stderr`, `oom_killed`, `peak`, `cpu_usec`,
    `oom_kill`; None leaves the child running until its caller stops it (`ctl/stop`: exit 202). The supervisor's
    files in `stat/` are written here. `requests` keeps every request, `removed` the names removed."""

    def __init__(self, work, script=None, toolchain=None):
        super().__init__()
        self.work, self.script, self.requests, self.removed = work, script, [], []
        self.answer = dict({"ok": True, "image": FAKE_IMAGE, "architecture": "amd64", "verified": True,
                            "docker": "0.0.test"}, **(toolchain or {}))

    def run(self, kind, name, image, argv, variables, limits):
        answer = super().run(kind, name, image, argv, variables, limits)
        self.requests.append(self.request)
        call = self.work / "call" / name
        result = self.script(self.request, call) if self.script else 0
        if result is None:
            return answer
        if isinstance(result, int):
            result = {"exit": result}
        elif isinstance(result, tuple):
            result = dict(zip(("exit", "stdout", "stderr"), result))
        if "stat" in containers.CALL_ROLES[kind]:
            (call / "stat" / "stdout").write_bytes(result.get("stdout", b""))
            (call / "stat" / "stderr").write_bytes(result.get("stderr", b""))
            (call / "stat" / "metrics").write_bytes(b"mdf-metrics/1 %d %d %d 5 end\n" % (
                result.get("peak", 1 << 20), result.get("cpu_usec", 1000), result.get("oom_kill", 0)))
            (call / "stat" / "exit").write_bytes(b"%d\n" % result["exit"])
        else:
            self.children[name]["log"] = result.get("stdout", b"")
        self.exits(name, result["exit"], oom_killed=bool(result.get("oom_killed")))
        return answer

    def inspect(self, name):
        child = self.children.get(name)
        if child is not None and child["state"] == "running" and (self.work / "call" / name / "ctl" / "stop").exists():
            self.exits(name, 202)
        return super().inspect(name)

    def remove(self, name):
        answer = super().remove(name)
        self.removed.append(name)
        return answer

    def toolchain(self, name=None, candidate=False):
        self._answering("toolchain", "candidate" if candidate else None)
        if "refuse" in self.answer:
            raise containers.Refused(*self.answer["refuse"])
        return dict(self.answer)


def patched(test, work, script=None, **kw):
    """Give a test a `ScriptedLauncher` in place of the launcher of the process, on the work and run directories
    `work` of the test; the constants of the volumes and the names of the process are restored when it ends."""
    from unittest import mock
    from pathlib import Path
    work = Path(work)
    (work / "run").mkdir(parents=True, exist_ok=True)
    fake = ScriptedLauncher(work, script, **kw)
    for target, name, value in ((containers, "launcher", lambda: fake), (containers, "WORK_ROOT", str(work)),
                                (containers, "RUN_ROOT", str(work / "run"))):
        patcher = mock.patch.object(target, name, value)
        patcher.start()
        test.addCleanup(patcher.stop)
    return fake
