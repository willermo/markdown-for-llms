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


class FakeLauncher:
    def __init__(self):
        self.children = {}              # name: {"state", "exit", "oom_killed"}
        self.down = False               # the launcher does not answer
        self.refuse = None              # (code, detail) for `run`
        self.unconfirmed = set()        # names whose removal Docker does not confirm
        self.reaped = {}                # name: why the launcher itself killed or removed the child (`reaped` of `inspect`)
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

    def list(self, prefix):
        self._answering("list", prefix)
        return [{"name": name, "state": child["state"]} for name, child in sorted(self.children.items())
                if containers.split_name(name)[0] == prefix]


def name(kind="light", n=1, prefix=PREFIX):
    return "%s-%s-%d" % (prefix, kind, n)
