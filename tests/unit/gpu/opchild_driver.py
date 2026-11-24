"""Runs `markdown_for_llms.gpu.opchild` with a fake provider kept in a JSON file (MFL023 B1), so that a test can
read what the provider was asked, make a call hang or drip past its limit, start a grandchild in the group of the
child (as `modal deploy` does) and kill the parent from outside.

    python -I -B opchild_driver.py STATE.json

The checkout comes first on sys.path (as `tests/unit/servers/session_driver.py` puts `scripts/servers/`): the
child runs the code under test also outside the full check, where the installed wheel may be older.

STATE.json: {"provider": "runpod"|"modal", "behave": {OP: {"sleep": s, "spawn": true, "raise": {"local", "op",
"http", "remote"}, "result": {...}}}}; the driver appends to "calls", sets "pid" (its own), "env" (the names of its
environment) and "grandchild" (the pid of a `sleep` it started) and, while it sleeps, "ticks".
"""
import json
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from markdown_for_llms.gpu import opchild, providers  # noqa: E402


class FileFake:
    def __init__(self, path, request):
        self.path, self.request = Path(path), request
        s = self._load()
        self.provider = s.get("provider", request["provider"])
        self._change(pid=__import__("os").getpid(), env=sorted(__import__("os").environ))

    def _load(self):
        return json.loads(self.path.read_text())

    def _change(self, **change):
        s = self._load()
        s.update(change)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(s))
        tmp.replace(self.path)
        return s

    def _act(self, name, **kw):
        s = self._load()
        self._change(calls=s.get("calls", []) + [[name, kw]])
        how = (s.get("behave") or {}).get(name) or {}
        if how.get("spawn"):
            child = subprocess.Popen(["sleep", "60"])
            self._change(grandchild=child.pid)
        for i in range(int(how.get("sleep") or 0) * 10):
            time.sleep(0.1)
            self._change(ticks=i + 1)
        if how.get("raise"):
            r = how["raise"]
            raise providers.ProviderError(r["local"], op=r.get("op"), http=r.get("http"), remote=r.get("remote"))
        return how.get("result")

    # the face of a provider
    @property
    def url(self):
        return "https://fake-endpoint.api.runpod.ai/"

    def endpoint(self, name=None, refresh=False):
        return self._load().get("endpoint") or {}

    def pin_facts(self):
        return [], [], []

    def state(self):
        return self._act("state") or {"armed": False, "busy": False}

    def arm(self, fault="", allow_download=False, levers="", deadline=None, preset=None):
        got = self._act("arm", fault=fault, levers=levers, deadline=deadline, preset=preset)
        return got if got is not None else ({"url": "https://ws--app-serve.modal.run", "volume": providers.VOLUME,
                                             "proxy_token": {"proxy_tokens": 1, "in_workspace": True}}
                                            if self.provider == "modal" else {"env_changed": False})

    def disarm(self):
        self._act("disarm")

    def zero(self):
        got = self._act("zero")
        return (True, {"armed": False, "busy": False}) if got is None else (got["zero"], got["state"])

    def restore_flashboot(self):
        self._act("restore_flashboot")


if __name__ == "__main__":
    state = sys.argv[1]
    sys.exit(opchild.main(factory=lambda request: FileFake(state, request)))
