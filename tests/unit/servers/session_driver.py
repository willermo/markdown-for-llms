"""Runs `scripts/servers/session.py` with a fake provider kept in a JSON file, so that a test can kill
the session from outside (SIGKILL, SIGTERM) and read what the provider was asked.

    python -I -B session_driver.py STATE.json PREEXEC_DELAY SESSION-ARGS...
"""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts" / "servers"))
import session  # noqa: E402


class FileFake:
    provider = "runpod"
    url = "https://fake-endpoint.api.runpod.ai"

    def __init__(self, path):
        self.path = Path(path)

    def _load(self):
        return json.loads(self.path.read_text())

    def _save(self, s):
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(s))
        tmp.replace(self.path)

    def _call(self, name, **change):
        s = self._load()
        s.setdefault("calls", []).append(name)
        s.update(change)
        self._save(s)
        return s

    def state(self):
        s = self._load()
        return {"armed": s.get("armed", False), "busy": s.get("busy", False), "summary": {}, "workers": []}

    def arm(self, fault="", allow_download=False, levers="", deadline=None):    # levers: N5b; deadline: S-B4
        s = self._call("arm", armed=True, busy=True)
        if s.get("arm_sleep"):                     # S-B4: the change accepted, its reading back slow
            import time
            time.sleep(s["arm_sleep"])
            self._call("arm-read")
        return {}

    def disarm(self):
        self._call("disarm", armed=False, busy=False)

    def zero(self):
        s = self._call("zero")
        return not s.get("armed") and not s.get("busy"), {}

    def balance(self):
        return None


if __name__ == "__main__":
    state, delay = sys.argv[1], float(sys.argv[2])
    session.PREEXEC_DELAY = delay
    sys.exit(session.main(sys.argv[3:], factory=lambda: FileFake(state)))
