"""A fake rented provider for the GPU sessions of the product (MFL023 B4), kept in a JSON file and served to the children of
`opchild` (`python -I -B provider_driver.py STATE.json`): a test reads what the provider was asked, arms it, makes a call
fail, hang, wait at a barrier or show the endpoint armed again, and kills the worker from outside.

The checkout comes first on sys.path, as in `opchild_driver.py`. STATE.json (every key optional but `provider`):

    provider      "runpod" | "modal"
    armed, busy   the state at rest (false)
    zero_after    readings with a container still running after a disarm, before the zero (0)
    never_zero    true: no zero ever
    rearm         readings after a disarm that show the endpoint armed again (0)
    fail          {OP: N}: the next N calls of OP raise a ProviderError whose remote text carries `leak`
    hang          {OP: S}: OP sleeps S seconds of real time, ticking (`ticks`)
    barrier       {OP: PATH}: OP writes PATH, then waits for PATH + ".go" (30 s at most)
    url           the URL of the endpoint (RunPod) or of the deploy (Modal)
    endpoint      fields that replace those of the RunPod endpoint at rest; volume: those of its volume
    system        the system lines of the Modal app; containers: its containers
    leak          a text the provider repeats in its errors (a synthetic key)

The driver appends each call to `calls` ([op, pid]) under a lock, so that the children of two threads never lose one.
"""
import fcntl
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from markdown_for_llms.gpu import opchild, providers  # noqa: E402

RUNPOD_URL = "https://ep1.api.runpod.ai"
MODAL_URL = "https://ws--mdf-surya-serve.modal.run"


class Locked:
    def __init__(self, path):
        self.path = Path(path)

    def __enter__(self):
        self.fd = os.open(str(self.path) + ".lock", os.O_RDWR | os.O_CREAT, 0o600)
        fcntl.flock(self.fd, fcntl.LOCK_EX)
        self.state = json.loads(self.path.read_text())
        return self.state

    def __exit__(self, *exc):
        if exc[0] is None:
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.state))
            tmp.replace(self.path)
        os.close(self.fd)


def endpoint(over=None):
    ep = {"id": "ep1", "name": "mdf-surya", "type": "LOAD_BALANCER", "image": providers.IMAGE,
          "entrypoint": providers.ENTRYPOINT, "env": providers.base_env(), "workers": providers.workers_block(0),
          "scaling": {"type": "REQUEST_COUNT", "requestCount": 32}, "flashboot": "OFF", "ports": ["8000/http"],
          "dataCenterIds": ["CA-MTL-1"], "mounts": {"network": [{"volumeId": "vol1"}]},
          "gpu": {"pools": [providers.POOL], "minCudaVersion": providers.CUDA_MIN, "count": 1}}
    ep.update(over or {})
    return ep


class Fake:
    def __init__(self, path):
        self.path = path
        with Locked(path) as s:
            self.provider = s["provider"]

    def act(self, op):
        with Locked(self.path) as s:
            s.setdefault("calls", []).append([op, os.getpid()])
            fail = (s.get("fail") or {}).get(op, 0)
            if fail:
                s["fail"][op] = fail - 1
            how = {"hang": (s.get("hang") or {}).get(op), "barrier": (s.get("barrier") or {}).get(op)}
            leak = s.get("leak", "")
        if how["barrier"]:
            Path(how["barrier"]).write_text(str(os.getpid()))
            end = time.monotonic() + 30
            while not Path(how["barrier"] + ".go").exists() and time.monotonic() < end:
                time.sleep(0.02)
        for i in range(int((how["hang"] or 0) * 10)):
            time.sleep(0.1)
            with Locked(self.path) as s:
                s["ticks"] = s.get("ticks", 0) + 1
        if fail:
            raise providers.ProviderError("%s %s: exit 1" % (self.provider, op), op="%s.%s" % (self.provider, op),
                                          remote="Error: " + leak)

    # --- the face of a provider ---------------------------------------------------------------------------

    def state(self):
        self.act("state")
        with Locked(self.path) as s:
            return {"armed": bool(s.get("armed")), "busy": bool(s.get("busy")), "workers": [],
                    "flashboot": (s.get("endpoint") or {}).get("flashboot", "OFF") if self.provider == "runpod" else None}

    @property
    def url(self):
        with Locked(self.path) as s:
            return s.get("url") or RUNPOD_URL

    def endpoint(self, name=None, refresh=False):
        with Locked(self.path) as s:
            return endpoint(s.get("endpoint"))

    def pin_facts(self):
        with Locked(self.path) as s:
            volume = dict({"id": "vol1", "name": providers.VOLUME, "dataCenter": "CA-MTL-1", "type": "HIGH_PERFORMANCE",
                           "size": 10}, **(s.get("volume") or {}))
        return ([volume], [{"id": "NVIDIA RTX A5000", "pool": providers.POOL, "secure": True, "price": {"serverless": 0.75}}],
                [{"id": "CA-MTL-1", "networkVolumeTypes": ["HIGH_PERFORMANCE"]}])

    def arm(self, fault="", allow_download=False, levers="", deadline=None, preset=None):
        self.act("arm")
        with Locked(self.path) as s:
            s.update(armed=True, busy=True, zeros=0, levers=levers)
            url = s.get("url")
        if self.provider == "modal":
            return {"url": url or MODAL_URL, "volume": providers.VOLUME,
                    "proxy_token": {"proxy_tokens": 1, "in_workspace": True, "scoped": False}}
        return {"env_changed": True, "gpu_changed": False, "flashboot": "OFF", "flashboot_changed": False}

    def disarm(self):
        self.act("disarm")
        with Locked(self.path) as s:
            s["armed"] = False
            s["zeros"] = 0

    def zero(self):
        self.act("zero")
        with Locked(self.path) as s:
            s["zeros"] = s.get("zeros", 0) + 1
            if s.get("rearm", 0) > 0:
                s["rearm"] -= 1
                s["armed"] = True
            if s.get("armed") or s.get("never_zero") or s["zeros"] <= s.get("zero_after", 0):
                state = {"armed": bool(s.get("armed")), "busy": True, "workers": [{"id": "w1", "status": "RUNNING"}]}
                return False, state
            s["busy"] = False
            return True, {"armed": False, "busy": False, "workers": []}

    def restore_flashboot(self):
        self.act("restore_flashboot")

    def system_logs(self, since):
        self.act("system_logs")
        with Locked(self.path) as s:
            return list(s.get("system") or [])

    def app_containers(self):
        with Locked(self.path) as s:
            return list(s.get("containers") or [])


if __name__ == "__main__":
    state = sys.argv[1]
    sys.exit(opchild.main(factory=lambda request: Fake(state)))
