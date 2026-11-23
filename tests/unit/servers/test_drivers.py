"""RunPod and Modal from the laptop (`scripts/servers/providers.py`, `drivers.py`, `runpod_endpoint.py`,
`modal_tasks.py`; MFL033 plan r4 App. D and G). No network: `httpx.MockTransport` stands for the
REST v2 and the balancer, a fake runner for the Modal CLI, a fake clock for the readings."""
import io
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import tempfile
import threading
import time
import unittest
from contextlib import redirect_stdout
from unittest import mock

import httpx

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts" / "servers"))
import drivers as dv  # noqa: E402
import providers as pv  # noqa: E402
import runpod_endpoint as rpe  # noqa: E402

from markdown_for_llms.bench import spend  # noqa: E402

KEY = "rp-secret-value"


class Clock:
    def __init__(self, t=1_800_000_000.0):
        self.t = t

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.t += s


def stored(env):
    """The env as RunPod stores it: the empty values are dropped (seen on 19/11)."""
    return {k: v for k, v in env.items() if v != ""}


class FakeRunPod:
    """The REST v2 of one endpoint and its volume, the balancer of the endpoint."""

    def __init__(self, env=None, workers=None, init_timeout=960, cuda="13.0"):
        # `workers.initTimeout` is always in the answers of the REST v2 (null when not set, 19/11);
        # `gpu` as read on 19/11: pools, the CUDA constraints, the count
        self.ep = {"id": "ep123", "name": "mdf-surya-fault", "type": "LOAD_BALANCER", "image": pv.IMAGE,
                   "entrypoint": pv.ENTRYPOINT, "env": stored(env if env is not None else pv.base_env()),
                   "workers": {"min": 0, "max": 0, "idleTimeout": 60, "initTimeout": init_timeout},
                   "scaling": {"type": "REQUEST_COUNT", "requestCount": 32},
                   "flashboot": "OFF", "ports": ["8000/http"], "dataCenterIds": ["EU-RO-1"],
                   "mounts": {"network": [{"volumeId": "vol1"}]},
                   "gpu": {"pools": [pv.POOL], "minCudaVersion": cuda, "allowedCudaVersions": [], "count": 1}}
        self.workers = list(workers or [])        # successive answers of /workers; the last one repeats
        self.calls, self.lb = [], {}
        self.exists = True
        # the catalog as read on 18/11 (n4/catalog.out): the pool Secure, 0.75 USD/h serverless
        self.gpus = [{"id": "NVIDIA RTX A5000", "pool": pv.POOL, "secure": True, "price": {"serverless": 0.75}}]
        self.data_centers = [{"id": "EU-RO-1", "networkVolumeTypes": ["STANDARD"]}, {"id": "US-KS-2"}]

    def __call__(self, request):
        url, method = str(request.url), request.method
        auth = request.headers.get("authorization")
        self.calls.append((method, url, json.loads(request.content) if request.content else None, auth))
        if ".api.runpod.ai" in url:
            path = request.url.path
            answer = self.lb.get((method, path), (404, {}, None))
            answer = answer() if callable(answer) else answer
            if isinstance(answer, Exception):
                raise answer
            status, body, proxy = answer
            if auth != "Bearer " + KEY:
                status, body, proxy = 401, {"error": "unauthorized"}, None
            return httpx.Response(status, json=body, headers={"x-mdf-proxy": proxy} if proxy else {})
        path = url.split("/v2", 1)[1].split("?")[0]
        if path == "/serverless" and method == "GET":
            return httpx.Response(200, json={"endpoints": [self.ep] if self.exists else [], "pagination": {}})
        if path == "/serverless" and method == "POST":
            return httpx.Response(201, json=dict(json.loads(request.content), id="ep999"))
        if path == "/serverless/ep123" and method == "GET":
            return httpx.Response(200, json=self.ep)
        if path == "/serverless/ep123" and method == "PATCH":
            body = json.loads(request.content)
            if "workers" in body:                 # the keys of `workers` left out stay as they are
                body["workers"] = dict(self.ep["workers"], **body["workers"])
            if "gpu" in body:                     # a partial update: an omitted field is left unchanged
                body["gpu"] = dict(self.ep["gpu"], **body["gpu"])
            if "env" in body:
                body["env"] = stored(body["env"])
            self.ep.update(body)
            return httpx.Response(200, json=self.ep)
        if path == "/serverless/ep123" and method == "DELETE":
            self.exists = False
            return httpx.Response(204)
        if path == "/serverless/ep123/workers":
            w = self.workers.pop(0) if len(self.workers) > 1 else (self.workers[0] if self.workers else [])
            return httpx.Response(200, json={"workers": w, "summary": {"total": len(w), "unhealthy": sum(
                1 for x in w if x.get("status") == "UNHEALTHY")}})
        if path == "/network-volumes" and method == "GET":
            return httpx.Response(200, json={"networkVolumes": [{"id": "vol1", "name": pv.VOLUME, "dataCenter": "EU-RO-1"}]})
        if path == "/network-volumes" and method == "POST":
            return httpx.Response(201, json={"id": "vol2"})
        if path == "/catalog/gpus":
            return httpx.Response(200, json={"gpus": self.gpus})
        if path == "/catalog/datacenters":
            return httpx.Response(200, json={"dataCenters": self.data_centers})
        return httpx.Response(404, json={})


class Base(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.dir, True)
        self.env = self.dir / ".env"
        self.env.write_text("RUNPOD_API_KEY=%s\n" % KEY)
        self.env.chmod(0o600)
        self.clock = Clock()

    def driver(self, fake, *argv):
        a = rpe.parser().parse_args(["--env-file", str(self.env)] + list(argv))
        return rpe.Driver(a, transport=httpx.MockTransport(fake), clock=self.clock, sleep=self.clock.sleep)

    def run_quiet(self, fn):
        out = io.StringIO()
        with redirect_stdout(out):
            code = fn()
        return code, [json.loads(x) for x in out.getvalue().splitlines() if x.strip()]


class TestRunPod(Base):
    def test_create_body_is_disarmed_in_the_volume_data_center(self):
        body = pv.create_body("mdf-surya", "vol1", "EU-RO-1")
        self.assertEqual(body["workers"], {"min": 0, "max": 0, "idleTimeout": 60, "initTimeout": 960})
        self.assertEqual((body["dataCenterIds"], body["mounts"]), (["EU-RO-1"], {"network": [{"volumeId": "vol1"}]}))
        self.assertEqual((body["type"], body["ports"], body["image"]), ("LOAD_BALANCER", ["8000/http"], pv.IMAGE))
        self.assertEqual(body["gpu"], {"pools": [pv.POOL], "minCudaVersion": "13.0"})   # the CUDA floor (c7.4)
        env = body["env"]
        self.assertEqual((env["RUNPOD_INIT_TIMEOUT"], env["MDF_ALLOW_DOWNLOAD"]), ("960", "0"))
        self.assertNotIn("MDF_FAULT", env)                                      # no empty value: RunPod drops it
        self.assertNotIn("MDF_EXPIRES_AT", env)
        import base64
        import hashlib
        data = base64.b64decode(env["MDF_WATCHDOG_B64"])
        self.assertEqual(data, (ROOT / "scripts" / "servers" / "watchdog.py").read_bytes())
        self.assertEqual(hashlib.sha256(data).hexdigest(), env["MDF_WATCHDOG_SHA256"])

    def test_create_dry_run_writes_nothing_then_creates(self):
        fake = FakeRunPod()
        fake.exists = False
        code, _ = self.run_quiet(self.driver(fake, "create", "--name", "mdf-surya", "--data-center", "EU-RO-1",
                                             "--dry-run").create)
        self.assertEqual(code, 0)
        self.assertFalse([c for c in fake.calls if c[0] == "POST"])
        code, lines = self.run_quiet(self.driver(fake, "create", "--name", "mdf-surya", "--data-center", "EU-RO-1").create)
        posts = [c for c in fake.calls if c[0] == "POST"]
        self.assertEqual([p[1].split("/v2")[1] for p in posts], ["/serverless"])  # the volume exists already
        self.assertEqual(posts[0][2]["mounts"], {"network": [{"volumeId": "vol1"}]})

    def test_create_refuses_a_volume_in_another_data_center(self):
        fake = FakeRunPod()
        fake.exists = False
        d = self.driver(fake, "create", "--name", "mdf-surya", "--data-center", "US-KS-2")
        with self.assertRaises(pv.ProviderError):
            d.create()

    def test_arm_sends_the_env_only_when_it_changes_and_reads_back(self):
        fake = FakeRunPod()
        rp = pv.RunPod(envfile_secret(), "mdf-surya-fault", transport=httpx.MockTransport(fake))
        off = {"flashboot": "OFF", "flashboot_changed": False}                  # MFL045 c3: FlashBoot off at the arm
        self.assertEqual(rp.arm(), dict(off, env_changed=False, gpu_changed=False))
        patch = [c for c in fake.calls if c[0] == "PATCH"][-1][2]
        self.assertEqual(patch, {"workers": {"min": 0, "max": 1, "idleTimeout": 60, "initTimeout": 960}})
        self.assertEqual(rp.arm("hang-after-first"), dict(off, env_changed=True, gpu_changed=False))
        patch = [c for c in fake.calls if c[0] == "PATCH"][-1][2]
        self.assertEqual(patch["env"]["MDF_FAULT"], "hang-after-first")
        rp.disarm()
        self.assertEqual(fake.ep["workers"]["max"], 0)
        rp.arm()                                                                # the next arm restores the base
        self.assertEqual(fake.ep["env"], pv.base_env())

    def test_arm_reads_back_the_env_runpod_stores(self):
        # c7.3: RunPod drops the empty values of the env; an empty MDF_FAULT made every read-back differ
        for fault in ("", "hang-after-first", "crash-after-first"):
            for allow in (False, True):
                self.assertNotIn("", pv.base_env(fault, allow).values())
        fake = FakeRunPod()
        rp = pv.RunPod(envfile_secret(), "mdf-surya-fault", transport=httpx.MockTransport(fake))
        self.assertTrue(rp.arm(allow_download=True)["env_changed"])
        self.assertFalse(rp.arm(allow_download=True)["env_changed"])           # no needless release
        rp.disarm()
        self.assertTrue(rp.arm("crash-after-first")["env_changed"])
        self.assertEqual(fake.ep["env"]["MDF_FAULT"], "crash-after-first")
        rp.disarm()
        self.assertTrue(rp.arm()["env_changed"])
        self.assertNotIn("MDF_FAULT", fake.ep["env"])

    def test_the_populate_arm_follows_its_w1(self):
        # c7.4: W1 of the populate 1500 s, so initTimeout 1560 s in the arm and in the env; the disarm
        # and the next arm bring back 960 s; verify-pins between them leaves out the keys of the arm
        fake = FakeRunPod()
        rp = pv.RunPod(envfile_secret(), "mdf-surya-fault", transport=httpx.MockTransport(fake))
        rp.arm(allow_download=True)
        patch = [c for c in fake.calls if c[0] == "PATCH"][-1][2]
        self.assertEqual(patch["workers"], {"min": 0, "max": 1, "idleTimeout": 60, "initTimeout": 1560})
        self.assertEqual((patch["env"]["RUNPOD_INIT_TIMEOUT"], patch["env"]["MDF_ALLOW_DOWNLOAD"]), ("1560", "1"))
        rp.disarm()
        self.assertEqual(fake.ep["workers"], {"min": 0, "max": 0, "idleTimeout": 60, "initTimeout": 960})
        code, lines = self.run_quiet(self.driver(fake, "verify-pins", "--name", "mdf-surya-fault").verify)
        self.assertEqual(code, 0, lines)
        rp.arm()
        self.assertEqual((fake.ep["workers"]["initTimeout"], fake.ep["env"]), (960, pv.base_env()))

    def test_the_cuda_floor(self):
        # c7.4: the image asks for CUDA >= 13.0; the endpoint of 19/11 has the default 12.0
        fake = FakeRunPod(cuda="12.0")
        code, lines = self.run_quiet(self.driver(fake, "verify-pins", "--name", "mdf-surya-fault").verify)
        self.assertEqual(code, 6)
        self.assertEqual([k for k, v in lines[0]["checks"].items() if not v], ["gpu"])
        rp = pv.RunPod(envfile_secret(), "mdf-surya-fault", transport=httpx.MockTransport(fake))
        self.assertTrue(rp.arm()["gpu_changed"])
        patch = [c for c in fake.calls if c[0] == "PATCH"][-1][2]
        self.assertEqual(patch["gpu"], {"minCudaVersion": "13.0"})              # no `pools`: the selection stays
        self.assertEqual(fake.ep["gpu"]["pools"], [pv.POOL])
        self.assertFalse(rp.arm()["gpu_changed"])
        for gpu in ({"minCudaVersion": "13.0"}, {"pools": [pv.POOL], "minCudaVersion": "12.8"},
                    {"pools": [pv.POOL], "minCudaVersion": "13.0", "allowedCudaVersions": ["13.0"]},
                    {"pools": [pv.POOL], "minCudaVersion": "13.0", "excludedTypes": ["NVIDIA L4"]},
                    {"pools": [pv.POOL, "ADA_24"], "minCudaVersion": "13.0"}):
            self.assertFalse(pv.gpu_ok(gpu), gpu)
        fake = FakeRunPod(cuda="12.0")
        original = fake.__call__

        def ignoring_gpu(request):
            if request.method == "PATCH":
                body = json.loads(request.content)
                body.pop("gpu", None)
                request = httpx.Request("PATCH", request.url, json=body, headers=request.headers)
            return original(request)
        rp = pv.RunPod(envfile_secret(), "mdf-surya-fault", transport=httpx.MockTransport(ignoring_gpu))
        with self.assertRaises(pv.ProviderError):
            rp.arm()

    def test_align_a_disarmed_endpoint(self):
        old = dict(pv.base_env(allow_download=True), MDF_WATCHDOG_B64="b2xk", MDF_WATCHDOG_SHA256="0" * 64)
        fake = FakeRunPod(env=old, cuda="12.0", init_timeout=None)
        code, lines = self.run_quiet(self.driver(fake, "verify-pins", "--name", "mdf-surya-fault").verify)
        self.assertEqual(code, 6)
        code, lines = self.run_quiet(self.driver(fake, "align", "--name", "mdf-surya-fault").align)
        self.assertEqual(code, 0, lines)
        self.assertEqual(lines[0]["patched"], ["env", "gpu", "workers"])
        self.assertEqual(fake.ep["env"], pv.base_env())
        code, lines = self.run_quiet(self.driver(fake, "verify-pins", "--name", "mdf-surya-fault").verify)
        self.assertEqual(code, 0, lines)
        code, lines = self.run_quiet(self.driver(fake, "align", "--name", "mdf-surya-fault").align)
        self.assertEqual((code, lines[0]["patched"]), (0, []))                  # nothing to change
        fake.ep["workers"]["max"] = 1                                           # armed: refused, no PATCH
        n = len([c for c in fake.calls if c[0] == "PATCH"])
        code, lines = self.run_quiet(lambda: rpe.main(["--env-file", str(self.env), "align", "--name", "mdf-surya-fault"],
                                                      transport=httpx.MockTransport(fake)))
        self.assertEqual((code, lines[0]["code"]), (4, "provider_error"))
        self.assertEqual(len([c for c in fake.calls if c[0] == "PATCH"]), n)

    def test_arm_read_back_mismatch(self):
        fake = FakeRunPod()
        original = fake.__call__

        def ignoring(request):
            if request.method == "PATCH":
                return httpx.Response(200, json=fake.ep)                       # the change is not applied
            return original(request)
        rp = pv.RunPod(envfile_secret(), "mdf-surya-fault", transport=httpx.MockTransport(ignoring))
        with self.assertRaises(pv.ProviderError):
            rp.arm()

    def test_state_and_zero(self):
        fake = FakeRunPod(workers=[[{"id": "w1", "status": "RUNNING", "gpuTypeId": "NVIDIA RTX A5000",
                                     "dataCenterId": "EU-SE-1", "uptimeSeconds": 7}], []])
        rp = pv.RunPod(envfile_secret(), "mdf-surya-fault", transport=httpx.MockTransport(fake))
        zero, state = rp.zero()
        self.assertEqual(zero, False)
        self.assertEqual(state["workers"], [{"id": "w1", "status": "RUNNING", "uptime": 7, "gpu": "NVIDIA RTX A5000",
                                             "data_center": "EU-SE-1", "version": None,
                                             "stale": None}])     # the evidence of each reading (MFL045 c3: version)
        self.assertEqual(rp.zero()[0], True)

    def test_verify_pins(self):
        fake = FakeRunPod()
        code, lines = self.run_quiet(self.driver(fake, "verify-pins", "--name", "mdf-surya-fault").verify)
        self.assertEqual(code, 0, lines)
        self.assertTrue(all(lines[0]["checks"].values()))
        self.assertEqual(lines[0]["url"], "https://ep123.api.runpod.ai")
        fake.ep["env"] = dict(pv.base_env(), MDF_WATCHDOG_SHA256="0" * 64)
        code, lines = self.run_quiet(self.driver(fake, "verify-pins", "--name", "mdf-surya-fault").verify)
        self.assertEqual(code, 6)
        self.assertFalse(lines[0]["checks"]["watchdog_sha256"])

    def test_levers_in_the_env_of_the_arm_only(self):
        """MFL033 N5b: MDF_LEVERS in the env of an arm with levers, absent without; verify-pins ignores it
        like the other keys of the arm; an unknown lever is refused before any call."""
        self.assertNotIn("MDF_LEVERS", pv.base_env())
        self.assertEqual(pv.base_env(levers=" skip-mm-profiling ")["MDF_LEVERS"], "skip-mm-profiling")
        self.assertIn("MDF_LEVERS", pv.ARM_ENV_KEYS)
        with self.assertRaises(pv.ProviderError):
            pv.base_env(levers="nope")
        fake = FakeRunPod()
        rp = pv.RunPod(envfile_secret(), "mdf-surya-fault", transport=httpx.MockTransport(fake))
        rp.arm(levers="skip-mm-profiling")
        sent = [c[2] for c in fake.calls if c[0] == "PATCH"][-1]
        self.assertEqual(sent["env"]["MDF_LEVERS"], "skip-mm-profiling")
        self.assertEqual(fake.ep["env"]["MDF_LEVERS"], "skip-mm-profiling")
        fake.ep["workers"]["max"] = 0
        code, lines = self.run_quiet(self.driver(fake, "verify-pins", "--name", "mdf-surya-fault").verify)
        self.assertEqual(code, 0, lines)
        rp.arm()                                          # the next arm without levers takes it away
        self.assertNotIn("MDF_LEVERS", fake.ep["env"])

    def test_init_timeout_null_fails_the_pins_and_the_disarm_sets_it(self):
        # c7.2: an endpoint created without `workers.initTimeout` reads it back as null, and the
        # platform default (420 s in the docs) could replace a worker that loads vLLM in 900 s
        fake = FakeRunPod(init_timeout=None)
        code, lines = self.run_quiet(self.driver(fake, "verify-pins", "--name", "mdf-surya-fault").verify)
        self.assertEqual(code, 6)
        self.assertEqual([k for k, v in lines[0]["checks"].items() if not v], ["workers"])
        pv.RunPod(envfile_secret(), "mdf-surya-fault", transport=httpx.MockTransport(fake)).disarm()
        self.assertEqual(fake.ep["workers"], {"min": 0, "max": 0, "idleTimeout": 60, "initTimeout": 960})
        code, lines = self.run_quiet(self.driver(fake, "verify-pins", "--name", "mdf-surya-fault").verify)
        self.assertEqual(code, 0, lines)

    def test_arm_read_back_without_the_init_timeout(self):
        fake = FakeRunPod(init_timeout=None)
        original = fake.__call__

        def dropping(request):
            if request.method == "PATCH":                                      # initTimeout not applied
                body = json.loads(request.content)
                body["workers"].pop("initTimeout", None)
                request = httpx.Request("PATCH", request.url, json=body, headers=request.headers)
            return original(request)
        rp = pv.RunPod(envfile_secret(), "mdf-surya-fault", transport=httpx.MockTransport(dropping))
        with self.assertRaises(pv.ProviderError):
            rp.arm()

    def test_errors_never_show_the_key(self):
        def fail(request):
            return httpx.Response(500, text="internal")
        rp = pv.RunPod(envfile_secret(), "x", transport=httpx.MockTransport(fail))
        with self.assertRaises(pv.ProviderError) as cm:
            rp.endpoints()
        self.assertNotIn(KEY, str(cm.exception))
        self.assertIn("Secret(***)", repr(envfile_secret()))


def envfile_secret():
    from markdown_for_llms.engine.envfile import Secret
    return Secret(KEY)


class TestPinsBeforeTheArm(Base):
    """S-B2 of the final review of MFL033: one function of the pins (`providers.pin_checks`) for verify-pins
    and every arm; a pin that align cannot change refuses the arm on the same endpoint, before any PATCH."""

    def drifts(self):
        def volume_elsewhere(f):
            f.ep["dataCenterIds"] = ["US-KS-2"]
        cases = {
            "image": lambda f: f.ep.update(image="vllm/vllm-openai:latest"),
            "entrypoint": lambda f: f.ep.update(entrypoint=["python3", "-m", "vllm.entrypoints.openai.api_server"]),
            "volume": lambda f: f.ep.update(mounts={"network": [{"volumeId": "vol9"}]}),
            "data_centers": volume_elsewhere,
            "secure": lambda f: f.gpus[0].update(secure=False),
            "rate": lambda f: f.gpus[0].update(price={"serverless": 0.76}),
            "pool": lambda f: f.ep["gpu"].update(excludedTypes=["NVIDIA RTX A5000"]),
            "type": lambda f: f.ep.update(type="QUEUE"),
            # `flashboot` is alignable since MFL045 c3 (plan r2 §6, A7): test_flashboot_runpod.py
            "ports": lambda f: f.ep.update(ports=["8000/http", "22/tcp"]),
            "scaling": lambda f: f.ep.update(scaling={"type": "QUEUE_DELAY", "requestCount": 4}),
        }
        for name, drift in cases.items():
            fake = FakeRunPod()
            drift(fake)
            yield name, fake

    def patches(self, fake):
        return [c for c in fake.calls if c[0] == "PATCH"]

    def test_the_pins_of_a_fresh_endpoint_hold(self):
        fake = FakeRunPod()
        rp = pv.RunPod(envfile_secret(), "mdf-surya-fault", transport=httpx.MockTransport(fake))
        checks, _ = rp.pins()
        self.assertTrue(all(checks.values()), checks)
        self.assertEqual(rp.pin_refusal(), [])
        for key in ("image", "entrypoint", "volume", "data_centers", "secure", "rate", "pool"):
            self.assertIn(key, checks)
            self.assertNotIn(key, pv.PIN_ALIGNABLE)

    def test_a_drift_outside_align_refuses_the_arm_without_a_patch(self):
        for name, fake in self.drifts():
            with self.subTest(pin=name):
                code, lines = self.run_quiet(self.driver(fake, "verify-pins", "--name", "mdf-surya-fault").verify)
                self.assertEqual(code, 6)
                self.assertEqual(lines[0]["outside_align"], [name])
                rp = pv.RunPod(envfile_secret(), "mdf-surya-fault", transport=httpx.MockTransport(fake))
                with self.assertRaises(pv.ProviderError) as caught:
                    rp.arm()
                self.assertIn(name, str(caught.exception))
                self.assertEqual(self.patches(fake), [])
                self.assertEqual(fake.ep["workers"]["max"], 0)

    def test_the_session_refuses_before_its_arm_row(self):
        """The case of the probe of Sol: image and entrypoint outside the pins; the session refuses at its
        start, with no arm row, no window lock and no PATCH."""
        import session
        fake = FakeRunPod()
        fake.ep.update(image="other-image", entrypoint=["other-entrypoint"])
        ledger = self.dir / "spend.jsonl"
        a = session.parser().parse_args(["run", "--provider", "runpod", "--name", "mdf-surya-fault", "--env-file",
                                         str(self.env), "--spend-ledger", str(ledger), "--spend-cap-usd", "4",
                                         "--cap-usd", "0.60", "--", "true"])
        a.command = a.command[1:] if a.command[:1] == ["--"] else a.command
        rp = pv.RunPod(envfile_secret(), "mdf-surya-fault", transport=httpx.MockTransport(fake))
        with self.assertRaises(session.Refusal) as caught:
            session.Session(a, rp, clock=self.clock, sleep=self.clock.sleep, out=io.StringIO()).run()
        self.assertEqual((caught.exception.code, caught.exception.exit_code), ("endpoint_out_of_pins", 3))
        self.assertIn("entrypoint, image", caught.exception.detail)
        self.assertEqual(self.patches(fake), [])
        self.assertFalse(ledger.exists() and ledger.read_text().strip())
        self.assertEqual(list(self.dir.glob("*.lock")), [])

    def test_an_arm_that_changes_an_invariant_pin_is_refused_on_the_read_back(self):
        fake = FakeRunPod()
        original = fake.__call__

        def drifting(request):
            answer = original(request)
            if request.method == "PATCH":
                fake.ep["image"] = "vllm/vllm-openai:latest"
            return answer
        rp = pv.RunPod(envfile_secret(), "mdf-surya-fault", transport=httpx.MockTransport(drifting))
        with self.assertRaises(pv.ProviderError):
            rp.arm()


class TestTermDuringTheArm(Base):
    """S-B4 of the final review of MFL033 on RunPod: the term T of the session holds during the arm. The
    PATCH of the arm is accepted, then its read-back ends after T: the outcome is uncertain, no command
    starts, the session disarms and reads the zero."""

    def session(self, fake, until_s=300):
        import session
        ledger = self.dir / "spend.jsonl"
        a = session.parser().parse_args(["run", "--provider", "runpod", "--name", "mdf-surya-fault", "--env-file",
                                         str(self.env), "--spend-ledger", str(ledger), "--spend-cap-usd", "4",
                                         "--cap-usd", "0.60", "--window-seconds", str(until_s), "--", "true"])
        a.command = a.command[1:] if a.command[:1] == ["--"] else a.command
        timeouts = []

        def transport(request):
            timeouts.append((request.method, request.url.path, dict(request.extensions.get("timeout") or {})))
            return fake(request)
        rp = pv.RunPod(envfile_secret(), "mdf-surya-fault", transport=httpx.MockTransport(transport), clock=self.clock,
                       sleep=self.clock.sleep)
        return session.Session(a, rp, clock=self.clock, sleep=self.clock.sleep, out=io.StringIO()), ledger, timeouts

    def test_a_read_back_after_the_term_disarms_without_the_command(self):
        from unittest import mock
        import session
        fake = FakeRunPod()
        original = fake.__call__
        armed_at = []

        def slow(request):
            answer = original(request)
            if request.method == "PATCH" and fake.ep["workers"]["max"] == 1:
                armed_at.append(self.clock())
                self.clock.t += 400                       # the PATCH accepted; its answer comes after T
            return answer
        ses, ledger, timeouts = self.session(slow, until_s=20)
        with mock.patch.object(session.subprocess, "Popen") as popen:
            code = ses.run()
        self.assertEqual(code, 5)
        popen.assert_not_called()
        self.assertEqual(len(armed_at), 1)
        self.assertEqual(fake.ep["workers"]["max"], 0)                   # the disarm after the uncertain arm
        rows = spend.read(ledger)
        disarm, = [r for r in rows if r.get("kind") == "disarm"]
        self.assertFalse(disarm["clean"])
        detail, = [e["detail"] for e in ses.events if e["event"] == "provider-error"]
        self.assertIn("the term of the session came during the arm", detail)
        # every call of the arm got at most the 20 s left to T, none after the PATCH but the disarm
        bound = [(m, p, t["read"]) for m, p, t in timeouts if t.get("read") is not None and t["read"] < 30]
        self.assertGreaterEqual(len(bound), 6, timeouts)                 # endpoint, pins, PATCH
        self.assertTrue(all(read <= 20 for _, _, read in bound), bound)
        after = timeouts[[m for m, _, _ in timeouts].index("PATCH") + 1:]
        self.assertEqual([m for m, _, t in after if t.get("read", 30) < 30], [])

    def test_no_call_of_the_arm_after_the_term(self):
        fake = FakeRunPod()
        rp = pv.RunPod(envfile_secret(), "mdf-surya-fault", transport=httpx.MockTransport(fake), clock=self.clock)
        with self.assertRaises(pv.TermReached):
            rp.arm(deadline=self.clock() - 1)
        self.assertEqual(fake.calls, [])
        rp.disarm()                                       # the disarm after it is not bound by the term
        self.assertEqual(fake.ep["workers"]["max"], 0)


class StateOnly:
    def __init__(self, states):
        self.states = list(states)

    def state(self):
        return self.states.pop(0) if len(self.states) > 1 else self.states[0]


def st(*workers, armed=True, total=None):
    w = [{"id": i, "status": s, "uptime": u} for i, s, u in workers]
    return {"armed": armed, "busy": bool(w), "summary": {"total": len(w) if total is None else total,
                                                         "unhealthy": sum(1 for x in w if x["status"] == "UNHEALTHY")},
            "workers": w}


class TestCriteria(unittest.TestCase):
    def watch(self, *states):
        c = Clock()
        return dv.Watch(StateOnly(states), c, c.sleep)

    def test_total_over_one(self):
        with self.assertRaises(dv.Violation):
            self.watch(st(total=2)).read()

    def test_unhealthy_and_restart_in_a(self):
        w = self.watch(st(("w1", "RUNNING", 10)), st(("w1", "UNHEALTHY", 20)))
        w.read(no_unhealthy=True)
        with self.assertRaises(dv.Violation):
            w.read(no_unhealthy=True)
        w = self.watch(st(("w1", "RUNNING", 10)), st(("w1", "RUNNING", 3)))
        w.read(no_restart=True)
        with self.assertRaises(dv.Violation):
            w.read(no_restart=True)
        w = self.watch(st(("w1", "RUNNING", 10)), st(), st(("w2", "INITIALIZING", 1)))
        w.read(no_restart=True)
        w.read(no_restart=True)
        with self.assertRaises(dv.Violation):
            w.read(no_restart=True)

    def test_a_worker_outside_the_data_center(self):
        c = Clock()
        state = st(("w1", "INITIALIZING", None))
        state["workers"][0]["data_center"] = "EU-SE-1"
        dv.Watch(StateOnly([state]), c, c.sleep, data_centers=["EU-SE-1"]).read()
        state["workers"][0]["data_center"] = "US-KS-2"
        with self.assertRaises(dv.Violation):
            dv.Watch(StateOnly([state]), c, c.sleep, data_centers=["EU-SE-1"]).read()
        dv.Watch(StateOnly([state]), c, c.sleep).read()                       # no constraint given

    def test_restarts_counted_in_b(self):
        w = self.watch(st(("w1", "RUNNING", 10)), st(), st(("w2", "INITIALIZING", 1)), st())
        w.for_seconds(40)
        self.assertEqual((w.restarts, w.ids), (1, ["w1", "w2"]))

    def test_worker_while_disarmed_and_zero(self):
        with self.assertRaises(dv.Violation):
            self.watch(st(("w1", "INITIALIZING", 1), armed=False)).read(allow_workers=False)
        w = self.watch(st(("w1", "RUNNING", 10)), st(("w1", "IDLE", 20)), st())
        self.assertEqual(w.until_zero(100), 20.0)
        with self.assertRaises(dv.Violation):
            self.watch(st(("w1", "RUNNING", 10))).until_zero(30)

    def test_extra_workers_in_the_readings(self):
        """c7.19: with `extra_ok` only the workers RUNNING count; a replacement without `judge` is a
        restart, a violation with `no_restart`."""
        c = Clock()

        def watch(*states):
            return dv.Watch(StateOnly(states), c, c.sleep, extra_ok=True)
        w = watch(st(("w1", "INITIALIZING", None)), st(("w1", "RUNNING", 5), ("w2", "INITIALIZING", None)),
                  st(("w1", "RUNNING", 15), ("w2", "THROTTLED", None)), st(("w2", "IDLE", None)))
        for _ in range(3):
            w.read(no_unhealthy=True, no_restart=True)
        self.assertEqual([(x["id"], x["states"]) for x in w.extra_workers()], [("w2", ["INITIALIZING", "THROTTLED"])])
        self.assertEqual((w.restarts, w.running_ids), (0, ["w1"]))
        self.assertEqual(w.until_zero(30, no_unhealthy=True, no_restart=True), 0.0)   # the extra worker is no worker
        two = st(("w1", "RUNNING", 5), ("w2", "INITIALIZING", None))
        two["summary"]["running"] = 2                                         # the summary counts one more
        with self.assertRaises(dv.Violation):
            watch(two).read()
        w = watch(st(("w1", "RUNNING", 5)), st(), st(("w2", "RUNNING", 1)))
        w.for_seconds(30)
        self.assertEqual((w.restarts, w.replacements[0]["old"], w.replacements[0]["new"]), (1, "w1", "w2"))
        w = watch(st(("w1", "RUNNING", 5)), st(("w2", "RUNNING", 1)))
        w.read(no_restart=True)
        with self.assertRaises(dv.Violation) as ctx:
            w.read(no_restart=True)
        self.assertIn("a restart: worker w2 RUNNING after w1", str(ctx.exception))

    def test_guarded_runs_the_cleanup_on_sigterm(self):
        done = []

        def body():
            os.kill(os.getpid(), signal.SIGTERM)
            time.sleep(5)
            done.append("body")

        before = signal.getsignal(signal.SIGTERM)
        with self.assertRaises(dv.Stop):
            dv.guarded(body, lambda: done.append("cleanup"))
        self.assertEqual(done, ["cleanup"])
        self.assertIs(signal.getsignal(signal.SIGTERM), before)              # the previous handler is back


class TestDriversRunPod(Base):
    def test_probe_valid(self):
        fake = FakeRunPod()
        fake.lb = {("GET", "/v1/models"): (200, {"data": [{"id": dv.MODEL}]}, "forwarded"),
                   ("GET", "/version"): (200, {"version": "0.25.1"}, "forwarded"),
                   ("GET", "/metrics"): (200, {}, "forwarded"), ("GET", "/health"): (200, {}, "health"),
                   ("POST", "/v1/chat/completions"): (200, {"choices": []}, "forwarded")}
        code, lines = self.run_quiet(self.driver(fake, "probe-valid", "--name", "mdf-surya-fault").probe_valid)
        self.assertEqual(code, 0, lines)
        fake.lb[("POST", "/v1/chat/completions")] = (200, {}, "error")
        code, _ = self.run_quiet(self.driver(fake, "probe-valid", "--name", "mdf-surya-fault").probe_valid)
        self.assertEqual(code, 6)

    def test_phase_d_a_worker_breaks_g1_and_the_endpoint_is_not_deleted(self):
        # c7.20: deleting is irreversible (prompt of the closure of N4, §4): disarmed again, a stop for Davide
        fake = FakeRunPod(workers=[[], [], [{"id": "w1", "status": "INITIALIZING"}]])
        fake.lb = {("POST", "/v1/chat/completions"): (503, {}, None)}
        code, lines = self.run_quiet(self.driver(fake, "fault-test", "--name", "mdf-surya-fault", "--phase", "D").fault_test)
        self.assertEqual((code, lines[-1]["guarantee"]), (6, "G1"))
        self.assertNotIn("DELETE", [c[0] for c in fake.calls])
        self.assertEqual([c[2] for c in fake.calls if c[0] == "PATCH"][-1]["workers"]["max"], 0)
        self.assertIn({"delete_not_done": "mdf-surya-fault"}, [{k: v for k, v in x.items() if k == "delete_not_done"}
                                                              for x in lines])

    def test_phase_d_a_request_without_an_outcome_breaks_g2(self):
        fake = FakeRunPod(workers=[[]])
        fake.lb = {}
        d = self.driver(fake, "fault-test", "--name", "mdf-surya-fault", "--phase", "D")
        def hold(inf, timeout):                         # a request whose thread ends without a result
            t = threading.Thread(target=lambda: None)
            t.start()
            return t, {}
        with mock.patch.object(dv.Inference, "hold", hold):
            code, lines = self.run_quiet(d.fault_test)
        self.assertEqual((code, lines[-1]["guarantee"]), (6, "G2"))
        self.assertIn("has no outcome", lines[-1]["violation"])

    def test_phase_d_passes_without_workers(self):
        fake = FakeRunPod(workers=[[]])
        fake.lb = {("POST", "/v1/chat/completions"): (503, {}, None)}
        code, lines = self.run_quiet(self.driver(fake, "fault-test", "--name", "mdf-surya-fault", "--phase", "D").fault_test)
        self.assertEqual(code, 0, lines)
        self.assertNotIn("DELETE", [c[0] for c in fake.calls])

    def test_observe_read_only_writes_nothing(self):
        fake = FakeRunPod(workers=[[]])
        ledger = self.dir / "spend.jsonl"
        d = self.driver(fake, "observe", "--name", "mdf-surya-fault", "--closed-utc-hours", "1", "--read-only")
        code, lines = self.run_quiet(d.observe)
        self.assertEqual(code, 0, lines)
        self.assertFalse(ledger.exists())
        self.assertGreaterEqual(lines[-1]["readings"], 60)
        self.assertEqual(lines[-1]["zero_s"], 0.0)                              # the finally: disarm and zero
        self.assertEqual([c[2] for c in fake.calls if c[0] == "PATCH"][-1],
                         {"workers": {"min": 0, "max": 0, "idleTimeout": 60, "initTimeout": 960}})

    def test_observe_with_a_reservation(self):
        fake = FakeRunPod(workers=[[]])
        ledger = self.dir / "spend.jsonl"
        d = self.driver(fake, "observe", "--name", "mdf-surya-fault", "--closed-utc-hours", "1", "--spend-ledger",
                        str(ledger), "--spend-cap-usd", "4", "--cap-usd", "0.05")
        code, lines = self.run_quiet(d.observe)
        self.assertEqual(code, 0)
        self.assertEqual([r["kind"] for r in spend.read(ledger)], ["activity"])

    def test_observe_stops_on_a_worker(self):
        fake = FakeRunPod(workers=[[], [{"id": "w1", "status": "IDLE"}], []])
        d = self.driver(fake, "observe", "--name", "mdf-surya-fault", "--closed-utc-hours", "1", "--read-only")
        code, lines = self.run_quiet(d.observe)
        self.assertEqual((code, lines[-1]["guarantee"]), (6, "G1"))
        self.assertNotIn("DELETE", [c[0] for c in fake.calls])                  # c7.20: never deleted

    def test_observe_a_failed_reading_is_a_fact(self):
        fake = FakeRunPod(workers=[[]])
        failed = []

        def flaky(request):
            if request.url.path.endswith("/workers") and len(failed) < 2:
                failed.append(1)
                return httpx.Response(502, text="bad gateway")
            return fake(request)
        a = rpe.parser().parse_args(["--env-file", str(self.env), "observe", "--name", "mdf-surya-fault",
                                     "--closed-utc-hours", "1", "--read-only"])
        d = rpe.Driver(a, transport=httpx.MockTransport(flaky), clock=self.clock, sleep=self.clock.sleep)
        code, lines = self.run_quiet(d.observe)
        self.assertEqual(code, 0, lines[-1])
        self.assertEqual((len(lines[-1]["read_errors"]), [f["fact"] for f in lines[-1]["facts"]]),
                         (2, ["read_error", "read_error"]))


def wrow(since, worker, source, text, recv="2026-11-19T12:00:00.000Z", ts=""):
    """A line of the worker log of the session, as the follower writes it."""
    return "\t".join([recv, "+%.1f" % since, worker, source, ts, text]) + "\n"


def listing(since, worker, status, running=None):
    """A `worker` line of the follower; `running` (c7.19) the workers RUNNING in that listing."""
    ev = {"event": "worker", "status": status, "gpu": "NVIDIA RTX A5000", "data_center": "EU-RO-1", "uptime": None}
    if running is not None:
        ev["running"] = running
    return wrow(since, worker, "session", json.dumps(ev, sort_keys=True))


def pline(since, worker, event, **fields):
    return wrow(since, worker, "container", json.dumps(dict(event=event, **fields), sort_keys=True))


def proxy_line(since, event, **fields):
    """A line of the worker log of the session with an event of the proxy."""
    return "\t".join(["2026-11-19T10:00:00.000Z", "+%.1f" % since, "w1", "container", "2026-11-19T10:00:00Z",
                      json.dumps(dict(event=event, **fields))]) + "\n"


class InSession(Base):
    """A driver as the session runs it: the arm row in the ledger, MDF_SESSION_ARM and the worker log."""

    def setUp(self):
        super().setUp()
        self.ledger = self.dir / "spend.jsonl"
        self.log = self.dir / "worker-log.partial"
        self.log.write_text("")

    def arm(self, allow_download=True, before=100.0):
        from markdown_for_llms.bench import serverspend
        self.ledger = self.dir / ("spend-%d.jsonl" % len(list(self.dir.glob("spend-*.jsonl"))))  # one window each
        t_arm = self.clock.t - before
        with spend.Ledger.file(self.ledger).locked() as locked:
            row = serverspend.arm(locked, self.ledger, "runpod", "mdf-surya-fault", "https://ep123.api.runpod.ai", "60",
                                  t_arm + 2689, "400", pv.RATES["runpod"], t_arm,
                                  extra={"allow_download": allow_download})
        return row["arm_id"]

    def in_session(self, fake, argv, arm_id=None, transport=None, ledger=True):
        a = rpe.parser().parse_args(["--env-file", str(self.env)] + list(argv) + (
            ["--spend-ledger", str(self.ledger)] if ledger else []))
        environ = {"SESSION_WORKER_LOG": str(self.log)}
        if arm_id:
            environ["MDF_SESSION_ARM"] = arm_id
        return rpe.Driver(a, transport=transport or httpx.MockTransport(fake), clock=self.clock, sleep=self.clock.sleep,
                          environ=environ)


class TestPopulate(InSession):
    """MFL033 c7.4: readiness up to 1800 s from the arm with the download, the fast stop from the
    worker log, `total > 1` and the data center at each reading."""

    def populate(self, fake, arm_id=None, transport=None):
        d = self.in_session(fake, ["populate", "--name", "mdf-surya-fault"], arm_id, transport)
        return self.run_quiet(d.populate)

    def models_after(self, fake, n):
        calls = []

        def answer():
            calls.append(1)
            if len(calls) <= n:
                return httpx.ReadTimeout("held by the balancer")
            return (200, {"data": [{"id": dv.MODEL}]}, "forwarded")
        fake.lb = {("GET", "/v1/models"): answer}
        return calls

    def test_ready_within_the_window_of_the_arm(self):
        fake = FakeRunPod(workers=[[{"id": "w1", "status": "RUNNING", "dataCenterId": "EU-RO-1"}]])
        self.models_after(fake, 50)
        code, lines = self.populate(fake, self.arm())
        self.assertEqual(code, 0, lines)
        res = lines[-1]
        self.assertEqual((res["allow_download"], res["deadline_from_arm_s"]), (True, 1800.0))
        self.assertEqual(res["ready"], {"ready": True, "since_arm_s": 350.0, "since_driver_s": 250.0,
                                        "proxy": "forwarded"})
        self.assertGreaterEqual(res["readings"], 8)                            # a reading every 30 s

    def test_not_ready_at_1800_s_from_the_arm(self):
        fake = FakeRunPod(workers=[[]])
        calls = self.models_after(fake, 10 ** 6)
        code, lines = self.populate(fake, self.arm())
        self.assertEqual(code, 6)
        self.assertEqual(lines[-1]["violation"], "not ready within 1800 s of the arm")
        self.assertEqual(lines[-1]["ready"]["since_arm_s"], 1800.0)
        self.assertEqual(len(calls), 340)                                       # (1800 - 100) / 5

    def test_without_the_download_the_plan_limit(self):
        fake = FakeRunPod(workers=[[]])
        self.models_after(fake, 10 ** 6)
        code, lines = self.populate(fake, self.arm(allow_download=False))
        self.assertEqual(code, 6)
        self.assertEqual((lines[-1]["deadline_from_arm_s"], lines[-1]["ready"]["since_arm_s"]), (1060.0, 1060.0))

    def test_outside_a_session(self):
        fake = FakeRunPod()
        code, lines = self.populate(fake)
        self.assertEqual((code, lines[-1]["code"]), (2, "invalid_setting"))
        self.assertFalse([c for c in fake.calls if ".api.runpod.ai" in c[1]])

    def test_fast_stops_from_the_worker_log(self):
        cases = [
            ([proxy_line(60, "mdf_download_start", mono=100.0, disk=0, rx=1000),
              proxy_line(70, "mdf_download_progress", mono=110.0, disk=5000, rx=900000),
              proxy_line(190, "mdf_download_progress", mono=229.0, disk=5000, rx=900100),
              proxy_line(200, "mdf_download_progress", mono=240.0, disk=5000, rx=900200)], "does not move for 130 s"),
            ([proxy_line(60, "mdf_download_start", mono=100.0, disk=0, rx=0),
              proxy_line(90, "mdf_download_end", mono=130.0, exit=1, timeout=False)], "the download failed: exit 1"),
            ([proxy_line(90, "mdf_weights_check", mono=130.0, phase="after_download", ok=False)], "not verified"),
            (["\t".join(["2026-11-19T10:00:00.000Z", "+400.0", "w1", "container", "",
                         "(EngineCore_DP0 pid=212) ERROR 11-19 10:06:40 [core.py:700] EngineCore failed to start."])
              + "\n"], "an error of vLLM"),
            ([proxy_line(400, "mdf_vllm_exit", mono=440.0, code=1)], "mdf_vllm_exit"),
            ([proxy_line(1500, "mdf_trip", mono=1540.0, rule="W1")], "mdf_trip"),
        ]
        for lines_in, expected in cases:
            self.log.write_text("".join(lines_in))
            fake = FakeRunPod(workers=[[]])
            calls = self.models_after(fake, 10 ** 6)
            code, lines = self.populate(fake, self.arm())
            self.assertEqual(code, 6, lines)
            self.assertIn(expected, lines[-1]["violation"])
            self.assertEqual(calls, [])                                         # stopped before any request

    def test_the_log_read_while_waiting(self):
        fake = FakeRunPod(workers=[[]])
        calls = []

        def answer():
            calls.append(1)
            if len(calls) == 3:
                with open(self.log, "a") as fh:
                    fh.write(proxy_line(80, "mdf_inert", mono=120.0, code="mdf_weights_missing"))
            return httpx.ReadTimeout("held")
        fake.lb = {("GET", "/v1/models"): answer}
        code, lines = self.populate(fake, self.arm())
        self.assertEqual((code, len(calls)), (6, 3))
        self.assertIn("mdf_weights_missing", lines[-1]["violation"])
        self.assertEqual(lines[-1]["proxy_events"]["mdf_inert"]["since_arm"], 80.0)

    def test_progress_by_bytes_received_and_an_info_line(self):
        self.log.write_text("".join([
            proxy_line(60, "mdf_download_start", mono=100.0, disk=0, rx=0),
            proxy_line(200, "mdf_download_progress", mono=240.0, disk=0, rx=5 << 20),     # moving: bytes received
            proxy_line(400, "mdf_download_progress", mono=440.0, disk=1 << 30, rx=1 << 30),
            proxy_line(410, "mdf_download_end", mono=450.0, exit=0, timeout=False),
            "\t".join(["x", "+420.0", "w1", "container", "", "(APIServer pid=1) INFO 11-19 10:07:00 [api.py:1] ok"]) + "\n",
            "\t".join(["x", "+421.0", "w1", "system", "", "ERROR pulling nothing"]) + "\n"]))
        fake = FakeRunPod(workers=[[]])
        self.models_after(fake, 2)
        code, lines = self.populate(fake, self.arm())
        self.assertEqual(code, 0, lines)

    def test_criteria_of_the_readings(self):
        fake = FakeRunPod(workers=[[{"id": "w1", "status": "RUNNING", "dataCenterId": "US-KS-2"}]])
        self.models_after(fake, 10)
        code, lines = self.populate(fake, self.arm())
        self.assertEqual(code, 6)
        self.assertIn("not in ['EU-RO-1']", lines[-1]["violation"])
        fake = FakeRunPod(workers=[[{"id": "w1", "status": "RUNNING"}, {"id": "w2", "status": "INITIALIZING"}]])
        self.models_after(fake, 10)
        code, lines = self.populate(fake, self.arm())
        self.assertEqual(code, 6)
        self.assertIn("total 2 > 1", lines[-1]["violation"])

    def test_a_failed_reading_is_not_a_criterion(self):
        fake = FakeRunPod(workers=[[]])
        self.models_after(fake, 10)
        failed = []

        def flaky(request):
            if request.url.path.endswith("/workers") and not failed:
                failed.append(1)
                return httpx.Response(502, text="bad gateway")
            return fake(request)
        code, lines = self.populate(fake, self.arm(), transport=httpx.MockTransport(flaky))
        self.assertEqual(code, 0, lines)
        self.assertEqual(len(lines[-1]["read_errors"]), 1)


class TestReadiness(InSession):
    """MFL033 c7.5 (prompt N4d §5.3 and §5.5): every readiness wait of the RunPod drivers reads the
    worker log of the session until readiness (the fast stop) and gives the readiness from the arm."""

    VLLM_ERROR = "\t".join(["2026-11-19T10:00:00.000Z", "+392.2", "w1", "container", "",
                            "(EngineCore_DP0 pid=212) ERROR 11-19 09:23:54 [core.py:700] EngineCore failed to start."]) + "\n"

    def valid(self, fake, models_503=3, on_chat=None):
        calls = []

        def models():
            calls.append(1)
            return (503, {}, None) if len(calls) <= models_503 else (200, {"data": [{"id": dv.MODEL}]}, "forwarded")

        def chat():
            if on_chat:
                on_chat()
            return (200, {"choices": []}, "forwarded")
        fake.lb = {("GET", "/v1/models"): models, ("GET", "/version"): (200, {"version": "0.25.1"}, "forwarded"),
                   ("GET", "/metrics"): (200, {}, "forwarded"), ("GET", "/health"): (200, {}, "health"),
                   ("POST", "/v1/chat/completions"): chat}
        return calls

    def test_probe_valid_gives_the_readiness_from_the_arm(self):
        fake = FakeRunPod()
        self.valid(fake)
        d = self.in_session(fake, ["probe-valid", "--name", "mdf-surya-fault"], self.arm(allow_download=False))
        code, lines = self.run_quiet(d.probe_valid)
        self.assertEqual(code, 0, lines)
        ready = lines[-1]["probe_valid"]["ready"]
        self.assertEqual((ready["ready"], ready["seconds"], ready["since_arm_s"], ready["proxy_events"]),
                         (True, 15.0, 115.0, {}))

    def test_probe_valid_stops_on_the_log_before_any_request(self):
        for text, expected in ((self.VLLM_ERROR, "an error of vLLM"),
                               (proxy_line(350, "mdf_inert", mono=10.0, code="mdf_config_invalid"), "mdf_config_invalid"),
                               (proxy_line(395, "mdf_vllm_exit", mono=50.0, code=1), "mdf_vllm_exit")):
            with self.subTest(expected=expected):
                self.log.write_text(text)
                fake = FakeRunPod()
                calls = self.valid(fake)
                d = self.in_session(fake, ["probe-valid", "--name", "mdf-surya-fault"], self.arm(allow_download=False))
                code, lines = self.run_quiet(d.probe_valid)
                self.assertEqual(code, 6)
                ready = lines[-1]["probe_valid"]["ready"]
                self.assertFalse(ready["ready"])
                self.assertIn(expected, ready["stopped"])
                self.assertEqual(calls, [])

    def test_the_log_is_not_read_after_readiness(self):
        fake = FakeRunPod()

        def trip():
            with open(self.log, "a") as fh:
                fh.write(proxy_line(500, "mdf_trip", mono=200.0, rule="W3"))
        self.valid(fake, on_chat=trip)
        d = self.in_session(fake, ["probe-valid", "--name", "mdf-surya-fault"], self.arm(allow_download=False))
        code, lines = self.run_quiet(d.probe_valid)
        self.assertEqual(code, 0, lines)
        self.assertIn("mdf_trip", self.log.read_text())

    def test_phase_a_stops_at_readiness_on_the_log(self):
        self.log.write_text(proxy_line(395, "mdf_vllm_exit", mono=50.0, code=1))
        fake = FakeRunPod(workers=[[]])
        calls = self.valid(fake)
        d = self.in_session(fake, ["fault-test", "--name", "mdf-surya-fault", "--phase", "A"], self.arm(allow_download=False))
        code, lines = self.run_quiet(d.fault_test)
        self.assertEqual(code, 8)                                       # c7.20: the fault not exercised
        self.assertIn("mdf_vllm_exit", lines[-1]["inconclusive"])
        self.assertNotIn("violation", lines[-1])
        self.assertEqual(calls, [])
        self.assertNotIn("POST", [c[0] for c in fake.calls])

    def test_hold_without_a_ledger(self):
        self.log.write_text(self.VLLM_ERROR)
        fake = FakeRunPod()
        self.valid(fake)
        d = self.in_session(fake, ["hold", "--name", "mdf-surya-fault", "--marker", str(self.dir / "m")],
                            self.arm(allow_download=False), ledger=False)
        r = d.ready(d.inference())
        self.assertFalse(r["ready"])
        self.assertIn("an error of vLLM", r["stopped"])
        self.assertNotIn("since_arm_s", r)

    def test_phase_b_reads_the_data_center(self):
        # N4b §2 in every reading of workers (c7.6); since c7.20 a worker outside the data center is a
        # fact (no guarantee broken). The worker goes after the requests: the zero Z1 of B.
        for dc, facts in (("EU-RO-1", []), ("US-KS-2", ["outside_data_center"])):
            with self.subTest(dc=dc):
                w1 = {"id": "w1", "status": "RUNNING", "dataCenterId": dc, "uptimeSeconds": 5}
                fake = FakeRunPod(workers=[[w1]])
                self.valid(fake, on_chat=lambda: setattr(fake, "workers", [[]]))
                d = self.in_session(fake, ["fault-test", "--name", "mdf-surya-fault", "--phase", "B",
                                           "--observe-seconds", "30"], self.arm(allow_download=False))
                code, lines = self.run_quiet(d.fault_test)
                self.assertEqual(code, 0, lines[-1])
                self.assertEqual([f["fact"] for f in lines[-1]["facts"]], facts)
                if facts:
                    self.assertEqual(lines[-1]["facts"][0]["allowed"], ["EU-RO-1"])


class TestReadinessWorkers(InSession):
    """MFL033 c7.7 (prompt N4f §3.4): every readiness wait of the RunPod drivers (`probe-valid`, phases
    A and B, `hold`) reads the workers of the endpoint every 30 s, with GETs of 30 s, and stops at once
    on `total > 1`, a worker unhealthy, replaced or outside the data center; a failed reading is not a
    criterion. `probe-valid` keeps reading them for 30 s after the chat (on 19/11 a second worker came
    1.3 s after readiness)."""

    W1 = {"id": "w1", "status": "INITIALIZING", "gpuTypeId": "NVIDIA RTX A5000", "dataCenterId": "EU-RO-1"}

    def worker(self, **changes):
        return dict(self.W1, **changes)

    def balancer(self, fake, held=0, on_chat=None):
        """The balancer holds the first `held` GETs of /v1/models until their timeout, then answers."""
        calls, timeouts = [], []

        def models():
            calls.append(1)
            if len(calls) <= held:
                self.clock.t += 30.0
                return httpx.ReadTimeout("held by the balancer")
            return (200, {"data": [{"id": dv.MODEL}]}, "forwarded")

        def chat():
            if on_chat:
                on_chat()
            return (200, {"choices": []}, "forwarded")
        fake.lb = {("GET", "/v1/models"): models, ("GET", "/version"): (200, {"version": "0.25.1"}, "forwarded"),
                   ("GET", "/metrics"): (200, {}, "forwarded"), ("GET", "/health"): (200, {}, "health"),
                   ("GET", "/ping"): (200, {}, "health"), ("POST", "/v1/chat/completions"): chat}

        def transport(request):
            if request.url.path == "/v1/models":
                timeouts.append(request.extensions["timeout"]["read"])
            return fake(request)
        return calls, timeouts, httpx.MockTransport(transport)

    def run_driver(self, fake, argv, transport):
        d = self.in_session(fake, argv, self.arm(allow_download=False), transport=transport)
        return self.run_quiet({"probe-valid": d.probe_valid, "fault-test": d.fault_test}[argv[0]])

    def test_gets_of_30_s_and_a_reading_every_30_s(self):
        fake = FakeRunPod(workers=[[self.worker()], [self.worker(status="RUNNING")]])
        calls, timeouts, transport = self.balancer(fake, held=10)
        code, lines = self.run_driver(fake, ["probe-valid", "--name", "mdf-surya-fault"], transport)
        self.assertEqual(code, 0, lines)
        res = lines[-1]
        ready = res["probe_valid"]["ready"]
        self.assertEqual((ready["ready"], ready["seconds"], ready["readings"], ready["workers"]),
                         (True, 350.0, 11, ["w1"]))                            # 10 x (30 s held + 5 s)
        self.assertEqual(set(timeouts), {30.0})
        self.assertEqual(len(calls), 11)
        self.assertEqual((res["pass"], res["violation"], res["workers"], res["max_total"]), (True, None, ["w1"], 1))
        self.assertGreaterEqual(res["readings"], ready["readings"] + 3)        # 30 s after the chat
        self.assertEqual(res["probe_valid"]["ping"]["proxy"], "health")

    DRIVERS = (["probe-valid", "--name", "mdf-surya-fault"],
               ["fault-test", "--name", "mdf-surya-fault", "--phase", "A"],
               ["fault-test", "--name", "mdf-surya-fault", "--phase", "B", "--observe-seconds", "30"])

    def test_a_criterion_stops_the_wait_before_readiness(self):
        # c7.19: two workers RUNNING (a second one INITIALIZING is an extra worker, admitted); c7.20: G1
        running = self.worker(status="RUNNING")
        workers = [[self.worker()], [running, self.worker(id="w2", status="RUNNING")]]
        for argv in self.DRIVERS:
            with self.subTest(driver=" ".join(argv[:1] + argv[3:5])):
                fake = FakeRunPod(workers=workers)
                calls, _, transport = self.balancer(fake, held=10 ** 6)
                code, lines = self.run_driver(fake, argv, transport)
                self.assertEqual(code, 6, lines)
                res = lines[-1]
                ready = res["probe_valid"]["ready"] if argv[0] == "probe-valid" else res["ready"]
                self.assertFalse(ready["ready"])
                self.assertIn("two workers RUNNING at once", ready["stopped"])
                self.assertEqual(ready["guarantee"], "G1")
                if argv[0] == "fault-test":
                    self.assertEqual(res["guarantee"], "G1")
                self.assertLessEqual(len(calls), len(workers))                # at once, at the reading
                self.assertNotIn("POST", [c[0] for c in fake.calls])

    def test_what_breaks_no_guarantee_is_a_fact_during_readiness(self):
        # c7.20: unhealthy, a replacement the log (empty here) does not know, a worker outside the data
        # center stop nothing; the wait goes on to readiness (the balancer answers after 10 GETs)
        running = self.worker(status="RUNNING")
        for name, workers, fact in (
                ("unhealthy", [[self.worker()], [self.worker(status="UNHEALTHY")], [running]], "unhealthy"),
                ("replaced", [[running], [], [self.worker(id="w2", status="RUNNING")]], "replacement_not_judged"),
                ("outside the data center", [[self.worker(dataCenterId="US-KS-2", status="RUNNING")]],
                 "outside_data_center")):
            with self.subTest(name=name):
                fake = FakeRunPod(workers=workers)
                calls, _, transport = self.balancer(fake, held=10)
                code, lines = self.run_driver(fake, self.DRIVERS[0], transport)
                res = lines[-1]
                self.assertEqual(code, 0, res)
                self.assertTrue(res["probe_valid"]["ready"]["ready"])
                self.assertIn(fact, [f["fact"] for f in res["facts"]])
                self.assertIn(fact, [x.get("fact") for x in lines])               # a line at once
                self.assertEqual(len(calls), 11)

    def test_a_failed_reading_is_not_a_criterion(self):
        fake = FakeRunPod(workers=[[self.worker()]])
        _, _, transport = self.balancer(fake, held=3)
        failed = []

        def flaky(request):
            if request.url.path.endswith("/workers") and not failed:
                failed.append(1)
                return httpx.Response(502, text="bad gateway")
            return transport.handle_request(request)
        code, lines = self.run_driver(fake, ["probe-valid", "--name", "mdf-surya-fault"], httpx.MockTransport(flaky))
        self.assertEqual(code, 0, lines)
        self.assertEqual(len(lines[-1]["probe_valid"]["ready"]["read_errors"]), 1)

    def test_probe_valid_reads_the_workers_after_the_chat(self):
        for second_status, code_expected in (("RUNNING", 6), ("INITIALIZING", 0)):
            with self.subTest(second=second_status):
                fake = FakeRunPod(workers=[[self.worker(status="RUNNING")]])

                def second():
                    fake.workers = [[self.worker(status="RUNNING"), self.worker(id="w2", status=second_status)]]
                _, _, transport = self.balancer(fake, held=2, on_chat=second)
                code, lines = self.run_driver(fake, ["probe-valid", "--name", "mdf-surya-fault"], transport)
                self.assertEqual(code, code_expected, lines)
                res = lines[-1]
                self.assertTrue(res["probe_valid"]["ready"]["ready"])
                self.assertEqual(res["probe_valid"]["chat"]["status"], 200)
                if code_expected:                      # c7.19: two RUNNING at once, the fast stop
                    self.assertIn("two workers RUNNING at once", res["violation"])
                    self.assertFalse(res["pass"])
                else:                                  # an extra worker, recorded (the readings only: empty log)
                    self.assertEqual((res["violation"], res["pass"], res["max_total"], res["max_running"]),
                                     (None, True, 2, 1))
                    self.assertEqual([(x["id"], x["states"]) for x in res["extra_workers_readings"]],
                                     [("w2", ["INITIALIZING"])])

    def test_phase_b_keeps_the_readings_of_the_readiness(self):
        fake = FakeRunPod(workers=[[self.worker(status="RUNNING")]])
        self.log.write_text(listing(10.0, "w1", "RUNNING", ["w1"]))

        def replaced():                                   # the crash of B: a worker after the first
            fake.workers = [[self.worker(id="w2", status="RUNNING")]] * 3 + [[self.worker(id="w2", status="IDLE")]]
            with self.log.open("a") as fh:                # c7.19: the log of the session names it too
                fh.write(listing(171.0, "w1", "GONE", []) + listing(181.0, "w2", "RUNNING", ["w2"])
                         + listing(185.0, "w2", "IDLE", []) + wrow(186.0, "w2", "system", "stop container c2"))
        _, _, transport = self.balancer(fake, held=2, on_chat=replaced)
        code, lines = self.run_driver(fake, ["fault-test", "--name", "mdf-surya-fault", "--phase", "B",
                                             "--observe-seconds", "30"], transport)
        self.assertEqual(code, 0, lines[-1])
        self.assertEqual((lines[-1]["worker_ids"], lines[-1]["restarts"]), (["w1", "w2"], 1))
        self.assertEqual([(r["old"]["id"], r["new"]["id"]) for r in lines[-1]["restarts_after_fault"]], [("w1", "w2")])

    def test_hold_stops_on_two_workers(self):
        fake = FakeRunPod(workers=[[self.worker(status="RUNNING"), self.worker(id="w2", status="RUNNING")]])
        calls, _, transport = self.balancer(fake, held=10 ** 6)
        d = self.in_session(fake, ["hold", "--name", "mdf-surya-fault", "--marker", str(self.dir / "m")],
                            self.arm(allow_download=False), transport=transport, ledger=False)
        r = d.ready(d.inference())
        self.assertFalse(r["ready"])
        self.assertIn("two workers RUNNING at once", r["stopped"])             # c7.19
        self.assertEqual(calls, [])


class NotHandled(Exception):
    """The signal reached the test's own handler: the driver did not handle it."""


def fallback_handler(test, sig):
    """A handler of the test for `sig` while it runs: without it a driver that does not handle the
    signal would end the whole test run."""
    def handler(signum, frame):
        raise NotHandled(signal.Signals(signum).name)
    old = signal.signal(sig, handler)
    test.addCleanup(signal.signal, sig, old)
    return handler


class TestEdgeRefusal(InSession):
    """MFL033 c7.13, as on Modal: in every readiness wait of the RunPod drivers (the populate's too) a
    401, 403 or 404 without `X-MDF-Proxy` is the edge refusing the key; the wait stops at the first
    one, with the status, the header names and the start of the body, never the key. Each attempt is
    a line as soon as it happens; SIGTERM and SIGINT print the result so far."""

    W1 = {"id": "w1", "status": "RUNNING", "dataCenterId": "EU-RO-1"}

    def refusing(self, fake, status):
        calls = []

        def handle(request):
            if request.url.path == "/v1/models":
                calls.append(1)
                return httpx.Response(status, json={"error": "denied for %s" % request.headers.get("authorization")},
                                      headers={"x-edge-value": "header-value-xyz"})
            return fake(request)
        return calls, httpx.MockTransport(handle)

    def test_a_refusal_of_the_edge_stops_every_wait_at_once(self):
        for status in (401, 403, 404):
            for argv in (["probe-valid"], ["fault-test", "--phase", "A"], ["fault-test", "--phase", "B"], ["populate"],
                         ["hold", "--marker", str(self.dir / "m")]):
                with self.subTest(status=status, driver=" ".join(argv[:1] + argv[2:3])):
                    fake = FakeRunPod(workers=[[self.W1]])
                    calls, transport = self.refusing(fake, status)
                    d = self.in_session(fake, argv[:1] + ["--name", "mdf-surya-fault"] + argv[1:],
                                        self.arm(allow_download=argv[0] == "populate"), transport=transport,
                                        ledger=argv[0] != "hold")
                    out = io.StringIO()
                    with redirect_stdout(out):
                        if argv[0] == "hold":
                            res, code = {"ready": d.ready(d.inference())}, 6
                        else:
                            code = {"probe-valid": d.probe_valid, "fault-test": d.fault_test, "populate": d.populate}[argv[0]]()
                            res = json.loads(out.getvalue().splitlines()[-1])
                    # c7.20: phases A and B that never reach readiness are inconclusive (exit 8)
                    self.assertEqual((code, len(calls)), (8 if argv[0] == "fault-test" else 6, 1))
                    self.assertNotIn("POST", [c[0] for c in fake.calls])
                    expected = "refused at the edge: HTTP %d" % status
                    if argv[0] == "populate":
                        self.assertEqual((res["violation"], res["ready"]["ready"], res["tries"]), (expected, False, 1))
                        edge = res["edge"]
                    else:
                        ready = res["probe_valid"]["ready"] if argv[0] == "probe-valid" else res["ready"]
                        self.assertEqual((ready["ready"], ready["stopped"], ready["tries"]), (False, expected, 1))
                        edge = ready["edge"]
                    self.assertEqual((edge["status"], edge["content_type"]), (status, "application/json"))
                    self.assertIn("x-edge-value", edge["header_names"])
                    self.assertEqual(json.loads(edge["body_start"]), {"error": "denied for [key]"})
                    tries = [json.loads(x) for x in out.getvalue().splitlines() if '"ready_try"' in x]
                    self.assertEqual([(x["ready_try"], x["status"]) for x in tries], [(1, status)])
                    self.assertNotIn(KEY, out.getvalue())
                    self.assertNotIn("header-value-xyz", out.getvalue())

    def test_the_populate_writes_each_attempt_at_once(self):
        fake = FakeRunPod(workers=[[self.W1]])
        out, seen, calls = io.StringIO(), [], []

        def models():
            seen.append(sum(1 for x in out.getvalue().splitlines() if '"ready_try"' in x))
            calls.append(1)
            return httpx.ReadTimeout("held") if len(calls) <= 2 else (200, {"data": [{"id": dv.MODEL}]}, "forwarded")
        fake.lb = {("GET", "/v1/models"): models}
        d = self.in_session(fake, ["populate", "--name", "mdf-surya-fault"], self.arm())
        with redirect_stdout(out):
            code = d.populate()
        self.assertEqual(code, 0)
        self.assertEqual(seen, [0, 1, 2])
        self.assertEqual(json.loads(out.getvalue().splitlines()[-1])["tries"], 3)

    def test_sigterm_or_sigint_prints_the_result_so_far(self):
        for sig in (signal.SIGTERM, signal.SIGINT):
            with self.subTest(signal=sig.name):
                fake = FakeRunPod(workers=[[self.W1]])
                calls = []

                def models():
                    calls.append(1)
                    if len(calls) == 3:
                        os.kill(os.getpid(), sig)                       # the session stops the driver
                        time.sleep(5)
                    return httpx.ReadTimeout("held by the balancer")
                fake.lb = {("GET", "/v1/models"): models}
                argv = ["--env-file", str(self.env), "probe-valid", "--name", "mdf-surya-fault", "--spend-ledger",
                        str(self.ledger)]
                arm_id = self.arm(allow_download=False)
                argv[-1] = str(self.ledger)
                before = fallback_handler(self, sig)          # a driver without its own handler fails, not pytest
                out = io.StringIO()
                with redirect_stdout(out):
                    code = rpe.main(argv, transport=httpx.MockTransport(fake), clock=self.clock, sleep=self.clock.sleep,
                                    environ={"SESSION_WORKER_LOG": str(self.log), "MDF_SESSION_ARM": arm_id})
                lines = [json.loads(x) for x in out.getvalue().splitlines()]
                self.assertEqual(code, 130)
                self.assertEqual([x["ready_try"] for x in lines if "ready_try" in x], [1, 2])
                self.assertEqual((lines[-1]["stopped_by"], lines[-1]["partial"]["ready_wait"]["tries"]), (sig.name, 2))
                self.assertEqual(lines[-1]["partial"]["workers"], ["w1"])
                self.assertIs(signal.getsignal(sig), before)
                self.assertNotIn(KEY, out.getvalue())

    def test_a_driver_without_a_signal_keeps_its_exit(self):
        fake = FakeRunPod(workers=[[]])
        fake.lb = {("POST", "/v1/chat/completions"): (503, {}, None)}
        out = io.StringIO()
        with redirect_stdout(out):
            code = rpe.main(["--env-file", str(self.env), "fault-test", "--name", "mdf-surya-fault", "--phase", "D"],
                            transport=httpx.MockTransport(fake), clock=self.clock, sleep=self.clock.sleep)
        self.assertEqual(code, 0)
        self.assertNotIn("stopped_by", out.getvalue())


def request_line(since, kind, status, method="GET", path="/v1/models", code=None):
    """An `mdf_request` line of the proxy in the worker log (c7.6)."""
    fields = dict(kind=kind, method=method, path=path, status=status, t=1795083980.0, mono=12.0)
    if code:
        fields["code"] = code
    return proxy_line(since, "mdf_request", **fields)


class TestProbeNegative(InSession):
    """MFL033 c7.6 (prompt N4e §4.1). The arm alone starts a worker on the load-balanced endpoint (H2,
    window 20261119T110021Z-0b5aa77b of 19/11), so `probe-negative` passes with 401 of the edge for
    both probes, no request answered by the proxy in the worker log, and at most one worker, in the
    data center of the endpoint, never unhealthy nor replaced; the zero after the disarm is the
    session's. The first violation stops at once."""

    INIT = {"id": "w1", "status": "INITIALIZING", "gpuTypeId": "NVIDIA RTX A5000", "dataCenterId": "EU-RO-1"}

    def worker(self, **changes):
        return dict(self.INIT, **changes)

    def probe(self, fake, transport=None, environ=None):
        d = self.in_session(fake, ["probe-negative", "--name", "mdf-surya-fault"], self.arm(allow_download=False),
                            transport=transport)
        if environ is not None:
            d.environ = environ
        return self.run_quiet(d.probe_negative)

    def edge(self, fake, answer):
        """The balancer of `fake`, but `answer(authorization)` -> (status, X-MDF-Proxy) for the probes."""
        def transport(request):
            auth = request.headers.get("authorization")
            if ".api.runpod.ai" in str(request.url) and auth != "Bearer " + KEY:
                status, proxy = answer(auth)
                if status is not None:
                    fake.calls.append((request.method, str(request.url), None, auth))
                    return httpx.Response(status, json={}, headers={"x-mdf-proxy": proxy} if proxy else {})
            return fake(request)
        return httpx.MockTransport(transport)

    def test_one_worker_from_the_arm_passes(self):
        fake = FakeRunPod(workers=[[self.worker()]])
        t0 = self.clock.t
        code, lines = self.probe(fake)
        self.assertEqual(code, 0, lines)
        r = lines[-1]
        refused = {"status": 401, "proxy": None, "error": None}
        self.assertEqual(r["probe_negative"], {"no_key": refused, "wrong_key": refused})
        self.assertEqual((r["violation"], r["workers"], r["max_total"], r["proxy_requests"]), (None, ["w1"], 1, []))
        self.assertGreaterEqual(len(r["readings"]), 12)
        self.assertGreaterEqual(self.clock.t - t0, 120)
        # two requests to the endpoint URL, neither with the key
        self.assertEqual([c[3] for c in fake.calls if ".api.runpod.ai" in c[1]], [None, "Bearer mdf-wrong-key"])

    def test_no_worker_or_one_that_goes_passes(self):
        for workers in ([[]], [[], [self.worker()], [self.worker(status="RUNNING")], []]):
            with self.subTest(workers=workers):
                code, lines = self.probe(FakeRunPod(workers=workers))
                self.assertEqual(code, 0, lines)
                self.assertLessEqual(lines[-1]["max_total"], 1)

    def test_lines_that_are_not_requests_pass(self):
        session = "\t".join(["2026-11-19T10:00:00.000Z", "+3.0", "w1", "session", "",
                             json.dumps({"event": "mdf_request", "kind": "forwarded"})]) + "\n"
        self.log.write_text(proxy_line(5, "mdf_start", mono=1.0) + session
                            + proxy_line(6, "mdf_request", kind="health", status=200, mono=2.0)
                            + "2026-11-19T10:00:00.000Z\t+4.8\tw1\tsystem\t\tloading container image from cache\n")
        code, lines = self.probe(FakeRunPod(workers=[[self.worker()]]))
        self.assertEqual(code, 0, lines)
        self.assertEqual(lines[-1]["proxy_requests"], [])

    def test_not_a_401_of_the_edge(self):
        for name, answer in (("403 to the wrong key", lambda auth: (403, None) if auth else (None, None)),
                             ("401 of the proxy", lambda auth: (401, "error")),
                             ("200 without a key", lambda auth: (200, "forwarded") if auth is None else (None, None))):
            with self.subTest(name=name):
                fake = FakeRunPod(workers=[[]])
                code, lines = self.probe(fake, transport=self.edge(fake, answer))
                self.assertEqual(code, 6)
                self.assertIn("not a 401 of the edge", lines[-1]["violation"])
                self.assertEqual(lines[-1]["readings"], [])          # stopped before any reading

    def test_workers_criteria(self):
        for name, workers, expected in (
                ("two workers", [[self.worker(), self.worker(id="w2")]], "total 2 > 1"),
                ("a replaced worker", [[self.worker()], [], [self.worker(id="w2")]], "a restart"),
                ("outside the data center", [[self.worker(dataCenterId="US-KS-2")]], "not in ['EU-RO-1']"),
                ("unhealthy", [[self.worker(status="UNHEALTHY")]], "unhealthy")):
            with self.subTest(name=name):
                code, lines = self.probe(FakeRunPod(workers=workers))
                self.assertEqual(code, 6)
                self.assertIn(expected, lines[-1]["violation"])

    def test_a_request_answered_by_the_proxy(self):
        for line in (request_line(3.1, "forwarded", 200), request_line(3.2, "error", 503, code="mdf_inert"),
                     request_line(3.3, "forwarded", 401, method="HEAD", path="other")):
            with self.subTest(line=line):
                self.log.write_text(line)
                code, lines = self.probe(FakeRunPod(workers=[[self.worker()]]))
                self.assertEqual(code, 6)
                self.assertIn("a request answered by the proxy", lines[-1]["violation"])
                self.assertEqual(len(lines[-1]["proxy_requests"]), 1)
                self.assertEqual(len(lines[-1]["readings"]), 1)     # the fast stop at the first reading

    def test_a_request_line_during_the_observation_stops_at_once(self):
        fake = FakeRunPod(workers=[[self.worker()]])
        reads = []

        def transport(request):
            if str(request.url).endswith("/workers"):
                reads.append(1)
                if len(reads) == 3:
                    with open(self.log, "a") as fh:
                        fh.write(request_line(25.0, "forwarded", 200))
            return fake(request)
        code, lines = self.probe(fake, transport=httpx.MockTransport(transport))
        self.assertEqual(code, 6)
        self.assertEqual(len(lines[-1]["readings"]), 3)
        self.assertEqual(lines[-1]["proxy_requests"][0]["since_arm"], 25.0)

    def test_without_the_worker_log_it_cannot_pass(self):
        for name, environ, expected in (("no log", {}, "no worker log of the session"),
                                        ("missing file", {"SESSION_WORKER_LOG": "/nonexistent/worker-log.partial"},
                                         "cannot be read")):
            with self.subTest(name=name):
                code, lines = self.probe(FakeRunPod(workers=[[]]), environ=environ)
                self.assertEqual(code, 6)
                self.assertIn(expected, lines[-1]["violation"])


class TestModal(unittest.TestCase):
    TOKEN = json.dumps([{"name": "mdf-surya-proxy", "token_id": "wk-the-proxy-token", "scoped": False}])

    def env_file(self):
        """The env file with the proxy token that the deploy looks for in the workspace (c7.13)."""
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp, True)
        path = tmp / ".env"
        path.write_text("MODAL_PROXY_TOKEN_ID=wk-the-proxy-token\n")
        path.chmod(0o600)
        return str(path)

    def runner(self, answers):
        calls = []

        def run(argv, **kw):
            calls.append(argv)
            key = " ".join(argv[len(pv.MODAL_SDK):len(pv.MODAL_SDK) + 2])
            code, out = answers.get(key, (0, "[]"))
            return type("R", (), {"returncode": code, "stdout": out, "stderr": ""})()
        return run, calls

    def test_state_arm_disarm(self):
        apps = json.dumps([{"App ID": "ap-1", "Description": "mdf-surya", "State": "deployed"},
                           {"App ID": "ap-2", "Description": "other", "State": "deployed"}])
        conts = json.dumps([{"Container ID": "ta-1", "App ID": "ap-1"}, {"Container ID": "ta-2", "App ID": "ap-2"}])
        run, calls = self.runner({"app list": (0, apps), "container list": (0, conts),
                                  "volume list": (0, json.dumps([{"name": pv.VOLUME}])),
                                  "workspace proxy-tokens": (0, self.TOKEN),
                                  "deploy " + str(pv.HERE / "modal_app.py"):
                                  (0, "Created web function serve => https://ws--mdf-surya-serve.modal.run\n")})
        m = pv.Modal("mdf-surya", runner=run, environ={"PATH": "/usr/bin", "RUNPOD_API_KEY": "x"}, env_file=self.env_file())
        s = m.state()
        self.assertEqual((s["armed"], s["busy"], s["containers"]), (True, True, ["ta-1"]))
        self.assertEqual(m.arm("hang-after-first"), {"url": "https://ws--mdf-surya-serve.modal.run", "volume": pv.VOLUME,
                                                     "proxy_token": {"proxy_tokens": 1, "in_workspace": True,
                                                                     "scoped": False}})
        self.assertEqual(m.disarm(), None)
        self.assertEqual(calls[-1][len(pv.MODAL_SDK):], ["app", "stop", "mdf-surya", "--yes"])
        self.assertTrue(all(c[:len(pv.MODAL_SDK)] == pv.MODAL_SDK for c in calls))

    def test_deploy_env_carries_the_fault_and_no_key(self):
        seen = {}

        def run(argv, **kw):
            seen.update(kw.get("env") or {})
            out = (json.dumps([{"name": pv.VOLUME}]) if "volume" in argv else self.TOKEN if "proxy-tokens" in argv
                   else "https://ws--mdf-surya-fault-serve.modal.run")
            return type("R", (), {"returncode": 0, "stdout": out, "stderr": ""})()
        m = pv.Modal("mdf-surya-fault", runner=run, environ={"PATH": "/usr/bin", "MODAL_PROXY_TOKEN_SECRET": "s"},
                     env_file=self.env_file())
        m.arm("crash-after-first")
        self.assertEqual((seen["MDF_FAULT"], seen["MDF_MODAL_APP"]), ("crash-after-first", "mdf-surya-fault"))
        self.assertNotIn("MODAL_PROXY_TOKEN_SECRET", seen)

    def test_stopped_app_and_no_container_is_zero(self):
        apps = json.dumps([{"App ID": "ap-1", "Description": "mdf-surya", "State": "stopped"}])
        run, _ = self.runner({"app list": (0, apps), "container list": (0, "[]")})
        self.assertTrue(pv.Modal("mdf-surya", runner=run, environ={}).zero()[0])


if __name__ == "__main__":
    unittest.main()


# The windows of 19/11 where RunPod listed a second worker (MFL033 N4f, N4r): the times since the arm,
# the states and the system lines of the worker logs 20261119T121138Z-fc3106de (phase A on
# `mdf-surya-fault`) and 20261119T112148Z-250a0406 (`probe-valid` on `mdf-surya`); the worker,
# container, image and volume ids replaced, the other lines of the container left out. The listing
# lines of 19/11 have no `running` field (c7.19 adds it).
PHASE_A_19_11 = {
    "start": [listing(2.6, "w1", "INITIALIZING"), wrow(7.7, "w1", "system", "loading container image from cache"),
              listing(120.7, "w1", "THROTTLED"), listing(141.9, "w1", "INITIALIZING"), listing(184.3, "w1", "THROTTLED"),
              listing(195.1, "w1", "INITIALIZING"), listing(206.9, "w1", "THROTTLED"), listing(217.5, "w1", "INITIALIZING"),
              wrow(318.6, "w1", "system", "Loaded image ID: sha256:img"), listing(324.2, "w1", "RUNNING"),
              wrow(325.8, "w1", "system", "create container img"),
              wrow(326.6, "w1", "system", "Network volume vol mounted at /runpod-volume (read-write)"),
              pline(338.2, "w1", "mdf_start", fault="hang-after-first")],
    "ready": [pline(522.4, "w1", "mdf_vllm_first_health", status=200), pline(522.4, "w1", "mdf_vllm_first_models", status=200),
              pline(522.4, "w1", "mdf_ready"),
              pline(523.7, "w1", "mdf_request", kind="forwarded", method="GET", path="/v1/models", status=200)],
    "chat": [pline(524.1, "w1", "mdf_request", kind="forwarded", method="POST", path="/v1/chat/completions", status=200),
             pline(524.1, "w1", "mdf_fault", fault="hang-after-first"),
             listing(526.5, "w2", "INITIALIZING"), wrow(527.3, "w2", "system", "worker is ready"),
             wrow(540.3, "w1", "system", "stop container c1")],
    "ready_s": 523.4,
}
PROBE_VALID_19_11 = {
    "start": [listing(2.4, "w1", "INITIALIZING"), wrow(406.1, "w1", "system", "Loaded image ID: sha256:img"),
              listing(407.9, "w1", "IDLE"), wrow(414.4, "w1", "system", "create container img"),
              pline(415.9, "w1", "mdf_start"), listing(418.6, "w1", "RUNNING")],
    "ready": [pline(599.6, "w1", "mdf_request", kind="forwarded", method="GET", path="other", status=200),
              pline(600.1, "w1", "mdf_request", kind="forwarded", method="GET", path="/v1/models", status=200)],
    "chat": [pline(601.0, "w1", "mdf_ready"), listing(601.3, "w2", "INITIALIZING"),
             wrow(602.3, "w2", "system", "worker is ready"),
             pline(602.4, "w1", "mdf_request", kind="forwarded", method="POST", path="/v1/chat/completions", status=200)],
    "ready_s": 600.0,
}


class TestExtraWorkers(InSession):
    """MFL033 c7.19 (prompt N4s §4.1, road S1 of Davide, 20/11): RunPod's extra workers. A second
    worker INITIALIZING, IDLE or THROTTLED is recorded and does not stop the phase; two workers
    RUNNING at once, or a worker RUNNING after another one after readiness (a restart), stop it. Rule
    (a) on RunPod: a replacement before the first answer of vLLM, without overlap and without an
    inference, is a fact. The readings and the log of the session judge alike."""

    W1 = {"id": "w1", "status": "RUNNING", "gpuTypeId": "NVIDIA RTX A5000", "dataCenterId": "EU-RO-1"}
    HELD = 2                                        # readiness at +70 s of the driver: two GETs held (30 s + 5 s)

    def w(self, wid="w1", status="RUNNING"):
        return dict(self.W1, id=wid, status=status)

    def append(self, rows):
        with self.log.open("a") as fh:
            fh.write("".join(rows))

    def scenario(self, argv, ready_s, start=(), ready=(), chat=(), workers=None, after=None, later=None):
        """A driver in a session whose readiness comes at `ready_s` s since the arm: `start` in the log
        at the start, `ready` appended with the answer of readiness, `chat` at the first chat; the
        workers read are `workers` until the first chat, then `after`. Later chats (phase A: the held
        one and the abandoned one) get `later(n)`."""
        arm_id = self.arm(allow_download=False, before=ready_s - self.HELD * 35)
        self.append(start)
        fake = FakeRunPod(workers=workers or [[self.w()]])
        calls, chats = [], []

        def models():
            calls.append(1)
            if len(calls) <= self.HELD:
                self.clock.t += 30.0
                return httpx.ReadTimeout("held by the balancer")
            if len(calls) == self.HELD + 1:
                self.append(ready)
            return (200, {"data": [{"id": dv.MODEL}]}, "forwarded")

        def answer_chat():
            chats.append(1)
            if len(chats) == 1:
                self.append(chat)
                if after is not None:
                    fake.workers = [list(x) for x in after]
                return (200, {"choices": []}, "forwarded")
            return later(len(chats)) if later else (200, {"choices": []}, "forwarded")
        fake.lb = {("GET", "/v1/models"): models, ("GET", "/version"): (200, {"version": "0.25.1"}, "forwarded"),
                   ("GET", "/metrics"): (200, {}, "forwarded"), ("GET", "/health"): (200, {}, "health"),
                   ("GET", "/ping"): (200, {}, "health"), ("POST", "/v1/chat/completions"): answer_chat}
        d = self.in_session(fake, argv, arm_id)
        return self.run_quiet({"probe-valid": d.probe_valid, "fault-test": d.fault_test}[argv[0]])

    PROBE = ["probe-valid", "--name", "mdf-surya-fault"]
    A = ["fault-test", "--name", "mdf-surya-fault", "--phase", "A"]
    B = ["fault-test", "--name", "mdf-surya-fault", "--phase", "B", "--observe-seconds", "30"]

    @staticmethod
    def tripped(n):
        return (503, {"error": "mdf_tripped"}, "error")

    # --- the windows of 19/11 -------------------------------------------------------------------
    def test_phase_a_of_19_11_passes_with_its_extra_worker(self):
        t = PHASE_A_19_11
        # after the hang the worker that ran goes; the extra worker stays listed: the zero of A (c7.20:
        # the continuation of the log names the GONE of the worker that ran, as the follower would)
        code, lines = self.scenario(self.A, t["ready_s"], t["start"], t["ready"], t["chat"] + [listing(541.0, "w1", "GONE")],
                                    after=[[self.w(), self.w("w2", "INITIALIZING")]] * 2 + [[self.w("w2", "IDLE")]],
                                    later=self.tripped)
        res = lines[-1]
        self.assertEqual(code, 0, res.get("violation"))
        self.assertTrue(res["pass"])
        self.assertEqual(res["ready"]["since_arm_s"], 523.4)
        self.assertEqual(res["ready"]["served"], "w1")
        self.assertEqual([(x["id"], x["states"], x["running_s"], x["system"][0]["line"]) for x in res["extra_workers"]],
                         [("w2", [[526.5, "INITIALIZING"]], None, "worker is ready")])
        self.assertEqual([x["id"] for x in res["extra_workers_readings"]], ["w2"])
        self.assertEqual((res["restarts"], res["replacements"], res["restarts_after_fault"]), (0, [], []))
        self.assertIn({"extra_worker": 1}, [{k: v for k, v in x.items() if k == "extra_worker"} for x in lines])
        self.assertIsNotNone(res["zero_since_hang_s"])

    def test_phase_a_the_worker_that_ran_still_listed(self):
        """c7.20, zero Z1 (Davide, 20/11 19:44): the worker that ran listed IDLE with its container
        stopped is the zero; without the "stop container" line it is not (G1)."""
        t = PHASE_A_19_11
        after = [[self.w(), self.w("w2", "INITIALIZING")]] * 2 + [[self.w("w1", "IDLE"), self.w("w2", "INITIALIZING")]]
        for name, chat, expected in (("stopped", t["chat"], 0),
                                     ("not stopped", [x for x in t["chat"] if "stop container" not in x], 6)):
            with self.subTest(name):
                self.log.write_text("")
                code, lines = self.scenario(self.A, t["ready_s"], t["start"], t["ready"],
                                            chat + [listing(530.0, "w1", "IDLE")], after=after, later=self.tripped)
                self.assertEqual(code, expected, lines[-1])
                if expected:
                    self.assertEqual(lines[-1]["guarantee"], "G1")
                    self.assertIn("no zero within", lines[-1]["violation"])
                    self.assertIn('no "stop container" in the log for [\'w1\']', lines[-1]["violation"])
                else:
                    self.assertTrue(lines[-1]["pass"])
                    self.assertIsNotNone(lines[-1]["zero_since_hang_s"])

    def test_probe_valid_of_19_11_passes_with_its_extra_worker(self):
        t = PROBE_VALID_19_11
        code, lines = self.scenario(self.PROBE, t["ready_s"], t["start"], t["ready"], t["chat"],
                                    workers=[[self.w("w1", "IDLE")], [self.w()]],
                                    after=[[self.w(), self.w("w2", "INITIALIZING")]])
        res = lines[-1]
        self.assertEqual(code, 0, res.get("violation"))
        self.assertEqual((res["pass"], res["max_total"], res["max_running"]), (True, 2, 1))
        self.assertEqual([x["id"] for x in res["extra_workers"]], ["w2"])

    def test_the_worker_that_ran_is_not_an_extra_worker_in_the_result(self):
        """c7.21 (phase B of 20/11, `9aeff068`): three workers INITIALIZING before any RUNNING; the
        third ran. Declared an extra worker while none ran, it is not one in the result."""
        start = [listing(4.2, "w1", "INITIALIZING", []), listing(15.8, "w2", "INITIALIZING", []),
                 listing(37.3, "w3", "INITIALIZING", [])]
        init = [self.w(x, "INITIALIZING") for x in ("w1", "w2", "w3")]
        code, lines = self.scenario(self.PROBE, 410.0, start, ready=[listing(227.1, "w3", "RUNNING", ["w3"])],
                                    workers=[init], after=[[self.w("w1", "IDLE"), self.w("w2", "IDLE"), self.w("w3")]])
        res = lines[-1]
        self.assertEqual(code, 0, res.get("violation"))
        self.assertEqual(res["probe_valid"]["ready"]["served"], "w3")
        self.assertEqual(sorted(x["id"] for x in res["extra_workers"]), ["w1", "w2"])
        self.assertEqual([x["id"] for x in res["extra_workers_readings"]], ["w1", "w2"])

    # --- what stops a phase ----------------------------------------------------------------------
    def test_two_workers_running_at_once(self):
        for name, chat, after in (
                ("readings", [], [[self.w(), self.w("w2")]]),
                ("log", [listing(171.0, "w2", "RUNNING", ["w1", "w2"])], [[self.w()]])):
            for argv in (self.PROBE, self.A, self.B):
                with self.subTest(source=name, driver=" ".join(argv[:1] + argv[3:5])):
                    self.log.write_text("")
                    code, lines = self.scenario(argv, 170.0, [listing(100.0, "w1", "RUNNING", ["w1"])], chat=chat,
                                                after=after, later=self.tripped)
                    self.assertEqual(code, 6, lines[-1])
                    self.assertIn("two workers RUNNING at once", lines[-1]["violation"])

    def test_a_second_worker_in_a_state_not_admitted(self):
        # c7.20: a fact (no guarantee broken); the phase goes on to its zero
        gone = [listing(172.0, "w1", "GONE", [])]
        for status in ("UNHEALTHY", "EXITED"):
            for name, chat, after in (
                    ("readings", gone, [[self.w(), self.w("w2", status)], [self.w("w2", status)]]),
                    ("log", [listing(171.0, "w2", status, ["w1"])] + gone, [[self.w()], []])):
                with self.subTest(status=status, source=name):
                    self.log.write_text("")
                    code, lines = self.scenario(self.B, 170.0, [listing(100.0, "w1", "RUNNING", ["w1"])], chat=chat,
                                                after=after)
                    self.assertEqual(code, 0, lines[-1])
                    facts = [(f["fact"], f["worker"], f["status"]) for f in lines[-1]["facts"]]
                    self.assertIn(("extra_worker_state", "w2", status), facts)

    def test_more_workers_than_max_and_the_extra_workers(self):
        # c7.20: more than 1 + 2 listed, none RUNNING but one, is a fact
        after = [[self.w(), self.w("w2", "INITIALIZING"), self.w("w3", "IDLE"), self.w("w4", "THROTTLED")]]
        code, lines = self.scenario(self.PROBE, 170.0, [listing(100.0, "w1", "RUNNING", ["w1"])], after=after)
        self.assertEqual(code, 0, lines[-1])
        self.assertEqual([(f["fact"], f["total"]) for f in lines[-1]["facts"]], [("more_workers", 4)])

    def test_a_worker_replaced_after_readiness_or_after_the_hang(self):
        # c7.20: a replacement after readiness is a fact; after the hang of A a restart is a fact too,
        # and the worker that stays RUNNING breaks G1 (no zero within 215 s of the hang)
        chat = [listing(171.0, "w1", "GONE", []), listing(181.0, "w2", "RUNNING", ["w2"])]
        start = [listing(100.0, "w1", "RUNNING", ["w1"]), pline(165.0, "w1", "mdf_ready")]
        code, lines = self.scenario(self.PROBE, 170.0, start, chat=chat, after=[[self.w("w2")]])
        self.assertEqual(code, 0, lines[-1])
        self.assertEqual([(r["old"]["id"], r["new"]["id"], r["rule"]) for r in lines[-1]["replacements"]],
                         [("w1", "w2", "after_readiness")])
        self.log.write_text("")
        code, lines = self.scenario(self.A, 170.0, start, chat=chat, after=[[self.w("w2")]], later=self.tripped)
        self.assertEqual((code, lines[-1]["guarantee"]), (6, "G1"), lines[-1])
        self.assertIn("no zero within", lines[-1]["violation"])
        self.assertIn("RUNNING in the reading", lines[-1]["violation"])
        self.assertEqual([(r["old"]["id"], r["new"]["id"]) for r in lines[-1]["restarts_after_fault"]], [("w1", "w2")])

    def test_a_restart_after_the_crash_of_phase_b_is_observed(self):
        # c7.20: the restart stops by the end of the observation (the zero Z1 of B); the log is written
        # no later than the readings that follow it, as the follower lists every 10 s
        chat = [listing(171.0, "w1", "GONE", []), listing(172.0, "w2", "INITIALIZING", []),
                listing(173.0, "w2", "RUNNING", ["w2"]), listing(175.0, "w2", "IDLE", []),
                wrow(176.0, "w2", "system", "stop container c2")]
        code, lines = self.scenario(self.B, 170.0, [listing(100.0, "w1", "RUNNING", ["w1"])], chat=chat,
                                    after=[[self.w("w2")], [self.w("w2", "IDLE")]])
        res = lines[-1]
        self.assertEqual(code, 0, res.get("violation"))
        self.assertEqual([(r["old"]["id"], r["new"]["id"], r["new"]["running_s"]) for r in res["restarts_after_fault"]],
                         [("w1", "w2", 170.0)])                         # the first reading of w2 RUNNING, before the log
        self.assertEqual((res["restarts"], res["extra_workers"]), (1, []))

    def test_a_restart_in_the_readings_unknown_to_the_log(self):
        # c7.20: a fact; the readings alone judge two RUNNING and a RUNNING after the zero
        code, lines = self.scenario(self.B, 170.0, [listing(100.0, "w1", "RUNNING", ["w1"])],
                                    chat=[listing(171.0, "w1", "GONE", [])], after=[[self.w("w2")], []])
        self.assertEqual(code, 0, lines[-1])
        facts = [(f["fact"], f["old"], f["new"]) for f in lines[-1]["facts"]]
        self.assertEqual(facts, [("replacement_not_judged", "w1", "w2")])
        self.assertIn("not in the log of the session after 15 s", lines[-1]["facts"][0]["why"])

    # --- rule (a) before readiness ---------------------------------------------------------------
    def rule_a(self, old_lines=()):
        """w1 RUNNING at +100 s, gone at +150 s; w2 RUNNING at +185 s serves readiness at +190 s."""
        start = [listing(95.0, "w1", "INITIALIZING", []), listing(100.0, "w1", "RUNNING", ["w1"]),
                 pline(101.0, "w1", "mdf_start")] + list(old_lines) + [
            listing(150.0, "w1", "GONE", []), listing(155.0, "w2", "INITIALIZING", []),
            listing(185.0, "w2", "RUNNING", ["w2"]), pline(186.0, "w2", "mdf_start")]
        return self.scenario(self.PROBE, 190.0, start, ready=[pline(188.0, "w2", "mdf_ready")],
                             workers=[[self.w("w1")], [self.w("w2", "INITIALIZING")], [self.w("w2")]])

    def test_rule_a_a_replacement_before_any_answer_of_vllm_is_a_fact(self):
        code, lines = self.rule_a()
        res = lines[-1]
        self.assertEqual(code, 0, res.get("violation"))
        self.assertEqual(res["probe_valid"]["ready"]["served"], "w2")
        rep, = res["replacements"]
        self.assertEqual((rep["old"]["id"], rep["old"]["running_s"], rep["old"]["gone_s"], rep["new"]["id"],
                          rep["new"]["running_s"], rep["old_vllm_s"], rep["old_inference_lines"]),
                         ("w1", 100.0, 150.0, "w2", 185.0, None, 0))
        declared = [x for x in lines if "replacement" in x]
        self.assertEqual([(x["replacement"], x["rule"]) for x in declared], [(1, "a")])
        self.assertEqual(res["extra_workers"], [])

    def test_rule_a_not_after_an_answer_of_vllm_or_with_an_inference(self):
        # c7.20: outside rule (a), still a fact (no overlap, no guarantee broken), with its reason
        for name, old, expected in (
                ("answer", [pline(140.0, "w1", "mdf_vllm_first_health", status=200)], "after the first answer of vLLM"),
                ("access", [wrow(140.0, "w1", "container", '(APIServer pid=1) INFO:     127.0.0.1:1 - "GET /health '
                                 'HTTP/1.1" 200 OK')], "after the first answer of vLLM"),
                ("inference", [pline(140.0, "w1", "mdf_request", kind="error", method="POST",
                                     path="/v1/chat/completions", status=502, code="mdf_upstream")],
                 "with an inference on it")):
            with self.subTest(old=name):
                self.log.write_text("")
                code, lines = self.rule_a(old)
                self.assertEqual(code, 0, lines[-1])
                rep, = lines[-1]["replacements"]
                self.assertEqual(rep["rule"], "outside_a")
                self.assertIn(expected, " ".join(rep.get("outside_rule_a") or []) or str(lines))


class LoadRunPod(FakeRunPod):
    """The balancer with `/metrics` in vLLM's text and pages that take time on the fake clock (MFL033 N5b)."""

    def __init__(self, test, running=32, warm_s=3.0, page_s=0.1, workers=None):
        super().__init__(workers=workers or [[{"id": "w1", "status": "RUNNING", "gpuTypeId": "NVIDIA RTX A5000",
                                                "dataCenterId": "EU-RO-1"}]])
        self.test, self.running, self.warm_s, self.page_s = test, running, warm_s, page_s
        self.pages, self.lock = 0, threading.Lock()
        self.lb = {("GET", "/v1/models"): (200, {"data": [{"id": dv.MODEL}]}, "forwarded")}

    def __call__(self, request):
        url = str(request.url)
        if ".api.runpod.ai" in url and request.url.path == "/metrics":
            return httpx.Response(200, text="# HELP vllm:x\nprocess_cpu 1\nvllm:num_requests_running{m=\"a\"} %d\n"
                                            "vllm:num_requests_waiting{m=\"a\"} 0\n" % self.running,
                                  headers={"x-mdf-proxy": "forwarded"})
        if ".api.runpod.ai" in url and request.url.path == "/v1/chat/completions":
            self.calls.append(("POST", url, None, request.headers.get("authorization")))
            if request.extensions.get("timeout", {}).get("read") == rpe.ABANDON_AFTER_S:
                raise httpx.ReadTimeout("the client gave up")       # the abandoned page, while vLLM runs it
            with self.lock:
                self.pages += 1
                first = self.pages == 1
            body = json.loads(request.content)
            self.test.assertEqual(body["messages"][0]["content"][1]["text"], "OCR this image to HTML.")
            self.test.clock.sleep(self.warm_s if first else self.page_s)
            return httpx.Response(200, json={"choices": []}, headers={"x-mdf-proxy": "forwarded"})
        return super().__call__(request)


class TestConcurrencyRunPod(InSession):
    """MFL033 N5b: App. D on RunPod, the driver of Modal (c7.16); the criteria a measure, G1 and G2 the
    failures."""

    def setUp(self):
        super().setUp()
        self.png = self.dir / "page.png"
        self.png.write_bytes(b"\x89PNG\r\n\x1a\nfake")

    def concurrency(self, fake, *extra):
        arm_id = self.arm(allow_download=False)
        d = self.in_session(fake, ["concurrency", "--name", "mdf-surya-fault", "--input", str(self.png)] + list(extra),
                            arm_id)
        return self.run_quiet(d.concurrency)

    def test_the_load_measured_with_the_criteria_of_app_d(self):
        fake = LoadRunPod(self)
        code, lines = self.concurrency(fake, "--n", "32", "--warmup", "1", "--abandon", "1", "--probe-while-busy")
        res = lines[-1]
        self.assertEqual(code, 0, res.get("violation"))
        self.assertTrue(res["pass"])
        self.assertEqual(res["statuses"], [200] * 32)
        self.assertEqual(res["warm_s"], 3.0)
        self.assertEqual(res["peak"], {"running": 32})
        self.assertEqual(res["abandon"]["requests"][0]["error"], "ReadTimeout")
        self.assertTrue(res["abandon"]["exercised"])
        self.assertEqual(res["criteria"], {"running_at_least_24": True, "pages_200": True, "within_4_warm": True,
                                           "probes_200": True, "abandoned": True})
        self.assertTrue(res["measure_pass"])
        self.assertEqual(fake.pages, 33)                         # the warm-up page and the 32 of the load
        self.assertEqual(len(res["after_load"]), 3)              # 30 s after the load, every 10 s
        self.assertIn("page_s", res)

    def test_metrics_read_apart_from_the_probes_see_the_real_peak(self):
        """MFL033 N6 (21/11, window aa09739b: vLLM logged «Running: 29 reqs» while the driver read
        /metrics 3 times in 14 s and saw 6, because each round waited first for the probe of /v1/models,
        up to 5 s). Here the probes during the load answer only once the load is over, and the pages
        only once /metrics has seen them all running: read apart from the probes, /metrics sees 8."""
        fake, n = LoadRunPod(self), 8
        lock, inflight, all_in, done = threading.Lock(), {"n": 0}, threading.Event(), threading.Event()
        loaded = {"pages": 0}
        real = fake.__call__

        def handle(request):
            path = request.url.path
            if path == "/metrics":
                with lock:
                    running = inflight["n"]
                if running >= n:
                    all_in.set()
                return httpx.Response(200, text="vllm:num_requests_running %d\nvllm:num_requests_waiting 0\n" % running,
                                      headers={"x-mdf-proxy": "forwarded"})
            if path == "/v1/models" and loaded["pages"]:
                done.wait(10.0)                                  # a probe held by the busy server
            if path == "/v1/chat/completions":
                with lock:
                    loaded["pages"] += 1
                    load = loaded["pages"] > 1                   # the first is the warm-up page
                    inflight["n"] += load
                if load:
                    all_in.wait(10.0)
                    with lock:
                        inflight["n"] -= 1
                        if inflight["n"] == 0:
                            done.set()
                    return httpx.Response(200, json={"choices": []}, headers={"x-mdf-proxy": "forwarded"})
            return real(request)
        arm_id = self.arm(allow_download=False)
        d = self.in_session(fake, ["concurrency", "--name", "mdf-surya-fault", "--input", str(self.png), "--n", str(n),
                                   "--probe-while-busy"], arm_id, transport=httpx.MockTransport(handle))
        code, lines = self.run_quiet(d.concurrency)
        res = lines[-1]
        self.assertEqual(code, 0, res.get("violation"))
        self.assertTrue(all_in.is_set())
        self.assertEqual(res["peak"], {"running": n})
        self.assertEqual(res["statuses"], [200] * n)
        self.assertGreaterEqual(res["metrics"]["with_running"], 1)

    def test_a_criterion_not_met_is_a_measure_not_a_failure(self):
        code, lines = self.concurrency(LoadRunPod(self, running=8), "--n", "8")
        res = lines[-1]
        self.assertEqual(code, 0)
        self.assertFalse(res["criteria"]["running_at_least_24"])
        self.assertFalse(res["measure_pass"])
        self.assertIsNone(res["abandon"].get("exercised"))

    def test_two_workers_running_at_once_break_g1(self):
        two = [{"id": "w1", "status": "RUNNING", "dataCenterId": "EU-RO-1"},
               {"id": "w2", "status": "RUNNING", "dataCenterId": "EU-RO-1"}]
        code, lines = self.concurrency(LoadRunPod(self, workers=[two]), "--n", "4")
        res = lines[-1]
        self.assertEqual(code, 6)
        self.assertEqual(res["guarantee"], "G1")
        self.assertFalse(res["pass"])

    def test_no_readiness_is_inconclusive(self):
        fake = LoadRunPod(self)
        fake.lb = {("GET", "/v1/models"): (502, {"error": "upstream"}, "error")}
        code, lines = self.concurrency(fake, "--n", "4")
        self.assertEqual(code, dv.INCONCLUSIVE_EXIT)
        self.assertIn("not ready", lines[-1]["inconclusive"])
        self.assertEqual(fake.pages, 0)

    def test_the_parser_and_the_page_body(self):
        a = rpe.parser().parse_args(["--env-file", "e", "concurrency", "--name", "n", "--input", "p.png"])
        self.assertEqual((a.n, a.warmup, a.abandon, a.probe_while_busy), (32, 1, 0, False))
        body = dv.page_body(self.png)
        self.assertEqual(body["model"], dv.MODEL)
        self.assertEqual(body["max_tokens"], 1024)
        self.assertTrue(body["messages"][0]["content"][0]["image_url"]["url"].startswith("data:image/png;base64,"))
