"""The adapters of the providers of the GPU presets (MFL051 B; the interface D11, App. A of the plan): the contract for
each adapter, without a network and without a provider.

- `zero()` returns the state with the meaning of today: RunPod at zero only with no worker listed and nothing queued,
  whatever their state (a worker INITIALIZING counts, `running` does not decide it); Modal with no app alive and no
  container;
- `arm` keeps the parameter `preset`, whose pins RunPod reads again before its PATCH;
- `disarm` is idempotent: the stop of an app of Modal already stopped is a disarm;
- `read()` of an adapter that costs never asks the URL of the service: the REST v2 of RunPod, the lists of the CLI of
  Modal; `running` counts the RUNNING alone and is for the panel;
- `balance()`: RunPod `GET /v2/billing/balance` (REST v2, never GraphQL) with the key of the worker; with a key that does
  not read it (401, 403) None, and no request leaves with another key; Modal None without a call;
- a GPU of the user: traits, `read()` from `GET /v1/models`, nothing armed, a zero always true, no balance;
- `requirements` is pure; `direct` gives the endpoint and the keys of a preset that costs nothing and refuses a rented
  one."""
import json
from pathlib import Path
import shutil
import tempfile
import types
import unittest

import httpx

from markdown_for_llms.engine import envfile, servers
from markdown_for_llms.gpu import adapters, lifecycle, opchild, presets
from markdown_for_llms.gpu import providers as pv
from markdown_for_llms.settings import Settings

KEY = "rk-b-synthetic-key/0123+456789=="
API = "api.runpod.io"


class FakeRunPod:
    """The REST v2 of one endpoint (`test_gpu_opchild.FakeRunPod`), with its workers and the balance of the account;
    every request is recorded with its host and its Authorization header."""

    def __init__(self, workers=None, summary=None, balance=200):
        self.ep = {"id": "ep1", "name": "mdf-surya", "type": "LOAD_BALANCER", "image": pv.IMAGE,
                   "entrypoint": pv.ENTRYPOINT, "env": pv.base_env(), "workers": pv.workers_block(0),
                   "scaling": {"type": "REQUEST_COUNT", "requestCount": 32}, "flashboot": "OFF",
                   "ports": ["8000/http"], "dataCenterIds": ["CA-MTL-1"], "mounts": {"network": [{"volumeId": "vol1"}]},
                   "gpu": {"pools": [pv.POOL], "minCudaVersion": pv.CUDA_MIN, "count": 1}}
        self.volume = {"id": "vol1", "name": pv.VOLUME, "dataCenter": "CA-MTL-1", "type": "HIGH_PERFORMANCE", "size": 10}
        self.workers = workers or []
        self.summary = summary if summary is not None else {"total": len(self.workers)}
        self.balance = balance
        self.calls = []

    def __call__(self, request):
        self.calls.append((request.method, request.url.host, request.url.path, request.headers.get("Authorization")))
        if request.url.host != API:
            return httpx.Response(599, json={"error": "a call outside the management API"})
        path, method = request.url.path.split("/v2", 1)[1], request.method
        if path == "/serverless" and method == "GET":
            return httpx.Response(200, json={"endpoints": [self.ep], "pagination": {}})
        if path == "/serverless/ep1" and method == "GET":
            return httpx.Response(200, json=self.ep)
        if path == "/serverless/ep1" and method == "PATCH":
            self.ep.update({k: v for k, v in json.loads(request.content).items() if k != "gpu"})
            return httpx.Response(200, json=self.ep)
        if path == "/serverless/ep1/workers":
            return httpx.Response(200, json={"summary": self.summary, "workers": self.workers})
        if path == "/network-volumes":
            return httpx.Response(200, json={"networkVolumes": [self.volume]})
        if path == "/catalog/gpus":
            return httpx.Response(200, json={"gpus": [{"id": "NVIDIA RTX A5000", "pool": pv.POOL, "secure": True,
                                                       "price": {"serverless": 0.75}}]})
        if path == "/catalog/datacenters":
            return httpx.Response(200, json={"dataCenters": [{"id": "CA-MTL-1",
                                                              "networkVolumeTypes": ["HIGH_PERFORMANCE"]}]})
        if path == "/billing/balance" and method == "GET":
            if self.balance == 200:
                return httpx.Response(200, json={"balanceAmount": 12.3456, "billingMode": "prepaid"})
            return httpx.Response(self.balance, json={"error": "this API key does not grant billing:read"})
        return httpx.Response(404, json={"error": "not found"})


def runpod(**kw):
    fake = FakeRunPod(**kw)
    return fake, adapters.RunPod(pv.RunPod(envfile.Secret(KEY), "mdf-surya", transport=httpx.MockTransport(fake)))


class FakeModal:
    """The CLI of Modal 1.6.3 as a runner: the calls recorded by their first two words."""

    def __init__(self, apps=(), containers=(), stop_text=None):
        self.apps, self.containers, self.stop_text = list(apps), list(containers), stop_text
        self.calls = []

    def __call__(self, argv, **kw):
        args = argv[len(pv.MODAL_SDK):]
        word = " ".join(args[:2])
        self.calls.append(word)
        if word == "app list":
            return types.SimpleNamespace(returncode=0, stdout=json.dumps(self.apps), stderr="")
        if word == "container list":
            return types.SimpleNamespace(returncode=0, stdout=json.dumps(self.containers), stderr="")
        if word.startswith("app stop"):
            if self.stop_text:
                return types.SimpleNamespace(returncode=1, stdout="", stderr=self.stop_text)
            return types.SimpleNamespace(returncode=0, stdout="", stderr="")
        return types.SimpleNamespace(returncode=0, stdout="[]", stderr="")


def modal(**kw):
    fake = FakeModal(**kw)
    return fake, adapters.Modal(pv.Modal("mdf-surya", runner=fake, environ={}))


class TraitsTests(unittest.TestCase):
    def test_the_traits_of_the_four_adapters(self):
        _, rp = runpod()
        _, md = modal()
        ep = servers.endpoint("local", {"kind": "gpu", "socket": "/run/mdf/run/gpu/front.sock"})
        local = adapters.of_endpoint("local", ep)
        remote = adapters.of_endpoint("remote", servers.endpoint("remote", {"kind": "user", "url": "https://gpu.lan"}))
        self.assertEqual([(a.provider, a.costs, a.arms, a.leaves_machine) for a in (rp, md, local, remote)], [
            ("runpod", True, True, True), ("modal", True, True, True), ("local", False, False, False),
            ("remote", False, False, True)])


class RunPodTests(unittest.TestCase):
    def test_zero_with_the_state_and_its_meaning_of_today(self):
        _, rp = runpod()
        zero, state = rp.zero()
        self.assertTrue(zero)
        self.assertFalse(state["busy"])
        # a worker INITIALIZING (or IDLE) is listed: not at zero, although none is RUNNING
        _, rp = runpod(workers=[{"id": "w1", "status": "INITIALIZING"}])
        zero, state = rp.zero()
        self.assertFalse(zero)
        self.assertEqual(pv.running_count(state), 0)
        # nothing listed but a request in the queue
        _, rp = runpod(summary={"total": 1})
        self.assertFalse(rp.zero()[0])

    def test_read_never_asks_the_url_of_the_service(self):
        fake, rp = runpod(workers=[{"id": "w1", "status": "RUNNING"}, {"id": "w2", "status": "INITIALIZING"}])
        got = rp.read()
        self.assertEqual({c[1] for c in fake.calls}, {API})
        self.assertEqual(set(got), {"armed", "busy", "running", "workers", "health", "read_at"})
        self.assertEqual((got["armed"], got["busy"], got["running"], got["health"]), (False, True, 1, "ok"))
        self.assertEqual(got["workers"], [{"id": "w1", "status": "RUNNING"}, {"id": "w2", "status": "INITIALIZING"}])
        _, rp = runpod(workers=[{"id": "w2", "status": "INITIALIZING"}])
        self.assertEqual(rp.read()["health"], "starting")
        _, rp = runpod()
        self.assertEqual((rp.read()["running"], rp.read()["health"]), (0, "unknown"))

    def test_arm_keeps_its_preset_and_reads_its_pins_before_the_patch(self):
        preset = presets.load()["runpod-serverless-volume-dc"]
        fake, rp = runpod()
        rp.arm(deadline=None, levers=pv.DEFAULT_LEVERS, fault="", preset=preset)
        self.assertIn(("PATCH", API, "/v2/serverless/ep1", "Bearer " + KEY), fake.calls)
        fake, rp = runpod()
        fake.volume["type"] = "STANDARD"                                     # outside the pins of the preset
        with self.assertRaises(pv.ProviderError):
            rp.arm(preset=preset)
        self.assertNotIn("PATCH", [c[0] for c in fake.calls])
        with self.assertRaises(pv.ProviderError):
            rp.verify_pins(preset)

    def test_disarm_is_idempotent(self):
        fake, rp = runpod()
        rp.disarm()
        rp.disarm()                                                          # already at 0: the same PATCH, no error
        self.assertEqual([c[2] for c in fake.calls if c[0] == "PATCH"], ["/v2/serverless/ep1"] * 2)

    def test_the_balance_of_the_rest_v2_with_the_key_of_the_worker(self):
        fake, rp = runpod()
        got = rp.balance()
        self.assertEqual(got["usd"], "12.3456")
        self.assertIn("read_at", got)
        self.assertEqual(fake.calls, [("GET", API, "/v2/billing/balance", "Bearer " + KEY)])

    def test_a_key_that_does_not_read_the_balance_gives_none_and_no_other_request(self):
        for status in (401, 403):
            with self.subTest(status=status):
                fake, rp = runpod(balance=status)
                self.assertIsNone(rp.balance())
                self.assertEqual(fake.calls, [("GET", API, "/v2/billing/balance", "Bearer " + KEY)])
        fake, rp = runpod(balance=500)
        self.assertIsNone(rp.balance())
        self.assertNotIn("/graphql", [c[2] for c in fake.calls])

    def test_the_balance_through_opchild(self):
        import io
        fake, rp = runpod()
        out = io.StringIO()
        request = {"op": "balance", "provider": "runpod", "resource": "mdf-surya", "env_file": "x", "args": {},
                   "parent": 1, "until": None}
        code = opchild.main(stdin=io.BytesIO(json.dumps(request).encode()), stdout=out, factory=lambda r: rp.prov,
                            watch=lambda p: True)
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(out.getvalue())["result"]["balance"]["usd"], "12.3456")
        out = io.StringIO()
        request["op"] = "read"
        opchild.main(stdin=io.BytesIO(json.dumps(request).encode()), stdout=out, factory=lambda r: rp.prov,
                     watch=lambda p: True)
        self.assertEqual(json.loads(out.getvalue())["result"]["reading"]["running"], 0)
        self.assertNotIn(KEY, out.getvalue())


class ModalTests(unittest.TestCase):
    APP = {"app_id": "ap-1", "description": "mdf-surya", "state": "deployed"}

    def test_zero_and_read_with_the_lists_of_the_cli_alone(self):
        fake, md = modal(apps=[self.APP], containers=[{"container_id": "ta-1", "app_id": "ap-1",
                                                       "app_name": "mdf-surya", "start_time": "2026-11-27 10:00"}])
        zero, state = md.zero()
        self.assertFalse(zero)
        got = md.read()
        self.assertEqual((got["armed"], got["busy"], got["running"], got["health"]), (True, True, 1, "ok"))
        self.assertEqual(set(fake.calls), {"app list", "container list"})
        fake, md = modal(apps=[dict(self.APP, state="stopped")])
        self.assertTrue(md.zero()[0])

    def test_disarm_is_idempotent(self):
        for text in ("App is already stopped.", "No App with name 'mdf-surya' found"):
            with self.subTest(text=text):
                fake, md = modal(stop_text=text)
                md.disarm()
                self.assertEqual(fake.calls, ["app stop"])

    def test_no_balance_and_no_call(self):
        fake, md = modal()
        self.assertIsNone(md.balance())
        self.assertEqual(fake.calls, [])

    def test_the_keys_of_modal_include_the_token_of_the_sdk(self):
        _, md = modal()
        self.assertEqual(md.key_names(), ["MODAL_PROXY_TOKEN_ID", "MODAL_PROXY_TOKEN_SECRET", "MODAL_TOKEN_ID",
                                          "MODAL_TOKEN_SECRET"])


class UserGpuTests(unittest.TestCase):
    def local(self, handler):
        ep = servers.endpoint("local", {"kind": "gpu", "socket": "/run/mdf/run/gpu/front.sock"})
        return adapters.of_endpoint("local", ep, transport=httpx.MockTransport(handler))

    def test_the_reading_of_the_local_gpu(self):
        seen = []

        def models(request):
            seen.append((request.method, request.url.path, request.headers.get("Authorization")))
            return httpx.Response(200, json={"data": [{"id": "datalab-to/surya-ocr-2.5"}]})
        a = self.local(models)
        got = a.read()
        self.assertEqual((got["armed"], got["busy"], got["running"], got["health"]), (False, False, 1, "ok"))
        self.assertEqual(seen, [("GET", "/v1/models", None)])
        for status, health in ((503, "starting"), (204, "starting"), (500, "unhealthy"), (401, "unhealthy")):
            with self.subTest(status=status):
                self.assertEqual(self.local(lambda r, s=status: httpx.Response(s)).read()["health"], health)
        self.assertEqual(self.local(lambda r: httpx.Response(200, json={"data": [{"id": "other"}]})).read()["health"],
                         "unhealthy")

        def down(request):
            raise httpx.ConnectError("refused")
        self.assertEqual(self.local(down).read()["health"], "unknown")

    def test_nothing_armed_and_no_balance(self):
        a = self.local(lambda r: httpx.Response(200))
        self.assertEqual((a.arm(), a.disarm(), a.zero(), a.balance(), a.verify_pins(None)), ({}, None, (True, {}), None,
                                                                                            None))

    def test_the_remote_server_sends_its_key(self):
        ep = servers.endpoint("remote", {"kind": "user", "url": "https://gpu.lan:23467", "key_env": ["GPU_SERVER_KEY"]})
        seen = []

        def models(request):
            seen.append((request.url.host, request.headers.get("Authorization")))
            return httpx.Response(200, json={"data": [{"id": "datalab-to/surya-ocr-2.5"}]})
        a = adapters.of_endpoint("remote", ep, {"GPU_SERVER_KEY": envfile.Secret("sk-remote")},
                                 transport=httpx.MockTransport(models))
        self.assertEqual(a.read()["health"], "ok")
        self.assertEqual(seen, [("gpu.lan", "Bearer sk-remote")])


class RequirementsTests(unittest.TestCase):
    def settings(self, **kw):
        return Settings(**dict({"gpu_cap_usd_runpod": "0", "gpu_session_cap_usd": "0.75"}, **kw))

    def test_pure_and_closed(self):
        _, rp = runpod()
        ep = servers.endpoint("rp", {"kind": "runpod", "url": "https://ep1.api.runpod.ai", "rate_usd_per_s": "0.0002"})
        self.assertEqual(rp.requirements(self.settings(), [], None), [
            {"name": "the endpoint of the GPU preset", "where": "state/config/servers.json", "kind": "servers.json"},
            {"name": "RUNPOD_API_KEY", "where": ".env", "kind": "key"},
            {"name": "MDF_GPU_CAP_USD_RUNPOD", "where": ".env", "kind": "setting"}])
        self.assertEqual(rp.requirements(self.settings(gpu_cap_usd_runpod="1.5"), ["RUNPOD_API_KEY"], ep), [])
        local = adapters.of_endpoint("local", servers.endpoint("l", {"kind": "gpu", "socket": "/s"}))
        self.assertEqual(local.requirements(self.settings(), [], None), [
            {"name": "the endpoint of the GPU preset", "where": "state/config/servers.json", "kind": "servers.json"}])
        remote = adapters.of_endpoint("remote", servers.endpoint("r", {"kind": "user", "url": "https://h",
                                                                       "key_env": ["GPU_SERVER_KEY"]}))
        self.assertEqual([r["name"] for r in remote.requirements(self.settings(), [], None)][1:], ["GPU_SERVER_KEY"])


class DirectTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root, True)
        self.servers = self.root / "servers.json"
        self.servers.write_text(json.dumps({"default": "local", "servers": {
            "local": {"kind": "gpu", "preset": "gpu-local", "socket": "/run/mdf/run/gpu/front.sock"},
            "remote": {"kind": "user", "preset": "gpu-remote", "url": "https://gpu.lan:23467"},
            "rp": {"kind": "runpod", "preset": "runpod-serverless-volume-dc", "resource": "mdf-surya",
                   "url": "https://ep1.api.runpod.ai"}}}))
        self.env = self.root / ".env"
        self.env.write_text("GPU_SERVER_KEY=sk-remote-0123456789\nRUNPOD_API_KEY=%s\n" % KEY)
        self.env.chmod(0o600)

    def settings(self, enabled="gpu-local,gpu-remote,runpod-serverless-volume-dc"):
        return Settings(data_root=str(self.root / "data"), gpu_presets=enabled, servers=str(self.servers),
                        gpu_cap_usd_runpod="0", gpu_session_cap_usd="0")

    def test_the_endpoint_and_the_keys_of_a_gpu_of_the_user(self):
        ep, keys = adapters.direct(self.settings(), "gpu-local", str(self.env))
        self.assertEqual((ep.kind, ep.socket, ep.costs, ep.leaves_machine(), keys), ("gpu", "/run/mdf/run/gpu/front.sock",
                                                                                     False, False, {}))
        ep, keys = adapters.direct(self.settings(), "gpu-remote", str(self.env))
        self.assertEqual((ep.kind, ep.costs, ep.leaves_machine(), sorted(keys)), ("user", False, True,
                                                                                  ["GPU_SERVER_KEY"]))
        self.assertEqual(keys["GPU_SERVER_KEY"].reveal(), "sk-remote-0123456789")

    def test_refusals(self):
        for name, settings, code in (
                ("runpod-serverless-volume-dc", self.settings(), "server_cap"),
                ("gpu-local", self.settings("gpu-remote"), "gpu_preset_not_enabled"),
                ("nope", self.settings(), "gpu_preset_unknown")):
            with self.subTest(name=name):
                with self.assertRaises(lifecycle.Refusal) as ctx:
                    adapters.direct(settings, name, str(self.env))
                self.assertEqual(ctx.exception.code, code)
        settings = Settings(data_root=str(self.root / "data"), gpu_presets="runpod-serverless-volume-dc",
                            servers=str(self.servers), gpu_cap_usd_runpod="1", gpu_session_cap_usd="0.5")
        with self.assertRaises(lifecycle.Refusal) as ctx:
            adapters.direct(settings, "runpod-serverless-volume-dc", str(self.env))
        self.assertEqual(ctx.exception.code, "route_option_unsupported")
        self.env.write_text("RUNPOD_API_KEY=%s\n" % KEY)
        self.env.chmod(0o600)
        with self.assertRaises(lifecycle.Refusal) as ctx:
            adapters.direct(self.settings(), "gpu-remote", str(self.env))
        self.assertEqual(ctx.exception.code, "server_key_missing")


if __name__ == "__main__":
    unittest.main()
