"""The GPUs of the user in the product (MFL051 B; rilievo H, J.7.3 of the plan): the local GPU of the `gpu` profile
(`gpu-local`) and the remote server of `make server-up` (`gpu-remote`).

- a job of their presets opens no session: no row of `gpu_sessions`, nothing in the ledger, no call to a provider,
  `runs.gpu` left empty; the worker converts with the endpoint of servers.json and its keys (`adapters.direct`);
- the transit: never asked for the local GPU, and a confirmation given to it is refused (`route_option_unsupported`);
  asked for the remote server (`transit_not_confirmed` without it), in the API, the worker and the presets alike;
- the caps are not read: with every cap at 0 the two presets stay available, a rented one is `server_cap`;
- the readiness and the routes: no ledger, no reserve, no window; the local GPU is `broken` with `make gpu-up` while
  the socket of its front is missing; a job of these presets is a local job for a session that lingers."""
import io
import json
from pathlib import Path
import shutil
import socket
import sys
import tempfile
import unittest
from unittest import mock

from fastapi.testclient import TestClient
import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "unit" / "gpu"))
from gpu_harness import FakeClocks, PDF  # noqa: E402

from markdown_for_llms import services  # noqa: E402
from markdown_for_llms.api import create_app  # noqa: E402
from markdown_for_llms.engine import routes  # noqa: E402
from markdown_for_llms.engine.codes import EngineError  # noqa: E402
from markdown_for_llms.files import identify, write, write_json  # noqa: E402
from markdown_for_llms.gpu import adapters  # noqa: E402
from markdown_for_llms.services import Refused, Service  # noqa: E402
from markdown_for_llms.settings import Settings  # noqa: E402
from markdown_for_llms.worker import Worker  # noqa: E402

HERE = Path(__file__).resolve().parents[1] / "unit" / "gpu"
DRIVER = HERE / "provider_driver.py"
REMOTE_KEY = "sk-remote-gpu-synthetic-0123456789abcdef"
LOCAL = {"route": "marker-server", "server_mode": "balanced", "gpu_preset": "gpu-local"}
REMOTE = {"route": "marker-server", "server_mode": "balanced", "gpu_preset": "gpu-remote", "transit_confirmed": True}


def code_of(call):
    try:
        call()
    except EngineError as exc:
        return exc.code.value
    except Refused as exc:
        return exc.code
    return None


class Base(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="mdf-ugpu-"))
        self.addCleanup(shutil.rmtree, self.root, True)
        self.socket = self.root / "front.sock"
        self.servers = self.root / "servers.json"
        self.servers.write_text(json.dumps({"default": "local", "servers": {
            "local": {"kind": "gpu", "preset": "gpu-local", "socket": str(self.socket)},
            "remote": {"kind": "user", "preset": "gpu-remote", "url": "https://gpu.example.lan:23467"},
            "rp": {"kind": "runpod", "preset": "runpod-serverless-volume-dc", "resource": "mdf-surya",
                   "url": "https://ep1.api.runpod.ai"}}}))
        self.env = self.root / ".env"
        self.env.write_text("GPU_SERVER_KEY=%s\nRUNPOD_API_KEY=rk-synthetic\n" % REMOTE_KEY)
        self.env.chmod(0o600)
        self.ledger = self.root / "ledger"
        self.state = self.root / "provider.json"
        self.state.write_text(json.dumps({"provider": "runpod"}))

    def settings(self, **values):
        return Settings(**dict({"data_root": str(self.root / "data"), "model_root": str(self.root / "models"),
                                "gpu_presets": "gpu-local,gpu-remote,runpod-serverless-volume-dc",
                                "servers": str(self.servers), "gpu_ledger_dir": str(self.ledger),
                                "gpu_cap_usd_modal": "0", "gpu_cap_usd_runpod": "0", "gpu_session_cap_usd": "0"},
                               **values))

    def front(self):
        """The socket of the front of the local GPU as a file on disk (only its kind is checked by the readiness)."""
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.bind(str(self.socket))
        s.close()

    def worker(self, settings):
        w = Worker(settings, env_file=str(self.env), gpu={"argv": [sys.executable, "-I", "-B", str(DRIVER),
                                                                    str(self.state)], "clocks": FakeClocks()})
        self.addCleanup(w.storage.close)
        return w

    def provider_calls(self):
        return json.loads(self.state.read_text()).get("calls", [])


def direct_convert(calls):
    """A fake of `engine.convert` on the route marker-server for an endpoint without a provider: it records what the
    worker gives it and writes what the engine writes; it never sends a reservation."""
    def convert(source, out, request=None, limits=None, work=None, observer=None, name=None, model_root=None,
                datalab_key=None, server_endpoint=None, server_keys=None):
        calls.append({"server": dict(request.get("server") or {}), "endpoint": server_endpoint,
                      "keys": {k: v.reveal() for k, v in (server_keys or {}).items()}})
        out = Path(out)
        out.mkdir(parents=True)
        observer({"event": "child"})
        write(out, "document.md", b"converted on a GPU of the user")
        inventory = [identify(out, "document.md")]
        result = {"schema": "mdf-engine-result/1", "attempt": request.get("attempt"), "status": "complete",
                  "route": {"converter": "marker-server", "family": "pages", "chosen_by": "user", "reason": "imposed",
                            "steps": []}, "components": {}, "timings": {"total_seconds": 3}, "limits": [],
                  "problems": [], "problem_counts": {}, "problem_codes": {}, "inventory": inventory,
                  "inventory_complete": True,
                  "transit": {"left_machine": server_endpoint.leaves_machine(), "service": server_endpoint.kind}}
        write_json(out, "engine.json", result)
        return result
    return convert


class OptionsTests(Base):
    def test_the_transit_follows_the_machine_of_the_server(self):
        s = self.settings()
        self.assertEqual(services.job_options("pdf", LOCAL, s)["transit_confirmed"], False)
        self.assertEqual(code_of(lambda: services.job_options("pdf", dict(LOCAL, transit_confirmed=True), s)),
                         "route_option_unsupported")
        self.assertEqual(code_of(lambda: services.job_options("pdf", dict(REMOTE, transit_confirmed=False), s)),
                         "transit_not_confirmed")
        self.assertEqual(services.job_options("pdf", REMOTE, s)["gpu_preset"], "gpu-remote")
        # the rule of today stands for a rented preset and by default in the engine
        self.assertEqual(code_of(lambda: routes.job_options("pdf", dict(LOCAL))), "transit_not_confirmed")
        self.assertEqual(code_of(lambda: routes.job_options("pdf", dict(LOCAL, transit_confirmed=True),
                                                            leaves_machine=False)), "route_option_unsupported")

    def test_the_caps_are_not_read(self):
        s = self.settings()
        self.assertEqual(services.gpu_availability("gpu-local", s), {"available": True, "code": None})
        self.assertEqual(services.gpu_availability("gpu-remote", s), {"available": True, "code": None})
        self.assertEqual(services.gpu_availability("runpod-serverless-volume-dc", s)["code"], "server_cap")
        self.assertEqual(services.gpu_availability("gpu-local", self.settings(gpu_presets="gpu-remote"))["code"],
                         "gpu_preset_not_enabled")

    def test_the_api_and_the_conversion_presets(self):
        settings = self.settings()
        origin = "http://127.0.0.1:%d" % settings.port
        client = TestClient(create_app(settings), base_url=origin)
        self.addCleanup(client.close)
        headers = {"origin": origin, "x-mdf-csrf": client.get("/api/v1/session").json()["csrf"]}

        def post(query):
            return client.post("/api/v1/runs" + query, content=PDF, headers=dict(
                headers, **{"Content-Type": "application/octet-stream", "X-Filename": "a.pdf"}))
        ok = post("?preset=local-gpu")
        self.assertIn(ok.status_code, (200, 201, 202), ok.text)
        refused = post("?preset=local-gpu&transit_confirmed=true")
        self.assertEqual(refused.json().get("code"), "route_option_unsupported", refused.text)
        self.assertEqual(post("?preset=remote-gpu").json().get("code"), "transit_not_confirmed")
        self.assertIn(post("?preset=remote-gpu&transit_confirmed=true").status_code, (200, 201, 202))
        estimate = client.get("/api/v1/gpu/estimate?preset=gpu-local").json()
        self.assertEqual((estimate["costs"], estimate["leaves_machine"], estimate["start_cents"], estimate["region"]),
                         (False, False, "0.00", "this computer"))
        self.assertIn("stay on this computer", estimate["notice"])
        self.assertEqual(client.get("/api/v1/gpu/estimate?preset=gpu-remote").json()["leaves_machine"], True)
        armed = client.post("/api/v1/gpu/arm", json={"gpu_preset": "gpu-local", "transit_confirmed": True},
                            headers=headers)
        self.assertEqual(armed.json().get("code"), "route_option_unsupported")


class WorkerTests(Base):
    def run_job(self, options):
        s = self.settings()
        service = Service(s)
        self.addCleanup(service.storage.close)
        job = service.submit(io.BytesIO(PDF), "a.pdf", options)
        w = self.worker(s)
        calls = []
        row = w.claim()
        with mock.patch("markdown_for_llms.worker.convert", direct_convert(calls)):
            w.process(row)
        return service, w, service.storage.get(job["id"]), calls

    def test_no_session_no_ledger_no_provider(self):
        for options, kind, keys in ((LOCAL, "gpu", {}), (REMOTE, "user", {"GPU_SERVER_KEY": REMOTE_KEY})):
            with self.subTest(preset=options["gpu_preset"]):
                service, w, run, calls = self.run_job(options)
                self.assertEqual(run["state"], "completed", run.get("error"))
                with service.storage.connect() as conn:
                    self.assertEqual(conn.execute("SELECT COUNT(*) FROM gpu_sessions").fetchone()[0], 0)
                self.assertIsNone(run["gpu"])
                self.assertFalse((self.ledger / "spend.jsonl").exists())
                self.assertEqual(self.provider_calls(), [])
                call, = calls
                self.assertEqual((call["endpoint"].kind, call["endpoint"].costs, call["keys"]), (kind, False, keys))
                self.assertEqual(set(call["server"]), {"mode", "transit_confirmed", "name"})   # no budget, no window
                self.assertIsNone(w.gpu.session)

    def test_a_job_of_a_gpu_of_the_user_is_local_for_a_session_that_lingers(self):
        s = self.settings()
        service = Service(s)
        self.addCleanup(service.storage.close)
        w = self.worker(s)
        self.assertFalse(w.gpu.local_queued())
        service.submit(io.BytesIO(PDF), "a.pdf", LOCAL)
        self.assertTrue(w.gpu.local_queued())


class ReadinessTests(Base):
    def test_no_ledger_no_reserve_no_window(self):
        self.front()
        w = self.worker(self.settings())
        gpu = w.gpu.readiness()
        local, remote = gpu["presets"]["gpu-local"], gpu["presets"]["gpu-remote"]
        for item in (local, remote):
            self.assertEqual((item["available"], item["code"], item["costs"], item["a_cents"], item["w_s"],
                              item["minimum_reserve_cents"], item["open_session"]), (True, None, False, None, None,
                                                                                    None, False))
        self.assertEqual(gpu["presets"]["runpod-serverless-volume-dc"]["code"], "server_cap")
        self.assertEqual(self.provider_calls(), [])

    def test_the_local_gpu_without_its_front(self):
        w = self.worker(self.settings(gpu_presets="gpu-local"))
        item = w.gpu.readiness()["presets"]["gpu-local"]
        self.assertEqual((item["available"], item["code"], item["detail"]),
                         (False, "server_unavailable", "start the local GPU: make gpu-up"))
        # a refresh reads only a server that can serve: nothing is asked of a front that is not there
        with mock.patch.object(adapters, "of_endpoint", side_effect=AssertionError("read")):
            self.assertIsNone(w.gpu.readiness(read=True)["presets"]["gpu-local"]["reading"])

    def test_the_routes_of_the_gpus_of_the_user(self):
        self.front()
        w = self.worker(self.settings())
        w.readiness["gpu"] = w.gpu.readiness()
        body = w.routes_now({})
        by = {r["id"]: r for r in body["routes"]}
        local, remote = by["marker-server:gpu-local"], by["marker-server:gpu-remote"]
        self.assertEqual((local["transit"]["required"], local["spend"], local["ledger"], local["preset"], local["state"]),
                         (False, "none", None, None, "available"))
        self.assertEqual((remote["transit"], remote["spend"], remote["provider"]),
                         ({"required": True, "what": "page_images"}, "none", "remote"))
        rented = by["marker-server:runpod-serverless-volume-dc"]
        self.assertEqual((rented["transit"]["required"], rented["spend"], rented["state"]), (True, "capped", "disabled"))

    def test_the_readings_are_made_only_on_request(self):
        self.front()
        w = self.worker(self.settings())
        seen = []

        def models(request):
            seen.append(request.url.path)
            return httpx.Response(200, json={"data": [{"id": "datalab-to/surya-ocr-2.5"}]})
        real = adapters.of_endpoint
        with mock.patch.object(adapters, "of_endpoint",
                               lambda p, ep, keys=None, transport=None: real(p, ep, keys, httpx.MockTransport(models))):
            self.assertIsNone(w.gpu.readiness()["presets"]["gpu-local"]["reading"])
            self.assertEqual((seen, self.provider_calls()), ([], []))
            item = w.gpu.readiness(read=True)["presets"]["gpu-local"]
            self.assertEqual((item["reading"]["health"], item["balance"]), ("ok", None))
            self.assertEqual(seen, ["/v1/models", "/v1/models"])                 # the local GPU and the remote one
            self.assertEqual(w.gpu.readiness()["presets"]["gpu-local"]["reading"]["health"], "ok")   # kept
        # the rented preset is read by a child of opchild, once, on the request alone
        self.assertEqual([c[0] for c in self.provider_calls()], ["state"])


if __name__ == "__main__":
    unittest.main()
