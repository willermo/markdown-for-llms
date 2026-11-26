"""The `marker-server` route in the product (MFL023 B3; plan r2 §4.1): only on a GPU preset of the instance, with a
server mode and the transit confirmed; a local or own inference server stays for `mdf convert` and `mdf bench`
(`route_not_in_product`, D4). The product does not list the route among the alternatives (it is reached through the
conversion presets); the API and the worker validate the same options with the settings of the instance (the GPU preset
enabled, the caps above 0); a retry confirms the transit again; the readiness of the worker never contacts an inference
server. Rewritten from the test of MFL033, which proved the refusal of the route until MFL023."""
import io
import json
import socket
import sys
import tempfile
import unittest

from fastapi.testclient import TestClient

from markdown_for_llms import services
from markdown_for_llms.api import create_app
from markdown_for_llms.engine import routes
from markdown_for_llms.engine.codes import Code, EngineError
from markdown_for_llms.services import Refused, Service
from markdown_for_llms.settings import Settings
from markdown_for_llms.worker import Worker, engine_request

CONNECTS = []
WATCH = [False]
PDF = b'%PDF-1.4\nsynthetic contract input; not a valid conversion fixture\n'
GPU = {"route": "marker-server", "server_mode": "balanced", "gpu_preset": "modal-base", "transit_confirmed": True}
ENABLED = {"gpu_presets": "modal-base,runpod-serverless-volume-dc", "gpu_cap_usd_modal": "3", "gpu_cap_usd_runpod": "2",
           "gpu_session_cap_usd": "0.50"}


def _audit(event, args):
    if WATCH[0] and event in ("socket.connect", "socket.getaddrinfo", "http.client.connect"):
        CONNECTS.append((event, repr(args[1:2])[:80]))


sys.addaudithook(_audit)


def code_of(call):
    try:
        call()
    except EngineError as exc:
        return exc.code.value
    except Refused as exc:
        return exc.code
    return None


class Instance(unittest.TestCase):
    def settings(self, **values):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        return Settings(data_root=tmp.name + "/data", model_root=tmp.name + "/models", **values)


class TestTheRouteInTheProduct(Instance):
    def test_the_product_view_of_the_matrix(self):
        for fmt in routes.MATRIX:
            self.assertNotIn(routes.MARKER_SERVER, routes.routes_for(fmt)["alternatives"])
        for fmt in ("pdf", "png", "jpeg", "tiff", "djvu"):
            self.assertIn(routes.MARKER_SERVER, routes.engine_routes_for(fmt)["alternatives"])
        self.assertEqual(routes.routes_for("pdf"), {"default": "marker", "alternatives": ["datalab"]})

    def test_without_a_gpu_preset_the_route_stays_out(self):
        for options in ({"route": "marker-server"}, {"route": "marker-server", "ocr": "force"},
                        {"route": "marker-server", "server_mode": "balanced", "transit_confirmed": True}):
            with self.subTest(options=options):
                self.assertEqual(code_of(lambda: routes.job_options("pdf", options)), "route_not_in_product")
        message = str(Refused(Code.ROUTE_NOT_IN_PRODUCT.value))
        self.assertIn("only on a GPU preset of the instance", message)
        self.assertIn("mdf convert and mdf bench only", message)

    def test_the_options_of_the_gpu_route(self):
        self.assertEqual(routes.job_options("pdf", GPU), {"route": "marker-server", "ocr": "auto", "lang": None,
                                                          "datalab_mode": None, "transit_confirmed": True,
                                                          "server_mode": "balanced", "gpu_preset": "modal-base"})
        self.assertEqual(routes.job_options("png", dict(GPU, server_mode="accurate", ocr="force", lang="it"))["ocr"],
                         "force")
        for change, code in (({"transit_confirmed": False}, "transit_not_confirmed"),
                             ({"server_mode": None}, "route_option_unsupported"),
                             ({"server_mode": "fast"}, "route_option_unsupported"),
                             ({"gpu_preset": "Modal Base"}, "invalid_setting"), ({"gpu_preset": 3}, "invalid_setting"),
                             ({"datalab_mode": "fast"}, "route_option_unsupported"), ({"ocr": "maybe"}, "route_option_unsupported")):
            with self.subTest(change=change):
                self.assertEqual(code_of(lambda: routes.job_options("pdf", dict(GPU, **change))), code)
        self.assertEqual(code_of(lambda: routes.job_options("docx", GPU)), "route_invalid")
        for options in ({"server_mode": "balanced"}, {"gpu_preset": "modal-base"}, {"route": "marker", "transit_confirmed": True},
                        {"route": "datalab", "transit_confirmed": True, "gpu_preset": "modal-base"}):
            with self.subTest(options=options):
                self.assertEqual(code_of(lambda: routes.job_options("pdf", options)), "route_option_unsupported")

    def test_the_instance_decides_the_gpu_preset(self):
        cases = [(None, "modal-base", "gpu_preset_not_enabled"),
                 ({}, "modal-base", "gpu_preset_not_enabled"),
                 ({"gpu_presets": "runpod-serverless-volume-dc"}, "modal-base", "gpu_preset_not_enabled"),
                 ({"gpu_presets": "modal-base"}, "modal-base", "server_cap"),                   # the caps at 0
                 (dict(ENABLED, gpu_cap_usd_modal="0"), "modal-base", "server_cap"),
                 (dict(ENABLED, gpu_session_cap_usd="0"), "runpod-serverless-volume-dc", "server_cap"),
                 (ENABLED, "runpod-serverless-volume-eu-se-1", "gpu_preset_superseded"),
                 (ENABLED, "no-such-preset", "gpu_preset_unknown"),
                 (ENABLED, "modal-base", None), (ENABLED, "runpod-serverless-volume-dc", None)]
        for values, gpu, code in cases:
            with self.subTest(values=values, gpu=gpu):
                settings = None if values is None else self.settings(**values)
                self.assertEqual(code_of(lambda: services.job_options("pdf", dict(GPU, gpu_preset=gpu), settings)), code)
                self.assertEqual(services.gpu_availability(gpu, settings), {"available": code is None, "code": code})

    def test_the_request_of_the_worker(self):
        settings = self.settings(**ENABLED)
        request = engine_request(json.dumps(GPU), "att", "pdf", settings=settings)
        self.assertEqual(request, {"route": "marker-server", "ocr": "auto", "lang": None, "attempt": "att",
                                   "server": {"mode": "balanced", "transit_confirmed": True}})
        self.assertEqual(code_of(lambda: engine_request(json.dumps(GPU), "att", "pdf", settings=self.settings())),
                         "gpu_preset_not_enabled")

    def test_the_readiness_of_the_worker_contacts_no_server(self):
        with tempfile.TemporaryDirectory() as tmp:
            worker = Worker(Settings(data_root=tmp + "/data", model_root=tmp + "/models", **ENABLED))
            self.addCleanup(worker.storage.close)
            CONNECTS.clear()
            WATCH[0] = True
            try:
                state = worker.ready()
            finally:
                WATCH[0] = False
        self.assertNotIn(routes.MARKER_SERVER, state)
        # the launcher is asked whether it answers (its socket, the one address the suite admits): no inference server is
        self.assertEqual([c for c in CONNECTS if "AF_UNIX" not in c[1] and "/run/mdf/launcher/launcher.sock" not in c[1]], [])


class TestTheArchive(Instance):
    def test_a_job_on_a_gpu_preset_and_its_retry(self):
        service = Service(self.settings(**ENABLED))
        self.addCleanup(service.storage.close)
        job = service.submit(io.BytesIO(PDF), "a.pdf", GPU)
        stored = json.loads(service.storage.get(job["id"])["options"])
        self.assertEqual(stored, {"route": "marker-server", "ocr": "auto", "lang": None, "datalab_mode": None,
                                  "transit_confirmed": True, "server_mode": "balanced", "gpu_preset": "modal-base"})
        service.cancel(job["id"])
        with self.assertRaises(Refused) as caught:                  # the transit is confirmed for one sending
            service.retry(job["id"])
        self.assertEqual(caught.exception.code, "transit_not_confirmed")
        again = service.retry(job["id"], {"transit_confirmed": True})
        self.assertEqual(json.loads(service.storage.get(again["id"])["options"])["gpu_preset"], "modal-base")

    def test_the_api_with_the_presets_of_the_instance(self):
        settings = self.settings(**ENABLED)
        client = TestClient(create_app(settings), base_url=f"http://127.0.0.1:{settings.port}")
        csrf = client.get("/api/v1/session").json()["csrf"]
        headers = {"origin": f"http://127.0.0.1:{settings.port}", "x-mdf-csrf": csrf, "x-filename": "a.pdf"}
        refused = client.post("/api/v1/runs?preset=quality-now", content=PDF, headers=headers)
        self.assertEqual((refused.status_code, refused.json()["code"]), (409, "transit_not_confirmed"))
        made = client.post("/api/v1/runs?preset=quality-now&transit_confirmed=true", content=PDF, headers=headers)
        self.assertEqual(made.status_code, 202, made.text)
        self.assertEqual(made.json()["options"], {"route": "marker-server", "ocr": "auto", "lang": None,
                                                  "datalab_mode": None, "transit_confirmed": True,
                                                  "server_mode": "balanced", "gpu_preset": "modal-base",
                                                  "preset": "quality-now"})
        presets = {p["name"]: p for p in client.get("/api/v1/presets").json()["presets"]}
        self.assertEqual({n: (presets[n]["available"], presets[n]["code"]) for n in ("quality-now", "batch")},
                         {"quality-now": (True, None), "batch": (True, None)})
        bare = client.post("/api/v1/runs?route=marker-server&server_mode=balanced&gpu_preset=modal-base"
                           "&transit_confirmed=true", content=PDF, headers=headers)
        self.assertEqual(bare.status_code, 202, bare.text)                # the options without a preset name


if __name__ == "__main__":
    unittest.main()
