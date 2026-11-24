"""The API and the worker refuse the same job options (MFL023 A5): one function, `services.job_options`, behind
`Service.upload`, `Service.retry` and `worker.engine_request`, for the options of schema 3 and the name of a conversion
preset; a job without a preset stores the five options of schema 3 and nothing more."""
import io
import json
import tempfile
import unittest

from markdown_for_llms import services
from markdown_for_llms.engine.codes import EngineError
from markdown_for_llms.services import Refused, Service
from markdown_for_llms.settings import Settings
from markdown_for_llms.worker import engine_request

PDF = b'%PDF-1.4\nsynthetic contract input; not a valid conversion fixture\n'
CASES = [
    {}, {"route": "marker"}, {"route": "marker", "ocr": "force"}, {"route": "marker", "lang": "it"},
    {"route": "datalab", "transit_confirmed": True}, {"route": "datalab", "transit_confirmed": True, "datalab_mode": "fast"},
    {"preset": "local-fast"}, {"route": "datalab", "transit_confirmed": True, "datalab_mode": "balanced", "preset": "datalab"},
    # refused
    {"route": "nope"}, {"route": 3}, {"route": "pandoc"}, {"route": "marker-server"}, {"ocr": "maybe"},
    {"lang": "not a tag"}, {"route": "datalab"}, {"route": "marker", "datalab_mode": "fast"},
    {"transit_confirmed": "true"}, {"unknown": 1}, {"server_mode": "balanced"}, {"gpu_preset": "modal-base"},
    {"preset": None}, {"preset": ""}, {"preset": "Bad Name"}, {"preset": 3},
]


class ParityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.service = Service(Settings(data_root=self.temp.name + '/data', model_root=self.temp.name + '/models'))
        self.addCleanup(self.service.storage.close)

    def api(self, options):
        try:
            job = self.service.submit(io.BytesIO(PDF), 'a.pdf', options)
        except Refused as exc:
            return ("refused", exc.code)
        stored = json.loads(self.service.storage.get(job['id'])['options'])
        return ("accepted", stored)

    def worker(self, options):
        try:
            engine_request(json.dumps(options), 'attempt', 'pdf')
        except EngineError as exc:
            return ("refused", exc.code.value)
        return ("accepted", services.job_options('pdf', options))

    def test_the_same_refusals_and_the_same_options(self):
        refused = 0
        for options in CASES:
            with self.subTest(options=options):
                api, worker = self.api(options), self.worker(options)
                self.assertEqual(api, worker)
                refused += api[0] == "refused"
        self.assertEqual(refused, 16)

    def test_a_retry_uses_the_same_function(self):
        job = self.service.submit(io.BytesIO(PDF), 'a.pdf', {"preset": "local-fast", "route": "marker"})
        with self.service.storage.connect() as conn:
            conn.execute("UPDATE runs SET state='failed' WHERE id=?", (job['id'],))
        again = self.service.retry(job['id'])
        self.assertEqual(json.loads(self.service.storage.get(again['id'])['options'])['preset'], 'local-fast')
        with self.service.storage.connect() as conn:
            conn.execute("UPDATE runs SET state='failed', options=? WHERE id=?",
                         (json.dumps({"route": "marker", "preset": "Bad Name"}), job['id']))
        with self.assertRaises(Refused) as ctx:
            self.service.retry(job['id'])
        self.assertEqual(ctx.exception.code, 'invalid_setting')


GPU = {"route": "marker-server", "server_mode": "balanced", "gpu_preset": "modal-base", "transit_confirmed": True}
GPU_CASES = [
    GPU, dict(GPU, server_mode="accurate", ocr="force", lang="it"), dict(GPU, gpu_preset="runpod-serverless-volume-dc"),
    dict(GPU, preset="quality-now"),
    # refused
    {"route": "marker-server"}, dict(GPU, transit_confirmed=False), dict(GPU, server_mode=None), dict(GPU, server_mode="x"),
    dict(GPU, gpu_preset="Bad"), dict(GPU, gpu_preset="nope"), dict(GPU, gpu_preset="runpod-serverless-volume-eu-se-1"),
    dict(GPU, datalab_mode="fast"), {"route": "marker", "gpu_preset": "modal-base"}, {"server_mode": "balanced"},
]


class GpuParityTests(unittest.TestCase):
    """MFL023 B3: on the route marker-server the API and the worker refuse the same options with the settings of the
    instance (`services.job_options(fmt, options, settings)`): the same codes of the engine and of the GPU presets."""

    def service(self, **values):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        service = Service(Settings(data_root=temp.name + '/data', model_root=temp.name + '/models', **values))
        self.addCleanup(service.storage.close)
        return service

    def outcomes(self, service, options):
        try:
            job = service.submit(io.BytesIO(PDF), 'a.pdf', options)
            api = ("accepted", json.loads(service.storage.get(job['id'])['options']))
        except Refused as exc:
            api = ("refused", exc.code)
        try:
            engine_request(json.dumps(options), 'attempt', 'pdf', settings=service.settings)
            worker = ("accepted", services.job_options('pdf', options, service.settings))
        except EngineError as exc:
            worker = ("refused", exc.code.value)
        except Refused as exc:
            worker = ("refused", exc.code)
        return api, worker

    def test_the_same_refusals_with_the_settings_of_the_instance(self):
        enabled = self.service(gpu_presets="modal-base,runpod-serverless-volume-dc", gpu_cap_usd_modal="3",
                               gpu_cap_usd_runpod="2", gpu_session_cap_usd="0.5")
        bare = self.service()
        capped = self.service(gpu_presets="modal-base", gpu_session_cap_usd="0.5")
        counts = {}
        for name, service in (("enabled", enabled), ("bare", bare), ("capped", capped)):
            for options in GPU_CASES:
                with self.subTest(instance=name, options=options):
                    api, worker = self.outcomes(service, options)
                    self.assertEqual(api, worker)
                    counts.setdefault(name, []).append(api[1] if api[0] == "refused" else "accepted")
        self.assertEqual(counts["enabled"][:4], ["accepted"] * 4)
        self.assertEqual(counts["bare"][:4], ["gpu_preset_not_enabled"] * 4)
        self.assertEqual(counts["capped"][:4], ["server_cap", "server_cap", "gpu_preset_not_enabled", "server_cap"])
        self.assertEqual(counts["enabled"][4:], ["route_not_in_product", "transit_not_confirmed", "route_option_unsupported",
                                                 "route_option_unsupported", "invalid_setting", "gpu_preset_unknown",
                                                 "gpu_preset_superseded", "route_option_unsupported",
                                                 "route_option_unsupported", "route_option_unsupported"])


if __name__ == '__main__':
    unittest.main()
