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


if __name__ == '__main__':
    unittest.main()
