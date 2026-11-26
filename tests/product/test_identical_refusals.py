"""The same refusal, with the same closed code, in the API, in the worker and in `mdf convert` (MFL051, plan r2.1
J.5.4, App. F.2): for a route and for a conversion preset.

One function validates the options of a job (`services.job_options`) for the API and for the worker, and one
resolves a conversion preset (`Service.preset_options`) for the API, `mdf submit` and, since MFL051, `mdf
convert --preset`. The engine validates a route the same way when `mdf convert` runs it directly.

The one declared difference: a preset on a rented GPU. The archive answers with what the instance says of the
preset; `mdf convert`, which has no session, no ledger and no cap of its own, answers `preset_needs_archive`.
"""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from fastapi.testclient import TestClient

from markdown_for_llms import cli
from markdown_for_llms.api import create_app
from markdown_for_llms.engine.codes import EngineError
from markdown_for_llms.services import Refused
from markdown_for_llms.settings import Settings
from markdown_for_llms.worker import engine_request

PDF = b"%PDF-1.4\nsynthetic contract input; not a valid conversion fixture\n"
ACCEPTED = "accepted"


class IdenticalRefusals(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.settings = Settings(data_root=str(self.root / "data"), model_root=str(self.root / "models"))
        self.client = TestClient(create_app(self.settings), base_url=f"http://127.0.0.1:{self.settings.port}")
        self.headers = {"origin": f"http://127.0.0.1:{self.settings.port}",
                        "x-mdf-csrf": self.client.get("/api/v1/session").json()["csrf"]}
        self.files = {"txt": self.root / "note.txt", "pdf": self.root / "paper.pdf"}
        self.files["txt"].write_text("Una riga.\n", encoding="utf-8")
        self.files["pdf"].write_bytes(PDF)
        self.count = 0

    def api(self, fmt, query):
        response = self.client.post("/api/v1/runs", params=query, content=self.files[fmt].read_bytes(),
                                    headers=dict(self.headers, **{"x-filename": self.files[fmt].name}))
        if response.status_code == 202:
            return ACCEPTED
        self.assertGreaterEqual(response.status_code, 400)
        return response.json().get("code")

    def worker(self, fmt, options):
        """What the worker says of the same options when it takes the job."""
        try:
            engine_request(json.dumps(options), "attempt", fmt, settings=self.settings)
        except EngineError as exc:
            return exc.code.value
        except Refused as exc:
            return exc.code
        return ACCEPTED

    def convert(self, fmt, *options):
        self.count += 1
        out = io.StringIO()

        def recorder(*args, **kw):                  # a request the CLI accepts ends here: the engine is not started
            raise KeyboardInterrupt
        with mock.patch("markdown_for_llms.engine.core.convert", recorder), contextlib.redirect_stdout(out):
            code = cli.main(["--data-root", self.settings.data_root, "--model-root", self.settings.model_root, "convert",
                             str(self.files[fmt]), *options, "--out", str(self.root / ("out-%d" % self.count))])
        answer = json.loads(out.getvalue())
        return ACCEPTED if code == 130 else answer["error"]

    def test_routes_and_their_options(self):
        cases = [("txt", {}), ("pdf", {}), ("pdf", {"route": "marker"}), ("pdf", {"route": "marker", "ocr": "force"}),
                 # refused
                 ("txt", {"route": "marker"}), ("pdf", {"route": "pandoc"}), ("txt", {"ocr": "force"}),
                 ("pdf", {"route": "marker", "datalab_mode": "fast"})]
        seen = {}
        for fmt, options in cases:
            with self.subTest(fmt=fmt, options=options):
                api, worker = self.api(fmt, options), self.worker(fmt, options)
                self.assertEqual(api, worker)
                seen[(fmt, json.dumps(options, sort_keys=True))] = api
        self.assertEqual(sum(1 for code in seen.values() if code != ACCEPTED), 4)
        self.assertEqual(seen[("txt", '{"route": "marker"}')], "route_invalid")
        # `mdf convert` asks the engine, which validates the route and its options with the functions of the API: the
        # refused requests end with the same code, before any child is started (the accepted ones would convert)
        for fmt, flags, key in (("txt", ["--route", "marker"], '{"route": "marker"}'), ("pdf", ["--route", "pandoc"], '{"route": "pandoc"}'),
                                ("txt", ["--ocr", "force"], '{"ocr": "force"}')):
            with self.subTest(convert=flags):
                self.assertNotEqual(seen[(fmt, key)], ACCEPTED)
                self.assertEqual(self.convert_refused(fmt, flags), seen[(fmt, key)])

    def convert_refused(self, fmt, flags):
        """`mdf convert` with the real engine, for a request the engine refuses before it starts anything."""
        self.count += 1
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = cli.main(["--data-root", self.settings.data_root, "--model-root", self.settings.model_root, "convert",
                             str(self.files[fmt]), *flags, "--out", str(self.root / ("eng-%d" % self.count))])
        self.assertIn(code, (1, 2))
        return json.loads(out.getvalue())["error"]

    def test_conversion_presets(self):
        cases = [("local-fast", {}), ("no-such-preset", {}), ("Bad Name", {})]
        for name, extra in cases:
            with self.subTest(preset=name):
                api = self.api("pdf", dict(extra, preset=name))
                convert = self.convert("pdf", "--preset", name)
                self.assertEqual(api, convert)
        self.assertEqual(self.api("pdf", {"preset": "no-such-preset"}), "preset_unknown")
        self.assertEqual(self.api("pdf", {"preset": "local-fast"}), ACCEPTED)

    def test_an_option_beside_a_preset_is_refused_everywhere(self):
        self.assertEqual(self.api("pdf", {"preset": "local-fast", "route": "marker"}), "invalid_setting")
        self.assertEqual(self.convert("pdf", "--preset", "local-fast", "--route", "marker"), "invalid_setting")
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = cli.main(["--data-root", self.settings.data_root, "--model-root", self.settings.model_root, "submit",
                             str(self.files["pdf"]), "--preset", "local-fast", "--route", "marker"])
        self.assertEqual((code, json.loads(out.getvalue())["code"]), (1, "invalid_setting"))

    def test_the_declared_difference_a_preset_on_a_rented_gpu(self):
        for name in ("quality-now", "batch"):
            with self.subTest(preset=name):
                self.assertEqual(self.api("pdf", {"preset": name, "transit_confirmed": "true"}), "gpu_preset_not_enabled")
                self.assertEqual(self.convert("pdf", "--preset", name, "--confirm-transit"), "preset_needs_archive")


if __name__ == "__main__":
    unittest.main()
