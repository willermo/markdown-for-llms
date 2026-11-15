"""`mdf convert` on the engine, `mdf submit`, `mdf prepare` (A4 of MFL032)."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import os
import unittest
from unittest import mock

from engine_test_support import FIXTURES
from markdown_for_llms import cli


class CliTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.base = ["--data-root", str(self.root / "data"), "--model-root", str(self.root / "models")]

    def run_cli(self, argv):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            try:
                code = cli.main(self.base + argv)
            except SystemExit as exc:
                code = exc.code
        return code, out.getvalue(), err.getvalue()

    def test_convert_writes_the_contract_and_a_summary(self):
        src = self.root / "a.txt"
        src.write_text("Una riga.\n", encoding="utf-8")
        code, out, err = self.run_cli(["convert", str(src), "--out", str(self.root / "o"), "--lang", "it"])
        self.assertEqual(code, 0)
        summary = json.loads(out)
        self.assertEqual((summary["status"], summary["error"], summary["route"]["converter"]), ("complete", None, "text"))
        self.assertEqual(summary["files"], 1)
        self.assertEqual(err, "")                                   # no terminal: no step lines
        self.assertEqual((self.root / "o" / "document.md").read_text(), "Una riga.\n")
        self.assertNotIn("Una riga", out)                           # no document text on stdout
        self.assertTrue((self.root / "o" / "engine.json").is_file())

    def test_convert_failures_and_usage(self):
        empty = self.root / "no-models"
        empty.mkdir()
        code, out, _ = self.run_cli(["--model-root", str(empty), "convert", str(FIXTURES / "C06-pdf-text-it.pdf"),
                                     "--out", str(self.root / "p")])
        self.assertEqual((code, json.loads(out)["error"]), (1, "weights_missing"))
        with mock.patch.dict(os.environ, {"DATALAB_API_KEY": ""}):
            code, out, _ = self.run_cli(["convert", str(FIXTURES / "C06-pdf-text-it.pdf"), "--route", "datalab",
                                         "--out", str(self.root / "p2")])
        self.assertEqual((code, json.loads(out)["error"]), (1, "datalab_key_missing"))       # B4: no key, nothing sent
        notes = self.root / "n.md"
        notes.write_text("x")
        code, out, _ = self.run_cli(["convert", str(notes), "--out", str(self.root / "q")])
        self.assertEqual((code, json.loads(out)["error"]), (1, "format_unsupported"))
        code, out, _ = self.run_cli(["convert", str(notes), "--out", str(self.root / "q")])     # q is not empty now
        self.assertEqual((code, json.loads(out)["status"]), (2, "refused"))
        code, _, _ = self.run_cli(["convert", str(notes)])                                    # --out is required
        self.assertEqual(code, 2)
        code, _, _ = self.run_cli(["convert", str(notes), "--out", str(self.root / "r"), "--route", "nope"])
        self.assertEqual(code, 2)
        code, out, _ = self.run_cli(["convert", str(notes), "--out", str(self.root / "s"), "--max-pages", "0"])
        self.assertEqual(code, 2)

    def test_convert_passes_the_configured_threads_and_stall(self):                      # FC4
        src = self.root / "a.txt"
        src.write_text("x\n", encoding="utf-8")
        seen = []
        with mock.patch.dict(os.environ, {"MDF_THREADS": "1", "MDF_STALL_SECONDS": "60", "MDF_TIMEOUT_SECONDS": "0"}), \
                mock.patch("markdown_for_llms.engine.core.convert",
                           side_effect=lambda *a, **k: seen.append(a[3]) or {"status": "complete"}):
            code, out, _ = self.run_cli(["convert", str(src), "--out", str(self.root / "lim"), "--max-pages", "7"])
        self.assertEqual(code, 0, out)
        self.assertEqual({k: seen[0].get(k) for k in ("threads", "stall_seconds", "max_pages", "poll_seconds")},
                         {"threads": 1, "stall_seconds": 60, "max_pages": 7, "poll_seconds": 1})
        from markdown_for_llms.settings import Settings
        settings = Settings.load(environ={"MDF_THREADS": "1", "MDF_STALL_SECONDS": "60"})
        self.assertEqual(seen[0], settings.engine_limits(7))      # the worker uses the same limits

    def test_datalab_key_file_is_checked_before_the_settings(self):                     # HC2
        dangling = self.root / "dangling.env"
        dangling.symlink_to(self.root / "absent")
        fifo = self.root / "fifo.env"
        os.mkfifo(fifo, 0o600)
        for env in (dangling, fifo):
            with self.subTest(env=env.name):
                code, out, _ = self.run_cli(["--env-file", str(env), "convert", str(FIXTURES / "C06-pdf-text-it.pdf"),
                                             "--route", "datalab", "--out", str(self.root / ("k-" + env.name))])
                self.assertEqual((code, json.loads(out)["error"]), (1, "datalab_key_unsafe"))

    def test_submit_is_the_archive_command(self):
        seen = []

        class FakeService:
            def __init__(self, settings):
                pass

            def submit(self, stream, name, overrides):
                seen.append((name, overrides))
                return dict(id="r", state="queued", bundle="pending", progress=None)
        src = self.root / "p.pdf"
        src.write_bytes((FIXTURES / "C06-pdf-text-it.pdf").read_bytes())
        with mock.patch("markdown_for_llms.cli.Service", FakeService):
            code, out, _ = self.run_cli(["submit", "--ocr", "force", "--lang", "it", str(src)])
            self.assertEqual((code, json.loads(out)["state"]), (0, "queued"))
            code, out, _ = self.run_cli(["submit", "--route", "datalab", "--datalab-mode", "fast", "--confirm-transit", str(src)])
            self.assertEqual(code, 0)
        self.assertEqual(seen, [("p.pdf", {"route": None, "ocr": "force", "lang": "it", "datalab_mode": None,
                                           "transit_confirmed": False}),
                                ("p.pdf", {"route": "datalab", "ocr": "auto", "lang": None, "datalab_mode": "fast",
                                           "transit_confirmed": True})])

    def test_submit_reports_the_code_of_a_refusal(self):
        src = self.root / "fake.docx"
        src.write_bytes(b"%PDF-1.4\n")
        code, out, _ = self.run_cli(["submit", str(src)])
        self.assertEqual(code, 1)
        self.assertEqual(json.loads(out)["code"], "format_mismatch")
        code, out, _ = self.run_cli(["submit", "--route", "datalab", str(src)])
        self.assertEqual(json.loads(out)["code"], "route_invalid")

    def test_prepare_is_the_bench_prepare(self):
        with mock.patch("markdown_for_llms.bench.cli.main", return_value=0) as bench:
            self.assertEqual(cli.main(["prepare", "--toolchain", "--no-cache"]), 0)
        bench.assert_called_once_with(["prepare", "--toolchain", "--no-cache"])


if __name__ == "__main__":
    unittest.main()
