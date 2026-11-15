"""`mdf convert` on the engine, `mdf submit`, `mdf prepare` (A4 of MFL032)."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
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
        code, out, _ = self.run_cli(["convert", str(FIXTURES / "C06-pdf-text-it.pdf"), "--out", str(self.root / "p")])
        self.assertEqual((code, json.loads(out)["error"]), (1, "route_unavailable"))
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

    def test_submit_is_the_archive_command(self):
        class FakeService:
            def __init__(self, settings):
                pass

            def submit(self, stream, name, overrides):
                self.seen = (name, overrides)
                return dict(id="r", state="queued", bundle="pending", progress=None)
        src = self.root / "p.pdf"
        src.write_bytes((FIXTURES / "C06-pdf-text-it.pdf").read_bytes())
        with mock.patch("markdown_for_llms.cli.Service", FakeService):
            code, out, _ = self.run_cli(["submit", "--force-ocr", str(src)])
        self.assertEqual((code, json.loads(out)["state"]), (0, "queued"))

    def test_prepare_is_the_bench_prepare(self):
        with mock.patch("markdown_for_llms.bench.cli.main", return_value=0) as bench:
            self.assertEqual(cli.main(["prepare", "--toolchain", "--no-cache"]), 0)
        bench.assert_called_once_with(["prepare", "--toolchain", "--no-cache"])


if __name__ == "__main__":
    unittest.main()
