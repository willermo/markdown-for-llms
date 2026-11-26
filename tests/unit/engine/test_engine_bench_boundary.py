"""K2 (PA2) of MFL032: a failure raised inside the engine reaches the handlers of the bench.

`BenchError` is `EngineError` and the bench codes compare equal to the engine's, so the real
callers of the bench (`bench/cli.py` per document and at the top, `bench/selftest.py`) give the
closed reports of before for errors raised by `engine.tools`: a non-fatal `tool_failed` marks one
document unavailable, a fatal one stops the command, `toolchain_missing` prints its hint.
"""
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import containers_test_support as launcher_fake
from markdown_for_llms.bench import cli as bench_cli
from markdown_for_llms.bench import fetch, selftest
from markdown_for_llms.bench.codes import Code as BenchCode
from markdown_for_llms.engine import containers, toolchain, tools
from markdown_for_llms.engine.codes import Code, EngineError


class BoundaryTests(unittest.TestCase):
    """The tool of a call exits 2 (`tool_failed`), or times out and the launcher cannot confirm that its
    container is gone (fatal): a scripted launcher, no container."""

    def setUp(self):
        saved = dict(toolchain._state)
        self.addCleanup(toolchain._state.update, saved)
        toolchain.reset()
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = tmp.name
        self.mode = "fail"
        self.launcher = launcher_fake.patched(self, Path(tmp.name) / "volume", self.script)

    def script(self, request, call):
        if self.mode == "timeout":
            return None                     # the tool never ends; its removal will not be confirmed
        return 2, b"", b"qpdf: not a PDF file\n"

    def derive(self, doc, manifest, cache, transport, offline=False):
        """Stands in for the fetch of a document: a derivation through the engine's runner."""
        if doc["id"] == "D05":
            self.mode = "timeout"

            class Every:
                def __contains__(self, name):
                    return True
            self.launcher.unconfirmed = Every()
        with mock.patch.object(containers, "UNREACHABLE_SECONDS", 0.3):
            tools.run("qpdf", ["--check", "x"], cache.tool_work_dir, timeout=1)
        raise AssertionError("not reached")

    def test_fetch_handlers_of_the_bench(self):
        out, err = io.StringIO(), io.StringIO()
        with mock.patch.object(fetch, "resolve_document", self.derive), contextlib.redirect_stdout(out), \
                contextlib.redirect_stderr(err):
            code = bench_cli.main(["fetch", "--docs", "D04,D05", "--offline", "--cache", self.tmp])
        self.assertEqual(code, 1)
        self.assertIn("D04 unavailable tool_failed", out.getvalue())         # not fatal: one document
        self.assertNotIn("D05 unavailable", out.getvalue())                   # fatal: stops the command
        self.assertEqual(err.getvalue().strip(), "error: tool_failed")

    def test_toolchain_missing_reaches_the_top_handler_with_its_hint(self):
        err = io.StringIO()
        self.launcher.answer["refuse"] = ("toolchain_missing", "no image mdf-bench-toolchain:candidate")
        with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
            code = bench_cli.main(["prepare", "--toolchain", "--cache", self.tmp])
        self.assertEqual(code, 1)
        self.assertIn("error: toolchain_missing", err.getvalue())
        self.assertIn("make prepare", err.getvalue())

    def test_selftest_expects_engine_codes(self):
        detail = selftest._expect_code("step", BenchCode.TOOL_FAILED, lambda: tools.run("qpdf", ["x"], self.tmp))
        self.assertIn("exit 2", detail)
        with self.assertRaises(EngineError) as ctx:
            selftest._expect_code("step", BenchCode.TOOL_MEMORY_LIMIT, lambda: tools.run("qpdf", ["x"], self.tmp))
        self.assertEqual(ctx.exception.code, Code.TOOLCHAIN_MISMATCH)


if __name__ == "__main__":
    unittest.main()
