"""Forwarding of `mdf bench`, the MDF_BENCH_ prefix in the product and the bench settings."""
import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from markdown_for_llms import cli
from markdown_for_llms.settings import ConfigurationError, Settings
from markdown_for_llms.bench.codes import BenchError, Code
from markdown_for_llms.bench.settings import BenchSettings


class ForwardingTests(unittest.TestCase):
    def test_bench_is_forwarded_before_the_product_settings(self):
        out = io.StringIO()
        with patch.object(Settings, "load", side_effect=AssertionError("product settings loaded")), \
                contextlib.redirect_stdout(out):
            self.assertEqual(cli.main(["bench", "list", "--level", "quick"]), 0)
        text = out.getvalue()
        for doc_id in ("D04", "D15", "D18", "D19"):
            self.assertIn(doc_id, text)
        self.assertNotIn("D01 ", text)

    def test_bench_error_prints_a_code_only(self):
        err = io.StringIO()
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stderr(err):
            env = Path(tmp) / "bench.env"
            env.write_text("MDF_BENCH_SECRET_SENTINEL=value-sentinel\n")
            self.assertEqual(cli.main(["bench", "list", "--env-file", str(env)]), 1)
        self.assertEqual(err.getvalue().strip(), "error: unknown_setting")


class ProductPrefixTests(unittest.TestCase):
    def test_product_ignores_bench_variables_only(self):
        settings = Settings.load(environ={"MDF_BENCH_CACHE_DIR": "/x", "MDF_BENCH_ANYTHING": "y"})
        self.assertEqual(settings.port, 23465)
        with self.assertRaises(ConfigurationError):
            Settings.load(environ={"MDF_BENCHX": "1"})
        with self.assertRaises(ConfigurationError):
            Settings.load(environ={"MDF_UNKNOWN": "1"})


class BenchSettingsTests(unittest.TestCase):
    def test_defaults_follow_xdg(self):
        s = BenchSettings.load(environ={"XDG_CACHE_HOME": "/c", "XDG_DATA_HOME": "/d"})
        self.assertEqual((s.relay_port, s.inference_port), (23461, 23463))
        with patch.dict("os.environ", {"XDG_CACHE_HOME": "/c", "XDG_DATA_HOME": "/d"}):
            s = BenchSettings.load(environ={})
        self.assertEqual(s.cache_dir, Path("/c/markdown-for-llms/bench"))
        self.assertEqual(s.runs_dir, Path("/d/markdown-for-llms/bench/runs"))
        self.assertEqual(s.data_dir, Path("/d/markdown-for-llms/bench"))

    def test_env_file_environment_and_options(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = Path(tmp) / "bench.env"
            env.write_text("MDF_BENCH_RELAY_PORT=23465\nMDF_BENCH_CACHE_DIR=/from-file\nDATALAB_API_KEY=ignored\n")
            s = BenchSettings.load(env, environ={"MDF_BENCH_CACHE_DIR": "/from-env"})
            self.assertEqual(s.relay_port, 23465)
            self.assertEqual(s.cache_dir, Path("/from-env"))
            s = BenchSettings.load(env, environ={}, cache="/from-option")
            self.assertEqual(s.cache_dir, Path("/from-option"))
            with self.assertRaises(BenchError) as ctx:
                BenchSettings.load(Path(tmp) / "missing.env", environ={})
            self.assertEqual(ctx.exception.code, Code.ENV_FILE_UNAVAILABLE)

    def test_rejections(self):
        cases = {
            Code.UNKNOWN_SETTING: {"MDF_BENCH_TYPO": "1"},
            Code.INVALID_SETTING: {"MDF_BENCH_RELAY_PORT": "port"},
            Code.PORT_OUTSIDE_BLOCK: {"MDF_BENCH_RELAY_PORT": "8000"},
        }
        for code, environ in cases.items():
            with self.subTest(code=code), self.assertRaises(BenchError) as ctx:
                BenchSettings.load(environ=environ)
            self.assertEqual(ctx.exception.code, code)
        with self.assertRaises(BenchError):
            BenchSettings.load(environ={"MDF_BENCH_RELAY_PORT": "23463"})

    def test_tool_memory_cap(self):
        from markdown_for_llms.bench import tools
        self.assertEqual(BenchSettings.load(environ={}).tool_memory_mb, tools.DEFAULT_MEMORY_MB)
        self.assertEqual(BenchSettings.load(environ={"MDF_BENCH_TOOL_MEMORY_MB": "2048"}).tool_memory_mb, 2048)
        for value in ("0", "lots", "2097152"):
            with self.subTest(value=value), self.assertRaises(BenchError) as ctx:
                BenchSettings.load(environ={"MDF_BENCH_TOOL_MEMORY_MB": value})
            self.assertEqual(ctx.exception.code, Code.INVALID_SETTING)
        # the command line hands the setting to the tools before any command runs
        before = tools.memory_limit()
        self.addCleanup(tools.set_memory_limit, before)
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            env = Path(tmp) / "bench.env"
            env.write_text("MDF_BENCH_TOOL_MEMORY_MB=3072\n")
            self.assertEqual(cli.main(["bench", "list", "--env-file", str(env)]), 0)
        self.assertEqual(tools.memory_limit(), 3072)
