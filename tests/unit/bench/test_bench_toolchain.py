"""Reference toolchain: identity, verification of the local image (E2), engine locality, reports."""
import contextlib
import hashlib
import io
import json
import os
import unittest
from unittest import mock

from bench_test_support import FakeEngine, require_toolchain
from markdown_for_llms.bench import cli, toolchain
from markdown_for_llms.bench.codes import BenchError, Code

GOOD_ID = "sha256:" + "a" * 64


class Base(unittest.TestCase):
    def setUp(self):
        saved = dict(toolchain._state)
        self.addCleanup(toolchain._state.update, saved)
        toolchain.reset()

    def engine(self, manifest=None, config=None, image_id=GOOD_ID, present=True):
        spec = toolchain.spec()
        image = {"id": image_id, "config": spec["config"] if config is None else config,
                 "manifest": b"stored manifest" if manifest is None else manifest} if present else None
        fake = FakeEngine(image=image)
        for target in (mock.patch.object(toolchain, "engine_run", fake),
                       mock.patch.object(toolchain, "engine_path", return_value="/usr/bin/docker")):
            target.start()
            self.addCleanup(target.stop)
        return fake

    def identity(self, manifest):
        patcher = mock.patch.object(toolchain, "identity", return_value=hashlib.sha256(manifest).hexdigest())
        patcher.start()
        self.addCleanup(patcher.stop)


class VerificationTests(Base):
    def test_verified_once_then_called_by_id(self):
        fake = self.engine()
        self.identity(b"stored manifest")
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("DOCKER_HOST", None)
            self.assertEqual(toolchain.image(), GOOD_ID)
            self.assertEqual(toolchain.image(), GOOD_ID)
        verbs = [c[0] for c in fake.calls]
        self.assertEqual(verbs, ["context", "image", "run", "version"])        # once per process
        self.assertEqual(fake.calls[1][-1], toolchain.tag())                    # located by tag ...
        run = fake.calls[2]
        self.assertEqual(run[-3:], [GOOD_ID, "cat", toolchain.MANIFEST_PATH])  # ... read by ID
        for flag in ("--network", "--read-only", "--cap-drop"):
            self.assertIn(flag, run)

    def test_missing_engine_image_and_a_remote_endpoint(self):
        with mock.patch.object(toolchain.shutil, "which", return_value=None):
            with self.assertRaises(BenchError) as ctx:
                toolchain.image()
        self.assertEqual(ctx.exception.code, Code.TOOLCHAIN_MISSING)
        self.engine(present=False)
        with mock.patch.dict(os.environ, {"DOCKER_HOST": "unix:///var/run/docker.sock"}):
            with self.assertRaises(BenchError) as ctx:
                toolchain.image()
        self.assertEqual(ctx.exception.code, Code.TOOLCHAIN_MISSING)
        for host in ("tcp://build.example:2376", "ssh://user@host"):
            with self.subTest(host=host), mock.patch.dict(os.environ, {"DOCKER_HOST": host}):
                with self.assertRaises(BenchError) as ctx:
                    toolchain.image()
                self.assertEqual((ctx.exception.code, "local socket" in ctx.exception.detail),
                                 (Code.TOOLCHAIN_MISSING, True))
        self.assertIsNone(toolchain._state["image"])

    def test_an_image_that_differs_is_a_mismatch(self):
        self.identity(b"stored manifest")
        cases = {"content manifest": dict(manifest=b"another manifest"),
                 "config": dict(config={"Env": ["PATH=/usr/bin"], "Entrypoint": None, "User": ""}),
                 "entrypoint": dict(config=dict(toolchain.spec()["config"], Entrypoint=["/bin/sh"]))}
        for name, kw in cases.items():
            with self.subTest(case=name):
                toolchain.reset()
                self.engine(**kw)
                with mock.patch.dict(os.environ, {"DOCKER_HOST": "unix:///run/docker.sock"}):
                    with self.assertRaises(BenchError) as ctx:
                        toolchain.image()
                self.assertEqual(ctx.exception.code, Code.TOOLCHAIN_MISMATCH)

    def test_the_client_environment_is_closed(self):
        with mock.patch.dict(os.environ, {"DOCKER_HOST": "unix:///x.sock", "MDF_SECRET_KEY": "k", "INFERENCE_API_KEY": "k",
                                          "HTTPS_PROXY": "http://p"}):
            env = toolchain.engine_env()
        self.assertEqual(set(env) - {"XDG_RUNTIME_DIR", "DOCKER_CONFIG", "DOCKER_CONTEXT"}, {"PATH", "HOME", "LC_ALL", "DOCKER_HOST"})

    def test_verify_entry_of_the_full_check(self):
        self.engine(present=False)
        out = io.StringIO()
        with mock.patch.dict(os.environ, {"DOCKER_HOST": "unix:///run/docker.sock"}), contextlib.redirect_stdout(out):
            self.assertEqual(toolchain.main(["verify"]), 1)
        self.assertIn("toolchain toolchain_missing", out.getvalue())
        self.assertIn("mdf bench prepare --toolchain", out.getvalue())
        self.assertEqual(toolchain.main(["other"]), 2)


class ReportTests(Base):
    def test_no_container_started_means_engine_null(self):
        r = toolchain.report()
        self.assertEqual((r["image_id"], r["engine"], r["reference_image"]), (None, None, None))
        self.assertEqual(r["content_sha256"], toolchain.identity())
        toolchain._state.update(image=toolchain.spec()["image_id"], engine="29.0")
        r = toolchain.report()
        self.assertEqual((r["engine"], r["reference_image"]), ({"name": "docker", "version": "29.0"}, True))

    def test_compare_flags_another_toolchain_even_with_equal_checks(self):
        a = {"checks_sha256": "c" * 64, "derived": "d" * 64, "toolchain": toolchain.report()}
        b = json.loads(json.dumps(a))
        self.assertFalse(toolchain.differs(a["toolchain"], b["toolchain"]))
        b["toolchain"]["content_sha256"] = "0" * 64
        self.assertEqual((a["checks_sha256"], a["derived"]), (b["checks_sha256"], b["derived"]))
        self.assertTrue(toolchain.differs(a["toolchain"], b["toolchain"]))
        self.assertTrue(toolchain.differs(a["toolchain"], None))

    def test_packaged_identity(self):
        spec = toolchain.spec()
        self.assertEqual(toolchain.tag(), "mdf-bench-toolchain:" + spec["content_sha256"][:16])
        self.assertEqual(set(spec["versions"]), {"ddjvu", "qpdf", "pdftoppm", "pdftotext", "pdfinfo", "pandoc", "soffice", "gs"})
        self.assertTrue(spec["builder"]["image"].startswith("moby/buildkit@sha256:"))
        with self.assertRaises(ValueError):
            toolchain._validate(dict(spec, extra=1))


class CliTests(Base):
    def test_toolchain_errors_print_the_code_and_a_fixed_hint(self):
        err = io.StringIO()
        with mock.patch.object(toolchain, "prepare", side_effect=BenchError(Code.TOOLCHAIN_MISMATCH, "/srv/private/secret path")), \
                contextlib.redirect_stderr(err):
            self.assertEqual(cli.main(["prepare", "--toolchain", "--cache", "/nonexistent-cache"]), 1)
        self.assertIn("error: toolchain_mismatch (", err.getvalue())
        self.assertNotIn("/srv/private", err.getvalue())                    # details stay local
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(cli.main(["prepare", "--cache", "/nonexistent-cache"]), 2)

    def test_a_fatal_error_stops_the_command(self):
        calls = []

        def resolve(doc, m, cache, transport, offline=True):
            calls.append(doc["id"])
            raise BenchError(Code.TOOL_FAILED, "container left", fatal=True)
        err = io.StringIO()
        with mock.patch("markdown_for_llms.bench.fetch.resolve_document", resolve), contextlib.redirect_stderr(err), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(cli.main(["fetch", "--offline", "--docs", "D01,D02", "--cache", "/nonexistent-cache"]), 1)
        self.assertEqual(calls, ["D01"])
        self.assertIn("error: tool_failed", err.getvalue())


class RealImageTests(unittest.TestCase):
    def test_the_local_image_is_the_reference_toolchain(self):
        """The image of the full check: verified against toolchain.json and called by ID."""
        image_id = require_toolchain()
        self.assertRegex(image_id, r"^sha256:[0-9a-f]{64}$")
        r = toolchain.report()
        self.assertEqual((r["image_id"], r["engine"]["name"]), (image_id, "docker"))


if __name__ == "__main__":
    unittest.main()
