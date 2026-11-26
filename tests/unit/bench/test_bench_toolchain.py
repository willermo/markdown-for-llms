"""Reference toolchain: identity, the image the launcher verifies (E2), the candidate of `prepare`, reports.

The verification of the local image (located by its tag, its config and its content manifest compared with
`toolchain.json`, then started by ID) is the launcher's: `tests/unit/launcher/test_launcher_spec.py`
(`TestToolchainRole`) and, on a real daemon, `tests/container/`. Here what a process of the engine does with
its answers.
"""
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from bench_test_support import require_toolchain, scripted
from markdown_for_llms.bench import cli, toolchain
from markdown_for_llms.bench.codes import BenchError, Code
from markdown_for_llms.engine import containers

GOOD_ID = "sha256:" + "a" * 64


class Base(unittest.TestCase):
    def setUp(self):
        saved = dict(toolchain._state)
        self.addCleanup(toolchain._state.update, saved)
        toolchain.reset()
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)

    def launcher(self, script=None, **answer):
        return scripted(self, self.root / "volume", script, toolchain=answer)

    def identity(self, manifest):
        patcher = mock.patch.object(toolchain, "identity", return_value=hashlib.sha256(manifest).hexdigest())
        patcher.start()
        self.addCleanup(patcher.stop)


class VerificationTests(Base):
    def test_asked_of_the_launcher_once_then_called_by_id(self):
        fake = self.launcher(image=GOOD_ID, docker="29.0")
        self.assertEqual(toolchain.image(), GOOD_ID)
        self.assertEqual(toolchain.image(), GOOD_ID)
        self.assertEqual(fake.calls, [("toolchain", None)])                    # once per process, and no container
        self.assertEqual((toolchain.role(), toolchain.report()["engine"]), ("toolchain", {"name": "docker", "version": "29.0"}))

    def test_a_missing_image_one_that_differs_and_no_launcher(self):
        cases = ((dict(refuse=("toolchain_missing", "no image mdf-bench-toolchain:x")), Code.TOOLCHAIN_MISSING),
                 (dict(refuse=("toolchain_mismatch", "content manifest")), Code.TOOLCHAIN_MISMATCH),
                 (dict(refuse=("toolchain_mismatch", "image config")), Code.TOOLCHAIN_MISMATCH),
                 (dict(verified=False), Code.TOOLCHAIN_MISMATCH),              # an image nobody verified is never used
                 (dict(image="mdf-bench-toolchain:latest"), Code.TOOLCHAIN_MISMATCH))     # an ID, never a tag
        for answer, code in cases:
            with self.subTest(answer=answer):
                toolchain.reset()
                self.launcher(**answer)
                with self.assertRaises(BenchError) as ctx:
                    toolchain.image()
                self.assertEqual(ctx.exception.code, code)
                self.assertIsNone(toolchain._state["image"])
        fake = self.launcher()
        fake.down = True
        with self.assertRaises(BenchError) as ctx:
            toolchain.image()
        self.assertEqual(ctx.exception.code, Code.ISOLATION_UNAVAILABLE)       # no launcher: nothing else is tried
        self.assertIsNone(toolchain._state["image"])

    def test_verify_entry_of_the_full_check(self):
        self.launcher(refuse=("toolchain_missing", "no image"))
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(toolchain.main(["verify"]), 1)
        self.assertIn("toolchain toolchain_missing", out.getvalue())
        self.assertIn("make prepare", out.getvalue())
        self.assertEqual(toolchain.main(["other"]), 2)
        toolchain.reset()
        self.launcher(image=toolchain.spec()["image_id"], docker="29.0")
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(toolchain.main(["verify"]), 0)
        self.assertIn("reference build yes docker 29.0", out.getvalue())


class CandidateTests(Base):
    """`prepare_candidate`: the image the host built, verified file by file through `manifest` children."""

    def manifests(self, stored=b"a  /usr/bin/x\nb  /usr/bin/y\n", regenerated=None, status=0):
        def script(request, call):
            self.assertEqual((request["kind"], request["image"], request["env"]), ("manifest", "candidate", {}))
            if request["argv"] == containers.MANIFEST_READ:
                return status, stored, b""
            self.assertEqual(request["argv"], containers.MANIFEST_REGENERATE)
            return 0, stored if regenerated is None else regenerated, b""
        return script

    def test_the_candidate_is_verified_file_by_file_and_the_self_test_runs_on_it(self):
        stored = b"a  /usr/bin/x\nb  /usr/bin/y\n"
        fake = self.launcher(self.manifests(stored), image=GOOD_ID, verified=False, docker="29.0")
        self.identity(stored)
        seen, lines = [], []

        def selftest(cache, out):
            seen.append((toolchain.role(), toolchain._state["image"], cache))
        self.assertEqual(toolchain.prepare_candidate("the cache", out=lines.append, selftest=selftest), GOOD_ID)
        self.assertEqual(seen, [("candidate", GOOD_ID, "the cache")])           # the tools of the self-test run on the candidate
        self.assertEqual((toolchain.role(), toolchain._state["image"]), ("toolchain", None))      # and nothing of it stays
        self.assertEqual(fake.calls[0], ("toolchain", "candidate"))
        self.assertEqual([r["argv"] for r in fake.requests], [containers.MANIFEST_READ, containers.MANIFEST_REGENERATE])
        self.assertEqual(lines[-1], "toolchain candidate ok %s %s docker 29.0" % (toolchain.tag(), GOOD_ID))
        self.assertIn("2 manifest lines, every file verified", lines[0])
        self.assertEqual(sorted(os.listdir(self.root / "volume" / "call")), [])

    def test_a_candidate_that_differs_is_a_mismatch_and_no_self_test_runs(self):
        stored = b"a  /usr/bin/x\n"
        cases = {"files": dict(script=self.manifests(stored, regenerated=b"c  /usr/bin/x\n"), identity=stored),
                 "content": dict(script=self.manifests(stored), identity=b"another manifest"),
                 "exit": dict(script=self.manifests(stored, status=1), identity=stored)}
        for name, case in cases.items():
            with self.subTest(case=name):
                toolchain.reset()
                self.launcher(case["script"], image=GOOD_ID, verified=False)
                self.identity(case["identity"])
                ran = []
                with self.assertRaises(BenchError) as ctx:
                    toolchain.prepare_candidate("cache", out=lambda line: None, selftest=lambda cache, out: ran.append(1))
                self.assertEqual((ctx.exception.code, ran), (Code.TOOLCHAIN_MISMATCH, []))
        self.launcher(refuse=("toolchain_missing", "no image mdf-bench-toolchain:candidate"))
        with self.assertRaises(BenchError) as ctx:
            toolchain.prepare_candidate("cache", out=lambda line: None, selftest=lambda cache, out: None)
        self.assertEqual(ctx.exception.code, Code.TOOLCHAIN_MISSING)

    def test_a_failed_self_test_leaves_no_candidate_behind(self):
        stored = b"a  /usr/bin/x\n"
        self.launcher(self.manifests(stored), image=GOOD_ID, verified=False)
        self.identity(stored)

        def selftest(cache, out):
            raise BenchError(Code.TOOL_FAILED, "self-test")
        with self.assertRaises(BenchError):
            toolchain.prepare_candidate("cache", out=lambda line: None, selftest=selftest)
        self.assertEqual((toolchain.role(), toolchain._state["image"]), ("toolchain", None))


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
        with mock.patch.object(toolchain, "prepare_candidate", side_effect=BenchError(Code.TOOLCHAIN_MISMATCH, "/srv/private/secret path")), \
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
