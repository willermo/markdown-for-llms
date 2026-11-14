"""Derivations: PDF date normalization, qpdf pages, pdftoppm PNG, ddjvu, references and page counts.

The derivations run for real in the reference toolchain image (`require_toolchain`)."""
import copy
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock
from unittest.mock import patch

from markdown_for_llms.bench import checks, derive, toolchain, tools
from markdown_for_llms.bench.cache import Cache
from markdown_for_llms.bench.codes import BenchError, Code

from bench_test_support import make_pdf, require_toolchain
from markdown_for_llms.bench.selftest import SYNTHETIC_DJVU


def sha(data):
    return hashlib.sha256(data).hexdigest()


class DateNormalizationTests(unittest.TestCase):
    def test_dates_fixed_and_length_kept(self):
        a = make_pdf(["one"], date=b"D:20261114005345")
        b = make_pdf(["one"], date=b"D:20261114005347")
        self.assertNotEqual(a, b)
        na, nb = derive.normalize_pdf_dates(a), derive.normalize_pdf_dates(b)
        self.assertEqual(na, nb)
        self.assertEqual(len(na), len(a))
        self.assertIn(b"/CreationDate (D:19700101000000)", na)
        self.assertEqual(derive.normalize_pdf_dates(na), na)
        tz = derive.normalize_pdf_dates(make_pdf(["one"], date=b"D:20261114005345+01'00'"))
        self.assertIn(b"(D:19700101000000+00'00')", tz)
        other = b"BT (D:20261114005345) Tj ET /Title (D:20261114005345)"
        self.assertEqual(derive.normalize_pdf_dates(other), other)

    def test_djvu_step_normalizes_the_ddjvu_output(self):
        def fake_run(name, args, work_root, timeout=0):
            args[-1].dest.write_bytes(make_pdf(["scan"], date=b"D:20261114005345"))
            return b""
        with tempfile.TemporaryDirectory() as tmp, patch.object(tools, "run", fake_run):
            out = derive._run_step({"op": "djvu-to-pdf"}, Path(tmp) / "in.djvu", Path(tmp), Path(tmp))
            self.assertIn(b"D:19700101000000", out.read_bytes())
            self.assertFalse((Path(tmp) / "ddjvu.pdf").exists())


class ToolDerivationTests(unittest.TestCase):
    def setUp(self):
        require_toolchain()
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.cache = Cache(self.root / "cache")
        self.pdf = self.root / "src.pdf"
        self.pdf.write_bytes(make_pdf(["alpha", "beta", "gamma", "delta"]))

    def doc(self, steps, pages, reference=None):
        inp = {"source": "s", "derive": steps}
        if reference:
            inp["reference"] = {"sha256": reference, "toolchain": toolchain.identity()}
        return {"id": "D99", "pages": pages, "input": inp}

    def test_pdf_pages_deterministic_with_page_check(self):
        steps = [{"op": "pdf-pages", "first": 2, "last": 3}]
        out, warnings, meta = derive.derive_input(self.doc(steps, 2), self.pdf, self.cache)
        self.assertEqual((meta["pages"], warnings), (2, []))
        other = Cache(self.root / "other")
        out2, _, meta2 = derive.derive_input(self.doc(steps, 2), self.pdf, other)
        self.assertEqual(sha(out.read_bytes()), sha(out2.read_bytes()))
        self.assertEqual(meta["toolchain"], toolchain.identity())
        with self.assertRaises(BenchError) as ctx:
            derive.derive_input(self.doc([{"op": "pdf-pages", "first": 1, "last": 3}], 2), self.pdf, self.cache)
        self.assertEqual(ctx.exception.code, Code.PAGES_MISMATCH)

    def test_reference_mismatch_is_a_warning_and_altered_output_is_redone(self):
        steps = [{"op": "pdf-pages", "first": 1, "last": 1}]
        out, warnings, meta = derive.derive_input(self.doc(steps, 1, reference="0" * 64), self.pdf, self.cache)
        self.assertEqual(warnings, [Code.DERIVED_DIFFERS])
        good = out.read_bytes()
        out.write_bytes(good + b"%altered")
        out2, warnings, _ = derive.derive_input(self.doc(steps, 1, reference=sha(good)), self.pdf, self.cache)
        self.assertEqual((out2.read_bytes(), warnings), (good, []))

    def test_page_png(self):
        steps = [{"op": "pdf-pages", "first": 1, "last": 4}, {"op": "page-png", "page": 2, "dpi": 72, "color": "rgb"}]
        out, _, meta = derive.derive_input(self.doc(steps, 1), self.pdf, self.cache)
        data = out.read_bytes()
        self.assertTrue(data.startswith(b"\x89PNG\r\n\x1a\n"))
        out2, _, _ = derive.derive_input(self.doc(steps, 1), self.pdf, Cache(self.root / "other"))
        self.assertEqual(data, out2.read_bytes())
        self.assertEqual(meta["toolchain"], toolchain.identity())

    def test_djvu_to_pdf_with_normalized_dates(self):
        djvu = self.root / "box.djvu"
        djvu.write_bytes(SYNTHETIC_DJVU)
        out, warnings, meta = derive.derive_input(self.doc([{"op": "djvu-to-pdf"}], 1), djvu, self.cache)
        data = out.read_bytes()
        self.assertTrue(data.startswith(b"%PDF"))
        self.assertIn(b"(D:19700101000000", data)
        out2, _, _ = derive.derive_input(self.doc([{"op": "djvu-to-pdf"}], 1), djvu, Cache(self.root / "other"))
        self.assertEqual(data, out2.read_bytes())       # ddjvu writes the current time; normalized away


class CopiedCacheWithoutEngineTests(unittest.TestCase):
    """E5 (U4): a cache prepared where Docker runs and copied where it does not (a pod): the keys come
    from toolchain.json, a hit starts no container and reports `engine: null`, and a cache that is
    incomplete or altered fails closed instead of being used."""

    def setUp(self):
        require_toolchain()
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.pdf = self.root / "src.pdf"
        self.pdf.write_bytes(make_pdf(["alpha", "beta", "gamma"]))
        self.doc = {"id": "D99", "pages": 2, "input": {"source": "s", "derive": [{"op": "pdf-pages", "first": 1, "last": 2}]}}
        prepared = Cache(self.root / "prepared")
        self.out, _, self.meta = derive.derive_input(self.doc, self.pdf, prepared)
        shutil.copytree(prepared.root, self.root / "pod")
        self.cache = Cache(self.root / "pod")
        saved = dict(toolchain._state)
        self.addCleanup(toolchain._state.update, saved)
        toolchain.reset()

    def no_engine(self):
        """No Docker on the PATH, and any attempt to call it is a failure of the test."""
        return mock.patch.multiple(toolchain, engine_path=mock.Mock(side_effect=BenchError(Code.TOOLCHAIN_MISSING, "docker not found")),
                                   engine_run=mock.Mock(side_effect=AssertionError("engine called")))

    def test_a_copied_cache_is_used_without_docker(self):
        with self.no_engine():
            out, warnings, meta = derive.derive_input(self.doc, self.pdf, self.cache)
            self.assertEqual((sha(out.read_bytes()), meta), (sha(self.out.read_bytes()), self.meta))
            self.assertTrue(out.is_relative_to(self.cache.root))
            key = checks.cache_key({**self.doc, "twin": {"kind": "latex"}}, {"source": "0" * 64, "twin": "1" * 64, "derived": meta})
            self.assertRegex(key, r"^[0-9a-f]{64}$")
            self.assertIsNone(toolchain.report()["engine"])        # no container started

    def test_an_altered_or_incomplete_copy_fails_closed(self):
        target = self.cache.derived_dir / "D99"
        (output,) = target.glob("*/input.pdf")
        cases = {"altered output": lambda: output.write_bytes(output.read_bytes() + b"%altered"),
                 "altered meta": lambda: output.with_name("meta.json").write_text("{not json"),
                 "meta of another toolchain": lambda: output.with_name("meta.json").write_text(
                     json.dumps(dict(self.meta, toolchain="0" * 64))),
                 "missing derivation": lambda: shutil.rmtree(target)}
        for name, alter in cases.items():
            with self.subTest(case=name):
                shutil.rmtree(self.cache.root)
                shutil.copytree(self.root / "prepared", self.cache.root)
                (output,) = target.glob("*/input.pdf")
                alter()
                with self.no_engine(), self.assertRaises(BenchError) as ctx:
                    derive.derive_input(self.doc, self.pdf, self.cache)
                self.assertEqual(ctx.exception.code, Code.TOOLCHAIN_MISSING)
