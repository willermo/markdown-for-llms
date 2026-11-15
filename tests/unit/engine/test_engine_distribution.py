"""K1 of MFL032: the engine as installed from the canonical wheel, and the guard of the full check.

The full check reinstalls the wheel it has just built into the development venv before the
suite; this test reads the installed distribution (not the checkout, which `tests/conftest.py`
puts first on `sys.path`) and converts a synthetic docx with it in a separate interpreter.
"""
import importlib.metadata
import importlib.util
import json
import subprocess
import sys
import sysconfig
import tempfile
import unittest
from pathlib import Path

from engine_test_support import ROOT, drawing, make_docx, para, png, require_toolchain

PACKAGE_DATA = ("markdown_for_llms/engine/filters/contract.lua", "markdown_for_llms/engine/data/toolchain.json",
                "markdown_for_llms/engine/filters/references.lua")


class InstalledWheelTests(unittest.TestCase):
    def setUp(self):
        # only the site-packages of this interpreter: the checkout, first on sys.path, may hold an egg-info
        found = [d for d in importlib.metadata.distributions(path=[sysconfig.get_path("purelib")])
                 if d.metadata["Name"] == "markdown-for-llms"]
        if len(found) != 1:
            self.fail("markdown-for-llms is not installed: run the full check, which installs the canonical wheel")
        self.dist = found[0]

    def test_package_data_of_the_engine_is_installed_and_equal_to_the_source(self):
        listed = {str(p) for p in self.dist.files or []}
        for name in PACKAGE_DATA:
            with self.subTest(name=name):
                self.assertIn(name, listed)
                self.assertEqual(Path(self.dist.locate_file(name)).read_bytes(), (ROOT / name).read_bytes())
        self.assertIn("charset-normalizer>=3.5.2", self.dist.requires or [])

    def test_a_conversion_with_the_installed_engine(self):
        require_toolchain()
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            src = make_docx(tmp / "w.docx", para("Installed engine text.") + f"<w:p><w:r>{drawing('rId1')}</w:r></w:p>",
                            media={"fig.png": png()})
            code = ("import json, sys; from markdown_for_llms.engine import core, isolation, native; "
                    "r = core.convert(sys.argv[1], sys.argv[2], {}, {'memory_mib': 1024}); "
                    "print(json.dumps({'core': core.__file__, 'child': str(isolation.PACKAGE), 'filter': str(native.FILTER), "
                    "'status': r['status'], 'assets': len([i for i in r.get('inventory', []) if i['path'].startswith('assets/')])}))")
            proc = subprocess.run([sys.executable, "-I", "-B", "-c", code, str(src), str(tmp / "out")], cwd=tmp,
                                  capture_output=True, text=True, timeout=300)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            got = json.loads(proc.stdout)
        installed = Path(self.dist.locate_file("markdown_for_llms")).resolve()
        self.assertTrue(Path(got["core"]).resolve().is_relative_to(installed), got)
        self.assertNotEqual(installed, (ROOT / "markdown_for_llms").resolve())
        self.assertEqual(Path(got["child"]).resolve(), installed)
        self.assertEqual((got["status"], got["assets"]), ("complete", 1))
        self.assertEqual(Path(got["filter"]).read_bytes(), (ROOT / PACKAGE_DATA[0]).read_bytes())


class GuardTests(unittest.TestCase):
    """The JUnit guard counts the engine tests apart from the bench tests."""

    def setUp(self):
        spec = importlib.util.spec_from_file_location("bench_junit_guard", ROOT / "scripts" / "bench_junit_guard.py")
        self.guard = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.guard)

    def junit(self, cases):
        body = "".join(f'<testcase classname="{c}" name="{n}">' + (f'<skipped message="{m}"/>' if m else "") + "</testcase>"
                       for c, n, m in cases)
        f = tempfile.NamedTemporaryFile("w", suffix=".xml", delete=False)
        self.addCleanup(Path(f.name).unlink)
        f.write(f"<testsuites><testsuite>{body}</testsuite></testsuites>")
        f.close()
        return f.name

    def test_both_groups_must_run_without_skips(self):
        green = self.junit([("tests.unit.bench.test_x.T", "a", None), ("tests.unit.engine.test_y.T", "b", None)])
        self.assertEqual(self.guard.main(["g", green]), 0)
        for cases in ([("tests.unit.bench.test_x.T", "a", None)],
                      [("tests.unit.bench.test_x.T", "a", None), ("tests.unit.engine.test_y.T", "b", "no image")]):
            with self.subTest(cases=cases):
                self.assertEqual(self.guard.main(["g", self.junit(cases)]), 1)
        ran, problems = self.guard.check(green, "tests.unit.engine.", "engine")
        self.assertEqual((ran, problems), (1, []))

    def test_full_check_verifies_the_engine_toolchain(self):
        text = (ROOT / "scripts" / "full-check.sh").read_text()
        self.assertIn("-m markdown_for_llms.engine.toolchain verify", text)
        self.assertNotIn("markdown_for_llms.bench.toolchain", text)


if __name__ == "__main__":
    unittest.main()
