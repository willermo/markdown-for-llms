"""Structural guards of the bench: imports, skips, the JUnit guard of the full check and the build
context of the reference toolchain."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[3]
BENCH = ROOT / "markdown_for_llms" / "bench"
TESTS = Path(__file__).resolve().parent

spec = importlib.util.spec_from_file_location("bench_junit_guard", ROOT / "scripts" / "bench_junit_guard.py")
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)

# Engines and their libraries: the bench runs Marker and Surya only as child processes.
ENGINE_MODULES = {"marker", "surya", "torch", "torchvision", "transformers", "pdftext", "pypdfium2"}
# Libraries Marker uses to read the same formats: a twin reader or a self-check built on them
# would not be independent of the converter (A3 of the plan).
MARKER_FORMAT_LIBRARIES = {"openpyxl", "pptx", "mammoth", "ebooklib", "weasyprint", "lxml", "bs4", "markdownify"}


def imported_roots(path):
    tree = ast.parse(path.read_text(encoding="utf-8"), str(path))
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])
        elif isinstance(node, ast.Call) and getattr(node.func, "attr", getattr(node.func, "id", "")) in {"import_module", "__import__"}:
            if node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                roots.add(node.args[0].value.split(".")[0])
    return roots


class ImportGuardTests(unittest.TestCase):
    def test_bench_never_imports_an_engine(self):
        files = sorted(BENCH.rglob("*.py"))
        self.assertGreater(len(files), 3)
        for path in files:
            with self.subTest(path=path.relative_to(ROOT).as_posix()):
                self.assertFalse(imported_roots(path) & ENGINE_MODULES)

    def test_readers_and_self_check_avoid_marker_format_libraries(self):
        files = sorted((BENCH / "readers").rglob("*.py")) + sorted(BENCH.glob("selfcheck*.py")) + sorted(BENCH.glob("inventory*.py"))
        for path in files:
            with self.subTest(path=path.relative_to(ROOT).as_posix()):
                self.assertFalse(imported_roots(path) & MARKER_FORMAT_LIBRARIES)

    def test_guard_detects_a_forbidden_import(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "x.py"
            p.write_text("import json\nfrom openpyxl import load_workbook\nimportlib.import_module('marker.models')\n")
            self.assertEqual(imported_roots(p), {"json", "openpyxl", "marker"})


SKIP_NAMES = {"skip", "skipIf", "skipUnless", "SkipTest", "skipTest", "importorskip", "pytestmark",
              "allow_module_level", "xfail"}


def skip_references(path):
    tree = ast.parse(path.read_text(encoding="utf-8"), str(path))
    found = []
    for node in ast.walk(tree):
        name = node.attr if isinstance(node, ast.Attribute) else node.id if isinstance(node, ast.Name) else \
            node.arg if isinstance(node, ast.keyword) else None
        if name in SKIP_NAMES:
            found.append((name, getattr(node, "lineno", 0)))
    return found


class SkipGuardTests(unittest.TestCase):
    def test_bench_tests_never_skip(self):
        files = sorted(TESTS.glob("*.py"))
        self.assertGreater(len(files), 1)
        for path in files:
            with self.subTest(path=path.name):
                self.assertEqual(skip_references(path), [])

    def test_skip_detector(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "t.py"
            p.write_text("import pytest\npytestmark = pytest.mark.skip\npytest.skip('x', allow_module_level=True)\n")
            self.assertEqual({n for n, _ in skip_references(p)}, {"pytestmark", "skip", "allow_module_level"})


def junit(cases):
    body = "".join(
        f'<testcase classname="{c}" name="{n}" time="0">' + (f'<skipped type="pytest.skip" message="{m}">x</skipped>' if m else "")
        + "</testcase>" for c, n, m in cases)
    return f'<testsuites><testsuite name="pytest" tests="{len(cases)}">{body}</testsuite></testsuites>'


class JUnitGuardTests(unittest.TestCase):
    """C8: the guard is a contract; each way it must fail is exercised here."""

    def run_guard(self, cases):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "fast-suite.xml"
            p.write_text(junit(cases))
            return guard.check(p)

    def test_green_when_bench_tests_run_and_none_skips(self):
        ran, problems = self.run_guard([
            ("tests.unit.bench.test_bench_x.T", "a", None),
            ("tests.unit.bench.test_bench_x.T", "b", None),
            ("tests.unit.test_other.T", "c", "any other reason"),
        ])
        self.assertEqual((ran, problems), (2, []))

    def test_red_cases(self):
        cases = {
            "no bench test": [("tests.unit.test_other.T", "a", None)],
            "all bench tests skipped": [("tests.unit.bench.test_bench_x.T", "a", "bench-tool-missing: gs")],
            "one skip for a tool": [("tests.unit.bench.test_bench_x.T", "a", None),
                                    ("tests.unit.bench.test_bench_x.T", "b", "bench-tool-missing: pandoc")],
            "one skip for another reason": [("tests.unit.bench.test_bench_x.T", "a", None),
                                            ("tests.unit.bench.test_bench_x.T", "b", "network unavailable")],
        }
        for name, case in cases.items():
            with self.subTest(case=name):
                self.assertTrue(self.run_guard(case)[1])

    def test_main_exit_codes(self):
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "bad.xml"
            bad.write_text("<testsuites>")
            self.assertEqual(guard.main(["g", str(bad)]), 2)
            red = Path(tmp) / "red.xml"
            red.write_text(junit([("tests.unit.test_other.T", "a", None)]))
            self.assertEqual(guard.main(["g", str(red)]), 1)


CONTEXT = ROOT / "scripts" / "bench-toolchain"
TOOLCHAIN_JSON = ROOT / "markdown_for_llms" / "engine" / "data" / "toolchain.json"


class ToolchainContextTests(unittest.TestCase):
    """E4: the build context stays out of the wheel; toolchain.json, in the wheel, records its SHA-256
    and `prepare --toolchain` refuses a context that differs."""

    def test_context_matches_toolchain_json(self):
        recorded = json.loads(TOOLCHAIN_JSON.read_text())["context"]
        found = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(CONTEXT.iterdir())}
        self.assertEqual(found, recorded)

    def test_context_is_not_in_the_wheel_and_the_identity_is(self):
        spec = importlib.util.spec_from_file_location(
            "make_source_manifest", ROOT / "scripts" / "diagnostics" / "run-a001-fase0-uv" / "make_source_manifest.py")
        manifest = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(manifest)
        files = {p.relative_to(ROOT).as_posix() for p in manifest.application_files(ROOT)}
        self.assertIn("markdown_for_llms/engine/data/toolchain.json", files)
        self.assertFalse([f for f in files if "bench-toolchain" in f])

    def test_prepare_refuses_a_context_that_differs(self):
        from markdown_for_llms.bench import toolchain
        from markdown_for_llms.bench.codes import BenchError, Code
        with tempfile.TemporaryDirectory() as tmp:
            ctx = Path(tmp) / "ctx"
            ctx.mkdir()
            for p in CONTEXT.iterdir():
                (ctx / p.name).write_bytes(p.read_bytes())
            self.assertEqual(set(toolchain.verify_context(ctx)), set(json.loads(TOOLCHAIN_JSON.read_text())["context"]))
            changes = {
                "edited file": lambda: (ctx / "Dockerfile").write_bytes((ctx / "Dockerfile").read_bytes() + b"# x\n"),
                "extra file": lambda: (ctx / "extra.txt").write_text("x"),
                "missing file": lambda: (ctx / "pixi.lock").unlink(),
            }
            for name, change in changes.items():
                with self.subTest(change=name):
                    backup = {p.name: p.read_bytes() for p in ctx.iterdir()}
                    change()
                    with self.assertRaises(BenchError) as exc:
                        toolchain.verify_context(ctx)
                    self.assertEqual(exc.exception.code, Code.TOOLCHAIN_MISMATCH)
                    for p in ctx.iterdir():
                        p.unlink()
                    for n, data in backup.items():
                        (ctx / n).write_bytes(data)
            with self.assertRaises(BenchError) as exc:
                toolchain.verify_context(Path(tmp) / "absent")
            self.assertEqual(exc.exception.code, Code.TOOLCHAIN_MISSING)

    def test_full_check_verifies_the_image_before_the_suites(self):
        text = (ROOT / "scripts" / "full-check.sh").read_text()
        x = text.index("-m markdown_for_llms.engine.toolchain verify")
        self.assertLess(text.index('echo "F: PASS_CURRENT_I"'), x)
        self.assertLess(x, text.index("run_suite fast-suite"))
        self.assertIn('|| stop "X: ', text[x:x + 200])
