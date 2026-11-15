"""Formats by extension and signature, the matrix of routes and the closed codes (A1 of MFL032)."""
import os
from pathlib import Path
import tempfile
import unittest

from engine_test_support import FIXTURES, TEXT_DJVU, make_docx, make_odt, make_xlsx, para, png, write_zip
from markdown_for_llms.engine import formats, routes
from markdown_for_llms.engine.codes import Code, EngineError


class Base(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)

    def file(self, name, data):
        p = self.root / name
        p.write_bytes(data)
        return p

    def code(self, path, name=None):
        with self.assertRaises(EngineError) as ctx:
            formats.detect(path, name)
        return ctx.exception.code


class DetectTests(Base):
    def test_every_format_of_the_matrix_by_extension_and_signature(self):
        cases = {
            "a.pdf": (FIXTURES / "C06-pdf-text-it.pdf").read_bytes(),
            "a.png": png(), "a.jpg": b"\xff\xd8\xff\xe0" + b"\0" * 20, "a.TIFF": b"II*\x00" + b"\0" * 20,
            "a.djvu": TEXT_DJVU, "a.doc": b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + b"\0" * 20,
            "a.rtf": b"{\\rtf1 hello}", "a.html": b"\xef\xbb\xbf  \n<!DOCTYPE html><html></html>",
            "a.htm": b"<HTML><body>x</body></HTML>", "a.txt": "testo è".encode("cp1252"),
        }
        expected = {"a.pdf": "pdf", "a.png": "png", "a.jpg": "jpeg", "a.TIFF": "tiff", "a.djvu": "djvu", "a.doc": "doc",
                    "a.rtf": "rtf", "a.html": "html", "a.htm": "html", "a.txt": "txt"}
        for name, data in cases.items():
            with self.subTest(name=name):
                info = formats.detect(self.file(name, data))
                self.assertEqual(info["format"], expected[name])
                self.assertEqual(info["family"], formats.family(expected[name]))
        self.assertEqual(formats.detect(FIXTURES / "C02-docx-it.docx")["format"], "docx")
        self.assertEqual(formats.detect(FIXTURES / "C03-epub-en.epub")["format"], "epub")
        self.assertEqual(formats.detect(make_xlsx(self.root / "s.xlsx"))["format"], "xlsx")
        self.assertEqual(formats.detect(make_odt(self.root / "t.odt"))["format"], "odt")

    def test_a_zip_package_is_read_from_its_central_directory(self):
        """K3: the central directory is at the end of the file; a docx whose first member is a large
        stored one, so that no OOXML name is in its first 8 KiB, is still a docx (HA3, Sol H3)."""
        path = make_docx(self.root / "big.docx", para("x"), stored_first=("word/media/blob.bin", os.urandom(64 * 1024)))
        head = path.read_bytes()[:8192]
        for name in (b"[Content_Types].xml", b"word/document.xml", b"_rels/.rels"):
            self.assertNotIn(name, head)
        self.assertEqual(formats.detect(path)["signature"], "ZIP word/document.xml")

    def test_mismatch_unsupported_and_unreadable(self):
        pdf = (FIXTURES / "C06-pdf-text-it.pdf").read_bytes()
        self.assertEqual(self.code(self.file("fake.docx", pdf)), Code.FORMAT_MISMATCH)        # K3: PDF bytes named .docx
        self.assertEqual(self.code(self.file("fake.pdf", b"{\\rtf1}")), Code.FORMAT_MISMATCH)
        self.assertEqual(self.code(make_xlsx(self.root / "sheet.docx")), Code.FORMAT_MISMATCH)  # another OOXML package
        self.assertEqual(self.code(self.file("x.txt", b"a\x00b")), Code.FORMAT_MISMATCH)
        self.assertEqual(self.code(self.file("x.md", b"# hi")), Code.FORMAT_UNSUPPORTED)
        self.assertEqual(self.code(self.file("noext", b"hi")), Code.FORMAT_UNSUPPORTED)
        truncated = (FIXTURES / "C02-docx-it.docx").read_bytes()[:3000]
        self.assertEqual(self.code(self.file("broken.docx", truncated)), Code.INPUT_UNREADABLE)
        self.assertEqual(self.code(self.root / "absent.pdf"), Code.INPUT_UNREADABLE)

    def test_the_mimetype_member_is_bounded(self):
        path = write_zip(self.root / "x.odt", {"content.xml": "<a/>"},
                         stored_first=("mimetype", b"application/vnd.oasis.opendocument.text" + b" " * 200))
        self.assertEqual(self.code(path), Code.FORMAT_MISMATCH)

    def test_the_name_decides_the_extension(self):
        path = self.file("upload", b"{\\rtf1 x}")
        self.assertEqual(formats.detect(path, "doc.rtf")["format"], "rtf")


class RouteTests(unittest.TestCase):
    def test_matrix(self):
        self.assertEqual(routes.routes_for("docx"), {"default": "pandoc", "alternatives": ["marker"]})
        self.assertEqual(routes.routes_for("djvu"), {"default": "marker", "alternatives": ["djvutxt", "datalab"]})
        self.assertEqual(routes.routes_for("txt")["default"], "text")
        for fmt in ("pdf", "png", "jpeg", "tiff"):
            self.assertEqual(routes.routes_for(fmt), {"default": "marker", "alternatives": ["datalab"]})
        self.assertEqual(set(routes.MATRIX), set(formats.FORMATS))

    def test_resolve(self):
        self.assertEqual(routes.resolve("docx"), ("pandoc", "matrix", "matrix-default"))
        self.assertEqual(routes.resolve("docx", "pandoc"), ("pandoc", "matrix", "imposed"))
        self.assertEqual(routes.resolve("djvu", "djvutxt"), ("djvutxt", "user", "imposed"))
        for fmt, route in (("docx", "djvutxt"), ("txt", "marker"), ("pdf", "pandoc"), ("odt", "nonsense")):
            with self.subTest(fmt=fmt, route=route), self.assertRaises(EngineError) as ctx:
                routes.resolve(fmt, route)
            self.assertEqual(ctx.exception.code, Code.ROUTE_INVALID)

    def test_options_not_accepted_are_refused(self):
        routes.check_options("marker", {"ocr": "force"})
        for converter in ("pandoc", "text", "djvutxt", "datalab"):
            with self.subTest(converter=converter), self.assertRaises(EngineError) as ctx:
                routes.check_options(converter, {"ocr": "force"})
            self.assertEqual(ctx.exception.code, Code.ROUTE_OPTION_UNSUPPORTED)
        with self.assertRaises(EngineError):
            routes.check_options("pandoc", {"ocr": "maybe"})


class SharedCodesTests(unittest.TestCase):
    """K2 (PA2): the bench shares the exception and the values of the engine codes."""

    def test_bench_error_is_engine_error_and_codes_compare_equal(self):
        from markdown_for_llms.bench.codes import BenchError, Code as BenchCode
        self.assertIs(BenchError, EngineError)
        for member in Code:
            self.assertEqual(BenchCode[member.name], member)
            self.assertEqual(hash(BenchCode[member.name]), hash(member))
            self.assertEqual(BenchCode[member.name].value, member.value)
        self.assertEqual(BenchCode.DERIVED_DIFFERS.value, "derived_differs_from_reference")
        self.assertEqual(len({m.value for m in BenchCode}), len(list(BenchCode)))

    def test_the_engine_imports_nothing_of_the_bench(self):
        root = Path(__file__).resolve().parents[3] / "markdown_for_llms" / "engine"
        for path in sorted(root.rglob("*.py")):
            text = path.read_text(encoding="utf-8")
            with self.subTest(module=path.name):
                self.assertNotIn("from ..bench", text)
                self.assertNotIn("markdown_for_llms.bench", text)

    def test_formats_and_routes_need_the_standard_library_only(self):
        import subprocess
        import sys
        code = ("import sys; sys.path.insert(0, sys.argv[1]); import markdown_for_llms.engine.formats, "
                "markdown_for_llms.engine.routes; heavy = {'marker', 'surya', 'torch', 'PIL', 'pypdfium2', "
                "'onnxruntime', 'charset_normalizer', 'markdown_it'} & set(m.split('.')[0] for m in sys.modules); "
                "print(sorted(heavy))")
        out = subprocess.run([sys.executable, "-I", "-B", "-c", code, str(Path(__file__).resolve().parents[3])],
                             capture_output=True, text=True, timeout=60)
        self.assertEqual(out.stdout.strip(), "[]", out.stderr)


if __name__ == "__main__":
    unittest.main()
