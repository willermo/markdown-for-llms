"""The `libreoffice` route (M5 of MFL026): xlsx and pptx through the HTML of LibreOffice, read by
pandoc with the contract filter, for real in the toolchain image.

pptx: the HTML of Impress has one `h1` per slide and no image; the primary images of the
pictures of each slide (`p:sldIdLst` order, never the SVG twin) follow the heading of their
slide, each occurrence once; when the headings are not as many as the slides no image is placed
and each is `media_lost` with `slide_boundaries_unknown`. xlsx: the images Calc writes beside its
HTML are the images of the document. An empty export is `tool_failed`.
"""
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from engine_test_support import (GREEN, RED, make_pptx, make_xlsx, png, require_toolchain, slide_picture,
                                 slide_title)
from markdown_for_llms.engine import core, native, routes, tools
from markdown_for_llms.engine.codes import Code, EngineError
from markdown_for_llms.engine.core import convert
from markdown_for_llms.engine.native import FILTER

LIMITS = {"memory_mib": 2048, "output_mib": 256}
# slide 1: a red picture with an SVG twin; slide 2: no picture; slide 3: green, then red again
SLIDES = [slide_title("First slide") + slide_picture("rId1", "red", svg="rId3"),
          slide_title("Second slide"),
          slide_title("Third slide") + slide_picture("rId2", "green") + slide_picture("rId1", "red again")]


class Base(unittest.TestCase):
    def setUp(self):
        require_toolchain()
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.n = 0

    def convert(self, path, route="libreoffice"):
        self.n += 1
        out = self.root / f"out{self.n}"
        return convert(path, out, {"route": route}, LIMITS, work=self.root / "work"), out

    def complete(self, path):
        result, out = self.convert(path)
        self.assertEqual(result["status"], "complete", result.get("error"))
        self.assertEqual(result["route"]["converter"], "libreoffice")
        return result, out, (out / "document.md").read_text(encoding="utf-8")

    def sha1_of_assets(self, out, md):
        """The SHA-1 of each image `md` references, in the order of the document."""
        import hashlib
        import re
        return [hashlib.sha1((out / p).read_bytes()).hexdigest() for p in re.findall(r"\]\((assets/[^ )]+)", md)]


class MatrixTests(unittest.TestCase):
    def test_libreoffice_is_an_alternative_of_xlsx_and_pptx_only(self):
        for fmt in ("xlsx", "pptx"):
            self.assertIn("libreoffice", routes.routes_for(fmt)["alternatives"])
            self.assertEqual(routes.resolve(fmt, "libreoffice"), ("libreoffice", "user", "imposed"))
        for fmt in set(routes.MATRIX) - {"xlsx", "pptx"}:
            with self.assertRaises(EngineError) as caught:
                routes.resolve(fmt, "libreoffice")
            self.assertEqual(caught.exception.code, Code.ROUTE_INVALID)

    def test_ocr_force_is_refused_and_the_options_are_those_of_a_native_route(self):
        with self.assertRaises(EngineError) as caught:
            routes.job_options("pptx", {"route": "libreoffice", "ocr": "force"})
        self.assertEqual(caught.exception.code, Code.ROUTE_OPTION_UNSUPPORTED)
        self.assertEqual(routes.job_options("xlsx", {"route": "libreoffice"})["route"], "libreoffice")

    def test_availability_follows_the_toolchain_and_the_isolation(self):
        with mock.patch("markdown_for_llms.engine.toolchain.image", return_value="sha256:x"), \
                mock.patch("markdown_for_llms.engine.isolation.available", return_value=False), \
                mock.patch.object(core, "_marker_availability", return_value={"available": False, "code": "x"}):
            state = core.availability()
        self.assertEqual(state["libreoffice"], {"available": False, "code": "isolation_unavailable"})
        self.assertEqual(state["djvutxt"], {"available": True, "code": None})
        with mock.patch("markdown_for_llms.engine.toolchain.image", side_effect=EngineError(Code.TOOLCHAIN_MISSING)), \
                mock.patch.object(core, "_marker_availability", return_value={"available": False, "code": "x"}):
            self.assertEqual(core.availability()["libreoffice"], {"available": False, "code": "toolchain_missing"})


class PptxTests(Base):
    def test_images_follow_the_heading_of_their_slide(self):
        result, out, md = self.complete(make_pptx(self.root / "s.pptx", SLIDES))
        lines = [line for line in md.splitlines() if line.strip()]
        headings = [i for i, line in enumerate(lines) if line.startswith("# ")]
        images = [i for i, line in enumerate(lines) if line.startswith("![")]
        self.assertEqual([lines[i] for i in headings], ["# First slide", "# Second slide", "# Third slide"])
        # one image after slide 1, none after slide 2, two after slide 3 (the red one again)
        self.assertEqual(len(images), 3)
        self.assertTrue(headings[0] < images[0] < headings[1])
        self.assertTrue(headings[2] < images[1] < images[2])
        import hashlib
        red, green = hashlib.sha1(RED).hexdigest(), hashlib.sha1(GREEN).hexdigest()
        self.assertEqual(self.sha1_of_assets(out, md), [red, green, red])
        self.assertEqual(len(list((out / "assets").iterdir())), 2)          # the same bytes share one file
        self.assertIn('"slide 3"', lines[images[2]])
        record = result["libreoffice"]
        self.assertEqual({k: record[k] for k in ("export", "reader", "images", "grouped", "slide_boundaries",
                                                 "slide_images")},
                         {"export": "html:impress_html_Export", "reader": "html", "images": "package",
                          "grouped": "slide", "slide_boundaries": "known", "slide_images": 3})
        self.assertNotIn("media_lost", result["problem_codes"])             # the SVG twin is no loss
        self.assertEqual(result["package"]["format"], "pptx")
        self.assertEqual(result["corrections"]["counts"]["slide_image_placed"], 3)

    def test_slides_come_in_the_order_of_the_slide_list(self):
        result, out, md = self.complete(make_pptx(self.root / "o.pptx", SLIDES, order=[3, 1, 2]))
        lines = [line for line in md.splitlines() if line.strip()]
        self.assertEqual([line for line in lines if line.startswith("# ")],
                         ["# Third slide", "# First slide", "# Second slide"])
        import hashlib
        red, green = hashlib.sha1(RED).hexdigest(), hashlib.sha1(GREEN).hexdigest()
        self.assertEqual(self.sha1_of_assets(out, md), [green, red, red])
        first = next(i for i, line in enumerate(lines) if line.startswith("!["))
        self.assertIn('"slide 1"', lines[first])                              # slide 1 of the list: slide3.xml

    def test_unknown_boundaries_lose_every_image_and_move_none(self):
        """When the slides cannot be matched to the headings, no image goes to another slide."""
        real = native.audit

        def unknown_order(ctx, path, fmt, flags=()):
            package, directory = real(ctx, path, fmt, flags)
            package["slide_order"] = "unknown"
            (directory / "audit.json").write_text(json.dumps(package))
            return package, directory
        with mock.patch.object(native, "audit", side_effect=unknown_order):
            result, out, md = self.complete(make_pptx(self.root / "u.pptx", SLIDES))
        self.assertNotIn("![", md)
        self.assertFalse((out / "assets").exists())
        self.assertEqual(result["libreoffice"]["slide_boundaries"], "unknown")
        lost = [p for p in result["problems"] if p["code"] == "media_lost"]
        self.assertEqual(sorted(p["path"] for p in lost), ["ppt/media/a.png", "ppt/media/b.png"])
        self.assertEqual({p["detail"] for p in lost}, {"slide_boundaries_unknown"})

    def test_empty_export_is_tool_failed(self):
        real = tools.run

        def empty_html(tool, args, *rest, **kwargs):
            done = real(tool, args, *rest, **kwargs)
            if tool == "soffice":
                out = next(a for a in args if isinstance(a, tools.Out))
                (out.dest / "source.html").write_bytes(b"")
            return done
        with mock.patch.object(tools, "run", side_effect=empty_html):
            result, out = self.convert(make_pptx(self.root / "e.pptx", SLIDES))
        self.assertEqual((result["status"], result["error"]["code"]), ("failed", "tool_failed"))
        self.assertFalse((out / "document.md").exists())


class XlsxTests(Base):
    def test_the_images_of_calc_are_the_images_of_the_document(self):
        result, out, md = self.complete(make_xlsx(self.root / "s.xlsx"))
        self.assertIn("Hello", md)
        import hashlib
        self.assertEqual(self.sha1_of_assets(out, md) or self.html_images(out, md), [hashlib.sha1(png()).hexdigest()])
        self.assertEqual(result["libreoffice"]["images"], "export")
        self.assertEqual(result["problem_codes"], {})
        self.assertEqual(result["package"]["format"], "xlsx")

    def html_images(self, out, md):
        import hashlib
        import re
        return [hashlib.sha1((out / p).read_bytes()).hexdigest() for p in re.findall(r'<img src="(assets/[^"]+)"', md)]


class FilterTests(unittest.TestCase):
    """The filter alone: as many headings as slides, or none of the images placed."""

    def run_html(self, html, slides):
        require_toolchain()
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        (root / "media").mkdir()
        (root / "media" / "m-0001.png").write_bytes(png())
        audit = {"format": "pptx", "media": [], "slide_order": "known",
                 "slides": [{"index": i + 1, "part": f"ppt/slides/slide{i + 1}.xml",
                             "images": [{"member": "ppt/media/a.png", "file": "m-0001.png", "name": "", "descr": "fig"}]
                             * n} for i, n in enumerate(slides)]}
        (root / "audit.json").write_text(json.dumps(audit))
        (root / "source.html").write_text(html, encoding="utf-8")
        tools.run("pandoc", ["-f", "html", "-t", "commonmark_x", "--wrap=none", "--extract-media=assets",
                             tools.In(FILTER, name="contract.lua", prefix="--lua-filter="),
                             tools.In(root / "audit.json", name="package-media.json", argument=False),
                             tools.In(root / "media", name="media", directory=True, argument=False),
                             "-o", tools.Out(root / "document.md", arg="document.md", allow_empty=True),
                             tools.Out(root / "assets", arg="assets", directory=True, optional=True, argument=False),
                             tools.Out(root / "pre.md", arg="pre-filter.md", allow_empty=True, argument=False),
                             tools.Out(root / "report.json", arg="contract-report.json", argument=False),
                             tools.In(root / "source.html")], root / "tools", timeout=120, memory_mb=1024)
        return (root / "document.md").read_text(), json.loads((root / "report.json").read_text())

    def test_headings_and_slides_match(self):
        md, report = self.run_html("<h1>A</h1><p>a</p><h1></h1><p>b</p>", [1, 2])
        self.assertEqual(report["slide_boundaries"], "known")
        lines = [line for line in md.splitlines() if line.strip()]
        self.assertEqual([line[:2] for line in lines], ["# ", "a", "![", "# ", "b", "![", "!["])

    def test_one_heading_too_many_places_no_image(self):
        md, report = self.run_html("<h1>A</h1><p>a</p><h1>B</h1><h1>C</h1>", [1, 1])
        self.assertEqual(report["slide_boundaries"], "unknown")
        self.assertNotIn("![", md)


if __name__ == "__main__":
    unittest.main()
