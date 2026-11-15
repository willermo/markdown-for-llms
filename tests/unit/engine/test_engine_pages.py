"""The pages family on the Marker route (MFL032 tranche B: Δ2-Δ4, D13, D14, PB1-PB7, 12.3-12.5).

Pure parts (cornice, figures, gate, image names, progress) run on synthetic data; `pdfprep`
runs on synthetic PDFs built in pure Python; the isolation of the Marker child (network,
scope cap) and one real conversion with Marker 2.2.0 and the weights of `FULL_CHECK_MODEL_ROOT`
(D7) run for real: a missing variable or scope is a failure, never a skip.
"""
import json
import os
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

from markdown_for_llms.engine import contract, isolation, marker_child, pages, pdfprep, progress
from markdown_for_llms.engine.codes import Code, EngineError

from engine_test_support import FIGURE, GLYPHLESS_MODE0, INVISIBLE, NATIVE_TEXT, SCAN, checker_rgb, make_pdf, scan_of

IMG = (100, 100, checker_rgb(100, 100))


def raster(box, digest):
    return {"box": box, "area": 0.01, "digest": digest, "depth": 0}


def page(n, rasters, cls="native"):
    return {"page": n, "class": cls, "size": [595, 842], "rotation": 0, "rasters": rasters, "invisible_chars": 0}


class FurnitureTests(unittest.TestCase):
    """D14 with PB12 of MFL032: the outcome of every raster of the native pages."""

    def test_same_image_on_three_pages_keeps_the_first_use(self):                   # PB12 case 1
        ps = [page(i, [raster([100, 300, 200, 400], "fig")]) for i in (1, 2, 3)]
        outcomes, groups = pages.furniture(ps)
        self.assertEqual(outcomes, {(2, 0): "cornice", (3, 0): "cornice"})
        self.assertEqual(groups, [(1, [2, 3], 2, 1)])

    def test_same_image_on_two_pages_only_in_the_band(self):                      # case 2
        band = [page(i, [raster([100, 10, 200, 60], "logo")]) for i in (1, 2)]
        self.assertEqual(pages.furniture(band), ({(1, 0): "cornice", (2, 0): "cornice"}, [(2, [1, 2], 2, None)]))
        body = [page(i, [raster([100, 300, 200, 400], "fig")]) for i in (1, 2)]
        self.assertEqual(pages.furniture(body), ({}, []))          # a figure repeated twice is content

    def test_same_place_in_the_footer_with_other_content(self):                    # case 3
        # the per-page codes of a running footer (D19): other bytes, same place, every page
        ps = [page(i, [raster([323, 765, 378, 820], "code%d" % i)]) for i in range(1, 8)]
        outcomes, groups = pages.furniture(ps)
        self.assertEqual(len(outcomes), 7)
        self.assertEqual(groups[0][:3], (3, list(range(1, 8)), 7))

    def test_one_pixel_images_are_fills(self):                                     # PB12, D01
        ps = [page(i, [dict(raster([100, 300, 104, 300.5], "dot"), pixels=[1, 1])] * 3) for i in (1, 2, 3, 4)]
        outcomes, groups = pages.furniture(ps)
        self.assertEqual(set(outcomes.values()), {"fill"})
        self.assertEqual((len(outcomes), groups), (12, []))

    def test_single_or_body_images_and_scans_stay_content(self):
        ps = [page(1, [raster([323, 765, 378, 820], "a")]), page(2, [raster([100, 300, 200, 400], "b")]),
              page(3, [raster([0, 0, 595, 842], "s1")], cls="image"), page(4, [raster([0, 0, 595, 842], "s1")], cls="image"),
              page(5, [raster([0, 0, 595, 842], "s1")], cls="image")]
        self.assertEqual(pages.furniture(ps), ({}, []))


def ctx_stub():
    return SimpleNamespace(problems=[])


# a figure at the centre of the page, a logo in the footer band and a fill of one pixel drawn
# as a short rule (as D01): three images of every page of the PB12 tests
CENTRE = b"q 200 0 0 150 197 346 cm /Im1 Do Q"
FOOTER_LOGO = b"q 60 0 0 30 500 20 cm /Im2 Do Q"
RULE = b"q 120 0 0 0.5 72 600 cm /Im3 Do Q"
LOGO, DOT = (30, 15, checker_rgb(30, 15, cell=5)), (1, 1, bytes((90, 90, 90)))


def beamer_pdf(n=3, centre=True, logo=False, rule=False):
    content = NATIVE_TEXT + (b" " + CENTRE if centre else b"") + (b" " + FOOTER_LOGO if logo else b"") + (b" " + RULE if rule else b"")
    return make_pdf([{"content": content, "image": IMG, "images": {"Im2": LOGO, "Im3": DOT}}] * n)


class FigurePlanTests(unittest.TestCase):
    """PB12 on synthetic PDFs: classification for real, then the plan of the figures."""

    def plan(self, data):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "s.pdf"
            path.write_bytes(data)
            ps = pdfprep.classify(path, 10)["page"]
        ctx = ctx_stub()
        _, pictures, rasters = pages.figures(ctx, ps)
        return sum(len(v) for v in pictures.values()), [p["code"] for p in ctx.problems], rasters

    def test_same_figure_at_the_centre_of_three_pages(self):
        figures, codes, rasters = self.plan(beamer_pdf())
        self.assertEqual((figures, codes), (1, ["media_furniture"]))
        self.assertEqual([rasters[i] for i in range(3)], [{"picture": 1, "total": 1}] + [{"cornice": 1, "total": 1}] * 2)

    def test_same_image_in_the_band_of_every_page(self):
        figures, codes, _ = self.plan(beamer_pdf(centre=False, logo=True))
        self.assertEqual((figures, codes), (0, ["media_furniture"]))

    def test_repeated_one_pixel_image(self):
        figures, codes, rasters = self.plan(beamer_pdf(centre=False, rule=True))
        self.assertEqual((figures, codes), (0, []))
        self.assertEqual(rasters[0], {"fill": 1, "total": 1})


class FakePage:
    def __init__(self, blocks=None, lines=None):
        self.blocks, self.lines = blocks or [], lines or []


class PictureTests(unittest.TestCase):
    def test_top_then_left_and_contained_boxes(self):
        p = FakePage(blocks=[{"label": "Text", "bbox": [0, 0, 600, 50], "text": "title"},
                             {"label": "Text", "bbox": [0, 500, 600, 550], "text": "caption"}])
        counts = marker_child.add_pictures(p, [[300, 100, 500, 300], [0, 100, 200, 300], [10, 110, 50, 150],
                                                [5, 5, 5, 5]])
        self.assertEqual(counts, (2, 2, 0))                     # one inside a bigger one, one empty
        labels = [(b["label"], b["bbox"][0]) for b in p.blocks]
        self.assertEqual(labels, [("Text", 0), ("Picture", 0.0), ("Picture", 300.0), ("Text", 0)])

    def test_a_partial_overlap_widens_the_crop_to_the_union(self):                     # FC1
        # two pictures overlapping on a slide: covered at 67 %, not contained; nothing is cut
        p = FakePage(blocks=[{"label": "Text", "bbox": [0, 0, 600, 50], "text": "title"}])
        counts = marker_child.add_pictures(p, [[100, 100, 300, 250], [160, 100, 340, 250]])
        self.assertEqual(counts, (1, 0, 1))
        self.assertEqual([b["bbox"] for b in p.blocks if b["label"] == "Picture"], [[100.0, 100.0, 340.0, 250.0]])

    def test_only_a_complete_containment_is_part_of_another_figure(self):             # FC1
        for inner, expected in (([150, 120, 250, 200], (1, 1, 0)),                   # inside
                                ([100.5, 99.5, 300.5, 250.5], (1, 1, 0)),             # inside within the tolerance
                                ([150, 120, 302, 200], (1, 0, 1))):                   # 2 px out: union
            with self.subTest(inner=inner):
                p = FakePage(blocks=[{"label": "Text", "bbox": [0, 0, 600, 50], "text": "t"}])
                self.assertEqual(marker_child.add_pictures(p, [[100, 100, 300, 250], inner]), expected)
                if expected[2]:
                    self.assertEqual(p.blocks[1]["bbox"], [100.0, 100.0, 302.0, 250.0])

    def test_boxes_that_only_touch_stay_two_figures(self):
        p = FakePage(blocks=[{"label": "Text", "bbox": [0, 0, 600, 50], "text": "t"}])
        self.assertEqual(marker_child.add_pictures(p, [[100, 100, 200, 200], [200, 100, 300, 200]]), (2, 0, 0))

    def test_a_partial_overlap_with_a_picture_of_indra_widens_it(self):
        p = FakePage(blocks=[{"label": "Picture", "bbox": [100, 100, 300, 250]}])
        self.assertEqual(marker_child.add_pictures(p, [[250, 100, 400, 250]]), (0, 0, 1))
        self.assertEqual(p.blocks[0]["bbox"], [100, 100, 400.0, 250])

    def test_blocks_are_made_from_lines_first(self):
        line = SimpleNamespace(box=[0, 400, 600, 420], text="the only line")
        p = FakePage(lines=[line])
        marker_child.add_pictures(p, [[0, 100, 200, 300]])
        self.assertEqual([b["label"] for b in p.blocks], ["Picture", "Text"])


def doc(*ids):
    return SimpleNamespace(pages=[SimpleNamespace(page_id=i, children=None) for i in ids])


def prov(source="text", text_blocks=3, blocks=None, errors=0, skipped=0):
    return {"indra_source": source, "note": "n", "text_blocks": text_blocks,
            "blocks": text_blocks if blocks is None else blocks, "block_errors": errors,
            "pictures_added": 0, "pictures_contained": 0, "pictures_skipped_rotated": skipped}


class GateTests(unittest.TestCase):
    def test_native_page_without_text_fails_even_with_a_figure(self):            # PB1
        with self.assertRaises(marker_child.PageFailed) as e:
            marker_child.gate(doc(0), {0: prov("text", text_blocks=0, blocks=1)}, {0: {"class": "native"}})
        self.assertEqual(e.exception.page, 1)

    def test_error_source_and_block_errors_fail(self):
        for p in (prov("error"), prov("text", errors=1)):
            with self.subTest(p=p), self.assertRaises(marker_child.PageFailed):
                marker_child.gate(doc(0), {0: p}, {0: {"class": "native"}})

    def test_missing_provenance_is_an_adapter_mismatch(self):
        with self.assertRaises(marker_child.AdapterMismatch):
            marker_child.gate(doc(0, 1), {0: prov()}, {})

    def test_ocr_layer_read_from_its_text_asks_for_force_ocr(self):
        with self.assertRaises(marker_child.OcrLayerRead):
            marker_child.gate(doc(0), {0: prov("hybrid")}, {0: {"class": "ocr_layer"}})

    def test_blank_and_lost_pages(self):
        d = doc(0, 1, 2)
        problems = marker_child.gate(d, {0: prov("ocr", 0, 0), 1: prov("ocr", 0, 0), 2: prov("text", skipped=2)},
                                     {0: {"class": "image"}, 1: {"class": "ocr_layer", "invisible_chars": 40},
                                      2: {"class": "native"}})
        self.assertEqual([(p["code"], p["page"]) for p in problems],
                         [("page_blank", 1), ("page_text_lost", 2), ("pictures_skipped_rotated", 3)])
        self.assertEqual(d.pages[0].children, [])
        for item in problems:
            contract.problem(item["code"], page=item["page"])        # every code is a problem of the contract


class ImageNameTests(unittest.TestCase):
    def test_only_image_destinations_are_renamed(self):
        text = "![](_page_0_Picture_1.jpeg)\n\nThe file _page_0_Picture_1.jpeg is named in the text.\n\n![](_page_1_Picture_0.png)\n"
        out, mapping = pages.rename_images(text, ["_page_1_Picture_0.png", "_page_0_Picture_1.jpeg"])
        self.assertEqual(mapping, {"_page_0_Picture_1.jpeg": "assets/asset-0001.jpg", "_page_1_Picture_0.png": "assets/asset-0002.png"})
        self.assertIn("The file _page_0_Picture_1.jpeg is named", out)
        self.assertIn("![](assets/asset-0001.jpg)", out)

    def test_a_reference_the_parser_does_not_see_is_refused(self):
        text = "```\n![](_page_0_Picture_1.jpeg)\n```\n"
        with self.assertRaises(EngineError) as e:
            pages.rename_images(text, ["_page_0_Picture_1.jpeg"])
        self.assertEqual(e.exception.code, Code.OUTPUT_INVALID)


class ProgressTests(unittest.TestCase):
    def test_pages_move_the_estimate_and_stages_never_go_back(self):
        p = progress.Progress(wall=lambda: 100.0)
        p.feed({"event": "models"})
        p.feed({"event": "page", "page": 5, "pages": 10})
        snap = p.snapshot()
        self.assertEqual((snap["stage"], snap["current"], snap["total"], snap["estimate_percent"]), ("pages", 5, 10, 50.0))
        p.feed({"event": "render"})
        p.feed({"event": "models"})
        self.assertEqual(p.snapshot()["stage"], "processors")
        self.assertFalse(p.feed({"event": "unknown"}))
        self.assertEqual(progress.finishing(p.snapshot(), wall=lambda: 1.0)["stage"], "finishing")


class PdfprepTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def pdf(self, name, spec, **kw):
        path = Path(self.tmp.name) / (name + ".pdf")
        path.write_bytes(make_pdf(spec, **kw))
        return path

    def classes(self, path):
        return [p["class"] for p in pdfprep.classify(path, 10)["page"]]

    def test_classes(self):
        self.assertEqual(self.classes(self.pdf("n", [{"content": NATIVE_TEXT + b" " + FIGURE, "image": IMG}])), ["native"])
        self.assertEqual(self.classes(self.pdf("s", [{"content": SCAN, "image": IMG}])), ["image"])
        self.assertEqual(self.classes(self.pdf("l", [{"content": NATIVE_TEXT}, {"content": SCAN + b" " + INVISIBLE, "image": IMG}])),
                         ["native", "ocr_layer"])

    def test_spaces_and_text_outside_the_page_do_not_make_it_native(self):             # PB4
        self.assertEqual(self.classes(self.pdf("sp", [{"content": b"BT /F1 12 Tf 72 700 Td (     ) Tj ET"}])), ["image"])
        self.assertEqual(self.classes(self.pdf("out", [{"content": b"BT /F1 12 Tf 900 900 Td (OUTSIDE) Tj ET"}])), ["image"])

    def test_glyphless_layer_in_mode_0_is_invisible(self):                              # PB6
        path = self.pdf("g", [{"content": NATIVE_TEXT}, {"content": SCAN + b" " + GLYPHLESS_MODE0, "image": IMG}],
                        font="GlyphlessFont")
        self.assertEqual(self.classes(path), ["native", "ocr_layer"])
        report = pdfprep.strip(path, Path(self.tmp.name) / "g-s.pdf", [2])
        self.assertFalse(report["removable"])               # the copy renders differently: discarded
        self.assertEqual(report["rendering_changed"], [2])

    def test_raster_boxes_follow_a_translated_mediabox_and_rotation_is_recorded(self):   # PB5
        plain = pdfprep.classify(self.pdf("p", [{"content": NATIVE_TEXT + b" " + FIGURE, "image": IMG}]), 10)["page"][0]
        moved = pdfprep.classify(self.pdf("m", [{"content": NATIVE_TEXT + b" " + FIGURE, "image": IMG,
                                                 "mediabox": (100, 100, 695, 942)}]), 10)["page"][0]
        self.assertEqual(plain["rasters"][0]["box"], [72.0, 192.0, 272.0, 342.0])
        self.assertEqual(moved["rasters"][0]["box"], [-28.0, 292.0, 172.0, 442.0])
        rotated = pdfprep.classify(self.pdf("r", [{"content": NATIVE_TEXT, "rotate": 90}]), 10)["page"][0]
        self.assertEqual(rotated["rotation"], 90)

    def test_every_raster_is_inventoried_with_a_digest(self):                          # PB2
        path = self.pdf("two", [{"content": NATIVE_TEXT + b" " + FIGURE + b" q 2 0 0 2 10 10 cm /Im1 Do Q", "image": IMG}])
        rasters = pdfprep.classify(path, 10)["page"][0]["rasters"]
        self.assertEqual(len(rasters), 2)                    # a 2x2 pt icon too, before any threshold
        self.assertEqual(rasters[0]["digest"], rasters[1]["digest"])

    def test_strip_direct_form_and_mixed_form(self):                                     # Δ4
        cases = {"direct": ({"content": SCAN + b" " + INVISIBLE, "image": IMG}, True),
                 "form": ({"content": SCAN + b" /Fm1 Do", "image": IMG, "form": INVISIBLE}, True),
                 "mixed": ({"content": b"/Fm1 Do", "image": IMG, "form": SCAN + b" " + INVISIBLE}, False)}
        for name, (spec, removable) in cases.items():
            with self.subTest(name):
                path = self.pdf(name, [{"content": NATIVE_TEXT}, spec])
                out = Path(self.tmp.name) / (name + "-s.pdf")
                report = pdfprep.strip(path, out, [2])
                self.assertEqual(report["removable"], removable)
                self.assertEqual(report["rendering_changed"], [])
                if removable:
                    self.assertEqual(self.classes(out), ["native", "image"])

    def test_images_become_one_page_per_frame(self):                                   # Δ3
        from PIL import Image
        import pypdfium2 as pdfium
        frames = [Image.new("RGB", (300, 200), c) for c in ("white", "black")]
        tiff = Path(self.tmp.name) / "two.tiff"
        frames[0].save(tiff, save_all=True, append_images=frames[1:], dpi=(150, 150))
        out = Path(self.tmp.name) / "two.pdf"
        self.assertEqual(pdfprep.image_to_pdf(tiff, out, "tiff", 10, 10_000_000), 2)
        doc = pdfium.PdfDocument(str(out))
        self.assertEqual((len(doc), [round(v) for v in doc[0].get_size()]), (2, [144, 96]))
        doc.close()
        with self.assertRaises(ValueError):
            pdfprep.image_to_pdf(tiff, out, "tiff", 1, 10_000_000)               # page limit before any page
        png = Path(self.tmp.name) / "one.png"
        frames[0].save(png)
        self.assertEqual(pdfprep.image_to_pdf(png, Path(self.tmp.name) / "one.pdf", "png", 10, 10_000_000), 1)

    def test_page_limit_reads_no_page(self):
        path = self.pdf("many", [{"content": NATIVE_TEXT}] * 3)
        self.assertEqual(pdfprep.classify(path, 2), {"pages": 3, "page": []})


class FakeChildTests(unittest.TestCase):
    """12.5 (scenarios 2, 3 and 6): a Marker child that exits 0 with an empty output or a missing
    page, a failed page, and the single retry with force_ocr of PB7. The light child of pdfprep
    runs for real; the Marker child is replaced by `fake`, which writes what a child would."""

    def run_with(self, script, spec=None):
        from unittest import mock
        from markdown_for_llms.engine.core import convert
        real_run = isolation.run
        calls = []

        def fake_run(scratch, module, args=(), **kw):
            if module != pages.CHILD:
                return real_run(scratch, module, args, **kw)
            request = json.loads((Path(scratch) / "request.json").read_text())
            calls.append(request["force_ocr"])
            rc, result, markdown = script[len(calls) - 1](request)
            (Path(scratch) / "result.json").write_text(json.dumps(result))
            (Path(scratch) / "out" / "images").mkdir(parents=True, exist_ok=True)
            (Path(scratch) / "out" / "document.md").write_text(markdown)
            kw["steps"].append({"tool": "bwrap:marker_child", "args": list(args), "seconds": 0, "exit": rc})
            return rc, b""
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        src = Path(tmp.name) / "s.pdf"
        src.write_bytes(make_pdf(spec or [{"content": NATIVE_TEXT}, {"content": NATIVE_TEXT}]))
        with mock.patch.object(pages.isolation, "run", fake_run), \
                mock.patch.object(pages.weights, "verify", return_value={"files": 18}):
            result = convert(src, Path(tmp.name) / "out", None, None, work=Path(tmp.name) / "work",
                             model_root=Path(tmp.name))
        return result, calls

    @staticmethod
    def ok(n=2, source="text", markdown=None, pages_out=None):
        def step(request):
            prov = {str(i): dict(prov_(source), pictures_added=0) for i in range(1, n + 1)}
            result = {"versions": {}, "pages": pages_out or n, "images": [], "provenance": prov,
                      "methods": {str(i): "pdftext" for i in range(1, n + 1)}, "page_problems": [],
                      "force_ocr": request["force_ocr"], "seconds": {}, "network_calls": 0}
            return 0, result, markdown if markdown is not None else "{0}" + "-" * 48 + "\n\nText of the page.\n"
        return step

    def test_exit_0_with_only_page_separators_is_output_empty(self):
        result, _ = self.run_with([self.ok(markdown="{0}" + "-" * 48 + "\n\n{1}" + "-" * 48 + "\n")])
        self.assertEqual(result["error"]["code"], "output_empty")
        self.assertNotIn("document.md", [i["path"] for i in result.get("inventory") or []])

    def test_exit_0_with_a_missing_page_fails(self):
        result, _ = self.run_with([self.ok(pages_out=1)])
        self.assertEqual((result["status"], result["error"]["code"]), ("failed", "converter_failed"))

    def test_a_failed_page_fails_the_document(self):
        result, _ = self.run_with([lambda r: (4, {"error": "page_failed", "page": 2, "note": "RuntimeError"}, "")])
        self.assertEqual(result["error"]["code"], "page_failed")

    def test_ocr_layer_read_from_its_text_is_retried_once_with_force_ocr(self):          # PB7
        retry = lambda r: (5, {"error": "ocr_layer_read", "page": 2, "source": "text"}, "")
        result, calls = self.run_with([retry, self.ok()])
        self.assertEqual((result["status"], calls), ("complete", [False, True]))
        self.assertEqual(result["marker"]["retried_with_force_ocr"], {"page": 2, "source": "text"})
        self.assertEqual(result["options"]["effective"]["ocr"], "force")
        self.assertIn("ocr_forced_document", [p["code"] for p in result["problems"]])
        result, calls = self.run_with([retry, retry])
        self.assertEqual((result["error"]["code"], calls), ("page_failed", [False, True]))

    @staticmethod
    def gated(empty_pages):
        """A child whose OCR returns nothing on `empty_pages`; the real gate of marker_child judges
        the pages with the page_info that marker_route sent."""
        def step(request):
            info = {int(k): v for k, v in request["pages"].items()}
            prov = {}
            for i, page_info in info.items():
                empty = i + 1 in empty_pages
                prov[i] = dict(prov_("text" if page_info["class"] == "native" else "ocr"), pictures_added=0,
                               text_blocks=0 if empty else 1, blocks=0 if empty else 1)
            document = SimpleNamespace(pages=[SimpleNamespace(page_id=i, children=None) for i in sorted(info)])
            problems = marker_child.gate(document, prov, info)
            md = "".join("{%d}%s\n\n%s\n\n" % (i, "-" * 48, "" if i + 1 in empty_pages else "Text.") for i in sorted(info))
            result = {"versions": {}, "pages": len(info), "images": [], "page_problems": problems,
                      "provenance": {str(i + 1): v for i, v in prov.items()},
                      "methods": {str(i + 1): "surya" for i in info}, "force_ocr": request["force_ocr"],
                      "seconds": {}, "network_calls": 0}
            return 0, result, md
        return step

    def test_ocr_layer_page_stripped_and_read_empty_is_text_lost(self):                   # FC2
        layer = (100, 100, checker_rgb(100, 100))
        result, _ = self.run_with([self.gated({2})], [{"content": NATIVE_TEXT}, {"content": SCAN + b" " + INVISIBLE, "image": layer}])
        self.assertEqual(result["status"], "complete", result.get("error"))
        self.assertEqual(result["marker"]["reason"], "ocr_layer_removed")              # strip ran for real
        self.assertEqual([(p["code"], p["page"], p["severity"]) for p in result["problems"]],
                         [("page_text_lost", 2, "incomplete")])
        page = result["pages"]["pages"][1]
        self.assertEqual((page["class"], page["class_after_strip"]), ("ocr_layer", "image"))
        self.assertGreaterEqual(page["invisible_chars"], 20)                         # the count of the original layer

    def test_scanned_page_without_expected_text_stays_blank(self):                       # FC2
        scan = (100, 100, checker_rgb(100, 100))
        result, _ = self.run_with([self.gated({2})], [{"content": NATIVE_TEXT}, {"content": SCAN, "image": scan}])
        self.assertEqual([(p["code"], p["page"], p["severity"]) for p in result["problems"]],
                         [("page_blank", 2, "warning")])
        self.assertNotIn("class_after_strip", result["pages"]["pages"][1])

    def test_no_page_with_error_in_a_complete_result(self):                              # PB7
        result, _ = self.run_with([self.ok(source="error")])
        self.assertEqual(result["error"]["code"], "page_failed")


def prov_(source):
    return {"indra_source": source, "note": "", "text_blocks": 1, "blocks": 1, "block_errors": 0,
            "pictures_contained": 0, "pictures_skipped_rotated": 0}


def model_root():
    root = os.environ.get("FULL_CHECK_MODEL_ROOT")
    if not root:
        raise AssertionError("FULL_CHECK_MODEL_ROOT is not set: run the suite from scripts/full-check.sh with the weights "
                             "prepared by `mdf prepare --weights --model-root DIR`")
    return root


class ChildIsolationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        if not isolation.scope_available():
            self.fail("no systemd user scope: the Marker route needs `systemd-run --user --scope`")

    def test_probe_with_the_mounts_and_scope_of_the_marker_child(self):                # 12.3
        scratch = Path(self.tmp.name) / "probe"
        steps = []
        rc, _ = isolation.run(scratch, "markdown_for_llms.engine.selfprobe", [os.path.realpath(model_root())], memory_mib=512,
                              timeout=60, ro_binds=[(model_root(), "/models")], env=pages.child_env(model_root(), 2),
                              fonts=True, scope="mdf-eng-test%s-probe" % os.getpid(), steps=steps)
        result = json.loads((scratch / "probe.json").read_text())
        self.assertEqual(rc, 0)
        self.assertTrue(all(result.values()), result)
        self.assertTrue(steps[0]["scope"].endswith(".scope"))
        self.assertIn("memory_peak_bytes", steps[0])         # None when the child ends before a sample

    def test_a_child_over_the_scope_cap_is_killed(self):                               # 12.3
        with self.assertRaises(EngineError) as e:
            isolation.run(Path(self.tmp.name) / "hog", "markdown_for_llms.engine.selfprobe", ["allocate", "600"],
                          memory_mib=200, address_space_mib=4096, timeout=60, scope="mdf-eng-test%s-hog" % os.getpid())
        self.assertEqual(e.exception.code, Code.MEMORY_LIMIT)

    def test_marker_is_available_with_the_weights(self):
        from markdown_for_llms.engine.core import availability
        self.assertEqual(availability(model_root=model_root())["marker"], {"available": True, "code": None})

    def test_the_scope_is_named_to_the_observer_before_the_child_starts(self):         # HC1
        from unittest import mock
        seen = []
        with mock.patch.object(isolation.subprocess, "Popen") as popen, self.assertRaises(EngineError) as e:
            isolation.run(Path(self.tmp.name) / "w", "markdown_for_llms.engine.selfprobe", timeout=10,
                          scope="mdf-eng-window-marker-1", observer=lambda ev: seen.append(ev) or "cancelled")
        self.assertEqual(e.exception.code, Code.CANCELLED)
        popen.assert_not_called()
        self.assertEqual(seen, [{"event": "child", "module": "markdown_for_llms.engine.selfprobe", "pid": None,
                                 "scope": "mdf-eng-window-marker-1.scope"}])

    def test_scopes_are_listed_by_prefix_and_names_are_checked(self):                 # HC1
        from unittest import mock
        self.assertEqual(isolation.list_scopes("mdf-eng-absent%d-" % os.getpid()), [])
        for bad in ("other-", "mdf-eng-a b-", "mdf-eng-x"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                isolation.list_scopes(bad)
        with mock.patch.object(isolation.subprocess, "run") as run, self.assertRaises(ValueError):
            isolation.stop_scope("../../etc/x.scope")
        run.assert_not_called()

    def test_scope_state_of_an_absent_unit(self):                                     # PB8
        self.assertEqual(isolation.scope_state("mdf-eng-test-absent-%d" % os.getpid()), "absent")

    def test_an_unreachable_manager_is_not_an_absent_unit(self):                      # PB8
        from unittest import mock
        with mock.patch.object(isolation, "_user_env", return_value={"XDG_RUNTIME_DIR": "/nonexistent",
                                                                     "DBUS_SESSION_BUS_ADDRESS": "unix:path=/nonexistent"}):
            with self.assertRaises(isolation.ScopeUnreachable):
                isolation.scope_state("mdf-eng-test-absent-%d" % os.getpid())


class RealMarkerTests(unittest.TestCase):
    """D7 of MFL032: a native page with a raster figure and a scanned page, Marker 2.2.0 for real."""

    def test_native_page_with_figure_and_scanned_page(self):
        from markdown_for_llms.engine.core import convert
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "mixed.pdf"
            scan = scan_of(["Scanned page of the engine test", "Marker reads it with OCR"], scale=1.5)
            src.write_bytes(make_pdf([{"content": NATIVE_TEXT + b" " + FIGURE, "image": IMG},
                                      {"content": SCAN, "image": scan}]))
            events = []
            result = convert(src, Path(tmp) / "out", {"attempt": "realmarker"}, {"memory_mib": 4096, "threads": 2},
                             work=Path(tmp) / "work", observer=lambda e: events.append(e) and None,
                             model_root=model_root())
            self.assertEqual(result["status"], "complete", result.get("error"))
            recs = result["pages"]["pages"]
            self.assertEqual([(r["class"], r["indra_source"] in ("text", "hybrid")) for r in recs],
                             [("native", True), ("image", False)])
            self.assertEqual(recs[1]["indra_source"], "ocr")
            self.assertEqual(recs[0]["pictures_added"], 1)
            text = (Path(tmp) / "out" / "document.md").read_text()
            self.assertIn("Native text of the synthetic page", text)
            self.assertIn("Scanned page", text)
            self.assertIn("](assets/asset-0001.", text)
            self.assertEqual(result["marker"]["network_calls"], 0)                      # 12.12
            child = [s for s in result["route"]["steps"] if s["tool"] == "bwrap:marker_child"]
            self.assertGreater(child[0]["memory_peak_bytes"], 100 << 20)               # read from the scope
            self.assertEqual(result["limits"], pages.LIMITS)                            # D13
            self.assertEqual(result["components"]["weights"]["files"], 18)
            self.assertTrue(any(e.get("event") == "progress" for e in events))
            self.assertTrue(any(e.get("event") == "child" and e.get("scope") == "mdf-eng-realmarker-marker-1.scope"
                                for e in events))

    def test_repeated_figure_logo_and_fill(self):                                       # PB12
        from markdown_for_llms.engine.core import convert
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "beamer.pdf"
            src.write_bytes(beamer_pdf(logo=True, rule=True))
            result = convert(src, Path(tmp) / "out", {"attempt": "pb12"}, {"memory_mib": 4096, "threads": 2},
                             work=Path(tmp) / "work", model_root=model_root())
            self.assertEqual(result["status"], "complete", result.get("error"))
            assets = sorted(p.name for p in (Path(tmp) / "out" / "assets").iterdir())
            self.assertEqual(len(assets), 1)
            self.assertEqual([p["code"] for p in result["problems"]], ["media_furniture", "media_furniture"])
            self.assertEqual(result["marker"]["rasters"], {"picture": 1, "cornice": 5, "fill": 3, "total": 9})

    def test_overlapping_pictures_lose_no_pixel(self):                                  # FC1
        """`overlap.pdf` of the review of B (Sol): a green picture [72, 500, 272, 650] and a red one
        drawn after it [132, 500, 312, 650]; the asset must hold the union, 240 x 150 pt, with the
        red right edge, not the 200 x 150 pt of the green one."""
        from PIL import Image
        from markdown_for_llms.engine.core import convert
        content = NATIVE_TEXT + b" q 200 0 0 150 72 500 cm /Im1 Do Q q 180 0 0 150 132 500 cm /Im2 Do Q"
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "overlap.pdf"
            src.write_bytes(make_pdf([{"content": content, "image": (100, 100, bytes((10, 200, 10)) * 10000),
                                       "images": {"Im2": (100, 100, bytes((230, 10, 10)) * 10000)}}]))
            result = convert(src, Path(tmp) / "out", {"attempt": "overlap"}, {"memory_mib": 4096, "threads": 2},
                             work=Path(tmp) / "work", model_root=model_root())
            self.assertEqual(result["status"], "complete", result.get("error"))
            self.assertEqual(result["pages"]["pages"][0]["pictures_merged"], 1)
            assets = list((Path(tmp) / "out" / "assets").iterdir())
            self.assertEqual(len(assets), 1)
            with Image.open(assets[0]) as img:
                rgb = img.convert("RGB")
                self.assertAlmostEqual(rgb.width / rgb.height, 240 / 150, delta=0.05)
                r, g, b = rgb.getpixel((rgb.width - 3, rgb.height // 2))       # the right edge is red
                self.assertTrue(r > 150 and g < 100, (r, g, b))
                r, g, b = rgb.getpixel((3, rgb.height // 2))                    # the left edge is green
                self.assertTrue(g > 150 and r < 100, (r, g, b))

    def test_missing_weights_stop_before_the_child(self):                              # 12.12
        from markdown_for_llms.engine.core import convert
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "n.pdf"
            src.write_bytes(make_pdf([{"content": NATIVE_TEXT}]))
            empty = Path(tmp) / "models"
            empty.mkdir()
            result = convert(src, Path(tmp) / "out", None, None, work=Path(tmp) / "work", model_root=empty)
            self.assertEqual(result["error"]["code"], "weights_missing")
            self.assertFalse(any(s["tool"] == "bwrap:marker_child" for s in result["route"]["steps"]))


if __name__ == "__main__":
    unittest.main()
