"""The native routes for real (A3 of MFL032): every born-digital format of the matrix from a small
synthetic file, the images in `assets/`, the loss policy (D9, D11) and the text and DjVu routes.

Criteria 12.1, 12.2 and 12.10 of the plan: the expected text is in `document.md`, the images are
in `assets/` and referenced, the inventory matches the files, and `references.inspect` (the
product's profile) never finds an `invalid` bundle.
"""
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from engine_test_support import (FIXTURES, TEXT_DJVU, derive_epub, drawing, inspect_out, make_docx, make_fodp,
                                 make_odt, make_odt_chart, make_odt_nested, make_xlsx, para, png, replace_figure, require_toolchain)
from markdown_for_llms.engine import tools
from markdown_for_llms.engine.core import convert

FIGURE = "OEBPS/figure-en.png"


class Base(unittest.TestCase):
    def setUp(self):
        require_toolchain()
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.n = 0

    def convert(self, path, **request):
        self.n += 1
        out = self.root / f"out{self.n}"
        result = convert(path, out, request, {"memory_mib": 2048, "output_mib": 256}, work=self.root / "work")
        return result, out

    def complete(self, path, **request):
        result, out = self.convert(path, **request)
        self.assertEqual(result["status"], "complete", result.get("error"))
        self.assertEqual(json.loads((out / "engine.json").read_text()), result)
        files = sorted(p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file() and p.name != "engine.json")
        self.assertEqual([i["path"] for i in result["inventory"]], files)
        for item in result["inventory"]:
            data = (out / item["path"]).read_bytes()
            self.assertEqual((item["bytes"], item["sha256"]), (len(data), hashlib.sha256(data).hexdigest()))
        self.assertTrue(result["inventory_complete"])
        self.assertNotEqual(inspect_out(out)["bundle"], "invalid")
        return result, out, (out / "document.md").read_text(encoding="utf-8")

    def codes(self, result):
        return sorted(p["code"] for p in result["problems"])

    def assert_image_linked(self, out, md, count=1):
        assets = sorted(p.name for p in (out / "assets").iterdir())
        self.assertEqual(len(assets), count)
        for name in assets:
            self.assertIn(f"assets/{name}", md)
        return assets


class FormatTests(Base):
    def test_docx_with_a_figure(self):
        path = make_docx(self.root / "d.docx", para("Docx paragraph text.") + f"<w:p><w:r>{drawing('rId1')}</w:r></w:p>",
                         media={"fig.png": png()})
        result, out, md = self.complete(path)
        self.assertIn("Docx paragraph text.", md)
        self.assert_image_linked(out, md)
        self.assertEqual(result["route"]["converter"], "pandoc")
        self.assertEqual(self.codes(result), [])
        self.assertEqual(result["package"]["media_roles"], {"content": 1})
        self.assertTrue((out / "diagnostics" / "pre-filter.md").is_file())

    def test_c02_figure_is_recovered(self):
        """12.10: C02 binds the drawing namespaces to other prefixes; the normalized copy keeps its figure."""
        result, out, md = self.complete(FIXTURES / "C02-docx-it.docx")
        self.assert_image_linked(out, md)
        self.assertEqual(self.codes(result), [])
        self.assertEqual(sorted(result["package"]["normalized"]["word/document.xml"]), ["ns2->wp", "ns3->a", "ns4->pic"])
        self.assertIn("package_prefixes_normalized", result["corrections"]["counts"])

    def test_xlsx_image_after_its_table(self):
        """12.10: the image anchored at A4 of the sheet follows the table of that sheet."""
        result, out, md = self.complete(make_xlsx(self.root / "s.xlsx"))
        (asset,) = self.assert_image_linked(out, md)
        lines = md.splitlines()
        image = next(i for i, line in enumerate(lines) if line.startswith("!["))
        self.assertEqual(lines[image], f'![Synthetic figure](assets/{asset} "Synthetic!A4")')
        self.assertLess(next(i for i, line in enumerate(lines) if "Hello" in line), image)
        self.assertEqual(self.codes(result), [])

    def test_xlsx_with_an_unused_drawing_relation(self):
        result, out, md = self.complete(make_xlsx(self.root / "u.xlsx", drawing_used=False))
        self.assertFalse((out / "assets").exists())
        self.assertEqual(self.codes(result), ["media_furniture"])

    def test_odt_and_a_picture_lost_in_a_text_box(self):
        """12.10: a picture pandoc does not extract (a frame inside a text box) is media_lost, incomplete."""
        result, out, md = self.complete(make_odt(self.root / "t.odt"))
        self.assertIn("Odt paragraph text.", md)
        self.assert_image_linked(out, md)
        result, out, md = self.complete(make_odt(self.root / "box.odt", text_box=True))
        lost = [p for p in result["problems"] if p["code"] == "media_lost"]
        self.assertEqual([(p["path"], p["severity"]) for p in lost], [("Pictures/fig.png", "incomplete")])
        self.assertEqual(result["problem_counts"]["incomplete"], 1)

    def test_epub(self):
        result, out, md = self.complete(FIXTURES / "C03-epub-en.epub")
        self.assert_image_linked(out, md)
        self.assertEqual(self.codes(result), [])

    def test_epub_figure_referenced_by_srcset_is_declared_lost(self):
        """PA3 (Sol F2): the figure of C03 referenced by `srcset` only is not extracted: media_lost."""
        path = derive_epub(self.root / "c03-srcset.epub", lambda t: t.replace('src="figure-en.png"', 'srcset="figure-en.png 1x"'))
        result, out, md = self.complete(path)
        self.assertIn(("media_lost", "OEBPS/figure-en.png", "incomplete"),
                      [(p["code"], p.get("path"), p["severity"]) for p in result["problems"]])
        self.assertNotIn("media_unused", self.codes(result))

    def test_rtf_html_and_txt(self):
        result, out, md = self.complete(FIXTURES / "C04-rtf-it.rtf")
        self.assertTrue(md.strip())
        html = self.root / "page.html"
        html.write_text('<!doctype html><html><body><h1 id="top">Titolo</h1><p>Testo <a href="#top">su</a>.</p>'
                        f'<p><img src="data:image/png;base64,{__import__("base64").b64encode(png()).decode()}" alt="dati"></p>'
                        '</body></html>', encoding="utf-8")
        result, out, md = self.complete(html)
        self.assertIn("Titolo", md)
        self.assert_image_linked(out, md)
        self.assertIn("data_uri_extracted", result["corrections"]["counts"])
        result, out, md = self.complete(FIXTURES / "C01-txt-it.txt")
        self.assertEqual(md, (FIXTURES / "C01-txt-it.txt").read_text(encoding="utf-8").replace("\r\n", "\n"))
        self.assertEqual(result["text"]["encoding"], "utf_8")

    def test_txt_in_cp1252(self):
        """C01 encoded in cp1252 (characters outside it replaced), with CRLF line endings."""
        data = (FIXTURES / "C01-txt-it.txt").read_text(encoding="utf-8").replace("\n", "\r\n").encode("cp1252", "replace")
        path = self.root / "it.txt"
        path.write_bytes(data)
        result, out, md = self.complete(path)
        self.assertEqual(result["text"]["encoding"], "cp1252")
        self.assertEqual(md, data.decode("cp1252").replace("\r\n", "\n"))

    def test_doc_ods_and_pptx_through_libreoffice(self):
        """doc and ods pass through LibreOffice before pandoc; the pptx is made from a flat ODP."""
        docx = make_docx(self.root / "src.docx", para("Paragraph converted twice.") + f"<w:p><w:r>{drawing('rId1')}</w:r></w:p>",
                         media={"fig.png": png()})
        doc = tools.soffice_convert(docx, "word.docx", "doc", self.root / "doc", self.root / "tools")
        result, out, md = self.complete(doc)
        self.assertIn("Paragraph converted twice.", md)
        self.assertEqual(result["pandoc"]["pre_step"], {"tool": "soffice", "from": "doc", "to": "docx"})
        self.assert_image_linked(out, md)
        ods = tools.soffice_convert(make_xlsx(self.root / "src.xlsx", image=False), "sheet.xlsx", "ods", self.root / "ods",
                                    self.root / "tools")
        result, out, md = self.complete(ods)
        self.assertIn("Hello", md)
        self.assertEqual(result["pandoc"]["pre_step"]["to"], "xlsx")
        pptx = tools.soffice_convert(make_fodp(self.root / "slides.fodp"), "slides.fodp", "pptx", self.root / "pptx",
                                     self.root / "tools")
        result, out, md = self.complete(pptx)
        self.assertIn("Slide title text", md)
        self.assert_image_linked(out, md)
        self.assertNotIn("incomplete", [p["severity"] for p in result["problems"]])


class LossTests(Base):
    """The loss policy on the counterexamples of the review of the diff A (FA2, FA3, FA4, FA6)."""

    def problems(self, result):
        return sorted((p["code"], p.get("path"), p["severity"]) for p in result["problems"])

    def test_body_figure_referenced_by_css_is_lost_not_furniture(self):
        """FA2 (Sol F2): the figure of C03 as the background of a sized `div` in its `figure`."""
        path = derive_epub(self.root / "css.epub", lambda t: replace_figure(
            t, '<figure id="b05"><div style="background-image:url(figure-en.png);width:128px;height:96px"></div>'
               '<figcaption>Figure 1 — Three regions and laboratory axes.</figcaption></figure>'))
        result, out, md = self.complete(path)
        self.assertIn("Three regions", md)
        self.assertEqual(self.problems(result), [("media_lost", FIGURE, "incomplete")])
        self.assertFalse((out / "assets").exists())

    def test_odt_chart_without_preview_is_an_object_not_converted(self):
        """FA3 (Sol F3): an OpenDocument chart of the content is converted only as its preview image
        written to assets/; without one it is object_not_converted, incomplete."""
        chart = ("object_not_converted", "Object 1/content.xml", "incomplete")
        result, out, md = self.complete(make_odt_chart(self.root / "chart.odt"))
        self.assertIn("Text after the chart.", md)
        self.assertEqual(self.problems(result), [chart])
        self.assertEqual(result["problems"][0]["detail"], "chart")
        # pandoc 3.14 does not extract the replacement image LibreOffice keeps beside a chart
        preview = "ObjectReplacements/Object 1"
        result, out, md = self.complete(make_odt_chart(self.root / "replacement.odt", preview=preview))
        self.assertEqual(self.problems(result), sorted([chart, ("media_lost", preview, "incomplete"),
                                                        ("media_missing", "./" + preview, "incomplete")]))
        result, out, md = self.complete(make_odt_chart(self.root / "picture.odt", preview="Pictures/preview.png"))
        self.assert_image_linked(out, md)                  # the preview written stands for the chart
        self.assertEqual(self.problems(result), [])

    def test_odt_frame_with_a_fallback_image(self):
        """The images after the first in an ODF frame are alternatives of it (D16: an SVM picture and
        its PNG twin); pandoc writes the first, and the twin is no loss once the mediabag is pruned."""
        result, out, md = self.complete(make_odt(self.root / "twin.odt", fallback=True))
        self.assert_image_linked(out, md)
        self.assertEqual(self.problems(result), [])
        self.assertEqual(result["package"]["media_roles"], {"content": 1, "alternate": 1})

    def test_image_the_writer_leaves_out_is_lost_without_an_orphan(self):
        """FA4: an image nested in the alternative text of another (pandoc's ODT reader, D16) stays in
        a Markdown table; in an HTML table the writer turns alternative text into text, so its file
        would be an orphan in assets/: it is removed and the image declared media_lost."""
        result, out, md = self.complete(make_odt_nested(self.root / "pipe.odt", html_table=False))
        self.assert_image_linked(out, md, count=2)
        self.assertEqual(self.problems(result), [])
        result, out, md = self.complete(make_odt_nested(self.root / "html.odt", html_table=True))
        self.assertIn("<table>", md)
        self.assert_image_linked(out, md, count=1)
        self.assertEqual(self.problems(result), [("media_lost", "Pictures/b.png", "incomplete")])
        self.assertEqual(result["corrections"]["counts"].get("asset_unreferenced_removed"), 1)

    def test_inline_image_the_writer_leaves_out_is_lost(self):
        """FB1 (Sol F1, Carbon HF1): the audit lists the members of the package, not an image whose
        bytes are inline in the content (`office:binary-data`, a `data:` URI); when the writer leaves
        it out, the removed file is reconciled by its digest and declared media_lost all the same."""
        for inner in ("binary", "data"):
            with self.subTest(inner=inner):
                result, out, md = self.complete(make_odt_nested(self.root / f"pipe-{inner}.odt", False, inner=inner))
                self.assert_image_linked(out, md, count=2)              # control: a Markdown table keeps both
                self.assertEqual(self.problems(result), [])
                result, out, md = self.complete(make_odt_nested(self.root / f"html-{inner}.odt", True, inner=inner))
                self.assertIn("<table>", md)
                self.assert_image_linked(out, md, count=1)
                self.assertEqual([(p["code"], p["severity"], p.get("detail")) for p in result["problems"]],
                                 [("media_lost", "incomplete", "not in document.md")])
                self.assertEqual(result["corrections"]["counts"].get("asset_unreferenced_removed"), 1)

    def test_a_literal_asset_path_in_the_text_is_no_reference(self):
        """FB2 (Carbon HF2): the text names `assets/asset-0002.png`, the file of the image the writer
        left out; only references (images, links, img src and a href of HTML) keep a file."""
        text = "After the table: a literal example assets/asset-0002.png in the text."
        result, out, md = self.complete(make_odt_nested(self.root / "literal.odt", True, after=text))
        self.assertIn("assets/asset-0002.png in the text", md)
        self.assertFalse((out / "assets" / "asset-0002.png").exists())
        self.assert_image_linked(out, md, count=1)
        self.assertEqual(self.problems(result), [("media_lost", "Pictures/b.png", "incomplete")])

    def test_an_empty_image_is_not_the_first_of_its_frame(self):
        """FB4 (Carbon RR1 a): an image of zero bytes before the picture of a frame does not make the
        picture an alternate; lost in a text box, the picture is media_lost."""
        result, out, md = self.complete(make_odt(self.root / "zero.odt", text_box=True, empty_first=True))
        self.assertEqual(self.problems(result), [("media_lost", "Pictures/fig.png", "incomplete")])

    def test_picture_keeps_its_fallback_image(self):
        """FA4 (Carbon R1): `picture` is unwrapped, its `source` removed, the `img` kept with its text."""
        path = derive_epub(self.root / "picture.epub", lambda t: replace_figure(
            t, '<p><picture><source srcset="figure-en.png 2x"/><img src="figure-en.png" alt="Inline lab figure"/>'
               '</picture> Sentence after picture.</p>'))
        result, out, md = self.complete(path)
        self.assert_image_linked(out, md)
        self.assertIn("Inline lab figure", md)
        self.assertIn("Sentence after picture.", md)
        self.assertEqual(self.problems(result), [])

    def test_image_removed_with_an_active_element_is_lost_without_orphans(self):
        """FA4 (Carbon R1): an image removed with its active element is media_lost, and `assets/`
        holds only the images the document references."""
        path = derive_epub(self.root / "object.epub", lambda t: replace_figure(
            t, '<p><object data="none.svg" type="image/svg+xml"><img src="figure-en.png" alt="Inside object"/>'
               '</object> After object.</p>'))
        result, out, md = self.complete(path)
        self.assertIn("After object.", md)
        self.assertFalse((out / "assets").exists())
        self.assertEqual(self.problems(result), [("media_lost", FIGURE, "incomplete")])
        self.assertEqual([i["path"] for i in result["inventory"]], ["diagnostics/pre-filter.md", "document.md"])

    def test_epub_member_with_a_percent_encoded_name_is_extracted(self):
        """FA6 (Carbon R3): `OEBPS/figure space.png` referenced as `figure%20space.png`."""
        path = derive_epub(self.root / "space.epub", lambda t: t.replace('src="figure-en.png"', 'src="figure%20space.png"'),
                           rename={FIGURE: "OEBPS/figure space.png"},
                           edits={"OEBPS/content.opf": lambda s: s.replace('href="figure-en.png"', 'href="figure%20space.png"')})
        result, out, md = self.complete(path)
        (asset,) = self.assert_image_linked(out, md)
        self.assertEqual(asset, "asset-0001.png")
        self.assertEqual(self.problems(result), [])


class DjvuTests(Base):
    def test_djvutxt_route(self):
        path = self.root / "text.djvu"
        path.write_bytes(TEXT_DJVU)
        result, out, md = self.complete(path, route="djvutxt")
        self.assertEqual(md, "{0}" + "-" * 48 + "\n\nAlpha Beta \n\n{1}" + "-" * 48 + "\n\n\n")
        self.assertEqual(result["pages"]["pages"], [{"page": 1, "chars": 9}, {"page": 2, "chars": 0}])
        self.assertEqual([(p["code"], p["page"]) for p in result["problems"]], [("page_blank", 2)])
        self.assertEqual(result["route"]["chosen_by"], "user")

    def test_a_djvu_without_text_and_the_page_limit(self):
        from markdown_for_llms.bench.selftest import SYNTHETIC_DJVU
        path = self.root / "image.djvu"
        path.write_bytes(SYNTHETIC_DJVU)
        result, out = self.convert(path, route="djvutxt")
        self.assertEqual((result["status"], result["error"]["code"]), ("failed", "route_invalid"))
        self.assertFalse((out / "document.md").exists())
        path = self.root / "text.djvu"
        path.write_bytes(TEXT_DJVU)
        out = self.root / "limit"
        result = convert(path, out, {"route": "djvutxt"}, {"max_pages": 1}, work=self.root / "work")
        self.assertEqual(result["error"]["code"], "page_limit")


class FailureTests(Base):
    def test_pages_family_is_unavailable_until_tranche_b(self):
        result, out = self.convert(FIXTURES / "C06-pdf-text-it.pdf")
        self.assertEqual((result["status"], result["error"]["code"]), ("failed", "route_unavailable"))
        self.assertEqual(result["route"]["converter"], "marker")
        self.assertEqual(sorted(p.name for p in out.iterdir()), ["diagnostics", "engine.json"])

    def test_scenarios_1_and_6(self):
        """Damaged file, extension and signature that disagree, a format outside the matrix: a closed
        code in engine.json, no document.md."""
        damaged = self.root / "damaged.docx"
        damaged.write_bytes((FIXTURES / "C02-docx-it.docx").read_bytes()[:3000])
        named = self.root / "named.docx"
        shutil.copyfile(FIXTURES / "C06-pdf-text-it.pdf", named)
        other = self.root / "notes.md"
        other.write_text("# x\n")
        for path, code in ((damaged, "input_unreadable"), (named, "format_mismatch"), (other, "format_unsupported")):
            with self.subTest(code=code):
                result, out = self.convert(path)
                self.assertEqual((result["status"], result["error"]["code"]), ("failed", code))
                self.assertFalse((out / "document.md").exists())
                self.assertFalse(result["inventory_complete"])

    def test_an_unsafe_package_is_refused(self):
        bomb = self.root / "bomb.docx"
        make_docx(bomb, para("z" * 100) + " " * (70 << 20))          # 70 MiB of document.xml, a few KiB zipped
        result, out = self.convert(bomb)
        self.assertEqual(result["error"]["code"], "package_unsafe")

    def test_pandoc_heap_cap_in_a_conversion(self):
        html = self.root / "grow.html"
        html.write_text("<!doctype html><html><body>" + "".join(f"<p>w{i} <b>b</b> <i>i</i></p>" for i in range(120000))
                        + "</body></html>", encoding="utf-8")
        result = convert(html, self.root / "out", {}, {"memory_mib": 64}, work=self.root / "work")
        self.assertEqual((result["status"], result["error"]["code"]), ("failed", "tool_memory_limit"))
        self.assertFalse((self.root / "out" / "document.md").exists())

    def test_output_cap_of_a_conversion(self):
        path = make_docx(self.root / "long.docx", para("parola " * 20000) * 60)
        result = convert(path, self.root / "out", {}, {"memory_mib": 2048, "output_mib": 1}, work=self.root / "work")
        self.assertEqual((result["status"], result["error"]["code"]), ("failed", "output_limit"))

    def test_ocr_force_is_refused_for_pandoc(self):
        result, out = self.convert(FIXTURES / "C02-docx-it.docx", ocr="force")
        self.assertEqual(result["error"]["code"], "route_option_unsupported")


if __name__ == "__main__":
    unittest.main()
