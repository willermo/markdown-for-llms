"""The package audit (`engine/pkgaudit.py`, Δ1 of MFL032 with PA3 and PA6), in process.

The audit needs only the standard library; here it runs in the test process on synthetic
packages, in the light child in `test_engine_native`.
"""
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import zipfile

from engine_test_support import (A, PIC, R, W, WP, derive_epub, drawing, make_docx, make_odt, make_odt_chart, make_xlsx, para, png,
                                 replace_figure, write_zip)
from markdown_for_llms.engine import pkgaudit

ODD = f'xmlns:w="{W}" xmlns:r="{R}" xmlns:ns2="{WP}" xmlns:ns3="{A}" xmlns:ns4="{PIC}"'


class Base(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.out = self.root / "audit"
        self.out.mkdir()

    def audit(self, fmt, path):
        return pkgaudit.audit(fmt, path, self.out)

    def roles(self, report):
        return {m["member"]: m["role"] for m in report["media"]}

    def normalized_part(self, fmt, part):
        with zipfile.ZipFile(self.out / ("normalized." + fmt)) as z:
            return z.read(part).decode("utf-8")


class RolesTests(Base):
    def test_docx_content_furniture_and_unused(self):
        hdr = (f'<?xml version="1.0"?><w:hdr xmlns:w="{W}" xmlns:r="{R}" xmlns:wp="{WP}" xmlns:a="{A}" xmlns:pic="{PIC}">'
               f'<w:p><w:r>{drawing("rIdH")}</w:r></w:p></w:hdr>')
        path = make_docx(self.root / "d.docx", para("Body") + f"<w:p><w:r>{drawing('rId1')}</w:r></w:p>",
                         media={"fig.png": png(), "logo.png": png(4, 4, (1, 2, 3)), "spare.png": png(3, 3)},
                         rels=[("rId1", "image", "media/fig.png"), ("rId9", "header", "header1.xml")],
                         extra={"word/header1.xml": hdr,
                                "word/_rels/header1.xml.rels": f'<?xml version="1.0"?><Relationships xmlns="{pkgaudit.PKG_R}">'
                                f'<Relationship Id="rIdH" Type="{R}/image" Target="media/logo.png"/></Relationships>'})
        roles = self.roles(self.audit("docx", path))
        self.assertEqual(roles, {"word/media/fig.png": "content", "word/media/logo.png": "furniture",
                                 "word/media/spare.png": "unused"})

    def test_objects_of_the_content_are_listed(self):
        body = para("x") + '<w:p><w:r><w:object><o:OLEObject xmlns:o="urn:schemas-microsoft-com:office:office" r:id="rId2"/></w:object></w:r></w:p>'
        path = make_docx(self.root / "o.docx", body, rels=[("rId2", "oleObject", "embeddings/obj.bin")],
                         extra={"word/embeddings/obj.bin": b"ole"})
        rows = {m["member"]: m for m in self.audit("docx", path)["media"]}
        self.assertEqual((rows["word/embeddings/obj.bin"]["kind"], rows["word/embeddings/obj.bin"]["role"]),
                         ("oleObject", "content"))

    def test_xlsx_sheet_images_and_unused_drawing_relation(self):
        """PA6: the drawing of a sheet counts, and its images are placed, only when the sheet uses it."""
        report = self.audit("xlsx", make_xlsx(self.root / "s.xlsx"))
        (sheet,) = report["sheets"]
        self.assertEqual([(i["cell"], i["descr"], i["via"]) for i in sheet["images"]], [("A4", "Synthetic figure", "drawingml")])
        self.assertEqual(self.roles(report), {"xl/media/image1.png": "content"})
        self.assertTrue((self.out / "media" / sheet["images"][0]["file"]).is_file())
        out2 = self.root / "audit2"
        out2.mkdir()
        report = pkgaudit.audit("xlsx", make_xlsx(self.root / "u.xlsx", drawing_used=False), out2)
        self.assertEqual(report["sheets"][0]["drawings"], [])
        self.assertEqual(report["sheets"][0]["images"], [])
        self.assertEqual(self.roles(report), {"xl/media/image1.png": "furniture"})

    def test_epub_srcset_is_a_reference(self):
        """PA3 (Sol F2): C03 with the figure referenced by `srcset` only is content, never unused."""
        path = derive_epub(self.root / "c03-srcset.epub",
                           lambda t: t.replace('src="figure-en.png"', 'srcset="figure-en.png 1x"'))
        self.assertEqual(self.roles(self.audit("epub", path))["OEBPS/figure-en.png"], "content")
        path = derive_epub(self.root / "c03-source.epub", lambda t: t.replace(
            '<img src="figure-en.png"', '<picture><source srcset="figure-en.png 2x, other.png 1x"/></picture><img'))
        out2 = self.root / "audit2"
        out2.mkdir()
        self.assertEqual(self.roles(pkgaudit.audit("epub", path, out2))["OEBPS/figure-en.png"], "content")

    def test_epub_css_of_the_spine_is_content_or_unresolved(self):
        """FA2 (Sol F2): a CSS reference from a content document is never furniture: an inline
        style is content, a `style` block or a stylesheet of the spine (selectors not matched) is
        unresolved; only the cover named by the metadata is furniture (D11)."""
        figure = "OEBPS/figure-en.png"
        no_img = lambda t: replace_figure(t, "")
        cases = {
            "inline": (lambda t: replace_figure(t, '<div style="background-image:url(figure-en.png);width:9px"></div>'), {}),
            "block": (lambda t: no_img(t).replace("</head>", "<style>.f { background: url('figure-en.png') }</style></head>"), {}),
            "sheet": (no_img, {"OEBPS/style.css": lambda s: s + " .f { background: url(figure-en.png) }"}),
            "cover": (no_img, {"OEBPS/content.opf": lambda s: s.replace('id="fig" href', 'id="fig" properties="cover-image" href')}),
        }
        expected = {"inline": "content", "block": "unresolved", "sheet": "unresolved", "cover": "furniture"}
        for name, (edit, edits) in cases.items():
            with self.subTest(name=name):
                out = self.root / name
                out.mkdir()
                report = pkgaudit.audit("epub", derive_epub(self.root / f"{name}.epub", edit, edits=edits), out)
                self.assertEqual(self.roles(report)[figure], expected[name])

    def test_odt_objects_of_the_content_are_listed(self):
        """FA3 (Sol F3): a chart object of the content is listed with its member and origin, and with
        its preview image when it has one; formulas keep their own treatment."""
        rows = {m["member"]: m for m in self.audit("odt", make_odt_chart(self.root / "c.odt"))["media"]}
        chart = rows["Object 1/content.xml"]
        self.assertEqual((chart["kind"], chart["role"], chart["from"], chart.get("preview")),
                         ("chart", "content", ["content.xml"], None))
        self.assertGreater(chart["bytes"], 0)
        out = self.root / "preview"
        out.mkdir()
        rows = {m["member"]: m for m in pkgaudit.audit("odt", make_odt_chart(self.root / "p.odt", "ObjectReplacements/Object 1"),
                                                       out)["media"]}
        self.assertEqual(rows["Object 1/content.xml"]["preview"], "ObjectReplacements/Object 1")
        self.assertEqual(rows["ObjectReplacements/Object 1"]["role"], "content")

    def test_odt_empty_image_is_no_picture_of_its_frame(self):
        """FB4 (Carbon RR1 a): the first image of a frame with bytes is its picture, not an alternate."""
        roles = self.roles(self.audit("odt", make_odt(self.root / "z.odt", empty_first=True, fallback=True)))
        self.assertEqual((roles["Pictures/fig.png"], roles["Pictures/fallback.png"]), ("content", "alternate"))

    def test_epub_unresolved_reference_never_proves_unused(self):
        for bad in ("fig%ZZ.png", "fig%FF.png"):          # a malformed escape, a byte that is not UTF-8
            out = self.root / bad
            out.mkdir()
            path = derive_epub(self.root / (bad + ".epub"), lambda t, bad=bad: t.replace('src="figure-en.png"', f'src="{bad}"'))
            report = pkgaudit.audit("epub", path, out)
            self.assertEqual(report["unresolved_references"], 1)
        self.assertEqual(self.roles(report)["OEBPS/figure-en.png"], "unresolved")

    def test_unknown_structure_proves_nothing_unused(self):
        path = write_zip(self.root / "x.docx", {"[Content_Types].xml": "<Types/>", "word/document.xml": "<w:document/>",
                                                "word/media/a.png": png()})
        report = self.audit("docx", path)
        self.assertEqual((report["structure"], self.roles(report)), ("unknown", {"word/media/a.png": "unresolved"}))


class NormalizationTests(Base):
    def test_c02_prefixes_are_rewritten_with_the_standard_ones(self):
        path = make_docx(self.root / "c02like.docx", para("A") + f"<w:p><w:r>{drawing('rId1', ('ns2', 'ns3', 'ns4'))}</w:r></w:p>",
                         media={"fig.png": png()}, root_ns=ODD)
        report = self.audit("docx", path)
        self.assertEqual(report["normalized"], {"word/document.xml": ["ns2->wp", "ns3->a", "ns4->pic"]})
        xml = self.normalized_part("docx", "word/document.xml")
        self.assertIn("<wp:inline>", xml)
        self.assertIn('<a:blip r:embed="rId1"/>', xml)
        self.assertIn(f'xmlns:ns2="{WP}"', xml)                # the original declaration stays
        self.assertIn(f'xmlns:wp="{WP}"', xml)
        with zipfile.ZipFile(path) as a, zipfile.ZipFile(self.out / "normalized.docx") as b:
            self.assertEqual([i.filename for i in a.infolist()], [i.filename for i in b.infolist()])
            for name in a.namelist():
                if name != "word/document.xml":
                    self.assertEqual(a.read(name), b.read(name))

    def test_local_declarations_mce_values_comments_and_pis_are_kept(self):
        """PA6: a prefix declared on an inner element is found; `mc:Ignorable` keeps its binding;
        comments and processing instructions stay."""
        mc = "http://schemas.openxmlformats.org/markup-compatibility/2006"
        w14 = "http://schemas.microsoft.com/office/word/2010/wordml"
        body = (para("Alpha") + "<!-- a comment -->" + "<?mdf keep?>"
                + f'<w:p xmlns:mc="{mc}" xmlns:w14="{w14}" mc:Ignorable="w14" w14:paraId="1"><w:r>'
                + drawing("rId1", ("q1", "q2", "q3")).replace("<w:drawing>", f'<w:drawing xmlns:q1="{WP}" xmlns:q2="{A}" xmlns:q3="{PIC}">')
                + "</w:r></w:p>")
        path = make_docx(self.root / "local.docx", body, media={"fig.png": png()}, root_ns=f'xmlns:w="{W}" xmlns:r="{R}"')
        report = self.audit("docx", path)
        self.assertEqual(report["normalized"]["word/document.xml"], ["q1->wp", "q2->a", "q3->pic"])
        xml = self.normalized_part("docx", "word/document.xml")
        self.assertIn('mc:Ignorable="w14"', xml)
        self.assertIn(f'xmlns:w14="{w14}"', xml)
        self.assertIn("<!-- a comment -->", xml)
        self.assertIn("<?mdf keep?>", xml)
        self.assertIn("<wp:inline>", xml)

    def test_default_namespace_is_kept_and_prefixes_of_a_sheet_rewritten(self):
        """PA6: a default spreadsheetml namespace stays a default namespace (no `ns0:`)."""
        ns = f'xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:rr="{R}"'
        path = make_xlsx(self.root / "s.xlsx", sheet_ns=ns)
        with zipfile.ZipFile(path) as z:
            members = {n: z.read(n) for n in z.namelist()}
        members["xl/worksheets/sheet1.xml"] = members["xl/worksheets/sheet1.xml"].replace(b'r:id="rId1"', b'rr:id="rId1"')
        path = write_zip(self.root / "s2.xlsx", members)
        report = self.audit("xlsx", path)
        self.assertEqual(report["normalized"], {"xl/worksheets/sheet1.xml": ["rr->r"]})
        xml = self.normalized_part("xlsx", "xl/worksheets/sheet1.xml")
        self.assertIn('<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"', xml)
        self.assertIn('<drawing r:id="rId1"/>', xml)
        self.assertNotIn("ns0:", xml)

    def test_a_standard_prefix_bound_elsewhere_blocks_the_rewrite(self):
        decl = f'xmlns:w="{W}" xmlns:r="{R}" xmlns:ns2="{WP}" xmlns:a="urn:other" xmlns:ns3="{A}" xmlns:ns4="{PIC}"'
        path = make_docx(self.root / "c.docx", f"<w:p><w:r>{drawing('rId1', ('ns2', 'ns3', 'ns4'))}</w:r></w:p>",
                         media={"fig.png": png()}, root_ns=decl)
        report = self.audit("docx", path)
        self.assertEqual(report["normalized"], {})
        self.assertIn("word/document.xml", report["not_normalized"])
        self.assertFalse((self.out / "normalized.docx").exists())

    def test_a_part_over_the_normalization_limit_is_left_as_it_is(self):
        path = make_docx(self.root / "c.docx", f"<w:p><w:r>{drawing('rId1', ('ns2', 'ns3', 'ns4'))}</w:r></w:p>",
                         media={"fig.png": png()}, root_ns=ODD)
        with mock.patch.object(pkgaudit, "MAX_NORMALIZE", 100):
            report = self.audit("docx", path)
        self.assertEqual(report["not_normalized"], {"word/document.xml": "part over the normalization limit"})


class LimitTests(Base):
    def test_decompressed_bytes_are_counted_while_streaming(self):
        """PA6: limits on the bytes actually decompressed, per member and in total, the copy included."""
        path = make_docx(self.root / "c.docx", para("x" * 5000) + f"<w:p><w:r>{drawing('rId1', ('ns2', 'ns3', 'ns4'))}</w:r></w:p>",
                         media={"fig.png": png()}, root_ns=ODD)
        with mock.patch.object(pkgaudit, "MAX_PART", 1000), self.assertRaises(pkgaudit.Refused):
            self.audit("docx", path)
        report = self.audit("docx", path)
        size = sum(i.file_size for i in zipfile.ZipFile(path).infolist())
        self.assertGreater(report["bytes_read"], size)              # the copy into normalized.docx is counted
        (self.out / "normalized.docx").unlink()
        with mock.patch.object(pkgaudit, "MAX_TOTAL", size), self.assertRaises(pkgaudit.Refused):
            self.audit("docx", path)

    def test_a_lying_member_size_is_refused(self):
        path = make_docx(self.root / "c.docx", para("y" * 20000))
        data = bytearray(path.read_bytes())
        info = zipfile.ZipFile(path).getinfo("word/document.xml")
        # shrink the uncompressed size that the central directory records for word/document.xml
        pos = data.find(b"PK\x01\x02")
        while pos >= 0:
            name_len = int.from_bytes(data[pos + 28:pos + 30], "little")
            if data[pos + 46:pos + 46 + name_len] == b"word/document.xml":
                break
            pos = data.find(b"PK\x01\x02", pos + 4)
        self.assertGreaterEqual(pos, 0)
        self.assertEqual(int.from_bytes(data[pos + 24:pos + 28], "little"), info.file_size)
        data[pos + 24:pos + 28] = (100).to_bytes(4, "little")
        bad = self.root / "lying.docx"
        bad.write_bytes(bytes(data))
        with self.assertRaises(pkgaudit.Refused):
            self.audit("docx", bad)

    def test_doctype_and_too_many_members_are_refused(self):
        path = make_docx(self.root / "d.docx", para("x"))
        with zipfile.ZipFile(path) as z:
            members = {n: z.read(n) for n in z.namelist()}
        members["word/document.xml"] = (b'<?xml version="1.0"?><!DOCTYPE w [<!ENTITY a "x">]>'
                                        + members["word/document.xml"].split(b"?>", 1)[1])
        with self.assertRaises(pkgaudit.Refused):
            self.audit("docx", write_zip(self.root / "dtd.docx", members))
        with mock.patch.object(pkgaudit, "MAX_MEMBERS", 3), self.assertRaises(pkgaudit.Refused):
            self.audit("docx", path)

    def test_main_writes_the_refusal(self):
        with mock.patch.object(pkgaudit, "audit", side_effect=pkgaudit.Refused("x")), \
                mock.patch.object(pkgaudit, "Path") as fake_path:
            self.assertEqual(pkgaudit.main(["docx", "/in/source.docx"]), pkgaudit.REFUSED_EXIT)
        fake_path.return_value.write_text.assert_called_once_with(json.dumps({"refused": "x"}))


if __name__ == "__main__":
    unittest.main()
