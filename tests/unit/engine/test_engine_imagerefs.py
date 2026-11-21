"""MFL043: the `<img>` tags of a Datalab answer are rewritten towards `assets/` like the Markdown
images, from `datalab._collect` (the step that keeps or drops the images of the answer). The
answers are synthetic; no network. Criteria: every image the document uses in either form becomes
an asset, numbered by its first occurrence; only the value of `src` changes (`alt`, `data-bbox`,
quotes and the rest of the tag byte for byte); code and inline math are never touched; a source
that is not an image of the answer follows the partition of the contract filter (http(s) kept,
`media_external`; any other replaced by its alternative text, `media_missing`); the document
passes the product's reference check without `invalid`."""
import base64
from pathlib import Path
import tempfile
import types
import unittest

from markdown_for_llms import references
from markdown_for_llms.engine import datalab
from markdown_for_llms.engine.codes import EngineError

from engine_test_support import png

PNG = png()
JPEG = b"\xff\xd8\xff\xe0" + b"jpeg-bytes"


class CollectCase(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.stage = Path(tmp.name) / "stage"
        (self.stage / "diagnostics").mkdir(parents=True)

    def collect(self, markdown, images):
        ctx = types.SimpleNamespace(stage=self.stage, problems=[], cost={}, datalab_versions=None)
        body = {"success": True, "markdown": markdown, "page_count": 1,
                "images": {name: base64.b64encode(data).decode() for name, data in images.items()}}
        record = datalab._collect(ctx, body, 1, "balanced", {}, datalab.Decimal("1"))
        text = (self.stage / "document.md").read_text(encoding="utf-8")
        assets = sorted(p.relative_to(self.stage).as_posix() for p in (self.stage / "assets").iterdir())
        codes = sorted(p["code"] for p in ctx.problems if p["code"] != "datalab_cost_unknown")
        inventory = {a: (self.stage / a).read_bytes() for a in assets}
        check = references.inspect(text, inventory)
        self.assertNotEqual(check["bundle"], "invalid", check["issues"])
        return text, inventory, codes, record, check


class RewriteTests(CollectCase):
    def test_several_img_in_one_block_and_markdown_images_numbered_by_first_use(self):
        md = ('<p><img alt="Logo: a \'quoted\' mark" data-bbox="1104 201 1423 510" src="b_img.jpg"/>\n'
              '<img alt="Second" data-bbox="1 2 3 4" src="a_img.jpg"/></p>\n\n'
              'Text ![Icon](c_img.jpg) and again <img src="b_img.jpg" alt="again">\n')
        text, inv, codes, record, check = self.collect(md, {"a_img.jpg": JPEG, "b_img.jpg": JPEG, "c_img.jpg": PNG})
        self.assertEqual(text, '<p><img alt="Logo: a \'quoted\' mark" data-bbox="1104 201 1423 510" '
                               'src="assets/asset-0001.jpg"/>\n'
                               '<img alt="Second" data-bbox="1 2 3 4" src="assets/asset-0002.jpg"/></p>\n\n'
                               'Text ![Icon](assets/asset-0003.png) and again <img src="assets/asset-0001.jpg" '
                               'alt="again">\n')
        self.assertEqual(list(inv), ["assets/asset-0001.jpg", "assets/asset-0002.jpg", "assets/asset-0003.png"])
        self.assertEqual(codes, [])                                   # no media_unused
        self.assertEqual(record["images"], 3)
        self.assertEqual(check["bundle"], "ready", check["issues"])

    def test_inline_img_in_table_cells(self):
        md = ("| Figure | Caption |\n|---|---|\n"
              "| <img src='a_img.jpg' alt=\"first\"> | one |\n"
              "| two | <img data-bbox=\"9 9 9 9\" src=b_img.jpg> and ![x](a_img.jpg) |\n")
        text, inv, codes, _, check = self.collect(md, {"a_img.jpg": JPEG, "b_img.jpg": JPEG})
        self.assertIn("| <img src='assets/asset-0001.jpg' alt=\"first\"> | one |", text)
        self.assertIn('| two | <img data-bbox="9 9 9 9" src="assets/asset-0002.jpg"> and ![x](assets/asset-0001.jpg) |',
                      text)
        self.assertEqual((len(inv), codes, check["bundle"]), (2, [], "ready"))

    def test_alt_with_apostrophes_a_greater_than_and_src_inside(self):
        md = 'See <img alt="it\'s > src=b_img.jpg here" src="a_img.jpg"> now.\n'
        text, inv, codes, _, _ = self.collect(md, {"a_img.jpg": JPEG, "b_img.jpg": JPEG})
        self.assertEqual(text, 'See <img alt="it\'s > src=b_img.jpg here" src="assets/asset-0001.jpg"> now.\n')
        self.assertEqual((list(inv), codes), (["assets/asset-0001.jpg"], ["media_unused"]))

    def test_img_over_several_lines(self):
        md = '<div>\n<img\n  src="a_img.jpg"\n  alt="multi\nline" data-bbox="1 2 3 4">\n</div>\n'
        text, inv, codes, _, _ = self.collect(md, {"a_img.jpg": JPEG})
        self.assertEqual(text, md.replace('src="a_img.jpg"', 'src="assets/asset-0001.jpg"'))
        self.assertEqual((len(inv), codes), (1, []))

    def test_code_and_math_are_never_touched(self):
        md = ('Literal `<img src="a_img.jpg">` and a real <img src="a_img.jpg" alt="r"> and '
              '$<img src="a_img.jpg">$ math.\n\n```html\n<img src="a_img.jpg">\n```\n')
        text, inv, codes, _, _ = self.collect(md, {"a_img.jpg": JPEG})
        self.assertEqual(text, md.replace('a real <img src="a_img.jpg"', 'a real <img src="assets/asset-0001.jpg"'))
        self.assertEqual(text.count("assets/asset-0001.jpg"), 1)
        self.assertEqual((len(inv), codes), (1, []))


class UnextractedSourceTests(CollectCase):
    def test_http_source_stays_and_is_external(self):
        md = 'A <img src="https://example.org/f.png" alt="remote"> and <img src="a_img.jpg">\n'
        text, inv, codes, _, check = self.collect(md, {"a_img.jpg": JPEG})
        self.assertIn('<img src="https://example.org/f.png" alt="remote">', text)
        self.assertEqual(codes, ["media_external"])
        self.assertIn("external-no-fetch", [r["kind"] for r in check["references"]])

    def test_absolute_and_unknown_sources_become_their_alternative_text(self):
        md = ('A <img src="/absolute.jpg" alt="A &amp; B <c>"> b <img src="missing.jpg" alt="gone"> '
              'c <img alt="no source">\n')
        text, inv, codes, _, check = self.collect(md, {})
        self.assertEqual(text, "A A &amp; B &lt;c&gt; b gone c no source\n")
        self.assertEqual((inv, codes), ({}, ["media_missing"] * 3))
        self.assertEqual(check["bundle"], "ready", check["issues"])

    def test_alternative_text_keeps_a_table_row_whole(self):
        md = "| a | b |\n|---|---|\n| <img src=\"/x.jpg\" alt=\"one&#124;two&#10;three\"> | c |\n"
        text, _, codes, _, _ = self.collect(md, {})
        self.assertEqual(text, "| a | b |\n|---|---|\n| one&#124;two three | c |\n")
        self.assertEqual(codes, ["media_missing"])
        from markdown_it import MarkdownIt
        cells = [t for t in MarkdownIt("commonmark").enable("table").parse(text) if t.type == "td_open"]
        self.assertEqual(len(cells), 2)


class RefusalTests(CollectCase):
    def test_a_markdown_reference_in_code_is_refused_not_rewritten(self):
        with self.assertRaises(EngineError) as e:
            self.collect("Code `![x](a_img.jpg)` and ![x](a_img.jpg)\n", {"a_img.jpg": JPEG})
        self.assertEqual(e.exception.code.value, "output_invalid")


if __name__ == "__main__":
    unittest.main()
