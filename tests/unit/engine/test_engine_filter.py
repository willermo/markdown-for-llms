"""The contract filter (`engine/filters/contract.lua`) against hostile inputs: PA4 (destinations
the product's reference profile refuses) and PA5 (raw HTML), plus the hostile HTML of 12.2.

The raw HTML cases are read by pandoc's Markdown reader, which keeps raw HTML as nodes (the
born-digital readers of the matrix produce such nodes too, for example the EPUB reader); the
destination cases go through the html reader of the matrix with the full `convert`.
"""
import base64
import contextlib
import io
import json
from pathlib import Path
import re
import tempfile
import unittest

from engine_test_support import inspect_out, png, require_toolchain, run_filter
from markdown_for_llms import cli, references
from markdown_for_llms.engine import contract
from markdown_for_llms.engine.core import convert

# PA4: the forms of the two reviews (Sol F3, Carbon R2.1) that `references.inspect` refuses
UNSAFE = ["https://example.invalid/%5csecret", "https://example.invalid/%00x", "https://example.com/a%5Cb",
          "https://example.com/a\\b", "https://example.com/%E0%A4%A", "http://[bad]/path", "#sec%5C1",
          "mailto:user@example.com?subject=a%5Cb"]
SAFE = ["https://example.invalid/ok?q=1#f", "#top", "mailto:someone@example.invalid"]
PANDOC_API = [1, 23, 1, 2]       # pandoc 3.14 (the version of toolchain.json)


def references_kinds(md):
    return [r.get("kind") for r in references.inspect(md, set())["references"]]


class DestinationTests(unittest.TestCase):
    def setUp(self):
        require_toolchain()
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)

    def test_unsafe_destinations_are_unlinked_and_safe_ones_kept(self):
        links = "".join(f'<p><a href="{u}">link{i}</a></p>' for i, u in enumerate(UNSAFE + SAFE))
        images = "".join(f'<p><img src="{u}" alt="img{i}"></p>' for i, u in enumerate(UNSAFE[:3]))
        html = self.root / "dest.html"
        html.write_text(f'<!doctype html><html><body><h1 id="top">T</h1>{links}{images}</body></html>', encoding="utf-8")
        out = self.root / "out"
        result = convert(html, out, {}, {"memory_mib": 1024}, work=self.root / "work")
        self.assertEqual(result["status"], "complete")
        md = (out / "document.md").read_text(encoding="utf-8")
        inspected = inspect_out(out)
        self.assertNotIn("invalid", [i["code"] for i in inspected["issues"]])
        self.assertNotEqual(inspected["bundle"], "invalid")
        for i in range(len(UNSAFE) + len(SAFE)):
            self.assertIn(f"link{i}", md)                       # the text of every link stays
        for u in SAFE[:1]:
            self.assertIn(u, md)
        for u in UNSAFE:
            self.assertNotIn("(" + u, md)
        missing = [p for p in result["problems"] if p["code"] == "media_missing"]
        self.assertEqual(len(missing), 3)
        self.assertEqual({p["severity"] for p in missing}, {"incomplete"})
        self.assertEqual(result["corrections"]["counts"].get("link_unlinked"), len(UNSAFE))
        self.assertEqual(result["corrections"]["counts"].get("image_unlinked"), 3)

    def test_mailto_with_an_authority_and_an_encoded_mailto_image(self):
        """FA5 (Carbon R2): `mailto://...` has an authority that `references.inspect` refuses, and an
        image whose scheme is `mailto` once decoded is no remote image."""
        html = self.root / "mailto.html"
        html.write_text('<!doctype html><html><body><p><a href="mailto://[bad]/x">bad_ipv6</a></p>'
                        '<p><a href="mailto://%5Bbad%5D/x">bad_encoded</a></p>'
                        '<p><a href="mailto://%EF%BC%A1%EF%BC%8F/x">bad_nfkc</a></p>'
                        '<p><a href="mailto:someone@example.invalid">good_mailto</a></p>'
                        '<p><img src="%6d%61%69%6c%74%6f:user@example.com" alt="encoded_mailto_img"></p>'
                        '<p><img src="mailto:user@example.com" alt="plain_mailto_img"></p></body></html>', encoding="utf-8")
        out = self.root / "out"
        result = convert(html, out, {}, {"memory_mib": 1024}, work=self.root / "work")
        self.assertEqual(result["status"], "complete")
        md = (out / "document.md").read_text(encoding="utf-8")
        inspected = inspect_out(out)
        self.assertNotIn("invalid", [i["code"] for i in inspected["issues"]])
        self.assertNotEqual(inspected["bundle"], "invalid")
        for text in ("bad_ipv6", "bad_encoded", "bad_nfkc", "good_mailto", "encoded_mailto_img", "plain_mailto_img"):
            self.assertIn(text, md)
        self.assertNotIn("mailto://", md)
        self.assertNotIn("%6d%61", md)
        self.assertIn("(mailto:someone@example.invalid)", md)
        self.assertEqual(sorted((p["code"], p["severity"]) for p in result["problems"]),
                         [("media_missing", "incomplete")] * 2)
        self.assertEqual(result["corrections"]["counts"].get("link_unlinked"), 3)

    def test_hostile_html_document(self):
        """12.2 (Δ5): the hostile HTML of the plan, with the html reader of the matrix."""
        data = base64.b64encode(png()).decode()
        html = self.root / "nasty.html"
        html.write_text(f"""<!doctype html><html><head><title>t</title><script>alert(1)</script>
<style>p{{color:red}}</style></head><body><h1 id="sec">Sezione</h1>
<p onclick="x()">Testo <span style="color:red">rosso</span> e <a href="#sec">ancora</a>,
<a href="/abs/page">assoluto</a>, <a href="javascript:alert(1)">js</a>, <a href="ch2.html">capitolo</a>,
<a href="https://example.invalid/x">esterno</a>.</p><div style="width:10px"><p>Div con stile</p></div>
<p><img src="/img/a.png" alt="assoluta"> <img src="../b.png" alt="sopra"> <img src="//cdn.invalid/c.png" alt="rete">
<img src="file:///etc/passwd" alt="file"> <img src="local/d.png" alt="locale assente">
<img src="https://example.invalid/e.png" alt="remota"> <img src="data:image/png;base64,{data}" alt="incorporata"></p>
<video poster="/p.png" src="/v.mp4">video</video>
<table><tr><td style="width:3em"><img src="/t.png" srcset="/t2.png 2x" alt="in tabella"></td><td>x</td></tr></table>
</body></html>""", encoding="utf-8")
        out = self.root / "out"
        result = convert(html, out, {}, {"memory_mib": 1024}, work=self.root / "work")
        self.assertEqual(result["status"], "complete")
        md = (out / "document.md").read_text(encoding="utf-8")
        for bad in ("script", "onclick", "style=", "javascript:", "/abs/", "file:", "srcset", "poster", "//cdn", "/etc/passwd"):
            self.assertNotIn(bad, md)
        self.assertEqual(inspect_out(out)["bundle"], "ready")
        codes = sorted(p["code"] for p in result["problems"])
        self.assertEqual(codes.count("media_missing"), 6)
        self.assertEqual(codes.count("media_external"), 1)
        self.assertEqual(len(list((out / "assets").iterdir())), 1)


class ProblemCountTests(unittest.TestCase):
    """FA1 (Sol F1): the counts of the problems are taken before any truncation of their items."""

    def test_one_incomplete_after_2000_warnings(self):
        require_toolchain()
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        html = root / "many.html"
        html.write_text("<!doctype html><html><body>"
                        + '<p><img src="https://example.invalid/a.png" alt="remote"/></p>' * 2000
                        + '<p><img src="missing.png" alt="lost-final"/></p></body></html>', encoding="utf-8")
        result = convert(html, root / "out", {}, {"memory_mib": 1024}, work=root / "work")
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["problem_counts"], {"incomplete": 1, "warning": 2000})
        self.assertEqual(result["problem_codes"], {"media_external": 2000, "media_missing": 1})
        self.assertLessEqual(len(result["problems"]), contract.MAX_PROBLEMS)
        self.assertEqual(result["problems"][0], {"code": "media_missing", "severity": "incomplete", "path": "missing.png"})
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(io.StringIO()):
            code = cli.main(["--data-root", str(root / "data"), "--model-root", str(root / "models"),
                             "convert", str(html), "--out", str(root / "cli")])
        summary = json.loads(stdout.getvalue())
        self.assertEqual((code, summary["status"]), (0, "complete"))
        self.assertEqual(summary["problem_counts"], {"incomplete": 1, "warning": 2000})
        self.assertEqual(summary["problem_codes"], {"media_external": 2000, "media_missing": 1})
        self.assertEqual(summary["problems"][0]["code"], "media_missing")


class NestedImageTests(unittest.TestCase):
    """An image inside the alternative text of another image (as pandoc's ODT reader can leave it,
    D16): the Markdown writer keeps it inside the description, so the filter keeps its media."""

    def test_image_in_the_alternative_text_of_an_image(self):
        outer = "data:image/png;base64," + base64.b64encode(png()).decode()
        inner = "data:image/png;base64," + base64.b64encode(png(4, 4, (0, 0, 200))).decode()
        image = {"t": "Image", "c": [["", [], []], [{"t": "Str", "c": "outer"}, {"t": "Space"},
                 {"t": "Image", "c": [["", [], []], [{"t": "Str", "c": "inner"}], [inner, ""]]}], [outer, ""]]}
        doc = {"pandoc-api-version": PANDOC_API, "meta": {}, "blocks": [{"t": "Para", "c": [image]}]}
        md, report = run_filter(self, json.dumps(doc), reader="json")
        self.assertEqual(len(report["media"]), 2)
        self.assertIn("![outer ![inner](assets/asset-0001.png)](assets/asset-0002.png)", md)   # inner named first
        self.assertFalse(report["problems"])                       # an empty Lua table: {} in JSON


class RawHtmlTests(unittest.TestCase):
    """PA5 (Carbon R2.2, Sol H2): raw HTML nodes, also split over several nodes."""

    def clean(self, text):
        md, report = run_filter(self, text)
        return md, report

    def assert_inert(self, md):
        lowered = md.lower()
        for bad in ("onclick", "href=", "src=", "style=", "<script", "<svg", "alert(1)", "evil()", "/etc/passwd", "secret.png"):
            self.assertNotIn(bad, lowered)
        self.assertNotEqual(references.inspect(md, set())["bundle"], "invalid")

    def test_case_quotes_unquoted_values_and_duplicates(self):
        md, report = self.clean('Text <a HREF="/etc/passwd">leak</a> and <img SRC="../secret.png" ONCLICK="alert(1)" STYLE="color:red"> '
                                '<span title="1 > 0" src="/etc/passwd" onclick="evil()">body</span> '
                                "<b oNclick=alert(1) class=x class=y>bold</b> <i title='a\"b' id=k>it</i>\n")
        self.assert_inert(md)
        self.assertIn("leak", md)
        self.assertIn("body", md)
        self.assertIn('<b class="x">', md)                  # raw inline HTML is written as `...`{=html}
        self.assertIn('title="a&quot;b"', md)
        self.assertIn('[body]{title="1 > 0"}', md)                 # read as a Span: src and onclick removed from its Attr

    def test_active_elements_split_over_nodes_and_upper_case(self):
        md, report = self.clean('Before <script>alert("*hi*")</script> after.\n\n<SCRIPT>alert(1)</SCRIPT>\n\n'
                                'One <iframe src="https://x.invalid">inner *text*</iframe> two.\n\nEnd.\n')
        self.assert_inert(md)
        for kept in ("Before", "after.", "One", "two.", "End."):
            self.assertIn(kept, md)
        self.assertNotIn("hi", md)
        self.assertNotIn("inner", md)
        self.assertGreaterEqual(report["counts"].get("raw_active_removed", 0), 3)

    def test_svg_is_handled(self):
        md, report = self.clean('Fig: <svg><image xlink:href="https://x.invalid/a.png"/><script>evil()</script></svg> done.\n\n'
                                '<svg viewBox="0 0 1 1"><path d="M0 0"/></svg>\n')
        self.assert_inert(md)
        self.assertIn("done.", md)
        self.assertEqual([p["code"] for p in report["problems"]], ["object_not_converted", "object_not_converted"])

    def test_incomplete_tags_comments_and_declarations_are_removed(self):
        """Raw nodes as a reader can leave them (pandoc JSON): an unclosed tag, a comment hiding a
        script, a declaration, an active element opened in one node and never closed."""
        raw = [["html", '<b title="open'], ["html", "<!-- <script>evil()</script> -->"], ["html", "<!DOCTYPE x>"],
               ["html", '<object data="/etc/passwd">'], ["html", "<p onclick=evil()>"]]
        blocks = [{"t": "Para", "c": [{"t": "Str", "c": "Keep"}, {"t": "Space"}, {"t": "Str", "c": "this."}]}]
        blocks += [{"t": "RawBlock", "c": r} for r in raw[:3]]
        blocks += [{"t": "RawBlock", "c": raw[3]}, {"t": "Para", "c": [{"t": "Str", "c": "hidden"}]}]
        blocks += [{"t": "Para", "c": [{"t": "RawInline", "c": raw[4]}, {"t": "Str", "c": "shown"}]}]
        doc = {"pandoc-api-version": PANDOC_API, "meta": {}, "blocks": blocks}
        md, report = run_filter(self, json.dumps(doc), reader="json")
        self.assert_inert(md)
        self.assertIn("Keep this.", md)
        self.assertNotIn("hidden", md)                       # inside the unclosed active element: removed
        self.assertNotIn("shown", md)
        for kind in ("raw_html_rejected", "raw_markup_removed", "raw_active_unclosed"):
            self.assertIn(kind, report["counts"])

    def test_raw_text_elements_end_at_their_closing_tag(self):
        """HA6 (Carbon I3): inside `script` or `style` a `<` starts no tag; the element ends at its
        closing tag, so what follows in the same node and in the next nodes stays."""
        blocks = [{"t": "RawBlock", "c": ["html", "<script>if (a <b) { x() }</script><p>after script</p>"]},
                  {"t": "Para", "c": [{"t": "Str", "c": "next"}]},
                  {"t": "RawBlock", "c": ["html", "<STYLE>p < b { color: red }</style >"]},
                  {"t": "Para", "c": [{"t": "Str", "c": "last"}]}]
        doc = {"pandoc-api-version": PANDOC_API, "meta": {}, "blocks": blocks}
        md, report = run_filter(self, json.dumps(doc), reader="json")
        self.assert_inert(md)
        for kept in ("after script", "next", "last"):
            self.assertIn(kept, md)
        for gone in ("x()", "color"):
            self.assertNotIn(gone, md)
        self.assertNotIn("raw_active_unclosed", report["counts"])

    def test_attributes_of_every_element_are_cleaned(self):
        md, report = self.clean('::: {style="color:red" onload="evil()" data-x="1" #d .c}\ntext [span]{style="x" ONCLICK="evil()"}\n:::\n\n'
                                '| a |\n|---|\n| b |\n')
        self.assertNotIn("style", md)
        self.assertNotIn("onload", md.lower())
        self.assertNotIn("onclick", md.lower())
        self.assertIn("data-x", md)
        self.assertIsNone(re.search(r"\bstyle\b", md))


if __name__ == "__main__":
    unittest.main()
