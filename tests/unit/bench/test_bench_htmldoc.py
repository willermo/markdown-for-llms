"""HTML renderings read as an evaluable document and an inventory (`bench.htmldoc`, plan r2 §5):
one test per phenomenon of each dialect, the omission proof on a rendered inventory, and the
streaming removal of `data:` URIs (delta r3 Δ10)."""
import os
from pathlib import Path
import tempfile
import unittest

from markdown_for_llms.bench import selfcheck
from markdown_for_llms.bench.codes import BenchError, Code
from markdown_for_llms.bench.htmldoc import read_html, read_html_files, strip_data_uris


def kinds(doc):
    return [b.kind for b in doc.blocks]


class WriterTests(unittest.TestCase):
    def test_notes_header_footer_and_markers(self):
        html = b"""<html><body><div title="header"><p>Running head</p></div>
<p>Body text<a class="sdfootnoteanc" name="sdfootnote1anc" href="#sdfootnote1sym"><sup>1</sup></a> goes on.</p>
<div id="sdfootnote1"><p><a class="sdfootnotesym" name="sdfootnote1sym" href="#sdfootnote1anc">1</a>First part.</p>
<p>Second part.</p></div><div title="footer"><p>Page 1</p></div></body></html>"""
        doc, inv = read_html(html, "writer")
        self.assertEqual([(b.kind, b.text) for b in doc.blocks],
                         [("paragraph", "Body text goes on."), ("footnote", "First part. Second part.")])
        self.assertEqual(inv, {"paragraph": ["body text goes on"], "footnote": ["first part second part"]})

    def test_formula_objects_and_lines_are_not_figures_frames_are(self):
        html = b"""<p>Area <img src="a.gif" name="Object1"/> of the shape <img src="b.png" name="Image2"/> here.</p>
<p><img src="c.png" name="Horizontal line 1"/></p>
<p><img src="d.png" name="Image3"/> <br/><i>Figure 1: </i><i>The shape</i></p>
<h2>2.2<span id="Frame4"><p><img src="e.png" name="Image4"/>Figure 2: Inside</p></span>Your first shape</h2>"""
        doc, inv = read_html(html, "writer", outline=True)
        self.assertEqual(len(doc.images), 3)
        self.assertEqual(sorted(inv["figure"]), ["", "figure 1 the shape", "figure 2 inside"])
        self.assertEqual(inv["paragraph"], ["area of the shape here"])
        self.assertEqual(inv["heading"], ["your first shape"])
        caption = next(b for b in doc.blocks if b.text.startswith("Figure 1"))
        self.assertEqual(caption.emph, [("i", "Figure 1: The shape")])     # runs split by tags, merged

    def test_outline_number_only_for_an_odt_twin(self):
        html = b"<h1>5.5Exercise Solutions</h1><h2>10.4.1 Attribute depth</h2>"
        self.assertEqual(read_html(html, "writer", outline=True)[1]["heading"], ["exercise solutions", "attribute depth"])
        self.assertEqual(read_html(html, "writer")[1]["heading"], ["5 5exercise solutions", "10 4 1 attribute depth"])

    def test_indexes_lists_and_preformatted_lines(self):
        html = b"""<div id="Table of Contents1"><p><a href="#x">1 Intro 3</a></p></div>
<div id="Table of Figures1"><p>Figure 1: x 4</p></div>
<ol><li><h5>REFERENCES</h5></li></ol><ul><li><p>one</p><ul><li>two</li></ul></li></ul>
<pre>&lt;draw:a b=&quot;1&quot;
  c=&quot;2&quot;<br/>d=&quot;3&quot;</pre>"""
        doc, inv = read_html(html, "writer")
        self.assertEqual(inv["heading"], ["references"])
        self.assertEqual(inv["list_item"], ["one", "two"])
        self.assertEqual([b.level for b in doc.blocks if b.kind == "list_item"], [1, 2])
        self.assertEqual(inv["paragraph"], ["draw a b 1", "c 2 d 3"])

    def test_tables_with_spans_and_headers(self):
        html = b"""<table><thead><tr><th colspan="2">H</th></tr></thead>
<tbody><tr><td rowspan="2">A</td><td>B</td></tr><tr><td>C</td></tr></tbody></table>"""
        doc, inv = read_html(html, "writer")
        table = doc.blocks[0]
        self.assertEqual((table.kind, table.rows, table.header_rows), ("table", [["H", ""], ["A", "B"], ["", "C"]], 1))
        self.assertEqual(inv, {"table": ["table"], "cell": ["h", "a", "b", "c"]})


class CalcTests(unittest.TestCase):
    def test_sheet_index_and_prefix_are_not_content(self):
        html = b"""<body><hr><p><center><h1>Overview</h1><A HREF="#table0">Sheet A</A><br>
<A HREF="#table1">Sheet B</A><br></center></p><hr>
<A NAME="table0"><h1>Sheet 1: <em>Prices</em></h1></A><table><tr><td>Bread</td><td>2,5</td></tr></table></body>"""
        doc, inv = read_html(html, "calc")
        self.assertEqual(inv, {"heading": ["prices"], "table": ["table"], "cell": ["bread", "2 5"]})


class LatexmlTests(unittest.TestCase):
    PAGE = b"""<body><header>arXiv site</header><nav class="ltx_page_navbar">toc</nav>
<article class="ltx_document"><h1 class="ltx_title ltx_title_document">The paper</h1>
<div class="ltx_authors">A. Author</div><div class="ltx_abstract"><h6 class="ltx_title ltx_title_abstract">Abstract</h6>
<p class="ltx_p">We study <span class="ltx_text ltx_font_italic">waves</span>.</p></div>
<div class="ltx_keywords"><h6 class="ltx_title ltx_title_keywords">Keywords</h6>x, y</div>
<section><h2 class="ltx_title ltx_title_section"><span class="ltx_tag ltx_tag_section">1 </span>Introduction</h2>
<div class="ltx_para"><p class="ltx_p">Let <math alttext="t^{\\prime}" display="inline"></math> be a time<span class="ltx_note ltx_role_footnote"><sup class="ltx_note_mark">1</sup><span class="ltx_note_outer"><span class="ltx_note_content"><sup class="ltx_note_mark">1</sup>A footnote.</span></span></span>.</p>
<table class="ltx_equationgroup ltx_eqn_table"><tr><td><math alttext="a=b," display="block"></math></td><td><span class="ltx_tag ltx_tag_equation">(1)</span></td></tr>
<tr><td><math alttext="c=d\\hskip 7.11317pt" display="block"></math></td></tr></table></div>
<figure class="ltx_figure"><object data="f.svg"></object><figcaption><span class="ltx_tag ltx_tag_figure">Figure 1: </span>A plot.</figcaption></figure>
</section><section class="ltx_bibliography"><h2 class="ltx_title ltx_title_bibliography">References</h2>
<ul class="ltx_biblist"><li class="ltx_bibitem"><span class="ltx_tag ltx_tag_bibitem">[1]</span><span class="ltx_bibblock">B. Writer, Book.</span></li></ul></section>
</article><footer>site footer</footer></body>"""

    def test_article_only_tags_notes_equations_references(self):
        doc, inv = read_html(self.PAGE, "latexml")
        self.assertEqual(inv["heading"], ["the paper", "introduction"])
        self.assertEqual(inv["paragraph"], ["we study waves", "let be a time"])
        self.assertEqual(inv["footnote"], ["a footnote"])
        self.assertEqual(inv["inline_math"], ["t '"])
        self.assertEqual(inv["formula"], ["a = b , c = d"])
        self.assertEqual(inv["figure"], ["a plot"])
        self.assertEqual(inv["bib_entry"], ["b writer book"])
        self.assertEqual(doc.formulas, [(False, "t'"), (True, "a=b, c=d ")])
        self.assertIn(("i", "waves"), [e for b in doc.blocks for e in b.emph])


class WriterEpubTests(unittest.TestCase):
    def test_notes_by_file_and_anchor_across_documents(self):
        first = b"""<p>Text with a note<a href="#fn1">1</a> and one in the next file<a href="ch2.xhtml#fn9">9</a>.</p>
<hr/><ol><li><p><a name="fn1"></a>The note. <a href="#fnref1">back</a></p></li></ol>"""
        second = b"""<p>Second chapter.</p><ol><li><p><a name="fn9"></a>Far note.<a href="ch1.xhtml#ref9">back</a></p></li>
<li><p>A plain list item.</p></li></ol>"""
        doc, inv = read_html_files([("text/ch1.xhtml", first), ("text/ch2.xhtml", second)])
        self.assertEqual(inv["footnote"], ["the note", "far note"])
        self.assertEqual(inv["paragraph"], ["text with a note and one in the next file", "second chapter"])
        self.assertEqual(inv["list_item"], ["a plain list item"])


class OmissionTests(unittest.TestCase):
    def test_a_dropped_generated_unit_turns_the_rendered_coverage_low(self):
        html = b"".join(b"<p>paragraph number %d of the rendered page here</p>" % k for k in range(12))
        _, inv = read_html(html, "writer")
        generated = {"paragraph": list(inv["paragraph"])}
        self.assertEqual(selfcheck.coverage(inv, generated)["paragraph"][2], 1.0)
        self.assertLess(selfcheck.coverage(inv, generated, drop_every=10)["paragraph"][2], selfcheck.COVERAGE_MIN)


class StripDataUriTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)

    def strip(self, data, chunk):
        (self.dir / "a.html").write_bytes(data)
        strip_data_uris(self.dir / "a.html", self.dir / "b.html", chunk=chunk)
        return (self.dir / "b.html").read_bytes()

    def test_uris_replaced_whatever_the_block_size(self):
        data = (b'<p style="background: url(\'data:image/png;base64,AAAA\') x">data: in the text '
                b'<img src="data:image/gif;base64,BBB="> <i style=x:url(data:a,b)>z</i></p>')
        outs = {self.strip(data, n) for n in range(1, 65)}
        self.assertEqual(len(outs), 1)
        out = outs.pop()
        self.assertNotIn(b"base64", out)
        self.assertIn(b"data: in the text", out)
        self.assertEqual(out.count(b"data:omitted;sha256="), 3)
        self.assertIn(b";bytes=21", out)

    def test_unterminated_or_broken_uri_is_refused_without_a_copy(self):
        for data in (b'<p style="x: url(\'data:abc', b'<img src="data:abc<b>">', b'<img src="data:ab>c">'):
            for n in (1, 3, 64):
                with self.subTest(data=data, chunk=n):
                    with self.assertRaises(BenchError) as ctx:
                        self.strip(data, n)
                    self.assertEqual(ctx.exception.code, Code.TWIN_RENDER_INVALID)
                    self.assertFalse(os.path.exists(self.dir / "b.html"))

    def test_an_omitted_uri_is_an_existing_image(self):
        doc, inv = read_html(b'<p><img src="data:omitted;sha256=00;bytes=3" name="Image1"/> Caption</p>', "writer")
        self.assertEqual(doc.images, [("data:omitted;sha256=00;bytes=3", True)])
        self.assertEqual(inv["figure"], ["caption"])


if __name__ == "__main__":
    unittest.main()


class GutenbergTwinTests(unittest.TestCase):
    """The HTML twin of a Project Gutenberg text: its header and licence never become checks."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)

    def test_boilerplate_removed_or_refused(self):
        from markdown_for_llms.bench.readers.htmltwin import read_html_twin
        page = (b'<html><body><section class="pg-boilerplate pgheader" id="pg-header"><p>The Project Gutenberg eBook '
                b'of a story, licence terms</p></section><h1>CAPITOLO I</h1><p>C\xc3\xa8ra una volta <i>un pezzo di '
                b'legno</i>.</p><section class="pg-boilerplate" id="pg-footer"><p>End of the licence</p></section>'
                b'</body></html>')
        (self.dir / "t.html").write_bytes(page)
        twin = read_html_twin(self.dir / "t.html")
        self.assertEqual([(e.type, e.text) for e in twin.elements],
                         [("heading", "CAPITOLO I"), ("paragraph", "Cèra una volta un pezzo di legno.")])
        self.assertEqual(twin.stats, {"boilerplate_removed": 2})
        (self.dir / "u.html").write_bytes(page.replace(b"pg-boilerplate", b"intro"))
        with self.assertRaises(BenchError) as ctx:
            read_html_twin(self.dir / "u.html")
        self.assertEqual(ctx.exception.code, Code.TWIN_BOILERPLATE_UNKNOWN)
