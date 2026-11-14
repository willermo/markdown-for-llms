"""Calibration of the self-check (M4b): each test reproduces, on a minimal synthetic twin, a defect
found on the reference set, and checks that a real error of a converter is still penalized."""
from pathlib import Path
import tempfile
import unittest
import zipfile

from markdown_for_llms.bench import checks, selfcheck
from markdown_for_llms.bench.match import evaluate
from markdown_for_llms.bench.model import Builder, Cell
from markdown_for_llms.bench.outdoc import from_markdown
from markdown_for_llms.bench.readers.docx import read_docx
from markdown_for_llms.bench.readers.epub import read_epub
from markdown_for_llms.bench.readers.latex import read_latex
from markdown_for_llms.bench.readers.odt import read_odt
from markdown_for_llms.bench.readers.wikisource import read_wikisource
from markdown_for_llms.bench.readers.xlsx import read_xlsx
from markdown_for_llms.bench.text import math_atoms, unicode_atoms

W = 'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math"'
S = 'xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"'


def outcomes(twin, markdown, kind="docx"):
    body = checks.generate({"id": "D99"}, twin)
    body.update(twin_kind=kind, rules=[])
    return {r["id"]: r["outcome"] for r in evaluate(body, from_markdown(markdown))}, body


class Base(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)

    def zip(self, name, members):
        path = self.root / name
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
            for member, data in members.items():
                z.writestr(member, data)
        return path


class HeadingAlignmentTests(unittest.TestCase):
    def twin(self):
        b = Builder("jats")
        for text, level in (("A title", 1), ("Abstract", 2), ("Methods", 3), ("Results", 3), ("Introduction", 2),
                            ("Methods", 2), ("Results", 2)):
            b.add("heading", text, level=level)
        return b.twin

    OUT = "# A title\n\n## Abstract\n\n### Methods\n\n### Results\n\n## Introduction\n\n## Methods\n\n## Results\n"

    def test_repeated_heading_texts_align_in_order(self):
        got, _ = outcomes(self.twin(), self.OUT)
        self.assertEqual({k: v for k, v in got.items() if k.endswith(":level")},
                         {f"h{k}:level": "pass" for k in range(2, 8)})

    def test_wrong_level_of_the_body_heading_is_still_penalized(self):
        got, _ = outcomes(self.twin(), self.OUT.replace("\n## Methods\n", "\n### Methods\n"))
        self.assertEqual(got["h6:level"], "wrong_level")


class AmbiguousUnitTests(unittest.TestCase):
    def twin(self):
        b = Builder("docx")
        b.add("paragraph", "Implants made of metal-on-metal (MoM) were studied in detail by many groups.")
        b.add("paragraph", "The first section has 1 result and some more words to make it long.")
        b.add("paragraph", "(1)", math=("E=mc^2",))
        b.add("paragraph", "The last paragraph closes the document with a few more words.")
        b.add("paragraph", "MoM Metal-on-metal")
        b.add("list_item", "and", level=1)
        return b.twin

    OUT = ("Implants made of metal-on-metal (MoM) were studied in detail by many groups.\n\n"
           "The first section has 1 result and some more words to make it long.\n\n$E=mc^2$ (1)\n\n"
           "The last paragraph closes the document with a few more words.\n\n- MoM Metal-on-metal\n- and\n")

    def test_short_or_recurring_units_have_no_order_check(self):
        got, body = outcomes(self.twin(), self.OUT)
        orders = sorted(k for k in got if k.endswith(":order"))
        self.assertEqual(orders, ["p1:order", "p2:order"])      # the chain of unique units: p1 -> p2 -> p4
        self.assertEqual(body["not_determinable"]["structure"], 3)      # p3, p5 and li1 have no position
        self.assertEqual(got["p3:math"], "pass")         # the formula of a short unit, anywhere in the output
        self.assertEqual(got["li1:list"], "pass")        # some occurrence of the short unit is a list item

    def test_real_errors_are_still_penalized(self):
        got, _ = outcomes(self.twin(), self.OUT.replace("$E=mc^2$ ", ""))
        self.assertEqual(got["p3:math"], "math")
        swapped = self.OUT.split("\n\n")
        swapped[0], swapped[1] = swapped[1], swapped[0]
        got, _ = outcomes(self.twin(), "\n\n".join(swapped))
        self.assertEqual(got["p1:order"], "order")

    def test_window_starts_at_the_unit_not_in_the_previous_one(self):
        b = Builder("docx")
        b.add("paragraph", "Claim 3. The values and the bounds agree.")
        b.add("paragraph", "Proof. Suppose that the claim fails and let us take the bounds of the values in turn.")
        got, _ = outcomes(b.twin, "Claim 3. The values and the bounds agree.\n\n"
                                  "Proof. Suppose that the claim fails and let us take the bounds of the values in turn.\n")
        self.assertEqual(got["p1:order"], "pass")


class TableRowTests(unittest.TestCase):
    def twin(self):
        b = Builder("docx")
        b.add("table", cells=(Cell("alpha", 0, 0), Cell("beta", 0, 1), Cell("gamma", 1, 0), Cell("delta", 1, 1)), level=0)
        return b.twin

    def test_rows_without_content_are_not_rows(self):
        # GFM needs a header row: a table without header gets an empty one; empty rows carry nothing
        got, _ = outcomes(self.twin(), "|  |  |\n|---|---|\n| alpha | beta |\n|  |  |\n| gamma | delta |\n")
        self.assertEqual(got["t1:shape"], "pass")

    def test_missing_or_extra_rows_are_still_penalized(self):
        got, _ = outcomes(self.twin(), "|  |  |\n|---|---|\n| alpha | beta |\n")
        self.assertEqual(got["t1:shape"], "shape")
        got, _ = outcomes(self.twin(), "| alpha | beta |\n|---|---|\n| gamma | delta |\n| more | rows |\n")
        self.assertEqual(got["t1:shape"], "shape")


class MathAtomTests(unittest.TestCase):
    def test_bracket_commands_and_invisible_operators(self):
        self.assertEqual(math_atoms("\\lbrack a \\rbrack + \\lbrace b \\rbrace"), ["[", "a", "]", "+", "{", "b", "}"])
        self.assertEqual(unicode_atoms("cos⁡(θ)⁢x"), list("cos(θ)x"))
        # a visible atom is never dropped
        self.assertNotEqual(unicode_atoms("cos(θ)"), unicode_atoms("cos(φ)"))

    def test_what_tex_does_not_print_has_no_atom_and_what_it_prints_has_one(self):
        # D01: labels, comments and spacing arguments were counted as printed atoms; \var Greek
        # letters, \ll and \iint were unknown commands with no atom
        self.assertEqual(math_atoms(r"a\label{eq:1}+b"), ["a", "+", "b"])
        self.assertEqual(math_atoms("a % a comment"), ["a"])
        self.assertEqual(math_atoms(r"a\\[0.25cm]b"), ["a", "b"])
        self.assertEqual(math_atoms(r"\varGamma\varepsilon \ll \iint f"), ["Γ", "ε", "≪", "∬", "f"])
        self.assertNotEqual(math_atoms(r"\varGamma"), math_atoms(r"\Gamma x"))   # an atom more is still seen


class DocxEmphasisTests(Base):
    def test_runs_of_one_emphasis_join_without_spaces_and_plain_text_separates_them(self):
        it = '<w:rPr><w:i/></w:rPr>'
        doc = (f'<w:document {W}><w:body><w:p>'
               f'<w:r>{it}<w:t>Intro</w:t></w:r><w:r>{it}<w:t>du</w:t></w:r><w:r>{it}<w:t>ction</w:t></w:r>'
               f'<w:r><w:t xml:space="preserve"> of </w:t></w:r><w:r>{it}<w:t>E. coli</w:t></w:r>'
               f'<w:r><w:t xml:space="preserve"> and </w:t></w:r><w:r>{it}<w:t>S.</w:t></w:r><w:r><w:t xml:space="preserve"> </w:t></w:r>'
               f'<w:r>{it}<w:t>aureus</w:t></w:r></w:p></w:body></w:document>')
        p = read_docx(self.zip("e.docx", {"word/document.xml": doc})).elements[0]
        self.assertEqual(p.text, "Introduction of E. coli and S. aureus")
        self.assertEqual(p.emphasis, (("i", "Introduction"), ("i", "E. coli"), ("i", "S. aureus")))


LATEX = r"""\documentclass{article}
\newtheorem{claim}{Claim}
\begin{document}
\section{Intro}\label{sec:intro}
Before the float.
\begin{figure*}[htb!]\centering\includegraphics{x}\caption{A plot}\end{figure*}
\begin{table}[t]\begin{tabular}{cc}a & b\\\end{tabular}\end{table}
\begin{minipage}[t]{0.5\textwidth}Inside the box.\end{minipage}
\begin{claim}\label{c1}True.\end{claim}
By Claims~\ref{c1} and~\ref{c2}, we have that $x$, and $y$ as in~\cite{a,b}.
\end{document}"""


class LatexFloatAndGapTests(unittest.TestCase):
    def test_placement_options_are_not_text_and_dropped_references_leave_gaps(self):
        twin = read_latex(LATEX.encode(), b"")
        texts = [(e.type, e.text) for e in twin.elements if e.type in ("paragraph", "figure")]
        self.assertEqual(texts, [("paragraph", "Before the float."), ("figure", "A plot"), ("paragraph", "Inside the box."),
                                 ("paragraph", "Claim 1. True."), ("paragraph", "By Claims and , we have that , and as in .")])
        self.assertEqual(twin.elements[-1].gaps, 4)          # two references (2 tokens each) and two citation keys

    def test_gaps_accept_the_printed_numbers_only(self):
        twin = read_latex(LATEX.encode(), b"")
        good = "By Claims 1 and 2, we have that $x$, and $y$ as in [3, 4].\n"
        got, _ = outcomes(twin, good, "latex")
        last = [k for k in got if k.startswith("p") and ":" not in k][-1]
        self.assertEqual(got[last], "pass")
        got, _ = outcomes(twin, "By Claims 1 and 2, we have $x$ as in [3, 4].\n", "latex")   # words lost
        self.assertNotEqual(got[last], "pass")



class LatexLineEndTests(unittest.TestCase):
    r"""D02: the revtex `.bbl` has CRLF line ends and breaks lines between a command and its
    argument (`\bibinfo<CR><LF>  {journal} {...}`, `\href<CR><LF>  {url} {...}`). The carriage
    return ended the argument scan, so field names and URLs became text of the entry; the PDF
    prints "W.-J. Beyn, An integral method ..., Linear Algebra Appl. 436 (2012)"."""
    TEX = "\\documentclass{article}\n\\begin{document}\nText.\n\\bibliography{refs}\n\\end{document}\n"
    BBL = ("\\begin{thebibliography}{1}\n"
           "\\bibitem [{Beyn(2012)}]{Beyn2012}\n"
           "  \\BibitemOpen\n"
           "  \\bibfield  {author} {\\bibinfo {author} {W.-J. Beyn}},\\ \\bibfield  {title} {\\bibinfo\n"
           "  {title} {An integral method}},\\ \\href\n"
           "  {https://doi.org/10.1016/j.laa.2011.03.030} {\\bibfield  {journal} {\\bibinfo\n"
           "  {journal} {Linear Algebra Appl.}\\ }\\textbf {\\bibinfo {volume} {436}}},\\\n"   # control space at the line end
           "  \\bibinfo {pages} {3839} (\\bibinfo {year} {2012})"
           "\\BibitemShut {NoStop}\n"
           "\\end{thebibliography}\n")

    def entries(self, newline):
        twin = read_latex(self.TEX.replace("\n", newline).encode(), self.BBL.replace("\n", newline).encode())
        return [e.text for e in twin.elements if e.type == "bib_entry"]

    def test_crlf_source_reads_as_lf(self):
        (lf,) = self.entries("\n")
        self.assertEqual(lf, "W.-J. Beyn, An integral method, Linear Algebra Appl. 436, 3839 (2012)")
        self.assertEqual(self.entries("\r\n"), [lf])
        self.assertEqual(self.entries("\r"), [lf])         # a lone CR ends a line in TeX too

    def test_control_space_at_a_line_end_in_math_has_no_atom(self):
        self.assertEqual(math_atoms("a\\\nb"), math_atoms("a\\ b"))


class EpubBibliographyAndSvgTests(Base):
    """D13: the references of the EPUB are `csl-entry` divisions and its cover is an inline `svg`;
    the reader made paragraphs of the references and saw no figure (coverage of bib_entry 0,
    figures 0,52 against the EPUB's own markup)."""

    def test_csl_entries_are_references_and_an_inline_svg_is_a_figure(self):
        container = ('<container xmlns="urn:oasis:names:tc:opendocument:xmlns:container"><rootfiles>'
                     '<rootfile full-path="OPS/content.opf"/></rootfiles></container>')
        opf = ('<package xmlns="http://www.idpf.org/2007/opf"><manifest>'
               '<item id="c1" href="c1.xhtml" media-type="application/xhtml+xml"/></manifest>'
               '<spine><itemref idref="c1"/></spine></package>')
        page = ('<html xmlns="http://www.w3.org/1999/xhtml"><body>'
                '<div><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10"><image href="cover.jpg"/></svg></div>'
                '<h1>References</h1><div id="refs" class="references csl-bib-body" role="list">'
                '<div class="csl-entry" role="listitem">Author, A. (2020). <em>A title</em>. Publisher.</div>'
                '<div class="csl-entry" role="listitem">Bee, B. (2021). Another title. J. 1.</div></div></body></html>')
        twin = read_epub(self.zip("t.epub", {"META-INF/container.xml": container, "OPS/content.opf": opf, "OPS/c1.xhtml": page}))
        got = [(e.type, e.text) for e in twin.elements]
        self.assertEqual(got, [("figure", ""), ("heading", "References"),
                               ("bib_entry", "Author, A. (2020). A title. Publisher."),
                               ("bib_entry", "Bee, B. (2021). Another title. J. 1.")])

class XlsxDisplayTests(Base):
    def test_general_format_and_thousands_grouping(self):
        wb = f'<workbook {S}><sheets><sheet name="P" sheetId="1" r:id="rId1"/></sheets></workbook>'
        rels = '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Target="worksheets/sheet1.xml"/></Relationships>'
        styles = f'<styleSheet {S}><cellXfs><xf numFmtId="0"/><xf numFmtId="3"/><xf numFmtId="1"/></cellXfs></styleSheet>'
        sheet = (f'<worksheet {S}><sheetData><row r="1"><c r="A1"><v>8.6999999999999993</v></c><c r="B1"><v>2024</v></c>'
                 '<c r="C1" s="1"><v>12551</v></c><c r="D1" s="2"><v>2025</v></c><c r="E1"><v>0.1</v></c></row></sheetData></worksheet>')
        z = self.zip("n.xlsx", {"xl/workbook.xml": wb, "xl/_rels/workbook.xml.rels": rels, "xl/styles.xml": styles,
                                "xl/worksheets/sheet1.xml": sheet})
        cells = read_xlsx(z).elements[1].cells
        self.assertEqual([(c.text, c.digits) for c in cells], [("8,7", 1), ("2024", 0), ("12.551", 0), ("2025", 0), ("0,1", 1)])


class WikisourceBlockTemplateTests(unittest.TestCase):
    def test_a_block_template_on_its_own_line_ends_the_paragraph(self):
        page = "Prima frase del paragrafo.\n{{Centrato|⁂}}\nSeconda parte, nuovo paragrafo.\n{{Indent|Rientrato.}}\nUltima."
        twin = read_wikisource({1: {"wikitext": page}}, {1: 4})
        self.assertEqual([e.text for e in twin.elements], ["Prima frase del paragrafo.", "⁂", "Seconda parte, nuovo paragrafo.",
                                                          "Rientrato.", "Ultima."])

    def test_inline_templates_stay_in_the_paragraph(self):
        twin = read_wikisource({1: {"wikitext": "Testo con {{Sc|Nome}}\ne continua."}}, {1: 4})
        self.assertEqual([e.text for e in twin.elements], ["Testo con Nome e continua."])


ODT_NS = ('xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0" xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0" '
          'xmlns:table="urn:oasis:names:tc:opendocument:xmlns:table:1.0" xmlns:style="urn:oasis:names:tc:opendocument:xmlns:style:1.0" '
          'xmlns:fo="urn:oasis:names:tc:opendocument:xmlns:xsl-fo-compatible:1.0" xmlns:draw="urn:oasis:names:tc:opendocument:xmlns:drawing:1.0" '
          'xmlns:xlink="http://www.w3.org/1999/xlink"')
MATH = '<math xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>x</mi><mo>=</mo><mn>2</mn></mrow><annotation>x=2</annotation></semantics></math>'


class OdtFrameTests(Base):
    def test_formula_objects_and_caption_frames(self):
        content = f"""<office:document-content {ODT_NS}><office:body><office:text>
<text:p>Set <draw:frame text:anchor-type="as-char"><draw:object xlink:href="./Object 1"/><draw:image xlink:href="./ObjectReplacements/Object 1"/></draw:frame> here.</text:p>
<text:p><draw:frame text:anchor-type="as-char"><draw:object xlink:href="./Object 2"/><draw:image xlink:href="./ObjectReplacements/Object 2"/></draw:frame></text:p>
<text:h text:outline-level="1"><draw:frame text:anchor-type="paragraph"><draw:text-box><text:p>Figure 3: Desired shape<draw:frame><draw:image xlink:href="Pictures/a.png"/></draw:frame></text:p></draw:text-box></draw:frame>Your first shape</text:h>
<table:table><table:table-row><table:table-cell><text:p><draw:frame><draw:image xlink:href="Pictures/b.png"/></draw:frame>Cell text</text:p></table:table-cell></table:table-row></table:table>
</office:text></office:body></office:document-content>"""
        twin = read_odt(self.zip("f.odt", {"content.xml": content, "Object 1/content.xml": MATH, "Object 2/content.xml": MATH}))
        got = [(e.type, e.text) for e in twin.elements if e.type != "table"]
        self.assertEqual(got, [("paragraph", "Set here."), ("formula", "x=2"), ("heading", "Your first shape"),
                               ("figure", "Figure 3: Desired shape"), ("figure", "")])
        self.assertEqual(twin.elements[0].math, ("x=2",))
        self.assertEqual(twin.elements[0].math_kind, "mathml")


class FloatingCaptionTests(unittest.TestCase):
    def test_table_caption_is_not_in_the_order_chain(self):
        b = Builder("jats")
        b.add("paragraph", "A total of many studies were identified through the database search here.")
        b.add("paragraph", "Table 1 Summary of the studies included in this review", role="table-caption")
        b.add("paragraph", "With regard to all site cancer incidence the studies found no increase.")
        got, body = outcomes(b.twin, "A total of many studies were identified through the database search here.\n\n"
                                     "With regard to all site cancer incidence the studies found no increase.\n")
        self.assertEqual(sorted(k for k in got if k.endswith(":order")), ["p1:order"])
        self.assertEqual(got["p1:order"], "pass")
        self.assertEqual(body["not_determinable"]["structure"], 1)


def para(text):
    words = text.split()
    out = []
    for i, w in enumerate(words):
        out += [{"t": "Str", "c": w}] + ([{"t": "Space"}] if i < len(words) - 1 else [])
    return out


class InventoryAlignmentTests(unittest.TestCase):
    def test_display_math_splits_and_images_are_figures_by_caption(self):
        image = {"t": "Image", "c": [["", [], []], [{"t": "Str", "c": "C:\\\\path\\\\x.jpg"}], ["x.jpg", ""]]}
        display = {"t": "Math", "c": [{"t": "DisplayMath"}, "x^2"]}
        ast = {"blocks": [
            {"t": "Para", "c": para("the curl equations take the form") + [display] + para("where c is the speed of light")},
            {"t": "Para", "c": [image]},
            {"t": "Figure", "c": [["", [], []], [None, [{"t": "Plain", "c": para("A plot of the data")}]],
                                  [{"t": "Plain", "c": [image]}]]}]}
        units = selfcheck.pandoc_inventory(ast)
        self.assertEqual(units["paragraph"], ["the curl equations take the form", "where c is the speed of light"])
        self.assertEqual(units["figure"], ["", "a plot of the data"])
        self.assertEqual(selfcheck.pandoc_inventory(ast, split_display=False)["paragraph"],
                         ["the curl equations take the form where c is the speed of light"])

    def test_split_and_assembled_units_are_covered_and_omissions_still_seen(self):
        long_a = " ".join(f"alpha{k}" for k in range(20))
        long_b = " ".join(f"beta{k}" for k in range(12))
        caption = " ".join(f"gamma{k}" for k in range(10))
        gen = {"paragraph": [long_a + " " + long_b, "another unit with its own words only here"], "figure": [caption]}
        # the tool split the first unit in two, and glued a figure caption into the second
        inv = {"paragraph": [long_a, long_b, "another unit with its own words only here " + caption]}
        self.assertEqual(selfcheck.coverage(inv, gen)["paragraph"][:2], (3, 3))
        gen_missing = {"paragraph": ["another unit with its own words only here"], "figure": [caption]}
        self.assertEqual(selfcheck.coverage(inv, gen_missing)["paragraph"][0], 1)
        # a near duplicate never covers a missing unit
        body = "of a longer text that keeps going with many common words"
        inv = {"paragraph": [f"unit number {k} {body} 1" for k in range(12)]}
        gen = {"paragraph": [f"unit number {k} {body}" for k in range(12) if k != 9]}
        self.assertEqual(selfcheck.coverage(inv, gen)["paragraph"][0], 11)

    def test_declared_gaps_allow_the_printed_reference_tokens(self):
        checks_ = [{"type": "paragraph", "tokens": "using eq and eq together with eq we obtain".split(), "gaps": 3}]
        gen, allow = selfcheck.generated_units(checks_), selfcheck.allowances(checks_)
        inv = {"paragraph": ["using eq ft and eq orthog together with eq f_t we obtain"]}
        self.assertEqual(selfcheck.coverage(inv, gen)["paragraph"][0], 0)
        self.assertEqual(selfcheck.coverage(inv, gen, allowance=allow)["paragraph"][0], 1)


class DocxDrawingTests(Base):
    def test_one_figure_per_drawing(self):
        drawing = '<w:r><w:drawing/></w:r>'
        alt = ('<w:r><mc:AlternateContent xmlns:mc="http://schemas.openxmlformats.org/markup-compatibility/2006">'
               '<mc:Choice Requires="wps"><w:drawing/></mc:Choice><mc:Fallback><w:pict/></mc:Fallback></mc:AlternateContent></w:r>')
        doc = f'<w:document {W}><w:body><w:p>{drawing}{drawing}</w:p><w:p>{alt}<w:r><w:t>Text</w:t></w:r></w:p></w:body></w:document>'
        twin = read_docx(self.zip("d.docx", {"word/document.xml": doc}))
        self.assertEqual([e.type for e in twin.elements], ["figure", "figure", "paragraph", "figure"])


class PdfBibliographyTests(unittest.TestCase):
    TEXT = ("Some text cites [1] in a line.\n2\fReferences\n[1] C. Barefoot and L. Clark. Cycles of length 0 modulo 3 in\n"
            "graphs. preprint, 1991.\n[2] B. Bollobás. Cycles modulo k. Bulletin of the London Mathematical Society,\n"
            "9(1):97–98, 1977.\n[3] G. Chen. Graphs with a cycle of length divi-\nsible by three. J. Comb. 1994.\n17\n")

    def test_numbered_entries_of_the_last_list(self):
        self.assertEqual(selfcheck.pdf_bib_entries(self.TEXT), [
            "c barefoot and l clark cycles of length 0 modulo 3 in graphs preprint 1991",
            "b bollobas cycles modulo k bulletin of the london mathematical society 9 1 97 98 1977",
            "g chen graphs with a cycle of length divisible by three j comb 1994"])

    def test_the_journal_pandoc_drops_is_found_and_a_missing_one_is_not(self):
        b = Builder("latex")
        b.add("bib_entry", "B. Bollobás. Cycles modulo . Bulletin of the London Mathematical Society, 9(1):97--98, 1977.")
        body = checks.generate({"id": "D99"}, b.twin)
        from markdown_for_llms.bench.outdoc import from_text
        self.assertEqual(evaluate(body, from_text(self.TEXT))[0]["outcome"], "pass")
        cut = self.TEXT.replace("Bulletin of the London Mathematical Society,\n", "")
        self.assertNotEqual(evaluate(body, from_text(cut))[0]["outcome"], "pass")


class PictureInventoryTests(unittest.TestCase):
    def test_picture_divs_are_not_text(self):
        ast = {"blocks": [{"t": "Div", "c": [["", ["center"], []], [
            {"t": "Div", "c": [["", ["picture"], []], [{"t": "Para", "c": para("(145,125) (10,15)(50,50)")}]]},
            {"t": "Para", "c": para("Figure 1. Construction of the lemma.")}]]},
            {"t": "Para", "c": para("A body paragraph with words.")}]}
        self.assertEqual(selfcheck.pandoc_inventory(ast)["paragraph"], ["a body paragraph with words"])


if __name__ == "__main__":
    unittest.main()


class TightSpanTests(unittest.TestCase):
    """`match._tight` (M4b, R4-C item 16): the span of a unit found in an output is the unit itself,
    not the whole window of the matcher, which reaches into the next block. On D02 the window of a
    paragraph reached an HTML block after it: a false `html_residue`, and order checks moved."""

    def twin(self):
        b = Builder("docx")
        b.add("paragraph", "One two three four five six seven eight nine ten.")
        b.add("paragraph", "Eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen.")
        return b.twin

    def test_a_clean_paragraph_before_an_html_block_has_no_residue(self):
        got, _ = outcomes(self.twin(), "One two three four five six seven eight nine ten.\n\n"
                                       "<div>Eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen.</div>\n")
        self.assertEqual(got["p1:html"], "pass")
        self.assertEqual(got["p2:html"], "html")            # the residue of the converter is still penalized

    def test_html_inside_the_paragraph_is_still_a_residue(self):
        got, _ = outcomes(self.twin(), "<span>One two three four five six seven eight nine ten.</span>\n\n"
                                       "Eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen.\n")
        self.assertEqual(got["p1:html"], "html")
