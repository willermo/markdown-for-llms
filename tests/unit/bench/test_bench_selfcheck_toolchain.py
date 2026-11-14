"""Self-check on the reference toolchain: the format the self-check emits and the declared limits."""
import re
import tempfile
import unittest
from pathlib import Path

from bench_test_support import require_toolchain
from markdown_for_llms.bench import selfcheck
from markdown_for_llms.bench.outdoc import from_markdown


class RenderingFormatTests(unittest.TestCase):
    def setUp(self):
        require_toolchain()
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name) / "tool-work"

    def test_math_is_written_between_dollars_and_read_back(self):
        """Pandoc writes GitHub math as `` $`x`$ `` and ```` ```math ```` blocks, which no converter
        output uses and `outdoc` reads as code: the self-check writes TeX math between dollars."""
        source = b"Energy $a^2+b$ of the system.\n\n\\[ x = \\frac{1}{2} \\]\n\nAfter.\n"
        ast = selfcheck._pandoc_json(["-f", "latex"], self.root, stdin=source)
        text = selfcheck._pandoc_gfm(ast, self.root)
        self.assertIn("$a^2+b$", text)
        self.assertIn("$$", text)
        self.assertNotIn("$`", text)
        self.assertNotIn("```math", text)
        out = from_markdown(text)
        self.assertEqual(sorted(m for b in out.blocks for m in b.math), sorted(["a^2+b", r"x = \frac{1}{2}"]))


class DeclaredLimitTests(unittest.TestCase):
    def test_limits_name_no_tool_version(self):
        """The versions are those of toolchain.json; a limit describes the behaviour of the tool."""
        for kind, text in selfcheck.DECLARED_LIMITS.items():
            with self.subTest(kind=kind):
                self.assertIsNone(re.search(r"\b\d+\.\d+(\.\d+)?\b", text))



class FigureRenderingTests(unittest.TestCase):
    """Pandoc of the reference toolchain writes a Figure block as raw HTML (<figure>, <img>,
    <figcaption>), which outdoc reads as a paragraph without image: the self-check writes it as GitHub
    Markdown can, the image and then the caption (D02-D05: figure missing)."""

    def setUp(self):
        require_toolchain()
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name) / "tool-work"

    def test_a_figure_is_an_image_and_its_caption(self):
        source = (b"Before.\n\n\\begin{figure}\\includegraphics{a.png}\\caption{The caption of figure one.}"
                  b"\\end{figure}\n\nAfter.\n")
        ast = selfcheck._pandoc_json(["-f", "latex"], self.root, stdin=source)
        text = selfcheck._pandoc_gfm(ast, self.root)
        self.assertNotIn("<figure", text)
        out = from_markdown(text)
        self.assertEqual(len([i for i in out.images if i[1]]), 1)
        self.assertIn("The caption of figure one.", text)
        self.assertEqual(selfcheck.pandoc_inventory(ast)["figure"], ["the caption of figure one"])   # inventory unchanged


class InventoryAlignmentToolchainTests(unittest.TestCase):
    def test_table_foot_paragraphs_are_notes(self):
        """JATS table-wrap-foot paragraphs print as notes below the table; Pandoc writes them as
        paragraphs of the table-wrap Div (D05)."""
        para = lambda text: {"t": "Para", "c": [{"t": "Str", "c": w} if i % 2 == 0 else {"t": "Space"}
                                               for i, w in enumerate(" ".join(text.split()).replace(" ", "  ").split(" "))]}
        table = {"t": "Table", "c": [["", [], []], [None, []], [], [["", [], []], []], [], [["", [], []], []]]}
        ast = {"blocks": [para("A body paragraph of the article."),
                          {"t": "Div", "c": [["t1", ["table-wrap"], []],
                                             [{"t": "Div", "c": [["", ["caption"], []], [para("Table 1 Patients")]]},
                                              table, para("Values are medians."), para("Abbreviations of the table.")]]}]}
        units = selfcheck.pandoc_inventory(ast)
        self.assertEqual(units["footnote"], ["values are medians", "abbreviations of the table"])
        self.assertEqual(units["paragraph"], ["a body paragraph of the article"])

    def test_the_continuation_of_a_reference_note_is_a_note(self):
        """ProofreadPage writes the continuation of a note begun on the previous page as a block at the
        top of the reference list; Pandoc makes it a list item (D14)."""
        html = ('<div class="pagetext"><p>Body text of the page here.</p><ol class="references">'
                "<p>end of the note begun on the page before</p><li>a note of this page</li></ol></div>")
        item = lambda text: [{"t": "Plain", "c": [{"t": "Str", "c": text}]}]
        ast = {"blocks": [{"t": "Para", "c": [{"t": "Str", "c": "Body text of the page here."}]},
                          {"t": "OrderedList", "c": [[1, {"t": "Decimal"}, {"t": "Period"}],
                                                     [item("end of the note begun on the page before"),
                                                      item("a note of this page")]]}]}
        units = selfcheck.mediawiki_inventory(ast, html)
        self.assertEqual(units["list_item"], [])
        self.assertEqual(units["footnote"], ["end of the note begun on the page before", "a note of this page"])

    def test_a_caption_without_its_printed_label_is_covered(self):
        """Pandoc drops the <label> of a JATS fig ("Figure 1") that the twin keeps (D04)."""
        inv = ["survival curves of the two groups of patients"]
        gen = ["figure 1 survival curves of the two groups of patients"]
        self.assertEqual(selfcheck._assign_figures(inv, gen)[0], 1)
        self.assertEqual(selfcheck._assign_figures(inv, ["figure 2 a different chart of the cohort over time"])[0], 0)
        cov = selfcheck.coverage({"figure": inv}, {"figure": gen}, drop_every=10)
        self.assertLess(cov["figure"][2], selfcheck.COVERAGE_MIN)          # the omission is still detected


class ReferenceSourceTests(unittest.TestCase):
    def test_revtex_bbl_is_not_readable_by_pandoc(self):
        import io, tarfile
        data = io.BytesIO()
        with tarfile.open(fileobj=data, mode="w") as tar:
            for name, body in (("main.tex", b"x"), ("main.bbl", b"\\bibitem{a}\\bibinfo{author}{A. Author}\\BibitemShut{NoStop}")):
                info = tarfile.TarInfo(name)
                info.size = len(body)
                tar.addfile(info, io.BytesIO(body))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "src.tar"
            path.write_bytes(data.getvalue())
            files = type("F", (), {"twin": {"file": path}})()
            self.assertTrue(selfcheck._bbl_unreadable({"twin": {"main": "main.tex", "bbl": "main.bbl"}}, files))

    def test_latexml_bibliography_entries(self):
        """D03 is the arXiv HTML (LaTeXML): its entries are judged and inventoried on its own list."""
        html = (b'<section class="ltx_bibliography"><ul class="ltx_biblist">'
                b'<li id="bib.bib1" class="ltx_bibitem"><span class="ltx_tag ltx_tag_bibitem">[1]</span>'
                b'<span class="ltx_bibblock">A. Author and B. Author, </span><span class="ltx_bibblock">'
                b'<em>Journal</em> <b>12</b>, 3 (2020).</span></li>'
                b'<li class="ltx_bibitem"><span class="ltx_tag ltx_tag_bibitem">[2]</span>C. Writer, Book (2021).</li>'
                b'</ul></section>')
        self.assertEqual(selfcheck.html_bib_entries(html), ["A. Author and B. Author, Journal 12, 3 (2020).",
                                                            "C. Writer, Book (2021)."])


class PdfTextModeTests(unittest.TestCase):
    """poppler 26.11 drops word spaces in raw mode in publisher PDFs (D04, D19); its reading-order
    mode keeps them. LibreOffice PDFs (PPTX) and the arXiv PDF (LaTeX) keep -raw."""

    def test_mode_per_path(self):
        from unittest import mock
        from markdown_for_llms.bench import tools
        calls = []
        with mock.patch.object(tools, "run", lambda name, args, root, timeout=0: calls.append(args) or b"text"):
            for kind in ("jats", "akoma-ntoso", "pptx", "latex"):
                selfcheck.pdf_text(Path("in.pdf"), kind, Path("root"))
        self.assertEqual([a[0] == "-raw" for a in calls], [False, False, True, True])


class AtomNormalizationTests(unittest.TestCase):
    """One atom per printed glyph (D16: Pandoc writes the MathML of an ODT formula as TeX)."""

    def test_same_glyph_and_spacing(self):
        from markdown_for_llms.bench.text import math_atoms, unicode_atoms
        self.assertEqual(math_atoms(r"a \cdot b^{\circ}\mspace{9mu} c"), unicode_atoms("a⋅b° c"))
        self.assertEqual(math_atoms(r"x^\circ"), ["x", "°"])
        self.assertEqual(math_atoms(r"f \circ g"), ["f", "∘", "g"])         # a composition stays a ring
        self.assertNotEqual(math_atoms(r"a \times b"), unicode_atoms("a⋅b"))  # a wrong operator still differs
        self.assertNotEqual(math_atoms(r"a \cdot b^{2}"), unicode_atoms("a⋅b°"))

    def test_converter_error_still_penalized(self):
        from markdown_for_llms.bench import checks
        from markdown_for_llms.bench.match import evaluate
        from markdown_for_llms.bench.model import Builder
        b = Builder("odt")
        b.add("paragraph", "The angle of the beam is twelve degrees here.", math=["θ⋅12°"], math_kind="mathml")
        body = checks.generate({"id": "D99"}, b.twin)
        body.update(twin_kind="odt", rules=[])
        good = {r["id"]: r["outcome"] for r in evaluate(body, from_markdown(
            "The angle of the beam is $\\theta \\cdot 12^{\\circ}\\mspace{9mu}$ twelve degrees here.\n"))}
        bad = {r["id"]: r["outcome"] for r in evaluate(body, from_markdown(
            "The angle of the beam is $\\theta \\times 12$ twelve degrees here.\n"))}
        self.assertEqual((good["p1:math"], bad["p1:math"]), ("pass", "math"))


class GeneratorVersionTests(unittest.TestCase):
    """The generated checks of synthetic twins are fixed per GENERATOR_VERSION: changing the generator
    without raising the version makes this test fail (E8)."""
    EXPECTED = {"4": "3621fa56084c7db1cab42f7da7fbe90f2acffdd8fb80cae5b709842a5aae9508"}

    @staticmethod
    def twins():
        from markdown_for_llms.bench.model import Builder, Cell
        b = Builder("odt")
        b.add("heading", "Results of the study", level=1)
        b.add("paragraph", "The measured angle is twelve degrees in every sample.", math=["θ⋅12°"], math_kind="mathml")
        b.add("formula", r"E = m c^{2} \mspace{9mu}", math_kind="tex")
        b.add("list_item", "first item of the short list here", level=1)
        b.add("figure", "Figure 1 the beam")
        b.add("footnote", "A note about the samples of the study.")
        b.add("bib_entry", "A. Author, Journal 12, 3 (2020).")
        b.add("table", cells=[Cell("Group", 0, 0, 1, 1, True), Cell("Value", 0, 1, 1, 1, True),
                              Cell("A", 1, 0, 1, 1, False), Cell("1.5", 1, 1, 1, 1, False)], level=1)
        return [b.twin]

    def test_generated_checks_are_fixed_for_the_version(self):
        import hashlib
        from markdown_for_llms.bench import GENERATOR_VERSION, checks
        from markdown_for_llms.bench.cache import canonical
        digest = hashlib.sha256(b"".join(canonical(checks.generate({"id": "D99"}, t)) for t in self.twins())).hexdigest()
        self.assertIn(GENERATOR_VERSION, self.EXPECTED)
        self.assertEqual(digest, self.EXPECTED[GENERATOR_VERSION])


if __name__ == "__main__":
    unittest.main()
