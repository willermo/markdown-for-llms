"""Arbiters of the self-check (plan r2 §5, App. C; delta r3 Δ4): which rendering judges each document
and which routes it cannot arbitrate, cell formulas in the Pandoc inventory of an ODT twin, the
`\\vert` of table cells, coverage never silent and formulas paired by similarity."""
import unittest

from bench_test_support import require_toolchain

from markdown_for_llms.bench import manifest, selfcheck
from markdown_for_llms.bench.codes import BenchError, Code
from markdown_for_llms.bench.match import FORMULA_RATIO


class ArbiterTableTests(unittest.TestCase):
    """Appendix C of plan r2, document by document."""
    EXPECTED = {
        "D01": ("pandoc-latex", {}), "D03": ("latexml-html", {"pandoc": []}),
        "D04": ("pandoc-jats", {}), "D06": ("ghostscript", {}),
        "D07": ("calc-html", {"pandoc": [], "libreoffice": ["libreoffice"]}),
        "D08": ("libreoffice-poppler", {"pandoc": [], "libreoffice": ["libreoffice"]}),
        "D10": ("writer-html", {"pandoc": [], "marker": []}),
        "D13": ("writer-epub", {"pandoc": []}), "D14": ("pandoc-mediawiki", {"marker": []}),
        "D16": ("pandoc-odt", {"marker": [], "datalab": []}), "D19": ("poppler", {}),
        "D20": ("pandoc-mediawiki", {"djvutxt": []}),
        "D22": ("writer-html", {"pandoc": ["libreoffice"]}), "D23": ("writer-html", {"pandoc": ["libreoffice"]}),
        "D24": ("calc-html", {"pandoc": ["libreoffice"]}), "D25": ("writer-html", {"pandoc": []}),
        "D26": ("writer-html", {"pandoc": []}), "D29": ("writer-html", {"text": []}),
        "D31": ("pandoc-html", {"text": []}), "D34": ("pandoc-html", {"text": []}), "D35": ("oracle", {"text": []}),
        "D36": ("oracle", {"text": []}),
    }

    def test_arbiter_and_shared_tools(self):
        m = manifest.load()
        for doc_id, (name, routes) in self.EXPECTED.items():
            doc = m.document(doc_id)
            with self.subTest(doc=doc_id):
                self.assertEqual(selfcheck.arbiter(doc), name)
                for route, shared in routes.items():
                    self.assertEqual(selfcheck.shares(doc, route), shared)

    def test_oracle_and_gutenberg(self):
        txt = {"id": "D99", "format": "txt", "twin": None}
        self.assertEqual((selfcheck.arbiter(txt), selfcheck.shares(txt, "text")), ("oracle", []))
        pg = {"id": "D98", "format": "txt", "twin": {"kind": "html", "source": "x"}}
        self.assertEqual((selfcheck.arbiter(pg), selfcheck.shares(pg, "text")), ("pandoc-html", []))


class PandocCellFormulaTests(unittest.TestCase):
    TABLE = {"blocks": [{"t": "Table", "c": [["", [], []], [None, []], [], [["", [], []], [[["", [], []], [
        [["", [], []], {"t": "AlignDefault"}, 1, 1, [{"t": "Plain", "c": [
            {"t": "Str", "c": "x"}, {"t": "Math", "c": [{"t": "InlineMath"}, "|a|"]}]}]]]]]], ["", [], []]]}]}

    def test_only_an_odt_twin_counts_cell_formulas(self):
        self.assertEqual(selfcheck.pandoc_inventory(self.TABLE, cell_math=True)["formula"], ["| a |"])
        self.assertNotIn("formula", selfcheck.pandoc_inventory(self.TABLE))

    def test_vert_rewrite_in_cells_only(self):
        import copy
        ast = copy.deepcopy(self.TABLE)
        ast["blocks"].append({"t": "Para", "c": [{"t": "Math", "c": [{"t": "InlineMath"}, "|b|"]}]})
        selfcheck._vert_in_cells(ast)

        def maths(x):
            if isinstance(x, dict):
                return ([x["c"][1]] if x.get("t") == "Math" else []) + [m for v in x.values() for m in maths(v)]
            return [m for v in x for m in maths(v)] if isinstance(x, list) else []
        self.assertEqual(maths(ast["blocks"][0]), ["\\vert a\\vert "])
        self.assertEqual(ast["blocks"][1]["c"][0]["c"][1], "|b|")


class VertRenderingTests(unittest.TestCase):
    """Through the Pandoc of the reference toolchain: a formula with bars in a pipe table cell is read
    back whole by `from_markdown` only after the rewrite."""

    def setUp(self):
        require_toolchain()
        import tempfile
        from pathlib import Path
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)

    def test_absolute_value_in_a_cell_is_read(self):
        from markdown_for_llms.bench.outdoc import from_markdown
        from markdown_for_llms.bench.text import math_atoms
        ast = selfcheck._pandoc_json(["-f", "latex"], self.root,
                                     stdin=b"\\begin{tabular}{ll} h & k \\\\ v & $|a| + \\|b\\|$ \\end{tabular}")
        out = from_markdown(selfcheck._pandoc_gfm(ast, self.root))
        self.assertEqual([math_atoms(t) for _, t in out.formulas], [math_atoms("|a| + \\|b\\|")])


class CoverageTests(unittest.TestCase):
    def test_a_type_without_inventory_is_reported(self):
        checks = [{"type": "paragraph"}, {"type": "footnote"}, {"type": "footnote"}, {"type": "order"}]
        self.assertEqual(selfcheck.no_inventory({"paragraph": ["a"]}, checks, "docx"), {"footnote": 2})
        self.assertEqual(selfcheck.no_inventory({"paragraph": ["a"], "footnote": ["b"]}, checks, "docx"), {})
        # declared for a JATS twin, not for a Writer twin
        self.assertIn("footnote", selfcheck.NO_INVENTORY["jats"])
        self.assertNotIn("docx", selfcheck.NO_INVENTORY)

    def test_writer_formulas_cover_inline_and_display_checks(self):
        checks = [{"type": "formula"}, {"type": "inline_math"}]
        self.assertEqual(selfcheck.no_inventory({"formula": ["x"]}, checks, "docx"), {})
        self.assertEqual(selfcheck.no_inventory({"formula": ["x"]}, checks, "latex"), {"inline_math": 1})

    def test_formulas_pair_by_similarity_and_the_omission_is_detected(self):
        generated = [f"x _ {k} = a ^ 2 + b ^ 2 + c _ {k}" for k in range(20)]
        inventory = [g + " , ;" for g in generated]           # close (similarity 0.93), not 90 % of the tokens
        self.assertTrue(all(selfcheck.formula_similarity(a.split(), b.split()) >= FORMULA_RATIO
                            for a, b in zip(inventory, generated)))
        cov = selfcheck.coverage({"formula": inventory}, {"formula": generated}, twin_kind="odt")
        self.assertEqual(cov["formula"][:2], (20, 20))
        omitted = selfcheck.coverage({"formula": inventory}, {"formula": generated}, drop_every=10, twin_kind="odt")
        self.assertLess(omitted["formula"][2], selfcheck.COVERAGE_MIN)
        # the token rule of the other twins does not pair them
        self.assertLess(selfcheck.coverage({"formula": inventory}, {"formula": generated}, twin_kind="latex")["formula"][2], 1)

    def test_gutenberg_boilerplate_must_be_found(self):
        page = (b'<body><header class="pg-boilerplate pgheader" id="pg-header"><h2>The eBook</h2><div><p>x</p></div>'
                b'</header><h1>Text</h1><p>Body</p><footer class="pg-boilerplate pgheader" id="pg-footer"><ul><li>Licence'
                b'</li></ul></footer></body>')
        self.assertEqual(selfcheck.drop_boilerplate(page), b"<body><h1>Text</h1><p>Body</p></body>")
        with self.assertRaises(BenchError) as ctx:
            selfcheck.drop_boilerplate(b"<body><p>Body</p></body>")
        self.assertEqual(ctx.exception.code, Code.TWIN_BOILERPLATE_UNKNOWN)


if __name__ == "__main__":
    unittest.main()
