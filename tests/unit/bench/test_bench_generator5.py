"""Generator 5 (delta r3 Δ4.1, prescription C-VERSIONE): formulas inside table cells become `formula`
checks after every other element, the ids of generator 4 do not move, and the kept exclusions carry
the new version and pass `exclusions.validate` again."""
from pathlib import Path
import tempfile
import unittest
import zipfile

from markdown_for_llms.bench import GENERATOR_VERSION, checks, exclusions
from markdown_for_llms.bench.readers.docx import read_docx
from markdown_for_llms.bench.readers.odt import read_odt

ODT_NS = ('xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0" '
          'xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0" '
          'xmlns:table="urn:oasis:names:tc:opendocument:xmlns:table:1.0" '
          'xmlns:draw="urn:oasis:names:tc:opendocument:xmlns:drawing:1.0" xmlns:xlink="http://www.w3.org/1999/xlink"')
MATH = '<math xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>{0}</mi><mo>=</mo><mn>2</mn></mrow></semantics></math>'
W = ('xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" '
     'xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math"')


def frame(name):
    return f'<draw:frame><draw:object xlink:href="./{name}"/></draw:frame>'


class Base(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)

    def zip(self, name, members):
        path = self.root / name
        with zipfile.ZipFile(path, "w") as z:
            for member, data in members.items():
                z.writestr(member, data)
        return path


class OdtCellFormulaTests(Base):
    def odt(self, with_cells):
        cell = f"<text:p>v {frame('Obj2')}</text:p>" if with_cells else "<text:p>v</text:p>"
        content = f"""<office:document-content {ODT_NS}><office:body><office:text>
<text:p>Before the table {frame('Obj1')} here.</text:p>
<table:table><table:table-row><table:table-cell>{cell}</table:table-cell>
<table:table-cell><text:p>{frame('Obj3') if with_cells else ''}</text:p></table:table-cell></table:table-row></table:table>
<text:p>{frame('Obj4')}</text:p>
</office:text></office:body></office:document-content>"""
        members = {"content.xml": content}
        for k, name in enumerate(("Obj1", "Obj2", "Obj3", "Obj4")):
            members[f"{name}/content.xml"] = MATH.format("xyzw"[k])
        return read_odt(self.zip("t.odt", members))

    def test_cell_formulas_are_checks_after_every_other_element(self):
        twin = self.odt(True)
        cells = [e for e in twin.elements if e.role == "table-cell"]
        self.assertEqual([(e.type, e.text) for e in cells], [("formula", "y=2"), ("formula", "z=2")])
        self.assertEqual(twin.elements[-2:], cells)
        generated = checks.generate({"id": "D99"}, twin)["checks"]
        cell_checks = [c for c in generated if c.get("role") == "table-cell"]
        self.assertEqual([c["atoms"] for c in cell_checks], [["y", "=", "2"], ["z", "=", "2"]])

    def test_ids_of_generator_4_do_not_move(self):
        before = {c["id"]: c for c in checks.generate({"id": "D99"}, self.odt(False))["checks"]}
        after = {c["id"]: c for c in checks.generate({"id": "D99"}, self.odt(True))["checks"]}
        cell_ids = {k for k, c in after.items() if c.get("role") == "table-cell"}
        self.assertEqual(len(cell_ids), 2)
        # every check of the table-less formulas keeps its id and content; the cells add the new ones
        for k, c in before.items():
            if c["type"] not in ("cell", "table", "table_shape"):
                self.assertEqual(after[k], c)
        self.assertTrue(cell_ids.isdisjoint(before))


class DocxCellFormulaTests(Base):
    def docx(self, cell):
        doc = f"""<w:document {W}><w:body>
<w:p><w:r><w:t>Body with </w:t></w:r><m:oMath><m:r><m:t>a</m:t></m:r></m:oMath></w:p>
<w:tbl><w:tr><w:tc>{cell}</w:tc></w:tr></w:tbl>
<w:p><m:oMathPara><m:oMath><m:r><m:t>d</m:t></m:r></m:oMath></m:oMathPara></w:p>
</w:body></w:document>"""
        return read_docx(self.zip("t.docx", {"word/document.xml": doc}))

    def test_inline_cell_formula_is_a_check_at_the_end(self):
        twin = self.docx("<w:p><w:r><w:t>cell </w:t></w:r><m:oMath><m:r><m:t>b</m:t></m:r></m:oMath></w:p>")
        self.assertEqual([(e.type, e.text, e.role) for e in twin.elements if e.type == "formula"],
                         [("formula", "d", ""), ("formula", "b", "table-cell")])
        self.assertEqual(twin.elements[-1].role, "table-cell")

    def test_display_formula_of_a_cell_is_not_duplicated(self):
        twin = self.docx("<w:p><m:oMathPara><m:oMath><m:r><m:t>c</m:t></m:r></m:oMath></m:oMathPara></w:p>")
        self.assertEqual([(e.text, e.role) for e in twin.elements if e.type == "formula"], [("c", ""), ("d", "")])


class CheckSetKeyTests(unittest.TestCase):
    def doc(self, kind):
        return {"id": "D99", "twin": {"kind": kind, "source": "s"}}

    def test_a_derived_input_enters_the_key_by_its_source(self):
        a = {"source": "1" * 64, "twin": "2" * 64, "derived": {"sha256": "3" * 64}}
        b = dict(a, derived={"sha256": "4" * 64})       # LibreOffice wrote another .doc (D23)
        self.assertEqual(checks.cache_key(self.doc("docx"), a), checks.cache_key(self.doc("docx"), b))
        # the projection of D06 reads the PDF to convert: its derivation enters the key
        self.assertNotEqual(checks.cache_key(self.doc("xlsx-projection"), a),
                            checks.cache_key(self.doc("xlsx-projection"), b))


class ExclusionVersionTests(unittest.TestCase):
    """C-VERSIONE: the kept exclusions carry GENERATOR_VERSION; through `exclusions.validate` an entry
    of the new version whose precondition holds is `ok`, the same entry of version 4 `revalidate`."""

    def test_packaged_exclusions_carry_the_generator_version(self):
        data = exclusions.load()
        self.assertEqual(GENERATOR_VERSION, "5")
        self.assertEqual({e["generator_version"] for e in data["exclusions"]}, {"5"})
        self.assertEqual({e["code"] for e in data["exclusions"]}, {"self_check_tool_limit"})
        by_doc = {}
        for e in data["exclusions"]:
            by_doc[e["document"]] = by_doc.get(e["document"], 0) + 1
        self.assertEqual(by_doc, {"D16": 40, "D19": 3, "D26": 42})

    def test_validate_with_the_new_version(self):
        checkset = {"twin_kind": "odt", "checks": [{"id": "fig1", "type": "figure"}, {"id": "p1:order", "type": "order"}]}
        results = [{"id": "fig1", "outcome": "caption"}, {"id": "p1:order", "outcome": "order"}]
        entry = {"document": "D99", "check": "fig1", "code": "self_check_tool_limit", "reason": "r", "origin": "self-check"}
        not_a_limit = dict(entry, check="p1:order")
        statuses = exclusions.validate([dict(entry, generator_version=GENERATOR_VERSION),
                                        dict(not_a_limit, generator_version=GENERATOR_VERSION)],
                                       checkset, results, lambda c: True, lambda c: 1.0)
        self.assertEqual(statuses, {"fig1": "ok", "p1:order": "stale"})
        old = exclusions.validate([dict(entry, generator_version="4")], checkset, results, lambda c: True, lambda c: 1.0)
        self.assertEqual(old, {"fig1": "revalidate"})


if __name__ == "__main__":
    unittest.main()


class D26ExclusionTests(unittest.TestCase):
    """The 42 exclusions of D26 (M6): figure checks whose caption Writer HTML prints only in the figure
    index, under the declared limit of the Writer rendering, within the 5 % cap, and valid."""

    def test_entries_limit_and_validation(self):
        from markdown_for_llms.bench import selfcheck
        entries = [e for e in exclusions.load()["exclusions"] if e["document"] == "D26"]
        self.assertEqual(len(entries), 42)
        self.assertTrue(all(e["check"].startswith("fig") and e["check"][3:].isdigit() for e in entries))
        self.assertEqual({(e["code"], e["origin"], e["generator_version"]) for e in entries},
                         {("self_check_tool_limit", "self-check", GENERATOR_VERSION)})
        self.assertEqual({e["reason"] for e in entries},
                         {"Writer HTML does not print the caption of this frame; it prints it only in the figure index"})
        self.assertIn("only in the figure index", selfcheck.DECLARED_LIMITS["writer-html"])
        self.assertIn("figure", exclusions.LIMIT_TYPES["odt"])
        self.assertIn("figure", selfcheck.OMISSION_NOT_DETERMINABLE["D26"])
        self.assertLessEqual(len(entries) / 2076, selfcheck.EXCLUSION_CAP)        # content checks of D26
        checkset = {"twin_kind": "odt", "checks": [{"id": e["check"], "type": "figure"} for e in entries]}
        results = [{"id": e["check"], "outcome": "caption"} for e in entries]
        statuses = exclusions.validate(entries, checkset, results, lambda c: True, lambda c: 0.0)
        self.assertEqual(set(statuses.values()), {"ok"})
        # a caption that the rendering prints again makes the exclusion stale, never silently kept
        results[0]["outcome"] = "pass"
        self.assertEqual(exclusions.validate(entries[:1], checkset, results, lambda c: True, lambda c: 0.0),
                         {entries[0]["check"]: "stale"})


class WikisourceLongBookTests(unittest.TestCase):
    """The phenomena of D17 that D14 does not have (M6): `<nowiki/>`, the index entries `{{Vi}}`, a
    table continued from the previous page, and in the MediaWiki inventory the notes of several
    paragraphs or with a table, and images inside links."""

    def test_reader(self):
        from markdown_for_llms.bench.readers.wikisource import read_wikisource
        index = ("<noinclude><pagequality level=\"4\" /></noinclude>{{Vi|sezione=I.|titolo=La Terra non è piana|pagina={{Pg|4}}}}\n"
                 "{{Vi|titolo=Introduzione|pagina=» 1}}\n<nowiki/>Testo dopo.")
        follow = ("<noinclude><pagequality level=\"4\" />{{RigaIntestazione|206|Titolo}}\n{| align=center</noinclude>|-\n"
                  "| Fenice || Ottante\n|-\n| Tucano || Mensa\n|}")
        twin = read_wikisource({1: {"wikitext": index}, 2: {"wikitext": follow}}, {1: 4, 2: 4})
        self.assertEqual([(e.type, e.text) for e in twin.elements if e.page == 1],
                         [("paragraph", "I."), ("paragraph", "La Terra non è piana"), ("paragraph", "4"),
                          ("paragraph", "Introduzione"), ("paragraph", "» 1"), ("paragraph", "Testo dopo.")])
        table = next(e for e in twin.elements if e.type == "table")
        self.assertEqual([c.text for c in table.cells], ["Fenice", "Ottante", "Tucano", "Mensa"])
        self.assertEqual([e.type for e in twin.elements if e.page == 2], ["noise", "table"])

    def test_mediawiki_inventory_of_notes_and_linked_images(self):
        from markdown_for_llms.bench import selfcheck
        html = ('<p>Testo<sup><a href="#n1">1</a></sup> <a href="/wiki/File:x.png"><img src="x.png"/></a></p>'
                '<ol class="references"><li>Prima parte della nota. <dl><dd>Seconda parte della nota.</dd></dl>'
                '<table>\n<tr>\n<td>Mercurio</td>\n<td>4816</td>\n</tr>\n</table></li></ol>')
        para = [{"t": "Str", "c": "Testo"}, {"t": "Space"},
                {"t": "Link", "c": [["", [], []], [{"t": "Image", "c": [["", [], []], [], ["x.png", ""]]}], ["/wiki/File:x.png", ""]]}]
        cell = lambda t: [["", [], []], {"t": "AlignDefault"}, 1, 1, [{"t": "Plain", "c": [{"t": "Str", "c": t}]}]]
        table = {"t": "Table", "c": [["", [], []], [None, []], [], [["", [], []], []],
                                     [[["", [], []], 0, [], [[["", [], []], [cell("Mercurio"), cell("4816")]]]]], [["", [], []], []]]}
        item = [{"t": "Plain", "c": [{"t": "Str", "c": "Prima"}]},
                {"t": "DefinitionList", "c": [[[], [[{"t": "Plain", "c": [{"t": "Str", "c": "Seconda parte della nota."}]}]]]]},
                table]
        ast = {"blocks": [{"t": "Para", "c": para}, {"t": "OrderedList", "c": [[1, {"t": "Decimal"}, {"t": "Period"}], [item]]}]}
        units = selfcheck.mediawiki_inventory(ast, html)
        self.assertEqual(units["figure"], [""])
        self.assertEqual(units["footnote"], ["prima parte della nota seconda parte della nota mercurio 4816"])
        self.assertEqual((units["cell"], units["table"], units["list_item"]), ([], [], []))
