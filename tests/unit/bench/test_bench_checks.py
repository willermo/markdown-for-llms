"""Generator, cache key, output reader, matcher, normalizations (C3), coverage and omission (C4),
exclusions and scores, on synthetic twins and outputs."""
import json
from pathlib import Path
import tempfile
import unittest

from markdown_for_llms.bench import checks, exclusions, score, selfcheck
from markdown_for_llms.bench.cache import Cache
from markdown_for_llms.bench.fetch import DocumentFiles
from markdown_for_llms.bench.match import evaluate
from markdown_for_llms.bench.model import Builder, Cell
from markdown_for_llms.bench.outdoc import from_markdown, from_text
from markdown_for_llms.bench.text import accepts, align_key, score_tokens


def twin():
    b = Builder("docx")
    b.add("heading", "Introduction", level=1)
    b.add("paragraph", "The quick brown fox jumps over the lazy dog near the river bank.", emphasis=(("i", "lazy dog"),))
    b.add("list_item", "first item here", level=1)
    b.add("heading", "Results", level=2)
    b.add("paragraph", "A second paragraph explains the numbers of the table below in detail.", math=("x^2",))
    b.add("table", cells=(Cell("Region", 0, 0, header=True), Cell("Rate", 0, 1, header=True),
                          Cell("Nord", 1, 0), Cell("8,0", 1, 1, value="8.04", digits=1)), level=1)
    b.add("formula", "\\frac{a}{b} + \\sin x", display=True)
    b.add("formula", "\\mymacro{a}", display=True)
    b.add("footnote", "A note at the bottom of the page.")
    b.add("noise", "— 199 —")
    return b.twin


GOOD = """# Introduction

The quick brown fox jumps over the *lazy dog* near the river bank.[^1]

- first item here

## Results

A second paragraph explains the numbers of the table below in detail $x^2$.

| Region | Rate |
|---|---|
| Nord | 8.0 |

$$\\frac{a}{b}+\\sin x$$

[^1]: A note at the bottom of the page.
"""


class GeneratorTests(unittest.TestCase):
    def test_types_classes_and_not_determinable(self):
        gen = checks.generate({"id": "D99"}, twin())
        types = [c["type"] for c in gen["checks"]]
        for t in ("heading", "paragraph", "list_item", "emphasis", "inline_math", "cell", "table", "table_shape",
                  "table_header", "formula", "spaced_math", "footnote", "note_linked", "noise", "heading_level", "order"):
            self.assertIn(t, types)
        self.assertEqual(gen["not_determinable"]["content"], 1)        # \mymacro
        self.assertTrue(all(c["class"] == checks.CLASS_OF[c["type"]] for c in gen["checks"]))
        self.assertIn("t1/r2/c2", [c["id"] for c in gen["checks"]])

    def test_order_pairs_are_capped(self):
        b = Builder("docx")
        for k in range(500):
            b.add("paragraph", f"paragraph number {k} with words")
        gen = checks.generate({"id": "D99"}, b.twin)
        self.assertEqual(sum(c["type"] == "order" for c in gen["checks"]), checks.MAX_ORDER_PAIRS)


class CacheKeyTests(unittest.TestCase):
    """A1: the key includes the document; D06 and D07 share a workbook but never a cache entry."""

    def doc(self, doc_id, kind, select=None):
        t = {"kind": kind, "source": "x"}
        if select is not None:
            t["select"] = select
        return {"id": doc_id, "twin": t}

    def test_order_independent_and_projection_sensitive(self):
        files = DocumentFiles("D", Path("in"), {}, [], {"source": "a" * 64, "twin": "b" * 64})
        d06, d07 = self.doc("D06", "xlsx-projection", [{"sheet": "S"}]), self.doc("D07", "xlsx")
        outputs = []
        for order in ((d06, d07), (d07, d06)):
            with tempfile.TemporaryDirectory() as tmp:
                cache = Cache(tmp)
                for d in order:
                    checks.load_or_generate(d, files, cache, twin)
                outputs.append({p.name: p.read_bytes() for p in cache.checks_dir.iterdir()})
        self.assertEqual(outputs[0], outputs[1])
        self.assertEqual(len(outputs[0]), 2)
        self.assertNotEqual(checks.cache_key(d06, files.hashes),
                            checks.cache_key(self.doc("D06", "xlsx-projection", [{"sheet": "T"}]), files.hashes))


class MatcherTests(unittest.TestCase):
    def results(self, markdown, rules=()):
        body = checks.generate({"id": "D99"}, twin())
        body.update(rules=list(rules), twin_kind="docx")
        return {r["id"]: r["outcome"] for r in evaluate(body, from_markdown(markdown))}

    def test_good_output_passes(self):
        r = self.results(GOOD)
        failing = {k: v for k, v in r.items() if v != "pass"}
        self.assertEqual(failing, {})

    def test_failures_have_closed_outcomes(self):
        bad = GOOD.replace("# Introduction", "Introduction").replace("*lazy dog*", "lazy dog") \
                  .replace("- first item here", "first item here").replace("near the river", "near the\nriv-\ner") \
                  .replace("| Nord | 8.0 |", "| Nord | 8.1 |") + "\n— 199 —\n\n<br>A second paragraph explains the numbers of the table below in detail.\n"
        r = self.results(bad)
        self.assertEqual(r["h1"], "not_heading")
        self.assertEqual(r["p1:emph1"], "no_emphasis")
        self.assertEqual(r["li1:list"], "not_list")
        self.assertEqual(r["p1:hyphen"], "broken_word")
        self.assertEqual(r["t1/r2/c2"], "missing")
        self.assertEqual(r["z1:noise"], "repeated")

    def test_plain_text_types_not_applicable(self):
        body = checks.generate({"id": "D99"}, twin())
        r = {x["id"]: x["outcome"] for x in evaluate(body, from_text(GOOD))}
        self.assertEqual(r["h1"], "not_applicable")
        self.assertEqual(r["p1"], "pass")

    def test_display_math_block_does_not_swallow_paragraphs(self):
        doc = from_markdown("A line ends with $$\nx + 1,$$ and goes on.\n\n$$ y\n\nNext paragraph stays.\n\nLast $$z$$\n")
        self.assertIn("stays", doc.tokens)

    def test_page_separators_and_html_grid(self):
        doc = from_markdown("Text\n\n{0}------------------------------------------------\n\n"
                            "<table><tr><td colspan=\"2\">a</td></tr><tr><td>b</td><td>c</td></tr></table>\n")
        self.assertNotIn("0", doc.tokens)
        table = [b for b in doc.blocks if b.kind == "table"][0]
        self.assertEqual(table.rows, [["a", ""], ["b", "c"]])


class NormalizationTests(unittest.TestCase):
    """C3: three distinct normalizations; it-final-accent widens twin tokens only."""

    def run_rule(self, twin_text, output_text, rules):
        b = Builder("akoma-ntoso")
        b.add("paragraph", twin_text)
        body = checks.generate({"id": "D19"}, b.twin)
        body.update(rules=list(rules), twin_kind="akoma-ntoso")
        return evaluate(body, from_markdown(output_text))[0]["outcome"]

    def test_accent_rule_is_one_way_and_only_where_declared(self):
        twin_text = "La abusivita' e' punita"
        self.assertEqual(self.run_rule(twin_text, "La abusività è punita", ["it-final-accent"]), "pass")
        self.assertEqual(self.run_rule(twin_text, "La abusività é punita", ["it-final-accent"]), "pass")
        # an accent error of the converter is still penalized
        self.assertNotEqual(self.run_rule(twin_text, "La abusivita e punita", ["it-final-accent"]), "pass")
        self.assertNotEqual(self.run_rule("La abusività è punita", "La abusivita' e' punita", ["it-final-accent"]), "pass")
        # without the rule (any other document) the forms differ
        self.assertNotEqual(self.run_rule(twin_text, "La abusività è punita", []), "pass")

    def test_score_and_alignment_are_distinct(self):
        self.assertNotEqual(score_tokens("perché"), score_tokens("perchè"))
        self.assertEqual(align_key("perché"), align_key("perche'"))
        self.assertEqual(accepts(set())("e'"), {"e'"})
        self.assertEqual(accepts({"it-final-accent"})("e'"), {"e'", "è", "é"})


class CoverageTests(unittest.TestCase):
    def test_coverage_omission_and_fuzzy(self):
        body = "of a longer text that keeps going with many common words"
        inv = {"paragraph": [f"unit number {k} {body} 1" for k in range(12)]}
        gen = {"paragraph": [f"unit number {k} {body}" for k in range(12)]}
        self.assertEqual(selfcheck.coverage(inv, gen)["paragraph"][2], 1.0)          # fuzzy, absent rare token
        self.assertLess(selfcheck.coverage(inv, gen, drop_every=10)["paragraph"][2], selfcheck.COVERAGE_MIN)
        small = {"citation": ["a b c d", "e f g h"]}
        self.assertLess(selfcheck.coverage(small, {"citation": ["a b c d", "e f g h"]}, drop_every=10)["citation"][2], 0.95)

    def test_pandoc_inventory_discards_the_bibliography_width(self):
        para = lambda s: {"t": "Para", "c": [{"t": "Str", "c": w} if i % 2 == 0 else {"t": "Space"}
                                            for i, w in enumerate(sum(([x, " "] for x in s.split()), [])[:-1])]}
        ast = {"blocks": [{"t": "Div", "c": [["", ["thebibliography"], []], [para("10"), para("A. Author Title"), para("B. Bee Other")]]},
                          {"t": "DefinitionList", "c": [[[{"t": "Str", "c": "MoM"}], [[para("Metal on metal")]]]]}]}
        units = selfcheck.pandoc_inventory(ast)
        self.assertEqual(units["bib_entry"], ["a author title", "b bee other"])
        self.assertEqual(units["list_item"], ["mom metal on metal"])

    def test_plain_ast_needs_no_raw_html(self):
        ast = [{"t": "SmallCaps", "c": [{"t": "Str", "c": "Stolz"}]}, {"t": "RawInline", "c": ["html", "<span>"]},
               {"t": "Link", "c": [["id", ["c"], [["data-x", "1"]]], [{"t": "Str", "c": "1"}], ["#f", ""]]}]
        out = selfcheck._plain_ast(ast)
        self.assertEqual(out[0], {"t": "Str", "c": "Stolz"})
        self.assertEqual(out[1]["c"][0], ["", [], []])


class ExclusionAndScoreTests(unittest.TestCase):
    def setUp(self):
        self.body = checks.generate({"id": "D99"}, twin())
        self.body.update(twin_kind="jats", rules=[])
        self.results = evaluate(self.body, from_markdown(GOOD.replace("## Results", "Results")))

    def entry(self, check, code, version=None):
        return {"document": "D99", "check": check, "code": code, "reason": "r", "origin": {"kind": "self-check"},
                "generator_version": version or checks.GENERATOR_VERSION}

    def test_preconditions(self):
        entries = [self.entry("h2", "self_check_tool_limit"), self.entry("p1", "self_check_tool_limit"),
                   self.entry("p2", "twin_only"), self.entry("gone", "twin_only"), self.entry("h1", "twin_only", "0")]
        statuses = exclusions.validate(entries, self.body, self.results, lambda c: True, lambda c: 0.2)
        self.assertEqual(statuses, {"h2": "ok", "p1": "stale", "p2": "ok", "gone": "stale", "h1": "revalidate"})
        statuses = exclusions.validate([self.entry("p2", "twin_only")], self.body, self.results, lambda c: False, lambda c: 0.2)
        self.assertEqual(statuses, {"p2": "stale"})

    def test_scores_keep_tool_limits_and_drop_twin_only(self):
        table = score.scores(self.body, self.results, [self.entry("h2", "self_check_tool_limit"), self.entry("p2", "twin_only")], "office")
        content = table["content"]
        self.assertEqual(content["excluded_self_check"], 1)
        self.assertEqual(content["evaluated"], content["generated"] - 1)
        self.assertEqual(table["structure"]["excluded_self_check"], 0)
        self.assertTrue(table["structure"]["provisional"])

    def test_packaged_exclusions_file_is_valid(self):
        data = exclusions.load()
        self.assertEqual(data["schema"], exclusions.SCHEMA)
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "x.json"
            bad.write_text(json.dumps({"schema": exclusions.SCHEMA, "exclusions": [{"document": "D1"}], "feedback": []}))
            with self.assertRaises(Exception):
                exclusions.load(bad)
