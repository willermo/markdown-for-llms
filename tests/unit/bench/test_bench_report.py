"""The report of a measured run: scores of the txt inputs, the encoding check, the summaries and no text
of a document anywhere (plan r2 §9, App. E; delta r3 Δ10, Δ11)."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from bench_run_support import Engine, build_set, fake_identity, patches
from markdown_for_llms.bench import manifest, report, run
from markdown_for_llms.bench.match import evaluate
from markdown_for_llms.bench.outdoc import from_markdown
from markdown_for_llms.bench.settings import BenchSettings
from markdown_for_llms.engine import contract
from markdown_for_llms.engine.codes import Code, EngineError

SENTINEL = "zqsentinel7f3a"


def start(s, engine, out):
    settings = BenchSettings.load(None, environ={}, cache=s["cache"], runs=s["runs"])
    printed = io.StringIO()
    with contextlib.ExitStack() as stack:
        for obj, name, value in patches(s["venv"], engine):
            stack.enter_context(mock.patch.object(obj, name, value))
        stack.enter_context(contextlib.redirect_stdout(printed))
        code = run.start(settings, level="quick", manifest=str(s["manifest"]), out=str(out))
    return code, printed.getvalue()


class TxtMatchingTests(unittest.TestCase):
    CHECKS = {"rules": [], "checks": [
        {"id": "t1/r1/c1", "class": "content", "type": "cell", "table": "t1", "tokens": ["alpha", "beta", "gamma", "delta"],
         "value": None, "digits": -1},
        {"id": "t1:table", "class": "structure", "type": "table", "table": "t1", "cells": [["alpha"]]},
        {"id": "h1", "class": "structure", "type": "heading", "tokens": ["capitolo", "primo"], "level": 2}]}
    # a cell of a text table whose words wrap on the next line, next to the cell of another column
    TEXT = "alpha beta      10\ngamma delta     20\n"

    def test_txt_cells_are_judged_as_plain_text(self):
        outcomes = {r["id"]: r["outcome"] for r in evaluate(self.CHECKS, from_markdown(self.TEXT), txt=True)}
        self.assertEqual(outcomes, {"t1/r1/c1": "pass", "t1:table": "not_applicable", "h1": "not_applicable"})
        markdown = {r["id"]: r["outcome"] for r in evaluate(self.CHECKS, from_markdown(self.TEXT))}
        self.assertEqual(markdown["t1/r1/c1"], "missing")


class FigureImageTests(unittest.TestCase):
    """C R6.2 of the plan: the images of the HTML blocks (`<figure>`, the `<table>` of Calc), of the inline
    HTML tags and of the cells of Markdown tables are images of the output, their files checked under `out/`."""

    CHECKS = {"rules": [], "checks": [
        {"id": "fig1", "class": "content", "type": "figure", "tokens": ["first", "caption", "of", "the", "figure"],
         "index": 1},
        {"id": "fig2", "class": "content", "type": "figure", "tokens": ["second", "caption", "of", "the", "figure"],
         "index": 2}]}

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.out = Path(self.tmp.name)
        (self.out / "assets").mkdir()
        for n in (1, 2, 3):
            (self.out / "assets" / f"asset-000{n}.png").write_bytes(b"\x89PNG")

    def tearDown(self):
        self.tmp.cleanup()

    def doc(self, markdown):
        return from_markdown(markdown, assets_dir=str(self.out))

    def test_figures_with_captions_in_html_blocks(self):
        md = ("Text before the figures.\n\n"
              "<figure>\n<img src=\"assets/asset-0001.png\" />\n<figcaption>First caption of the figure</figcaption>\n"
              "</figure>\n\n"
              "<figure>\n<img src=\"assets/asset-0002.png\" alt=\"two\" />\n"
              "<figcaption>Second caption of the figure</figcaption>\n</figure>\n")
        doc = self.doc(md)
        self.assertEqual(doc.images, [("assets/asset-0001.png", True), ("assets/asset-0002.png", True)])
        outcomes = [r["outcome"] for r in evaluate(self.CHECKS, doc)]
        self.assertEqual(outcomes, ["pass", "pass"])

    def test_an_html_table_of_calc_with_an_image(self):
        md = ("<table>\n<tbody>\n<tr>\n<td>Value</td>\n<td><img src=\"assets/asset-0001.png\" /></td>\n</tr>\n"
              "</tbody>\n</table>\n")
        doc = self.doc(md)
        self.assertEqual(doc.images, [("assets/asset-0001.png", True)])
        self.assertEqual(doc.blocks[0].rows, [["Value", ""]])

    def test_a_pipe_table_with_an_image_and_an_inline_tag(self):
        md = ("| Name | Picture |\n| --- | --- |\n| one | ![chart](assets/asset-0003.png) |\n\n"
              "A paragraph with <img src=\"assets/asset-0002.png\" alt=\"x\"> inline.\n\n"
              "<figure><img src=\"assets/missing.png\" /></figure>\n\n"
              "<p><img src=\"../outside.png\"></p>\n")
        doc = self.doc(md)
        self.assertEqual(doc.images, [("assets/asset-0003.png", True), ("assets/asset-0002.png", True),
                                      ("assets/missing.png", False), ("../outside.png", False)])
        self.assertEqual([r[0] for r in doc.blocks[0].rows], ["Name", "one"])       # the table itself is unchanged


class RunReportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.s = build_set(self.tmp.name)
        self.out = self.s["runs"] / "r1"

    def tearDown(self):
        self.tmp.cleanup()

    def test_scores_encoding_and_summary(self):
        code, _ = start(self.s, Engine(), self.out)
        self.assertEqual(code, 0)
        rep = json.loads((self.out / "report.json").read_bytes())
        self.assertEqual((rep["schema"], rep["valid"], rep["bench"]["matcher"]), ("mdf-bench-report/1", True, "3"))
        docs = {d["id"]: d for d in rep["documents"]}
        for doc_id in ("D01", "D02"):
            a = docs[doc_id]["attempts"][0]
            self.assertEqual((a["status"], a["format_limit"], a["arbiter"]["tool"], a["arbiter"]["arbitrated"]),
                             ("complete", "txt", "pandoc-html", True))
            self.assertEqual(a["scores"]["content"]["ratio"], 1.0)
            self.assertEqual(a["scores"]["content"]["evaluated"], 3)
            self.assertIsNone(a["scores"]["form"]["ratio"])          # emphasis and the like: not applicable
            self.assertEqual(a["encoding"]["expected"], "utf_8" if doc_id == "D01" else "cp1252")
            self.assertEqual(a["encoding"]["decode_equals_oracle"], a["encoding"]["detected"] == a["encoding"]["expected"])
            results = json.loads((self.out / "attempts" / a["attempt_id"] / "results.json").read_bytes())
            self.assertEqual({r["outcome"] for r in results if r["class"] == "content"}, {"pass"})
        self.assertTrue(docs["D01"]["attempts"][0]["encoding"]["decode_equals_oracle"])
        oracle = docs["D03"]["attempts"][0]
        self.assertEqual((oracle["scores"], oracle["arbiter"]["tool"], docs["D03"]["checks"]), (None, "oracle", None))
        cell = rep["summary"]["format_route"][0]
        self.assertEqual((cell["format"], cell["route"], cell["complete"], cell["scores"]["content"]["n"]),
                         ("txt", "text", 3, 2))
        self.assertEqual(rep["summary"]["not_measured"], [])
        self.assertIn("| txt | text |", (self.out / "report.md").read_text())
        self.assertEqual((rep["run"]["status"], rep["run"]["stop"]), ("complete", None))
        profile = rep["summary"]["profile_route"]
        self.assertEqual([(p["profile"], p["route"], p["scores"]["content"]["n"], p["scores"]["content"]["threshold"])
                          for p in profile], [("office", "text", 2, 0.95)])
        self.assertIn("## Profile × route", (self.out / "report.md").read_text())

    def test_no_text_of_a_document_in_reports_and_output(self):
        """S20: the sentinel is in the text, the title, the name of the input as the engine records it,
        the members and paths of the problems, their details, the arguments of a tool, the declared
        limitations, the page methods, the stages and the error detail of a failed attempt."""
        from markdown_for_llms.engine import text as text_route
        text = f"Il primo capitolo {SENTINEL} racconta di un falegname che trovò un legno.\n\nAltro paragrafo qui.\n"
        html = ("<html><body><section class=\"pg-boilerplate\"><p>licence</p></section>"
                f"<p>Il primo capitolo {SENTINEL} racconta di un falegname che trovò un legno.</p>"
                "<p>Altro paragrafo qui.</p></body></html>")
        self.tmp.cleanup()
        self.tmp = tempfile.TemporaryDirectory()
        self.s = build_set(self.tmp.name, text=text, html=html)
        self.out = self.s["runs"] / "r1"
        data = json.loads(self.s["manifest"].read_text())
        for d in data["documents"]:
            d["title"] = d["role"] = SENTINEL
        self.s["manifest"].write_text(json.dumps(data))
        real_convert = text_route.convert

        def failing(source, stage, output_mib=None):
            if "-D03-" in Path(stage).parent.name:
                raise EngineError(Code.TOOL_FAILED, SENTINEL)
            return real_convert(source, stage, output_mib)

        class Tainted(Engine):
            def __call__(self, source, out, request, limits, work, **kwargs):
                result = super().__call__(source, out, request, limits, work, **kwargs)
                if result["status"] != "complete":
                    return result
                result["source"]["name"] = SENTINEL + ".txt"
                result["problems"] = [{"code": "media_lost", "severity": "incomplete", "path": f"word/{SENTINEL}.png",
                                       "detail": SENTINEL, "page": 3}]
                result["problem_codes"] = {"media_lost": 1, SENTINEL: 1}
                result["route"]["steps"] = [{"tool": SENTINEL, "args": [SENTINEL], "seconds": 1.0}]
                result["limits"] = [SENTINEL]
                result["pages"] = {"count": 1, "classes": {SENTINEL: 1}, "pages": [{"page": 1, "marker_method": SENTINEL}]}
                result["timings"]["stages"] = {SENTINEL: 1.0}
                result["text"]["encoding"] = SENTINEL
                contract.write_result(out, result)
                return result
        with mock.patch.object(text_route, "convert", failing):
            code, printed = start(self.s, Tainted(), self.out)
        self.assertEqual(code, 0)
        rep_json = (self.out / "report.json").read_text()
        rep_md = (self.out / "report.md").read_text()
        for name, body in (("report.json", rep_json), ("report.md", rep_md), ("stdout", printed)):
            self.assertNotIn(SENTINEL, body, name)
        rep = json.loads(rep_json)
        failed = next(d for d in rep["documents"] if d["id"] == "D03")["attempts"][0]
        self.assertEqual((failed["status"], failed["error"]), ("failed", "tool_failed"))
        tainted = next(d for d in rep["documents"] if d["id"] == "D01")["attempts"][0]
        self.assertEqual(tainted["problem_pages"], {"media_lost": [3]})
        self.assertIn("unrecognized", tainted["problem_codes"])
        # the outputs themselves keep the text: they are local
        self.assertIn(SENTINEL, (self.out / "attempts" / tainted["attempt_id"] / "out" / "document.md").read_text())


class SummaryTests(unittest.TestCase):
    def test_routes_sharing_a_tool_with_the_arbiter_are_not_arbitrated(self):
        m = manifest.load()
        d22, d26 = m.document("D22"), m.document("D26")
        self.assertEqual(report.arbiter_record(d22, "pandoc"),
                         {"tool": "writer-html", "shares": ["libreoffice"], "arbitrated": False})
        self.assertEqual(report.arbiter_record(d26, "pandoc")["formulas"], "source-coherence")

        def attempt(route, ratio, arbitrated=True):
            return {"route": route, "variant": {}, "variant_sha8": "44136fa3", "status": "complete",
                    "timing": {"s_per_page": 1.0}, "arbiter": {"arbitrated": arbitrated, "shares": [] if arbitrated else ["libreoffice"]},
                    "scores": {c: {"ratio": ratio} for c in report.CLASSES}}
        docs = [m.document(i) for i in ("D04", "D05", "D10", "D22")]
        records = {"D04": {"status": "unavailable", "attempts": []},
                   "D05": {"status": "measured", "attempts": [attempt("marker", 0.8)]},
                   "D10": {"status": "measured", "attempts": [attempt("pandoc", 0.9), attempt("marker", 0.5)]},
                   "D22": {"status": "measured", "attempts": [attempt("pandoc", 0.1, arbitrated=False)]}}
        summary = report._summary(docs, records, {})
        cells = {(c["format"], c["route"]): c for c in summary["format_route"]}
        self.assertEqual(cells[("docx", "pandoc")]["scores"]["content"],
                         {"mean": 0.9, "min": 0.9, "n": 1, "threshold": None, "mean_meets": None, "below_threshold": 0})
        self.assertIsNone(cells[("doc", "pandoc")]["scores"]["content"])
        self.assertEqual(summary["not_arbitrated"][0]["documents"], ["D22"])
        self.assertEqual(summary["not_measured"], [{"format": "pdf", "route": "datalab", "reason": "not_selected"}])
        self.assertEqual(cells[("pdf", "marker")]["scores"]["content"]["n"], 1)      # D04 unavailable
        self.assertEqual({s["script"] for s in summary["script_route"]}, {"Latn"})
        self.assertIsNone(cells[("pdf", "marker")]["cost"])                          # a local route costs nothing

    @staticmethod
    def attempt(route, ratios, thresholds, status="complete", v8="44136fa3", spp=1.0):
        """A scored attempt: per class its ratio and the threshold of the profile of its document."""
        return {"route": route, "variant": {}, "variant_sha8": v8, "status": status,
                "timing": {"s_per_page": spp}, "arbiter": {"arbitrated": True, "shares": []},
                "scores": None if status != "complete" else
                {c: {"ratio": r, "threshold": t} for c, r, t in zip(report.CLASSES, ratios, thresholds)}}

    def test_a_controlled_report_per_profile_against_the_thresholds(self):
        """B-SINTESI (plan r2 §9): per format × route and per profile × route the mean of the arbitrated
        documents, their minimum and number, against the provisional threshold of each profile."""
        m = manifest.load()
        docs = [m.document(i) for i in ("D04", "D05", "D19", "D06")]       # pdf: scientific ×2, legal, office
        sci, legal = (0.95, 0.85, 0.70), (0.90, 0.80, 0.60)
        records = {"D04": {"status": "measured", "attempts": [self.attempt("marker", (1.0, 0.9, 0.8), sci, spp=2.0),
                                                              self.attempt("pandoc", (0.5, 0.5, 0.5), sci)]},
                   "D05": {"status": "measured", "attempts": [self.attempt("marker", (0.9, 0.84, 0.5), sci, spp=4.0)]},
                   "D19": {"status": "measured", "attempts": [self.attempt("marker", (0.92, 0.82, 0.62), legal, spp=3.0)]},
                   "D06": {"status": "measured", "attempts": [self.attempt("marker", None, None, status="failed")]}}
        summary = report._summary(docs, records, {})
        cell = next(c for c in summary["format_route"] if (c["format"], c["route"]) == ("pdf", "marker"))
        self.assertEqual((cell["complete"], cell["failed"], cell["s_per_page_median"]), (3, 1, 3.0))
        content = cell["scores"]["content"]
        self.assertEqual((content["mean"], content["min"], content["n"]), (0.94, 0.9, 3))
        # the documents of the cell have two profiles, two thresholds: no single threshold for the mean, but
        # each document against the threshold of its own profile (D05 0.90 < 0.95; D19 0.92 >= 0.90)
        self.assertEqual((content["threshold"], content["mean_meets"], content["below_threshold"]), (None, None, 1))
        profiles = {(p["profile"], p["route"]): p for p in summary["profile_route"]}
        self.assertEqual(set(profiles), {("scientific", "marker"), ("scientific", "pandoc"), ("legal", "marker")})
        sci_marker = profiles[("scientific", "marker")]["scores"]
        self.assertEqual(sci_marker["content"], {"mean": 0.95, "min": 0.9, "n": 2, "threshold": 0.95,
                                                 "mean_meets": True, "below_threshold": 1})
        self.assertEqual(sci_marker["structure"], {"mean": 0.87, "min": 0.84, "n": 2, "threshold": 0.85,
                                                   "mean_meets": True, "below_threshold": 1})
        self.assertEqual(sci_marker["form"]["mean_meets"], False)              # 0.65 against 0.70
        self.assertEqual(profiles[("legal", "marker")]["scores"]["form"],
                         {"mean": 0.62, "min": 0.62, "n": 1, "threshold": 0.6, "mean_meets": True, "below_threshold": 0})
        self.assertEqual(summary["thresholds"]["provisional"], True)
        md = report.markdown(MarkdownTests.report(None, summary=summary))
        self.assertIn("## Profile × route", md)
        self.assertIn("| scientific | marker | 44136fa3 | 0.950 (min 0.900, n 2, 1 below 0.95) |", md)
        self.assertIn("| pdf | marker | 44136fa3 | 3 | 1 | 0.940 (min 0.900, n 3, 1 below) |", md)

    def test_the_cost_of_each_format_and_route_from_the_ledger(self):
        """B-SINTESI (plan r2 §9): the cost in the summary format × route, the billed cost known, the
        estimate and the uncertain cost (an estimate that stands for a cost not billed) apart."""
        docs = {"D05": {"format": "pdf"}, "D18": {"format": "png"}}
        attempts = {f"r-D05-datalab-aaaa-1-g{g}": {"document": "D05", "route": "datalab", "v8": "aaaaaaaa"} for g in (1, 2)}
        attempts.update({"r-D18-datalab-bbbb-1-g1": {"document": "D18", "route": "datalab", "v8": "bbbbbbbb"},
                         "r-D05-datalab-cccc-1-g1": {"document": "D05", "route": "datalab", "v8": "cccccccc"}})

        def reserve(aid, cents, run_id="run"):
            return {"kind": "reserve", "run_id": run_id, "attempt_id": aid, "estimate_cents": cents}

        def complete(aid, billed, run_id="run"):
            return {"kind": "complete", "run_id": run_id, "attempt_id": aid, "billed_cents": billed}
        entries = [reserve("r-D05-datalab-aaaa-1-g1", "1.10"), complete("r-D05-datalab-aaaa-1-g1", "0"),   # refused: 0
                   reserve("r-D05-datalab-aaaa-1-g2", "1.10"), complete("r-D05-datalab-aaaa-1-g2", "0.075"),
                   reserve("r-D18-datalab-bbbb-1-g1", "1.05"),                                        # no completion
                   reserve("r-D05-datalab-cccc-1-g1", "2.50"), complete("r-D05-datalab-cccc-1-g1", None),  # uncertain end
                   reserve("x-D05-datalab-aaaa-1-g1", "9.99", run_id="other")]                        # another run
        costs = report.cell_costs(entries, "run", attempts, docs)
        self.assertEqual(costs[("pdf", "datalab", "aaaaaaaa")],
                         {"sends": 2, "estimated_cents": "2.20", "billed_cents": "0.075", "billed_known": 2,
                          "uncertain_cents": "0", "uncertain": 0, "counted_cents": "0.075"})
        self.assertEqual(costs[("png", "datalab", "bbbbbbbb")],
                         {"sends": 1, "estimated_cents": "1.05", "billed_cents": "0", "billed_known": 0,
                          "uncertain_cents": "1.05", "uncertain": 1, "counted_cents": "1.05"})
        self.assertEqual(costs[("pdf", "datalab", "cccccccc")]["uncertain_cents"], "2.50")
        self.assertEqual(len(costs), 3)


class ReportRebuildTests(unittest.TestCase):
    """`mdf bench report RUN_DIR`: the report of a finished run again, from its saved outputs, converting
    nothing; the identity of the run, its check sets and the results it recorded are verified first."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.s = build_set(self.tmp.name)
        self.out = self.s["runs"] / "r1"
        code, _ = start(self.s, Engine(), self.out)
        self.assertEqual(code, 0)

    def tearDown(self):
        self.tmp.cleanup()

    def rebuild(self, engine=None):
        engine = engine or Engine()
        printed, err = io.StringIO(), io.StringIO()
        with contextlib.ExitStack() as stack:
            for obj, name, value in patches(self.s["venv"], engine):
                stack.enter_context(mock.patch.object(obj, name, value))
            stack.enter_context(contextlib.redirect_stdout(printed))
            stack.enter_context(contextlib.redirect_stderr(err))
            code = cli_main(["report", str(self.out)])
        return code, printed.getvalue(), err.getvalue().strip()

    def outputs(self):
        return {str(p.relative_to(self.out)): hashlib_sha(p) for p in sorted(self.out.rglob("*"))
                if p.is_file() and p.name in ("engine.json", "document.md", "state.json")}

    def test_the_report_is_written_again_without_converting(self):
        before = json.loads((self.out / "report.json").read_bytes())
        outputs = self.outputs()
        for path in list((self.s["cache"] / "checks").glob("*.json")):     # the same checks by an earlier generator
            body = json.loads(path.read_bytes())
            older = {**body, "generator": "4", "key": "0" * 64, "not_determinable": {"content": 9, "form": 9, "structure": 9}}
            path.with_name("older-" + path.name).write_text(json.dumps(older))
        (self.out / "report.json").unlink()
        (self.out / "report.md").unlink()
        engine = Engine()
        code, printed, _ = self.rebuild(engine)
        self.assertEqual(code, 0)
        self.assertEqual(engine.calls, [])                                    # nothing converted
        self.assertEqual(self.outputs(), outputs)                             # state and outputs untouched
        after = json.loads((self.out / "report.json").read_bytes())
        self.assertEqual(after["documents"], before["documents"])
        self.assertEqual((after["run"]["id"], after["run"]["status"], after["run"]["stop"]),
                         (before["run"]["id"], "complete", None))
        self.assertIsNone(before["rebuilt"])
        self.assertEqual(sorted(after["rebuilt"]), ["at", "package"])
        self.assertIn("Report written again", (self.out / "report.md").read_text())
        self.assertIn(f"report {before['run']['id']}: 3 results verified", printed)

    def test_whatever_does_not_match_is_refused_with_a_closed_code(self):
        state = json.loads((self.out / "state.json").read_bytes())
        attempt = next(a for a in state["attempts"].values() if a["document"] == "D01")
        doc_md = self.out / "attempts" / attempt["id"] / "out" / "document.md"
        good = doc_md.read_bytes()
        doc_md.write_bytes(good + b"x")                                         # the inventory no longer holds
        self.assertEqual(self.rebuild()[::2], (1, "error: report_result_changed"))
        doc_md.write_bytes(good)
        engine_json = doc_md.parent / "engine.json"
        data = engine_json.read_bytes()
        engine_json.write_bytes(data + b"\n")                                    # not the result the state recorded
        self.assertEqual(self.rebuild()[::2], (1, "error: report_result_changed"))
        engine_json.write_bytes(data)
        checks = next(p for p in (self.s["cache"] / "checks").glob("*.json")
                      if json.loads(p.read_bytes())["document"] == "D01")
        body = json.loads(checks.read_bytes())
        checks.write_text(json.dumps({**body, "checks": body["checks"][1:]}))    # another check set
        self.assertEqual(self.rebuild()[::2], (1, "error: report_checks_changed"))
        checks.write_text(json.dumps(body))
        manifest_bytes = self.s["manifest"].read_bytes()
        self.s["manifest"].write_bytes(manifest_bytes + b"\n")
        self.assertEqual(self.rebuild()[::2], (1, "error: report_identity_changed"))
        self.s["manifest"].write_bytes(manifest_bytes)
        state["status"] = "interrupted"
        (self.out / "state.json").write_text(json.dumps(state))
        self.assertEqual(self.rebuild()[::2], (1, "error: report_run_not_finished"))
        state["status"] = "complete"
        (self.out / "state.json").write_text(json.dumps(state))
        import fcntl
        import os
        fd = os.open(self.out / "run.lock", os.O_RDWR)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.assertEqual(self.rebuild()[::2], (1, "error: run_busy"))
        finally:
            os.close(fd)
        self.assertEqual(self.rebuild()[0], 0)


def hashlib_sha(path):
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cli_main(argv):
    from markdown_for_llms.bench import cli
    return cli.main(argv)


class MarkdownTests(unittest.TestCase):
    """`report.md` from a report: the encoding column and the notes."""

    @staticmethod
    def report(encoding, notes=None, summary=None, status="complete", stop=None):
        attempt = {"route": "text", "variant_sha8": "44136fa3", "repeat": 1, "generation": 1, "status": "complete",
                   "error": None, "scores": None, "encoding": encoding, "timing": {"total_s": 0.1}, "problem_codes": {}}
        return {"valid": True, "run": {"id": "r", "level": None, "documents": 1, "repeat": 1, "ocr": "auto", "executions": 1,
                                       "generations": {"adopted": 0, "abandoned": 0}, "exit": 0, "status": status,
                                       "stop": stop},
                "package": {"version": "0", "record_sha256": "0" * 64}, "toolchain": {"content_sha256": "0" * 64},
                "weights": {"manifest_sha256": "0" * 64}, "manifest": {"sha256": "0" * 64},
                "bench": {"version": "1", "generator": "5", "matcher": "3"}, "rebuilt": None,
                "summary": summary or {"format_route": [], "profile_route": [], "not_arbitrated": [], "not_measured": [],
                                       "thresholds": report.thresholds_summary()},
                "documents": [{"id": "P03", "status": "measured", "attempts": [attempt]}], "warnings": [],
                "notes": notes or []}

    def test_status_and_stop_of_the_run(self):
        """R4.3: the report says how the run ended, and why it stopped."""
        md = report.markdown(self.report(None, status="stopped", stop="datalab_billed_over_estimate"))
        self.assertIn("status stopped (datalab_billed_over_estimate)", md)
        self.assertIn("status complete", report.markdown(self.report(None)))

    def test_a_txt_without_an_oracle_is_no_oracle(self):
        """A txt of a personal set has no oracle: `decode_equals_oracle: null` is not `differs` (M8)."""
        lines = {}
        for value in (None, True, False):
            md = report.markdown(self.report({"expected": None, "detected": "ascii", "bom": False,
                                              "decode_equals_oracle": value}))
            lines[value] = next(x for x in md.splitlines() if x.startswith("| P03 |"))
        self.assertIn("| no oracle (ascii) |", lines[None])
        self.assertIn("| ok (ascii) |", lines[True])
        self.assertIn("| differs (ascii) |", lines[False])

    def test_djvutxt_note(self):
        md = report.markdown(self.report(None, notes=["djvutxt_plain_text"]))
        self.assertIn("## Notes", md)
        self.assertIn("djvutxt writes the text layer of a DjVu without structure", md)
        self.assertIn("suspects on those checks of D20 are expected", md)
        self.assertNotIn("## Notes", report.markdown(self.report(None)))


if __name__ == "__main__":
    unittest.main()
