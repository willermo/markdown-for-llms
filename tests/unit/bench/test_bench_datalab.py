"""`mdf bench run --route datalab` with a fake transport: no network, no cost (plan r2 §7 T5, §12; delta
r3 Δ5, Δ6, Δ7). The PDF preparation of the Datalab route runs for real in its light child; Datalab
is `bench_run_support.FakeDatalab`, whose POSTs and DELETEs are counted in a file shared by the
processes of a test. Criteria: the order of the refusals (the personal set and the options before
the key file is opened), the key from its file alone and nowhere in the outputs, a reservation in
the ledger before each POST and a completion after it, the caps checked under the lock of the
ledger, never a second POST for an attempt whose reservation is in the ledger."""
import builtins
import contextlib
import errno
from decimal import Decimal
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from bench_run_support import (DATALAB_SPACING_TEST, KEY, PARAGRAPHS, ROOT, Died, Engine, FakeDatalab, build_datalab_set,
                               fake_identity, make_docx, patches, spawn, wait_for, write_key)
from markdown_for_llms.bench import cli, report, run, spend
from markdown_for_llms.bench.codes import BenchError, Code
from markdown_for_llms.bench.settings import BenchSettings
from markdown_for_llms.engine import datalab, tools

SENTINEL = "zqdatalab5e2b"
MODES = ("lite", "fast", "balanced", "accurate")


class DatalabCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)
        self.s = build_datalab_set(self.base)
        self.key_file = write_key(self.base)
        self.out = self.s["runs"] / "r1"
        self.ledger = self.s["runs"] / spend.LEDGER

    def tearDown(self):
        self.tmp.cleanup()

    def options(self, **extra):
        return {"docs": "D06", "routes": "datalab", "datalab_modes": "lite", "confirm_transit": True,
                "max_cost_usd": "0.10", "spend_cap_usd": "4.00", "key_file": str(self.key_file), **extra}

    @contextlib.contextmanager
    def patched(self, engine=None, transport=None):
        transport = transport if transport is not None else FakeDatalab(self.base)
        with contextlib.ExitStack() as stack:
            for obj, name, value in patches(self.s["venv"], engine or Engine(), transport=transport):
                stack.enter_context(mock.patch.object(obj, name, value))
            printed = stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            yield printed

    def start(self, engine=None, transport=None, environ=None, **options):
        settings = BenchSettings.load(None, environ=environ or {}, cache=self.s["cache"], runs=self.s["runs"])
        options = {"manifest": str(self.s["manifest"]), "out": str(self.out), **self.options(), **options}
        options = {k: v for k, v in options.items() if v is not None}
        with self.patched(engine, transport) as printed:
            try:
                return run.start(settings, **options)
            finally:
                self.printed = printed.getvalue()

    def resume(self, engine=None, transport=None, **options):
        with self.patched(engine, transport) as printed:
            try:
                return run.resume(self.out, **options)
            finally:
                self.printed = printed.getvalue()

    def resume_datalab(self, engine=None, transport=None, **options):
        return self.resume(engine, transport, confirm_transit=True, key_file=str(self.key_file), **options)

    def state(self):
        return json.loads((self.out / "state.json").read_bytes())

    def report(self):
        return json.loads((self.out / "report.json").read_bytes())

    def entries(self):
        return spend.read(self.ledger)

    def posts(self):
        return FakeDatalab.calls(self.base, "POST")

    def engine_result(self, attempt_id):
        return json.loads((self.out / "attempts" / attempt_id / "out" / "engine.json").read_bytes())

    def assert_refused(self, code, **options):
        with self.assertRaises(BenchError) as ctx:
            self.start(**options)
        self.assertEqual(ctx.exception.code, code, options)
        self.assertFalse(self.s["runs"].exists(), options)
        self.assertEqual(self.posts(), [])


class CampaignTests(unittest.TestCase):
    def test_standard_campaign_without_d20_and_d16_accurate_last(self):
        from markdown_for_llms.bench import manifest
        docs = manifest.load().select(level="standard")
        keys = run.plan_keys(docs, ["datalab"], "auto", 1, list(MODES))
        self.assertEqual(len(keys), 44)                       # 11 documents x 4 modes (App. F)
        self.assertNotIn("D20", {k["document"] for k in keys})
        self.assertEqual((keys[-1]["document"], keys[-1]["variant"]), ("D16", {"datalab_mode": "accurate"}))
        estimates = [run.datalab_estimate(manifest.load().document(k["document"]), k["variant"]["datalab_mode"])
                     for k in keys]
        self.assertEqual(estimates, sorted(estimates))        # the cheapest first
        self.assertEqual(sum(estimates), Decimal("397.12"))   # the limit of App. F: 3.9712 USD
        ids = {run.attempt_id("0123abcd", k["document"], k["route"], k["v8"], k["k"], 1) for k in keys}
        self.assertEqual(len(ids), 44)
        self.assertEqual(len({run.variant_sha8({"datalab_mode": m}) for m in MODES}), 4)
        summary = report._summary([manifest.load().document("D20")], {"D20": {"status": "measured", "attempts": []}}, {})
        self.assertIn({"format": "djvu", "route": "datalab", "reason": "datalab_excluded",
                       "documents": {"D20": "same_pdf_as_d14"}}, summary["not_measured"])


class RefusalTests(DatalabCase):
    def test_options_are_checked_before_any_file(self):
        cases = [({"datalab_modes": None}, Code.INVALID_SETTING),
                 ({"datalab_modes": "lite,turbo"}, Code.ROUTE_OPTION_UNSUPPORTED),
                 ({"confirm_transit": False}, Code.TRANSIT_NOT_CONFIRMED),
                 ({"max_cost_usd": None}, Code.INVALID_SETTING),
                 ({"max_cost_usd": "-1"}, Code.INVALID_SETTING),
                 ({"spend_cap_usd": None}, Code.INVALID_SETTING),
                 ({"retry_unaccepted": True}, Code.INVALID_SETTING),
                 ({"docs": "D01"}, Code.INVALID_SETTING)]           # a txt has no Datalab route
        for options, code in cases:
            with self.subTest(options=options):
                self.assert_refused(code, **options)

    def test_key_file_only_and_never_the_environment(self):
        self.assert_refused(Code.DATALAB_KEY_MISSING, key_file=None, environ={"DATALAB_API_KEY": KEY})
        empty = write_key(self.base, text="OTHER=1\n")
        with mock.patch.dict(os.environ, {"DATALAB_API_KEY": KEY}):
            self.assert_refused(Code.DATALAB_KEY_MISSING, key_file=str(empty))
        os.chmod(self.key_file, 0o644)
        self.assert_refused(Code.DATALAB_KEY_UNSAFE)

    def test_key_file_without_datalab_is_refused_unopened(self):
        with spied() as seen:
            self.assert_refused(Code.INVALID_SETTING, routes=None, datalab_modes=None, confirm_transit=False,
                                max_cost_usd=None, spend_cap_usd=None, docs="D01")
        self.assertFalse(any(str(self.key_file) in call for call in seen))

    def test_personal_set_is_refused_before_the_key_file(self):
        """Δ5: an unreadable (0000) or unsafe (0644) key file is never opened for a personal set."""
        personal = self.base / "personal.json"
        doc = make_docx(self.base / "p.docx", ["Una frase."])
        personal.write_text(json.dumps({"schema": "mdf-bench-manifest/2", "set": "personal", "documents": [
            {"id": "P01", "format": "docx", "input": {"path": str(doc)}, "twin": None}]}))
        for mode in (0o000, 0o644):
            with self.subTest(mode=oct(mode)):
                os.chmod(self.key_file, mode)
                argv = ["run", "--manifest", str(personal), "--docs", "P01", "--route", "datalab", "--datalab-mode",
                        "lite", "--confirm-transit", "--max-cost-usd", "0.02", "--spend-cap-usd", "4.00",
                        "--env-file", str(self.key_file), "--runs-dir", str(self.s["runs"]),
                        "--out", str(self.out)]
                err = io.StringIO()
                with self.patched(), spied() as seen, contextlib.redirect_stderr(err):
                    code = cli.main(argv)
                self.assertEqual((code, err.getvalue().strip()), (1, "error: datalab_refused_personal"))
                self.assertFalse(any(str(self.key_file) in call for call in seen), seen)
                self.assertFalse(self.s["runs"].exists())
        os.chmod(self.key_file, 0o644)            # the same file on the public set is opened, and refused
        self.assert_refused(Code.DATALAB_KEY_UNSAFE)

    def test_inconsistent_ledger_is_refused_before_the_run(self):
        self.s["runs"].mkdir()
        for content, mode in (('{"kind": "complete", "run_id": "r", "attempt_id": "a", "billed_cents": 1}\n', 0o600),
                              ('{"kind": "reserve", "run_id": "r", "attempt_id": "a", "estimate_cents": 1}', 0o600),
                              ("", 0o644)):
            with self.subTest(content=content, mode=oct(mode)):
                self.ledger.write_text(content)
                os.chmod(self.ledger, mode)
                with self.assertRaises(BenchError) as ctx:
                    self.start()
                self.assertEqual(ctx.exception.code, Code.SPEND_LEDGER_INCONSISTENT)
                self.assertEqual(sorted(p.name for p in self.s["runs"].iterdir()), [spend.LEDGER])
        self.assertEqual(self.posts(), [])

    def test_spend_cap_from_the_setting(self):
        self.assertEqual(self.start(spend_cap_usd=None, environ={"MDF_BENCH_CAP_DATALAB_USD": "0.50"}), 0)
        self.assertEqual(self.state()["datalab"], {"max_cost_usd": "0.10", "spend_cap_usd": "0.50"})
        with self.assertRaises(BenchError) as ctx:
            BenchSettings.load(None, environ={"MDF_BENCH_CAP_DATALAB_USD": "nan"})
        self.assertEqual(ctx.exception.code, Code.INVALID_SETTING)


class SendTests(DatalabCase):
    def test_each_variant_is_reserved_sent_and_completed(self):
        self.assertEqual(self.start(docs="D05", datalab_modes="accurate,lite,balanced,fast", max_cost_usd="1"), 0)
        st, rep = self.state(), self.report()
        posts = self.posts()
        self.assertEqual([p[1] for p in posts], list(MODES))                         # the cheapest first
        times = [float(p[-1]) for p in posts]
        self.assertTrue(all(b - a >= DATALAB_SPACING_TEST for a, b in zip(times, times[1:])), times)
        self.assertEqual(len(FakeDatalab.calls(self.base, "DELETE")), 4)
        entries = self.entries()
        reserves = [e for e in entries if e["kind"] == "reserve"]
        completes = [e for e in entries if e["kind"] == "complete"]
        self.assertEqual([e["variant"]["datalab_mode"] for e in reserves], list(MODES))
        self.assertEqual(len({e["attempt_id"] for e in reserves}), 4)
        self.assertEqual({e["attempt_id"] for e in completes}, {e["attempt_id"] for e in reserves})
        for e in reserves:
            a = st["attempts"][e["attempt_id"]]
            estimate = datalab.estimate_cents(2, a["variant"]["datalab_mode"])
            self.assertEqual((e["estimate_cents"], e["run_id"], e["document"], e["pages"]),
                             (str(estimate), st["run_id"], "D05", 2))
            self.assertEqual(a["request"]["datalab"], {"mode": a["variant"]["datalab_mode"], "transit_confirmed": True,
                                                       "cap_usd": format(estimate / 100, "f")})
            result = self.engine_result(a["id"])
            self.assertEqual((result["status"], result["transit"]["deleted"], result["cost"]["billed_cents"]),
                             ("complete", True, "0.5"))
        _, total = spend.reconcile(entries)
        self.assertEqual(total, Decimal("2.0"))                                       # four billed costs of 0.5
        self.assertIn(run.DATALAB_WARNING, self.printed)
        self.assertIn(report.DATALAB_WARNING_CODE, rep["warnings"])
        self.assertEqual(rep["cost"]["by_mode"]["lite"], {"sends": 1, "pages": 2, "estimated_cents": "1.10",
                                                          "billed_cents": "0.5", "billed_unknown": 0,
                                                          "counted_cents": "0.5"})
        attempts = rep["documents"][0]["attempts"]
        self.assertEqual({a["datalab"]["deleted"] for a in attempts}, {True})
        self.assertEqual({a["scores"]["content"]["ratio"] for a in attempts}, {1.0})
        self.assertNotIn(KEY, json.dumps(st) + self.printed)
        self.assertFalse(any(KEY.encode() in p.read_bytes() for p in self.s["runs"].rglob("*") if p.is_file()))
        md = (self.out / "report.md").read_text()
        self.assertIn("## Datalab", md)
        self.assertIn("| lite | 1 | 2 | 1.10 | 0.5 | 0 | 0.5 |", md)
        # B-SINTESI: the cost of each format × route × variant, billed and uncertain apart
        cells = {c["variant"]["datalab_mode"]: c for c in rep["summary"]["format_route"] if c["route"] == "datalab"}
        self.assertEqual(cells["lite"]["cost"], {"sends": 1, "estimated_cents": "1.10", "billed_cents": "0.5",
                                                 "billed_known": 1, "uncertain_cents": "0", "uncertain": 0,
                                                 "counted_cents": "0.5"})
        self.assertEqual(len(cells), 4)
        self.assertIn("| 0.5 billed + 0 uncertain (estimate 1.10) |", md)

    def test_cost_unknown_keeps_the_estimate(self):
        self.assertEqual(self.start(transport=FakeDatalab(self.base, billed=None)), 0)
        completes = [e for e in self.entries() if e["kind"] == "complete"]
        self.assertEqual([e["billed_cents"] for e in completes], [None])
        self.assertEqual(spend.reconcile(self.entries())[1], Decimal("1.05"))
        a = self.report()["documents"][0]["attempts"][0]
        self.assertEqual((a["status"], a["problem_codes"].get("datalab_cost_unknown")), ("complete", 1))
        self.assertIsNone(a["datalab"]["billed_cents"])

    def test_billed_cost_over_its_estimate_stops_the_run(self):
        with self.assertRaises(BenchError) as ctx:
            self.start(transport=FakeDatalab(self.base, billed=5.0), datalab_modes="lite,fast")
        self.assertEqual(ctx.exception.code, Code.DATALAB_BILLED_OVER_ESTIMATE)
        self.assertEqual(len(self.posts()), 1)
        self.assertEqual((self.state()["status"], self.state()["stop"]), ("stopped", "datalab_billed_over_estimate"))
        self.assertTrue((self.out / "report.json").is_file())
        self.assertEqual((self.report()["run"]["status"], self.report()["run"]["stop"]),
                         ("stopped", "datalab_billed_over_estimate"))
        cell = next(c for c in self.report()["summary"]["format_route"] if c["route"] == "datalab")
        self.assertEqual((cell["cost"]["billed_cents"], cell["cost"]["estimated_cents"]), ("5.0", "1.05"))
        with self.assertRaises(BenchError) as ctx:                       # the stop stays for the maintainer
            self.resume_datalab()
        self.assertEqual(ctx.exception.code, Code.DATALAB_BILLED_OVER_ESTIMATE)
        self.assertEqual(len(self.posts()), 1)

    def test_billed_over_its_estimate_after_an_adoption_stops_before_any_send(self):
        """B-SPESA: the bench dies after the engine published a result billed over its estimate; the
        resumption adopts it, completes its reservation, stops the run and posts nothing more, then and later."""
        with self.assertRaises(Died):
            self.start(Engine("raise"), FakeDatalab(self.base, billed=5.0), datalab_modes="lite,fast")
        self.assertEqual([e["kind"] for e in self.entries()], ["reserve"])
        first = next(iter(self.state()["attempts"].values()))
        with self.assertRaises(BenchError) as ctx:
            self.resume_datalab()
        self.assertEqual(ctx.exception.code, Code.DATALAB_BILLED_OVER_ESTIMATE)
        self.assertEqual(len(self.posts()), 1)                            # no POST after the adoption
        self.assertEqual([(e["kind"], e.get("billed_cents")) for e in self.entries()],
                         [("reserve", None), ("complete", "5.0")])
        st = self.state()
        self.assertEqual((st["status"], st["stop"]), ("stopped", "datalab_billed_over_estimate"))
        self.assertEqual((st["attempts"][first["id"]]["status"], st["attempts"][first["id"]]["adopted"]), ("done", True))
        self.assertEqual(len(st["attempts"]), 1)                          # the second mode never started
        rep = self.report()
        self.assertEqual((rep["run"]["status"], rep["run"]["stop"]), ("stopped", "datalab_billed_over_estimate"))
        self.assertIn(f"D06 datalab:lite {first['v8']} k1 g1 adopted complete", self.printed)
        with self.assertRaises(BenchError) as ctx:
            self.resume_datalab()
        self.assertEqual(ctx.exception.code, Code.DATALAB_BILLED_OVER_ESTIMATE)
        self.assertEqual(len(self.posts()), 1)

    def test_the_completion_precedes_done_on_an_adoption(self):
        """B-SPESA, R4.2: an adopted attempt is done only once its completion is in the ledger."""
        with self.assertRaises(Died):
            self.start(Engine("raise"))
        with mock.patch.object(spend, "_write_line", side_effect=OSError(errno.ENOSPC, "No space left on device")):
            with self.assertRaises(BenchError) as ctx:
                self.resume_datalab()
        self.assertEqual(ctx.exception.code, Code.SPEND_LEDGER_UNWRITABLE)
        a = next(iter(self.state()["attempts"].values()))
        self.assertEqual((a["status"], a["adopted"]), ("started", False))
        self.assertEqual([e["kind"] for e in self.entries()], ["reserve"])
        self.assertEqual(self.resume_datalab(), 0)                         # not a stop that stays
        self.assertEqual(len(self.posts()), 1)
        self.assertEqual([(e["kind"], e.get("billed_cents")) for e in self.entries()],
                         [("reserve", None), ("complete", "0.5")])
        self.assertEqual(self.state()["attempts"][a["id"]]["status"], "done")
        self.assertIsNone(self.report()["run"]["stop"])

    def test_no_text_of_a_document_or_of_the_answer_in_the_reports(self):
        """S20 with a Datalab answer: the sentinel is in the Markdown, the versions, the cost breakdown and
        the request id of the answer, and a second attempt fails after its acceptance (success=false)."""
        class Answers(FakeDatalab):
            def post(self, url, headers, files, data):
                super().post(url, headers, files, data)
                if data["mode"] == "fast":
                    return 200, {"success": False, "request_id": SENTINEL + "-2"}
                return 200, {"success": True, "request_id": SENTINEL}

            def get(self, url, headers):
                self.log("GET")
                return 200, {"status": "complete", "success": True, "markdown": f"{PARAGRAPHS[0]} {SENTINEL}\n",
                             "images": {}, "versions": {SENTINEL: SENTINEL}, "runtime": SENTINEL,
                             "cost_breakdown": {SENTINEL: SENTINEL, "final_cost_cents": 0.25}}
        self.assertEqual(self.start(transport=Answers(self.base), datalab_modes="lite,fast"), 0)
        rep_json, rep_md = (self.out / "report.json").read_text(), (self.out / "report.md").read_text()
        for name, body in (("report.json", rep_json), ("report.md", rep_md), ("stdout", self.printed)):
            self.assertNotIn(SENTINEL, body, name)
        failed = [a for a in self.report()["documents"][0]["attempts"] if a["status"] == "failed"]
        self.assertEqual([a["error"] for a in failed], ["datalab_rejected"])
        completes = {e["request_id"]: e["billed_cents"] for e in self.entries() if e["kind"] == "complete"}
        self.assertEqual(completes, {SENTINEL: "0.25", SENTINEL + "-2": None})     # accepted: the estimate stays

    def test_t1_died_after_publish_is_adopted_and_completed(self):
        with self.assertRaises(Died):
            self.start(Engine("raise"))
        self.assertEqual([e["kind"] for e in self.entries()], ["reserve"])
        self.assertEqual(self.resume_datalab(), 0)
        self.assertEqual(len(self.posts()), 1)
        a = next(iter(self.state()["attempts"].values()))
        self.assertEqual((a["status"], a["adopted"]), ("done", True))
        self.assertEqual([(e["kind"], e.get("billed_cents")) for e in self.entries()],
                         [("reserve", None), ("complete", "0.5")])
        self.assertIn(f"D06 datalab:lite {a['v8']} k1 g1 adopted complete", self.printed)      # R4.1


class OverEstimateStopTests(DatalabCase):
    """B-SPESA: the bench dies right after the real save that publishes `done` for an attempt billed
    over its estimate (5.0 cents against 1.05), on the return of the call and on an adoption. The
    stop is in that same save, so every resumption stops before any POST, the first and the later ones;
    a state left without it by an earlier version is stopped by the completion in the ledger."""

    @contextlib.contextmanager
    def dying_after_done(self):
        """The first save of the run that has an attempt done is written, then the bench dies."""
        save, died = run.Run.save, []

        def save_then_die(r):
            save(r)
            if not died and any(a["status"] == "done" for a in r.attempts.values()):
                died.append(True)
                raise Died()
        with mock.patch.object(run.Run, "save", save_then_die):
            yield
        self.assertEqual(died, [True])

    def resume_code(self):
        try:
            return self.resume_datalab()
        except BenchError as exc:
            return exc.code

    def resume_cli(self):
        """`mdf bench run --resume RUN_DIR --confirm-transit --env-file KEY` in this process: (exit, stderr)."""
        self.addCleanup(signal.signal, signal.SIGINT, signal.getsignal(signal.SIGINT))
        err = io.StringIO()
        with self.patched(), mock.patch.dict(tools._memory), contextlib.redirect_stderr(err):
            code = cli.main(["run", "--resume", str(self.out), "--confirm-transit", "--env-file", str(self.key_file)])
        return code, err.getvalue().strip()

    def assert_stopped_before_any_send(self, died):
        """`died` is the state the death left: the first attempt done and billed over its estimate."""
        first = next(iter(died["attempts"].values()))
        self.assertEqual((first["status"], first["variant"]["datalab_mode"]), ("done", "lite"))
        self.assertEqual([(e["kind"], e.get("billed_cents")) for e in self.entries()],
                         [("reserve", None), ("complete", "5.0")])
        code = self.resume_code()
        self.assertEqual((len(self.posts()), code), (1, Code.DATALAB_BILLED_OVER_ESTIMATE))     # no POST
        st, rep = self.state(), self.report()
        self.assertEqual((st["status"], st["stop"], st["executions"][-1]["exit"]),
                         ("stopped", "datalab_billed_over_estimate", 1))
        self.assertEqual((rep["run"]["status"], rep["run"]["stop"]), ("stopped", "datalab_billed_over_estimate"))
        self.assertEqual(list(st["attempts"]), [first["id"]])                 # the second mode never started
        self.assertEqual(self.resume_cli(), (1, "error: datalab_billed_over_estimate"))     # and again later
        self.assertEqual(len(self.posts()), 1)
        self.assertEqual(len(self.entries()), 2)

    def test_died_after_done_on_the_return_of_the_call(self):
        with self.dying_after_done(), self.assertRaises(Died):
            self.start(transport=FakeDatalab(self.base, billed=5.0), datalab_modes="lite,fast")
        died, printed = self.state(), self.printed
        self.assert_stopped_before_any_send(died)
        # the stop is in the save that published `done`, before the line of the attempt and the stop of the run
        self.assertEqual((died["status"], died.get("stop")), ("running", "datalab_billed_over_estimate"))
        self.assertNotIn("D06 datalab:lite", printed)

    def test_died_after_done_on_an_adoption(self):
        with self.assertRaises(Died):
            self.start(Engine("raise"), FakeDatalab(self.base, billed=5.0), datalab_modes="lite,fast")
        with self.dying_after_done(), self.assertRaises(Died):
            self.resume_datalab()
        died, printed = self.state(), self.printed
        self.assert_stopped_before_any_send(died)
        # the stop is in the save that published the adopted attempt, before its line `adopted`
        self.assertEqual((died["status"], died.get("stop")), ("running", "datalab_billed_over_estimate"))
        self.assertTrue(next(iter(died["attempts"].values()))["adopted"])
        self.assertNotIn("adopted", printed)

    def test_a_stop_left_unrecorded_is_rebuilt_from_the_ledger(self):
        """The state an earlier version left in that window (the attempt done, no stop): the resumption
        finds the completion billed over the estimate of its reservation in the ledger."""
        with self.dying_after_done(), self.assertRaises(Died):
            self.start(transport=FakeDatalab(self.base, billed=5.0), datalab_modes="lite,fast")
        died = self.state()
        died.pop("stop", None)
        run.save_state(self.out, died)
        self.assert_stopped_before_any_send(died)


class InterruptionTests(DatalabCase):
    def args(self, *extra):
        return ["--docs", "D06", "--manifest", self.s["manifest"], "--cache", self.s["cache"], "--runs-dir",
                self.s["runs"], "--out", self.out, "--route", "datalab", "--datalab-mode", "lite", "--confirm-transit",
                "--max-cost-usd", "0.10", "--spend-cap-usd", "4.00", "--env-file", self.key_file, *extra]

    def test_t5_sigint_after_the_acceptance_then_resume_is_unresolved(self):
        """Δ7: the first GET waits; SIGINT after `datalab-request.json`; the resumption never posts again."""
        proc = spawn("datalab-hold", self.base, self.args())
        try:
            evidence = wait_for(self.base / "holding", proc, timeout=120)
            self.assertTrue(evidence)
            self.assertTrue(list((self.out / "work").glob("attempt-*-D06-datalab-*/datalab-request.json")))
            proc.send_signal(signal.SIGINT)
            _, err = proc.communicate(timeout=60)
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.communicate()
        self.assertEqual((proc.returncode, err.decode().strip().splitlines()[-1]), (1, "error: interrupted"))
        first = next(iter(self.state()["attempts"].values()))
        self.assertEqual(first["status"], "interrupted")
        result = self.engine_result(first["id"])
        self.assertEqual((result["error"]["code"], result["transit"]["deleted"]), ("interrupted", True))
        self.assertEqual([e["kind"] for e in self.entries()], ["reserve"])
        self.assertEqual(self.resume_datalab(), 0)
        self.assertEqual(len(self.posts()), 1)                        # one POST in all, resumption included
        self.assertIn(f"D06 datalab:lite {first['v8']} k1 g1 unresolved datalab_unresolved", self.printed)   # R4.1
        st = self.state()
        self.assertEqual(st["attempts"][first["id"]]["status"], "unresolved")
        self.assertEqual(len(st["attempts"]), 1)
        a = self.report()["documents"][0]["attempts"][0]
        self.assertEqual((a["status"], a["error"], a["scores"]), ("failed", "datalab_unresolved", None))
        self.assertEqual(spend.reconcile(self.entries())[1], Decimal("1.05"))      # the estimate stays
        self.assertEqual(self.resume(), 0)                # nothing left to send: neither transit nor key asked

    def test_interrupted_before_the_send_is_converted_again(self):
        def stop(_):
            raise KeyboardInterrupt
        with self.assertRaises(BenchError) as ctx:
            self.start(Engine(hooks={"classify": stop}))
        self.assertEqual(ctx.exception.code, Code.INTERRUPTED)
        self.assertEqual((self.posts(), self.entries()), ([], []))
        with self.assertRaises(BenchError) as ctx:
            self.resume()                                     # a Datalab key left: transit and key asked again
        self.assertEqual(ctx.exception.code, Code.TRANSIT_NOT_CONFIRMED)
        self.assertEqual(self.resume_datalab(), 0)
        self.assertEqual(len(self.posts()), 1)
        self.assertEqual(sorted(a["status"] for a in self.state()["attempts"].values()), ["abandoned", "done"])

    def test_resume_options(self):
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            code = cli.main(["run", "--resume", str(self.out), "--datalab-mode", "lite"])
        self.assertEqual((code, err.getvalue().strip()), (1, "error: resume_option_not_allowed"))
        with self.patched():
            self.assertEqual(run.start(BenchSettings.load(None, environ={}, cache=self.s["cache"], runs=self.s["runs"]),
                                       docs="D01", manifest=str(self.s["manifest"]), out=str(self.out)), 0)
        with spied() as seen, self.assertRaises(BenchError) as ctx:
            self.resume(key_file=str(self.key_file), confirm_transit=True)
        self.assertEqual(ctx.exception.code, Code.INVALID_SETTING)
        self.assertFalse(any(str(self.key_file) in call for call in seen))


class LedgerRuleTests(DatalabCase):
    def test_retry_unaccepted_counts_one_estimate(self):
        """Δ6: a 429 without a request id closes its reservation at 0; the retry counts one estimate."""
        self.assertEqual(self.start(transport=FakeDatalab(self.base, post=(429, {"detail": "slow down"}))), 0)
        first = next(iter(self.state()["attempts"].values()))
        self.assertEqual(self.engine_result(first["id"])["error"]["code"], "datalab_rejected")
        self.assertEqual([(e["kind"], e.get("billed_cents")) for e in self.entries()], [("reserve", None), ("complete", "0")])
        self.assertEqual(self.resume_datalab(), 0)                   # without --retry-unaccepted: an outcome
        self.assertEqual(len(self.posts()), 1)
        self.assertEqual(self.resume_datalab(transport=FakeDatalab(self.base, billed=None), retry_unaccepted=True), 0)
        self.assertEqual(len(self.posts()), 2)
        self.assertEqual(spend.reconcile(self.entries())[1], Decimal("1.05"))      # one estimate in all
        self.assertEqual(sorted(a["g"] for a in self.state()["attempts"].values()), [1, 2])

    def test_refused_after_the_acceptance_keeps_the_estimate_and_is_not_retried(self):
        self.assertEqual(self.start(transport=FakeDatalab(self.base, post=(200, {"success": False,
                                                                                 "request_id": "req-x"}))), 0)
        self.assertEqual([(e["kind"], e.get("billed_cents")) for e in self.entries()], [("reserve", None), ("complete", None)])
        self.assertEqual(self.resume_datalab(retry_unaccepted=True), 0)
        self.assertEqual(len(self.posts()), 1)
        self.assertEqual(spend.reconcile(self.entries())[1], Decimal("1.05"))

    def test_cap_taken_by_another_process_between_two_sends(self):
        """Δ6: between the call and the send of the second attempt a second process reserves the rest of
        the ledger cap: `datalab_cap` in `engine.json`, no POST, the run stops with exit 1."""
        def other_process(calls):
            if calls == 2:
                code = ("import sys; sys.path.insert(0, sys.argv[1]); from markdown_for_llms.bench import spend\n"
                        "with spend.Ledger(sys.argv[2]).locked() as l:\n"
                        "    l.append({'kind': 'reserve', 'run_id': 'other', 'attempt_id': 'x', 'estimate_cents': '399'})\n")
                subprocess.run([sys.executable, "-I", "-B", "-c", code, str(ROOT), str(self.s["runs"])], check=True)
        with self.assertRaises(BenchError) as ctx:
            self.start(Engine(hooks={"classify": other_process}), datalab_modes="lite,fast")
        self.assertEqual(ctx.exception.code, Code.DATALAB_CAP)
        self.assertEqual([p[1] for p in self.posts()], ["lite"])
        st = self.state()
        second = next(a for a in st["attempts"].values() if a["variant"] == {"datalab_mode": "fast"})
        self.assertEqual((st["status"], second["status"]), ("stopped", "stopped"))
        result = self.engine_result(second["id"])
        self.assertEqual((result["error"]["code"], result["transit"]["left_machine"]), ("datalab_cap", False))
        self.assertEqual(second["request"]["datalab"]["cap_usd"], "0.0114")        # frozen before the call
        failed = [a for a in self.report()["documents"][0]["attempts"] if a["status"] == "failed"]
        self.assertEqual([a["error"] for a in failed], ["datalab_cap"])
        with self.assertRaises(BenchError) as ctx:                  # still full: the resumption sends nothing
            self.resume_datalab()
        self.assertEqual(ctx.exception.code, Code.DATALAB_CAP)
        self.assertEqual(len(self.posts()), 1)
        self.assertEqual(self.state()["attempts"][second["id"]]["status"], "abandoned")

    def test_unwritable_ledger_sends_nothing(self):
        with mock.patch.object(spend, "_write_line", side_effect=OSError(errno.ENOSPC, "No space left on device")):
            with self.assertRaises(BenchError) as ctx:
                self.start()
        self.assertEqual(ctx.exception.code, Code.SPEND_LEDGER_UNWRITABLE)
        self.assertEqual(self.posts(), [])
        a = next(iter(self.state()["attempts"].values()))
        self.assertEqual(self.engine_result(a["id"])["error"]["code"], "datalab_cap")

    def test_two_runs_at_once_share_the_cap_of_the_ledger(self):
        """Two bench processes, one ledger whose cap holds one send: exactly one POST in all."""
        procs = []
        for name in ("a", "b"):
            out = self.s["runs"] / name
            args = ["--docs", "D06", "--manifest", self.s["manifest"], "--cache", self.s["cache"], "--runs-dir",
                    self.s["runs"], "--out", out, "--route", "datalab", "--datalab-mode", "lite", "--confirm-transit",
                    "--max-cost-usd", "0.10", "--spend-cap-usd", "0.015", "--env-file", self.key_file]
            procs.append(spawn("datalab", self.base, args))
        codes = sorted(p.wait(timeout=180) for p in procs)
        for p in procs:
            p.communicate()
        self.assertEqual(codes, [0, 1])
        self.assertEqual(len(self.posts()), 1)
        entries = self.entries()
        self.assertEqual([e["kind"] for e in entries], ["reserve", "complete"])
        self.assertLessEqual(spend.reconcile(entries)[1], Decimal("1.5"))

    def test_ledger_created_by_another_process_meanwhile_is_opened(self):
        """Two first writers: the ledger appears between the open that misses it and the exclusive create."""
        self.s["runs"].mkdir()
        real = os.open

        def racing(path, flags, *args):
            if str(path) == str(self.ledger) and flags & os.O_EXCL:
                os.close(real(path, os.O_WRONLY | os.O_CREAT, 0o600))          # the other process
            return real(path, flags, *args)
        with mock.patch.object(os, "open", racing):
            with spend.Ledger(self.s["runs"]).locked() as ledger:
                ledger.append({"kind": "reserve", "run_id": "r", "attempt_id": "a", "estimate_cents": "1"})
        self.assertEqual(len(self.entries()), 1)

    def test_concurrent_writers_never_mix_their_lines(self):
        code = ("import sys; sys.path.insert(0, sys.argv[1]); from markdown_for_llms.bench import spend\n"
                "for i in range(25):\n"
                "    with spend.Ledger(sys.argv[2]).locked() as l:\n"
                "        n = len(l.entries())\n"
                "        l.append({'kind': 'reserve', 'run_id': sys.argv[3], 'attempt_id': str(i), 'estimate_cents': '1',"
                " 'seen': n, 'pad': 'x' * 3000})\n")
        self.s["runs"].mkdir()
        procs = [subprocess.Popen([sys.executable, "-I", "-B", "-c", code, str(ROOT), str(self.s["runs"]), f"w{i}"])
                 for i in range(4)]
        self.assertEqual([p.wait(timeout=120) for p in procs], [0] * 4)
        entries = self.entries()
        self.assertEqual(len(entries), 100)
        self.assertEqual(sorted(e["seen"] for e in entries), list(range(100)))      # each one saw all the others
        self.assertEqual(spend.reconcile(entries)[1], Decimal(100))
        self.assertEqual(os.stat(self.ledger).st_mode & 0o777, 0o600)


@contextlib.contextmanager
def spied():
    """Every path given to os.open, open, io.open and dotenv_values while it is active."""
    import dotenv
    seen = []
    real = {"os": os.open, "open": builtins.open, "io": io.open, "dotenv": dotenv.dotenv_values}

    def spy(name):
        def call(*args, **kwargs):
            seen.append(" ".join(str(a) for a in args[:1]))
            return real[name](*args, **kwargs)
        return call
    with mock.patch.object(os, "open", spy("os")), mock.patch.object(builtins, "open", spy("open")), \
            mock.patch.object(io, "open", spy("io")), mock.patch.object(dotenv, "dotenv_values", spy("dotenv")):
        yield seen


if __name__ == "__main__":
    unittest.main()
