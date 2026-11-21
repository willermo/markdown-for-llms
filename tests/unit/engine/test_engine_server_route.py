"""The `marker-server` route (MFL033: plan r2 §6, §8; r3 §6; r4 §6 and §8): the real Marker child in
`balanced` and `accurate`, through the real relay socket, against the fake server (an
`httpx.MockTransport` behind the relay: the suite forbids connections in its own process; the
child is another process and connects to the relay for real). The weights come from
`FULL_CHECK_MODEL_ROOT`: a missing variable is a failure, never a skip."""
from decimal import Decimal
import json
import os
from pathlib import Path
import tempfile
import unittest

from engine_test_support import FIGURE, INVISIBLE, NATIVE_TEXT, SCAN, checker_rgb, make_pdf, scan_of
from fake_inference import Fake
from markdown_for_llms.engine import marker_child, pages, servers
from markdown_for_llms.engine.codes import Code
from markdown_for_llms.engine.envfile import Secret

IMG = (100, 100, checker_rgb(100, 100))
SCAN_TEXT = "Scanned page of the engine test"
KEY = "sk-route-test-0123456789"


def model_root():
    root = os.environ.get("FULL_CHECK_MODEL_ROOT")
    if not root:
        raise AssertionError("FULL_CHECK_MODEL_ROOT is not set: run the suite from scripts/full-check.sh")
    return root


def respond(request):
    """Answers of the fake for the prompts of Surya: a full page is a top-level div with a box."""
    from surya.inference.prompts import PROMPT_MAPPING
    text = request["messages"][0]["content"][1]["text"]
    if text == PROMPT_MAPPING["high_accuracy_bbox"]:
        return '<div data-bbox="60 60 940 160" data-label="Text"><p>%s</p></div>' % SCAN_TEXT
    return "<p>%s</p>" % SCAN_TEXT


def scanned():
    return {"content": SCAN, "image": scan_of([SCAN_TEXT, "Marker reads it with OCR"], scale=1.5)}


def native():
    return {"content": NATIVE_TEXT + b" " + FIGURE, "image": IMG}


def local(**extra):
    return servers.endpoint("fake", {"kind": "local", "url": "http://127.0.0.1:9", **extra})


def runpod(**extra):
    return servers.endpoint("rp", {"kind": "runpod", "url": "http://127.0.0.1:9", "rate_usd_per_s": "0.0002090948",
                                   **extra})


class RouteCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def run_route(self, pdf_pages, mode, ep=None, fake=None, server=None, observer=None, work=None, keys=None, attempt="srv"):
        from markdown_for_llms.engine.core import convert
        fake = fake or Fake(respond=respond)
        self.addCleanup(fake.close)
        ep = ep or local()
        src = self.root / "in.pdf"
        src.write_bytes(make_pdf(pdf_pages))
        request = {"route": "marker-server", "attempt": attempt, "server": {"name": ep.name, "mode": mode, **(server or {})}}
        out = self.root / ("out-" + attempt)
        result = convert(src, out, request, {"memory_mib": 4096, "threads": 2}, work=work or self.root / "work",
                         observer=observer, model_root=model_root(), server_endpoint=ep, server_keys=keys or {},
                         server_transport=fake.factory())
        return result, out, fake


class TestRealChild(RouteCase):
    def test_balanced_reads_the_scan_on_the_server_from_a_deep_work_directory(self):
        work = self.root / ("w" * 120) / ("x" * 100)            # the socket path does not depend on --work
        result, out, fake = self.run_route([native(), scanned()], "balanced", work=work, attempt="bal")
        self.assertEqual(result["status"], "complete", result.get("error"))
        self.assertGreater(len(os.fsencode(str(work))), 200)
        recs = result["pages"]["pages"]
        self.assertIn(recs[0]["indra_source"], ("text", "hybrid"))
        self.assertEqual((recs[1]["indra_source"], recs[0]["pictures_added"]), ("server", 1))
        text = (out / "document.md").read_text()
        self.assertIn("Native text of the synthetic page", text)
        self.assertIn(SCAN_TEXT, text)
        self.assertIn("](assets/asset-0001.", text)
        srv = result["marker_server_route"]["server"]
        self.assertEqual((srv["requests"], srv["final_errors"], srv["spawn_attempts"]), (fake.chat, 0, 0))
        self.assertGreaterEqual(srv["relay_connects"], 1)
        self.assertIn("RecordingOcrBuilder", srv["checked_users"])
        self.assertEqual(result["marker_server_route"]["network_calls"], 0)
        relay = result["marker_server"]
        self.assertTrue(relay["clean"])
        self.assertEqual(relay["requests"]["chat"], fake.chat)
        self.assertEqual(result["limits"], pages.SERVER_LIMITS + pages.BALANCED_SERVER_LIMITS)   # R10 (MFL033 N5a)
        self.assertEqual(result["components"]["server"]["endpoint"]["identity_sha256"], local().identity_sha256())
        self.assertEqual(result["components"]["server"]["versions"], {"vllm": "0.25.1"})
        self.assertFalse(result["transit"]["left_machine"])
        self.assertIsNone(result["cost"])
        self.assertEqual([p.name for p in Path("/run/user/%d" % os.getuid()).glob("mdf-*")] if Path(
            "/run/user/%d" % os.getuid()).exists() else [], [])

    def test_accurate_with_a_raster_figure_and_an_ocr_layer(self):
        """r4 §6: every page read by the server; the figure of the native page added through the view."""
        layer = {"content": SCAN + b" " + INVISIBLE, "image": scan_of([SCAN_TEXT], scale=1.5)}
        result, out, fake = self.run_route([native(), layer], "accurate", attempt="acc")
        self.assertEqual(result["status"], "complete", result.get("error"))
        recs = result["pages"]["pages"]
        self.assertEqual([(r["class"], r["indra_source"]) for r in recs], [("native", "server"), ("ocr_layer", "server")])
        self.assertEqual(recs[0]["pictures_added"], 1)
        self.assertEqual(fake.chat, 2)
        self.assertIn("](assets/asset-0001.", (out / "document.md").read_text())
        self.assertTrue(result["marker_server_route"]["force_ocr"])
        self.assertEqual(result["limits"], pages.SERVER_LIMITS)      # no local rebuild of the cells: no R10

    def test_a_500_then_success_is_a_warning_and_a_fixed_500_fails(self):
        for mode in ("balanced", "accurate"):
            with self.subTest(mode=mode):
                fake = Fake(respond=respond)
                fake.fail_next = 1
                result, _, _ = self.run_route([scanned()], mode, fake=fake, attempt="retry-" + mode)
                self.assertEqual(result["status"], "complete", result.get("error"))
                self.assertEqual(result["problem_codes"].get("server_retried"), 1)
                self.assertEqual(result["marker_server_route"]["server"]["exceptions"], 1)
                fake = Fake(respond=respond)
                fake.fail_always = True
                result, _, _ = self.run_route([scanned()], mode, fake=fake, attempt="fail-" + mode)
                self.assertEqual(result["status"], "failed")
                self.assertEqual(result["error"]["code"], Code.SERVER_FAILED.value)
                self.assertFalse(result["marker_server"]["clean"])

    def test_a_rented_endpoint_inside_its_session(self):
        events = []
        session = {"arm_id": "a1", "armed_until": __import__("time").time() + 7200}
        fake = Fake(respond=respond, mark="forwarded", key=KEY)
        result, out, _ = self.run_route([scanned()], "balanced", ep=runpod(), fake=fake, attempt="rented",
                                        server={"transit_confirmed": True, "budget_cents": "60", "session": session},
                                        observer=lambda e: events.append(e) and None,
                                        keys={"RUNPOD_API_KEY": Secret(KEY)})
        self.assertEqual(result["status"], "complete", result.get("error"))
        wake = [e for e in events if e.get("step") == "server-wake"]
        self.assertEqual(len(wake), 1)
        cost = result["cost"]
        self.assertTrue(cost["clean"])
        self.assertLessEqual(Decimal(cost["bound_cents"]), Decimal(cost["reserve_cents"]))
        self.assertEqual(wake[0]["reserve"]["reserve_cents"], cost["reserve_cents"])
        self.assertIn("server_cost_bound", result["problem_codes"])
        self.assertEqual(result["options"]["requested"]["server"]["arm_id"], "a1")
        dump = json.dumps(result) + "".join(p.read_text(errors="replace") for p in out.rglob("*") if p.is_file())
        self.assertNotIn(KEY, dump)

    def test_a_server_that_repeats_the_key_reaches_no_published_file(self):
        """S-B1 of the final review of MFL033: the server repeats the key in the body of a 500, a 401 and
        a 200, and in a header of every answer; Surya retries, and neither the child, its log, engine.json
        nor any published file holds the key. The same with every answer an error repeating it."""
        session = {"arm_id": "a1", "armed_until": __import__("time").time() + 7200}
        for name, script, fail in (("reflect-ok", [500, 401, 200], False), ("reflect-fail", [], True)):
            with self.subTest(name):
                fake = Fake(respond=respond, mark="forwarded", key=KEY)
                fake.reflect, fake.script, fake.fail_always = True, list(script), fail
                result, out, _ = self.run_route([scanned()], "balanced", ep=runpod(), fake=fake, attempt=name,
                                                server={"transit_confirmed": True, "budget_cents": "60",
                                                        "session": session}, keys={"RUNPOD_API_KEY": Secret(KEY)})
                files = [p for p in out.rglob("*") if p.is_file()]
                self.assertTrue(any(p.name.endswith(".log") for p in files), [p.name for p in files])
                for where, blob in [("engine result", json.dumps(result).encode())] + [
                        (str(p.relative_to(out)), p.read_bytes()) for p in files]:
                    self.assertNotIn(KEY.encode(), blob, where)
                if fail:
                    self.assertEqual(result["error"]["code"], Code.SERVER_FAILED.value)
                    self.assertGreater(result["marker_server"]["reflected"], 3)
                else:
                    self.assertEqual(result["status"], "complete", result.get("error"))
                    self.assertEqual(result["problem_codes"].get("server_retried"), 1)
                    self.assertEqual(result["marker_server"]["reflected"], 3)
                    self.assertIn(SCAN_TEXT, (out / "document.md").read_text())
                by_status = result["marker_server"]["by_status"]
                self.assertGreaterEqual(by_status.get("500", 0), 1)               # the server's statuses
                if not fail:
                    self.assertEqual((by_status["401"], result["marker_server_route"]["server"]["exceptions"]), (1, 3))


class TestBeforeTheWake(RouteCase):
    """Refusals before any request reaches the server (the fake sees nothing)."""

    def refused(self, code, **kw):
        self.n = getattr(self, "n", 0) + 1
        result, _, fake = self.run_route([scanned()], "balanced", attempt="refused%d" % self.n, **kw)
        self.assertEqual((result["status"], result["error"]["code"]), ("failed", code.value))
        self.assertEqual(fake.seen, [])
        self.assertFalse(result["transit"]["left_machine"])
        return result

    def test_transit_not_confirmed(self):
        ep = servers.endpoint("gpu", {"kind": "user", "url": "https://gpu.example"})
        self.refused(Code.TRANSIT_NOT_CONFIRMED, ep=ep)

    def test_a_provider_without_session_or_budget(self):
        self.refused(Code.SERVER_NOT_ARMED, ep=runpod(), server={"transit_confirmed": True, "budget_cents": "60"})
        near = {"arm_id": "a", "armed_until": __import__("time").time() + 900}
        self.refused(Code.SERVER_NOT_ARMED, ep=runpod(), server={"transit_confirmed": True, "budget_cents": "60",
                                                                  "session": near})
        far = {"arm_id": "a", "armed_until": __import__("time").time() + 7200}
        self.refused(Code.SERVER_CAP, ep=runpod(), server={"transit_confirmed": True, "budget_cents": "20",
                                                            "session": far})

    def test_a_refusal_of_the_reservation_stops_before_the_wake(self):
        from markdown_for_llms.engine.codes import EngineError

        def observer(event):
            if event.get("step") == "server-wake":
                raise EngineError(Code.SERVER_CAP, "the ledger")
        far = {"arm_id": "a", "armed_until": __import__("time").time() + 7200}
        self.refused(Code.SERVER_CAP, ep=runpod(), observer=observer,
                     server={"transit_confirmed": True, "budget_cents": "60", "session": far})


class TestPicturesInto(unittest.TestCase):
    """The view of r4 §6 on a `PageOCRResult` (pure)."""

    def test_objects_kept_figures_added_order_renumbered(self):
        from surya.recognition.schema import BlockOCRResult, PageOCRResult
        title = BlockOCRResult(polygon=[10, 10, 500, 40], label="SectionHeader", reading_order=0, html="<h1>T</h1>")
        body = BlockOCRResult(polygon=[10, 400, 500, 450], label="Text", reading_order=1, html="<p>b</p>")
        pic = BlockOCRResult(polygon=[10, 100, 200, 200], label="Picture", reading_order=2, html="", skipped=True)
        result = PageOCRResult(blocks=[body, pic, title], image_bbox=[0, 0, 600, 800])
        added, contained, merged = marker_child.pictures_into(result, [[300, 100, 400, 300], [150, 150, 250, 250]])
        self.assertEqual((added, contained, merged), (1, 0, 1))
        self.assertIs(result.blocks[0], title)
        self.assertEqual([b.reading_order for b in result.blocks], list(range(len(result.blocks))))
        self.assertEqual(pic.bbox, [10, 100, 250, 250])                      # widened, the same object
        new = [b for b in result.blocks if b not in (title, body, pic)]
        self.assertEqual(len(new), 1)
        self.assertEqual((new[0].label, new[0].skipped, new[0].error, new[0].html), ("Picture", True, False, ""))
        self.assertEqual((body.html, title.html), ("<p>b</p>", "<h1>T</h1>"))
        self.assertLess(result.blocks.index(new[0]), result.blocks.index(body))

    def test_the_gate_counts_a_page_read_by_the_server_as_ocr(self):
        from types import SimpleNamespace
        doc = SimpleNamespace(pages=[SimpleNamespace(page_id=0, children=[])])
        prov = {0: {"indra_source": "server", "note": "", "blocks": 1, "text_blocks": 1, "block_errors": 0,
                    "pictures_skipped_rotated": 0}}
        info = {0: {"class": "ocr_layer", "invisible_chars": 40}}
        self.assertEqual(marker_child.gate(doc, prov, info, server=True), [])
        with self.assertRaises(marker_child.OcrLayerRead):
            marker_child.gate(doc, prov, info)


class TestServerLimits(unittest.TestCase):
    """R10 of MFL033 (plan r2 §6), measured on D04 with the CPU, RunPod and Modal servers: a declared limit of
    `balanced` only (pure)."""

    def test_balanced_declares_the_cells_on_several_lines_and_advises_accurate(self):
        got = pages.server_limits("balanced")
        self.assertEqual(got, pages.SERVER_LIMITS + pages.BALANCED_SERVER_LIMITS)
        self.assertEqual(len(pages.BALANCED_SERVER_LIMITS), 1)
        text = pages.BALANCED_SERVER_LIMITS[0]
        for part in ("balanced", "rebuilds the tables on this machine", "several lines", "as measured on the table of a scientific "
                     "article", "CPU server", "GPU servers", "accurate", "has not been measured"):
            self.assertIn(part, text)

    def test_the_text_the_user_reads_names_no_internal_reference(self):
        """MFL033 N5b (review note of N5a), N6 for every limit of the route: the user reads them; no
        run, no check id."""
        for text in pages.SERVER_LIMITS + pages.BALANCED_SERVER_LIMITS:
            for internal in ("MFL", "R10", "N5a", "N6", "plan r", "D04", "D06", "reconstruct_table_html", "_"):
                self.assertNotIn(internal, text)

    def test_the_route_declares_that_a_gpu_server_may_vary_between_conversions(self):
        """MFL033 N6: on 21/11 D04 in balanced differed in 6 lines between two conversions without the
        cache of torch.compile on the same A10G of Modal (a letter, the case of a word, the cells of a
        table on several lines), and D06 differed in its bytes between RunPod and Modal: a limit of
        the route in both modes, in words the user understands."""
        for mode in ("balanced", "accurate"):
            text = " ".join(pages.server_limits(mode))
            for part in ("GPU server", "two conversions of the same document can differ", "same GPU",
                         "by their content, not by their bytes", "the accurate mode has not been measured"):
                self.assertIn(part, text)

    def test_accurate_keeps_the_limit_of_the_route_alone(self):
        self.assertEqual(pages.server_limits("accurate"), pages.SERVER_LIMITS)

    def test_a_new_list_each_time(self):
        got = pages.server_limits("balanced")
        got.append("x")
        self.assertNotIn("x", pages.SERVER_LIMITS + pages.BALANCED_SERVER_LIMITS)
        self.assertEqual(len(pages.server_limits("balanced")), 3)


if __name__ == "__main__":
    unittest.main()


class TestStallOnTheServerRoute(RouteCase):
    def test_a_child_waiting_on_a_slow_server_is_not_stalled(self):
        """The child is idle while its requests are in flight: that is not a stall (the CPU server
        may take minutes per request); the term of the relay bounds the attempt instead."""
        from markdown_for_llms.engine.core import convert
        fake = Fake(respond=respond, delay=9)
        self.addCleanup(fake.close)
        src = self.root / "in.pdf"
        src.write_bytes(make_pdf([scanned()]))
        result = convert(src, self.root / "out-stall", {"route": "marker-server", "attempt": "stall",
                                                         "server": {"name": "fake", "mode": "balanced"}},
                         {"memory_mib": 4096, "threads": 2, "stall_seconds": 4}, work=self.root / "work",
                         model_root=model_root(), server_endpoint=local(), server_keys={},
                         server_transport=fake.factory())
        self.assertEqual(result["status"], "complete", result.get("error"))
