"""FlashBoot on the RunPod side (MFL045 c3; plan r2 §6 A6, A7, A13; prescriptions P-6 and P-9): the lever in the arm
and its read-back, the return to OFF after the session, the pin `flashboot` alignable, `version` and `isStale` in the
readings, the releases, the quiet intervals of `drivers.Watch`, the zero Z1 with a paused worker, and `A-FB`, the
fault test of phase A with the lever: two zeros, a declared window between them, a resume that answers 503
`mdf_inert`, or a missed resume as a fact. No network: `httpx.MockTransport` for the REST v2 and the balancer, a
fake clock, the log of the session written as the follower writes it."""
import io
import json
from pathlib import Path
import sys
import unittest
from contextlib import redirect_stdout

import httpx

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts" / "servers"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import drivers as dv  # noqa: E402
import providers as pv  # noqa: E402
import runpod_endpoint as rpe  # noqa: E402
from test_drivers import FakeRunPod, InSession, listing, pline, wrow  # noqa: E402

LV_FB = pv.DEFAULT_LEVERS + ",flashboot"


class TestTheLeverOnTheEndpoint(InSession):
    def driver_of(self, fake, *argv):
        return self.driver(fake, *argv)

    def test_the_arm_sets_it_and_reads_it_back(self):
        fake = FakeRunPod()
        rp = pv.RunPod(pv.read_key(self.env, "RUNPOD_API_KEY"), "mdf-surya-fault", transport=httpx.MockTransport(fake))
        info = rp.arm("", False, LV_FB)
        patch = [c for c in fake.calls if c[0] == "PATCH"][-1][2]
        self.assertEqual(patch["flashboot"], "FLASHBOOT")
        self.assertIn("flashboot", patch["env"]["MDF_LEVERS"].split(","))
        self.assertEqual((info["flashboot"], fake.ep["flashboot"]), ("FLASHBOOT", "FLASHBOOT"))
        rp.disarm()
        self.assertEqual(fake.ep["flashboot"], "FLASHBOOT")            # the disarm of N4 is unchanged: workers only
        rp.restore_flashboot()
        last = [c for c in fake.calls if c[0] == "PATCH"][-1][2]
        self.assertEqual(last, {"flashboot": "OFF"})
        self.assertEqual(fake.ep["flashboot"], "OFF")
        info = rp.arm("", False, pv.DEFAULT_LEVERS)                      # without the lever: OFF, said in the PATCH
        self.assertEqual(info["flashboot"], "OFF")
        self.assertNotIn("flashboot", [c for c in fake.calls if c[0] == "PATCH"][-1][2])

    def test_a_read_back_that_differs(self):
        class Sticky(FakeRunPod):
            def __call__(self, request):
                r = FakeRunPod.__call__(self, request)
                self.ep["flashboot"] = "OFF"                               # RunPod ignores the field
                return r
        fake = Sticky()
        rp = pv.RunPod(pv.read_key(self.env, "RUNPOD_API_KEY"), "mdf-surya-fault", transport=httpx.MockTransport(fake))
        with self.assertRaisesRegex(pv.ProviderError, "read back differs"):
            rp.arm("", False, LV_FB)

        class Stuck(FakeRunPod):
            def __call__(self, request):
                r = FakeRunPod.__call__(self, request)
                self.ep["flashboot"] = "FLASHBOOT"
                return r
        rp = pv.RunPod(pv.read_key(self.env, "RUNPOD_API_KEY"), "mdf-surya-fault",
                       transport=httpx.MockTransport(Stuck()))
        with self.assertRaisesRegex(pv.ProviderError, "flashboot is not OFF"):
            rp.restore_flashboot()

    def test_the_pin_is_alignable(self):
        """A7: at rest `flashboot` is OFF, the pin of `create_body` and of `verify-pins`; left on, it is an
        alignable pin: no refusal of the arm, `verify-pins` exit 6, `align` puts it back."""
        self.assertIn("flashboot", pv.PIN_ALIGNABLE)
        self.assertEqual(pv.create_body("x", "v", "EU-RO-1")["flashboot"], "OFF")
        fake = FakeRunPod()
        fake.ep["flashboot"] = "FLASHBOOT"
        code, lines = self.run_quiet(self.driver(fake, "verify-pins", "--name", "mdf-surya-fault").verify)
        self.assertEqual((code, lines[-1]["checks"]["flashboot"], lines[-1]["outside_align"]), (6, False, []))
        code, lines = self.run_quiet(self.driver(fake, "align", "--name", "mdf-surya-fault").align)
        self.assertEqual((code, lines[-1]["ok"]), (0, True))
        self.assertIn("flashboot", lines[-1]["patched"])
        self.assertEqual(fake.ep["flashboot"], "OFF")

    def test_version_and_stale_in_the_readings(self):
        fake = FakeRunPod(workers=[[{"id": "w1", "status": "IDLE", "version": 4, "isStale": False}]])
        fake.ep["version"] = 4
        rp = pv.RunPod(pv.read_key(self.env, "RUNPOD_API_KEY"), "mdf-surya-fault", transport=httpx.MockTransport(fake))
        s = rp.state()
        self.assertEqual((s["flashboot"], s["version"]), ("OFF", 4))
        self.assertEqual((s["workers"][0]["version"], s["workers"][0]["stale"]), (4, False))

    def test_the_releases(self):
        fake = FakeRunPod()
        seen = []

        def releases(request):
            seen.append(str(request.url))
            return httpx.Response(200, json={"releases": [{"version": 4, "createdAt": "2026-11-22T10:00:00Z"}]})
        base = fake.__call__

        def handler(request):
            if request.url.path.endswith("/releases"):
                return releases(request)
            return base(request)
        a = rpe.parser().parse_args(["--env-file", str(self.env), "releases", "--name", "mdf-surya-fault"])
        d = rpe.Driver(a, transport=httpx.MockTransport(handler), clock=self.clock, sleep=self.clock.sleep)
        code, lines = self.run_quiet(d.releases)
        self.assertEqual(code, 0)
        self.assertTrue(seen[0].endswith("/v2/serverless/ep123/releases"))
        self.assertEqual(lines[-1]["releases"]["releases"][0]["version"], 4)

    def test_the_levers_by_provider(self):
        with self.assertRaisesRegex(pv.ProviderError, "flashboot.*runpod"):
            pv.levers_text("flashboot", "modal")

    def test_the_watchdog_fits_the_request_limit_of_runpod(self):
        """F0 of MFL045 (22/11): RunPod refuses a PATCH whose expanded request exceeds 102,400 bytes (HTTP 413), and the
        watchdog travels base64 in the env of the endpoint. The PATCH of the largest arm (the levers, a fault, FlashBoot,
        the CUDA floor) stays under the limit with a margin for the rest of the template; the code of the snapshot of
        Modal is not in it (`snapshot.py`, only in the image of Modal)."""
        env = pv.base_env("crash-after-first", True, LV_FB)
        body = {"workers": pv.workers_block(1, True), "env": env, "gpu": {"minCudaVersion": pv.CUDA_MIN},
                "flashboot": "FLASHBOOT"}
        self.assertEqual(pv.RUNPOD_REQUEST_MAX, 102_400)
        self.assertLessEqual(len(json.dumps(body)), pv.RUNPOD_REQUEST_MAX - pv.RUNPOD_REQUEST_MARGIN)
        self.assertNotIn(b"def snap_boot", pv.watchdog_bytes())


class Reading:
    """`state()` of an endpoint for `drivers.Watch`: the workers by the fake clock."""

    def __init__(self, clock, timeline):
        self.clock, self.timeline = clock, timeline

    def state(self):
        w = self.timeline(self.clock.t)
        return {"armed": True, "busy": bool(w), "workers_max": 1, "summary": {"total": len(w)},
                "workers": [{"id": i, "status": s} for i, s in w]}


class TestQuietIntervals(unittest.TestCase):
    """P-6: `Watch` keeps the closed intervals of quiet and the open one; the declared window closes the open one
    after a final reading; RUNNING in the open interval is a violation, after the window is not, two RUNNING at
    once always are."""

    def test_the_window_and_the_intervals(self):
        from test_drivers import Clock
        clock = Clock(1000.0)
        status = {"w1": "IDLE"}
        w = dv.Watch(Reading(clock, lambda t: [("w1", status["w1"])]), clock, clock.sleep, extra_ok=True)
        w.until_zero(30, settled=lambda listed: None)
        self.assertEqual(w.zero_t, 1000.0)
        clock.t += 60
        w.open_window()
        self.assertIsNone(w.zero_t)
        self.assertEqual(w.quiet, [(1000.0, 1060.0)])
        status["w1"] = "RUNNING"
        w.read()                                                        # inside the window: admitted
        status["w1"] = "IDLE"
        clock.t += 100
        w.until_zero(30, settled=lambda listed: None)
        status["w1"] = "RUNNING"
        with self.assertRaises(dv.Violation) as ctx:
            w.read()
        self.assertEqual(ctx.exception.guarantee, "G1")

    def test_the_final_reading_before_the_window(self):
        from test_drivers import Clock
        clock = Clock(1000.0)
        status = {"w1": "IDLE"}
        w = dv.Watch(Reading(clock, lambda t: [("w1", status["w1"])]), clock, clock.sleep, extra_ok=True)
        w.until_zero(30, settled=lambda listed: None)
        status["w1"] = "RUNNING"
        clock.t += 30
        with self.assertRaises(dv.Violation):
            w.open_window()                                             # the open interval read once more
        self.assertIsNotNone(w.zero_t)

    def test_two_running_at_once_even_in_the_window(self):
        from test_drivers import Clock
        clock = Clock(1000.0)
        w = dv.Watch(Reading(clock, lambda t: [("w1", "RUNNING"), ("w2", "RUNNING")]), clock, clock.sleep,
                     extra_ok=True)
        with self.assertRaises(dv.Violation):
            w.read()


class LogScript:
    """The log of a RunPod window: rows (append at s since the arm, the line), appended when the clock passes."""

    def __init__(self, path, clock, t_arm, rows):
        self.path, self.clock, self.t_arm = path, clock, t_arm
        self.rows = sorted(rows, key=lambda r: r[0])

    def flush(self):
        with open(self.path, "a") as fh:
            while self.rows and self.rows[0][0] <= self.clock.t - self.t_arm:
                fh.write(self.rows.pop(0)[1])


def at(since, line):
    return (since, line)


class TimedRunPod(FakeRunPod):
    """The REST v2 and the balancer with the workers by the fake clock: `timeline(s)` the (id, status) since the
    arm; every call first appends the lines of the log due by then."""

    def __init__(self, clock, t_arm, timeline, script):
        super().__init__()
        self.clock, self.t_arm, self.timeline, self.script = clock, t_arm, timeline, script

    def __call__(self, request):
        self.script.flush()
        if request.url.path == "/v2/serverless/ep123/workers":
            w = [{"id": i, "status": s, "gpuTypeId": "NVIDIA RTX A5000", "dataCenterId": "EU-RO-1"}
                 for i, s in self.timeline(self.clock.t - self.t_arm)]
            return httpx.Response(200, json={"workers": w, "summary": {"total": len(w)}})
        return FakeRunPod.__call__(self, request)


def spans(*rows):
    """(worker, from s, to s or None, status) since the arm; for each worker the last row that covers a moment."""
    def timeline(s):
        out = {}
        for w, a, b, st in rows:
            if a <= s and (b is None or s < b):
                out[w] = st
        return sorted((w, st) for w, st in out.items() if st != "GONE")     # GONE: no longer listed
    return timeline


class TestAFB(InSession):
    """`fault-test --phase A-FB` on `mdf-surya-fault` with the lever (plan r2 §6, A6; P-6): as phase A up to the zero
    Z1 (a worker paused by FlashBoot, IDLE in the current reading and not RUNNING in the log, is the zero:
    `paused_listed`) and 60 s; a declared window; a request that must get 503 `mdf_inert` from the same container
    (else `resume_missed`, a fact); the second zero within A_ZERO_S of its answer (READY_WITHIN_S + A_ZERO_S after
    a missed resume); 60 s without RUNNING. The workers and the log after the request follow the moment it came."""

    def scenario(self, on_request, extra_rows=(), levers=LV_FB):
        from markdown_for_llms.bench import serverspend
        from markdown_for_llms.bench import spend as sp
        self.clock.t = 1_800_000_000.0
        t_arm = self.clock.t - 100.0
        self.ledger = self.dir / "spend-afb.jsonl"
        with sp.Ledger.file(self.ledger).locked() as locked:
            arm_id = serverspend.arm(locked, self.ledger, "runpod", "mdf-surya-fault", "https://ep123.api.runpod.ai",
                                     "45", t_arm + 1942, "200", pv.RATES["runpod"], t_arm,
                                     extra={"levers": levers})["arm_id"]
        rows = [at(20.0, listing(20.0, "w1", "RUNNING", ["w1"])),
                at(20.5, pline(20.5, "w1", "mdf_start", provider="runpod", boot_id="a" * 16)),
                at(200.0, listing(200.0, "w1", "IDLE", []))] + list(extra_rows)
        script = LogScript(self.log, self.clock, t_arm, rows)
        dyn = [("w1", 20, 200, "RUNNING"), ("w1", 200, None, "IDLE")]
        fake = TimedRunPod(self.clock, t_arm, lambda s: spans(*dyn)(s), script)
        chats = []

        def models():
            return (200, {"data": [{"id": dv.MODEL}]}, "forwarded")

        def chat():
            chats.append(self.clock.t - t_arm)
            n = len(chats)
            if n == 1:
                return (200, {"choices": []}, "forwarded")
            if n in (2, 3):
                return (503, {"error": "mdf_tripped"}, "error")
            answer, more_spans, more_rows = on_request(chats[-1])
            dyn.extend(more_spans)
            script.rows = sorted(script.rows + list(more_rows), key=lambda r: r[0])
            return answer
        fake.lb = {("GET", "/v1/models"): models, ("POST", "/v1/chat/completions"): chat}
        d = self.in_session(fake, ["fault-test", "--name", "mdf-surya-fault", "--phase", "A-FB"], arm_id)
        code, lines = self.run_quiet(d.fault_test)
        self.chats = chats
        return code, lines

    @staticmethod
    def resumed(s, again=None):
        """The paused worker resumes for the request: RUNNING while it answers and its idle timeout, then IDLE."""
        more = [("w1", s - 0.5, s + 60, "RUNNING"), ("w1", s + 60, None, "IDLE")]
        rows = [at(s + 1, listing(s + 1, "w1", "RUNNING", ["w1"])),
                at(s + 1.5, pline(s + 1.5, "w1", "mdf_pause", boot_id="a" * 16, seconds=70.0, applied=True)),
                at(s + 60, listing(s + 60, "w1", "IDLE", []))]
        if again is not None:
            more.append(("w1", s + again, None, "RUNNING"))
            rows.append(at(s + again, listing(s + again, "w1", "RUNNING", ["w1"])))
        return (503, {"error": "mdf_inert"}, "error"), more, rows

    def test_a_resume_that_answers_inert_passes(self):
        code, lines = self.scenario(self.resumed)
        res = lines[-1]
        self.assertEqual(code, 0, res.get("violation"))
        self.assertTrue(res["pass"])
        self.assertTrue(res["resumed_inert"])
        self.assertIn("paused_listed", [f["fact"] for f in res["facts"]])
        self.assertEqual(len(res["quiet"]), 1)
        self.assertLessEqual(res["second_zero_since_answer_s"], rpe.A_ZERO_S)
        self.assertGreaterEqual(res["second_zero_since_answer_s"], 60.0)

    def test_running_after_the_second_zero_breaks_g1(self):
        code, lines = self.scenario(lambda s: self.resumed(s, again=100.0))
        self.assertEqual((code, lines[-1]["guarantee"]), (6, "G1"))
        self.assertIn("RUNNING after the zero", lines[-1]["violation"])

    def test_running_in_the_closed_interval_read_late_breaks_g1(self):
        """A listing RUNNING at +230 s (inside the first interval of quiet, which runs from about +200 s to the
        window) reaches the file only at +300 s, after the window: the closed interval still judges it."""
        late = [at(300.0, listing(230.0, "w1", "RUNNING", ["w1"])), at(300.1, listing(231.0, "w1", "IDLE", []))]
        code, lines = self.scenario(self.resumed, extra_rows=late)
        self.assertEqual((code, lines[-1]["guarantee"]), (6, "G1"))
        self.assertIn("quiet", lines[-1]["violation"])

    def test_two_running_at_once_in_the_window_break_g1(self):
        def both(s):
            answer, more, rows = self.resumed(s)
            more.append(("w2", s, s + 20, "RUNNING"))
            rows.append(at(s + 2, listing(s + 2, "w2", "RUNNING", ["w1", "w2"])))
            return answer, more, rows
        code, lines = self.scenario(both)
        self.assertEqual((code, lines[-1]["guarantee"]), (6, "G1"))

    def test_a_missed_resume_is_a_fact_and_its_second_zero_waits_longer(self):
        """P-6: an answer other than 503 `mdf_inert` (the paused worker gone, a new one starting cold) enters the
        branch `resume_missed` (criterion 3 of F4 false), not a violation."""
        def missed(s):
            more = [("w1", s, None, "GONE"), ("w2", s + 1, s + 20, "INITIALIZING"), ("w2", s + 20, s + 400, "RUNNING"),
                    ("w2", s + 400, None, "IDLE")]
            rows = [at(s + 1, listing(s + 1, "w1", "GONE", [])), at(s + 20, listing(s + 20, "w2", "RUNNING", ["w2"])),
                    at(s + 400, listing(s + 400, "w2", "IDLE", []))]
            return httpx.ReadTimeout("the balancer gave up"), more, rows
        code, lines = self.scenario(missed)
        res = lines[-1]
        self.assertEqual(code, 0, res.get("violation"))
        self.assertFalse(res["resumed_inert"])
        self.assertIn("resume_missed", [f["fact"] for f in res["facts"]])
        self.assertGreater(res["second_zero_since_answer_s"], rpe.A_ZERO_S)

    def test_a_fb_needs_the_lever(self):
        code, lines = self.scenario(self.resumed, levers=pv.DEFAULT_LEVERS)
        self.assertEqual((code, lines[-1]["code"]), (2, "invalid_setting"))


class TestSettled(InSession):
    def test_a_paused_worker_is_the_zero_only_with_the_lever(self):
        from markdown_for_llms.bench import serverspend
        from markdown_for_llms.bench import spend as sp
        for levers, settled in ((LV_FB, True), (pv.DEFAULT_LEVERS, False)):
            with self.subTest(levers=levers):
                self.log.write_text(listing(20.0, "w1", "RUNNING", ["w1"]) + listing(200.0, "w1", "IDLE", []))
                ledger = self.dir / ("spend-%s.jsonl" % settled)
                with sp.Ledger.file(ledger).locked() as locked:
                    arm_id = serverspend.arm(locked, ledger, "runpod", "mdf-surya-fault", "https://ep123.api.runpod.ai",
                                             "45", self.clock.t + 1000, "200", pv.RATES["runpod"], self.clock.t - 300,
                                             extra={"levers": levers})["arm_id"]
                self.ledger = ledger
                d = self.in_session(FakeRunPod(), ["fault-test", "--name", "mdf-surya-fault", "--phase", "A"], arm_id)
                d.watch(s1=True)
                with redirect_stdout(io.StringIO()):
                    why = d.settled({"w1": "IDLE"})
                    why_throttled = d.settled({"w1": "THROTTLED"})
                    why_running_listed = d.settled({"w1": "INITIALIZING"})
                self.assertEqual(why is None, settled, why)
                self.assertEqual(why_throttled is None, settled)
                self.assertIsNotNone(why_running_listed)
                self.assertEqual("paused_listed" in [f["fact"] for f in d.facts], settled)


if __name__ == "__main__":
    unittest.main()
