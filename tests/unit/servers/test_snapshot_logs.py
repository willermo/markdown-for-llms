"""The integrated proof of A9 and P-7 (MFL045 c2; plan r2 §7, App. D; review r2 Sol S-C4): the real lines of
the watchdog (`snap_boot`, `snap_wake`, the fallback) with a fake vLLM (`snapshot_driver.py`, a subprocess) become
the file of the follower as `ModalFollower` writes it, stdout and stderr delivered together or apart; then
`worker_logs.StartRules`, `worker_logs.Lives`, `start_series.phases` and `modal_tasks.Tasks.ready` read it.

- An ERROR of the adopted vLLM during the wake is held until the explicit outcome: a declared fallback lets it
  go (whether it reached the file before `mdf_restore` or after `mdf_fallback`), readiness without a fallback
  makes it the stop, and no outcome within WAKE_HOLD_S makes it the stop too;
- the vLLM of the fallback is never exempt, nor a container with the same pid;
- the warm-up before the snapshot is internal: it is no answer of vLLM through the proxy and no inference."""
from datetime import datetime, timedelta, timezone
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace

import httpx

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts" / "servers"))
import drivers as dv  # noqa: E402
import modal_tasks as mt  # noqa: E402
import providers as pv  # noqa: E402
import start_series as ss  # noqa: E402
import worker_logs as wl  # noqa: E402

from markdown_for_llms.bench import serverspend, spend  # noqa: E402

DRIVER = Path(__file__).resolve().parent / "snapshot_driver.py"
URL = "https://ws--mdf-surya-snap-serve-web.modal.run"
TOKEN_ID, TOKEN_SECRET = "wk-test-token-id", "ws-test-token-secret"
CREATOR, RESTORED, OTHER = "ta-01CREATE00000000000000000", "ta-01RESTORE0000000000000000", "ta-01OTHER000000000000000000"


def cli_time(t):
    """A time as `modal app logs --timestamps` prints it: whole seconds, an offset (MFL033, 20/11)."""
    return datetime.fromtimestamp(int(t), timezone(timedelta(hours=1))).isoformat(sep=" ")


def run_driver(scenario):
    tmp = Path(tempfile.mkdtemp())
    try:
        out = tmp / "out.json"
        r = subprocess.run([sys.executable, "-I", "-B", str(DRIVER), scenario, str(out)], capture_output=True,
                           text=True, timeout=180)
        if r.returncode or not out.exists():
            raise AssertionError(r.stderr[-3000:])
        res = json.loads(out.read_text())
    finally:
        shutil.rmtree(tmp, True)
    res["events"] = [json.loads(x) for x in r.stderr.splitlines() if x.startswith("{")]
    return res


class Window:
    """The follower's file of a window from the lines of the driver: the arm 10 s before the restore; the
    lines before the restore are those of the creator (another window: left out unless `creator`); the
    container listed every 5 s; stdout and stderr each with its own lag."""

    def __init__(self, res, path, lag_out=0.0, lag_err=0.0, creator=False, drop=(), cut_after=None, extra=(),
                 events_before=None):
        self.t_arm = res["t_restore"] - 10.0
        rows = []
        for ev in res["events"]:
            if ev.get("event") in drop or (cut_after is not None and ev["t"] > cut_after) \
                    or (events_before is not None and ev["t"] >= events_before):
                continue
            cid = RESTORED if ev["t"] >= res["t_restore"] else CREATOR
            rows.append((ev["t"] + lag_err, ev["t"], cid, json.dumps(ev, sort_keys=True)))
        for t, text, pid in res["lines"]:
            if (cut_after is not None and t > cut_after) or (events_before is not None and pid not in res["pids"]):
                continue
            cid = RESTORED if t >= res["t_restore"] else CREATOR
            rows.append((t + lag_out, t, cid, text))
        rows = [r for r in rows if creator or r[2] == RESTORED]
        rows += list(extra)
        end = max(r[0] for r in rows) + 320.0
        t = (res["t_restore"] if not creator else res["t0"]) - 1.0
        first = True
        while t < end:
            ev = {"event": "worker", "status": "RUNNING", "start": "x"} if not first else \
                {"event": "worker", "status": "PENDING", "start": None}
            rows.append((t, None, RESTORED if not creator else CREATOR, ev))
            first = False
            t += 5.0
        self.rows = sorted(rows, key=lambda r: r[0])
        self.path = Path(path)
        follow = json.dumps({"event": "follow", "provider": "modal", "t_arm": wl.iso(self.t_arm, ms=True)})
        self.path.write_text("%s\t+%.1f\t-\tsession\t\t%s\n" % (wl.iso(self.t_arm - 16, ms=True), -16.0, follow))
        self.at = 0

    def line(self, row):
        recv, t, cid, text = row
        if isinstance(text, dict):
            return "%s\t+%.1f\t%s\tsession\t\t%s\n" % (wl.iso(recv, ms=True), recv - self.t_arm, cid, json.dumps(text))
        return "%s\t+%.1f\t%s\tcontainer\t%s\t%s\n" % (wl.iso(recv, ms=True), recv - self.t_arm, cid, cli_time(t), text)

    def until(self, since):
        """Append the rows received by `since` s from the arm."""
        with open(self.path, "a") as fh:
            while self.at < len(self.rows) and self.rows[self.at][0] - self.t_arm <= since:
                fh.write(self.line(self.rows[self.at]))
                self.at += 1

    def each(self):
        """Append the rows one at a time: the second of each."""
        while self.at < len(self.rows):
            with open(self.path, "a") as fh:
                fh.write(self.line(self.rows[self.at]))
            self.at += 1
            yield self.rows[self.at - 1][0] - self.t_arm


RUNS = {}


class Scenarios(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not RUNS:                                    # once per module: each run starts real processes
            RUNS.update({s: run_driver(s) for s in ("awake", "fallback", "definitive")})
        cls.runs = RUNS

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def window(self, scenario, **kw):
        return Window(self.runs[scenario], self.tmp / ("log-%d" % len(list(self.tmp.iterdir()))), **kw)

    def first_stop(self, win, rules):
        for since in win.each():
            reason = rules.feed(win.path)
            if reason:
                return reason, since
        return None, None


class TestTheDriver(Scenarios):
    def test_the_three_scenarios_ran(self):
        self.assertTrue(self.runs["awake"]["ready"])
        self.assertTrue(self.runs["fallback"]["ready"])
        self.assertFalse(self.runs["definitive"]["ready"])
        for name, res in self.runs.items():
            names = [e["event"] for e in res["events"]]
            with self.subTest(name):
                self.assertEqual(names[names.index("mdf_snapshot")], "mdf_snapshot")
                self.assertLess(names.index("mdf_snapshot"), names.index("mdf_restore"))
                self.assertEqual([e["kind"] for e in res["events"] if e["event"] == "mdf_restore"], ["restore"])
                errors = [x for x in res["lines"] if " ERROR " in x[1]]
                self.assertEqual(bool(errors), name != "awake")
                if name != "awake":
                    self.assertIn(errors[0][2], res["pids"])          # the ERROR of the adopted group
                    self.assertIn("mdf_fallback", names)


class TestStartRules(Scenarios):
    """P-7: the errors of the adopted vLLM held during the wake, until `mdf_fallback` or readiness."""

    def test_a_declared_fallback_lets_the_error_of_the_adopted_pid_go(self):
        for lags in ((0.0, 0.0), (0.0, 25.0), (25.0, 0.0)):
            with self.subTest(lag_out=lags[0], lag_err=lags[1]):
                win = self.window("fallback", lag_out=lags[0], lag_err=lags[1])
                rules = wl.StartRules(snapshot=True)
                reason, _ = self.first_stop(win, rules)
                self.assertIsNone(reason)
                self.assertIsNone(rules.at_readiness())
                self.assertEqual(len(rules.exempt), 1)
                self.assertIn("wake_up failed", rules.exempt[0]["line"])

    def test_without_the_snapshot_the_same_error_stops_at_once(self):
        win = self.window("fallback")
        reason, _ = self.first_stop(win, wl.StartRules())
        self.assertIn("an error of vLLM", reason)
        self.assertIn("wake_up failed", reason)

    def test_the_vllm_of_the_fallback_is_never_exempt(self):
        for lags in ((0.0, 0.0), (0.0, 25.0), (25.0, 0.0)):
            with self.subTest(lag_out=lags[0], lag_err=lags[1]):
                win = self.window("definitive", lag_out=lags[0], lag_err=lags[1])
                reason, _ = self.first_stop(win, wl.StartRules(snapshot=True))
                self.assertIsNotNone(reason)
                self.assertTrue("the engine failed to start" in reason or "mdf_vllm_exit" in reason, reason)

    def test_no_outcome_of_the_wake_stops_after_the_hold(self):
        res = self.runs["fallback"]
        t_fb = [e["t"] for e in res["events"] if e["event"] == "mdf_fallback"][0]
        win = self.window("fallback", events_before=t_fb)               # no fallback, no vLLM of a fallback
        rules = wl.StartRules(snapshot=True)
        reason, since = self.first_stop(win, rules)
        self.assertIn("no outcome of the wake", reason)
        restore = [e["t"] for e in res["events"] if e["event"] == "mdf_restore"][0] - win.t_arm
        self.assertGreaterEqual(since - restore, wl.WAKE_HOLD_S)
        self.assertLess(since - restore, wl.WAKE_HOLD_S + 6)

    def test_readiness_without_a_fallback_makes_the_held_error_the_stop(self):
        win = self.window("fallback", drop=("mdf_fallback",))
        rules = wl.StartRules(snapshot=True)
        res = self.runs["fallback"]
        t_fb = [e["t"] for e in res["events"] if e["event"] == "mdf_fallback"][0]
        win.until(t_fb - win.t_arm)                      # up to the moment of the fallback, dropped here
        self.assertIsNone(rules.feed(win.path))
        self.assertIn("wake_up failed", rules.at_readiness())

    def test_another_container_with_the_same_pid(self):
        res = self.runs["fallback"]
        t, text, _ = [x for x in res["lines"] if " ERROR " in x[1]][0]
        win = self.window("fallback", extra=[(t + 1.0, t, OTHER, text)])
        reason, _ = self.first_stop(win, wl.StartRules(snapshot=True))
        self.assertIn("wake_up failed", reason)


class TestLivesAndPhases(Scenarios):
    def test_the_warm_up_is_internal(self):
        """The two chats and the probes of `snap_boot` are vLLM's own access lines before the snapshot: no answer
        through the proxy, no inference, so a container replaced then is judged by rule (a)."""
        res = self.runs["awake"]
        t_shot = [e["t"] for e in res["events"] if e["event"] == "mdf_snapshot"][0]
        for lags in ((0.0, 0.0), (0.0, 20.0)):
            with self.subTest(lag_err=lags[1]):
                win = self.window("awake", creator=True, cut_after=t_shot + 0.0001, lag_out=lags[0], lag_err=lags[1])
                win.until(10_000)
                lv = wl.Lives().feed(win.path)
                life = lv.by[CREATOR]
                self.assertEqual((life["vllm"], life["inferences"]), (None, 0))
                self.assertGreaterEqual(life["internal"], 2 + 3)             # the chats, the reset, sleep, is_sleeping
                self.assertTrue(life["snapshot"]["ok"])

    def test_after_the_restore_the_lines_count_as_before(self):
        win = self.window("awake")
        win.until(10_000)
        life = wl.Lives().feed(win.path).by[RESTORED]
        self.assertEqual(life["restore"]["kind"], "restore")
        self.assertIsNotNone(life["vllm"])
        self.assertEqual(life["internal"], 0)
        self.assertTrue(life["wake"]["ok"])

    def test_the_phases_of_a_restore_with_its_fallback(self):
        win = self.window("fallback")
        win.until(10_000)
        (start,) = ss.phases(win.path, win.t_arm)
        self.assertEqual(start["snapshot"]["kind"], "restore")
        self.assertIn("wake_up 500", start["snapshot"]["fallback"])
        self.assertIsNotNone(start["ready_s"])
        win = self.window("awake")
        win.until(10_000)
        (start,) = ss.phases(win.path, win.t_arm)
        self.assertEqual((start["snapshot"]["kind"], start["snapshot"].get("fallback")), ("restore", None))
        self.assertTrue(start["snapshot"]["wake"]["ok"])


class Modal:
    """The app as `providers.Modal` reads it: the one container from the restore on."""

    def __init__(self, win, clock):
        self.win, self.clock = win, clock

    def state(self):
        self.win.until(self.clock.t - self.win.t_arm)
        return {"armed": True, "busy": True, "apps": [], "containers": [RESTORED],
                "workers": [{"id": RESTORED, "status": "RUNNING", "start": None}]}

    def system_logs(self, since):
        return []


class Clock:
    def __init__(self, t):
        self.t = t

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.t += s


class TestReadiness(Scenarios):
    """`modal_tasks.Tasks.ready` on the file of the window, read as it grows: the fallback declared and readiness
    reached; the definitive failure stopped with its reason."""

    def ready(self, scenario):
        res = self.runs[scenario]
        win = self.window(scenario)
        clock = Clock(win.t_arm + 5.0)
        env = self.tmp / ".env"
        env.write_text("MODAL_PROXY_TOKEN_ID=%s\nMODAL_PROXY_TOKEN_SECRET=%s\n" % (TOKEN_ID, TOKEN_SECRET))
        env.chmod(0o600)
        sv = self.tmp / "servers.json"
        sv.write_text(json.dumps({"default": "modal-snap", "servers": {"modal-snap": {
            "kind": "modal", "url": URL, "rate_usd_per_s": "0.00036772", "gpu": "L4,A10G"}}}))
        ledger = self.tmp / "spend.jsonl"
        with spend.Ledger.file(ledger).locked() as locked:
            arm_id = serverspend.arm(locked, ledger, "modal", "mdf-surya-snap", URL, "100", win.t_arm + 2267, "500",
                                     pv.RATES["modal"], win.t_arm, extra={"levers": "snapshot," + pv.DEFAULT_LEVERS})[
                "arm_id"]
        t_ready = ([e["t"] for e in res["events"] if e["event"] == "mdf_ready"] or [None])[0]

        def models(request):
            win.until(clock.t - win.t_arm)
            if t_ready is not None and clock.t >= t_ready:
                return httpx.Response(200, json={"data": [{"id": dv.MODEL}]}, headers={"x-mdf-proxy": "forwarded"})
            return httpx.Response(503, json={"error": "mdf_waking"}, headers={"x-mdf-proxy": "error"})
        a = mt.parser().parse_args(["--env-file", str(env), "--server", "modal-snap", "--servers", str(sv),
                                    "fault-test", "--phase", "A", "--app", "mdf-surya-snap", "--spend-ledger",
                                    str(ledger)])
        t = mt.Tasks(a, modal=Modal(win, clock), transport=httpx.MockTransport(models), clock=clock,
                     sleep=clock.sleep, environ={serverspend.SESSION_ENV: arm_id, wl.LOG_ENV: str(win.path)})
        out = io.StringIO()
        with redirect_stdout(out):
            r = t.ready(t.inference())
        return r, t

    def test_a_fallback_reaches_readiness_and_is_declared(self):
        r, t = self.ready("fallback")
        self.assertTrue(r["ready"], r.get("stopped"))
        self.assertEqual(r["snapshot"]["kind"], "restore")
        self.assertIn("wake_up 500", r["snapshot"]["fallback"])
        self.assertTrue(t.snapshot)

    def test_a_definitive_failure_stops_with_its_reason(self):
        r, _ = self.ready("definitive")
        self.assertFalse(r["ready"])
        self.assertTrue("the engine failed to start" in r["stopped"] or "mdf_vllm_exit" in r["stopped"], r["stopped"])

    def test_a_wake_that_succeeds(self):
        r, _ = self.ready("awake")
        self.assertTrue(r["ready"], r.get("stopped"))
        self.assertEqual((r["snapshot"]["kind"], r["snapshot"].get("fallback")), ("restore", None))


class TestTheHoldOnTheClock(unittest.TestCase):
    """MFL045 c8 (S-C1 of the review of the diff; P-7, "a limited duration"): the real `ModalFollower` writes a
    container only when its listing changes, so a container listed the same while its vLLM is silent adds no line to
    the file. The holds of `StartRules` (an error of the adopted vLLM, WAKE_HOLD_S from its restore; a line waiting for
    its restore, PENDING_S) run on the clock they are given, from the arm of the window, and expire without new lines;
    `modal_tasks.Tasks.ready` gives them its clock and the arm of its row."""

    ERROR = "(EngineCore pid=42) ERROR 11-23 10:00:00 [core.py:1] wake_up failed: CUDA error"

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.clock = Clock(1_800_000_000.0)
        self.t_arm = self.clock()
        self.path = self.tmp / "worker-log.partial"
        listing = [{"id": RESTORED, "status": "RUNNING", "start": "2026-11-23 10:00:00+01:00"}]
        self.follower = wl.ModalFollower(SimpleNamespace(containers=lambda: listing), self.path, self.t_arm,
                                         clock=self.clock)
        self.addCleanup(self.follower.fh.close)
        self.follower.list_once()

    def restore(self):
        self.follower.write(RESTORED, "container", cli_time(self.clock()), json.dumps(
            {"event": "mdf_restore", "kind": "restore", "t": self.clock(), "pids": [42]}))

    def error(self):
        self.follower.write(RESTORED, "container", cli_time(self.clock()), self.ERROR)

    def silent(self, seconds):
        """`seconds` of silence: the follower lists the same container every 5 s and writes nothing."""
        size = self.path.stat().st_size
        end = self.clock() + seconds
        while self.clock() < end:
            self.clock.sleep(min(wl.MODAL_LIST_EVERY_S, end - self.clock()))
            self.follower.list_once()
        self.assertEqual(self.path.stat().st_size, size)

    def test_a_held_error_expires_without_new_lines(self):
        """The case of the review: 192 s of silence after an error of the adopted vLLM during the wake."""
        self.clock.sleep(1)
        self.restore()
        self.clock.sleep(1)
        self.error()
        rules = wl.StartRules(snapshot=True, clock=self.clock, t_arm=self.t_arm)
        self.assertIsNone(rules.feed(self.path))
        self.assertEqual(len(rules.gens[RESTORED]["held"]), 1)
        self.silent(wl.WAKE_HOLD_S - 2)                    # 179 s from the restore: still held
        self.assertIsNone(rules.feed(self.path))
        self.silent(12)                                     # 192 s from the arm, 191 s from the restore
        self.assertEqual(self.clock() - self.t_arm, 192.0)
        reason = rules.feed(self.path)
        self.assertIsNotNone(reason)
        self.assertIn("no outcome of the wake of %s within 180 s of its restore" % wl.short(RESTORED), reason)
        self.assertIn("wake_up failed", reason)

    def test_a_line_waiting_for_its_restore_expires_without_new_lines(self):
        self.clock.sleep(1)
        self.error()                                        # its `mdf_restore` never comes
        rules = wl.StartRules(snapshot=True, clock=self.clock, t_arm=self.t_arm)
        self.assertIsNone(rules.feed(self.path))
        self.silent(wl.PENDING_S - 1)
        self.assertIsNone(rules.feed(self.path))
        self.silent(2)
        self.assertEqual(rules.feed(self.path), "an error of vLLM: " + self.ERROR)

    def test_an_unreadable_file_does_not_stop_the_clock(self):
        self.clock.sleep(1)
        self.restore()
        self.clock.sleep(1)
        self.error()
        rules = wl.StartRules(snapshot=True, clock=self.clock, t_arm=self.t_arm)
        self.assertIsNone(rules.feed(self.path))
        self.clock.sleep(wl.WAKE_HOLD_S + 10)
        self.assertIn("no outcome of the wake", rules.feed(self.path.with_name("missing")))

    def test_without_the_snapshot_the_clock_changes_nothing(self):
        rules = wl.StartRules(clock=self.clock, t_arm=self.t_arm)
        self.clock.sleep(1)
        self.follower.write(RESTORED, "container", cli_time(self.clock()), "INFO 11-23 10:00:00 [api_server.py:1] ok")
        self.assertIsNone(rules.feed(self.path))
        self.silent(3600)
        self.assertIsNone(rules.feed(self.path))

    def test_the_readiness_wait_stops_with_its_reason(self):
        """`Tasks.ready` on the file of the real follower: the restore and an error of the adopted vLLM, then the
        container listed the same and the proxy answering 503 `mdf_waking`: the wait stops after WAKE_HOLD_S with the
        reason of the hold, not at the end of READY_WITHIN_S."""
        env = self.tmp / ".env"
        env.write_text("MODAL_PROXY_TOKEN_ID=%s\nMODAL_PROXY_TOKEN_SECRET=%s\n" % (TOKEN_ID, TOKEN_SECRET))
        env.chmod(0o600)
        sv = self.tmp / "servers.json"
        sv.write_text(json.dumps({"default": "modal-snap", "servers": {"modal-snap": {
            "kind": "modal", "url": URL, "rate_usd_per_s": "0.00036772", "gpu": "L4,A10G"}}}))
        ledger = self.tmp / "spend.jsonl"
        with spend.Ledger.file(ledger).locked() as locked:
            arm_id = serverspend.arm(locked, ledger, "modal", "mdf-surya-snap", URL, "100", self.t_arm + 2267, "500",
                                     pv.RATES["modal"], self.t_arm, extra={"levers": "snapshot," + pv.DEFAULT_LEVERS})[
                "arm_id"]
        follower, clock = self.follower, self.clock

        class Modal:
            def state(self):
                follower.list_once()
                return {"armed": True, "busy": True, "apps": [], "containers": [RESTORED],
                        "workers": [{"id": RESTORED, "status": "RUNNING", "start": None}]}

            def system_logs(self, since):
                return []

        self.clock.sleep(10)
        self.restore()
        t_restore = self.clock()
        self.clock.sleep(1)
        self.error()

        def models(request):
            follower.list_once()
            return httpx.Response(503, json={"error": "mdf_waking"}, headers={"x-mdf-proxy": "error"})
        a = mt.parser().parse_args(["--env-file", str(env), "--server", "modal-snap", "--servers", str(sv),
                                    "fault-test", "--phase", "A", "--app", "mdf-surya-snap", "--spend-ledger",
                                    str(ledger)])
        t = mt.Tasks(a, modal=Modal(), transport=httpx.MockTransport(models), clock=clock, sleep=clock.sleep,
                     environ={serverspend.SESSION_ENV: arm_id, wl.LOG_ENV: str(self.path)})
        size = self.path.stat().st_size
        with redirect_stdout(io.StringIO()):
            r = t.ready(t.inference())
        self.assertEqual(self.path.stat().st_size, size)    # no line came after the error
        self.assertFalse(r["ready"])
        self.assertIn("no outcome of the wake of %s" % wl.short(RESTORED), r.get("stopped") or "")
        self.assertGreaterEqual(clock() - t_restore, wl.WAKE_HOLD_S)
        self.assertLess(clock() - t_restore, wl.WAKE_HOLD_S + 10)


if __name__ == "__main__":
    unittest.main()
