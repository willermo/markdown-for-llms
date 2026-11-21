"""The drivers of N4 replayed on the windows that were recorded (MFL033 c7.20, prompt of the closure of
N4 on RunPod, §6.1). No network: the readings the driver recorded (or the states of the follower's
listings, when the driver left none) at a fake clock, the worker log of the session written as the
clock reaches its lines, and an edge that answers as the provider did (readiness at the recorded
time, the first chat, then the trip, the cut or the crash). The window ends at its disarm: the
reading at the end is the last one, the next one stops the driver, as the disarm of the session did.

The fixtures (`tests/fixtures/servers/replay-*.json`) come from `temp/mfl033-gpu/n4/replay/
make_fixtures.py`, an untracked tool of the run: provenance and the changes made (ids, digests and
the workspace name replaced, vLLM's info lines left out) are in each file."""
import copy
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
import threading
import unittest
from contextlib import redirect_stdout

import httpx

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts" / "servers"))
import drivers as dv  # noqa: E402
import modal_tasks as mt  # noqa: E402
import providers as pv  # noqa: E402
import runpod_endpoint as rpe  # noqa: E402
import worker_logs as wl  # noqa: E402

from markdown_for_llms.bench import serverspend, spend  # noqa: E402

FIXTURES = ROOT / "tests" / "fixtures" / "servers"
KEY = "rp-replay-secret"


def load(name):
    return json.loads((FIXTURES / name).read_text())


class Clock:
    def __init__(self, t):
        self.t = t
        self.lock = threading.Lock()

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.t += s

    def at_least(self, t):
        with self.lock:
            self.t = max(self.t, t)


class Window:
    """A recorded window at the fake clock (module docstring): `state()` for the drivers, `flush()`
    for the edge."""

    def __init__(self, fix, clock, log_path):
        self.fix, self.clock, self.path = fix, clock, Path(log_path)
        self.t_arm, self.end, self.ended = fix["t_arm"], fix["end_s"], False
        self.rows = sorted(fix["log"], key=lambda r: r[1])
        self.readings = fix["readings"]
        self.lock = threading.Lock()
        self.url = "https://ep00000000000a.api.runpod.ai"
        self.disarmed = 0

    def now(self):
        return self.clock.t - self.t_arm

    def flush(self, upto=None):
        upto = self.now() if upto is None else upto
        with self.lock, self.path.open("a") as fh:
            while self.rows and self.rows[0][1] <= upto:
                recv, since, worker, source, ts, line = self.rows.pop(0)
                fh.write("%s\t+%.1f\t%s\t%s\t%s\t%s\n" % (recv, since, worker, source, ts, line))

    def state(self):
        s = self.now()
        if s > self.end:
            if self.ended:
                raise dv.Stop("the end of the recorded window")
            self.ended, s = True, self.end
        self.flush(s)
        found = [st for t, st in self.readings if t <= s]
        return copy.deepcopy(found[-1] if found else self.readings[0][1])

    # --- the provider's calls the drivers make -------------------------------------------------
    def endpoint(self, refresh=False):
        return {"id": "ep00000000000a", "dataCenterIds": ["EU-SE-1"]}

    def disarm(self):
        self.disarmed += 1

    def exists(self, name):
        return True

    def system_logs(self, since):
        return []

    def inputs(self, since):
        return []


def edge(win, fix, modal=False):
    """The balancer (RunPod) or the edge (Modal) of the window: /v1/models held 30 s (RunPod) or 502
    from the proxy (Modal) until the recorded readiness, then 200; the first chat at its time; the
    later chats as the window had them (`later`)."""
    chats, t_arm = [], fix["t_arm"]
    later = fix.get("later") or {}

    def handle(request):
        win.flush()
        path, s = request.url.path, win.now()
        if path == "/v1/models":
            if s < fix["ready_s"]:
                if modal:
                    return httpx.Response(502, json={"error": "mdf_upstream"}, headers={"x-mdf-proxy": "error"})
                if s + 30 < fix["ready_s"]:
                    win.clock.t += 30.0
                    raise httpx.ReadTimeout("held by the balancer")
                win.clock.at_least(t_arm + fix["ready_s"])
            return httpx.Response(200, json={"data": [{"id": dv.MODEL}]}, headers={"x-mdf-proxy": "forwarded"})
        if path == "/version":
            return httpx.Response(200, json={"version": dv.VLLM_VERSION}, headers={"x-mdf-proxy": "forwarded"})
        if path in ("/metrics", "/health", "/ping"):
            return httpx.Response(200, json={}, headers={"x-mdf-proxy": "health" if path != "/metrics" else "forwarded"})
        if request.method == "POST" and path == "/v1/chat/completions":
            with win.lock:
                chats.append(1)
                n = len(chats)
            if n == 1 and later.get("kind") != "crash":
                if fix.get("first_chat_s") is not None:
                    win.clock.at_least(t_arm + fix["first_chat_s"])
                return httpx.Response(200, json={"choices": []}, headers={"x-mdf-proxy": "forwarded"})
            kind = later.get("kind")
            if kind == "tripped":
                win.clock.at_least(t_arm + later["at_s"])
                return httpx.Response(503, json={"error": "mdf_tripped"}, headers={"x-mdf-proxy": "error"})
            if kind == "crash":
                win.clock.at_least(t_arm + later["at_s"])
                return httpx.Response(500, text="modal-http: internal error: function was terminated by signal")
            raise httpx.ReadTimeout("cut by the end of the window")
        return httpx.Response(404, json={})
    return httpx.MockTransport(handle)


class Replay(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.dir, True)
        self.log = self.dir / "worker-log.partial"
        self.log.write_text("")

    def arm(self, fix, provider, name):
        ledger = self.dir / "spend.jsonl"
        with spend.Ledger.file(ledger).locked() as locked:
            row = serverspend.arm(locked, ledger, provider, name, "https://ep00000000000a.api.runpod.ai", "45",
                                  fix["t_arm"] + 1942, "400", pv.RATES[provider], fix["t_arm"])
        return ledger, row["arm_id"]

    def run_driver(self, fn, live):
        out = io.StringIO()
        with redirect_stdout(out):
            code = dv.run_driver(fn, live, rpe.out)
        return code, [json.loads(x) for x in out.getvalue().splitlines() if x.strip()]

    def runpod(self, fix, argv):
        clock = Clock(fix["t_arm"] + fix["driver_start_s"])
        win = Window(fix, clock, self.log)
        env = self.dir / ".env"
        env.write_text("RUNPOD_API_KEY=%s\n" % KEY)
        env.chmod(0o600)
        ledger, arm_id = self.arm(fix, "runpod", "mdf-surya-fault")
        a = rpe.parser().parse_args(["--env-file", str(env)] + argv + ["--spend-ledger", str(ledger)])
        d = rpe.Driver(a, rp=win, transport=edge(win, fix), clock=clock, sleep=clock.sleep,
                       environ={wl.LOG_ENV: str(self.log), serverspend.SESSION_ENV: arm_id})
        fn = {"probe-valid": d.probe_valid, "fault-test": d.fault_test}[argv[0]]
        code, lines = self.run_driver(fn, d.live)
        return code, lines, win

    def modal(self, fix, argv):
        clock = Clock(fix["t_arm"] + fix["driver_start_s"])
        win = Window(fix, clock, self.log)
        env = self.dir / ".env"
        env.write_text("MODAL_PROXY_TOKEN_ID=wk-replay\nMODAL_PROXY_TOKEN_SECRET=ws-replay\n")
        env.chmod(0o600)
        servers = self.dir / "servers.json"
        servers.write_text(json.dumps({"default": "modal-l4", "servers": {"modal-l4": {
            "kind": "modal", "url": "https://ws--mdf-surya-fault-serve.modal.run", "rate_usd_per_s": "0.00036772"}}}))
        ledger, arm_id = self.arm(fix, "modal", "mdf-surya-fault")
        a = mt.parser().parse_args(["--env-file", str(env), "--server", "modal-l4", "--servers", str(servers)] + argv
                                   + ["--app", "mdf-surya-fault", "--spend-ledger", str(ledger)])
        t = mt.Tasks(a, modal=win, transport=edge(win, fix, modal=True), clock=clock, sleep=clock.sleep,
                     environ={wl.LOG_ENV: str(self.log), serverspend.SESSION_ENV: arm_id})
        fn = {"probe-valid": t.probe_valid, "fault-test": t.fault_test}[argv[0]]
        code, lines = self.run_driver(fn, t.live)
        return code, lines, win


A = ["fault-test", "--name", "mdf-surya-fault", "--phase", "A"]


def at(fix, since, change):
    """`fix` with each reading from `since` s changed by `change(state)` (in place)."""
    for t, st in fix["readings"]:
        if t >= since:
            change(st)
    return fix


def worker(st, wid, status):
    for w in st["workers"]:
        if w["id"] == wid:
            w["status"] = status
    st["summary"]["running"] = sum(1 for w in st["workers"] if w["status"] == "RUNNING")


def listing(recv, since, wid, status, running):
    return [recv, since, wid, "session", "", json.dumps({"data_center": "EU-SE-1", "event": "worker",
                                                          "gpu": "NVIDIA RTX A5000", "running": running,
                                                          "status": status, "uptime": None}, sort_keys=True)]


class TestRunPodWindows(Replay):
    """Phase A of 20/11 (`636a1558`, c7.19): the worker that ran stopped by RunPod 132.4 s after the
    hang, then listed IDLE without a container next to two extra workers. With the zero Z1 it passes;
    with c7.19 it was "no zero within 157 s"."""

    def test_phase_a_of_20_11_passes_with_the_zero_z1(self):
        code, lines, win = self.runpod(load("replay-runpod-636a1558-phase-a.json"), A)
        res = lines[-1]
        self.assertEqual(code, 0, res.get("violation"))
        self.assertTrue(res["pass"])
        self.assertEqual((res["ready"]["ready"], res["ready"]["since_arm_s"], res["ready"]["served"]), (True, 318.6, "w1"))
        self.assertEqual((res["held"]["status"], res["held"]["body"]), (503, {"error": "mdf_tripped"}))
        self.assertEqual(res["abandoned"]["status"], 503)                 # G2: the abandoned request too
        self.assertEqual(res["t_trip_s"], 67.9)
        # the zero: the first reading without RUNNING after the stop of the container (+451.7 s)
        self.assertGreaterEqual(res["zero_since_arm_s"], 459.1)
        self.assertLessEqual(res["zero_since_hang_s"], rpe.A_ZERO_S)
        self.assertEqual([(x["id"], x["running_s"]) for x in res["extra_workers"]], [("w2", None), ("w3", None)])
        self.assertEqual((res["replacements"], res["restarts_after_fault"], res["facts"]), ([], [], []))
        stop, = res["workers_log"]["stops"]
        self.assertEqual((stop["worker"], stop["since_arm"]), ("w1", 451.7))

    def test_a_worker_that_stays_running_breaks_g1(self):
        fix = load("replay-runpod-636a1558-phase-a.json")
        at(fix, 450.0, lambda st: worker(st, "w1", "RUNNING"))
        fix["log"] = [r for r in fix["log"] if not (r[2] == "w1" and r[1] >= 450.0
                                                     and (r[3] == "session" or r[5].startswith("stop container")))]
        code, lines, _ = self.runpod(fix, A)
        res = lines[-1]
        self.assertEqual((code, res["pass"], res["guarantee"]), (6, False, "G1"), res)
        self.assertTrue(res["violation"].startswith("no zero within"), res["violation"])
        self.assertIn("RUNNING in the reading: ['w1']", res["violation"])

    def test_two_workers_running_at_once_break_g1(self):
        fix = load("replay-runpod-636a1558-phase-a.json")
        at(fix, 400.0, lambda st: worker(st, "w2", "RUNNING"))
        code, lines, _ = self.runpod(fix, A)
        res = lines[-1]
        self.assertEqual((code, res["guarantee"]), (6, "G1"), res)
        self.assertTrue(res["violation"].startswith("two workers RUNNING at once"), res["violation"])

    def test_a_worker_running_after_the_zero_breaks_g1(self):
        fix = load("replay-runpod-636a1558-phase-a.json")
        at(fix, 481.0, lambda st: worker(st, "w1", "RUNNING"))
        fix["log"].append(listing("2026-11-20T16:22:27.200Z", 481.7, "w1", "RUNNING", ["w1"]))
        code, lines, _ = self.runpod(fix, A)
        res = lines[-1]
        self.assertEqual((code, res["guarantee"]), (6, "G1"), res)
        self.assertTrue(res["violation"].startswith("a worker RUNNING after the zero of the phase"), res["violation"])

    def test_phase_a_of_19_11_up_to_its_disarm_with_its_extra_worker(self):
        """`fc3106de`: c7.18 stopped it at +536.5 s on its extra worker (total 2 > 1), the session
        disarmed at +537.0 s. Replayed up to there: no guarantee broken, the extra worker declared."""
        code, lines, _ = self.runpod(load("replay-runpod-fc3106de-phase-a.json"), A)
        self.assertEqual(code, 130)
        last = lines[-1]
        self.assertEqual(last["stopped_by"], "the end of the recorded window")
        res = last["partial"]["result"]
        self.assertNotIn("violation", res)
        self.assertEqual((res["ready"]["ready"], res["first"]["status"]), (True, 200))
        extra = last["partial"]["workers_log"]["extra_workers"]
        self.assertEqual([(x["id"], x["states"][0][1], x["running_s"], x["system"][0]["line"]) for x in extra],
                         [("w2", "INITIALIZING", None, "worker is ready")])
        self.assertIn({"extra_worker": 1}, [{k: v for k, v in x.items() if k == "extra_worker"} for x in lines])

    def test_probe_valid_of_19_11_up_to_its_disarm_with_its_extra_worker(self):
        """`250a0406` (c7.6): the extra worker listed 1.3 s after readiness; the driver of that day closed
        without a line (c7.13), so the readings are the states of the follower's listings."""
        code, lines, _ = self.runpod(load("replay-runpod-250a0406-probe-valid.json"), ["probe-valid", "--name",
                                                                                       "mdf-surya"])
        self.assertEqual(code, 130, lines[-1])
        last = lines[-1]
        res = last["partial"]["result"]
        self.assertEqual((res["ready"]["ready"], res["chat"]["status"], res["chat"]["proxy"]), (True, 200, "forwarded"))
        self.assertNotIn("stopped", res["ready"])
        self.assertEqual([x["id"] for x in last["partial"]["workers_log"]["extra_workers"]], ["w2"])


class TestModalWindows(Replay):
    """The windows of Modal of 20/11 that the tests of c7.17 stood for: phases A and B passed on the day,
    and the readiness of `concurrency-2` (a container replaced before vLLM answered: rule (a))."""

    def test_phase_a(self):
        code, lines, _ = self.modal(load("replay-modal-50b1b363-phase-a.json"), ["fault-test", "--phase", "A"])
        res = lines[-1]
        self.assertEqual(code, 0, res.get("violation"))
        self.assertTrue(res["pass"])
        self.assertAlmostEqual(res["t_trip_s"], 67.4, delta=1.0)        # 67.4 on the day; the GETs come every 5 s
        self.assertLessEqual(res["zero_since_hang_s"], mt.A_ZERO_S + 10)
        self.assertEqual((res["replacements"], res["restarts_after_fault"]), ([], []))

    def test_phase_b(self):
        code, lines, _ = self.modal(load("replay-modal-82ef8610-phase-b.json"),
                                    ["fault-test", "--phase", "B", "--observe-seconds", "1300"])
        res = lines[-1]
        self.assertEqual(code, 0, res.get("violation"))
        self.assertEqual([r["status"] for r in res["requests"]], [500, 500, 500])
        self.assertEqual((res["restarts"], res["restarts_after_fault"]), (0, []))

    def test_the_readiness_of_concurrency_2_declares_its_replacement(self):
        code, lines, _ = self.modal(load("replay-modal-eb304b3e-readiness.json"), ["probe-valid"])
        res = lines[-1]
        self.assertEqual(code, 0, res.get("violation"))
        ready = res["probe_valid"]["ready"]
        self.assertEqual((ready["ready"], ready["served"]), (True, wl.short("ta-02REPLAY" + "X" * 18)))
        row, = res["replacements"]
        self.assertEqual((row["old"]["id"], row["new"]["id"]), (wl.short("ta-01REPLAY" + "X" * 18),
                                                                wl.short("ta-02REPLAY" + "X" * 18)))
        self.assertEqual((row["old"]["seen_s"], row["old"]["gone_s"], row["new"]["seen_s"]), (10.8, 22.3, 26.7))
        self.assertEqual((row["old_vllm_s"], row["old_inference_lines"]), (None, 0))


if __name__ == "__main__":
    unittest.main()
