"""The worker logs of a RunPod window (`scripts/servers/worker_logs.py`, MFL033 c7.4). No network:
`httpx.MockTransport` stands for the log stream of the REST v2, a fake source for the follower."""
import json
from pathlib import Path
import shutil
import sys
import tempfile
import threading
import time
import unittest

import httpx

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts" / "servers"))
import worker_logs as wl  # noqa: E402

KEY = "rp-secret-value"
T_ARM = 1_800_000_000.0


def sse(*events):
    return "".join("id: %s\ndata: %s\n\n" % (eid, json.dumps(data)) for eid, data in events)


class TestStream(unittest.TestCase):
    def source(self, handler):
        def client(timeout):
            return httpx.Client(transport=httpx.MockTransport(handler), timeout=timeout,
                                headers={"Authorization": "Bearer " + KEY})
        return wl.RunPodLogs("https://api.runpod.io/v2", "ep123", client)

    def test_events_since_then_resumed_with_the_last_id(self):
        seen = []

        def handler(request):
            seen.append((request.url.path, dict(request.url.params), request.headers.get("last-event-id"),
                         request.headers.get("accept"), request.headers.get("authorization")))
            return httpx.Response(200, text=sse(("2026-11-19T10:00:01Z/1", {"source": "system", "line": "pull",
                                                                             "ts": "2026-11-19T10:00:01Z"}),
                                                ("2026-11-19T10:00:02Z/2", {"source": "container", "line": "{\"event\": "
                                                                            "\"mdf_start\"}", "ts": "t2"})),
                                  headers={"content-type": "text/event-stream"})
        src = self.source(handler)
        got = list(src.stream("w1", "2026-11-19T09:59:00Z", None))
        self.assertEqual([e for e, _ in got], ["2026-11-19T10:00:01Z/1", "2026-11-19T10:00:02Z/2"])
        self.assertEqual(got[1][1]["source"], "container")
        self.assertEqual(seen[0], ("/v2/serverless/ep123/workers/w1/logs", {"since": "2026-11-19T09:59:00Z"}, None,
                                   "text/event-stream", "Bearer " + KEY))
        list(src.stream("w1", "2026-11-19T09:59:00Z", "2026-11-19T10:00:02Z/2"))
        self.assertEqual(seen[1][1:3], ({}, "2026-11-19T10:00:02Z/2"))          # Last-Event-ID, no since

    def test_gone_and_errors(self):
        src = self.source(lambda request: httpx.Response(404, json={"error": "worker not found"}))
        with self.assertRaises(wl.Gone):
            list(src.stream("w1", "x", None))
        src = self.source(lambda request: httpx.Response(429, json={}))
        with self.assertRaises(RuntimeError):
            list(src.stream("w1", "x", None))

    def test_workers_listed(self):
        src = self.source(lambda request: httpx.Response(200, json={"workers": [
            {"id": "w1", "status": "INITIALIZING", "gpuTypeId": "NVIDIA RTX A5000", "dataCenterId": "EU-SE-1"}]}))
        self.assertEqual(src.workers(), [{"id": "w1", "status": "INITIALIZING", "gpu": "NVIDIA RTX A5000",
                                          "data_center": "EU-SE-1", "uptime": None}])


class FakeSource:
    """Workers as successive answers; for each worker a list of rounds, each a list of events or an
    exception (a cut, a quiet stream, a gone worker)."""

    def __init__(self, lists, rounds):
        self.lists, self.rounds = list(lists), {k: list(v) for k, v in rounds.items()}
        self.calls = []

    def workers(self):
        if len(self.lists) > 1:
            return self.lists.pop(0)
        return self.lists[0]

    def stream(self, wid, since, last_id):
        self.calls.append((wid, since, last_id))
        todo = self.rounds.get(wid) or [wl.Gone(wid)]
        step = todo.pop(0) if len(todo) > 1 else todo[0]
        if isinstance(step, Exception):
            raise step
        for item in step:
            yield item


class TestFollower(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, True)
        self.path = Path(tmp) / "worker-log.partial"

    def follow(self, src):
        f = wl.Follower(src, self.path, T_ARM, clock=lambda: T_ARM + 12.5, list_every=0.05)
        f.reconnect_s, f.retry_s = 0.2, 0.2
        return f.start()

    def wait_for(self, cond, within=5.0):
        t0 = time.monotonic()
        while not cond() and time.monotonic() - t0 < within:
            time.sleep(0.02)

    def test_lines_reconnects_and_the_end_of_a_worker(self):
        e1 = ("i1", {"source": "system", "line": "image pulled", "ts": "t1"})
        e2 = ("i2", {"source": "container", "line": json.dumps({"event": "mdf_start", "mono": 5.0}), "ts": "t2"})
        e3 = ("i3", {"source": "container", "line": "line\twith a tab\nand a newline", "ts": "t3"})
        rounds = {"w1": [wl.Gone("w1"), [e1, e2], httpx.ReadTimeout("quiet"), [e2, e3], wl.Gone("w1")]}
        w1 = {"id": "w1", "status": "RUNNING", "gpu": "G", "data_center": "EU-SE-1", "uptime": 3}
        src = FakeSource([[dict(w1, status="INITIALIZING", uptime=None)]] + [[w1]] * 40 + [[]], rounds)
        f = self.follow(src)
        self.wait_for(lambda: "logs_gone" in self.path.read_text())
        info = f.stop(grace=2)
        lines = [wl.parse(x) for x in self.path.read_text().splitlines()]
        self.assertTrue(all(x is not None for x in lines))
        self.assertEqual([x["line"] for x in lines if x["source"] in ("system", "container")],
                         ["image pulled", json.dumps({"event": "mdf_start", "mono": 5.0}),
                          "line with a tab\\nand a newline"])              # e2 not twice: deduplicated by id
        notes = [json.loads(x["line"]) for x in lines if x["source"] == "session"]
        self.assertEqual([n.get("status") for n in notes if n["event"] == "worker"], ["INITIALIZING", "RUNNING", "GONE"])
        # a 404 while listed (starting) is tried again; once unlisted it ends the reading
        self.assertEqual([n["event"] for n in notes if n["event"].startswith("logs")], ["logs_not_found_yet", "logs_gone"])
        self.assertEqual(lines[0]["since_arm"], 12.5)
        self.assertEqual([c[2] for c in src.calls][:5], [None, None, "i2", "i2", "i3"])  # resumed from the last id
        self.assertEqual(set([c[2] for c in src.calls][5:]), {"i3"} if len(src.calls) > 5 else set())
        self.assertTrue(all(c[1] == "2027-01-15T07:59:00Z" for c in src.calls))  # 60 s before the arm
        self.assertEqual(info["workers"], ["w1"])
        self.assertEqual(info["errors"], {})                                    # a quiet stream is not an error
        self.assertEqual([x["event"]["event"] for x in lines if x["source"] == "container" and x["event"]],
                         ["mdf_start"])
        f.write("w1", "container", "", "after the stop")                        # closed: dropped
        self.assertNotIn("after the stop", self.path.read_text())

    def test_errors_noted_once_and_the_file_private(self):
        class Failing(FakeSource):
            def workers(self):
                raise RuntimeError("workers: HTTP 500")
        f = self.follow(Failing([[]], {}))
        time.sleep(0.3)
        info = f.stop(grace=1)
        self.assertGreaterEqual(info["errors"]["workers: RuntimeError"], 2)
        text = self.path.read_text()
        self.assertEqual(text.count("follower_error"), 1)
        self.assertEqual(self.path.stat().st_mode & 0o777, 0o600)


class TestParse(unittest.TestCase):
    def test_parse_and_partial_lines(self):
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, True)
        path = Path(tmp) / "log"
        start = "\t".join(["r", "+1.0", "w1", "container", "", json.dumps({"event": "mdf_download_start", "mono": 1.0,
                                                                         "disk": 0, "rx": 0})])
        end = "\t".join(["r", "+9.0", "w1", "container", "", json.dumps({"event": "mdf_download_end", "exit": 2})])
        rules = wl.PopulateRules()
        path.write_text(start + "\n" + end[:20])
        self.assertIsNone(rules.feed(path))                                     # the half line waits
        with open(path, "a") as fh:
            fh.write(end[20:] + "\n")
        self.assertIn("the download failed: exit 2", rules.feed(path))
        self.assertIsNone(wl.parse("not\ta line"))
        self.assertIsNone(wl.PopulateRules().feed(path.with_name("missing")))

    def test_vllm_error_lines(self):
        for line, error in (("ERROR 11-19 10:00:00 [x.py:1] boom", True),
                            ("(EngineCore_DP0 pid=7) CRITICAL 11-19 10:00:00 [x.py:1] boom", True),
                            ("(APIServer pid=1) WARNING 11-19 10:00:00 [x.py:1] slow", False),
                            ("INFO 11-19 10:00:00 [x.py:1] no ERROR here", False)):
            self.assertEqual(bool(wl.VLLM_ERROR.search(line)), error, line)


class TestRequestLines(unittest.TestCase):
    """MFL033 c7.6: the requests answered by the proxy (`mdf_request`, `forwarded` or `error`)."""

    def test_requests_found_as_they_come(self):
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, True)
        path = Path(tmp) / "log"

        def line(since, source, event):
            return "\t".join(["r", "+%.1f" % since, "w1", source, "", json.dumps(event)]) + "\n"
        forwarded = line(3.0, "container", {"event": "mdf_request", "kind": "forwarded", "method": "GET",
                                            "path": "/v1/models", "status": 200})
        error = line(4.0, "container", {"event": "mdf_request", "kind": "error", "method": "POST",
                                        "path": "/v1/chat/completions", "status": 503, "code": "mdf_busy"})
        ignored = (line(1.0, "session", {"event": "mdf_request", "kind": "forwarded"})       # the follower's own
                   + line(2.0, "container", {"event": "mdf_request", "kind": "health"})
                   + line(2.5, "container", {"event": "mdf_ready"}) + "r\t+2.6\tw1\tsystem\t\tplain text\n")
        reqs = wl.RequestLines()
        path.write_text(ignored + forwarded[:15])
        self.assertEqual(reqs.feed(path), [])                                   # the half line waits
        with open(path, "a") as fh:
            fh.write(forwarded[15:] + error)
        found = reqs.feed(path)
        self.assertEqual([(r["kind"], r["status"], r["since_arm"], r["worker"]) for r in found],
                         [("forwarded", 200, 3.0, "w1"), ("error", 503, 4.0, "w1")])
        self.assertEqual(reqs.feed(path), found)                                # nothing new, nothing twice
        with self.assertRaises(OSError):                                        # the caller decides
            wl.RequestLines().feed(path.with_name("missing"))


# The two system lines of `mdf-surya` fetched on 20/11 (n4/modal/after-applogs-system.out of the run).
WAITING_LINE = ("2026-11-20 09:09:07+01:00 fu-sF4Wg5rE4HlmgcKfKPvp5Q Function 'serve' (fu-sF4Wg5rE4HlmgcKfKPvp5Q) "
                "is waiting to be scheduled on a GPU_L4 worker. We are actively working on acquiring more capacity "
                "for your workload.")
STOP_LINE = "2026-11-20 09:20:44+01:00 Stopping app - user stopped from CLI."
T_WAITING = 1795162147.0                       # 2026-11-20T08:09:07Z


class Now:
    def __init__(self, t):
        self.t = t

    def __call__(self):
        return self.t


class TestScheduler(unittest.TestCase):
    """c7.14: the system lines of a Modal app fetched without `-f`, the scheduler's waiting line and
    its deadline, on the lines Modal wrote on 20/11."""

    def test_the_lines_of_20_11(self):
        now, emitted = Now(T_WAITING + 18.0), []
        s = wl.Scheduler(lambda since: [WAITING_LINE, STOP_LINE], T_WAITING - 110, clock=now, emit=emitted.append,
                         requested=("L4", "A10G"))
        s.read()
        self.assertEqual([(x["ts"], x["function"], x["container"]) for x in emitted],
                         [("2026-11-20 09:09:07+01:00", "fu-sF4Wg5rE4HlmgcKfKPvp5Q", None),
                          ("2026-11-20 09:20:44+01:00", None, None)])
        self.assertEqual(emitted[1]["text"], "Stopping app - user stopped from CLI.")
        self.assertTrue(emitted[0]["text"].startswith("Function 'serve' (fu-sF4Wg5rE4HlmgcKfKPvp5Q) is waiting"))
        w = s.summary()["waiting"]
        self.assertEqual((w["gpu"], w["t_from"], w["t_seen"]), ("GPU_L4", T_WAITING, T_WAITING + 18.0))
        self.assertEqual(s.deadline(), T_WAITING + 300)
        s.read()
        self.assertEqual(len(emitted), 2)                                       # a line once
        self.assertEqual(s.summary()["reads"], 2)

    def test_the_deadline_and_the_verdict(self):
        now = Now(T_WAITING + 299.0)
        s = wl.Scheduler(lambda since: [WAITING_LINE], T_WAITING - 110, clock=now, requested=("L4", "A10G"))
        self.assertIsNone(s.verdict([], now.t))                                 # nothing read yet
        s.read()
        self.assertIsNone(s.verdict([], now.t))                                 # 299 s
        now.t = T_WAITING + 300.0
        self.assertIsNone(s.verdict([], T_WAITING + 299.0))                     # no reading since the deadline
        self.assertIsNone(s.verdict(["ta-1"], now.t))                           # a container: capacity came
        reason = s.verdict([], now.t)
        self.assertTrue(reason.startswith("modal_no_capacity: no container 300 s after \"Function 'serve'"), reason)
        self.assertTrue(reason.endswith("(requested L4/A10G)"), reason)

    def test_a_clock_behind_the_server_counts_from_the_reading(self):
        now = Now(T_WAITING - 40.0)                                             # the laptop 40 s behind
        s = wl.Scheduler(lambda since: [WAITING_LINE], T_WAITING - 150, clock=now)
        s.read()
        self.assertEqual(s.deadline(), T_WAITING - 40.0 + 300)                  # never later than 300 s from here

    def test_other_lines_a_failed_fetch_and_a_waiting_line_without_a_gpu(self):
        rows = iter([RuntimeError("modal app logs: exit 1"), ["not a line of the CLI"],
                     ["2026-11-20T08:09:07Z fu-a ta-b Function 'serve' is waiting to be scheduled."]])

        def fetch(since):
            r = next(rows)
            if isinstance(r, Exception):
                raise r
            return r
        s = wl.Scheduler(fetch, 0, clock=Now(T_WAITING))
        s.read()
        self.assertEqual(s.summary()["errors"], ["RuntimeError: modal app logs: exit 1"])
        s.read()
        self.assertEqual((s.lines[0]["ts"], s.lines[0]["text"], s.waiting), (None, "not a line of the CLI", None))
        s.read()
        self.assertEqual((s.lines[1]["function"], s.lines[1]["container"]), ("fu-a", "ta-b"))
        self.assertEqual((s.waiting["gpu"], s.waiting["t_from"]), (None, T_WAITING))

    def test_gpus_of_a_window(self):
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, True)
        path = Path(tmp) / "worker-log.txt"

        def line(since, worker, source, event):
            return "\t".join(["r", "+%.1f" % since, worker, source, "", json.dumps(event)]) + "\n"
        path.write_text(line(1.0, "-", "session", {"event": "mdf_gpu", "names": ["not a proxy"]})
                        + line(150.2, "ta-1", "container", {"event": "mdf_start"})
                        + line(150.4, "ta-1", "container", {"event": "mdf_gpu", "names": ["NVIDIA L4"],
                                                            "source": "nvidia-smi", "error": None})
                        + line(900.0, "ta-2", "container", {"event": "mdf_gpu", "names": ["NVIDIA A10G"],
                                                            "source": "proc", "error": "nvidia-smi: FileNotFoundError"}))
        self.assertEqual(wl.gpus(path), [
            {"worker": "ta-1", "names": ["NVIDIA L4"], "source": "nvidia-smi", "since_arm": 150.4},
            {"worker": "ta-2", "names": ["NVIDIA A10G"], "source": "proc", "since_arm": 900.0}])
        self.assertEqual(wl.gpus(path.with_name("missing")), [])


if __name__ == "__main__":
    unittest.main()


def row(since, worker, source, text, recv="2026-11-19T12:00:00.000Z", ts=""):
    return "\t".join([recv, "+%.1f" % since, worker, source, ts, text]) + "\n"


def listing(since, worker, status, running=None):
    ev = {"event": "worker", "status": status, "gpu": "NVIDIA RTX A5000", "data_center": "EU-SE-1", "uptime": None}
    if running is not None:
        ev["running"] = running
    return row(since, worker, "session", json.dumps(ev, sort_keys=True))


class TestRunPodWorkers(unittest.TestCase):
    """MFL033 c7.19 (prompt N4s §4): the follower writes the workers RUNNING of each listing, and
    `Workers` reads the workers of a RunPod window: states, first RUNNING, answers of vLLM,
    inference lines, the stop of the containers and two workers RUNNING at once."""

    def setUp(self):
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, True)
        self.path = Path(tmp) / "worker-log.partial"

    def test_each_listing_line_holds_the_workers_running(self):
        w = {"gpu": "G", "data_center": "EU-SE-1", "uptime": None}
        lists = [[dict(w, id="w1", status="RUNNING")],
                 [dict(w, id="w1", status="RUNNING"), dict(w, id="w2", status="INITIALIZING")],
                 [dict(w, id="w1", status="IDLE"), dict(w, id="w2", status="RUNNING")],
                 [dict(w, id="w2", status="RUNNING")]] + [[dict(w, id="w2", status="RUNNING")]] * 30
        f = wl.Follower(FakeSource(lists, {}), self.path, T_ARM, clock=lambda: T_ARM + 3.0, list_every=0.02)
        f.start()
        t0 = time.monotonic()
        while '"GONE"' not in self.path.read_text() and time.monotonic() - t0 < 5:
            time.sleep(0.02)
        f.stop(grace=1)
        notes = [json.loads(x["line"]) for x in map(wl.parse, self.path.read_text().splitlines())
                 if x["source"] == "session" and json.loads(x["line"]).get("event") == "worker"]
        self.assertEqual([(n["status"], n["running"]) for n in notes],
                         [("RUNNING", ["w1"]), ("INITIALIZING", ["w1"]), ("IDLE", ["w2"]), ("RUNNING", ["w2"]),
                          ("GONE", ["w2"])])
        lv = wl.Workers().feed(self.path)
        self.assertEqual((lv.overlaps, lv.running_order()), ([], ["w1", "w2"]))   # written in one listing: no overlap

    def test_the_workers_of_a_window(self):
        self.path.write_text("".join([
            row(2.0, "-", "session", json.dumps({"event": "follow", "t_arm": "2026-11-19T12:11:38.905Z"})),
            listing(2.6, "w1", "INITIALIZING"),                                  # a log of 19/11: no `running`
            listing(120.7, "w1", "THROTTLED"),
            listing(324.2, "w1", "RUNNING"),
            row(338.2, "w1", "container", json.dumps({"event": "mdf_start"})),
            row(338.3, "w1", "container", json.dumps({"event": "mdf_gpu", "names": ["NVIDIA RTX A5000"]})),
            row(522.4, "w1", "container", json.dumps({"event": "mdf_vllm_first_health", "status": 200})),
            row(524.1, "w1", "container", json.dumps({"event": "mdf_request", "kind": "forwarded", "method": "POST",
                                                      "path": "/v1/chat/completions", "status": 200})),
            listing(526.5, "w2", "INITIALIZING"),
            row(527.3, "w2", "system", "worker is ready"),
            row(540.3, "w1", "system", "stop container e961", recv="2026-11-19T12:20:39.179Z"),
            "a partial line"]))
        lv = wl.Workers().feed(self.path)
        self.assertEqual(lv.t_arm, wl._utc("2026-11-19T12:11:38.905Z"))
        self.assertEqual(lv.running_order(), ["w1"])
        self.assertEqual(lv.overlaps, [])
        self.assertEqual(lv.states("w1"), ["INITIALIZING", "THROTTLED", "RUNNING"])
        w1, w2 = (lv.brief(x) for x in ("w1", "w2"))
        self.assertEqual((w1["running_s"], w1["vllm_s"], w1["inference_lines"], w1["stop_s"], w1["gpu_proxy"]),
                         (324.2, 522.4, 1, 540.3, ["NVIDIA RTX A5000"]))
        self.assertEqual((w2["states"], w2["running_s"], w2["vllm_s"], w2["system"]),
                         ([[526.5, "INITIALIZING"]], None, None, [{"since_arm": 527.3, "line": "worker is ready"}]))
        self.assertEqual(lv.facts(), {"t_stop": wl._utc("2026-11-19T12:20:39.179Z"), "overlap_since_arm": None,
                                      "stops": 1, "error": None})

    def test_two_workers_running_at_once(self):
        self.path.write_text("".join([listing(100.0, "w1", "RUNNING", ["w1"]), listing(130.0, "w2", "RUNNING", ["w1", "w2"]),
                                      listing(140.0, "w1", "GONE", ["w2"])]))
        lv = wl.Workers().feed(self.path)
        self.assertEqual(lv.overlaps, [{"workers": ["w1", "w2"], "since_arm": 130.0, "utc": "2026-11-19T12:00:00.000Z",
                                        "by": "listing"}])
        self.assertEqual(lv.facts()["overlap_since_arm"], 130.0)
        self.path.write_text("".join([listing(100.0, "w1", "RUNNING"), listing(130.0, "w2", "RUNNING")]))
        self.assertEqual([o["by"] for o in wl.Workers().feed(self.path).overlaps], ["states"])   # a log of 19/11

    def test_bound_facts_never_raise(self):
        self.assertEqual(wl.bound_facts(self.path), {"t_stop": None, "overlap_since_arm": None, "stops": 0,
                                                     "error": "FileNotFoundError"})
