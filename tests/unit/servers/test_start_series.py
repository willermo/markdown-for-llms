"""The series of starts of a rented server (`scripts/servers/start_series.py`, MFL033 N5b). No network:
a fake provider for the readings, a fake runner for `mdf convert`, `httpx.MockTransport` for the page,
a fake clock; the log of the session is written as the follower writes it."""
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from datetime import datetime, timezone

import httpx

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts" / "servers"))
import drivers as dv  # noqa: E402
import providers as pv  # noqa: E402
import start_series as ss  # noqa: E402

from markdown_for_llms.bench import serverspend, spend  # noqa: E402

KEY = "rp-secret-value"
T_ARM = 1_800_000_000.0


class Clock:
    def __init__(self, t=T_ARM + 5.0):
        self.t = t

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.t += s


def iso(t, ns=False):
    text = datetime.fromtimestamp(t, timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")
    return (text + "123Z") if ns else (text[:-3] + "Z")


def row(since, worker, source, text, ts=""):
    """A line of the log of the session, as the follower writes it."""
    return "\t".join([iso(T_ARM + since), "+%.1f" % since, worker, source, ts, text]) + "\n"


def listing(since, worker, status, running=None):
    ev = {"event": "worker", "status": status, "gpu": "NVIDIA RTX A5000", "data_center": "EU-RO-1", "uptime": None,
          "running": running if running is not None else ([worker] if status == "RUNNING" else [])}
    return row(since, worker, "session", json.dumps(ev, sort_keys=True))


def proxy(since, worker, event, **fields):
    return row(since, worker, "container", json.dumps(dict(event=event, t=T_ARM + since, **fields), sort_keys=True),
               ts=iso(T_ARM + since, ns=True))


def vllm(since, worker, text):
    return row(since, worker, "container", "(EngineCore pid=714) INFO 11-21 07:53:27 [x.py:1] " + text,
               ts=iso(T_ARM + since, ns=True))


def system(since, worker, text):
    return row(since, worker, "system", text, ts=iso(T_ARM + since)[:19] + "Z")


def start_lines(base, worker, cached=False):
    """One container start of vLLM 0.25.1, the times of N5a (window b4f05497) from `base`."""
    lines = [proxy(base + 0.6, worker, "mdf_start", provider="runpod"),
             proxy(base + 0.6, worker, "mdf_gpu", names=["NVIDIA RTX A5000"], since_start=0.0),
             proxy(base + 4.1, worker, "mdf_weights_check", ok=True, phase="start", seconds=3.5),
             proxy(base + 4.1, worker, "mdf_vllm_start", since_start=3.5),
             vllm(base + 49.1, worker, "Initializing a V1 LLM engine (v0.25.1) with config: model='/x'"),
             vllm(base + 58.2, worker, "Model loading took 1.42 GiB memory and 0.818937 seconds")]
    if cached:
        lines.append(vllm(base + 60.0, worker, "Directly load the compiled graph(s) for compile range (1, 8192) "
                                                "from the cache, took 1.802 s"))
    lines += [vllm(base + 88.4, worker, "torch.compile took %s s in total" % ("2.19" if cached else "28.13")),
              vllm(base + 151.7, worker, "Initial profiling/warmup run took 63.34 s"),
              vllm(base + 170.4, worker, "Graph capturing finished in 2 secs, took 0.09 GiB"),
              vllm(base + 170.6, worker, "init engine (profile, create kv cache, warmup model) took 112.43 s "
                                         "(compilation: 28.13 s)"),
              row(base + 178.9, worker, "container", "(APIServer pid=66) INFO:     Application startup complete.",
                  ts=iso(T_ARM + base + 178.9, ns=True)),
              proxy(base + 180.8, worker, "mdf_ready", since_start=180.1)]
    return lines


def runpod_log():
    lines = [row(1.9, "-", "session", json.dumps({"event": "follow", "t_arm": iso(T_ARM)})),
             listing(2.7, "w1", "INITIALIZING"),
             system(3.8, "w1", "loading container image from cache"),
             system(77.9, "w1", "Loaded image ID: sha256:" + "a" * 64),
             system(86.9, "w1", "create container vllm/vllm-openai@sha256:" + "b" * 64),
             listing(87.9, "w1", "RUNNING")]
    lines += start_lines(87.9, "w1")
    lines += [listing(400.0, "w1", "IDLE"), system(401.0, "w1", "stop container " + "c" * 64),
              listing(500.0, "w1", "RUNNING"), system(500.5, "w1", "start container " + "c" * 64)]
    lines += start_lines(500.0, "w1", cached=True)
    return "".join(lines)


class TestPhases(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.dir, True)

    def test_the_phases_of_two_container_starts_of_one_worker(self):
        path = self.dir / "worker-log.txt"
        path.write_text(runpod_log())
        first, second = ss.phases(path, T_ARM)
        self.assertEqual(first["worker"], "w1")
        self.assertEqual(first["image"], {"listed_s": 2.7, "from_cache": True, "loaded_s": 77.9, "create_s": 86.9,
                                          "seconds": 75.2})
        self.assertEqual(first["gpu"], ["NVIDIA RTX A5000"])
        self.assertEqual((first["running_s"], first["start_s"], first["ready_s"]), (87.9, 88.5, 268.7))
        self.assertEqual(first["weights_s"], 3.5)
        self.assertEqual(first["import_s"], 45.0)
        self.assertEqual((first["load_s"], first["compile_s"], first["profiling_s"]), (0.818937, 28.13, 63.34))
        self.assertEqual((first["capture_s"], first["init_engine_s"], first["graphs_s"]), (2.0, 112.43, 21.0))
        self.assertEqual((first["api_s"], first["start_to_ready_s"]), (10.2, 180.1))
        self.assertFalse(first["compile_cached"])
        # the second start of the worker: no image loaded again, the compiled graphs from the cache
        self.assertIsNone(second["image"])
        self.assertEqual((second["running_s"], second["start_s"]), (500.0, 500.6))
        self.assertTrue(second["compile_cached"])
        self.assertEqual(second["compile_s"], 2.19)
        self.assertFalse([k for s in (first, second) for k in s if k.startswith("_")])

    def test_the_arm_from_the_follower_and_the_times_of_modal(self):
        """Modal: a container per start, its lines with times to the second, no image lines."""
        cid = "ta-01M7YJ233S7ZJ2JWJM7F570HQR"
        lines = [row(3.8, "-", "session", json.dumps({"event": "follow", "provider": "modal", "t_arm": iso(T_ARM)})),
                 row(10.4, cid, "session", json.dumps({"event": "worker", "start": None, "status": "PENDING"})),
                 proxy(10.5, cid, "mdf_start", provider="modal"),
                 proxy(12.2, cid, "mdf_vllm_start"),
                 row(16.1, cid, "session", json.dumps({"event": "worker", "start": "x", "status": "RUNNING"})),
                 row(93.3, cid, "container", "(EngineCore pid=9) INFO 11-21 08:04:11 [core.py:114] Initializing a "
                                             "V1 LLM engine (v0.25.1)", ts="2026-11-21 09:04:11+01:00"),
                 proxy(260.6, cid, "mdf_ready", since_start=250.1)]
        path = self.dir / "worker-log.txt"
        path.write_text("".join(lines))
        (s,) = ss.phases(path)                               # the arm from the follower's first line
        self.assertEqual((s["start_s"], s["running_s"], s["ready_s"], s["start_to_ready_s"]), (10.5, 16.1, 260.6, 250.1))
        self.assertIsNone(s["image"])
        t_engine = datetime.fromisoformat("2026-11-21 09:04:11+01:00").timestamp()
        self.assertEqual(s["import_s"], round(t_engine - (T_ARM + 12.2), 1))


class FakeProvider:
    """`state()` of a provider: successive readings, the last one repeats."""

    def __init__(self, states):
        self.states, self.calls = list(states), 0

    def state(self):
        self.calls += 1
        s = self.states.pop(0) if len(self.states) > 1 else self.states[0]
        return json.loads(json.dumps(s))


def rp_state(*statuses):
    workers = [{"id": "w%d" % i, "status": st, "uptime": None, "gpu": "NVIDIA RTX A5000", "data_center": "EU-RO-1"}
               for i, st in enumerate(statuses, 1)]
    return {"armed": True, "busy": bool(workers), "workers_max": 1, "summary": {"total": len(workers)},
            "workers": workers}


class SeriesCase(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.dir, True)
        self.env = self.dir / ".env"
        self.env.write_text("RUNPOD_API_KEY=%s\n" % KEY)
        self.env.chmod(0o600)
        self.servers = self.dir / "servers.json"
        self.servers.write_text(json.dumps({"default": "rp", "servers": {"rp": {
            "kind": "runpod", "url": "https://ep123.api.runpod.ai", "gpu": "AMPERE_24", "rate_usd_per_s": "0.0002090948",
            "pins": {"vllm": "0.25.1"}}}}))
        self.png = self.dir / "page.png"
        self.png.write_bytes(b"\x89PNG\r\n\x1a\nfake")
        self.ledger = self.dir / "runs" / "spend.jsonl"
        self.ledger.parent.mkdir()
        self.clock = Clock()

    def arm(self, provider="runpod", window=3616):
        with spend.Ledger.file(self.ledger).locked() as locked:
            r = serverspend.arm(locked, self.ledger, provider, "mdf-surya", "https://ep123.api.runpod.ai", "80",
                                T_ARM + window, "400", pv.RATES[provider], T_ARM)
        return r["arm_id"]

    def argv(self, *extra):
        return ["--env-file", str(self.env), "--server", "rp", "--servers", str(self.servers), "--input", str(self.png),
                "--spend-ledger", str(self.ledger), "--spend-cap-usd", "4"] + list(extra)

    def run_main(self, argv, environ, **inject):
        out = io.StringIO()
        with redirect_stdout(out):
            code = ss.main(argv, clock=self.clock, sleep=self.clock.sleep, environ=environ, **inject)
        return code, [json.loads(x) for x in out.getvalue().splitlines() if x.strip()]


class FakeConvert:
    """`mdf convert` as the series runs it: the argv, an `engine.json` in `--out`, the summary on stdout.
    Each conversion takes `ready` seconds to readiness and `after` seconds more (the fake clock)."""

    def __init__(self, test, ready=(270.0, 1.0, 200.0, 1.0), after=20.0, fail_at=None):
        self.test, self.ready, self.after, self.fail_at, self.calls = test, list(ready), after, fail_at, []

    def __call__(self, cmd, capture_output, text, timeout, env):
        self.calls.append({"cmd": cmd, "env": env, "t": self.test.clock()})
        out = Path(cmd[cmd.index("--out") + 1])
        out.mkdir(parents=True)
        k = len(self.calls)
        t_wake = self.test.clock()
        wait = self.ready[(k - 1) % len(self.ready)]
        self.test.clock.sleep(wait + self.after)
        failed = self.fail_at == k
        (out / "engine.json").write_text(json.dumps({
            "status": "failed" if failed else "complete", "pages": {"count": 1},
            "timings": {"total_seconds": wait + self.after, "stages": {"models": 4.0}},
            "marker_server": {"t_wake": t_wake, "readiness": {"wait_seconds": wait}, "t_last": self.test.clock() - 1,
                              "t_close": self.test.clock(), "clean": True, "requests": {"chat": 1, "read": 0},
                              "by_status": {"200": 1}}}))
        summary = {"status": "failed" if failed else "complete", "error": "page_failed" if failed else None}
        # as `cli.emit` prints it: indented over several lines (c8.2.1: the series read single lines only)
        return subprocess.CompletedProcess(cmd, 1 if failed else 0, json.dumps(summary, indent=2) + "\n", "")


class TestSeries(SeriesCase):
    def test_cold_and_warm_in_order_with_the_quiet_before_each_cold_start(self):
        arm_id = self.arm()
        # readings: the quiet before the second cold start sees RUNNING twice, then IDLE
        prov = FakeProvider([rp_state("RUNNING"), rp_state("RUNNING"), rp_state("IDLE")])
        conv = FakeConvert(self)
        code, lines = self.run_main(self.argv("--cold", "2", "--warm", "2", "--cap-usd", "0.80"),
                                    {"MDF_SESSION_ARM": arm_id, "PATH": "/bin"}, prov=prov, runner=conv)
        self.assertEqual(code, 0, lines[-1])
        res = lines[-1]
        self.assertTrue(res["pass"])
        self.assertEqual([r["kind"] for r in res["starts"]], ["cold", "warm", "cold", "warm"])
        self.assertEqual([Path(c["cmd"][c["cmd"].index("--out") + 1]).name for c in conv.calls],
                         ["01-cold", "02-warm", "03-cold", "04-warm"])
        self.assertEqual(Path(conv.calls[0]["cmd"][conv.calls[0]["cmd"].index("--out") + 1]).parent,
                         self.ledger.parent / "starts" / arm_id)
        cmd = conv.calls[0]["cmd"]
        self.assertLess(cmd.index("--env-file"), cmd.index("convert"))       # an option of mdf (c7.18)
        self.assertEqual(cmd[cmd.index("--server-cap-usd") + 1], "0.80")
        self.assertEqual(cmd[cmd.index("--server-mode") + 1], "balanced")
        self.assertEqual(Path(cmd[0]).name, "mdf")
        self.assertNotIn("RUNPOD_API_KEY", conv.calls[0]["env"])
        self.assertEqual(conv.calls[0]["env"]["MDF_SESSION_ARM"], arm_id)
        self.assertEqual(prov.calls, 3)                     # only the quiet reads, and only before the 3rd start
        self.assertNotIn("quiet_after_s", res["starts"][1])
        self.assertEqual(res["starts"][2]["quiet_after_s"], 20.0)
        first = res["starts"][0]
        self.assertEqual((first["send_s"], first["ready_wait_s"], first["ready_s"]), (5.0, 270.0, 275.0))
        self.assertEqual((first["after_ready_s"], first["s_per_page"], first["pages"]), (20.0, 20.0, 1))
        self.assertTrue(all(r["ok"] for r in res["starts"]))
        self.assertEqual([x for x in lines if "start_row" in x][0]["start_row"]["start"], 1)   # a line as it ends

    def test_a_failed_start_stops_the_series(self):
        arm_id = self.arm()
        conv = FakeConvert(self, fail_at=2)
        code, lines = self.run_main(self.argv("--cold", "3", "--warm", "3", "--cap-usd", "0.80"),
                                    {"MDF_SESSION_ARM": arm_id}, prov=FakeProvider([rp_state()]), runner=conv)
        self.assertEqual(code, 6)
        self.assertEqual(len(conv.calls), 2)
        self.assertIn("start 2 (warm) failed", lines[-1]["failed"])
        self.assertFalse(lines[-1]["pass"])

    def test_never_quiet_stops_the_series(self):
        arm_id = self.arm()
        conv = FakeConvert(self)
        code, lines = self.run_main(self.argv("--cold", "2", "--cap-usd", "0.80"), {"MDF_SESSION_ARM": arm_id},
                                    prov=FakeProvider([rp_state("RUNNING")]), runner=conv)
        self.assertEqual(code, 6)
        self.assertEqual(len(conv.calls), 1)
        self.assertIn("not quiet within 600 s", lines[-1]["failed"])

    def test_the_log_waits_for_the_stop_of_the_container(self):
        """RunPod: IDLE in the readings, but the log has no "stop container" after the RUNNING yet."""
        arm_id = self.arm()
        log = self.dir / "worker-log.partial"
        log.write_text(listing(10.0, "w1", "RUNNING"))
        writes = iter([None, None, system(300.0, "w1", "stop container " + "c" * 64)])

        class Prov(FakeProvider):
            def state(prov):
                line = next(writes, None)
                if line:
                    with open(log, "a") as fh:
                        fh.write(line)
                return FakeProvider.state(prov)
        code, lines = self.run_main(self.argv("--cold", "2", "--cap-usd", "0.80"),
                                    {"MDF_SESSION_ARM": arm_id, "SESSION_WORKER_LOG": str(log)},
                                    prov=Prov([rp_state("IDLE")]), runner=FakeConvert(self))
        self.assertEqual(code, 0, lines[-1])
        self.assertEqual(lines[-1]["starts"][1]["quiet_after_s"], 20.0)
        self.assertNotIn("no_stop_line", lines[-1]["starts"][1])

    def test_no_stop_line_for_90_s_is_quiet_and_a_fact(self):
        arm_id = self.arm()
        log = self.dir / "worker-log.partial"
        log.write_text(listing(10.0, "w1", "RUNNING"))
        code, lines = self.run_main(self.argv("--cold", "2", "--cap-usd", "0.80"),
                                    {"MDF_SESSION_ARM": arm_id, "SESSION_WORKER_LOG": str(log)},
                                    prov=FakeProvider([rp_state("IDLE")]), runner=FakeConvert(self))
        self.assertEqual(code, 0)
        self.assertTrue(lines[-1]["starts"][1]["no_stop_line"])
        self.assertEqual([f["fact"] for f in lines[-1]["facts"]], ["no_stop_line"])

    def test_two_workers_running_at_once_break_g1(self):
        arm_id = self.arm()
        code, lines = self.run_main(self.argv("--cold", "2", "--cap-usd", "0.80"), {"MDF_SESSION_ARM": arm_id},
                                    prov=FakeProvider([rp_state("RUNNING", "RUNNING")]), runner=FakeConvert(self))
        self.assertEqual(code, 6)
        self.assertEqual(lines[-1]["guarantee"], "G1")

    def test_the_phases_of_the_log_go_with_the_cold_starts(self):
        arm_id = self.arm()
        log = self.dir / "worker-log.partial"
        log.write_text(runpod_log())
        conv = FakeConvert(self, ready=(270.0, 1.0), after=20.0)
        self.clock.t = T_ARM + 1.0
        prov = FakeProvider([rp_state("IDLE")] * 5)
        code, lines = self.run_main(self.argv("--cold", "2", "--warm", "1", "--cap-usd", "0.80"),
                                    {"MDF_SESSION_ARM": arm_id, "SESSION_WORKER_LOG": str(log)}, prov=prov,
                                    runner=conv)
        self.assertEqual(code, 0, lines[-1])
        starts = lines[-1]["starts"]
        self.assertEqual(starts[0]["container"]["start_s"], 88.5)
        self.assertNotIn("container", starts[1])
        self.assertNotIn("new_container", starts[1])
        self.assertEqual(starts[2]["container"]["start_s"], 500.6)
        self.assertTrue(starts[2]["container"]["compile_cached"])
        self.assertEqual(len(lines[-1]["phases"]), 2)


class TestSummary(unittest.TestCase):
    def test_the_summary_of_mdf_convert_over_several_lines(self):
        """MFL033 c8.2.1 (21/11, window ebddfbf1): `mdf convert` prints its summary indented (`cli.emit`)."""
        doc = {"status": "complete", "out": "x", "pages": {"count": 1}}
        self.assertEqual(ss.summary_of(json.dumps(doc, indent=2) + "\n"), doc)
        self.assertEqual(ss.summary_of("noise\n" + json.dumps(doc) + "\n"), doc)
        self.assertIsNone(ss.summary_of("not json\n{"))
        self.assertIsNone(ss.summary_of(""))


class TestPageAndArmAhead(SeriesCase):
    def fake(self, models_after=0):
        n = {"models": 0}

        def handler(request):
            self.assertEqual(request.headers.get("authorization"), "Bearer " + KEY)
            if request.url.path == "/v1/models":
                n["models"] += 1
                if n["models"] <= models_after:
                    return httpx.Response(502, json={"error": "upstream"}, headers={"x-mdf-proxy": "error"})
                return httpx.Response(200, json={"data": [{"id": dv.MODEL}]}, headers={"x-mdf-proxy": "forwarded"})
            if request.url.path == "/v1/chat/completions":
                body = json.loads(request.content)
                self.assertEqual(body["messages"][0]["content"][1]["text"], "OCR this image to HTML.")
                self.clock.sleep(3.0)
                return httpx.Response(200, json={"choices": []}, headers={"x-mdf-proxy": "forwarded"})
            return httpx.Response(404)
        return httpx.MockTransport(handler), n

    def test_arm_ahead_then_one_page_straight_to_vllm(self):
        arm_id = self.arm(window=1224)
        transport, n = self.fake(models_after=3)
        prov = FakeProvider([rp_state("INITIALIZING")])
        code, lines = self.run_main(self.argv("--cold", "1", "--arm-ahead", "420", "--via", "page"),
                                    {"MDF_SESSION_ARM": arm_id}, prov=prov, transport=transport,
                                    runner=lambda *a, **k: self.fail("no mdf convert with --via page"))
        self.assertEqual(code, 0, lines[-1])
        res = lines[-1]
        self.assertEqual(len(res["before_first"]), 15)          # +5, +35, ... +395 and +420: every 30 s to 420 s
        self.assertEqual(res["before_first"][-1]["s"], 420.0)
        self.assertFalse(res["running_before_request"])
        (start,) = res["starts"]
        self.assertEqual(start["send_s"], 420.0)
        self.assertEqual((start["ready_wait_s"], start["tries"], start["ready_s"]), (15.0, 4, 435.0))
        self.assertEqual((start["page"]["status"], start["page"]["proxy"], start["s_per_page"]), (200, "forwarded", 3.0))
        self.assertTrue(start["ok"])

    def test_a_worker_running_before_the_request_is_a_fact(self):
        arm_id = self.arm(window=1224)
        transport, _ = self.fake()
        code, lines = self.run_main(self.argv("--cold", "1", "--arm-ahead", "60", "--via", "page"),
                                    {"MDF_SESSION_ARM": arm_id}, prov=FakeProvider([rp_state("INITIALIZING"),
                                                                                    rp_state("RUNNING")]),
                                    transport=transport)
        self.assertEqual(code, 0)
        self.assertTrue(lines[-1]["running_before_request"])
        self.assertEqual(lines[-1]["facts"][0]["fact"], "running_before_request")


class TestPopulate(SeriesCase):
    """MFL033 c8.4: `--via populate`, readiness then the lines of the populate in the log of the session."""

    def models(self):
        def handler(request):
            if request.url.path == "/v1/models":
                return httpx.Response(200, json={"data": [{"id": dv.MODEL}]}, headers={"x-mdf-proxy": "forwarded"})
            self.fail("no inference while the cache is populated: %s" % request.url.path)
        return httpx.MockTransport(handler)

    def test_the_sha256_of_the_archive_from_the_log(self):
        arm_id = self.arm(window=1224)
        log = self.dir / "worker-log.partial"
        log.write_text("")
        lines = iter([proxy(300.0, "w1", "mdf_compile_cache", phase="populate", written=True, sha256="e" * 64,
                            bytes=1000, files=12, seconds=4.2)])

        class Prov(FakeProvider):
            pass

        def sleep(s):
            self.clock.sleep(s)
            line = next(lines, None)
            if line:
                with open(log, "a") as fh:
                    fh.write(line)
        out = io.StringIO()
        with redirect_stdout(out):
            code = ss.main(self.argv("--cold", "1", "--via", "populate"), clock=self.clock, sleep=sleep,
                           environ={"MDF_SESSION_ARM": arm_id, "SESSION_WORKER_LOG": str(log)},
                           prov=Prov([rp_state("RUNNING")]), transport=self.models())
        res = json.loads(out.getvalue().splitlines()[-1])
        self.assertEqual(code, 0, res)
        (start,) = res["starts"]
        self.assertEqual((start["sha256"], start["populate"]["written"], start["pages"]), ("e" * 64, True, 0))
        self.assertTrue(start["ok"])

    def test_a_populate_not_written_fails(self):
        arm_id = self.arm(window=1224)
        log = self.dir / "worker-log.partial"
        log.write_text(proxy(30.0, "w1", "mdf_compile_cache", phase="populate", written=False, error="ValueError"))
        code, lines = self.run_main(self.argv("--cold", "1", "--via", "populate"),
                                    {"MDF_SESSION_ARM": arm_id, "SESSION_WORKER_LOG": str(log)},
                                    prov=FakeProvider([rp_state("RUNNING")]), transport=self.models())
        self.assertEqual(code, 6)
        self.assertFalse(lines[-1]["starts"][0]["ok"])

    def test_the_use_of_the_cache_in_the_phases(self):
        lines = [row(1.9, "-", "session", json.dumps({"event": "follow", "t_arm": iso(T_ARM)})),
                 listing(87.9, "w1", "RUNNING")]
        lines += start_lines(87.9, "w1", cached=True)
        lines.insert(4, proxy(91.0, "w1", "mdf_compile_cache", phase="use", used=True, sha256="e" * 64, seconds=3.1))
        path = self.dir / "worker-log.txt"
        path.write_text("".join(lines))
        (s,) = ss.phases(path, T_ARM)
        self.assertEqual(s["compile_cache"], {"used": True, "sha256": "e" * 64, "seconds": 3.1})
        self.assertTrue(s["compile_cached"])


class FakeModalGpus:
    """Modal behind the proxy for `--gpu` (MFL033 N6): a GET of /v1/models with no container starts one
    on the next GPU of `gpus`, its `mdf_start` and `mdf_gpu` written in the log of the session as the
    follower writes them; a container leaves after `idle_reads` readings of the provider. On the GPU
    `populate_on` the container populates the cache after `ready_after` GETs."""

    def __init__(self, test, log, gpus, idle_reads=2, populate_on=None, ready_after=2):
        self.test, self.log, self.gpus, self.idle_reads = test, log, list(gpus), idle_reads
        self.populate_on, self.ready_after = populate_on, ready_after
        self.current, self.started, self.models, self.reads, self.calls = None, [], 0, 0, 0

    def write(self, *lines):
        with open(self.log, "a") as fh:
            fh.write("".join(lines))

    def handler(self, request):
        if request.url.path != "/v1/models":
            self.test.fail("no inference before the GPU is known: %s" % request.url.path)
        self.models += 1
        since = round(self.test.clock() - T_ARM, 1)
        if self.current is None:
            cid = "ta-%d" % (len(self.started) + 1)
            gpu = self.gpus[min(len(self.started), len(self.gpus) - 1)]
            self.current = {"id": cid, "gpu": gpu, "gets": 0, "idle": 0}
            self.started.append(self.current)
            self.write(proxy(since, cid, "mdf_start", provider="modal"),
                       proxy(since, cid, "mdf_gpu", names=[gpu], since_start=0.0))
        c = self.current
        c["gets"], c["idle"] = c["gets"] + 1, 0
        if c["gpu"] == self.populate_on and c["gets"] == self.ready_after:
            self.write(proxy(since, c["id"], "mdf_compile_cache", phase="populate", written=True, sha256="f" * 64,
                             bytes=1000, files=12, gpu=[c["gpu"]], seconds=2.6),
                       proxy(since + 4.4, c["id"], "mdf_compile_cache", phase="commit", ok=True, error=None))
        return httpx.Response(200, json={"data": [{"id": dv.MODEL}]}, headers={"x-mdf-proxy": "forwarded"})

    def state(self):
        self.calls += 1
        c = self.current
        if c is not None:
            c["idle"] += 1
            if c["idle"] > self.idle_reads:
                self.current = c = None
        return {"app": "mdf-surya", "containers": [{"id": c["id"], "status": "RUNNING"}] if c else []}


class TestGpu(SeriesCase):
    """MFL033 N6 (21/11): Modal serves its type `A10` (the `A10G` of `gpu=`) with an NVIDIA A10 or an
    NVIDIA A10G, as it likes, and an archive of the cache of torch.compile serves only the GPU of its
    manifest. `--gpu NAME`: before each cold start a GET wakes the container and the log of the session
    gives its GPU; on another one no more requests (the container leaves before vLLM is ready, so no
    archive is written and no document is sent), the provider quiet, then the next try, at most
    `--gpu-tries`; none on NAME: the start fails with `gpu_not_assigned`."""

    def setUp(self):
        super().setUp()
        self.env.write_text("MODAL_PROXY_TOKEN_ID=wk-id-value\nMODAL_PROXY_TOKEN_SECRET=ws-secret-value\n")
        self.servers.write_text(json.dumps({"default": "md", "servers": {"md": {
            "kind": "modal", "url": "https://ws--mdf-surya-serve.us-east.modal.run", "gpu": "L4,A10G",
            "rate_usd_per_s": "0.00036772", "pins": {"vllm": "0.25.1"}}}}))
        self.log = self.dir / "worker-log.partial"
        self.log.write_text(row(3.8, "-", "session", json.dumps({"event": "follow", "provider": "modal",
                                                                  "t_arm": iso(T_ARM)})))

    def argv(self, *extra):
        return super().argv(*extra)[:3] + ["md"] + super().argv(*extra)[4:]

    def series(self, fake, *extra, runner=None):
        arm_id = self.arm(provider="modal", window=907)
        return self.run_main(self.argv(*extra), {"MDF_SESSION_ARM": arm_id, "SESSION_WORKER_LOG": str(self.log)},
                             prov=fake, transport=httpx.MockTransport(fake.handler),
                             runner=runner or (lambda *a, **k: self.fail("no mdf convert")))

    def test_another_gpu_is_let_go_then_the_gpu_named_populates(self):
        fake = FakeModalGpus(self, self.log, ["NVIDIA A10", "NVIDIA A10G"], populate_on="NVIDIA A10G")
        code, lines = self.series(fake, "--cold", "1", "--via", "populate", "--gpu", "NVIDIA A10G", "--gpu-tries", "3")
        res = lines[-1]
        self.assertEqual(code, 0, res)
        (start,) = res["starts"]
        tries = start["gpu_tries"]
        self.assertEqual([(t["try"], t["gpu"], t["taken"]) for t in tries],
                         [(1, ["NVIDIA A10"], False), (2, ["NVIDIA A10G"], True)])
        self.assertEqual([t["worker"] for t in tries], ["ta-1", "ta-2"])
        self.assertGreater(tries[1]["quiet_after_s"], 0)            # the A10 left before the second try
        self.assertEqual(start["send_s"], tries[1]["send_s"])       # the start is the try taken
        self.assertEqual(fake.started[0]["gets"], 1)                # one GET only to the A10: never ready
        self.assertEqual((start["sha256"], start["populate"]["gpu"], start["ok"]), ("f" * 64, ["NVIDIA A10G"], True))
        self.assertEqual(start["container"]["worker"], "ta-2")
        self.assertEqual(start["container"]["gpu"], ["NVIDIA A10G"])

    def test_no_container_on_the_gpu_named_within_the_tries(self):
        fake = FakeModalGpus(self, self.log, ["NVIDIA A10"])
        code, lines = self.series(fake, "--cold", "1", "--via", "populate", "--gpu", "NVIDIA A10G", "--gpu-tries", "2")
        res = lines[-1]
        self.assertEqual(code, 6)
        (start,) = res["starts"]
        self.assertEqual((start["ok"], start["stopped"]), (False, "gpu_not_assigned"))
        self.assertEqual([t["gpu"] for t in start["gpu_tries"]], [["NVIDIA A10"]] * 2)
        self.assertIn("gpu_not_assigned", res["failed"])
        self.assertEqual([c["gets"] for c in fake.started], [1, 1])  # neither was asked again

    def test_one_try_by_default_and_no_quiet_before_it(self):
        fake = FakeModalGpus(self, self.log, ["NVIDIA A10", "NVIDIA A10G"])
        code, lines = self.series(fake, "--cold", "1", "--via", "populate", "--gpu", "NVIDIA A10G")
        self.assertEqual(code, 6)
        (t,) = lines[-1]["starts"][0]["gpu_tries"]
        self.assertNotIn("quiet_after_s", t)
        self.assertEqual(fake.calls, 0)                             # no reading of the provider before it

    def test_the_engine_converts_on_the_container_of_the_gpu_named(self):
        fake = FakeModalGpus(self, self.log, ["NVIDIA A10G"])
        conv = FakeConvert(self, ready=(160.0,))
        code, lines = self.series(fake, "--cold", "1", "--cap-usd", "0.45", "--gpu", "NVIDIA A10G", runner=conv)
        res = lines[-1]
        self.assertEqual(code, 0, res)
        self.assertEqual(len(conv.calls), 1)
        (start,) = res["starts"]
        self.assertEqual([t["taken"] for t in start["gpu_tries"]], [True])
        self.assertLessEqual(start["gpu_tries"][0]["send_s"], start["wake_s"])   # woken before mdf convert
        self.assertTrue(start["ok"])

    def test_a_populate_written_on_another_gpu_is_not_taken(self):
        fake = FakeModalGpus(self, self.log, ["NVIDIA A10G"], populate_on="NVIDIA A10G")
        real = fake.handler

        def handler(request):                                       # the manifest names another GPU
            r = real(request)
            text = self.log.read_text().replace('"gpu": ["NVIDIA A10G"], "phase": "populate"',
                                                '"gpu": ["NVIDIA A10"], "phase": "populate"')
            self.log.write_text(text)
            return r
        fake.handler = handler
        code, lines = self.series(fake, "--cold", "1", "--via", "populate", "--gpu", "NVIDIA A10G")
        self.assertEqual(code, 6)
        self.assertFalse(lines[-1]["starts"][0]["ok"])

    def test_the_tries_are_checked(self):
        arm_id = self.arm(provider="modal")
        for extra in (["--gpu", "NVIDIA A10G", "--gpu-tries", "0"], ["--gpu-tries", "2"], ["--gpu", ""]):
            with self.subTest(extra=extra):
                code, lines = self.run_main(self.argv("--cold", "1", "--via", "populate", *extra),
                                            {"MDF_SESSION_ARM": arm_id})
                self.assertEqual(code, 2)
                self.assertEqual(lines[-1]["code"], "invalid_setting")


class TestRefusals(SeriesCase):
    def test_outside_a_session(self):
        code, lines = self.run_main(self.argv("--cold", "1", "--cap-usd", "0.80"), {}, prov=FakeProvider([rp_state()]),
                                    runner=FakeConvert(self))
        self.assertEqual(code, 2)
        self.assertIn("inside session.py run", lines[-1]["detail"])

    def test_the_engine_needs_the_cap_and_the_counts_are_checked(self):
        arm_id = self.arm()
        code, lines = self.run_main(self.argv("--cold", "1"), {"MDF_SESSION_ARM": arm_id},
                                    prov=FakeProvider([rp_state()]), runner=FakeConvert(self))
        self.assertEqual(code, 2)
        self.assertIn("--cap-usd", lines[-1]["detail"])
        code, _ = self.run_main(self.argv("--cold", "0", "--cap-usd", "0.80"), {"MDF_SESSION_ARM": arm_id})
        self.assertEqual(code, 2)

    def test_the_session_of_another_provider(self):
        arm_id = self.arm(provider="modal")
        code, lines = self.run_main(self.argv("--cold", "1", "--cap-usd", "0.80"), {"MDF_SESSION_ARM": arm_id},
                                    prov=FakeProvider([rp_state()]), runner=FakeConvert(self))
        self.assertEqual(code, 2)
        self.assertIn("the session armed modal", lines[-1]["detail"])


GEN = "0123456789abcdef"


def snap_boot_lines(base, worker, gen=GEN, enabled=True):
    """The lines of `snap_boot` in a container that creates a snapshot (MFL045 c2), from `base`."""
    return [proxy(base, worker, "mdf_gpu", names=["NVIDIA A10G"], uuids=["GPU-a"], since_start=0.0),
            proxy(base + 3.0, worker, "mdf_weights_check", ok=True, phase="snap_boot", seconds=2.9),
            proxy(base + 3.1, worker, "mdf_vllm_args", ok=True, snap_boot=True, args=["--enable-sleep-mode"]),
            proxy(base + 3.1, worker, "mdf_vllm_start", pid=23, snap_boot=True),
            vllm(base + 50.0, worker, "Initializing a V1 LLM engine (v0.25.1) with config: model='/x'"),
            vllm(base + 80.0, worker, "torch.compile took 28.13 s in total"),
            proxy(base + 200.0, worker, "mdf_snap_boot", ready_s=196.9),
            proxy(base + 230.0, worker, "mdf_snapshot", ok=True, gen=gen, seconds=230.0, snap_enabled=enabled,
                  pids=[23, 57])]


def woken(base, worker, kind, gen=GEN, fallback=None, chat=True):
    """`snap_wake` in a container: the kind, the proxy, the wake (or the fallback), readiness, a chat."""
    out = [proxy(base, worker, "mdf_restore", kind=kind, gen=gen, age_s=5.0 if kind == "create" else 3600.0,
                 pids=[23, 57]),
           proxy(base + 0.1, worker, "mdf_start", provider="modal", snapshot=kind, boot_id="b" * 16),
           proxy(base + 0.2, worker, "mdf_gpu", names=["NVIDIA A10G"], uuids=["GPU-a"], since_start=0.1)]
    if fallback:
        out += [proxy(base + 2.0, worker, "mdf_fallback", reason=fallback, pids=[23, 57], since_start=1.9),
                proxy(base + 6.0, worker, "mdf_vllm_start", pid=91, since_start=5.9),
                proxy(base + 170.0, worker, "mdf_ready", since_start=169.9)]
    else:
        out += [proxy(base + 4.0, worker, "mdf_wake", ok=True, seconds=3.8, since_start=3.9),
                proxy(base + 5.0, worker, "mdf_ready", since_start=4.9)]
    if chat:
        end = base + (180.0 if fallback else 12.0)
        out.append(proxy(end, worker, "mdf_request", kind="forwarded", method="POST", path="/v1/chat/completions",
                         status=200))
    return out


class TestSnapshotPhases(unittest.TestCase):
    """MFL045 c2 (plan r2 §8 and E.7): the kind of every container start of a window of the snapshot from the
    log (`create`, `restore`, `boot`, with a declared fallback), the SDK's "GPU memory snapshot restore failed"
    (exit 222) as `restore_failed`, the creation of the snapshot (its seconds), and the first useful answer
    (the first `mdf_request` forwarded POST /v1/chat/completions 200 of the container)."""

    def test_the_kinds_of_the_starts(self):
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp, True)
        log = tmp / "worker-log.partial"
        rows = (snap_boot_lines(10.0, "ta-A") + woken(240.0, "ta-A", "create")
                + woken(600.0, "ta-B", "restore")
                + [row(900.0, "ta-C", "container", "GPU memory snapshot restore failed: CUDA error. Retrying task "
                                                   "without snapshot", ts=iso(T_ARM + 900.0, ns=True))]
                + snap_boot_lines(905.0, "ta-D", enabled=False) + woken(1140.0, "ta-D", "boot")
                + woken(1500.0, "ta-E", "restore", fallback="wake_up 500"))
        log.write_text("".join(rows))
        starts = ss.phases(log, T_ARM)
        kinds = [(s["worker"], s["snapshot"]["kind"]) for s in starts]
        self.assertEqual(kinds, [("ta-A", "create"), ("ta-B", "restore"), ("ta-C", "restore_failed"),
                                 ("ta-D", "boot"), ("ta-E", "restore")])
        a, b, c, d, e = starts
        self.assertEqual((a["snapshot"]["created"]["seconds"], a["snapshot"]["created"]["ready_s"]), (230.0, 196.9))
        self.assertEqual(a["snapshot"]["gen"], GEN)
        self.assertEqual(a["snap_boot"]["compile_s"], 28.13)
        self.assertEqual((a["start_s"], a["ready_s"], a["first_useful_s"]), (240.1, 245.0, 252.0))
        self.assertEqual((b["snapshot"]["age_s"], b["snapshot"]["wake"]["seconds"], b["first_useful_s"]),
                         (3600.0, 3.8, 612.0))
        self.assertEqual(c["start_s"], 900.0)
        self.assertIsNone(d["snapshot"].get("fallback"))
        self.assertEqual((e["snapshot"]["fallback"], e["ready_s"], e["first_useful_s"]), ("wake_up 500", 1670.0, 1680.0))


class TestRepeatedInput(SeriesCase):
    """MFL045 c2 (plan r2 §8, M2): `--input` repeated, one per cold start in order (D04, D16, D18); one input for
    every start as before; another count is refused."""

    def test_one_input_per_cold_start(self):
        arm_id = self.arm()
        inputs = []
        for name in ("d04.pdf", "d16.pdf", "d18.pdf"):
            (self.dir / name).write_bytes(b"%PDF-1.4\n")
            inputs += ["--input", str(self.dir / name)]
        argv = [x for x in self.argv("--cold", "3", "--cap-usd", "1.00") if x != str(self.png)]
        argv.remove("--input")
        conv = FakeConvert(self)
        code, lines = self.run_main(argv + inputs, {"MDF_SESSION_ARM": arm_id},
                                    prov=FakeProvider([rp_state("IDLE")]), runner=conv)
        self.assertEqual(code, 0, lines[-1])
        self.assertEqual([Path(c["cmd"][c["cmd"].index("convert") + 1]).name for c in conv.calls],
                         ["d04.pdf", "d16.pdf", "d18.pdf"])
        self.assertEqual([r["input"] for r in lines[-1]["starts"]], ["d04.pdf", "d16.pdf", "d18.pdf"])
        code, lines = self.run_main(argv + inputs[:4], {"MDF_SESSION_ARM": arm_id},
                                    prov=FakeProvider([rp_state("IDLE")]), runner=FakeConvert(self))
        self.assertEqual((code, lines[-1]["code"]), (2, "invalid_setting"))


class TestStartKindInTheRows(SeriesCase):
    def test_each_cold_start_gets_the_kind_of_its_container(self):
        self.servers.write_text(json.dumps({"default": "md", "servers": {"md": {
            "kind": "modal", "url": "https://ws--mdf-surya-snap-serve-web.modal.run", "gpu": "L4,A10G",
            "rate_usd_per_s": "0.00036772", "pins": {"vllm": "0.25.1"}}}}))
        self.env.write_text("MODAL_PROXY_TOKEN_ID=wk-test\nMODAL_PROXY_TOKEN_SECRET=ws-test\n")
        arm_id = self.arm(provider="modal")
        log = self.dir / "worker-log.partial"
        log.write_text("".join(snap_boot_lines(10.0, "ta-A") + woken(240.0, "ta-A", "create")
                               + woken(600.0, "ta-B", "restore")))
        self.clock.t = T_ARM + 1.0
        conv = FakeConvert(self, ready=(244.0, 4.0), after=10.0)

        class Quiet(FakeProvider):
            def state(prov):
                self.clock.t = max(self.clock.t, T_ARM + 590.0)
                return FakeProvider.state(prov)
        argv = [x if x != "rp" else "md" for x in self.argv("--cold", "2", "--cap-usd", "0.90")]
        code, lines = self.run_main(argv, {"MDF_SESSION_ARM": arm_id, "SESSION_WORKER_LOG": str(log)},
                                    prov=Quiet([{"armed": True, "busy": False, "workers": [], "containers": []}]),
                                    runner=conv)
        self.assertEqual(code, 0, lines[-1])
        self.assertEqual([r["start_kind"] for r in lines[-1]["starts"]], ["create", "restore"])
        self.assertEqual(lines[-1]["starts"][1]["container"]["first_useful_s"], 612.0)


class TestResumes(SeriesCase):
    """MFL045 c3 (plan r2 §6; prescription P-4): `--resume N` sends N requests after the cold starts, each after
    `--idle` seconds without requests from the end of the one before (`--idle-last` for the last one), by the way
    of `--resume-via` (one per resume: engine or page). A resume is told by the same `boot_id` and an `mdf_pause` of
    it after the request before; a new `mdf_start` is a missed resume; the same boot without a pause a warm start."""

    fake = TestPageAndArmAhead.fake

    def log(self, rows):
        path = self.dir / "worker-log.partial"
        path.write_text("".join(rows))
        return path

    def test_idle_waits_ways_and_kinds(self):
        arm_id = self.arm(window=2420)
        transport, n = self.fake()
        conv = FakeConvert(self, ready=(170.0, 1.0), after=20.0)
        self.clock.t = T_ARM + 20.0
        boot = "b" * 16
        log = self.log([proxy(25.0, "w1", "mdf_start", provider="runpod", boot_id=boot),
                        proxy(190.0, "w1", "mdf_ready", since_start=165.0),
                        proxy(340.0, "w1", "mdf_pause", boot_id=boot, seconds=90.0, applied=True, during="sleep"),
                        proxy(472.0, "w1", "mdf_pause", boot_id=boot, seconds=95.0, applied=True, during="sleep"),
                        proxy(1258.0, "w2", "mdf_start", provider="runpod", boot_id="c" * 16)])
        code, lines = self.run_main(self.argv("--cold", "1", "--resume", "4", "--idle", "120", "--idle-last", "660",
                                              "--resume-via", "engine,page,page,page", "--cap-usd", "0.55"),
                                    {"MDF_SESSION_ARM": arm_id, "SESSION_WORKER_LOG": str(log)},
                                    prov=FakeProvider([rp_state("IDLE")]), runner=conv, transport=transport)
        self.assertEqual(code, 0, lines[-1])
        rows = lines[-1]["starts"]
        self.assertEqual([r["kind"] for r in rows], ["cold", "resume", "resume", "resume", "resume"])
        self.assertEqual([r["via"] for r in rows], ["engine", "engine", "page", "page", "page"])
        self.assertEqual(len(conv.calls), 2)
        for before, row, idle in zip(rows, rows[1:], (120, 120, 120, 660)):
            self.assertGreaterEqual(round(row["send_s"] - before["end_s"], 1), idle)
            self.assertLess(row["send_s"] - before["end_s"], idle + 1)
            self.assertEqual(row["idle_s"], idle)
        self.assertEqual([r["start_kind"] for r in rows], ["cold", "resume", "resume", "warm", "missed"])
        self.assertEqual(rows[1]["boot_id"], boot)
        self.assertEqual(rows[4]["new_container"]["boot_id"], "c" * 16)

    def test_the_options_are_checked(self):
        arm_id = self.arm(window=2420)
        for extra in (("--resume", "2"), ("--resume", "2", "--idle", "120", "--resume-via", "page"),
                      ("--resume", "1", "--idle", "120", "--resume-via", "fax"), ("--idle", "120")):
            with self.subTest(extra=extra):
                code, lines = self.run_main(self.argv("--cold", "1", "--via", "page", *extra),
                                            {"MDF_SESSION_ARM": arm_id}, prov=FakeProvider([rp_state("IDLE")]))
                self.assertEqual((code, lines[-1]["code"]), (2, "invalid_setting"))


if __name__ == "__main__":
    unittest.main()
