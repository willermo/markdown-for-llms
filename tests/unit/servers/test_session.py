"""The session around a paid command (`scripts/servers/session.py`, MFL033 plan r4 §5.4 and App. G,
review r4 prescriptions 2, 3, 4 and 5). No network: a fake provider, a fake clock where time
matters, real child processes."""
from decimal import Decimal
import fcntl
import io
import json
import os
from pathlib import Path
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import time
import unittest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts" / "servers"))
import session  # noqa: E402
import providers as pv  # noqa: E402
import worker_logs  # noqa: E402

from markdown_for_llms.bench import serverspend, spend  # noqa: E402

DRIVER = Path(__file__).resolve().parent / "session_driver.py"
RATE = pv.RATES["runpod"]


class Clock:
    def __init__(self):
        self.t = 1_800_000_000.0

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.t += s
        time.sleep(0.002)


class FakeProv:
    provider = "runpod"

    def __init__(self, armed=False, busy=False, zero_after=0, arm_error=None, on_url=None):
        self.armed, self.busy, self.zero_after = armed, busy, zero_after
        self.arm_error, self.on_url = arm_error, on_url
        self.calls, self.zeros = [], 0

    @property
    def url(self):
        if self.on_url:
            self.on_url()
        return "https://fake-endpoint.api.runpod.ai"

    def state(self):
        return {"armed": self.armed, "busy": self.busy, "summary": {"total": int(self.busy)}, "workers": []}

    def arm(self, fault, allow_download, deadline=None, **levers):
        self.calls.append(("arm", fault, allow_download) + ((levers["levers"],) if levers else ()))
        if self.arm_error:
            raise pv.ProviderError(self.arm_error)
        self.armed = self.busy = True
        return {"env_changed": bool(fault)}

    def disarm(self):
        self.calls.append("disarm")
        self.armed = False

    def zero(self):
        self.zeros += 1
        if self.zero_after is None or self.zeros <= self.zero_after:
            return False, self.state()
        self.busy = False
        return True, self.state()

    def balance(self):
        return 9.37


class Base(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.dir, True)
        self.ledger = self.dir / "spend.jsonl"
        self.env = self.dir / ".env"
        self.env.write_text("RUNPOD_API_KEY=rp-secret-value\n")
        self.env.chmod(0o600)

    def args(self, *extra, cap="0.60", command=("true",)):
        return ["run", "--provider", "runpod", "--name", "mdf-surya-fault", "--env-file", str(self.env),
                "--spend-ledger", str(self.ledger), "--spend-cap-usd", "4", "--cap-usd", cap, *extra, "--", *command]

    def main(self, argv, prov, clock=None):
        out = io.StringIO()
        a = session.parser().parse_args(argv)
        if a.cmd == "run" and a.command and a.command[0] == "--":
            a.command = a.command[1:]
        clock = clock or Clock()
        try:
            if a.cmd == "run":
                code = session.Session(a, prov, clock=clock, sleep=clock.sleep, out=out).run()
            elif a.cmd == "status":
                code = session.status(a, prov, out=out)
            else:
                code = session.disarm(a, prov, clock=clock, sleep=clock.sleep, out=out)
        except session.Refusal as exc:
            out.write(json.dumps({"code": exc.code, "detail": exc.detail}) + "\n")
            code = exc.exit_code
        lines = [json.loads(x) for x in out.getvalue().splitlines() if x.strip()]
        return code, lines

    def rows(self, kind=None):
        rows = spend.read(self.ledger) if self.ledger.exists() else []
        return [r for r in rows if kind is None or r.get("kind") == kind]

    def orphan_arm(self, provider="runpod", endpoint="mdf-surya-fault"):
        with spend.Ledger.file(self.ledger).locked() as locked:
            return serverspend.arm(locked, self.ledger, provider, endpoint, "https://fake-endpoint.api.runpod.ai",
                                   "30", 2e9, "400", RATE, 1.7e9)

    def locks(self):
        return sorted(p.name for p in self.dir.glob("*.lock"))


class TestStart(Base):
    def test_orphan_comes_before_provider_not_idle(self):
        self.orphan_arm()
        code, lines = self.main(self.args(), FakeProv(armed=True, busy=True))
        self.assertEqual(code, 3)
        self.assertEqual(lines[-1]["code"], "session_orphaned")

    def test_provider_not_idle(self):
        for prov in (FakeProv(armed=True), FakeProv(busy=True)):
            code, lines = self.main(self.args(), prov)
            self.assertEqual((code, lines[-1]["code"]), (3, "provider_not_idle"))
            self.assertEqual(prov.calls, [])
        self.assertEqual(self.rows("arm"), [])

    def test_a_window_open_by_a_live_session(self):
        row = self.orphan_arm()
        fd = os.open(serverspend.lock_path(self.ledger, row["arm_id"]), os.O_RDWR | os.O_CREAT, 0o600)
        fcntl.flock(fd, fcntl.LOCK_EX)
        try:
            code, lines = self.main(self.args(), FakeProv())
        finally:
            os.close(fd)
        self.assertEqual((code, lines[-1]["code"]), (3, "provider_not_idle"))

    def test_w_floored_and_refused_when_not_positive(self):
        self.assertEqual(session.window_seconds(Decimal("60"), RATE, 120), 2689)         # floor of 2689.5
        self.assertEqual(session.window_seconds(Decimal("60"), RATE, 120, limit=600), 600)
        # c7.19: W = A / r - 60 - Z - 30 on RunPod (the tail of the bound after the stop of the container)
        self.assertEqual(session.window_seconds(Decimal("60"), RATE, 120, tail=pv.BILLING_TAIL_S["runpod"]), 2659)
        self.assertEqual(pv.BILLING_TAIL_S, {"runpod": 30, "modal": 0})
        prov = FakeProv()
        code, lines = self.main(self.args(cap="0.03"), prov)                            # 143 s - 60 - 120 < 0
        self.assertEqual((code, lines[-1]["code"]), (3, "server_cap"))
        self.assertEqual((prov.calls, self.rows("arm"), self.locks()), ([], [], []))

    def test_ledger_read_again_under_its_lock(self):
        # another session arms between the check at the start and the arm row: refused, nothing armed
        held = {}

        def race():
            if "fd" not in held:
                row = self.orphan_arm()
                held["fd"] = os.open(serverspend.lock_path(self.ledger, row["arm_id"]), os.O_RDWR | os.O_CREAT, 0o600)
                fcntl.flock(held["fd"], fcntl.LOCK_EX)
        prov = FakeProv(on_url=race)
        try:
            code, lines = self.main(self.args(), prov)
        finally:
            os.close(held["fd"])
        self.assertEqual((code, lines[-1]["code"]), (3, "provider_not_idle"))
        self.assertEqual(len(self.rows("arm")), 1)
        self.assertEqual(prov.calls, [])

        def orphan():
            if not self.rows("arm")[1:]:
                self.orphan_arm(endpoint="other")
        prov = FakeProv(on_url=orphan)
        code, lines = self.main(self.args(), prov)
        self.assertEqual((code, lines[-1]["code"]), (3, "session_orphaned"))
        self.assertEqual(prov.calls, [])


class TestWindow(Base):
    def test_arm_command_disarm_rows(self):
        prov, clock = FakeProv(zero_after=2), Clock()
        code, lines = self.main(self.args("--fault", "hang-after-first"), prov, clock)
        self.assertEqual(code, 0)
        arm, = self.rows("arm")
        self.assertEqual(Decimal(arm["a_cents"]), Decimal("60.00"))
        self.assertEqual(arm["window_s"], 2659)                                  # c7.19: A / r - 60 - Z - 30
        self.assertEqual(Decimal(str(arm["until"])) - Decimal(str(arm["t_arm"])), 2659)
        self.assertEqual((arm["fault"], arm["z_s"], arm["tail_s"], arm["url"]),
                         ("hang-after-first", 120, 30, "https://fake-endpoint.api.runpod.ai"))
        dis, = self.rows("disarm")
        self.assertEqual((dis["clean"], dis["orphan"], dis["exit"]), (False, False, 0))
        self.assertEqual(dis["zero_latency_s"], 20.0)
        self.assertEqual((dis["t_stop"], dis["double_s"]), (None, 0.0))       # no log: the zero, plus the tail
        self.assertEqual(prov.calls, [("arm", "hang-after-first", False, pv.DEFAULT_LEVERS), "disarm"])   # N5b step 8
        self.assertEqual(self.locks(), [])

    def test_a_is_the_residual_when_smaller(self):
        with spend.Ledger.file(self.ledger).locked() as locked:
            serverspend.activity(locked, "runpod", "x", "370", "400")
        code, _ = self.main(self.args(), FakeProv())
        self.assertEqual(code, 0)
        self.assertEqual(Decimal(self.rows("arm")[0]["a_cents"]), Decimal("30.00"))

    def mdf(self, code=0):
        tool = self.dir / "bin" / "mdf"
        tool.parent.mkdir(exist_ok=True)
        tool.write_text("#!/bin/sh\nexit %d\n" % code)
        tool.chmod(0o755)
        return str(tool)

    def test_clean_only_for_the_engine_with_exit_0_and_zero_within_z(self):
        code, _ = self.main(self.args(command=(self.mdf(), "convert", "x")), FakeProv(zero_after=1))
        self.assertEqual((code, self.rows("disarm")[-1]["clean"]), (0, True))
        code, _ = self.main(self.args(command=(self.mdf(), "convert", "x")), FakeProv(zero_after=13))
        dis = self.rows("disarm")[-1]
        self.assertEqual((code, dis["clean"], dis["zero_latency_s"]), (0, False, 130.0))   # zero later than Z
        code, _ = self.main(self.args(command=(self.mdf(3), "convert", "x")), FakeProv())
        self.assertEqual((code, self.rows("disarm")[-1]["clean"]), (3, False))
        code, _ = self.main(self.args("--fault", "crash-after-first", command=(self.mdf(), "bench", "run")), FakeProv())
        self.assertFalse(self.rows("disarm")[-1]["clean"])

    def test_no_zero_within_600_s_is_a_stop_and_leaves_an_orphan(self):
        prov = FakeProv(zero_after=None)
        code, lines = self.main(self.args(), prov)
        self.assertEqual(code, 4)
        self.assertEqual(self.rows("disarm"), [])
        self.assertGreaterEqual(prov.calls.count("disarm"), 2)                  # tried again
        s = serverspend.summary(spend.read(self.ledger), "runpod", self.ledger)
        self.assertEqual(s["orphans"], [self.rows("arm")[0]["arm_id"]])
        self.assertEqual(s["total_cents"], "400.00")                            # the provider at its cap

    def test_disarm_after_an_arm_that_fails(self):
        prov = FakeProv(arm_error="HTTP 500")
        code, _ = self.main(self.args(), prov)
        self.assertEqual(code, 4)
        self.assertEqual(prov.calls, [("arm", "", False, pv.DEFAULT_LEVERS), "disarm"])   # N5b step 8
        self.assertFalse(self.rows("disarm")[0]["clean"])

    def test_term_t_stops_the_command(self):
        prov = FakeProv()
        t0 = time.monotonic()
        code, _ = self.main(self.args("--window-seconds", "2", command=("sleep", "60")), prov,
                            clock=_RealClock())
        self.assertEqual(code, 5)
        self.assertLess(time.monotonic() - t0, 25)
        self.assertEqual(self.rows("disarm")[0]["exit"], -15)
        self.assertIn("disarm", prov.calls)

    def test_child_sees_the_session_and_no_key_nor_lock(self):
        probe = self.dir / "probe.json"
        code = ("import json, os, sys\nfds = []\nfor f in os.listdir('/proc/self/fd'):\n"
                " try: fds.append(os.readlink('/proc/self/fd/' + f))\n except OSError: pass\n"
                "from markdown_for_llms.bench import serverspend as s;"
                "a = os.environ.get('MDF_SESSION_ARM');"
                "json.dump({'arm': a, 'keys': [k for k in os.environ if k in %r], 'locks': [f for f in fds if '.lock' in f],"
                "'alive': s.alive(sys.argv[1], a)}, open(sys.argv[2], 'w'))" % (pv.KEYS,))
        prov = FakeProv()
        environ = dict(os.environ, RUNPOD_API_KEY="rp-secret-value", DATALAB_API_KEY="dl-secret")
        a = session.parser().parse_args(self.args(command=(sys.executable, "-I", "-c", code, str(self.ledger),
                                                           str(probe))))
        a.command = a.command[1:]
        clock = Clock()
        self.assertEqual(session.Session(a, prov, clock=clock, sleep=clock.sleep, environ=environ,
                                         out=io.StringIO()).run(), 0)
        seen = json.loads(probe.read_text())
        self.assertEqual(seen["arm"], self.rows("arm")[0]["arm_id"])
        self.assertEqual((seen["keys"], seen["locks"]), ([], []))              # the lock is never inherited
        self.assertTrue(seen["alive"])                                          # held by the session itself

    def test_runpod_drivers_after_the_double_dash(self):
        a = session.parser().parse_args(self.args(command=("probe-valid",)))
        a.command = a.command[1:]
        cmd = session.Session(a, FakeProv(), out=io.StringIO()).command()
        self.assertEqual(cmd[3:], [str(pv.HERE / "runpod_endpoint.py"), "--env-file", str(self.env), "probe-valid",
                                   "--name", "mdf-surya-fault", "--spend-ledger", str(self.ledger)])

    def test_the_levers_go_into_the_arm_and_its_row(self):
        """MFL033 N5b: `--levers` checked before the row, written in it, passed to the arm."""
        prov = FakeProv()
        code, _ = self.main(self.args("--levers", "skip-mm-profiling"), prov)
        self.assertEqual(code, 0)
        self.assertEqual(self.rows("arm")[-1]["levers"], "skip-mm-profiling")
        self.assertIn(("arm", "", False, "skip-mm-profiling"), prov.calls)
        prov = FakeProv()
        code, _ = self.main(self.args("--levers", "none"), prov)
        self.assertNotIn("levers", self.rows("arm")[-1])
        self.assertIn(("arm", "", False), prov.calls)

    def test_without_levers_an_arm_takes_the_levers_adopted(self):
        """MFL033 N5b step 8: the cache of torch.compile of both providers, by SHA-256; a disarmed activity
        takes none and is not refused for them."""
        prov = FakeProv()
        code, _ = self.main(self.args(), prov)
        self.assertEqual(code, 0)
        self.assertEqual(self.rows("arm")[-1]["levers"], pv.DEFAULT_LEVERS)
        self.assertIn(("arm", "", False, pv.DEFAULT_LEVERS), prov.calls)
        self.assertEqual(pv.levers_text(pv.DEFAULT_LEVERS), pv.DEFAULT_LEVERS)
        self.assertNotIn("skip-mm-profiling", pv.DEFAULT_LEVERS)          # lever 7 gave no gain
        prov = FakeProv()
        code, _ = self.main(self.args("--disarmed"), prov)
        self.assertEqual(code, 0)
        self.assertFalse([c for c in prov.calls if c[0] == "arm"])

    def test_the_levers_adopted_hold_an_archive_for_each_gpu_populated(self):
        """MFL033 N6: Modal serves its type A10 with an NVIDIA A10 or an NVIDIA A10G, and an archive serves only
        the GPU of its manifest; the A10G one (window 20261121T155237Z-3107a62b), checked on 21/11 by a cold
        start on an A10G (window 20261121T155812Z-fe8c2ca4: the A10 one tried, `another_gpu`, then the A10G one
        used); the watchdog tries them in the order of `watchdog.levers` (the values sorted), as in that start."""
        sys.path.insert(0, str(ROOT / "scripts" / "servers"))
        import watchdog
        self.assertEqual(watchdog.cache_shas(watchdog.levers(pv.DEFAULT_LEVERS)), [
            "29012f5b05f56b51698c99f6d5f573d3511ae06191946be983554a8b35a55b2b",       # Modal, NVIDIA A10
            "cd31ce211155dae8d34230a9988f7fd82e855c54b94dc4e79607a7bc93608224",       # Modal, NVIDIA A10G
            "d38180544831bf3c0d1a45712e25f125ff4d18a9929012376f844a2224e0ff73"])      # RunPod, RTX A5000

    def test_an_unknown_lever_or_a_disarmed_activity_is_refused(self):
        n = len(self.rows())
        for extra in (("--levers", "nope"), ("--levers", "skip-mm-profiling", "--disarmed")):
            with self.subTest(extra=extra):
                prov = FakeProv()
                code, out = self.main(self.args(*extra), prov)
                self.assertEqual(code, 3)
                self.assertEqual(out[-1]["code"], "invalid_setting")
                self.assertEqual(prov.calls, [])
        self.assertEqual(len(self.rows()), n)

    def test_mdf_convert_after_the_options_of_mdf_is_a_command_of_the_engine(self):
        """MFL033 c8.6 (21/11, window af24d9ab): `mdf --env-file F convert` is the form since c7.18; S-NB1 of
        the final review: `<python> [-I] [-B] -m markdown_for_llms.cli bench run` too, the form of the guide."""
        a = session.parser().parse_args(self.args())
        ses = session.Session(a, FakeProv(), out=io.StringIO())
        for cmd, engine in (([".venv-dev/bin/mdf", "convert", "x"], True),
                            ([".venv-dev/bin/mdf", "--env-file", ".env", "convert", "x"], True),
                            (["mdf", "--env-file=.env", "--model-root", "m", "bench", "run"], True),
                            (["mdf", "--env-file", ".env"], False), (["mdf", "serve"], False),
                            (["python", "convert"], False), ([], False),
                            # S-NB1: the module, as the guides write it
                            ([".venv-dev/bin/python", "-I", "-B", "-m", "markdown_for_llms.cli", "bench", "run"], True),
                            (["python3", "-m", "markdown_for_llms.cli", "--env-file", ".env", "convert", "x"], True),
                            (["/usr/bin/python3.12", "-IB", "-m", "markdown_for_llms.cli", "convert", "x"], True),
                            ([".venv-dev/bin/python", "-I", "-B", "-m", "markdown_for_llms.cli", "serve"], False),
                            ([".venv-dev/bin/python", "-m", "markdown_for_llms.bench.cli", "run"], False),
                            ([".venv-dev/bin/python", "-c", "markdown_for_llms.cli", "convert"], False),
                            (["python", "-u", "-m", "markdown_for_llms.cli", "convert"], False),
                            (["sh", "-c", "mdf convert x"], False)):
            with self.subTest(cmd=cmd):
                self.assertEqual(ses.engine_command(cmd), engine)

    def test_concurrency_is_a_runpod_driver_too(self):
        """MFL033 N5b: `concurrency --input PNG ...` after the double dash runs RunPod's driver."""
        a = session.parser().parse_args(self.args(command=("concurrency", "--input", "p.png", "--n", "32")))
        a.command = a.command[1:]
        cmd = session.Session(a, FakeProv(), out=io.StringIO()).command()
        self.assertEqual(cmd[3:], [str(pv.HERE / "runpod_endpoint.py"), "--env-file", str(self.env), "concurrency",
                                   "--name", "mdf-surya-fault", "--spend-ledger", str(self.ledger), "--input", "p.png",
                                   "--n", "32"])


class LogSource:
    """One worker whose log stream gives `lines` once, then is gone."""

    def __init__(self, lines):
        self.lines, self.done = lines, False

    def workers(self):
        return [{"id": "w1", "status": "RUNNING", "gpu": "NVIDIA RTX A5000", "data_center": "EU-SE-1"}]

    def stream(self, wid, since, last_id):
        if self.done:
            raise worker_logs.Gone(wid)
        self.done = True
        for i, line in enumerate(self.lines):
            yield "i%d" % i, {"source": "container", "line": line, "ts": "t%d" % i}


class LogProv(FakeProv):
    def __init__(self, lines=(), fail=False, **kw):
        super().__init__(**kw)
        self.lines, self.fail = list(lines), fail

    def log_follower(self, path, t_arm, clock=time.time):
        if self.fail:
            raise pv.ProviderError("RunPod GET /serverless: HTTP 502")
        return worker_logs.Follower(LogSource(self.lines), path, t_arm, clock=clock, list_every=0.05)


# the wrapped command waits for the first line of the proxy in the live log, as the populate does
WAIT_LOG = ("import json, os, sys, time\np = os.environ.get('SESSION_WORKER_LOG')\nseen = False\n"
            "for _ in range(500):\n"
            "    if p and os.path.exists(p) and 'mdf_start' in open(p).read():\n        seen = True\n        break\n"
            "    time.sleep(0.02)\n"
            "json.dump({'path': p, 'seen': seen, 'mdf': sorted(k for k in os.environ if k.startswith('MDF_'))},"
            " open(sys.argv[1], 'w'))\n")


class TestWorkerLog(Base):
    """MFL033 c7.4: the logs of the worker followed in every RunPod window, scanned, then saved."""

    def run_logged(self, prov, *extra):
        probe = self.dir / "probe.json"
        code, lines = self.main(self.args(*extra, command=(sys.executable, "-I", "-c", WAIT_LOG, str(probe))), prov)
        return code, lines, (json.loads(probe.read_text()) if probe.exists() else None)

    def test_saved_after_the_scan(self):
        logs = self.dir / "n4"
        prov = LogProv(['{"event": "mdf_start", "mono": 5.0}', '{"event": "mdf_download_start", "mono": 6.0}'],
                       zero_after=1)
        code, lines, probe = self.run_logged(prov, "--allow-download", "--worker-log-dir", str(logs))
        self.assertEqual(code, 0, lines)
        self.assertTrue(probe["seen"])                                          # the live file, for the driver
        # Settings refuses an unknown MDF_ variable: the session adds MDF_SESSION_ARM only (mdf convert in N5)
        self.assertEqual(sorted(set(probe["mdf"]) - {k for k in os.environ if k.startswith("MDF_")}),
                         ["MDF_SESSION_ARM"])
        live = Path(probe["path"])
        self.assertTrue(live.name == "worker-log.partial" and live.parent.name.startswith("mdf-wlog-"))
        self.assertFalse(live.parent.exists())                                  # removed after the save
        arm_id = self.rows("arm")[0]["arm_id"]
        saved = logs / arm_id / "worker-log.txt"
        text = saved.read_text()
        self.assertIn('"mdf_download_start"', text)
        self.assertIn('"status": "RUNNING"', text)                              # the worker as read by the session
        dis, = self.rows("disarm")
        self.assertEqual(dis["worker_log"]["saved"], str(saved))
        self.assertEqual(dis["worker_log"]["sha256"], __import__("hashlib").sha256(saved.read_bytes()).hexdigest())
        self.assertEqual(dis["worker_log"]["workers"], ["w1"])
        self.assertEqual(lines[-1]["worker_log"]["saved"], str(saved))

    def test_the_gpu_in_the_report_of_the_window(self):
        """c7.14: the report of the window and its `disarm` row name the GPU each proxy found."""
        gpu = '{"event": "mdf_gpu", "names": ["NVIDIA A10G"], "source": "nvidia-smi", "error": null}'
        code, lines, _ = self.run_logged(LogProv(['{"event": "mdf_start"}', gpu], zero_after=1),
                                         "--worker-log-dir", str(self.dir / "n4"))
        self.assertEqual(code, 0, lines)
        dis, = self.rows("disarm")
        self.assertEqual([(g["worker"], g["names"], g["source"]) for g in dis["worker_log"]["gpus"]],
                         [("w1", ["NVIDIA A10G"], "nvidia-smi")])
        self.assertEqual(lines[-1]["worker_log"]["gpus"], dis["worker_log"]["gpus"])

    def test_a_key_in_the_log_is_not_saved(self):
        logs = self.dir / "n4"
        prov = LogProv(['{"event": "mdf_start"}', "echo rp-secret-value"], zero_after=1)
        code, lines, probe = self.run_logged(prov, "--worker-log-dir", str(logs))
        self.assertEqual(code, 4)
        self.assertFalse(logs.exists())
        self.assertFalse(Path(probe["path"]).parent.exists())
        dis, = self.rows("disarm")                                              # the window is closed all the same
        self.assertIsNone(dis["worker_log"]["saved"])
        self.assertEqual(lines[-1]["worker_log"]["secret_found"], 1)
        self.assertNotIn("rp-secret-value", json.dumps(lines))

    def test_default_directory_next_to_the_ledger(self):
        code, lines, probe = self.run_logged(LogProv(['{"event": "mdf_start"}'], zero_after=1))
        self.assertEqual(code, 0, lines)
        arm_id = self.rows("arm")[0]["arm_id"]
        self.assertTrue((self.dir / "worker-logs" / arm_id / "worker-log.txt").is_file())

    def test_an_error_while_saving_never_skips_the_disarm_row(self):
        prov = LogProv(['{"event": "mdf_start"}'], zero_after=1)
        logs = self.dir / "n4"
        logs.write_text("a file where the directory should be")
        code, lines, probe = self.run_logged(prov, "--worker-log-dir", str(logs))
        self.assertEqual(code, 4)
        dis, = self.rows("disarm")
        self.assertIsNone(dis["worker_log"]["saved"])
        self.assertEqual(lines[-1]["worker_log"]["error"], "NotADirectoryError")
        self.assertFalse(Path(probe["path"]).parent.exists())
        self.assertEqual(self.locks(), [])

    def test_no_follower_no_command(self):
        prov = LogProv(fail=True)
        code, lines, probe = self.run_logged(prov)
        self.assertEqual(code, 4)
        self.assertIsNone(probe)                                                # the command never ran
        self.assertEqual(prov.calls, [("arm", "", False, pv.DEFAULT_LEVERS), "disarm"])   # N5b step 8
        dis, = self.rows("disarm")
        self.assertEqual((dis["clean"], dis["exit"]), (False, None))
        self.assertNotIn("worker_log", dis)


class _RealClock:
    def __call__(self):
        return time.time()

    def sleep(self, s):
        time.sleep(min(s, 0.2))


class TestActivities(Base):
    def test_disarmed_reserves_the_cap_and_arms_nothing(self):
        prov = FakeProv()
        code, lines = self.main(self.args("--disarmed", cap="0.25"), prov)
        self.assertEqual(code, 0)
        act, = self.rows("activity")
        self.assertEqual(Decimal(act["reserve_cents"]), Decimal("25.00"))
        self.assertEqual((prov.calls, self.rows("arm")), ([], []))
        code, lines = self.main(self.args("--disarmed", cap="0.25"), FakeProv(armed=True))
        self.assertEqual((code, lines[-1]["code"]), (3, "provider_not_idle"))

    POPULATE = ("uv", "tool", "run", "--from", "modal==1.6.3", "modal", "run",
                str(ROOT / "scripts" / "servers" / "modal_populate.py") + "::populate")

    def cpu(self, cap, command=POPULATE):
        """A `--cpu-activity` of Modal whose command never runs (a fake Popen)."""
        from unittest import mock
        argv = ["run", "--provider", "modal", "--name", "mdf-surya", "--env-file", str(self.env), "--spend-ledger",
                str(self.ledger), "--spend-cap-usd", "15", "--cap-usd", cap, "--cpu-activity", "--", *command]
        started = []

        def popen(cmd, **kw):
            started.append(cmd)
            return mock.Mock(poll=mock.Mock(return_value=0), returncode=0, pid=1)
        prov = FakeProv()
        prov.provider = "modal"
        with mock.patch.object(session.subprocess, "Popen", popen):
            code, lines = self.main(argv, prov)
        return code, lines, started

    def test_cpu_activity_rate(self):
        code, lines, started = self.cpu("0.10")
        self.assertEqual((code, len(started)), (0, 1), lines)
        self.assertEqual(Decimal(self.rows("activity")[0]["rate"]), pv.RATES["modal-cpu"])
        limit, = [e for e in lines[-1]["events"] if e["event"] == "activity-limit"]
        self.assertEqual((limit["timeout_s"], limit["limit_cents"]), (1800, "6.32"))      # 1800 s x 0.00003508 USD/s

    def test_a_cpu_activity_needs_a_reserve_over_the_timeout_of_the_populate(self):
        """S-B3 of the final review of MFL033, the case of the probe of Sol: a reserve of one cent for the
        populate, whose Function runs up to 1800 s; refused before the row and before the command."""
        for cap in ("0.01", "0.06"):
            with self.subTest(cap=cap):
                code, lines, started = self.cpu(cap)
                self.assertEqual((code, lines[-1]["code"], started), (3, "server_cap", []))
                self.assertIn("1800 s", lines[-1]["detail"])
                self.assertEqual(self.rows("activity"), [])
        code, lines, started = self.cpu("0.07")
        self.assertEqual((code, len(started)), (0, 1))
        self.assertEqual(Decimal(self.rows("activity")[0]["reserve_cents"]), Decimal("7.00"))

    def test_a_cpu_activity_runs_only_the_populate(self):
        others = (("true",), ("modal", "run", "elsewhere/modal_populate.py::populate"),
                  ("modal", "deploy", str(ROOT / "scripts" / "servers" / "modal_populate.py")),
                  ("python3", str(ROOT / "scripts" / "servers" / "modal_populate.py") + "::populate"))
        for command in others:
            with self.subTest(command=command):
                code, lines, started = self.cpu("0.10", command)
                self.assertEqual((code, lines[-1]["code"], started), (3, "invalid_setting", []))
                self.assertEqual(self.rows("activity"), [])

    def test_the_timeout_comes_from_the_source_of_the_populate(self):
        import ast
        tree = ast.parse((ROOT / "scripts" / "servers" / "modal_populate.py").read_text())
        literal = [n.value.value for n in tree.body if isinstance(n, ast.Assign)
                   and [t.id for t in n.targets if isinstance(t, ast.Name)] == ["TIMEOUT_S"]]
        self.assertEqual(pv.modal_populate_timeout(), literal[0])
        for text in ("@app.function(timeout=600)\ndef populate():\n    pass\n",          # retries not 0
                     "@app.function(timeout=600, retries=2)\ndef populate():\n    pass\n",
                     "@app.function(retries=0)\ndef populate():\n    pass\n"):
            f = self.dir / "modal_populate.py"
            f.write_text(text)
            with self.assertRaises(pv.ProviderError):
                pv.modal_populate_timeout(f)


class TestStatusDisarm(Base):
    def status(self, prov):
        return self.main(["status", "--provider", "runpod", "--name", "mdf-surya-fault", "--env-file", str(self.env),
                          "--spend-ledger", str(self.ledger)], prov)

    def disarm(self, prov):
        return self.main(["disarm", "--provider", "runpod", "--name", "mdf-surya-fault", "--env-file", str(self.env),
                          "--spend-ledger", str(self.ledger)], prov)

    def test_status_reads_only(self):
        code, lines = self.status(FakeProv())
        self.assertEqual(code, 0)
        self.assertFalse(self.ledger.exists())
        self.assertEqual(lines[0]["balance_usd"], 9.37)
        self.assertIn("session.py disarm", lines[0]["commands"][0])

    def test_orphan_listed_disarmed_and_released_only_by_hand(self):
        row = self.orphan_arm()
        code, lines = self.status(FakeProv(armed=True))
        self.assertEqual(code, 3)
        self.assertEqual([w["arm_id"] for w in lines[0]["orphan"]], [row["arm_id"]])
        prov = FakeProv(armed=True, busy=True, zero_after=3)
        code, lines = self.disarm(prov)
        self.assertEqual(code, 0)
        self.assertEqual(lines[0]["orphan_closed"], [row["arm_id"]])
        dis, = self.rows("disarm")
        self.assertTrue(dis["orphan"])
        entries = spend.read(self.ledger)
        self.assertEqual(serverspend.summary(entries, "runpod", self.ledger)["total_cents"], "400.00")
        with spend.Ledger.file(self.ledger).locked() as locked:
            serverspend.reconcile(locked, self.ledger, "runpod", "12", manual=True, arms=[row["arm_id"]], orphan=True)
        self.assertEqual(serverspend.summary(spend.read(self.ledger), "runpod", self.ledger)["total_cents"], "12.00")
        code, _ = self.status(FakeProv())
        self.assertEqual(code, 0)

    def test_disarm_refused_under_a_live_session(self):
        row = self.orphan_arm()
        fd = os.open(serverspend.lock_path(self.ledger, row["arm_id"]), os.O_RDWR | os.O_CREAT, 0o600)
        fcntl.flock(fd, fcntl.LOCK_EX)
        try:
            code, lines = self.disarm(FakeProv(armed=True))
        finally:
            os.close(fd)
        self.assertEqual((code, lines[-1]["code"]), (3, "provider_not_idle"))


class TestTermDuringTheArm(Base):
    """S-B4 of the final review of MFL033: the term T holds during the arm."""

    def test_an_arm_that_ends_after_t_starts_no_command(self):
        """The case of the probe of Sol: the arm of the provider takes the clock past T (it ignores the
        deadline it is given); the command is not started, the session disarms and exits with 5."""
        from unittest import mock
        clock = Clock()
        prov = FakeProv()
        given = []
        original = prov.arm

        def slow(fault, allow_download, deadline=None, **levers):
            given.append(deadline)
            clock.t = deadline + 59                       # past T
            return original(fault, allow_download, **levers)
        prov.arm = slow
        with mock.patch.object(session.subprocess, "Popen") as popen:
            code, lines = self.main(self.args(), prov, clock=clock)
        self.assertEqual(code, 5)
        popen.assert_not_called()
        arm, = self.rows("arm")
        self.assertEqual(given, [float(arm["until"])])                    # the provider got T
        self.assertIn("disarm", prov.calls)
        dis, = self.rows("disarm")
        self.assertFalse(dis["clean"])
        self.assertIn("term T before the command", [e.get("reason") for e in lines[-1]["events"]])

    def test_the_alarm_at_t_interrupts_an_arm_on_the_real_clock(self):
        """An arm stuck past T (a provider that ignores its deadline) ends at T by the alarm of the
        session: no command, the disarm, exit 5, within seconds."""
        from unittest import mock
        prov = FakeProv()

        def stuck(fault, allow_download, deadline=None, **levers):
            prov.calls.append("arm")
            prov.armed = prov.busy = True
            time.sleep(30)
        prov.arm = stuck
        argv = self.args("--window-seconds", "2")
        a = session.parser().parse_args(argv)
        a.command = a.command[1:] if a.command[:1] == ["--"] else a.command
        out = io.StringIO()
        ses = session.Session(a, prov, out=out)                         # the real clock
        started = time.monotonic()
        try:
            with mock.patch.object(session.subprocess, "Popen") as popen:
                code = ses.run()
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGALRM):
                signal.signal(sig, signal.SIG_DFL if sig != signal.SIGINT else signal.default_int_handler)
        self.assertLess(time.monotonic() - started, 15)
        self.assertEqual(code, 5)
        popen.assert_not_called()
        self.assertEqual(prov.calls[:2], ["arm", "disarm"])
        report = json.loads(out.getvalue().splitlines()[-1])
        self.assertIn({"event": "arm-interrupted", "reason": "term T"},
                      [{k: e[k] for k in ("event", "reason")} for e in report["events"] if e["event"] == "arm-interrupted"])


class TestProcesses(Base):
    """The session as a process: SIGKILL, parent death before prctl, SIGTERM, normal close."""

    def start(self, delay, command, state=None):
        self.state = self.dir / "provider.json"
        self.state.write_text(json.dumps(state or {}))
        argv = [sys.executable, "-I", "-B", str(DRIVER), str(self.state), str(delay)] + self.args(command=command)
        return subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    def wait_for(self, path, seconds=20):
        t0 = time.monotonic()
        while not path.exists():
            if time.monotonic() - t0 > seconds:
                self.fail("no %s" % path.name)
            time.sleep(0.05)

    def child(self, marker, extra=""):
        return (sys.executable, "-I", "-c", "import os, sys, time; open(sys.argv[1], 'w').write(str(os.getpid()));"
                + extra + "time.sleep(120)", str(marker))

    def test_sigkill_of_the_session_kills_the_command_and_leaves_an_orphan(self):
        marker = self.dir / "pid"
        ses = self.start(0, self.child(marker))
        self.wait_for(marker)
        pid = int(marker.read_text())
        ses.kill()
        ses.wait()
        t0 = time.monotonic()
        while _alive(pid) and time.monotonic() - t0 < 10:
            time.sleep(0.1)
        self.assertFalse(_alive(pid))                                            # PR_SET_PDEATHSIG
        s = serverspend.summary(spend.read(self.ledger), "runpod", self.ledger)
        self.assertEqual(len(s["orphans"]), 1)
        again = subprocess.run([sys.executable, "-I", "-B", str(DRIVER), str(self.state), "0"] + self.args(),
                               capture_output=True, text=True, timeout=60)
        self.assertEqual(again.returncode, 3)                                    # provider still armed: orphan first
        self.assertEqual(json.loads(again.stdout.splitlines()[-1])["code"], "session_orphaned")

    def test_parent_dead_before_prctl(self):
        marker = self.dir / "ran"
        ses = self.start(3.0, (sys.executable, "-I", "-c", "open(%r, 'w').write('ran')" % str(marker)))
        t0 = time.monotonic()
        while not self.rows("arm") and time.monotonic() - t0 < 20:
            time.sleep(0.05)
        time.sleep(0.8)                                                          # inside the delay before prctl
        ses.kill()
        ses.wait()
        time.sleep(4.0)
        self.assertFalse(marker.exists())                                        # getppid() changed: the child left

    def test_sigterm_stops_disarms_and_records(self):
        marker = self.dir / "pid"
        ses = self.start(0, self.child(marker))
        self.wait_for(marker)
        pid = int(marker.read_text())
        ses.send_signal(signal.SIGTERM)
        out, _ = ses.communicate(timeout=60)
        self.assertEqual(ses.returncode, 130)
        self.assertFalse(_alive(pid))
        dis, = self.rows("disarm")
        self.assertEqual((dis["signal"], dis["clean"]), ("SIGTERM", False))
        self.assertEqual(json.loads(self.state.read_text())["calls"][:2], ["arm", "disarm"])

    def test_sigterm_during_the_arm_stops_it_disarms_and_starts_no_command(self):
        """S-B4 of the final review of MFL033: the PATCH accepted, its reading back slow (30 s); SIGTERM
        then ends the arm at once, no command starts, the session disarms, reads the zero and records."""
        marker = self.dir / "ran"
        ses = self.start(0, (sys.executable, "-I", "-c", "open(%r, 'w').write('ran')" % str(marker)),
                         state={"arm_sleep": 30})
        t0 = time.monotonic()
        while "arm" not in json.loads(self.state.read_text() or "{}").get("calls", []):
            if time.monotonic() - t0 > 20:
                self.fail("no arm")
            time.sleep(0.05)
        time.sleep(0.3)
        sent = time.monotonic()
        ses.send_signal(signal.SIGTERM)
        out, _ = ses.communicate(timeout=60)
        self.assertLess(time.monotonic() - sent, 10)                             # not the 30 s of the arm
        self.assertEqual(ses.returncode, 130)
        self.assertFalse(marker.exists())
        calls = json.loads(self.state.read_text())["calls"]
        self.assertEqual(calls[:2], ["arm", "disarm"])
        self.assertNotIn("arm-read", calls)
        dis, = self.rows("disarm")
        self.assertEqual((dis["signal"], dis["clean"]), ("SIGTERM", False))
        report = json.loads(out.splitlines()[-1])
        self.assertIn("arm-interrupted", [e["event"] for e in report["events"]])

    def test_normal_close_while_the_command_ends(self):
        ses = self.start(0, (sys.executable, "-I", "-c", "import time; time.sleep(1.0)"))
        ses.communicate(timeout=60)
        self.assertEqual(ses.returncode, 0)
        dis, = self.rows("disarm")
        self.assertEqual(dis["exit"], 0)
        self.assertEqual(self.locks(), [])
        self.assertEqual(serverspend.summary(spend.read(self.ledger), "runpod", self.ledger)["orphans"], [])


def _alive(pid):
    try:
        with open("/proc/%d/stat" % pid) as fh:
            return fh.read().split(")")[-1].split()[0] != "Z"
    except OSError:
        return False


class TestRefusalOutput(Base):
    def test_main_prints_the_commands_and_never_the_key(self):
        self.orphan_arm()
        out = io.StringIO()
        code = session.main(self.args(), factory=lambda: FakeProv(), out=out)
        self.assertEqual(code, 3)
        text = out.getvalue()
        self.assertIn("session_orphaned", text)
        self.assertIn("mdf bench spend reconcile", text)
        self.assertNotIn("rp-secret-value", text)


if __name__ == "__main__":
    unittest.main()


class ListingSource:
    """Workers listed as successive answers (the last repeats); each worker's stream gives its own
    events once, then the worker is gone."""

    def __init__(self, lists, events):
        self.lists, self.events, self.done = list(lists), dict(events), set()

    def workers(self):
        return self.lists.pop(0) if len(self.lists) > 1 else self.lists[0]

    def stream(self, wid, since, last_id):
        if wid in self.done:
            raise worker_logs.Gone(wid)
        self.done.add(wid)
        for i, (source, line) in enumerate(self.events.get(wid, [])):
            yield "%s-%d" % (wid, i), {"source": source, "line": line, "ts": ""}


class BoundProv(FakeProv):
    def __init__(self, lists, events, readings=None, **kw):
        super().__init__(**kw)
        self.src = ListingSource(lists, events)
        self.readings = list(readings or [])

    def zero(self):
        done, state = super().zero()
        if self.readings:
            state = self.readings.pop(0)
        return done, state

    def log_follower(self, path, t_arm, clock=time.time):
        return worker_logs.Follower(self.src, path, t_arm, clock=clock, list_every=0.05)


class TestBound(Base):
    """MFL033 c7.19 (prompt N4s §4.2): the `disarm` row of a RunPod window holds `t_stop`, the last
    "stop container" of the worker log, and `double_s`, the seconds counted twice when two workers
    were RUNNING at once; the bound of the window runs to the later of the zero and the stop, plus
    the 30 s of the tail."""

    W = {"gpu": "NVIDIA RTX A5000", "data_center": "EU-SE-1", "uptime": None}

    def window(self, prov, clock):
        code, lines = self.main(self.args(command=(sys.executable, "-I", "-c", "import time; time.sleep(0.6)")), prov, clock)
        arm, = self.rows("arm")
        dis, = self.rows("disarm")
        amount = serverspend.summary(spend.read(self.ledger), "runpod", self.ledger)["windows"][0]["amount_cents"]
        return code, arm, dis, Decimal(amount)

    def test_the_stop_of_the_container_and_the_tail(self):
        clock = Clock()
        prov = BoundProv([[dict(self.W, id="w1", status="RUNNING")]],
                         {"w1": [("system", "create container abc"), ("system", "stop container abc"),
                                 ("system", "remove container")]}, zero_after=1)
        code, arm, dis, amount = self.window(prov, clock)
        self.assertEqual(code, 0)
        t_arm, t_zero = float(arm["t_arm"]), float(dis["t_zero"])
        self.assertIsNotNone(dis["t_stop"])
        self.assertEqual(dis["double_s"], 0.0)
        end = max(t_zero, float(dis["t_stop"])) + 30
        self.assertEqual(amount, (Decimal(str(end - t_arm)) * RATE * 100).quantize(Decimal("0.01"), rounding="ROUND_CEILING"))
        self.assertGreater(amount, (Decimal(str(t_zero - t_arm)) * RATE * 100).quantize(Decimal("0.01"), rounding="ROUND_CEILING"))

    def test_two_workers_running_count_twice(self):
        clock = Clock()
        lists = [[dict(self.W, id="w1", status="RUNNING")],
                 [dict(self.W, id="w1", status="RUNNING"), dict(self.W, id="w2", status="RUNNING")]]
        prov = BoundProv(lists, {}, zero_after=1)
        code, arm, dis, amount = self.window(prov, clock)
        text = (self.dir / "worker-logs" / arm["arm_id"] / "worker-log.txt").read_text()
        overlap = worker_logs.Workers().feed(self.dir / "worker-logs" / arm["arm_id"] / "worker-log.txt").overlaps
        self.assertTrue(overlap, text)
        t_arm, t_zero = float(arm["t_arm"]), float(dis["t_zero"])
        end = t_zero + 30
        start = max(t_arm, t_arm + overlap[0]["since_arm"] - worker_logs.LIST_EVERY_S)
        self.assertAlmostEqual(dis["double_s"], end - start, places=2)
        self.assertEqual(amount, (Decimal(str(end - t_arm + dis["double_s"])) * RATE * 100).quantize(
            Decimal("0.01"), rounding="ROUND_CEILING"))

    def test_two_workers_running_in_a_reading_of_the_zero(self):
        clock = Clock()
        two = {"armed": False, "busy": True, "summary": {"total": 2, "running": 2}, "workers": []}
        prov = BoundProv([[]], {}, readings=[two], zero_after=1)
        code, arm, dis, amount = self.window(prov, clock)
        first = dis["readings"][0]["t"]
        self.assertAlmostEqual(dis["double_s"], float(dis["t_zero"]) + 30 - (first - worker_logs.LIST_EVERY_S), places=2)
