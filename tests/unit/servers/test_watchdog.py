"""The proxy in front of vLLM (`scripts/servers/watchdog.py`, MFL033 plan r4 App. C.2).

The rules and the inert mode run here on a fake clock; the sockets (forwarding, limits, terms,
trips, faults) run in `watchdog_driver.py`, a subprocess, because the guard of `conftest.py`
refuses `connect` in the pytest process.
"""
import contextlib
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import threading
import types
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[3]
DRIVER = Path(__file__).resolve().parent / "watchdog_driver.py"


def load():
    spec = importlib.util.spec_from_file_location("mdf_watchdog_t", ROOT / "scripts" / "servers" / "watchdog.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


wd = load()


class Clock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t


class FakeProcess:
    def __init__(self):
        self.signals = []

    def signal(self, sig):
        self.signals.append(sig)


def rules(**over):
    cfg = dict(wd.RULES, **wd.DEFAULTS["runpod"])
    cfg.update(over)
    return wd.Rules(cfg, 0.0)


class TestRules(unittest.TestCase):
    def run_until(self, r, health, metrics, inflight, until, step=5.0, start=0.0):
        t = start
        while t <= until:
            rule = r.step(t, health(t), metrics(t), inflight(t))
            if rule:
                return rule, t
            t += step
        return None, t

    def test_w1_never_healthy(self):
        rule, t = self.run_until(rules(), lambda t: False, lambda t: None, lambda t: 0, 2000)
        self.assertEqual((rule, t), ("W1", 900.0))

    def test_w2_unhealthy_after_readiness(self):
        r = rules()
        rule, t = self.run_until(r, lambda t: t < 100, lambda t: {"tokens": 1}, lambda t: 0, 2000)
        self.assertEqual(rule, "W2")
        self.assertEqual(t, 160.0)

    def test_w3_work_and_no_token(self):
        r = rules()
        rule, t = self.run_until(r, lambda t: True, lambda t: {"tokens": 5, "running": 1 if t >= 50 else 0},
                                 lambda t: 0, 2000)
        self.assertEqual((rule, t), ("W3", 165.0))     # from the last reading without work

    def test_w3_counts_the_requests_of_the_proxy(self):
        rule, t = self.run_until(rules(), lambda t: True, lambda t: {"tokens": 5}, lambda t: 1 if t >= 20 else 0, 2000)
        self.assertEqual((rule, t), ("W3", 135.0))

    def test_tokens_keep_w3_away(self):
        rule, _ = self.run_until(rules(w4_s=10_000), lambda t: True, lambda t: {"tokens": t, "running": 1},
                                 lambda t: 1, 3000)
        self.assertIsNone(rule)

    def test_w4_idle_from_readiness(self):
        rule, t = self.run_until(rules(), lambda t: t >= 30, lambda t: {"tokens": 5}, lambda t: 0, 2000)
        self.assertEqual((rule, t), ("W4", 630.0))

    def test_w4_resets_with_inference(self):
        rule, t = self.run_until(rules(), lambda t: True, lambda t: {"tokens": 5}, lambda t: 1 if 400 <= t < 410 else 0,
                                 3000)
        self.assertEqual((rule, t), ("W4", 1005.0))

    def test_w5_life(self):
        rule, t = self.run_until(rules(w4_s=99_999), lambda t: True, lambda t: {"tokens": t},
                                 lambda t: 0, 9000, step=60)
        self.assertEqual((rule, t), ("W5", 7200.0))
        self.assertEqual(wd.DEFAULTS["modal"]["w5_s"], 14400)

    def test_metrics_sum_the_label_sets(self):
        text = ('# HELP x\nvllm:num_requests_running{model="a"} 2\nvllm:num_requests_running{model="b"} 1\n'
                'vllm:generation_tokens_total{model="a"} 10.0\nvllm:num_requests_waiting 0\nother 5\n')
        self.assertEqual(wd.metrics(text), {"running": 3.0, "tokens": 10.0, "waiting": 0.0})


class TestWatchdog(unittest.TestCase):
    def make(self, fault=""):
        clock = FakeProcess()
        c = Clock()
        exits = []
        w = wd.Watchdog(wd.config({"MDF_PROVIDER": "runpod", "MDF_FAULT": fault}), clock=c, process=FakeProcess(),
                        on_exit=exits.append)
        return w, c, exits

    def test_trip_kills_and_goes_inert_then_an_hour_without_exit(self):
        w, c, exits = self.make()
        w.probe = lambda path: (None, "")                     # vLLM never healthy
        for _ in range(int(900 / 5) + 1):
            c.t += 5
            w.tick()
            w.control()
        self.assertEqual(w.inert, "mdf_inert")
        self.assertEqual(w.process.signals, [9])
        for _ in range(3600):                                 # an hour of fake clock in the inert mode
            c.t += 1
            w.control()
            self.assertIsNone(w.tick())
        self.assertEqual(exits, [])
        self.assertEqual(w.process.signals, [9])

    def test_ready_after_204(self):
        w, c, _ = self.make()
        w.probe = lambda path: (None, "")
        w.tick()
        self.assertFalse(w.ready)
        w.probe = lambda path: (200, "vllm:generation_tokens_total 1\n")
        c.t += 5
        w.tick()
        self.assertTrue(w.ready)

    def test_faults(self):
        w, _, exits = self.make("hang-after-first")
        w.after_success()
        w.after_success()
        self.assertEqual(w.process.signals, [19])
        w, _, exits = self.make("crash-after-first")
        w.after_success()
        self.assertEqual(exits, [1])
        w, _, exits = self.make("")
        w.after_success()
        self.assertEqual((exits, w.process.signals), ([], []))

    def test_request_path(self):
        # c7.6: the path an `mdf_request` names, without the query, only a known one
        for path, named in (("/v1/models", "/v1/models"), ("/v1/models?token=x", "/v1/models"),
                            ("/v1/chat/completions", "/v1/chat/completions"), ("/version", "/version"),
                            ("/metrics", "/metrics"), ("/health", "other"), ("/e500", "other"), ("", "other"),
                            (None, "other"), ("/v1/models/../x", "other")):
            self.assertEqual(wd.request_path(path), named, path)

    def test_config_by_provider(self):
        self.assertEqual(wd.config({"MDF_PROVIDER": "runpod"})["req_max_s"], 290)
        self.assertEqual(wd.config({"MDF_PROVIDER": "modal"})["req_max_s"], 130)
        cfg = wd.config({"MDF_ALLOW_DOWNLOAD": "1", "MDF_FAULT": "hang-after-first"})
        self.assertTrue(cfg["allow_download"])
        self.assertEqual((cfg["port"], cfg["upstream"]), (8000, 8001))

    def test_vllm_arguments(self):
        argv = wd.vllm_argv(wd.config({}), False)
        for flag in ("--revision", "--no-enable-log-requests", "--served-model-name"):
            self.assertIn(flag, argv)
        self.assertNotIn("--speculative-config", argv)                     # MTP only with a head (c7.5)
        self.assertNotIn("--disable-log-stats", argv)
        self.assertNotIn("--api-key", argv)
        self.assertEqual(argv[argv.index("--host") + 1], "127.0.0.1")
        self.assertEqual(argv[argv.index("--port") + 1], "8001")
        self.assertEqual((argv[argv.index("--max-num-seqs") + 1], argv[argv.index("--max-num-batched-tokens") + 1]),
                         ("32", "8192"))                                    # Surya's spawn for 24 GB
        mtp = wd.vllm_argv(wd.config({}), True)
        self.assertEqual(mtp[:len(argv)], argv)
        self.assertEqual(json.loads(mtp[mtp.index("--speculative-config") + 1]),
                         {"method": "mtp", "num_speculative_tokens": 2})
        self.assertEqual(argv[argv.index("--gpu-memory-utilization") + 1], "0.85")
        self.assertNotIn("--skip-mm-profiling", argv)

    def test_levers_of_the_start(self):
        """MFL033 N5b: a closed set in MDF_LEVERS; `skip-mm-profiling` adds the flag and lowers the memory
        of the cache to 0.80; another name is an error of the config, never a default."""
        self.assertEqual(wd.levers(""), ())
        self.assertEqual(wd.levers(" skip-mm-profiling , skip-mm-profiling"), ("skip-mm-profiling",))
        with self.assertRaises(wd.LeversInvalid):
            wd.levers("skip-mm-profiling,enforce-eager")
        cfg = wd.config({"MDF_LEVERS": "skip-mm-profiling"})
        self.assertEqual((cfg["levers"], cfg["levers_error"]), (("skip-mm-profiling",), None))
        argv = wd.vllm_argv(cfg, False)
        self.assertIn("--skip-mm-profiling", argv)
        self.assertEqual(argv[argv.index("--gpu-memory-utilization") + 1], "0.80")
        base = wd.vllm_argv(wd.config({}), False)
        self.assertEqual([a for a in argv if a not in ("--skip-mm-profiling", "0.80")],
                         [a for a in base if a != "0.85"])
        bad = wd.config({"MDF_LEVERS": "nope"})
        self.assertEqual((bad["levers"], bad["levers_error"]), ((), "unknown levers: nope"))


def events(text):
    return [json.loads(x) for x in text.splitlines() if x.startswith("{")]


class TestHealthChecks(unittest.TestCase):
    """MFL033 c7.7 (prompt N4f §3.1-§3.2): RunPod's load balancer polls `/ping` (seen on 19/11 although
    HEALTH_CHECK_PATH is /health; docs.runpod.io/serverless/load-balancing/overview, read on 19/11:
    `/ping` by default, or HEALTH_CHECK_PATH). The proxy serves both like /health and never forwards
    them; the checks are counted and summarized, never logged one by one."""

    def make(self):
        c = Clock()
        return wd.Watchdog(wd.config({"MDF_PROVIDER": "runpod"}), clock=c, process=FakeProcess()), c

    def capture(self, fn):
        import contextlib
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            fn()
        return events(err.getvalue())

    def test_the_paths_of_the_platform(self):
        self.assertEqual(set(wd.HEALTH_PATHS), {"/health", "/ping"})
        for path, served in (("/ping", "/ping"), ("/ping?x=1", "/ping"), ("/health", "/health"),
                             ("/health?probe", "/health"), ("/pingx", None), ("/ping/", None), ("/v1/models", None),
                             ("", None), (None, None)):
            self.assertEqual(wd.health_path(path), served, path)

    def test_counts_and_summaries(self):
        w, c = self.make()

        def checks(path, status, n):
            for _ in range(n):
                w.count_check(path, status)

        ev = self.capture(lambda: checks("/ping", 204, 3))
        self.assertEqual([(e["event"], e["first"], e["counts"]) for e in ev],
                         [("mdf_health_checks", ["/ping", 204], {"/ping": {"204": 1}})])    # the first of a status
        ev = self.capture(lambda: checks("/health", 204, 1))
        self.assertEqual(ev[0]["counts"], {"/health": {"204": 1}, "/ping": {"204": 3}})
        c.t += 10
        self.assertEqual(self.capture(w.control), [])                         # within the period
        c.t += 25
        self.assertEqual(self.capture(w.control), [])                         # nothing new since the last line
        checks("/ping", 204, 2)
        c.t += 5
        ev = self.capture(w.control)
        self.assertEqual([(e["event"], e["first"], e["counts"], e["ready"], e["inert"]) for e in ev],
                         [("mdf_health_checks", None, {"/health": {"204": 1}, "/ping": {"204": 5}}, False, None)])
        self.assertEqual(self.capture(w.control), [])                         # once a period
        w.ready = True
        ev = self.capture(lambda: checks("/ping", 200, 4))
        self.assertEqual([(e["first"], e["counts"]["/ping"]) for e in ev], [(["/ping", 200], {"204": 5, "200": 1})])
        self.capture(lambda: w.trip("W3"))
        checks("/ping", 200, 7)
        c.t += 30
        ev = self.capture(w.control)                                          # the inert mode keeps counting
        self.assertEqual([(e["counts"]["/ping"], e["inert"]) for e in ev], [({"204": 5, "200": 11}, "mdf_inert")])
        for _ in range(3600):                                                 # an hour without checks: no line
            c.t += 1
            self.assertEqual(self.capture(w.control), [])


class PollProcess(FakeProcess):
    def __init__(self, code=None):
        super().__init__()
        self.code = code

    def poll(self):
        return self.code


class TestStartAndPopulate(unittest.TestCase):
    """MFL033 c7.4: W1 from the start of the proxy (r2:181), 1500 s for the populate, the download
    under the supervisor, the log lines of the start."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.saved = wd.WEIGHT
        self.data = b"weights-of-the-test"
        wd.WEIGHT = ("model.safetensors", len(self.data), hashlib.sha256(self.data).hexdigest())

    def tearDown(self):
        wd.WEIGHT = self.saved

    def cfg(self, **env):
        cfg = wd.config(dict({"MDF_PROVIDER": "runpod"}, **env))
        cfg["hf_home"] = str(self.tmp / "hf")
        return cfg

    def capture(self, fn):
        import contextlib
        import io
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            result = fn()
        return result, events(err.getvalue())

    def test_w1_of_the_populate_and_from_the_start(self):
        self.assertEqual(wd.config({})["w1_s"], 900)
        self.assertEqual(wd.config({"MDF_ALLOW_DOWNLOAD": "1"})["w1_s"], 1500)
        self.assertEqual(wd.config({"MDF_ALLOW_DOWNLOAD": "1", "MDF_W1_S": "60"})["w1_s"], 60)
        for env, w1 in (({}, 900), ({"MDF_ALLOW_DOWNLOAD": "1"}, 1500)):
            c = Clock()
            w = wd.Watchdog(self.cfg(**env), clock=c, process=FakeProcess())
            self.assertEqual(w.rules.t0, c.t)                 # the rules start with the proxy
            w.probe = lambda path: (None, "")
            c.t += w1 - 1
            self.assertIsNone(w.tick())
            c.t += 1
            rule, ev = self.capture(w.tick)
            self.assertEqual(rule, "W1")
            self.assertEqual([e["event"] for e in ev][-1], "mdf_trip")
            self.assertEqual((ev[-1]["rule"], ev[-1]["since_start"]), ("W1", float(w1)))

    def test_the_trip_kills_the_download_and_nothing_starts_after(self):
        c = Clock()
        cfg = self.cfg(MDF_ALLOW_DOWNLOAD="1")
        w = wd.Watchdog(cfg, clock=c)
        sleeper = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True)
        w.download_proc = sleeper
        w.probe = lambda path: (None, "")
        c.t += 1500
        _, ev = self.capture(w.tick)
        self.assertEqual(sleeper.wait(10), -9)
        self.assertTrue(ev[-1]["download"])
        w.download_proc = None
        _, ev = self.capture(lambda: wd.prepare(w, cfg, argv=["/bin/false"], download_argv=["/bin/false"]))
        self.assertIsNone(w.process)                          # no download and no vLLM after a trip
        self.assertEqual([e["event"] for e in ev], ["mdf_weights_check", "mdf_weights_check"])

    def test_download_progress_end_and_vllm_start(self):
        cfg = self.cfg(MDF_ALLOW_DOWNLOAD="1")
        w = wd.Watchdog(cfg)
        snap = wd.snapshot(cfg["hf_home"])
        writer = ("import pathlib, sys, time\n"
                  "snap = pathlib.Path(sys.argv[1]); snap.mkdir(parents=True)\n"
                  "for name in sys.argv[3:]: (snap / name).write_text('{}')\n"
                  "time.sleep(0.5)\n"
                  "(snap / 'model.safetensors').write_bytes(bytes.fromhex(sys.argv[2]))\n")
        argv = [sys.executable, "-c", writer, str(snap), self.data.hex(), *wd.FILES]
        vllm = [sys.executable, "-c", "import time; time.sleep(30)"]
        try:
            with mock.patch.object(wd, "DOWNLOAD_TICK_S", 0.1):
                _, ev = self.capture(lambda: wd.prepare(w, cfg, argv=vllm, download_argv=argv))
        finally:
            if w.process is not None:
                w.process.signal(9)
        names = [e["event"] for e in ev]
        self.assertEqual(names[:2], ["mdf_weights_check", "mdf_download_start"])
        self.assertIn("mdf_download_progress", names)
        self.assertEqual(names[-4:], ["mdf_download_end", "mdf_weights_check", "mdf_vllm_args", "mdf_vllm_start"])
        end = ev[names.index("mdf_download_end")]
        self.assertEqual((end["exit"], end["timeout"]), (0, False))
        self.assertGreater(end["bytes"], len(self.data))
        self.assertGreaterEqual(end["seconds"], 0.5)
        self.assertEqual((ev[0]["phase"], ev[0]["ok"], ev[-3]["phase"], ev[-3]["ok"]),
                         ("start", False, "after_download", True))
        for e in ev:
            self.assertIn("mono", e)                          # monotonic time on every line
        self.assertIsNotNone(w.process)

    def test_download_failed_gives_the_inert_mode(self):
        cfg = self.cfg(MDF_ALLOW_DOWNLOAD="1")
        w = wd.Watchdog(cfg)
        _, ev = self.capture(lambda: wd.prepare(w, cfg, argv=["/bin/true"], download_argv=[sys.executable, "-c",
                                                                                          "raise SystemExit(3)"]))
        names = [e["event"] for e in ev]
        self.assertEqual(names, ["mdf_weights_check", "mdf_download_start", "mdf_download_end", "mdf_weights_check",
                                 "mdf_inert"])
        self.assertEqual((ev[2]["exit"], ev[-1]["code"], w.process), (3, "mdf_weights_missing", None))

    def test_an_error_at_the_start_is_the_inert_mode_not_an_exit(self):
        snap = wd.snapshot(str(self.tmp / "hf"))
        snap.mkdir(parents=True)
        for name in wd.FILES:
            (snap / name).write_text("{}")
        (snap / "model.safetensors").write_bytes(self.data)
        cfg = self.cfg()
        w = wd.Watchdog(cfg)
        _, ev = self.capture(lambda: wd.boot(w, cfg, argv=[str(self.tmp / "no-vllm")]))
        self.assertEqual([e["event"] for e in ev], ["mdf_weights_check", "mdf_vllm_args", "mdf_error", "mdf_inert"])
        self.assertEqual((w.inert, ev[2]["error"]), ("mdf_start_error", "FileNotFoundError"))

    def test_first_answers_of_vllm_and_its_exit(self):
        c = Clock()
        proc = PollProcess()
        w = wd.Watchdog(self.cfg(), clock=c, process=proc)
        answers = {"/health": (None, "")}
        w.probe = lambda path: answers.get(path, (None, ""))
        _, ev = self.capture(w.tick)
        self.assertEqual(ev, [])
        answers.update({"/health": (200, ""), "/v1/models": (200, json.dumps({"data": [{"id": wd.MODEL}]})),
                        "/metrics": (200, "vllm:generation_tokens_total 0\n")})
        c.t += 5
        _, ev = self.capture(w.tick)
        self.assertEqual([e["event"] for e in ev], ["mdf_vllm_first_health", "mdf_vllm_first_models",
                                                    "mdf_vllm_models_200", "mdf_ready"])
        self.assertEqual((ev[0]["status"], ev[1]["models"], ev[2]["models"], ev[3]["since_start"]),
                         (200, [wd.MODEL], [wd.MODEL], 5.0))
        c.t += 5
        _, ev = self.capture(w.tick)
        self.assertEqual(ev, [])                              # only the first answers
        proc.code = -11
        answers["/health"] = (None, "")
        c.t += 5
        _, ev = self.capture(w.tick)
        self.assertEqual([(e["event"], e["code"]) for e in ev], [("mdf_vllm_exit", -11)])
        c.t += 5
        _, ev = self.capture(w.tick)
        self.assertEqual(ev, [])                              # logged once


class TestWeights(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.saved = wd.WEIGHT
        data = b"weights"
        wd.WEIGHT = ("model.safetensors", len(data), hashlib.sha256(data).hexdigest())
        self.snap = wd.snapshot(self.tmp)
        self.snap.mkdir(parents=True)
        for name in wd.FILES:
            (self.snap / name).write_text("{}")
        (self.snap / "model.safetensors").write_bytes(data)

    def tearDown(self):
        wd.WEIGHT = self.saved

    def test_present_and_verified(self):
        self.assertTrue(wd.weights_ok(self.tmp))

    def test_changed_missing_or_short(self):
        (self.snap / "model.safetensors").write_bytes(b"weightz")
        self.assertFalse(wd.weights_ok(self.tmp))
        (self.snap / "model.safetensors").write_bytes(b"short")
        self.assertFalse(wd.weights_ok(self.tmp))
        (self.snap / "model.safetensors").write_bytes(b"weights")
        (self.snap / "tokenizer.json").unlink()
        self.assertFalse(wd.weights_ok(self.tmp))

    def test_pinned_revision(self):
        self.assertEqual(self.saved[1], 1378772208)
        self.assertEqual(wd.REVISION, "00c843e9cae45611dca0a6dc2c9bbbf2dadaaed3")


class TestMTP(unittest.TestCase):
    """MFL033 c7.5: the MTP speculation only when the config.json of the verified snapshot declares a
    head, by Surya's rule; a config.json that cannot be read gives the inert mode, never a default."""

    CASES = [({}, False), ({"num_nextn_predict_layers": 1}, True), ({"mtp_num_hidden_layers": 3}, True),
             ({"text_config": {"mtp_num_hidden_layers": 2}}, True), ({"text_config": {"num_nextn_predict_layers": 1}}, True),
             ({"num_nextn_predict_layers": 0}, False), ({"num_nextn_predict_layers": 0, "text_config": {"mtp_num_hidden_layers": 0}}, False),
             ({"num_nextn_predict_layers": True}, False), ({"num_nextn_predict_layers": "1"}, False),
             ({"num_nextn_predict_layers": 1.0}, False), ({"mtp_num_hidden_layers": -1}, False),
             ({"text_config": "x", "mtp_num_hidden_layers": None}, False), ({"text_config": [1]}, False)]

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.addCleanup(setattr, wd, "WEIGHT", wd.WEIGHT)
        data = b"weights"
        wd.WEIGHT = ("model.safetensors", len(data), hashlib.sha256(data).hexdigest())
        self.hf = str(self.tmp / "hf")
        self.snap = wd.snapshot(self.hf)
        self.snap.mkdir(parents=True)
        for name in wd.FILES:
            (self.snap / name).write_text("{}")
        (self.snap / "model.safetensors").write_bytes(data)

    def config(self, value):
        path = self.snap / "config.json"
        if isinstance(value, bytes):
            path.write_bytes(value)
        else:
            path.write_text(value if isinstance(value, str) else json.dumps(value))

    def prepare(self, cfg_value, env=None):
        import contextlib
        self.config(cfg_value)
        cfg = wd.config(dict({"MDF_PROVIDER": "runpod"}, **(env or {})))
        cfg["hf_home"] = self.hf
        started, self.envs = [], []
        envs = self.envs

        class Recorder:
            def __init__(self, argv, env=None):
                started.append(argv)
                envs.append(env)
                self.proc = types.SimpleNamespace(pid=4242)

        w = wd.Watchdog(cfg)
        err = io.StringIO()
        with mock.patch.object(wd, "VLLM", Recorder), contextlib.redirect_stderr(err):
            wd.prepare(w, cfg)
        return w, started, events(err.getvalue())

    def test_surya_rule(self):
        for value, head in self.CASES:
            with self.subTest(config=value):
                self.config(value)
                self.assertEqual(wd.mtp_head(self.hf)[0], head)
        self.config({"num_nextn_predict_layers": "1", "text_config": {"mtp_num_hidden_layers": 2}})
        self.assertEqual(wd.mtp_head(self.hf), (True, {"config.num_nextn_predict_layers": "str",
                                                       "text_config.mtp_num_hidden_layers": 2}))

    def test_the_same_answers_as_surya(self):
        from surya.inference.backends import vllm as surya_vllm
        for value, head in self.CASES:
            with self.subTest(config=value):
                self.config(value)
                self.assertEqual(surya_vllm._checkpoint_has_mtp_head(str(self.snap)), head)

    def test_a_config_that_cannot_be_read(self):
        for value, reason in ((b"\xff\xfe{}", "unreadable"), ("nope", "not_json"), ("[1]", "not_an_object"),
                              ("", "not_json")):
            with self.subTest(reason=reason):
                self.config(value)
                with self.assertRaises(wd.ConfigInvalid) as ctx:
                    wd.mtp_head(self.hf)
                self.assertEqual(str(ctx.exception), reason)
        (self.snap / "config.json").unlink()
        with self.assertRaises(wd.ConfigInvalid) as ctx:
            wd.mtp_head(self.hf)
        self.assertEqual(str(ctx.exception), "absent")
        (self.snap / "config.json").mkdir()
        with self.assertRaises(wd.ConfigInvalid) as ctx:
            wd.mtp_head(self.hf)
        self.assertEqual(str(ctx.exception), "unreadable")

    def test_prepare_without_a_head_starts_vllm_without_mtp(self):
        w, started, ev = self.prepare({"text_config": {"mtp_num_hidden_layers": 0}})
        self.assertEqual([e["event"] for e in ev], ["mdf_weights_check", "mdf_vllm_args", "mdf_vllm_start"])
        self.assertEqual(len(started), 1)
        self.assertNotIn("--speculative-config", started[0])
        args = ev[1]
        self.assertEqual((args["ok"], args["mtp"], args["mtp_tokens"], args["config"]),
                         (True, False, None, {"text_config.mtp_num_hidden_layers": 0}))
        self.assertEqual(args["args"], [a for a in started[0] if a.startswith("--")])
        self.assertTrue(all(a.startswith("--") for a in args["args"]))    # names only, no values
        self.assertNotIn(wd.MODEL, json.dumps(args))
        self.assertNotIn(wd.REVISION, json.dumps(args))
        self.assertIsNone(w.inert)

    def test_prepare_with_a_lever_names_it(self):
        """MFL033 N5b: the arguments of the lever, and the levers and the memory in `mdf_vllm_args`."""
        w, started, ev = self.prepare({}, {"MDF_LEVERS": "skip-mm-profiling"})
        self.assertIn("--skip-mm-profiling", started[0])
        self.assertEqual((ev[1]["levers"], ev[1]["gpu_memory_utilization"]), (["skip-mm-profiling"], "0.80"))
        self.assertIsNone(w.inert)
        _, started, ev = self.prepare({})
        self.assertEqual((ev[1]["levers"], ev[1]["gpu_memory_utilization"]), ([], "0.85"))

    def test_an_unknown_lever_is_the_inert_mode(self):
        w, started, ev = self.prepare({}, {"MDF_LEVERS": "skip-mm-profiling,fast"})
        self.assertEqual((started, w.process, w.inert), ([], None, "mdf_levers_invalid"))
        self.assertEqual((ev[1]["event"], ev[1]["ok"], ev[1]["error"]), ("mdf_vllm_args", False, "unknown levers: fast"))

    def test_prepare_with_a_head_passes_mtp(self):
        _, started, ev = self.prepare({"num_nextn_predict_layers": 1})
        self.assertIn("--speculative-config", started[0])
        self.assertEqual((ev[1]["mtp"], ev[1]["mtp_tokens"]), (True, 2))
        self.assertIn("--speculative-config", ev[1]["args"])

    def script(self, name, first_line):
        """An executable `name` in a bin directory of the test, with `first_line` as its first line."""
        bin_dir = self.tmp / "bin"
        bin_dir.mkdir(exist_ok=True)
        path = bin_dir / name
        path.write_bytes(first_line + b"\nimport sys\n")
        path.chmod(0o755)
        return bin_dir, path

    def test_the_runtime_of_vllm_in_the_args_line(self):
        """MFL033 c7.11 (prompt N4i §3.2): the entries of PYTHONPATH by name and never what they hold,
        the `vllm` of the PATH and its interpreter, the proxy's Python; the env logged is the one vLLM
        gets, PYTHONPATH unchanged."""
        bin_dir, vllm = self.script("vllm", b"#!/usr/bin/python3.12")
        deps = self.tmp / "deps"
        deps.mkdir()
        (deps / "mdf_hidden_module.py").write_text("x = 1\n")
        pythonpath = os.pathsep.join([str(deps), "/root", ""])
        with mock.patch.dict(os.environ, {"PATH": str(bin_dir), "PYTHONPATH": pythonpath}):
            _, started, ev = self.prepare({})
        args = ev[1]
        self.assertEqual(args["event"], "mdf_vllm_args")
        self.assertEqual(args["pythonpath"], [str(deps), "/root", ""])
        self.assertNotIn("mdf_hidden_module", json.dumps(args))                       # names, not contents
        self.assertEqual((args["vllm_bin"], args["interpreter"], args["interpreter_error"]),
                         (str(vllm), "/usr/bin/python3.12", None))
        self.assertEqual(args["interpreter_real"], os.path.realpath("/usr/bin/python3.12"))
        self.assertEqual(args["proxy_python"], sys.executable)
        self.assertEqual(started[0][:2], ["vllm", "serve"])
        self.assertEqual(self.envs[0]["PYTHONPATH"], pythonpath)
        self.assertEqual((self.envs[0]["HF_HOME"], self.envs[0]["HF_HUB_OFFLINE"]), (self.hf, "1"))

    def test_the_runtime_without_pythonpath_and_through_env(self):
        python = Path(sys.executable)
        bin_dir, vllm = self.script("vllm", b"#!/usr/bin/env " + python.name.encode())
        (bin_dir / python.name).symlink_to(python)
        env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
        env["PATH"] = str(bin_dir)
        with mock.patch.dict(os.environ, env, clear=True):
            _, _, ev = self.prepare({})
        args = ev[1]
        self.assertIsNone(args["pythonpath"])
        self.assertEqual((args["vllm_bin"], args["interpreter"]), (str(vllm), "/usr/bin/env " + python.name))
        self.assertEqual(args["interpreter_real"], os.path.realpath(python))

    def test_the_runtime_when_vllm_cannot_be_read(self):
        """No `vllm` on the PATH, a binary, an unreadable file: the line says so and vLLM is still
        started (its own error is the arrest), never an exception of the log."""
        empty = self.tmp / "empty"
        empty.mkdir()
        with mock.patch.dict(os.environ, {"PATH": str(empty)}):
            _, started, ev = self.prepare({})
        self.assertEqual([e["event"] for e in ev], ["mdf_weights_check", "mdf_vllm_args", "mdf_vllm_start"])
        self.assertEqual((ev[1]["vllm_bin"], ev[1]["interpreter"], ev[1]["interpreter_error"]),
                         (None, None, "not_found"))
        self.assertEqual(len(started), 1)
        bin_dir, vllm = self.script("vllm", b"\x7fELF\x02\x01\x01")
        self.assertEqual(wd.runtime("vllm", {"PATH": str(bin_dir)})["interpreter_error"], "no_shebang")
        with mock.patch("builtins.open", side_effect=PermissionError):
            got = wd.runtime("vllm", {"PATH": str(bin_dir)})
        self.assertEqual((got["vllm_bin"], got["interpreter_error"]), (str(vllm), "unreadable"))
        many = os.pathsep.join("/p%d" % i + "x" * 300 for i in range(50))
        got = wd.runtime("vllm", {"PATH": str(bin_dir), "PYTHONPATH": many})
        self.assertEqual(len(got["pythonpath"]), wd.RUNTIME_ENTRIES)
        self.assertTrue(all(len(e) <= wd.RUNTIME_TEXT for e in got["pythonpath"]))

    def test_prepare_with_an_unreadable_config_is_inert(self):
        for value, reason in (("nope", "not_json"), (b"\xff", "unreadable"), ("7", "not_an_object")):
            with self.subTest(reason=reason):
                w, started, ev = self.prepare(value)
                self.assertEqual([e["event"] for e in ev], ["mdf_weights_check", "mdf_vllm_args", "mdf_inert"])
                self.assertEqual((ev[1]["ok"], ev[1]["error"], ev[2]["code"]), (False, reason, "mdf_config_invalid"))
                self.assertEqual((started, w.process, w.inert), ([], None, "mdf_config_invalid"))

    def test_first_200_of_models_after_another_answer(self):
        c = Clock()
        w = wd.Watchdog(wd.config({}), clock=c, process=PollProcess())
        answers = {"/health": (200, ""), "/v1/models": (503, "{}"), "/metrics": (200, "vllm:generation_tokens_total 0\n")}
        w.probe = lambda path: answers.get(path, (None, ""))
        err = io.StringIO()
        import contextlib
        with contextlib.redirect_stderr(err):
            w.tick()
        ev = events(err.getvalue())
        self.assertEqual([(e["event"], e.get("status")) for e in ev],
                         [("mdf_vllm_first_health", 200), ("mdf_vllm_first_models", 503), ("mdf_ready", None)])
        answers["/v1/models"] = (200, json.dumps({"data": [{"id": wd.MODEL}]}))
        c.t += 5
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            w.tick()
            c.t += 5
            w.tick()
        ev = events(err.getvalue())
        self.assertEqual([(e["event"], e["models"], e["since_start"]) for e in ev],
                         [("mdf_vllm_models_200", [wd.MODEL], 5.0)])                  # once


class TestCompileCache(unittest.TestCase):
    """MFL033 c8.4 (Davide, 21/11 10:36): the cache of torch.compile on the volume, written only by our
    populate, its SHA-256 checked before every use (a cache whose SHA differs is not used), nothing of
    the documents in it (the proxy refuses every inference while it populates)."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.vllm = self.tmp / "vllm"
        graphs = self.vllm / wd.CACHE_SUB / "b9678b66f8" / "rank_0_0" / "backbone"
        graphs.mkdir(parents=True)
        (graphs / "graph.py").write_text("compiled graph\n")
        (self.vllm / wd.CACHE_SUB / "link").symlink_to("/etc/hostname")          # never archived
        (self.vllm / "other").write_text("not the torch.compile cache")
        self.marker = self.tmp / "written"
        self.cfg = {"hf_home": str(self.tmp / "vol" / "hf"), "gpu_names": ["NVIDIA A10"], "levers": ("compile-cache-populate",)}

    def capture(self, fn):
        import contextlib
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            value = fn()
        return value, events(err.getvalue())

    def populate(self):
        return self.capture(lambda: wd.populate_cache(self.cfg, env={"VLLM_CACHE_ROOT": str(self.vllm)},
                                                      written=str(self.marker)))

    def use(self, sha, gpu=("NVIDIA A10",), dest=None):
        cfg = dict(self.cfg, levers=("compile-cache:%s" % sha,), gpu_names=list(gpu))
        dest = dest or (self.tmp / "fresh")
        return self.capture(lambda: wd.use_cache(cfg, env={"VLLM_CACHE_ROOT": str(dest)})), dest

    def test_the_levers_of_the_cache(self):
        sha = "a" * 64
        self.assertEqual(wd.levers("compile-cache:%s, skip-mm-profiling" % sha),
                         ("skip-mm-profiling", "compile-cache:" + sha))
        self.assertEqual(wd.cache_shas(wd.levers("compile-cache:%s,compile-cache:%s" % ("b" * 64, sha))), [sha, "b" * 64])
        for bad in ("compile-cache", "compile-cache:" + "A" * 64, "compile-cache:abc", "skip-mm-profiling:1",
                    "compile-cache-populate:" + sha, "compile-cache-populate,compile-cache:" + sha):
            with self.subTest(bad=bad):
                with self.assertRaises(wd.LeversInvalid):
                    wd.levers(bad)

    def test_the_populate_writes_one_archive_named_by_its_sha256(self):
        sha, ev = self.populate()
        final = self.tmp / "vol" / wd.CACHE_DIR / ("torch-compile-%s.tar" % sha)
        self.assertTrue(final.is_file())
        self.assertEqual(wd.sha256_file(final), sha)
        self.assertEqual(self.marker.read_text(), str(final))
        self.assertEqual([x.name for x in final.parent.iterdir()], [final.name])        # no temporary left
        with tarfile.open(final) as tar:
            names = sorted(tar.getnames())
            manifest = json.loads(tar.extractfile(wd.CACHE_MANIFEST).read())
        self.assertIn(wd.CACHE_SUB + "/b9678b66f8/rank_0_0/backbone/graph.py", names)
        self.assertFalse([n for n in names if n.endswith("link") or n == "other"])
        self.assertEqual(manifest, {"gpu": ["NVIDIA A10"], "levers": [], "files": 1, "bytes": 15})
        (line,) = ev
        self.assertEqual((line["event"], line["phase"], line["written"], line["sha256"], line["files"]),
                         ("mdf_compile_cache", "populate", True, sha, 1))

    def test_a_populate_without_graphs_writes_nothing(self):
        shutil.rmtree(self.vllm / wd.CACHE_SUB)
        sha, ev = self.populate()
        self.assertIsNone(sha)
        self.assertFalse(self.marker.exists())
        self.assertEqual((ev[0]["written"], ev[0]["error"]), (False, "ValueError"))
        self.assertEqual(list((self.tmp / "vol" / wd.CACHE_DIR).glob("*")) if (self.tmp / "vol" / wd.CACHE_DIR).exists()
                         else [], [])

    def test_a_use_checks_the_sha256_and_the_gpu_then_extracts(self):
        sha, _ = self.populate()
        (got, ev), dest = self.use(sha)
        self.assertEqual(got, sha)
        self.assertEqual((dest / wd.CACHE_SUB / "b9678b66f8" / "rank_0_0" / "backbone" / "graph.py").read_text(),
                         "compiled graph\n")
        self.assertFalse((dest / wd.CACHE_MANIFEST).exists())
        self.assertEqual((ev[0]["phase"], ev[0]["used"], ev[0]["sha256"]), ("use", True, sha))
        (got, ev), _ = self.use(sha, gpu=("NVIDIA L4",), dest=self.tmp / "l4")
        self.assertIsNone(got)
        self.assertEqual(ev[0]["tried"][0]["reason"], "another_gpu")
        self.assertFalse((self.tmp / "l4").exists())
        (got, ev), _ = self.use("c" * 64, dest=self.tmp / "absent")
        self.assertEqual((got, ev[0]["used"], ev[0]["tried"][0]["reason"]), (None, False, "absent"))

    def test_a_cache_whose_sha256_differs_is_not_used(self):
        sha, _ = self.populate()
        final = self.tmp / "vol" / wd.CACHE_DIR / ("torch-compile-%s.tar" % sha)
        with open(final, "ab") as fh:
            fh.write(b"changed on the volume")
        (got, ev), dest = self.use(sha)
        self.assertIsNone(got)
        self.assertEqual(ev[0]["tried"][0]["reason"], "sha256_differs")
        self.assertFalse(dest.exists())

    def test_an_archive_with_a_member_outside_the_cache_is_not_used(self):
        import io as _io
        root = self.tmp / "vol" / wd.CACHE_DIR
        root.mkdir(parents=True)
        bad = root / "bad.tar"
        with tarfile.open(bad, "w") as tar:
            for name, data in ((wd.CACHE_MANIFEST, json.dumps({"gpu": ["NVIDIA A10"]}).encode()),
                               (wd.CACHE_SUB + "/../../escape.py", b"x")):
                info = tarfile.TarInfo(name)
                info.size = len(data)
                tar.addfile(info, _io.BytesIO(data))
        sha = wd.sha256_file(bad)
        bad.rename(root / ("torch-compile-%s.tar" % sha))
        (got, ev), dest = self.use(sha)
        self.assertIsNone(got)
        self.assertFalse((self.tmp / "escape.py").exists())
        self.assertFalse((dest.parent / "escape.py").exists())
        self.assertEqual(ev[0]["used"], False)

    def archive(self, members, manifest=None):
        """An archive named by its SHA-256 in the cache directory: `members` [(TarInfo fields, data)]."""
        import io as _io
        root = self.tmp / "vol" / wd.CACHE_DIR
        root.mkdir(parents=True, exist_ok=True)
        path = root / "build.tar"
        with tarfile.open(path, "w") as tar:
            data = manifest if manifest is not None else json.dumps({"gpu": ["NVIDIA A10"], "files": 1}).encode()
            info = tarfile.TarInfo(wd.CACHE_MANIFEST)
            info.size = len(data)
            tar.addfile(info, _io.BytesIO(data))
            for fields, body in members:
                info = tarfile.TarInfo(fields["name"])
                for k, v in fields.items():
                    setattr(info, k, v)
                if body is not None:
                    info.size = len(body)
                tar.addfile(info, _io.BytesIO(body) if body is not None else None)
        sha = wd.sha256_file(path)
        path.rename(root / ("torch-compile-%s.tar" % sha))
        return sha

    def refused(self, sha, reason, **patches):
        with mock.patch.multiple(wd, **patches) if patches else contextlib.nullcontext():
            (got, ev), dest = self.use(sha)
        self.assertIsNone(got)
        self.assertEqual(ev[0]["tried"][0]["reason"], reason)
        self.assertFalse((dest / wd.CACHE_SUB).exists())
        self.assertEqual([p.name for p in dest.iterdir()] if dest.exists() else [], [])   # no temporary left
        return ev

    def test_the_limits_of_the_populate_hold_when_the_archive_is_read(self):
        """S-NB4 of the final review of MFL033: the archive, its manifest, the types and names and the bytes of
        its members are checked before anything is extracted."""
        good = {"name": wd.CACHE_SUB + "/g/graph.py"}
        cases = {
            "member_type": [({"name": wd.CACHE_SUB + "/link", "type": tarfile.SYMTYPE, "linkname": "graph.py"}, None)],
            "hard link": [(good, b"x"), ({"name": wd.CACHE_SUB + "/hard", "type": tarfile.LNKTYPE,
                                         "linkname": wd.CACHE_SUB + "/g/graph.py"}, None)],
            "device": [({"name": wd.CACHE_SUB + "/dev", "type": tarfile.CHRTYPE}, None)],
            "dot dot": [({"name": wd.CACHE_SUB + "/g/../../escape.py"}, b"x")],
            "absolute": [({"name": "/" + wd.CACHE_SUB + "/abs.py"}, b"x")],
        }
        reasons = {"member_type": "member_type", "hard link": "member_type", "device": "member_type",
                   "dot dot": "outside_" + wd.CACHE_SUB, "absolute": "outside_" + wd.CACHE_SUB}
        for name, members in cases.items():
            with self.subTest(case=name):
                self.refused(self.archive(members), reasons[name])
                shutil.rmtree(self.tmp / "vol" / wd.CACHE_DIR)
        with self.subTest(case="the bytes of the members"):
            sha = self.archive([({"name": wd.CACHE_SUB + "/a.py"}, b"x" * 600), ({"name": wd.CACHE_SUB + "/b.py"}, b"y" * 600)])
            self.refused(sha, "over_the_limit", CACHE_MAX_BYTES=1000)
        with self.subTest(case="the members"):
            sha = self.archive([({"name": wd.CACHE_SUB + "/%d.py" % i}, b"x") for i in range(5)])
            self.refused(sha, "too_many_members", CACHE_MAX_MEMBERS=3)
        with self.subTest(case="the manifest"):
            sha = self.archive([(good, b"x")], manifest=b"[" * 100)
            self.refused(sha, "manifest", CACHE_MAX_MANIFEST=50)

    def test_an_archive_over_its_limit_is_not_even_hashed(self):
        sha, _ = self.populate()
        hashed = []
        real = wd.sha256_file

        def counting(path):
            hashed.append(path)
            return real(path)
        self.refused(sha, "over_the_limit", CACHE_MAX_ARCHIVE=100, sha256_file=counting)
        self.assertEqual(hashed, [])

    def test_a_link_in_place_of_the_archive(self):
        sha, _ = self.populate()
        final = self.tmp / "vol" / wd.CACHE_DIR / ("torch-compile-%s.tar" % sha)
        moved = self.tmp / "elsewhere.tar"
        final.rename(moved)
        final.symlink_to(moved)
        self.refused(sha, "not_a_file")

    def test_an_extraction_that_fails_publishes_nothing(self):
        sha, _ = self.populate()
        real = tarfile.TarFile.extractall

        def half(tar, path, members=None, **kw):
            real(tar, path, members=members[:1], **kw)               # a first member on disk, then a failure
            raise OSError("disk full")
        with mock.patch.object(tarfile.TarFile, "extractall", half):
            (got, ev), dest = self.use(sha)
        self.assertIsNone(got)
        self.assertEqual(ev[0]["tried"][0]["reason"], "OSError")
        self.assertFalse((dest / wd.CACHE_SUB).exists())
        self.assertEqual(list(dest.iterdir()), [])

    def test_the_proxy_refuses_inference_while_it_populates(self):
        cfg = wd.config({"MDF_LEVERS": "compile-cache-populate"})
        w = wd.Watchdog(cfg)
        self.assertTrue(w.populating)
        self.assertEqual(w.refusal("/v1/chat/completions"), "mdf_populating")
        self.assertEqual(w.refusal("/v1/completions?x=1"), "mdf_populating")
        self.assertIsNone(w.refusal("/v1/models"))
        self.assertIsNone(wd.Watchdog(wd.config({})).refusal("/v1/chat/completions"))

    def test_the_populate_starts_at_readiness(self):
        cfg = wd.config({"MDF_LEVERS": "compile-cache-populate"})
        w = wd.Watchdog(cfg)
        w.probe = lambda path: (200, "")
        w.first_answers = lambda status: None
        started = threading.Event()
        with mock.patch.object(wd, "populate_cache", lambda c: started.set()), \
                mock.patch.object(wd, "log", lambda *a, **k: None):
            w.tick()
            self.assertTrue(started.wait(5))
            self.assertTrue(w.ready)

    def test_prepare_uses_the_cache_before_vllm(self):
        calls = []
        cfg = wd.config({"MDF_LEVERS": "compile-cache:" + "d" * 64})
        w = wd.Watchdog(cfg)
        with mock.patch.object(wd, "check_weights", lambda *a: True), \
                mock.patch.object(wd, "use_cache", lambda c: calls.append("cache")), \
                mock.patch.object(wd, "mtp_head", lambda h: calls.append("vllm") or (False, {})), \
                mock.patch.object(wd, "VLLM", lambda argv, env=None: types.SimpleNamespace(proc=types.SimpleNamespace(pid=1))), \
                mock.patch.object(wd, "log", lambda *a, **k: None), mock.patch.object(wd, "runtime", lambda *a: {}):
            wd.prepare(w, cfg)
        self.assertEqual(calls, ["cache", "vllm"])


class TestGpu(unittest.TestCase):
    """MFL033 c7.14 (prompt N4m §3.3): at its start the proxy names in one line the GPU it got, from
    `nvidia-smi --query-gpu=name --format=csv,noheader` or else /proc/driver/nvidia/gpus; never raising."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def proc(self, *models):
        for i, model in enumerate(models):
            d = self.tmp / ("0000:00:%02x.0" % (30 + i))
            d.mkdir()
            (d / "information").write_text("Model: \t\t %s\nIRQ:   \t\t 42\nBus Type: \t PCIe\n" % model)
        return str(self.tmp)

    def test_from_nvidia_smi(self):
        seen = []

        def run(argv, **kw):
            seen.append((argv, kw.get("timeout")))
            return types.SimpleNamespace(returncode=0, stdout="NVIDIA L4\n", stderr="")
        self.assertEqual(wd.gpu_names(run=run, proc=self.proc("NVIDIA A10G")),
                         {"names": ["NVIDIA L4"], "source": "nvidia-smi", "error": None})
        self.assertEqual(seen, [(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], wd.GPU_QUERY_S)])

    def test_from_the_driver_when_nvidia_smi_gives_none(self):
        def missing(argv, **kw):
            raise FileNotFoundError(argv[0])

        def timeout(argv, **kw):
            raise subprocess.TimeoutExpired(argv, kw["timeout"])

        def failing(argv, **kw):
            return types.SimpleNamespace(returncode=9, stdout="", stderr="NVIDIA-SMI has failed")
        for run, error in ((missing, "nvidia-smi: FileNotFoundError"), (timeout, "nvidia-smi: TimeoutExpired"),
                           (failing, "nvidia-smi: exit 9")):
            with self.subTest(error=error):
                shutil.rmtree(self.tmp)
                self.tmp.mkdir()
                self.assertEqual(wd.gpu_names(run=run, proc=self.proc("NVIDIA A10G")),
                                 {"names": ["NVIDIA A10G"], "source": "proc", "error": error})
                self.assertEqual(wd.gpu_names(run=run, proc=str(self.tmp / "none")),
                                 {"names": [], "source": None, "error": error})

    def test_one_line_after_mdf_start(self):
        import contextlib
        err = io.StringIO()
        cfg = wd.config({"MDF_PROVIDER": "modal"})
        with mock.patch.object(wd.Watchdog, "serve"), mock.patch.object(wd.Watchdog, "loops"), \
                mock.patch.object(wd, "boot") as boot, contextlib.redirect_stderr(err), \
                mock.patch.object(wd, "gpu_names", lambda: {"names": ["NVIDIA A10G"], "source": "nvidia-smi",
                                                            "error": None}):
            wd.start(cfg)
        ev = events(err.getvalue())
        self.assertEqual([e["event"] for e in ev], ["mdf_start", "mdf_gpu"])
        self.assertEqual((ev[1]["names"], ev[1]["source"]), (["NVIDIA A10G"], "nvidia-smi"))
        self.assertIn("since_start", ev[1])
        boot.assert_called_once()                                               # the start goes on


class TestModalPopulate(unittest.TestCase):
    """MFL033 c7.5: the Modal populate calls `download` with its signature of c7.4 (a `Standalone`
    holder, a limit inside the timeout of the function); c7.4 called `download("/weights/hf")`."""

    def test_populate_downloads_outside_the_proxy(self):
        import contextlib
        import inspect
        calls, real = [], wd.download

        def download(*a, **k):
            calls.append(inspect.signature(real).bind(*a, **k).arguments)
            return real(*a, **dict(k, argv=[sys.executable, "-c", "pass"]))      # never the network

        class App:
            def __init__(self, name):
                self.name = name

            def function(self, **kw):
                self.kw = kw
                return lambda f: f

        image, vol = mock.Mock(), mock.Mock()
        image.entrypoint.return_value = image
        image.add_local_file.return_value = image
        fake = types.ModuleType("modal")
        fake.Image = types.SimpleNamespace(from_registry=lambda ref, **kw: image)
        fake.Volume = types.SimpleNamespace(from_name=lambda name, create_if_missing=False: vol)
        fake.App = App
        verified = iter([False, True])
        out = io.StringIO()
        with mock.patch.dict(sys.modules, {"modal": fake, "watchdog": wd}), \
                mock.patch.object(sys, "path", list(sys.path)), \
                mock.patch.object(wd, "weights_ok", lambda hf: next(verified)), \
                mock.patch.object(wd, "download", download):
            spec = importlib.util.spec_from_file_location("mdf_modal_populate_t",
                                                          ROOT / "scripts" / "servers" / "modal_populate.py")
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
                mod.populate()
        self.assertEqual(len(calls), 1)
        self.assertIsInstance(calls[0]["wd"], wd.Standalone)
        self.assertEqual(calls[0]["hf_home"], "/weights/hf")
        self.assertLess(calls[0]["limit_s"], mod.app.kw["timeout"])
        line = json.loads(out.getvalue().strip().splitlines()[-1])
        self.assertEqual((line["present_before"], line["download_exit"], line["verified"]), (False, 0, True))
        vol.commit.assert_called_once_with()


class TestProxySockets(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        r = subprocess.run([sys.executable, "-I", "-B", str(DRIVER)], capture_output=True, text=True, timeout=300)
        if r.returncode:
            raise AssertionError(r.stderr[-3000:])
        cls.r = json.loads(r.stdout.strip().splitlines()[-1])
        cls.stderr = r.stderr
        cls.requests, cls.checks = [], []
        for text in r.stderr.splitlines():
            try:
                ev = json.loads(text)
            except ValueError:
                continue
            if isinstance(ev, dict) and ev.get("event") == "mdf_request":
                cls.requests.append(ev)
            if isinstance(ev, dict) and ev.get("event") == "mdf_health_checks":
                cls.checks.append(ev)

    def request(self, kind, method, path, status, code=None):
        return [e for e in self.requests if (e.get("kind"), e.get("method"), e.get("path"), e.get("status"),
                                             e.get("code")) == (kind, method, path, status, code)]

    def test_a_line_for_every_answer_but_health(self):
        # MFL033 c7.6: `mdf_request` for every answer of the proxy, `forwarded` or `error`; /health never
        for kind, method, path, status, code in (("forwarded", "GET", "/v1/models", 200, None),
                                                 ("forwarded", "GET", "/metrics", 200, None),
                                                 ("forwarded", "POST", "/v1/chat/completions", 200, None),
                                                 ("forwarded", "HEAD", "/v1/models", 200, None),
                                                 ("forwarded", "GET", "other", 500, None),
                                                 ("error", "GET", "/v1/models", 502, "mdf_upstream"),
                                                 ("error", "POST", "/v1/chat/completions", 411, "mdf_length_required"),
                                                 ("error", "GET", "/v1/models", 503, "mdf_busy"),
                                                 ("error", "GET", "other", 504, "mdf_request_timeout"),
                                                 ("error", "GET", "/v1/models", 503, "mdf_weights_missing")):
            with self.subTest(kind=kind, path=path, status=status):
                found = self.request(kind, method, path, status, code)
                self.assertTrue(found)
                self.assertTrue(all(isinstance(e.get("t"), float) and isinstance(e.get("mono"), float) for e in found))
        self.assertEqual({e["kind"] for e in self.requests}, {"forwarded", "error"})
        self.assertNotIn("/health", {e["path"] for e in self.requests})
        self.assertLessEqual({e["path"] for e in self.requests}, set(wd.LOGGED_PATHS) | {"other"})

    def test_the_query_never_in_the_log(self):
        self.assertEqual(self.r["query"][1], "forwarded")
        self.assertTrue(self.request("forwarded", "GET", "/v1/models", self.r["query"][0]))
        self.assertNotIn("mdf-query-secret", self.stderr)

    def test_health_204_then_200(self):
        self.assertEqual(self.r["health_before"], 204)
        self.assertEqual(self.r["health_after"], [200, "health"])

    def test_ping_is_served_like_health_and_never_forwarded(self):
        # MFL033 c7.7: 204 while starting, 200 after readiness and in the inert mode, never vLLM's
        self.assertEqual(self.r["ping_before"], [204, "health"])
        for key in ("ping_after", "ping_query", "ping_head", "ping_upstream_down_ready", "inert_ping"):
            self.assertEqual(self.r[key], [200, "health"], key)
        self.assertEqual(self.r["ping_upstream_down_starting"], [204, "health"])   # 502 from the proxy on 19/11
        self.assertEqual(self.r["weights_missing"][3:], [200, 200])
        self.assertTrue(self.r["upstream_paths"])
        self.assertFalse([p for p in self.r["upstream_paths"] if p.split("?")[0] in ("/ping", "/health")])

    def test_health_checks_summarized_never_one_line_each(self):
        # MFL033 c7.7: the first answer of each path and status is a summary line, with the counts
        def first(path, status):
            return [e["counts"] for e in self.checks if e.get("first") == [path, status]]
        self.assertIn({"/health": {"204": 1}, "/ping": {"204": 1}}, first("/ping", 204))      # the main proxy
        self.assertIn({"/health": {"200": 1, "204": 1}, "/ping": {"200": 1, "204": 1}}, first("/ping", 200))
        self.assertLess(len(self.checks), 20)                 # dozens of checks, a few lines
        self.assertFalse([e for e in self.requests if e.get("path") in ("/ping", "/health")])

    def test_reads_without_a_body_and_the_post_of_surya(self):
        # prescription 1: a GET or HEAD without a body has length 0; the probes and Surya's POST go through
        for path in ("/v1/models", "/metrics", "/version"):
            self.assertEqual(self.r["get" + path], [200, "forwarded"])
        self.assertEqual(self.r["chat"], [200, "forwarded", True])
        self.assertEqual(self.r["head"], [200, "forwarded"])
        self.assertEqual(self.r["get_no_length"], 200)

    def test_framing(self):
        self.assertEqual(self.r["post_no_length"], 411)
        self.assertEqual(self.r["both_framings"], 400)
        self.assertEqual(self.r["two_lengths"], 400)
        self.assertEqual(self.r["too_large"], 413)
        self.assertEqual(self.r["get_body"], 400)

    def test_hop_by_hop_both_ways(self):
        # prescription 6: also the headers named by Connection
        seen = self.r["hop_request_seen"]
        for h in ("x-drop", "keep-alive", "proxy-authorization"):
            self.assertNotIn(h, seen)
        self.assertIn("x-keep", seen)
        self.assertEqual(self.r["hop_response"], {"x-up": None, "keep-alive": None, "x-kept": "yes",
                                                  "x-mdf-proxy": "forwarded", "connection": "close"})

    def test_errors_of_vllm_are_forwarded_those_of_the_proxy_are_error(self):
        self.assertEqual(self.r["e500"], [500, "forwarded"])
        self.assertEqual(self.r["upstream_down"], [502, "error", "mdf_upstream"])

    def test_chunked_answer_rewritten_with_a_length(self):
        self.assertEqual(self.r["chunked"], [200, "19", None, "hello chunked world"])

    def test_terms(self):
        status, kind, code, seconds = self.r["timeout"]
        self.assertEqual((status, kind, code), (504, "error", "mdf_request_timeout"))
        self.assertLess(seconds, 3.5)
        status, body, seconds = self.r["partial"]
        self.assertEqual((status, body), (200, 1000))       # the partial answer, then closed at the term
        self.assertLess(seconds, 4.5)
        cut, closed = self.r["no_read"]                       # a client that does not read: closed at req_max + 2 s
        self.assertTrue(cut)
        self.assertLessEqual(closed, 7.0)

    def test_busy_trip_and_inert(self):
        self.assertEqual(self.r["busy_41st"], [503, "error", "mdf_busy", 40])
        self.assertEqual(self.r["trip_40"], [40, ["SIGKILL"]])
        self.assertEqual(self.r["inert"], [503, "error", "mdf_inert"])
        self.assertEqual(self.r["inert_health"], 200)

    def test_32_bodies_of_30_mb_health_and_trip(self):
        self.assertEqual(self.r["big_pending"], 32)
        tripped, total, seconds = self.r["big_trip"]
        self.assertEqual((tripped, total), (32, 32))
        self.assertLess(seconds, 5)
        self.assertLess(self.r["big_health_max_s"], 1.0)

    def test_faults_on_sockets(self):
        self.assertEqual(self.r["hang_after_first"], ["SIGSTOP"])
        self.assertEqual(self.r["crash_before"], [])
        self.assertEqual(self.r["crash_after_first"], [1])

    def test_weights_missing(self):
        self.assertEqual(self.r["weights_missing"][:4], [503, "error", "mdf_weights_missing", 200])


if __name__ == "__main__":
    unittest.main()
