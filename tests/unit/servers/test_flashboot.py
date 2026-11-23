"""FlashBoot of RunPod in the proxy (`scripts/servers/watchdog.py`, MFL045 c3; plan r2 §6 A3, App. D; prescriptions
P-1, P-2, P-3, P-9).

With the lever `flashboot` a worker paused by RunPod (its idle timeout, no job: the process frozen) resumes with
the monotonic clock ahead by the pause. Every blocking step of the supervisor has its bound: the sleep tick_s, a
probe PROBE_S. A suspension is a step that overran its bound by PAUSE_GAP_S or more, judged only from the instants
before and after the step and its bound (P-1); the round drops its readings and starts again, and only without work
before the step (`inflight > 0` sampled just before it, or the busy of the last step: P-3) the references of W2-W4
move by the overrun, never W1 nor W5; `mdf_pause` says so. Without the lever: no compensation at all (P-2).

The supervisor runs here on fake services: a fake clock and sleep injected, a fake vLLM as the probe; for each case
the trip is compared with the rule of before MFL045 on the same sequence (the lever off), in active time, the
compensated jump and the real deadlines (P-1). With the lever the proxy also resets vLLM's caches at each passage to
idle after work, in vLLM's dev mode that only the lever turns on (P-9)."""
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import threading
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[3]


def load():
    spec = importlib.util.spec_from_file_location("mdf_watchdog_fb", ROOT / "scripts" / "servers" / "watchdog.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


wd = load()
TICK = 5.0


class Clock:
    def __init__(self, t=0.0):
        self.t = t

    def __call__(self):
        return self.t


class Proc:
    def __init__(self):
        self.signals = []

    def signal(self, sig):
        self.signals.append(sig)

    def poll(self):
        return None


def metrics_text(m):
    return "".join("vllm:%s %s\n" % ({"running": "num_requests_running", "waiting": "num_requests_waiting",
                                       "tokens": "generation_tokens_total"}[k], v) for k, v in m.items())


class Supervisor:
    """The supervisor of a proxy on fake services. `script(i)` gives round i: {"health": (s, ok), "metrics": (s, ok,
    m) or None, "models": (s, ok), "inflight": n, "inflight_in_sleep": n, "freeze": (step, s)}; a step is a probe
    path or "sleep". Each probe advances the clock by its seconds (at most the bound of its socket operations, as
    the real probe), the sleep by tick_s; a freeze adds its seconds to its step."""

    def __init__(self, script, levers="", provider="runpod"):
        self.script, self.clock, self.events = script, Clock(), []
        cfg = wd.config({"MDF_PROVIDER": provider, "MDF_LEVERS": levers})
        self.w = wd.Watchdog(cfg, clock=self.clock, process=Proc())
        self.w.probe, self.w.sleep = self.probe, self.sleep
        self.i, self.round = 0, None

    def freeze(self, step):
        f = self.round.get("freeze")
        return f[1] if f and f[0] == step else 0.0

    def probe(self, path):
        key = {"/health": "health", "/metrics": "metrics", "/v1/models": "models"}[path]
        spec = self.round.get(key) or ((0.01, True) if key != "metrics" else None)
        seconds, ok = spec[0], spec[1]
        self.clock.t += seconds + self.freeze(path)
        if not ok:
            return None, ""
        if key == "health":
            return 200, ""
        if key == "models":
            return 200, json.dumps({"data": [{"id": wd.MODEL}]})
        return 200, metrics_text(spec[2])

    def sleep(self, s):
        if "inflight_in_sleep" in self.round:            # P-3: a request accepted during the sleep
            self.w.inflight = self.round["inflight_in_sleep"]
        self.clock.t += s + self.freeze("sleep")

    def run(self, horizon=2400.0):
        """Rounds until a trip or the horizon: (rule, t, the applied seconds of the pauses, the mdf_pause lines)."""
        with mock.patch.object(wd, "log", lambda event, **f: self.events.append(dict(f, event=event))):
            while self.clock.t < horizon:
                self.round = self.script(self.i)
                self.w.inflight = self.round.get("inflight", 0)
                self.i += 1
                rule = self.w.round()
                if rule:
                    pauses = [e for e in self.events if e["event"] == "mdf_pause"]
                    return rule, round(self.clock.t, 2), sum(p["seconds"] for p in pauses if p["applied"]), pauses
        pauses = [e for e in self.events if e["event"] == "mdf_pause"]
        return None, round(self.clock.t, 2), sum(p["seconds"] for p in pauses if p["applied"]), pauses


IDLE = {"health": (0.01, True), "metrics": (0.01, True, {"tokens": 0.0}), "inflight": 0}


def healthy_pause(at, seconds, step="sleep"):
    """Idle until round `at`, a freeze of `seconds` in `step`, then the waking page (work, tokens growing)."""
    def script(i):
        if i < at:
            return dict(IDLE)
        if i == at:
            return dict(IDLE, freeze=(step, seconds))
        k = i - at - 1                                   # the first round after the pause: the page, no token yet
        if k <= 4:
            return {"health": (0.01, True), "metrics": (0.01, True, {"tokens": 100.0 * k, "running": 1}), "inflight": 1}
        return {"health": (0.01, True), "metrics": (0.01, True, {"tokens": 500.0}), "inflight": 0}
    return script


def slow_probes(health_s, metrics_s):
    def script(i):
        if i == 0:
            return dict(IDLE)
        return {"health": (health_s, True), "metrics": (metrics_s, True, {"tokens": 0.0, "running": 1}), "inflight": 1}
    return script


def body_stall():
    """/metrics: headers then a stalled body, 5 s each, then a failure (the per-operation bound, P-2)."""
    def script(i):
        if i == 0:
            return dict(IDLE)
        return {"health": (0.5, True), "metrics": (10.0, False, None), "inflight": 1}
    return script


def hang_then_freeze(at, seconds):
    def script(i):
        r = {"health": (0.01, True), "metrics": (0.01, True, {"tokens": 0.0, "running": 1}), "inflight": 1}
        if i == 0:
            return dict(IDLE)
        if i == at:
            r["freeze"] = ("sleep", seconds)
        return r
    return script


def unhealthy_then_pause(at, seconds):
    def script(i):
        r = {"health": (0.01, i < at - 6), "metrics": (0.01, True, {"tokens": 0.0}), "inflight": 0}
        if i == 0:
            return dict(IDLE)
        if i == at:
            r["freeze"] = ("sleep", seconds)
        return r
    return script


def without(script):
    """The same sequence without its freeze."""
    def s(i):
        r = dict(script(i))
        r.pop("freeze", None)
        return r
    return s


def run(script, levers, provider="runpod", horizon=2400.0):
    return Supervisor(script, levers=levers, provider=provider).run(horizon)


class TestThePauseRule(unittest.TestCase):
    def test_a_pause_of_flashboot_resumed_by_its_page(self):
        """Case 1: an idle worker frozen 700 s or 130 s, in the sleep, during /metrics or during /v1/models, then
        the page that wakes it: with the lever no trip, and in active time the same as the sequence without the
        freeze; before MFL045 (the lever off) a trip at once after the pause."""
        for name, script, seconds in (("700 s in the sleep", healthy_pause(30, 700.0), 700.0),
                                      ("130 s in the sleep", healthy_pause(30, 130.0), 130.0),
                                      ("700 s in /metrics", healthy_pause(30, 700.0, "/metrics"), 700.0)):
            with self.subTest(name):
                horizon = 30 * 5.04 + seconds + 300                     # before the idle W4 after the page
                rule, t, shifted, pauses = run(script, "flashboot", horizon=horizon)
                self.assertIsNone(rule)
                (p,) = pauses
                self.assertTrue(p["applied"])
                self.assertFalse(p["busy_before"])
                self.assertAlmostEqual(p["seconds"], seconds - (0.0 if p["during"] == "sleep" else 0.0)
                                       + (0.01 if p["during"] != "sleep" else 0.0) - (wd.PROBE_S if p["during"] != "sleep"
                                                                                      else 0.0), delta=0.1)
                self.assertEqual(len(p["boot_id"]), 16)
                today = run(script, "", horizon=horizon)
                self.assertIn(today[0], ("W3", "W4"))
                self.assertLess(today[1], 30 * 5.04 + seconds + 20)       # at once after the pause

    def test_the_detector_knows_only_the_instants_and_the_bound(self):
        """P-1: a freeze during a probe is the probe's time over PROBE_S (a probe of 0.01 s frozen 700 s overran its
        bound by 685.01 s), never the freeze the simulation of r2 knew; a freeze during /v1/models (the first
        answers, before readiness) counts like any probe."""
        def models_pause(i):
            if i < 3:
                return {"health": (0.01, False), "inflight": 0}
            if i == 3:
                return {"health": (0.01, True), "models": (0.01, True), "metrics": (0.01, True, {"tokens": 0.0}),
                        "freeze": ("/v1/models", 700.0)}
            return dict(IDLE)
        s = Supervisor(healthy_pause(30, 700.0, "/metrics"), levers="flashboot")
        _, _, _, pauses = s.run(horizon=1150.0)
        self.assertAlmostEqual(pauses[0]["seconds"], 700.01 - wd.PROBE_S, places=2)
        self.assertEqual(pauses[0]["during"], "metrics")
        s = Supervisor(models_pause, levers="flashboot")
        rule, _, _, pauses = s.run(horizon=900.0)
        self.assertEqual([(p["during"], p["applied"]) for p in pauses], [("models", True)])
        self.assertAlmostEqual(pauses[0]["seconds"], 700.01 - wd.PROBE_S, places=2)

    def test_slow_probes_and_a_stalled_body_trip_as_before(self):
        """Cases 2 and 4: probes slow but within their bound (5 + 5 s, 4.6 + 5.5 s), a body stalled after its
        headers: no suspension, W3 at the same instant as before MFL045."""
        for name, script in (("5 + 5 s", slow_probes(5.0, 5.0)), ("4.6 + 5.5 s", slow_probes(4.6, 5.5)),
                             ("body stall", body_stall())):
            with self.subTest(name):
                with_lever, before = run(script, "flashboot"), run(script, "")
                self.assertEqual(with_lever[:2], before[:2])
                self.assertEqual(with_lever[0], "W3")
                self.assertEqual(with_lever[3], [])

    def test_a_hang_with_work_then_a_freeze_is_not_compensated(self):
        """Case 3: work in flight and the counter still, then a freeze: recorded, not applied; W3 counts the real
        time, at the same instant as before."""
        script = hang_then_freeze(10, 700.0)
        rule, t, shifted, pauses = run(script, "flashboot")
        self.assertEqual((rule, t), run(script, "")[:2])
        self.assertEqual(rule, "W3")
        self.assertEqual([(p["applied"], p["busy_before"]) for p in pauses], [(False, True)])
        self.assertEqual(shifted, 0)

    def test_w2_counts_the_active_time_of_an_unhealthy_vllm(self):
        """Case 5: unhealthy and idle, then a freeze: `bad_since` moved by the jump, never erased; W2 trips after 60 s
        of active unhealthy time, as on the sequence without the freeze; before MFL045 at once after the pause."""
        script = unhealthy_then_pause(20, 700.0)
        rule, t, shifted, pauses = run(script, "flashboot")
        self.assertEqual(rule, "W2")
        self.assertEqual(shifted, 700.0)
        plain = run(without(script), "")
        self.assertEqual(plain[0], "W2")
        self.assertLessEqual(abs((t - shifted) - plain[1]), TICK + 0.1)
        before = run(script, "")
        self.assertEqual(before[0], "W2")
        self.assertLess(before[1], t)

    def test_a_request_accepted_during_the_sleep_is_work(self):
        """P-3: the last step saw no work; a request comes during the sleep; the next probe is frozen: `busy_before`
        is the `inflight` sampled just before the probe, so nothing moves."""
        def script(i):
            if i < 10:
                return dict(IDLE)
            if i == 10:
                return dict(IDLE, inflight_in_sleep=1)
            if i == 11:
                return {"health": (0.01, True), "metrics": (0.01, True, {"tokens": 0.0, "running": 1}), "inflight": 1,
                        "freeze": ("/health", 300.0)}
            return {"health": (0.01, True), "metrics": (0.01, True, {"tokens": 0.0, "running": 1}), "inflight": 1}
        rule, t, shifted, pauses = run(script, "flashboot")
        self.assertEqual([(p["busy_before"], p["applied"]) for p in pauses], [(True, False)])
        self.assertEqual((rule, shifted), ("W3", 0))

    def test_without_the_lever_no_compensation_on_both_providers(self):
        """P-2: without the lever nothing is compensated, on RunPod and on Modal: the pause trips as before, and no
        `mdf_pause`; the only change of MFL045 there is the deadline of the probes (test_watchdog)."""
        for provider in ("runpod", "modal"):
            with self.subTest(provider):
                rule, t, shifted, pauses = run(healthy_pause(30, 700.0), "", provider=provider)
                self.assertIn(rule, ("W3", "W4"))
                self.assertEqual(pauses, [])
                self.assertLess(t, 30 * 5.04 + 720)

    def test_w1_and_w5_never_move(self):
        r = wd.Rules(dict(wd.RULES, **wd.DEFAULTS["runpod"]), 0.0)
        r.step(10.0, True, {"tokens": 0.0}, 0)
        before = (r.t0, r.ready_at)
        r.shift(700.0)
        self.assertEqual((r.t0, r.ready_at), before)
        self.assertEqual((r.progress_at, r.activity_at), (710.0, 710.0))

    def test_a_request_keeps_its_real_deadline(self):
        """The requests accepted before a jump keep their term: the controller closes them at the real deadline."""
        clock = Clock(100.0)
        w = wd.Watchdog(wd.config({"MDF_PROVIDER": "runpod", "MDF_LEVERS": "flashboot"}), clock=clock, process=Proc())
        p = wd.Pending(clock() + w.cfg["req_max_s"])
        w.pending.add(p)
        clock.t += 700.0
        w.control()
        self.assertEqual(p.reason, "timeout")


class TestTheLever(unittest.TestCase):
    def test_the_closed_set(self):
        self.assertIn("flashboot", wd.LEVERS)
        self.assertEqual(wd.levers("flashboot,compile-cache:" + "a" * 64)[-1], "flashboot")
        with self.assertRaises(wd.LeversInvalid):
            wd.levers("flashboot,snapshot")
        w = wd.Watchdog(wd.config({"MDF_PROVIDER": "runpod", "MDF_LEVERS": "flashboot"}))
        self.assertTrue(w.pausing)
        self.assertFalse(wd.Watchdog(wd.config({"MDF_PROVIDER": "runpod"})).pausing)

    def test_the_dev_mode_of_vllm_only_with_the_lever(self):
        """P-9: VLLM_SERVER_DEV_MODE=1 in the env of vLLM only with the lever (its routes stay closed at the edge:
        the closed list of the proxy); the compile cache keeps its hash (the variable is not a factor of it)."""
        started = []

        class Fake:
            def __init__(self, argv, env=None):
                started.append(env)
                self.proc = type("P", (), {"pid": 9})()
        for levers, dev in (("flashboot", "1"), ("", None)):
            with self.subTest(levers=levers):
                cfg = wd.config({"MDF_PROVIDER": "runpod", "MDF_LEVERS": levers})
                w = wd.Watchdog(cfg)
                with mock.patch.object(wd, "VLLM", Fake), mock.patch.object(wd, "check_weights", lambda *a: True), \
                        mock.patch.object(wd, "use_cache", lambda c: None), \
                        mock.patch.object(wd, "mtp_head", lambda h: (False, {})), \
                        contextlib.redirect_stderr(io.StringIO()):
                    wd.prepare(w, cfg)
                self.assertEqual(started[-1].get("VLLM_SERVER_DEV_MODE"), dev)
        self.assertNotIn("VLLM_SERVER_DEV_MODE", os.environ)


class TestTheResetOfTheCaches(unittest.TestCase):
    """P-9: with the lever, at each passage to idle after work (`inflight` back to 0) the proxy resets vLLM's prefix,
    multimodal and encoder caches, on the loopback, in a thread of its own (a new request is never held), and writes
    the outcome in one line without contents; nothing without the lever."""

    def test_the_three_resets_and_their_line(self):
        calls, events = [], []

        def call(method, path, body=None, timeout=None):
            calls.append((method, path))
            return (500, "") if path == "/reset_encoder_cache" else (200, "")
        w = wd.Watchdog(wd.config({"MDF_PROVIDER": "runpod", "MDF_LEVERS": "flashboot"}), process=Proc())
        with mock.patch.object(wd, "log", lambda event, **f: events.append(dict(f, event=event))):
            ok = w.reset_caches(call)
        self.assertFalse(ok)
        self.assertEqual(calls, [("POST", "/reset_prefix_cache"), ("POST", "/reset_mm_cache"),
                                 ("POST", "/reset_encoder_cache")])
        (line,) = events
        self.assertEqual((line["event"], line["ok"]), ("mdf_cache_reset", False))
        self.assertEqual(line["routes"], {"/reset_prefix_cache": 200, "/reset_mm_cache": 200,
                                          "/reset_encoder_cache": 500})
        self.assertEqual(line["boot_id"], w.boot_id)

    def test_only_at_the_passage_to_idle_after_work_and_only_with_the_lever(self):
        for levers, wanted in (("flashboot", True), ("", False)):
            with self.subTest(levers=levers):
                w = wd.Watchdog(wd.config({"MDF_PROVIDER": "runpod", "MDF_LEVERS": levers}), process=Proc())
                w.inflight = 2
                w.request_done(True)                       # one of two inferences ends: still work
                self.assertFalse(w.reset_wanted.is_set())
                w.request_done(False)                      # a GET: no work changed
                self.assertFalse(w.reset_wanted.is_set())
                w.request_done(True)
                self.assertEqual(w.reset_wanted.is_set(), wanted)
                self.assertEqual(w.inflight, 0)

    def test_a_reset_never_holds_a_request(self):
        release, calls = threading.Event(), []

        def slow(method, path, body=None, timeout=None):
            calls.append(path)
            release.wait(5)
            return 200, ""
        w = wd.Watchdog(wd.config({"MDF_PROVIDER": "runpod", "MDF_LEVERS": "flashboot"}), process=Proc())
        with contextlib.redirect_stderr(io.StringIO()):
            t = w.start_resetter(slow)
            w.inflight = 1
            w.request_done(True)
            for _ in range(200):
                if calls:
                    break
                threading.Event().wait(0.01)
            self.assertEqual(calls[:1], ["/reset_prefix_cache"])
            self.assertIsNone(w.refusal("/v1/chat/completions"))     # a new request goes on meanwhile
            release.set()
            w.stop_resetter()
            t.join(5)


if __name__ == "__main__":
    unittest.main()
