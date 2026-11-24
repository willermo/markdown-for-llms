"""The rule of the close in `session.py run` (MFL023 B7, point 1 of the plan r3 approved by Davide; SOL2-B1 of the plan r2).

The command line writes in the same ledger as the product (D6) and closes its window by the same rule,
`serverspend.closing`, with the largest drift of the wall clock on the monotonic clock read in its waits: a suspend
(the wall clock jumps, the monotonic one does not) or a zero after the bound of the window leaves an orphan with
`uncertain`, the event `uncertain` and exit 4 after 130 and 5; the provider at its cap, new arms refused until a manual
reconciliation; a zero in time keeps the row of today. A drift that goes back is judged at its largest. No network: a
fake provider and the CLI of Modal faked."""
from decimal import Decimal
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
import types
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts" / "servers"))
import providers as pv  # noqa: E402
import session  # noqa: E402

from markdown_for_llms.bench import serverspend, spend  # noqa: E402
from markdown_for_llms.bench.codes import BenchError, Code  # noqa: E402

URL = "https://b7--mdf-surya-serve.modal.run"


def env_file(test):
    d = Path(tempfile.mkdtemp())
    test.addCleanup(shutil.rmtree, d, True)
    path = d / ".env"
    path.write_text("MODAL_PROXY_TOKEN_ID=wk-b7\nMODAL_PROXY_TOKEN_SECRET=ws-b7\nRUNPOD_API_KEY=rk-b7\n")
    path.chmod(0o600)
    return path


class Clocks:
    """A wall clock and a monotonic clock: `sleep` moves both; `jumps` move the wall clock alone at the next sleeps (a
    suspend: CLOCK_MONOTONIC does not count it on Linux)."""

    def __init__(self):
        self.wall, self.mono, self.jumps = 1_800_000_000.0, 1000.0, []

    def time(self):
        return self.wall

    def monotonic(self):
        return self.mono

    def sleep(self, s):
        self.wall += s + (self.jumps.pop(0) if self.jumps else 0.0)
        self.mono += s


class FakeRunPod:
    provider = "runpod"
    url = "https://fake-endpoint.api.runpod.ai"

    def __init__(self, zero_after=0):
        self.armed = self.busy = False
        self.zero_after, self.zeros = zero_after, 0

    def state(self):
        return {"armed": self.armed, "busy": self.busy, "summary": {"total": int(self.busy)}, "workers": []}

    def arm(self, fault, allow_download, deadline=None, **levers):
        self.armed = self.busy = True
        return {}

    def disarm(self):
        self.armed = False

    def zero(self):
        self.zeros += 1
        if self.zeros <= self.zero_after:
            return False, self.state()
        self.busy = False
        return True, self.state()

    def balance(self):
        return 1.0


class TestTheCloseOfTheCommandLine(unittest.TestCase):
    """RunPod, A = 0.35 USD, cap 2 USD (W = 1463 s, the R4 of the plan r2)."""

    def run_session(self, prov, jumps=(), runs_to_t=False):
        env = env_file(self)
        self.ledger = env.parent / "spend.jsonl"
        a = session.parser().parse_args(["run", "--provider", "runpod", "--name", "mdf-surya", "--env-file", str(env),
                                         "--spend-ledger", str(self.ledger), "--spend-cap-usd", "2",
                                         "--cap-usd", "0.35", "--", "true"])
        a.command = a.command[1:] if a.command[:1] == ["--"] else a.command
        clocks = Clocks()
        self.out = io.StringIO()
        ses = session.Session(a, prov, clock=clocks.time, sleep=clocks.sleep, out=self.out)
        ses.mono = clocks.monotonic                       # the monotonic clock of the test, beside its wall clock
        child = mock.MagicMock()
        polls = {"n": 0}

        def poll():
            polls["n"] += 1
            return None if (runs_to_t or polls["n"] <= len(jumps) + 1) else 0
        child.poll.side_effect = poll
        child.returncode = 0
        child.pid = 4242

        def popen(*args, **kw):
            clocks.jumps = list(jumps)                  # the laptop sleeps while the command runs
            return child
        with mock.patch.object(session.subprocess, "Popen", side_effect=popen):
            code = ses.run()
        rows = spend.read(self.ledger)
        arm, = [r for r in rows if r.get("kind") == "arm"]
        dis, = [r for r in rows if r.get("kind") == "disarm"]
        return code, arm, dis, serverspend.summary(rows, "runpod", self.ledger), ses

    def assertHeldAtTheCap(self, arm, dis, summary, reason):
        self.assertEqual((dis["orphan"], dis["clean"], dis.get("uncertain")), (True, False, reason))
        report = json.loads(self.out.getvalue().splitlines()[-1])
        self.assertIn(reason, [e.get("reason") for e in report["events"] if e["event"] == "uncertain"])
        self.assertEqual(Decimal(summary["total_cents"]), Decimal("200.00"))
        self.assertEqual(summary["orphans"], [arm["arm_id"]])
        with spend.Ledger.file(self.ledger).locked() as locked:
            with self.assertRaises(BenchError) as caught:
                serverspend.arm(locked, self.ledger, "runpod", "mdf-surya", arm["url"], "10", 9e9, "200",
                                "0.0002090948", 9e9)
            self.assertEqual(caught.exception.code, Code.SESSION_ORPHANED)
            serverspend.reconcile(locked, self.ledger, "runpod", "20", manual=True, arms=[arm["arm_id"]], orphan=True)
            serverspend.arm(locked, self.ledger, "runpod", "mdf-surya", arm["url"], "10", 9e9, "200", "0.0002090948",
                            9e9)

    def test_a_suspend_of_three_hours(self):
        code, arm, dis, summary, ses = self.run_session(FakeRunPod(), jumps=[10800.0])
        self.assertEqual(arm["window_s"], 1463)
        self.assertHeldAtTheCap(arm, dis, summary, "unobserved_interval")
        self.assertEqual(code, 5)                       # the term T came first (exit 5); the row holds the block

    def test_a_short_suspend_before_the_term(self):
        code, arm, dis, summary, ses = self.run_session(FakeRunPod(), jumps=[10.0])
        self.assertLess(dis["t_zero"], arm["until"])
        self.assertHeldAtTheCap(arm, dis, summary, "unobserved_interval")
        self.assertEqual(code, 4)

    def test_a_drift_that_goes_back_counts_at_its_largest(self):
        """+10 s, then -10 s: the last difference is 0, the largest is 10 s (note of Sol on the plan r3)."""
        code, arm, dis, summary, ses = self.run_session(FakeRunPod(), jumps=[10.0, -10.0])
        self.assertAlmostEqual(ses.max_drift, 10.0)
        self.assertAlmostEqual(dis["drift_s"], 10.0)
        self.assertHeldAtTheCap(arm, dis, summary, "unobserved_interval")
        self.assertEqual(code, 4)

    def test_a_drift_within_the_limit(self):
        code, arm, dis, summary, ses = self.run_session(FakeRunPod(), jumps=[3.0])
        self.assertEqual((dis["orphan"], dis.get("uncertain")), (False, None))
        self.assertEqual(code, 0)

    def test_a_zero_after_the_bound_of_the_window(self):
        code, arm, dis, summary, ses = self.run_session(FakeRunPod(zero_after=19), runs_to_t=True)
        self.assertGreater(float(dis["t_zero"]), float(arm["until"]) + 180)        # T + 60 + Z passed, no jump
        self.assertHeldAtTheCap(arm, dis, summary, "over_window")
        self.assertEqual(code, 5)

    def test_a_zero_in_time_is_unchanged(self):
        code, arm, dis, summary, ses = self.run_session(FakeRunPod(zero_after=2))
        self.assertEqual((dis["orphan"], dis.get("uncertain")), (False, None))
        window, = summary["windows"]
        self.assertEqual(window["status"], "unclean")
        self.assertLessEqual(Decimal(window["amount_cents"]), Decimal(arm["a_cents"]))
        self.assertEqual(code, 0)

    def test_the_real_clock_reads_the_monotonic_one(self):
        a = types.SimpleNamespace(spend_ledger="/nonexistent/spend.jsonl")
        import time
        self.assertIs(session.Session(a, FakeRunPod()).mono, time.monotonic)
        fake = lambda: 1.0                               # noqa: E731
        self.assertIs(session.Session(a, FakeRunPod(), clock=fake).mono, fake)


class TestTheDeployPastTheTerm(unittest.TestCase):
    def test_the_window_of_test_modal_deploy_past_the_term(self):
        """`test_modal.py:283` (S-B4 of MFL033): the fake deploy moves the clock 2000 s; that test pins exit 5, `clean`
        false and the text of the event. The window was 73.55 cents with A at 30 (plan r3, fact 11): now an orphan
        held at the cap, never a window over A."""
        env = env_file(self)
        ledger = env.parent / "spend.jsonl"
        clocks = Clocks()

        def run(argv, **kw):
            args = argv[len(pv.MODAL_SDK):]
            word = args[0] if args[0] == "deploy" else " ".join(args[:2])
            if word == "deploy":
                clocks.wall += 2000
                clocks.mono += 2000                      # a slow deploy, not a suspend
                return types.SimpleNamespace(returncode=0, stdout="Created web function serve => %s\n" % URL, stderr="")
            out = {"volume list": json.dumps([{"name": pv.VOLUME}]),
                   "workspace proxy-tokens": json.dumps([{"token_id": "wk-b7", "scoped": False}])}.get(word, "[]")
            return types.SimpleNamespace(returncode=0, stdout=out, stderr="")
        a = session.parser().parse_args(["run", "--provider", "modal", "--name", "mdf-surya", "--env-file", str(env),
                                         "--spend-ledger", str(ledger), "--spend-cap-usd", "15", "--cap-usd", "0.30",
                                         "--", "true"])
        a.command = a.command[1:] if a.command[:1] == ["--"] else a.command
        m = pv.Modal("mdf-surya", runner=run, clock=clocks.time, sleep=clocks.sleep, environ={}, env_file=str(env))
        m.log_follower = None
        ses = session.Session(a, m, clock=clocks.time, sleep=clocks.sleep, out=io.StringIO())
        ses.mono = clocks.monotonic
        with mock.patch.object(session.subprocess, "Popen") as popen:
            code = ses.run()
        popen.assert_not_called()
        rows = spend.read(ledger)
        arm, = [r for r in rows if r.get("kind") == "arm"]
        dis, = [r for r in rows if r.get("kind") == "disarm"]
        self.assertEqual(code, 5)
        self.assertEqual((dis["orphan"], dis["uncertain"]), (True, "over_window"))
        window, = serverspend.summary(rows, "modal", ledger)["windows"]
        self.assertEqual(window["status"], "orphan")


if __name__ == "__main__":
    unittest.main()
