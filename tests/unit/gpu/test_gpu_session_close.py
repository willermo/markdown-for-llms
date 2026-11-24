"""The close of a GPU session of the product that cannot be certified (MFL023 B4; SOL2-B1 of the plan r2, plan r3 §4.3,
§6 point 5, and the notes of Sol in the review of r3).

The clock thread keeps the largest drift of the wall clock on the monotonic one (a suspend of the laptop, a step of its
clock): over 5 s it asks for the close at once, also before T, and `serverspend.closing` makes the window an orphan
with `uncertain` (`unobserved_interval`), the provider at its cap until a reconciliation with the invoice, and no new
session on it until then. A zero observed too late for the reserve of the window (a slow disarm, a slow zero, after T)
is `over_window`. The cleanup has its own budget of 600 s of the monotonic clock, also after T, and its children are
never killed at T; without a zero within it the session is STOPPED (no row: the window stays orphan). Clocks given to
the worker; the fake provider of `provider_driver.py`."""
from decimal import Decimal
from pathlib import Path
import sys
import threading
import time
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gpu_harness import Instance, gpu_convert, wait_for  # noqa: E402

from markdown_for_llms.bench import serverspend, spend  # noqa: E402

W_RUNPOD = 1703          # RunPod at 0.40 USD
W_MODAL = 1179           # Modal at 0.50 USD


def process(inst, w, convert=None):
    row = w.claim()
    with mock.patch("markdown_for_llms.worker.convert", convert or gpu_convert(inst.clocks)):
        w.process(row)
    return inst.run(row["id"])


class Accelerator:
    """Time that passes fast, both clocks, while CLOSING runs: `step` seconds every 10 ms of real time."""

    def __init__(self, clocks, step):
        self.clocks, self.step, self.done = clocks, step, threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)

    def run(self):
        while not self.done.wait(0.01):
            self.clocks.advance(self.step)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.done.set()
        self.thread.join()


class Base(unittest.TestCase):
    def final(self, inst, state, within=30):
        wait_for(lambda: inst.sessions() and inst.sessions()[-1]["state"] == state, within, "the session " + state)

    def assertUncertain(self, inst, w, reason, cap="200.00"):
        session = inst.sessions()[-1]
        self.assertEqual((session["state"], session["reason"]), ("uncertain", reason))
        dis, = [r for r in inst.rows("disarm") if r["arm_id"] == session["arm_id"]]
        self.assertEqual((dis["orphan"], dis["clean"], dis["uncertain"]), (True, False, reason))
        summary = inst.summary()
        self.assertEqual(summary["total_cents"], cap)               # the cap of the provider, never over it
        self.assertIn(session["arm_id"], summary["orphans"])
        return session, dis

    def assertNoNewSession(self, inst, w):
        """A new job on the provider: refused with session_orphaned before any lock, row or call."""
        calls, sessions = inst.calls(), len(inst.sessions())
        inst.submit()
        run = process(inst, w)
        self.assertIn("session_orphaned", run["error"])
        self.assertEqual((inst.calls(), len(inst.sessions())), (calls, sessions))
        self.assertFalse(list(inst.ledger_dir.glob("*.lock")))

    def reconcile(self, inst, w, arm_id, cents="12.40"):
        ledger = spend.Ledger.file(inst.ledger_dir / spend.LEDGER)
        with ledger.locked() as locked:
            serverspend.reconcile(locked, ledger.path, inst.provider, cents, manual=True, arms=[arm_id], orphan=True)
        w.gpu.reconciled()
        session = [s for s in inst.sessions() if s["arm_id"] == arm_id][0]
        self.assertEqual((session["state"], session["reason"]), ("closed", "reconciled"))


class TestAnIntervalNotObserved(Base):
    def test_a_suspend_of_three_hours_during_a_job(self):
        inst = Instance(self)
        w = inst.worker()
        inst.submit()

        def during(observer):
            inst.clocks.jump(10800)                                 # the laptop sleeps: the wall clock alone
            self.final(inst, "uncertain")
        run = process(inst, w, gpu_convert(inst.clocks, during=during))
        self.assertIn("server_deadline", run["error"])
        session, dis = self.assertUncertain(inst, w, "unobserved_interval")
        self.assertGreater(dis["drift_s"], 10000)
        self.assertGreater(session["drift_s"], 10000)
        self.assertEqual(len(inst.rows("reserve")), 1)              # nothing reserved after the jump
        self.assertNoNewSession(inst, w)
        self.reconcile(inst, w, session["arm_id"])
        inst.submit()
        self.assertEqual(process(inst, w)["state"], "completed")    # released by the figure of the invoice

    def test_a_short_jump_long_before_the_term(self):
        """Ten seconds of suspend with T far: the same outcome. That no request leaves after the jump, before the clock
        thread reads its clocks again, is the relay's (`test_engine_relay_wall_term.py::test_a_short_jump_before_the_term`,
        with the request right after the jump)."""
        inst = Instance(self)
        w = inst.worker()
        inst.submit()

        def during(observer):
            inst.clocks.jump(10)
            self.final(inst, "uncertain")
        process(inst, w, gpu_convert(inst.clocks, during=during))
        arm, = inst.rows("arm")
        self.assertLess(inst.sessions()[0]["t_zero"], arm["until"])        # well before T
        self.assertUncertain(inst, w, "unobserved_interval")

    def test_a_drift_within_the_limit(self):
        inst = Instance(self)
        w = inst.worker()
        inst.submit()

        def during(observer):
            inst.clocks.jump(3)
            time.sleep(0.2)
        self.assertEqual(process(inst, w, gpu_convert(inst.clocks, during=during))["state"], "completed")
        inst.clocks.advance(61)
        w.gpu.poll()
        inst.settle(w)
        session = inst.sessions()[0]
        self.assertEqual((session["state"], session["reason"]), ("closed", "idle"))
        self.assertAlmostEqual(session["drift_s"], 3, places=1)

    def test_a_drift_that_goes_back_counts_at_its_largest(self):
        """+10 s, seen by the clock thread, then -10 s before the close: the difference at the end is 0, the largest
        drift observed is 10 s, and it is that one that decides (note of Sol)."""
        inst = Instance(self)
        w = inst.worker()
        inst.submit()

        def during(observer):
            inst.clocks.jump(10)
            wait_for(lambda: w.gpu.session.state.closing.is_set(), what="the drift seen")
            inst.clocks.jump(-10)
            self.final(inst, "uncertain")
        process(inst, w, gpu_convert(inst.clocks, during=during))
        session, dis = self.assertUncertain(inst, w, "unobserved_interval")
        self.assertAlmostEqual(dis["drift_s"], 10, places=1)

    def test_a_suspend_during_the_cleanup(self):
        """The laptop sleeps while CLOSING waits for the zero: the budget of the cleanup counts the monotonic clock and
        is not spent; the close is not certified."""
        inst = Instance(self, state={"never_zero": True})
        w = inst.worker()
        inst.submit()
        process(inst, w)
        inst.clocks.advance(61)
        w.gpu.poll()
        wait_for(lambda: inst.calls().count("zero") >= 2, what="CLOSING waiting for the zero")
        inst.clocks.jump(3600)
        inst.change(never_zero=False)
        inst.settle(w)
        self.assertUncertain(inst, w, "unobserved_interval")


class TestALateZero(Base):
    def test_a_zero_after_the_bound_of_the_window(self):
        """No jump: the zero comes 200 s after T, later than T + 60 s + Z = T + 180 s."""
        inst = Instance(self, state={"zero_after": 20})
        w = inst.worker()
        inst.submit()
        process(inst, w)
        inst.clocks.advance(W_RUNPOD + 1)                           # T, both clocks: the clock thread closes
        self.final(inst, "uncertain")
        arm, = inst.rows("arm")
        self.assertGreater(inst.sessions()[0]["t_zero"], arm["until"] + 180)
        self.assertUncertain(inst, w, "over_window")

    def test_a_slow_disarm_twice(self):
        """Modal at 0.50 USD: the close at T, a disarm that fails and the next one, 150 s each; the zero after the bound."""
        inst = Instance(self, provider="modal", session_cap="0.50", provider_caps=("3", "2"), state={"fail": {"disarm": 1}})
        w = inst.worker()
        inst.submit()
        process(inst, w)
        stop = threading.Event()

        def slow():
            seen = 0
            while not stop.is_set():
                n = inst.calls().count("disarm")
                if n > seen:
                    inst.clocks.advance(150 * (n - seen))
                    seen = n
                time.sleep(0.01)
        t = threading.Thread(target=slow, daemon=True)
        t.start()
        self.addCleanup(stop.set)
        inst.clocks.advance(W_MODAL + 1)
        self.final(inst, "uncertain")
        stop.set()
        self.assertEqual(inst.calls().count("disarm"), 2)
        self.assertUncertain(inst, w, "over_window", cap="300.00")

    def test_the_cleanup_after_t_is_never_killed_at_t(self):
        """The close at T with a disarm that takes its time: the child is not killed at T (the cleanup has its own
        budget), and the row is the ordinary one."""
        inst = Instance(self, state={"hang": {"disarm": 2}})
        w = inst.worker()
        inst.submit()
        process(inst, w)
        inst.clocks.advance(W_RUNPOD + 1)
        self.final(inst, "closed")
        dis, = inst.rows("disarm")
        self.assertFalse(dis["orphan"])
        self.assertEqual(inst.sessions()[0]["reason"], "term")
        self.assertGreaterEqual(inst.provider_state()["ticks"], 20)       # the whole disarm ran

    def test_no_zero_within_the_budget_is_stopped(self):
        inst = Instance(self, state={"never_zero": True})
        w = inst.worker()
        inst.submit()
        process(inst, w)
        inst.clocks.advance(61)
        w.gpu.poll()
        with Accelerator(inst.clocks, 2):
            self.final(inst, "stopped", within=60)
        inst.settle(w)
        session = inst.sessions()[0]
        self.assertEqual(session["reason"], "zero_missing")
        self.assertEqual(inst.rows("disarm"), [])                    # no row: the window stays orphan
        summary = inst.summary()
        self.assertEqual((summary["total_cents"], summary["orphans"]), ("200.00", [session["arm_id"]]))
        calls = inst.calls()
        self.assertGreaterEqual(calls.count("zero"), 2)


if __name__ == "__main__":
    unittest.main()
