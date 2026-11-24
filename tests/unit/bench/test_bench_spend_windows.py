"""The ledger of the rented servers (MFL033), without network, with a fake price and a fake clock:
plan r2 §11.9, r3 §11.9, r4 §11.10 and App. G, prescription 2 of the review of r4."""
from decimal import Decimal
import fcntl
import io
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from contextlib import redirect_stdout

from markdown_for_llms.bench import cli as bench_cli, serverspend, spend
from markdown_for_llms.bench.codes import BenchError, Code
from markdown_for_llms.engine import servers
from markdown_for_llms.engine.codes import EngineError

RATE = Decimal("0.75") / 3600 + Decimal(20) * Decimal("0.10") / (Decimal("30.4") * 86400)
URL = "https://abc.api.runpod.ai"
EP = servers.endpoint("rp", {"kind": "runpod", "url": URL, "rate_usd_per_s": str(RATE)})


def cents(seconds):
    return (Decimal(seconds) * RATE * 100).quantize(Decimal("0.01"), rounding="ROUND_CEILING")


class Ledger(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.dir, True)
        self.ledger = spend.Ledger(self.dir)
        self.now = [100_000.0]
        self.held = {}

    def clock(self):
        return self.now[0]

    def entries(self):
        return spend.read(self.ledger.path)

    def summary(self):
        return serverspend.summary(self.entries(), "runpod", self.ledger.path)

    def arm(self, a_usd="1.00", until=None, cap_usd="4", live=True):
        with self.ledger.locked() as locked:
            row = serverspend.arm(locked, self.ledger.path, "runpod", "rp", URL, Decimal(a_usd) * 100,
                                  until if until is not None else self.now[0] + 7200, Decimal(cap_usd) * 100, RATE,
                                  self.now[0])
        if live:
            self.hold(row["arm_id"])
        return row["arm_id"]

    def hold(self, arm_id):
        fd = os.open(serverspend.lock_path(self.ledger.path, arm_id), os.O_RDWR | os.O_CREAT | os.O_CLOEXEC, 0o600)
        fcntl.flock(fd, fcntl.LOCK_EX)
        self.held[arm_id] = fd
        self.addCleanup(self.release, arm_id)

    def release(self, arm_id):
        fd = self.held.pop(arm_id, None)
        if fd is not None:
            os.close(fd)

    def disarm(self, arm_id, clean=True, zero_after=30, orphan=False):
        self.release(arm_id)
        with self.ledger.locked() as locked:
            serverspend.disarm(locked, "runpod", arm_id, self.now[0] + zero_after, clean, orphan=orphan)

    def attempt(self, arm_id, aid, pages=1, ready=True, clean=True, run_s=300, last_before_close=5, run="r1",
                complete=True, command_left=10_000):
        """A reservation at the wake (the term of the engine with the budget of the session), then,
        when `complete`, the completion with the bound of the engine. Returns the reservation."""
        s = serverspend.session(self.ledger.path, "runpod", URL, {serverspend.SESSION_ENV: arm_id}, self.entries())
        t = servers.term(EP, pages, budget_cents=min(Decimal(s["window_left_cents"]), Decimal(command_left)),
                         until=s["armed_until"], now=self.now[0])
        info = {"reserve_cents": t["reserve_cents"], "rate": RATE, "tail_idle_s": EP.tail_idle_s,
                "identity": EP.identity_sha256(), "endpoint": "rp", "url": URL, "D": t["D"]}
        with self.ledger.locked() as locked:
            row = serverspend.reserve(locked, self.ledger.path, "runpod", arm_id, run, aid, info, command_left,
                                      clock=self.clock)
        wake = self.now[0] + 1
        close = wake + min(run_s, t["D"] - 5)
        last = close - last_before_close
        if complete:
            bound = servers.bound_cents(EP, ready, clean, wake, last if ready else None, close)
            result = {"status": "complete" if clean else "failed", "cost": {"bound_cents": bound, "clean": clean and ready},
                      "marker_server": {"t_wake": wake, "t_last": last if ready else None, "t_close": close}}
            with self.ledger.locked() as locked:
                serverspend.complete(locked, "runpod", run, aid, result)
        self.now[0] = close
        return row, t


class TestCaps(Ledger):
    def test_the_cap_is_reached_in_the_middle_of_a_batch_before_a_wake(self):
        arm_id = self.arm("1.00")
        done = 0
        for i in range(10):
            try:
                self.attempt(arm_id, "a%d" % i, run_s=600)
            except (EngineError, BenchError) as exc:
                self.assertEqual(exc.code, Code.SERVER_CAP)
                break
            done += 1
            self.now[0] += 200
        self.assertGreaterEqual(done, 2)
        rows = [e for e in self.entries() if e.get("kind") == "reserve"]
        self.assertEqual(len(rows), done)                    # nothing reserved for the refused attempt
        s = self.summary()
        self.assertLessEqual(Decimal(s["total_cents"]), 100)

    def test_the_session_reservation_is_checked_again_under_the_lock(self):
        arm_id = self.arm("0.50")
        info = {"reserve_cents": "40.00", "rate": RATE, "tail_idle_s": 90, "identity": "x", "endpoint": "rp", "url": URL,
                "D": 1}
        with self.ledger.locked() as locked:
            serverspend.reserve(locked, self.ledger.path, "runpod", arm_id, "r", "a", info, 1000, clock=self.clock)
            with self.assertRaises(BenchError) as caught:
                serverspend.reserve(locked, self.ledger.path, "runpod", arm_id, "r", "b", info, 1000, clock=self.clock)
            self.assertEqual(caught.exception.code, Code.SERVER_CAP)
            with self.assertRaises(BenchError) as caught:      # the command cap
                serverspend.reserve(locked, self.ledger.path, "runpod", arm_id, "r", "c",
                                    dict(info, reserve_cents="5"), 4, clock=self.clock)
            self.assertEqual(caught.exception.code, Code.SERVER_CAP)

    def test_arm_within_the_cap_of_the_provider(self):
        self.arm("3.00")
        with self.ledger.locked() as locked:
            with self.assertRaises(BenchError) as caught:
                serverspend.arm(locked, self.ledger.path, "runpod", "rp", URL, 150, 1e9, 400, RATE, self.now[0])
        self.assertEqual(caught.exception.code, Code.SERVER_CAP)


class TestAttempts(Ledger):
    def test_a_crash_between_reservation_and_completion_keeps_the_reservation(self):
        arm_id = self.arm("2.00")
        row, _ = self.attempt(arm_id, "g1", complete=False)
        self.disarm(arm_id, clean=True)
        s = self.summary()
        self.assertEqual(s["windows"][0]["status"], "unclean")       # an attempt without completion
        self.assertGreaterEqual(Decimal(s["total_cents"]), Decimal(row["reserve_cents"]))
        arm2 = self.arm("2.00")
        row2, _ = self.attempt(arm2, "g2")                            # the resumption: a new generation
        self.assertNotEqual(row2["attempt_id"], row["attempt_id"])
        self.assertTrue(spend.has(self.entries(), "reserve", "r1", "g1", provider="runpod"))

    def test_the_short_tail_is_cut_by_a_following_attempt(self):
        arm_id = self.arm("2.00")
        self.attempt(arm_id, "a1", run_s=300)
        self.now[0] += 20                                             # inside the 90 s of the tail
        second, _ = self.attempt(arm_id, "a2", run_s=300)
        self.assertEqual(second["follows"], {"run_id": "r1", "attempt_id": "a1"})
        p = serverspend.parse(self.entries())["runpod"]
        amounts = p.window_attempts(arm_id)
        first = p.attempts[("r1", "a1")]["complete"]
        self.assertEqual(amounts[("r1", "a1")], cents(Decimal(str(second["t_reserve"])) - Decimal(str(first["t_wake"]))))
        self.disarm(arm_id)
        s = self.summary()
        self.assertEqual(s["windows"][0]["status"], "clean")
        self.assertEqual(Decimal(s["total_cents"]), sum(amounts.values()))
        # without a follower the whole short tail stays
        arm2 = self.arm("2.00")
        self.attempt(arm2, "b1", run_s=300, run="r2")
        p = serverspend.parse(self.entries())["runpod"]
        c = p.attempts[("r2", "b1")]["complete"]
        self.assertEqual(p.window_attempts(arm2)[("r2", "b1")], Decimal(c["bound_cents"]))

    def test_no_follows_after_an_unclean_close(self):
        arm_id = self.arm("3.00")
        self.attempt(arm_id, "a1", clean=False)
        self.now[0] += 20
        second, _ = self.attempt(arm_id, "a2")
        self.assertIsNone(second["follows"])

    def test_a_crash_between_follows_and_the_wake(self):
        """The tail of the previous attempt is cut, the next reservation stays whole and covers the
        rest: the total stays over what a provider billing every second the GPU was on charges."""
        arm_id = self.arm("2.00")
        self.attempt(arm_id, "a1", run_s=300)
        first = serverspend.parse(self.entries())["runpod"].attempts[("r1", "a1")]["complete"]
        self.now[0] += 20
        second, _ = self.attempt(arm_id, "a2", complete=False)
        self.disarm(arm_id, clean=False)
        p = serverspend.parse(self.entries())["runpod"]
        amounts = p.window_attempts(arm_id)
        self.assertEqual(amounts[("r1", "a2")], Decimal(second["reserve_cents"]))
        billed = cents(Decimal(str(first["t_last"])) + 60 - Decimal(str(first["t_wake"])))   # the idle of the provider
        self.assertGreaterEqual(sum(amounts.values()), billed)

    def test_a_local_timeout_with_a_provider_that_bills_360_s_more(self):
        arm_id = self.arm("2.00")
        self.attempt(arm_id, "a1", clean=False, run_s=500)
        c = serverspend.parse(self.entries())["runpod"].attempts[("r1", "a1")]["complete"]
        billed = cents(Decimal(str(c["t_close"])) + 360 - Decimal(str(c["t_wake"])))
        self.assertGreaterEqual(Decimal(c["bound_cents"]), billed)

    def test_an_interruption_during_the_readiness(self):
        arm_id = self.arm("2.00")
        _, t = self.attempt(arm_id, "a1", ready=False, clean=False, run_s=10)
        c = serverspend.parse(self.entries())["runpod"].attempts[("r1", "a1")]["complete"]
        self.assertEqual(Decimal(c["bound_cents"]), cents(900 + 385))
        self.assertLessEqual(Decimal(c["bound_cents"]), Decimal(t["reserve_cents"]))


class TestWindows(Ledger):
    def test_open_clean_unclean_and_the_fault_window_without_attempts(self):
        a1 = self.arm("1.00")
        s = self.summary()
        self.assertEqual((s["windows"][0]["status"], Decimal(s["total_cents"])), ("open", Decimal(100)))
        self.attempt(a1, "x1")
        self.assertEqual(Decimal(self.summary()["total_cents"]), Decimal(100))      # attempts draw from A
        self.disarm(a1, clean=True)
        s = self.summary()
        self.assertEqual(s["windows"][0]["status"], "clean")
        clean_total = Decimal(s["total_cents"])
        self.assertLess(clean_total, 100)
        a2 = self.arm("1.00")                                   # a driver: exit 0, no attempt of the engine
        self.now[0] += 400
        self.disarm(a2, clean=True, zero_after=30)
        s = self.summary()
        w2 = next(w for w in s["windows"] if w["arm_id"] == a2)
        self.assertEqual(w2["status"], "unclean")
        self.assertEqual(Decimal(w2["amount_cents"]), cents(430))

    def test_orphan_pins_the_provider_at_its_cap_until_the_manual_reconciliation(self):
        a1 = self.arm("0.30", cap_usd="4", live=False)          # SIGKILL of the session: the lock is free
        s = self.summary()
        self.assertEqual((s["orphans"], Decimal(s["total_cents"])), ([a1], Decimal(400)))
        with self.assertRaises(BenchError) as caught:
            serverspend.session(self.ledger.path, "runpod", URL, {serverspend.SESSION_ENV: "none"}, self.entries())
        self.assertEqual(caught.exception.code, Code.SESSION_ORPHANED)       # before server_not_armed
        with self.ledger.locked() as locked:
            with self.assertRaises(BenchError) as caught:
                serverspend.arm(locked, self.ledger.path, "runpod", "rp", URL, 10, 1e9, 400, RATE, self.now[0])
            self.assertEqual(caught.exception.code, Code.SESSION_ORPHANED)
            with self.assertRaises(BenchError) as caught:                     # no disarm with orphan: true yet
                serverspend.reconcile(locked, self.ledger.path, "runpod", 12, True, arms=[a1], orphan=True)
            self.assertEqual(caught.exception.code, Code.INVALID_SETTING)
        self.disarm(a1, orphan=True)                                          # session.py disarm
        self.assertEqual(Decimal(self.summary()["total_cents"]), Decimal(400))
        with self.ledger.locked() as locked:
            self.assertEqual(serverspend.orphan_windows(locked.entries(), "runpod", self.ledger.path), [a1])
            serverspend.reconcile(locked, self.ledger.path, "runpod", Decimal("12.50"), True, arms=[a1], orphan=True)
        s = self.summary()
        self.assertEqual((s["orphans"], Decimal(s["total_cents"])), ([], Decimal("12.50")))
        self.arm("1.00")                                                      # the provider is usable again

    def test_reconciliations(self):
        a1 = self.arm("1.00")
        self.attempt(a1, "x1")
        self.disarm(a1, clean=True)
        a2 = self.arm("1.00")
        self.attempt(a2, "y1", clean=False)
        self.disarm(a2, clean=True)
        base = {w["arm_id"]: Decimal(w["amount_cents"]) for w in self.summary()["windows"]}
        with self.ledger.locked() as locked:
            with self.assertRaises(BenchError) as caught:      # two equal readings never free a fault window
                serverspend.reconcile(locked, self.ledger.path, "runpod", 1, False, arms=[a2])
            self.assertEqual(caught.exception.code, Code.INVALID_SETTING)
            row, s = serverspend.reconcile(locked, self.ledger.path, "runpod", 3, False, arms=[a1])
            self.assertIsNotNone(row)
            again, s2 = serverspend.reconcile(locked, self.ledger.path, "runpod", 3, False, arms=[a1])
            self.assertIsNone(again)                           # idempotent
            self.assertEqual(s["total_cents"], s2["total_cents"])
            self.assertEqual(Decimal(s["total_cents"]), 3 + base[a2])
            serverspend.reconcile(locked, self.ledger.path, "runpod", 2, True, arms=[a2])     # manual: any window
        self.assertEqual(Decimal(self.summary()["total_cents"]), 5)
        with self.ledger.locked() as locked:                   # a figure over the ledger: a persistent stop
            with self.assertRaises(BenchError) as caught:
                serverspend.reconcile(locked, self.ledger.path, "runpod", base[a1] + 1, True, arms=[a1])
            self.assertEqual(caught.exception.code, Code.SERVER_BILLED_OVER_BOUND)
            with self.assertRaises(BenchError) as caught:
                serverspend.arm(locked, self.ledger.path, "runpod", "rp", URL, 10, 1e9, 400, RATE, self.now[0])
            self.assertEqual(caught.exception.code, Code.SERVER_BILLED_OVER_BOUND)

    def test_without_a_session(self):
        for environ in ({}, {serverspend.SESSION_ENV: "nope"}):
            with self.assertRaises(BenchError) as caught:
                serverspend.session(self.ledger.path, "runpod", URL, environ, self.entries())
            self.assertEqual(caught.exception.code, Code.SERVER_NOT_ARMED)
        a1 = self.arm("1.00")
        with self.assertRaises(BenchError) as caught:          # another endpoint
            serverspend.session(self.ledger.path, "runpod", "https://other.api.runpod.ai",
                                {serverspend.SESSION_ENV: a1}, self.entries())
        self.assertEqual(caught.exception.code, Code.SERVER_NOT_ARMED)
        s = serverspend.session(self.ledger.path, "runpod", URL, {serverspend.SESSION_ENV: a1}, self.entries())
        self.assertEqual((s["arm_id"], s["window_left_cents"]), (a1, "100.00"))
        self.disarm(a1)
        with self.assertRaises(BenchError) as caught:
            serverspend.session(self.ledger.path, "runpod", URL, {serverspend.SESSION_ENV: a1}, self.entries())
        self.assertEqual(caught.exception.code, Code.SERVER_NOT_ARMED)


class TestLedgerShared(Ledger):
    def test_datalab_rows_and_server_rows_are_apart(self):
        with self.ledger.locked() as locked:
            locked.append({"kind": "reserve", "run_id": "d", "attempt_id": "1", "estimate_cents": "18.60"})
            locked.append({"kind": "complete", "run_id": "d", "attempt_id": "1", "billed_cents": "13.20"})
        a1 = self.arm("1.00")
        self.attempt(a1, "x1", run="d")
        self.assertEqual(spend.totals(self.entries(), "d"), (Decimal("13.20"), Decimal("13.20")))
        self.assertFalse(spend.has(self.entries(), "reserve", "d", "x1"))
        with self.ledger.locked() as locked:
            locked.append({"kind": "mystery", "provider": "runpod"})
        with self.assertRaises(BenchError) as caught:
            self.summary()
        self.assertEqual(caught.exception.code, Code.SPEND_LEDGER_INCONSISTENT)

    def test_the_cli_shows_and_reconciles(self):
        a1 = self.arm("0.30", live=False)
        self.disarm(a1, orphan=True)
        with redirect_stdout(io.StringIO()) as out:
            self.assertEqual(bench_cli.main(["spend", "show", "--runs-dir", str(self.dir)]), 0)
        shown = json.loads(out.getvalue())
        self.assertEqual(shown["providers"]["runpod"]["orphans"], [a1])
        self.assertEqual(shown["providers"]["modal"]["total_cents"], "0.00")
        with redirect_stdout(io.StringIO()) as out:
            self.assertEqual(bench_cli.main(["spend", "reconcile", "--runs-dir", str(self.dir), "--provider", "runpod",
                                             "--orphan", "--manual-cents", "7"]), 0)
        self.assertEqual(json.loads(out.getvalue())["summary"]["total_cents"], "7.00")
        self.assertEqual(bench_cli.main(["spend", "reconcile", "--runs-dir", str(self.dir), "--provider", "runpod",
                                         "--orphan", "--manual-cents", "7"]), 1)       # nothing orphan left


if __name__ == "__main__":
    unittest.main()


class TestBoundTail(Ledger):
    """MFL033 c7.19 (prompt N4s §4.2): the bound of an unclean window runs to the later of the zero
    and the stop of the container (`t_stop` of the disarm row), plus the `tail_s` of the arm row;
    `double_s` counts again the seconds with two workers RUNNING. A row without them counts as before."""

    def window(self, arm_extra=None, disarm_extra=None, zero_after=30):
        with self.ledger.locked() as locked:
            row = serverspend.arm(locked, self.ledger.path, "runpod", "rp", URL, Decimal(100), self.now[0] + 7200,
                                  Decimal(400), RATE, self.now[0], extra=arm_extra)
        with self.ledger.locked() as locked:
            serverspend.disarm(locked, "runpod", row["arm_id"], self.now[0] + zero_after, False, extra=disarm_extra)
        w, = [x for x in self.summary()["windows"] if x["arm_id"] == row["arm_id"]]
        self.assertEqual(w["status"], "unclean")
        return Decimal(w["amount_cents"])

    def test_the_tail_the_stop_and_two_workers(self):
        t = self.now[0]
        self.assertEqual(self.window(), cents(30))                                       # as before c7.19
        self.assertEqual(self.window({"tail_s": 30}), cents(60))
        self.assertEqual(self.window({"tail_s": 30}, {"t_stop": t + 31.2, "double_s": 0.0}), cents("61.2"))
        self.assertEqual(self.window({"tail_s": 30}, {"t_stop": t + 10}), cents(60))     # a stop before the zero
        self.assertEqual(self.window({"tail_s": 30}, {"t_stop": None, "double_s": 40}), cents(100))
        self.assertEqual(self.window({"tail_s": 0}, {"t_stop": t + 31.2}), cents("31.2"))

    def test_negative_or_unreadable_values_are_refused(self):
        for arm_extra, disarm_extra in (({"tail_s": -1}, None), (None, {"double_s": -0.5}), (None, {"t_stop": "x"}),
                                        ({"tail_s": True}, None)):
            with self.subTest(arm=arm_extra, disarm=disarm_extra):
                self.ledger = spend.Ledger(Path(tempfile.mkdtemp(dir=self.dir)))
                with self.assertRaises(BenchError) as ctx:
                    self.window(arm_extra, disarm_extra)
                self.assertEqual(ctx.exception.code, Code.SPEND_LEDGER_INCONSISTENT)


class TestTheRateOfTheArm(Ledger):
    """S-B6 of the final review of MFL033, the case of the probe of Sol: the session armed Modal at the rate
    of its dearest GPU (0.00036772 USD/s) and servers.json still holds the L4 alone (0.00028372 USD/s). The
    attempt would reserve and bound its cost at the lower rate; it is refused before any request, and the
    reservation checks it again under the lock. The same for an endpoint `user` with a provider."""

    ARMED, STALE = Decimal("0.00036772"), Decimal("0.00028372")
    MODAL_URL = "https://ws--mdf-surya-serve.modal.run"

    def modal_arm(self):
        with self.ledger.locked() as locked:
            row = serverspend.arm(locked, self.ledger.path, "modal", "mdf-surya", self.MODAL_URL, Decimal(500),
                                  self.now[0] + 7200, Decimal(1500), self.ARMED, self.now[0])
        self.hold(row["arm_id"])
        return row["arm_id"]

    def endpoints(self, rate):
        return [servers.endpoint("md", {"kind": "modal", "url": self.MODAL_URL, "rate_usd_per_s": str(rate)}),
                servers.endpoint("md-user", {"kind": "user", "provider": "modal", "url": self.MODAL_URL,
                                             "rate_usd_per_s": str(rate), "ready_within": 1200})]

    def test_a_stale_rate_is_refused_before_the_wake_and_under_the_lock(self):
        arm_id = self.modal_arm()
        environ = {serverspend.SESSION_ENV: arm_id}
        for ep in self.endpoints(self.STALE):
            with self.subTest(kind=ep.kind):
                # the reservation, as the engine asks it at the wake, under the lock of the ledger
                s = serverspend.session(self.ledger.path, "modal", self.MODAL_URL, environ, self.entries())
                t = servers.term(ep, 1, budget_cents=Decimal(s["window_left_cents"]), until=s["armed_until"],
                                 now=self.now[0])
                info = {"reserve_cents": t["reserve_cents"], "rate": ep.rate_usd_per_s, "tail_idle_s": ep.tail_idle_s,
                        "identity": ep.identity_sha256(), "endpoint": ep.name, "url": self.MODAL_URL, "D": t["D"]}
                with self.ledger.locked() as locked:
                    with self.assertRaises(EngineError) as caught:
                        serverspend.reserve(locked, self.ledger.path, "modal", arm_id, "r1", ep.kind, info, 10_000,
                                            clock=self.clock)
                self.assertEqual(caught.exception.code, Code.SERVER_NOT_CONFIGURED)
                # and the session that the callers open before the attempt
                with self.assertRaises(EngineError) as caught:
                    serverspend.session(self.ledger.path, "modal", self.MODAL_URL, environ, self.entries(), rate=ep.rate)
                self.assertEqual(caught.exception.code, Code.SERVER_NOT_CONFIGURED)
                self.assertIn("0.00028372", caught.exception.detail)
                self.assertIn("0.00036772", caught.exception.detail)
        self.assertEqual([e["kind"] for e in self.entries() if e.get("provider") == "modal"], ["arm"])

    def test_the_rate_of_the_arm_passes(self):
        arm_id = self.modal_arm()
        environ = {serverspend.SESSION_ENV: arm_id}
        for ep in self.endpoints(self.ARMED):
            with self.subTest(kind=ep.kind):
                s = serverspend.session(self.ledger.path, "modal", self.MODAL_URL, environ, self.entries(), rate=ep.rate)
                t = servers.term(ep, 1, budget_cents=Decimal(s["window_left_cents"]), until=s["armed_until"],
                                 now=self.now[0])
                info = {"reserve_cents": t["reserve_cents"], "rate": ep.rate_usd_per_s, "tail_idle_s": ep.tail_idle_s,
                        "identity": ep.identity_sha256(), "endpoint": ep.name, "url": self.MODAL_URL, "D": t["D"]}
                with self.ledger.locked() as locked:
                    row = serverspend.reserve(locked, self.ledger.path, "modal", arm_id, "r1", ep.kind, info, 10_000,
                                              clock=self.clock)
                self.assertEqual(Decimal(row["rate"]), self.ARMED)
