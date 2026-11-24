"""The close of a window by its own session and the URL of a reservation (MFL023 B2; SOL2-B1 of the plan r2, C-B2 of the
plan r1).

`serverspend.closing` certifies a close only when the session observed the whole window (the largest drift of its wall
clock on its monotonic clock within `DRIFT_LIMIT_S`) and the unclean bound up to the zero it observed stays within `A`;
otherwise the reason (`unobserved_interval`, `over_window`), and the `disarm` row with `orphan: true` and `uncertain`
holds the provider at its cap until a manual reconciliation, never a window cut at `A`. `unclean_cents` is the
calculation of the parser. A reservation draws only from the window of its own endpoint (`url`)."""
from decimal import Decimal
from pathlib import Path
import shutil
import tempfile
import unittest

from markdown_for_llms.bench import serverspend, spend
from markdown_for_llms.bench.codes import BenchError, Code
from markdown_for_llms.engine import servers

RATES = {"modal": "0.00036772", "runpod": "0.0002090948"}
CAPS = {"modal": "300", "runpod": "200"}


def arm(provider, a_cents, window_s, tail_s, t_arm=1000, arm_id="w"):
    return dict(kind="arm", provider=provider, arm_id=arm_id, endpoint="mdf-surya", url="https://x.example", t_arm=t_arm,
                until=t_arm + window_s, a_cents=a_cents, spend_cap_cents=CAPS[provider], rate=RATES[provider],
                tail_s=tail_s, window_s=window_s, z_s=120)


class TestClosing(unittest.TestCase):
    """The bounds of the live tests of the plan (r2 §7): Modal at 0.50 USD (W 1179 s), RunPod at 0.35 USD (W 1463 s,
    tail 30 s)."""

    def test_the_bound_of_each_provider(self):
        for provider, a, w, tail, at, over in (("modal", "50", 1179, 0, "49.98", "50.01"),
                                                ("runpod", "35", 1463, 30, "34.99", "35.01")):
            with self.subTest(provider=provider):
                row = arm(provider, a, w, tail)
                t_end = row["until"] + 60 + 120                               # T + 60 s + Z
                self.assertEqual(str(serverspend.unclean_cents(row, {"t_zero": t_end})), at)
                self.assertIsNone(serverspend.closing(row, t_end))
                self.assertEqual(str(serverspend.unclean_cents(row, {"t_zero": t_end + 1})), over)
                self.assertEqual(serverspend.closing(row, t_end + 1), "over_window")
                self.assertEqual(serverspend.closing(row, row["until"] + 10800), "over_window")
                self.assertEqual(serverspend.closing(row, row["until"], double_s=300), "over_window")
                self.assertEqual(serverspend.closing(row, row["until"], t_stop=t_end + 1), "over_window")

    def test_the_drift(self):
        row = arm("modal", "50", 1179, 0)
        for drift, reason in ((0, None), (5, None), (-5, None), (3.2, None), (5.001, "unobserved_interval"),
                              (-6, "unobserved_interval"), (10800, "unobserved_interval")):
            with self.subTest(drift=drift):
                self.assertEqual(serverspend.closing(row, row["until"], drift_s=drift), reason)
        self.assertEqual(serverspend.DRIFT_LIMIT_S, servers.DRIFT_LIMIT_S)
        self.assertEqual(serverspend.closing(row, row["until"] + 10800, drift_s=10800), "unobserved_interval")

    def test_unclean_cents_is_the_calculation_of_the_parser(self):
        cases = [({"t_zero": 2000}, 0), ({"t_zero": 2000, "t_stop": 2500}, 30), ({"t_zero": 2600, "t_stop": 2100}, 30),
                 ({"t_zero": 2000, "double_s": 45.5}, 0), ({"t_zero": 900}, 0)]
        for extra, tail in cases:
            with self.subTest(extra=extra):
                row = arm("runpod", "1000", 5000, tail)
                dis = dict(kind="disarm", provider="runpod", arm_id="w", clean=False, orphan=False, **extra)
                window, = serverspend.parse([row, dis])["runpod"].summary(lambda _: False)["windows"]
                self.assertEqual(window["status"], "unclean")
                self.assertEqual(Decimal(window["amount_cents"]), serverspend.unclean_cents(row, dis))


class Ledger(unittest.TestCase):
    def setUp(self):
        d = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, d, True)
        self.ledger = spend.Ledger.file(d / "spend.jsonl")

    def live_arm(self, provider="runpod", url="https://ep1.api.runpod.ai", a_cents="35"):
        arm_id = serverspend.new_arm_id()
        fd = serverspend.lock_path(self.ledger.path, arm_id)
        import fcntl
        import os
        handle = os.open(fd, os.O_RDWR | os.O_CREAT, 0o600)
        fcntl.flock(handle, fcntl.LOCK_EX)
        self.addCleanup(os.close, handle)
        with self.ledger.locked() as locked:
            return serverspend.arm(locked, self.ledger.path, provider, "mdf-surya", url, a_cents, 1e10, CAPS[provider],
                                   RATES[provider], 1000.0, arm_id=arm_id, extra={"tail_s": 30})


class TestAnUncertainClose(Ledger):
    def test_the_provider_stays_at_its_cap_until_the_invoice(self):
        row = self.live_arm()
        with self.ledger.locked() as locked:
            serverspend.disarm(locked, "runpod", row["arm_id"], 1000.0 + 13000, False, orphan=True,
                               extra={"uncertain": "unobserved_interval", "drift_s": 10800.0})
        entries = spend.read(self.ledger.path)
        s = serverspend.summary(entries, "runpod", self.ledger.path)
        self.assertEqual((s["total_cents"], s["orphans"]), ("200.00", [row["arm_id"]]))   # the cap, never over it
        self.assertEqual(serverspend.orphan_windows(entries, "runpod", self.ledger.path), [row["arm_id"]])
        with self.ledger.locked() as locked:
            with self.assertRaises(BenchError) as caught:
                serverspend.arm(locked, self.ledger.path, "runpod", "mdf-surya", row["url"], "10", 1e10, "200",
                                RATES["runpod"], 2e9)
            self.assertEqual(caught.exception.code, Code.SESSION_ORPHANED)
        with self.assertRaises(BenchError) as caught:
            serverspend.session(self.ledger.path, "runpod", row["url"], {serverspend.SESSION_ENV: row["arm_id"]})
        self.assertEqual(caught.exception.code, Code.SESSION_ORPHANED)
        with self.ledger.locked() as locked:
            serverspend.reconcile(locked, self.ledger.path, "runpod", "12.40", manual=True, arms=[row["arm_id"]],
                                  orphan=True)
        s = serverspend.summary(spend.read(self.ledger.path), "runpod", self.ledger.path)
        self.assertEqual((s["total_cents"], s["orphans"]), ("12.40", []))      # the figure of the invoice
        with self.ledger.locked() as locked:
            serverspend.arm(locked, self.ledger.path, "runpod", "mdf-surya", row["url"], "10", 1e10, "200",
                            RATES["runpod"], 2e9)                         # released by the reconciliation


class TestTheUrlOfAReservation(Ledger):
    def info(self, **kw):
        return dict({"reserve_cents": "10.00", "rate": RATES["runpod"], "tail_idle_s": 90, "identity": "x",
                     "endpoint": "mdf-surya", "url": "https://ep1.api.runpod.ai", "D": 100}, **kw)

    def test_only_the_window_of_its_endpoint(self):
        row = self.live_arm()
        for info in (self.info(url="https://ep2.api.runpod.ai"), self.info(url=None),
                     {k: v for k, v in self.info().items() if k != "url"}):
            with self.subTest(url=info.get("url")):
                with self.ledger.locked() as locked:
                    with self.assertRaises(BenchError) as caught:
                        serverspend.reserve(locked, self.ledger.path, "runpod", row["arm_id"], "r", "a", info, 1000)
                self.assertEqual(caught.exception.code, Code.SERVER_NOT_ARMED)
        self.assertEqual([e["kind"] for e in spend.read(self.ledger.path)], ["arm"])
        with self.ledger.locked() as locked:
            got = serverspend.reserve(locked, self.ledger.path, "runpod", row["arm_id"], "r", "a", self.info(), 1000)
        self.assertEqual(got["kind"], "reserve")

    def test_the_observer_of_the_engine(self):
        """`Reservation`, the observer of `mdf convert`, of the bench and of the worker: the event of an engine that
        names another URL stops the attempt with `server_not_armed`, an engine code, before any request."""
        row = self.live_arm()
        hook = serverspend.Reservation(self.ledger, "runpod", row["arm_id"], "run", "att", Decimal(1000))
        with self.assertRaises(BenchError) as caught:
            hook({"event": "step", "step": "server-wake", "reserve": self.info(url="https://ep9.api.runpod.ai")})
        self.assertEqual(caught.exception.code, Code.SERVER_NOT_ARMED)
        self.assertIsNone(hook({"event": "step", "step": "server-wake", "reserve": self.info()}))
        self.assertEqual((hook.row["kind"], hook.row["arm_id"]), ("reserve", row["arm_id"]))


if __name__ == "__main__":
    unittest.main()
