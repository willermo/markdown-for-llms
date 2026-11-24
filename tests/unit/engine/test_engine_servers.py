"""Endpoints, keys, terms and bounds of the `marker-server` route (MFL033: plan r2 §5.1, §11; r3 §11;
r4 §11.2, §11.4, App. G; prescription 4 of the review of r4)."""
from decimal import Decimal
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest

from markdown_for_llms.engine import servers
from markdown_for_llms.engine.codes import Code, EngineError

HOUR = Decimal(3600)
RUNPOD_RATE = Decimal("0.75") / HOUR + Decimal(20) * Decimal("0.10") / (Decimal("30.4") * 86400)
MODAL_RATE = Decimal("0.000222") + 2 * Decimal("0.0000131") + 16 * Decimal("0.00000222")


def ep(name="e", **raw):
    return servers.endpoint(name, raw)


def runpod(**extra):
    return ep(kind="runpod", url="https://abc.api.runpod.ai", rate_usd_per_s=str(RUNPOD_RATE), **extra)


def modal(**extra):
    return ep(kind="modal", url="https://ws--mdf-surya-serve.modal.run", rate_usd_per_s=str(MODAL_RATE), **extra)


class Tmp(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.dir, True)

    def write(self, name, text, mode=0o600):
        path = self.dir / name
        path.write_text(text)
        os.chmod(path, mode)
        return path


class TestConfiguration(Tmp):
    def test_default_and_override(self):
        path = self.write("servers.json", json.dumps({"default": "fake", "servers": {
            "fake": {"kind": "local", "url": "http://127.0.0.1:23463"},
            "rp": {"kind": "runpod", "url": "https://x.api.runpod.ai", "rate_usd_per_s": "0.0002"}}}))
        self.assertEqual(servers.load(path).name, "fake")
        rp = servers.load(path, "rp")
        self.assertEqual((rp.kind, rp.provider, rp.tail_fault_s, rp.ready_within, rp.read_timeout), ("runpod", "runpod", 385, 900, 300))
        self.assertEqual(rp.key_env, ["RUNPOD_API_KEY"])
        md = modal()
        self.assertEqual((md.tail_fault_s, md.ready_within, md.read_timeout, md.cold_start_s), (225, 900, 140, 480))
        for bad in ("nope", None):
            with self.assertRaises(EngineError) as caught:
                servers.load(path, bad) if bad else servers.load(self.dir / "absent.json")
            self.assertEqual(caught.exception.code, Code.SERVER_NOT_CONFIGURED)
        self.assertEqual(servers.default_path({"XDG_CONFIG_HOME": "/c"}), Path("/c/markdown-for-llms/servers.json"))

    def test_refused_entries(self):
        cases = [
            ({"kind": "local", "url": "http://localhost:23463"}, Code.SERVER_NOT_LOOPBACK),
            ({"kind": "local", "url": "https://127.0.0.1:23463"}, Code.SERVER_NOT_LOOPBACK),
            ({"kind": "local", "url": "http://127.0.0.1"}, Code.SERVER_NOT_LOOPBACK),
            ({"kind": "user", "url": "http://gpu.example:8000"}, Code.SERVER_NOT_CONFIGURED),
            ({"kind": "user", "url": "https://u:p@gpu.example"}, Code.SERVER_NOT_CONFIGURED),
            ({"kind": "user", "url": "https://gpu.example/?k=v"}, Code.SERVER_NOT_CONFIGURED),
            ({"kind": "user", "url": "https://gpu.example/#f"}, Code.SERVER_NOT_CONFIGURED),
            ({"kind": "user", "url": "https://gpu.example", "api_key": "x"}, Code.SERVER_NOT_CONFIGURED),
            ({"kind": "runpod", "url": "https://x.api.runpod.ai"}, Code.SERVER_NOT_CONFIGURED),
            ({"kind": "runpod", "url": "https://x", "rate_usd_per_s": "0.0002", "provider": "modal"},
             Code.SERVER_NOT_CONFIGURED),
            ({"kind": "modal", "url": "https://x", "rate_usd_per_s": "0.0002", "ready_within": 600},
             Code.SERVER_NOT_CONFIGURED),
            ({"kind": "cpu", "url": "http://127.0.0.1:1"}, Code.SERVER_NOT_CONFIGURED),
            ({"kind": "local", "url": "http://127.0.0.1:1", "provider": "runpod"}, Code.SERVER_NOT_CONFIGURED),
            ({"kind": "user", "url": "https://x", "rate_usd_per_s": "0.1"}, Code.SERVER_NOT_CONFIGURED),
            ({"kind": "user", "url": "https://x", "parallel": 0}, Code.SERVER_NOT_CONFIGURED),
            ({"kind": "user", "url": "https://x", "key_env": "lower"}, Code.SERVER_NOT_CONFIGURED),
            ({"kind": "user", "url": "https://x", "key_env": "MDF_KEY"}, Code.SERVER_NOT_CONFIGURED),
            ({"kind": "modal", "url": "https://x", "rate_usd_per_s": "0.0002", "key_env": "ONE"},
             Code.SERVER_NOT_CONFIGURED),
        ]
        for raw, code in cases:
            with self.subTest(raw=raw):
                with self.assertRaises(EngineError) as caught:
                    servers.endpoint("e", raw)
                self.assertEqual(caught.exception.code, code)

    def test_identity_and_transit(self):
        a = ep(kind="user", url="https://GPU.example:8443/")
        b = ep(kind="user", url="https://gpu.example:8443")
        c = ep(kind="user", url="https://other.example:8443")
        self.assertEqual(a.identity_sha256(), b.identity_sha256())
        self.assertNotEqual(a.identity_sha256(), c.identity_sha256())
        self.assertNotEqual(a.identity_sha256(), ep(kind="user", url="https://gpu.example:8443", parallel=8).identity_sha256())
        self.assertTrue(a.leaves_machine())
        self.assertFalse(ep(kind="local", url="http://[::1]:23463").leaves_machine())
        self.assertFalse(ep(kind="cpu").leaves_machine())
        self.assertTrue(ep(kind="user", url="http://127.0.0.2:9").leaves_machine() is False)
        self.assertEqual(ep(kind="cpu").requests(), {"allow_stop_token_ids": False, "allow_guided_json": False})


class TestKeys(Tmp):
    def test_env_file_environment_and_headers(self):
        env = self.write(".env", "# keys\nexport RUNPOD_API_KEY='rp-secret'\nMODAL_PROXY_TOKEN_ID=wk-1\n"
                                 "MODAL_PROXY_TOKEN_SECRET=ws-2\nOTHER=x\n")
        rp, md = runpod(), modal()
        keys = servers.read_keys(rp, env)
        self.assertNotIn("rp-secret", repr(keys))
        self.assertEqual(servers.headers(rp, keys), {"Authorization": "Bearer rp-secret"})
        self.assertEqual(servers.headers(md, servers.read_keys(md, env)), {"Modal-Key": "wk-1", "Modal-Secret": "ws-2"})
        self.assertEqual(servers.read_keys(ep(kind="cpu"), env), {})
        user = ep(kind="user", url="https://gpu.example", key_env="MY_KEY")
        with self.assertRaises(EngineError) as caught:
            servers.read_keys(user, env)
        self.assertEqual(caught.exception.code, Code.SERVER_KEY_MISSING)
        self.assertEqual(servers.headers(user, servers.read_keys(user, env, {"MY_KEY": "k"})), {"Authorization": "Bearer k"})
        with self.assertRaises(EngineError) as caught:
            servers.read_keys(user, None)
        self.assertEqual(caught.exception.code, Code.SERVER_KEY_MISSING)

    def test_unsafe_env_file(self):
        loose = self.write(".env", "RUNPOD_API_KEY=x\n", mode=0o644)
        link = self.dir / "link.env"
        link.symlink_to(self.write("real.env", "RUNPOD_API_KEY=x\n"))
        for path in (loose, link, self.dir):
            with self.assertRaises(EngineError) as caught:
                servers.read_keys(runpod(), path)
            self.assertEqual(caught.exception.code, Code.SERVER_KEY_UNSAFE)


class TestTerm(unittest.TestCase):
    """r4 App. G: the floor before the wake and the truncations of prescription 4."""

    def test_d18_on_runpod_at_35_cents(self):
        t = servers.term(runpod(), 1, budget_cents=35, until=10_000, now=0)
        self.assertEqual((t["D"], t["reserve_cents"]), (1230, "33.77"))
        with self.assertRaises(EngineError) as caught:           # with the tail of r3
            servers.term(runpod(tail_fault_s=1045), 1, budget_cents=35, until=10_000, now=0)
        self.assertEqual(caught.exception.code, Code.SERVER_CAP)

    def test_near_the_term_of_the_session(self):
        with self.assertRaises(EngineError) as caught:
            servers.term(runpod(), 1, budget_cents=35, until=959, now=0)
        self.assertEqual(caught.exception.code, Code.SERVER_NOT_ARMED)
        self.assertEqual(servers.term(runpod(), 1, budget_cents=35, until=960, now=0)["D"], 900)
        with self.assertRaises(EngineError) as caught:
            servers.term(modal(), 1, budget_cents=60, until=959, now=0)
        self.assertEqual(caught.exception.code, Code.SERVER_NOT_ARMED)
        self.assertEqual(servers.term(modal(), 1, budget_cents=60, until=960, now=0)["D"], 900)
        with self.assertRaises(EngineError) as caught:
            servers.term(runpod(), 1, budget_cents=35)
        self.assertEqual(caught.exception.code, Code.SERVER_NOT_ARMED)

    def test_the_cap_is_checked_before_the_session(self):
        with self.assertRaises(EngineError) as caught:
            servers.term(runpod(), 1, budget_cents=10)
        self.assertEqual(caught.exception.code, Code.SERVER_CAP)

    def test_no_reservation_over_the_budget(self):
        """No reserve exceeds the budget, not even by 0.01 cents; D is never under the floor."""
        for endpoint in (runpod(), modal()):
            for pages in (1, 3, 12, 50, 167):
                for cents in range(20, 400, 7):
                    budget = Decimal(cents) + Decimal("0.37")
                    for left in (Decimal(900), Decimal("1261.5"), Decimal(5000), Decimal(50000)):
                        try:
                            t = servers.term(endpoint, pages, budget_cents=budget, until=left, now=0)
                        except EngineError as exc:
                            self.assertIn(exc.code, (Code.SERVER_CAP, Code.SERVER_NOT_ARMED))
                            continue
                        self.assertLessEqual(Decimal(t["reserve_cents"]), budget)
                        self.assertGreaterEqual(t["D"], Decimal(t["d_min"]))
                        self.assertLessEqual(t["D"], left - 60)

    def test_without_a_provider(self):
        t = servers.term(ep(kind="local", url="http://127.0.0.1:9"), 4)
        self.assertEqual((t["D"], t["reserve_cents"]), (300 + 4 * 30 + 300, None))
        self.assertIsNone(servers.bound_cents(ep(kind="cpu"), True, True, 0, 1, 2))


class TestBound(unittest.TestCase):
    def test_cases_within_the_reservation(self):
        e = runpod()
        t = servers.term(e, 3, budget_cents=200, until=100_000, now=0)
        reserve, d = Decimal(t["reserve_cents"]), t["D"]
        close = d - 5
        clean = Decimal(servers.bound_cents(e, True, True, 0, 300, close))
        unclean = Decimal(servers.bound_cents(e, True, False, 0, 300, close))
        unready = Decimal(servers.bound_cents(e, False, False, 0, None, 10))
        unready_late = Decimal(servers.bound_cents(e, False, False, 0, None, close))
        self.assertEqual(clean, ((300 + 90) * e.rate * 100).quantize(Decimal("0.01"), rounding="ROUND_CEILING"))
        self.assertEqual(unready, ((900 + 385) * e.rate * 100).quantize(Decimal("0.01"), rounding="ROUND_CEILING"))
        for value in (clean, unclean, unready, unready_late):
            self.assertLessEqual(value, reserve)


if __name__ == "__main__":
    unittest.main()
