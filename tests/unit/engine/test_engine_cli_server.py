"""`mdf convert --route marker-server` (MFL033: plan r2 §8, r4 §8, prescription 2): refusals before any
conversion, and two files on a rented endpoint inside a live session, with the reservations and the
completions in the ledger. The fake server is an `httpx.MockTransport` given to the engine by a
wrapper of `core.convert` (the suite forbids connections in its process)."""
import contextlib
from decimal import Decimal
import fcntl
import io
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest import mock

from engine_test_support import SCAN, make_pdf, scan_of
from fake_inference import Fake
from markdown_for_llms import cli
from markdown_for_llms.bench import serverspend, spend
from markdown_for_llms.engine import core

KEY = "rp-cli-test-key-0123456789"
URL = "http://127.0.0.1:9"
RATE = "0.0002090948"


def respond(request):
    from surya.inference.prompts import PROMPT_MAPPING
    if request["messages"][0]["content"][1]["text"] == PROMPT_MAPPING["high_accuracy_bbox"]:
        return '<div data-bbox="60 60 940 160" data-label="Text"><p>Scanned for the CLI</p></div>'
    return "<p>Scanned for the CLI</p>"


class CliServer(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.servers = self.root / "servers.json"
        self.servers.write_text(json.dumps({"default": "local", "servers": {
            "local": {"kind": "local", "url": URL},
            "gpu": {"kind": "user", "url": "https://gpu.example"},
            "rp": {"kind": "runpod", "url": URL, "rate_usd_per_s": RATE}}}))
        self.env = self.root / ".env"
        self.env.write_text("RUNPOD_API_KEY=%s\n" % KEY)
        os.chmod(self.env, 0o600)
        self.ledger = self.root / "runs" / "spend.jsonl"
        self.ledger.parent.mkdir()
        self.src = self.root / "scan.pdf"
        self.src.write_bytes(make_pdf([{"content": SCAN, "image": scan_of(["Scanned for the CLI"], scale=1.5)}]))
        model_root = os.environ.get("FULL_CHECK_MODEL_ROOT")
        if not model_root:
            raise AssertionError("FULL_CHECK_MODEL_ROOT is not set: run the suite from scripts/full-check.sh")
        self.base = ["--env-file", str(self.env), "--data-root", str(self.root / "data"), "--model-root", model_root]

    def run_cli(self, argv, environ=None):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err), \
                mock.patch.dict(os.environ, environ or {}, clear=False):
            os.environ.pop(serverspend.SESSION_ENV, None) if not environ or serverspend.SESSION_ENV not in environ else None
            try:
                code = cli.main(self.base + argv)
            except SystemExit as exc:
                code = exc.code
        return code, out.getvalue()

    def convert(self, *extra, environ=None, files=None):
        files = files or [str(self.src)]
        return self.run_cli(["convert", *files, "--out", str(self.root / "out"), "--route", "marker-server",
                             "--servers", str(self.servers), *extra], environ)

    def arm(self, a_usd="1.00", live=True):
        with spend.Ledger.file(self.ledger).locked() as locked:
            row = serverspend.arm(locked, self.ledger, "runpod", "rp", URL, Decimal(a_usd) * 100, time.time() + 7200,
                                  400, RATE, time.time())
        if live:
            fd = os.open(serverspend.lock_path(self.ledger, row["arm_id"]), os.O_RDWR | os.O_CREAT | os.O_CLOEXEC, 0o600)
            fcntl.flock(fd, fcntl.LOCK_EX)
            self.addCleanup(os.close, fd)
        return row["arm_id"]


class TestRefusals(CliServer):
    def test_options_and_configuration(self):
        cases = [
            (["--server-mode", "balanced", "--datalab-mode", "fast"], 2, "route_option_unsupported"),
            ([], 2, "route_option_unsupported"),
            (["--server-mode", "fast"], 2, None),
            (["--server-mode", "balanced", "--server", "nope"], 1, "server_not_configured"),
            (["--server-mode", "balanced", "--server-cap-usd", "1"], 2, "route_option_unsupported"),
            (["--server-mode", "balanced", "--server", "rp"], 2, "server_ledger_required"),
        ]
        for extra, status, code in cases:
            with self.subTest(extra=extra):
                got, out = self.convert(*extra)
                self.assertEqual(got, status)
                if code:
                    self.assertEqual(json.loads(out)["error"], code)
        got, out = self.run_cli(["convert", str(self.src), "--out", str(self.root / "o2"), "--server-mode", "balanced"])
        self.assertEqual((got, json.loads(out)["error"]), (2, "route_option_unsupported"))

    def test_a_rented_endpoint_needs_a_live_session_and_an_orphan_comes_first(self):
        rented = ["--server-mode", "balanced", "--server", "rp", "--server-cap-usd", "0.60", "--spend-ledger",
                  str(self.ledger), "--spend-cap-usd", "4"]
        got, out = self.convert(*rented)
        self.assertEqual((got, json.loads(out)["error"]), (1, "server_not_armed"))
        self.assertIn("session.py run", json.loads(out)["detail"])
        orphan = self.arm(live=False)
        got, out = self.convert(*rented, environ={serverspend.SESSION_ENV: orphan})
        self.assertEqual((got, json.loads(out)["error"]), (1, "session_orphaned"))
        self.assertFalse((self.root / "out").exists())

    def test_a_rate_of_servers_json_that_is_not_the_rate_of_the_arm(self):
        """S-B6 of the final review of MFL033: refused before anything reaches the server."""
        arm_id = self.arm("1.00")
        doc = json.loads(self.servers.read_text())
        doc["servers"]["rp"]["rate_usd_per_s"] = "0.0001"
        self.servers.write_text(json.dumps(doc))
        fake = Fake(respond=respond, mark="forwarded", key=KEY)
        self.addCleanup(fake.close)
        real = core.convert

        def convert(*args, **kwargs):
            return real(*args, server_transport=fake.factory(), **kwargs)
        with mock.patch.object(core, "convert", convert):
            got, out = self.convert("--server-mode", "balanced", "--server", "rp", "--server-cap-usd", "0.60",
                                    "--spend-ledger", str(self.ledger), "--spend-cap-usd", "4",
                                    environ={serverspend.SESSION_ENV: arm_id})
        summary = json.loads(out)
        self.assertEqual((got, summary["error"]), (1, "server_not_configured"))
        self.assertIn("rate_usd_per_s", summary["detail"])
        self.assertEqual(fake.seen, [])
        self.assertEqual([e["kind"] for e in spend.read(self.ledger) if e.get("provider") == "runpod"], ["arm"])
        self.assertNotIn(KEY, out)

    def test_transit_of_a_server_outside_the_loopback(self):
        got, out = self.convert("--server-mode", "balanced", "--server", "gpu")
        summary = json.loads(out)
        self.assertEqual((got, summary["error"], summary["transit"]["left_machine"]), (1, "transit_not_confirmed", False))


class TestRentedRun(CliServer):
    def test_two_files_inside_a_session_reserve_and_complete(self):
        arm_id = self.arm("1.00")
        fake = Fake(respond=respond, mark="forwarded", key=KEY)
        self.addCleanup(fake.close)
        real = core.convert

        def convert(*args, **kwargs):
            return real(*args, server_transport=fake.factory(), **kwargs)
        second = self.root / "scan2.pdf"
        second.write_bytes(self.src.read_bytes())
        with mock.patch.object(core, "convert", convert):
            got, out = self.convert("--server-mode", "balanced", "--server", "rp", "--server-cap-usd", "0.60",
                                    "--spend-ledger", str(self.ledger), "--spend-cap-usd", "4",
                                    environ={serverspend.SESSION_ENV: arm_id}, files=[str(self.src), str(second)])
        self.assertEqual(got, 0, out)
        summaries = json.loads(out)
        self.assertEqual([s["status"] for s in summaries], ["complete", "complete"])
        self.assertEqual(sorted(p.name for p in (self.root / "out").iterdir()), ["001-scan", "002-scan2"])
        rows = [e for e in spend.read(self.ledger) if e.get("provider") == "runpod"]
        self.assertEqual([r["kind"] for r in rows], ["arm", "reserve", "complete", "reserve", "complete"])
        reserves = [r for r in rows if r["kind"] == "reserve"]
        self.assertEqual({r["arm_id"] for r in reserves}, {arm_id})
        self.assertEqual(reserves[1]["follows"]["attempt_id"], reserves[0]["attempt_id"])
        completes = [r for r in rows if r["kind"] == "complete"]
        self.assertTrue(all(c["clean"] and Decimal(c["bound_cents"]) <= Decimal(r["reserve_cents"])
                            for c, r in zip(completes, reserves)))
        text = self.ledger.read_text() + out + "".join(p.read_text(errors="replace")
                                                        for p in (self.root / "out").rglob("*") if p.is_file())
        self.assertNotIn(KEY, text)
        self.assertEqual(set(fake.auth) - {None}, {"Bearer " + KEY})


if __name__ == "__main__":
    unittest.main()
