"""`ca_file` of a `user` endpoint (MFL051 B; J.7.4 of the plan): the remote server of the user is pinned by the
authorities of a PEM file, on BOTH clients of the relay (the requests of the child, `relay.py` `make` with its
transport, and the probes of the readiness), with the name of the certificate checked and no redirect followed.

TLS is synthetic and real: `ca_file_driver.py`, a subprocess (the guard of `conftest.py` forbids the sockets here),
makes two authorities and the certificates of `localhost` and `other.test`, serves them on 127.0.0.1 and runs the
relay of the engine against them. The field itself is checked here, without a network."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from markdown_for_llms.engine import servers
from markdown_for_llms.engine.codes import EngineError

DRIVER = Path(__file__).resolve().parent / "ca_file_driver.py"


class TlsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        work = Path(tempfile.mkdtemp(prefix="cf-"))           # short: the relay binds a Unix socket in it
        try:
            r = subprocess.run([sys.executable, "-I", "-B", str(DRIVER), str(work)], capture_output=True, text=True,
                               timeout=180)
        finally:
            shutil.rmtree(work, True)
        if r.returncode != 0:
            raise AssertionError(r.stderr[-3000:])
        out = json.loads(r.stdout.strip().splitlines()[-1])
        cls.results = {c["case"]: c for c in out["results"]}
        cls.seen = out["seen"]

    def test_the_right_authority_on_both_clients(self):
        c = self.results["right"]
        self.assertTrue(c["ready"])                                           # the client of the probes
        self.assertEqual(c["chat_status"], 200)                               # the client of the requests
        self.assertEqual(c["stats"]["transport_errors"], 0)

    def test_another_authority_the_system_ones_or_another_name_on_both_clients(self):
        for name in ("wrong", "system", "name"):
            with self.subTest(case=name):
                c = self.results[name]
                self.assertEqual((c["ready"], c["ready_code"]), (False, "server_unavailable"))
                self.assertEqual((c["chat_status"], c["chat_body"]), (502, {"error": "transport"}))
                self.assertEqual(c["stats"]["transport_errors"], 1)
        self.assertEqual(self.seen["named"], [])                              # nothing past the handshake

    def test_a_redirect_is_never_followed(self):
        c = self.results["redirect"]
        self.assertTrue(c["ready"])
        self.assertEqual((c["chat_status"], c["chat_body"]), (502, {"error": "redirect not followed"}))
        self.assertEqual(c["stats"]["redirects"], 1)
        self.assertNotIn(["GET", "/elsewhere"], self.seen["moved"])

    def test_a_file_that_cannot_be_read_or_holds_no_authority(self):
        for name in ("missing", "empty"):
            with self.subTest(case=name):
                self.assertEqual((self.results[name]["built"], self.results[name]["code"]),
                                 (False, "server_not_configured"))


class FieldTests(unittest.TestCase):
    def refused(self, raw):
        with self.assertRaises(EngineError) as ctx:
            servers.endpoint("e", raw)
        self.assertEqual(ctx.exception.code.value, "server_not_configured")

    def test_only_an_absolute_path_on_an_https_user_endpoint(self):
        ep = servers.endpoint("e", {"kind": "user", "url": "https://gpu.lan:23467", "ca_file": "/state/config/ca.pem"})
        self.assertEqual(ep.ca_file, "/state/config/ca.pem")
        for raw in ({"kind": "user", "url": "http://127.0.0.1:8000", "ca_file": "/a.pem"},
                    {"kind": "user", "url": "https://gpu.lan", "ca_file": "ca.pem"},
                    {"kind": "user", "url": "https://gpu.lan", "ca_file": "/state/../etc/ca.pem"},
                    {"kind": "user", "url": "https://gpu.lan", "ca_file": "/a\n.pem"},
                    {"kind": "user", "url": "https://gpu.lan", "ca_file": 1},
                    {"kind": "runpod", "url": "https://ep1.api.runpod.ai", "ca_file": "/a.pem"},
                    {"kind": "local", "url": "http://127.0.0.1:8000", "ca_file": "/a.pem"}):
            with self.subTest(raw=raw):
                self.refused(raw)

    def test_the_record_says_whether_never_the_path(self):
        ep = servers.endpoint("e", {"kind": "user", "url": "https://gpu.lan", "ca_file": "/state/config/ca.pem"})
        record = ep.record()
        self.assertIs(record["ca_file"], True)
        self.assertNotIn("/state/config", json.dumps(record))
        self.assertNotIn("ca_file", servers.endpoint("f", {"kind": "user", "url": "https://gpu.lan"}).record())


if __name__ == "__main__":
    unittest.main()
