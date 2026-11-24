"""The absolute limit of the restore of FlashBoot by `session.py disarm` (MFL023 B10, SOL-B-B2 of the review of the diff B;
MFL049 (b) of the plan r3: the alarm of an arm, besides the `_deadline` of each call).

The timeout of HTTPX bounds the wait for each piece of an answer, not the whole answer: RunPod answering the PATCH of the
restore with a valid JSON in small pieces, each well under the read timeout, used to hold the command past its limit.
Here the transport is the real one of HTTPX and httpcore (`HTTPTransport`, its connection pool, `SyncStream` on a
socket); the socket is one end of an AF_UNIX `socketpair` (the guard of the suite forbids `connect`), whose other end a
thread of the test serves as the REST v2 of one endpoint. The clock is the real one, with the limit reduced for the
test. No network."""
import io
import json
from pathlib import Path
import shutil
import socket
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

import httpcore
from httpcore._backends.sync import SyncStream
import httpx

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts" / "servers"))
import providers as pv  # noqa: E402
import session  # noqa: E402

from markdown_for_llms.bench import serverspend, spend  # noqa: E402
from markdown_for_llms.engine import envfile  # noqa: E402

KEY = "rk-b10-restore-0123"
LIMIT_S = 1.0                     # the limit of the restore in this test (90 s in the command)
MARGIN_S = 1.5
PIECE_S = 0.1                     # one byte of the answer of the PATCH every 0.1 s: about 10 s for the whole body


class RunPodOverAPair:
    """The REST v2 of one endpoint disarmed at its zero with FlashBoot on; the PATCH of the restore answered one byte at
    a time. Every connection of the client is a new `socketpair`, served by a thread of its own."""

    def __init__(self):
        self.ep = {"id": "ep1", "name": "mdf-surya", "workers": pv.workers_block(0), "flashboot": "FLASHBOOT"}
        self.calls, self.threads, self.lock = [], [], threading.Lock()

    def answer(self, method, path, body):
        with self.lock:
            self.calls.append((method, path, body))
        if path == "/v2/serverless" and method == "GET":
            return {"endpoints": [self.ep], "pagination": {}}, False
        if path == "/v2/serverless/ep1" and method == "GET":
            return self.ep, False
        if path == "/v2/serverless/ep1" and method == "PATCH":
            slow = "flashboot" in body
            self.ep.update(body)
            return self.ep, slow
        if path == "/v2/serverless/ep1/workers":
            return {"workers": [], "summary": {"total": 0}}, False
        return {}, False

    def serve(self, sock):
        reader = sock.makefile("rb")
        try:
            while True:
                line = reader.readline()
                if not line:
                    return
                method, target, _ = line.decode("ascii").split(" ", 2)
                length = 0
                while True:
                    header = reader.readline()
                    if header in (b"\r\n", b""):
                        break
                    name, _, value = header.decode("ascii").partition(":")
                    if name.strip().lower() == "content-length":
                        length = int(value.strip())
                body = json.loads(reader.read(length)) if length else {}
                doc, slow = self.answer(method, target.split("?", 1)[0], body)
                data = json.dumps(doc).encode()
                sock.sendall(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: %d\r\n\r\n"
                             % len(data))
                if not slow:
                    sock.sendall(data)
                    continue
                for byte in data:
                    time.sleep(PIECE_S)
                    sock.sendall(bytes([byte]))
        except OSError:
            return                                       # the client closed its end (the limit came)
        finally:
            reader.close()
            sock.close()

    def transport(self):
        fake = self

        class Pairs(httpcore.NetworkBackend):
            def connect_tcp(self, host, port, timeout=None, local_address=None, socket_options=None):
                ours, theirs = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
                t = threading.Thread(target=fake.serve, args=(theirs,), daemon=True)
                fake.threads.append(t)
                t.start()
                return SyncStream(ours)

            def connect_unix_socket(self, path, timeout=None, socket_options=None):
                return self.connect_tcp(None, None)

            def sleep(self, seconds):
                time.sleep(seconds)

        real = httpx.HTTPTransport()
        real._pool = httpcore.ConnectionPool(network_backend=Pairs())

        class Plain(httpx.BaseTransport):
            """The REST URL in plain HTTP towards the pair (no TLS on a local socket); the rest is the real transport."""

            def handle_request(self, request):
                request.url = request.url.copy_with(scheme="http", host="runpod.test", port=None)
                return real.handle_request(request)

            def close(self):
                real.close()
        return Plain()


class TestTheAbsoluteLimitOfTheRestore(unittest.TestCase):
    def setUp(self):
        d = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, d, True)
        self.env = d / ".env"
        self.env.write_text("RUNPOD_API_KEY=%s\n" % KEY)
        self.env.chmod(0o600)
        self.ledger = d / "spend.jsonl"
        now = time.time()
        with spend.Ledger.file(self.ledger).locked() as locked:
            self.row = serverspend.arm(locked, self.ledger, "runpod", "mdf-surya", "https://ep1.api.runpod.ai", "30",
                                       now - 50, "200", "0.0002090948", now - 1000)

    def test_an_answer_in_small_pieces_ends_at_the_limit(self):
        fake = RunPodOverAPair()
        a = session.parser().parse_args(["disarm", "--provider", "runpod", "--name", "mdf-surya", "--env-file",
                                         str(self.env), "--spend-ledger", str(self.ledger)])
        prov = pv.RunPod(envfile.Secret(KEY), "mdf-surya", transport=fake.transport())
        self.addCleanup(prov.close)
        out = io.StringIO()
        with mock.patch.object(session, "RESTORE_LIMIT_S", LIMIT_S):
            start = time.monotonic()
            code = session.disarm(a, prov, out=out)
            elapsed = time.monotonic() - start
        report = json.loads(out.getvalue().splitlines()[-1])
        restores = [c for c in fake.calls if c[0] == "PATCH" and "flashboot" in c[2]]
        self.assertEqual(len(restores), 1)                                  # the restore was asked, its answer slow
        self.assertLess(elapsed, LIMIT_S + MARGIN_S)
        self.assertEqual((code, report["zero"], report["flashboot_restored"]), (4, True, False))
        self.assertEqual(report["orphan_closed"], [self.row["arm_id"]])
        dis, = [r for r in spend.read(self.ledger) if r.get("kind") == "disarm"]
        self.assertEqual((dis["arm_id"], dis["orphan"], dis["clean"], dis["flashboot_restored"]),
                         (self.row["arm_id"], True, False, False))
        self.assertNotIn(KEY, out.getvalue())
        patch_at = fake.calls.index(restores[0])
        self.assertEqual(fake.calls[patch_at + 1:], [])                     # no reading after the limit


if __name__ == "__main__":
    unittest.main()
