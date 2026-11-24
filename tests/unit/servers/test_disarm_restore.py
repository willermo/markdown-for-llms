"""The restore of FlashBoot by `session.py disarm`, the recovery of a session that died (MFL023 B7; MFL049 (a) and (b),
C8-R1 and C8-R2 of the review of MFL045).

(a) a reading of RunPod without the field `flashboot` asks for no restore (only a lever of the window or a reading
explicitly not OFF does); (b) the PATCH of the restore and its reading back run outside the lock of the ledger, under an
absolute limit for all their calls (`RESTORE_LIMIT_S`), then the windows are read again under the lock. No network:
`httpx.MockTransport`, a fake clock."""
import fcntl
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

import httpx

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts" / "servers"))
import providers as pv  # noqa: E402
import session  # noqa: E402

from markdown_for_llms.bench import serverspend, spend  # noqa: E402
from markdown_for_llms.engine import envfile  # noqa: E402

KEY = "rk-b7-restore-0123"


class Clock:
    def __init__(self):
        self.t = 1_800_000_000.0

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.t += s


class Endpoint:
    """The REST v2 of one endpoint disarmed at its zero: `flashboot` as given (or absent), a PATCH of the restore that
    takes `patch_s` of the fake clock; the lock of the ledger tried during each PATCH."""

    def __init__(self, clock, flashboot="absent", patch_s=0.0, ledger=None):
        self.clock, self.patch_s, self.ledger = clock, patch_s, ledger
        self.ep = {"id": "ep1", "name": "mdf-surya", "workers": pv.workers_block(0)}
        if flashboot != "absent":
            self.ep["flashboot"] = flashboot
        self.calls, self.lock_free = [], []

    def __call__(self, request):
        path, method = request.url.path.split("/v2", 1)[1], request.method
        self.calls.append((method, path, json.loads(request.content) if request.content else None))
        if path == "/serverless" and method == "GET":
            return httpx.Response(200, json={"endpoints": [self.ep], "pagination": {}})
        if path == "/serverless/ep1" and method == "GET":
            return httpx.Response(200, json=self.ep)
        if path == "/serverless/ep1" and method == "PATCH":
            body = json.loads(request.content)
            if "flashboot" in body and self.ledger is not None:
                fd = os.open(self.ledger, os.O_RDWR)
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    self.lock_free.append(True)
                    fcntl.flock(fd, fcntl.LOCK_UN)
                except BlockingIOError:
                    self.lock_free.append(False)
                finally:
                    os.close(fd)
                self.clock.t += self.patch_s
            self.ep.update(body)
            return httpx.Response(200, json=self.ep)
        if path == "/serverless/ep1/workers":
            return httpx.Response(200, json={"workers": [], "summary": {"total": 0}})
        return httpx.Response(404, json={})


class TestTheRestoreOfTheRecovery(unittest.TestCase):
    def setUp(self):
        d = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, d, True)
        self.env = d / ".env"
        self.env.write_text("RUNPOD_API_KEY=%s\n" % KEY)
        self.env.chmod(0o600)
        self.ledger = d / "spend.jsonl"
        with spend.Ledger.file(self.ledger).locked() as locked:
            self.row = serverspend.arm(locked, self.ledger, "runpod", "mdf-surya", "https://ep1.api.runpod.ai", "30",
                                       1.8e9 + 1000, "200", "0.0002090948", 1.8e9 - 100)

    def disarm(self, fake, clock):
        a = session.parser().parse_args(["disarm", "--provider", "runpod", "--name", "mdf-surya", "--env-file",
                                         str(self.env), "--spend-ledger", str(self.ledger)])
        prov = pv.RunPod(envfile.Secret(KEY), "mdf-surya", transport=httpx.MockTransport(fake), clock=clock)
        out = io.StringIO()
        code = session.disarm(a, prov, clock=clock, sleep=clock.sleep, out=out)
        return code, json.loads(out.getvalue().splitlines()[-1])

    def patches(self, fake):
        return [c for c in fake.calls if c[0] == "PATCH" and "flashboot" in (c[2] or {})]

    def test_a_reading_without_flashboot_asks_for_no_restore(self):
        clock = Clock()
        fake = Endpoint(clock)
        code, report = self.disarm(fake, clock)
        self.assertEqual(code, 0)
        self.assertEqual(self.patches(fake), [])
        self.assertNotIn("flashboot_restored", report)
        dis, = [r for r in spend.read(self.ledger) if r.get("kind") == "disarm"]
        self.assertTrue(dis["orphan"])

    def test_a_reading_explicitly_not_off_restores_outside_the_lock(self):
        clock = Clock()
        fake = Endpoint(clock, flashboot="FLASHBOOT", ledger=self.ledger)
        code, report = self.disarm(fake, clock)
        self.assertEqual((code, report["flashboot_restored"]), (0, True))
        self.assertEqual(fake.lock_free, [True])                   # the ledger was not locked during the PATCH
        dis, = [r for r in spend.read(self.ledger) if r.get("kind") == "disarm"]
        self.assertTrue(dis["flashboot_restored"])

    def test_an_endpoint_at_off_asks_for_nothing(self):
        clock = Clock()
        fake = Endpoint(clock, flashboot="OFF")
        code, report = self.disarm(fake, clock)
        self.assertEqual((code, self.patches(fake)), (0, []))

    def test_a_slow_restore_ends_at_its_limit(self):
        """The PATCH takes 100 s of the clock: its reading back comes after the limit of 90 s and is not made; the rows
        are written anyway, and the exit is 4."""
        clock = Clock()
        fake = Endpoint(clock, flashboot="FLASHBOOT", patch_s=100, ledger=self.ledger)
        start = clock()
        code, report = self.disarm(fake, clock)
        self.assertEqual((code, report["flashboot_restored"]), (4, False))
        self.assertLessEqual(clock() - start, 200)
        self.assertIn("term of the session", report["flashboot_error"])
        gets = [c for c in fake.calls if c[0] == "GET" and c[1] == "/serverless/ep1"]
        patch_at = fake.calls.index(self.patches(fake)[0])
        self.assertFalse([c for c in fake.calls[patch_at + 1:] if c[0] == "GET"])       # no reading after the limit
        self.assertTrue(gets)
        dis, = [r for r in spend.read(self.ledger) if r.get("kind") == "disarm"]
        self.assertEqual((dis["orphan"], dis["flashboot_restored"]), (True, False))


if __name__ == "__main__":
    unittest.main()
