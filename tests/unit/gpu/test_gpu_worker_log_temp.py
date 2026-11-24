"""The temporary worker log of a GPU session left by a dead worker (MFL023 B10, SOL-B-R1 of the review of the diff B, in
part; B12, SOL-B11-B1 of the review delta of B11).

The follower of the worker logs writes into a private folder of the session under the ledger
(`<ledger dir>/worker-logs/<arm_id>/partial/`, 0700, no link), never into the temporary directory of the system. A
SIGKILL of the worker skips the scan and the save of `keep_log`: at the next start the recovery of that session removes
the folder unread and keeps nothing of the partial log, whatever the env files say (both readable, rotated in place,
gone, no longer private, unreadable): the keys the session used cannot be attested after its death. The diary of the
session records `partial_discarded`, with no line of the log and no key; the recovery touches the folder of the session
it recovers only."""
from dataclasses import asdict
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import time
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gpu_harness import KEY, Instance, wait_for  # noqa: E402

DRIVER = Path(__file__).resolve().parent / "worker_driver.py"
LINES = ["2026-11-24T09:51:00.000Z\t+1.0\tw1\tcontainer\t2026-11-24 10:51:00+01:00\t{\"event\": \"mdf_ready\"}",
         "2026-11-24T09:51:01.000Z\t+2.0\tw1\tcontainer\t2026-11-24 10:51:01+01:00\tINFO: the proxy is up"]


class TestTheLogOfADeadWorker(unittest.TestCase):
    def killed(self, lines):
        """A worker in a subprocess with a fake follower, killed with SIGKILL once its session is armed; its TMPDIR."""
        inst = Instance(self)
        tmp = inst.root / "tmp"
        tmp.mkdir()
        barrier, where = inst.root / "barrier", inst.root / "follower-path"
        inst.submit()
        cfg = inst.root / "config.json"
        cfg.write_text(json.dumps({"settings": asdict(inst.settings), "env_file": str(inst.env), "state": str(inst.state),
                                   "barrier": "armed_before_reserve", "file": str(barrier), "close": False,
                                   "follower": {"lines": lines, "file": str(where)}}))
        p = subprocess.Popen([sys.executable, "-I", "-B", str(DRIVER), str(cfg)], stdout=subprocess.DEVNULL,
                             stderr=subprocess.PIPE, env=dict(os.environ, TMPDIR=str(tmp)))
        try:
            wait_for(lambda: barrier.exists() or p.poll() is not None, within=60, what="the armed session")
            if p.poll() is not None:
                self.fail("the worker ended first: %s" % p.stderr.read().decode()[-2000:])
            time.sleep(0.1)
            p.kill()
            p.wait(10)
        finally:
            if p.poll() is None:
                p.kill()
                p.wait(10)
            p.stderr.close()
        log = Path(where.read_text())
        session, = inst.sessions()
        return inst, tmp, log, session["arm_id"]

    def assertNoTemporary(self, inst, tmp, arm_id):
        self.assertEqual(sorted(p.name for p in tmp.iterdir()), [])                 # no mdf-wlog-* in TMPDIR
        self.assertFalse((inst.ledger_dir / "worker-logs" / arm_id / "partial").exists())

    def assertDiscarded(self, inst, arm_id, partial, *keys):
        """Nothing of the partial log kept: no `worker-log-partial.txt`, no key and no line of the log under the ledger,
        and a diary event with the outcome only."""
        self.assertFalse((inst.ledger_dir / "worker-logs" / arm_id / "worker-log-partial.txt").exists())
        for path in inst.ledger_dir.rglob("*"):
            if path.is_file():
                text = path.read_text(errors="replace")
                for needle in keys + (LOG_TEXT,):
                    self.assertNotIn(needle, text)
        self.assertEqual(partial, {"event": "worker-log-partial", "t": partial["t"], "partial_discarded": True})

    def diary_event(self, inst, arm_id):
        diary = json.loads((inst.ledger_dir / "worker-logs" / arm_id / "session.json").read_text())
        partial, = [e for e in diary["events"] if e["event"] == "worker-log-partial"]
        return partial

    def test_the_log_lives_under_the_ledger_and_is_discarded(self):
        inst, tmp, log, arm_id = self.killed(LINES)
        folder = inst.ledger_dir / "worker-logs" / arm_id
        self.assertEqual(log.parent, folder / "partial")
        self.assertEqual(sorted(p.name for p in tmp.iterdir()), [])
        for path in (folder.parent, folder, log.parent):
            info = os.lstat(path)
            self.assertTrue(stat.S_ISDIR(info.st_mode))
            self.assertEqual(stat.S_IMODE(info.st_mode), 0o700)
        self.assertIn(LOG_TEXT, log.read_text())                                    # the follower wrote it there
        other = inst.ledger_dir / "worker-logs" / "20261124T000000Z-other000" / "partial"
        other.mkdir(parents=True)
        (other / "worker-log.partial").write_text("a session this start does not recover\n")
        w = inst.worker()
        w.gpu.recover()
        self.assertNoTemporary(inst, tmp, arm_id)
        self.assertEqual((other / "worker-log.partial").read_text(), "a session this start does not recover\n")
        self.assertDiscarded(inst, arm_id, self.diary_event(inst, arm_id), KEY)
        self.assertEqual(inst.sessions()[0]["state"], "orphan")

    def test_a_partial_log_with_a_key_is_dropped(self):
        inst, tmp, log, arm_id = self.killed(LINES + [LEAK % KEY])
        w = inst.worker()
        w.gpu.recover()
        self.assertNoTemporary(inst, tmp, arm_id)
        self.assertDiscarded(inst, arm_id, self.diary_event(inst, arm_id), KEY)

    # The keys rotated between the death and the next start (MFL023 B11, SOL-B10-B1; B12, SOL-B11-B1). The next worker
    # reads the same env file overwritten with another key, or another env file while the one of the session is kept,
    # gone, no longer private, a link or unreadable. In every case the partial log goes, unread.

    def rotated(self, lines, session="kept", worker_mode=0o600):
        """A worker killed with `lines` in its log, then the recovery of the next start. `session="same"`: the env file
        of the session overwritten with another key (the next worker reads it); otherwise the next worker reads
        `rotated.env` (another key, mode `worker_mode`) and the env file of the session is `session`: kept, absent,
        0644, a link or unreadable. (instance, arm_id, the event `worker-log-partial` of the diary)."""
        inst, tmp, log, arm_id = self.killed(lines)
        old, new = inst.env, inst.root / "rotated.env"
        content = "RUNPOD_API_KEY=%s\nMODAL_PROXY_TOKEN_ID=wk-b11-id\nMODAL_PROXY_TOKEN_SECRET=ws-b11-secret\n" % NEW
        if session == "same":
            old.write_text(content)
            self.assertEqual(stat.S_IMODE(os.lstat(old).st_mode), 0o600)
        else:
            new.write_text(content)
            new.chmod(worker_mode)
            inst.env = new
        if session == "absent":
            old.unlink()
        elif session == "0644":
            old.chmod(0o644)
        elif session == "unreadable":
            old.chmod(0o000)
        elif session == "link":
            moved = inst.root / "moved.env"
            old.rename(moved)
            old.symlink_to(moved)
        w = inst.worker()
        w.gpu.recover()
        self.assertNoTemporary(inst, tmp, arm_id)
        self.assertEqual(inst.sessions()[0]["state"], "orphan")                    # the window recovered all the same
        return inst, arm_id, self.diary_event(inst, arm_id)

    def test_the_keys_rotated_in_the_same_env_file(self):
        """MFL023 B12 (SOL-B11-B1): the env file of the session readable, with the new key only; the old key in the log
        is not among the keys it gives, and the log still goes."""
        inst, arm_id, partial = self.rotated(LINES + [LEAK % KEY], session="same")
        self.assertDiscarded(inst, arm_id, partial, KEY, NEW)

    def test_the_env_file_of_the_session_absent_after_a_rotation(self):
        inst, arm_id, partial = self.rotated(LINES + [LEAK % KEY], session="absent")
        self.assertDiscarded(inst, arm_id, partial, KEY, NEW)

    def test_the_env_file_of_the_session_not_private_after_a_rotation(self):
        for how in ("0644", "link"):
            with self.subTest(how):
                inst, arm_id, partial = self.rotated(LINES + [LEAK % KEY], session=how)
                self.assertDiscarded(inst, arm_id, partial, KEY, NEW)

    def test_the_env_file_of_the_session_unreadable_after_a_rotation(self):
        inst, arm_id, partial = self.rotated(LINES + [LEAK % KEY], session="unreadable")
        self.assertDiscarded(inst, arm_id, partial, KEY, NEW)

    def test_the_env_file_of_the_worker_not_private(self):
        inst, arm_id, partial = self.rotated(LINES + [LEAK % NEW], worker_mode=0o644)
        self.assertDiscarded(inst, arm_id, partial, KEY, NEW)

    def test_both_env_files_readable_after_a_rotation(self):
        """Both files readable: the log goes, with the old key in it or without any key."""
        for lines in (LINES + [LEAK % KEY], LINES):
            with self.subTest(key=len(lines) > len(LINES)):
                inst, arm_id, partial = self.rotated(lines)
                self.assertDiscarded(inst, arm_id, partial, KEY, NEW)


NEW = "rk-b11-rotated-synthetic-key/9876543210=="
LEAK = "2026-11-24T09:51:02.000Z\t+3.0\tw1\tcontainer\t-\tBearer %s"
LOG_TEXT = "INFO: the proxy is up"


if __name__ == "__main__":
    unittest.main()
