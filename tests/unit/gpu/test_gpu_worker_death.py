"""The death of the worker (SIGKILL) in every narrow window of a GPU session, and its next start (MFL023 B4; plan r2 §6
point 6, r3 §6 point 6).

A worker in a subprocess (`worker_driver.py`) runs a job of the route marker-server on a fake provider and is killed at a
barrier: after the row `opening` and before the row `arm`; after the row `arm` and before the arm; during the arm (its
child dies with it); armed before the first reservation; after the reservation; after the completion and before the
publication; during the publication; in LINGER; in CLOSING before the zero, after the zero before the row, after the row
before the lock. At the next start, before any claim: the fake provider disarmed whenever there is an arm row; no job
without an outcome; a coherent ledger (the windows armed and never disarmed held as orphans at the cap); three distinct
facts: the completion in the ledger, the outcome of the job and its publication, with `runs.gpu` kept."""
from dataclasses import asdict
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gpu_harness import Instance, wait_for  # noqa: E402

from markdown_for_llms.bench import serverspend  # noqa: E402

DRIVER = Path(__file__).resolve().parent / "worker_driver.py"
TERMINAL = {"completed", "failed", "interrupted", "cancelled"}


class TestDeath(unittest.TestCase):
    def kill_at(self, barrier, close=False, state=None):
        inst = Instance(self, state=state)
        file = inst.root / "barrier"
        if barrier == "during_arm":
            inst.change(barrier={"arm": str(file)})
        elif barrier == "closing_before_zero":
            inst.change(barrier={"zero": str(file)})
        job = inst.submit()
        cfg = inst.root / "config.json"
        cfg.write_text(json.dumps({"settings": asdict(inst.settings), "env_file": str(inst.env), "state": str(inst.state),
                                   "barrier": barrier, "file": str(file), "close": close}))
        p = subprocess.Popen([sys.executable, "-I", "-B", str(DRIVER), str(cfg)], stdout=subprocess.DEVNULL,
                             stderr=subprocess.PIPE)
        try:
            wait_for(lambda: file.exists() or p.poll() is not None, within=60, what="the barrier " + barrier)
            if p.poll() is not None:
                self.fail("the worker ended before %s: %s" % (barrier, p.stderr.read().decode()[-2000:]))
            time.sleep(0.1)
            p.kill()
            p.wait(10)
        finally:
            if p.poll() is None:
                p.kill()
                p.wait(10)
            p.stderr.close()
        child = int(file.read_text()) if barrier in ("during_arm", "closing_before_zero") else None
        before = len(inst.provider_state().get("calls", []))
        w = inst.worker()
        w.gpu.recover()
        w.reconcile()
        after = [c[0] for c in inst.provider_state().get("calls", [])][before:]
        return inst, w, job, after, child

    def assertCoherent(self, inst, job, armed=True):
        rows = inst.rows()
        serverspend.parse(rows)
        run = inst.run(job["id"])
        self.assertIn(run["state"], TERMINAL)
        if armed:
            self.assertFalse(inst.provider_state().get("armed"))             # disarmed before any claim
        return run

    def test_after_the_session_row_and_before_the_arm_row(self):
        inst, w, job, after, _ = self.kill_at("before_arm_row")
        self.assertEqual(after, [])                                          # never armed: no call
        session, = inst.sessions()
        self.assertEqual((session["state"], session["reason"]), ("closed", "never_armed"))
        self.assertEqual(inst.rows(), [])
        self.assertEqual(self.assertCoherent(inst, job, armed=False)["state"], "interrupted")

    def test_after_the_arm_row_before_the_arm(self):
        inst, w, job, after, _ = self.kill_at("before_arm")
        self.assertEqual(after[:1], ["disarm"])
        session, = inst.sessions()
        self.assertEqual(session["state"], "orphan")
        self.assertEqual(inst.summary()["orphans"], [session["arm_id"]])
        self.assertEqual(self.assertCoherent(inst, job)["state"], "interrupted")

    def test_during_the_arm(self):
        inst, w, job, after, child = self.kill_at("during_arm")
        wait_for(lambda: not Path("/proc/%d" % child).exists()
                 or Path("/proc/%d/stat" % child).read_text().rsplit(")", 1)[1].split()[0] == "Z",
                 what="the arm child gone with its worker")
        self.assertEqual(after[:1], ["disarm"])
        self.assertEqual(inst.sessions()[0]["state"], "orphan")
        self.assertEqual(self.assertCoherent(inst, job)["state"], "interrupted")

    def test_armed_before_the_first_reservation(self):
        inst, w, job, after, _ = self.kill_at("armed_before_reserve")
        self.assertEqual(after[:1], ["disarm"])
        self.assertEqual(inst.rows("reserve"), [])
        self.assertEqual(self.assertCoherent(inst, job)["state"], "interrupted")

    def test_after_the_reservation(self):
        inst, w, job, after, _ = self.kill_at("after_reserve")
        self.assertEqual(after[:1], ["disarm"])
        self.assertEqual((len(inst.rows("reserve")), inst.rows("complete")), (1, []))
        self.assertEqual(self.assertCoherent(inst, job)["state"], "interrupted")

    def test_after_the_completion_and_before_the_publication(self):
        inst, w, job, after, _ = self.kill_at("after_complete")
        self.assertEqual(len(inst.rows("complete")), 1)                       # the completion of the ledger
        run = self.assertCoherent(inst, job)
        self.assertEqual(run["state"], "interrupted")                        # the outcome: no publication
        self.assertTrue(json.loads(run["gpu"])["completed"])                 # runs.gpu kept
        self.assertIsNone(run["result_path"])

    def test_during_the_publication(self):
        inst, w, job, after, _ = self.kill_at("publishing")
        run = self.assertCoherent(inst, job)
        self.assertEqual(run["state"], "completed")                          # the reconciliation completes it
        self.assertEqual(len(inst.rows("complete")), 1)
        self.assertTrue(json.loads(run["gpu"])["completed"])
        self.assertEqual(inst.sessions()[0]["state"], "orphan")

    def test_in_linger(self):
        inst, w, job, after, _ = self.kill_at("linger")
        self.assertEqual(after[:1], ["disarm"])
        self.assertEqual(self.assertCoherent(inst, job)["state"], "completed")
        self.assertEqual(inst.sessions()[0]["state"], "orphan")

    def test_in_closing_before_the_zero(self):
        inst, w, job, after, child = self.kill_at("closing_before_zero", close=True)
        self.assertIn("zero", after)
        self.assertEqual(inst.sessions()[0]["state"], "orphan")
        self.assertEqual(self.assertCoherent(inst, job)["state"], "completed")

    def test_after_the_zero_before_the_row(self):
        inst, w, job, after, _ = self.kill_at("after_zero_before_row", close=True)
        self.assertEqual(after[:1], ["disarm"])                              # no row yet: disarmed again
        dis, = inst.rows("disarm")
        self.assertTrue(dis["orphan"])
        self.assertEqual(self.assertCoherent(inst, job)["state"], "completed")

    def test_after_the_row_before_the_lock(self):
        inst, w, job, after, _ = self.kill_at("after_row_before_lock", close=True)
        self.assertEqual(after, [])                                          # the row closed it: no call
        session, = inst.sessions()
        dis, = inst.rows("disarm")
        self.assertFalse(dis["orphan"])
        self.assertEqual(session["state"], "closed")
        self.assertFalse(serverspend.lock_path(inst.ledger_dir / "spend.jsonl", session["arm_id"]).exists())
        self.assertEqual(self.assertCoherent(inst, job)["state"], "completed")


if __name__ == "__main__":
    unittest.main()
