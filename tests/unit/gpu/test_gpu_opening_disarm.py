"""A disarm while the session of its preset is OPENING (MFL023 B10, SOL-B-B1 of the review of the diff B).

The main thread of the worker is busy in the children of OPENING (the reading at rest, the arm), so the requests of the
API are not read by `Manager.requests()` then: the stop of those children reads the requests to disarm the preset and
asks for the close (`user_disarm`). The worker is the real one, the request goes through the service of the API
(`Service.gpu_disarm`), the provider is a fake whose call waits at a barrier (`provider_driver.py`) for 30 s at most:
the child is reaped before the fake is let go, then one close only (the disarm and the zero), the jobs of the preset
cancelled, no request left pending."""
import os
from pathlib import Path
import sys
import threading
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gpu_harness import Instance, gpu_convert, wait_for  # noqa: E402

PRESET = "runpod-serverless-volume-dc"


def alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


class TestADisarmDuringTheOpening(unittest.TestCase):
    def blocked(self, op):
        """A job of the preset claimed and processed in a thread until the child of `op` waits at its barrier."""
        inst = Instance(self)
        barrier = inst.root / (op + ".barrier")
        inst.change(barrier={op: str(barrier)})
        w = inst.worker()
        job, queued = inst.submit(), inst.submit()
        row = w.claim()
        errors = []

        def process():
            try:
                with mock.patch("markdown_for_llms.worker.convert", gpu_convert(inst.clocks)):
                    w.process(row)
            except Exception as exc:                    # noqa: BLE001 - asserted below
                errors.append(exc)
        thread = threading.Thread(target=process, daemon=True)
        thread.start()
        self.addCleanup(thread.join, 40)
        self.addCleanup(lambda: Path(str(barrier) + ".go").write_text("go"))      # never left waiting on a red run
        wait_for(barrier.exists, within=20, what="the child of %s at its barrier" % op)
        return inst, w, job, queued, barrier, thread, errors

    def assertDisarmed(self, inst, answer, job, queued):
        self.assertEqual(answer["cancelled"], {"running": 1, "queued": 1})
        self.assertEqual((inst.run(job["id"])["state"], inst.run(queued["id"])["state"]), ("cancelled", "cancelled"))
        with inst.service.storage.connect() as conn:
            requests = [dict(r) for r in conn.execute("SELECT kind, state FROM gpu_requests")]
        self.assertEqual(requests, [{"kind": "disarm", "state": "done"}])

    def test_a_disarm_during_the_arm(self):
        inst, w, job, queued, barrier, thread, errors = self.blocked("arm")
        child = int(barrier.read_text())
        answer = inst.service.gpu_disarm(PRESET)
        wait_for(lambda: not alive(child), within=10, what="the arm child reaped before its barrier lets it go")
        Path(str(barrier) + ".go").write_text("go")
        thread.join(30)
        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        inst.settle(w)
        session, = inst.sessions()
        self.assertEqual((session["state"], session["reason"]), ("closed", "user_disarm"))
        self.assertEqual(inst.calls(), ["state", "arm", "disarm", "zero"])        # one close: one disarm, one zero
        self.assertEqual(len(inst.rows("arm")), 1)
        self.assertEqual(len(inst.rows("disarm")), 1)
        self.assertFalse(inst.provider_state().get("armed"))
        self.assertDisarmed(inst, answer, job, queued)
        self.assertIsNone(w.gpu.session)
        self.assertIsNone(w.claim())

    def test_a_disarm_during_the_reading_at_rest(self):
        """Before the durable row: nothing was armed, no row, no window; the job cancelled."""
        inst, w, job, queued, barrier, thread, errors = self.blocked("state")
        child = int(barrier.read_text())
        answer = inst.service.gpu_disarm(PRESET)
        wait_for(lambda: not alive(child), within=10, what="the child of the reading reaped before its barrier")
        Path(str(barrier) + ".go").write_text("go")
        thread.join(30)
        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        w.gpu.poll()
        self.assertEqual((inst.sessions(), inst.calls()), ([], ["state"]))
        self.assertEqual(inst.rows(), [])
        self.assertDisarmed(inst, answer, job, queued)

    def test_a_disarm_older_than_the_job_does_not_close_its_opening(self):
        """A request made before the job that opens the session (it cancelled only the jobs that existed then) is not
        the disarm of this opening: the session opens; the request is handled as before, by the next `poll`."""
        inst = Instance(self)
        w = inst.worker()
        inst.service.gpu_disarm(PRESET)
        with inst.service.storage.connect() as conn:
            conn.execute("UPDATE gpu_requests SET created=created-60")
        job = inst.submit()
        row = w.claim()
        with mock.patch("markdown_for_llms.worker.convert", gpu_convert(inst.clocks)):
            w.process(row)
        self.assertEqual(inst.run(job["id"])["state"], "completed")
        w.gpu.poll()
        inst.settle(w)
        session, = inst.sessions()
        self.assertEqual((session["state"], session["reason"]), ("closed", "user_disarm"))
        with inst.service.storage.connect() as conn:
            self.assertEqual([r[0] for r in conn.execute("SELECT state FROM gpu_requests")], ["done"])


if __name__ == "__main__":
    unittest.main()
