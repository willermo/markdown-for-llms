"""The discipline between the threads of a GPU session (MFL023 B4; SOL2-N1 of the plan r2, plan r3 §4.2.1 and §6 point
9, and the note of Sol in the review of r3 on a child started with the first request to close).

The clock thread alone runs CLOSING and writes the final states; the rows of `gpu_sessions` change by compare-and-set;
each child has one owner; no lock of the ledger is held during a call, a wait or a join. Barriers: T with the wake of
an attempt, T with its completion, a stop (SIGTERM) with CLOSING, a disarm with the end of LINGER, a late write of
`linger` after `closing`, the arm child started at the moment of the first request to close."""
from pathlib import Path
import sys
import threading
import time
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gpu_harness import Instance, gpu_convert, wait_for  # noqa: E402

from markdown_for_llms.bench import serverspend, spend  # noqa: E402
from markdown_for_llms.gpu import lifecycle  # noqa: E402

W_RUNPOD = 1703


def process(inst, w, convert=None):
    row = w.claim()
    with mock.patch("markdown_for_llms.worker.convert", convert or gpu_convert(inst.clocks)):
        w.process(row)
    return inst.run(row["id"])


class Base(unittest.TestCase):
    def coherent(self, inst):
        """The ledger parses, every attempt is in a window, no reservation after its window's disarm."""
        rows = inst.rows()
        serverspend.parse(rows)
        order = {}
        for i, r in enumerate(rows):
            order.setdefault((r.get("kind"), r.get("arm_id")), i)
        for i, r in enumerate(rows):
            if r.get("kind") == "reserve":
                dis = order.get(("disarm", r["arm_id"]))
                self.assertTrue(dis is None or dis > i, "a reservation after the disarm of its window")
        return rows


class TestBarriers(Base):
    def test_t_with_the_wake_of_an_attempt(self):
        """The close at T asked and CLOSING written while the attempt reaches its wake: the reservation is refused
        (`server_not_armed`), nothing is reserved after the window closed, the job has its outcome."""
        inst = Instance(self)
        w = inst.worker()
        inst.submit()

        def before_wake():                                       # no lock of the ledger is held here
            w.gpu.session.state.request_close("term")
            wait_for(lambda: inst.rows("disarm"), what="the disarm row of CLOSING")
        run = process(inst, w, gpu_convert(inst.clocks, before_wake=before_wake))
        self.assertEqual(run["state"], "failed")
        rows = self.coherent(inst)
        self.assertEqual([r["kind"] for r in rows], ["arm", "disarm"])
        self.assertLessEqual(len([r for r in rows if r["kind"] == "reserve"]), 1)

    def test_t_with_the_completion(self):
        inst = Instance(self)
        w = inst.worker()
        inst.submit()
        real = w.gpu.done
        started = threading.Event()

        def done(row, hook, result):
            w.gpu.session.state.request_close("term")
            wait_for(lambda: w.gpu.session.state.snapshot()[0] == "closing", what="CLOSING started")
            started.set()
            return real(row, hook, result)
        with mock.patch.object(w.gpu, "done", side_effect=done):
            process(inst, w)
        inst.settle(w)
        rows = self.coherent(inst)
        self.assertEqual(sorted(r["kind"] for r in rows), ["arm", "complete", "disarm", "reserve"])
        self.assertTrue(started.is_set())

    def test_a_stop_with_closing(self):
        """SIGTERM sets the stop while CLOSING waits at the disarm: one disarm row, the exit after the final state."""
        inst = Instance(self)
        barrier = str(inst.root / "disarm.barrier")
        inst.change(barrier={"disarm": barrier})
        w = inst.worker()
        inst.submit()
        process(inst, w)
        inst.request("disarm")
        w.gpu.poll()
        wait_for(lambda: Path(barrier).exists(), what="the disarm in progress")
        w.stop = True
        done = threading.Event()
        threading.Thread(target=lambda: (w.gpu.shutdown(), done.set()), daemon=True).start()
        time.sleep(0.3)
        self.assertFalse(done.is_set())                          # the exit waits for the final state
        Path(barrier + ".go").write_text("")
        self.assertTrue(done.wait(20))
        self.assertEqual(len(inst.rows("disarm")), 1)
        self.assertEqual(inst.sessions()[0]["state"], "closed")
        self.assertEqual(inst.sessions()[0]["reason"], "user_disarm")

    def test_a_disarm_with_the_end_of_linger(self):
        inst = Instance(self)
        w = inst.worker()
        inst.submit()
        process(inst, w)
        inst.request("disarm")
        inst.clocks.advance(61)
        s = w.gpu.session
        threads = [threading.Thread(target=w.gpu.poll), threading.Thread(target=s.state.request_close, args=("idle",))]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        inst.settle(w)
        self.assertEqual(len(inst.rows("disarm")), 1)
        self.assertIn(inst.sessions()[0]["reason"], ("idle", "user_disarm"))
        self.assertEqual(inst.calls().count("disarm"), 1)

    def test_a_late_write_never_undoes_a_final_state(self):
        inst = Instance(self)
        w = inst.worker()
        inst.submit()
        process(inst, w)
        s = w.gpu.session
        arm_id = s.arm_id
        s.state.request_close("idle")
        inst.settle(w)
        self.assertFalse(lifecycle.transition(inst.service.storage, arm_id, "linger", ("open",)))
        self.assertFalse(lifecycle.transition(inst.service.storage, arm_id, "open", ("linger", "opening")))
        self.assertFalse(s.state.move("linger", {"open"}))
        self.assertEqual(inst.sessions()[0]["state"], "closed")

    def test_the_arm_child_started_with_the_first_request_to_close(self):
        """The arm child waits at its barrier; a disarm is asked at that moment: the child is killed by its owner (never
        by the clock thread), CLOSING disarms, one disarm row."""
        inst = Instance(self)
        barrier = str(inst.root / "arm.barrier")
        inst.change(barrier={"arm": barrier})
        w = inst.worker()
        inst.submit()

        def close_at_the_launch():
            wait_for(lambda: Path(barrier).exists(), what="the arm child started")
            w.gpu.session.state.request_close("user_disarm")
        threading.Thread(target=close_at_the_launch, daemon=True).start()
        run = process(inst, w)
        self.assertEqual(run["state"], "failed")
        inst.settle(w)
        self.assertEqual(len(inst.rows("disarm")), 1)
        session = inst.sessions()[0]
        self.assertEqual((session["state"], session["reason"]), ("closed", "user_disarm"))
        pid = int(Path(barrier).read_text())
        wait_for(lambda: not Path("/proc/%d" % pid).exists() or "Z" in Path("/proc/%d/stat" % pid).read_text().split(")")[1][:3],
                 what="the arm child gone")


if __name__ == "__main__":
    unittest.main()
