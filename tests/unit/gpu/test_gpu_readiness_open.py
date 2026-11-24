"""The readiness of a GPU preset whose session is open (MFL023 B10, SOL-B-N1 of the review of the diff B).

The readiness tells two things apart: whether a new session could open now (the reserve `A` of the ledger, the minimum
reserve), and whether the open session of the preset can still host a job (the floor of the engine on what is left of
its window and on `T`, with `MDF_MAX_PAGES` pages). With the cap of the provider equal to the reserve of the open
session nothing is left for a new opening, yet the preset stays available for that session; once its window can no
longer host a job, the preset is `server_cap` again. The floor and the checks of the engine are unchanged."""
import io
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gpu_harness import JOBS, PDF, Instance  # noqa: E402

PRESET = "runpod-serverless-volume-dc"


class TestTheReadinessOfAnOpenSession(unittest.TestCase):
    def test_coinciding_caps_and_a_job_through_the_preset(self):
        inst = Instance(self, session_cap="0.40", provider_caps=("3", "0.40"))
        w = inst.worker()
        before = w.gpu.readiness()["presets"][PRESET]
        self.assertEqual((before["available"], before["a_cents"], before["w_s"]), (True, "40.00", 1703))
        opened = w.gpu.open(PRESET, "job")
        ready = w.gpu.readiness()
        item = ready["presets"][PRESET]
        self.assertEqual((item["available"], item["code"]), (True, None))       # B9: (False, "server_cap")
        self.assertTrue(item["open_session"])
        self.assertEqual((item["a_cents"], item["w_s"]), ("0.00", 0))         # a new opening would get nothing
        self.assertFalse(ready["presets"]["modal-base"].get("open_session"))
        w.heartbeat({"ready": True, "message": "Local worker ready", "converters": None, "gpu": ready})
        listed = {p["name"]: p for p in inst.service.presets()["presets"]}
        self.assertEqual((listed["batch"]["available"], listed["batch"]["code"]), (True, None))
        estimate = inst.service.gpu_estimate(PRESET)
        self.assertEqual((estimate["available"], estimate["open_session"]), (True, True))
        options = inst.service.preset_options("batch", {"transit_confirmed": True})
        job = inst.service.submit(io.BytesIO(PDF), "a.pdf", options)
        row = w.claim()
        self.assertEqual(row["id"], job["id"])
        ep, keys, srv, hook = w.gpu.prepare(row, JOBS["runpod"], {"server": {"mode": "balanced",
                                                                            "transit_confirmed": True}})
        self.assertEqual((srv["session"]["arm_id"], srv["budget_cents"]), (opened.arm_id, "40.00"))

    def test_a_window_that_can_no_longer_host_a_job(self):
        inst = Instance(self, session_cap="0.40", provider_caps=("3", "0.40"))
        w = inst.worker()
        w.gpu.open(PRESET, "job")
        inst.clocks.advance(1500)                       # 203 s left of W = 1703 s: under the floor of one job
        item = w.gpu.readiness()["presets"][PRESET]
        self.assertEqual((item["available"], item["code"], item["open_session"]), (False, "server_cap", False))

    def test_a_session_closing_does_not_serve(self):
        inst = Instance(self, session_cap="0.40", provider_caps=("3", "0.40"))
        w = inst.worker()
        s = w.gpu.open(PRESET, "job")
        s.state.request_close("user_disarm")
        inst.settle(w)
        item = w.gpu.readiness()["presets"][PRESET]
        self.assertFalse(item["open_session"])


if __name__ == "__main__":
    unittest.main()
