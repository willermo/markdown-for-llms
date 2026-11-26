"""The heartbeat of a worker that left (MFL023 B10, SOL-B-N2 and C-DB-NB2.1 of the review of the diff B).

The API reads one row `worker`, by its age only (`Service.worker_gpu`, `api.worker_state`): a worker that ended cleanly
(`--once`, SIGINT, SIGTERM) left its readiness readable for the lease (15 s), with the presets and the caps of its own
configuration. At a clean exit of `Worker.run()` the worker expires its heartbeat, only while the row is still its own:
a worker that ends never expires the heartbeat of the one after it. A worker stopped by a GPU window it cannot close
keeps its heartbeat with the code and the commands (`test_gpu_recover.py`)."""
from pathlib import Path
import sys
import time
import unittest
from unittest import mock

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gpu_harness import Instance  # noqa: E402

from markdown_for_llms.api import create_app  # noqa: E402


def run_once(w):
    with mock.patch("markdown_for_llms.worker.Worker.probe", return_value=[]), \
            mock.patch("markdown_for_llms.worker.availability", return_value={}):
        w.run(once=True)


class TestTheHeartbeatAtTheExit(unittest.TestCase):
    def row(self, inst):
        with inst.service.storage.connect() as conn:
            return dict(conn.execute("SELECT token, heartbeat FROM worker WHERE id=1").fetchone())

    def test_a_worker_once_that_ends(self):
        inst = Instance(self)
        w = inst.worker()
        seen = []
        real = w.claim

        def claim():
            seen.append(inst.service.worker_gpu())           # while it runs, the API reads its GPU readiness
            return real()
        with mock.patch.object(w, "claim", side_effect=claim):
            run_once(w)
        self.assertIsNotNone(seen[0])
        self.assertIn("runpod-serverless-volume-dc", seen[0]["presets"])
        self.assertIsNone(inst.service.worker_gpu())
        client = TestClient(create_app(inst.settings), base_url="http://127.0.0.1:%d" % inst.settings.port)
        ready = client.get("/api/v1/readiness").json()
        self.assertEqual((ready["ready"], ready["worker"]), (False, "Worker absent; archive available"))
        self.assertNotIn("gpu", ready)
        self.assertEqual(self.row(inst)["token"], w.token)

    def test_the_end_of_a_worker_leaves_the_heartbeat_of_the_next(self):
        inst = Instance(self)
        first, second = inst.worker(), inst.worker()

        def claim():
            second.ready()
            second.heartbeat()                                  # the next worker writes its row before the first ends
            return None
        with mock.patch.object(first, "claim", side_effect=claim):
            run_once(first)
        row = self.row(inst)
        self.assertEqual(row["token"], second.token)
        self.assertGreater(row["heartbeat"], time.time() - inst.settings.lease_seconds)
        self.assertIsNotNone(inst.service.worker_gpu())


if __name__ == "__main__":
    unittest.main()
