"""The worker and the code of its launcher, while the worker runs (MFL051, via of A2-M2, point 2).

Every child of the engine is a container of the image of the launcher. The probe at the start of the worker
compares its own code with the launcher's (`hello`: `payload_sha256`, and `inputs_sha256` when the image recorded
them). That is not enough for a worker that lives for days: `make up` after a change may recreate the launcher
while a worker of the old image goes on, and its children would then run another code than the worker. So:

- the comparison, which starts no container, is repeated at every check of the routes (READY_SECONDS);
- when the code differs the worker takes no job, and its heartbeat says `stale engine image ...: run `make up``;
- when the launcher is another one than the one of the last probe (`instance`, the ID of its container), the
  whole probe is repeated, and a probe that fails stops the claims in the same way;
- a launcher that does not answer is not judged here: the routes that need it say so, the others go on.

The worker is the real one; the launcher is the scripted one of the tests, which changes what it reports while
the worker runs.
"""
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from markdown_for_llms.services import Service
from markdown_for_llms.settings import Settings
from markdown_for_llms.storage import StorageError
from markdown_for_llms import worker as worker_module
from markdown_for_llms.worker import Worker

from engine_fake import fake_convert, launcher_fake

ROUTES = {name: {"available": True, "code": None} for name in ("text", "pandoc", "libreoffice", "djvutxt", "marker")}
OTHER = "0" * 64


class StaleEngineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.settings = Settings(data_root=self.tmp.name + "/data", model_root=self.tmp.name + "/models")
        self.service = Service(self.settings)
        self.addCleanup(self.service.storage.close)
        self.launcher = launcher_fake.patched(self, Path(self.tmp.name) / "volume")
        for target, value in (("markdown_for_llms.worker.availability", lambda **kw: dict(ROUTES)),
                              ("markdown_for_llms.worker.convert", fake_convert())):
            patcher = mock.patch(target, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.worker = Worker(self.settings)
        self.addCleanup(self.worker.storage.close)

    def submit(self):
        return self.service.submit(io.BytesIO(b"%PDF-1.4\nsynthetic stale\n"), "p.pdf")["id"]

    def state(self, job):
        return self.service.get(job)["state"]

    def turn(self):
        """One turn of the loop after READY_SECONDS have passed; returns the readiness of the heartbeat."""
        self.worker._ready_at = None
        self.worker.turn()
        with self.service.storage.connect() as conn:
            return json.loads(conn.execute("SELECT readiness FROM worker").fetchone()[0])

    def test_the_same_code_is_asked_at_every_check_of_the_routes_without_a_container(self):
        job = self.submit()
        readiness = self.turn()
        self.assertIs(readiness["ready"], True)
        self.assertEqual(self.state(job), "completed")
        self.assertEqual(len(self.launcher.ops("hello")), 1)
        self.worker.turn()                                              # within READY_SECONDS: not asked again
        self.assertEqual(len(self.launcher.ops("hello")), 1)
        self.turn()
        self.assertEqual(len(self.launcher.ops("hello")), 2)
        self.assertEqual(self.launcher.ops("run"), [])                  # the comparison starts no child

    def test_a_launcher_of_another_build_stops_the_claims_until_the_code_is_the_same_again(self):
        first = self.submit()
        self.assertIs(self.turn()["ready"], True)
        self.assertEqual(self.state(first), "completed")
        # `make up` rebuilt the image and recreated the launcher; this worker is still the one of the old image
        self.launcher.code = {"payload_sha256": OTHER}
        second = self.submit()
        for _ in range(3):
            readiness = self.turn()
            self.assertEqual((readiness["ready"], readiness["code"], readiness["converters"]),
                             (False, "isolation_unavailable", None))
            self.assertIn("stale engine image (payload_sha256): run `make up`", readiness["message"])
            self.assertEqual(self.state(second), "queued")              # not taken, not failed
        self.assertEqual(self.launcher.ops("run"), [])
        self.launcher.code = {}
        self.assertIs(self.turn()["ready"], True)
        self.assertEqual(self.state(second), "completed")

    def test_the_build_inputs_count_when_the_image_of_the_worker_recorded_them(self):
        mine = dict(self.worker.code())
        self.launcher.code = {"inputs_sha256": OTHER}
        self.assertEqual(self.worker.launcher_code()[1], [] if mine["inputs_sha256"] is None else
                         ["stale engine image (inputs_sha256): run `make up`"])
        self.worker._code = dict(mine, inputs_sha256="1" * 64)          # a worker that runs from an image
        self.assertEqual(self.worker.launcher_code()[1], ["stale engine image (inputs_sha256): run `make up`"])
        self.launcher.code = {"inputs_sha256": "1" * 64}
        self.assertEqual(self.worker.launcher_code()[1], [])
        self.launcher.code = {"inputs_sha256": "1" * 64, "payload_sha256": OTHER}
        self.assertEqual(self.worker.launcher_code()[1], ["stale engine image (payload_sha256): run `make up`"])

    def test_with_once_a_stale_image_is_an_error_and_no_job_is_taken(self):
        job = self.submit()
        self.launcher.code = {"payload_sha256": OTHER}
        with self.assertRaises(StorageError) as ctx:
            self.worker.turn(once=True)
        self.assertIn("stale engine image", str(ctx.exception))
        self.assertEqual(self.state(job), "queued")

    def test_a_launcher_that_does_not_answer_is_not_judged_here(self):
        self.launcher.down = True
        self.assertEqual(self.worker.same_engine(), [])
        self.assertIsNotNone(self.worker.ready())                       # the routes say what they need (`availability`)
        self.assertIs(self.worker.readiness["ready"], True)

    def test_another_launcher_than_the_one_of_the_probe_means_the_whole_probe_again(self):
        probes = []

        def probe():
            probes.append(self.launcher.instance)
            self.worker.launcher_instance = self.launcher.instance      # what a probe that passes records
            return []
        with mock.patch.object(self.worker, "probe", probe):
            self.worker.launcher_instance = self.launcher.instance      # the probe of the start
            job = self.submit()
            self.assertIs(self.turn()["ready"], True)
            self.assertEqual(probes, [])                                # the same launcher: no child
            self.launcher.instance = "fixture-launcher-2"               # recreated: the same code, another container
            self.assertIs(self.turn()["ready"], True)
            self.assertEqual(probes, ["fixture-launcher-2"])
            self.turn()
            self.assertEqual(probes, ["fixture-launcher-2"])            # once per launcher
        self.assertEqual(self.state(job), "completed")

    def test_a_repeated_probe_that_fails_stops_the_claims_and_is_tried_again(self):
        outcomes = [["probe: exit 1, None"], ["seccomp: off"], []]
        asked = []

        def probe():
            asked.append(1)
            failed = outcomes[len(asked) - 1]
            if not failed:
                self.worker.launcher_instance = self.launcher.instance
            return failed
        with mock.patch.object(self.worker, "probe", probe):
            self.worker.launcher_instance = "the-launcher-of-the-start"
            job = self.submit()
            for expected in ("probe: exit 1, None", "seccomp: off"):
                readiness = self.turn()
                self.assertEqual((readiness["ready"], readiness["code"]), (False, "isolation_unavailable"))
                self.assertIn(expected, readiness["message"])
                self.assertEqual(self.state(job), "queued")
            self.assertIs(self.turn()["ready"], True)
        self.assertEqual((len(asked), self.state(job)), (3, "completed"))

    def test_the_probe_records_its_launcher_only_when_it_passes_and_each_one_has_its_own_folder(self):
        folders = []

        def execute(settings, scratch, module, **kw):
            folders.append(Path(scratch).name)
            work = Path(scratch) / "probe"
            work.mkdir()
            (work / "probe.json").write_text("{}")
            return {"exit": 0, "reason": None, "asked": False, "elapsed_seconds": 0.0, "dir": work}
        verdicts = [["no_new_privs: 0"], [], []]
        with mock.patch.object(worker_module, "execute", execute), \
                mock.patch.object(worker_module.selfprobe, "verdict", lambda report, **kw: verdicts.pop(0)):
            self.assertEqual(self.worker.probe(), ["no_new_privs: 0"])
            self.assertIsNone(self.worker.launcher_instance)            # a probe that failed is repeated
            self.assertEqual(self.worker.probe(), [])
            self.assertEqual(self.worker.launcher_instance, "fixture-launcher-1")
            self.launcher.instance = "fixture-launcher-2"
            self.assertEqual(self.worker.same_engine(), [])             # the whole probe, against the new launcher
            self.assertEqual(self.worker.launcher_instance, "fixture-launcher-2")
        token = self.worker.token
        self.assertEqual(folders, ["worker-" + token, "worker-%s-1" % token, "worker-%s-2" % token])
        self.assertEqual(self.worker.kept, {})                          # every commitment of a probe was released


if __name__ == "__main__":
    unittest.main()
