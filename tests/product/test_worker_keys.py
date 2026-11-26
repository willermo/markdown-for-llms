"""The keys of the worker in containers (MFL051, plan r2.1 §5.2, J.5.2; rilievo I; E-1; via A2-M2 point 5).

- The worker starts whatever the state of its keys file: its settings do not come from it. A file that is not
  safe (another mode, a link, absent) makes the routes with a key unavailable with their closed code
  (`datalab_key_unsafe`), the local routes go on, no key is read and nothing falls back to the environment.
- The key of Datalab is read from the file at every use: a `chmod` given while the worker runs holds from the
  next read, in both directions.
- The file is ONE file bound from the host: the worker keeps the inode it started with. Once that file is
  replaced (`os.replace`, what an editor or `sed -i` does) or removed, no key is read any more, and the worker
  says to run `make restart`. The mount is stood for by a descriptor held on the file.
- Declared limit (E-1): an old file that still has a name (a backup) is still read until the restart.
- No key is ever in the environment, in the readiness, in the archive.
"""
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from markdown_for_llms.engine import envfile
from markdown_for_llms.services import Service
from markdown_for_llms.settings import Settings
from markdown_for_llms.worker import Worker

from engine_fake import fake_convert, launcher_fake

SECRET = "dl-synthetic-0123456789"
PDF = b"%PDF-1.4\nsynthetic worker keys\n"
LOCAL = {name: {"available": True, "code": None} for name in ("text", "pandoc", "libreoffice", "djvutxt", "marker")}
MOUNT = "/run/mdf/env"


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.settings = Settings(data_root=str(self.root / "data"), model_root=str(self.root / "models"), datalab_cap_usd="0.05")
        self.service = Service(self.settings)
        self.addCleanup(self.service.storage.close)
        self.launcher = launcher_fake.patched(self, self.root / "volume")        # it attests the end of every attempt
        patcher = mock.patch("markdown_for_llms.worker.availability", lambda **kw: dict(LOCAL))
        patcher.start()
        self.addCleanup(patcher.stop)
        self.keys = self.root / "env"
        self.write(self.keys, SECRET)

    def write(self, path, value, mode=0o600):
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as stream:
            stream.write("MDF_PORT=23465\nDATALAB_API_KEY=%s\n" % value)
        os.chmod(path, mode)

    def worker(self, keys):
        worker = Worker(self.settings, env_file=keys)
        self.addCleanup(worker.storage.close)
        return worker

    def checked(self, worker):
        """The readiness after a new check of the routes, as the heartbeat carries it."""
        worker._ready_at = None
        worker.ready()
        worker.heartbeat()
        with self.service.storage.connect() as conn:
            text = conn.execute("SELECT readiness FROM worker").fetchone()[0]
        self.assertNotIn(SECRET, text)
        return json.loads(text)

    def datalab_job(self):
        return self.service.submit(io.BytesIO(PDF), "p.pdf", {"route": "datalab", "transit_confirmed": True,
                                                              "datalab_mode": "lite"})

    def convert(self, worker):
        """One Datalab job through the worker: (the job, the calls of the engine)."""
        job, calls = self.datalab_job(), []
        with mock.patch("markdown_for_llms.worker.convert", side_effect=fake_convert(calls=calls)):
            worker.process(worker.claim())
        return self.service.get(job["id"]), calls

    def mount(self):
        """The keys file as the container sees it: the inode of the start, whatever the path holds afterwards."""
        fd = os.open(self.keys, os.O_RDONLY)
        self.addCleanup(os.close, fd)
        real = envfile._open
        patcher = mock.patch.object(envfile, "_open", lambda path: (
            os.open("/proc/self/fd/%d" % fd, os.O_RDONLY) if str(path) == MOUNT else real(path)))
        patcher.start()
        self.addCleanup(patcher.stop)
        return self.worker(MOUNT)


class UnsafeFileTests(Base):
    def test_the_worker_starts_and_only_the_routes_with_a_key_are_unavailable(self):
        link = self.root / "link"
        link.symlink_to(self.keys)
        loose = self.root / "loose"
        self.write(loose, SECRET, 0o644)
        for keys, code in ((loose, "datalab_key_unsafe"), (link, "datalab_key_unsafe"), (self.root / "absent", "datalab_key_unsafe"),
                           (None, "datalab_key_missing")):
            with self.subTest(keys=getattr(keys, "name", None)), mock.patch.dict(os.environ, {"DATALAB_API_KEY": SECRET}):
                worker = self.worker(keys)                              # it starts: nothing is raised
                self.assertNotIn("DATALAB_API_KEY", os.environ)         # and no key stays in its environment
                readiness = self.checked(worker)
                self.assertIs(readiness["ready"], True)
                self.assertEqual(readiness["converters"]["datalab"], {"available": False, "code": code})
                for name in LOCAL:
                    self.assertEqual(readiness["converters"][name], {"available": True, "code": None})
                self.assertIn("datalab (%s)" % code, readiness["message"])

    def test_a_local_conversion_goes_on_and_a_datalab_one_fails_with_the_code_of_the_route(self):
        self.write(self.keys, SECRET, 0o644)
        worker = self.worker(self.keys)
        self.checked(worker)
        refused, local = self.datalab_job(), self.service.submit(io.BytesIO(PDF), "local.pdf")
        calls = []
        with mock.patch("markdown_for_llms.worker.convert", side_effect=fake_convert(calls=calls)):
            worker.process(worker.claim())
            worker.process(worker.claim())
        self.assertEqual(len(calls), 1)                                 # nothing was asked of the engine for Datalab
        self.assertIsNone(calls[0]["datalab_key"])
        failed = self.service.get(refused["id"])
        self.assertEqual(failed["state"], "failed")
        self.assertIn("route_unavailable (datalab: datalab_key_unsafe)", failed["error"])
        self.assertEqual(self.service.get(local["id"])["state"], "completed")


class EveryUseTests(Base):
    def test_the_key_is_read_again_at_every_conversion(self):
        worker = self.worker(self.keys)
        self.checked(worker)
        done, calls = self.convert(worker)
        self.assertEqual((done["state"], calls[0]["datalab_key"].header()), ("completed", {"X-API-Key": SECRET}))
        # a chmod while the worker runs, with the routes not checked again yet: the conversion itself reads the file
        os.chmod(self.keys, 0o640)
        failed, calls = self.convert(worker)
        self.assertEqual((failed["state"], calls), ("failed", []))
        self.assertIn("datalab_key_unsafe", failed["error"])
        os.chmod(self.keys, 0o600)
        done, calls = self.convert(worker)
        self.assertEqual((done["state"], calls[0]["datalab_key"].header()), ("completed", {"X-API-Key": SECRET}))
        # a key changed in place (the same inode): the next conversion has the new one
        with open(self.keys, "w") as stream:
            stream.write("DATALAB_API_KEY=dl-second-synthetic\n")
        _, calls = self.convert(worker)
        self.assertEqual(calls[0]["datalab_key"].header(), {"X-API-Key": "dl-second-synthetic"})

    def test_no_key_is_kept_by_the_worker_or_written_in_the_archive(self):
        worker = self.worker(self.keys)
        self.checked(worker)
        self.convert(worker)
        self.assertNotIn(SECRET, repr(vars(worker)))
        for path in Path(self.settings.data_root).rglob("*"):
            if path.is_file():
                self.assertNotIn(SECRET.encode(), path.read_bytes(), path)


class ReplacedFileTests(Base):
    def test_a_file_replaced_while_the_worker_runs_gives_no_key_and_asks_for_a_restart(self):
        worker = self.mount()
        self.assertEqual(self.checked(worker)["converters"]["datalab"], {"available": True, "code": None})
        fresh = self.root / "env.tmp"
        self.write(fresh, "dl-new-synthetic")
        os.replace(fresh, self.keys)                                    # the atomic save of an editor, `sed -i`
        readiness = self.checked(worker)
        self.assertEqual(readiness["converters"]["datalab"], {"available": False, "code": "datalab_key_unsafe"})
        self.assertIn("make restart", readiness["message"])
        self.assertIs(readiness["ready"], True)                         # the local routes go on
        failed, calls = self.convert(worker)                            # neither the old key nor the new one is used
        self.assertEqual((failed["state"], calls), ("failed", []))
        self.assertIn("route_unavailable (datalab: datalab_key_unsafe)", failed["error"])
        # the restart: a new container mounts the file of the path
        again = self.worker(self.keys)
        self.assertEqual(self.checked(again)["converters"]["datalab"], {"available": True, "code": None})
        _, calls = self.convert(again)
        self.assertEqual(calls[0]["datalab_key"].header(), {"X-API-Key": "dl-new-synthetic"})

    def test_a_file_removed_while_the_worker_runs(self):
        worker = self.mount()
        self.checked(worker)
        self.keys.unlink()
        readiness = self.checked(worker)
        self.assertEqual(readiness["converters"]["datalab"], {"available": False, "code": "datalab_key_unsafe"})
        self.assertIn("make restart", readiness["message"])
        self.assertEqual(self.convert(worker)[1], [])

    def test_limit_an_old_file_kept_by_a_backup_is_still_read_until_the_restart(self):
        """E-1, the declared limit: the old inode still has a name, so the worker cannot tell. It ends with
        `make restart`."""
        worker = self.mount()
        self.checked(worker)
        os.link(self.keys, self.root / "env.bak")                       # a backup made with a hard link
        fresh = self.root / "env.tmp"
        self.write(fresh, "dl-new-synthetic")
        os.replace(fresh, self.keys)
        self.assertEqual(self.checked(worker)["converters"]["datalab"], {"available": True, "code": None})
        _, calls = self.convert(worker)
        self.assertEqual(calls[0]["datalab_key"].header(), {"X-API-Key": SECRET})            # the old key, until the restart
        _, calls = self.convert(self.worker(self.keys))
        self.assertEqual(calls[0]["datalab_key"].header(), {"X-API-Key": "dl-new-synthetic"})


if __name__ == "__main__":
    unittest.main()
