"""B1 of MFL051 (SOL-R2-B1): the end of a child is not attested while the worker lives.

The launcher, or the daemon behind it, stops answering in the middle of a conversion. The engine gives the launcher
its twenty seconds, writes the stop its supervisor reads, keeps the folder of the call and ends the step with
`isolation_unavailable`. The job gets its terminal state as always; what must NOT happen is the release of the
space commitment of the attempt, and a new job taken, before Docker attests that no container of the attempt is
left. The durable condition is the record of that commitment in `operations/`, which is there from before the first
child: it is kept, the worker claims nothing while it is, says so in its heartbeat, asks again, and releases it only
after the attestation; a worker that restarts meanwhile finds the record and starts from it, whatever the state of
the job.

The launcher here is scripted (no container); the engine, the worker, the storage and the capacity are the real
ones. With a real launcher killed and held down: `tests/container/test_child_stop.py`.
"""
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

from markdown_for_llms import capacity
from markdown_for_llms.engine import containers
from markdown_for_llms.services import Service
from markdown_for_llms.settings import Settings
from markdown_for_llms.storage import StorageError
from markdown_for_llms.worker import Unattested, Worker

from engine_fake import fake_convert, launcher_fake

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'unit' / 'engine'))
from engine_test_support import make_docx, para                    # noqa: E402

PDF = b'%PDF-1.4\nsynthetic unattested contract\n'
ROUTES = {name: {"available": True, "code": None} for name in ("text", "pandoc", "libreoffice", "djvutxt", "marker")}


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.volume = Path(self.tmp.name) / 'volume'
        self.settings = Settings(data_root=self.tmp.name + '/data', model_root=self.tmp.name + '/models')
        self.service = Service(self.settings)
        self.addCleanup(self.service.storage.close)
        self.launcher = launcher_fake.patched(self, self.volume, self.script)
        for target, value in (('markdown_for_llms.worker.availability', lambda **kw: dict(ROUTES)),
                              ('markdown_for_llms.worker.KEPT_SECONDS', 0.0)):
            patcher = mock.patch(target, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.released = []                  # the children the launcher listed each time a commitment was released
        real = capacity.cleanup

        def cleanup(storage, record):
            if record['kind'] == 'worker':
                self.released.append((record['token'], sorted(self.launcher.children)))
            return real(storage, record)
        patcher = mock.patch.object(capacity, 'cleanup', cleanup)
        patcher.start()
        self.addCleanup(patcher.stop)

    def script(self, request, call):
        return 0

    def worker(self):
        worker = Worker(self.settings)
        self.addCleanup(worker.storage.close)
        self.addCleanup(lambda: [lease.abandon() for lease in worker.kept.values()])
        return worker

    def records(self):
        operations = self.service.storage.root / 'operations'
        return sorted(json.loads(p.read_text())['token'] for p in operations.glob('*.json'))

    def readiness(self):
        with self.service.storage.connect() as conn:
            return json.loads(conn.execute('SELECT readiness FROM worker').fetchone()[0])

    def state(self, job):
        return self.service.get(job['id'])['state']

    def calls_left(self):
        calls = self.volume / 'call'
        return sorted(os.listdir(calls)) if calls.is_dir() else []


class EngineTests(Base):
    """The whole path with the real engine: the launcher goes away while the first child of a conversion runs."""

    def script(self, request, call):
        self.launcher.down = True           # the request was accepted, the child runs; then nobody answers
        return None

    def test_the_launcher_goes_away_during_a_conversion(self):
        docx = make_docx(Path(self.tmp.name) / 'letter.docx', para('Text of the synthetic letter'))
        job = self.service.submit(docx.open('rb'), 'letter.docx')
        second = self.service.submit(io.BytesIO(PDF), 'second.pdf')
        worker = self.worker()
        with mock.patch.object(containers, 'UNREACHABLE_SECONDS', 0.3):        # its twenty seconds, shortened
            self.assertEqual(worker.turn()['id'], job['id'])
        row = self.service.storage.get(job['id'])
        attempt, prefix = row['attempt_id'], containers.prefix_of(row['attempt_id'])
        # the job has its terminal state, with the closed code, and may be retried
        self.assertEqual(row['state'], 'failed')
        self.assertIn('Conversion failed: isolation_unavailable', row['error'])
        self.assertIn('retry available', row['error'])
        engine = json.loads((self.service.storage.root / 'attempts' / attempt / 'engine-out' / 'engine.json').read_text())
        self.assertEqual(engine['error']['code'], 'isolation_unavailable')
        # the child was asked to stop without Docker, and the folder of its call stays where it is
        (name,) = self.calls_left()
        self.assertTrue(name.startswith(prefix + '-light-'))
        self.assertTrue((self.volume / 'call' / name / 'ctl' / 'stop').exists())
        # the commitment of the attempt is still there, and its budget still counts
        self.assertEqual((self.records(), sorted(worker.kept)), ([attempt], [attempt]))
        self.assertEqual(self.released, [])
        with self.service.storage.lock('admission.lock'):
            self.assertEqual(capacity.live_workers(self.service.storage), {attempt})
        # no job is taken for as many turns as the launcher stays away, and the heartbeat says why
        for _ in range(3):
            self.assertIsNone(worker.turn())
            self.assertEqual(self.state(second), 'queued')
        beat = self.readiness()
        self.assertEqual((beat['ready'], beat['code'], beat['pending']), (False, 'isolation_unavailable', [prefix]))
        self.assertIn('make restart', beat['message'])
        # (a) the launcher comes back with the container still listed: kill, remove, the list empty, and ONLY THEN
        # the record goes; the folder of the call goes with it, and the second job starts
        self.launcher.down = False
        self.assertEqual(sorted(self.launcher.children), [name])
        with mock.patch('markdown_for_llms.worker.convert', side_effect=fake_convert()):
            self.assertEqual(worker.turn()['id'], second['id'])
        self.assertEqual(self.launcher.removed, [name])
        self.assertEqual(self.released[0], (attempt, []))            # at the release Docker listed no child of it
        self.assertEqual((self.calls_left(), worker.kept), ([], {}))
        self.assertEqual(self.state(second), 'completed')
        self.assertEqual(self.records(), [])
        self.assertTrue(self.readiness()['ready'])


class WorkerTests(Base):
    """The release rule by itself, with the engine replaced: what it ended with does not matter to the rule."""

    def unattested(self, engine=None, name='p.pdf'):
        """A job processed while the launcher does not answer; returns (job, worker, attempt)."""
        job = self.service.submit(io.BytesIO(PDF + name.encode()), name)
        worker = self.worker()
        self.launcher.down = True
        with mock.patch('markdown_for_llms.worker.convert', side_effect=engine or fake_convert(code='isolation_unavailable')):
            self.assertEqual(worker.turn()['id'], job['id'])
        return job, worker, self.service.storage.get(job['id'])['attempt_id']

    def left(self, attempt, kind='marker', n=1):
        name = containers.child_name(attempt, kind, n)
        self.launcher.children[name] = {"state": "running", "exit": 0, "oom_killed": False}
        (self.volume / 'call' / name / 'ctl').mkdir(parents=True)
        (self.volume / 'call' / name / 'owner').write_text('')
        return name

    def test_a_restart_starts_from_the_record_whatever_the_state_of_the_job(self):
        """(b) The worker dies with the commitment kept and the launcher still away. The next worker finds the job
        already `failed` and does not skip it: it attests nothing, releases nothing and takes no job; once the
        launcher is back it attests, the folder of the call goes, the commitment is released, the queue goes on."""
        job, worker, attempt = self.unattested()
        second = self.service.submit(io.BytesIO(PDF + b'2'), 'second.pdf')
        name = self.left(attempt)
        self.assertEqual((self.state(job), self.records()), ('failed', [attempt]))
        worker.kept[attempt].abandon()                               # the death of the worker: its lock goes, the record stays
        with self.service.storage.lock('admission.lock'):
            self.assertEqual(capacity.dead_workers(self.service.storage), {attempt})
        restarted = self.worker()
        for _ in range(2):
            with self.assertRaises(Unattested):
                restarted.reconcile()
            self.assertEqual((self.records(), self.state(second), self.state(job)), ([attempt], 'queued', 'failed'))
        with self.assertRaisesRegex(StorageError, 'not attested absent'):
            restarted.run(once=True)                                 # the service ends, compose starts it again
        beat = self.readiness()
        self.assertEqual((beat['ready'], beat['code']), (False, 'isolation_unavailable'))
        self.assertEqual((self.records(), self.released, self.state(second)), ([attempt], [], 'queued'))
        self.launcher.down = False
        restarted.reconcile()
        self.assertEqual(self.launcher.removed, [name])
        self.assertEqual(self.released, [(attempt, [])])
        self.assertEqual((self.records(), self.calls_left()), ([], []))
        with mock.patch('markdown_for_llms.worker.convert', side_effect=fake_convert()):
            self.assertEqual(restarted.turn()['id'], second['id'])
        self.assertEqual(self.state(second), 'completed')

    def test_an_attestation_missing_after_a_conversion_that_succeeded_keeps_the_commitment_too(self):
        job, worker, attempt = self.unattested(engine=fake_convert())
        second = self.service.submit(io.BytesIO(PDF + b'2'), 'second.pdf')
        self.assertEqual(self.state(job), 'completed')               # the outcome of the job is its own
        self.assertEqual((self.records(), sorted(worker.kept)), ([attempt], [attempt]))
        self.assertIsNone(worker.turn())
        self.assertEqual(self.state(second), 'queued')
        self.launcher.down = False
        with mock.patch('markdown_for_llms.worker.convert', side_effect=fake_convert()):
            self.assertEqual(worker.turn()['id'], second['id'])
        self.assertEqual((self.records(), self.released[0]), ([], (attempt, [])))

    def test_a_removal_that_is_not_confirmed_keeps_the_commitment_like_a_launcher_that_is_away(self):
        job, worker, attempt = self.unattested()
        name = self.left(attempt, 'tool')
        self.launcher.down = False
        self.launcher.unconfirmed = {name}
        self.assertIsNone(worker.turn())
        self.assertEqual((self.records(), self.released), ([attempt], []))
        self.launcher.unconfirmed = set()
        self.assertTrue(worker.release_kept())
        self.assertEqual((self.records(), self.released), ([], [(attempt, [])]))

    def test_the_attestation_is_asked_again_only_every_few_seconds(self):
        job, worker, attempt = self.unattested()
        with mock.patch('markdown_for_llms.worker.KEPT_SECONDS', 5.0):
            asked = len(self.launcher.ops('list'))
            self.assertFalse(worker.release_kept(now=100.0))
            self.assertFalse(worker.release_kept(now=102.0))         # too soon: the launcher is not asked
            self.assertFalse(worker.release_kept(now=104.9))
            self.assertEqual(len(self.launcher.ops('list')) - asked, 1)
            self.launcher.down = False
            self.assertFalse(worker.release_kept(now=104.95))
            self.assertTrue(worker.release_kept(now=105.1))
        self.assertEqual(self.records(), [])

    def test_a_gpu_session_still_closes_while_the_worker_waits(self):
        """`gpu.poll` runs at every turn, also while no job is taken: the spending of a session does not wait for
        the attestation."""
        job, worker, attempt = self.unattested()
        worker.gpu = mock.Mock()
        self.assertIsNone(worker.turn())
        self.assertIsNone(worker.turn())
        self.assertEqual(worker.gpu.poll.call_count, 2)
        worker.gpu.claim_preset.assert_not_called()

    def test_a_worker_that_ends_leaves_the_record_for_the_next_one(self):
        job, worker, attempt = self.unattested()
        worker.gpu = mock.Mock()
        self.launcher.down = False                                   # for the start of `run`: its reconciliation and its probe
        with mock.patch.object(worker, 'probe', return_value=[]), mock.patch.object(worker, 'release_kept', return_value=False):
            worker.run(once=True)
        self.assertEqual(self.records(), [attempt])
        self.assertIsNone(worker.kept[attempt].fd)
        with self.service.storage.lock('admission.lock'):
            self.assertEqual(capacity.dead_workers(self.service.storage), {attempt})

    def test_the_ordinary_end_asks_for_the_attestation_and_releases(self):
        job = self.service.submit(io.BytesIO(PDF), 'p.pdf')
        worker = self.worker()
        with mock.patch('markdown_for_llms.worker.convert', side_effect=fake_convert()):
            worker.turn()
        attempt = self.service.storage.get(job['id'])['attempt_id']
        self.assertEqual(self.launcher.ops('list'), [('list', containers.prefix_of(attempt))] * 2)     # before and after: empty
        self.assertEqual((self.records(), worker.kept, self.released), ([], {}, [(attempt, [])]))


if __name__ == '__main__':
    unittest.main()
