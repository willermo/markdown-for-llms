"""A2-c of MFL051: a child stopped by the launcher itself ends its job as `interrupted`.

When the launcher is stopped (the service going down, a restart) it kills and removes the children it started and
says why in `inspect` (`reaped: stop`). That is an interruption, not a failure of the engine: the closed code of
the conversion is `interrupted` and the job becomes `interrupted`, the terminal state of a worker that was stopped,
open to `mdf retry`. Nothing retries it by itself. The same for a child whose supervisor ended it because its
lease was not renewed (the worker was stuck). A stop the worker itself asked for keeps its own meaning, and a
child killed at its term is still a timeout.

The engine and the worker are the real ones; the launcher is scripted (no container).
"""
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

from markdown_for_llms.engine import supervisor
from markdown_for_llms.services import Service
from markdown_for_llms.settings import Settings
from markdown_for_llms.worker import Worker

from engine_fake import fake_convert, launcher_fake

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'unit' / 'engine'))
from engine_test_support import make_docx, para                    # noqa: E402

ROUTES = {name: {"available": True, "code": None} for name in ("text", "pandoc", "libreoffice", "djvutxt", "marker")}


class InterruptedTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.settings = Settings(data_root=self.tmp.name + '/data', model_root=self.tmp.name + '/models')
        self.service = Service(self.settings)
        self.addCleanup(self.service.storage.close)
        self.end = None                     # what the launcher says of the first child of the conversion
        self.launcher = launcher_fake.patched(self, Path(self.tmp.name) / 'volume', self.script)
        patcher = mock.patch('markdown_for_llms.worker.availability', lambda **kw: dict(ROUTES))
        patcher.start()
        self.addCleanup(patcher.stop)

    def script(self, request, call):
        reaped, status = self.end
        if reaped is not None:
            self.launcher.reaped[request["name"]] = reaped
        return status

    def convert(self, reaped, status, name='letter.docx'):
        self.end = (reaped, status)
        docx = make_docx(Path(self.tmp.name) / name, para('Text of the synthetic letter ' + name))
        job = self.service.submit(docx.open('rb'), name)
        worker = Worker(self.settings)
        self.addCleanup(worker.storage.close)
        worker.turn()
        row = self.service.storage.get(job['id'])
        engine = json.loads((self.service.storage.root / 'attempts' / row['attempt_id'] / 'engine-out' / 'engine.json').read_text())
        return row, engine['error']['code'], worker

    def test_a_child_the_launcher_stopped_makes_the_job_interrupted(self):
        row, code, worker = self.convert('stop', 137)
        self.assertEqual((code, row['state']), ('interrupted', 'interrupted'))
        self.assertIn('retry available', row['error'])
        self.assertNotIn('memory', row['error'])
        self.assertIsNone(row['token'])
        # terminal: the worker does not take it again by itself, and the user may retry it
        self.assertIsNone(worker.turn())
        self.assertEqual(self.service.storage.get(row['id'])['state'], 'interrupted')
        again = self.service.retry(row['id'])
        self.assertEqual((again['state'], again['id'] != row['id'] or again['attempt_id'] != row['attempt_id']), ('queued', True))

    def test_a_lease_that_was_not_renewed_is_an_interruption_too(self):
        for reaped, status in (('lease', 137), (None, supervisor.EXIT_LEASE)):
            with self.subTest(reaped=reaped, status=status):
                row, code, _ = self.convert(reaped, status, 'lease-%s.docx' % status)
                self.assertEqual((code, row['state']), ('interrupted', 'interrupted'))

    def test_a_term_is_still_a_timeout_and_a_kill_for_memory_a_memory_limit(self):
        for reaped, status, wanted in (('term', 137, 'timeout'), (None, supervisor.EXIT_TERM, 'timeout'), (None, 137, 'memory_limit')):
            with self.subTest(reaped=reaped, status=status):
                row, code, _ = self.convert(reaped, status, 'other-%s-%s.docx' % (reaped, status))
                self.assertEqual((code, row['state']), (wanted, 'failed'))

    def test_a_stop_the_worker_asked_for_keeps_its_own_meaning(self):
        """`interrupted` from the engine after the worker's own observer asked for it: the job follows the worker
        (`interrupted` when it is stopping, `cancelled` when the user asked), as before."""
        job = self.service.submit(io.BytesIO(b'%PDF-1.4\nsynthetic\n'), 'p.pdf')
        worker = Worker(self.settings)
        self.addCleanup(worker.storage.close)

        def stopping(observer):
            worker.stop = True
        engine = fake_convert(events=[{"event": "step", "step": "x"}, {"event": "tick", "module": "m", "container": "c"}],
                              before=stopping, code='interrupted')
        with mock.patch('markdown_for_llms.worker.convert', side_effect=engine):
            worker.process(worker.claim())
        self.assertEqual(self.service.storage.get(job['id'])['state'], 'interrupted')
        # and an engine that says `interrupted` when nobody of the worker asked, and the worker is not stopping
        other = self.service.submit(io.BytesIO(b'%PDF-1.4\nsynthetic 2\n'), 'q.pdf')
        worker = Worker(self.settings)
        self.addCleanup(worker.storage.close)
        with mock.patch('markdown_for_llms.worker.convert', side_effect=fake_convert(code='interrupted')):
            worker.process(worker.claim())
        row = self.service.storage.get(other['id'])
        self.assertEqual(row['state'], 'interrupted')
        self.assertIn('a child of the engine was stopped', row['error'])


if __name__ == '__main__':
    unittest.main()
