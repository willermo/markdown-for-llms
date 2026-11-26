"""Progress of the engine in the product (MFL032): the observer of the worker, persistence of the
snapshot, stall and deadline messages, progress view and CLI line. The events of the engine and
the isolation of its children are tested in tests/unit/engine; no ML import here."""
from dataclasses import replace
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from markdown_for_llms.api import create_app
from markdown_for_llms.cli import ProgressLine, main
from markdown_for_llms.engine.progress import Progress, finishing
from markdown_for_llms.files import read_json
from markdown_for_llms.services import Service, progress_view
from markdown_for_llms.settings import ConfigurationError, Settings
from markdown_for_llms.worker import Worker

from engine_fake import fake_convert

PDF = b'%PDF-1.4\nsynthetic progress contract\n'


def pages_progress(current, total):
    progress = Progress(wall=lambda: 1.0)
    progress.feed({"event": "models"})
    progress.feed({"event": "page", "page": current, "pages": total})
    return progress.snapshot()


class EngineProgressTests(unittest.TestCase):
    def test_engine_snapshot_is_a_valid_stored_progress(self):
        snapshot = pages_progress(5, 9)
        view = progress_view('running', json.dumps(snapshot), 900, now=1.0)
        self.assertEqual((view['stage'], view['current'], view['total']), ('pages', 5, 9))
        self.assertEqual(view['label'], 'Reading pages 5/9 (≈53%, estimate)')

    def test_finishing_snapshot(self):
        value = finishing(pages_progress(9, 9), wall=lambda: 5.0)
        self.assertEqual((value['stage'], value['estimate_percent'], value['phase'], value['updated'], value['last_advance']),
                         ('finishing', 95.0, 'Reading pages', 5.0, 1.0))
        self.assertEqual(finishing(None, wall=lambda: 5.0)['stage'], 'finishing')


class ProgressPersistenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.settings = Settings(data_root=self.tmp.name + '/data', model_root=self.tmp.name + '/models')
        self.service = Service(self.settings)
        self.addCleanup(self.service.storage.close)

    def stored(self, run_id, column='progress'):
        with self.service.storage.connect() as conn:
            value = conn.execute('SELECT %s FROM runs WHERE id=?' % column, (run_id,)).fetchone()[0]
        return value and json.loads(value)

    def test_tick_without_snapshot_keeps_the_last_phase(self):
        job = self.service.submit(io.BytesIO(PDF), 'p.pdf')
        worker = Worker(self.settings)
        self.addCleanup(worker.storage.close)
        row = worker.claim()
        self.assertFalse(worker.tick(row, None, pages_progress(3, 4)))
        self.assertEqual(self.stored(job['id'])['current'], 3)
        self.assertFalse(worker.tick(row, None, finishing(pages_progress(3, 4))))
        self.assertFalse(worker.tick(row, None))
        stored = self.stored(job['id'])
        self.assertEqual((stored['stage'], stored['phase'], stored['current']), ('finishing', 'Reading pages', 3))

    def test_conversion_on_the_engine_records_child_progress_and_finishing(self):
        job = self.service.submit(io.BytesIO(PDF), 'p.pdf')
        calls, seen = [], []
        events = [{"event": "child", "module": "markdown_for_llms.engine.marker_child", "container": "mdf-eng-x-marker-1"},
                  {"event": "progress", "progress": pages_progress(9, 9)}]
        engine = fake_convert(events=events, calls=calls,
                              before=lambda observer: seen.append(self.stored(job['id'], 'child')))
        worker = Worker(self.settings)
        self.addCleanup(worker.storage.close)
        with patch('markdown_for_llms.worker.convert', side_effect=engine):
            worker.process(worker.claim())
        self.assertEqual(self.service.get(job['id'])['state'], 'completed')
        call = calls[0]
        self.assertEqual({k: call['limits'][k] for k in ('stall_seconds', 'timeout_seconds', 'max_pages', 'memory_mib')},
                         {'stall_seconds': 900, 'timeout_seconds': 0, 'max_pages': 50, 'memory_mib': 16384})
        row = self.service.storage.get(job['id'])
        self.assertEqual(call['request'], {'route': None, 'ocr': 'auto', 'lang': None, 'attempt': row['attempt_id']})
        self.assertEqual((call['name'], call['model_root']), ('p.pdf', self.settings.model_root))
        self.assertEqual(seen, [{'container': 'mdf-eng-x-marker-1'}])                            # K7
        stored = self.stored(job['id'])
        self.assertEqual((stored['stage'], stored['estimate_percent'], stored['current']), ('finishing', 95.0, 9))
        result = self.service.storage.root / row['result_path']
        self.assertTrue((result / 'engine-result.json').is_file())                                   # K9
        self.assertTrue((result / 'diagnostics' / 'marker-1.log').is_file())
        self.assertEqual(sorted(p.name for p in (result / 'original').iterdir()), ['document.md'])

    def test_force_ocr_is_ocr_force_and_strip_is_dropped(self):
        # a run queued by a previous version, with the options of schema 2 (Marker 1)
        job = self.service.submit(io.BytesIO(PDF), 'p.pdf')
        v2 = dict(use_llm=False, force_ocr=True, strip_existing_ocr=True, disable_image_extraction=False,
                  paginate_output=True, pdftext_workers=1, layout_batch_size=1, recognition_batch_size=1,
                  detection_batch_size=1, table_rec_batch_size=1, ocr_error_batch_size=1,
                  lowres_image_dpi=96, highres_image_dpi=192)
        with self.service.storage.connect() as conn:
            conn.execute('UPDATE runs SET options=? WHERE id=?', (json.dumps(v2), job['id']))
        calls = []
        worker = Worker(self.settings)
        self.addCleanup(worker.storage.close)
        row = worker.claim()
        with patch('markdown_for_llms.worker.convert', side_effect=fake_convert(calls=calls)):
            worker.process(row)
        self.assertEqual(calls[0]['request']['ocr'], 'force')
        stored = read_json(self.service.storage.root / self.service.storage.get(job['id'])['result_path'], 'request.json')
        self.assertEqual(stored['dropped_options'], ['strip_existing_ocr'])

    def test_stalled_and_timeout_messages(self):
        for code, settings, text in [('stalled', self.settings, 'No conversion progress for 900 s (MDF_STALL_SECONDS)'),
                                     ('timeout', replace(self.settings, timeout_seconds=7200), 'Conversion exceeded 7200 s (MDF_TIMEOUT_SECONDS)')]:
            with self.subTest(code=code):
                job = self.service.submit(io.BytesIO(PDF), 'p.pdf')
                worker = Worker(settings)
                self.addCleanup(worker.storage.close)
                calls = []
                with patch('markdown_for_llms.worker.convert', side_effect=fake_convert(code=code, calls=calls)):
                    worker.process(worker.claim())
                result = self.service.get(job['id'])
                self.assertEqual(result['state'], 'failed')
                self.assertTrue(result['error'].startswith(text), result['error'])
                self.assertIn('source kept, retry available', result['error'])
                self.assertEqual(calls[0]['limits']['timeout_seconds'], settings.timeout_seconds)

    def test_an_error_of_the_lease_stops_the_engine_and_is_raised(self):                           # K7
        from markdown_for_llms.storage import CapacityError
        job = self.service.submit(io.BytesIO(PDF), 'p.pdf')
        worker = Worker(self.settings)
        self.addCleanup(worker.storage.close)
        row = worker.claim()
        results = []
        engine = fake_convert(events=[{"event": "tick"}, {"event": "tick"}], calls=results)
        with patch.object(worker, 'tick', side_effect=CapacityError('Operation exceeded its space commitment')), \
                patch('markdown_for_llms.worker.convert', side_effect=engine):
            worker.process(row)
        result = self.service.get(job['id'])
        self.assertEqual(result['state'], 'failed')
        self.assertIn('space commitment', result['error'])


class ProgressViewTests(unittest.TestCase):
    SNAPSHOT = dict(stage='pages', estimate_percent=42.7, phase='Reading pages', current=12, total=47,
                    updated=1000.0, last_advance=990.0)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.settings = Settings(data_root=self.tmp.name + '/data', model_root=self.tmp.name + '/models')
        self.service = Service(self.settings)
        self.addCleanup(self.service.storage.close)

    def set_run(self, run_id, state, progress):
        with self.service.storage.connect() as conn:
            conn.execute('UPDATE runs SET state=?,progress=? WHERE id=?', (state, progress, run_id))

    def test_progress_is_separate_from_state_and_never_an_error(self):
        job = self.service.submit(io.BytesIO(PDF), 'p.pdf')
        self.assertIsNone(job['progress'])
        self.set_run(job['id'], 'running', None)
        starting = self.service.get(job['id'])['progress']
        self.assertEqual((starting['stage'], starting['live'], starting['label'], starting['idle_seconds']),
                         ('starting', True, 'Starting engine (≈0%, estimate)', None))
        self.set_run(job['id'], 'running', json.dumps(self.SNAPSHOT))
        with patch('markdown_for_llms.services.time.time', return_value=1003.0):
            running = self.service.get(job['id'])
        self.assertEqual(running['progress'], dict(estimate_percent=42.7, estimated=True, stage='pages',
                         phase='Reading pages', current=12, total=47, last_advance=990.0, idle_seconds=13.0,
                         stall_seconds=900, live=True, label='Reading pages 12/47 (≈42%, estimate)'))
        base = {k: running[k] for k in ('state', 'bundle', 'quality', 'message')}
        for state, label in [('failed', 'Stopped during Reading pages 12/47 (≈42%, estimate)'),
                             ('cancelled', 'Stopped during Reading pages 12/47 (≈42%, estimate)')]:
            self.set_run(job['id'], state, json.dumps(self.SNAPSHOT))
            stopped = self.service.get(job['id'])['progress']
            self.assertEqual((stopped['live'], stopped['label'], stopped['idle_seconds']), (False, label, None))
        self.set_run(job['id'], 'interrupted', json.dumps(finishing(self.SNAPSHOT)))
        self.assertEqual(self.service.get(job['id'])['progress']['label'], 'Stopped while finishing (≈95%, estimate)')
        self.set_run(job['id'], 'completed', json.dumps(self.SNAPSHOT))
        self.assertIsNone(self.service.get(job['id'])['progress'])
        for stored in ['{broken', '[]', json.dumps(dict(self.SNAPSHOT, stage='unknown')),
                       json.dumps(dict(self.SNAPSHOT, estimate_percent='42')), json.dumps(dict(self.SNAPSHOT, current=True)),
                       json.dumps(dict(self.SNAPSHOT, estimate_percent=101)), '{"stage": "pages", "estimate_percent": NaN}',
                       json.dumps({k: v for k, v in self.SNAPSHOT.items() if k != 'phase'}),
                       json.dumps({k: v for k, v in self.SNAPSHOT.items() if k not in ('current', 'last_advance')}),
                       json.dumps(dict(self.SNAPSHOT, stage=['pages'])), json.dumps(dict(self.SNAPSHOT, last_advance=float('nan'))),
                       json.dumps(dict(self.SNAPSHOT, updated=float('inf')))]:
            with self.subTest(stored=stored):
                self.set_run(job['id'], 'running', stored)
                result = self.service.get(job['id'])
                self.assertIsNone(result['progress'])
                self.assertEqual({k: result[k] for k in base}, base)
                self.assertEqual([row['progress'] for row in self.service.list()], [None])
        self.set_run(job['id'], 'failed', None)
        self.assertIsNone(self.service.get(job['id'])['progress'])

    def test_api_lists_and_shows_progress(self):
        job = self.service.submit(io.BytesIO(PDF), 'p.pdf')
        self.set_run(job['id'], 'running', json.dumps(self.SNAPSHOT))
        client = TestClient(create_app(self.settings), base_url=f'http://127.0.0.1:{self.settings.port}')
        self.assertEqual(client.get('/api/v1/runs/' + job['id']).json()['progress']['phase'], 'Reading pages')
        self.assertEqual(client.get('/api/v1/runs').json()[0]['progress']['current'], 12)

    def test_cli_progress_line_only_on_a_terminal(self):
        class Stream(io.StringIO):
            def __init__(self, tty):
                super().__init__()
                self.tty = tty
            def isatty(self):
                return self.tty
        running = dict(id='r', state='running', bundle='pending',
                       progress=progress_view('running', json.dumps(self.SNAPSHOT), 900, now=1000.0))
        done = dict(id='r', state='completed', bundle='ready', progress=None)
        class FakeService:
            def __init__(self, settings):
                self.states = [running, running, done]
            def submit(self, stream, name, overrides):
                return dict(id='r', state='queued', bundle='pending', progress=None)
            def get(self, run_id):
                return self.states.pop(0)
        source = Path(self.tmp.name) / 'p.pdf'
        source.write_bytes(PDF)
        for tty in (False, True):
            with self.subTest(tty=tty):
                out, err = io.StringIO(), Stream(tty)
                with patch('markdown_for_llms.cli.Service', FakeService), patch('markdown_for_llms.cli.time.sleep'), \
                        patch.object(sys, 'stdout', out), patch.object(sys, 'stderr', err):
                    code = main(['--data-root', self.settings.data_root, '--model-root', self.settings.model_root,
                                 'submit', '--wait', str(source)])
                self.assertEqual(code, 0)
                self.assertEqual(json.loads(out.getvalue())['state'], 'completed')
                if tty:
                    self.assertIn('\r[########------------] Reading pages 12/47 (≈42%, estimate)', err.getvalue())
                    self.assertTrue(err.getvalue().endswith('\n'))
                else:
                    self.assertEqual(err.getvalue(), '')
        line = ProgressLine(Stream(True))
        line.show(dict(progress=None))
        self.assertEqual(line.stream.getvalue(), '')


class SettingsTests(unittest.TestCase):
    def test_time_limit_settings(self):
        settings = Settings.load(environ={})
        self.assertEqual((settings.timeout_seconds, settings.stall_seconds), (0, 900))
        self.assertEqual(Settings.load(environ={'MDF_TIMEOUT_SECONDS': '0', 'MDF_STALL_SECONDS': '60'}).stall_seconds, 60)
        for values in [{'MDF_TIMEOUT_SECONDS': '-1'}, {'MDF_STALL_SECONDS': '59'}, {'MDF_STALL_SECONDS': '0'}]:
            with self.subTest(values=values), self.assertRaises(ConfigurationError):
                Settings.load(environ=values)
        changed = replace(settings, timeout_seconds=600, stall_seconds=1200)
        self.assertEqual(changed.identity(), settings.identity())
        # schema 3 (MFL032 C): the port, the model root and the Datalab cap are operational too
        for other in (dict(port=23460), dict(model_root='/elsewhere/models'), dict(datalab_cap_usd='0.25')):
            with self.subTest(other=other):
                self.assertEqual(replace(settings, **other).identity(), settings.identity())
        self.assertNotEqual(replace(settings, memory_mib=8192).identity(), settings.identity())
        from markdown_for_llms import legacy_identity
        root = settings.model_root
        self.assertEqual(legacy_identity.v1(replace(settings, timeout_seconds=450), root), legacy_identity.v1(settings, root))
        self.assertEqual((changed.public()['timeout_seconds'], changed.public()['stall_seconds']), (600, 1200))


if __name__ == '__main__':
    unittest.main()
