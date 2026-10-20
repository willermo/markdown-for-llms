"""Engine progress parser, stall guard, optional deadline and progress persistence (ADR 0011); no ML imports."""
from dataclasses import replace
from importlib import metadata
import io
import json
from pathlib import Path
import random
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from tqdm import tqdm

from markdown_for_llms import child
from markdown_for_llms.api import create_app
from markdown_for_llms.cli import ProgressLine, main
from markdown_for_llms.files import identify, write, write_json
from markdown_for_llms.progress import MAX_SEGMENT, Progress, finishing
from markdown_for_llms.runtime import PINS
from markdown_for_llms.sandbox import execute
from markdown_for_llms.services import Service, progress_view
from markdown_for_llms.settings import ConfigurationError, Settings
from markdown_for_llms.worker import Worker

PDF = b'%PDF-1.4\nsynthetic progress contract\n'
# Bar sequence of the pinned Marker/Surya path (counts are synthetic, not from a private log).
SEQUENCE = [('Recognizing Layout', 3), ('Running OCR Error Detection', 3), ('Detecting bboxes', 2),
            ('Recognizing Text', 5), ('Detecting bboxes', 1), ('Recognizing Text', 4), ('Recognizing tables', 2)]


class Recorder(Progress):
    def __init__(self, *args, **kwargs):
        self.log = []
        super().__init__(*args, **kwargs)

    def observe(self, desc, n, total):
        advanced = super().observe(desc, n, total)
        if advanced:
            self.log.append((self.stage, desc, n, total, round(self.estimate, 3)))
        return advanced


def tqdm_output(sequence):
    stream = io.StringIO()
    for desc, total in sequence:
        for _ in tqdm(range(total), desc=desc, file=stream, mininterval=0, miniters=1):
            pass
    return stream.getvalue().encode()


def bar(desc, n, total):
    return '\r%s: %3d%%|%s| %d/%d [00:01<00:01, 1.00it/s]' % (desc, 100 * n // total, '#' * (10 * n // total), n, total)


class ProgressParserTests(unittest.TestCase):
    def test_real_tqdm_output_in_any_chunking(self):
        data = tqdm_output(SEQUENCE)
        whole = Recorder()
        whole.feed(data)
        self.assertEqual(len(whole.log), sum(total + 1 for _, total in SEQUENCE))
        self.assertEqual(whole.unparsed, 0)
        stages = [entry[0] for entry in whole.log if entry[2] == 0]
        self.assertEqual(stages, ['layout', 'ocr-error', 'detection', 'text', 'processors', 'processors', 'processors'])
        estimates = [entry[4] for entry in whole.log]
        self.assertEqual(estimates, sorted(estimates))
        self.assertLessEqual(estimates[-1], 95)
        rng = random.Random(13)
        for _ in range(20):
            chunked, offset = Recorder(), 0
            while offset < len(data):
                size = rng.randint(1, 17)
                chunked.feed(data[offset:offset + size])
                offset += size
            self.assertEqual(chunked.log, whole.log)
            self.assertEqual(chunked.unparsed, 0)

    def test_bar_is_read_when_its_closing_bracket_arrives(self):
        progress = Recorder()
        progress.feed(b'\rRecognizing Layout:   0%|          | 0/11 [00:00<?, ?it/s]')
        self.assertEqual(progress.log, [('layout', 'Recognizing Layout', 0, 11, 5.0)])
        first, second = bar('Recognizing Layout', 1, 11).encode()[:30], bar('Recognizing Layout', 1, 11).encode()[30:]
        progress.feed(first)
        self.assertEqual(len(progress.log), 1)
        progress.feed(second)
        self.assertEqual(progress.log[-1][2], 1)
        progress.feed(b'   ')                        # tqdm pads a shorter redraw with spaces
        self.assertEqual((len(progress.log), progress.unparsed), (2, 0))

    def test_utf8_bar_split_inside_a_character(self):
        data = '\rRecognizing Text:  50%|█████     | 1/2 [00:01<00:01, 1.00it/s]'.encode()
        cut = data.index('█'.encode()) + 1
        progress = Recorder()
        progress.feed(data[:cut])
        progress.feed(data[cut:])
        self.assertEqual(progress.log, [('text', 'Recognizing Text', 1, 2, 62.5)])

    def test_redraw_is_not_an_advance_and_idle_grows(self):
        now = [100.0]
        progress = Recorder(clock=lambda: now[0])
        progress.feed(bar('Recognizing Text', 1, 4).encode())
        for step in range(5):
            now[0] += 10
            progress.feed(bar('Recognizing Text', 1, 4).encode())
        self.assertEqual((progress.events, progress.idle()), (1, 50.0))
        progress.feed(bar('Recognizing Text', 2, 4).encode())
        self.assertEqual((progress.events, progress.idle(), progress.receipt()['max_silence_seconds']), (2, 0.0, 50.0))

    def test_unrecognized_input_is_counted_and_resynchronized(self):
        progress = Recorder()
        progress.feed(b'\xff\xfe not utf-8\nWARNING: something\n')
        progress.feed(b'x' * (MAX_SEGMENT + 1))
        progress.feed(b'y' * 100)                     # still the same oversized segment
        progress.feed(b'z\r' + bar('Recognizing Layout', 1, 2).encode())
        self.assertEqual(progress.unparsed, 3)
        self.assertEqual(progress.log, [('layout', 'Recognizing Layout', 1, 2, 20.0)])
        progress.feed(b'\rRecognizing Layout:  50%|#   | 3/2 [00:01<00:01, 1.00it/s]')
        progress.feed(b'\rRecognizing Layout:  50%|#   | 1/20000000 [00:01<00:01, 1.00it/s]')
        self.assertEqual((progress.unparsed, progress.events), (5, 1))

    def test_unknown_bar_restart_and_missing_phase_keep_the_estimate_monotone(self):
        progress = Recorder()
        for desc, n, total in [('Recognizing Layout', 2, 2), ('Some new step', 0, 3), ('Some new step', 1, 3),
                               ('Recognizing Text', 3, 4), ('Recognizing Text', 0, 4), ('Recognizing Layout', 0, 1),
                               ('Recognizing Text', 1, 9), ('Recognizing Text', 9, 9), ('Recognizing Text', 0, 24)]:
            progress.feed(bar(desc, n, total).encode())
        # Layout -> text skips two phases; a restarted or unfinished text bar stays in text;
        # layout never comes back; a text bar after a finished one belongs to the processors.
        self.assertEqual([entry[0] for entry in progress.log],
                         ['layout', 'layout', 'layout', 'text', 'text', 'text', 'text', 'text', 'processors'])
        estimates = [entry[4] for entry in progress.log]
        self.assertEqual(estimates, sorted(estimates))

    def test_finishing_snapshot(self):
        value = finishing({'stage': 'processors', 'estimate_percent': 90.0, 'phase': 'Recognizing Text',
                           'current': 24, 'total': 24, 'updated': 1.0, 'last_advance': 1.0}, wall=lambda: 5.0)
        self.assertEqual((value['stage'], value['estimate_percent'], value['phase'], value['updated'], value['last_advance']),
                         ('finishing', 95.0, 'Recognizing Text', 5.0, 1.0))
        self.assertEqual(finishing(None, wall=lambda: 5.0)['stage'], 'finishing')


class ExecuteLimitTests(unittest.TestCase):
    """execute() with the Bubblewrap command replaced by a plain child; the real sandbox is proved in M6."""
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.settings = Settings(data_root=self.tmp.name + '/data', model_root=self.tmp.name + '/models')
        self.count = 0

    def run_child(self, script, **kwargs):
        self.count += 1
        scratch = Path(self.tmp.name) / ('scratch-%d' % self.count)
        scratch.mkdir()
        seen = []
        argv = [sys.executable, '-I', '-B', '-c', script]
        with patch('markdown_for_llms.sandbox.command', return_value=argv):
            receipt = execute(self.settings, scratch, tick=lambda child, progress: seen.append(progress) or False, **kwargs)
        return receipt, seen, json.loads((scratch / 'process-limits.json').read_text())

    def test_stalled_child_is_stopped(self):
        script = ('import sys, time\n'
                  'line = lambda n: "\\rRecognizing Layout: %3d%%|#   | %d/3 [00:01<00:01, 1.00it/s]" % (33 * n, n)\n'
                  'for n in range(3):\n    sys.stderr.write(line(n)); sys.stderr.flush(); time.sleep(0.1)\n'
                  'while True:\n    sys.stderr.write(line(2)); sys.stderr.flush(); time.sleep(0.1)\n')
        started = time.monotonic()
        receipt, seen, limits = self.run_child(script, stall=1)
        self.assertEqual(receipt['reason'], 'stalled')
        self.assertLess(time.monotonic() - started, 10)
        self.assertNotIn('cpu_seconds', limits)
        self.assertEqual((receipt['progress']['phase'], receipt['progress']['current'], receipt['progress_events']),
                         ('Recognizing Layout', 2, 3))
        self.assertEqual((receipt['stall_seconds'], receipt['deadline_seconds']), (1, None))
        self.assertGreater(receipt['max_silence_seconds'], 1)
        self.assertTrue(receipt['children_reaped'])
        self.assertFalse(Path('/proc/%d' % receipt['process']['pid']).exists())
        self.assertEqual(seen[-1]['current'], 2)

    def test_advancing_child_has_no_total_deadline(self):
        script = ('import sys, time\n'
                  'for n in range(7):\n'
                  '    sys.stderr.write("\\rRecognizing Text: %3d%%|#   | %d/6 [00:01<00:01, 1.00it/s]" % (16 * n, n)); sys.stderr.flush(); time.sleep(0.4)\n')
        receipt, seen, limits = self.run_child(script, stall=1)
        self.assertEqual((receipt['reason'], receipt['exit'], receipt['progress_events']), (None, 0, 7))
        self.assertGreater(receipt['elapsed_seconds'], 2.5)
        self.assertLess(receipt['max_silence_seconds'], 1)
        self.assertNotIn('cpu_seconds', limits)

    def test_explicit_deadline_without_stall_guard(self):
        receipt, seen, limits = self.run_child('import time; time.sleep(30)', deadline=1)
        self.assertEqual(receipt['reason'], 'timeout')
        self.assertEqual(limits['cpu_seconds'], self.settings.threads)
        self.assertNotIn('progress', receipt)
        self.assertEqual(set(seen), {None})


class ChildLimitTests(unittest.TestCase):
    def run_main(self, limits):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'limits.json'
            path.write_text(json.dumps(limits))
            calls = []
            target = SimpleNamespace(main=lambda args: ('ran', args))
            with patch.object(sys, 'argv', ['child', str(path), 'synthetic.target', 'a']), \
                    patch('markdown_for_llms.child.resource.setrlimit', side_effect=lambda *a: calls.append(a)), \
                    patch('markdown_for_llms.child.os.sched_getaffinity', return_value={0, 1, 2, 3, 4}), \
                    patch('markdown_for_llms.child.os.sched_setaffinity') as affinity, \
                    patch('markdown_for_llms.child.importlib.import_module', return_value=target) as module:
                result = child.main()
            module.assert_called_once_with('synthetic.target')
            affinity.assert_called_once_with(0, [0, 1])
            return result, dict(calls)

    def test_cpu_limit_only_with_a_deadline(self):
        import resource
        base = dict(memory_mib=64, output_mib=8, threads=2, process_limit=32)
        result, calls = self.run_main(base)
        self.assertEqual(result, ('ran', ['a']))
        self.assertNotIn(resource.RLIMIT_CPU, calls)
        self.assertEqual(calls[resource.RLIMIT_AS], (64 * 1048576,) * 2)
        result, calls = self.run_main(dict(base, cpu_seconds=120))
        self.assertEqual(calls[resource.RLIMIT_CPU], (120, 120))


class ProgressPersistenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.settings = Settings(data_root=self.tmp.name + '/data', model_root=self.tmp.name + '/models')
        self.service = Service(self.settings)
        self.addCleanup(self.service.storage.close)

    def stored(self, run_id):
        with self.service.storage.connect() as conn:
            value = conn.execute('SELECT progress FROM runs WHERE id=?', (run_id,)).fetchone()[0]
        return value and json.loads(value)

    def test_tick_without_snapshot_keeps_the_last_phase(self):
        job = self.service.submit(io.BytesIO(PDF), 'p.pdf')
        worker = Worker(self.settings)
        self.addCleanup(worker.storage.close)
        row = worker.claim()
        progress = Progress()
        progress.feed(bar('Recognizing Text', 3, 4).encode())
        self.assertFalse(worker.tick(row, None, progress.snapshot()))
        self.assertEqual(self.stored(job['id'])['current'], 3)
        self.assertFalse(worker.tick(row, None, finishing(progress.snapshot())))
        self.assertFalse(worker.tick(row, None))
        stored = self.stored(job['id'])
        self.assertEqual((stored['stage'], stored['phase'], stored['current']), ('finishing', 'Recognizing Text', 3))

    def test_conversion_receives_stall_and_optional_deadline_and_records_finishing(self):
        job = self.service.submit(io.BytesIO(PDF), 'p.pdf')
        calls = []
        def engine(settings, scratch, source=None, **kwargs):
            calls.append(kwargs)
            kwargs['tick'](None, {'stage': 'text', 'estimate_percent': 60.0, 'phase': 'Recognizing Text',
                                  'current': 5, 'total': 9, 'updated': 1.0, 'last_advance': 1.0})
            original = scratch / 'original'
            original.mkdir()
            write(original, 'document.md', b'converted')
            write_json(scratch, 'engine.json', {'inventory': [identify(original, 'document.md')], 'inventory_complete': True})
            return {'exit': 0, 'reason': None, 'elapsed_seconds': 1.0,
                    'progress': {'stage': 'processors', 'estimate_percent': 90.0, 'phase': 'Recognizing Text',
                                 'current': 9, 'total': 9, 'updated': 2.0, 'last_advance': 2.0}}
        worker = Worker(self.settings)
        self.addCleanup(worker.storage.close)
        with patch('markdown_for_llms.worker.check_runtime', return_value={}), \
                patch('markdown_for_llms.worker.execute', side_effect=engine):
            worker.process(worker.claim())
        self.assertEqual((calls[0]['deadline'], calls[0]['stall']), (None, 900))
        self.assertEqual(self.service.get(job['id'])['state'], 'completed')
        stored = self.stored(job['id'])
        self.assertEqual((stored['stage'], stored['estimate_percent'], stored['current']), ('finishing', 95.0, 9))

    def test_stalled_and_timeout_messages(self):
        for reason, settings, text in [('stalled', self.settings, 'No conversion progress for 900 s (MDF_STALL_SECONDS)'),
                                       ('timeout', replace(self.settings, timeout_seconds=7200), 'Conversion exceeded 7200 s (MDF_TIMEOUT_SECONDS)')]:
            with self.subTest(reason=reason):
                job = self.service.submit(io.BytesIO(PDF), 'p.pdf')
                worker = Worker(settings)
                self.addCleanup(worker.storage.close)
                calls = []
                receipt = {'exit': -15, 'reason': reason, 'elapsed_seconds': 1.0}
                with patch('markdown_for_llms.worker.check_runtime', return_value={}), \
                        patch('markdown_for_llms.worker.execute', side_effect=lambda *a, **k: calls.append(k) or receipt):
                    worker.process(worker.claim())
                result = self.service.get(job['id'])
                self.assertEqual(result['state'], 'failed')
                self.assertTrue(result['error'].startswith(text), result['error'])
                self.assertIn('source kept, retry available', result['error'])
                self.assertEqual(calls[0]['deadline'], settings.timeout_seconds or None)


class ProgressViewTests(unittest.TestCase):
    SNAPSHOT = dict(stage='text', estimate_percent=42.7, phase='Recognizing Text', current=12, total=47,
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
        self.assertEqual(running['progress'], dict(estimate_percent=42.7, estimated=True, stage='text',
                         phase='Recognizing Text', current=12, total=47, last_advance=990.0, idle_seconds=13.0,
                         stall_seconds=900, live=True, label='Recognizing Text 12/47 (≈42%, estimate)'))
        base = {k: running[k] for k in ('state', 'bundle', 'quality', 'message')}
        for state, label in [('failed', 'Stopped during Recognizing Text 12/47 (≈42%, estimate)'),
                             ('cancelled', 'Stopped during Recognizing Text 12/47 (≈42%, estimate)')]:
            self.set_run(job['id'], state, json.dumps(self.SNAPSHOT))
            stopped = self.service.get(job['id'])['progress']
            self.assertEqual((stopped['live'], stopped['label'], stopped['idle_seconds']), (False, label, None))
        self.set_run(job['id'], 'interrupted', json.dumps(finishing(self.SNAPSHOT)))
        self.assertEqual(self.service.get(job['id'])['progress']['label'], 'Stopped while finishing (≈95%, estimate)')
        self.set_run(job['id'], 'completed', json.dumps(self.SNAPSHOT))
        self.assertIsNone(self.service.get(job['id'])['progress'])
        for stored in ['{broken', '[]', json.dumps(dict(self.SNAPSHOT, stage='unknown')),
                       json.dumps(dict(self.SNAPSHOT, estimate_percent='42')), json.dumps(dict(self.SNAPSHOT, current=True)),
                       json.dumps(dict(self.SNAPSHOT, estimate_percent=101)), '{"stage": "text", "estimate_percent": NaN}',
                       json.dumps({k: v for k, v in self.SNAPSHOT.items() if k != 'phase'}),
                       json.dumps({k: v for k, v in self.SNAPSHOT.items() if k not in ('current', 'last_advance')}),
                       json.dumps(dict(self.SNAPSHOT, stage=['text'])), json.dumps(dict(self.SNAPSHOT, last_advance=float('nan'))),
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
        client = TestClient(create_app(self.settings), base_url='http://127.0.0.1:8765')
        self.assertEqual(client.get('/api/v1/runs/' + job['id']).json()['progress']['phase'], 'Recognizing Text')
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
                                 'convert', '--wait', str(source)])
                self.assertEqual(code, 0)
                self.assertEqual(json.loads(out.getvalue())['state'], 'completed')
                if tty:
                    self.assertIn('\r[########------------] Recognizing Text 12/47 (≈42%, estimate)', err.getvalue())
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
        self.assertEqual(replace(settings, timeout_seconds=450).legacy_identity(), settings.legacy_identity())
        self.assertEqual((changed.public()['timeout_seconds'], changed.public()['stall_seconds']), (600, 1200))

    def test_tqdm_is_pinned(self):
        self.assertEqual(PINS['tqdm'], '4.70.1')
        self.assertEqual(metadata.version('tqdm'), PINS['tqdm'])


if __name__ == '__main__':
    unittest.main()
