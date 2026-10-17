"""Independent publication-location, manifest roundtrip and passive polling regressions."""
from contextlib import contextmanager
import errno
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
import zipfile

from fastapi.testclient import TestClient
from markdown_for_llms.api import create_app
from markdown_for_llms.capacity import CONTROL, sizes
from markdown_for_llms.files import identify, write, write_json, FileSafetyError
from markdown_for_llms.sandbox import process_identity
from markdown_for_llms.services import Service
from markdown_for_llms.settings import Settings
from markdown_for_llms.worker import Worker
from markdown_for_llms.storage import CapacityError

PDF = b'%PDF-1.4\nr004 inert fixture\n'


class PublicationRoundtripTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.settings = Settings(data_root=self.tmp.name+'/data', model_root=self.tmp.name+'/models',
                                 upload_mib=1, output_mib=3, log_mib=1, storage_mib=64, reserve_mib=1)
        self.s = Service(self.settings)
        self.addCleanup(self.s.storage.close)

    def prepare(self, extra=None, raw=b'Original 0012, -7; formula $x_i$; final \xc3\xa8'):
        self.s.submit(io.BytesIO(PDF), 'fixture.pdf')
        w = Worker(self.settings)
        self.addCleanup(w.storage.close)
        row = w.claim()
        scratch = w.storage.root/'attempts'/row['attempt_id']
        (scratch/'original').mkdir(parents=True)
        write(scratch/'original', 'document.md', raw)
        engine = dict(inventory_complete=True, inventory=[identify(scratch/'original', 'document.md')])
        engine.update(extra or {})
        write_json(scratch, 'engine.json', engine)
        return w, row, scratch

    def process(self, w, row, scratch):
        with patch.object(w, '_process', side_effect=lambda r: w.publish(r, scratch, {}, {'elapsed_seconds': 0})):
            w.process(row)

    def pending(self, moved=False):
        w, row, scratch = self.prepare()
        real = os.rename
        class Crash(BaseException):
            pass
        def crash(src, dst, *args, **kwargs):
            if src == row['attempt_id']:
                if moved:
                    real(src, dst, *args, **kwargs)
                raise Crash()
            return real(src, dst, *args, **kwargs)
        with patch('markdown_for_llms.files.os.rename', side_effect=crash), self.assertRaises(Crash):
            self.process(w, row, scratch)
        return w, row, scratch

    def roundtrip(self, row, raw):
        job = self.s.get(row['id'])
        self.assertEqual((job['state'], job['bundle']), ('completed', 'ready'))
        self.assertIn(row['id'], [j['id'] for j in self.s.list()])
        db = self.s.storage.get(row['id'])
        body = (self.s.storage.root/db['result_path']/'manifest.json').read_bytes()
        self.assertLessEqual(len(body), 1048576)
        self.assertEqual(body, db['manifest'].encode())
        self.assertEqual(json.loads(body), job['manifest'])
        _, _, paths = self.s.verified_files(row['id'])
        with zipfile.ZipFile(self.s.bundle(row['id'])) as archive:
            self.assertIsNone(archive.testzip())
            self.assertEqual(archive.read('manifest.json'), body)
            self.assertEqual(archive.read('result/document.md'), raw)
        app = create_app(self.settings)
        self.addCleanup(app.state.service.storage.close)
        with TestClient(app, base_url='http://127.0.0.1:8765') as client:
            self.assertEqual(client.get('/api/v1/runs').status_code, 200)
            offset, observed = 0, b''
            while True:
                page = client.get('/api/v1/runs/'+row['id']+'/text', params={'offset': offset}).json()
                observed += page['text'].encode()
                offset = page['next_offset']
                if page['complete']:
                    break
            self.assertEqual(observed, raw)
        Worker(self.settings).reconcile(row['id'])
        self.assertTrue(self.s.get(row['id'])['download_available'])
        return len(body)

    def test_before_rename_syscall_errors_persistent_and_single(self):
        for code in [errno.EIO, errno.EACCES, errno.ENOSPC, errno.EDQUOT]:
            for persistent in [False, True]:
                with self.subTest(errno=code, persistent=persistent):
                    w, row, scratch = self.prepare()
                    real = os.rename
                    faults = []
                    def rename(src, dst, *args, **kwargs):
                        if src == row['attempt_id'] and (persistent or not faults):
                            faults.append(code)
                            raise OSError(code, 'before syscall')
                        return real(src, dst, *args, **kwargs)
                    with patch('markdown_for_llms.files.os.rename', side_effect=rename):
                        self.process(w, row, scratch)
                        if persistent:
                            self.assertEqual(self.s.get(row['id'])['state'], 'publishing')
                            self.assertTrue(scratch.is_dir())
                            w.reconcile(row['id'])
                            self.assertEqual(self.s.get(row['id'])['state'], 'publishing')
                    w.reconcile(row['id'])
                    self.roundtrip(row, b'Original 0012, -7; formula $x_i$; final \xc3\xa8')

    def test_location_and_integrity_negatives_preserve_trees(self):
        for case in ['both', 'neither', 'manifest', 'raw', 'missing', 'symlink', 'extra', 'source', 'identity', 'revoked']:
            with self.subTest(case=case):
                w, row, scratch = self.pending()
                result = w.storage.root/'results'/row['attempt_id']
                if case == 'both': shutil.copytree(scratch, result)
                if case == 'neither': scratch.rename(Path(self.tmp.name)/('retained-'+row['attempt_id']))
                if case == 'manifest': (scratch/'manifest.json').write_text('{}')
                if case == 'raw': (scratch/'original/document.md').write_bytes(b'altered')
                if case == 'missing': (scratch/'original/document.md').unlink()
                if case == 'symlink':
                    (scratch/'original/document.md').unlink()
                    (scratch/'original/document.md').symlink_to('/etc/hostname')
                if case == 'extra': (scratch/'original/extra').write_bytes(b'uninventoried')
                if case == 'source': (w.storage.root/row['source_path']).write_bytes(b'wrong')
                if case == 'identity':
                    manifest = json.loads((scratch/'manifest.json').read_bytes())
                    manifest['document_id'] = 'f'*32
                    (scratch/'manifest.json').write_text(json.dumps(manifest))
                    with w.storage.connect() as c:
                        c.execute('UPDATE runs SET manifest=? WHERE id=?', (json.dumps(manifest), row['id']))
                if case == 'revoked':
                    with w.storage.connect() as c: c.execute('UPDATE runs SET token=NULL WHERE id=?', (row['id'],))
                w.reconcile(row['id'])
                job = self.s.get(row['id'])
                self.assertEqual(job['state'], 'interrupted')
                self.assertIn('integrity', job['error'].lower())
                self.assertFalse(job['download_available'])
                if case == 'both': self.assertTrue(scratch.exists() and result.exists())

    def test_manifest_depth_and_non_ascii_roundtrip(self):
        for levels in [125, 126, 127]:
            with self.subTest(levels=levels):
                deep = 0
                for _ in range(levels): deep = [deep]
                w, row, scratch = self.prepare({'extra': deep})
                self.process(w, row, scratch)
                if levels == 127:
                    self.assertEqual(self.s.get(row['id'])['state'], 'failed')
                    self.assertTrue((scratch/'original/document.md').exists())
                    self.assertIsNone(self.s.storage.get(row['id'])['manifest'])
                else:
                    self.roundtrip(row, (w.storage.root/'results'/row['attempt_id']/'original/document.md').read_bytes())
        for extra, raw in [({'extra': 'è'*180000}, b'Unicode metadata'),
                           ({'notes': ['説明'*100]*1100}, b'Unicode notes'),
                           ({}, ''.join(f'<a href="https://例え.テスト/{"資料"*40}/{i}">r{i}</a>\n\n' for i in range(1800)).encode())]:
            with self.subTest(non_ascii=len(raw)):
                w, row, scratch = self.prepare(extra, raw)
                self.process(w, row, scratch)
                self.roundtrip(row, raw)

    def test_exact_file_limit_includes_newline(self):
        w, row, scratch = self.prepare({'pad': 'a'*1000})
        self.process(w, row, scratch)
        size = self.roundtrip(row, (w.storage.root/'results'/row['attempt_id']/'original/document.md').read_bytes())
        for target in [1048575, 1048576, 1048577]:
            with self.subTest(bytes=target):
                w, row, scratch = self.prepare({'pad': 'a'*(1000+target-size)})
                self.process(w, row, scratch)
                if target > 1048576:
                    self.assertEqual(self.s.get(row['id'])['state'], 'failed')
                    self.assertIsNone(self.s.storage.get(row['id'])['manifest'])
                    self.assertTrue((scratch/'original/document.md').exists())
                else:
                    observed = self.roundtrip(row, (w.storage.root/'results'/row['attempt_id']/'original/document.md').read_bytes())
                    self.assertEqual(observed, target)

    def test_malformed_historical_document_is_confined(self):
        w, row, scratch = self.prepare()
        self.process(w, row, scratch)
        original = self.s.get(row['id'])['manifest']
        other = self.s.submit(io.BytesIO(PDF), 'other.pdf')
        app = create_app(self.settings)
        self.addCleanup(app.state.service.storage.close)
        with TestClient(app, base_url='http://127.0.0.1:8765') as client:
            for case in ['bundle', 'path']:
                with self.subTest(case=case):
                    manifest = json.loads(json.dumps(original))
                    if case == 'bundle': manifest['bundle'] = []
                    else: manifest['artifacts'][0]['path'] = '../document.md'
                    with self.s.storage.connect() as c:
                        c.execute('UPDATE runs SET manifest=? WHERE id=?', (json.dumps(manifest), row['id']))
                    job = client.get('/api/v1/runs/'+row['id'])
                    self.assertEqual(job.status_code, 200)
                    self.assertEqual(job.json()['bundle'], 'invalid')
                    self.assertIn('Malformed', job.json()['error'])
                    self.assertEqual(client.get('/api/v1/runs').status_code, 200)
                    self.assertEqual(client.get('/api/v1/runs/'+other['id']).json()['state'], 'queued')

    def test_polling_is_passive_even_with_unsafe_unattributed_attempt(self):
        pending = [self.pending()[1] for _ in range(3)]
        outside = Path(self.tmp.name)/'outside'
        outside.write_bytes(b'unchanged')
        unsafe = self.s.storage.root/'attempts/unattributed'
        unsafe.mkdir()
        (unsafe/'link').symlink_to(outside)
        before = sizes(unsafe.parent.parent, isolated=('attempts/unattributed',))
        app = create_app(self.settings)
        self.addCleanup(app.state.service.storage.close)
        with TestClient(app, base_url='http://127.0.0.1:8765') as client:
            with patch('markdown_for_llms.capacity.archive_sizes', side_effect=AssertionError('global scan on GET')):
                with self.s.storage.lock('admission.lock'):
                    t = time.monotonic()
                    for _ in range(10):
                        self.assertEqual(client.get('/api/v1/runs').status_code, 200)
                        for row in pending:
                            self.assertIn('recovery', client.get('/api/v1/runs/'+row['id']).json())
                    self.assertLess(time.monotonic()-t, 2)
                with self.s.storage.lock('worker.lock'):
                    for row in pending:
                        job = client.get('/api/v1/runs/'+row['id']).json()
                        self.assertNotIn('recovery', job)
                        self.assertIsNone(job['error'])
                for exc in [OSError(errno.EACCES, 'diagnosis'), FileSafetyError('Unsafe diagnostic lock')]:
                    with patch('markdown_for_llms.services.open_file', side_effect=exc):
                        self.assertEqual(client.get('/api/v1/runs').status_code, 200)
        self.assertEqual(outside.read_bytes(), b'unchanged')
        self.assertEqual(before, sizes(unsafe.parent.parent, isolated=('attempts/unattributed',)))

    def test_real_crash_before_and_after_rename_one_restart_tight_quota(self):
        for moved in [False, True]:
            with self.subTest(moved=moved):
                base = Path(self.tmp.name)/str(moved)
                base.mkdir()
                script = base/'crash.py'
                script.write_text(CRASH_CHILD)
                p = subprocess.run([sys.executable, '-I', '-B', str(script), str(base), str(int(moved))],
                                   capture_output=True, timeout=20)
                self.assertEqual(p.returncode, 9, p.stderr.decode())
                settings = Settings(data_root=str(base/'data'), model_root=str(base/'models'),
                                    upload_mib=1, output_mib=3, log_mib=1, storage_mib=16, reserve_mib=1)
                service = Service(settings)
                self.addCleanup(service.storage.close)
                row = service.storage.list()[0]
                child = json.loads(row['child'])
                def stop_child():
                    now = process_identity(child['pid'])
                    if now and now['start'] == child['start'] and now['state'] != 'Z':
                        os.kill(child['pid'], 9)
                self.addCleanup(stop_child)
                self.assertEqual(row['state'], 'publishing')
                self.assertEqual(len(list((base/'data/operations').iterdir())), 2)
                total = sum(sizes(base/'data').values())
                (base/'data/padding').write_bytes(b'p'*(16*1048576-total-CONTROL-4*1048576))
                w = Worker(settings)
                self.addCleanup(w.storage.close)
                w.reconcile()
                final = service.get(row['id'])
                self.assertEqual((final['state'], final['id'], final['attempt_id']),
                                 ('completed', row['id'], row['attempt_id']))
                now = process_identity(child['pid'])
                self.assertTrue(not now or now['start'] != child['start'] or now['state'] == 'Z')
                self.assertEqual(list((base/'data/operations').iterdir()), [])

    def test_live_worker_and_independent_leases_are_not_collected(self):
        w, row, scratch = self.prepare()
        child = subprocess.Popen([sys.executable, '-I', '-B', '-c', 'import time; time.sleep(30)'])
        self.addCleanup(lambda: child.poll() is None and child.kill())
        self.addCleanup(lambda: child.wait(timeout=5) if child.poll() is not None else None)
        w.tick(row, process_identity(child.pid))
        with w.commitment(row), self.s.storage.reservation('control', 65536):
            before = list((w.storage.root/'operations').iterdir())
            Worker(self.settings).reconcile()
            self.assertEqual(set(before), set((w.storage.root/'operations').iterdir()))
            self.assertIsNone(child.poll())
            self.assertEqual(self.s.get(row['id'])['state'], 'running')
        child.terminate()
        child.wait(timeout=5)

    def test_unrecordable_deferral_releases_owner_before_readiness(self):
        w, row, scratch = self.pending()
        before = sizes(w.storage.root)
        with patch.object(w.storage, 'reservation', side_effect=CapacityError('No diagnostic space')):
            with patch('markdown_for_llms.worker.execute', side_effect=AssertionError('readiness after deferral')):
                w.run(once=True)
        self.assertTrue(w.stop)
        job = self.s.get(row['id'])
        self.assertEqual(job['state'], 'publishing')
        self.assertIn('recovery', job)
        self.assertEqual(before, sizes(w.storage.root))


CRASH_CHILD = r'''
import io, os, subprocess, sys
from pathlib import Path
from unittest.mock import patch
from markdown_for_llms.settings import Settings
from markdown_for_llms.services import Service
from markdown_for_llms.worker import Worker
from markdown_for_llms.sandbox import process_identity
from markdown_for_llms.files import write, write_json, identify
base = Path(sys.argv[1])
s = Settings(data_root=str(base/'data'), model_root=str(base/'models'), upload_mib=1,
             output_mib=3, log_mib=1, storage_mib=16, reserve_mib=1)
Service(s).submit(io.BytesIO(b'%PDF-inert crash'), 'crash.pdf')
w = Worker(s); row = w.claim()
scratch = base/'data/attempts'/row['attempt_id']; (scratch/'original').mkdir(parents=True)
write(scratch/'original', 'document.md', b'Preserved crash output 0012')
write_json(scratch, 'engine.json', dict(inventory_complete=True, inventory=[identify(scratch/'original', 'document.md')]))
child = subprocess.Popen([sys.executable, '-I', '-B', '-c', 'import time; time.sleep(60)'],
                         start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
w.tick(row, process_identity(child.pid))
real = os.rename
def crash(src, dst, *a, **kw):
    if src == row['attempt_id']:
        if int(sys.argv[2]): real(src, dst, *a, **kw)
        os._exit(9)
    return real(src, dst, *a, **kw)
with patch('markdown_for_llms.files.os.rename', side_effect=crash), patch.object(w, '_process', side_effect=lambda r: w.publish(r, scratch, {}, {'elapsed_seconds': 0})):
    w.process(row)
'''
