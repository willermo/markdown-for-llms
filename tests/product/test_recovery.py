"""Independent regressions for shared capacity, temporary ownership and recovery."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import replace
import errno
import hashlib
import io
import json
import os
from pathlib import Path
import sqlite3
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
from markdown_for_llms.capacity import recover, sizes
from markdown_for_llms.files import identify, write, write_json
from markdown_for_llms.references import inspect
from markdown_for_llms.sandbox import scratch_size
from markdown_for_llms.services import Service, ReadUnavailable
from markdown_for_llms.settings import Settings
from markdown_for_llms.storage import Storage, StorageError
from markdown_for_llms.worker import Worker, validate_raster

PDF = b'%PDF-1.4\nsynthetic recovery contract\n'


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.settings = Settings(data_root=self.tmp.name+'/data', model_root=self.tmp.name+'/models',
                                 upload_mib=1, output_mib=3, log_mib=1, storage_mib=12, reserve_mib=1)
        self.s = Service(self.settings)
        self.addCleanup(self.s.storage.close)

    def output(self, row, raw=b'preserved output', assets=None):
        scratch = self.s.storage.root/'attempts'/row['attempt_id']
        original = scratch/'original'
        original.mkdir(parents=True, exist_ok=True)
        write(original, 'document.md', raw)
        for name, data in (assets or {}).items():
            (original/name).parent.mkdir(parents=True, exist_ok=True)
            write(original, name, data)
        write_json(scratch, 'engine.json', {'inventory_complete': True,
            'inventory': [identify(original, n) for n in ['document.md', *(assets or {})]]})
        return scratch

    def completed(self, assets=None):
        j = self.s.submit(io.BytesIO(PDF), 'x.pdf')
        w = Worker(self.settings); self.addCleanup(w.storage.close)
        row = w.claim(); scratch = self.output(row, assets=assets)
        with patch('markdown_for_llms.worker.execute', return_value={'exit': 1, 'reason': 'fixture'}):
            w.publish(row, scratch, {'mock': True}, {'elapsed_seconds': 0})
        return self.s.get(j['id'])

    def assert_clean(self):
        self.assertEqual(list((self.s.storage.root/'operations').iterdir()), [])
        self.assertEqual(list((self.s.storage.root/'staging').iterdir()), [])

    def test_reject_stream_failure_replay_conflict_leave_no_bytes(self):
        accepted = self.s.submit(io.BytesIO(PDF), 'x.pdf', key='key')
        before = sizes(self.s.storage.root)
        for body, name in [(b'wrong', 'x.pdf'), (PDF, 'x.txt'), (PDF+b'x'*1048576, 'x.pdf')]:
            with self.assertRaises(StorageError): self.s.submit(io.BytesIO(body), name)
        class Broken:
            def read(self, n): raise OSError(errno.EIO, 'synthetic stream fault')
        with self.assertRaises(OSError): self.s.submit(Broken(), 'x.pdf')
        for _ in range(30):
            self.assertEqual(self.s.submit(io.BytesIO(PDF), 'x.pdf', key='key')['id'], accepted['id'])
        for body, opts in [(PDF+b'x', None), (PDF, {'force_ocr': True})]:
            with self.assertRaises(StorageError): self.s.submit(io.BytesIO(body), 'x.pdf', opts, key='key')
        self.assert_clean()
        self.assertEqual(before, sizes(self.s.storage.root))
        self.assertEqual(len(self.s.list()), 1)

    def test_replay_at_quota_does_not_reserve_another_source(self):
        j = self.s.submit(io.BytesIO(PDF), 'x.pdf', key='key')
        from markdown_for_llms.capacity import CONTROL
        available = self.settings.storage_mib*1048576 - self.s.storage.capacity() - CONTROL - 1024
        write(self.s.storage.root, 'owned-fixture-padding', b'x'*available)
        before = sizes(self.s.storage.root)
        self.assertEqual(self.s.submit(io.BytesIO(PDF), 'x.pdf', key='key')['id'], j['id'])
        with self.assertRaises(StorageError): self.s.submit(io.BytesIO(PDF), 'other.pdf')
        self.assertEqual(before, sizes(self.s.storage.root)); self.assert_clean()

    def test_deterministic_worker_upload_interleaving(self):
        pdf = b'%PDF-'+b'x'*(1048576-5)
        j = self.s.submit(io.BytesIO(pdf), 'x.pdf')
        w = Worker(self.settings); self.addCleanup(w.storage.close); row = w.claim()
        admitted = []
        def engine(settings, scratch, source, **kw):
            # A distinct service sees the outstanding commitment before growth.
            other = Service(settings)
            try:
                for _ in range(20):
                    try: admitted.append(other.submit(io.BytesIO(pdf), 'concurrent.pdf')['id'])
                    except StorageError: break
                self.assertLess(len(admitted), 8)
                self.output(row, raw=b'x'*(2*1048576))
                self.s.storage.capacity()
                return {'exit': 0, 'reason': None, 'elapsed_seconds': 0}
            finally: other.storage.close()
        with patch('markdown_for_llms.worker.check_runtime', return_value={'mock': True}), patch('markdown_for_llms.worker.execute', side_effect=engine): w.process(row)
        self.assertEqual(self.s.get(j['id'])['state'], 'completed')
        self.s.storage.capacity(); self.assert_clean()

    def test_reservations_do_not_double_count_existing_growth(self):
        token = 'a'*32
        with self.s.storage.reservation('worker', 6*1048576, token, ('attempts/'+token,)):
            p = self.s.storage.root/'attempts'/token; p.mkdir()
            write(p, 'growth', b'x'*(2*1048576))
            self.s.storage.capacity(4*1048576)
            with self.assertRaises(StorageError): self.s.storage.capacity(6*1048576)
        self.s.storage.capacity(6*1048576)

    def test_crash_owned_upload_recovery_and_live_owner(self):
        code = """import sys
from markdown_for_llms.settings import Settings
from markdown_for_llms.services import Service
s=Service(Settings(**__import__('json').loads(sys.argv[1])))
with s.upload('x.pdf') as u:
 u.write(b'%PDF-'+b'x'*10000); u.stream.flush()
 print('ready',flush=True)
 __import__('time').sleep(60)
"""
        from dataclasses import asdict
        p = subprocess.Popen([sys.executable, '-I', '-B', '-c', code, json.dumps(asdict(self.settings))], stdout=subprocess.PIPE, text=True)
        self.addCleanup(lambda: p.poll() is None and p.kill())
        self.assertEqual(p.stdout.readline().strip(), 'ready')
        with self.s.storage.lock('admission.lock'): recover(self.s.storage)
        self.assertEqual(len(list((self.s.storage.root/'staging').iterdir())), 1)
        p.kill(); p.wait(); p.stdout.close()
        # Registration of the next request triggers cleanup, without touching its source.
        self.s.submit(io.BytesIO(PDF), 'next.pdf'); self.assert_clean()
        self.assertEqual(len(list((self.s.storage.root/'sources').iterdir())), 1)

    def test_rename_commit_gap_collects_only_unregistered_source(self):
        from markdown_for_llms import services
        real = services.rename
        def broken(*a): real(*a); raise OSError(errno.EIO, 'after rename')
        with patch('markdown_for_llms.services.rename', side_effect=broken), self.assertRaises(OSError): self.s.submit(io.BytesIO(PDF), 'x.pdf')
        self.assertEqual(self.s.list(), []); self.assert_clean()
        self.assertEqual(list((self.s.storage.root/'sources').iterdir()), [])

    def test_zip_repeated_concurrent_verified_cache(self):
        j = self.completed()
        first = self.s.bundle(j['id']); before = sizes(self.s.storage.root)
        with ThreadPoolExecutor(max_workers=4) as pool:
            paths = list(pool.map(lambda _: self.s.bundle(j['id']), range(8)))
        self.assertEqual(set(paths), {first}); self.assertEqual(before, sizes(self.s.storage.root))
        self.assertFalse(list((self.s.storage.root/'downloads').glob('*.part')))
        with zipfile.ZipFile(first) as z:
            self.assertIsNone(z.testzip()); self.assertEqual(set(z.namelist()), {'source/document.pdf', 'result/document.md', 'manifest.json'})
            self.assertEqual(z.read('source/document.pdf'), PDF)
            self.assertEqual(json.loads(z.read('manifest.json'))['attempt_id'], j['attempt_id'])
        row = self.s.storage.get(j['id']); (self.s.storage.root/row['result_path']/'original/document.md').unlink()
        with self.assertRaises(StorageError): self.s.bundle(j['id'])
        self.assertEqual(self.s.get(j['id'])['bundle'], 'invalid')

    def test_zip_failure_cleans_partial_and_can_retry(self):
        j = self.completed()
        with patch('markdown_for_llms.services.zipfile.ZipFile', side_effect=OSError(errno.ENOSPC, 'synthetic')), self.assertRaises(OSError): self.s.bundle(j['id'])
        self.assertFalse(list((self.s.storage.root/'downloads').glob('*.part')))
        self.s.bundle(j['id']); self.assert_clean()

    def test_transient_read_errors_retry_same_result(self):
        j = self.completed()
        for error in (errno.EMFILE, errno.EIO, errno.ENOMEM, errno.EACCES):
            with patch('markdown_for_llms.services.identify', side_effect=OSError(error, 'synthetic')), self.assertRaises(ReadUnavailable): self.s.bundle(j['id'])
            with patch('markdown_for_llms.services.identify', side_effect=OSError(error, 'synthetic')): Worker(self.settings).reconcile()
            self.assertEqual(self.s.get(j['id'])['bundle'], 'ready')
            self.s.bundle(j['id'])
        self.assertEqual(self.s.get(j['id'])['attempt_id'], j['attempt_id'])

    def test_http_transient_source_text_gallery_and_manifest(self):
        j = self.completed(assets={'img.png': b'opaque'})
        client = TestClient(create_app(self.settings), base_url='http://127.0.0.1:8765')
        for endpoint in ('source', 'text', 'artifacts/'+j['manifest']['artifacts'][1]['id']):
            with patch('markdown_for_llms.api.open_file', side_effect=OSError(errno.EIO, 'synthetic')):
                self.assertEqual(client.get('/api/v1/runs/'+j['id']+'/'+endpoint).status_code, 503)
            self.assertEqual(self.s.get(j['id'])['bundle'], 'ready')
        with patch('markdown_for_llms.services.read_json', side_effect=OSError(errno.EMFILE, 'synthetic')), self.assertRaises(ReadUnavailable): self.s.bundle(j['id'])
        self.assertEqual(client.get('/api/v1/runs/'+j['id']+'/text').status_code, 200)
        client.close()

    def test_missing_tampered_unsafe_files_invalidate(self):
        for fault in ('missing', 'digest', 'symlink'):
            j = self.completed(); row = self.s.storage.get(j['id'])
            p = self.s.storage.root/row['result_path']/'original/document.md'
            if fault == 'digest': p.write_bytes(b'tampered')
            else:
                p.unlink()
                if fault == 'symlink': p.symlink_to('/etc/passwd')
            with self.assertRaises(StorageError): self.s.bundle(j['id'])
            self.assertEqual(self.s.get(j['id'])['bundle'], 'invalid')

    def test_fsync_files_nested_directories_before_publication(self):
        j = self.s.submit(io.BytesIO(PDF), 'x.pdf'); w = Worker(self.settings); self.addCleanup(w.storage.close)
        row = w.claim(); scratch = self.output(row, assets={'nested/img.png': b'opaque'})
        synced = []; real = os.fsync
        def sync(fd):
            synced.append(os.readlink('/proc/self/fd/'+str(fd))); real(fd)
        from markdown_for_llms.worker import rename
        def move(*args):
            for name in ('original/document.md', 'original/nested/img.png', 'original/nested', 'original'):
                self.assertIn(str(scratch/name), synced)
            return rename(*args)
        with patch('markdown_for_llms.worker.execute', return_value={'exit': 1, 'reason': 'fixture'}), patch('markdown_for_llms.files.os.fsync', side_effect=sync), patch('markdown_for_llms.worker.rename', side_effect=move):
            w.publish(row, scratch, {}, {'elapsed_seconds': 0})
        self.assertEqual(self.s.get(j['id'])['state'], 'completed')

    def test_fsync_failure_blocks_publication(self):
        j = self.s.submit(io.BytesIO(PDF), 'x.pdf'); w = Worker(self.settings); self.addCleanup(w.storage.close)
        row = w.claim(); scratch = self.output(row)
        with patch('markdown_for_llms.worker.sync_tree', side_effect=OSError(errno.EIO, 'synthetic')), self.assertRaises(OSError): w.publish(row, scratch, {}, {'elapsed_seconds': 0})
        self.assertNotEqual(self.s.get(j['id'])['state'], 'completed')

    def test_scratch_churn_only_enoent_is_tolerated(self):
        root = Path(self.tmp.name)/'churn'; root.mkdir(); stop = threading.Event()
        def churn():
            while not stop.is_set():
                p = root/'tmp'; p.write_bytes(b'x'); p.unlink()
        thread = threading.Thread(target=churn); thread.start()
        try:
            for _ in range(1000): scratch_size(root)
        finally: stop.set(); thread.join()
        with patch('markdown_for_llms.capacity.os.walk', side_effect=OSError(errno.EIO, 'synthetic')), self.assertRaises(OSError): scratch_size(root)

    def test_worker_stop_distinct_from_user_cancel(self):
        for cancel in (False, True):
            j = self.s.submit(io.BytesIO(PDF), 'x.pdf'); w = Worker(self.settings); self.addCleanup(w.storage.close); row = w.claim()
            def engine(*a, **kw):
                if cancel: self.s.cancel(j['id'])
                else: w.stop = True
                reason = kw['tick'](None)
                return {'exit': -15, 'reason': reason or 'fixture'}
            with patch('markdown_for_llms.worker.check_runtime', return_value={}), patch('markdown_for_llms.worker.execute', side_effect=engine): w.process(row)
            self.assertEqual(self.s.get(j['id'])['state'], 'cancelled' if cancel else 'interrupted')
            self.assertEqual(identify(self.s.storage.root, self.s.storage.get(j['id'])['source_path'])['bytes'], len(PDF))

    def test_safe_reference_forms_and_negative_paths(self):
        for raw in ('[x]()', '<a href="">x</a>', '![x](./img.png)'):
            self.assertEqual(inspect(raw, ['img.png'])['bundle'], 'ready')
        for path in ('../img.png', './%2e%2e/x', '/x', '//host/x', 'file:///etc/passwd', '%252e%252e/x', 'C:/x', r'\\host\x'):
            self.assertEqual(inspect('![x]('+path+')', [])['bundle'], 'invalid')
        self.assertEqual(inspect('`![x](../bad)`\n\n$$<img src="../bad">$$', [])['bundle'], 'ready')

    def test_raster_allowlist_cannot_replace_inventory_identity(self):
        candidates = [{'path': 'img.png'}]
        good = dict(path='img.png', gallery=True, reason=None, mime='image/png', width=1, height=1)
        self.assertNotIn('path', validate_raster([good], candidates, self.settings)['img.png'])
        for bad in ([dict(good, id='evil')], [dict(good, sha256='evil')], [dict(good, path='other')],
                    [dict(good, mime='text/html')], [good, good], None, [dict(good, width=True)], [dict(good, gallery='true')]):
            with self.assertRaises(StorageError): validate_raster(bad, candidates, self.settings)

    def test_malformed_engine_and_sqlite_faults_have_visible_states(self):
        for fault in ('metadata', 'sqlite'):
            j = self.s.submit(io.BytesIO(PDF), 'x.pdf'); w = Worker(self.settings); self.addCleanup(w.storage.close); row = w.claim()
            def engine(settings, scratch, source, **kw):
                if fault == 'sqlite': raise sqlite3.OperationalError('database is locked')
                write_json(scratch, 'engine.json', {'inventory_complete': True, 'inventory': [None]})
                return {'exit': 0, 'reason': None, 'elapsed_seconds': 0}
            with patch('markdown_for_llms.worker.check_runtime', return_value={}), patch('markdown_for_llms.worker.execute', side_effect=engine): w.process(row)
            self.assertEqual(self.s.get(j['id'])['state'], 'failed'); self.assertTrue(self.s.get(j['id'])['error'])
        j = self.s.submit(io.BytesIO(PDF), 'x.pdf'); w = Worker(self.settings); row = w.claim()
        with patch.object(w, '_process', side_effect=sqlite3.OperationalError('readonly')), patch.object(w.storage, 'connect', side_effect=sqlite3.OperationalError('readonly')), self.assertRaisesRegex(StorageError, 'requires reconciliation'): w.process(row)
        self.assertEqual(self.s.get(j['id'])['state'], 'running')
        w.reconcile(); self.assertEqual(self.s.get(j['id'])['state'], 'interrupted'); w.storage.close()

    def test_administrative_update_requires_exclusive_old_identity(self):
        j = self.completed(); manifest = j['manifest']
        with self.assertRaises(StorageError): Storage.configure(self.settings, self.settings.identity(), storage_mib=20)
        # All service instances must release their shared configuration locks.
        for cleanup in list(self._cleanups):
            if getattr(cleanup[0], '__name__', '') == 'close': cleanup[0]()
        with self.assertRaises(StorageError): Storage.configure(self.settings, 'wrong', storage_mib=20)
        result = Storage.configure(self.settings, self.settings.identity(), storage_mib=20)
        new = replace(self.settings, storage_mib=20)
        self.assertEqual(result['config'], new.identity())
        with self.assertRaises(StorageError): Service(self.settings)
        s = Service(new)
        self.assertEqual(s.get(j['id'])['manifest'], manifest)
        s.bundle(j['id']); s.storage.close()

    def test_zero_link_churn_is_not_a_hardlink_bypass(self):
        import stat
        from types import SimpleNamespace
        root = Path(self.tmp.name)/'stat-race'; root.mkdir(); write(root, 'temporary', b'x')
        real = Path.lstat
        def vanished(path, *a, **kw):
            if path.name == 'temporary': return SimpleNamespace(st_mode=stat.S_IFREG|0o600, st_nlink=0, st_size=1)
            return real(path, *a, **kw)
        with patch('pathlib.Path.lstat', vanished): self.assertEqual(scratch_size(root), 0)
        def hardlinked(path, *a, **kw):
            if path.name == 'temporary': return SimpleNamespace(st_mode=stat.S_IFREG|0o600, st_nlink=2, st_size=1)
            return real(path, *a, **kw)
        with patch('pathlib.Path.lstat', hardlinked), self.assertRaises(ValueError): scratch_size(root)

    def test_interrupted_reservation_creation_recovers_owned_pending_only(self):
        root = self.s.storage.root
        # This protocol's pending record is locked until its owner dies; it
        # cannot have associated source or scratch before its atomic rename.
        name = 'operations/lease-'+'b'*32+'.pending'
        write(root, name, b'{partial')
        with self.s.storage.lock('admission.lock'): recover(self.s.storage)
        self.assertFalse((root/name).exists())
        write(root, 'operations/unrelated.part', b'preserve')
        with self.s.storage.lock('admission.lock'), self.assertRaises(ValueError): recover(self.s.storage)
        self.assertTrue((root/'operations/unrelated.part').exists())
