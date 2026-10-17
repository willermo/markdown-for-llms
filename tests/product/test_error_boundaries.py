"""Independent r003 error semantics, durable recovery and isolation regressions."""
from contextlib import contextmanager
import errno
import io
import json
import os
from pathlib import Path
import socket
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from fastapi.testclient import TestClient
from markdown_for_llms.api import create_app
from markdown_for_llms.capacity import CONTROL, sizes
from markdown_for_llms.files import (IntegrityError, FileSafetyError, MetadataError,
                                    corruption, decode_json, identify, write, write_json)
from markdown_for_llms.references import inspect
from markdown_for_llms.services import Service, ReadUnavailable
from markdown_for_llms.settings import Settings
from markdown_for_llms.storage import CapacityError
from markdown_for_llms.worker import Worker

PDF = b'%PDF-1.4\nindependent r003 fixture\n'


class ErrorBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.settings = Settings(data_root=self.tmp.name+'/data', model_root=self.tmp.name+'/models',
                                 upload_mib=1, output_mib=3, log_mib=1, storage_mib=16, reserve_mib=1)
        self.s = Service(self.settings)
        self.addCleanup(self.s.storage.close)

    def prepared(self, raw=b'intact output', assets=None):
        job = self.s.submit(io.BytesIO(PDF), 'fixture.pdf')
        w = Worker(self.settings)
        self.addCleanup(w.storage.close)
        row = w.claim()
        scratch = w.storage.root/'attempts'/row['attempt_id']
        original = scratch/'original'
        original.mkdir(parents=True)
        write(original, 'document.md', raw)
        for name, body in (assets or {}).items():
            (original/name).parent.mkdir(parents=True, exist_ok=True)
            write(original, name, body)
        write_json(scratch, 'engine.json', dict(inventory_complete=True,
            inventory=[identify(original, n) for n in ['document.md', *(assets or {})]]))
        return w, row, scratch

    def publish(self, w, row, scratch):
        with patch('markdown_for_llms.worker.execute', return_value={'exit': 1, 'reason': 'synthetic gallery'}):
            w.publish(row, scratch, {'synthetic': True}, {'elapsed_seconds': 0})

    def pending(self):
        w, row, scratch = self.prepared(assets={'nested/image.png': b'image bytes'})
        class Crash(BaseException):
            pass
        from markdown_for_llms.worker import rename
        def move(*args):
            rename(*args)
            raise Crash()
        with patch('markdown_for_llms.worker.rename', side_effect=move), self.assertRaises(Crash):
            self.publish(w, row, scratch)
        self.assertEqual(self.s.get(row['id'])['state'], 'publishing')
        return w, row

    def test_explicit_classification_and_unknown_failures(self):
        for exc in [ValueError('offset'), KeyError('unknown'), TypeError('unknown'),
                    CapacityError('quota'), MetadataError('child'),
                    *[OSError(e, 'transient') for e in [errno.EIO, errno.EMFILE, errno.ENOMEM, errno.EACCES]]]:
            self.assertFalse(corruption(exc))
        for exc in [IntegrityError('mismatch'), FileSafetyError('unsafe'),
                    *[OSError(e, 'missing') for e in [errno.ENOENT, errno.ENOTDIR, errno.ELOOP]]]:
            self.assertTrue(corruption(exc))
        w, row, scratch = self.prepared()
        self.publish(w, row, scratch)
        for exc in [ValueError('unknown'), KeyError('unknown'), TypeError('unknown')]:
            with self.assertRaises(ReadUnavailable): self.s.read_failure(row['id'], exc)
            self.assertEqual(self.s.get(row['id'])['bundle'], 'ready')

    def test_request_offsets_utf8_tail_eof_and_pagination(self):
        raw = ('a'*262143+'€'+'città'*100+'終').encode()
        w, row, scratch = self.prepared(raw)
        self.publish(w, row, scratch)
        app = create_app(self.settings)
        self.addCleanup(app.state.service.storage.close)
        with TestClient(app, base_url='http://127.0.0.1:8765') as c:
            url = '/api/v1/runs/'+row['id']
            for offset in [-1, len(raw)+1, 262144, len(raw)-1]:
                self.assertEqual(c.get(url+'/text', params={'offset': offset}).status_code, 400)
                self.assertTrue(c.get(url).json()['download_available'])
            offset, body = 0, ''
            while True:
                page = c.get(url+'/text', params={'offset': offset}).json()
                body += page['text']
                self.assertEqual(page['next_offset']-offset, len(page['text'].encode()))
                offset = page['next_offset']
                if page['complete']: break
            self.assertEqual(body.encode(), raw)
            self.assertEqual(c.get(url+'/text', params={'offset': len(raw)}).json()['text'], '')
            z = c.get(url+'/bundle')
            self.assertEqual(z.status_code, 200)
            with zipfile.ZipFile(io.BytesIO(z.content)) as archive:
                self.assertEqual(archive.read('result/document.md'), raw)

    def test_real_corrupt_utf8_not_a_request_error(self):
        w, row, scratch = self.prepared()
        self.publish(w, row, scratch)
        path = self.s.storage.root/'results'/row['attempt_id']
        bad = b'broken\xff\x80'
        (path/'original/document.md').write_bytes(bad)
        manifest = json.loads((path/'manifest.json').read_text())
        manifest['artifacts'][0].update(identify(path/'original', 'document.md'))
        (path/'manifest.json').write_text(json.dumps(manifest))
        with self.s.storage.connect() as conn:
            conn.execute('UPDATE runs SET manifest=? WHERE id=?', (json.dumps(manifest), row['id']))
        app = create_app(self.settings)
        self.addCleanup(app.state.service.storage.close)
        with TestClient(app, base_url='http://127.0.0.1:8765') as c:
            self.assertEqual(c.get('/api/v1/runs/'+row['id']+'/text', params={'offset': len(bad)-1}).status_code, 409)
            self.assertEqual(self.s.get(row['id'])['bundle'], 'invalid')

    def test_sync_failure_once_or_persistent_then_same_run_recovery(self):
        for parent, persistent in [('attempts', False), ('results', False), ('attempts', True), ('results', True)]:
            with self.subTest(parent=parent, persistent=persistent):
                w, row, scratch = self.prepared(assets={'nested/image.png': b'image'})
                root = w.storage.root
                events, faults = [], []
                real = os.fsync
                broken = True
                def fsync(fd):
                    path = os.readlink('/proc/self/fd/'+str(fd))
                    fail = path == str(root/parent) and broken and (persistent or not faults)
                    events.append((path, not fail))
                    if fail:
                        faults.append(path)
                        raise OSError(errno.EIO, 'sync injected')
                    real(fd)
                connect = w.storage.connect
                @contextmanager
                def observed_connect():
                    with connect() as conn:
                        conn.set_trace_callback(lambda sql: events.append(('completed SQL', True))
                                                if "SET state='completed'" in sql else None)
                        yield conn
                with patch('markdown_for_llms.files.os.fsync', side_effect=fsync), patch.object(w.storage, 'connect', observed_connect):
                    with patch.object(w, '_process', side_effect=lambda r: self.publish(w, r, scratch)):
                        w.process(row)
                    current = self.s.get(row['id'])
                    self.assertEqual(current['state'], 'publishing' if persistent else 'completed')
                    if persistent:
                        self.assertFalse(current['download_available'])
                        self.assertTrue(current['error'])
                        w.reconcile()
                        self.assertEqual(self.s.get(row['id'])['state'], 'publishing')
                        self.assertNotIn(('completed SQL', True), events)
                        broken = False
                        w.reconcile()
                    commit = events.index(('completed SQL', True))
                    for p in [root/'attempts', root/'results', root/'results'/row['attempt_id']/'original/document.md',
                              root/'results'/row['attempt_id']/'original/nested/image.png']:
                        self.assertIn((str(p), True), events[:commit])
                final = self.s.get(row['id'])
                self.assertEqual(final['id'], row['id'])
                self.assertEqual(final['state'], 'completed')
                self.assertIsNone(final['error'])
                self.assertEqual(identify(root, row['source_path'])['sha256'], row['source_sha256'])
                self.s.bundle(row['id'])

    def test_capacity_refusal_no_diagnostic_growth_then_same_id(self):
        for fault in ['quota', 'free-disk']:
            with self.subTest(fault=fault):
                w, row = self.pending()
                root = w.storage.root
                padding = root/'fixture-padding'
                if fault == 'quota':
                    free = self.settings.storage_mib*1048576-self.s.storage.capacity()-CONTROL
                    padding.write_bytes(b'x'*(free-16384))
                    context = patch('markdown_for_llms.capacity.CONTROL', CONTROL)
                else:
                    from collections import namedtuple
                    usage = namedtuple('usage', 'total used free')(10**12, 10**12, 0)
                    context = patch('markdown_for_llms.capacity.shutil.disk_usage', return_value=usage)
                before = sizes(root)
                before_error = self.s.storage.get(row['id'])['error']
                with context:
                    w.reconcile()
                    mid = self.s.get(row['id'])
                    self.assertEqual(mid['state'], 'publishing')
                    self.assertFalse(mid['download_available'])
                    self.assertIn('recovery', mid)
                    # No admission or diagnostic write on polling a full archive.
                    self.assertTrue(mid['error'])
                    self.assertEqual(before_error, self.s.storage.get(row['id'])['error'])
                self.assertEqual(before, sizes(root))
                if padding.exists(): padding.unlink()
                w.reconcile()
                self.assertEqual(self.s.get(row['id'])['state'], 'completed')
                self.s.bundle(row['id'])

    def test_sync_failure_and_capacity_refusal_in_same_path(self):
        w, row, scratch = self.prepared()
        from markdown_for_llms.worker import rename
        moved = False
        def move(*args):
            nonlocal moved
            rename(*args)
            moved = True
            raise OSError(errno.EIO, 'after rename')
        capacity = w.storage.reservation
        def reserve(kind, *args, **kwargs):
            if moved and kind == 'control': raise CapacityError('Synthetic quota refusal')
            return capacity(kind, *args, **kwargs)
        with patch('markdown_for_llms.worker.rename', side_effect=move), patch.object(w.storage, 'reservation', side_effect=reserve):
            with patch.object(w, '_process', side_effect=lambda r: self.publish(w, r, scratch)):
                w.process(row)
        self.assertEqual(self.s.get(row['id'])['state'], 'publishing')
        w.reconcile()
        self.assertEqual(self.s.get(row['id'])['state'], 'completed')

    def test_decoder_and_serializers_reject_nonfinite_depth_and_size(self):
        for body in [b'{"n":NaN}', b'{"n":Infinity}', b'{"n":-Infinity}', b'{"n":1e999}',
                     b'['*1500+b'0'+b']'*1500, b'['*200000, b' '*1048577]:
            with self.subTest(size=len(body)), self.assertRaises(MetadataError): decode_json(body)
        for value in [float('nan'), float('inf'), -float('inf')]:
            with self.assertRaises(ValueError): write_json(Path(self.tmp.name), 'should-not-exist', {'n': value})
        self.assertFalse((Path(self.tmp.name)/'should-not-exist').exists())

    def test_child_metadata_errors_visible_and_archive_readable(self):
        for stage, body in [('engine', b'['*1500+b'0'+b']'*1500), ('engine', b'{"n":NaN}'),
                            ('engine', b'{"n":1e999}'), ('raster', b'['*200000),
                            ('raster', b' '*1048577)]:
            with self.subTest(stage=stage, size=len(body)):
                w, row, scratch = self.prepared(assets={'image.png': b'opaque'})
                if stage == 'engine': (scratch/'engine.json').write_bytes(body)
                else: write(scratch, 'raster-result.json', body)
                def publish(r):
                    with patch('markdown_for_llms.worker.execute', return_value={'exit': 0, 'reason': None}):
                        w.publish(r, scratch, {}, {'elapsed_seconds': 0})
                with patch.object(w, '_process', side_effect=publish): w.process(row)
                result = self.s.get(row['id'])
                self.assertEqual(result['state'], 'failed')
                self.assertTrue(result['error'])
                self.assertIsNone(result['manifest'])
        other = self.s.submit(io.BytesIO(PDF), 'other.pdf')
        app = create_app(self.settings)
        self.addCleanup(app.state.service.storage.close)
        with TestClient(app, base_url='http://127.0.0.1:8765') as c:
            self.assertEqual(c.get('/api/v1/runs').status_code, 200)
            self.assertEqual(c.get('/api/v1/runs/'+other['id']).status_code, 200)

    def test_nonfinite_runtime_cannot_reach_publication(self):
        w, row, scratch = self.prepared()
        with patch.object(w, '_process', side_effect=lambda r: w.publish(r, scratch, {'n': float('inf')}, {'elapsed_seconds': 0})):
            w.process(row)
        self.assertEqual(self.s.get(row['id'])['state'], 'failed')
        self.assertIsNone(self.s.get(row['id'])['manifest'])

    def test_malformed_historical_manifest_does_not_break_list(self):
        w, row, scratch = self.prepared()
        self.publish(w, row, scratch)
        with self.s.storage.connect() as conn:
            conn.execute('UPDATE runs SET manifest=? WHERE id=?', ('{"n":Infinity}', row['id']))
        other = self.s.submit(io.BytesIO(PDF), 'other.pdf')
        jobs = self.s.list()
        damaged = next(j for j in jobs if j['id'] == row['id'])
        self.assertEqual(damaged['bundle'], 'invalid')
        self.assertFalse(damaged['download_available'])
        self.assertTrue(damaged['error'])
        self.assertEqual(self.s.get(other['id'])['state'], 'queued')

    def test_unsafe_failed_attempt_does_not_poison_archive(self):
        good_w, good_row, good_scratch = self.prepared()
        self.publish(good_w, good_row, good_scratch)
        outside = Path(self.tmp.name)/'outside'
        outside.mkdir()
        (outside/'secret').write_bytes(b'private')
        for kind in ['symlink', 'directory-link', 'hardlink', 'fifo', 'socket']:
            with self.subTest(kind=kind):
                w, row, scratch = self.prepared()
                special = scratch/'special'
                sock = None
                if kind == 'symlink': special.symlink_to(outside/'secret')
                if kind == 'directory-link': special.symlink_to(outside, target_is_directory=True)
                if kind == 'hardlink': os.link(outside/'secret', special)
                if kind == 'fifo': os.mkfifo(special)
                if kind == 'socket':
                    sock = socket.socket(socket.AF_UNIX)
                    self.addCleanup(sock.close)
                    previous = Path.cwd()
                    try:
                        os.chdir(scratch)
                        sock.bind('special')
                    finally:
                        os.chdir(previous)
                with patch.object(w, '_process', side_effect=lambda r: w.lease.check(local_only=True)):
                    w.process(row)
                self.assertEqual(self.s.get(row['id'])['state'], 'failed')
                self.assertTrue(special.lstat())
                self.s.storage.capacity()
                self.s.submit(io.BytesIO(PDF), 'new.pdf')
                self.s.bundle(good_row['id'])
                self.s.retry(row['id'])
                # Remove only queued synthetic follow-up jobs from selection,
                # leaving their sources and all failed partials untouched.
                with self.s.storage.connect() as conn:
                    conn.execute("UPDATE runs SET state='cancelled' WHERE state='queued'")
                if sock: sock.close()
        self.assertEqual((outside/'secret').read_bytes(), b'private')

    def test_unsafe_unattributed_and_published_paths_stay_strict(self):
        for base in ['sources', 'results', 'attempts/unattributed']:
            target = self.s.storage.root/base
            target.mkdir(exist_ok=True)
            (target/'evil').symlink_to('/etc/passwd')
            with self.assertRaises(FileSafetyError): self.s.storage.capacity()
            (target/'evil').unlink()

    def test_ticks_check_only_owned_budget_and_cancel_without_global_scan(self):
        w, row, scratch = self.prepared()
        with w.commitment(row) as lease:
            w.lease = lease
            with patch('markdown_for_llms.capacity.archive_sizes', side_effect=AssertionError('global scan at tick')):
                for _ in range(5): self.assertFalse(w.tick(row, None))
                self.s.cancel(row['id'])
                self.assertTrue(w.tick(row, None))
            w.lease = None

    def test_nonfile_safe_reference_forms_preserve_negative_paths(self):
        for form in ['.', './', 'docs/', '././', './././docs/']:
            for markup in ['[x]({})', '<a href="{}">x</a>']:
                self.assertEqual(inspect(markup.format(form), [])['bundle'], 'unchecked')
        self.assertEqual(inspect('![x](./././img.png)', ['img.png'])['bundle'], 'ready')
        for form in ['../x', '././../x', '/abs', '//host/x', '%252e%252e/x', 'file:///x', r'\\host\x']:
            self.assertEqual(inspect('[x]('+form+')', [])['bundle'], 'invalid')
        self.assertEqual(inspect('`[x](../x)`\n\n$$<img src="../x">$$', [])['bundle'], 'ready')

    def test_failure_before_completed_commit_repeats_barrier(self):
        w, row, scratch = self.prepared()
        connect = w.storage.connect
        events = []
        failed = False
        real_fsync = os.fsync
        def fsync(fd):
            events.append(os.readlink('/proc/self/fd/'+str(fd)))
            real_fsync(fd)
        class Proxy:
            def __init__(self, conn): self.conn = conn
            def execute(self, sql, *args):
                nonlocal failed
                if "SET state='completed'" in sql and not failed:
                    failed = True
                    events.append('failed before completed commit')
                    raise OSError(errno.EIO, 'commit boundary')
                return self.conn.execute(sql, *args)
        @contextmanager
        def connection():
            with connect() as conn: yield Proxy(conn)
        with patch.object(w.storage, 'connect', connection), patch('markdown_for_llms.files.os.fsync', side_effect=fsync):
            with patch.object(w, '_process', side_effect=lambda r: self.publish(w, r, scratch)):
                w.process(row)
        self.assertTrue(failed)
        failure = events.index('failed before completed commit')
        for parent in ['attempts', 'results']:
            self.assertIn(str(w.storage.root/parent), events[:failure])
            self.assertIn(str(w.storage.root/parent), events[failure+1:])
        self.assertEqual(self.s.get(row['id'])['state'], 'completed')

    def test_transient_errno_at_all_http_reads_and_publishing(self):
        w, row, scratch = self.prepared()
        self.publish(w, row, scratch)
        app = create_app(self.settings)
        self.addCleanup(app.state.service.storage.close)
        with TestClient(app, base_url='http://127.0.0.1:8765') as c:
            for err in [errno.EMFILE, errno.EIO, errno.ENOMEM, errno.EACCES]:
                for endpoint in ['source', 'text']:
                    with self.subTest(errno=err, endpoint=endpoint), patch('markdown_for_llms.api.open_file', side_effect=OSError(err, 'injected')):
                        response = c.get('/api/v1/runs/'+row['id']+'/'+endpoint)
                        self.assertEqual(response.status_code, 503)
                        self.assertTrue(response.json()['retryable'])
                    self.assertEqual(self.s.get(row['id'])['bundle'], 'ready')
                pending, prow = self.pending()
                with patch('markdown_for_llms.worker.identify', side_effect=OSError(err, 'injected')):
                    pending.reconcile(prow['id'])
                self.assertEqual(self.s.get(prow['id'])['state'], 'publishing')
                pending.reconcile(prow['id'])
                self.assertEqual(self.s.get(prow['id'])['state'], 'completed')
            self.assertEqual(c.get('/api/v1/runs/'+row['id']+'/bundle').status_code, 200)
