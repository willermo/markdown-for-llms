"""Behavioral contracts for the installed experimental application; no ML imports."""
import io
import json
import os
import sqlite3
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import uuid
import zipfile

from markdown_for_llms.api import create_app
from markdown_for_llms.files import identify, open_file, relative, write, write_json
from markdown_for_llms.references import inspect
from markdown_for_llms.services import Service
from markdown_for_llms.settings import ConfigurationError, Settings, options
from markdown_for_llms.storage import StorageError
from markdown_for_llms.worker import Worker
from fastapi.testclient import TestClient

FIXTURES = json.loads(Path(__file__).with_name('reference_contract.json').read_text())['cases']
PDF = b'%PDF-1.4\nsynthetic contract input; not a valid conversion fixture\n'


class ProductTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.settings = Settings(data_root=self.temp.name + '/data', model_root=self.temp.name + '/models')
        self.service = Service(self.settings)

    def completed(self, raw='hello', assets=None):
        job = self.service.submit(io.BytesIO(PDF), 'same.pdf')
        worker = Worker(self.settings)
        row = worker.claim()
        scratch = worker.storage.root / 'attempts' / row['attempt_id']
        original = scratch / 'original'
        original.mkdir(parents=True)
        write(original, 'document.md', raw.encode())
        for name, content in (assets or {}).items():
            (original / name).parent.mkdir(exist_ok=True, parents=True)
            write(original, name, content)
        inventory = [identify(original, name) for name in ['document.md', *(assets or {})]]
        write_json(scratch, 'engine.json', {'inventory': inventory, 'inventory_complete': True})
        with patch('markdown_for_llms.worker.execute', return_value={'exit': 1, 'reason': 'synthetic gallery unavailable'}):
            worker.publish(row, scratch, {'test': 'mock'}, {'elapsed_seconds': 0})
        return self.service.get(job['id'])

    def test_reference_fixture_contracts(self):
        for fixture in FIXTURES:
            with self.subTest(case=fixture['id']):
                result = inspect(fixture['raw'], fixture['inventory'])
                expected = fixture['expected']['bundle']
                if fixture['fault'] != 'digest-mismatch':
                    self.assertEqual(result['bundle'], expected)
                self.assertIn(fixture['expected']['quality'], {'unverified'})

    def test_fixture_publication_manifest_download_http_cli(self):
        for fixture in FIXTURES:
            with self.subTest(case=fixture['id']):
                job = self.completed(fixture['raw'], {n: b'opaque asset' for n in fixture['inventory']})
                if fixture['fault'] == 'digest-mismatch':
                    row = self.service.storage.get(job['id'])
                    with open_file(self.service.storage.root, row['result_path'] + '/original/figure.jpeg', 'r+b') as f:
                        f.write(b'tampered')
                if fixture['expected']['ordinary_download']:
                    bundle = self.service.bundle(job['id'])
                    with zipfile.ZipFile(bundle) as archive:
                        manifest = json.loads(archive.read('manifest.json'))
                        self.assertEqual(archive.read('result/document.md'), fixture['raw'].encode())
                        self.assertEqual(manifest['bundle'], fixture['expected']['bundle'])
                        self.assertEqual(manifest['quality'], 'unverified')
                else:
                    with self.assertRaises(StorageError):
                        self.service.bundle(job['id'])
                job = self.service.get(job['id'])
                self.assertEqual(job['bundle'], fixture['expected']['bundle'])
                app = create_app(self.settings)
                client = TestClient(app, base_url='http://127.0.0.1:8765')
                self.assertEqual(client.get('/api/v1/runs/' + job['id']).json()['bundle'], job['bundle'])
                result = subprocess.run([sys.executable, '-I', '-B', '-m', 'markdown_for_llms.cli', '--data-root', self.settings.data_root, '--model-root', self.settings.model_root, 'show', job['id']], capture_output=True, text=True, cwd=self.temp.name)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(json.loads(result.stdout)['bundle'], job['bundle'])

    def test_idempotency_and_names(self):
        a = self.service.submit(io.BytesIO(PDF), 'same.pdf', key='key')
        b = self.service.submit(io.BytesIO(PDF), 'same.pdf', key='key')
        c = self.service.submit(io.BytesIO(PDF + b'other'), 'same.pdf')
        self.assertEqual(a['id'], b['id'])
        self.assertNotEqual(a['document_id'], c['document_id'])
        with self.assertRaises(StorageError):
            self.service.submit(io.BytesIO(PDF + b'other'), 'same.pdf', key='key')

    def test_claim_cancel_and_retry(self):
        job = self.service.submit(io.BytesIO(PDF), 'test.pdf')
        worker = Worker(self.settings)
        row = worker.claim()
        self.assertEqual(row['id'], job['id'])
        self.assertIsNone(Worker(self.settings).claim())
        self.service.cancel(job['id'])
        self.assertTrue(worker.tick(row, None))
        worker.reconcile()
        retry = self.service.retry(job['id'], {'force_ocr': True})
        self.assertNotEqual(retry['attempt_id'], job['attempt_id'])
        self.assertEqual(retry['retry_of'], job['id'])
        self.assertTrue(retry['options']['force_ocr'])

    def test_safe_paths_and_opening(self):
        for name in ['../x','/etc/passwd','a/../b','a//b','C:/x','x\\y','x\x00','x%2fy','file:x','x?y','x#y']:
            with self.subTest(name=name), self.assertRaises(ValueError):
                relative(name)
        outside = Path(self.temp.name) / 'outside'
        outside.write_text('private')
        root = self.service.storage.root
        (root / 'symlink').symlink_to(outside)
        with self.assertRaises(OSError):
            with open_file(root, 'symlink'):
                self.fail('Symlink opened')
        (root / 'parent').symlink_to(outside.parent, target_is_directory=True)
        with self.assertRaises(OSError):
            with open_file(root, 'parent/outside'):
                self.fail('Parent symlink opened')
        write(root, 'legal name-α.txt', b'ok')
        self.assertEqual(identify(root, 'legal name-α.txt')['bytes'], 2)

    def test_configuration(self):
        env = Path(self.temp.name) / '.env'
        env.write_text('MDF_PORT=8766\nMDF_TIMEOUT_SECONDS=20\n')
        settings = Settings.load(env, {'port': 8768}, {'MDF_PORT': '8767'})
        self.assertEqual(settings.port, 8768)
        self.assertEqual(settings.timeout_seconds, 20)
        for values in [{'MDF_UNKNOWN': 'secret'}, {'MDF_PORT': 'bad'}, {'MDF_HOST': '0.0.0.0'}, {'MDF_TIMEOUT_SECONDS': '-1'}]:
            with self.assertRaises(ConfigurationError):
                Settings.load(environ=values)
        with self.assertRaises(ConfigurationError):
            options({'force_ocr': 'false'})
        for data, models in [('/tmp/model/data', '/tmp/model'), ('/tmp/data', '/tmp/data/models')]:
            with self.assertRaises(ConfigurationError):
                Settings(data_root=data, model_root=models)
        with self.assertRaises(StorageError):
            Service(Settings(data_root=self.settings.data_root, port=8766))

    def test_host_origin_csrf(self):
        client = TestClient(create_app(self.settings), base_url='http://127.0.0.1:8765')
        self.assertEqual(client.get('/health', headers={'host':'evil.example'}).status_code,403)
        csrf = client.get('/api/v1/session').json()['csrf']
        url='/api/v1/runs'
        self.assertEqual(client.post(url,content=PDF).status_code,403)
        self.assertEqual(client.post(url,content=PDF,headers={'origin':'https://evil.example','x-mdf-csrf':csrf}).status_code,403)
        response=client.post(url,content=PDF,headers={'origin':'http://127.0.0.1:8765','x-mdf-csrf':csrf,'x-filename':'ok.pdf'})
        self.assertEqual(response.status_code,202,response.text)
        self.assertIn("default-src 'none'", response.headers['content-security-policy'])
        self.assertEqual(client.get('/docs').status_code,404)

    def test_no_ml_imports(self):
        result=subprocess.run([sys.executable,'-I','-B','-c',"import markdown_for_llms.api,markdown_for_llms.cli,sys; assert not {'torch','marker','surya','PIL'} & sys.modules.keys()"],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)

    def test_missing_source_and_published_asset_reconciliation(self):
        job = self.completed('![x](figure.jpeg)', {'figure.jpeg': b'original'})
        row = self.service.storage.get(job['id'])
        (self.service.storage.root / row['result_path'] / 'original/figure.jpeg').unlink()
        Worker(self.settings).reconcile()
        self.assertEqual(self.service.get(job['id'])['bundle'], 'invalid')
        (self.service.storage.root / row['source_path']).unlink()
        client = TestClient(create_app(self.settings), base_url='http://127.0.0.1:8765')
        self.assertEqual(client.get('/api/v1/runs/' + job['id'] + '/source').status_code, 409)

    def test_publication_crash_boundaries(self):
        from markdown_for_llms.files import rename as real_rename
        for stage in ('before_manifest', 'before_rename', 'after_rename'):
            with self.subTest(stage=stage):
                job = self.service.submit(io.BytesIO(PDF), 'crash.pdf')
                worker = Worker(self.settings)
                row = worker.claim()
                scratch = worker.storage.root / 'attempts' / row['attempt_id']
                original = scratch / 'original'
                original.mkdir(parents=True)
                write(original, 'document.md', b'untouched content')
                write_json(scratch, 'engine.json', {'inventory': [identify(original, 'document.md')], 'inventory_complete': True})
                def crash(*args, **kwargs):
                    if stage == 'after_rename':
                        real_rename(*args, **kwargs)
                    raise RuntimeError('simulated crash')
                target = 'markdown_for_llms.worker.write_json' if stage == 'before_manifest' else 'markdown_for_llms.worker.rename'
                with patch(target, side_effect=crash), self.assertRaises(RuntimeError):
                    worker.publish(row, scratch, {'mock': True}, {'elapsed_seconds': 0})
                Worker(self.settings).reconcile()
                final = self.service.get(job['id'])
                self.assertEqual(final['state'], 'interrupted' if stage == 'before_manifest' else 'completed')
                self.assertEqual(self.service.storage.get(job['id'])['source_sha256'], __import__('hashlib').sha256(PDF).hexdigest())

    def test_worker_lock_and_stale_token(self):
        with self.service.storage.lock('worker.lock'):
            with self.assertRaises(BlockingIOError):
                with self.service.storage.lock('worker.lock', blocking=False):
                    self.fail('Second owner admitted')
        job = self.service.submit(io.BytesIO(PDF), 'token.pdf')
        worker = Worker(self.settings)
        row = worker.claim()
        with self.service.storage.connect() as conn:
            conn.execute("UPDATE runs SET token='new-owner' WHERE id=?", (row['id'],))
        self.assertTrue(worker.tick(row, None))
        self.assertEqual(self.service.get(job['id'])['state'], 'running')

    def test_future_schema_and_storage_identity(self):
        with self.service.storage.connect() as conn:
            conn.execute('PRAGMA user_version=99')
        with self.assertRaises(StorageError):
            Service(self.settings)

    def test_concurrent_short_connections_and_admission(self):
        from concurrent.futures import ThreadPoolExecutor
        def use_storage(index):
            for step in range(20):
                if index == 0:
                    self.service.submit(io.BytesIO(PDF), 'concurrent.pdf', key='admission-' + str(step))
                else:
                    self.service.list()
                    self.service.storage.capacity()
        with ThreadPoolExecutor(max_workers=6) as pool:
            list(pool.map(use_storage, range(6)))
        self.assertEqual(len(self.service.list()), 20)

    def test_storage_identity_mismatch(self):
        with self.service.storage.connect() as conn:
            conn.execute("UPDATE metadata SET value='different-storage'")
        with self.assertRaises(StorageError):
            Service(self.settings)

    def test_unsafe_special_files_and_collisions(self):
        root = self.service.storage.root
        os.mkfifo(root / 'fifo')
        with self.assertRaises(ValueError):
            with open_file(root, 'fifo'):
                self.fail('FIFO admitted')
        write(root, 'first', b'private')
        os.link(root / 'first', root / 'hardlink')
        with self.assertRaises(ValueError):
            identify(root, 'hardlink')
        with self.assertRaises(FileExistsError):
            write(root, 'first', b'overwrite')

    def test_quotas_and_unsupported_pdf(self):
        from collections import namedtuple
        Usage = namedtuple('Usage', 'total used free')
        with patch('markdown_for_llms.storage.shutil.disk_usage', return_value=Usage(100,99,1)):
            with self.assertRaises(StorageError):
                self.service.submit(io.BytesIO(PDF), 'space.pdf')
        for payload, name in [(b'plain text', 'x.pdf'), (PDF, 'x.txt')]:
            with self.assertRaises(StorageError):
                self.service.submit(io.BytesIO(payload), name)

    def test_additional_reference_forms(self):
        cases = [('[x][r]\n\n[r]: a.png', ['a.png'], 'ready'),
                 ('~~~html\n<img src="missing">\n~~~', [], 'ready'),
                 ('$$<img src="missing">$$', [], 'ready'),
                 ('<img src="x" src="y">', [], 'incomplete'),
                 ('<video poster="a.png"></video>', ['a.png'], 'unchecked'),
                 ('<p style="background:url(x)">x</p>', [], 'unchecked'),
                 ('<img src="missing', [], 'unchecked'),
                 ('&lt;img src="missing">', [], 'ready'),
                 ('![x](%2e%2e/x)', [], 'invalid'),
                 ('[x](file:///etc/passwd)', [], 'invalid'),
                 ('[x](C:/path)', [], 'invalid')]
        for raw, inventory, expected in cases:
            with self.subTest(raw=raw):
                self.assertEqual(inspect(raw, inventory)['bundle'], expected)

    def test_link_destination_not_a_file_name(self):
        warning = 'Local link destination is not a file name; not checked as an asset'
        cases = [('k[[y, z, v]](y,z)', 'unchecked', [('unchecked', 'y,z')]),
                 ('[x](docs/notes)', 'unchecked', [('unchecked', 'docs/notes')]),
                 ('<a href="docs/notes">x</a>', 'unchecked', [('unchecked', 'docs/notes')]),
                 ('[x](./docs/notes#part)', 'unchecked', [('unchecked', './docs/notes')]),
                 ('![x](missing.jpeg)', 'incomplete', [('incomplete', 'missing.jpeg')]),
                 ('[x](missing.jpeg)', 'incomplete', [('incomplete', 'missing.jpeg')]),
                 ('<img src="x">', 'incomplete', [('incomplete', 'x')]),
                 ('![x](docs/notes) [x](docs/notes)', 'incomplete',
                  [('incomplete', 'docs/notes'), ('unchecked', 'docs/notes')])]
        for raw, expected, issues in cases:
            with self.subTest(raw=raw):
                result = inspect(raw, [])
                self.assertEqual(result['bundle'], expected)
                self.assertEqual([(i['code'], i['path']) for i in result['issues']], issues)
                for issue in result['issues']:
                    if issue['code'] == 'unchecked':
                        self.assertEqual(issue['reason'], warning)
        self.assertEqual([r['kind'] for r in inspect('k[[y, z, v]](y,z)', [])['references']], ['local-unresolved'])
        job = self.completed('k[[y, z, v]](y,z)')
        self.assertEqual((job['state'], job['bundle'], job['download_available']), ('completed', 'unchecked', True))
        with zipfile.ZipFile(self.service.bundle(job['id'])) as archive:
            self.assertEqual(json.loads(archive.read('manifest.json'))['bundle'], 'unchecked')
            self.assertEqual(archive.read('result/document.md'), b'k[[y, z, v]](y,z)')

    def test_tracked_fixture_markdown_outcomes_unchanged(self):
        # Outcomes of the code before the link rule (17de44f), with an empty inventory
        # and with the tracked files next to each document as inventory.
        root = Path(__file__).resolve().parents[2]
        missing = lambda path: [{'code': 'incomplete', 'reason': 'Missing local reference', 'path': path}]
        baseline = {'tests/fixtures/conversion-corpus/README.md': ('ready', []),
                    'tests/fixtures/conversion-corpus/controls/K01-full/document.md': ('incomplete', missing('assets/figure-it.png')),
                    'tests/fixtures/conversion-corpus/controls/K02-three-sections/document.md': ('ready', []),
                    'tests/fixtures/conversion-corpus/controls/K03-repeated/document.md': ('ready', []),
                    'tests/fixtures/run-a001-fase0-uv/f1-stable.md': ('ready', []),
                    'tests/fixtures/run-a001-fase0-uv/f2-sensitive.md': ('incomplete', missing('assets/f5.png')),
                    'tests/fixtures/run-a001-fase0-uv/f3-cleaned-sensitive.md': ('incomplete', missing('assets/f5.png')),
                    'tests/fixtures/run-a001-fase0-uv/f3-cleaned-stable.md': ('ready', []),
                    'tests/fixtures/run-a001-fase0-uv/f5-asset.md': ('incomplete', missing('assets/f5.png')),
                    'tests/fixtures/run-a001-fase0-uv/f6-multichunk.md': ('ready', [])}
        listed = lambda *spec: subprocess.run(['git', '-C', str(root), 'ls-files', '-z', *spec], check=True,
                                              capture_output=True).stdout.decode().split('\0')[:-1]
        self.assertEqual(sorted(listed('tests/fixtures/*.md')), sorted(baseline))
        tracked = listed('tests/fixtures')
        for name, (bundle, issues) in baseline.items():
            with self.subTest(fixture=name):
                raw = (root / name).read_text(encoding='utf-8')
                result = inspect(raw, [])
                self.assertEqual((result['bundle'], result['issues']), (bundle, issues))
                folder = Path(name).parent
                inventory = [str(Path(t).relative_to(folder)) for t in tracked
                             if Path(t).is_relative_to(folder) and t != name]
                result = inspect(raw, inventory)
                self.assertEqual((result['bundle'], result['issues']), ('ready', []))

    def test_full_utf8_pagination_and_manifest_integrity(self):
        raw = ('αè正文\n' * 50000) + 'END-SENTINEL'
        job = self.completed(raw)
        client = TestClient(create_app(self.settings), base_url='http://127.0.0.1:8765')
        offset, parts = 0, []
        while True:
            response = client.get('/api/v1/runs/' + job['id'] + '/text', params={'offset': offset})
            self.assertEqual(response.status_code, 200, response.text)
            data = response.json()
            parts.append(data['text'])
            self.assertGreater(data['next_offset'], offset)
            offset = data['next_offset']
            if data['complete']:
                break
        self.assertEqual(''.join(parts), raw)
        row = self.service.storage.get(job['id'])
        (self.service.storage.root / row['result_path'] / 'manifest.json').unlink()
        with self.assertRaises(StorageError):
            self.service.bundle(job['id'])
        self.assertEqual(self.service.get(job['id'])['bundle'], 'invalid')

    def test_failed_engine_and_empty_output_never_succeed(self):
        for fault in ('abnormal_exit', 'empty_output', 'publication_io_error'):
            with self.subTest(fault=fault):
                job = self.service.submit(io.BytesIO(PDF), 'fault.pdf')
                worker = Worker(self.settings)
                row = worker.claim()
                def engine(settings, scratch, source, **kwargs):
                    if fault == 'abnormal_exit':
                        return {'reason': None, 'exit': 17}
                    original = scratch / 'original'
                    original.mkdir()
                    write(original, 'document.md', b'' if fault == 'empty_output' else b'preserved content')
                    write_json(scratch, 'engine.json', {'inventory': [identify(original, 'document.md')], 'inventory_complete': True})
                    return {'reason': None, 'exit': 0, 'elapsed_seconds': 0}
                from markdown_for_llms.files import rename as real_rename
                def rename_or_fail(*args, **kwargs):
                    if fault == 'publication_io_error':
                        raise OSError('controlled publication failure')
                    return real_rename(*args, **kwargs)
                with patch('markdown_for_llms.worker.check_runtime', return_value={'mock': True}), patch('markdown_for_llms.worker.execute', side_effect=engine), patch('markdown_for_llms.worker.rename', side_effect=rename_or_fail):
                    worker.process(row)
                final = self.service.get(job['id'])
                # A complete publishing result can recover through the shared barrier.
                self.assertEqual(final['state'], 'completed' if fault == 'publication_io_error' else 'failed')
                self.assertEqual(final['download_available'], fault == 'publication_io_error')
                if fault == 'publication_io_error':
                    self.service.bundle(job['id'])
                else:
                    with self.assertRaises(StorageError):
                        self.service.bundle(job['id'])

    def test_engine_asset_error_marks_bundle_incomplete(self):
        job = self.service.submit(io.BytesIO(PDF), 'error.pdf')
        worker = Worker(self.settings)
        row = worker.claim()
        scratch = worker.storage.root / 'attempts' / row['attempt_id']
        original = scratch / 'original'
        original.mkdir(parents=True)
        write(original, 'document.md', b'partial content')
        write_json(scratch, 'engine.json', {'inventory': [identify(original, 'document.md')], 'inventory_complete': True, 'issues': [{'code': 'asset_error'}]})
        worker.publish(row, scratch, {'mock': True}, {'elapsed_seconds': 0})
        self.assertEqual(self.service.get(job['id'])['bundle'], 'incomplete')
        with self.assertRaises(StorageError):
            self.service.bundle(job['id'])


if __name__ == '__main__':
    unittest.main()
