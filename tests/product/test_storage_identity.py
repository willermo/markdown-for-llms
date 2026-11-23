"""Storage identity and database schema 3: automatic upgrade from 1 and 2, configure and the return
to schema 2 (ADR 0011, K5 of MFL032)."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stdout
from dataclasses import replace
import ast
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import sqlite3
import tempfile
import threading
import unittest
from unittest.mock import patch
import uuid
import zipfile

from markdown_for_llms import storage
from markdown_for_llms.cli import main
from markdown_for_llms.files import identify, write, write_json
from markdown_for_llms.services import Service
from markdown_for_llms.settings import Settings
from markdown_for_llms.storage import Storage, StorageError
from markdown_for_llms.worker import Worker

# Identity schemas 1 and 2 as computed at 17de44f and 59ed8ca, frozen here independently of the
# code under test (markdown_for_llms/legacy_identity.py).
V1_FIELDS = ("data_root", "model_root", "host", "port", "timeout_seconds", "memory_mib",
             "upload_mib", "max_pages", "output_mib", "log_mib", "storage_mib", "reserve_mib",
             "max_pixels", "max_dimension", "threads", "poll_seconds", "lease_seconds",
             "grace_seconds", "process_limit", "bind_profile", "review_provider")
MIGRATIONS = Path(__file__).parents[2] / 'markdown_for_llms' / 'migrations'
MIGRATION_001 = (MIGRATIONS / '001.sql').read_text()
MIGRATION_002 = (MIGRATIONS / '002.sql').read_text()
V2_OPTIONS = dict(use_llm=False, force_ocr=False, strip_existing_ocr=False, disable_image_extraction=False,
                  paginate_output=True, pdftext_workers=1, layout_batch_size=1, recognition_batch_size=1,
                  detection_batch_size=1, table_rec_batch_size=1, ocr_error_batch_size=1,
                  lowres_image_dpi=96, highres_image_dpi=192)
PDF = b'%PDF-1.4\nsynthetic identity input\n'


def legacy_value(settings, fields, model_root=None):
    value = {name: getattr(settings, name) for name in fields}
    value['bind_profile'] = 'marker-cpu-v1'
    value['model_root'] = settings.model_root if model_root is None else model_root
    return value


def v1_identity(settings, timeout=450, model_root=None):
    value = legacy_value(settings, V1_FIELDS, model_root)
    value['timeout_seconds'] = timeout
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def v2_identity(settings, model_root=None):
    value = legacy_value(settings, [f for f in V1_FIELDS if f != 'timeout_seconds'], model_root)
    value['identity_schema'] = 2
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def catalog(conn):
    return conn.execute("SELECT type,name,tbl_name,sql FROM sqlite_master ORDER BY name").fetchall()


class StorageIdentityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.settings = Settings(data_root=self.tmp.name + '/data', model_root=self.tmp.name + '/models',
                                 upload_mib=1, output_mib=3, log_mib=1, storage_mib=16, reserve_mib=1)
        self.root = Path(self.settings.data_root)

    def make_legacy(self, schema=1, timeout=450, settings=None, model_root=None):
        """An archive as a previous version leaves it: identity and database of schema 1 (17de44f)
        or 2 (59ed8ca), one failed run with the job options of schema 2."""
        settings = settings or self.settings
        self.root.mkdir(mode=0o700)
        for name in ("sources", "attempts", "results", "staging", "downloads", "operations"):
            (self.root / name).mkdir(mode=0o700)
        storage_id = uuid.uuid4().hex
        config = v1_identity(settings, timeout, model_root) if schema == 1 else v2_identity(settings, model_root)
        (self.root / 'identity.json').write_text(json.dumps({'id': storage_id, 'config': config}))
        conn = sqlite3.connect(self.root / 'archive.sqlite3')
        conn.executescript(MIGRATION_001 + (MIGRATION_002 if schema == 2 else ''))
        conn.execute("INSERT INTO metadata VALUES ('storage_id',?)", (storage_id,))
        conn.execute("INSERT INTO documents VALUES ('d1','old.pdf','sources/old.pdf',?,3,1.0)", ('0' * 64,))
        conn.execute("INSERT INTO runs (id,document_id,attempt_id,options,state,created,updated,error) "
                     "VALUES ('r1','d1','a1',?,'failed',1.0,2.0,'old failure')", (json.dumps(V2_OPTIONS),))
        conn.commit()
        conn.close()

    def snapshot(self):
        conn = sqlite3.connect(self.root / 'archive.sqlite3')
        try:
            return ((self.root / 'identity.json').read_bytes(), conn.execute('PRAGMA user_version').fetchone()[0],
                    catalog(conn), conn.execute('SELECT id,document_id,attempt_id,options,state,created,updated,error FROM runs').fetchall())
        finally:
            conn.close()

    def identity(self):
        return json.loads((self.root / 'identity.json').read_text())

    def columns(self, table):
        with sqlite3.connect(self.root / 'archive.sqlite3') as conn:
            return [row[1] for row in conn.execute('PRAGMA table_info(%s)' % table)]

    def test_v1_archive_upgrades_through_schema_2_on_first_open(self):
        self.make_legacy(1)
        before = self.snapshot()
        service = Service(self.settings)
        service.storage.close()
        identity = self.identity()
        self.assertEqual(identity['config'], self.settings.identity())
        self.assertEqual([(e['config'], e['reason']) for e in identity['history']],
                         [(v1_identity(self.settings), 'identity-v2'), (v2_identity(self.settings), 'identity-v3')])
        self.assertEqual(identity['history'][1]['model_root'], self.settings.model_root)
        after = self.snapshot()
        self.assertEqual(after[1], storage.SCHEMA)
        self.assertEqual(after[3], before[3])
        self.assertIn('progress', self.columns('runs'))
        self.assertEqual([c for c in self.columns('runs') if c in ('owner', 'scope_recorded')], ['owner', 'scope_recorded'])
        self.assertIn('owner', self.columns('documents'))
        with sqlite3.connect(self.root / 'archive.sqlite3') as conn:
            # an attempt born before tranche C has no scope indicator: NULL, not 0
            self.assertEqual(conn.execute("SELECT owner,scope_recorded,progress FROM runs WHERE id='r1'").fetchone(),
                             ('local', None, None))
            self.assertEqual(conn.execute("SELECT owner FROM documents WHERE id='d1'").fetchone(), ('local',))
        self.assertEqual(Service(self.settings).get('r1')['error'], 'old failure')
        # schema 3: time limits, port and model root change without configure; any other setting needs it
        for other in (dict(timeout_seconds=900), dict(port=8799), dict(model_root=self.tmp.name + '/other-models'),
                      dict(datalab_cap_usd='0.10')):
            with self.subTest(other=other):
                Service(replace(self.settings, **other)).storage.close()
        with self.assertRaisesRegex(StorageError, 'configuration mismatch'):
            Service(replace(self.settings, memory_mib=8192))
        self.assertEqual(len(self.identity()['history']), 2)

    def test_v2_archive_upgrades_on_first_open(self):
        self.make_legacy(2)
        before = self.snapshot()
        Service(self.settings).storage.close()
        identity = self.identity()
        self.assertEqual(identity['config'], self.settings.identity())
        self.assertEqual([(e['config'], e['reason']) for e in identity['history']],
                         [(v2_identity(self.settings), 'identity-v3')])
        self.assertEqual((self.snapshot()[1], self.snapshot()[3]), (storage.SCHEMA, before[3]))

    def test_v1_archive_with_configured_timeout(self):
        self.make_legacy(1, timeout=5400)
        with self.assertRaisesRegex(StorageError, 'MDF_TIMEOUT_SECONDS'):
            Service(self.settings)
        self.assertEqual(self.snapshot()[1], 1)
        Service(replace(self.settings, timeout_seconds=5400)).storage.close()
        self.assertEqual(self.identity()['history'][0]['config'], v1_identity(self.settings, 5400))
        Service(self.settings).storage.close()

    def test_legacy_archive_of_the_former_default_port_needs_it_once(self):
        former = replace(self.settings, port=23460)
        self.assertEqual(self.settings.port, 23465)
        self.make_legacy(2, settings=former)
        before = (self.root / 'identity.json').read_bytes()
        with self.assertRaises(StorageError) as raised:
            Storage(self.settings)
        self.assertIn('MDF_PORT', str(raised.exception))
        self.assertIn('23460', str(raised.exception))
        self.assertEqual((self.root / 'identity.json').read_bytes(), before)
        Storage(former).close()                        # upgraded to schema 3: the port leaves the identity
        Storage(self.settings).close()

    def test_former_default_model_root_by_its_absolute_path(self):                  # FD4 of the fix of C
        # an archive created with ./mdf-models of another directory: the default model root of
        # this directory does not open it, the absolute path of that mdf-models does
        elsewhere = self.tmp.name + '/old-cwd/mdf-models'
        self.make_legacy(2, model_root=elsewhere)
        with self.assertRaises(StorageError) as raised:
            Storage(self.settings)
        self.assertIn('MDF_MODEL_ROOT set to the absolute path of that mdf-models directory', str(raised.exception))
        self.assertIn('./mdf-models from the directory the archive was used from', str(raised.exception))
        Storage(replace(self.settings, model_root=elsewhere)).close()
        self.assertEqual(self.identity()['history'][-1]['model_root'], elsewhere)
        Storage(self.settings).close()                                  # schema 3: the model root is operational

    def test_legacy_archive_of_the_former_default_model_root(self):
        # created by a previous version with MDF_MODEL_ROOT unset (./mdf-models of the working directory)
        former = str(Path('./mdf-models').absolute())
        self.make_legacy(2, model_root=former)
        Storage(self.settings).close()
        self.assertEqual(self.identity()['history'][-1]['model_root'], former)
        self.assertEqual(self.identity()['config'], self.settings.identity())

    def test_new_archive_is_born_with_schema_3(self):
        Service(self.settings).storage.close()
        self.assertEqual(self.identity()['config'], self.settings.identity())
        self.assertNotIn('history', self.identity())
        self.assertEqual(self.snapshot()[1], storage.SCHEMA)

    def test_the_default_model_root_is_the_one_of_the_engine(self):            # D10
        from markdown_for_llms.engine.weights import default_root
        with patch.dict(os.environ, {'XDG_CACHE_HOME': self.tmp.name + '/cache'}):
            self.assertEqual(Settings(data_root=self.tmp.name + '/d').model_root, str(default_root()))
            self.assertEqual(Settings(data_root=self.tmp.name + '/d').model_root,
                             self.tmp.name + '/cache/markdown-for-llms/models')

    def test_scope_indicator_is_monotonic(self):
        service = Service(self.settings)
        job = service.submit(io.BytesIO(PDF), 'a.pdf')
        with service.storage.connect() as conn:
            self.assertEqual(conn.execute('SELECT scope_recorded FROM runs WHERE id=?', (job['id'],)).fetchone()[0], 0)
            conn.execute('UPDATE runs SET scope_recorded=1 WHERE id=?', (job['id'],))
        for value in (0, None):
            with self.subTest(value=value), self.assertRaisesRegex(sqlite3.IntegrityError, 'monotonic'):
                with service.storage.connect() as conn:
                    conn.execute('UPDATE runs SET scope_recorded=? WHERE id=?', (value, job['id']))
        service.storage.close()

    def test_live_previous_version_process_refuses_upgrade_without_changes(self):
        # (a) The previous version holds configuration.lock shared for its whole life.
        self.make_legacy(2)
        before = self.snapshot()
        fd = os.open(self.root / 'configuration.lock', os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_SH)
            with self.assertRaisesRegex(StorageError, 'stop the API, worker and CLI of every version'):
                Service(self.settings)
            self.assertEqual(self.snapshot(), before)
        finally:
            os.close(fd)
        Service(self.settings).storage.close()
        self.assertEqual(self.snapshot()[1], storage.SCHEMA)

    def test_concurrent_new_openers_upgrade_once(self):
        # (b) Racing openers either upgrade, observe the upgrade or are refused; never twice.
        self.make_legacy(1)
        barrier = threading.Barrier(4)
        def open_storage(_):
            barrier.wait()
            try:
                Storage(self.settings).close()
                return 'opened'
            except StorageError as exc:
                return str(exc)
        with ThreadPoolExecutor(max_workers=4) as pool:
            outcomes = list(pool.map(open_storage, range(4)))
        for outcome in outcomes:
            self.assertIn(outcome, {'opened', 'Storage upgrade needs exclusive access: stop the API, worker and CLI of every version, then restart'})
        self.assertIn('opened', outcomes)
        self.assertEqual([e['reason'] for e in self.identity()['history']], ['identity-v2', 'identity-v3'])
        self.assertEqual(self.snapshot()[1], storage.SCHEMA)
        Storage(self.settings).close()
        self.assertEqual([e['reason'] for e in self.identity()['history']], ['identity-v2', 'identity-v3'])

    def test_interrupted_upgrade_is_completed_by_the_next_opening(self):
        # (c) The database is migrated first; identity.json is rewritten last.
        for schema in (1, 2):
            with self.subTest(schema=schema):
                if self.root.exists():
                    import shutil
                    shutil.rmtree(self.root)
                self.make_legacy(schema)
                before = self.snapshot()
                with patch.object(Storage, 'replace_identity', side_effect=OSError('synthetic interruption')):
                    with self.assertRaises(OSError):
                        Service(self.settings)
                interrupted = self.snapshot()
                self.assertEqual((interrupted[0], interrupted[1], interrupted[3]), (before[0], storage.SCHEMA, before[3]))
                self.assertEqual(list(self.root.glob('configuration-*.pending')), [])
                Service(self.settings).storage.close()
                self.assertEqual(self.identity()['config'], self.settings.identity())
                self.assertEqual([e['reason'] for e in self.identity()['history']],
                                 ['identity-v2', 'identity-v3'][schema - 1:])

    def test_configure_checks_expected_before_upgrading(self):
        # (d) A wrong expected identity on a legacy archive changes nothing.
        self.make_legacy(2)
        before = self.snapshot()
        for expected in ('wrong', self.settings.identity()):
            with self.assertRaisesRegex(StorageError, 'Previous configuration identity mismatch'):
                Storage.configure(self.settings, expected, storage_mib=20)
            self.assertEqual(self.snapshot(), before)
        with self.assertRaisesRegex(StorageError, 'Only the storage quota'):
            Storage.configure(self.settings, v2_identity(self.settings), timeout_seconds=600)
        result = Storage.configure(self.settings, v2_identity(self.settings), storage_mib=20)
        new = replace(self.settings, storage_mib=20)
        self.assertEqual(result['config'], new.identity())
        history = self.identity()['history']
        self.assertEqual([(e['config'], e.get('reason')) for e in history],
                         [(v2_identity(self.settings), 'identity-v3'), (self.settings.identity(), None)])
        self.assertEqual(self.snapshot()[1], storage.SCHEMA)
        Service(new).storage.close()

    def test_downgrade_round_trip_to_schema_2(self):
        self.make_legacy(2)
        before = self.snapshot()
        service = Service(self.settings)
        with service.storage.connect() as conn:
            conn.execute("UPDATE runs SET progress='{}' WHERE id='r1'")
        with self.assertRaisesRegex(StorageError, 'Stop API, worker and CLI'):
            Storage.downgrade(self.settings, self.settings.identity())
        service.storage.close()
        for expected in ('wrong', v2_identity(self.settings)):
            with self.assertRaisesRegex(StorageError, 'Previous configuration identity mismatch'):
                Storage.downgrade(self.settings, expected)
        env = Path(self.tmp.name) / '.env'
        env.write_text('MDF_UPLOAD_MIB=1\nMDF_OUTPUT_MIB=3\nMDF_LOG_MIB=1\nMDF_STORAGE_MIB=16\nMDF_RESERVE_MIB=1\n')
        output = io.StringIO()
        with redirect_stdout(output):
            code = main(['--env-file', str(env), '--data-root', self.settings.data_root, '--model-root', self.settings.model_root,
                         'downgrade-storage', '--expected-config', self.settings.identity()])
        self.assertEqual(code, 0, output.getvalue())
        self.assertEqual(json.loads(output.getvalue()), {'previous_config': self.settings.identity(),
                                                          'config': v2_identity(self.settings), 'schema': 2})
        after = self.snapshot()
        fresh = sqlite3.connect(':memory:')
        fresh.executescript(MIGRATION_001 + MIGRATION_002)
        self.assertEqual((after[1], after[2]), (2, catalog(fresh)))
        self.assertEqual(after[3], before[3])
        identity = self.identity()
        self.assertEqual(identity['config'], v2_identity(self.settings))
        self.assertEqual([e['reason'] for e in identity['history']], ['identity-v3', 'downgrade-v2'])
        # Already at schema 2: a repeated downgrade changes nothing.
        Storage.downgrade(self.settings, v2_identity(self.settings))
        self.assertEqual(self.snapshot(), after)
        Service(self.settings).storage.close()
        self.assertEqual(self.snapshot()[1], storage.SCHEMA)

    def test_v1_archive_returns_to_schema_2_and_keeps_the_way_back_to_1(self):
        self.make_legacy(1)
        Service(self.settings).storage.close()
        Storage.downgrade(self.settings, self.settings.identity())
        self.assertEqual((self.snapshot()[1], self.identity()['config']), (2, v2_identity(self.settings)))
        # the previous version finds its own record of the upgrade from schema 1 (identity-v2)
        self.assertEqual([(e['reason'], e['config']) for e in self.identity()['history']][0],
                         ('identity-v2', v1_identity(self.settings)))

    def test_interrupted_downgrade_completes(self):
        self.make_legacy(2)
        Service(self.settings).storage.close()
        with patch.object(Storage, 'replace_identity', side_effect=OSError('synthetic interruption')):
            with self.assertRaises(OSError):
                Storage.downgrade(self.settings, self.settings.identity())
        self.assertEqual(self.snapshot()[1], 2)
        self.assertEqual(self.identity()['config'], self.settings.identity())
        Storage.downgrade(self.settings, self.settings.identity())
        self.assertEqual((self.snapshot()[1], self.identity()['config']), (2, v2_identity(self.settings)))

    def test_downgrade_limits(self):
        Service(self.settings).storage.close()
        before = self.snapshot()
        with self.assertRaisesRegex(StorageError, 'not upgraded from schema 2'):
            Storage.downgrade(self.settings, self.settings.identity())
        self.assertEqual(self.snapshot(), before)
        self.tmp2 = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp2.cleanup)
        self.settings = replace(self.settings, data_root=self.tmp2.name + '/data')
        self.root = Path(self.settings.data_root)
        self.make_legacy(2)
        Service(self.settings).storage.close()
        with sqlite3.connect(self.root / 'archive.sqlite3') as conn:
            conn.execute("UPDATE runs SET state='running'")
        conn.close()
        before = self.snapshot()
        with self.assertRaisesRegex(StorageError, 'reconcile running runs'):
            Storage.downgrade(self.settings, self.settings.identity())
        self.assertEqual(self.snapshot(), before)
        with sqlite3.connect(self.root / 'archive.sqlite3') as conn:
            conn.execute("UPDATE runs SET state='failed'")
        conn.close()
        # the port is no part of schema 3, but it is of schema 2: the archive opens, the return does not
        with self.assertRaisesRegex(StorageError, 'settings differ'):
            Storage.downgrade(replace(self.settings, port=23460), self.settings.identity())
        before = self.snapshot()
        cases = [("UPDATE documents SET source_path='sources/old.docx'", 'sources other than PDF'),
                 ("UPDATE documents SET owner='someone'", 'another owner'),
                 ("UPDATE runs SET options='{\"route\": null, \"ocr\": \"auto\"}'", 'job options of schema 3'),
                 ("UPDATE runs SET owner='someone'", 'another owner')]
        for statement, message in cases:
            with self.subTest(statement=statement):
                with sqlite3.connect(self.root / 'archive.sqlite3') as conn:
                    conn.execute(statement)
                conn.close()
                changed = self.snapshot()
                with self.assertRaisesRegex(StorageError, message):
                    Storage.downgrade(self.settings, self.settings.identity())
                self.assertEqual(self.snapshot(), changed)
                with sqlite3.connect(self.root / 'archive.sqlite3') as conn:
                    conn.execute("UPDATE documents SET source_path='sources/old.pdf', owner='local'")
                    conn.execute("UPDATE runs SET owner='local', options=?", (json.dumps(V2_OPTIONS),))
                conn.close()
        self.assertEqual(self.snapshot(), before)

    def test_downgrade_refused_after_a_docx(self):
        self.make_legacy(2)
        service = Service(self.settings)
        body = io.BytesIO()
        with zipfile.ZipFile(body, 'w') as z:
            z.writestr('[Content_Types].xml', '<Types/>')
            z.writestr('word/document.xml', '<w:document/>')
        service.submit(io.BytesIO(body.getvalue()), 'letter.docx')
        service.storage.close()
        before = self.snapshot()
        with self.assertRaisesRegex(StorageError, 'sources other than PDF'):
            Storage.downgrade(self.settings, self.settings.identity())
        self.assertEqual(self.snapshot(), before)

    def test_manifest_of_a_schema_2_run_stays_downloadable(self):
        """A completed PDF run of a schema 2 archive (the manifest as 59ed8ca publishes it) is
        downloaded after the upgrade, with its source as source/document.pdf."""
        service = Service(self.settings)
        job = service.submit(io.BytesIO(PDF), 'old.pdf')
        worker = Worker(self.settings)
        row = worker.claim()
        scratch = worker.storage.root / 'attempts' / row['attempt_id']
        (scratch / 'original').mkdir(parents=True)
        write(scratch / 'original', 'document.md', b'old text')
        write_json(scratch, 'engine.json', {'inventory': [identify(scratch / 'original', 'document.md')],
                                            'inventory_complete': True})
        with patch('markdown_for_llms.worker.execute', return_value={'exit': 1, 'reason': 'synthetic'}):
            worker.publish(row, scratch, {'test': 'mock'}, {'elapsed_seconds': 0})
        worker.storage.close()
        service.storage.close()
        # back to what 59ed8ca leaves: database and identity of schema 2, options of schema 2
        storage_id = self.identity()['id']
        with sqlite3.connect(self.root / 'archive.sqlite3') as conn:
            conn.execute('UPDATE runs SET options=?', (json.dumps(V2_OPTIONS),))
            conn.executescript(''.join((MIGRATIONS / ('%03d-down.sql' % n)).read_text() for n in range(storage.SCHEMA, 2, -1)))
        conn.close()
        (self.root / 'identity.json').write_text(json.dumps({'id': storage_id, 'config': v2_identity(self.settings)}))
        service = Service(self.settings)
        self.assertEqual(self.snapshot()[1], storage.SCHEMA)
        got = service.get(job['id'])
        self.assertEqual((got['state'], got['download_available']), ('completed', True))
        with zipfile.ZipFile(service.bundle(job['id'])) as archive:
            self.assertEqual(sorted(archive.namelist()), ['manifest.json', 'result/document.md', 'source/document.pdf'])
            self.assertEqual(archive.read('source/document.pdf'), PDF)
        service.storage.close()

    def test_the_frozen_module_imports_nothing_of_the_engine(self):            # 12.7 with K5
        path = Path(__file__).parents[2] / 'markdown_for_llms' / 'legacy_identity.py'
        tree = ast.parse(path.read_text())
        imported = {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
        imported |= {('.' * node.level) + (node.module or '') for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
        self.assertEqual(imported, {'hashlib', 'json', 'pathlib'})


if __name__ == '__main__':
    unittest.main()
