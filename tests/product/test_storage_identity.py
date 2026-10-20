"""Storage identity and database schema 2: automatic upgrade, configure and downgrade (ADR 0011)."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stdout
from dataclasses import replace
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

from markdown_for_llms.cli import main
from markdown_for_llms.services import Service
from markdown_for_llms.settings import Settings
from markdown_for_llms.storage import Storage, StorageError

# Identity schema 1 as computed at 17de44f, frozen here independently of the code under test.
V1_FIELDS = ("data_root", "model_root", "host", "port", "timeout_seconds", "memory_mib",
             "upload_mib", "max_pages", "output_mib", "log_mib", "storage_mib", "reserve_mib",
             "max_pixels", "max_dimension", "threads", "poll_seconds", "lease_seconds",
             "grace_seconds", "process_limit", "bind_profile", "review_provider")
MIGRATION_001 = (Path(__file__).parents[2] / 'markdown_for_llms' / 'migrations' / '001.sql').read_text()


def v1_identity(settings, timeout=450):
    value = {name: getattr(settings, name) for name in V1_FIELDS}
    value['timeout_seconds'] = timeout
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

    def make_v1(self, timeout=450):
        """A schema 1 archive as the previous version leaves it: identity v1, 001.sql, one run."""
        self.root.mkdir(mode=0o700)
        for name in ("sources", "attempts", "results", "staging", "downloads", "operations"):
            (self.root / name).mkdir(mode=0o700)
        storage_id = uuid.uuid4().hex
        (self.root / 'identity.json').write_text(json.dumps({'id': storage_id, 'config': v1_identity(self.settings, timeout)}))
        conn = sqlite3.connect(self.root / 'archive.sqlite3')
        conn.executescript(MIGRATION_001)
        conn.execute("INSERT INTO metadata VALUES ('storage_id',?)", (storage_id,))
        conn.execute("INSERT INTO documents VALUES ('d1','old.pdf','sources/old.pdf',?,3,1.0)", ('0' * 64,))
        conn.execute("INSERT INTO runs (id,document_id,attempt_id,options,state,created,updated,error) "
                     "VALUES ('r1','d1','a1','{}','failed',1.0,2.0,'old failure')")
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

    def test_v1_archive_upgrades_on_first_open_and_time_limits_leave_identity(self):
        self.make_v1()
        before = self.snapshot()
        service = Service(self.settings)
        service.storage.close()
        identity = self.identity()
        self.assertEqual(identity['config'], self.settings.identity())
        self.assertEqual([(e['config'], e['reason']) for e in identity['history']], [(v1_identity(self.settings), 'identity-v2')])
        after = self.snapshot()
        self.assertEqual(after[1], 2)
        self.assertEqual(after[3], before[3])
        with sqlite3.connect(self.root / 'archive.sqlite3') as conn:
            self.assertIn('progress', [row[1] for row in conn.execute('PRAGMA table_info(runs)')])
            self.assertIsNone(conn.execute("SELECT progress FROM runs WHERE id='r1'").fetchone()[0])
        self.assertEqual(Service(self.settings).get('r1')['error'], 'old failure')
        # Time limits change without configure; any other setting still needs it.
        Service(replace(self.settings, timeout_seconds=900)).storage.close()
        with self.assertRaisesRegex(StorageError, 'configuration mismatch'):
            Service(replace(self.settings, port=8799))
        self.assertEqual(len(self.identity()['history']), 1)

    def test_v1_archive_with_configured_timeout(self):
        self.make_v1(timeout=5400)
        with self.assertRaisesRegex(StorageError, 'MDF_TIMEOUT_SECONDS'):
            Service(self.settings)
        self.assertEqual(self.snapshot()[1], 1)
        Service(replace(self.settings, timeout_seconds=5400)).storage.close()
        self.assertEqual(self.identity()['history'][0]['config'], v1_identity(self.settings, 5400))
        Service(self.settings).storage.close()

    def test_new_archive_is_born_with_schema_2(self):
        Service(self.settings).storage.close()
        self.assertEqual(self.identity()['config'], self.settings.identity())
        self.assertNotIn('history', self.identity())
        self.assertEqual(self.snapshot()[1], 2)

    def test_live_previous_version_process_refuses_upgrade_without_changes(self):
        # (a) The previous version holds configuration.lock shared for its whole life.
        self.make_v1()
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
        self.assertEqual(self.snapshot()[1], 2)

    def test_concurrent_new_openers_upgrade_once(self):
        # (b) Racing openers either upgrade, observe the upgrade or are refused; never twice.
        self.make_v1()
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
        # The last opener to release its shared lock always finds the upgrade possible or done.
        self.assertIn('opened', outcomes)
        self.assertEqual([e['reason'] for e in self.identity()['history']], ['identity-v2'])
        self.assertEqual(self.snapshot()[1], 2)
        Storage(self.settings).close()
        self.assertEqual([e['reason'] for e in self.identity()['history']], ['identity-v2'])

    def test_interrupted_upgrade_is_completed_by_the_next_opening(self):
        # (c) The database is migrated first; identity.json is rewritten last.
        self.make_v1()
        before = self.snapshot()
        with patch.object(Storage, 'replace_identity', side_effect=OSError('synthetic interruption')):
            with self.assertRaises(OSError):
                Service(self.settings)
        interrupted = self.snapshot()
        self.assertEqual((interrupted[0], interrupted[1], interrupted[3]), (before[0], 2, before[3]))
        self.assertEqual(list(self.root.glob('configuration-*.pending')), [])
        Service(self.settings).storage.close()
        self.assertEqual(self.identity()['config'], self.settings.identity())
        self.assertEqual([e['reason'] for e in self.identity()['history']], ['identity-v2'])

    def test_configure_checks_expected_before_upgrading(self):
        # (d) A wrong expected identity on a v1 archive changes nothing.
        self.make_v1()
        before = self.snapshot()
        for expected in ('wrong', self.settings.identity()):
            with self.assertRaisesRegex(StorageError, 'Previous configuration identity mismatch'):
                Storage.configure(self.settings, expected, storage_mib=20)
            self.assertEqual(self.snapshot(), before)
        with self.assertRaisesRegex(StorageError, 'Only the storage quota'):
            Storage.configure(self.settings, v1_identity(self.settings), timeout_seconds=600)
        result = Storage.configure(self.settings, v1_identity(self.settings), storage_mib=20)
        new = replace(self.settings, storage_mib=20)
        self.assertEqual(result['config'], new.identity())
        history = self.identity()['history']
        self.assertEqual([(e['config'], e.get('reason')) for e in history],
                         [(v1_identity(self.settings), 'identity-v2'), (self.settings.identity(), None)])
        self.assertEqual(self.snapshot()[1], 2)
        Service(new).storage.close()

    def test_downgrade_round_trip(self):
        self.make_v1()
        before = self.snapshot()
        service = Service(self.settings)
        with service.storage.connect() as conn:
            conn.execute("UPDATE runs SET progress='{}' WHERE id='r1'")
        with self.assertRaisesRegex(StorageError, 'Stop API, worker and CLI'):
            Storage.downgrade(self.settings, self.settings.identity())
        service.storage.close()
        for expected in ('wrong', v1_identity(self.settings)):
            with self.assertRaisesRegex(StorageError, 'Previous configuration identity mismatch'):
                Storage.downgrade(self.settings, expected)
        env = Path(self.tmp.name) / '.env'
        env.write_text('MDF_UPLOAD_MIB=1\nMDF_OUTPUT_MIB=3\nMDF_LOG_MIB=1\nMDF_STORAGE_MIB=16\nMDF_RESERVE_MIB=1\n')
        output = io.StringIO()
        with redirect_stdout(output):
            code = main(['--env-file', str(env), '--data-root', self.settings.data_root, '--model-root', self.settings.model_root,
                         'downgrade-storage', '--expected-config', self.settings.identity()])
        self.assertEqual(code, 0, output.getvalue())
        self.assertEqual(json.loads(output.getvalue())['config'], v1_identity(self.settings))
        after = self.snapshot()
        fresh = sqlite3.connect(':memory:')
        fresh.executescript(MIGRATION_001)
        self.assertEqual((after[1], after[2], after[3]), (1, catalog(fresh), before[3]))
        identity = self.identity()
        self.assertEqual(identity['config'], v1_identity(self.settings))
        self.assertEqual([e['reason'] for e in identity['history']], ['identity-v2', 'downgrade-v1'])
        # Already at schema 1: a repeated downgrade changes nothing.
        Storage.downgrade(self.settings, v1_identity(self.settings))
        self.assertEqual(self.snapshot(), after)
        Service(self.settings).storage.close()
        self.assertEqual(self.snapshot()[1], 2)

    def test_interrupted_downgrade_completes(self):
        self.make_v1()
        Service(self.settings).storage.close()
        with patch.object(Storage, 'replace_identity', side_effect=OSError('synthetic interruption')):
            with self.assertRaises(OSError):
                Storage.downgrade(self.settings, self.settings.identity())
        self.assertEqual(self.snapshot()[1], 1)
        self.assertEqual(self.identity()['config'], self.settings.identity())
        Storage.downgrade(self.settings, self.settings.identity())
        self.assertEqual((self.snapshot()[1], self.identity()['config']), (1, v1_identity(self.settings)))

    def test_downgrade_limits(self):
        Service(self.settings).storage.close()
        before = self.snapshot()
        with self.assertRaisesRegex(StorageError, 'not upgraded from schema 1'):
            Storage.downgrade(self.settings, self.settings.identity())
        self.assertEqual(self.snapshot(), before)
        self.tmp2 = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp2.cleanup)
        self.settings = replace(self.settings, data_root=self.tmp2.name + '/data')
        self.root = Path(self.settings.data_root)
        self.make_v1()
        Service(self.settings).storage.close()
        with sqlite3.connect(self.root / 'archive.sqlite3') as conn:
            conn.execute("UPDATE runs SET state='running'")
        conn.close()
        before = self.snapshot()
        with self.assertRaisesRegex(StorageError, 'reconcile running runs'):
            Storage.downgrade(self.settings, self.settings.identity())
        self.assertEqual(self.snapshot(), before)
        with self.assertRaisesRegex(StorageError, 'settings differ'):
            Storage.downgrade(replace(self.settings, timeout_seconds=900), self.settings.identity())
        self.assertEqual(self.snapshot(), before)


if __name__ == '__main__':
    unittest.main()
