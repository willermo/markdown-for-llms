"""Single-host SQLite storage, schema lock and durable source admission."""
from contextlib import contextmanager
from importlib import resources
from pathlib import Path
import fcntl
import json
import os
import shutil
import sqlite3
import stat
import time
import uuid

from .files import directory, open_file, read_json, write_json, sync_directory


class StorageError(ValueError):
    pass


class CapacityError(StorageError):
    """A finite space commitment was refused; stored bytes are not corrupt."""


SCHEMA = 2
MISMATCH = ("Storage configuration mismatch; for an archive of a previous version, one possible cause "
            "is an MDF_TIMEOUT_SECONDS different from the one it was created or configured with")
UPGRADE_BUSY = ("Storage upgrade needs exclusive access: stop the API, worker and CLI "
                "of every version, then restart")


def migration(name):
    return resources.files("markdown_for_llms").joinpath("migrations/" + name + ".sql").read_text()


class Storage:
    def __init__(self, settings, administrative=False):
        self.settings = settings
        self.root = Path(settings.data_root)
        self._config_lock = None
        with directory(self.root, create=True):
            pass
        info = self.root.stat()
        if info.st_uid != os.getuid() or info.st_mode & 0o022:
            raise StorageError("Storage must be owned by this user and not writable by others")
        for name in ("sources", "attempts", "results", "staging", "downloads", "operations"):
            with directory(self.root / name, create=True):
                pass
        try:
            self._acquire(administrative)
            if self._open() and not administrative:
                # A schema 1 identity or database is upgraded only by an opener that
                # holds the configuration exclusively: a shared lock is never promoted.
                self.close()
                self._acquire(True, UPGRADE_BUSY)
                self.upgrade()
                self.close()
                self._acquire(False)
                if self._open():
                    raise StorageError("Storage upgrade incomplete")
        except BaseException:
            self.close()
            raise

    def _acquire(self, administrative, busy="Stop API, worker and CLI before changing configuration"):
        lock = self.lock("configuration.lock", blocking=not administrative, shared=not administrative)
        try:
            lock.__enter__()
        except BlockingIOError:
            raise StorageError(busy) from None
        self._config_lock = lock

    def _open(self):
        """Create or verify the archive; True while it still has schema 1 parts."""
        with self.lock("schema.lock"):
            schema = self._load()
        return schema != SCHEMA or self.identity["config"] != self.settings.identity()

    def _load(self):
        """Caller holds schema.lock: read and verify identity, schema and storage id."""
        identity_path = self.root / "identity.json"
        if not identity_path.exists():
            if (self.root / "archive.sqlite3").exists():
                raise StorageError("Storage identity missing")
            write_json(self.root, "identity.json", {"id": uuid.uuid4().hex, "config": self.settings.identity()})
            sync_directory(self.root)
        self.identity = read_json(self.root, "identity.json")
        if self.identity["config"] not in (self.settings.identity(), self.settings.legacy_identity()):
            raise StorageError(MISMATCH)
        with self.connect() as conn:
            schema = conn.execute("PRAGMA user_version").fetchone()[0]
            if schema not in (0, 1, SCHEMA):
                raise StorageError("Unsupported database schema")
            if schema == 0:
                if conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' LIMIT 1").fetchone():
                    raise StorageError("Unrecognized unversioned database")
                conn.executescript("BEGIN IMMEDIATE;\n" + migration("001") + migration("002"))
                conn.execute("INSERT INTO metadata VALUES ('storage_id',?)", (self.identity["id"],))
                conn.commit()
                schema = SCHEMA
            stored = conn.execute("SELECT value FROM metadata WHERE key='storage_id'").fetchone()
            if stored is None or stored[0] != self.identity["id"]:
                raise StorageError("Database/storage identity mismatch")
        return schema

    def upgrade(self):
        """Caller holds configuration.lock exclusively. Database first, identity.json last:
        an interrupted upgrade is completed by the next exclusive opening."""
        with self.lock("schema.lock"):
            schema = self._load()
            if schema == 1:
                with self.connect() as conn:
                    conn.executescript("BEGIN IMMEDIATE;\n" + migration("002"))
            if self.identity["config"] != self.settings.identity():
                self.replace_identity(self.settings.identity(), dict(
                    config=self.identity["config"], settings=self.settings.public(),
                    changed=time.time(), reason="identity-v2"))

    def replace_identity(self, config, entry):
        """Caller holds configuration.lock exclusively; the change is recorded in history."""
        token = uuid.uuid4().hex
        pending = 'configuration-' + token + '.pending'
        with self.reservation('admin', 65536, token, (pending,)):
            value = dict(self.identity)
            value['history'] = [*value.get('history', []), entry]
            value['config'] = config
            write_json(self.root, pending, value)
            os.replace(self.root / pending, self.root / 'identity.json')
            sync_directory(self.root)
        self.identity = value

    def quiesce(self, expected):
        """Administrative checks that precede any change, including an upgrade."""
        if expected != self.identity['config']:
            raise StorageError('Previous configuration identity mismatch')
        with self.lock('admission.lock'):
            from .capacity import recover
            recover(self)
            if any((self.root / 'operations').iterdir()):
                raise StorageError('Restart worker to reconcile interrupted operations first')

    @contextmanager
    def lock(self, name, blocking=True, shared=False):
        with directory(self.root) as fd:
            lock_fd = os.open(name, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600, dir_fd=fd)
        try:
            info = os.fstat(lock_fd)
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                raise StorageError("Unsafe lock file")
            fcntl.flock(lock_fd, (fcntl.LOCK_SH if shared else fcntl.LOCK_EX) | (0 if blocking else fcntl.LOCK_NB))
            yield lock_fd
        finally:
            os.close(lock_fd)

    @contextmanager
    def connect(self):
        with directory(self.root):
            for name in ("archive.sqlite3", "archive.sqlite3-wal", "archive.sqlite3-shm"):
                path = self.root / name
                try:
                    info = path.lstat()
                except FileNotFoundError:
                    # SQLite removes transient WAL/SHM files when its last
                    # connection closes. A single lstat avoids mixed observations.
                    continue
                # SQLite may unlink a last-connection sidecar while lstat is
                # returning its inode. A zero-link sidecar is already absent.
                if name != 'archive.sqlite3' and info.st_nlink == 0:
                    continue
                if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                    raise StorageError("Unsafe database path")
            conn = sqlite3.connect(self.root / "archive.sqlite3", timeout=5)
        try:
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys=ON")
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=FULL")
            yield conn
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
        finally:
            conn.close()

    def capacity(self, required=0):
        with self.lock('admission.lock'):
            return self._capacity(required)

    def _capacity(self, required=0):
        from .capacity import capacity
        return capacity(self, required)

    def reservation(self, kind, budget, token=None, paths=()):
        from .capacity import reserve
        return reserve(self, kind, budget, token, paths)

    def get(self, run_id):
        with self.connect() as conn:
            row = conn.execute("SELECT r.*,d.name,d.source_path,d.source_sha256,d.source_bytes FROM runs r JOIN documents d ON d.id=r.document_id WHERE r.id=?", (run_id,)).fetchone()
        if row is None:
            raise StorageError("Run not found")
        return dict(row)

    def list(self):
        with self.connect() as conn:
            return [dict(row) for row in conn.execute("SELECT r.*,d.name FROM runs r JOIN documents d ON d.id=r.document_id ORDER BY r.created DESC")]

    def close(self):
        if getattr(self, '_config_lock', None):
            self._config_lock.__exit__(None, None, None)
            self._config_lock = None

    def __del__(self):
        self.close()

    @classmethod
    def configure(cls, settings, expected, **changes):
        from dataclasses import replace
        if not changes or set(changes) - {'storage_mib'}:
            raise StorageError('Only the storage quota can be changed')
        new = replace(settings, **changes)
        storage = cls(settings, administrative=True)
        try:
            storage.quiesce(expected)
            storage.upgrade()
            storage.settings = new
            storage.replace_identity(new.identity(), dict(config=settings.identity(),
                                     settings=settings.public(), changed=time.time()))
            return {'previous_config': expected, 'config': new.identity(), 'settings': new.public()}
        finally:
            storage.close()

    @classmethod
    def downgrade(cls, settings, expected):
        """Return an upgraded archive to identity and database schema 1 (ADR 0011)."""
        storage = cls(settings, administrative=True)
        try:
            storage.quiesce(expected)
            legacy = settings.legacy_identity()
            if storage.identity['config'] != legacy:
                upgrades = [entry for entry in storage.identity.get('history', [])
                            if isinstance(entry, dict) and entry.get('reason') == 'identity-v2']
                if not upgrades:
                    raise StorageError('Downgrade unavailable: this archive was not upgraded from schema 1')
                if upgrades[-1].get('config') != legacy:
                    raise StorageError('Downgrade unavailable: settings differ from those of the upgrade')
            with storage.connect() as conn:
                if conn.execute("SELECT 1 FROM runs WHERE state IN ('running','publishing') LIMIT 1").fetchone():
                    raise StorageError('Downgrade unavailable: start the worker once to reconcile running runs')
            with storage.lock('schema.lock'):
                with storage.connect() as conn:
                    if conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA:
                        conn.executescript("BEGIN IMMEDIATE;\n" + migration("002-down"))
                if storage.identity['config'] != legacy:
                    storage.replace_identity(legacy, dict(config=storage.identity['config'],
                                             settings=settings.public(), changed=time.time(),
                                             reason='downgrade-v1'))
            return {'previous_config': expected, 'config': legacy, 'schema': 1}
        finally:
            storage.close()
