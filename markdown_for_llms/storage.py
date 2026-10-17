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


class Storage:
    def __init__(self, settings, administrative=False):
        self.settings = settings
        self.root = Path(settings.data_root)
        with directory(self.root, create=True):
            pass
        info = self.root.stat()
        if info.st_uid != os.getuid() or info.st_mode & 0o022:
            raise StorageError("Storage must be owned by this user and not writable by others")
        for name in ("sources", "attempts", "results", "staging", "downloads", "operations"):
            with directory(self.root / name, create=True):
                pass
        self._config_lock = self.lock("configuration.lock", blocking=not administrative,
                                      shared=not administrative)
        try:
            self._config_lock.__enter__()
        except BlockingIOError:
            self._config_lock = None
            raise StorageError("Stop API, worker and CLI before changing configuration") from None
        try:
            with self.lock("schema.lock"):
                identity_path = self.root / "identity.json"
                if not identity_path.exists():
                    if (self.root / "archive.sqlite3").exists():
                        raise StorageError("Storage identity missing")
                    write_json(self.root, "identity.json", {"id": uuid.uuid4().hex, "config": settings.identity()})
                    sync_directory(self.root)
                self.identity = read_json(self.root, "identity.json")
                if self.identity["config"] != settings.identity():
                    raise StorageError("Storage configuration mismatch")
                with self.connect() as conn:
                    schema = conn.execute("PRAGMA user_version").fetchone()[0]
                    if schema not in (0, 1):
                        raise StorageError("Unsupported database schema")
                    if schema == 0:
                        if conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' LIMIT 1").fetchone():
                            raise StorageError("Unrecognized unversioned database")
                        conn.executescript("BEGIN IMMEDIATE;\n" + resources.files("markdown_for_llms").joinpath("migrations/001.sql").read_text())
                        conn.execute("INSERT INTO metadata VALUES ('storage_id',?)", (self.identity["id"],))
                        conn.commit()
                    stored = conn.execute("SELECT value FROM metadata WHERE key='storage_id'").fetchone()
                    if stored is None or stored[0] != self.identity["id"]:
                        raise StorageError("Database/storage identity mismatch")
        except BaseException:
            self.close()
            raise

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
        from .files import write_json
        if not changes or set(changes) - {'storage_mib', 'timeout_seconds'}:
            raise StorageError('Only storage quota and timeout can be changed')
        new = replace(settings, **changes)
        storage = cls(settings, administrative=True)
        try:
            if expected != settings.identity() or storage.identity['config'] != expected:
                raise StorageError('Previous configuration identity mismatch')
            with storage.lock('admission.lock'):
                from .capacity import recover
                recover(storage)
                if any((storage.root / 'operations').iterdir()):
                    raise StorageError('Restart worker to reconcile interrupted operations first')
            storage.settings = new
            token = uuid.uuid4().hex
            pending = 'configuration-' + token + '.pending'
            with storage.reservation('admin', 65536, token, (pending,)):
                value = dict(storage.identity)
                value['history'] = [*value.get('history', []),
                                    dict(config=expected, settings=settings.public(), changed=time.time())]
                value['config'] = new.identity()
                write_json(storage.root, pending, value)
                os.replace(storage.root / pending, storage.root / 'identity.json')
                sync_directory(storage.root)
            return {'previous_config': expected, 'config': new.identity(), 'settings': new.public()}
        finally:
            storage.close()
