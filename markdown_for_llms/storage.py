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

from . import legacy_identity
from .files import directory, open_file, read_json, write_json, sync_directory


class StorageError(ValueError):
    pass


class CapacityError(StorageError):
    """A finite space commitment was refused; stored bytes are not corrupt."""


SCHEMA = 4              # 4: the conversion presets of the users and their default (MFL023)
OWNER = "local"         # the owner of every document and run until the product has users
MISMATCH = ("Storage configuration mismatch; for an archive of a previous version (identity schema 1 or 2), "
            "possible causes are an MDF_TIMEOUT_SECONDS, an MDF_PORT or an MDF_MODEL_ROOT different from "
            "the ones it was created or configured with (an archive created with a former default port needs "
            "it in MDF_PORT: 23460, or 8765 for archives created before 2026-11-07; one with the former "
            "default model root needs MDF_MODEL_ROOT set to the absolute path of that mdf-models directory, "
            "or ./mdf-models from the directory the archive was used from). "
            "Its first opening upgrades it to schema %d, which depends on none of them" % SCHEMA)
UPGRADE_BUSY = ("Storage upgrade needs exclusive access: stop the API, worker and CLI "
                "of every version, then restart")


def migration(name):
    return resources.files("markdown_for_llms").joinpath("migrations/" + name + ".sql").read_text()


def migrations(after, upto=SCHEMA):
    """The SQL of the migrations that bring a database from schema `after` to `upto`, in order."""
    return "".join(migration("%03d" % n) for n in range(after + 1, upto + 1))


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
        if (self.identity["config"] != self.settings.identity()
                and self.identity["config"] not in legacy_identity.candidates(self.settings)):
            raise StorageError(MISMATCH)
        with self.connect() as conn:
            schema = conn.execute("PRAGMA user_version").fetchone()[0]
            if schema not in range(SCHEMA + 1):
                raise StorageError("Unsupported database schema")
            if schema == 0:
                if conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' LIMIT 1").fetchone():
                    raise StorageError("Unrecognized unversioned database")
                conn.executescript("BEGIN IMMEDIATE;\n" + migrations(0))
                conn.execute("INSERT INTO metadata VALUES ('storage_id',?)", (self.identity["id"],))
                conn.commit()
                schema = SCHEMA
            stored = conn.execute("SELECT value FROM metadata WHERE key='storage_id'").fetchone()
            if stored is None or stored[0] != self.identity["id"]:
                raise StorageError("Database/storage identity mismatch")
        return schema

    def upgrade(self):
        """Caller holds configuration.lock exclusively. Database first, identity.json last:
        an interrupted upgrade is completed by the next exclusive opening. An identity of schema 1
        goes through schema 2 as the previous version upgraded it (`identity-v2`, so that version
        can still return it to schema 1), then to schema 3 (`identity-v3`, with the model root its
        legacy identity was recognized with, for the return to schema 2). The database takes the
        migrations it misses in order, in one transaction (schema 4, MFL023: the conversion presets);
        the identity does not depend on the database schema after 3."""
        with self.lock("schema.lock"):
            schema = self._load()
            if 0 < schema < SCHEMA:
                with self.connect() as conn:
                    conn.executescript("BEGIN IMMEDIATE;\n" + migrations(schema))
            legacy = legacy_identity.candidates(self.settings).get(self.identity["config"])
            if legacy and legacy[0] == 1:
                self.replace_identity(legacy_identity.v2(self.settings, legacy[1]), dict(
                    config=self.identity["config"], settings=self.settings.public(),
                    changed=time.time(), reason="identity-v2"))
                legacy = (2, legacy[1])
            if self.identity["config"] != self.settings.identity():
                self.replace_identity(self.settings.identity(), dict(
                    config=self.identity["config"], settings=self.settings.public(),
                    changed=time.time(), reason="identity-v3", model_root=legacy[1]))

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
    def downgrade(cls, settings, expected, to=2):
        """Return the archive to the database schema `to`, from 2 to SCHEMA - 1 (MFL023, C-B4 of the
        plan r1): 2 by default, the meaning of the command before schema 4; 3 for the binary of
        schema 3. The checks of every level crossed run before anything changes, then the down
        migrations run in one transaction. Only the step from 3 to 2 changes the identity, back to
        the one recorded by the upgrade from schema 2 (K5 of MFL032: only while every source is a
        PDF, every run has the job options of schema 2 and the owner `local`); the steps above it
        leave identity.json as it is, byte for byte. From 4 to 3: no conversion preset of a user, no
        default preset, no run whose options (read as `stored_options` reads them) have a key outside
        those of schema 3. No run may be running. A database already at `to` is left as it is (the
        identity of a step to 2 is completed); one under `to` is refused."""
        if isinstance(to, bool) or not isinstance(to, int) or not 2 <= to < SCHEMA:
            raise StorageError('Downgrade target: a database schema from 2 to %d' % (SCHEMA - 1))
        storage = cls(settings, administrative=True)
        try:
            storage.quiesce(expected)
            target = storage._downgrade_target() if to == 2 else storage.identity['config']
            with storage.connect() as conn:
                if conn.execute("SELECT 1 FROM runs WHERE state IN ('running','publishing') LIMIT 1").fetchone():
                    raise StorageError('Downgrade unavailable: start the worker once to reconcile running runs')
                version = conn.execute("PRAGMA user_version").fetchone()[0]
                if version < to:
                    raise StorageError('Downgrade unavailable: the database is already under schema %d' % to)
                for level in range(version, to, -1):
                    storage.DOWN_CHECKS[level](conn)
            with storage.lock('schema.lock'):
                with storage.connect() as conn:
                    version = conn.execute("PRAGMA user_version").fetchone()[0]
                    if version > to:
                        conn.executescript("BEGIN IMMEDIATE;\n" + "".join(
                            migration("%03d-down" % level) for level in range(version, to, -1)))
                if to == 2 and storage.identity['config'] != target:
                    storage.replace_identity(target, dict(config=storage.identity['config'],
                                             settings=settings.public(), changed=time.time(),
                                             reason='downgrade-v2'))
            return {'previous_config': expected, 'config': target, 'schema': to}
        finally:
            storage.close()

    @staticmethod
    def _check_v3_content(conn):
        """What schema 3 cannot hold refuses the return from 4 (MFL023): the conversion presets of the
        users, a default preset, a run with job options outside those of schema 3 (a preset's name)."""
        from .engine.routes import JOB_OPTIONS
        from .services import stored_options
        if conn.execute("SELECT 1 FROM conversion_presets LIMIT 1").fetchone():
            raise StorageError('Downgrade unavailable: the archive holds conversion presets of its users')
        if conn.execute("SELECT 1 FROM user_settings WHERE default_preset IS NOT NULL LIMIT 1").fetchone():
            raise StorageError('Downgrade unavailable: a user has a default conversion preset')
        for (options,) in conn.execute("SELECT options FROM runs"):
            try:
                value = stored_options(options)
            except ValueError:
                value = None
            if not isinstance(value, dict) or set(value) - set(JOB_OPTIONS):
                raise StorageError('Downgrade unavailable: a run has job options of schema 4')

    def _downgrade_target(self):
        """The identity of schema 2 this archive returns to: the one recorded by its upgrade."""
        current = self.identity['config']
        legacy = legacy_identity.candidates(self.settings).get(current)
        if legacy is not None:
            if legacy[0] != 2:
                raise StorageError('Downgrade unavailable: the archive is at identity schema 1')
            return current                       # an interrupted downgrade: the database is completed
        upgrades = [entry for entry in self.identity.get('history', [])
                    if isinstance(entry, dict) and entry.get('reason') == 'identity-v3']
        if not upgrades:
            raise StorageError('Downgrade unavailable: this archive was not upgraded from schema 2')
        root = upgrades[-1].get('model_root')
        if not isinstance(root, str) or upgrades[-1].get('config') != legacy_identity.v2(self.settings, root):
            raise StorageError('Downgrade unavailable: settings differ from those of the upgrade')
        return upgrades[-1]['config']

    @staticmethod
    def _check_v2_content(conn):
        """What schema 2 cannot hold refuses the return (K5)."""
        if conn.execute("SELECT 1 FROM documents WHERE owner != ? OR lower(source_path) NOT LIKE '%.pdf' LIMIT 1",
                        (OWNER,)).fetchone():
            raise StorageError('Downgrade unavailable: the archive holds sources other than PDF or of another owner')
        for (owner, options) in conn.execute("SELECT owner, options FROM runs"):
            try:
                shaped = legacy_identity.v2_options(json.loads(options))
            except ValueError:
                shaped = False
            if owner != OWNER or not shaped:
                raise StorageError('Downgrade unavailable: a run has job options of schema 3 or another owner')

    # The checks of the return from the database schema N to N - 1, by N.
    DOWN_CHECKS = {3: _check_v2_content, 4: _check_v3_content}
