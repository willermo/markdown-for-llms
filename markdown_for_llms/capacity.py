"""Cross-process space commitments with flock-owned, crash-recoverable leases."""
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import shutil
import stat
import uuid

from .files import directory, open_file, read_json, sync_directory, rename, FileSafetyError, decode_json

MIB = 1048576
CONTROL = 262144


def sizes(root, isolated=()):
    """Count existing bytes once, including private scratch and SQLite sidecars."""
    found = {}
    try:
        with directory(root):
            pass
    except FileNotFoundError:
        return found
    def retained(path):
        name = path.relative_to(root).as_posix()
        return any(name.startswith(prefix + '/') for prefix in isolated)
    def failed(exc):
        if not isinstance(exc, FileNotFoundError):
            raise exc
    for parent, dirs, files in os.walk(root, followlinks=False, onerror=failed):
        for name in dirs[:]:
            try:
                info = (Path(parent) / name).lstat()
            except FileNotFoundError:
                continue
            if not stat.S_ISDIR(info.st_mode):
                if not retained(Path(parent) / name):
                    raise FileSafetyError('Unsafe storage directory')
                dirs.remove(name)
                found[(Path(parent) / name).relative_to(root).as_posix()] = max(info.st_size, 4096)
        for name in files:
            path = Path(parent) / name
            try:
                info = path.lstat()
            except FileNotFoundError:
                continue
            # A zero-link inode was unlinked while lstat was being filled.
            if info.st_nlink == 0:
                continue
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                if not retained(path):
                    raise FileSafetyError('Unsafe storage file in attempt; partial preserved')
                # Count the entry itself conservatively; never open special
                # files or follow links, including links outside the archive.
                found[path.relative_to(root).as_posix()] = max(info.st_size, 4096)
                continue
            found[path.relative_to(root).as_posix()] = info.st_size
    return found


def archive_sizes(storage):
    with storage.connect() as conn:
        isolated = tuple('attempts/' + row[0] for row in conn.execute(
            "SELECT attempt_id FROM runs WHERE state IN ('failed','interrupted','cancelled')"))
    return sizes(storage.root, isolated)


class Lease:
    def __init__(self, storage, record, fd, name):
        self.storage, self.record, self.fd, self.name = storage, record, fd, name

    def check(self, local_only=False):
        if not local_only:
            with self.storage.lock('admission.lock'):
                self.storage._capacity()
            return
        from .storage import CapacityError
        owned = 0
        for path in self.record['paths']:
            owned += sum(sizes(self.storage.root / path).values())
        if owned > self.record['budget']:
            raise CapacityError('Operation exceeded its space commitment')
        if shutil.disk_usage(self.storage.root).free < (
                self.record['budget'] - owned + CONTROL + self.storage.settings.reserve_mib * MIB):
            raise CapacityError('Insufficient free disk space; archive preserved')

    def close(self):
        if self.fd is None:
            return
        try:
            with self.storage.lock('admission.lock'):
                cleanup(self.storage, self.record)
                (self.storage.root / self.name).unlink()
                sync_directory(self.storage.root / 'operations')
        finally:
            os.close(self.fd)
            self.fd = None


def cleanup(storage, record):
    """Delete only paths explicitly owned by this protocol, never by suffix."""
    token = record['token']
    if len(token) != 32 or any(c not in '0123456789abcdef' for c in token):
        raise ValueError('Invalid capacity owner')
    if record['kind'] == 'upload':
        stage = storage.root / 'staging' / ('upload-' + token)
        if stage.exists():
            shutil.rmtree(stage)
        path = 'sources/' + token + '.pdf'
        with storage.connect() as conn:
            accepted = conn.execute('SELECT 1 FROM documents WHERE source_path=?', (path,)).fetchone()
        if not accepted:
            (storage.root / path).unlink(missing_ok=True)
        sync_directory(storage.root / 'sources')
        sync_directory(storage.root / 'staging')
    elif record['kind'] == 'zip':
        for name in (token + '.part',):
            (storage.root / 'downloads' / name).unlink(missing_ok=True)
        sync_directory(storage.root / 'downloads')
    elif record['kind'] == 'admin':
        (storage.root / ('configuration-' + token + '.pending')).unlink(missing_ok=True)
        sync_directory(storage.root)
    elif record['kind'] not in {'worker', 'control'}:
        raise ValueError('Unknown capacity owner')


def recover(storage, workers=False):
    """Caller holds admission lock; live flock owners are never collected."""
    for path in sorted((storage.root / 'operations').iterdir()):
        name = path.name
        stem, dot, suffix = name.partition('.')
        token = stem.removeprefix('lease-')
        if (not stem.startswith('lease-') or len(token) != 32
                or any(c not in '0123456789abcdef' for c in token)
                or suffix not in {'json', 'pending'}):
            raise ValueError('Unrecognized capacity operation')
        with open_file(storage.root, path.relative_to(storage.root).as_posix(), 'r+b') as stream:
            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                continue
            if os.fstat(stream.fileno()).st_uid != os.getuid():
                raise ValueError('Capacity owner mismatch')
            if suffix == 'pending':
                # No writer may create its owned paths until this record has
                # been fsynced and renamed to .json under admission.lock.
                path.unlink()
                continue
            record = decode_json(stream.read(1048577))
            # An orphan conversion may still have a child. Only the worker,
            # after PID/starttime reconciliation under worker.lock, can release it.
            if record['kind'] == 'worker' and (not workers or
                    workers is not True and record['token'] not in workers):
                continue
            cleanup(storage, record)
            path.unlink()
    sync_directory(storage.root / 'operations')


def live_workers(storage):
    """Caller holds admission.lock; identify live commitments without collecting them."""
    tokens = set()
    for path in (storage.root / 'operations').glob('*.json'):
        with open_file(storage.root, path.relative_to(storage.root).as_posix()) as stream:
            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                record = decode_json(stream.read(1048577))
                if record['kind'] == 'worker':
                    tokens.add(record['token'])
    return tokens


def capacity(storage, required=0, found=None):
    from .storage import CapacityError
    found = archive_sizes(storage) if found is None else found
    remaining = 0
    for name in (storage.root / 'operations').glob('*.json'):
        record = read_json(storage.root, name.relative_to(storage.root).as_posix())
        owned = sum(size for path, size in found.items()
                    if any(path == p or path.startswith(p + '/') for p in record['paths']))
        if owned > record['budget']:
            raise CapacityError('Operation exceeded its space commitment')
        remaining += record['budget'] - owned
    committed = required + remaining + CONTROL
    total = sum(found.values())
    if total + committed > storage.settings.storage_mib * MIB:
        raise CapacityError('Storage quota exceeded; archive preserved')
    if shutil.disk_usage(storage.root).free < committed + storage.settings.reserve_mib * MIB:
        raise CapacityError('Insufficient free disk space; archive preserved')
    return total


@contextmanager
def reserve(storage, kind, budget, token=None, paths=()):
    token = token or uuid.uuid4().hex
    name = 'operations/lease-' + uuid.uuid4().hex + '.json'
    pending = name.removesuffix('.json') + '.pending'
    fd = None
    with storage.lock('admission.lock'):
        recover(storage)
        found = archive_sizes(storage)
        owned = sum(size for path, size in found.items()
                    if any(path == p or path.startswith(p + '/') for p in paths))
        if owned > budget:
            from .storage import CapacityError
            raise CapacityError('Operation already exceeds space commitment')
        capacity(storage, budget - owned + 4096, found=found)
        record = dict(token=token, kind=kind, budget=budget, paths=list(paths))
        try:
            with open_file(storage.root, pending, 'xb') as stream:
                fd = os.dup(stream.fileno())
                fcntl.flock(fd, fcntl.LOCK_EX)
                stream.write((json.dumps(record) + '\n').encode())
                stream.flush()
                os.fsync(fd)
            rename(storage.root, pending, name)
        except BaseException:
            if fd is not None:
                os.close(fd)
            (storage.root / pending).unlink(missing_ok=True)
            (storage.root / name).unlink(missing_ok=True)
            raise
    lease = Lease(storage, record, fd, name)
    try:
        yield lease
    finally:
        lease.close()
