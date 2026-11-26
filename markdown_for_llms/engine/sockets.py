"""The socket directories of an attempt, on the run volume (MFL033, MFL051).

A Unix socket path holds at most 107 bytes on Linux (`sun_path` is 108 with the final NUL), and
the work directory of a conversion can be much longer; so the sockets never live under `work`.
They live on the run volume, which the process that converts and the launcher both see, at a
fixed base: `<containers.RUN_ROOT>/sock/<prefix of the attempt>/`, with `relay/` (mounted
read-only on `/relay` of the Marker child) and `llama/` (mounted on `/relay` of the local CPU
server). The launcher derives the two mounts from the name of the child and checks every
component: directories of the user, no link.

- `Private(prefix)` makes the directory of the attempt (mode 0700, owner and mode checked again
  with `lstat`), `relay/`, `llama/` and `lock`, held with an exclusive `flock` for the life of the
  directory. Everything after the first `mkdir` is inside a `try`: on any failure the lock is
  closed, the directory removed and the error raised again (CNB2 of the review of r3).
- `check(path)`: a socket path over 107 bytes is `socket_path_too_long`, before any `bind`.
- `sweep()`: at the start of each conversion that needs the directory, the directories of the
  attempts whose lock is free (left by a process that died) and older than 60 seconds are
  removed; a held lock is a live conversion, kept.

The mode 0700 keeps other users out, not the processes of the same user (a risk accepted for the
single-user target). Standard library only.
"""
import fcntl
import os
from pathlib import Path
import shutil
import stat
import time

from . import containers
from .codes import Code, EngineError

SUN_PATH_MAX = 107
ORPHAN_AGE = 60.0
RELAY_SOCKET = "relay/relay.sock"
LLAMA_SOCKET = "llama/llama.sock"


def _private_dir(path):
    try:
        info = os.lstat(path)
    except OSError:
        return False
    return stat.S_ISDIR(info.st_mode) and info.st_uid == os.getuid() and stat.S_IMODE(info.st_mode) == 0o700


def base(run_root=None):
    """The directory of the socket directories of the attempts on the run volume."""
    return Path(run_root or containers.RUN_ROOT) / "sock"


def check(path):
    """The length in bytes of a socket path, or `socket_path_too_long`."""
    n = len(os.fsencode(str(path)))
    if n > SUN_PATH_MAX:
        raise EngineError(Code.SOCKET_PATH_TOO_LONG, "%d bytes, at most %d" % (n, SUN_PATH_MAX))
    return n


class Private:
    """The socket directory of an attempt, held by its lock until `close`. `prefix` is the prefix of the names of
    its children (`containers.prefix_of`). `isolation_unavailable` when the run volume is not there, or the
    directory is held by another conversion."""

    def __init__(self, prefix, run_root=None):
        self.lock_fd = None
        self.path = base(run_root) / prefix
        try:
            containers.socket_dirs(run_root or containers.RUN_ROOT, prefix)
        except OSError as exc:
            raise EngineError(Code.ISOLATION_UNAVAILABLE, "run volume: " + type(exc).__name__) from None
        try:
            if not _private_dir(self.path) or not all(_private_dir(self.path / sub) for sub in containers.SOCKET_ROLES):
                raise EngineError(Code.ISOLATION_UNAVAILABLE, "socket directory not private")
            self.lock_fd = os.open(self.path / "lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
            try:
                fcntl.flock(self.lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                os.close(self.lock_fd)
                self.lock_fd = None
                raise EngineError(Code.ISOLATION_UNAVAILABLE, "socket directory of the attempt in use") from None
            self.relay_socket = self.path / RELAY_SOCKET
            self.llama_socket = self.path / LLAMA_SOCKET
            check(self.relay_socket)
            check(self.llama_socket)
        except EngineError:
            if self.lock_fd is not None:
                self.close()
            raise
        except BaseException:
            self.close()
            raise

    @property
    def relay_dir(self):
        return self.path / "relay"

    @property
    def llama_dir(self):
        return self.path / "llama"

    def close(self):
        shutil.rmtree(self.path, ignore_errors=True)
        if self.lock_fd is not None:
            os.close(self.lock_fd)
            self.lock_fd = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def sweep(root=None, min_age=ORPHAN_AGE):
    """Remove the orphan socket directories of the attempts under `root` (default: `base()`); returns (removed,
    kept) names. A directory younger than `min_age` seconds is kept: it may be being made."""
    root = Path(root) if root is not None else base()
    removed, kept = [], []
    try:
        names = sorted(os.listdir(root))
    except OSError:
        return removed, kept
    for name in names:
        entry = root / name
        if not containers.PREFIX.fullmatch(name) or not _private_dir(entry):
            continue
        try:
            try:
                made = os.lstat(entry / "lock").st_mtime
            except FileNotFoundError:            # killed before the lock, or made for a child outside a conversion: no owner
                if time.time() - os.lstat(entry).st_mtime >= min_age:
                    shutil.rmtree(entry, ignore_errors=True)
                    removed.append(entry.name)
                else:
                    kept.append(entry.name)
                continue
            if time.time() - made < min_age:
                kept.append(entry.name)
                continue
            fd = os.open(entry / "lock", os.O_RDWR | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK)
        except OSError:
            continue
        try:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                kept.append(entry.name)
                continue
            shutil.rmtree(entry, ignore_errors=True)
            removed.append(entry.name)
        finally:
            os.close(fd)
    return removed, kept
