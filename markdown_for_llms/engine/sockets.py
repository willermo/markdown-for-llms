"""Private, short directories for the Unix sockets of the `marker-server` route (MFL033).

A Unix socket path holds at most 107 bytes on Linux (`sun_path` is 108 with the final NUL), and
the work directory of a conversion can be much longer; so the sockets never live under `work`:

- `Private()` makes `mdf-<random>` with `mkdtemp` under `$XDG_RUNTIME_DIR` when it is an
  absolute directory of the user with mode 0700 (not a link), else under `/tmp`: an
  unpredictable name, created exclusively with mode 0700, owner and mode checked again with
  `lstat`. Inside: `relay/` (bound read-only on `/relay` of the Marker child), `llama/` (bound
  on `/relay` of the local CPU server) and `lock`, held with an exclusive `flock` for the life of
  the directory. Everything after `mkdtemp` is inside a `try`: on any failure the lock is closed,
  the directory removed and the error raised again (CNB2 of the review of r3).
- `check(path)`: a socket path over 107 bytes is `socket_path_too_long`, before any `bind`.
- `sweep()`: at the start of each conversion on the route, the `mdf-*` directories of the user
  whose lock is free (left by a process that died) and older than 60 seconds are removed; a held
  lock is a live conversion, kept.

The mode 0700 keeps other users out, not the processes of the same user (a risk accepted for the
single-user target). Standard library only.
"""
import fcntl
import os
from pathlib import Path
import shutil
import stat
import tempfile
import time

from .codes import Code, EngineError

SUN_PATH_MAX = 107
PREFIX = "mdf-"
ORPHAN_AGE = 60.0
RELAY_SOCKET = "relay/relay.sock"
LLAMA_SOCKET = "llama/llama.sock"


def _private_dir(path):
    try:
        info = os.lstat(path)
    except OSError:
        return False
    return stat.S_ISDIR(info.st_mode) and info.st_uid == os.getuid() and stat.S_IMODE(info.st_mode) == 0o700


def base(environ=None):
    """(directory, kind): `$XDG_RUNTIME_DIR` when private, else `/tmp`."""
    runtime = (os.environ if environ is None else environ).get("XDG_RUNTIME_DIR")
    if runtime and os.path.isabs(runtime) and _private_dir(runtime):
        return Path(runtime), "XDG_RUNTIME_DIR"
    return Path("/tmp"), "/tmp"


def check(path):
    """The length in bytes of a socket path, or `socket_path_too_long`."""
    n = len(os.fsencode(str(path)))
    if n > SUN_PATH_MAX:
        raise EngineError(Code.SOCKET_PATH_TOO_LONG, "%d bytes, at most %d" % (n, SUN_PATH_MAX))
    return n


class Private:
    """A private socket directory, held by its lock until `close`."""

    def __init__(self, environ=None):
        root, self.base_kind = base(environ)
        self.path = Path(tempfile.mkdtemp(prefix=PREFIX, dir=root))
        self.lock_fd = None
        try:
            if not _private_dir(self.path):
                raise EngineError(Code.ISOLATION_UNAVAILABLE, "socket directory not private")
            for sub in ("relay", "llama"):
                (self.path / sub).mkdir(mode=0o700)
            self.lock_fd = os.open(self.path / "lock", os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
                                   0o600)
            fcntl.flock(self.lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.relay_socket = self.path / RELAY_SOCKET
            self.llama_socket = self.path / LLAMA_SOCKET
            check(self.relay_socket)
            check(self.llama_socket)
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


def sweep(root=None, min_age=ORPHAN_AGE, environ=None):
    """Remove the orphan `mdf-*` directories under `root` (default: `base()`); returns (removed,
    kept) names. A directory younger than `min_age` seconds is kept: it may be being made."""
    root = Path(root) if root is not None else base(environ)[0]
    removed, kept = [], []
    for entry in sorted(root.glob(PREFIX + "*")):
        if not _private_dir(entry):
            continue
        try:
            try:
                made = os.lstat(entry / "lock").st_mtime
            except FileNotFoundError:            # killed between mkdtemp and the lock: no owner
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
