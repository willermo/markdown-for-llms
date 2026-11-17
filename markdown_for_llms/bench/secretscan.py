"""`mdf bench secret-scan --env-file FILE PATH…`: the Datalab key must appear in no file of the run.

The key is read only with `engine.datalab.read_key(FILE, environ={})` (a regular file of the
user with mode 0600; never the environment, never `BenchSettings`, delta r3 Δ5) and searched,
as bytes, in every regular file under the paths, without following symbolic links. The search is
literal: a key encoded (base64, URL or JSON escapes), compressed or split is not found.

Coverage is counted (C-SCAN of the review of the final diff): the files read whole, apart from the
errors, a path that could not be scanned: a path given that is missing, is a symbolic link or is
neither a regular file nor a directory, a directory that cannot be listed, a file that cannot be
opened or read. Symbolic links and special files found inside a directory are skipped, not errors.
The output names the files where the key occurs and their number, the paths not read, never the
key. Exit 0 only when every path was read and the key occurs nowhere; 1 with `secret_found`, or
with `secret_scan_incomplete` when something could not be read.
"""
import os
from pathlib import Path
import stat

CHUNK = 1 << 20


def _count(path, needle):
    count, tail = 0, b""
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as fh:
        if not stat.S_ISREG(os.fstat(fh.fileno()).st_mode):
            raise OSError("not a regular file")
        for chunk in iter(lambda: fh.read(CHUNK), b""):
            data = tail + chunk
            count += data.count(needle)
            # the tail is shorter than the needle: an occurrence across two chunks is found once
            tail = data[-(len(needle) - 1):] if len(needle) > 1 else b""
    return count


def _files(paths, errors):
    """The regular files under `paths`; what cannot be scanned goes to `errors`."""
    for root in map(Path, paths):
        try:
            mode = os.lstat(root).st_mode
        except OSError:
            errors.append(str(root))
            continue
        if stat.S_ISREG(mode):
            yield root
            continue
        if not stat.S_ISDIR(mode):           # a symbolic link given as a path is not followed: not scanned
            errors.append(str(root))
            continue
        for parent, dirs, names in os.walk(root, followlinks=False,
                                           onerror=lambda exc: errors.append(str(exc.filename or root))):
            for name in names:
                path = Path(parent) / name
                try:
                    regular = stat.S_ISREG(os.lstat(path).st_mode)
                except OSError:
                    errors.append(str(path))
                    continue
                if regular:
                    yield path


def scan(paths, needle):
    """(files read, [paths not read], [(path, occurrences)])."""
    if not needle:
        raise ValueError("empty needle")
    read, errors, found = 0, [], []
    for path in _files(paths, errors):
        try:
            n = _count(path, needle)
        except OSError:
            errors.append(str(path))
            continue
        read += 1
        if n:
            found.append((str(path), n))
    return read, errors, found


def key_bytes(env_file):
    from ..engine.datalab import read_key
    return read_key(env_file, environ={}).header()["X-API-Key"].encode("utf-8")
