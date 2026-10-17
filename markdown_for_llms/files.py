"""Descriptor-relative storage operations with no symlink traversal."""
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
import hashlib
import json
import math
import os
import stat


class FileSafetyError(ValueError):
    pass


class IntegrityError(FileSafetyError):
    """Positive evidence that stored bytes or their manifest are inconsistent."""


class MetadataError(ValueError):
    """Bounded child/stored JSON could not be decoded safely."""


def decode_json(body):
    """Use the standard decoder, rejecting oversized/deep/non-finite metadata."""
    if len(body if isinstance(body, bytes) else body.encode('utf-8')) > 1048576:
        raise MetadataError('JSON metadata limit exceeded')
    def constant(value):
        raise MetadataError('Non-finite JSON number')
    def number(value):
        parsed = float(value)
        if not math.isfinite(parsed):
            raise MetadataError('Non-finite JSON number')
        return parsed
    try:
        value = json.loads(body, parse_constant=constant, parse_float=number)
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise MetadataError('Malformed, deep or non-finite JSON metadata') from exc
    pending = [(value, 0)]
    while pending:
        item, depth = pending.pop()
        if isinstance(item, (dict, list)):
            if depth >= 128:
                raise MetadataError('JSON metadata nesting limit exceeded')
            pending.extend((child, depth + 1) for child in
                           (item.values() if isinstance(item, dict) else item))
    return value


def relative(value):
    if (not isinstance(value, str) or not value or len(value) > 1024
            or any(ord(c) < 32 or ord(c) == 127 for c in value)
            or any(c in value for c in "\\:%?#")
            or value.startswith("/") or any(p in {"", ".", ".."} for p in value.split("/"))
            or PurePosixPath(value).as_posix() != value):
        raise FileSafetyError("Unsafe relative file name")
    return value


@contextmanager
def directory(path, create=False):
    path = Path(path).absolute()
    fd = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
    try:
        for part in path.parts[1:]:
            if part in {".", ".."}:
                raise FileSafetyError("Noncanonical storage path")
            if create:
                try:
                    os.mkdir(part, 0o700, dir_fd=fd)
                except FileExistsError:
                    pass
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = child
        yield fd
    finally:
        os.close(fd)


@contextmanager
def open_file(root, name, mode="rb"):
    relative(name)
    if mode not in {"rb", "xb", "r+b"}:
        raise ValueError("Unsupported storage open mode")
    parts = name.split("/")
    with directory(Path(root).joinpath(*parts[:-1])) as parent:
        flags = {"rb": os.O_RDONLY, "xb": os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                 "r+b": os.O_RDWR}[mode] | os.O_NOFOLLOW | os.O_NONBLOCK
        fd = os.open(parts[-1], flags, 0o600, dir_fd=parent)
        try:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                raise FileSafetyError("File is not an exclusive regular file")
            with os.fdopen(fd, mode, closefd=False) as stream:
                yield stream
        finally:
            os.close(fd)


def digest(stream):
    h = hashlib.sha256()
    size = 0
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        h.update(chunk)
        size += len(chunk)
    return {"sha256": h.hexdigest(), "bytes": size}


def identify(root, name):
    with open_file(root, name) as stream:
        return dict(path=name, **digest(stream))


def write(root, name, data):
    with open_file(root, name, "xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def write_json(root, name, value):
    write(root, name, json_bytes(value))


def json_bytes(value):
    """Canonical UTF-8 representation, including the final newline."""
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode('utf-8')


def read_json(root, name):
    with open_file(root, name) as stream:
        return decode_json(stream.read(1048577))


def sync_directory(path):
    with directory(path) as fd:
        os.fsync(fd)


def make_directory(root, name):
    relative(name)
    parts = name.split('/')
    with directory(Path(root).joinpath(*parts[:-1])) as fd:
        os.mkdir(parts[-1], 0o700, dir_fd=fd)
        os.fsync(fd)


def rename(root, source, destination):
    relative(source)
    relative(destination)
    src, dst = source.split('/'), destination.split('/')
    with directory(Path(root).joinpath(*src[:-1])) as src_fd:
        with directory(Path(root).joinpath(*dst[:-1])) as dst_fd:
            # Internal destinations are random exclusive IDs, never user names.
            try:
                os.stat(dst[-1], dir_fd=dst_fd, follow_symlinks=False)
            except FileNotFoundError:
                pass
            else:
                raise FileSafetyError('Destination already exists')
            os.rename(src[-1], dst[-1], src_dir_fd=src_fd, dst_dir_fd=dst_fd)
            os.fsync(src_fd)
            os.fsync(dst_fd)


def corruption(exc):
    """Only positive evidence of missing/unsafe bytes warrants invalidation."""
    import errno
    return (isinstance(exc, FileSafetyError)
            or isinstance(exc, OSError) and exc.errno in
            {errno.ENOENT, errno.ENOTDIR, errno.ELOOP})


def sync_tree(root):
    """Persist every regular file, then directories from leaves to root."""
    def failed(exc):
        raise exc
    for parent, dirs, files in os.walk(root, followlinks=False, onerror=failed):
        for name in dirs:
            if (Path(parent) / name).is_symlink():
                raise FileSafetyError('Unsafe publication directory')
        for name in files:
            with open_file(root, (Path(parent) / name).relative_to(root).as_posix()) as stream:
                os.fsync(stream.fileno())
    for parent, _, _ in os.walk(root, topdown=False, followlinks=False, onerror=failed):
        sync_directory(parent)


def publication_barrier(root, attempt_id, *, moved):
    """Establish the same durable boundary after a crash or a normal rename."""
    relative(attempt_id)
    location = 'results/' if moved else 'attempts/'
    sync_tree(Path(root) / location / attempt_id)
    if not moved:
        rename(root, 'attempts/' + attempt_id, 'results/' + attempt_id)
    # A prior rename may have failed after the syscall, before either fsync.
    # Reading the destination or its hashes does not establish durability.
    sync_directory(Path(root) / 'attempts')
    sync_directory(Path(root) / 'results')
