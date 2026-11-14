"""Layout of the bench cache and atomic writes.

`sources/<sha256>` downloaded files by expected hash; `quarantine/` files that did not match;
`wikisource/<revid>.json` revisions; `derived/<doc>/<key>/` derivations; `checks/<key>.json`
generated checks; `tool-work/` the private /work of each tool call (`tools.py`), removed after the
call; `toolchain/` the build logs of `prepare --toolchain`; `work/` scratch space.
Everything here may contain document text and stays local.
"""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":")).encode()


def write_atomic(path, data: bytes):
    """Write through a temporary file in the same directory, fsync, rename, fsync the directory."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".tmp-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    dfd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(dfd)
    finally:
        os.close(dfd)


def utc_stamp():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


class Cache:
    def __init__(self, root):
        self.root = Path(root)

    def __getattr__(self, name):
        dirs = {"sources_dir": "sources", "quarantine_dir": "quarantine", "wiki_dir": "wikisource",
                "derived_dir": "derived", "checks_dir": "checks", "tool_work_dir": "tool-work", "work_dir": "work"}
        if name in dirs:
            return self.root / dirs[name]
        raise AttributeError(name)

    def source_path(self, src):
        return self.sources_dir / src["sha256"]

    def quarantine(self, source_id, path=None, data=None):
        """Keep a mismatching file for local inspection; return its new path."""
        digest = sha256_file(path) if path is not None else hashlib.sha256(data).hexdigest()
        target = self.quarantine_dir / f"{source_id}-{utc_stamp()}-{digest[:16]}"
        target.parent.mkdir(parents=True, exist_ok=True)
        if path is not None:
            shutil.move(str(path), target)
        else:
            write_atomic(target, data)
        return target

    def work(self, prefix):
        self.work_dir.mkdir(parents=True, exist_ok=True)
        return Path(tempfile.mkdtemp(prefix=prefix + "-", dir=self.work_dir))
