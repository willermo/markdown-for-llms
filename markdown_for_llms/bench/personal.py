"""The personal set of a run (plan r2 §13, delta r3 Δ9): local documents, copied before they are converted.

Each input (and twin) is opened with `O_NOFOLLOW` and must be a regular file of the user (`fstat`
on the descriptor); from that descriptor a stable copy is written to `RUN/inputs/<id>.<ext>`
(mode 0600), with its SHA-256 computed while copying, and the engine converts the copy, so a file
changed or replaced during the run cannot reach the engine. The copy goes to a temporary file of
the same directory, made durable with fsync and published with `os.replace` only after the whole
file was read (B-COPIA of the review of the final diff): a copy cut short (Ctrl-C, SIGKILL, a read
or write error, a full disk) is never published, and stays as `.partial-<name>-*`, as evidence. An
input that cannot be read so is `input_unreadable`: the document is unavailable. A resumed run
reuses a copy only when the state registered its SHA-256 (`input_sha256`, `twin_sha256`) and the
copy still has it; without a registration it copies again, and a registered copy that changed is
`resume_identity_changed`. The check set of
a document with a twin is generated in the run (`RUN/checks/<id>.json`), never in the shared
cache: it holds tokens of the document. Paths and SHA-256 stay in `state.json`; the report of a
personal set has numbers only.
"""
import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile

from .cache import write_atomic
from .codes import BenchError, Code
from .fetch import DocumentFiles

CHUNK = 1 << 20
PARTIAL = ".partial-"


def stable_copy(path, target):
    """Copy a regular file of the user to `target` (0600); its SHA-256. The bytes go to a new temporary
    file `.partial-<name>-*` of the same directory, made durable, and `os.replace` publishes it only
    at the end of the file; a copy cut short stays there as evidence and never becomes `target`."""
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK)
    except OSError:
        raise BenchError(Code.INPUT_UNREADABLE, "personal input cannot be opened") from None
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid():
            raise BenchError(Code.INPUT_UNREADABLE, "personal input: a regular file of the user expected")
        target = Path(target)
        target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        out, partial = tempfile.mkstemp(prefix=f"{PARTIAL}{target.name}-", dir=target.parent)    # new, 0600
        digest = hashlib.sha256()
        try:
            while True:
                chunk = os.read(fd, CHUNK)
                if not chunk:
                    break
                digest.update(chunk)
                view = memoryview(chunk)
                while view:
                    view = view[os.write(out, view):]
            os.fsync(out)
        finally:
            os.close(out)
        os.replace(partial, target)
        dfd = os.open(target.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
        return digest.hexdigest()
    finally:
        os.close(fd)


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def _copy(path, target, registered):
    """The SHA-256 of the copy of `path` at `target`. An existing copy is reused only when the state
    registered a SHA-256 for it (the run compares the two: a registered copy that changed is
    `resume_identity_changed`); without a registration the copy may be incomplete or absent, and a
    new stable copy replaces it."""
    target = Path(target)
    if registered is not None:
        try:
            regular = stat.S_ISREG(os.lstat(target).st_mode)
        except FileNotFoundError:
            regular = False
        if regular:
            return _sha256(target)
    return stable_copy(path, target)


def resolve(doc, run_dir, input_sha256=None, twin_sha256=None):
    """The files of a personal document: its stable copies under `RUN/inputs/`; `input_sha256` and
    `twin_sha256` are the SHA-256 the state registered for them (None: no copy is reused)."""
    from .run import input_name
    inputs = Path(run_dir) / "inputs"
    name = input_name(doc)
    source = _copy(doc["input"]["path"], inputs / name, input_sha256)
    twin, hashes = {}, {"source": source, "twin": None}
    if doc["twin"] is not None:
        twin_name = f"{doc['id']}-twin.{doc['twin']['kind']}"
        hashes["twin"] = _copy(doc["twin"]["path"], inputs / twin_name, twin_sha256)
        twin["file"] = inputs / twin_name
    return DocumentFiles(doc["id"], inputs / name, twin, [], hashes)


def checkset(doc, files, run_dir):
    """The check set of a personal document with a twin, kept in the run."""
    from .cache import canonical
    from .checks import cache_key, checkset as generate_set
    from .readers import read_twin
    path = Path(run_dir) / "checks" / f"{doc['id']}.json"
    key = cache_key(doc, files.hashes)
    if path.is_file():
        try:
            body = json.loads(path.read_bytes())
            if body.get("key") == key and body.get("checks_sha256") == hashlib.sha256(canonical(body["checks"])).hexdigest():
                return body
        except ValueError:
            pass
    body = generate_set(doc, files, read_twin(doc, files))
    write_atomic(path, json.dumps(body, ensure_ascii=False, sort_keys=True).encode("utf-8"))
    os.chmod(path, 0o600)
    return body
