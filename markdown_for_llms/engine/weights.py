"""The weights of the Marker route: the 18 files of `data/weights.json` under a model root.

    <root>/indra-ocr/<commit>/<file>     the 17 files of indra-ocr v1.1.0, by commit
    <root>/fonts/GoNotoCurrent-Regular.ttf

`verify(root)` checks every file before each start of the Marker child: a regular file reached
without symbolic links, the bytes and the SHA-256 (read in streaming). A missing file is
`weights_missing`, any other difference `weights_mismatch`; nothing is ever downloaded there.
The child then gets the root read-only, in a container without network, so the two
download fallbacks of Marker and indra-ocr cannot run.

`prepare(root)` (`mdf prepare --weights`) is the only step with network: each missing file is
fetched by curl, over HTTPS only, from the URL of the manifest into `<file>.partial`, with a
bounded number of retries that resume the partial file, checked (bytes and SHA-256, and for the
two JSON files also the git blob SHA-1 that Hugging Face publishes) and renamed. A file already
present is kept when it verifies; one that does not stops the preparation without being
touched. The command line is also the precondition W of the full check:

    python -I -B -m markdown_for_llms.engine.weights verify ROOT
    python -I -B -m markdown_for_llms.engine.weights prepare ROOT
"""
import hashlib
from importlib import resources
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys

from .codes import Code, EngineError

CHUNK = 1 << 20
CURL_RETRIES = 8                    # curl's own retries of one transfer, each resuming the partial file
ATTEMPTS = 3                        # transfers of one file before the preparation stops


def manifest():
    return json.loads(resources.files(__package__).joinpath("data/weights.json").read_text(encoding="utf-8"))


def manifest_sha256():
    data = resources.files(__package__).joinpath("data/weights.json").read_bytes()
    return hashlib.sha256(data).hexdigest()


def default_root():
    """`${XDG_CACHE_HOME:-~/.cache}/markdown-for-llms/models` (D10 of MFL032), like the bench cache."""
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "markdown-for-llms" / "models"


def indra_dir(m=None):
    """The directory of the indra-ocr files, relative to the model root."""
    return (m or manifest())["indra_ocr"]["directory"]


def font(m=None):
    return (m or manifest())["font"]


def _digests(stream, size):
    """(bytes read, SHA-256, git blob SHA-1 for the declared size)."""
    h256, sha1, total = hashlib.sha256(), hashlib.sha1(b"blob %d\0" % size), 0
    for chunk in iter(lambda: stream.read(CHUNK), b""):
        total += len(chunk)
        h256.update(chunk)
        sha1.update(chunk)
    return total, h256.hexdigest(), sha1.hexdigest()


def _open_confined(root, rel):
    """A file descriptor of `root/rel`, every component reached without following a link."""
    fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        parts = rel.split("/")
        for part in parts[:-1]:
            nxt = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=fd)
            os.close(fd)
            fd = nxt
        return os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=fd)
    finally:
        os.close(fd)


def check_file(root, entry):
    """None when `root/<entry path>` is the file of the manifest, else (code, reason)."""
    try:
        fd = _open_confined(root, entry["path"])
    except FileNotFoundError:
        return Code.WEIGHTS_MISSING, "missing"
    except OSError as exc:              # ELOOP: a link; ENOTDIR: a link or a file in the way
        return Code.WEIGHTS_MISMATCH, "not reachable without links (%s)" % type(exc).__name__
    info = os.fstat(fd)
    if not stat.S_ISREG(info.st_mode):
        os.close(fd)
        return Code.WEIGHTS_MISMATCH, "not a regular file"
    with os.fdopen(fd, "rb") as stream:
        if info.st_size != entry["bytes"]:
            return Code.WEIGHTS_MISMATCH, "%d bytes, %d expected" % (info.st_size, entry["bytes"])
        total, digest, sha1 = _digests(stream, entry["bytes"])
    if total != entry["bytes"] or digest != entry["sha256"]:
        return Code.WEIGHTS_MISMATCH, "SHA-256 differs"
    if "git_sha1" in entry and sha1 != entry["git_sha1"]:
        return Code.WEIGHTS_MISMATCH, "git SHA-1 differs"
    return None


def verify(root):
    """Check all the files of the manifest under `root`; returns what `engine.json` records.
    The first missing file is `weights_missing`, any other difference `weights_mismatch`."""
    m = manifest()
    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise EngineError(Code.WEIGHTS_MISSING, "model root is not a directory")
    worst = None
    for entry in m["files"]:
        bad = check_file(str(root), entry)
        if bad is not None:
            if bad[0] == Code.WEIGHTS_MISMATCH:
                raise EngineError(bad[0], entry["path"] + ": " + bad[1])
            worst = worst or EngineError(bad[0], entry["path"] + ": " + bad[1])
    if worst is not None:
        raise worst
    return {"manifest_sha256": manifest_sha256(), "indra_ocr": m["indra_ocr"],
            "files": len(m["files"]), "bytes": sum(e["bytes"] for e in m["files"])}


def _curl(url, partial):
    return subprocess.run(
        ["curl", "-fsSL", "--proto", "=https", "--proto-redir", "=https", "--retry", str(CURL_RETRIES),
         "--retry-all-errors", "--retry-delay", "3", "--connect-timeout", "30", "--max-time", "1800",
         "-C", "-", "-o", str(partial), url],
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=3600, check=False)


def prepare(root, fetch=None, log=None):
    """Download the missing files of the manifest into `root`; returns {kept, downloaded}.
    `fetch(url, partial)` (tests) replaces curl and returns an object with `returncode`."""
    fetch = fetch or _curl
    log = log or (lambda line: print(line, file=sys.stderr))
    m = manifest()
    root = Path(root).expanduser().absolute()
    if fetch is _curl and shutil.which("curl") is None:
        raise EngineError(Code.WEIGHTS_MISSING, "curl is needed to download the weights")
    kept, downloaded = [], []
    for entry in m["files"]:
        target = root / entry["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        bad = check_file(str(root), entry)
        if bad is None:
            kept.append(entry["path"])
            continue
        if bad[0] == Code.WEIGHTS_MISMATCH or os.path.lexists(target):
            raise EngineError(Code.WEIGHTS_MISMATCH, entry["path"] + ": present and different (" + bad[1]
                              + "); left untouched")
        partial = target.with_name(target.name + ".partial")
        if partial.is_symlink() or (partial.exists() and not partial.is_file()):
            raise EngineError(Code.WEIGHTS_MISMATCH, str(partial) + ": not a regular file")
        if partial.exists() and partial.stat().st_size > entry["bytes"]:
            partial.unlink()                    # longer than the file: no resumable prefix
        for attempt in range(1, ATTEMPTS + 1):
            done = fetch(entry["url"], partial)
            if done.returncode == 0 and partial.is_file() and partial.stat().st_size == entry["bytes"]:
                break
            log("weights: %s transfer %d of %d failed (exit %s)" % (entry["path"], attempt, ATTEMPTS, done.returncode))
        else:
            raise EngineError(Code.WEIGHTS_MISSING, entry["path"] + ": download failed after %d transfers" % ATTEMPTS)
        bad = check_file(str(root), dict(entry, path=entry["path"] + ".partial"))
        if bad is not None:
            partial.unlink()
            raise EngineError(Code.WEIGHTS_MISMATCH, entry["path"] + ": downloaded file differs (" + bad[1] + ")")
        os.rename(partial, target)
        downloaded.append(entry["path"])
        log("weights: downloaded " + entry["path"])
    verify(root)
    return {"root": str(root), "kept": len(kept), "downloaded": len(downloaded), "manifest_sha256": manifest_sha256()}


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 2 or argv[0] not in ("verify", "prepare"):
        print("usage: python -m markdown_for_llms.engine.weights verify|prepare ROOT", file=sys.stderr)
        return 2
    try:
        result = verify(argv[1]) if argv[0] == "verify" else prepare(argv[1])
    except EngineError as exc:
        print("weights: %s (%s)" % (exc.code.value, exc.detail))
        return 1
    print("weights ok: %d files, %d bytes, manifest %s" % (
        result.get("files", len(manifest()["files"])), result.get("bytes", sum(e["bytes"] for e in manifest()["files"])),
        result["manifest_sha256"][:16]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
