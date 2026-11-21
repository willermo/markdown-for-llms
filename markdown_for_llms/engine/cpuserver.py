"""The local CPU server of the `marker-server` route (MFL033, plan r2 §7): `llama-server` of llama.cpp
with the GGUF weights of Surya OCR 2.5, on the user's CPU.

`engine/data/cpu-server.json` pins the release (`b12040`, its tarball by URL, bytes and SHA-256) and
the two GGUF files (repository, revision, bytes, SHA-256), the arguments and the memory cap.
`prepare` (the only download, `mdf prepare --cpu-server`) fetches what is missing with curl into
`<root>` (default `${XDG_CACHE_HOME:-~/.cache}/markdown-for-llms/cpu-server`), each file through a
`.partial` checked before its rename, never touching a present file that differs; it extracts the
tarball (members checked: regular files and directories under one tree, links only inside it)
into `llama-<release>/` with a manifest of its files, and records whether `--host` of that build
binds a Unix socket (`.sock` in its `--help`): without it the route cannot run in the sandbox.

`verify` checks all of it before every start. `start` runs `llama-server` in Bubblewrap without
network (`isolation.program_command`), the tree and the weights read-only, the `llama/` directory
of the private socket directory on `/relay`, `--host /relay/llama.sock`, under a transient scope
`mdf-eng-<attempt>-llama` with its memory cap; `stop` kills it and its scope.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

from . import isolation, weights
from .codes import Code, EngineError

DATA = Path(__file__).resolve().parent / "data" / "cpu-server.json"
SOCKET_IN_SANDBOX = "/relay/llama.sock"
TREE = "/opt/llama"
MODELS = "/models"


def manifest(path=None):
    return json.loads(Path(path or DATA).read_text())


def default_root():
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "markdown-for-llms" / "cpu-server"


def _entries(m):
    """The downloads of the manifest as `weights.check_file` entries, paths relative to the root."""
    tar = m["llama_server"]
    return [{"path": tar["path"], "url": tar["url"], "bytes": tar["bytes"], "sha256": tar["sha256"]}] + [
        {"path": "gguf/" + f["path"], "url": f["url"], "bytes": f["bytes"], "sha256": f["sha256"]} for f in m["gguf"]["files"]]


def _tree_dir(root, m):
    return Path(root) / ("llama-" + m["llama_server"]["release"])


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _inventory(tree):
    out = {}
    for parent, _, names in os.walk(tree, followlinks=False):
        for name in sorted(names):
            path = Path(parent) / name
            rel = str(path.relative_to(tree))
            out[rel] = ("link:" + os.readlink(path)) if path.is_symlink() else _sha256(path)
    return dict(sorted(out.items()))


def _extract(archive, dest):
    """Extract the tarball under `dest`: regular files, directories and links that stay inside it."""
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar.getmembers():
            name = Path(member.name)
            if name.is_absolute() or ".." in name.parts:
                raise EngineError(Code.TOOLCHAIN_MISMATCH, "cpu server archive: member outside its tree")
            if member.issym() or member.islnk():
                target = (Path(dest) / name.parent / member.linkname).resolve()
                if not target.is_relative_to(Path(dest).resolve()) or Path(member.linkname).is_absolute():
                    raise EngineError(Code.TOOLCHAIN_MISMATCH, "cpu server archive: link outside its tree")
            elif not (member.isfile() or member.isdir()):
                raise EngineError(Code.TOOLCHAIN_MISMATCH, "cpu server archive: special member")
        tar.extractall(dest, filter="data")


def _binary(tree):
    found = sorted(p for p in Path(tree).rglob("llama-server") if p.is_file() and not p.is_symlink())
    if len(found) != 1:
        raise EngineError(Code.SERVER_NOT_CONFIGURED, "llama-server not found once in the prepared tree")
    return found[0]


def prepare(root=None, fetch=None, log=None, m=None):
    """Download, check and extract (see the module docstring); returns the record of `verify` with
    `downloaded` and `kept`."""
    fetch = fetch or weights._curl
    log = log or (lambda line: print(line, file=sys.stderr))
    m = m or manifest()
    root = Path(root or default_root()).expanduser().absolute()
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    if fetch is weights._curl and shutil.which("curl") is None:
        raise EngineError(Code.SERVER_NOT_CONFIGURED, "curl is needed to download the CPU server")
    kept, downloaded = [], []
    for entry in _entries(m):
        target = root / entry["path"]
        target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        bad = weights.check_file(str(root), entry)
        if bad is None:
            kept.append(entry["path"])
            continue
        if bad[0] == Code.WEIGHTS_MISMATCH or os.path.lexists(target):
            raise EngineError(Code.SERVER_NOT_CONFIGURED, entry["path"] + ": present and different (" + bad[1] + ")")
        partial = target.with_name(target.name + ".partial")
        if partial.is_symlink() or (partial.exists() and not partial.is_file()):
            raise EngineError(Code.SERVER_NOT_CONFIGURED, str(partial) + ": not a regular file")
        if partial.exists() and partial.stat().st_size > entry["bytes"]:
            partial.unlink()
        for attempt in range(1, weights.ATTEMPTS + 1):
            done = fetch(entry["url"], partial)
            if done.returncode == 0 and partial.is_file() and partial.stat().st_size == entry["bytes"]:
                break
            log("cpu server: %s transfer %d of %d failed (exit %s)" % (entry["path"], attempt, weights.ATTEMPTS,
                                                                      done.returncode))
        else:
            raise EngineError(Code.SERVER_NOT_CONFIGURED, entry["path"] + ": download failed")
        bad = weights.check_file(str(root), dict(entry, path=entry["path"] + ".partial"))
        if bad is not None:
            partial.unlink()
            raise EngineError(Code.SERVER_NOT_CONFIGURED, entry["path"] + ": downloaded file differs (" + bad[1] + ")")
        os.rename(partial, target)
        downloaded.append(entry["path"])
        log("cpu server: downloaded " + entry["path"])
    tree = _tree_dir(root, m)
    if not tree.exists():
        staging = root / (tree.name + ".partial")
        shutil.rmtree(staging, ignore_errors=True)
        staging.mkdir(mode=0o700)
        try:
            _extract(root / m["llama_server"]["path"], staging)
            binary = _binary(staging)
            (root / (tree.name + ".json")).write_text(json.dumps({"binary": str(binary.relative_to(staging)),
                                                                  "files": _inventory(staging)}, indent=1))
            os.rename(staging, tree)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        os.chmod(tree / str(binary.relative_to(staging)), 0o755)
    record = verify(root, m, help_check=True)
    return dict(record, downloaded=len(downloaded), kept=len(kept))


def verify(root=None, m=None, help_check=False):
    """The checks before a start (downloads, extracted tree, socket support); returns what
    `engine.json` records. `server_not_configured` when something is missing or differs."""
    m = m or manifest()
    root = Path(root or default_root()).expanduser().absolute()
    for entry in _entries(m):
        bad = weights.check_file(str(root), entry)
        if bad is not None:
            raise EngineError(Code.SERVER_NOT_CONFIGURED, "%s: %s (mdf prepare --cpu-server)" % (entry["path"], bad[1]))
    tree = _tree_dir(root, m)
    try:
        saved = json.loads((root / (tree.name + ".json")).read_text())
    except (OSError, ValueError):
        raise EngineError(Code.SERVER_NOT_CONFIGURED, "the extracted llama-server (mdf prepare --cpu-server)") from None
    if not tree.is_dir() or tree.is_symlink() or _inventory(tree) != saved["files"]:
        raise EngineError(Code.SERVER_NOT_CONFIGURED, "the extracted llama-server differs (mdf prepare --cpu-server)")
    record = {"root": str(root), "release": m["llama_server"]["release"], "binary": saved["binary"],
              "tarball_sha256": m["llama_server"]["sha256"], "gguf_revision": m["gguf"]["revision"],
              "gguf_sha256": {f["path"]: f["sha256"] for f in m["gguf"]["files"]}, "arguments": m["arguments"],
              "memory_mib": m["memory_mib"]}
    if help_check:
        record["unix_socket"] = unix_socket(root, m)
    return record


def argv(root, m, relay_dir, threads=None, help_only=False):
    root = Path(root)
    tree = _tree_dir(root, m)
    binary = json.loads((root / (tree.name + ".json")).read_text())["binary"]
    exe = TREE + "/" + binary
    libdir = str(Path(exe).parent)
    args = ["--help"] if help_only else (
        ["--host", SOCKET_IN_SANDBOX, "--model", MODELS + "/" + m["gguf"]["files"][0]["path"],
         "--mmproj", MODELS + "/" + m["gguf"]["files"][1]["path"]] + list(m["arguments"]))
    if threads is not None and not help_only:
        i = args.index("--threads")
        args[i + 1] = str(threads)
    binds = [] if relay_dir is None else [(str(relay_dir), "/relay")]
    return isolation.program_command(exe, args, ro_binds=[(str(tree), TREE), (str(root / "gguf"), MODELS)],
                                     binds=binds, env={"LD_LIBRARY_PATH": libdir})


def unix_socket(root, m=None):
    """True when `--host` of the prepared build binds a Unix socket (`.sock` in its `--help`)."""
    m = m or manifest()
    done = subprocess.run(argv(root, m, None, help_only=True), stdin=subprocess.DEVNULL, capture_output=True,
                          timeout=60, env={"PATH": "/usr/bin:/bin"})
    text = (done.stdout + done.stderr).decode("utf-8", "replace")
    host = [line for line in text.splitlines() if "--host" in line]
    following = text[text.find("--host"):text.find("--host") + 400] if "--host" in text else ""
    return bool(host) and ".sock" in following


def start(root, relay_dir, scope, log_path, m=None, threads=None):
    """`llama-server` in its sandbox and scope (see the module docstring); `isolation.stop` ends it."""
    m = m or manifest()
    return isolation.start(argv(root, m, relay_dir, threads), scope, m["memory_mib"], log_path)


def main(argv_=None):
    """`python -m markdown_for_llms.engine.cpuserver prepare|verify [ROOT]`."""
    args = sys.argv[1:] if argv_ is None else list(argv_)
    if not args or args[0] not in ("prepare", "verify") or len(args) > 2:
        print("usage: python -m markdown_for_llms.engine.cpuserver prepare|verify [ROOT]", file=sys.stderr)
        return 2
    try:
        result = prepare(args[1] if len(args) > 1 else None) if args[0] == "prepare" else \
            verify(args[1] if len(args) > 1 else None, help_check=True)
    except EngineError as exc:
        print(json.dumps({"status": "failed", "error": exc.code.value, "detail": exc.detail}))
        return 1
    print(json.dumps(result, indent=1))
    return 0 if result.get("unix_socket", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
