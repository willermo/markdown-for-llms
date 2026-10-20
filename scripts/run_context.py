#!/usr/bin/env python3
"""Fingerprint review inputs of a local run context and check them later. Standard library only."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys


RUN_ID = re.compile(r"run-[a-z][0-9]{3,}-fase[a-z0-9]+-[a-z0-9]+(?:-[a-z0-9]+)*\Z")
LABEL = re.compile(r"[a-z0-9][a-z0-9-]{0,79}\Z")
EXCLUDED_PREFIXES = ("temp/", "documentation/runs/")


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], stderr=subprocess.PIPE)


def timestamp():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def repository_root():
    return Path(git(Path.cwd(), "rev-parse", "--show-toplevel").decode().strip()).resolve()


def check_local_path(root, path):
    """Reject symlinks in paths used for context reads/writes, including ancestors."""
    relative = path.relative_to(root)
    current = root
    for component in relative.parts:
        current = current / component
        if current.is_symlink():
            raise ValueError(f"Symbolic link not allowed in the context path: {current}")


def context_root(root):
    path = root / "temp"
    check_local_path(root, path)
    result = subprocess.run(
        ["git", "-C", str(root), "check-ignore", "--quiet", "--no-index", "temp/_context_probe_"],
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        raise ValueError("temp/ must be ignored by Git before it holds a local context.")
    return path


def run_path(root, run_id, must_exist=True):
    if not RUN_ID.fullmatch(run_id):
        raise ValueError("Invalid run ID; example: run-a001-fase0-uv")
    path = context_root(root) / run_id
    check_local_path(root, path)
    if must_exist and not path.is_dir():
        raise ValueError(f"Run not found: {run_id}")
    return path


def write_new(root, path, body):
    check_local_path(root, path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(body)


def digest_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(root, relative):
    path = root / relative
    check_local_path(root, path.parent)
    if path.is_symlink():
        return {"path": relative, "kind": "symlink", "sha256": hashlib.sha256(os.fsencode(os.readlink(path))).hexdigest()}
    if not path.exists():
        return {"path": relative, "kind": "deleted"}
    if not path.is_file():
        raise ValueError(f"Object not supported in a snapshot (for example a submodule): {relative}")
    return {
        "path": relative, "kind": "file", "sha256": digest_file(path),
        "executable": bool(path.stat().st_mode & 0o111),
    }


def capture(root, run_dir, artifacts):
    names = git(root, "ls-files", "--cached", "--others", "--exclude-standard", "-z").split(b"\0")
    relative_paths = sorted({os.fsdecode(name) for name in names if name})
    files = [file_record(root, name) for name in relative_paths if not name.startswith(EXCLUDED_PREFIXES)]
    artifact_records = []
    for name in sorted(set(artifacts)):
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Artifacts must have relative paths confined to the run.")
        path = root / relative
        path.relative_to(run_dir)
        check_local_path(root, path)
        if not path.is_file():
            raise ValueError(f"Missing artifact: {name}")
        artifact_records.append(file_record(root, relative.as_posix()))
    canonical = json.dumps(files, sort_keys=True, ensure_ascii=True, separators=(",", ":")).encode()
    return {
        "schema": 1,
        "head": git(root, "rev-parse", "HEAD").decode().strip(),
        "branch": git(root, "branch", "--show-current").decode().strip() or "DETACHED",
        "excluded_prefixes": list(EXCLUDED_PREFIXES),
        "worktree_sha256": hashlib.sha256(canonical).hexdigest(),
        "files": files,
        "artifacts": artifact_records,
    }


def snapshot_path(root, run_dir, label):
    if not LABEL.fullmatch(label):
        raise ValueError("Invalid snapshot label.")
    path = run_dir / "snapshots" / (label + ".json")
    check_local_path(root, path)
    return path


def snapshot(root, args):
    run_dir = run_path(root, args.run_id)
    output = snapshot_path(root, run_dir, args.label)
    if output.exists():
        raise ValueError("Snapshot already present: use a new revision.")
    data = capture(root, run_dir, args.artifact)
    data.update({"created_at": timestamp(), "run_id": args.run_id, "label": args.label})
    write_new(root, output, json.dumps(data, indent=2, ensure_ascii=True) + "\n")
    print(f"Snapshot: {output.relative_to(root)}\nWorking tree SHA-256: {data['worktree_sha256']}")


def verify(root, args):
    run_dir = run_path(root, args.run_id)
    path = snapshot_path(root, run_dir, args.label)
    saved = json.loads(path.read_text(encoding="utf-8"))
    if saved.get("schema") != 1 or saved.get("run_id") != args.run_id or saved.get("label") != args.label:
        raise ValueError("Invalid snapshot schema or identity.")
    current = capture(root, run_dir, [item["path"] for item in saved["artifacts"]])
    differences = [key for key in current if current[key] != saved.get(key)]
    if differences:
        print("STALE: snapshot differs; changed fields: " + ", ".join(differences))
        return 2
    print("MATCH: objects unchanged. This check is not a review or a GO.")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    snap = commands.add_parser("snapshot", help="Fingerprint working tree and selected run artifacts")
    snap.add_argument("run_id")
    snap.add_argument("--label", required=True)
    snap.add_argument("--artifact", action="append", default=[])
    check = commands.add_parser("verify", help="Check snapshot without modifying files")
    check.add_argument("run_id")
    check.add_argument("--label", required=True)
    args = parser.parse_args()
    try:
        root = repository_root()
        if args.command == "snapshot":
            snapshot(root, args)
        else:
            return verify(root, args)
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
