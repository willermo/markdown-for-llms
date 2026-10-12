#!/usr/bin/env python3
"""Inventario esterno degli helper, corpus e tool: complementa S/B/I."""
import argparse
from pathlib import Path
from data import descriptor, digest, load, write_new
from safety import no_symlink_root, regular


def capture(repo, extras=()):
    repo = no_symlink_root(repo)
    names = set()
    for directory in (repo / "scripts/benchmarks", repo / "tests/fixtures/conversion-corpus"):
        for path in directory.rglob("*"):
            if path.is_symlink():
                raise ValueError("Input esterno symlink")
            if path.is_file() and "__pycache__" not in path.parts:
                names.add(path)
    names.update(repo / name for name in ("tests/unit/test_conversion_benchmark.py", "tests/unit/test_conversion_comparison.py", "tests/unit/test_conversion_fix_r002.py", "tests/integration/test_conversion_writer.py") if (repo / name).is_file())
    names.update(Path(p) for p in extras)
    files = []
    for path in sorted(names):
        if path.is_relative_to(repo):
            regular(repo, path.relative_to(repo).as_posix())
        elif any(p.is_symlink() for p in (path, *path.parents)):
            raise ValueError("Tool/font esterno non canonico")
        files.append(descriptor(path))
    payload = {"schema_version": 1, "scope": "benchmark-inputs", "files": files}
    return dict(payload, digest=digest(payload))


def verify(value):
    payload = {k: v for k, v in value.items() if k != "digest"}
    if value.get("schema_version") != 1 or value.get("scope") != "benchmark-inputs" or digest(payload) != value.get("digest"):
        raise ValueError("Manifest esterno invalido")
    for row in value["files"]:
        path = Path(row["path"])
        if any(p.is_symlink() for p in (path, *path.parents)) or descriptor(path) != row:
            raise ValueError("Input esterno STALE: " + str(path))
    return value["digest"]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest="command", required=True)
    c = commands.add_parser("capture")
    c.add_argument("--repo", type=Path, required=True)
    c.add_argument("--extra", type=Path, action="append", default=[])
    c.add_argument("--output", type=Path, required=True)
    v = commands.add_parser("verify")
    v.add_argument("manifest", type=Path)
    args = p.parse_args()
    if args.command == "capture":
        value = capture(args.repo, args.extra)
        write_new(args.output, value)
        print("CAPTURED " + value["digest"])
    else:
        print("PASS_EXTERNAL_INPUTS " + verify(load(args.manifest)))


if __name__ == "__main__":
    main()
