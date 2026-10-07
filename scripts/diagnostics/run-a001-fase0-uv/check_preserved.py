#!/usr/bin/env python3
"""Verifica il target s007 chiuso e misura budget senza seguire link storici."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat


def measure(root):
    logical = allocated = count = 0
    for directory, dirs, names in os.walk(root, followlinks=False):
        for name in names + [n for n in dirs if (Path(directory) / n).is_symlink()]:
            info = (Path(directory) / name).lstat()
            logical += info.st_size
            allocated += info.st_blocks * 512
            count += 1
    return dict(logical=logical, allocated=allocated, files=count)


def verify(repo, manifest):
    data = json.loads(manifest.read_text())
    expected = data['files'] + [data['lock']]
    if len({r['path'] for r in expected}) != len(expected):
        raise ValueError('Inventario duplicato')
    for item in expected:
        relative = Path(item['path'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Path evasivo')
        path = repo / relative
        if any(p.is_symlink() for p in (path, *path.parents)):
            raise ValueError('Link nel target')
        info = path.lstat()
        content = path.read_bytes()
        if (not stat.S_ISREG(info.st_mode) or len(content) != item['bytes'] or
                hashlib.sha256(content).hexdigest() != item['sha256']):
            raise ValueError('Target alterato: ' + item['path'])
    target = repo / data['target']
    actual = set()
    directories = 1  # L'inventario s007 include la radice del target.
    for directory, dirs, names in os.walk(target, followlinks=False):
        directories += len(dirs)
        for name in dirs + names:
            if (Path(directory) / name).is_symlink():
                raise ValueError('Link nel target chiuso')
        actual.update((Path(directory)/n).relative_to(repo).as_posix() for n in names)
    if actual != {r['path'] for r in data['files']} or directories != data['totals']['directories']:
        raise ValueError('Albero target non chiuso')
    return dict(status='PASS_PRESERVATION_ONLY', files=len(expected), directories=directories,
                manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    result = verify(a.repo, a.manifest)
    result['run'] = measure(a.repo/'temp/run-a001-fase0-uv')
    result['free'] = shutil.disk_usage(a.repo).free
    result['reserves'] = 80 * 1024**2
    result['admission'] = (max(result['run']['logical'], result['run']['allocated']) +
                           result['reserves'] < 384*1024**2 and result['free'] >= 516*1024**2)
    with a.output.open('x') as f:
        json.dump(result, f, indent=2); f.write('\n')
    print(json.dumps(result))
    return 0 if result['admission'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
