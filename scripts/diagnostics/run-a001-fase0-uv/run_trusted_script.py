#!/usr/bin/env python3
"""Avvia una copia legacy fidata con -I -B e root esplicita, senza PYTHONPATH."""
import argparse
from pathlib import Path
import runpy
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('script', type=Path)
    p.add_argument('args', nargs=argparse.REMAINDER)
    a = p.parse_args()
    if not sys.flags.isolated or not sys.dont_write_bytecode:
        p.error('Richiesto -I -B')
    for path in (a.root, a.script):
        if not path.is_absolute() or '..' in path.parts or any(x.is_symlink() for x in (path,*path.parents)):
            p.error('Path fidati assoluti senza link richiesti')
    if a.script.parent != a.root:
        p.error('Script deve appartenere alla root dichiarata')
    sys.path.insert(0, str(a.root))
    sys.argv = [str(a.script), *a.args]
    runpy.run_path(str(a.script), run_name='__main__')


if __name__ == '__main__':
    main()
