#!/usr/bin/env python3
"""Preflight stdlib dell'origine CPython, prima di qualunque import applicativo."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import sysconfig


def check_origin(expected_managed):
    expected = Path(expected_managed)
    if (not expected.is_absolute() or '..' in expected.parts or
            any(p.is_symlink() for p in (expected, *expected.parents)) or not expected.is_dir()):
        raise ValueError('Directory managed assoluta, esistente e senza symlink richiesta')
    base = Path(sys.base_prefix).resolve(strict=True)
    executable = Path(sys.executable).resolve(strict=True)
    stdlib = Path(sysconfig.get_path('stdlib')).resolve(strict=True)
    if platform.python_implementation() != 'CPython' or sys.version_info[:3] != (3, 12, 13):
        raise ValueError('Richiesto CPython 3.12.13 esatto')
    if not sys.flags.isolated:
        raise ValueError('Eseguire il diagnostico con -I')
    if not base.is_relative_to(expected) or base == expected:
        raise ValueError('sys.base_prefix non appartiene al managed atteso')
    if not executable.is_relative_to(base) or not stdlib.is_relative_to(base):
        raise ValueError('Binario o stdlib esterno al managed atteso')
    records = []
    for name in (os.__file__, json.__file__, str(stdlib/'pathlib.py'), str(executable)):
        p = Path(name).resolve(strict=True)
        if not p.is_relative_to(base) or not p.is_file():
            raise ValueError('Origine stdlib/binario difforme: '+str(p))
        data = p.read_bytes()
        records.append({'path':str(p),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()})
    return {'implementation':'CPython','version':list(sys.version_info[:3]),
            'executable':sys.executable,'executable_realpath':str(executable),
            'prefix':sys.prefix,'base_prefix':str(base),'stdlib':str(stdlib),
            'expected_managed':str(expected),'isolated':True,'files':records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--expected-managed', type=Path, required=True)
    args = parser.parse_args()
    try:
        print(json.dumps(check_origin(args.expected_managed), sort_keys=True))
        return 0
    except (OSError, ValueError) as exc:
        print('FAIL origine: '+str(exc), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
