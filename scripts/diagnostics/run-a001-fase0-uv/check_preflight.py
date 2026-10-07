#!/usr/bin/env python3
"""Valida I/E corrente in un processo -I -B prima di collection/import."""
import argparse
import importlib.util
from pathlib import Path
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', required=True, type=Path)
    p.add_argument('--source', required=True, type=Path)
    p.add_argument('--receipt', required=True, type=Path)
    p.add_argument('--standalone', action='store_true')
    a = p.parse_args()
    spec = importlib.util.spec_from_file_location('distribution_gate', Path(__file__).with_name('verify_distribution.py'))
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    try:
        if not sys.flags.isolated or not sys.dont_write_bytecode:
            raise ValueError('Richiesto Python -I -B')
        result = tool.validate_receipt(a.receipt, a.source, a.repo, official=not a.standalone)
        if a.standalone and result['scope'] != 'standalone':
            raise ValueError('Modalità standalone richiede S/I standalone')
        print('PASS_CURRENT_' + result['kind'])
        return 0
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print('FAIL preflight: '+str(exc), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
