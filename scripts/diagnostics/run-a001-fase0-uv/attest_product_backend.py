#!/usr/bin/env python3
"""Attestazione reale dell'import backend84, solo dopo freeze e guardia filesystem.

Con --source chiama solo i due hook get_requires prima della build nativa e
conserva i risultati reali. Non importa moduli del prodotto. Il launcher conserva
separatamente i comandi hook osservati dal wrapper durante uv build nativo.
"""
import hashlib
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import sys


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path)
    args=parser.parse_args()
    if not sys.flags.isolated or not sys.dont_write_bytecode:raise ValueError('Richiesti -I -B')
    import setuptools
    import setuptools.build_meta as backend
    if importlib.metadata.version('setuptools')!='84.0.0':raise ValueError('Backend diverso')
    paths=[Path(setuptools.__file__),Path(backend.__file__)]
    if not all(p.is_relative_to(Path(sys.prefix)) for p in paths):raise ValueError('Backend esterno al prefix')
    requirements={}
    if args.source:
        if args.source!=Path.cwd() or not args.source.is_absolute():raise ValueError('CWD sorgente atteso')
        # Setuptools scrive messaggi su stdout: la receipt conserva l'intero log
        # e l'ultima riga JSON distingue i risultati davvero restituiti.
        for name in ('get_requires_for_build_sdist','get_requires_for_build_wheel'):
            requirements[name]=getattr(backend,name)(config_settings={})
            if requirements[name]:raise ValueError('Nuovi requisiti backend fuori supply84: '+repr(requirements[name]))
    print(json.dumps({'version':'84.0.0','python':sys.executable,'base_prefix':sys.base_prefix,
                      'backend':'setuptools.build_meta','files':[{'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size} for p in paths],
                      'prebuild_hook_requirements':requirements,'native_build_hooks':'OBSERVE_SEPARATELY; these prebuild calls do not prove native build success'}))
    return 0


if __name__=='__main__':raise SystemExit(main())
