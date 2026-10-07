#!/usr/bin/env python3
"""Guardia filesystem prima di avviare interpreti S/B/I/E; niente import prodotto.

Il backend già acquisito è verificato contro la wheel84 prima del suo startup.
L'import attestato del backend è un comando distinto, riservato al dopo freeze.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import sys
import zipfile


def descriptor(path):
    data=path.read_bytes()
    return {'path':str(path),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}


def regular(path):
    if any(p.is_symlink() for p in (path,*path.parents)) or not stat.S_ISREG(path.lstat().st_mode):
        raise ValueError('File canonico richiesto: '+str(path))
    return descriptor(path)


def inspect_prefix(python, managed, backend_wheel=None):
    if not python.is_absolute() or '..' in python.parts:
        raise ValueError('Interprete assoluto richiesto')
    managed_binary=managed/'cpython-3.12.13-linux-x86_64-gnu/bin/python3.12'
    if python.resolve(strict=True)!=managed_binary or descriptor(managed_binary)['sha256']!='8392ec78d090ab7e1b1c058cff454de64a0780e263e6a9e66cdb1256be322fb0':
        raise ValueError('Binario managed diverso')
    prefix=python.parent.parent
    for parent in (prefix,*prefix.parents):
        if parent.is_symlink():raise ValueError('Prefix evasivo')
    base=managed_binary.parent.parent
    for directory in (base/'lib/python3.12',base/'lib/python3.12/site-packages'):
        if list(directory.glob('*.pth')) or list(directory.glob('*customize*')):
            raise ValueError('Startup managed non vuoto')
    site=prefix/'lib/python3.12/site-packages'
    pth=sorted(site.glob('*.pth'))
    expected=[] if prefix==base else ['_virtualenv.pth']
    if backend_wheel:expected.append('distutils-precedence.pth')
    if sorted(p.name for p in pth)!=sorted(expected):raise ValueError('Startup .pth inatteso')
    if list(site.glob('*customize*')):raise ValueError('Customize inatteso')
    files=[regular(p) for p in pth]
    if prefix!=base:
        if (site/'_virtualenv.pth').read_bytes()!=b'import _virtualenv':raise ValueError('Hook virtualenv inatteso')
        virtual=regular(site/'_virtualenv.py')
        if virtual['sha256']!='6cf30c56faf2a55228914dbbd17f8088ed371ebb08f5e7fa6fd931f913fcaf1d':raise ValueError('Hook uv diverso da quello letto')
        files.extend([virtual,regular(prefix/'pyvenv.cfg')])
        cfg=(prefix/'pyvenv.cfg').read_text()
        if 'include-system-site-packages = false' not in cfg or 'version_info = 3.12.13' not in cfg:
            raise ValueError('Venv/config non chiusa')
    backend_files=[]
    if backend_wheel:
        regular(backend_wheel)
        if descriptor(backend_wheel)['sha256']!='51a52592b3b99e102b609654876bd65f19f999935166d1352678931132b0c670':
            raise ValueError('Wheel backend non ricevuta')
        with zipfile.ZipFile(backend_wheel) as wheel:
            for name in wheel.namelist():
                if name.endswith('/') or name.endswith('/RECORD'):continue
                rel=Path(name)
                if rel.is_absolute() or '..' in rel.parts:raise ValueError('Backend ZIP evasivo')
                p=site/rel;r=regular(p)
                if p.read_bytes()!=wheel.read(name):raise ValueError('Backend installato diverso dalla wheel: '+name)
                backend_files.append(r)
        expected_pth=b"import os; var = 'SETUPTOOLS_USE_DISTUTILS'; enabled = os.environ.get(var, 'local') == 'local'; enabled and __import__('_distutils_hack').add_shim(); \n"
        if (site/'distutils-precedence.pth').read_bytes()!=expected_pth:raise ValueError('Hook setuptools inatteso')
        names=sorted(p.name for p in site.glob('*.dist-info'))
        if names!=['setuptools-84.0.0.dist-info']:raise ValueError('Backend env contiene altre distribuzioni')
    return {'python':str(python),'real_python':str(managed_binary),'prefix':str(prefix),
            'startup':files,'backend_wheel':descriptor(backend_wheel) if backend_wheel else None,
            'backend_files':backend_files,'base':str(base)}


def check_configuration(repo, environment):
    if any(k.startswith('COVERAGE') for k in environment) or 'PYTHONPATH' in environment:
        raise ValueError('Environment startup aperto')
    absent=[Path('/etc/uv/uv.toml'),Path('/etc/uv.toml'),repo/'uv.toml',
            Path('/home/davide/.netrc'),Path('/home/davide/.pydistutils.cfg'),
            Path('/home/davide/.local/share/uv/credentials/credentials.toml')]
    for p in absent:
        if os.path.lexists(p):raise ValueError('Configurazione inattesa, contenuto non letto: '+str(p))
    config=Path(environment['XDG_CONFIG_HOME'])
    if not config.is_dir() or list(config.iterdir()):raise ValueError('Config XDG non vuota')
    return {'absent_paths':[str(p) for p in absent],'XDG_CONFIG_HOME':str(config),'contents_read':False}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--python',type=Path,required=True)
    parser.add_argument('--managed',type=Path,required=True)
    parser.add_argument('--backend-wheel',type=Path)
    args=parser.parse_args()
    try:
        print(json.dumps(inspect_prefix(args.python,args.managed,args.backend_wheel),sort_keys=True))
        return 0
    except (OSError,ValueError,zipfile.BadZipFile) as e:
        print('FAIL guardia: '+str(e),file=sys.stderr);return 2


if __name__=='__main__':raise SystemExit(main())
