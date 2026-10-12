"""E08: risoluzione del token Pandoc nel PATH chiuso, prima e dopo il figlio."""
import hashlib
import os
from pathlib import Path
import shutil
import subprocess

ALLOWED_PATH='/usr/bin'


def check(environment,expected,version='3.1.3'):
    path=environment.get('PATH')
    if path!=ALLOWED_PATH:raise ValueError('Pandoc PATH outside closed allowlist')
    resolved=shutil.which('pandoc',path=path)
    if resolved!='/usr/bin/pandoc':raise ValueError('Pandoc token resolution mismatch')
    real=Path(resolved).resolve(strict=True)
    if str(real)!=str(Path(expected['path']).resolve(strict=True)):raise ValueError('Pandoc realpath mismatch')
    before=hashlib.sha256(real.read_bytes()).hexdigest()
    if before!=expected['sha256']:raise ValueError('Pandoc binary STALE')
    result=subprocess.run([str(real),'--version'],env=environment,stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=5,check=False)
    # This is the already hash-verified system tool; document data is absent.
    line=result.stdout.splitlines()[0].decode('ascii') if result.stdout else ''
    if result.returncode or line!='pandoc '+version:raise ValueError('Pandoc version mismatch')
    after=hashlib.sha256(real.read_bytes()).hexdigest()
    if after!=before:raise ValueError('Pandoc binary changed during resolution')
    return {'token':'pandoc','PATH':path,'which':resolved,'realpath':str(real),'sha256':before,'version':line,'version_argv':[str(real),'--version']}
