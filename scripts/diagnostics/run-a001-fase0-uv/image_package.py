#!/usr/bin/env python3
"""Catena container: binding S fidato → sdist/wheel → /opt/venv; niente import ML."""
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import os


def helper():
    spec=importlib.util.spec_from_file_location('image_verify',Path(__file__).with_name('verify_distribution.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',required=True,type=Path)
    p.add_argument('--repo',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--install',action='store_true')
    a=p.parse_args();v=helper();s=v.source_tool.load_json(a.source)
    if set(s)!={'schema','S_id','S_sha256','modules','build_inputs'} or s['schema']!=1:
        raise ValueError('Binding S immagine incompleto')
    if {x['name'] for x in s['modules']}!=v.MODULE_FILES or len(s['modules'])!=10:
        raise ValueError('Dieci moduli obbligatori')
    records=s['modules']+s['build_inputs']
    if len({x['name'] for x in records})!=len(records):raise ValueError('Input duplicati')
    for x in records:
        v.safe_name(x['name']);q=a.repo/x['name']
        if not q.is_file() or q.is_symlink() or q.stat().st_size!=x['bytes'] or v.source_tool.digest(q)!=x['sha256']:
            raise ValueError('Input immagine diverso da S: '+x['name'])
    dist=a.output/'dist'
    if not a.install:
        a.output.mkdir(exist_ok=False);dist.mkdir()
        cmd=['/usr/local/bin/uv','build','--python','/usr/local/bin/python','--no-managed-python',
             '--no-python-downloads','--offline','--no-build-isolation','--build-constraints',str(a.repo/'build-constraints.txt'),'--out-dir',str(dist)]
        prep=Path(__file__).with_name('image_prepare.py')
        spec=importlib.util.spec_from_file_location('image_build_monitor',prep)
        monitor=importlib.util.module_from_spec(spec);spec.loader.exec_module(monitor)
        import setuptools.build_meta as backend
        if importlib.metadata.version('setuptools')!='84.0.0':raise ValueError('Backend84 richiesto')
        os.chdir(a.repo)
        requirements={n:getattr(backend,n)({}) for n in ('get_requires_for_build_sdist','get_requires_for_build_wheel')}
        if any(requirements.values()):raise ValueError('Nuovi requisiti backend: '+repr(requirements))
        trace=monitor.observed_run(cmd,a.repo,a.output)
        hooks=[r for r in trace['observed_processes'] if 'setuptools' in r['cmdline'] and 'build_meta' in r['cmdline']]
        if not all(any(n in r['cmdline'] for r in hooks) for n in ('build_sdist','build_wheel')):
            raise ValueError('Hook nativi B incompleti')
    tar=list(dist.glob('*.tar.gz'));wheel=list(dist.glob('*.whl'))
    if len(tar)!=1 or len(wheel)!=1:raise ValueError('Archivi canonici mancanti/ambigui')
    payload={'host_repo':str(a.repo),'modules':[dict(path=x['name'],kind='file',bytes=x['bytes'],sha256=x['sha256']) for x in s['modules']],
             'build_inputs':[dict(path=x['name'],kind='file',bytes=x['bytes'],sha256=x['sha256']) for x in s['build_inputs']]}
    files=v.check_archives(payload,tar[0],wheel[0])
    result={'schema':1,'S_id':s['S_id'],'S_sha256':s['S_sha256'],'archives':[v.descriptor(tar[0]),v.descriptor(wheel[0])],
            'modules':s['modules'],'status':'PASS_ARCHIVES_ONLY'}
    if a.install:
        cmd=['/usr/local/bin/uv','pip','install','--python','/opt/venv/bin/python','--no-deps','--no-build',
             '--reinstall-package','markdown-for-llms',str(wheel[0])]
        subprocess.run(cmd,check=True)
        result['installed']=v.check_installed(payload,files,wheel[0],'cpu');result['status']='PASS_IMAGE_PACKAGE_ONLY'
    with (a.output/('I.json' if a.install else 'B.json')).open('x') as f:json.dump(result,f,indent=2);f.write('\n')


if __name__=='__main__':main()
