#!/usr/bin/env python3
"""Identifica gli input pubblici espliciti prima di S; nessuna acquisizione."""
import argparse,hashlib,importlib.util,json,os,stat,sys
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['repo','managed','backend-wheel','uv','tokenizer-cache','output']:
        p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args()
    if not sys.flags.isolated or not sys.dont_write_bytecode:raise ValueError('Richiesto -I -B')
    spec=importlib.util.spec_from_file_location('guard',Path(__file__).with_name('guard_product_inputs.py'));guard=importlib.util.module_from_spec(spec);spec.loader.exec_module(guard)
    guard.inspect_prefix(Path(sys.executable),a.managed)
    files={a.backend_wheel,a.uv}
    for directory in [a.managed,a.tokenizer_cache]:
        pending=[directory]
        while pending:
            path=pending.pop();info=path.lstat()
            if stat.S_ISDIR(info.st_mode):pending.extend(path.iterdir())
            elif stat.S_ISREG(info.st_mode):files.add(path)
    for name in ['.python-version','README.md','LICENSE','MANIFEST.in','pyproject.toml','uv.lock','build-constraints.txt',
                 'config.py','logging_config.py','exceptions.py','unified_converter.py','master_workflow.py','clean_markdown.py',
                 'validate_markdown.py','chunk_markdown.py','batch_monitor.py','marker_api_server.py']:
        files.add(a.repo/name)
    files.update(Path(__file__).parent.glob('*.py'))
    rows=[]
    for path in sorted(files):
        if not path.is_absolute():path=path.absolute()
        if path.is_symlink() or not path.is_file():raise ValueError('Input non regolare: '+str(path))
        h=hashlib.sha256()
        with path.open('rb') as f:
            for part in iter(lambda:f.read(65536),b''):h.update(part)
        rows.append(dict(path=str(path),bytes=path.stat().st_size,sha256=h.hexdigest()))
    backend=next(r for r in rows if r['path']==str(a.backend_wheel.absolute()))
    if backend['sha256']!='51a52592b3b99e102b609654876bd65f19f999935166d1352678931132b0c670':raise ValueError('Backend84 non identificato')
    raw=json.dumps(dict(schema=1,scope='package-inputs',files=rows,backend=dict(name='setuptools',version='84.0.0',artifacts=[str(a.backend_wheel.absolute())],config_settings={},environment={})),separators=(',',':'))
    if len(raw.encode())>8388608:raise ValueError('Inventario eccessivo')
    with a.output.open('x') as f:f.write(raw)
    print(json.dumps(dict(status='PASS_INPUT_CAPTURE_ONLY',files=len(rows),output=str(a.output))))
    return 0

if __name__=='__main__':raise SystemExit(main())
