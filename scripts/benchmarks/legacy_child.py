"""Solo PandocConverter installato, avvio -I -B senza root prodotto."""
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import sysconfig


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    if not sys.flags.isolated or not sys.dont_write_bytecode:raise ValueError('Require -I -B')
    cfg=json.loads(Path(sys.argv[1]).read_text())
    if sha(cfg['installation'])!=cfg['installation_sha256']:raise ValueError('Installation STALE')
    receipt=json.loads(Path(cfg['installation']).read_text())['payload'];origins=receipt['installed']['origins'];files={r['path']:r['sha256'] for r in receipt['installed']['files']}
    installed=[];site=Path(sysconfig.get_path('purelib'))
    for name,path in origins.items():
        spec=importlib.util.find_spec(name.removesuffix('.py'));actual=Path(spec.origin).absolute()
        if actual!=Path(path) or not actual.is_relative_to(site) or sha(actual)!=files[str(actual)]:raise ValueError('Installed origin/hash diverge')
        installed.append({'module':name,'path':str(actual),'sha256':sha(actual)})
    guard_spec=importlib.util.spec_from_file_location('a002_pandoc_guard',Path(__file__).with_name('pandoc_guard.py'));guard=importlib.util.module_from_spec(guard_spec);guard_spec.loader.exec_module(guard)
    token_before=guard.check(dict(os.environ),cfg['pandoc'],cfg['pandoc_version_required'])
    if token_before!=cfg['pandoc_resolution_before']:raise ValueError('Parent/child Pandoc token diverge')
    # No ML/cloud discovery; installed module imports are the ten verified
    # modules already covered by preflight. The only method called is Pandoc.
    from unified_converter import PandocConverter
    if Path(importlib.import_module('unified_converter').__file__).absolute()!=Path(origins['unified_converter.py']):raise ValueError('Converter runtime __file__ diverge')
    result=PandocConverter({'pandoc_options':' '.join(cfg['options']),'conversion_timeout':cfg['timeout_s']}).convert(cfg['input'])
    token_after=guard.check(dict(os.environ),cfg['pandoc'],cfg['pandoc_version_required'])
    if token_after!=token_before:raise ValueError('Pandoc changed during legacy call')
    if result.markdown_content is not None:Path(cfg['output']).write_text(result.markdown_content,encoding='utf-8')
    data={'schema_version':1,'installed_modules':installed,'success':result.success,'error':result.error,'api_processing_time_s':result.processing_time,'internal_pandoc_exit':None,'internal_argv':'reconstructed externally; not observed by this API','returned_text_origin':'serialized from ConversionResult; original temp removed by API','images_api':result.images,'pandoc_resolution_before':token_before,'pandoc_resolution_after':token_after}
    with Path(cfg['receipt']).open('x') as f:json.dump(data,f)
    return 0 if result.success else 2

if __name__=='__main__':raise SystemExit(main())
