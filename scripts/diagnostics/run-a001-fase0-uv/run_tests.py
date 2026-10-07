#!/usr/bin/env python3
"""E v1 nel runner R corrente: preflight I prima di pytest/collection, zero build."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import subprocess


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['repo','source','installation','runner-receipt','output']:
        p.add_argument('--'+name,required=True,type=Path)
    p.add_argument('--test-mode', choices=('official','standalone'))
    p.add_argument('pytest_args',nargs=argparse.REMAINDER)
    a=p.parse_args()
    mode=os.environ.get('RUN_TEST_MODE','official')
    if mode not in ('official','standalone'):
        raise ValueError('RUN_TEST_MODE deve essere official o standalone')
    if a.test_mode:
        if 'RUN_TEST_MODE' in os.environ and mode != a.test_mode:
            raise ValueError('Modalità --test-mode e RUN_TEST_MODE discordanti')
        mode = a.test_mode
    spec=importlib.util.spec_from_file_location('tests_gate',Path(__file__).with_name('verify_distribution.py'))
    v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)
    for path in [a.repo,a.source,a.installation,a.runner_receipt,a.output]:v.source_tool.no_symlinks(path)
    if a.output.exists():raise ValueError('Output E già presente')
    if not sys.flags.isolated or not sys.dont_write_bytecode:raise ValueError('Richiesto -I -B')
    receipt=v.validate_receipt(a.installation,a.source,a.repo,official=mode=='official')
    if mode == 'standalone' and receipt['scope'] != 'standalone':
        raise ValueError('Modalità standalone richiede S/I standalone')
    runner=v.source_tool.load_json(a.runner_receipt)
    ns=os.readlink('/proc/self/ns/net')
    if runner.get('status')!='PASS' or runner['parent']['network']['netns']!=ns or runner['child']['network']['netns']!=ns:
        raise ValueError('R corrente padre/figlio richiesto')
    files=[a.runner_receipt,Path(__file__),a.repo/'pyproject.toml']
    files+=sorted((a.repo/'tests').rglob('*.py'))+sorted(p for p in (a.repo/'tests/fixtures/run-a001-fase0-uv').rglob('*') if p.is_file())
    cache=Path(os.environ['TIKTOKEN_CACHE_DIR'])
    files+=sorted(p for p in cache.iterdir() if p.is_file())
    inputs=[v.descriptor(x) for x in files]
    observed={'collected':[],'reports':[],'collection_errors':[],'warnings':[]}
    execution_environment = dict(os.environ)  # ambiente pubblico chiuso ricevuto da R
    os.environ['RUN_TEST_MODE'] = mode
    execution_environment['RUN_TEST_MODE'] = mode
    import pytest
    class Observe:
        def pytest_collection_finish(self,session):observed['collected']=[x.nodeid for x in session.items]
        def pytest_collectreport(self,report):
            if report.failed:observed['collection_errors'].append(str(report.longrepr))
        def pytest_runtest_logreport(self,report):
            observed['reports'].append({'nodeid':report.nodeid,'when':report.when,'outcome':report.outcome,
                                        'longrepr':str(report.longrepr) if report.failed else ''})
        def pytest_warning_recorded(self,warning_message,when,nodeid,location):
            observed['warnings'].append({'when':when,'nodeid':nodeid,'message':str(warning_message.message)})
    argv=a.pytest_args[1:] if a.pytest_args[:1]==['--'] else a.pytest_args
    if not argv:raise ValueError('Selezione pytest esplicita richiesta')
    exit_code=int(pytest.main(argv,plugins=[Observe()]))
    current=True
    # Pytest può aver importato moduli dal clone e modificato sys.path: il
    # controllo installazione usa un figlio isolato nello stesso runner/interprete.
    post_command=[sys.executable,'-I','-B',str(Path(__file__).with_name('check_preflight.py')),
                  '--repo',str(a.repo),'--source',str(a.source),'--receipt',str(a.installation)]
    if mode == 'standalone':
        post_command.append('--standalone')
    post_check=None
    try:
        post=subprocess.run(post_command,env=execution_environment,cwd=a.repo,
                            shell=False,close_fds=True,capture_output=True,text=True,timeout=60)
        post_check={'argv':post_command,'exit':post.returncode,'stdout':post.stdout,'stderr':post.stderr}
        current=post.returncode==0 and inputs==[v.descriptor(x) for x in files]
    except (OSError,ValueError,KeyError,subprocess.SubprocessError) as error:
        current=False
        post_check={'argv':post_command,'error':repr(error)}
    skipped=[x for x in observed['reports'] if x['outcome']=='skipped']
    status='PASS' if exit_code==0 and current and observed['collected'] and not skipped and not observed['collection_errors'] else 'FAIL'
    observed.update(exit_code=exit_code,skipped=skipped,inputs_unchanged=current,netns=ns,isolated_postcheck=post_check)
    source=v.source_tool.load_json(a.source)
    result=v.source_tool.envelope(dict(kind='E',scope=receipt['scope'],source=dict(v.descriptor(a.source),id=source['id']),
        snapshot=receipt['snapshot'],archives=receipt['archives'],interpreter=receipt['interpreter'],profile=receipt['profile'],
        inputs=inputs,argv=sys.argv,cwd=str(Path.cwd()),status=status,
        installation=dict(v.descriptor(a.installation),id=v.source_tool.load_json(a.installation)['id']),result=observed))
    with a.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({'status':status,'collected':len(observed['collected']),'receipt':str(a.output)}))
    return 0 if status=='PASS' else (exit_code or 2)


if __name__=='__main__':raise SystemExit(main())
