"""Reader indipendente stdlib: non importa né esegue il prodotto."""
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import time
import traceback
import urllib.parse

ROOT = Path('/home/davide/workarea/markdown-for-llms')
RUN = ROOT / 'temp/run-a001-fase0-uv'
HERE = Path(__file__).parent
FIX = RUN / 'evidence/implementation-r001/fix-review-r003'
OLD = FIX.parent / 'fix-review-r002'
RECEPTION = RUN / 'evidence/supervisor-implementation-r003/reception-r001'
PY = ROOT / '.venv-python/cpython-3.12.13-linux-x86_64-gnu/bin/python3.12'
reads = {}
results = {}

def read(p):
    p = Path(p)
    b = p.read_bytes()
    reads[str(p)] = dict(bytes=len(b), sha256=hashlib.sha256(b).hexdigest())
    return b

def load(p):
    return json.loads(read(p))

def sha(p):
    return hashlib.sha256(read(p)).hexdigest()

def check_record(row, base=ROOT):
    p = Path(row['path'])
    if not p.is_absolute():
        p = base / p
    b = read(p)
    assert len(b) == row['bytes'] and hashlib.sha256(b).hexdigest() == row['sha256'], str(p)

def write(name, obj):
    (HERE / name).write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n')

def command(argv, cwd=ROOT, timeout=20):
    t = time.monotonic()
    p = subprocess.run([str(a) for a in argv], cwd=cwd, capture_output=True,
                       text=True, close_fds=True, timeout=timeout,
                       env={'PATH':'/usr/bin:/bin', 'LANG':'C.UTF-8',
                            'PYTHONDONTWRITEBYTECODE':'1', 'PYTHONNOUSERSITE':'1'})
    row = dict(argv=[str(a) for a in argv], cwd=str(cwd), exit=p.returncode,
               stdout=p.stdout, stderr=p.stderr, seconds=time.monotonic()-t)
    assert p.returncode == 0, row
    return row

def ledger():
    c = load(RUN/'evidence/supervisor-arbitration-implementation-r002/authorized-scope.json')['core']
    excluded = {str(ROOT/p) for p in c['exclude_new_docker_subtrees']}
    logical = allocated = 0
    for root in c['roots']:
        p = Path(root)
        pending = [str(p if p.is_absolute() else ROOT/p)]
        while pending:
            name = pending.pop()
            if name in excluded:
                continue
            try:
                s = os.lstat(name)
            except FileNotFoundError:
                continue
            logical += s.st_size
            allocated += s.st_blocks*512
            if stat.S_ISDIR(s.st_mode):
                with os.scandir(name) as es:
                    pending.extend(e.path for e in es)
    H = max(logical,allocated)
    delta = max(0,H-c['Hentry'])
    remaining = c['incremental_bytes']-delta
    total = H+max(0,remaining)+c['external_reserve_bytes']
    rf = shutil.disk_usage(ROOT).free
    tf = shutil.disk_usage('/tmp').free
    return dict(logical=logical, allocated=allocated, H=H, Hentry=c['Hentry'],
                delta=delta, remaining=remaining, with_reserves=total,
                roots=c['roots'], exclusions=sorted(excluded), repo_free=rf, tmp_free=tf,
                admitted=(delta<c['incremental_bytes'] and H<c['max_pool_bytes'] and
                          total<c['stop_bytes'] and rf>=c['repository_free_required_bytes'] and
                          tf>=c['external_tmp_free_required_bytes']))

def identity():
    expected=load(RECEPTION/'freeze-identity.json')
    p=ROOT/expected['path']
    assert len(read(p))==expected['bytes'] and sha(p)==expected['sha256']
    snap=load(p)
    assert snap['worktree_sha256']==expected['worktree_sha256']
    assert len(snap['files'])==expected['file_count']
    assert len(snap['artifacts'])==expected['artifact_count']
    cmds=[command([PY,'-I','-B',ROOT/'scripts/run_context.py','verify',RUN.name,'--label',snap['label']])]
    for args in [('rev-parse','HEAD'),('rev-parse','dev'),('merge-base','HEAD','dev'),('branch','--show-current'),('diff','--cached','--name-only')]:
        cmds.append(command(['git',*args]))
    assert all(c['stdout'].strip()=='66ba82200e5def5a4db76f9bafccb0731b506091' for c in cmds[1:4])
    assert cmds[4]['stdout'].strip()=='feature/run-a001-uv' and not cmds[5]['stdout']
    write('match-before.json',dict(identity=expected,commands=cmds,ledger=ledger()))
    received=load(RECEPTION/'reception.json')
    for row in received['received_identities']:
        if row['path'].endswith('/handovers/implementation-r001.md'):
            p=RECEPTION/'implementation-checkpoint-received-r020.md'
            b=read(p)
            assert len(b)==row['bytes'] and hashlib.sha256(b).hexdigest()==row['sha256']
        else:
            check_record(row)
    return dict(received_identities=len(received['received_identities']),all_exact=True)

def real_R():
    out=[]
    for run_id in [RUN.name,'run-b002-fase0-uv']:
        d=FIX/('R-'+run_id)
        S=load(d/'S.json'); I=load(d/'I.json')
        for e in [S,I]:
            enc=json.dumps(e['payload'],sort_keys=True,separators=(',',':'),ensure_ascii=True).encode()
            assert e['schema']==1 and e['id']=='sha256:'+hashlib.sha256(enc).hexdigest()
        assert S['payload']['scope']==I['payload']['scope']=='run-stage'
        assert I['payload']['source']['id']==S['id']
        check_record(I['payload']['source'])
        assert I['payload']['snapshot']==S['payload']['snapshot']
        assert I['payload']['status']=='PASS' and I['payload']['kind']=='I'
        interp=I['payload']['interpreter']
        assert interp['version']==[3,12,13] and interp['isolated'] and interp['executable_realpath']==str(PY)
        for row in interp['files']+I['payload']['installed']['files']+I['payload']['inputs']:
            check_record(row)
        w=load(d/'wrapper/receipt.json'); r=load(d/'wrapper/runner.json'); inside=load(d/'wrapper/inside.json')
        t=load(d/'effective-target.json'); mon=load(FIX/('monitor-'+run_id)/'receipt.json')
        inv=load(d/'invocation.json'); res=load(d/'result.json')
        assert w['status']==r['status']==inside['status']=='PASS'
        assert w['wrapper_sha256']==sha(ROOT/'scripts/diagnostics/run-a001-fase0-uv/run_offline.py')
        assert w['diagnostic_sha256']==sha(ROOT/'scripts/diagnostics/run-a001-fase0-uv/check_runner.py')
        assert w['target_sha256']==sha(d/'effective-target.json') and w['inputs_unchanged'] and w['temporary_socket_cleaned']
        assert w['binding_before']==w['binding_after']
        binding=w['binding_before']
        assert binding['exit_code']==0 and 'MATCH:' in binding['stdout']
        assert binding['path']==S['payload']['snapshot']['path']
        assert binding['sha256']==sha(binding['path'])==S['payload']['snapshot']['sha256']
        assert run_id in binding['argv']
        assert '--bundle' not in inv['argv'] and '--snapshot' in inv['argv']
        assert '--net=none' in w['invocation']['argv'] and w['invocation']['exit_code']==0
        assert w['synthetic_host_before']['connected'] and w['synthetic_host_after']['connected']
        assert not Path(w['synthetic_socket']).exists()
        assert r['same_namespace'] and inside['netns']!=w['host_netns']
        for party in ['parent','child']:
            q=r[party]
            assert q['status']=='PASS' and q['realpath']==str(PY) and q['executable']==t['python']
            assert q['network']['netns']==inside['netns'] and q['network']['no_external_routes_or_addresses']
            assert [v['name'] for v in q['network']['links']]==['lo']
            assert all(not v['connected'] for v in q['internet_probes']+q['daemon_probes'])
            assert q['socketpair_positive'] and not q['synthetic_probe']['connected']
            assert all(v.get('matches',True) for v in q['prerequisites'])
        assert inside['preflights']==[dict(argv=load(d/'preflight.json')['commands'][0],exit_code=0,stdout='PASS_CURRENT_I\n',stderr='')]
        c=inside['command']
        assert c['argv'][:3]==[t['python'],'-I','-B'] and c['exit_code']==0 and not c['timed_out']
        assert c['command_namespace_observed'] and c['observed_namespaces_match']
        assert mon['exit']==0 and mon['stop_reason'] is None
        assert not load(d/'process-collection.json')['survivors']
        for row in t['files']:
            check_record(row)
        for n in ['target.json','startup-before.json','invocation.json','wrapper/runner-stdout.txt','wrapper/runner-stderr.txt','wrapper/command-stdout.txt','wrapper/command-stderr.txt']:
            read(d/n)
        read(FIX/('monitor-'+run_id)/'invocation.json')
        out.append(dict(run_id=run_id,S=S['id'],I=I['id'],binding=binding,
                        target_python=t['python'],realpath=interp['executable_realpath'],
                        namespace=inside['netns'],host_namespace=w['host_netns'],
                        preflight=inside['preflights'],monitor_seconds=mon['duration_seconds'],charged_at=res['charged_at'],
                        socket_and_processes_collected=True,current_payload_hashes_exact=True))
    clone=FIX/'clone-b002'
    for p in (clone/'scripts').rglob('*.py'):
        assert sha(p)==sha(ROOT/p.relative_to(clone))
    alternates=read(clone/'.git/objects/info/alternates').decode()
    assert alternates==str(ROOT/'.git/objects')+'\n'
    calls=load(FIX/'calls.json')
    gitcalls=[v for v in calls if v['argv'][0]=='git']
    assert len(gitcalls)==3 and all(v['cwd']==str(clone) and v['exit']==0 for v in gitcalls)
    for row in calls:
        assert row['exit']==0
    return dict(real_received_R=out,clone_tools_all_equal=True,alternate_readonly_usage=alternates.strip(),git_calls_only_clone=gitcalls,
                received_mock_call=calls[0],new_product_probes_by_reviewer=0)

def reuse_docs():
    eq=load(OLD/'equivalence-final.json')
    for pair in eq['Docker_r007_context_objects']:
        for row in pair.values():check_record(row)
    for row in eq['C_docker_driver_probe_and_test_exact']:
        assert sha(ROOT/row['path'])==row['sha256']
    for row in load(OLD/'actual-B-artifacts-s056.json').values():check_record(row)
    before=load(OLD/'S-s060.json')['payload'];now=load(FIX/'R-run-a001-fase0-uv/S.json')['payload']
    assert all(now[k]==before[k] for k in ['modules','build_inputs','backend','toolchain'])
    for row in now['modules']+now['build_inputs']:
        p=ROOT/row['path']
        if row['kind']=='absent':assert not p.exists()
        else:assert sha(p)==row['sha256']
    inv=load(FIX/'inventory-a001.json')
    for row in inv['files']:check_record(row)
    delta=load(FIX/'inventory-delta.json')
    assert len(delta)==1 and delta[0]['after']['path'].endswith('/run_offline.py')
    links=[]
    for name in ['docs/how-to/ambiente-uv.md','docs/how-to/marker-legacy-docker.md','docs/reference/toolchain-legacy.md']:
        p=ROOT/name;txt=read(p).decode()
        assert 'RUN_TEST_MODE=official|standalone' in txt and 'solo flag' in txt
        for no,line in enumerate(txt.splitlines(),1):
            assert line.rstrip()==line
            for target in re.findall(r'(?<!!)\[[^\]]*\]\(([^)]+)\)',line):
                if target.startswith(('http:','https:','mailto:','#')):continue
                assert (p.parent/urllib.parse.unquote(target.split('#')[0])).exists()
                links.append(dict(path=name,line=no,target=target))
    uv=read(ROOT/'docs/how-to/ambiente-uv.md').decode()
    assert 'verify_distribution.py --standalone' not in uv and 'I deriva lo scope da S' in uv
    ref=read(ROOT/'docs/reference/toolchain-legacy.md').decode()
    assert 'RUN_IMAGE_CONTEXT_RECEIPT' in ref
    parser=ast.parse(read(ROOT/'scripts/diagnostics/run-a001-fase0-uv/verify_distribution.py').decode())
    options=[n.value for n in ast.walk(parser) if isinstance(n,ast.Constant) and isinstance(n.value,str) and n.value.startswith('--')]
    assert '--standalone' not in options
    docs_delta=load(RECEPTION/'supervisor-documentation-delta.json')
    technical=load(RUN/'snapshots/impl-r001-stage-launcher-s076.json')
    current=load(RUN/'snapshots/impl-r001-stage-final-s003.json')
    oldfiles={v['path']:v for v in technical['files']};newfiles={v['path']:v for v in current['files']}
    changed=[p for p in oldfiles if oldfiles[p]!=newfiles[p]]
    assert set(changed)=={v['path'] for v in docs_delta['delta']}
    for p in changed:
        read(RECEPTION/'before'/p);read(ROOT/p)
    for name in ['static.json','reuse.json','matrix.json','delivery.json','outer-invocations.json','execution-result.json']:
        load(FIX/name)
    for p in [ROOT/'scripts/run_context.py',ROOT/'tests/conftest.py',ROOT/'tests/docker/test_marker_contract.py',*list((ROOT/'scripts/diagnostics/run-a001-fase0-uv').glob('*.py')),ROOT/'tests/unit/test_core_diagnostics.py']:
        ast.parse(read(p).decode())
    return dict(Docker_context_pairs=len(eq['Docker_r007_context_objects']),Docker_driver_objects=len(eq['C_docker_driver_probe_and_test_exact']),
                B56_exact=True,build_fields_exact=True,inventory_records_exact=len(inv['files']),inventory_delta_only_launcher=True,
                supervisor_only_documentation_changed=changed,local_links=links,verify_options=sorted(set(options)),
                whitespace=command(['git','diff','--check']),old_R_historical=True,old_fast_not_transferred_to_new_tests=True)

def accounting():
    e=load(FIX/'entry.json');close=load(FIX/'closing-accounting.json')
    formula=e['historical_conservative_entry']+e['initial_reading_charge_seconds']+close['monotonic']-e['monotonic_start']
    assert abs(formula-close['observed_after_all_report_checkpoint_manifest_rewrites'])<0.001
    assert close['prospective_charged_upper']==close['observed_after_all_report_checkpoint_manifest_rewrites']+1
    assert close['overrun']==close['prospective_charged_upper']-8880
    assert not close['historical_budget_PASS'] and not close['historical_upper_known']
    assert close['status']=='NON_PASS_PROSPECTIVE_TIME'
    hist=load(RUN/'evidence/supervisor-arbitration-implementation-r002/corrected-time-ledger.json')
    failures=load(FIX/'finalize-failure.json');assert failures['exit']==1 and not failures['workload_started']
    for name in ['execute.py','static_and_reuse.py','finalize.py','close_cost.py']:
        ast.parse(read(FIX/name).decode())
    mismatches={}
    for name in ['manifest.json','manifest-closing.json']:
        rows=load(FIX/name);bad=[]
        for row in rows:
            p=Path(row['path']);b=read(p)
            if len(b)!=row['bytes'] or hashlib.sha256(b).hexdigest()!=row['sha256']:
                bad.append(row['path'])
        mismatches[name]=bad
    assert mismatches['manifest.json']==[str(FIX/'delivery.json')]
    assert not mismatches['manifest-closing.json']
    timeline=[]
    for p in [*FIX.rglob('*'),RUN/'implementation/report-r020.md',RECEPTION/'implementation-checkpoint-received-r020.md']:
        if not p.is_file() or p.is_symlink():continue
        # Receipt del checkpoint ricevuto è copia del supervisore: esclusa dal tempo autore.
        if p==RECEPTION/'implementation-checkpoint-received-r020.md':continue
        s=p.stat();charge=8440+s.st_mtime-e['wall_start']
        if p.name in ['result.json','static.json','reuse.json','finalize.py','close_cost.py','delivery.json','manifest.json','manifest-closing.json','closing-accounting.json','report-r020.md','execution-result.json'] or charge>8880:
            timeline.append(dict(path=str(p.relative_to(ROOT)),mtime=s.st_mtime,charge_from_mtime=charge,seconds_after_observed=s.st_mtime-close['wall']))
    later=[v for v in timeline if v['seconds_after_observed']>1]
    assert not later, later
    return dict(formula_reconstructed=formula,closing=close,historical_ledger=hist,finalize_failure=failures,
                manifest_mismatches=mismatches,mtime_timeline=sorted(timeline,key=lambda v:v['mtime']),unaccounted_writes_later_than_margin=later,
                max_real_R_charge=max(load(FIX/('R-'+run_id)/'result.json')['charged_at'] for run_id in [RUN.name,'run-b002-fase0-uv']))

for name,fn in [('identity',identity),('real_R',real_R),('reuse_docs',reuse_docs),('accounting',accounting)]:
    try:
        t=time.monotonic();value=fn();results[name]=dict(status='PASS_READER_ONLY',seconds=time.monotonic()-t,result=value)
    except Exception:
        results[name]=dict(status='READER_ERROR',error=traceback.format_exc())
    write('audit.json',results)
write('read-inventory.json',reads)
print(json.dumps({k:{kk:vv for kk,vv in v.items() if kk!='result'} for k,v in results.items()},ensure_ascii=False))
