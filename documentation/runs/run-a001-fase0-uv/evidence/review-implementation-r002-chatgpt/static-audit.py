"""Review in sola lettura: hash, AST, receipt, archivi e ledger. Nessun workload."""
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import tarfile
import time
import traceback
import urllib.parse
import zipfile

R = Path('/home/davide/workarea/markdown-for-llms')
RUN = R/'temp/run-a001-fase0-uv'
H = RUN/'evidence/implementation-r001/fix-review-r002'
OLD = H.parent/'fix-review-r001'
D = H.parent/'docker-cpu-r001/fix-review-r001'
OUT = Path(__file__).parent
reads = {}
checks = {}
start = time.monotonic()

def raw(path):
    path = Path(path)
    assert 'review-implementation-r002-claude' not in str(path)
    data = path.read_bytes()
    reads[str(path)] = dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),mode='content')
    return data

def load(path):
    return json.loads(raw(path))

def ident(path):
    path = Path(path)
    h = hashlib.sha256()
    with path.open('rb') as f:
        for data in iter(lambda:f.read(1048576), b''):h.update(data)
    d = dict(bytes=path.stat().st_size,sha256=h.hexdigest())
    reads.setdefault(str(path),dict(**d,mode='hash'))
    return d

def match(row):
    actual = ident(row['path'])
    return all(actual[k] == row[k] for k in ('bytes','sha256') if k in row)

def envelope(path):
    d = load(path)
    canonical = json.dumps(d['payload'],sort_keys=True,ensure_ascii=True,separators=(',',':')).encode()
    assert d['schema']==1 and d['id']=='sha256:'+hashlib.sha256(canonical).hexdigest()
    return d

def section(name, fn):
    try:checks[name] = fn()
    except Exception as e:checks[name] = dict(audit_error=repr(e),traceback=traceback.format_exc())

def identity():
    freeze = load(RUN/'evidence/supervisor-implementation-r002/reception-r001/freeze-identity.json')
    assert match(freeze['snapshot'])
    snap = load(freeze['snapshot']['path'])
    assert snap['worktree_sha256']==freeze['worktree_sha256']
    for row in snap['files']:
        if row['kind']=='file':raw(R/row['path'])
    git = {}
    for n,argv in [('head',['rev-parse','HEAD']),('dev',['rev-parse','dev']),('branch',['branch','--show-current']),('index',['diff','--cached','--name-only']),('whitespace',['diff','--check'])]:
        p = subprocess.run(['git','-C',str(R),*argv],capture_output=True,text=True,timeout=10)
        git[n] = dict(exit=p.returncode,stdout=p.stdout,stderr=p.stderr)
    return dict(freeze=freeze,git=git,delta=load(RUN/'evidence/supervisor-implementation-r002/reception-r001/delta-s001-to-received.json'))

def host():
    S = envelope(H/'S-s060.json')
    previous = envelope(OLD/'S-s056.json')
    inputs_equal = {k:previous['payload'][k]==S['payload'][k] for k in ['modules','build_inputs','backend','toolchain']}
    assert all(inputs_equal.values())
    inv = load(H/'stage-s060/package-inputs.json')
    assert all(match(x) for x in inv['files'])
    profiles = {}
    for n in ['root','base-a','base-b','external','dev','api']:
        I = envelope(H/f'I-{n}-s060.json')
        execution = load(H/f'proof-I-{n}-r001/workload.json')
        assert I['payload']['source']['id']==S['id'] and I['payload']['status']=='PASS'
        assert execution['status']=='PASS' and len(execution['calls'])==2 and all(x['exit']==0 for x in execution['calls'])
        profiles[n] = dict(I_id=I['id'],execution_preflight=execution['calls'],profile=I['payload']['profile'],interpreter=I['payload']['interpreter'],archive_hashes=[x['sha256'] for x in I['payload']['archives']])
    tests = {}
    for p in sorted(H.glob('proof-*/E-pytest.json')):
        q = envelope(p)['payload'];z = q['result']
        tests[p.parent.name] = dict(status=q['status'],collected=len(z['collected']),skipped=len(z['skipped']),collection_errors=z['collection_errors'],exit=z['exit_code'],postcheck=z['inputs_unchanged'],isolated_postcheck=z['isolated_postcheck'],outcomes={v:sum(x['outcome']==v for x in z['reports']) for v in ['passed','failed','skipped']})
    proofs = []
    for p in sorted(H.glob('proof-*/result.json')):
        q = load(p);wrap = load(p.parent/'wrapper/receipt.json');inside = load(p.parent/'wrapper/inside.json');runner = load(p.parent/'wrapper/runner.json');collection = load(p.parent/'process-collection.json')
        proofs.append(dict(name=p.parent.name,status=q['status'],spent_before=q['spent_before'],cumulative_after=q['cumulative_after'],wrapper_status=wrap['status'],wrapper_final_exact=wrap['wrapper_sha256']==ident(R/'scripts/diagnostics/run-a001-fase0-uv/run_offline.py')['sha256'],socket_cleaned=wrap['temporary_socket_cleaned'],runner_status=runner['status'],inside_status=inside['status'],survivors=collection['survivors']))
    standalone = load(H/'proof-standalone-runid-r003/workload.json')
    v1 = load(H/'proof-V1-r003/workload.json')
    return dict(S_id=S['id'],historical_S60_not_revalidated_after_supervisor_docs=True,B56_inputs_equal=inputs_equal,input_hashes_checked=len(inv['files']),profiles=profiles,tests=tests,proofs=proofs,standalone=dict(status=standalone['status'],observations=standalone['observations'],calls=[dict(argv=x['argv'],exit=x['exit'],stderr=x['stderr']) for x in standalone['calls']]),V1=dict(status=v1['status'],calls=len(v1['calls']),phases=[dict(name=x['name'],status=x['status'],changed_modules=x['changed_modules'],immediate_hashes=x['immediately_after_native_sync_before_canonical'],calls=x['calls']) for x in v1['phases']],clone_unchanged=v1['original_clone_unchanged']))

def reuse():
    q = load(H/'equivalence-final.json')
    assert all(match(x['context']) and match(x['current']) and x['context']['sha256']==x['current']['sha256'] for x in q['Docker_r007_context_objects'])
    assert all(match(dict(x,path=str(R/x['path']))) for x in q['C_docker_driver_probe_and_test_exact'])
    refs = load(H/'actual-B-artifacts-s056.json');assert all(match(x) for x in refs.values())
    with zipfile.ZipFile(refs['wheel']['path']) as wheel:
        modules = [n for n in wheel.namelist() if n.endswith('.py')]
        assert len(modules)==10 and all(wheel.read(n)==raw(R/n) for n in modules)
        metadata = wheel.read('markdown_for_llms-1.0.0.dist-info/METADATA')
        assert raw(R/'README.md') in metadata
    with tarfile.open(refs['sdist']['path']) as archive:
        names=archive.getnames();readme=next(n for n in names if n.endswith('/README.md'))
        assert archive.extractfile(readme).read()==raw(R/'README.md')
    oldeq = load(q['previous_equivalence'])
    b20 = oldeq['wheel_B20_to_B46']['old_archive'];assert match(b20)
    with zipfile.ZipFile(b20['path']) as original,zipfile.ZipFile(refs['wheel']['path']) as current:
        payloads = modules+['markdown_for_llms-1.0.0.dist-info/entry_points.txt']
        assert all(original.read(n)==current.read(n) for n in payloads)
        metadata_name='markdown_for_llms-1.0.0.dist-info/METADATA'
        assert original.read(metadata_name).split(b'\n\n',1)[0]==current.read(metadata_name).split(b'\n\n',1)[0]
    dependencies = {}
    for x in q['profiles']:
        old_I = envelope(x['old_I']['path'])['payload'];new_I=envelope(x['new_I'])['payload']
        dependencies[x['name']]=old_I['installed']['dependencies']==new_I['installed']['dependencies']
    assert all(dependencies.values())
    return dict(context_files_exact=len(q['Docker_r007_context_objects']),C_docker_objects_exact=len(q['C_docker_driver_probe_and_test_exact']),canonical=refs,modules_exact=modules,README_in_sdist_and_METADATA_exact=True,B20_B56_payload_entrypoints_headers_exact=True,dependencies_exact=dependencies,previous_equivalence=oldeq)

def docker():
    configs = [load(D/f'compose-r004/config-{i}.json') for i in range(3)]
    b0,b1 = [x['services']['marker-api']['build'] for x in configs[:2]]
    assert b0==b1 and b0['network']=='none'
    result = load(D/'compose-r004/result.json')
    native = [x for x in result['calls'] if 'history' in x['argv'] or 'inspect' in x['argv']]
    supply = load(RUN/'work/docker-cpu-r001/fix-supply-r004.receipt.json')
    assert all(match(x) for x in supply['wheels']['payloads']+supply['apt']['selected'])
    c = load(D/'C-docker-r002/contract/result.json')
    before = load(D/'C-docker-r002/contract/container-before.json')
    startup = load(D/'C-docker-r002/contract/filesystem-startup-before-python.json')
    assert len(c['contract']['subcases'])==9 and not c['contract']['network_attempts']
    hc=before['HostConfig']
    assert not before['Mounts'] and hc['NetworkMode']=='none' and hc['CapDrop']==['ALL'] and 'no-new-privileges' in hc['SecurityOpt'] and not hc['Privileged']
    return dict(CPU_build_equal=b0==b1,build=b0,GPU_extra=configs[2]['services']['marker-api']['build']['args']['MARKER_EXTRA'],image=result['image'],history=[x for x in result['history'] if x['name'].endswith('fix-context-r004')],native_completion_historical=result['native_completion'],native_calls=native,supply_status=supply['status'],supply_network=supply['network_accounted_upper'],wheels_checked=len(supply['wheels']['payloads']),debs_checked=len(supply['apt']['selected']),APT_bindings=supply['apt']['bindings'],C_docker=dict(status=c['status'],subcases=c['contract']['subcases'],collected=c['owned_container_collected'],startup=startup,hostconfig=hc),ledger=load(D/'final-docker-ledger.json'))

def docs():
    rows = load(OLD/'readme-reconciliation-verified.json')
    spec = importlib.util.spec_from_file_location('review_doc_scanner',R/'scripts/diagnostics/run-a001-fase0-uv/inventory_docs.py');scanner=importlib.util.module_from_spec(spec);spec.loader.exec_module(scanner)
    raw(rows['original']['path']);assert match(rows['original']);old=scanner.scan(Path(rows['original']['path']))
    blocks={b['first_line']:b for b in old['blocks']}
    detail=[]
    for x in rows['blocks']:
        b=blocks[x['first_line']]
        assert x['last_line']==b['last_line'] and x['text_sha256']==hashlib.sha256(b['text'].strip().encode()).hexdigest()
        assert x['reason'] and x['disposition'] in ['modificato','rimosso','spostato','mantenuto']
        if x['disposition']!='rimosso':assert (R/x['destination']).is_file()
        detail.append(dict(id=x['id'],first_line=x['first_line'],planning_text_sha256=x['planning_text_sha256'],actual_text_sha256=x['text_sha256'],reason=x['reason'],disposition=x['disposition'],destination=x.get('destination')))
    for x in rows['inline']:
        assert any(x['text']==y['text'] and x['line']==y['line'] and y['command'] for y in old['inline'])
        assert x['text_sha256']==hashlib.sha256(x['text'].encode()).hexdigest() and x['reason']
        if x.get('destination'):assert (R/x['destination']).is_file()
    assert len(detail)==74 and len(rows['inline'])==9
    links=[]
    for p in [R/'README.md',*sorted((R/'docs').rglob('*.md'))]:
        for line,txt in enumerate(raw(p).decode().splitlines(),1):
            for target in re.findall(r'(?<!!)\[[^\]]*\]\(([^)]+)\)',txt):
                if target.startswith(('http:','https:','mailto:','#')):continue
                links.append(dict(file=str(p.relative_to(R)),line=line,target=target,exists=(p.parent/urllib.parse.unquote(target.split('#')[0])).exists()))
    options=[n.args[0].value for n in ast.walk(ast.parse(raw(R/'scripts/diagnostics/run-a001-fase0-uv/verify_distribution.py'))) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='add_argument' and n.args and isinstance(n.args[0],ast.Constant)]
    return dict(blocks=detail,inline=rows['inline'],links=links,missing_links=[x for x in links if not x['exists']],verify_distribution_literal_options=options,standalone_option_accepted='--standalone' in options)

def costs():
    scope=load(RUN/'evidence/supervisor-implementation-r001/final-host-time-r001/authorized-scope.json')['core']
    excluded={str(R/p) for p in scope['exclude_new_docker_subtrees']};logical=allocated=0
    for root in scope['roots']:
        pending=[R/root]
        while pending:
            p=pending.pop()
            if str(p) in excluded:continue
            try:s=p.lstat()
            except FileNotFoundError:continue
            logical+=s.st_size;allocated+=s.st_blocks*512
            if stat.S_ISDIR(s.st_mode):pending.extend(p.iterdir())
    Hnow=max(logical,allocated);delta=max(0,Hnow-scope['Hentry'])
    final=load(H/'final-checks.json');entry=load(H/'entry.json');auth=load(H/'resource-authorization-r002.json')
    latest=max(load(p)['cumulative_after'] for p in H.glob('proof-*/result.json'))
    return dict(entry=entry,authorization=auth,latest_workload_cumulative=latest,final_cumulative=final['core_cumulative_seconds'],overrun=final['core_cumulative_seconds']-8280,time_quota_PASS=False,accounting_formula='7194.886280257022 + 45 + monotonic-now - 1287939.063062153 - 27.52434803196229',logical_now=logical,allocated_now=allocated,H_now=Hnow,delta_now=delta,remaining_bytes=scope['incremental_bytes']-delta,reserve_included_total=Hnow+max(0,scope['incremental_bytes']-delta)+scope['external_reserve_bytes'],storage_admitted=delta<scope['incremental_bytes'] and Hnow+max(0,scope['incremental_bytes']-delta)+scope['external_reserve_bytes']<scope['stop_bytes'])

prior=OUT/'static-audit.json'
if prior.exists() and not (OUT/'static-audit-attempt1.json').exists():
    (OUT/'static-audit-attempt1.json').write_bytes(prior.read_bytes())
elif prior.exists() and not (OUT/'static-audit-attempt2.json').exists():
    (OUT/'static-audit-attempt2.json').write_bytes(prior.read_bytes())
for n,fn in [('identity',identity),('host',host),('reuse',reuse),('docker',docker),('docs',docs),('costs',costs)]:section(n,fn)
for relative in ['AGENTS.md','.agents/skills/manage-implementation-run/SKILL.md','documentation/development/run-lifecycle.md','documentation/development/templates/review.md','documentation/README.md','documentation/decisions/README.md','documentation/roadmap.md','documentation/decisions/0006-python-toolchain-uv.md','documentation/decisions/0007-supervised-development-runs.md']:
    raw(R/relative)
for relative in ['prompts/57-review-implementation-r002-chatgpt.md','prompts/review-implementation-r002-common.md','prompts/55-implementation-r001-fix-review-findings.md','prompts/56-implementation-r001-complete-host-proofs-with-time.md','plans/plan-r003.md','arbitrations/arbitration-plan-r003.md','arbitrations/arbitration-implementation-r001.md','arbitrations/addendum-operational-protocol-r019.md','arbitrations/addendum-operational-protocol-r020.md','implementation/report-r018.md','implementation/report-r019.md','reviews/review-implementation-r001-chatgpt.md','reviews/review-implementation-r001-claude.md']:
    raw(RUN/relative)
for p in sorted((R/'scripts/diagnostics/run-a001-fase0-uv').glob('*.py')):ast.parse(raw(p))
for p in sorted(H.glob('*.py')):ast.parse(raw(p))
with (OUT/'static-audit.json').open('w') as f:json.dump(dict(scope='READ_HASH_AST_ONLY_NOT_NEW_PRODUCT_PASS',seconds=time.monotonic()-start,checks=checks),f,indent=2)
with (OUT/'read-inventory.json').open('w') as f:json.dump(reads,f,indent=2)
print(json.dumps(dict(sections={n:('AUDIT_ERROR '+x['audit_error'] if 'audit_error' in x else 'READ_CHECK_COMPLETE') for n,x in checks.items()},files_read_or_hashed=len(reads),seconds=time.monotonic()-start)))
