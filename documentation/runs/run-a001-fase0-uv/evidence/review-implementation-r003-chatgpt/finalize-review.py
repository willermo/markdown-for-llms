"""Chiusura propria: identità, link e spazio, senza prodotto o scritture comuni."""
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
import time
import urllib.parse

ROOT=Path('/home/davide/workarea/markdown-for-llms')
RUN=ROOT/'temp/run-a001-fase0-uv'
HERE=Path(__file__).parent
REPORT=RUN/'reviews/review-implementation-r003-chatgpt.md'
CHECKPOINT=RUN/'handovers/review-implementation-r003-chatgpt.md'
OUT=HERE/'final-checks.json'
SCOPE=json.loads((RUN/'evidence/supervisor-arbitration-implementation-r002/authorized-scope.json').read_text())['core']

def identity(p):
    b=p.read_bytes()
    return dict(path=str(p.relative_to(ROOT)),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())

def ledger():
    logical=allocated=0
    excludes={str(ROOT/p) for p in SCOPE['exclude_new_docker_subtrees']}
    for root in SCOPE['roots']:
        p=Path(root);pending=[str(p if p.is_absolute() else ROOT/p)]
        while pending:
            p=pending.pop()
            if p in excludes:continue
            try:s=os.lstat(p)
            except FileNotFoundError:continue
            logical+=s.st_size;allocated+=s.st_blocks*512
            if stat.S_ISDIR(s.st_mode):
                with os.scandir(p) as es:pending.extend(e.path for e in es)
    H=max(logical,allocated);delta=max(0,H-SCOPE['Hentry']);remaining=SCOPE['incremental_bytes']-delta
    reserve=H+max(0,remaining)+SCOPE['external_reserve_bytes']
    repo_free=shutil.disk_usage(ROOT).free;tmp_free=shutil.disk_usage('/tmp').free
    return dict(logical=logical,allocated=allocated,H=H,Hentry=SCOPE['Hentry'],delta=delta,remaining=remaining,
                with_reserves=reserve,repo_free=repo_free,tmp_free=tmp_free,roots=SCOPE['roots'],exclusions=sorted(excludes),
                admitted=(delta<SCOPE['incremental_bytes'] and H<SCOPE['max_pool_bytes'] and reserve<SCOPE['stop_bytes']
                          and repo_free>=SCOPE['repository_free_required_bytes'] and tmp_free>=SCOPE['external_tmp_free_required_bytes']))

t=time.monotonic();commands=[]
for argv in [[sys.executable,'-I','-B',str(ROOT/'scripts/run_context.py'),'verify',RUN.name,'--label','impl-r001-stage-final-s003'],
             ['git','diff','--check'],['git','diff','--cached','--name-only']]:
    p=subprocess.run(argv,cwd=ROOT,capture_output=True,text=True,close_fds=True,timeout=15,
                     env={'PATH':'/usr/bin:/bin','LANG':'C.UTF-8','PYTHONDONTWRITEBYTECODE':'1','PYTHONNOUSERSITE':'1'})
    row=dict(argv=argv,cwd=str(ROOT),exit=p.returncode,stdout=p.stdout,stderr=p.stderr)
    commands.append(row);assert p.returncode==0,row
assert not commands[-1]['stdout']
expected=json.loads((RUN/'evidence/supervisor-implementation-r003/reception-r001/freeze-identity.json').read_text())
snap=ROOT/expected['path'];actual=identity(snap)
assert actual['bytes']==expected['bytes'] and actual['sha256']==expected['sha256']
data=json.loads(snap.read_text());assert data['worktree_sha256']==expected['worktree_sha256']
assert all(v['status']=='PASS_READER_ONLY' for v in json.loads((HERE/'audit.json').read_text()).values())
probe=json.loads((HERE/'probe.json').read_text());assert probe['exit']==0 and probe['wall_seconds']<30
OUT.write_text('{}\n')
links=[]
for p in [REPORT,CHECKPOINT]:
    for no,line in enumerate(p.read_text().splitlines(),1):
        assert line.rstrip()==line,(p,no)
        for target in re.findall(r'(?<!!)\[[^\]]*\]\(([^)]+)\)',line):
            if target.startswith(('http:','https:','mailto:','#')):continue
            path=p.parent/urllib.parse.unquote(target.split('#',1)[0])
            assert path.exists(),(p,no,target)
            links.append(dict(path=str(p.relative_to(ROOT)),line=no,target=target,exists=True))
result=dict(status='PASS_REVIEW_FINAL_CHECKS_ONLY',review_verdict='GO',identity=expected,commands=commands,local_links=links,
            probe_count=1,probe_child_wall_seconds=probe['wall_seconds'],probe_charged_upper_with_launcher_seconds=2,
            probe_cap_seconds=30,real_R_started_by_reviewer=0,application_suites_by_reviewer=0,
            product_install_build_Docker_network_ML=0,competing_r003_review_read=False,
            shared_or_product_writes=False,own_processes_complete=True,
            accounting_status='HISTORICAL_AND_PROSPECTIVE_TIME_NON_PASS_PRESERVED',
            final_reader_seconds=time.monotonic()-t,
            execution=dict(argv=[sys.executable,'-I','-B',str(Path(__file__))],cwd=str(ROOT),outer_timeout_seconds=25))
for _ in range(2):
    files=[p for p in HERE.rglob('*') if p.is_file() and not p.is_symlink()]+[REPORT,CHECKPOINT]
    total=sum(p.stat().st_size for p in files)
    result['own_outputs_bytes_before_final_receipt_rewrite']=total
    result['own_outputs_charged_upper_with_receipt_margin']=total+8192
    result['own_output_limit_bytes']=1048576
    result['own_file_identities_excluding_this_receipt']=[identity(p) for p in files if p!=OUT]
    result['ledger']=ledger();assert result['ledger']['admitted']
    assert result['ledger']['remaining']>8192 and total+8192<1048576
    OUT.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
print(json.dumps(dict(status=result['status'],verdict='GO',MATCH=True,own_output_upper=result['own_outputs_charged_upper_with_receipt_margin'],ledger=result['ledger']),ensure_ascii=False))
