"""Identità finale e limiti degli output reviewer; nessun workload prodotto."""
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import urllib.parse

R=Path('/home/davide/workarea/markdown-for-llms')
RUN=R/'temp/run-a001-fase0-uv'
OUT=Path(__file__).parent
PY=R/'.venv-python/cpython-3.12.13-linux-x86_64-gnu/bin/python3.12'
snapshot=RUN/'snapshots/impl-r001-stage-final-s002.json'
argv=[str(PY),'-I','-B',str(R/'scripts/run_context.py'),'verify','run-a001-fase0-uv','--label','impl-r001-stage-final-s002']
p=subprocess.run(argv,cwd=R,shell=False,close_fds=True,capture_output=True,text=True,timeout=60)
assert p.returncode==0 and p.stdout.startswith('MATCH:'),p.stdout+p.stderr
raw=snapshot.read_bytes();snap=json.loads(raw)
assert hashlib.sha256(raw).hexdigest()=='e0cb33d16a5766f653923c2d25cf365f910d7e738d542893db2a1ac589272c1d'
files=[RUN/'reviews/review-implementation-r002-chatgpt.md',RUN/'handovers/review-implementation-r002-chatgpt.md']
links=[]
for f in files:
    text=f.read_text()
    assert all(line.rstrip()==line for line in text.splitlines())
    for target in re.findall(r'(?<!!)\[[^\]]*\]\(([^)]+)\)',text):
        if target.startswith(('http:','https:','mailto:','#')):continue
        path=(f.parent/urllib.parse.unquote(target.split('#')[0])).resolve();links.append(dict(file=str(f),target=target,exists=path.exists() or path==(OUT/'final-checks.json').resolve()))
assert all(x['exists'] for x in links)
diff=subprocess.run(['git','-C',str(R),'diff','--check'],capture_output=True,text=True,timeout=10)
assert diff.returncode==0,diff.stdout+diff.stderr
scope=json.loads((RUN/'evidence/supervisor-implementation-r001/final-host-time-r001/authorized-scope.json').read_text())['core']
excluded={str(R/p) for p in scope['exclude_new_docker_subtrees']}
logical=allocated=0
for root in scope['roots']:
    pending=[R/root]
    while pending:
        f=pending.pop()
        if str(f) in excluded:continue
        try:info=f.lstat()
        except FileNotFoundError:continue
        logical+=info.st_size;allocated+=info.st_blocks*512
        if stat.S_ISDIR(info.st_mode):pending.extend(f.iterdir())
H=max(logical,allocated);delta=max(0,H-scope['Hentry'])
output_files=[x for x in OUT.iterdir() if x.is_file() and x.name!='final-checks.json']+files
logical_output=sum(x.stat().st_size for x in output_files)
allocated_output=sum(x.stat().st_blocks*512 for x in output_files)
probes=[json.loads((OUT/n).read_text()) for n in ['probe-run-identity-first.json','probe-run-identity.json']]
result=dict(snapshot=dict(path=str(snapshot),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),worktree_sha256=snap['worktree_sha256']),verify_after=dict(argv=argv,exit=p.returncode,stdout=p.stdout,stderr=p.stderr),git_diff_check_exit=diff.returncode,own_local_links=links,probe_internal_seconds_cumulative=sum(x['seconds'] for x in probes),probe_wall_seconds_cumulative_observed=0.478677209+0.295863294,probe_quota_seconds=30,probe_real_subprocesses=0,output_file_count=len(output_files)+1,output_logical_bytes=0,output_allocated_bytes_before_receipt=allocated_output,output_limit_bytes=2097152,core_ledger=dict(logical=logical,allocated=allocated,H=H,delta=delta,remaining_bytes=scope['incremental_bytes']-delta,total_with_reserve=H+max(0,scope['incremental_bytes']-delta)+scope['external_reserve_bytes'],measurement_before_this_small_receipt=True),product_workloads_executed=0,docker_commands_executed=0,network_commands_executed=0,other_r002_report_read=False)
for _ in range(4):
    result['output_logical_bytes']=logical_output+len((json.dumps(result,indent=2)+'\n').encode())
assert result['output_logical_bytes']<2097152 and allocated_output+8192<2097152
assert delta<scope['incremental_bytes'] and result['core_ledger']['total_with_reserve']<scope['stop_bytes']
(OUT/'final-checks.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(dict(MATCH=True,output_logical_bytes=result['output_logical_bytes'],core_H=H,remaining_bytes=result['core_ledger']['remaining_bytes'],probe_seconds=result['probe_internal_seconds_cumulative'])))
