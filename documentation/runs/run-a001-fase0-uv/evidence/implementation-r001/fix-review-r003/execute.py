import ast,copy,hashlib,importlib.util,json,subprocess,time,sys,shutil
from pathlib import Path
R=Path('/home/davide/workarea/markdown-for-llms');RUN=R/'temp/run-a001-fase0-uv';H=Path(__file__).parent;OLD=H.parent/'fix-review-r002';PY=R/'.venv-python/cpython-3.12.13-linux-x86_64-gnu/bin/python3.12';EXT=Path('/tmp/run-a001-product-wheel-env-r001/bin/python');DIAG=R/'scripts/diagnostics/run-a001-fase0-uv'
def load(p):return json.loads(p.read_text())
def module(p):
 s=importlib.util.spec_from_file_location('current',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
m=module(OLD/'run_core_proof_r002.py');d=m.d;resource=module(OLD/'resource_scope.py');resource.CORE=load(RUN/'evidence/supervisor-arbitration-implementation-r002/authorized-scope.json')['core'];resource.apply(d);entry=load(H/'entry.json');env=dict(d.ENV);calls=[]
def costs():return entry['historical_conservative_entry']+entry['initial_reading_charge_seconds']+time.monotonic()-entry['monotonic_start']
def write(p,v):p.write_text(json.dumps(v,separators=(',',':')))
def command(argv,cwd=R,expected=0,environment=env):
 assert costs()<8820,'Reserve60s';argv=[str(x) for x in argv];start=time.monotonic();p=subprocess.run(argv,cwd=cwd,env=environment,shell=False,close_fds=True,capture_output=True,text=True,timeout=min(180,8820-costs()));row=dict(argv=argv,cwd=str(cwd),environment=environment,exit=p.returncode,stdout=p.stdout,stderr=p.stderr,seconds=time.monotonic()-start,shell=False,close_fds=True);calls.append(row);write(H/'calls.json',calls);assert p.returncode==expected,p.stdout+p.stderr;return p
# Admission before tests; scope exact roots/excludes, no fresh pool.
ledger=d.ledger();assert d.gate(ledger)['admitted'];write(H/'admission.json',dict(ledger=ledger,gate=d.gate(ledger),cost=costs()))
# Pure mock branches; no collection/application imports.
(H/'mock-tmp').mkdir();mockenv=dict(env,TMPDIR=str(H/'mock-tmp'))
command([PY,'-I','-B','-m','unittest','discover','-s',R/'tests/unit','-p','test_core_diagnostics.py','-v'],environment=mockenv)
refs=load(OLD/'actual-B-artifacts-s056.json')
for v in refs.values():assert d.identity(Path(v['path']),True)==v
base=load(OLD/'S-s060.json')['payload']
for v in base['modules']+base['build_inputs']:
 p=R/v['path'];assert (not p.exists()) if v['kind']=='absent' else hashlib.sha256(p.read_bytes()).hexdigest()==v['sha256']
inv=load(OLD/'stage-s060/package-inputs.json');changes=[]
for v in inv['files']:
 p=Path(v['path']);sha=hashlib.sha256(p.read_bytes()).hexdigest()
 if sha!=v['sha256']:
  assert p==DIAG/'run_offline.py',str(p);changes.append(dict(before=copy.deepcopy(v),after=d.identity(p,True)));v.update(bytes=p.stat().st_size,sha256=sha)
write(H/'inventory-a001.json',inv);write(H/'inventory-delta.json',changes)
# Small public synthetic clone, not an environment copy. Git writes only here.
clone=H/'clone-b002';clone.mkdir();names={v['path'] for v in base['modules']+base['build_inputs'] if v['kind']=='file'};names.update(str(p.relative_to(R)) for p in DIAG.glob('*.py'));names.update(['scripts/run_context.py','.gitignore'])
for name in sorted(names):p=clone/name;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(R/name,p)
command(['git','init','--initial-branch','synthetic-r021'],clone);(clone/'.git/objects/info/alternates').write_text(str(R/'.git/objects')+'\n')
for name in ['refs/heads/synthetic-r021','refs/heads/dev']:command(['git','update-ref',name,'66ba82200e5def5a4db76f9bafccb0731b506091'],clone)
bi=copy.deepcopy(inv)
for v in bi['files']:
 p=Path(v['path'])
 if p.is_relative_to(R) and (clone/p.relative_to(R)).is_file():v['path']=str(clone/p.relative_to(R))
local=clone/'temp/run-b002-fase0-uv';local.mkdir(parents=True);write(local/'inputs.json',bi)
# Technical snapshots: only present immutable inputs, no S/I/output/checkpoint.
for repo,id,label,artifact in [(R,'run-a001-fase0-uv','impl-r001-stage-launcher-s076',H/'inventory-a001.json'),(clone,'run-b002-fase0-uv','launcher-r021-s001',local/'inputs.json')]:
 command([PY,'-I','-B',repo/'scripts/run_context.py','snapshot',id,'--label',label,'--artifact',artifact.relative_to(repo)],repo);command([PY,'-I','-B',repo/'scripts/run_context.py','verify',id,'--label',label],repo)

def run_R(repo,id,label,inventory):
 out=H/('R-'+id);out.mkdir();snap=repo/'temp'/id/'snapshots'/(label+'.json');S=out/'S.json';I=out/'I.json'
 # Origin/config/startup BEFORE target Python; native guard, exact B56 payload.
 d.guard.inspect_prefix(EXT,R/'.venv-python');d.guard.check_configuration(R,env)
 command([EXT,'-I','-B',repo/'scripts/diagnostics/run-a001-fase0-uv/make_source_manifest.py','--repo',repo,'--snapshot',snap,'--input-inventory',inventory,'--uv','/home/davide/.local/bin/uv','--output',S],repo)
 command([EXT,'-I','-B',repo/'scripts/diagnostics/run-a001-fase0-uv/verify_distribution.py','--repo',repo,'--source-manifest',S,'--sdist',refs['sdist']['path'],'--wheel',refs['wheel']['path'],'--profile','base','--expected-managed',R/'.venv-python','--receipt',I],repo)
 op=dict(id=id,R_python=str(EXT),environment=env,cwd=str(repo),backend_preimport=False);target=d.target_for(op,out);target['repo']=str(repo);target['workspace']=str(repo)
 target['files'].extend([d.identity(S,True),d.identity(I,True),d.identity(DIAG/'run_offline.py',True),d.identity(Path(__file__),True)]);target['files'].extend(load(I)['payload']['installed']['files']);write(out/'effective-target.json',target)
 preflight=[str(EXT),'-I','-B',str(repo/'scripts/diagnostics/run-a001-fase0-uv/check_preflight.py'),'--repo',str(repo),'--source',str(S),'--receipt',str(I)];write(out/'preflight.json',dict(schema=1,commands=[preflight]))
 seconds=min(60,180,8820-costs());assert seconds>5
 argv=[str(PY),'-I','-B',str(DIAG/'run_offline.py'),'--runner','firejail','--python',str(EXT),'--repo',str(repo),'--target',str(out/'effective-target.json'),'--snapshot',str(snap),'--output',str(out/'wrapper'),'--cwd',str(repo),'--preflight-config',str(out/'preflight.json'),'--timeout',str(seconds),'--',str(EXT),'-I','-B','-c',"import sys,time;assert sys.flags.isolated and sys.dont_write_bytecode;print('SYNTHETIC_OFFLINE_ONLY');time.sleep(.3)"]
 op.update(argv=argv,environment=env,shell=False,close_fds=True,child_seconds=seconds,outer_seconds=seconds+180,monitor_seconds=min(seconds+180,8820-costs()));write(out/'invocation.json',op);d.t.OUT=H;receipt=d.monitored(op,'monitor-'+id);survivors=m.b.survivors(receipt['observed_processes']);live=[]
 for v in survivors:
  try:
   if (Path('/proc')/str(v['pid'])/'stat').read_text().rsplit(')',1)[1].split()[0]!='Z':live.append(v)
  except FileNotFoundError:pass
 write(out/'process-collection.json',dict(survivors=live,zombies_are_not_active=True));assert not live
 outer=load(out/'wrapper/receipt.json');inside=load(out/'wrapper/inside.json');assert receipt['exit']==0 and receipt['stop_reason'] is None and outer['status']=='PASS' and inside['status']=='PASS';assert outer['temporary_socket_cleaned']
 command(preflight,repo);command([EXT,'-I','-B',repo/'scripts/run_context.py','verify',id,'--label',label],repo);write(out/'result.json',dict(status='PASS_REAL_R_CURRENT_TARGET_ONLY',monitor=receipt,status_R=outer['status'],status_inside=inside['status'],charged_at=costs(),S=d.identity(S,True),I=d.identity(I,True),snapshot=d.identity(snap,True)))
run_R(R,'run-a001-fase0-uv','impl-r001-stage-launcher-s076',H/'inventory-a001.json')
run_R(clone,'run-b002-fase0-uv','launcher-r021-s001',local/'inputs.json')
write(H/'execution-result.json',dict(status='PASS_REAL_R_BOTH_IDS',cost_at=costs(),B56=refs));print('PASS_REAL_R_BOTH_IDS',costs(),flush=True)
