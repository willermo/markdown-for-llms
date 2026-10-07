import ast,hashlib,json,re,subprocess,urllib.parse,time
from pathlib import Path
R=Path('/home/davide/workarea/markdown-for-llms');RUN=R/'temp/run-a001-fase0-uv';H=Path(__file__).parent;OLD=H.parent/'fix-review-r002';PY=R/'.venv-python/cpython-3.12.13-linux-x86_64-gnu/bin/python3.12'
def identity(p):return dict(path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def load(p):return json.loads(p.read_text())
commands=[]
def call(argv):
 r=subprocess.run([str(x) for x in argv],cwd=R,capture_output=True,text=True,close_fds=True,timeout=20);commands.append(dict(argv=[str(x) for x in argv],cwd=str(R),exit=r.returncode,stdout=r.stdout,stderr=r.stderr));assert r.returncode==0,r.stdout+r.stderr;return r
help=call([PY,'-I','-B',R/'scripts/diagnostics/run-a001-fase0-uv/verify_distribution.py','--help']).stdout;assert '--standalone' not in help
reference=(R/'docs/reference/toolchain-legacy.md').read_text();uv=(R/'docs/how-to/ambiente-uv.md').read_text();docker=(R/'docs/how-to/marker-legacy-docker.md').read_text();assert 'verify_distribution.py --standalone' not in uv
assert all('RUN_TEST_MODE=official|standalone' in s and 'solo flag' in s for s in [uv,reference,docker]);assert 'RUN_IMAGE_CONTEXT_RECEIPT' in reference
actual=ast.parse((R/'tests/docker/test_marker_contract.py').read_text());required={'RUN_IMAGE_MANIFEST','RUN_IMAGE_INPUTS','RUN_BACKEND_WHEEL','RUN_IMAGE_CONTEXT_RECEIPT','RUN_TEST_OUTPUT','RUN_IMAGE','RUN_TEST_MODE'};assert required<={n.value for n in ast.walk(actual) if isinstance(n,ast.Constant) and isinstance(n.value,str)}
links=[]
for p in [R/'docs/how-to/ambiente-uv.md',R/'docs/reference/toolchain-legacy.md',R/'docs/how-to/marker-legacy-docker.md']:
 for number,line in enumerate(p.read_text().splitlines(),1):
  assert line.rstrip()==line
  for target in re.findall(r'(?<!!)\[[^\]]*\]\(([^)]+)\)',line):
   if target.startswith(('http:','https:','mailto:','#')):continue
   exists=(p.parent/urllib.parse.unquote(target.split('#',1)[0])).exists();links.append(dict(file=str(p.relative_to(R)),line=number,target=target,exists=exists));assert exists
call(['git','diff','--check']);ast.parse((R/'scripts/diagnostics/run-a001-fase0-uv/run_offline.py').read_text());ast.parse((R/'tests/unit/test_core_diagnostics.py').read_text())
# Independent exact consumed identities: never relabel historical R as current.
eq=load(OLD/'equivalence-final.json');reused=[]
for pair in eq['Docker_r007_context_objects']:
 for row in pair.values():assert identity(Path(row['path']))==row
 reused.append(pair)
for row in eq['C_docker_driver_probe_and_test_exact']:
 p=R/row['path'];assert hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256']
B=load(OLD/'actual-B-artifacts-s056.json')
for row in B.values():assert identity(Path(row['path']))==row
S=load(H/'R-run-a001-fase0-uv/S.json')['payload'];before=load(OLD/'S-s060.json')['payload'];assert all(S[k]==before[k] for k in ['modules','build_inputs','backend','toolchain'])
notes=dict(status='PASS_EXACT_REUSE_ONLY',B56=B,S_current=identity(H/'R-run-a001-fase0-uv/S.json'),build_fields_exact=['modules','build_inputs','backend','toolchain'],Docker_r007_objects=reused,C_docker_exact=eq['C_docker_driver_probe_and_test_exact'],historical_host_V1_CLI_fidelity='retain their stages/receipts; product, installed payload, runtime, cache/fixtures and build inputs unchanged; launchers of old R remain historical; new R verifies corrected launcher only',invalidated='run_offline.py snapshot entry branches and their cleanup regression; new targeted mocks and two actual R replace current launcher assertions; docs are not consumed by B56 or Docker',not_invalidated='No README/pyproject/lock/ten modules/Compose/Docker test or driver changed; no build/install/suite/Docker repetition')
(H/'reuse.json').write_text(json.dumps(notes,separators=(',',':')));(H/'static.json').write_text(json.dumps(dict(status='PASS_STATIC_ONLY',commands=commands,links=links,verify_distribution_scope='derived from S; no standalone flag',docker_required_environment=sorted(required),mode='RUN_TEST_MODE required to select Docker; pytest flag alone unsupported',AST=True),separators=(',',':')));print('PASS_STATIC_AND_EXACT_REUSE')
