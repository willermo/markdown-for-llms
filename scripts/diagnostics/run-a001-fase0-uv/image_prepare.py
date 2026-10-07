#!/usr/bin/env python3
"""Ammissione CPU offline: fonte EbookLib, backend84 e startup runtime prima di I."""
import argparse,socket,hashlib,importlib.metadata,json,os,sys,sysconfig,tarfile,subprocess,time,zipfile
from pathlib import Path

def descriptor(p):
 b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def observed_run(argv,cwd,output):
 owned={};start=time.monotonic()
 with (output/'stdout.txt').open('xb') as out,(output/'stderr.txt').open('xb') as err:
  p=subprocess.Popen(argv,cwd=cwd,shell=False,close_fds=True,stdout=out,stderr=err)
  while True:
   pending=[p.pid];seen=set()
   while pending:
    pid=pending.pop()
    if pid in seen:continue
    seen.add(pid)
    try:
     q=Path('/proc')/str(pid);row=dict(pid=pid,starttime=(q/'stat').read_text().rsplit(')',1)[1].split()[19],cmdline=(q/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace'),netns=os.readlink(q/'ns/net'))
     # Conserva transizioni exec e argv prima dello zombie; non sovrascriverli
     # con cmdline vuota alla fine del processo.
     owned[(pid,row['starttime'],row['cmdline'])]=row
     for task in (q/'task').iterdir():pending.extend(int(x) for x in (task/'children').read_text().split())
    except (FileNotFoundError,ProcessLookupError,PermissionError):pass
   if p.poll() is not None:break
   if time.monotonic()-start>1800: p.terminate();p.wait(timeout=5);raise TimeoutError('Deadline build interna')
   if out.tell()>1048576 or err.tell()>1048576:p.terminate();p.wait(timeout=5);raise ValueError('Log cap')
   time.sleep(.002)
 r=dict(argv=argv,cwd=str(cwd),exit=p.returncode,seconds=time.monotonic()-start,observed_processes=list(owned.values()))
 with (output/'command.json').open('x') as f:json.dump(r,f,indent=2)
 if p.returncode:raise ValueError('UV workload fallito: '+(output/'stderr.txt').read_text())
 return r

def backend_files(wheel,site):
 with zipfile.ZipFile(wheel) as z:
  rows=[]
  for n in z.namelist():
   if n.endswith('/') or n.endswith('/RECORD'):continue
   p=site/n
   if p.is_symlink() or p.read_bytes()!=z.read(n):raise ValueError('Backend installato diverso da wheel: '+n)
   rows.append(descriptor(p))
 return rows

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--repo',type=Path,required=True);p.add_argument('--supply',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(exist_ok=False)
 if not sys.flags.isolated or not sys.dont_write_bytecode or sys.version_info[:3]!=(3,12,13):raise ValueError('Interprete image inatteso')
 namespace=os.readlink('/proc/self/ns/net');host_namespace=os.environ.get('RUN_HOST_NETNS')
 if not host_namespace or namespace==host_namespace or os.listdir('/sys/class/net')!=['lo']:raise ValueError('Namespace build non isolato')
 daemon_paths=['/run/docker.sock','/var/run/docker.sock','/run/systemd/private','/run/dbus/system_bus_socket']
 daemon_paths += [str(p) for p in Path('/run/user').glob('*/systemd/private')]
 if any(Path(x).exists() for x in daemon_paths):raise ValueError('Socket host visibile')
 left,right=socket.socketpair();left.close();right.close()
 local=socket.socket(socket.AF_UNIX);local.bind(str(a.output/'synthetic.sock'));local.close();(a.output/'synthetic.sock').unlink()
 with (a.output/'R-D4.json').open('x') as f:json.dump(dict(status='PASS',netns=namespace,host_netns=host_namespace,interfaces=['lo'],daemon_paths_absent=daemon_paths,AF_UNIX_positive=True,python=sys.executable,version=sys.version),f,indent=2)
 site=Path(sysconfig.get_path('purelib'));wheel=a.supply/'setuptools-84.0.0-py3-none-any.whl'
 if descriptor(wheel)['sha256']!='51a52592b3b99e102b609654876bd65f19f999935166d1352678931132b0c670':raise ValueError('Backend non ricevuto')
 if sorted(x.name for x in site.glob('*.pth'))!=['distutils-precedence.pth'] or list(site.glob('*customize*')):raise ValueError('Startup base inatteso')
 files=backend_files(wheel,site)
 source=a.supply/'EbookLib-0.18.tar.gz'
 if descriptor(source)['sha256']!='38562643a7bc94d9bf56e9930b4927e4e93b5d1d0917f697a6454db5a1c1a533':raise ValueError('EbookLib sdist diversa da lock')
 extracted=a.output/'ebooklib-source';extracted.mkdir()
 with tarfile.open(source,'r:gz') as t:
  for m in t.getmembers():
   q=Path(m.name)
   if q.is_absolute() or '..' in q.parts or not (m.isfile() or m.isdir()):raise ValueError('Tar evasivo')
   target=extracted/q
   if m.isdir():target.mkdir(parents=True,exist_ok=True)
   else:
    target.parent.mkdir(parents=True,exist_ok=True)
    with target.open('xb') as out:out.write(t.extractfile(m).read())
 ebook=extracted/'EbookLib-0.18';os.chdir(ebook)
 import setuptools, setuptools.build_meta as backend
 if importlib.metadata.version('setuptools')!='84.0.0':raise ValueError('Backend diverso')
 requires={n:getattr(backend.__legacy__,n)({}) for n in ('get_requires_for_build_sdist','get_requires_for_build_wheel')}
 if any(requires.values()):raise ValueError('Nuovi requisiti EbookLib fuori dalla supply84: '+repr(requires))
 with (a.output/'ebooklib-prebuild.json').open('x') as f:json.dump(dict(source=descriptor(source),backend='setuptools.build_meta:__legacy__',version='84.0.0',backend_files=files,requirements=requires,setup=descriptor(ebook/'setup.py')),f,indent=2)
 os.chdir(a.repo)
 binding=json.loads((a.repo/'image-source.json').read_text())
 for item in binding['modules']+binding['build_inputs']:
  q=a.repo/item['name'];got=descriptor(q)
  if q.is_symlink() or (got['bytes'],got['sha256'])!=(item['bytes'],item['sha256']):raise ValueError('Context diverso da S: '+item['name'])
 allowed={r['name'] for r in binding['modules']+binding['build_inputs']}|{'Dockerfile','.dockerignore','image-source.json'}
 allowed|={'scripts/diagnostics/run-a001-fase0-uv/'+n for n in ['image_package.py','image_prepare.py','verify_distribution.py','make_source_manifest.py','check_python_origin.py']}
 actual={q.relative_to(a.repo).as_posix() for q in a.repo.rglob('*') if q.is_file()}
 if actual-allowed:raise ValueError('File context non ammesso: '+repr(sorted(actual-allowed)))
 with (a.output/'filtered-context.json').open('x') as f:json.dump(dict(status='PASS',files=[descriptor(a.repo/n) for n in sorted(actual)],allowed=sorted(allowed),sentinels_absent=True,S_id=binding['S_id']),f,indent=2)
 # Export nativo del lock e sync hash-locked dalla supply locale verificata.
 # I riferimenti registry del lock non vengono riscritti o simulati nella cache.
 def native(name,argv):
  out=a.output/name;out.mkdir();r=observed_run(argv,a.repo,out)
  if any(row['netns']!=namespace for row in r['observed_processes']):raise ValueError('Figlio fuori namespace')
  return r
 native('export',['/usr/local/bin/uv','export','--locked','--offline','--no-default-groups','--extra','marker-server','--extra',os.environ['MARKER_EXTRA'],'--no-emit-project','--format','requirements-txt','--output-file',str(a.output/'runtime-locked.txt')])
 native('venv',['/usr/local/bin/uv','venv','/opt/venv','--python','/usr/local/bin/python','--no-managed-python','--no-python-downloads'])
 runtime=Path('/opt/venv');rs=runtime/'lib/python3.12/site-packages'
 def startup(backend_ready):
  if (runtime/'bin/python').resolve()!=Path('/usr/local/bin/python3.12').resolve():raise ValueError('Python runtime diverso dal builder')
  expected=['_virtualenv.pth','distutils-precedence.pth'] if backend_ready else ['_virtualenv.pth']
  if sorted(q.name for q in rs.glob('*.pth'))!=expected or list(rs.glob('*customize*')):raise ValueError('Startup runtime inatteso')
  if (rs/'_virtualenv.pth').read_bytes()!=b'import _virtualenv' or descriptor(rs/'_virtualenv.py')['sha256']!='6cf30c56faf2a55228914dbbd17f8088ed371ebb08f5e7fa6fd931f913fcaf1d':raise ValueError('Startup uv runtime inatteso')
  return backend_files(wheel,rs) if backend_ready else []
 startup(False)
 native('backend-runtime',['/usr/local/bin/uv','pip','install','--offline','--python',str(runtime/'bin/python'),'--no-deps','--no-build',str(wheel)])
 runtime_backend=startup(True)
 with (a.output/'runtime-startup-before.json').open('x') as f:json.dump(dict(python=str(runtime/'bin/python'),real_python=str((runtime/'bin/python').resolve()),pth=[descriptor(q) for q in sorted(rs.glob('*.pth'))],backend_files=runtime_backend,backend_role='setuptools84 runtime dal lock e backend84 EbookLib prima del sync'),f,indent=2)
 result=native('sync',['/usr/local/bin/uv','pip','sync','--offline','--no-index','--find-links',str(a.supply),'--python',str(runtime/'bin/python'),'--require-hashes','--no-build-isolation','--build-constraints',str(a.repo/'build-constraints.txt'),str(a.output/'runtime-locked.txt')])
 print(json.dumps(dict(native_sync_trace=result)),flush=True)
 hooks=[r for r in result['observed_processes'] if 'setuptools' in r['cmdline'] and 'build_meta' in r['cmdline']]
 if not any('build_wheel' in r['cmdline'] for r in hooks):raise ValueError('Hook nativo EbookLib build_wheel non osservato')
 startup(True)
 native('pip-check',['/usr/local/bin/uv','pip','check','--python',str(runtime/'bin/python')])
 print('PASS preparazione nativa offline')
if __name__=='__main__':main()
