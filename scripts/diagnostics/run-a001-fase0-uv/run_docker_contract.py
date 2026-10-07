#!/usr/bin/env python3
"""C-docker permanente: ID locale, startup prima Python, full I e nove contratti.

Nessun snapshot storico, download, mount, inferenza o normale servizio.
"""
import argparse,hashlib,json,os,subprocess,time,signal,zipfile,tarfile,threading,uuid,re,importlib.util
from pathlib import Path

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for key in ['repo','source','installation','expected','inputs-receipt','backend-wheel','context-receipt','output']:p.add_argument('--'+key,type=Path,required=True)
 p.add_argument('--mode',choices=['official','standalone'],default='official')
 p.add_argument('--image',required=True);p.add_argument('--docker',default='/usr/bin/docker');p.add_argument('--docker-host',default='unix:///var/run/docker.sock');p.add_argument('--timeout',type=float,default=900);p.add_argument('--write-budget-bytes',type=int,default=67108864)
 a=p.parse_args();ROOT=a.repo.resolve();out=a.output.absolute();out.mkdir(exist_ok=False);start=time.monotonic();deadline=min(900,a.timeout);spent=0
 if not re.fullmatch(r'sha256:[0-9a-f]{64}',a.image) or not a.docker_host.startswith('unix:///'):raise ValueError('ID immagine e daemon UNIX locale richiesti')
 expected=json.loads(a.expected.read_text());image=a.image;expected['host_netns']=os.readlink('/proc/self/ns/net')
 if expected.get('schema')!=1 or not expected.get('S_id','').startswith('sha256:'):raise ValueError('Binding S non valido')
 spec=importlib.util.spec_from_file_location('host_gate',Path(__file__).with_name('verify_distribution.py'));gate=importlib.util.module_from_spec(spec);spec.loader.exec_module(gate)
 gate.validate_receipt(a.installation,a.source,ROOT,official=a.mode=='official')
 current=gate.source_tool.load_json(a.source)
 context_receipt=json.loads(a.context_receipt.read_text())
 if context_receipt['status']!='PASS_PREPARATION_ONLY' or context_receipt['context']['binding']!={k:expected[k] for k in context_receipt['context']['binding']}:raise ValueError('Receipt/binding contesto difforme')
 context_root=Path(context_receipt['argv'][context_receipt['argv'].index('--output')+1])
 for row in context_receipt['context']['files']:
  path=ROOT/Path(row['path']).relative_to(context_root);gate.source_tool.no_symlinks(path)
  if path.stat().st_size!=row['bytes'] or hashlib.sha256(path.read_bytes()).hexdigest()!=row['sha256']:raise ValueError('Input immagine consumato diverso dal clone corrente: '+str(path))
 tool=context_receipt['tool'];path=Path(__file__).with_name('prepare_docker_inputs.py')
 if path.stat().st_size!=tool['bytes'] or hashlib.sha256(path.read_bytes()).hexdigest()!=tool['sha256']:raise ValueError('Producer contesto cambiato')
 # La label host può cambiare solo dopo equivalenza esatta dei reali input immagine.
 host_equivalence=dict(host_S_id=current['id'],image_S_id=expected['S_id'],consumed_files=context_receipt['context']['files'],status='PASS_EXACT_CONSUMED_INPUTS')
 inputs=json.loads(a.inputs_receipt.read_text());assert inputs['status']=='PASS_PREPARATION_ONLY'
 expected['locked_runtime']={r['name']:r['version'] for r in inputs['wheels']['payloads']};expected['locked_runtime']['markdown-for-llms']='1.0.0'
 expected['native_versions']={r['package']:r['version'] for r in inputs['apt']['selected']}
 assert hashlib.sha256(a.backend_wheel.read_bytes()).hexdigest()=='51a52592b3b99e102b609654876bd65f19f999935166d1352678931132b0c670'
 config=out/'docker-config';config.mkdir();ENV={'PATH':'/usr/bin:/bin','HOME':str(config),'DOCKER_CONFIG':str(config),'LANG':'C.UTF-8','LC_ALL':'C.UTF-8'};BASE=[a.docker,'--host',a.docker_host];name='mdllms-contract-'+uuid.uuid4().hex[:12];calls=[]
 def write(path,value):
  with path.open('x') as f:json.dump(value,f,separators=(',',':'))
 def command(args):
  argv=BASE+args;r=subprocess.run(argv,env=ENV,cwd=ROOT,shell=False,close_fds=True,capture_output=True,text=True,timeout=60);row=dict(argv=argv,exit=r.returncode,stdout=r.stdout,stderr=r.stderr);calls.append(row);write(out/f'control-{len(calls):03d}.json',row)
  if r.returncode:raise ValueError(r.stderr)
  return r.stdout
 error=None;exitcode=None;contract=None;stop=None;samples=[];max_gap=0;cleanup_error=None;cid=None
 try:
  binding=command(['image','inspect',image]);detail=json.loads(binding)[0];assert detail['Id']==image and detail['Os']=='linux' and detail['Architecture']=='amd64'
  # Un solo container diagnostico, mai CMD normale/healthcheck. Prima si copia
  # il payload startup dallo stato fermo; nessun interprete venv avviato prima.
  argv=['create','--pull','never','--name',name,'--label','mdllms.role=offline-contract','--network','none','--no-healthcheck','--cap-drop','ALL','--security-opt','no-new-privileges','--pids-limit','128','--memory','2g','--cpus','2','-i','--entrypoint','/opt/venv/bin/python',image,'-I','-B','-'];cid=command(argv).strip();write(out/'container-id.json',dict(id=cid,name=name,image=image));error=None;exitcode=None;contract=None;stop=None;samples=[];max_gap=0;last_tick=time.monotonic()
  try:
   # Export del solo container fermo: leggere header/payload startup in streaming,
   # senza estrarre l'immagine o avviare Python/shell prima della guardia.
   z=zipfile.ZipFile(a.backend_wheel);backend={n:z.read(n) for n in z.namelist() if not n.endswith('/') and not n.endswith('/RECORD')};z.close()
   export_argv=BASE+['export',cid];seen=[];pth=[];customize=[];python_links={};cfg=None;binary=None;export_start=time.monotonic();export_error=None
   with (out/'export-stderr.txt').open('wb') as export_stderr:
    exporter=subprocess.Popen(export_argv,env=ENV,cwd=ROOT,shell=False,close_fds=True,stdout=subprocess.PIPE,stderr=export_stderr)
    timer=threading.Timer(60,exporter.terminate);timer.start()
    try:
     with tarfile.open(fileobj=exporter.stdout,mode='r|') as archive:
      for member in archive:
       name=member.name.removeprefix('./')
       if name.startswith('opt/venv/lib/python3.12/site-packages/') or name.startswith('usr/local/lib/python3.12/site-packages/'):
        relative=name.split('site-packages/',1)[1]
        if '/' not in relative and relative.endswith('.pth'):pth.append(name)
        if '/' not in relative and relative.startswith(('sitecustomize','usercustomize')):customize.append(name)
        if name.startswith('opt/venv/') and relative in backend:
         assert member.isfile() and archive.extractfile(member).read()==backend[relative],relative;seen.append(relative)
       if name in ['opt/venv/bin/python','opt/venv/bin/python3','opt/venv/bin/python3.12']:python_links[name]=dict(link=member.linkname,symlink=member.issym())
       if name=='opt/venv/pyvenv.cfg':cfg=archive.extractfile(member).read().decode()
       if name=='usr/local/bin/python3.12':binary=dict(bytes=member.size,sha256=hashlib.sha256(archive.extractfile(member).read()).hexdigest())
       assert time.monotonic()-export_start<60
     exporter.stdout.close();assert exporter.wait(timeout=5)==0
    finally:
     timer.cancel()
     if exporter.poll() is None:exporter.terminate();exporter.wait(timeout=5)
   assert sorted(seen)==sorted(backend) and not customize and sorted(pth)==['opt/venv/lib/python3.12/site-packages/_virtualenv.pth','opt/venv/lib/python3.12/site-packages/distutils-precedence.pth']
   assert binary and 'home = /usr/local/bin' in cfg and 'include-system-site-packages = false' in cfg and all(v['symlink'] for v in python_links.values()) and len(python_links)==3
   write(out/'filesystem-startup-before-python.json',dict(status='PASS',export_argv=export_argv,export_exit=exporter.returncode,seconds=time.monotonic()-export_start,pth=pth,customize=customize,backend_wheel_sha256=hashlib.sha256((a.backend_wheel).read_bytes()).hexdigest(),backend_files_verified=seen,python_links=python_links,pyvenv_cfg=cfg,python_binary=binary,image=image,no_interpreter_started_yet=True))
   startup=out/'startup';startup.mkdir();command(['cp',cid+':/opt/venv/lib/python3.12/site-packages/_virtualenv.pth',str(startup/'_virtualenv.pth')]);command(['cp',cid+':/opt/venv/lib/python3.12/site-packages/_virtualenv.py',str(startup/'_virtualenv.py')]);command(['cp',cid+':/opt/venv/lib/python3.12/site-packages/distutils-precedence.pth',str(startup/'distutils-precedence.pth')]);command(['cp',cid+':/opt/venv/pyvenv.cfg',str(startup/'pyvenv.cfg')]);command(['cp',cid+':/opt/prepare-proof/runtime-startup-before.json',str(startup/'builder-startup.json')]);command(['cp',cid+':/opt/build-proof',str(out/'build-proof')]);command(['cp',cid+':/opt/prepare-proof',str(out/'prepare-proof')])
   assert (startup/'_virtualenv.pth').read_bytes()==b'import _virtualenv';assert hashlib.sha256((startup/'_virtualenv.py').read_bytes()).hexdigest()=='6cf30c56faf2a55228914dbbd17f8088ed371ebb08f5e7fa6fd931f913fcaf1d'
   with zipfile.ZipFile(a.backend_wheel) as z:assert (startup/'distutils-precedence.pth').read_bytes()==z.read('distutils-precedence.pth')
   inspection=json.loads(command(['container','inspect',cid]))[0];assert not inspection['Mounts'] and inspection['HostConfig']['NetworkMode']=='none' and not inspection['HostConfig']['Privileged'] and inspection['Config']['Entrypoint']==['/opt/venv/bin/python'] and inspection['State']['Status']=='created';write(out/'container-before.json',inspection)
   trusted='EXPECTED='+repr(expected)+'\n'+'''import importlib.util,json,zipfile,hashlib,sysconfig,os
  from pathlib import Path
  # Prima di import applicativi: stessa startup attestata e tutti i bytes canonici.
  site=Path(sysconfig.get_path('purelib'))
  assert sorted(p.name for p in site.glob('*.pth'))==['_virtualenv.pth','distutils-precedence.pth']
  assert not list(site.glob('*customize*'))
  assert site.joinpath('_virtualenv.pth').read_bytes()==b'import _virtualenv'
  assert hashlib.sha256(site.joinpath('_virtualenv.py').read_bytes()).hexdigest()=='6cf30c56faf2a55228914dbbd17f8088ed371ebb08f5e7fa6fd931f913fcaf1d'
  startup=json.loads(Path('/opt/prepare-proof/runtime-startup-before.json').read_text())
  for row in startup['backend_files']:
   p=Path(row['path']);data=p.read_bytes();assert len(data)==row['bytes'] and hashlib.sha256(data).hexdigest()==row['sha256']
  root=Path('/opt/build-proof');record=json.loads((root/'I.json').read_text());assert record['S_id']==EXPECTED['S_id'] and record['S_sha256']==EXPECTED['S_sha256']
  spec=importlib.util.spec_from_file_location('runtime_verify',root/'diagnostics/verify_distribution.py');verify=importlib.util.module_from_spec(spec);spec.loader.exec_module(verify)
  wheel=list((root/'dist').glob('*.whl'))[0]
  for row in record['archives']:
   p=Path(row['path']);data=p.read_bytes();assert len(data)==row['bytes'] and hashlib.sha256(data).hexdigest()==row['sha256']
  with zipfile.ZipFile(wheel) as z:files={n:z.read(n) for n in z.namelist() if not n.endswith('/')}
  payload={'host_repo':'/src','modules':[dict(path=x['name'],kind='file',bytes=x['bytes'],sha256=x['sha256']) for x in EXPECTED['modules']]}
  actual=verify.check_installed(payload,files,wheel,'cpu');assert actual==record['installed']
  print(json.dumps({'status':'PASS_RUNTIME_I','installed':actual,'S_id':EXPECTED['S_id']}),flush=True)
  # Inventari reali runtime: nessun compilatore/dev/cache/src/UV o peso incorporato.
  import importlib.metadata,subprocess
  native=subprocess.run(['/usr/bin/dpkg-query','-W','-f','${Package}\\t${Version}\\n'],capture_output=True,text=True,check=True)
  assert not any(Path(p).exists() for p in ['/src','/verified-wheels','/build-cache','/usr/local/bin/uv','/usr/bin/gcc','/usr/bin/g++','/verified-apt'])
  assert not list(Path('/var/cache/apt/archives').glob('*.deb'))
  print(json.dumps({'status':'PASS_RUNTIME_INVENTORY','dpkg':native.stdout,'python':os.sys.version,'packages':{d.metadata['Name']:d.version for d in importlib.metadata.distributions()}}),flush=True)
  ''' + '''
  actual_versions={d.metadata['Name'].lower().replace('_','-'):d.version for d in importlib.metadata.distributions()}
  assert actual_versions==EXPECTED['locked_runtime'],(actual_versions,EXPECTED['locked_runtime'])
  native_versions=dict(line.split('\t',1) for line in native.stdout.splitlines())
  for name,version in EXPECTED['native_versions'].items():assert native_versions.get(name)==version,(name,native_versions.get(name),version)
  ''' + (ROOT/'scripts/diagnostics/run-a001-fase0-uv/probe_image_contract.py').read_text()
   # Normalizzare soltanto il prefisso generato, conservando il probe versionato.
   probe=(ROOT/'scripts/diagnostics/run-a001-fase0-uv/probe_image_contract.py').read_text()
   prefix=trusted[:-len(probe)]
   trusted='\n'.join(line[2:] if line.startswith('  ') else line for line in prefix.split('\n'))+probe
   import ast
   ast.parse(trusted)  # Un errore di generazione fallisce prima del Python container.
   write(out/'stdin-identity.json',dict(bytes=len(trusted.encode()),sha256=hashlib.sha256(trusted.encode()).hexdigest(),probe_file=str(ROOT/'scripts/diagnostics/run-a001-fase0-uv/probe_image_contract.py'),probe_sha256=hashlib.sha256((ROOT/'scripts/diagnostics/run-a001-fase0-uv/probe_image_contract.py').read_bytes()).hexdigest(),expected=expected))
   (out/'trusted-stdin.py').write_text(trusted)
   real=BASE+['start','-a','-i',cid];write(out/'invocation.json',dict(argv=real,create_argv=BASE+argv,environment=ENV,cwd=str(ROOT),shell=False,close_fds=True,spent_before=spent,deadline_seconds=deadline,image=image))
   last_tick=time.monotonic()
   with (out/'stdout.txt').open('xb') as stdout,(out/'stderr.txt').open('xb') as stderr:
    child=subprocess.Popen(real,env=ENV,cwd=ROOT,shell=False,close_fds=True,start_new_session=True,stdin=subprocess.PIPE,stdout=stdout,stderr=stderr);child.stdin.write(trusted.encode());child.stdin.close()
    while child.poll() is None:
     tick=time.monotonic();max_gap=max(max_gap,tick-last_tick);last_tick=tick;container_size=json.loads(command(['container','inspect','--size',cid]))[0];size=sum(p.stat().st_size for p in out.rglob('*') if p.is_file());engine_H=size+max(0,container_size.get('SizeRw',0));samples.append(dict(seconds=tick-start,output_bytes=size,container_SizeRw=container_size.get('SizeRw'),own_write_upper=engine_H));
     if engine_H>a.write_budget_bytes:stop='storage'
     elif tick-start>deadline:stop='deadline'
     elif max((out/'stdout.txt').stat().st_size,(out/'stderr.txt').stat().st_size)>1048576:stop='stream'
     if stop:command(['stop','--time','5',cid]);break
     time.sleep(.5)
    child.wait(timeout=15)
   inspection=json.loads(command(['container','inspect','--size',cid]))[0];assert not inspection['State']['Running'];exitcode=inspection['State']['ExitCode'];write(out/'container-after.json',inspection)
   rows=[json.loads(line) for line in (out/'stdout.txt').read_text().splitlines() if line.startswith('{')];contract=rows[-1] if rows else None
   assert exitcode==0 and child.returncode==0 and not stop and contract['status']=='PASS_OFFLINE_CONTRACT_ONLY',contract
   assert len(contract['subcases'])==9
  except Exception as e:error=repr(e)
  finally:
   state=json.loads(command(['container','inspect',cid]))[0]
   if state['State']['Running']:command(['stop','--time','5',cid]);state=json.loads(command(['container','inspect',cid]))[0]
   assert not state['State']['Running'];write(out/'process-collection.json',dict(status='COLLECTED',container=cid,state=state['State'],owned_only=True));command(['rm',cid])
 except Exception as e:error=f'{type(e).__name__}: {e}'
 finally:
  if cid:
   try:
    state=json.loads(command(['container','inspect',cid]))[0]['State']
    if state['Running']:command(['stop','--time','5',cid])
    command(['rm',cid])
   except Exception as e:
    # Il corpo può avere già raccolto/rimosso il container. Confermare l'assenza.
    probe=subprocess.run(BASE+['container','inspect',cid],env=ENV,capture_output=True,text=True,close_fds=True,timeout=30)
    if probe.returncode==0:cleanup_error=repr(e)
  result=dict(status='PASS' if error is None and not cleanup_error else 'FAIL',error=error,cleanup_error=cleanup_error,exit=exitcode,stop=stop,image=image,contract=contract,charged_seconds=time.monotonic()-start,samples=samples,max_scan_gap_seconds=max_gap,periodic_not_atomic=True,owned_container_collected=not cleanup_error,host_image_equivalence=host_equivalence,inputs=dict(context_receipt_sha256=hashlib.sha256(a.context_receipt.read_bytes()).hexdigest(),expected_sha256=hashlib.sha256(a.expected.read_bytes()).hexdigest(),supply_receipt_sha256=hashlib.sha256(a.inputs_receipt.read_bytes()).hexdigest()),limits='CPU installed/native/mock offline; no inference/GPU/weights/fonts/normal CMD')
  write(out/'result.json',result)
 print(json.dumps(dict(status=result['status'],error=error,output=str(out))))
 return 0 if result['status']=='PASS' else 2

if __name__=='__main__':
 try:raise SystemExit(main())
 except Exception as exc:
  # Anche un gate host fallito conserva receipt, senza avviare container.
  import sys
  if '--output' in sys.argv:
   directory=Path(sys.argv[sys.argv.index('--output')+1]);directory.mkdir(parents=True,exist_ok=True)
   receipt=directory/'result.json'
   if not receipt.exists():receipt.write_text(json.dumps(dict(status='FAIL',error=f'{type(exc).__name__}: {exc}',phase='host/preparation',owned_container_collected=True)))
  raise SystemExit(2)
