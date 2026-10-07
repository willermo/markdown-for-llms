#!/usr/bin/env python3
"""Supply pubblica e contesto CPU espliciti; nessun riferimento a una run.

La preparazione può acquisire soltanto con --acquire. La build usa sempre
input locali verificati. Output nuovi, ricevute anche su errore, nessun prune.
"""
import argparse
import csv
import hashlib
import importlib.util
import io
import json
import lzma
import os
from pathlib import Path
import re
import shutil
import ssl
import subprocess
import time
import tomllib
import urllib.parse
import urllib.request
import uuid
import zipfile

BACKEND = ('setuptools-84.0.0-py3-none-any.whl', '51a52592b3b99e102b609654876bd65f19f999935166d1352678931132b0c670')
NATIVE = ['ca-certificates','curl','libpango-1.0-0','libpangoft2-1.0-0','libharfbuzz-subset0','fontconfig','fonts-dejavu-core','libgomp1']
DIAGNOSTICS = ['image_package.py','image_prepare.py','verify_distribution.py','make_source_manifest.py','check_python_origin.py']


def descriptor(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for part in iter(lambda:f.read(65536), b''): h.update(part)
    return dict(path=str(path),bytes=path.stat().st_size,sha256=h.hexdigest())


def regular(path):
    if not path.is_file() or any(p.is_symlink() for p in [path,*path.parents]):
        raise ValueError('File regolare senza symlink richiesto: '+str(path))
    return path


def check(path, sha, size=None):
    row=descriptor(regular(path))
    if row['sha256']!=sha or size is not None and row['bytes']!=size:
        raise ValueError('Payload difforme: '+str(path))
    return row


def check_wheel(path):
    import base64
    with zipfile.ZipFile(path) as z:
        names=z.namelist()
        if len(set(names))!=len(names):raise ValueError('ZIP duplicato')
        for member in z.infolist():
            p=Path(member.filename)
            if p.is_absolute() or '..' in p.parts or (member.external_attr>>16)&0o170000==0o120000:
                raise ValueError('ZIP evasivo')
        records=[n for n in names if len(n.split('/'))==2 and n.endswith('.dist-info/RECORD')]
        if len(records)!=1:raise ValueError('RECORD wheel mancante/duplicato')
        listed=set()
        for name, digest, size in csv.reader(io.StringIO(z.read(records[0]).decode())):
            if name in listed or name not in names:raise ValueError('RECORD incoerente')
            listed.add(name)
            if name==records[0]:
                if digest or size:raise ValueError('Self RECORD inatteso')
            else:
                data=z.read(name)
                expected='sha256='+base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b'=').decode()
                if digest!=expected or int(size)!=len(data):raise ValueError('Payload/RECORD difforme')
        if listed!={n for n in names if not n.endswith('/')}:raise ValueError('Payload non inventariato')


class Preparation:
    def __init__(self,a):
        self.a=a;self.calls=[];self.reads=[];self.network=0;self.start=time.monotonic()
        self.env={'PATH':'/usr/bin:/bin','HOME':str(a.output/'docker-config'),'DOCKER_CONFIG':str(a.output/'docker-config'),
                  'LANG':'C.UTF-8','LC_ALL':'C.UTF-8','UV_PYTHON_DOWNLOADS':'never','PYTHONDONTWRITEBYTECODE':'1'}
        (a.output/'docker-config').mkdir()
        self.docker=[a.docker,'--host',a.docker_host]

    def command(self, argv, timeout=180):
        self.admit()
        r=subprocess.run([str(x) for x in argv],env=self.env,cwd=self.a.repo,shell=False,close_fds=True,
                         capture_output=True,text=True,timeout=timeout)
        row=dict(argv=[str(x) for x in argv],environment=self.env,cwd=str(self.a.repo),exit=r.returncode,stdout=r.stdout,stderr=r.stderr)
        self.calls.append(row)
        if max(len(r.stdout.encode()),len(r.stderr.encode()))>1048576:raise ValueError('Stream cap')
        if r.returncode:raise ValueError('Comando preparazione fallito: '+r.stderr)
        return r.stdout

    def admit(self):
        if time.monotonic()-self.start>self.a.timeout:raise TimeoutError('Deadline preparazione')
        logical=allocated=0;pending=[self.a.output]
        while pending:
            path=pending.pop();info=path.lstat();logical+=info.st_size;allocated+=info.st_blocks*512
            if path.is_dir() and not path.is_symlink():pending.extend(path.iterdir())
        if max(logical,allocated)>=self.a.storage_budget_bytes:raise ValueError('Quota storage preparazione')
        if shutil.disk_usage(self.a.output).free<1073741824:raise ValueError('Spazio libero preparazione insufficiente')

    def fetch(self,url,path,sha=None,size=None):
        self.admit()
        if not self.a.acquire:raise ValueError('Supply assente: acquisizione esplicita --acquire richiesta per '+url)
        source=urllib.parse.urlsplit(url)
        if source.scheme!='https' or source.hostname not in {'snapshot.debian.org','files.pythonhosted.org','download.pytorch.org','download-r2.pytorch.org'}:
            raise ValueError('Fonte non configurata nel lock/ricetta: '+url)
        hosts={source.hostname}
        if source.hostname=='download.pytorch.org':hosts.add('download-r2.pytorch.org')
        redirects=[]
        class Redirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(self,req,fp,code,msg,headers,newurl):
                u=urllib.parse.urlsplit(newurl)
                if u.scheme!='https' or u.hostname not in hosts:raise ValueError('Redirect fuori provenienza')
                redirects.append(dict(source=req.full_url,target=newurl,status=code))
                return super().redirect_request(req,fp,code,msg,headers,newurl)
        opener=urllib.request.build_opener(urllib.request.ProxyHandler({}),Redirect(),urllib.request.HTTPSHandler(context=ssl.create_default_context()))
        bound=min(size if size is not None else self.a.max_payload_bytes,self.a.network_budget_bytes-self.network)
        if bound<=0:raise ValueError('Quota acquisizione esaurita')
        path.parent.mkdir(parents=True,exist_ok=True)
        with opener.open(urllib.request.Request(url,headers={'User-Agent':'Debian APT-HTTP/1.3 (verified preparation)'}),timeout=60) as response,path.open('xb') as out:
            declared=response.headers.get('Content-Length')
            if declared and int(declared)>bound:raise ValueError('Payload non ammesso')
            count=0
            while True:
                chunk=response.read(min(65536,bound-count+1))
                if not chunk:break
                count+=len(chunk);self.network+=len(chunk)
                if count>bound:raise ValueError('Cap acquisizione')
                out.write(chunk)
                self.admit()
        row=check(path,sha,size) if sha else descriptor(path)
        self.reads.append(dict(url=url,redirects=redirects,**row))
        return row

    def image(self,ref,origin):
        if origin=='registry-1.docker.io/library/python':
            ref=ref.replace('docker.io/library/python@','registry-1.docker.io/library/python@',1) if ref.startswith('docker.io/') else ref
        if re.fullmatch(re.escape(origin)+r'@sha256:[0-9a-f]{64}',ref) is None:
            raise ValueError('Riferimento pinned della fonte configurata richiesto')
        probe=subprocess.run(self.docker+['image','inspect',ref],env=self.env,capture_output=True,text=True,close_fds=True,timeout=30)
        if probe.returncode:
            if not self.a.acquire:raise ValueError('Base locale assente; nessun pull implicito')
            raw=self.command(self.docker+['buildx','imagetools','inspect','--raw',ref])
            metadata=json.loads(raw)
            if 'layers' not in metadata:raise ValueError('Pin del manifest linux/amd64 richiesto, non index multi-ABI')
            upper=sum(x['size'] for x in metadata['layers'])+metadata['config']['size']
            allowance=int(upper*1.1)+5242880
            if self.network+allowance>self.a.network_budget_bytes:raise ValueError('Quota OCI insufficiente')
            self.network+=allowance # upper dichiarato; non misura wire
            self.command(self.docker+['pull','--platform','linux/amd64',ref],timeout=900)
        obj=json.loads(self.command(self.docker+['image','inspect',ref]))[0]
        if obj['Os']!='linux' or obj['Architecture']!='amd64' or not any(x.endswith('@'+ref.split('@')[1]) for x in obj.get('RepoDigests',[])):
            raise ValueError('Origine/ABI OCI non confermata')
        return obj

    def apt(self):
        a=self.a;apt=a.output/'apt-repository';apt.mkdir();(apt/'snapshot.txt').write_text(a.apt_snapshot)
        if a.apt_repository and (a.apt_repository/'snapshot.txt').read_text().strip()!=a.apt_snapshot:
            raise ValueError('Snapshot della supply APT differente')
        keyring=a.output/'debian-archive-keyring.gpg'
        cid=self.command(self.docker+['create','--pull=never','--network','none','--no-healthcheck','--entrypoint','/bin/false',a.python_image]).strip()
        try:self.command(self.docker+['cp',cid+':/usr/share/keyrings/debian-archive-keyring.gpg',str(keyring)])
        finally:self.command(self.docker+['rm',cid])
        indices={};bindings=[]
        for repository,suite in [('debian','bookworm'),('debian-security','bookworm-security')]:
            rel=Path(repository)/'dists'/suite;dest=apt/rel;dest.mkdir(parents=True)
            base=f'https://snapshot.debian.org/archive/{repository}/{a.apt_snapshot}/'
            release=dest/'InRelease';old=a.apt_repository/rel/'InRelease' if a.apt_repository else None
            if old and old.is_file():shutil.copyfile(regular(old),release)
            else:self.fetch(base+'dists/'+suite+'/InRelease',release,size=None)
            signature=self.command(['/usr/bin/gpgv','--status-fd','1','--keyring',str(keyring),str(release)])
            if '[GNUPG:] VALIDSIG ' not in signature:raise ValueError('Firma APT non valida')
            entries={};active=False
            for line in release.read_text().splitlines():
                if line=='SHA256:':active=True;continue
                if active and not line.startswith(' '):break
                if active:
                    sha,size,name=line.split();entries[name]=(sha,int(size))
            name='main/binary-amd64/Packages.xz';sha,size=entries[name];packages=dest/name;packages.parent.mkdir(parents=True)
            old=a.apt_repository/rel/name if a.apt_repository else None
            if old and old.is_file():check(old,sha,size);shutil.copyfile(regular(old),packages)
            else:self.fetch(base+'dists/'+suite+'/'+name,packages,sha,size)
            record={}
            for line in lzma.decompress(packages.read_bytes()).decode().splitlines()+['']:
                if not line and record:
                    if 'Filename' in record:indices[(repository,record['Filename'])]=record
                    record={}
                elif ': ' in line and not line.startswith(' '):
                    k,v=line.split(': ',1);record[k]=v
            bindings.append(dict(repository=repository,suite=suite,snapshot=a.apt_snapshot,InRelease=descriptor(release),Packages=descriptor(packages),signature=signature,keyring=descriptor(keyring)))
        # APT nativo, solo dati pubblici readonly. Nessun servizio normale/Python.
        script="set -eu\nrm /etc/apt/sources.list.d/debian.sources\nprintf 'deb [check-valid-until=no] file:/verified-apt/debian bookworm main\\ndeb [check-valid-until=no] file:/verified-apt/debian-security bookworm-security main\\n' > /etc/apt/sources.list\napt-get -o Acquire::Languages=none -o Acquire::By-Hash=false -o APT::Sandbox::User=root update\napt-get -o APT::Sandbox::User=root --print-uris install -y --no-install-recommends "+' '.join(NATIVE)
        cid=self.command(self.docker+['create','--name','mdllms-apt-'+uuid.uuid4().hex[:12],'--label','mdllms.role=apt-preparation','--pull=never','--network','none','--no-healthcheck','--cap-drop=ALL','--security-opt','no-new-privileges','--pids-limit','64','--memory','256m','--mount',f'type=bind,source={apt},target=/verified-apt,readonly','--entrypoint','/bin/sh',a.python_image,'-c',script]).strip()
        try:raw=self.command(self.docker+['start','--attach',cid],timeout=180)
        finally:
            state=json.loads(self.command(self.docker+['inspect',cid]))[0]['State']
            if state['Running']:self.command(self.docker+['stop','--time','5',cid])
            self.command(self.docker+['rm',cid])
        selected=[]
        for line in raw.splitlines():
            match=re.match(r"^'file:/verified-apt/(debian(?:-security)?)/(pool/[^']+)' ([^ ]+) (\d+)",line)
            if not match:continue
            repository,name,filename,size=match.groups();name=urllib.parse.unquote(name);record=indices[(repository,name)]
            if '..' in Path(name).parts or Path(name).name!=filename or int(size)!=int(record['Size']):raise ValueError('APT selezione difforme')
            dest=apt/repository/name;dest.parent.mkdir(parents=True,exist_ok=True);sha=record['SHA256'];size=int(size)
            old=a.apt_repository/repository/name if a.apt_repository else None
            url=f'https://snapshot.debian.org/archive/{repository}/{a.apt_snapshot}/'+urllib.parse.quote(name,safe='/')
            if old and old.is_file():
                check(old,sha,size)
                if a.reuse_in_place:dest=old
                else:shutil.copyfile(regular(old),dest)
            else:self.fetch(url,dest,sha,size)
            selected.append(dict(package=record['Package'],version=record['Version'],url=url,**check(dest,sha,size)))
        if not 8<len(selected)<64:raise ValueError('Selezione APT incompleta')
        return dict(bindings=bindings,selected=selected,resolver='native apt-get --print-uris; offline signed local repositories')

    def wheels(self):
        from packaging.markers import Marker,default_environment
        from packaging.tags import sys_tags
        from packaging.utils import parse_wheel_filename
        a=self.a;lock=tomllib.loads((a.repo/'uv.lock').read_text());directory=a.output/'wheels';directory.mkdir()
        export=self.command([str(a.uv),'export','--locked','--offline','--no-default-groups','--extra','marker-cpu','--extra','marker-server','--no-emit-project','--no-hashes'])
        selected={};environment=default_environment();environment['python_version']='3.12';environment['python_full_version']='3.12.13'
        for line in export.splitlines():
            m=re.match(r'^([A-Za-z0-9_.-]+)==([^ ;]+)(?:\s*;\s*(.*))?$',line)
            if m and (not m[3] or Marker(m[3]).evaluate(environment)):selected[m[1].lower().replace('_','-')]=m[2]
        tags=set(sys_tags());payloads=[]
        for package in lock['package']:
            name=package['name'].lower().replace('_','-')
            if name not in selected:continue
            if package['version']!=selected[name]:continue # fork non selezionato dall'export nativo
            candidates=[]
            for row in package.get('wheels',[]):
                filename=Path(urllib.parse.unquote(urllib.parse.urlsplit(row['url']).path)).name
                if parse_wheel_filename(filename)[3]&tags:candidates.append((filename,row))
            if candidates:
                available=[x for x in candidates if a.wheel_dir and any(p.name.lower()==x[0].lower() for p in a.wheel_dir.iterdir())]
                filename,row=sorted(available or candidates)[0]
            elif name=='ebooklib' and package['version']=='0.18':
                row=package['sdist'];filename=Path(urllib.parse.unquote(urllib.parse.urlsplit(row['url']).path)).name
            else:raise ValueError('Wheel ABI mancante o backend sconosciuto: '+name)
            sha=row['hash'].removeprefix('sha256:');size=row.get('size');dest=directory/filename;old=a.wheel_dir/filename if a.wheel_dir else None
            if old and not old.is_file():
                aliases=[p for p in a.wheel_dir.iterdir() if p.name.lower()==filename.lower()]
                if len(aliases)==1:old=aliases[0]
            if old and old.is_file():
                check(old,sha,size)
                if a.reuse_in_place:dest=old
                else:shutil.copyfile(regular(old),dest)
            else:self.fetch(row['url'],dest,sha,size)
            if dest.suffix=='.whl':check_wheel(dest)
            payloads.append(dict(name=name,version=package['version'],url=row['url'],**check(dest,sha,size)))
        if len(payloads)!=len(selected):raise ValueError('Export CPU incompleto')
        backend=a.wheel_dir/BACKEND[0] if a.reuse_in_place else directory/BACKEND[0]
        if not backend.is_file():
            old=a.wheel_dir/BACKEND[0] if a.wheel_dir else None
            if old and old.is_file():check(old,BACKEND[1]);shutil.copyfile(regular(old),directory/BACKEND[0])
            else:raise ValueError('Backend84 esplicito da supply hash-pinned richiesto; nessun backend implicito')
        check(backend,BACKEND[1]);check_wheel(backend)
        if a.reuse_in_place:
            allowed={Path(r['path']).name for r in payloads}|{BACKEND[0]}
            if {p.name for p in a.wheel_dir.iterdir()}!=allowed:raise ValueError('Supply in-place contiene file non selezionati')
        return dict(native_export=export,payloads=payloads,backend=descriptor(backend),ABI=environment,profile='marker-server+marker-cpu; no dev/cu126')


def context(a):
    spec=importlib.util.spec_from_file_location('source',Path(__file__).with_name('make_source_manifest.py'));source=importlib.util.module_from_spec(spec);spec.loader.exec_module(source)
    S=source.load_json(a.source);payload=source.validate_current(S,a.repo,official=a.mode=='official')
    if payload['repo']!=payload['host_repo']:raise ValueError('Contesto richiede clone prodotto, non copia probe')
    binding=dict(schema=1,S_id=S['id'],S_sha256=source.digest(a.source),modules=[],build_inputs=[])
    paths=['Dockerfile','.dockerignore']+[f'scripts/diagnostics/run-a001-fase0-uv/{n}' for n in DIAGNOSTICS]
    for kind in ['modules','build_inputs']:
        for row in payload[kind]:
            if row['kind']=='file':
                binding[kind].append(dict(name=row['path'],bytes=row['bytes'],sha256=row['sha256']));paths.append(row['path'])
    for name in sorted(set(paths)):
        p=regular(a.repo/name);out=a.output/name;out.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,out)
    (a.output/'image-source.json').write_text(json.dumps(binding,indent=2))
    # Solo sentinelle nuove nella copia; il filtro effettivo è attestato nel builder.
    sentinels=['.env','.env.private','local.env','pipeline_config.json','.git/private','.venv/private','temp/private','tmp/private','source_documents/private','output/private','cache/private']
    for name in sentinels:
        p=a.output/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('SYNTHETIC_EXCLUDED_SENTINEL\n')
    return dict(S=descriptor(a.source),binding=binding,files=[descriptor(a.output/name) for name in sorted(set(paths))],sentinels=[descriptor(a.output/name) for name in sentinels])


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('operation',choices=['supply','context'])
    p.add_argument('--repo',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--source',type=Path);p.add_argument('--mode',choices=['official','standalone'],default='official')
    p.add_argument('--wheel-dir',type=Path);p.add_argument('--apt-repository',type=Path);p.add_argument('--uv',type=Path)
    p.add_argument('--python-image');p.add_argument('--uv-image');p.add_argument('--apt-snapshot')
    p.add_argument('--docker',default='/usr/bin/docker');p.add_argument('--docker-host',default='unix:///var/run/docker.sock')
    p.add_argument('--acquire',action='store_true');p.add_argument('--network-budget-bytes',type=int,default=1073741824)
    p.add_argument('--reuse-in-place',action='store_true',help='Verifica ogni file della supply locale senza duplicarne i payload')
    p.add_argument('--storage-budget-bytes',type=int,default=17179869184);p.add_argument('--timeout',type=float,default=1800)
    p.add_argument('--max-payload-bytes',type=int,default=1073741824);a=p.parse_args();a.repo=a.repo.resolve();a.output=a.output.absolute()
    if any(x.is_symlink() for x in [a.output,*a.output.parents]):raise ValueError('Output symlink')
    a.output.mkdir(exist_ok=False);result=dict(status='FAIL',operation=a.operation,repo=str(a.repo),argv=os.sys.argv,tool=descriptor(Path(__file__)));tool=None;start=time.monotonic()
    try:
        if not os.sys.flags.isolated or not os.sys.dont_write_bytecode or os.sys.version_info[:3]!=(3,12,13):
            raise ValueError('Richiesto managed CPython3.12.13 -I -B già verificato')
        if a.operation=='context':result['context']=context(a)
        else:
            if not all([a.uv,a.python_image,a.uv_image,a.apt_snapshot]) or re.fullmatch(r'\d{8}T\d{6}Z',a.apt_snapshot) is None:raise ValueError('uv/basi pinned/snapshot APT espliciti richiesti')
            if not a.docker_host.startswith('unix:///'):raise ValueError('Solo daemon UNIX locale esplicitamente ammesso')
            tool=Preparation(a);result['images']=[tool.image(a.python_image,'registry-1.docker.io/library/python'),tool.image(a.uv_image,'ghcr.io/astral-sh/uv')]
            if a.reuse_in_place and (not a.wheel_dir or not a.apt_repository or a.acquire):raise ValueError('In-place richiede due input locali e nessuna acquisizione')
            result['wheels']=tool.wheels();result['apt']=tool.apt()
            result['contexts']={'verified-wheels':str(a.wheel_dir if a.reuse_in_place else a.output/'wheels'),
                                'verified-apt':str(a.apt_repository if a.reuse_in_place else a.output/'apt-repository')}
        result['status']='PASS_PREPARATION_ONLY'
    except Exception as e:result['error']=f'{type(e).__name__}: {e}'
    finally:
        result['seconds']=time.monotonic()-start
        if tool:result.update(calls=tool.calls,reads=tool.reads,network_accounted_upper=tool.network,wire_measured=False)
        # La receipt resta fuori dal contesto/supply consumato dalla build.
        with a.output.with_suffix('.receipt.json').open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(dict(status=result['status'],output=str(a.output),error=result.get('error'))))
    return 0 if result['status']=='PASS_PREPARATION_ONLY' else 2


if __name__=='__main__':raise SystemExit(main())
