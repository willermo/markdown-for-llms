#!/usr/bin/env python3
"""Verifica stdlib S/archivi/installazione; non importa né ripara il prodotto.

Receipt I/E schema1: envelope canonico del produttore S, kind, scope, source
(ID e SHA file), snapshot, archives, interpreter, profile, inputs, argv/cwd,
status. E richiede inoltre installation (ID/SHA della receipt I verificata).
Ogni consumer deve chiamare validate_receipt prima di collection/import.
Le prove reali degli import appartengono a V3–V5, dopo questo gate.
"""
import argparse
import base64
import configparser
import csv
from email.parser import BytesParser
import hashlib
import importlib.metadata
import importlib.util
import io
import json
from pathlib import Path, PurePosixPath
import re
import stat
import sys
import sysconfig
import tarfile
import zipfile


def load_helper(name):
    path = Path(__file__).with_name(name+'.py')
    spec = importlib.util.spec_from_file_location('a001_'+name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


source_tool = load_helper('make_source_manifest')
origin_tool = load_helper('check_python_origin')
MODULE_FILES = {m+'.py' for m in source_tool.MODULES}
ENTRYPOINTS = {'markdown-pipeline':'master_workflow:main','markdown-config':'config:main',
               'markdown-clean':'clean_markdown:main','markdown-validate':'validate_markdown:main',
               'markdown-chunk':'chunk_markdown:main'}
DIST = 'markdown_for_llms-1.0.0.dist-info'
EGG = 'markdown_for_llms.egg-info'
SDIST_FILES = MODULE_FILES | {'pyproject.toml','MANIFEST.in','README.md','LICENSE',
                             '.python-version','build-constraints.txt','PKG-INFO','setup.cfg'}
SDIST_FILES |= {EGG+'/'+n for n in ('PKG-INFO','SOURCES.txt','dependency_links.txt',
                                  'entry_points.txt','requires.txt','top_level.txt')}
WHEEL_FILES = MODULE_FILES | {DIST+'/'+n for n in ('METADATA','WHEEL','RECORD',
                                                 'entry_points.txt','top_level.txt','licenses/LICENSE')}
MAX_FILE = 16 * 1024 * 1024
MAX_TOTAL = 64 * 1024 * 1024


def descriptor(path):
    source_tool.no_symlinks(path)
    data = path.read_bytes()
    return {'path':str(path),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}


def safe_name(name):
    p = PurePosixPath(name)
    if (not name or '\\' in name or ':' in name or '\x00' in name or p.is_absolute()
            or '..' in p.parts or '.' in name.split('/') or p.as_posix() != name):
        raise ValueError('Path archivio non sicuro: '+repr(name))
    return name


def read_archive(path, wheel):
    """Nessuna estrazione; allowlist esatta, limiti e nessun link/duplicato."""
    result, seen, total = {}, set(), 0
    if wheel:
        with zipfile.ZipFile(path) as archive:
            for member in archive.infolist():
                name = safe_name(member.filename.rstrip('/') if member.is_dir() else member.filename)
                if name in seen:
                    raise ValueError('Path ZIP duplicato')
                seen.add(name)
                mode = member.external_attr >> 16
                if stat.S_IFMT(mode) not in (0,stat.S_IFREG,stat.S_IFDIR) or member.flag_bits & 1:
                    raise ValueError('Link/tipo speciale/ZIP cifrato')
                if member.is_dir():
                    if name not in {DIST,DIST+'/licenses'}:
                        raise ValueError('Directory ZIP inattesa')
                    continue
                total += member.file_size
                if name not in WHEEL_FILES or member.file_size > MAX_FILE or total > MAX_TOTAL:
                    raise ValueError('Contenuto/dimensione wheel inatteso: '+name)
                result[name] = archive.read(member)
    else:
        with tarfile.open(path, 'r:*') as archive:
            for member in archive:
                name = safe_name(member.name.rstrip('/') if member.isdir() else member.name)
                if name in seen:
                    raise ValueError('Path tar duplicato')
                seen.add(name)
                parts = name.split('/')
                if parts[0] != 'markdown_for_llms-1.0.0':
                    raise ValueError('Radice sdist inattesa')
                if member.isdir():
                    if name not in {'markdown_for_llms-1.0.0','markdown_for_llms-1.0.0/'+EGG}:
                        raise ValueError('Directory tar inattesa')
                    continue
                if not member.isfile():
                    raise ValueError('Link/tipo tar non regolare')
                relative = '/'.join(parts[1:])
                total += member.size
                if relative not in SDIST_FILES or member.size > MAX_FILE or total > MAX_TOTAL:
                    raise ValueError('Contenuto/dimensione sdist inatteso: '+relative)
                result[relative] = archive.extractfile(member).read()
    return result


def check_metadata(data, entrypoints):
    metadata = BytesParser().parsebytes(data)
    for key, expected in (('Name','markdown-for-llms'),('Version','1.0.0'),('Requires-Python','<3.13,>=3.12')):
        values = metadata.get_all(key, [])
        matches = len(values) == 1 and (set(values[0].split(',')) == set(expected.split(',')) if key == 'Requires-Python' else values[0] == expected)
        if not matches:
            raise ValueError('Metadata difforme: '+key)
    parser = configparser.ConfigParser(interpolation=None, strict=True)
    parser.optionxform = str
    parser.read_string(entrypoints.decode('utf-8'))
    if parser.sections() != ['console_scripts'] or dict(parser['console_scripts']) != ENTRYPOINTS:
        raise ValueError('Cinque entrypoint difformi')
    requirements = metadata.get_all('Requires-Dist', [])
    base = {r.split(';')[0].strip() for r in requirements if ';' not in r}
    if base != {'requests>=2.31.0','tiktoken>=0.5.1','tqdm>=4.66.0','python-dotenv>=1.0.0'}:
        raise ValueError('Dipendenze runtime base difformi')
    extras = metadata.get_all('Provides-Extra', [])
    if len(extras) != 3 or set(extras) != {'marker-server','marker-cpu','marker-cu126'}:
        raise ValueError('Extra distribuzione difformi')
    for requirement in requirements:
        if ';' in requirement:
            dependency, marker = requirement.split(';',1)
            match = re.fullmatch(r'\s*extra\s*==\s*[\"\'](marker-server|marker-cpu|marker-cu126)[\"\']\s*',marker)
            if not match:
                raise ValueError('Marker Requires-Dist inatteso')
            allowed = ({'fastapi','uvicorn','python-multipart'} if match[1] == 'marker-server' else
                       {'marker-pdf[full]==1.10.2','surya-ocr==0.17.1','torch==2.7.1','beautifulsoup4'})
            if dependency.strip() not in allowed:
                raise ValueError('Dipendenza extra inattesa')
    for extra in extras:
        selected = [r.split(';',1)[0].strip() for r in requirements if re.search(r'extra\s*==\s*[\"\']'+extra+r'[\"\']',r)]
        expected = ({'fastapi','uvicorn','python-multipart'} if extra == 'marker-server' else
                    {'marker-pdf[full]==1.10.2','surya-ocr==0.17.1','torch==2.7.1','beautifulsoup4'})
        if len(selected) != len(expected) or set(selected) != expected:
            raise ValueError('Extra incompleto/duplicato')


def check_record(data, files, installed=False):
    rows = list(csv.reader(io.StringIO(data.decode('utf-8'))))
    seen = set()
    for row in rows:
        if len(row) != 3 or row[0] in seen:
            raise ValueError('RECORD malformato/duplicato')
        name, digest, size = row
        seen.add(name)
        # Solo i cinque script generati possono avere il path relativo fuori
        # purelib; il lettore installazione li vincola al bin del prefix.
        scripts = {'../../../bin/'+n for n in ENTRYPOINTS}
        if not (installed and name in scripts):
            safe_name(name)
        if name not in files:
            raise ValueError('RECORD contiene file sconosciuto: '+name)
        if name == DIST+'/RECORD':
            if digest or size:
                raise ValueError('RECORD deve omettere il proprio digest/dimensione')
            continue
        bytecode = {'__pycache__/'+m+'.cpython-312.pyc' for m in source_tool.MODULES}
        if installed and name in bytecode and not digest and not size:
            # PEP376 consente campi vuoti per pyc; i bytes effettivi sono
            # comunque inclusi nei descriptor della receipt installazione.
            continue
        if not digest.startswith('sha256=') or not size.isdecimal():
            raise ValueError('RECORD senza sha256/dimensione: '+name)
        encoded = digest.removeprefix('sha256=')
        try:
            raw = base64.b64decode(encoded+'='*((-len(encoded))%4), altchars=b'-_', validate=True)
        except ValueError as exc:
            raise ValueError('Digest RECORD invalido') from exc
        content = files[name]
        if (len(raw) != 32 or base64.urlsafe_b64encode(raw).rstrip(b'=').decode() != encoded or
                raw != hashlib.sha256(content).digest() or int(size) != len(content)):
            raise ValueError('RECORD hash/byte difformi: '+name)
    if set(files) != seen.intersection(files):
        raise ValueError('RECORD incompleto')


def installation_record_files(data, locate, roots, prefix):
    """Legge gli extra nativi dell'installer senza saltarne i digest RECORD."""
    extras = {DIST+'/'+n for n in ('INSTALLER','REQUESTED','direct_url.json','uv_cache.json')}
    bytecode = {'__pycache__/'+m+'.cpython-312.pyc' for m in source_tool.MODULES}
    scripts = {'../../../bin/'+n for n in ENTRYPOINTS}
    files, paths, seen, total = {}, {}, set(), 0
    for row in csv.reader(io.StringIO(data.decode('utf-8'))):
        if len(row) != 3 or row[0] in seen:
            raise ValueError('RECORD malformato/duplicato')
        name = row[0]; seen.add(name)
        if name in WHEEL_FILES:
            continue
        if name in scripts:
            p = prefix/'bin'/name.rsplit('/',1)[1]
        elif name in extras or name in bytecode:
            safe_name(name)
            p = Path(locate(name))
            if not any(p.is_relative_to(root) for root in roots):
                raise ValueError('Extra installazione fuori site-packages')
        else:
            raise ValueError('RECORD contiene file sconosciuto: '+name)
        source_tool.no_symlinks(p)
        info = p.lstat()
        total += info.st_size
        if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_FILE or total > MAX_TOTAL:
            raise ValueError('Tipo/dimensione extra installazione inattesi: '+name)
        files[name] = p.read_bytes()
        paths[name] = p
    return files, paths


def check_archives(payload, sdist, wheel):
    required_inputs = {'pyproject.toml','.python-version','uv.lock','build-constraints.txt','README.md','LICENSE','MANIFEST.in'}
    incoming = {Path(r['path']).name for r in payload['build_inputs'] if r['kind'] == 'file'}
    if not required_inputs <= incoming:
        raise ValueError('S package incompleto: lock e metadata reali obbligatori')
    tar, zipdata = read_archive(sdist, False), read_archive(wheel, True)
    required_tar = MODULE_FILES | (required_inputs - {'uv.lock'}) | {'PKG-INFO',EGG+'/entry_points.txt'}
    if not required_tar <= tar.keys() or set(zipdata) != WHEEL_FILES:
        raise ValueError('Archivio incompleto')
    host = Path(payload['host_repo'])
    for item in payload['modules']:
        name = Path(item['path']).name
        current = (host/item['path']).read_bytes()
        if len(current) != item['bytes'] or hashlib.sha256(current).hexdigest() != item['sha256']:
            raise ValueError('Sorgente diverso da S')
        if tar[name] != current or zipdata[name] != current:
            raise ValueError('Modulo archivio diverso da S: '+name)
    for item in payload['build_inputs']:
        name = Path(item['path']).name
        if item['kind'] == 'file' and name in tar and tar[name] != (host/item['path']).read_bytes():
            raise ValueError('Input backend nella sdist difforme: '+name)
    if zipdata[DIST+'/licenses/LICENSE'] != tar['LICENSE']:
        raise ValueError('Licenza wheel difforme')
    check_metadata(tar['PKG-INFO'],tar[EGG+'/entry_points.txt'])
    check_metadata(zipdata[DIST+'/METADATA'],zipdata[DIST+'/entry_points.txt'])
    if tar['PKG-INFO'] != zipdata[DIST+'/METADATA']:
        raise ValueError('Metadata sdist/wheel difformi')
    check_record(zipdata[DIST+'/RECORD'],zipdata)
    wheel_metadata = BytesParser().parsebytes(zipdata[DIST+'/WHEEL'])
    if wheel_metadata['Root-Is-Purelib'] != 'true' or wheel_metadata.get_all('Tag') != ['py3-none-any']:
        raise ValueError('Tipo/tag wheel inatteso')
    return zipdata


def check_direct_url(direct, wheel_path):
    """PEP610: URI obbligatorio; digest opzionale verificato quando dichiarato.

    uv0.10.10 lascia archive_info vuoto per una wheel locale. Il digest di B
    e il confronto di ogni file wheel rimangono obbligatori in check_installed.
    """
    if (not isinstance(direct,dict) or 'dir_info' in direct or
            direct.get('url') != wheel_path.as_uri() or
            not isinstance(direct.get('archive_info'),dict)):
        raise ValueError('Installazione non proveniente dalla wheel canonica/non editable')
    archive = direct['archive_info']
    expected = source_tool.digest(wheel_path)
    if set(archive) - {'hash','hashes'}:
        raise ValueError('archive_info inatteso')
    if 'hashes' in archive:
        if archive['hashes'] != {'sha256':expected}:
            raise ValueError('Hash direct_url diverso dalla wheel canonica')
    if 'hash' in archive and archive['hash'] != 'sha256='+expected:
        raise ValueError('Hash direct_url legacy diverso dalla wheel canonica')
    return 'DECLARED_AND_VERIFIED' if archive else 'NOT_DECLARED; verified B digest and all wheel payload bytes'


def check_installed(payload, wheel_files, wheel_path, profile):
    roots = {Path(sysconfig.get_path(k)).resolve() for k in ('purelib','platlib')}
    distributions = [d for d in importlib.metadata.distributions()
                     if (d.metadata.get('Name') or '').lower().replace('_','-') == 'markdown-for-llms']
    if len(distributions) != 1:
        raise ValueError('Distribuzione mancante/duplicata')
    dist = distributions[0]
    installed, installed_paths = {}, {}
    origins = {}
    for name in wheel_files:
        p = Path(dist.locate_file(name))
        source_tool.no_symlinks(p)
        if not any(p.resolve().is_relative_to(r) for r in roots):
            raise ValueError('Installazione fuori dal site-packages')
        if not stat.S_ISREG(p.lstat().st_mode) or p.stat().st_size > MAX_FILE:
            raise ValueError('File installazione non regolare/eccessivo')
        installed[name] = p.read_bytes()
        installed_paths[name] = p
        if name != DIST+'/RECORD' and installed[name] != wheel_files[name]:
            raise ValueError('Installazione stantia/difforme: '+name)
        if name in MODULE_FILES:
            spec = importlib.util.find_spec(name[:-3])
            if not spec or spec.origin != str(p):
                raise ValueError('Origine spec difforme: '+name)
            origins[name] = str(p)
    extra_files, extra_paths = installation_record_files(
        installed[DIST+'/RECORD'],dist.locate_file,roots,Path(sys.prefix))
    installed.update(extra_files); installed_paths.update(extra_paths)
    check_record(installed[DIST+'/RECORD'],installed,installed=True)
    check_metadata(installed[DIST+'/METADATA'],installed[DIST+'/entry_points.txt'])
    direct = dist.read_text('direct_url.json')
    if not direct:
        raise ValueError('direct_url mancante: wheel canonica non identificata')
    direct = json.loads(direct)
    direct_hash_status = check_direct_url(direct,wheel_path)
    packages = sorted(({'name':d.metadata['Name'],'version':d.version} for d in importlib.metadata.distributions()), key=lambda d:d['name'].lower())
    names = {d['name'].lower().replace('_','-') for d in packages}
    forbidden = {'fastapi','uvicorn','marker-pdf','surya-ocr','torch'}
    if profile in {'base','dev','cache-probe'} and names & forbidden:
        raise ValueError('Dipendenze server/ML nel profilo base/dev')
    if profile == 'cache-probe' and not any(x['name'].lower()=='setuptools' and x['version']=='84.0.0' for x in packages):
        raise ValueError('Backend84 mancante nel probe cache')
    if profile == 'dev' and not {'pytest','pytest-cov','pytest-mock'} <= names:
        raise ValueError('Gruppo dev incompleto')
    if profile in {'base','api','dev','cache-probe'} and names & {'marker-pdf','surya-ocr','torch'}:
        raise ValueError('Dipendenze ML fuori dal profilo CPU')
    if profile == 'base' and names & {'setuptools','pytest','pytest-cov','pytest-mock','black','flake8','mypy','httpx'}:
        raise ValueError('Dev/backend nel profilo base')
    if profile in {'api','cpu'} and not {'fastapi','uvicorn','python-multipart'} <= names:
        raise ValueError('Extra server incompleto')
    if profile == 'cpu' and not {'marker-pdf','surya-ocr','torch','beautifulsoup4'} <= names:
        raise ValueError('Extra CPU incompleto')
    files = [descriptor(installed_paths[n]) for n in sorted(installed)]
    if DIST+'/direct_url.json' not in installed_paths:
        files.append(descriptor(Path(dist.locate_file(DIST+'/direct_url.json'))))
    return {'origins':origins,'dependencies':packages,'direct_url':direct,'files':files,
            'canonical_archive':descriptor(wheel_path),'direct_url_hash_status':direct_hash_status}


def validate_receipt(receipt_path, source_path, repo, official=True):
    """Rifiuta ID corretto con SHA/file/input diverso prima del consumer."""
    source = source_tool.load_json(source_path)
    payload = source_tool.validate_current(source, repo, official=official)
    receipt = source_tool.load_json(receipt_path)
    r = source_tool.check_envelope(receipt)
    common = {'kind','scope','source','snapshot','archives','interpreter','profile','inputs','argv','cwd','status'}
    expected = common | ({'installation','result'} if r.get('kind') == 'E' else {'installed'})
    if set(r) != expected or r['kind'] not in {'I','E'} or r['status'] != 'PASS':
        raise ValueError('Receipt incompleta/non PASS I/E')
    if (r['source'] != dict(descriptor(source_path), id=source['id']) or
            r['scope'] != payload['scope'] or r['snapshot'] != payload['snapshot']):
        raise ValueError('Receipt vecchia rispetto a S/snapshot')
    for records in (r['archives'],r['inputs']):
        if not records or len({x['path'] for x in records}) != len(records):
            raise ValueError('Input receipt vuoto/duplicato')
        for item in records:
            if descriptor(Path(item['path'])) != item:
                raise ValueError('SHA/file/input receipt difforme')
    if len(r['archives']) != 2:
        raise ValueError('Richiesti esattamente sdist e wheel')
    if r['kind'] == 'E':
        installation = r['installation']
        path = Path(installation['path'])
        if dict(descriptor(path),id=source_tool.load_json(path)['id']) != installation:
            raise ValueError('Receipt I cambiata')
        if source_tool.check_envelope(source_tool.load_json(path)).get('kind') != 'I':
            raise ValueError('E deve riferire I, non altra E')
        validate_receipt(path, source_path, repo, official)
    else:
        origin = origin_tool.check_origin(Path(r['interpreter']['expected_managed']))
        if origin != r['interpreter']:
            raise ValueError('Interprete diverso dalla receipt I')
        tar, wheel = (Path(x['path']) for x in r['archives'])
        files = check_archives(payload,tar,wheel)
        if check_installed(payload,files,wheel,r['profile']) != r['installed']:
            raise ValueError('Installazione diversa dalla receipt I')
    return r


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('repo','source-manifest','sdist','wheel','expected-managed','receipt'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--profile',choices=('base','dev','api','cpu','cache-probe'),required=True)
    parser.add_argument('--archives-only',action='store_true')
    args = parser.parse_args()
    try:
        for path in (args.repo,args.source_manifest,args.sdist,args.wheel,args.receipt):
            source_tool.no_symlinks(path)
        if args.receipt.exists():
            raise ValueError('Output già presente')
        origin = origin_tool.check_origin(args.expected_managed)
        source = source_tool.load_json(args.source_manifest)
        payload = source_tool.validate_current(source,args.repo)
        before = [descriptor(p) for p in (args.source_manifest,args.sdist,args.wheel)]
        wheel_files = check_archives(payload,args.sdist,args.wheel)
        installed = None if args.archives_only else check_installed(payload,wheel_files,args.wheel,args.profile)
        source_tool.validate_current(source,args.repo)
        if before != [descriptor(p) for p in (args.source_manifest,args.sdist,args.wheel)]:
            raise ValueError('Input cambiato durante il preflight')
        result = source_tool.envelope(dict(
            kind='ARCHIVES' if args.archives_only else 'I', scope=payload['scope'],
            source=dict(before[0],id=source['id']), snapshot=payload['snapshot'], archives=before[1:],
            interpreter=origin, profile=args.profile,
            inputs=[descriptor(Path(__file__)),descriptor(Path(source_tool.__file__)),descriptor(Path(origin_tool.__file__))],
            argv=sys.argv, cwd=str(Path.cwd()), status='PASS', installed=installed))
        with args.receipt.open('x',encoding='utf-8') as f:
            json.dump(result,f,indent=2); f.write('\n')
        print(json.dumps({'status':'PASS','id':result['id'],'kind':result['payload']['kind']}))
        return 0
    except (OSError, ValueError, KeyError, TypeError, tarfile.TarError, zipfile.BadZipFile) as exc:
        print('FAIL distribuzione: '+str(exc),file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
