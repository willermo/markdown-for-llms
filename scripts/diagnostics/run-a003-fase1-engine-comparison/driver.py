"""Driver finito a003: singoli tentativi, standard R+SBI e consumer Bubblewrap."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import datetime
import stat


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


HERE = Path(__file__).absolute().parent
c = load_module('a003_confine', HERE / 'confinement.py')


def outcome(receipt, raw_present, raw_bytes, probes_passed):
    if not probes_passed:
        return 'NOT_MEASURED'
    reason = receipt.get('stop_reason')
    if reason == 'TIMEOUT':
        return 'TIMEOUT'
    if reason == 'INTERRUPTED':
        return 'INTERRUPTED'
    if reason or receipt.get('adapter_exit') != 0:
        return 'ERROR'
    if not raw_present:
        return 'ERROR'
    return 'SUCCESS' if raw_bytes else 'EMPTY'


def engine_exit(receipt, engine, probes_passed):
    # Bubblewrap/consumer admission exits are not exits of a document engine.
    return receipt.get('adapter_exit') if probes_passed and engine != 'pandoc-legacy-v1' else None


def artifact_inventory(root, safety, errors=None):
    """Inerte e confinato; rifiuta link/special prima di leggere."""
    root = Path(root)
    result = []
    def entries(directory, expected):
        # dirfd no-follow: scandir fallibile, non rglob che salta EACCES.
        directory = c.canonical_path(str(directory))
        fd = os.open('/', os.O_RDONLY | os.O_DIRECTORY)
        try:
            for part in directory.parts[1:]:
                child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
                os.close(fd); fd = child
            before = os.fstat(fd)
            if (before.st_dev, before.st_ino) != expected:
                raise ValueError('Inventory directory replaced before traversal')
            with os.scandir(fd) as stream:
                names = sorted(entry.name for entry in stream)
            current = directory.lstat()
            if (before.st_dev, before.st_ino) != (current.st_dev, current.st_ino):
                raise ValueError('Inventory directory replaced')
            return [directory / name for name in names]
        finally:
            os.close(fd)
    info = root.lstat()
    pending = [(root, (info.st_dev, info.st_ino))]
    paths = []
    while pending:
        directory, identity = pending.pop()
        try:
            for path in entries(directory, identity):
                info = path.lstat()
                if stat.S_ISDIR(info.st_mode):
                    pending.append((path, (info.st_dev, info.st_ino)))
                else:
                    paths.append(path)
        except (OSError, ValueError) as exc:
            if errors is None:
                raise
            errors.append({'file':str(directory),'phase':'inventory-traversal','error':repr(exc)})
    for path in sorted(paths):
        try:
            if path.is_symlink():
                raise ValueError('Asset symlink: ' + str(path))
            if path.is_dir():
                continue
            name = path.relative_to(root).as_posix()
            info = path.lstat()
            identity = {'dev': info.st_dev, 'ino': info.st_ino}
            content = c.safe_bytes(root, name, safety.MAX_FILE, expected_identity=identity)
            row = {'path':str(path),'bytes':len(content),'sha256':__import__('hashlib').sha256(content).hexdigest(), 'identity': identity}
            row['relative_path'] = name
            result.append(row)
        except (OSError, ValueError) as exc:
            if errors is None:
                raise
            errors.append({'file':str(path),'phase':'inventory','error':repr(exc)})
    return result


def wrapper_budget(child_timeout):
    if type(child_timeout) is not int or not 0 < child_timeout <= 690:
        raise ValueError('Child timeout leaves no standard wrapper margin')
    return child_timeout + 210


def final_status(status, wrapper, runner, integrity):
    if not integrity or wrapper.get('adapter_exit') != 0 or wrapper.get('stop_reason') or not wrapper.get('children_reaped') or runner.get('status') != 'PASS':
        return 'ERROR' if status in {'SUCCESS','EMPTY'} else status
    return status


def native(repo, stage, case, engine, attempt_id):
    """Staging fidato prima del consumer; nessuna modifica agli adapter standard."""
    sys.path.insert(0, str(repo / 'scripts/benchmarks'))
    import adapters
    import profile
    import safety
    import pandoc_guard
    from resources import run
    runroot = c.RUN
    delivery = stage.get('delivery', c.DELIVERY)
    import re
    if delivery != c.DELIVERY or not re.fullmatch(r'[a-z0-9][a-z0-9-]{0,79}', attempt_id):
        raise ValueError('New operational delivery destinations/attempt ID required')
    budget = wrapper_budget(stage.get('timeout_s',120))
    evidence = runroot / 'evidence' / delivery / 'attempts' / attempt_id
    scratch = runroot / 'work' / delivery / 'attempts' / attempt_id
    evidence.mkdir(parents=True, exist_ok=False)
    scratch.mkdir(parents=True, exist_ok=False)
    doc = next(d for d in json.loads((repo / 'tests/fixtures/conversion-corpus/manifest.json').read_text())['documents'] if d['document_id'] == case)
    corpus = repo / 'tests/fixtures/conversion-corpus'
    if engine in {'marker','docling'}:
        payload=safety.regular(corpus,doc['input'])
        if c.descriptor(corpus/doc['input'])['sha256'] != doc['input_sha256'] or not payload.startswith(b'%PDF-'):
            raise ValueError('PDF identity/header mismatch')
        related={}
    else:
        payload, related = adapters.validate(corpus, doc, engine)
    if related:
        raise ValueError('Unexpected related inputs in selected native case')
    canonical = c.descriptor(corpus / doc['input'])
    for name in ['source', 'data', 'home', 'tmp', 'cache', 'config']:
        (scratch / name).mkdir()
    input_leaf = scratch / 'source' / ('input.' + doc['format'])
    input_leaf.write_bytes(payload)
    source_before = c.descriptor(input_leaf)
    py = Path(stage['base_python'])
    prefix = py.parent.parent
    diagnostic = repo / 'scripts/diagnostics/run-a001-fase0-uv'
    base = py.resolve().parent.parent
    installation = Path(stage['base_installation'])
    conf = stage['profile'] if engine in {'marker','docling'} else profile.config(doc['format'], engine)
    c.write_new(evidence / 'profile.json', conf)
    env = {'PATH': '/usr/bin', 'HOME': str(scratch/'home'), 'LANG': 'C.UTF-8', 'LC_ALL': 'C.UTF-8',
           'TMPDIR': str(scratch/'tmp'), 'XDG_CACHE_HOME': str(scratch/'cache'), 'XDG_CONFIG_HOME': str(scratch/'config'),
           'XDG_CONFIG_DIRS': str(scratch/'config'), 'PIP_CONFIG_FILE': '/dev/null', 'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONNOUSERSITE': '1'}
    if engine in {'marker','docling'}:
        env.update(stage['model_environment'])
        env.update(OMP_NUM_THREADS=str(conf['threads']),MKL_NUM_THREADS=str(conf['threads']),OPENBLAS_NUM_THREADS=str(conf['threads']),
                   HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1',TOKENIZERS_PARALLELISM='false')
        child={'input':str(input_leaf),'output':str(scratch),'engine':engine,'threads':conf['threads'],'options':conf['options'],
               'page_range':stage.get('page_range'),'benchmark_helpers':str(repo/'scripts/benchmarks'),'artifacts_path':stage.get('model_root')}
        c.write_new(scratch/'pdf-config.json',child)
        command=[str(py),'-I','-B',str(HERE/'pdf_child.py'),str(scratch/'pdf-config.json')]
    elif engine == 'txt-literal-v1':
        command = [str(py), '-I', '-B', str(repo/'scripts/benchmarks/literal_child.py'), str(input_leaf), str(scratch/'document.md')]
    elif engine == 'pandoc-direct-v1':
        command = ['/usr/bin/pandoc', str(input_leaf), *conf['ordered_options'], '--output=document.md']
    else:
        child = {'input': str(input_leaf), 'options': conf['ordered_options'], 'output': str(scratch/'document.md'),
                 'receipt': str(scratch/'legacy-result.json'), 'installation': str(installation),
                 'installation_sha256': c.descriptor(installation)['sha256'], 'pandoc': conf['pandoc'], 'timeout_s': 60,
                 'pandoc_resolution_before': pandoc_guard.check(env, conf['pandoc']), 'pandoc_version_required': conf['pandoc_version_required']}
        c.write_new(scratch/'legacy-config.json', child)
        command = [str(py), '-I', '-B', str(repo/'scripts/benchmarks/legacy_child.py'), str(scratch/'legacy-config.json')]
    denied = runroot/'work'/delivery/'synthetic-denied'
    positive = scratch/'positive.txt'
    positive.write_bytes(b'a003 sentinel\n')
    # Alternative path to a synthetic outside file, never to real secrets/gold.
    (scratch/'outside-link').symlink_to(denied/'gold')
    sentinel_info = (scratch/'outside-link').lstat()
    sentinel = {'identity':{'dev':sentinel_info.st_dev,'ino':sentinel_info.st_ino}, 'target':str(denied/'gold')}
    readonly_directories = [Path('/usr'), prefix, base, HERE, repo/'scripts/benchmarks', diagnostic,
                            *map(Path,stage.get('model_readonly',[]))]
    if engine == 'marker': readonly_directories.append(Path(stage['model_environment']['FONT_DIR']))
    consumer = {'argv': command, 'scratch': str(scratch), 'readable': [str(input_leaf), str(py), str(positive)],
                'denied_read': [*[str(p) for p in denied.iterdir()], str(scratch/'outside-link'), '/proc/1/root'+str(denied/'gold'), '/proc/self/fd/99'],
                'denied_write': ['/a003-write', *[str(p/'a003-write') for p in dict.fromkeys(readonly_directories)]],
                'readonly_existing': [str(input_leaf)],
                'network_helper': str(diagnostic/'check_runner.py'), 'host_netns': os.readlink('/proc/self/ns/net'),
                'host_pidns': os.readlink('/proc/self/ns/pid'),
                'probe_receipt': str(scratch/'probes.json'), 'pandoc': '/usr/bin/pandoc' if engine.startswith('pandoc-') else None,
                'pandoc_positive': str(positive)}
    target = json.loads(Path(stage['target']).read_text())
    consumer['runtime'] = {k:target[k] for k in ('python','prefix','base_prefix','python_version','dependencies','startup')}
    consumer['daemon_paths'] = [row['path'] for row in target['daemons']]
    consumer['cpu_affinity'] = sorted(os.sched_getaffinity(0))[:4]
    consumer['denied_read'].extend('/proc/'+str(os.getpid())+'/'+name for name in ('root','environ','fd/0'))
    c.write_new(scratch/'consumer.json', consumer)
    files = [HERE/'confinement.py', HERE/'consumer.py', HERE/'driver.py', scratch/'consumer.json', input_leaf,
             installation, diagnostic/'check_runner.py', repo/'scripts/benchmarks/resources.py']
    manifest = {'schema': 1, 'phase': 'conversion', 'scratch': str(scratch),
                'readonly': list(map(str, [Path('/usr'), base, prefix, HERE, repo/'scripts/benchmarks', diagnostic, installation,*map(Path,stage.get('model_readonly',[]))])),
                'readonly_leaf': list(map(str,[input_leaf, scratch/'consumer.json', *([scratch/'legacy-config.json'] if engine=='pandoc-legacy-v1' else []),*([scratch/'pdf-config.json'] if engine in {'marker','docling'} else [])])),
                'files': [c.descriptor(p) for p in files], 'argv': command, 'python': str(py), 'consumer': str(HERE/'consumer.py'),
                'consumer_config': str(scratch/'consumer.json'), 'environment': env, 'host_environment': stage['environment'],
                'monitor': str(repo/'scripts/benchmarks/resources.py'), 'process_output': str(evidence/'consumer-process'),
                'result': str(evidence/'consumer-result.json'), 'timeout_s': stage.get('timeout_s',120),
                'rss_limit_bytes':stage.get('rss_limit_bytes',6*1024**3)}
    manifest['staging'] = {str(p.relative_to(scratch)):c.descriptor(p) for p in
                           [scratch/'consumer.json', input_leaf, positive,
                            *([scratch/'legacy-config.json'] if engine=='pandoc-legacy-v1' else []),
                            *([scratch/'pdf-config.json'] if engine in {'marker','docling'} else [])]}
    c.write_new(evidence/'manifest.json', manifest)
    started = {'attempt_id': attempt_id, 'document_id': case, 'engine': engine, 'scope': 'pilot-page-range' if stage.get('page_range') else 'whole-document',
               'page_range':stage.get('page_range'),
               'started_at': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'canonical': canonical, 'source_copy': source_before,
               'configuration': c.descriptor(evidence/'profile.json'), 'manifest': c.descriptor(evidence/'manifest.json'),
               'runtime': c.descriptor(installation), 'driver': [c.descriptor(p) for p in HERE.glob('*.py')], 'argv': command, 'cwd': str(scratch), 'environment': env}
    c.write_new(evidence/'started.json', started)
    label = attempt_id.lower()
    snapshot_command = [str(py), '-I', '-B', str(repo/'scripts/run_context.py'), 'snapshot', runroot.name, '--label', label,
                        '--artifact', str((evidence/'manifest.json').relative_to(repo)), '--artifact', str((evidence/'started.json').relative_to(repo))]
    snapshot = runroot/'snapshots'/ (label+'.json')
    host_errors = []
    try:
        snap_receipt = run(snapshot_command, repo, evidence/'snapshot-process', stage['environment'], 120)
        c.write_new(evidence/'snapshot-process.json', snap_receipt)
        if snap_receipt['adapter_exit']:
            raise ValueError('Snapshot creation failed')
        preflight = {'schema': 1, 'commands': [[str(py), '-I', '-B', str(diagnostic/'check_preflight.py'), '--repo', str(repo), '--source', stage['source']['path'], '--receipt', str(installation)]]}
        c.write_new(evidence/'preflight.json', preflight)
        wrapper = [str(py), '-I', '-B', str(diagnostic/'run_offline.py'), '--runner', 'bwrap', '--python', str(py), '--repo', str(repo),
                   '--target', stage['target'], '--snapshot', str(snapshot), '--output', str(evidence/'R'), '--cwd', str(scratch),
                   '--preflight-config', str(evidence/'preflight.json'), '--timeout', str(stage.get('timeout_s',120)+30),
                   '--confinement', str(evidence/'manifest.json')]
        result = run(wrapper, repo, evidence/'wrapper-process', stage['environment'], budget,
                     rss_limit_bytes=stage.get('rss_limit_bytes',6*1024**3))
        c.write_new(evidence/'wrapper-process.json', result)
    except (OSError, ValueError, RuntimeError) as exc:
        result = {'adapter_exit':None,'children_reaped':False,'stop_reason':'HOST_RUN_ERROR'}
        host_errors.append({'file':'wrapper-process.json','phase':'host-run','error':repr(exc)})
    return finalize_attempt(evidence=evidence, scratch=scratch, safety=safety, attempt_id=attempt_id,
                            case=case, engine=engine, wrapper=result, snapshot=snapshot, sentinel=sentinel,
                            canonical=canonical, canonical_path=corpus/doc['input'], input_leaf=input_leaf,
                            payload=payload, host_errors=host_errors)


def finalize_attempt(*, evidence, scratch, safety, attempt_id, case, engine, wrapper, snapshot,
                     sentinel, canonical, canonical_path, input_leaf, payload, host_errors=()):
    """Post-run finito: errori e parziali persistiti anche quando audit/copia falliscono."""
    result = wrapper
    read_errors = list(host_errors)
    def output_json(root, name, default):
        try: return c.safe_json(root, name, limit=safety.MAX_FILE)
        except (OSError, ValueError, TypeError) as exc:
            read_errors.append({'file':name,'error':repr(exc)}); return default
    # Fonte R catturata dal runner host, mai un JSON selezionato dal motore.
    probes = output_json(evidence, 'R/admission.json', {'status':'UNAVAILABLE'})
    consumer_result = output_json(evidence, 'consumer-result.json', {'adapter_exit':None,'stop_reason':None})
    raw = scratch/'document.md'
    try: raw_data = c.safe_bytes(scratch,'document.md',safety.MAX_FILE)
    except (OSError,ValueError) as exc:
        raw_data = None; read_errors.append({'file':'document.md','error':repr(exc)})
    status = outcome(consumer_result, raw_data is not None, len(raw_data) if raw_data is not None else 0, probes['status']=='PASS')
    r_receipt = output_json(evidence, 'R/receipt.json', {'status':'UNAVAILABLE'})
    if r_receipt.get('stop_reason') == 'HOST_RUN_ERROR':
        read_errors.append({'file':'R/receipt.json','phase':'host-run',
                            'error':r_receipt.get('error','HOST_RUN_ERROR')})
    sentinel_replacement = None
    try:
        info = (scratch/'outside-link').lstat()
        if not stat.S_ISLNK(info.st_mode) or {'dev':info.st_dev,'ino':info.st_ino} != sentinel['identity'] or os.readlink(scratch/'outside-link') != sentinel['target']:
            sentinel_replacement = {'mode':info.st_mode,'bytes':info.st_size,'identity':{'dev':info.st_dev,'ino':info.st_ino},'preserved_in_place':True}
            if stat.S_ISREG(info.st_mode):
                data = c.safe_bytes(scratch,'outside-link',safety.MAX_FILE,expected_identity=sentinel_replacement['identity'])
                dest = evidence/'sentinel-replacement.bin'
                with dest.open('xb') as stream: stream.write(data)
                sentinel_replacement.update(reading_copy=str(dest),sha256=__import__('hashlib').sha256(data).hexdigest())
    except (OSError, ValueError) as exc:
        read_errors.append({'file':'outside-link','phase':'preserve-sentinel','error':repr(exc)})
    # Remove only the known synthetic symlink after confinement; never follow it.
    try: c.unlink_sentinel(scratch, sentinel)
    except (OSError, ValueError) as exc:
        read_errors.append({'file':'outside-link','phase':'cleanup','error':repr(exc)})
    try: artifacts = artifact_inventory(scratch,safety,read_errors)
    except (OSError,ValueError) as exc:
        artifacts = []; read_errors.append({'file':'artifact inventory','error':repr(exc)})
    # Copies for passive reading, raw originals remain unchanged.
    for row in artifacts:
        try:
            path=Path(row['relative_path'])
            if path.suffix.lower() in {'.md','.png','.jpg','.jpeg','.json'}:
                data=c.safe_bytes(scratch,str(path),safety.MAX_FILE,expected_identity=row['identity'])
                if __import__('hashlib').sha256(data).hexdigest()!=row['sha256']:raise ValueError('Artifact changed before copy')
                dest=evidence/'raw'/path;dest.parent.mkdir(parents=True,exist_ok=True)
                with dest.open('xb') as stream: stream.write(data)
                row['reading_copy'] = str(dest)
        except (OSError,ValueError) as exc:
            read_errors.append({'file':row['relative_path'],'phase':'copy','error':repr(exc)})
    try: integrity = canonical == c.descriptor(canonical_path) and c.safe_bytes(scratch,str(input_leaf.relative_to(scratch)),safety.MAX_FILE)==payload
    except (OSError,ValueError) as exc:
        integrity=False
        read_errors.append({'file':str(input_leaf),'phase':'integrity','error':repr(exc)})
    try: snapshot_identity = c.descriptor(snapshot)
    except (OSError,ValueError) as exc:
        snapshot_identity=None
        read_errors.append({'file':str(snapshot),'phase':'snapshot-identity','error':repr(exc)})
    status=final_status(status,result,r_receipt,integrity)
    if result.get('stop_reason') == 'HOST_RUN_ERROR' or r_receipt.get('stop_reason') == 'HOST_RUN_ERROR' or read_errors and status in {'SUCCESS','EMPTY'}:status='ERROR'
    final = {'attempt_id':attempt_id,'document_id':case,'engine':engine,'operational_status':status,
             'integrity_unchanged':integrity,'confinement_status':probes['status'],'R_status':r_receipt['status'],
             'engine_exit':engine_exit(consumer_result,engine,probes['status']=='PASS'),
             'wrapper_exit':result['adapter_exit'],'stop_reason':result.get('stop_reason') or r_receipt.get('stop_reason') or consumer_result.get('stop_reason'), 'signal':consumer_result.get('signal'),
             'runner_host_error':r_receipt.get('error') if r_receipt.get('stop_reason') == 'HOST_RUN_ERROR' else None,
             'runner_children_reaped':r_receipt.get('children_reaped'),
             'sentinel_replacement':sentinel_replacement,
             'raw':{'path':str(raw),'bytes':len(raw_data),'sha256':__import__('hashlib').sha256(raw_data).hexdigest()} if raw_data is not None else None,'artifacts':artifacts,'snapshot':snapshot_identity,'output_read_errors':read_errors,
             'raw_origin':'returned-text serialized; original temporary and internal exit not observable' if engine=='pandoc-legacy-v1' else 'engine-output-file',
             'human_assessment':'PENDING','fidelity':'INDETERMINATE' if status=='SUCCESS' else 'NOT_MEASURED' if status=='NOT_MEASURED' else 'UNAVAILABLE',
             'resources':str(evidence/'consumer-result.json'),'accounting_outer_includes_inner':True,'children_reaped':result['children_reaped']}
    try: c.write_new(evidence/'final.json', final)
    except (OSError,ValueError,TypeError) as exc:
        print(json.dumps({'attempt_id':attempt_id,'phase':'final-record-write','status':'ERROR',
                          'path':str(evidence/'final.json'),'error':repr(exc),
                          'evidence_preserved':str(evidence),'output_read_errors':read_errors}),file=sys.stderr,flush=True)
        raise
    return final


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',type=Path,required=True)
    parser.add_argument('--stage',type=Path,required=True)
    parser.add_argument('--case',required=True)
    parser.add_argument('--engine',choices=['txt-literal-v1','pandoc-direct-v1','pandoc-legacy-v1','marker','docling'],required=True)
    parser.add_argument('--attempt-id',required=True)
    args=parser.parse_args()
    import re
    if not re.fullmatch(r'[a-z0-9][a-z0-9-]{0,79}',args.attempt_id):
        parser.error('Attempt ID non valido')
    result=native(args.repo,json.loads(args.stage.read_text()),args.case,args.engine,args.attempt_id)
    print(json.dumps({k:result[k] for k in ['attempt_id','operational_status','R_status','confinement_status']}))
    return 0 if result['operational_status']=='SUCCESS' else 2


if __name__=='__main__':
    raise SystemExit(main())
