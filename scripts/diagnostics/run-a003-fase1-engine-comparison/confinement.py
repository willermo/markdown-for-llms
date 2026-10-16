"""Manifest e argv Bubblewrap per a003; monitor e runner restano quelli standard."""
import hashlib
import json
import os
import stat
import fcntl
from pathlib import Path

REPO = Path(__file__).absolute().parents[3]
RUN = REPO / 'temp/run-a003-fase1-engine-comparison'
DELIVERY = 'implementation-r007'
RECEIPT_LIMIT = 1024 * 1024
# ABI Linux verificata in linux/fcntl.h e linux/memfd.h; managed omette wrapper.
SEALS = 15
F_ADD_SEALS = 1033
F_GET_SEALS = 1034


def memfd_create():
    import ctypes
    libc = ctypes.CDLL(None, use_errno=True)
    libc.memfd_create.argtypes = [ctypes.c_char_p, ctypes.c_uint]
    libc.memfd_create.restype = ctypes.c_int
    fd = libc.memfd_create(b'a003-consumer-admission', 3)
    if fd < 0: raise OSError(ctypes.get_errno(), 'memfd_create')
    os.set_inheritable(fd, False)
    return fd


def safe_bytes(root, name, limit=RECEIPT_LIMIT, *, expected_identity=None):
    """Walk dirfd O_NOFOLLOW da /; verifica l'oggetto letto, mai il target link."""
    root = Path(root)
    relative = Path(name)
    if not root.is_absolute() or relative.is_absolute() or '..' in (*root.parts, *relative.parts):
        raise ValueError('Unsafe output path')
    components = [*root.parts[1:], *relative.parts]
    fd = os.open('/', os.O_RDONLY | os.O_DIRECTORY)
    try:
        for component in components[:-1]:
            next_fd = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd); fd = next_fd
        leaf = os.open(components[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=fd)
        try:
            before = os.fstat(leaf)
            if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or before.st_size > limit:
                raise ValueError('Output special/hardlink/oversize')
            if expected_identity is not None and expected_identity != {'dev':before.st_dev,'ino':before.st_ino}:
                raise ValueError('Output inode replaced after admission')
            chunks = []; size = 0
            while True:
                chunk = os.read(leaf, min(65536, limit + 1 - size))
                if not chunk: break
                chunks.append(chunk); size += len(chunk)
                if size > limit: raise ValueError('Output oversize')
            after = os.fstat(leaf)
            current = os.stat(components[-1], dir_fd=fd, follow_symlinks=False)
            identity = lambda s: (s.st_dev, s.st_ino, s.st_mode, s.st_nlink, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
            if identity(before) != identity(after) or identity(after) != identity(current) or size != after.st_size:
                raise ValueError('Output substituted during read')
            return b''.join(chunks)
        finally:
            os.close(leaf)
    finally:
        os.close(fd)


def safe_json(root, name, limit=RECEIPT_LIMIT):
    return json.loads(safe_bytes(root, name, limit))


def unlink_sentinel(root, expected):
    """Rimuove solo outside-link tramite dirfd no-follow, mai directory/target."""
    root = canonical_path(str(root))
    fd = os.open('/', os.O_RDONLY | os.O_DIRECTORY)
    try:
        for component in root.parts[1:]:
            child = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd); fd = child
        info = os.stat('outside-link', dir_fd=fd, follow_symlinks=False)
        if stat.S_ISDIR(info.st_mode):
            raise ValueError('Outside sentinel replaced with directory; preserved')
        identity = {'dev': info.st_dev, 'ino': info.st_ino}
        valid = stat.S_ISLNK(info.st_mode) and identity == expected['identity'] and os.readlink('outside-link', dir_fd=fd) == expected['target']
        if not valid:
            raise ValueError('Outside sentinel identity/type/target changed; replacement preserved')
        os.unlink('outside-link', dir_fd=fd)
    finally:
        os.close(fd)


def sealed_admission(fd):
    if fcntl.fcntl(fd, F_GET_SEALS) != SEALS:
        raise ValueError('Admission not sealed before exec')
    info = os.fstat(fd)
    if not 0 < info.st_size <= RECEIPT_LIMIT:
        raise ValueError('Admission missing/oversize')
    raw = os.pread(fd, info.st_size, 0)
    value = json.loads(raw)
    if not isinstance(value,dict) or value.get('phase') != 'consumer-before-exec' or value.get('status') not in {'PASS', 'FAIL'} or not isinstance(value.get('probes'),list) or not value['probes'] or not all(isinstance(p,dict) and type(p.get('passed')) is bool for p in value['probes']) or value['status'] == 'PASS' and not all(p['passed'] for p in value['probes']):
        raise ValueError('Invalid admission schema')
    return raw, value


def allowed_bind(p):
    """Allowlist locale finita; il chiamante non può dichiarare nuovi root."""
    fixed = {Path('/usr'), Path('/etc/fonts'), REPO/'.venv-python/cpython-3.12.13-linux-x86_64-gnu',
             REPO/'scripts/benchmarks', Path(__file__).absolute().parent,
             REPO/'scripts/diagnostics/run-a001-fase0-uv', Path('/home/davide/.local/bin/uv'),
             REPO/'temp/run-a001-fase0-uv/work/docker-cpu-r001/wheels',
             RUN/'work/implementation-r001/docling-preparation-r001/wheels',
             RUN/'work/implementation-r001/marker-models-r001',
             RUN/'work/implementation-r002/docling-models',
             RUN/'work/implementation-r002/setup-ml-r001/ebook-wheel',
             RUN/'work/implementation-r002/setup-ml-r001/dist-fix-r005-001',
             RUN/'work/implementation-r002/setup-ml-r001/dist-fix-r005-002',
             RUN/'work/implementation-r002/setup-ml-r001/dist-fix-r006-001',
             RUN/'work/implementation-r002/setup-ml-r001/dist-fix-r006-002',
             RUN/'work/implementation-r002/setup-ml-r001/dist-fix-r007-001',
             RUN/'work/implementation-r002/setup-ml-r001/dist-fix-r007-002'}
    if p in fixed: return True
    prefixes = [RUN/'work/implementation-r001/setup-base-dev-r008', RUN/'work/implementation-r002/setup-ml-r001']
    if any(p == parent / role for parent in prefixes for role in ('base', 'dev', 'marker', 'docling')): return True
    # Un singolo receipt I, non l'intera evidence/preflight o il corpus.
    return (p.is_file() and p.name.startswith('I-') and p.suffix == '.json'
            and any(p.is_relative_to(RUN/'evidence'/version) for version in ('implementation-r001', 'implementation-r002', 'implementation-r003', 'implementation-r004', 'implementation-r005', 'implementation-r006', DELIVERY)))


def descriptor(path):
    path = Path(path)
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(65536), b''):
            h.update(chunk)
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': h.hexdigest()}


def write_new(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=True)
        stream.write('\n')


def canonical_path(value):
    p = Path(value)
    if not isinstance(value, str) or str(p) != value or not p.is_absolute() or '..' in p.parts or any(q.is_symlink() for q in (p, *p.parents)):
        raise ValueError('Path assoluto canonico richiesto: ' + str(p))
    return p


def writable_scratch(manifest):
    """Policy finita run/delivery/attempt, oppure setup espressamente disposto."""
    scratch = canonical_path(manifest['scratch'])
    if not scratch.is_dir():
        raise ValueError('Scratch esclusivo richiesto')
    if manifest['phase'] == 'conversion':
        import re
        # Il nome del leaf è autonomo; i parent e la relazione non sono liberi.
        parents = {RUN/'work'/DELIVERY/'attempts'}
        if scratch.parent not in parents or not re.fullmatch(r'[a-z0-9][a-z0-9-]{0,79}', scratch.name):
            raise ValueError('RW requires one run/delivery/attempt leaf')
        delivery = scratch.parent.parent.name
        evidence = RUN/'evidence'/delivery/'attempts'/scratch.name
        if canonical_path(manifest.get('process_output', '')) != evidence/'consumer-process' or canonical_path(manifest.get('result', '')) != evidence/'consumer-result.json' or not evidence.is_dir():
            raise ValueError('RW attempt/evidence relation absent or different')
        if canonical_path(manifest['consumer_config']) != scratch/'consumer.json':
            raise ValueError('Consumer config outside attempt leaf')
        new_attempt_staging(manifest, scratch, evidence)
    else:
        # Destinazioni reali E04 e disposizioni successive, non root di run/temp.
        approved = {RUN/'work/implementation-r001/setup-base-dev-r008',
                    RUN/'work/implementation-r002/setup-ml-r001'}
        if scratch not in approved:
            raise ValueError('RW setup destination not authorized')
    return scratch


def new_attempt_staging(manifest, scratch, evidence):
    """Solo staging finito prima del boundary; il manifest non dispone nuovi leaf."""
    allowed_evidence = {'manifest.json', 'profile.json', 'started.json', 'preflight.json',
                        'snapshot-process', 'snapshot-process.json', 'wrapper-process', 'R'}
    # Il runner crea R prima del preflight. Un R già usato non è nuovo.
    for entry in evidence.iterdir():
        if entry.name not in allowed_evidence and not entry.name.startswith(('snapshot-process-monitor-', 'wrapper-process-monitor-')):
            raise ValueError('RW evidence belongs to completed/used attempt')
        if entry.is_symlink():
            raise ValueError('RW evidence staging link')
        if entry.name == 'R' and (not entry.is_dir() or any(entry.iterdir())):
            raise ValueError('RW runner evidence already used')
    permitted_dirs = {'source', 'data', 'home', 'tmp', 'cache', 'config'}
    permitted_files = {'consumer.json', 'positive.txt', 'legacy-config.json', 'pdf-config.json'}
    actual = {}
    for entry in scratch.iterdir():
        info = entry.lstat()
        if entry.name == 'outside-link' and stat.S_ISLNK(info.st_mode):
            if os.readlink(entry) != str(RUN/'work'/DELIVERY/'synthetic-denied/gold'):
                raise ValueError('RW staging sentinel target')
            continue
        if entry.name in permitted_dirs and stat.S_ISDIR(info.st_mode):
            children = list(entry.iterdir())
            if entry.name != 'source' and children:
                raise ValueError('RW scratch already used/nonempty staging directory')
            for child in children:
                if child.name not in {'input.txt', 'input.docx', 'input.epub', 'input.pdf'}:
                    raise ValueError('RW unsupported source staging')
                actual[str(child.relative_to(scratch))] = child
            if len(children) > 1:
                raise ValueError('RW requires one source leaf')
        elif entry.name in permitted_files and stat.S_ISREG(info.st_mode):
            actual[entry.name] = entry
        else:
            raise ValueError('RW scratch belongs to completed/used attempt or unsupported staging')
    staging = manifest.get('staging')
    if not isinstance(staging, dict) or set(staging) != set(actual) or 'consumer.json' not in actual:
        raise ValueError('RW declared finite staging absent/different')
    for name, path in actual.items():
        row = staging[name]
        info = path.lstat()
        data = safe_bytes(scratch, name, 32*1024*1024, expected_identity={'dev':info.st_dev,'ino':info.st_ino})
        if row != {'path':str(path),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}:
            raise ValueError('RW staging identity STALE')
    required_ro = {str(path) for name, path in actual.items() if name != 'positive.txt'}
    if not required_ro <= set(manifest.get('readonly_leaf', [])):
        raise ValueError('RW staging inputs/configs require readonly overlays')


def validate(manifest):
    if manifest.get('schema') != 1 or manifest.get('phase') not in {'setup', 'conversion'}:
        raise ValueError('Schema/fase confinement')
    scratch = writable_scratch(manifest)
    if manifest['phase'] == 'setup':
        # I due scratch storici sono container di prefix, non singole venv.
        # Nuove scritture solo a prefix/temporanei/output r006 identificati.
        writes = manifest.get('setup_writable', [])
        if not writes:
            raise ValueError('Setup requires explicit writable destinations')
        permitted = {'base', 'dev', 'marker', 'docling', 'setup-r007-002', 'dist-fix-r007-002'}
        for value in writes:
            path = canonical_path(value)
            if path.parent != scratch or path.name not in permitted or not path.is_dir():
                raise ValueError('Setup writable destination not disposed')
    seen = set()
    for value in manifest['readonly']:
        p = canonical_path(value)
        if not p.exists() or p in seen or scratch.is_relative_to(p) or not allowed_bind(p):
            raise ValueError('Bind ambiguo o troppo ampio')
        seen.add(p)
    for row in manifest['files']:
        if descriptor(canonical_path(row['path'])) != row:
            raise ValueError('Confinement input STALE')
    if not manifest['argv'] or not Path(manifest['argv'][0]).is_absolute():
        raise ValueError('argv assoluto richiesto')
    env = manifest['environment']
    if any(not isinstance(k, str) or not isinstance(v, str) for k, v in env.items()):
        raise ValueError('Environment non testuale')
    if any(k in env for k in ['PYTHONPATH', 'LD_PRELOAD', 'COVERAGE_PROCESS_START', 'COVERAGE_PROCESS_CONFIG']):
        raise ValueError('Startup environment non chiuso')
    if any(k.startswith('A003_') for k in env):
        raise ValueError('Admission environment reserved to runner')
    keys = {'HOME', 'TMPDIR', 'TMP', 'TEMP', 'XDG_CACHE_HOME', 'XDG_CONFIG_HOME', 'XDG_CONFIG_DIRS', 'UV_CACHE_DIR',
            'TORCH_HOME', 'HF_HOME', 'HUGGINGFACE_HUB_CACHE', 'TRANSFORMERS_CACHE'}
    for key in keys & env.keys():
        if key == 'XDG_CONFIG_DIRS' and ':' in env[key]:
            raise ValueError('One scratch XDG_CONFIG_DIRS directory required')
        if not canonical_path(env[key]).is_relative_to(scratch):
            raise ValueError('Environment fuori scratch')
    if not {'HOME', 'TMPDIR', 'XDG_CACHE_HOME', 'XDG_CONFIG_HOME'} <= env.keys():
        raise ValueError('Scratch environment absent')
    for value in manifest.get('readonly_leaf', []):
        p = canonical_path(value)
        if not p.is_file() or not p.is_relative_to(scratch):
            raise ValueError('Leaf non confinato')
    return scratch


def argv(manifest, *, netns=None, synthetic_socket=None, admission_fd=None):
    scratch = validate(manifest)
    command = ['/usr/bin/bwrap', '--unshare-user', '--unshare-pid', '--unshare-ipc',
               '--unshare-uts', '--cap-drop', 'ALL', '--die-with-parent', '--new-session']
    # Un solo confine: procfs nasce nel nuovo PID namespace, senza Firejail esterno.
    command.append('--unshare-net')
    command += ['--ro-bind' if manifest['phase'] == 'setup' else '--bind', str(scratch), str(scratch)]
    if manifest['phase'] == 'setup':
        for path in manifest['setup_writable']:
            command += ['--bind', path, path]
    for path in manifest['readonly']:
        command += ['--ro-bind', path, path]
    # Leaf immutable overlays writable legacy parent.
    for path in manifest.get('readonly_leaf', []):
        p = canonical_path(path)
        if not p.is_file() or not p.is_relative_to(scratch):
            raise ValueError('Leaf non confinato')
        command += ['--ro-bind', str(p), str(p)]
    command += ['--symlink', 'usr/bin', '/bin', '--symlink', 'usr/lib', '/lib',
                '--symlink', 'usr/lib64', '/lib64', '--symlink', str(scratch / 'tmp'), '/tmp',
                '--proc', '/proc', '--remount-ro', '/proc', '--dev', '/dev', '--clearenv']
    environment = dict(manifest['environment'])
    if netns:
        environment['A003_HOST_NETNS'] = netns
    if synthetic_socket:
        environment['A003_SYNTHETIC_SOCKET'] = synthetic_socket
    if admission_fd is not None:
        environment['A003_ADMISSION_FD'] = str(admission_fd)
    for key, value in sorted(environment.items()):
        command += ['--setenv', key, value]
    command += ['--chdir', str(scratch), '--remount-ro', '/', '--',
                manifest['python'], '-I', '-B', manifest['consumer'], manifest['consumer_config']]
    return command


def main():
    import sys
    import importlib.util
    manifest = json.loads(Path(sys.argv[1]).read_text())
    netns = os.readlink('/proc/self/ns/net')
    command = argv(manifest, netns=netns if manifest['phase'] == 'conversion' else None)
    spec = importlib.util.spec_from_file_location('a003_resources', manifest['monitor'])
    monitor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(monitor)
    result = monitor.run(command, manifest['scratch'], Path(manifest['process_output']),
                         manifest['host_environment'], manifest['timeout_s'])
    write_new(manifest['result'], result)
    return 0 if result['adapter_exit'] == 0 and not result['stop_reason'] and result['children_reaped'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
