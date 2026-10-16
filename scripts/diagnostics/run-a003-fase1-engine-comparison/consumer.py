"""Sonde nello stesso figlio/env/mount/PID/netns prima del comando effettivo."""
import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import errno
import fcntl


def denied_errno(exc, *, absence=False):
    return exc.errno in ({errno.ENOENT, errno.EACCES, errno.EPERM, errno.ELOOP} if absence else {errno.EROFS, errno.EACCES, errno.EPERM})


def main():
    cfg = json.loads(Path(sys.argv[1]).read_text())
    probes = []
    def record(name, ok, **details):
        probes.append(dict(name=name, passed=bool(ok), **details))
    if cfg.get('cpu_affinity'):
        os.sched_setaffinity(0,cfg['cpu_affinity'])
        record('cpu-affinity', sorted(os.sched_getaffinity(0)) == cfg['cpu_affinity'], cpus=cfg['cpu_affinity'])
    for name in cfg['readable']:
        try:
            with open(name, 'rb') as stream:
                content = stream.read(64)
            record('read:' + name, bool(content), bytes_read=len(content))
        except OSError as exc:
            record('read:' + name, False, error=repr(exc))
    for name in cfg['denied_read']:
        try:
            with open(name, 'rb') as stream:
                stream.read(1)
            record('denied-read:' + name, False)
        except OSError as exc:
            record('denied-read:' + name, denied_errno(exc, absence=True), errno=exc.errno, error=repr(exc))
    for name in cfg['denied_write']:
        try:
            # Only synthetic sentinels: never truncate a real runtime/input.
            fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            os.close(fd)
            record('denied-write:' + name, False)
            os.unlink(name)
        except OSError as exc:
            record('denied-write:' + name, denied_errno(exc), errno=exc.errno, error=repr(exc))
    for name in cfg.get('readonly_existing', []):
        try:
            fd = os.open(name, os.O_WRONLY)
            os.close(fd)
            record('readonly-open:' + name, False)
        except OSError as exc:
            record('readonly-open:' + name, denied_errno(exc), errno=exc.errno, error=repr(exc))
    scratch = Path(cfg['scratch']) / 'positive-probe'
    scratch.write_bytes(b'a003-synthetic')
    record('scratch-roundtrip', scratch.read_bytes() == b'a003-synthetic')
    scratch.unlink()
    namespaces = {name: os.readlink('/proc/self/ns/' + name) for name in ('net','pid','mnt','user')}
    proc_pids = [p.name for p in Path('/proc').iterdir() if p.name.isdecimal()]
    record('pid-namespace', namespaces['pid'] != cfg.get('host_pidns') and os.getpid() < 100,
           pid=os.getpid(), proc_pids=proc_pids, namespaces=namespaces)
    fd_paths = list(Path('/proc/self/fd').iterdir())
    fds = {p.name: os.readlink(p) for p in fd_paths if p.exists()}
    admission_fd = int(os.environ['A003_ADMISSION_FD']) if 'A003_ADMISSION_FD' in os.environ else None
    allowed = {0,1,2} | ({admission_fd} if admission_fd is not None else set())
    record('closed-fds', all(int(name) in allowed for name in fds), fds=fds,
           admission_fd=admission_fd, phase='consumer before seal; sole extra FD is empty host memfd')
    if admission_fd is not None:
        record('admission-fd', os.fstat(admission_fd).st_size == 0 and fcntl.fcntl(admission_fd,1034) == 0 and fds.get(str(admission_fd),'').startswith('/memfd:a003-consumer-admission'), fd=admission_fd)
    spec = importlib.util.spec_from_file_location('a003_netprobe', cfg['network_helper'])
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    network = helper.network_state()
    record('no-egress', network['no_external_routes_or_addresses'], network=network)
    host_netns = os.environ.get('A003_HOST_NETNS', cfg['host_netns'])
    record('private-netns', network['netns'] != host_netns, host=host_netns, consumer=network['netns'])
    if cfg.get('runtime'):
        from importlib import metadata
        runtime = cfg['runtime']
        checks = [sys.executable == runtime['python'], str(Path(sys.prefix).resolve()) == runtime['prefix'],
                  str(Path(sys.base_prefix).resolve()) == runtime['base_prefix'], sys.version.split()[0] == runtime['python_version'],
                  bool(sys.flags.isolated and sys.dont_write_bytecode)]
        for item in runtime['dependencies']:
            checks.append(metadata.version(item['name']) == item['version'])
        actual = [helper_record(p) for directory in runtime['startup']['site_dirs'] for p in sorted(Path(directory).glob('*.pth'))]
        checks.append(actual == runtime['startup']['pth'])
        record('consumer-runtime-startup', all(checks), checks=checks, startup=actual)
    left, right = socket.socketpair()
    with left, right:
        left.send(b'x')
        record('socketpair', right.recv(1) == b'x')
    if network['no_external_routes_or_addresses']:
        for family, address in [(socket.AF_INET, ('192.0.2.1', 9)), (socket.AF_INET6, ('2001:db8::1', 9, 0, 0))]:
            response = helper.connect_only(family, address)
            record('IP-denied', not response['connected'] and response.get('errno') in {errno.ENETUNREACH, errno.EHOSTUNREACH, errno.EAFNOSUPPORT, errno.EACCES, errno.EPERM}, response=response)
    # File sentinella lunghi non sono socket: ENAMETOOLONG non prova isolamento.
    sockets = list(cfg.get('daemon_paths', []))
    if os.environ.get('A003_SYNTHETIC_SOCKET'):
        sockets.append(os.environ['A003_SYNTHETIC_SOCKET'])
    elif cfg.get('runtime'):
        record('synthetic-socket-present', False)
    for name in sockets:
        response = helper.connect_only(socket.AF_UNIX, name)
        record('unix-denied:' + name, not response['connected'] and response.get('errno') in {errno.ENOENT, errno.EACCES, errno.EPERM}, response=response)
    if cfg.get('pandoc'):
        for name, positive in [(cfg['pandoc_positive'], True), *[(p, False) for p in cfg['denied_read']]]:
            command = [cfg['pandoc'], '-f', 'markdown', '-t', 'html', '-s', '--include-in-header=' + name]
            child = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, close_fds=True)
            try:
                child_ns = {key: os.readlink('/proc/' + str(child.pid) + '/ns/' + key) for key in namespaces}
                record('Pandoc-namespaces:' + name, child_ns == namespaces, namespaces=child_ns)
                out, err = child.communicate(b'a003 synthetic\n', timeout=20)
            finally:
                if child.poll() is None:
                    child.kill()
                    child.wait()
            record('Pandoc-resource:' + name, (child.returncode == 0 and b'a003 sentinel' in out) if positive else child.returncode != 0,
                   argv=command, exit=child.returncode, stdout=out.decode(errors='replace'), stderr=err.decode(errors='replace'))
    passed = bool(probes) and all(p['passed'] for p in probes)
    value = {'status': 'PASS' if passed else 'FAIL', 'probes': probes, 'namespaces': namespaces,
             'argv_after_probes': cfg['argv'], 'environment': dict(os.environ), 'phase':'consumer-before-exec'}
    with Path(cfg['probe_receipt']).open('x') as stream:
        info = os.fstat(stream.fileno())
        value['copy_identity'] = {'dev':info.st_dev,'ino':info.st_ino}
        data = json.dumps(value, indent=2).encode('utf-8')
        stream.write(data.decode('utf-8'))
    if cfg.get('runtime') and admission_fd is None:
        return 3
    if admission_fd is not None:
        fd = admission_fd
        try:
            if os.fstat(fd).st_size or len(data) > 1024 * 1024:
                raise ValueError('Admission not empty/oversize')
            view = memoryview(data)
            while view:
                written = os.write(fd, view); view = view[written:]
            seals = 15
            fcntl.fcntl(fd, 1033, seals)
            if fcntl.fcntl(fd, 1034) != seals:
                raise ValueError('Admission seal FAIL')
        finally:
            os.close(fd)
        os.environ.pop('A003_ADMISSION_FD', None)
    if not passed:
        return 3
    # exec retains interpreter identity/env/namespaces and the monitor's tree.
    os.execve(cfg['argv'][0], cfg['argv'], dict(os.environ))


def helper_record(path):
    import hashlib
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


if __name__ == '__main__':
    raise SystemExit(main())
