"""Measure the isolation boundary before admitting conversion work."""
from pathlib import Path
import os
import socket
import sys

from ..files import write_json


def main(args):
    interfaces = [line.split(':', 1)[0].strip() for line in Path('/proc/net/dev').read_text().splitlines()[2:]]
    attempts = []
    for address in [('198.51.100.1', 9), ('127.0.0.1', 9)]:
        try:
            with socket.socket() as sock:
                sock.settimeout(0.25)
                sock.connect(address)
            attempts.append({'address': address[0], 'connected': True})
        except OSError as exc:
            attempts.append({'address': address[0], 'connected': False, 'errno': exc.errno})
    readonly = False
    try:
        fd = os.open(Path(sys.prefix) / 'mdf-write-probe', os.O_CREAT | os.O_WRONLY | os.O_EXCL, 0o600)
    except OSError:
        readonly = True
    else:
        os.close(fd)
        os.unlink(Path(sys.prefix) / 'mdf-write-probe')
    result = dict(interfaces=interfaces, attempts=attempts, runtime_readonly=readonly,
                  netns=os.readlink('/proc/self/ns/net'), pidns=os.readlink('/proc/self/ns/pid'),
                  environment_keys=sorted(os.environ), host_sockets_visible=any(Path(p).exists() for p in ('/run/docker.sock', '/var/run/docker.sock', '/run/user')))
    result['pass'] = interfaces == ['lo'] and all(not x['connected'] for x in attempts) and readonly and not result['host_sockets_visible']
    write_json(Path('/scratch'), 'isolation.json', result)
    return 0 if result['pass'] else 2
