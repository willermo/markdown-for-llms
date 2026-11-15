"""Isolation probe of the light child, run inside it by `availability` and the tests.

It tries what the child must not be able to do and writes the outcomes to /scratch/probe.json:
a TCP connection to the loopback and a DNS lookup (no network namespace but its own empty
loopback), reading a host path given by the caller (not mounted), writing outside /scratch (a
read-only root), and the Docker socket (absent). Each entry is true when the attempt failed.
"""
import json
import os
from pathlib import Path
import socket


def _fails(action):
    try:
        action()
    except Exception:
        return True
    return False


def main(args):
    if args[:1] == ["allocate"]:          # a process that grows past a memory cap (tests of the scope)
        held = []
        for _ in range(int(args[1])):
            held.append(bytearray(1 << 20))
            held[-1][::4096] = b"x" * len(held[-1][::4096])
        return 0
    host_path = args[0] if args else "/home"

    def connect():
        s = socket.create_connection(("127.0.0.1", 9), timeout=1)
        s.close()

    def write_root():
        Path("/usr/mdf-probe").write_text("x")
    result = {
        "loopback_refused": _fails(connect),
        "dns_failed": _fails(lambda: socket.getaddrinfo("example.com", 443)),
        "host_path_hidden": not os.path.exists(host_path),
        "root_read_only": _fails(write_root),
        "docker_socket_absent": not os.path.exists("/var/run/docker.sock") and not os.path.exists("/run/docker.sock"),
        "scratch_writable": not _fails(lambda: Path("/scratch/probe-write").write_text("x")),
    }
    Path("/scratch/probe.json").write_text(json.dumps(result))
    return 0
