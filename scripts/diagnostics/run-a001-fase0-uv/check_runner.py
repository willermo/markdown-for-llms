#!/usr/bin/env python3
"""Preflight R stdlib: osserva l'isolamento, non lo configura né lo ripara.

Il target contiene soltanto path/hash pubblici della preparazione. Nessun import
applicativo, DNS, installazione o richiesta ai daemon. Le sonde IP avvengono solo
dopo aver verificato un namespace distinto, sola lo e nessuna rotta esterna.
"""

import argparse
import hashlib
import importlib.metadata
import ipaddress
import json
import os
from pathlib import Path
import socket
import struct
import subprocess
import sys
import tempfile
from datetime import datetime, timezone


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, value):
    path = Path(path)
    for part in (path, *path.parents):
        if part.is_symlink():
            raise ValueError(f"Output symlink: {part}")
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=True)
        stream.write("\n")


def attributes(data):
    result = {}
    while data:
        if len(data) < 4:
            raise ValueError("Attributo netlink troncato")
        size, kind = struct.unpack_from("=HH", data)
        if size < 4 or size > len(data):
            raise ValueError("Dimensione attributo netlink non valida")
        result[kind] = data[4:size]
        data = data[(size + 3) & ~3:]
    return result


def netlink_dump(kind, body):
    """RTNETLINK locale, sole richieste dump; non configura interfacce/rotte."""
    with socket.socket(socket.AF_NETLINK, socket.SOCK_RAW, socket.NETLINK_ROUTE) as sock:
        sock.settimeout(2)
        sock.bind((0, 0))
        sequence = 1
        sock.sendto(struct.pack("=IHHII", 16 + len(body), kind, 0x301, sequence, 0) + body, (0, 0))
        result = []
        for _ in range(128):
            data = sock.recv(65536)
            while data:
                if len(data) < 16:
                    raise ValueError("Header netlink troncato")
                size, reply, flags, seq, pid = struct.unpack_from("=IHHII", data)
                if size < 16 or size > len(data) or seq != sequence or flags & 0x10:
                    raise ValueError("Dump netlink invalido/interrotto")
                payload = data[16:size]
                if reply == 3:
                    if payload and struct.unpack_from("=i", payload)[0] != 0:
                        raise ValueError("Dump netlink fallito")
                    return result
                if reply == 2:
                    error = struct.unpack_from("=i", payload)[0]
                    if error:
                        raise OSError(-error, "netlink")
                else:
                    result.append((reply, payload))
                data = data[(size + 3) & ~3:]
        raise ValueError("Dump netlink incompleto")


def network_state():
    links, addresses, routes = [], [], []
    for kind, data in netlink_dump(18, struct.pack("=BBHiII", 0, 0, 0, 0, 0, 0)):
        if kind != 16:
            raise ValueError("Risposta link inattesa")
        family, pad, typ, index, flags, change = struct.unpack_from("=BBHiII", data)
        attr = attributes(data[16:])
        links.append({"index": index, "name": attr[3].rstrip(b"\0").decode(), "flags": flags, "up": bool(flags & 1)})
    for kind, data in netlink_dump(22, struct.pack("=BBBBI", 0, 0, 0, 0, 0)):
        if kind != 20:
            raise ValueError("Risposta address inattesa")
        family, prefix, flags, scope, index = struct.unpack_from("=BBBBI", data)
        attr = attributes(data[8:])
        if family not in (socket.AF_INET, socket.AF_INET6):
            raise ValueError("Famiglia address inattesa")
        for key in (1, 2):
            if key in attr:
                addresses.append({"index": index, "address": socket.inet_ntop(family, attr[key]), "prefix": prefix, "scope": scope})
    for kind, data in netlink_dump(26, struct.pack("=BBBBBBBBI", 0, 0, 0, 0, 0, 0, 0, 0, 0)):
        if kind != 24:
            raise ValueError("Risposta route inattesa")
        family, prefix, src_len, tos, table, protocol, scope, typ, flags = struct.unpack_from("=BBBBBBBBI", data)
        attr = attributes(data[12:])
        if family not in (socket.AF_INET, socket.AF_INET6):
            raise ValueError("Famiglia route inattesa")
        routes.append({"family": family, "prefix": prefix, "dst": socket.inet_ntop(family, attr[1]) if 1 in attr else None,
                       "gateway": socket.inet_ntop(family, attr[5]) if 5 in attr else None,
                       "oif": struct.unpack("=I", attr[4])[0] if 4 in attr else None,
                       "multipath": 9 in attr, "table": table, "type": typ})
    proc = {p: Path(p).read_text() for p in ("/proc/net/dev", "/proc/net/route", "/proc/net/ipv6_route", "/proc/net/if_inet6")}
    lo = {item["index"] for item in links if item["name"] == "lo"}
    safe = len(links) == 1 and len(lo) == 1
    safe = safe and all(item["index"] in lo and ipaddress.ip_address(item["address"]).is_loopback for item in addresses)
    for route in routes:
        safe = safe and not route["gateway"] and not route["multipath"] and route["oif"] in (lo | {None})
        # BLACKHOLE/UNREACHABLE/PROHIBIT non forniscono egress, anche con dst /0.
        if route["type"] in (6, 7, 8):
            continue
        safe = safe and route["oif"] in lo
        # Un prefisso deve essere interamente loopback, non solo iniziare da 127.
        if route["dst"] is None:
            safe = False
        else:
            network = ipaddress.ip_network(f'{route["dst"]}/{route["prefix"]}', strict=False)
            safe = safe and network.network_address.is_loopback and network.broadcast_address.is_loopback
    proc_names = {line.split(":", 1)[0].strip() for line in proc["/proc/net/dev"].splitlines()[2:] if ":" in line}
    safe = safe and proc_names == {"lo"}
    for line in proc["/proc/net/route"].splitlines()[1:]:
        fields = line.split()
        iface, dst, gateway, flags = fields[:4]
        address = ipaddress.IPv4Address(struct.pack("<I", int(dst, 16)))
        mask = ipaddress.IPv4Address(struct.pack("<I", int(fields[7], 16)))
        net = ipaddress.ip_network(f"{address}/{mask}", strict=False)
        safe = safe and iface == "lo" and int(gateway, 16) == 0
        if not int(flags, 16) & 0x200:
            safe = safe and net.network_address.is_loopback and net.broadcast_address.is_loopback
    for line in proc["/proc/net/ipv6_route"].splitlines():
        fields = line.split()
        net = ipaddress.ip_network(f"{ipaddress.IPv6Address(int(fields[0], 16))}/{int(fields[1], 16)}", strict=False)
        safe = safe and fields[-1] == "lo" and int(fields[4], 16) == 0
        if not int(fields[8], 16) & 0x200:
            safe = safe and net.network_address.is_loopback and net.broadcast_address.is_loopback
    return {"netns": os.readlink("/proc/self/ns/net"), "links": links, "addresses": addresses,
            "routes": routes, "proc": proc, "no_external_routes_or_addresses": bool(safe)}


def connect_only(family, address):
    result = {"family": int(family), "address": address, "connected": False}
    try:
        with socket.socket(family, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.25)
            sock.connect(address)
            result["connected"] = True
    except OSError as exc:
        result.update(errno=exc.errno, error=type(exc).__name__)
    return result


def prerequisites(target):
    checks = []
    for item in target["files"]:
        path = Path(item["path"])
        ok = path.is_file() and path.stat().st_size == item["bytes"] and digest(path) == item["sha256"]
        checks.append({"path": str(path), "matches": ok})
    for name in target.get("absent_paths", []):
        path = Path(name)
        checks.append({"path": name, "expected_absent": True, "matches": not path.exists() and not path.is_symlink()})
    for item in target["dependencies"]:
        try:
            version = importlib.metadata.version(item["name"])
        except importlib.metadata.PackageNotFoundError:
            version = None
        checks.append({"package": item["name"], "version": version, "matches": version == item["version"]})
    executable = str(Path(sys.executable).absolute())
    checks.append({"python": executable, "matches": executable == target["python"] and
                   str(Path(sys.prefix).resolve()) == target["prefix"] and
                   str(Path(sys.base_prefix).resolve()) == target["base_prefix"] and
                   sys.version.split()[0] == target["python_version"]})
    checks.append({'isolated_no_bytecode': True, 'matches': bool(sys.flags.isolated and sys.dont_write_bytecode)})
    if 'startup' in target:
        observed = []
        for directory in target['startup']['site_dirs']:
            for p in sorted(Path(directory).glob('*.pth')):
                observed.append({'path':str(p),'bytes':p.stat().st_size,'sha256':digest(p)})
        checks.append({'startup':observed, 'matches':observed == target['startup']['pth'] and
                       not any(k.startswith('COVERAGE_') for k in os.environ)})
    for directory in (target["tmpdir"], target["workspace"]):
        try:
            with tempfile.NamedTemporaryFile(prefix="a001-r-", dir=directory) as stream:
                stream.write(b"synthetic-runner-probe\n")
                stream.flush()
                matches = Path(stream.name).read_bytes() == b"synthetic-runner-probe\n"
            checks.append({"writable_directory": directory, "matches": matches})
        except OSError as exc:
            checks.append({"writable_directory": directory, "matches": False, "error": type(exc).__name__})
    return checks


def observe(target, host_netns, synthetic_socket):
    observation = {"pid": os.getpid(), "ppid": os.getppid(), "uid": os.getuid(), "gid": os.getgid(),
                   "cwd": str(Path.cwd()), "executable": sys.executable, "realpath": str(Path(sys.executable).resolve()),
                   "prefix": sys.prefix, "base_prefix": sys.base_prefix, "stdlib": os.__file__, "version": sys.version,
                   "status": "IMPEDITA", "internet_probes": [], "daemon_probes": []}
    try:
        network = network_state()
        observation["network"] = network
        isolated = network["netns"] != host_netns and network["no_external_routes_or_addresses"]
        # Anche nel caso di runner inefficace non si inviano sonde IP all'host.
        if isolated:
            observation["internet_probes"] = [connect_only(socket.AF_INET, ("192.0.2.1", 9)),
                                               connect_only(socket.AF_INET6, ("2001:db8::1", 9, 0, 0))]
        left, right = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
        with left, right:
            left.settimeout(1)
            right.settimeout(1)
            left.sendall(b"a001-local")
            observation["socketpair_positive"] = right.recv(10) == b"a001-local"
        for item in target["daemons"]:
            path = Path(item["path"])
            probe = {"path": str(path), "exists_on_target": item["exists"], "exists_inside": path.exists()}
            if item["exists"] or path.exists():
                probe.update(connect_only(socket.AF_UNIX, str(path)))
            else:
                probe.update(connected=False, absent=True, blacklist_demonstrated=False)
            observation["daemon_probes"].append(probe)
        observation["synthetic_probe"] = connect_only(socket.AF_UNIX, synthetic_socket)
        observation["prerequisites"] = prerequisites(target)
        observation["environment"] = {key: os.environ.get(key) for key in
                                       ("TMPDIR", "TIKTOKEN_CACHE_DIR", "PYENV_VERSION", "VIRTUAL_ENV", "PYTHONPATH", "PYTHONHOME")}
        environment_ok = (os.environ.get("TMPDIR") == target["tmpdir"] and
                          os.environ.get("TIKTOKEN_CACHE_DIR") == target["tokenizer_cache"] and
                          not any(key in os.environ for key in
                                  ("DOCKER_HOST", "DOCKER_CONTEXT", "CONTAINER_HOST", "CONTAINERD_ADDRESS", "PYTHONPATH", "PYTHONHOME")))
        observation["fds"] = sorted(os.readlink(path) for path in Path("/proc/self/fd").iterdir() if path.exists())
        fd_ok = not any(value.startswith("socket:") for value in observation["fds"])
        binary_checks = []
        if all(check["matches"] for check in observation["prerequisites"]):
            for name in ("pandoc", "git"):
                command = [target["binaries"][name], "--version"]
                proc = subprocess.run(command, capture_output=True, text=True, timeout=10, close_fds=True)
                binary_checks.append({"argv": command, "exit_code": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr})
        observation["binary_checks"] = binary_checks
        denied_errno = {1, 2, 13, 20, 111}  # EPERM, ENOENT, EACCES, ENOTDIR, ECONNREFUSED
        synthetic_denied = (not observation["synthetic_probe"]["connected"] and
                            observation["synthetic_probe"].get("errno") in denied_errno)
        daemon_denied = all(not probe["connected"] and (probe.get("absent") or probe.get("errno") in denied_errno)
                            for probe in observation["daemon_probes"])
        passed = (isolated and len(observation["internet_probes"]) == 2 and
                  all(not probe["connected"] for probe in observation["internet_probes"]) and
                  observation["socketpair_positive"] and synthetic_denied and daemon_denied and environment_ok and fd_ok and
                  all(check["matches"] for check in observation["prerequisites"]) and len(binary_checks) == 2 and
                  all(check["exit_code"] == 0 for check in binary_checks))
        observation["status"] = "PASS" if passed else "IMPEDITA"
    except Exception as exc:
        observation["error"] = f"{type(exc).__name__}: {exc}"
    return observation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--host-netns", required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--synthetic-socket", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--child", action="store_true")
    args = parser.parse_args()
    target = json.loads(args.target.read_text())
    if target.get("schema") != 1 or str(args.repo.resolve()) != target["repo"]:
        parser.error("Target/schema/repo incompatibile")
    parent = observe(target, args.host_netns, args.synthetic_socket)
    if args.child:
        print(json.dumps(parent, ensure_ascii=True))
        return 0 if parent["status"] == "PASS" else 2
    if args.output is None:
        parser.error("--output obbligatorio per il padre")
    argv = [sys.executable, "-I", "-B", str(Path(__file__).resolve()), "--target", str(args.target.resolve()),
            "--host-netns", args.host_netns, "--repo", str(args.repo.resolve()), "--synthetic-socket", args.synthetic_socket, "--child"]
    process = subprocess.run(argv, env=dict(os.environ), capture_output=True, text=True, close_fds=True, timeout=30)
    try:
        child = json.loads(process.stdout)
    except ValueError:
        child = {"status": "IMPEDITA", "stdout": process.stdout}
    same_namespace = parent.get("network", {}).get("netns") == child.get("network", {}).get("netns")
    passed = parent["status"] == child["status"] == "PASS" and same_namespace and process.returncode == 0
    result = {"schema": 1, "captured_at_utc": datetime.now(timezone.utc).isoformat(), "status": "PASS" if passed else "IMPEDITA",
              "host_netns": args.host_netns, "diagnostic_sha256": digest(__file__), "target_sha256": digest(args.target),
              "parent": parent, "child": child, "child_command": {"argv": argv, "exit_code": process.returncode, "stderr": process.stderr},
              "same_namespace": same_namespace, "limits": "Solo fixture fidate e canali inventariati; nessuna sandbox generale."}
    write_new(args.output, result)
    print(json.dumps({"status": result["status"], "receipt": str(args.output)}, ensure_ascii=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
