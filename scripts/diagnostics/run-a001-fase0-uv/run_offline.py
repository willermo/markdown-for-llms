#!/usr/bin/env python3
"""R, preflight e comando in una sola invocazione non interattiva del runner.

Non prepara dipendenze e non crea snapshot. Richiede uno snapshot identificato.
La receipt distingue il processo del comando e i discendenti osservati; quelli
troppo brevi per /proc richiedono anche le receipt del relativo harness.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
import resource
import re
import signal
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
import time


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=True)
        stream.write("\n")


def clean_env(target):
    # Ambiente pubblico dichiarato dal bundle, mai un merge con os.environ.
    env = target['environment']
    allowed = {'PATH', 'HOME', 'LANG', 'LC_ALL', 'TMPDIR', 'TIKTOKEN_CACHE_DIR',
               'PYTEST_DISABLE_PLUGIN_AUTOLOAD', 'PYTHONDONTWRITEBYTECODE', 'PYTHONNOUSERSITE',
               'PIP_CONFIG_FILE', 'TZ', 'UV_CACHE_DIR', 'UV_PYTHON_DOWNLOADS',
               'XDG_CACHE_HOME', 'XDG_CONFIG_HOME', 'XDG_CONFIG_DIRS'}
    if (not isinstance(env, dict) or set(env) != allowed or
            any(not isinstance(v, str) for v in env.values()) or
            env['TMPDIR'] != target['tmpdir'] or env['TIKTOKEN_CACHE_DIR'] != target['tokenizer_cache'] or
            env['PYTEST_DISABLE_PLUGIN_AUTOLOAD'] != '1' or env['PYTHONDONTWRITEBYTECODE'] != '1'):
        raise ValueError('Environment pubblico chiuso mancante/difforme')
    workload = target.get('workload_environment', {})
    public_inputs = {'RUN_SOURCE_MANIFEST','RUN_INSTALLATION_RECEIPT','RUN_WHEEL_PY',
                     'RUN_WHEEL_RECEIPT','RUN_BASELINE','RUN_BASELINE_WORKSPACES','RUN_IMAGE',
                     'RUN_IMAGE_MANIFEST','RUN_TEST_OUTPUT'}
    if not isinstance(workload, dict) or not workload.keys() <= public_inputs or any(not isinstance(v,str) for v in workload.values()):
        raise ValueError('Environment delle prove non ammesso')
    return dict(env, **workload)


def binding_check(args):
    if args.snapshot:
        return snapshot_check(args.repo, args.snapshot, args.python)
    data = json.loads(args.bundle.read_text())
    if data.get('schema') != 1 or data.get('scope') != 'preliminary-inputs' or not data.get('files'):
        raise ValueError('Bundle preliminare non valido')
    paths = set()
    for record in data['files']:
        path = Path(record['path'])
        if (path.is_absolute() or '..' in path.parts or record['path'] in paths or
                any(p.is_symlink() for p in (args.repo/path, *(args.repo/path).parents))):
            raise ValueError('Path bundle non valido')
        paths.add(record['path'])
        p = args.repo/path
        if p.stat().st_size != record['bytes'] or digest(p) != record['sha256']:
            raise ValueError('Input bundle cambiato: '+str(path))
    required = {str(Path(__file__).relative_to(args.repo)),
                str(Path(__file__).with_name('check_runner.py').relative_to(args.repo)),
                str(args.target.relative_to(args.repo))}
    if not required <= paths:
        raise ValueError('Wrapper/diagnostico/target non legati al bundle')
    return {'scope': 'preliminary-inputs', 'path':str(args.bundle), 'sha256':digest(args.bundle),
            'entry_context': data.get('entry_context'), 'files':len(paths)}


def target_files_check(target):
    """Legge startup/prerequisiti prima di avviare l'interprete target."""
    monitor=Path(__file__).resolve().parents[2]/'benchmarks/resources.py'
    if not any(item['path']==str(monitor) for item in target['files']):
        raise ValueError('Monitor dei figli non legato al target')
    for item in target['files']:
        p = Path(item['path'])
        if not p.is_file() or p.stat().st_size != item['bytes'] or digest(p) != item['sha256']:
            raise ValueError('Prerequisito target alterato prima del lancio: '+str(p))
    startup = target.get('startup')
    if startup:
        actual = [str(p) for directory in startup['site_dirs'] for p in sorted(Path(directory).glob('*.pth'))]
        if actual != [item['path'] for item in startup['pth']]:
            raise ValueError('Startup .pth non chiuso')


def snapshot_check(repo, snapshot, python):
    # Stesso contratto del produttore S; controlli locali prima del verificatore.
    if (not repo.is_absolute() or not snapshot.is_absolute() or
            '..' in repo.parts or '..' in snapshot.parts or
            any(p.is_symlink() for p in (snapshot, *snapshot.parents, repo, *repo.parents))):
        raise ValueError('Path snapshot/clone assoluto senza symlink richiesto')
    data = json.loads(snapshot.read_text())
    required = {'schema', 'run_id', 'label', 'head', 'branch', 'files', 'artifacts', 'worktree_sha256'}
    if (not isinstance(data, dict) or not required <= data.keys() or
            type(data['schema']) is not int or data['schema'] != 1):
        raise ValueError("Snapshot/schema non valido")
    if (not isinstance(data['run_id'], str) or
            re.fullmatch(r'run-[a-z][0-9]{3,}-fase[a-z0-9]+-[a-z0-9]+(?:-[a-z0-9]+)*', data['run_id']) is None or
            not isinstance(data['label'], str) or
            re.fullmatch(r'[a-z0-9][a-z0-9-]{0,79}', data['label']) is None):
        raise ValueError('Run ID/label snapshot non validi')
    expected = repo / "temp" / data["run_id"] / "snapshots" / (data["label"] + ".json")
    if expected != snapshot or snapshot.is_symlink():
        raise ValueError("Snapshot non appartenente al clone/stage")
    argv = [python, "-I", "-B", str(repo / "scripts/run_context.py"), "verify", data["run_id"], "--label", data["label"]]
    proc = subprocess.run(argv, cwd=repo, capture_output=True, text=True, close_fds=True, timeout=60)
    result = {"argv": argv, "exit_code": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr,
              "path": str(snapshot), "sha256": digest(snapshot), "worktree_sha256": data["worktree_sha256"], "label": data["label"]}
    if proc.returncode != 0:
        raise ValueError(f'Stage STALE: {proc.stdout.strip()}')
    return result


def monitored_command(argv, cwd, output, timeout):
    # Trusted stdlib-only resource helper, bound by the target and snapshot.
    path=Path(__file__).resolve().parents[2]/'benchmarks/resources.py'
    spec=importlib.util.spec_from_file_location('a002_resources_monitor',path)
    monitor=importlib.util.module_from_spec(spec);spec.loader.exec_module(monitor)
    # R's output already exists. The helper writes into a fresh child directory.
    temporary=output/'command-monitor'
    ns=os.readlink('/proc/self/ns/net')
    receipt=monitor.run(argv,cwd,temporary,dict(os.environ),timeout,
                        stdout_name='command-stdout.txt',stderr_name='command-stderr.txt',expected_netns=ns)
    for name in ('command-stdout.txt','command-stderr.txt'):
        (temporary/name).rename(output/name)
    write_new(output/'command-resources.json',receipt)
    observed=receipt['observed_processes']
    return {'argv':argv,'cwd':str(cwd),'pid':receipt['pid'],'exit_code':receipt['adapter_exit'],
            'timed_out':bool(receipt['stop_reason']),'deadline_exceeded':receipt['timeout'],'guard_reason':receipt['stop_reason'],
            'duration_seconds':receipt['conversion_elapsed_s'],'children_reaped':receipt['children_reaped'],
            'observed_processes':observed,'command_namespace_observed':any(i['pid']==receipt['pid'] and i['netns']==ns for i in observed),
            'observed_namespaces_match':all(i['netns'] in (None,ns) for i in observed),
            'resources':'command-resources.json','stdout':'command-stdout.txt','stderr':'command-stderr.txt'}


def inside(args, target):
    output = args.output
    argv = [args.python, "-I", "-B", str(Path(__file__).with_name("check_runner.py")), "--target", str(args.target),
            "--host-netns", args.host_netns, "--repo", str(args.repo), "--synthetic-socket", args.synthetic_socket,
            "--output", str(output / "runner.json")]
    proc = subprocess.run(argv, capture_output=True, text=True, close_fds=True, timeout=60)
    result = {"schema": 1, "netns": os.readlink("/proc/self/ns/net"), "runner": {"argv": argv, "exit_code": proc.returncode,
              "stdout": proc.stdout, "stderr": proc.stderr}, "preflights": [], "status": "IMPEDITA"}
    if proc.returncode != 0:
        write_new(output / "inside.json", result)
        return 2
    # Ogni preflight è argv esplicito, nel medesimo namespace; nessuna shell.
    if args.preflight_config:
        conf = json.loads(args.preflight_config.read_text())
        if conf.get("schema") != 1 or not isinstance(conf.get("commands"), list):
            raise ValueError("Configurazione preflight non valida")
        for command in conf["commands"]:
            if not command or not all(isinstance(item, str) for item in command) or not Path(command[0]).is_absolute():
                raise ValueError("argv preflight non fidato")
            proc = subprocess.run(command, cwd=args.cwd, capture_output=True, text=True, close_fds=True, timeout=60)
            result["preflights"].append({"argv": command, "exit_code": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr})
            if proc.returncode != 0:
                result["status"] = "FAIL"
                write_new(output / "inside.json", result)
                return 2
    if args.command:
        command = args.command[1:] if args.command[0] == "--" else args.command
        if not command or not Path(command[0]).is_absolute():
            raise ValueError("Il comando deve iniziare con un eseguibile assoluto")
        try:
            result["command"] = monitored_command(command, args.cwd, output, args.timeout)
        except (OSError,ValueError,RuntimeError) as exc:
            result.update(status='FAIL',error=f'{type(exc).__name__}: {exc}')
            write_new(output/'inside.json',result)
            return 2
        item = result["command"]
        passed = (item["exit_code"] == args.expected_exit and not item["timed_out"] and not item["guard_reason"] and item["children_reaped"] and
                  item["command_namespace_observed"] and item["observed_namespaces_match"])
        result["status"] = "PASS" if passed else "FAIL"
    else:
        result["status"] = "PASS"
        result["scope"] = "solo R; nessuna prova applicativa"
    write_new(output / "inside.json", result)
    return 0 if result["status"] == "PASS" else 2


def confined_command(args, target, helper, synthetic_socket):
    """Backend a003: preflight host, R nel consumer, un solo mount/PID/netns."""
    if args.bundle or args.inside or args.command or not args.confinement or not args.preflight_config:
        raise ValueError('bwrap richiede snapshot, manifest e preflight, senza comando alternativo')
    manifest = json.loads(args.confinement.read_text())
    path = args.repo/'scripts/diagnostics/run-a003-fase1-engine-comparison/confinement.py'
    spec = importlib.util.spec_from_file_location('a003_standard_confinement', path)
    confinement = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(confinement)
    confinement.validate(manifest)
    if manifest['phase'] != 'conversion' or manifest['python'] != args.python or manifest['scratch'] != str(args.cwd):
        raise ValueError('Consumer incompatibile con target/cwd')
    cfg = json.loads(Path(manifest['consumer_config']).read_text())
    if manifest.get('consumer') != str(path.with_name('consumer.py')) or cfg.get('argv') != manifest['argv']:
        raise ValueError('Authentic consumer/argv required')
    if cfg.get('runtime') != {k: target[k] for k in ('python','prefix','base_prefix','python_version','dependencies','startup')}:
        raise ValueError('Runtime consumer non legato al target')
    if cfg.get('daemon_paths') != [row['path'] for row in target['daemons']]:
        raise ValueError('Sonde daemon non legate al target')
    checks = helper.prerequisites(target)
    if not all(row['matches'] for row in checks):
        raise ValueError('Prerequisiti host FAIL')
    conf = json.loads(args.preflight_config.read_text())
    if conf.get('schema') != 1 or not conf.get('commands'):
        raise ValueError('Preflight host obbligatorio')
    result = {'schema':1, 'host_prerequisites':checks, 'preflights':[],
              'probe_phase':'consumer before engine; private net/pid/proc/mount',
              'preflight_phase':'host before confinement; source/archives/installation', 'status':'IMPEDITA'}
    for command in conf['commands']:
        if not command or command[0] != args.python or not all(isinstance(x,str) for x in command):
            raise ValueError('argv preflight incompatibile')
        child = subprocess.run(command, cwd=args.repo, env=clean_env(target), capture_output=True, text=True, close_fds=True, timeout=60)
        result['preflights'].append({'argv':command,'exit_code':child.returncode,'stdout':child.stdout,'stderr':child.stderr})
        if child.returncode:
            result['status']='FAIL'
            write_new(args.output/'inside.json',result)
            return 2
    monitor_path=args.repo/'scripts/benchmarks/resources.py'
    spec=importlib.util.spec_from_file_location('a003_standard_monitor',monitor_path)
    monitor=importlib.util.module_from_spec(spec);spec.loader.exec_module(monitor)
    probes_path=Path(cfg['probe_receipt'])
    if not probes_path.is_relative_to(args.cwd) or '..' in probes_path.parts:
        raise ValueError('Probe receipt outside scratch')
    admission_fd = confinement.memfd_create()
    try:
        command = confinement.argv(manifest, netns=args.host_netns or os.readlink('/proc/self/ns/net'), synthetic_socket=synthetic_socket, admission_fd=admission_fd)
        receipt=monitor.run(command,args.cwd,Path(manifest['process_output']),clean_env(target),manifest['timeout_s'],
                            rss_limit_bytes=manifest.get('rss_limit_bytes',6*1024**3), pass_fds=(admission_fd,))
        write_new(manifest['result'],receipt)
        try:
            authentic, probes = confinement.sealed_admission(admission_fd)
            (args.output/'admission.json').write_bytes(authentic)
            actual = confinement.safe_bytes(args.cwd, str(probes_path.relative_to(args.cwd)), expected_identity=probes['copy_identity'])
            if actual != authentic:
                raise ValueError('Probe copy replaced after admission')
            if probes['argv_after_probes'] != manifest['argv']:
                raise ValueError('Admission argv mismatch')
            result['admission'] = {'method':'host memfd inherited by consumer, sealed and closed before exec; no engine FD', 'sha256':hashlib.sha256(authentic).hexdigest(), 'copy_matches':True}
        except (OSError, ValueError, KeyError, TypeError) as exc:
            result['admission_error'] = repr(exc)
            probes = {'status':'UNAVAILABLE'}
    finally:
        os.close(admission_fd)
    result.update(command={'argv':command,'resources':manifest['result']},
                  consumer_receipt={'path':str(probes_path),'sha256':result.get('admission',{}).get('sha256')},
                  namespaces=probes.get('namespaces'), monitor_phase='host; includes boundary and children')
    # R attesta l'ammissione; l'exit del motore resta un risultato separato.
    passed=probes['status']=='PASS' and receipt['children_reaped'] and not receipt['stop_reason']
    result['status']='PASS' if passed else 'FAIL'
    write_new(args.output/'inside.json',result)
    return 0 if passed else 2


def outside(args, target):
    args.output.mkdir(exist_ok=False)
    result = {"schema": 1, "captured_at_utc": datetime.now(timezone.utc).isoformat(), "runner_candidate": args.runner,
              "wrapper_sha256": digest(__file__), "diagnostic_sha256": digest(Path(__file__).with_name("check_runner.py")),
              "target_sha256": digest(args.target), "status": "IMPEDITA", "host_netns": os.readlink("/proc/self/ns/net")}
    # Import fidato del solo diagnostico stdlib a path assoluto, anche con Python -I.
    spec = importlib.util.spec_from_file_location("a001_runner", Path(__file__).with_name("check_runner.py"))
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    socket_dir = None
    try:
        result["binding_before"] = binding_check(args)
        target_files_check(target)
        if args.bundle and (args.command or args.preflight_config):
            raise ValueError('Bundle preliminare ammette solo R stdlib')
        result["host_daemons_before"] = [dict(item, probe=helper.connect_only(socket.AF_UNIX, item["path"]))
                                          for item in target["daemons"] if item["exists"]]
        socket_dir = Path(tempfile.mkdtemp(prefix="s-", dir=target.get('socket_root', target["tmpdir"])))
        path = socket_dir / "s"
        if len(os.fsencode(path)) >= 108:
            raise ValueError('Socket sintetico richiede path UNIX <108 byte; usare socket_root proprio più breve')
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener:
            listener.bind(str(path))
            listener.listen(16)
            result["synthetic_socket"] = str(path)
            result["synthetic_host_before"] = helper.connect_only(socket.AF_UNIX, str(path))
            if not result["synthetic_host_before"]["connected"]:
                raise ValueError("Socket sintetico non raggiungibile fuori dal runner")
            inner = [args.python, "-I", "-B", str(Path(__file__).resolve()), "--inside", "--python", args.python,
                     "--repo", str(args.repo), "--target", str(args.target),
                     "--output", str(args.output), "--cwd", str(args.cwd), "--host-netns", result["host_netns"],
                     "--synthetic-socket", str(path), "--expected-exit", str(args.expected_exit), "--timeout", str(args.timeout)]
            inner += ['--snapshot', str(args.snapshot)] if args.snapshot else ['--bundle', str(args.bundle)]
            if args.preflight_config:
                inner += ["--preflight-config", str(args.preflight_config)]
            inner += args.command
            if args.runner == 'bwrap':
                exit_code=confined_command(args,target,helper,str(path))
                proc=type('Exit', (), {'returncode':exit_code})()
                result['invocation']={'backend':'single bwrap boundary','exit_code':exit_code,'manifest':str(args.confinement),
                                      'manifest_sha256':digest(args.confinement)}
            elif args.runner == "unshare":
                runner = [target["binaries"]["unshare"], "--user", "--map-root-user", "--net", "--"]
            else:
                blacklist = sorted({item["path"] for item in target["daemons"]} | {str(path)})
                runner = [target["binaries"]["firejail"], "--noprofile", "--net=none", f"--env=TMPDIR={target['tmpdir']}", *[f"--blacklist={p}" for p in blacklist], "--"]
            if args.runner != 'bwrap':
                proc = subprocess.run(runner + inner, env=clean_env(target), capture_output=True, text=True,
                                      close_fds=True, timeout=args.timeout + 180)
                (args.output / "runner-stdout.txt").write_text(proc.stdout)
                (args.output / "runner-stderr.txt").write_text(proc.stderr)
                result["invocation"] = {"argv": runner + inner, "exit_code": proc.returncode,
                                        "stdout": "runner-stdout.txt", "stderr": "runner-stderr.txt"}
            result["synthetic_host_after"] = helper.connect_only(socket.AF_UNIX, str(path))
        result["host_daemons_after"] = [dict(item, probe=helper.connect_only(socket.AF_UNIX, item["path"]))
                                         for item in target["daemons"] if item["exists"]]
        result["binding_after"] = binding_check(args)
        target_files_check(target)
        result["inputs_unchanged"] = digest(args.target) == result["target_sha256"] and digest(__file__) == result["wrapper_sha256"]
        if (args.output / "inside.json").is_file():
            inner_result = json.loads((args.output / "inside.json").read_text())
            result["status"] = inner_result["status"] if proc.returncode == 0 else "IMPEDITA" if inner_result["status"] == "IMPEDITA" else "FAIL"
        if not result["inputs_unchanged"] or not result["synthetic_host_after"]["connected"]:
            result["status"] = "FAIL"
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        if isinstance(exc, ValueError):
            result["status"] = "FAIL"
        result["error"] = f"{type(exc).__name__}: {exc}"
        if isinstance(exc, RuntimeError):
            result.update(status='FAIL', stop_reason='HOST_RUN_ERROR', children_reaped=None)
    finally:
        if socket_dir is not None:
            # Unica directory socket appena creata dal wrapper, mai cleanup globale.
            try:
                probe = socket_dir / 's'
                if probe.exists():
                    probe.unlink()
                socket_dir.rmdir()
            except OSError as cleanup_error:
                result['cleanup_error'] = f'{type(cleanup_error).__name__}: {cleanup_error}'
                # L'errore secondario non sostituisce la diagnosi del runner.
                if result['status'] == 'PASS':
                    result['status'] = 'FAIL'
        result["temporary_socket_cleaned"] = socket_dir is None or not socket_dir.exists()
        write_new(args.output / "receipt.json", result)
    print(json.dumps({"status": result["status"], "receipt": str(args.output / "receipt.json")}, ensure_ascii=True))
    return 0 if result["status"] == "PASS" else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", choices=("unshare", "firejail", "bwrap"), default="unshare")
    parser.add_argument('--confinement', type=Path, help='Manifest conversione a003 per backend bwrap')
    parser.add_argument("--python", required=True)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--target", required=True, type=Path)
    binding = parser.add_mutually_exclusive_group(required=True)
    binding.add_argument("--snapshot", type=Path)
    binding.add_argument("--bundle", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cwd", required=True, type=Path)
    parser.add_argument("--preflight-config", type=Path)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--expected-exit", type=int, default=0)
    parser.add_argument("--inside", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--host-netns", help=argparse.SUPPRESS)
    parser.add_argument("--synthetic-socket", help=argparse.SUPPRESS)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not 0 < args.timeout <= 900:
        parser.error('Deadline figlio deve essere <=900s')
    resource.setrlimit(resource.RLIMIT_FSIZE, (33554432, 33554432))
    for name in ("repo", "target", "snapshot", "bundle", "output", "cwd", "preflight_config", "confinement"):
        path = getattr(args, name)
        if path is not None:
            if not path.is_absolute() or any(part.is_symlink() for part in (path, *path.parents)):
                parser.error(f"Path assoluto senza symlink richiesto: {name}")
    target = json.loads(args.target.read_text())
    if target.get("schema") != 1 or target["repo"] != str(args.repo) or target["python"] != args.python:
        parser.error("Schema/repo/interprete del target incompatibile")
    clean_env(target)
    if (args.runner == 'bwrap') != bool(args.confinement):
        parser.error('Manifest confinement esclusivo e obbligatorio per bwrap')
    if args.inside:
        if not args.host_netns or not args.synthetic_socket:
            parser.error("Parametri inside mancanti")
        return inside(args, target)
    return outside(args, target)


if __name__ == "__main__":
    raise SystemExit(main())
