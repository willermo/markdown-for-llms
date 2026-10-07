#!/usr/bin/env python3
"""Produce S v1 dopo il freeze del supervisore, oppure con scope standalone.

Per la baseline --repo è la copia identificata e --host-repo il clone Git;
i suoi input devono essere artifacts dello snapshot, non i moduli del prodotto.
Nessuna build, installazione, import applicativo o scrittura di snapshot.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tomllib


MODULES = ("config", "logging_config", "exceptions", "unified_converter", "master_workflow",
           "clean_markdown", "validate_markdown", "chunk_markdown", "batch_monitor", "marker_api_server")
INPUTS = ("pyproject.toml", ".python-version", "uv.lock", "build-constraints.txt", "README.md", "LICENSE",
          "MANIFEST.in", "setup.cfg", "setup.py", "requirements.txt", "uv.toml")


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":")).encode()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def no_symlinks(path):
    if not path.is_absolute() or '..' in path.parts or any(part.is_symlink() for part in (path, *path.parents)):
        raise ValueError(f"Path assoluto senza symlink richiesto: {path}")
    return path


def relative_path(value):
    path = Path(value)
    if (not isinstance(value, str) or not value or path.is_absolute() or
            '..' in path.parts or '\\' in value or path.as_posix() != value or value == '.'):
        raise ValueError(f"Path relativo non canonico: {value}")
    return path


def load_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Chiave JSON duplicata: {key}")
            result[key] = value
        return result
    return json.loads(no_symlinks(path).read_text(), object_pairs_hook=unique)


def envelope(payload):
    return {"schema": 1, "id": "sha256:" + hashlib.sha256(canonical(payload)).hexdigest(),
            "payload": payload, "captured_at_utc": datetime.now(timezone.utc).isoformat()}


def check_envelope(value):
    if (set(value) != {"schema", "id", "payload", "captured_at_utc"} or
            type(value['schema']) is not int or value['schema'] != 1 or
            not isinstance(value['payload'], dict) or
            value['id'] != 'sha256:' + hashlib.sha256(canonical(value['payload'])).hexdigest()):
        raise ValueError("Envelope schema/ID non valido")
    return value['payload']


def entry(path, host_repo):
    no_symlinks(path)
    relative = path.relative_to(host_repo).as_posix()
    if not path.exists():
        return {"path": relative, "kind": "absent"}
    if path.is_dir():
        raise ValueError(f"Input directory non supportato, identificarne prima i file: {path}")
    data = path.read_bytes()
    return {"path": relative, "kind": "file", "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def git(host_repo, *argv):
    result = subprocess.run(["git", "-C", str(host_repo), *argv], capture_output=True, text=True, close_fds=True, timeout=15)
    if result.returncode:
        raise ValueError(f"Git {argv}: exit {result.returncode}")
    return result.stdout.rstrip("\n")


def snapshot_check(path, host_repo):
    no_symlinks(path)
    data = load_json(path)
    required = {"schema", "run_id", "label", "head", "branch", "files", "artifacts", "worktree_sha256"}
    if not required <= data.keys() or type(data["schema"]) is not int or data["schema"] != 1:
        raise ValueError("Snapshot schema sconosciuto/incompleto")
    import re
    if (not isinstance(data['run_id'], str) or
            re.fullmatch(r'run-[a-z][0-9]{3,}-fase[a-z0-9]+-[a-z0-9]+(?:-[a-z0-9]+)*', data['run_id']) is None or
            not isinstance(data['label'], str) or
            re.fullmatch(r'[a-z0-9][a-z0-9-]{0,79}', data['label']) is None):
        raise ValueError('Run ID/label snapshot non validi')
    expected = host_repo / "temp" / data["run_id"] / "snapshots" / (data["label"] + ".json")
    if path != expected:
        raise ValueError("Snapshot non appartenente al clone/stage")
    result = subprocess.run([sys.executable, "-I", "-B", str(host_repo / "scripts/run_context.py"), "verify",
                             data["run_id"], "--label", data["label"]], cwd=host_repo,
                            capture_output=True, text=True, close_fds=True, timeout=60)
    if result.returncode:
        raise ValueError(f"Stage non MATCH: {result.stdout.strip()}")
    indexed = {}
    for item in [*data["files"], *data["artifacts"]]:
        key = item["path"]
        relative_path(key)
        if key in indexed:
            raise ValueError(f"Input duplicato/incompatibile nello snapshot: {key}")
        indexed[key] = item
    return data, indexed


def match_snapshot(entries, indexed):
    if len({e['path'] for e in entries}) != len(entries):
        raise ValueError("Input duplicato")
    for item in entries:
        relative_path(item['path'])
        stored = indexed.get(item["path"])
        if item["kind"] == "absent":
            if stored is not None and stored.get("kind") != "deleted":
                raise ValueError(f"Assenza difforme nello snapshot: {item['path']}")
        elif (stored is None or stored.get("kind", "file") != "file" or
              stored.get("sha256") != item["sha256"] or
              ('bytes' in stored and stored['bytes'] != item['bytes'])):
            raise ValueError(f"Input assente/difforme nello snapshot: {item['path']}")


def git_identity(repo, standalone=False):
    if not standalone:
        return {'branch': git(repo, 'branch', '--show-current'), 'head': git(repo, 'rev-parse', 'HEAD'),
                'dev': git(repo, 'rev-parse', 'dev'), 'merge_base': git(repo, 'merge-base', 'HEAD', 'dev'),
                'status': git(repo, 'status', '--porcelain=v1', '--untracked-files=all')}
    if Path(git(repo, 'rev-parse', '--show-toplevel')).resolve() != repo.resolve():
        raise ValueError('Git standalone appartiene a un altro clone')
    def optional_ref(ref):
        r = subprocess.run(['git', '-C', str(repo), 'rev-parse', '--verify', '--quiet', ref],
                           capture_output=True, text=True, shell=False, close_fds=True, timeout=15)
        if r.returncode not in (0, 1):
            raise ValueError('Git standalone non leggibile')
        return r.stdout.strip() if r.returncode == 0 else None
    head, dev = optional_ref('HEAD'), optional_ref('refs/heads/dev')
    return {'branch': git(repo, 'branch', '--show-current'), 'head': head, 'dev': dev,
            'merge_base': git(repo, 'merge-base', 'HEAD', 'dev') if head and dev else None,
            'status': git(repo, 'status', '--porcelain=v1', '--untracked-files=all')}


def validate_current(source, repo, official=False):
    """Gate corrente ↔ S ↔ snapshot, da chiamare prima di collection/import."""
    payload = check_envelope(source)
    required = {'scope', 'repo', 'host_repo', 'git', 'modules', 'build_inputs',
                'diagnostics', 'input_inventory', 'backend', 'toolchain', 'snapshot'}
    if set(payload) != required or payload['scope'] not in {'run-stage', 'standalone'}:
        raise ValueError('Payload S ignoto/incompleto')
    no_symlinks(repo)
    if str(repo) != payload['repo']:
        raise ValueError('S appartiene a un altro clone/copia')
    host = no_symlinks(Path(payload['host_repo']))
    repo.relative_to(host)
    if official and payload['scope'] != 'run-stage':
        raise ValueError('S standalone non è una prova ufficiale della run')
    snapshot = payload['snapshot']
    indexed = None
    if payload['scope'] == 'run-stage':
        if not isinstance(snapshot, dict):
            raise ValueError('Snapshot mancante')
        path = no_symlinks(Path(snapshot['path']))
        if digest(path) != snapshot['sha256']:
            raise ValueError('SHA snapshot difforme')
        data, indexed = snapshot_check(path, host)
        if data['worktree_sha256'] != snapshot['worktree_sha256'] or data['label'] != snapshot['label']:
            raise ValueError('Identità snapshot difforme')
    elif snapshot is not None or host != repo:
        raise ValueError('Standalone richiede clone esplicito senza snapshot/host diverso')
    expected_modules = sorted((repo / (m+'.py')).relative_to(host).as_posix() for m in MODULES)
    if [x['path'] for x in payload['modules']] != expected_modules:
        raise ValueError('S deve identificare esattamente dieci moduli ordinati')
    required_inputs = {(repo/n).relative_to(host).as_posix() for n in (*INPUTS, 'src')}
    if not required_inputs <= {x['path'] for x in payload['build_inputs']}:
        raise ValueError('Input build obbligatori mancanti')
    records = payload['modules'] + payload['build_inputs'] + payload['diagnostics']
    if payload['input_inventory']:
        records += [payload['input_inventory']]
    if len({x['path'] for x in records}) != len(records):
        raise ValueError('Input S duplicati')
    for item in records:
        p = host / relative_path(item['path'])
        if entry(p, host) != item:
            raise ValueError('Input corrente diverso da S: ' + item['path'])
    if indexed is not None:
        match_snapshot(records, indexed)
    current_git = git_identity(host, standalone=payload['scope']=='standalone')
    if current_git != payload['git']:
        raise ValueError('Git corrente diverso da S')
    if payload['backend']['build_system']:
        if not payload['input_inventory']:
            raise ValueError('Inventario ingresso package mancante')
        verify_input_inventory(host / payload['input_inventory']['path'])
        config = tomllib.loads((repo/'pyproject.toml').read_text())
        if (payload['backend'] != {'build_system':config.get('build-system'),
                                  'tool_uv':config.get('tool',{}).get('uv'),
                                  'config_settings':{},'build_environment':{}}):
            raise ValueError('Backend/config di S diversi dagli input reali')
    return payload


def verify_input_inventory(path):
    """Inventario futuro di ingresso: riferimenti reali, nessun output S/B/I/E."""
    data = load_json(path)
    if set(data) != {'schema','scope','files','backend'} or data['schema'] != 1 or data['scope'] != 'package-inputs':
        raise ValueError('Schema inventario package ignoto')
    if not data['files'] or len({r['path'] for r in data['files']}) != len(data['files']):
        raise ValueError('Inventario vuoto/duplicato')
    for r in data['files']:
        p = no_symlinks(Path(r['path']))
        if set(r) != {'path','bytes','sha256'} or not p.is_file():
            raise ValueError('File inventario sconosciuto/mancante')
        b = p.read_bytes()
        if len(b) != r['bytes'] or hashlib.sha256(b).hexdigest() != r['sha256']:
            raise ValueError('Input inventario cambiato: '+str(p))
    backend = data['backend']
    if (set(backend) != {'name','version','artifacts','config_settings','environment'} or
            backend['name'] != 'setuptools' or backend['version'] != '84.0.0' or
            not backend['artifacts'] or backend['config_settings'] != {} or backend['environment'] != {}):
        raise ValueError('Backend/config non identificati o non supportati')
    if not set(backend['artifacts']) <= {r['path'] for r in data['files']}:
        raise ValueError('Artefatti backend assenti dall’inventario')
    return data


def capture(args, snapshot_data, indexed):
    host_repo = args.host_repo or args.repo
    uv_path = getattr(args, 'uv', None) or shutil.which('uv')
    if uv_path:
        uv_path = no_symlinks(Path(uv_path).absolute())
        if not uv_path.is_file():
            raise ValueError('Binario uv esplicito mancante')
    modules = sorted((entry(args.repo / f"{name}.py", host_repo) for name in MODULES), key=lambda e: e["path"])
    if any(item["kind"] != "file" for item in modules):
        raise ValueError("Dieci moduli obbligatori mancanti")
    inputs = [entry(args.repo / name, host_repo) for name in INPUTS]
    pyproject = args.repo / "pyproject.toml"
    config = tomllib.loads(pyproject.read_text()) if pyproject.exists() else {}
    project = config.get("project", {})
    if project.get("dynamic"):
        raise ValueError("Metadata dinamici non identificati: riesame degli input necessario")
    extra_names = set()
    readme = project.get("readme")
    if isinstance(readme, str):
        extra_names.add(readme)
    elif isinstance(readme, dict) and "file" in readme:
        extra_names.add(readme["file"])
    license_value = project.get("license")
    if isinstance(license_value, dict) and "file" in license_value:
        extra_names.add(license_value["file"])
    for pattern in project.get("license-files", []):
        if not isinstance(pattern, str) or Path(pattern).is_absolute() or ".." in Path(pattern).parts:
            raise ValueError("Pattern licenza evasivo")
        extra_names.update(path.relative_to(args.repo).as_posix() for path in args.repo.glob(pattern))
    for name in sorted(extra_names):
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Input metadata evasivo")
        item = entry(args.repo / relative, host_repo)
        if item["kind"] != "file":
            raise ValueError("Input consumato dal backend mancante")
        if name not in INPUTS:
            inputs.append(item)
    # src non è previsto nel layout flat: non accettarlo senza inventario nuovo.
    inputs.append(entry(args.repo / "src", host_repo))
    inventory = entry(args.input_inventory, host_repo) if args.input_inventory else None
    if inventory and inventory["kind"] != "file":
        raise ValueError("Inventario di ingresso mancante")
    if config:
        if inventory is None:
            raise ValueError('Package: inventario ingresso obbligatorio prima di S')
        verify_input_inventory(args.input_inventory)
        for name in ('pyproject.toml', '.python-version', 'uv.lock', 'build-constraints.txt'):
            if not (args.repo / name).is_file():
                raise ValueError('Package: input obbligatorio assente: '+name)
        if (args.repo/'uv.toml').exists() or (args.repo/'setup.cfg').exists():
            raise ValueError('Configurazione aggiuntiva non inventariata nel backend: riesame necessario')
        if config.get('tool',{}).get('setuptools',{}).get('py-modules') != list(MODULES):
            raise ValueError('Elenco py-modules inatteso')
    diagnostics = sorted((entry(p, host_repo) for p in Path(__file__).parent.glob("*.py")), key=lambda e: e["path"])
    all_entries = modules + inputs + diagnostics + ([inventory] if inventory else [])
    if len({item["path"] for item in all_entries}) != len(all_entries):
        raise ValueError("Input S duplicati")
    if indexed is not None:
        match_snapshot(all_entries, indexed)
    packages = sorted(({"name": d.metadata["Name"], "version": d.version,
                        "metadata_sha256": hashlib.sha256((d.read_text("METADATA") or "").encode()).hexdigest(),
                        "record_sha256": hashlib.sha256((d.read_text("RECORD") or "").encode()).hexdigest()}
                       for d in importlib.metadata.distributions()
                       if d.metadata["Name"].lower().replace("_", "-") != "markdown-for-llms"), key=lambda e: e["name"].lower())
    payload = {"scope": "run-stage" if args.snapshot else "standalone", "repo": str(args.repo), "host_repo": str(host_repo),
               "git": git_identity(host_repo, standalone=args.snapshot is None),
               "modules": modules, "build_inputs": sorted(inputs, key=lambda e: e["path"]), "diagnostics": diagnostics,
               "input_inventory": inventory, "backend": {"build_system": config.get("build-system"),
                                                         "tool_uv": config.get("tool", {}).get("uv"),
                                                         "config_settings": {}, "build_environment": {}},
               "toolchain": {"executable": sys.executable, "executable_realpath": str(Path(sys.executable).resolve()),
                             "python_sha256": digest(sys.executable), "prefix": sys.prefix, "base_prefix": sys.base_prefix,
                             "version": sys.version, "stdlib": {"os_path": os.__file__, "os_sha256": digest(os.__file__)},
                             "platform": platform.platform(), "uv_path": str(uv_path) if uv_path else None,
                             "uv_sha256": digest(uv_path) if uv_path else None,
                             "dependencies_incoming": packages},
               "snapshot": None}
    if snapshot_data:
        payload["snapshot"] = {"label": snapshot_data["label"], "path": str(args.snapshot),
                               "sha256": digest(args.snapshot), "worktree_sha256": snapshot_data["worktree_sha256"]}
        if payload["git"]["head"] != snapshot_data["head"] or payload["git"]["branch"] != snapshot_data["branch"]:
            raise ValueError("Git incompatibile con snapshot")
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--snapshot", type=Path)
    group.add_argument("--standalone", action="store_true")
    parser.add_argument("--host-repo", type=Path)
    parser.add_argument("--input-inventory", type=Path)
    parser.add_argument("--uv", type=Path, help="Binario uv canonico, anche fuori dal PATH chiuso")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        for path in (args.repo, args.host_repo, args.snapshot, args.output, args.input_inventory, args.uv):
            if path is not None:
                no_symlinks(path)
        if args.output.exists():
            raise ValueError("Output già presente")
        host_repo = args.host_repo or args.repo
        data, indexed = snapshot_check(args.snapshot, host_repo) if args.snapshot else (None, None)
        before = capture(args, data, indexed)
        after = capture(args, data, indexed)
        if before != after:
            raise ValueError("Input cambiati durante la cattura")
        if args.snapshot:
            repeated, _ = snapshot_check(args.snapshot, host_repo)
            if repeated != data:
                raise ValueError("Snapshot cambiato durante la cattura")
        result = envelope(before)
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2, ensure_ascii=True)
            stream.write("\n")
        print(json.dumps({"id": result["id"], "path": str(args.output), "sha256": digest(args.output)}))
        return 0
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        print(f"FAIL S: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
