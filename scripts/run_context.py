#!/usr/bin/env python3
"""Initialize local run contexts and fingerprint review inputs. Standard library only."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys


RUN_ID = re.compile(r"run-[a-z][0-9]{3,}-fase[a-z0-9]+-[a-z0-9]+(?:-[a-z0-9]+)*\Z")
LABEL = re.compile(r"[a-z0-9][a-z0-9-]{0,79}\Z")
EXCLUDED_PREFIXES = ("temp/", "documentation/runs/")
RUN_DIRECTORIES = (
    "prompts", "plans", "reviews", "arbitrations", "implementation", "handovers",
    "snapshots", "evidence", "probes", "notes",
)


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], stderr=subprocess.PIPE)


def timestamp():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def repository_root():
    return Path(git(Path.cwd(), "rev-parse", "--show-toplevel").decode().strip()).resolve()


def check_local_path(root, path):
    """Reject symlinks in paths used for context reads/writes, including ancestors."""
    relative = path.relative_to(root)
    current = root
    for component in relative.parts:
        current = current / component
        if current.is_symlink():
            raise ValueError(f"Percorso simbolico non ammesso per il contesto: {current}")


def context_root(root):
    path = root / "temp"
    check_local_path(root, path)
    result = subprocess.run(
        ["git", "-C", str(root), "check-ignore", "--quiet", "--no-index", "temp/_context_probe_"],
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        raise ValueError("temp/ deve essere esclusa da Git prima di creare contesto locale.")
    return path


def run_path(root, run_id, must_exist=True):
    if not RUN_ID.fullmatch(run_id):
        raise ValueError("ID non valido; esempio: run-a001-fase0-uv")
    path = context_root(root) / run_id
    check_local_path(root, path)
    if must_exist and not path.is_dir():
        raise ValueError(f"Run inesistente: {run_id}")
    return path


def write_new(root, path, body):
    check_local_path(root, path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(body)


def bootstrap(root):
    path = context_root(root)
    files = {
        "README.md": """# Contesto operativo locale

Questa cartella è ignorata da Git. Per riprendere leggere HANDOVER.md, PROJECT-CONTEXT.md
e RUNS.md, poi STATE.md e il checkpoint del ruolo nella run indicata.
Il protocollo versionato è `documentation/development/run-lifecycle.md`.
Git non trasferisce né salva questi file: per altre macchine/chat fornire il contesto
e, se necessario, il lavoro non committato. Non conservare qui credenziali.
""",
        "PROJECT-CONTEXT.md": """# Contesto comune

Progetto: convertitore di documenti in Markdown completo con asset e provenienza.
Leggere AGENTS.md, documentation/README.md, registro ADR e roadmap nel repository.
Decisioni e autorizzazioni aggiuntive vanno registrate qui dal supervisore quando
comuni a più run; i dettagli di una sola run rimangono nella relativa sottocartella.
""",
        "RUNS.md": """# Registro locale delle run

Gli stati aggiornati sono nei rispettivi STATE.md; l'handover globale indica la run corrente.

| Run | Fase | Titolo | Stato autorevole |
| --- | --- | --- | --- |
""",
        "HANDOVER.md": """# Prompt di recupero

<!-- NO_ACTIVE_RUN -->
Leggi AGENTS.md, temp/PROJECT-CONTEXT.md e temp/RUNS.md. Non è ancora assegnata una
run corrente. Riprendi la run indicata dall'utente verificandone STATE.md, HANDOVER.md,
branch e HEAD; non presumere che piani o review siano stati completati.
""",
    }
    for name, body in files.items():
        target = path / name
        check_local_path(root, target)
        if not target.exists():
            write_new(root, target, body)
    return path


def initialize(root, args):
    path = run_path(root, args.run_id, must_exist=False)
    if path.exists():
        raise ValueError(f"Run già presente; nessun file sovrascritto: {args.run_id}")
    base_commit = git(root, "rev-parse", "--verify", "--end-of-options", args.base + "^{commit}").decode().strip()
    head = git(root, "rev-parse", "HEAD").decode().strip()
    branch = git(root, "branch", "--show-current").decode().strip() or "DETACHED"
    values = {
        "RUN_ID": args.run_id, "TITLE": args.title, "PHASE": args.phase,
        "BRANCH": args.branch, "BASE": args.base, "BASE_COMMIT": base_commit,
        "OBSERVED_BRANCH": branch, "HEAD": head, "TIMESTAMP": timestamp(),
    }
    state = (root / "documentation/development/templates/run-state.md").read_text(encoding="utf-8")
    for key, value in values.items():
        state = state.replace("{{" + key + "}}", value)
    if re.search(r"\{\{[A-Z_]+\}\}", state):
        raise ValueError("Template di stato contiene variabili non risolte.")
    context = bootstrap(root)
    path.mkdir()
    for directory in RUN_DIRECTORIES:
        (path / directory).mkdir()
    files = {
        "STATE.md": state,
        "brief.md": f"# Brief — {args.run_id}\n\nTitolo: {args.title}\n\nDa completare nell'analisi: obiettivo, perimetro, criteri, dipendenze e decisioni.\n",
        "HANDOVER.md": f"# Handover — {args.run_id}\n\nStato: PREPARATION. Contesto creato; nessun piano, review o implementazione prodotti.\nLeggere STATE.md e brief.md, verificare branch e dipendenze, poi assegnare la supervisione.\n",
        "events.md": f"# Eventi — {args.run_id}\n\n- {values['TIMESTAMP']}: contesto inizializzato su {branch}, HEAD {head}; branch previsto {args.branch}.\n",
        "artifacts.md": f"# Artefatti — {args.run_id}\n\nIniziali: STATE.md, brief.md, HANDOVER.md, events.md.\nPiani, review, arbitrati e prove non sono ancora stati prodotti.\n",
    }
    for name, body in files.items():
        write_new(root, path / name, body)
    index = context / "RUNS.md"
    check_local_path(root, index)
    safe_title = args.title.replace("|", "\\|").replace("\n", " ")
    with index.open("a", encoding="utf-8") as stream:
        stream.write(f"| {args.run_id} | {args.phase} | {safe_title} | [{args.run_id}/STATE.md]({args.run_id}/STATE.md) |\n")
    handover = context / "HANDOVER.md"
    check_local_path(root, handover)
    if "<!-- NO_ACTIVE_RUN -->" in handover.read_text(encoding="utf-8"):
        handover.write_text(
            f"# Prompt di recupero — {args.run_id}\n\n"
            f"Riprendi la supervisione della run `{args.run_id}`. Leggi AGENTS.md,\n"
            f"temp/PROJECT-CONTEXT.md, temp/{args.run_id}/STATE.md,\n"
            f"temp/{args.run_id}/HANDOVER.md e temp/{args.run_id}/brief.md.\n"
            "Verifica branch, HEAD e stato Git; segui il ciclo in documentation/development/run-lifecycle.md.\n"
            "Non considerare completate attività senza i relativi report.\n",
            encoding="utf-8",
        )
    print(f"Creata {path.relative_to(root)}; stato PREPARATION, nessuna modifica Git eseguita.")


def digest_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(root, relative):
    path = root / relative
    check_local_path(root, path.parent)
    if path.is_symlink():
        return {"path": relative, "kind": "symlink", "sha256": hashlib.sha256(os.fsencode(os.readlink(path))).hexdigest()}
    if not path.exists():
        return {"path": relative, "kind": "deleted"}
    if not path.is_file():
        raise ValueError(f"Oggetto non supportato nello snapshot (es. submodule): {relative}")
    return {
        "path": relative, "kind": "file", "sha256": digest_file(path),
        "executable": bool(path.stat().st_mode & 0o111),
    }


def capture(root, run_dir, artifacts):
    names = git(root, "ls-files", "--cached", "--others", "--exclude-standard", "-z").split(b"\0")
    relative_paths = sorted({os.fsdecode(name) for name in names if name})
    files = [file_record(root, name) for name in relative_paths if not name.startswith(EXCLUDED_PREFIXES)]
    artifact_records = []
    for name in sorted(set(artifacts)):
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Gli artefatti devono avere percorsi relativi confinati alla run.")
        path = root / relative
        path.relative_to(run_dir)
        check_local_path(root, path)
        if not path.is_file():
            raise ValueError(f"Artefatto mancante: {name}")
        artifact_records.append(file_record(root, relative.as_posix()))
    canonical = json.dumps(files, sort_keys=True, ensure_ascii=True, separators=(",", ":")).encode()
    return {
        "schema": 1,
        "head": git(root, "rev-parse", "HEAD").decode().strip(),
        "branch": git(root, "branch", "--show-current").decode().strip() or "DETACHED",
        "excluded_prefixes": list(EXCLUDED_PREFIXES),
        "worktree_sha256": hashlib.sha256(canonical).hexdigest(),
        "files": files,
        "artifacts": artifact_records,
    }


def snapshot_path(root, run_dir, label):
    if not LABEL.fullmatch(label):
        raise ValueError("Etichetta snapshot non valida.")
    path = run_dir / "snapshots" / (label + ".json")
    check_local_path(root, path)
    return path


def snapshot(root, args):
    run_dir = run_path(root, args.run_id)
    output = snapshot_path(root, run_dir, args.label)
    if output.exists():
        raise ValueError("Snapshot già presente: usare una nuova revisione.")
    data = capture(root, run_dir, args.artifact)
    data.update({"created_at": timestamp(), "run_id": args.run_id, "label": args.label})
    write_new(root, output, json.dumps(data, indent=2, ensure_ascii=True) + "\n")
    print(f"Snapshot: {output.relative_to(root)}\nWorking tree SHA-256: {data['worktree_sha256']}")


def verify(root, args):
    run_dir = run_path(root, args.run_id)
    path = snapshot_path(root, run_dir, args.label)
    saved = json.loads(path.read_text(encoding="utf-8"))
    if saved.get("schema") != 1 or saved.get("run_id") != args.run_id or saved.get("label") != args.label:
        raise ValueError("Schema o identità snapshot non validi.")
    current = capture(root, run_dir, [item["path"] for item in saved["artifacts"]])
    differences = [key for key in current if current[key] != saved.get(key)]
    if differences:
        print("STALE: snapshot diverso; campi modificati: " + ", ".join(differences))
        return 2
    print("MATCH: oggetti invariati. Questo controllo non costituisce una review o un GO.")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("bootstrap", help="Create global context files without overwriting existing ones")
    init = commands.add_parser("init", help="Create a new run, without switching or creating branches")
    init.add_argument("run_id")
    init.add_argument("--phase", required=True)
    init.add_argument("--title", required=True)
    init.add_argument("--branch", required=True)
    init.add_argument("--base", default="dev")
    snap = commands.add_parser("snapshot", help="Fingerprint working tree and selected run artifacts")
    snap.add_argument("run_id")
    snap.add_argument("--label", required=True)
    snap.add_argument("--artifact", action="append", default=[])
    check = commands.add_parser("verify", help="Check snapshot without modifying files")
    check.add_argument("run_id")
    check.add_argument("--label", required=True)
    args = parser.parse_args()
    try:
        root = repository_root()
        if args.command == "bootstrap":
            print(f"Contesto globale disponibile: {bootstrap(root).relative_to(root)}")
        elif args.command == "init":
            initialize(root, args)
        elif args.command == "snapshot":
            snapshot(root, args)
        else:
            return verify(root, args)
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
        print(f"Errore: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
