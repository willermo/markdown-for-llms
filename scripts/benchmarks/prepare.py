#!/usr/bin/env python3
"""Prepara uno stage locale offline con gli strumenti standard S/B/I/R.

Non acquisisce software e non modifica lock/pin o prefix storici. La wheel
canonica viene reinstallata per ultima in una venv nuova dello stage.
"""
import argparse
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import sys
import tomllib
import zipfile

from data import descriptor, load, write_new
from inventory import capture
from resources import run


def audit_wheels(repo):
    """Solo wheel che saranno candidate locali, tutte legate a uv.lock."""
    lock = tomllib.loads((repo / "uv.lock").read_text())
    accepted = {w["hash"].removeprefix("sha256:") for p in lock["package"] for w in p.get("wheels", [])}
    directories = [repo / "temp/run-a001-fase0-uv/work/core-profiles-r001/wheels", repo / "temp/run-a001-fase0-uv/work/product-sbi-r001/wheel-inputs"]
    records = []
    for directory in directories:
        for path in sorted(directory.glob("*.whl")):
            record = descriptor(path)
            if record["sha256"] not in accepted:
                raise ValueError("Wheel locale non nel lock: " + str(path))
            records.append(record)
    return directories, records


def audit_dev_startup(venv, managed, wheels):
    """Audit statico prima del primo avvio del nuovo interprete dev."""
    site = venv / "lib/python3.12/site-packages"
    expected = {"_virtualenv.pth": b"import _virtualenv"}
    for row in wheels:
        with zipfile.ZipFile(row["path"]) as archive:
            for name in archive.namelist():
                if name.endswith(".pth"):
                    if name != "a1_coverage.pth":
                        raise ValueError("Startup runtime inatteso")
                    content = archive.read(name)
                    literal = b'import sys; exec(\'import os\\n\\nif os.getenv("COVERAGE_PROCESS_START") or os.getenv("COVERAGE_PROCESS_CONFIG"):\\n try:\\n  import coverage\\n except:\\n  pass\\n else:\\n  coverage.process_startup(slug="pth")\')'
                    if content != literal + b"\n":
                        raise ValueError("Hook coverage non ricevuto")
                    expected[name] = content
    if sorted(p.name for p in site.glob("*.pth")) != sorted(expected):
        raise ValueError("Startup dev non chiuso")
    for name, content in expected.items():
        if (site / name).read_bytes() != content:
            raise ValueError("Startup dev bytes difformi")
    if list(site.glob("*customize*")):
        raise ValueError("Customize dev inatteso")
    cfg = (venv / "pyvenv.cfg").read_text()
    if "version_info = 3.12.13" not in cfg or "include-system-site-packages = false" not in cfg:
        raise ValueError("Prefix dev non chiuso")
    if (venv / "bin/python").resolve() != managed / "cpython-3.12.13-linux-x86_64-gnu/bin/python3.12":
        raise ValueError("Managed dev diverso")
    if descriptor(site / "_virtualenv.py")["sha256"] != "6cf30c56faf2a55228914dbbd17f8088ed371ebb08f5e7fa6fd931f913fcaf1d":
        raise ValueError("Hook uv dev diverso")
    return [descriptor(p) for p in [venv / "pyvenv.cfg", site / "_virtualenv.py", *site.glob("*.pth")]]


def stage(repo, output, label, base_target):
    output.mkdir(exist_ok=False)
    env = dict(base_target["environment"])
    managed = repo / ".venv-python"
    python = managed / "cpython-3.12.13-linux-x86_64-gnu/bin/python3.12"
    uv = Path.home() / ".local/bin/uv"
    backend = repo / "temp/run-a001-fase0-uv/work/docker-cpu-r001/wheels/setuptools-84.0.0-py3-none-any.whl"
    diagnostic = repo / "scripts/diagnostics/run-a001-fase0-uv"
    env["UV_PYTHON_INSTALL_DIR"] = str(managed)
    venv = output / "dev"
    env["UV_PROJECT_ENVIRONMENT"] = str(venv)
    commands = []
    def call(argv, timeout=900):
        ident = f"step-{len(commands)+1:02d}"
        result = run(list(map(str, argv)), repo, output / ident, env, timeout)
        commands.append(result)
        write_new(output / (ident + ".json"), result)
        if result["adapter_exit"] or result["stop_reason"]:
            raise RuntimeError("Preparazione fallita: " + ident)
        return result
    call([python, "-I", "-B", diagnostic / "guard_product_inputs.py", "--python", python, "--managed", managed])
    call([python, "-I", "-B", diagnostic / "check_python_origin.py", "--expected-managed", managed])
    incoming = output / "package-inputs.json"
    call([python, "-I", "-B", diagnostic / "make_input_inventory.py", "--repo", repo, "--managed", managed, "--backend-wheel", backend, "--uv", uv, "--tokenizer-cache", base_target["tokenizer_cache"], "--output", incoming])
    localdirs, wheels = audit_wheels(repo)
    # Add actual runtime/dev payloads before snapshot/S; not covered by a
    # historical inventory nor by the mere fact that paths still exist.
    inventory = load(incoming)
    inventory["files"].extend(wheels)
    incoming.write_text(json.dumps(inventory, separators=(",", ":")) + "\n")
    write_new(output / "local-wheel-audit.json", {"lock": descriptor(repo / "uv.lock"), "wheels": wheels})
    extras = [Path("/usr/bin") / name for name in ("pandoc", "mutool", "pdfinfo", "pdftoppm", "pdftotext", "pdffonts", "inkscape", "firejail", "unshare")]
    extras += [Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"), Path("/usr/share/doc/fonts-dejavu-core/copyright")]
    extras += [Path('/opt/google/chrome/chrome'), Path('/opt/google/chrome/chrome-sandbox'),Path('/opt/google/chrome/chrome_crashpad_handler')]
    extras += [Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
               Path("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"),
               Path("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"),
               Path("/usr/share/fonts/truetype/libreoffice/opens___.ttf"),
               Path("/usr/share/doc/fonts-opensymbol/copyright"),
               Path("/usr/share/doc/fonts-liberation/copyright")]
    extras += sorted(Path("/usr/share/fonts/truetype/liberation").glob("*.ttf"))
    # LibreOffice/Chrome entrypoints are symlinks: identify resolved tool bytes.
    extras += [Path("/usr/bin/libreoffice").resolve(), Path("/usr/bin/google-chrome").resolve()]
    external = output / "benchmark-inputs.json"
    write_new(external, capture(repo, extras))
    call([python, "-I", "-B", repo / "scripts/run_context.py", "snapshot", "run-a002-fase1-corpus", "--label", label, "--artifact", incoming.relative_to(repo), "--artifact", external.relative_to(repo)])
    snapshot = repo / "temp/run-a002-fase1-corpus/snapshots" / (label + ".json")
    source = output / "S.json"
    call([python, "-I", "-B", diagnostic / "make_source_manifest.py", "--repo", repo, "--snapshot", snapshot, "--input-inventory", incoming, "--uv", uv, "--output", source])
    call([uv, "build", "--offline", "--python", python, "--managed-python", "--no-python-downloads", "--build-constraints", repo / "build-constraints.txt", "--out-dir", output / "dist"])
    sdist, = (output / "dist").glob("*.tar.gz")
    wheel, = (output / "dist").glob("*.whl")
    args = ["--repo", repo, "--source-manifest", source, "--sdist", sdist, "--wheel", wheel, "--profile", "dev", "--expected-managed", managed]
    call([python, "-I", "-B", diagnostic / "verify_distribution.py", *args, "--archives-only", "--receipt", output / "B.json"])
    runtime = output / "runtime-dev.txt"
    call([uv, "export", "--offline", "--locked", "--no-default-groups", "--group", "dev", "--no-emit-project", "--format", "requirements.txt", "--output-file", runtime])
    call([uv, "venv", "--python", python, "--managed-python", "--no-python-downloads", venv])
    links = [x for path in localdirs for x in ("--find-links", path)]
    call([uv, "pip", "sync", "--offline", "--no-index", "--no-build", "--require-hashes", "--python", venv / "bin/python", *links, runtime])
    startup = audit_dev_startup(venv, managed, wheels)
    write_new(output / "startup-audit.json", {"files": startup, "COVERAGE_environment": "absent in closed environment"})
    call([uv, "pip", "install", "--offline", "--python", venv / "bin/python", "--no-deps", "--no-build", "--reinstall-package", "markdown-for-llms", wheel])
    call([venv / "bin/python", "-I", "-B", diagnostic / "verify_distribution.py", *args, "--receipt", output / "I.json"])
    call([venv / "bin/python", "-I", "-B", diagnostic / "check_preflight.py", "--repo", repo, "--source", source, "--receipt", output / "I.json"])
    # Safe metadata capture in the new interpreter, after I. No product import.
    query = 'import importlib.metadata,json,sys,sysconfig;print(json.dumps({"python":sys.executable,"prefix":sys.prefix,"base_prefix":sys.base_prefix,"site":sysconfig.get_path("purelib"),"dependencies":[{"name":d.metadata["Name"],"version":d.version} for d in importlib.metadata.distributions()]}))'
    call([venv / "bin/python", "-I", "-B", "-c", query])
    metadata = json.loads((output / f"step-{len(commands):02d}/stdout.bin").read_bytes())
    target = dict(base_target)
    target.update(python=metadata["python"], prefix=metadata["prefix"], base_prefix=metadata["base_prefix"], dependencies=metadata["dependencies"])
    site = Path(metadata["site"])
    target["startup"] = {"site_dirs": [str(site)], "pth": [descriptor(p) for p in sorted(site.glob("*.pth"))]}
    # coverage hook may be present, but runs are closed with COVERAGE_* absent.
    target["files"] = [row for row in base_target["files"] if not row["path"].startswith(str(repo / ".venv") + "/")]
    target["files"] += [descriptor(p) for p in [venv / "pyvenv.cfg", *site.glob("*.pth"), site / "_virtualenv.py", *site.glob("*.dist-info/METADATA"), source, output / "I.json", external]]
    target["workload_environment"] = {"RUN_SOURCE_MANIFEST": str(source), "RUN_INSTALLATION_RECEIPT": str(output / "I.json")}
    write_new(output / "target.json", target)
    preflight = {"schema": 1, "commands": [[str(venv / "bin/python"), "-I", "-B", str(diagnostic / "check_preflight.py"), "--repo", str(repo), "--source", str(source), "--receipt", str(output / "I.json")], [str(venv / "bin/python"), "-I", "-B", str(repo / "scripts/benchmarks/launch.py"), "inventory", "verify", str(external)]]}
    write_new(output / "preflight.json", preflight)
    result = {"schema_version": 1, "snapshot": descriptor(snapshot), "source": descriptor(source), "archives": [descriptor(sdist), descriptor(wheel)], "installation": descriptor(output / "I.json"), "external": descriptor(external), "python": str(venv / "bin/python"), "commands": commands}
    write_new(output / "preparation.json", result)
    print("PREPARATION_COMPLETE " + str(output))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--target", type=Path, required=True)
    a = p.parse_args()
    stage(a.repo, a.output, a.label, load(a.target))


if __name__ == "__main__":
    main()
