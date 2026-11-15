"""The engine toolchain: identity, verification of the local image, `prepare --toolchain`.

The native conversions of the engine (pandoc, LibreOffice, DjVuLibre) and the external tools of
the bench (derivations, the projection of D06, the self-check) run only in this image, built on
the user's machine from `scripts/bench-toolchain/` of the checkout by `mdf prepare --toolchain`
(alias `mdf bench prepare --toolchain`). The image, its repository and its builder keep the names
of the bench until their renaming (MFL039): moving them to the engine changed no byte of the
build context, so `content_sha256` and the image stay the same. `data/toolchain.json`, in the
wheel, is the identity:

- `context`: the SHA-256 of each file of the build context, which is not in the wheel;
  `prepare --toolchain` refuses a context that differs.
- `content_sha256`: the SHA-256 of the content manifest that the build writes to
  /etc/mdf-bench-toolchain.sha256 (`content-manifest.sh` of the context); two builds from
  scratch give the same manifest, while the image ID changes with every build.
- `config`: environment, entrypoint and user of the image; `versions`: what the tools report.
- `image_id`: the image of the reference build, on which the reference hashes of the manifest
  and the calibration of the self-check were measured (provenance: another build with the same
  content is the same toolchain).
- `builder`: the BuildKit image, by digest, and the memory cap of the builder.
- `selftest`: SHA-256 of the derivations of synthetic files in the self-test (`bench/selftest.py`).

Each process verifies the image at its first tool call (`image()`): the local image tagged
`mdf-bench-toolchain:<content_sha256[:16]>` must exist (`toolchain_missing`) and have the
recorded config and a content manifest with the recorded SHA-256 (`toolchain_mismatch`); from
then on it is called by `sha256:<image id>`, never by tag. A bench process that finds everything
it needs in the cache starts no container and reports `engine: null` (U4: a cache prepared
elsewhere and copied). Docker is the only engine, through a local socket: a remote endpoint
would receive the documents.
"""
import functools
import hashlib
import itertools
import json
import os
from pathlib import Path
import re
import secrets
import shutil
import subprocess
import sys
import tempfile
import time

from .codes import Code, EngineError

ENGINE = "docker"
REPOSITORY = "mdf-bench-toolchain"
BUILDER = "mdf-bench-toolchain"
MANIFEST_PATH = "/etc/mdf-bench-toolchain.sha256"
CONTENT_SCRIPT = "/usr/local/bin/mdf-bench-content-manifest"
SCHEMA = "mdf-bench-toolchain/1"
DEFAULT_CONTEXT = Path("scripts") / "bench-toolchain"
CONTEXT_MTIME = 1767225600          # 2026-01-01T00:00:00Z, the time of every file of the copied context
# Isolation of every container of the toolchain; tools.py adds mounts, caps and environment.
ISOLATION = ("--network", "none", "--cap-drop", "ALL", "--security-opt", "no-new-privileges", "--read-only")
# The only variables of the caller that reach the Docker client (which socket, which config).
ENGINE_ENV = ("DOCKER_HOST", "DOCKER_CONTEXT", "DOCKER_CONFIG", "XDG_RUNTIME_DIR")
SHA256 = re.compile(r"[0-9a-f]{64}")
IMAGE_ID = re.compile(r"sha256:[0-9a-f]{64}")
HINTS = {
    Code.TOOLCHAIN_MISSING: "build the toolchain image with `mdf prepare --toolchain` (alias `mdf bench prepare "
                            "--toolchain`) at the root of the checkout (Docker Engine on Linux)",
    Code.TOOLCHAIN_MISMATCH: "the reference toolchain image or its build context differs from toolchain.json: rebuild "
                             "it with `mdf prepare --toolchain` (alias `mdf bench prepare --toolchain`) from a clean "
                             "checkout",
}

_state = {"image": None, "engine": None}
_counter = itertools.count(1)


# --- identity ---------------------------------------------------------------------------------

def _validate(data):
    keys = {"schema", "context", "content_sha256", "config", "versions", "image_id", "builder", "selftest", "notes"}
    if not isinstance(data, dict) or set(data) != keys or data["schema"] != SCHEMA:
        raise ValueError("toolchain.json: unexpected keys or schema")
    if not data["context"] or not all(isinstance(k, str) and "/" not in k and isinstance(v, str) and SHA256.fullmatch(v)
                                      for k, v in data["context"].items()):
        raise ValueError("toolchain.json: context")
    if not SHA256.fullmatch(data["content_sha256"]) or not IMAGE_ID.fullmatch(data["image_id"]):
        raise ValueError("toolchain.json: content_sha256 or image_id")
    if set(data["config"]) != {"Env", "Entrypoint", "User"}:
        raise ValueError("toolchain.json: config")
    if not all(isinstance(v, str) and re.fullmatch(r"[\d.]+", v) for v in data["versions"].values()):
        raise ValueError("toolchain.json: versions")
    if set(data["builder"]) != {"image", "memory"} or not all(isinstance(v, str) and SHA256.fullmatch(v)
                                                               for v in data["selftest"].values()):
        raise ValueError("toolchain.json: builder or selftest")
    return data


@functools.lru_cache(maxsize=None)
def spec():
    """The packaged identity of the reference toolchain (`data/toolchain.json`)."""
    from importlib import resources
    raw = resources.files("markdown_for_llms.engine").joinpath("data/toolchain.json").read_bytes()
    return _validate(json.loads(raw))


def identity():
    """The identity that enters derivation and check keys: the SHA-256 of the content manifest."""
    return spec()["content_sha256"]


def versions():
    return dict(spec()["versions"])


def tag():
    return f"{REPOSITORY}:{identity()[:16]}"


def report():
    """The toolchain of this process for reports: identity, versions, the image and the engine when a
    container was started (null otherwise: everything came from the cache)."""
    s, image_id = spec(), _state["image"]
    return {"content_sha256": s["content_sha256"], "versions": dict(s["versions"]), "image_id": image_id,
            "reference_image": None if image_id is None else image_id == s["image_id"],
            "engine": None if image_id is None else {"name": ENGINE, "version": _state["engine"]}}


def differs(a, b):
    """True when two reports were measured with different toolchains; `compare` flags it even when the
    checks and the derived inputs coincide (E2)."""
    return (a or {}).get("content_sha256") != (b or {}).get("content_sha256")


def reset():
    """Forget the verified image (tests, and a failed `prepare`)."""
    _state.update(image=None, engine=None)


# --- engine -----------------------------------------------------------------------------------

def engine_path():
    path = shutil.which(ENGINE)
    if path is None:
        raise EngineError(Code.TOOLCHAIN_MISSING, "docker not found on the PATH")
    return path


def engine_env():
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": os.environ.get("HOME", "/"), "LC_ALL": "C"}
    env.update({k: os.environ[k] for k in ENGINE_ENV if k in os.environ})
    return env


def engine_run(args, timeout, stdin=b""):
    """The Docker client with a closed environment and no shell; `subprocess.TimeoutExpired` propagates."""
    try:
        return subprocess.run([engine_path(), *args], input=stdin, capture_output=True, timeout=timeout,
                              env=engine_env(), close_fds=True)
    except OSError as exc:
        raise EngineError(Code.ENGINE_FAILED, f"docker: {type(exc).__name__}") from None


def engine_popen(args, stdout, stderr):
    """The Docker client started in the background (the watched calls of the engine), with the same
    closed environment as `engine_run`; stdin is a pipe."""
    try:
        return subprocess.Popen([engine_path(), *args], stdin=subprocess.PIPE, stdout=stdout, stderr=stderr,
                                env=engine_env(), close_fds=True)
    except OSError as exc:
        raise EngineError(Code.ENGINE_FAILED, f"docker: {type(exc).__name__}") from None


def container_name(attempt=None):
    """A name per call: pid, random part and counter for the bench; `mdf-eng-<attempt>-<n>` for a
    conversion of the engine, known before the start (K7). A timed-out container is removed by name."""
    if attempt is not None:
        return f"mdf-eng-{attempt}-{next(_counter)}"
    return f"mdf-bench-{os.getpid()}-{secrets.token_hex(4)}-{next(_counter)}"


def remove_container(name):
    """Remove a container that may have outlived its client; True when it no longer exists."""
    try:
        engine_run(["rm", "-f", name], 60)
        proc = engine_run(["container", "inspect", name], 60)
    except (subprocess.TimeoutExpired, EngineError):
        return False
    return proc.returncode != 0 and b"no such container" in proc.stderr.lower()


def _check_local():
    host = os.environ.get("DOCKER_HOST")
    if host is None:
        proc = engine_run(["context", "inspect", "--format", "{{.Endpoints.docker.Host}}"], 60)
        if proc.returncode != 0:
            raise EngineError(Code.TOOLCHAIN_MISSING, "docker context unreadable")
        host = proc.stdout.decode("utf-8", "replace").strip()
    if not host.startswith("unix://"):
        raise EngineError(Code.TOOLCHAIN_MISSING, "the docker endpoint is not a local socket")


def _inspect(ref):
    proc = engine_run(["image", "inspect", "--format", "{{json .}}", ref], 60)
    if proc.returncode != 0:
        return None
    return json.loads(proc.stdout)


def config_of(data):
    config = data.get("Config") or {}
    return {"Env": config.get("Env") or [], "Entrypoint": config.get("Entrypoint"), "User": config.get("User") or ""}


def _manifest(image_id, command, timeout=120, root=False):
    """stdout of a read-only command in the image (the content manifest, stored or regenerated).
    The regeneration runs as root of the container, still without capabilities, network or a
    writable root: some files of the conda environment are readable by their owner only."""
    name = container_name()
    user = "0:0" if root else f"{os.getuid()}:{os.getgid()}"
    args = ["run", "--rm", "-i", "--name", name, *ISOLATION, "--user", user,
            "--tmpfs", "/tmp:rw,nosuid,nodev,size=64m", "--memory", "512m", "--memory-swap", "512m", image_id, *command]
    try:
        proc = engine_run(args, timeout)
    except subprocess.TimeoutExpired:
        fatal = not remove_container(name)
        raise EngineError(Code.TOOLCHAIN_MISMATCH, "content manifest: timeout", fatal=fatal) from None
    if proc.returncode != 0:
        raise EngineError(Code.TOOLCHAIN_MISMATCH, f"content manifest: exit {proc.returncode}")
    return proc.stdout


def verify(ref):
    """Image ID of `ref` when it is the reference toolchain (config and content manifest)."""
    data = _inspect(ref)
    if data is None:
        raise EngineError(Code.TOOLCHAIN_MISSING, f"no image {ref}")
    image_id = data.get("Id", "")
    if not IMAGE_ID.fullmatch(image_id):
        raise EngineError(Code.TOOLCHAIN_MISMATCH, "image id")
    if config_of(data) != spec()["config"]:
        raise EngineError(Code.TOOLCHAIN_MISMATCH, "image config")
    if hashlib.sha256(_manifest(image_id, ["cat", MANIFEST_PATH])).hexdigest() != identity():
        raise EngineError(Code.TOOLCHAIN_MISMATCH, "content manifest")
    return image_id


def _engine_version():
    proc = engine_run(["version", "--format", "{{.Server.Version}}"], 60)
    return proc.stdout.decode("utf-8", "replace").strip() if proc.returncode == 0 else "unknown"


def image():
    """ID of the verified reference image; verified once per process, at the first tool call."""
    if _state["image"] is None:
        engine_path()
        _check_local()
        image_id = verify(tag())
        _state.update(image=image_id, engine=_engine_version())
    return _state["image"]


# --- prepare --toolchain ----------------------------------------------------------------------

def _sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_context(directory):
    """The build context of the checkout against `toolchain.json` (E4): exactly the recorded files,
    regular files with the recorded SHA-256. Returns {name: sha256}."""
    directory = Path(directory)
    if not directory.is_dir():
        raise EngineError(Code.TOOLCHAIN_MISSING, "build context not found: run at the root of the checkout or pass --context")
    found = {}
    for entry in sorted(directory.iterdir()):
        if entry.is_symlink() or not entry.is_file():
            raise EngineError(Code.TOOLCHAIN_MISMATCH, f"build context: {entry.name} is not a regular file")
        found[entry.name] = _sha256_file(entry)
    expected = spec()["context"]
    if found != expected:
        names = sorted(set(found) ^ set(expected)) or sorted(k for k in found if found[k] != expected[k])
        raise EngineError(Code.TOOLCHAIN_MISMATCH, "build context differs from toolchain.json: " + ", ".join(names))
    return found


def ensure_builder(out):
    """The BuildKit builder of the toolchain: a docker-container builder with the recorded image and
    memory cap, created when absent; one with other options is refused."""
    b = spec()["builder"]
    proc = engine_run(["buildx", "inspect", BUILDER], 120)
    if proc.returncode != 0:
        create = engine_run(["buildx", "create", "--name", BUILDER, "--driver", "docker-container",
                             "--driver-opt", "image=" + b["image"], "--driver-opt", "memory=" + b["memory"],
                             "--driver-opt", "memory-swap=" + b["memory"]], 300)
        if create.returncode != 0:
            raise EngineError(Code.ENGINE_FAILED, "buildx create: " + create.stderr.decode("utf-8", "replace")[-300:])
        out(f"toolchain builder created {BUILDER}")
        proc = engine_run(["buildx", "inspect", BUILDER], 120)
    text = proc.stdout.decode("utf-8", "replace")
    driver = re.search(r"^Driver:\s+(\S+)\s*$", text, re.M)
    options = re.search(r"^Driver Options:\s*(.*)$", text, re.M)
    have = set(options.group(1).split()) if options else set()
    want = {f'image="{b["image"]}"', f'memory="{b["memory"]}"', f'memory-swap="{b["memory"]}"'}
    if not driver or driver.group(1) != "docker-container" or not want <= have:
        raise EngineError(Code.TOOLCHAIN_MISMATCH, f"builder {BUILDER} has other options: remove it with "
                                                  f"`docker buildx rm {BUILDER}`")
    out(f"toolchain builder ok {BUILDER} memory {b['memory']}")


def build(context_dir, log_dir, no_cache=False):
    """Build the image from a private copy of the verified context; returns its image ID."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="mdf-bench-toolchain-") as tmp:
        ctx, meta = Path(tmp) / "context", Path(tmp) / "meta"
        ctx.mkdir()
        meta.mkdir()
        expected = spec()["context"]
        for name in expected:
            target = ctx / name
            shutil.copyfile(Path(context_dir) / name, target)
            if _sha256_file(target) != expected[name]:
                raise EngineError(Code.TOOLCHAIN_MISMATCH, f"build context changed while copying: {name}")
            # the same mode and time for every copy: BuildKit keys its cache on them, whatever the checkout
            os.chmod(target, 0o644)
            os.utime(target, (CONTEXT_MTIME, CONTEXT_MTIME))
        iid = meta / "iid"
        args = ["buildx", "build", "--builder", BUILDER, "--load", "--provenance=false", "--sbom=false",
                "--progress=plain", "--iidfile", str(iid), *(["--no-cache"] if no_cache else []),
                "-f", str(ctx / "Dockerfile"), str(ctx)]
        try:
            proc = engine_run(args, 3600)
        except subprocess.TimeoutExpired:
            raise EngineError(Code.ENGINE_FAILED, "build: timeout") from None
        log = log_dir / f"build-{time.strftime('%Y%m%dT%H%M%S')}.log"
        log.write_bytes(proc.stdout + proc.stderr)
        if proc.returncode != 0 or not iid.is_file():
            raise EngineError(Code.ENGINE_FAILED, f"build failed (exit {proc.returncode}), log {log}")
        return iid.read_text().strip(), log


def full_verify(image_id):
    """Every file of the image against its stored content manifest, and the manifest against
    `toolchain.json`; returns the number of manifest lines."""
    data = _inspect(image_id)
    if data is None or data.get("Id") != image_id:
        raise EngineError(Code.TOOLCHAIN_MISSING, "built image not found")
    if config_of(data) != spec()["config"]:
        raise EngineError(Code.TOOLCHAIN_MISMATCH, f"image config {config_of(data)}")
    stored = _manifest(image_id, ["cat", MANIFEST_PATH])
    regenerated = _manifest(image_id, [CONTENT_SCRIPT], timeout=900, root=True)
    if stored != regenerated:
        raise EngineError(Code.TOOLCHAIN_MISMATCH, "the files of the image differ from its content manifest")
    content = hashlib.sha256(stored).hexdigest()
    if content != identity():
        raise EngineError(Code.TOOLCHAIN_MISMATCH, f"content manifest {content}, toolchain.json {identity()}")
    return stored.count(b"\n")


def prepare(context_dir, cache, no_cache=False, out=print, selftest=None):
    """`mdf prepare --toolchain`: verify the context, build, verify the image file by file, run
    `selftest(cache, out)` through the runner on the image just built, then tag it. The self-test
    is the bench's (`bench/selftest.py`), passed by the caller: the engine imports nothing of the
    bench. `cache.root / "toolchain"` receives the build logs."""
    engine_path()
    _check_local()
    files = verify_context(context_dir)
    out(f"toolchain context ok {len(files)} files ({', '.join(files)})")
    if selftest is None:
        raise ValueError("prepare needs the self-test of the toolchain")
    ensure_builder(out)
    start = time.monotonic()
    image_id, log = build(context_dir, cache.root / "toolchain", no_cache=no_cache)
    out(f"toolchain build ok {image_id} in {time.monotonic() - start:.0f} s{' (no cache)' if no_cache else ''}")
    lines = full_verify(image_id)
    out(f"toolchain content ok {identity()} ({lines} manifest lines, every file verified)")
    out(f"toolchain reference build {'yes' if image_id == spec()['image_id'] else 'no (same content, another build)'}")
    _state.update(image=image_id, engine=_engine_version())
    try:
        selftest(cache, out)
    except BaseException:
        reset()
        raise
    proc = engine_run(["tag", image_id, tag()], 60)
    if proc.returncode != 0:
        reset()
        raise EngineError(Code.ENGINE_FAILED, "docker tag")
    out(f"toolchain ready {tag()} {image_id} docker {_state['engine']}")
    return image_id


# --- `python -m markdown_for_llms.engine.toolchain verify` (precondition X of the full check) ----

def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv != ["verify"]:
        print("usage: python -m markdown_for_llms.engine.toolchain verify", file=sys.stderr)
        return 2
    try:
        image()
    except EngineError as exc:
        print(f"toolchain {exc.code.value}: {exc.detail}; {HINTS.get(exc.code, '')}")
        return 1
    r = report()
    print(f"toolchain ok content {r['content_sha256'][:16]} image {r['image_id']} reference build "
          f"{'yes' if r['reference_image'] else 'no'} docker {r['engine']['version']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
