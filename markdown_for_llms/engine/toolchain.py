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

Each process asks the launcher for the image at its first tool call (`image()`); the launcher
(`markdown_for_llms/launcher.py`, the one process with the Docker socket) verifies it before any tool
may use it: the local image tagged `mdf-bench-toolchain:<content_sha256[:16]>` must exist
(`toolchain_missing`) and have the recorded config and a content manifest with the recorded SHA-256
(`toolchain_mismatch`); it is then started by `sha256:<image id>`, never by tag. Without the
launcher: `isolation_unavailable`. A bench process that finds everything it needs in the cache starts
no container and reports `engine: null` (U4: a cache prepared elsewhere and copied). Docker is the
only engine, through the local socket the launcher holds: a remote endpoint would receive the
documents.

`prepare_candidate` is the part of `mdf prepare --toolchain` that runs in a container: the image the
host built and tagged `mdf-bench-toolchain:candidate` is verified file by file (its stored content
manifest against the one regenerated in it, both read by `manifest` children of the launcher) and
the self-test runs on it; the host tags it afterwards. No container has a Docker client: the build is
the host's, `scripts/prepare-toolchain.sh` (`make prepare`), which copies the build context, has it verified
here (`python -m markdown_for_llms.engine.toolchain context DIR`, in a container without network), builds
it with the capped builder and tags the result `mdf-bench-toolchain:candidate`.
"""
import functools
import hashlib
import json
from pathlib import Path
import re
import sys

from . import containers
from .codes import Code, EngineError

ENGINE = "docker"                    # the name of the engine in the reports
REPOSITORY = "mdf-bench-toolchain"
BUILDER = "mdf-bench-toolchain"      # the capped builder; scripts/prepare-toolchain.sh and the Makefile name it too
MANIFEST_PATH = "/etc/mdf-bench-toolchain.sha256"
CONTENT_SCRIPT = "/usr/local/bin/mdf-bench-content-manifest"
SCHEMA = "mdf-bench-toolchain/1"
DEFAULT_CONTEXT = Path("scripts") / "bench-toolchain"
# 2026-01-01T00:00:00Z, the time of every file of the context copied by scripts/prepare-toolchain.sh (mode 0644):
# BuildKit keys its cache on mode and time, whatever the checkout
CONTEXT_MTIME = 1767225600
SHA256 = re.compile(r"[0-9a-f]{64}")
IMAGE_ID = re.compile(r"sha256:[0-9a-f]{64}")
HINTS = {
    Code.TOOLCHAIN_MISSING: "build the toolchain image with `make prepare` at the root of the checkout (Docker Engine "
                            "on Linux)",
    Code.TOOLCHAIN_MISMATCH: "the reference toolchain image or its build context differs from toolchain.json: rebuild "
                             "it with `make prepare` from a clean checkout",
    Code.ISOLATION_UNAVAILABLE: "the launcher does not answer: start the product with `make up`",
}

_state = {"image": None, "engine": None, "role": "toolchain"}
MANIFEST_LIMITS = {"memory_mib": 512, "pids": 256, "fsize_mib": 64, "nofile": 1024, "tmpfs_mib": 64}


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
    _state.update(image=None, engine=None, role="toolchain")


def role():
    """The image role of the tool calls of this process: `toolchain`, or `candidate` while `prepare_candidate`
    runs the self-test on the image the host just built."""
    return _state["role"]


def _ask(call):
    """An operation of the launcher with its refusals as the closed codes of the engine."""
    try:
        return call(containers.launcher())
    except containers.Unreachable as exc:
        raise EngineError(Code.ISOLATION_UNAVAILABLE, "launcher: %s" % exc) from None
    except containers.Refused as exc:
        code = next((c for c in Code if c.value == exc.code), Code.ENGINE_FAILED)
        raise EngineError(code, exc.detail) from None


def config_of(data):
    config = data.get("Config") or {}
    return {"Env": config.get("Env") or [], "Entrypoint": config.get("Entrypoint"), "User": config.get("User") or ""}


def _manifest(image_role, command, timeout=120):
    """stdout of one of the two commands of a `manifest` child of the launcher in the image of `image_role`: the
    stored content manifest (`containers.MANIFEST_READ`), or the one regenerated from the files of the image
    (`containers.MANIFEST_REGENERATE`), which the launcher runs as root of the container, still without
    capabilities, network or any writable volume: some files of the conda environment are readable by their
    owner only."""
    limits = dict(MANIFEST_LIMITS, lease_seconds=containers.LEASE_SECONDS,
                  max_seconds=min(86400, int(timeout) + containers.TERM_MARGIN))
    with containers.open_child(None, "manifest") as child:
        outcome = child.run(image_role, list(command), {}, limits, timeout=timeout)
    if outcome.reason == "timeout" or outcome.cause == "term":
        raise EngineError(Code.TOOLCHAIN_MISMATCH, "content manifest: timeout")
    if outcome.cause != "exited" or outcome.exit != 0:
        raise EngineError(Code.TOOLCHAIN_MISMATCH, f"content manifest: exit {outcome.exit} ({outcome.cause})")
    return outcome.stdout


def image():
    """ID of the verified reference image; asked of the launcher once per process, at the first tool call."""
    if _state["image"] is None:
        answer = _ask(lambda launcher: launcher.toolchain())
        image_id = answer.get("image") or ""
        if not IMAGE_ID.fullmatch(image_id) or answer.get("verified") is not True:
            raise EngineError(Code.TOOLCHAIN_MISMATCH, "the launcher did not verify the image")
        _state.update(image=image_id, engine=str(answer.get("docker") or "unknown"), role="toolchain")
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


def full_verify(image_role="candidate"):
    """Every file of the image of `image_role` against its stored content manifest, and the manifest against
    `toolchain.json`; returns (image ID, number of manifest lines). The launcher checks the config of the image."""
    answer = _ask(lambda launcher: launcher.toolchain(candidate=image_role == "candidate"))
    image_id = answer.get("image") or ""
    if not IMAGE_ID.fullmatch(image_id):
        raise EngineError(Code.TOOLCHAIN_MISMATCH, "image id")
    stored = _manifest(image_role, containers.MANIFEST_READ)
    regenerated = _manifest(image_role, containers.MANIFEST_REGENERATE, timeout=900)
    if stored != regenerated:
        raise EngineError(Code.TOOLCHAIN_MISMATCH, "the files of the image differ from its content manifest")
    content = hashlib.sha256(stored).hexdigest()
    if content != identity():
        raise EngineError(Code.TOOLCHAIN_MISMATCH, f"content manifest {content}, toolchain.json {identity()}")
    return image_id, stored.count(b"\n"), str(answer.get("docker") or "unknown")


def prepare_candidate(cache, out=print, selftest=None):
    """The container part of `mdf prepare --toolchain`: the image the host built and tagged
    `mdf-bench-toolchain:candidate` is verified file by file, then `selftest(cache, out)` runs through the runner
    on it. The self-test is the bench's (`bench/selftest.py`), passed by the caller: the engine imports nothing
    of the bench. The last line it prints, `toolchain candidate ok <tag> <image ID>`, is what the host reads to
    give the image its tag. Returns the image ID."""
    if selftest is None:
        raise ValueError("prepare needs the self-test of the toolchain")
    image_id, lines, engine = full_verify("candidate")
    out(f"toolchain content ok {identity()} ({lines} manifest lines, every file verified)")
    out(f"toolchain reference build {'yes' if image_id == spec()['image_id'] else 'no (same content, another build)'}")
    _state.update(image=image_id, engine=engine, role="candidate")
    try:
        selftest(cache, out)
    finally:
        reset()
    out(f"toolchain candidate ok {tag()} {image_id} docker {engine}")
    return image_id


# --- `python -m markdown_for_llms.engine.toolchain verify | tag | context DIR` ------------------------

def main(argv=None):
    """`verify`: precondition X of the full check, the image through the launcher. `tag`: the tag the verified
    image has, for the host script that builds it. `context DIR`: the build context copied by that script
    against `toolchain.json`, before it is built."""
    argv = sys.argv[1:] if argv is None else argv
    if argv == ["tag"]:
        print(tag())
        return 0
    if len(argv) == 2 and argv[0] == "context":
        try:
            found = verify_context(argv[1])
        except EngineError as exc:
            print(f"toolchain {exc.code.value}: {exc.detail}")
            return 1
        print(f"toolchain context ok {len(found)} files")
        return 0
    if argv != ["verify"]:
        print("usage: python -m markdown_for_llms.engine.toolchain verify | tag | context DIR", file=sys.stderr)
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
