"""Read-only readiness checks; no model import or network acquisition."""
from importlib import metadata, resources
from pathlib import Path
import hashlib
import json
import shutil
import sys

from .files import directory, identify


# tqdm is pinned because the stall guard reads its bar format (ADR 0011).
PINS = {"marker-pdf": "1.10.2", "surya-ocr": "0.17.1", "torch": "2.7.1+cpu", "tqdm": "4.70.1"}


def model_inventory(root, verify=True):
    manifest = json.loads(resources.files("markdown_for_llms").joinpath("models.json").read_text())
    with directory(root):
        for row in manifest["files"]:
            if verify:
                observed = identify(root, row["path"])
                if observed != row:
                    raise ValueError("Model/font identity mismatch: " + row["path"])
            elif not (Path(root) / row["path"]).is_file():
                raise ValueError("Model/font unavailable")
    return manifest


def check_runtime(settings, verify_models=True):
    if sys.version_info[:3] != (3, 12, 13):
        raise ValueError("CPython 3.12.13 required")
    versions = {name: metadata.version(name) for name in PINS}
    if versions != PINS:
        raise ValueError("Incompatible ML runtime versions")
    if not shutil.which("bwrap"):
        raise ValueError("Bubblewrap unavailable")
    manifest = model_inventory(settings.model_root, verify_models)
    return dict(versions=versions, model_manifest_sha256=hashlib.sha256(
        json.dumps(manifest, sort_keys=True).encode()).hexdigest())
