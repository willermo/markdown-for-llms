"""Frozen identity schemas 1 and 2 of the archive (K5 of MFL032): constants and the calculation.

Schema 1 (before ADR 0011) hashed the fields of `V1_FIELDS`, with the timeout an unset variable
used to mean; schema 2 (ADR 0011) every setting of that time except the time limits. Both held
the profile `marker-cpu-v1` and the model root, whose default was `./mdf-models`. Schema 3 (the
product on the engine) leaves the port, the model root and the Datalab settings out, like the
time limits (`settings.Settings.identity`). An archive of a previous version is recognized by
these frozen values only, for the current model root and for the former default one, then the
first exclusive opening upgrades it (`storage.Storage.upgrade`).

This module never changes with the settings and imports nothing of the engine.
"""
import hashlib
import json
from pathlib import Path

V1_FIELDS = ("data_root", "model_root", "host", "port", "timeout_seconds", "memory_mib",
             "upload_mib", "max_pages", "output_mib", "log_mib", "storage_mib", "reserve_mib",
             "max_pixels", "max_dimension", "threads", "poll_seconds", "lease_seconds",
             "grace_seconds", "process_limit", "bind_profile", "review_provider")
V2_FIELDS = tuple(name for name in V1_FIELDS if name != "timeout_seconds")
LEGACY_TIMEOUT_SECONDS = 450
LEGACY_BIND_PROFILE = "marker-cpu-v1"
LEGACY_MODEL_ROOT = "./mdf-models"
# the job options of schema 2 (`settings.options()` until tranche C of MFL032)
V2_OPTION_KEYS = frozenset({"use_llm", "force_ocr", "strip_existing_ocr", "disable_image_extraction",
                            "paginate_output", "pdftext_workers", "layout_batch_size",
                            "recognition_batch_size", "detection_batch_size", "table_rec_batch_size",
                            "ocr_error_batch_size", "lowres_image_dpi", "highres_image_dpi"})


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _value(settings, name, model_root):
    if name == "model_root":
        return model_root
    if name == "bind_profile":
        return LEGACY_BIND_PROFILE
    return getattr(settings, name)


def v1(settings, model_root):
    value = {name: _value(settings, name, model_root) for name in V1_FIELDS}
    value["timeout_seconds"] = settings.timeout_seconds or LEGACY_TIMEOUT_SECONDS
    return digest(value)


def v2(settings, model_root):
    value = {name: _value(settings, name, model_root) for name in V2_FIELDS}
    value["identity_schema"] = 2
    return digest(value)


def model_roots(settings):
    """The model roots an archive of a previous version may have been created with: the current
    one and the former default (made absolute as the settings of that time made it)."""
    roots = [settings.model_root]
    former = str(Path(LEGACY_MODEL_ROOT).absolute())
    if former not in roots:
        roots.append(former)
    return roots


def candidates(settings):
    """{identity: (schema, model_root)} of the archives of schema 1 and 2 these settings open."""
    found = {}
    for root in model_roots(settings):
        found.setdefault(v2(settings, root), (2, root))
        found.setdefault(v1(settings, root), (1, root))
    return found


def v2_options(options):
    """True when stored job options have the shape of schema 2."""
    return isinstance(options, dict) and set(options) == V2_OPTION_KEYS
