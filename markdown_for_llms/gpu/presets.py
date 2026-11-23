"""GPU presets (MFL023 A3): where and how the inference of the marker-server route runs on a rented GPU.

A preset is an entry of the versioned file `data/presets.json`, with a closed schema (`schema: 1`): a field more or
less refuses the file (`gpu_preset_invalid`). Fields: `status` (`supported`, `experimental`, `superseded`);
`superseded_by` and `reason`, strings only with `superseded`, else null; `provider` (`runpod`, `modal`); `mode`
(`serverless`); `data_center` (null for Modal, which chooses the region); `volume` {`type` `network` or `global`,
`class` `STANDARD`, `HIGH_PERFORMANCE` or null (Modal), `name`, `size_gb` or null (Modal)}; `gpu` {`group`, a string or
a list in the order of preference, `memory_gb`, `observed`}; `image` (by digest); `model` {`id`, `revision`, `vllm`};
`server` (`vllm`); `parallel`; `rate` {`usd_per_s`, `source`, `date`}; `levers` {`compile_cache`: the SHA-256 of the
archives of the compile cache, `arm_ahead_s`: seconds or null}, never another lever (the experimental `snapshot` and
`flashboot` stay on the command line); `seconds` {`ready_within`, `read_timeout`, `page_seconds`, `cold_start_s`,
`w1_s`, `tail_idle_s`, `tail_fault_s`, `idle_s`}; `measures` (the measures shown before a conversion, with their date,
hardware and source).

A preset changes only by review of this file: data center, volume or GPU different from a preset are a new preset
(D3 of MFL023). A `superseded` preset is read and listed, never enabled; a `global` volume is recognized and refused
(`gpu_volume_unsupported`, run 4 of the GPU sprint); only `supported` presets on a `network` volume can be enabled.
Standard library only: the API process imports this module.
"""
from decimal import Decimal, InvalidOperation
import json
from pathlib import Path
import re

from .codes import (GPU_PRESET_INVALID, GPU_PRESET_NOT_ENABLED, GPU_PRESET_SUPERSEDED, GPU_PRESET_UNKNOWN,
                    GPU_VOLUME_UNSUPPORTED, GpuError)

PATH = Path(__file__).resolve().parent / "data" / "presets.json"
SCHEMA = 1
NAME = re.compile(r"[a-z0-9][a-z0-9-]{0,63}")
STATUSES = ("supported", "experimental", "superseded")
PROVIDERS = ("runpod", "modal")
VOLUME_TYPES = ("network", "global")
VOLUME_CLASSES = ("STANDARD", "HIGH_PERFORMANCE")
SECONDS = ("ready_within", "read_timeout", "page_seconds", "cold_start_s", "w1_s", "tail_idle_s", "tail_fault_s",
           "idle_s")
FIELDS = ("status", "superseded_by", "reason", "provider", "mode", "data_center", "volume", "gpu", "image", "model",
          "server", "parallel", "rate", "levers", "seconds", "measures")
MEASURE = ("what", "value", "date", "hardware", "source")
SHA256 = re.compile(r"[0-9a-f]{64}")
IMAGE = re.compile(r"[a-z0-9][a-z0-9./_-]{0,200}@sha256:[0-9a-f]{64}")
REVISION = re.compile(r"[0-9a-f]{40}")
VERSION = re.compile(r"[0-9]+(\.[0-9]+){1,3}")
DATE = re.compile(r"20[0-9]{2}-[01][0-9]-[0-3][0-9]")
MAX_BYTES = 1 << 20


def _bad(detail):
    return GpuError(GPU_PRESET_INVALID, detail)


def _text(value, what, limit=300):
    if not isinstance(value, str) or not value or len(value) > limit or any(c in value for c in "\r\n\0"):
        raise _bad(what)
    return value


def _whole(value, what, low=1, high=86400):
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise _bad(what)
    return value


def _keys(value, keys, what):
    if not isinstance(value, dict) or set(value) != set(keys):
        raise _bad(what + ": the fields " + ", ".join(keys))
    return value


def check_preset(name, p, names):
    """Validates the preset `name` against the closed schema; `names` are the presets of the file."""
    if not isinstance(name, str) or not NAME.fullmatch(name):
        raise _bad("preset name")
    _keys(p, FIELDS, name)
    if p["status"] not in STATUSES:
        raise _bad(name + ": status")
    if p["status"] == "superseded":
        if p["superseded_by"] not in names or p["superseded_by"] == name:
            raise _bad(name + ": superseded_by names another preset of the file")
        _text(p["reason"], name + ": reason")
    elif p["superseded_by"] is not None or p["reason"] is not None:
        raise _bad(name + ": superseded_by and reason only with the status superseded")
    if p["provider"] not in PROVIDERS:
        raise _bad(name + ": provider")
    if p["mode"] != "serverless":
        raise _bad(name + ": mode serverless")
    modal = p["provider"] == "modal"
    if modal and p["data_center"] is not None:
        raise _bad(name + ": Modal chooses the region: data_center null")
    if not modal:
        _text(p["data_center"], name + ": data_center", 64)
    vol = _keys(p["volume"], ("type", "class", "name", "size_gb"), name + ": volume")
    if vol["type"] not in VOLUME_TYPES:
        raise _bad(name + ": volume.type network or global")
    _text(vol["name"], name + ": volume.name", 64)
    if modal:
        if vol["class"] is not None or vol["size_gb"] is not None:
            raise _bad(name + ": a Modal volume has no class and no size: null")
    else:
        if vol["class"] not in VOLUME_CLASSES:
            raise _bad(name + ": volume.class STANDARD or HIGH_PERFORMANCE")
        _whole(vol["size_gb"], name + ": volume.size_gb", 1, 100000)
    gpu = _keys(p["gpu"], ("group", "memory_gb", "observed"), name + ": gpu")
    group = gpu["group"] if isinstance(gpu["group"], list) else [gpu["group"]]
    if not group or len(set(group)) != len(group):
        raise _bad(name + ": gpu.group")
    for g in group:
        _text(g, name + ": gpu.group", 64)
    _whole(gpu["memory_gb"], name + ": gpu.memory_gb", 1, 1024)
    if not isinstance(gpu["observed"], list):
        raise _bad(name + ": gpu.observed")
    for g in gpu["observed"]:
        _text(g, name + ": gpu.observed", 64)
    if not isinstance(p["image"], str) or not IMAGE.fullmatch(p["image"]):
        raise _bad(name + ": image by digest")
    model = _keys(p["model"], ("id", "revision", "vllm"), name + ": model")
    _text(model["id"], name + ": model.id", 200)
    if not isinstance(model["revision"], str) or not REVISION.fullmatch(model["revision"]):
        raise _bad(name + ": model.revision")
    if not isinstance(model["vllm"], str) or not VERSION.fullmatch(model["vllm"]):
        raise _bad(name + ": model.vllm")
    if p["server"] != "vllm":
        raise _bad(name + ": server vllm")
    _whole(p["parallel"], name + ": parallel", 1, 64)
    rate = _keys(p["rate"], ("usd_per_s", "source", "date"), name + ": rate")
    try:
        usd = Decimal(rate["usd_per_s"]) if isinstance(rate["usd_per_s"], str) else None
    except InvalidOperation:
        usd = None
    if usd is None or not usd.is_finite() or not 0 < usd < 1:
        raise _bad(name + ": rate.usd_per_s, a decimal string in USD per second")
    _text(rate["source"], name + ": rate.source")
    if not isinstance(rate["date"], str) or not DATE.fullmatch(rate["date"]):
        raise _bad(name + ": rate.date")
    levers = _keys(p["levers"], ("compile_cache", "arm_ahead_s"), name + ": levers (no other lever)")
    cache = levers["compile_cache"]
    if not isinstance(cache, list) or len(set(map(str, cache))) != len(cache) or any(
            not isinstance(x, str) or not SHA256.fullmatch(x) for x in cache):
        raise _bad(name + ": levers.compile_cache, SHA-256 of the archives")
    if levers["arm_ahead_s"] is not None:
        _whole(levers["arm_ahead_s"], name + ": levers.arm_ahead_s", 1, 3600)
    sec = _keys(p["seconds"], SECONDS, name + ": seconds")
    for key in SECONDS:
        _whole(sec[key], "%s: seconds.%s" % (name, key))
    if sec["ready_within"] < sec["w1_s"]:
        raise _bad(name + ": seconds.ready_within is at least w1_s")
    if not isinstance(p["measures"], list):
        raise _bad(name + ": measures")
    for m in p["measures"]:
        _keys(m, MEASURE, name + ": a measure")
        for key in MEASURE:
            _text(m[key], "%s: measures.%s" % (name, key))
        if not DATE.fullmatch(m["date"]):
            raise _bad(name + ": measures.date")
    return p


def check(doc):
    """The presets of the document `doc` ({"schema": 1, "presets": {...}}), validated; `gpu_preset_invalid`."""
    _keys(doc, ("schema", "presets"), "presets file")
    if doc["schema"] != SCHEMA or isinstance(doc["schema"], bool):
        raise _bad("presets file: schema %d" % SCHEMA)
    if not isinstance(doc["presets"], dict) or not doc["presets"]:
        raise _bad("presets file: presets")
    names = set(doc["presets"])
    return {name: check_preset(name, p, names) for name, p in doc["presets"].items()}


def load(path=None):
    """The validated presets of the versioned file (or of `path`), {name: preset}."""
    path = PATH if path is None else Path(path)
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise _bad("presets file: %s" % type(exc).__name__) from None
    if len(data) > MAX_BYTES:
        raise _bad("presets file too large")
    try:
        doc = json.loads(data)
    except ValueError:
        raise _bad("presets file: not JSON") from None
    return check(doc)


def refusal(preset):
    """The code that keeps `preset` from being enabled, or None for a `supported` preset on a network volume."""
    if preset["status"] == "superseded":
        return GPU_PRESET_SUPERSEDED
    if preset["volume"]["type"] != "network":
        return GPU_VOLUME_UNSUPPORTED
    if preset["status"] != "supported":
        return GPU_PRESET_NOT_ENABLED
    return None


def enable(names, presets=None):
    """The presets `names` (a list) checked for an instance: each known, `supported`, on a network volume, and named
    once; GpuError with the code of the first refusal."""
    presets = load() if presets is None else presets
    if len(set(names)) != len(names):
        raise GpuError(GPU_PRESET_INVALID, "a preset enabled twice")
    for name in names:
        if name not in presets:
            raise GpuError(GPU_PRESET_UNKNOWN)
        code = refusal(presets[name])
        if code is not None:
            raise GpuError(code, name)
    return list(names)


def gpu_text(preset):
    """The `gpu` of an endpoint: the group as servers.json writes it (`AMPERE_24`, `L4,A10G`)."""
    group = preset["gpu"]["group"]
    return ",".join(group) if isinstance(group, list) else group


def endpoint_fields(preset):
    """The fields of an endpoint of `servers.json` that the preset fixes (D3): every value but `kind`, `url`,
    `key_env` and `resource`, so that a field written by hand next to `preset` is refused."""
    fields = {"server": preset["server"], "model": preset["model"]["id"], "revision": preset["model"]["revision"],
              "pins": {"vllm": preset["model"]["vllm"]}, "parallel": preset["parallel"], "gpu": gpu_text(preset),
              "rate_usd_per_s": preset["rate"]["usd_per_s"], "rate_source": preset["rate"]["source"],
              "rate_date": preset["rate"]["date"]}
    fields.update({k: preset["seconds"][k] for k in SECONDS if k != "idle_s"})
    return fields


def endpoint_table(presets=None):
    """The table the engine receives to expand an endpoint of `servers.json` that names a preset, as a plain dict (the
    engine depends on nothing of this package): {name: {"provider", "refusal", "fields"}}, `refusal` the code that keeps
    the preset from being used, or None."""
    presets = load() if presets is None else presets
    return {name: {"provider": p["provider"], "refusal": refusal(p), "fields": endpoint_fields(p)}
            for name, p in presets.items()}


def compile_caches(presets=None):
    """The SHA-256 of the compile caches of the supported presets, in the order of the file (MFL049 d)."""
    presets = load() if presets is None else presets
    return [sha for p in presets.values() if p["status"] == "supported" for sha in p["levers"]["compile_cache"]]


def public(name, preset):
    """What the API and the Web UI show of a preset: no secret exists in it; the whole entry and its name."""
    return dict(preset, name=name)
