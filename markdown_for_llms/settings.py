"""Strict administrative settings and a closed per-job conversion profile."""
from dataclasses import asdict, dataclass, field, fields
from decimal import Decimal, InvalidOperation
from pathlib import Path
import hashlib
import io
import json
import os


class ConfigurationError(ValueError):
    """An invalid configuration; messages never contain supplied values. `code`: a closed code when
    there is one (the GPU presets, MFL023)."""

    def __init__(self, message, code=None):
        super().__init__(message)
        self.code = code


# Operational time limits change neither quotas, paths nor stored data: they are
# recorded per run in the manifest and stay out of the storage identity (ADR 0011).
TIME_LIMITS = frozenset({"timeout_seconds", "stall_seconds"})
# Identity schema 3 (MFL032 tranche C): the port (MFL021), the model root (the weights are
# identified by hash in every engine.json, D10) and the Datalab settings are operational too.
# Schemas 1 and 2 are frozen in legacy_identity.py.
# MFL023: the settings of the rented GPUs are operational too (the presets enabled, the endpoints, the
# spend ledger of the instance and its caps), so an archive of schema 3 still opens.
GPU = frozenset({"gpu_presets", "servers", "gpu_ledger_dir", "gpu_cap_usd_modal", "gpu_cap_usd_runpod",
                 "gpu_session_cap_usd"})
OPERATIONAL = TIME_LIMITS | {"port", "model_root", "datalab_cap_usd"} | GPU
# Paths of the machine: never in `public()`, which goes into manifest.json, the answers and the ZIPs.
PATHS = frozenset({"data_root", "model_root", "servers", "gpu_ledger_dir"})
AMOUNTS = ("datalab_cap_usd", "gpu_cap_usd_modal", "gpu_cap_usd_runpod", "gpu_session_cap_usd")
IDENTITY_SCHEMA = 3
PROFILE = "engine-v1"


def default_model_root():
    """`${XDG_CACHE_HOME:-~/.cache}/markdown-for-llms/models` (D10 of MFL032), the default of
    `engine.weights.default_root`."""
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return str(Path(base) / "markdown-for-llms" / "models")


def default_servers():
    """`${XDG_CONFIG_HOME:-~/.config}/markdown-for-llms/servers.json`, the default of
    `engine.servers.default_path` (written here: the API process does not import that module)."""
    base = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return str(Path(base) / "markdown-for-llms" / "servers.json")


def default_gpu_ledger():
    """`${XDG_DATA_HOME:-~/.local/share}/markdown-for-llms/gpu-ledger` (MFL023, C-B3 of the plan r1): the
    spend ledger of the instance outside the archive, so it never counts against MDF_STORAGE_MIB and
    recreating the archive keeps the record of the caps."""
    base = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
    return str(Path(base) / "markdown-for-llms" / "gpu-ledger")


def overlap(a, b):
    return Path(a).is_relative_to(b) or Path(b).is_relative_to(a)


@dataclass(frozen=True)
class Settings:
    data_root: str = "./mdf-data"
    model_root: str = field(default_factory=default_model_root)
    host: str = "127.0.0.1"
    port: int = 23465
    timeout_seconds: int = 0          # total deadline of a conversion; 0 = none (ADR 0011)
    stall_seconds: int = 900          # stop a conversion without engine progress for this long
    memory_mib: int = 16384
    upload_mib: int = 50
    max_pages: int = 50
    output_mib: int = 512
    log_mib: int = 10
    storage_mib: int = 20480
    reserve_mib: int = 1024
    max_pixels: int = 25000000
    max_dimension: int = 16000
    threads: int = 4
    poll_seconds: int = 1
    lease_seconds: int = 15
    grace_seconds: int = 5
    process_limit: int = 512
    # the profile of the storage identity: the conversions run on the engine, which records
    # the versions of its routes in every engine.json
    bind_profile: str = PROFILE
    review_provider: str = "disabled"
    # spending cap of one Datalab conversion, in USD (0: nothing is sent); the key is never a
    # setting: it is DATALAB_API_KEY in the environment of the worker
    datalab_cap_usd: str = "0"
    # The rented GPUs (MFL023): the GPU presets enabled (comma-separated names of
    # markdown_for_llms/gpu/data/presets.json, only supported ones on a network volume), the
    # servers.json of the instance, the folder of its spend ledger (spend.jsonl, its lock and the
    # worker logs) and the caps in USD: of the ledger for each provider (0: the presets of that
    # provider are not available) and of one session. The keys are never settings.
    gpu_presets: str = ""
    servers: str = field(default_factory=default_servers)
    gpu_ledger_dir: str = field(default_factory=default_gpu_ledger)
    gpu_cap_usd_modal: str = "0"
    gpu_cap_usd_runpod: str = "0"
    gpu_session_cap_usd: str = "0"

    def __post_init__(self):
        for field in fields(self):
            value = getattr(self, field.name)
            low = 0 if field.name == "timeout_seconds" else 1
            if field.type is int and (type(value) is not int or not low <= value <= 2**31 - 1):
                raise ConfigurationError("Invalid positive integer: " + field.name)
        if self.stall_seconds < 60:
            raise ConfigurationError("Stall limit below 60 seconds: stall_seconds")
        if self.host != "127.0.0.1" or self.port > 65535:
            raise ConfigurationError("Only IPv4 loopback and valid ports are supported")
        if self.bind_profile != PROFILE or self.review_provider != "disabled":
            raise ConfigurationError("Unsupported provider/profile")
        for name in AMOUNTS:
            value = getattr(self, name)
            try:
                cap = Decimal(value) if isinstance(value, str) else None
            except InvalidOperation:
                cap = None
            if cap is None or not cap.is_finite() or cap < 0 or len(value) > 32:
                raise ConfigurationError("Invalid amount: %s (a finite amount in USD, 0 or more)" % name)
        if not isinstance(self.gpu_presets, str) or len(self.gpu_presets) > 1024:
            raise ConfigurationError("Invalid setting: gpu_presets", "gpu_preset_invalid")
        names = [n.strip() for n in self.gpu_presets.split(",") if n.strip()]
        if names:
            from .gpu import codes, presets
            try:
                presets.enable(names)
            except codes.GpuError as exc:
                raise ConfigurationError("GPU preset not available for an instance: gpu_presets", exc.code) from None
        object.__setattr__(self, "gpu_presets", ",".join(names))
        if self.lease_seconds < 3 * self.poll_seconds or self.threads > 64:
            raise ConfigurationError("Inconsistent lease/poll/thread settings")
        for name in ("data_root", "model_root", "servers", "gpu_ledger_dir"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value or "\0" in value:
                raise ConfigurationError("Invalid path: " + name)
            # `~` is the home of the user for the operational paths, as in bench/settings.py (SOL-A-N2 of the review
            # of A); data_root, part of the identity, keeps its literal value
            path = Path(value) if name == "data_root" else Path(value).expanduser()
            object.__setattr__(self, name, str(path.absolute()))
        if overlap(self.data_root, self.model_root):
            raise ConfigurationError("Data and model directories must not overlap")
        if overlap(self.gpu_ledger_dir, self.data_root) or overlap(self.gpu_ledger_dir, self.model_root):
            raise ConfigurationError("The GPU spend ledger must be outside the data and model directories")
        if Path(self.servers).is_relative_to(self.data_root):
            raise ConfigurationError("The servers file must be outside the data directory")
        if self.storage_mib <= self.upload_mib + 2 * self.output_mib + self.log_mib:
            raise ConfigurationError("Storage quota too small for one attempt")

    @classmethod
    def load(cls, env_file=None, overrides=None, environ=None):
        from dotenv import dotenv_values
        known = {"MDF_" + f.name.upper(): f for f in fields(cls)}
        values = {}
        sources = []
        if env_file is not None:
            if not Path(env_file).is_file():
                raise ConfigurationError("Explicit environment file unavailable")
            sources.append(dotenv_values(stream=io.StringIO(settings_lines(env_file)), interpolate=False))
        sources.append(os.environ if environ is None else environ)
        for source in sources:
            for key, value in source.items():
                if not key.startswith("MDF_") or key.startswith("MDF_BENCH_") or key == "MDF_SESSION_ARM":
                    # MDF_BENCH_* belong to `mdf bench` (markdown_for_llms/bench/settings.py); MDF_SESSION_ARM
                    # is the window of a live session of scripts/servers/session.py (MFL033), not a setting
                    continue
                if key not in known:
                    raise ConfigurationError("Unknown MDF setting")
                f = known[key]
                try:
                    values[f.name] = int(value) if f.type is int else value
                except (ValueError, TypeError):
                    raise ConfigurationError("Invalid setting type: " + f.name) from None
        if overrides:
            if set(overrides) - {f.name for f in fields(cls)}:
                raise ConfigurationError("Unknown administrative argument")
            values.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**values)

    def engine_limits(self, max_pages=None):
        """The limits of `engine.convert` from these settings: the same for `mdf convert` and the
        worker (FC4 of the review of B of MFL032)."""
        return {"memory_mib": self.memory_mib, "output_mib": self.output_mib, "timeout_seconds": self.timeout_seconds,
                "max_pages": self.max_pages if max_pages is None else max_pages, "poll_seconds": self.poll_seconds,
                "threads": self.threads, "stall_seconds": self.stall_seconds}

    def public(self):
        return {k: v for k, v in asdict(self).items() if k not in PATHS}

    def gpu_presets_enabled(self):
        """The names of the GPU presets this instance enables, in their order."""
        return [n for n in self.gpu_presets.split(",") if n]

    def identity(self):
        """Storage identity schema 3: every setting except the operational ones."""
        value = {k: v for k, v in asdict(self).items() if k not in OPERATIONAL}
        value["identity_schema"] = IDENTITY_SCHEMA
        return digest(value)


def settings_lines(env_file):
    """The lines of `env_file` that set an `MDF_` name (MFL023, C-NB2 of the plan r1): the other lines,
    keys included, are dropped before the parser reads the text, so the settings never hold a key."""
    try:
        text = Path(env_file).read_text(encoding="utf-8")
    except UnicodeDecodeError:
        raise ConfigurationError("Environment file is not UTF-8 text") from None
    keep = []
    for line in text.splitlines():
        name = line.strip()
        if name.startswith("export "):
            name = name[7:].lstrip()
        if name.startswith("MDF_"):
            keep.append(line)
    return "\n".join(keep) + "\n"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
