"""Strict administrative settings and a closed per-job conversion profile."""
from dataclasses import asdict, dataclass, field, fields
from decimal import Decimal, InvalidOperation
from pathlib import Path
import hashlib
import json
import os


class ConfigurationError(ValueError):
    """An invalid configuration; messages never contain supplied values."""


# Operational time limits change neither quotas, paths nor stored data: they are
# recorded per run in the manifest and stay out of the storage identity (ADR 0011).
TIME_LIMITS = frozenset({"timeout_seconds", "stall_seconds"})
# Identity schema 3 (MFL032 tranche C): the port (MFL021), the model root (the weights are
# identified by hash in every engine.json, D10) and the Datalab settings are operational too.
# Schemas 1 and 2 are frozen in legacy_identity.py.
OPERATIONAL = TIME_LIMITS | {"port", "model_root", "datalab_cap_usd"}
IDENTITY_SCHEMA = 3
PROFILE = "engine-v1"


def default_model_root():
    """`${XDG_CACHE_HOME:-~/.cache}/markdown-for-llms/models` (D10 of MFL032), the default of
    `engine.weights.default_root`."""
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return str(Path(base) / "markdown-for-llms" / "models")


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
        try:
            cap = Decimal(self.datalab_cap_usd) if isinstance(self.datalab_cap_usd, str) else None
        except InvalidOperation:
            cap = None
        if cap is None or not cap.is_finite() or cap < 0 or len(self.datalab_cap_usd) > 32:
            raise ConfigurationError("Invalid amount: datalab_cap_usd (a finite amount in USD, 0 or more)")
        if self.lease_seconds < 3 * self.poll_seconds or self.threads > 64:
            raise ConfigurationError("Inconsistent lease/poll/thread settings")
        for name in ("data_root", "model_root"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value or "\0" in value:
                raise ConfigurationError("Invalid path: " + name)
            object.__setattr__(self, name, str(Path(value).absolute()))
        if (Path(self.data_root).is_relative_to(self.model_root)
                or Path(self.model_root).is_relative_to(self.data_root)):
            raise ConfigurationError("Data and model directories must not overlap")
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
            sources.append(dotenv_values(env_file, interpolate=False))
        sources.append(os.environ if environ is None else environ)
        for source in sources:
            for key, value in source.items():
                if not key.startswith("MDF_") or key.startswith("MDF_BENCH_"):
                    continue   # MDF_BENCH_* belong to `mdf bench` (markdown_for_llms/bench/settings.py)
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
        return {k: v for k, v in asdict(self).items() if k not in {"data_root", "model_root"}}

    def identity(self):
        """Storage identity schema 3: every setting except the operational ones."""
        value = {k: v for k, v in asdict(self).items() if k not in OPERATIONAL}
        value["identity_schema"] = IDENTITY_SCHEMA
        return digest(value)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
