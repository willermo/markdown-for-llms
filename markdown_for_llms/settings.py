"""Strict administrative settings and a closed per-job conversion profile."""
from dataclasses import asdict, dataclass, fields
from pathlib import Path
import hashlib
import json
import os


class ConfigurationError(ValueError):
    """An invalid configuration; messages never contain supplied values."""


# Operational time limits change neither quotas, paths nor stored data: they are
# recorded per run in the manifest and stay out of the storage identity (ADR 0011).
TIME_LIMITS = frozenset({"timeout_seconds", "stall_seconds"})
# Fields hashed by identity schema 1 (before ADR 0011), frozen for the automatic upgrade.
LEGACY_FIELDS = ("data_root", "model_root", "host", "port", "timeout_seconds", "memory_mib",
                 "upload_mib", "max_pages", "output_mib", "log_mib", "storage_mib", "reserve_mib",
                 "max_pixels", "max_dimension", "threads", "poll_seconds", "lease_seconds",
                 "grace_seconds", "process_limit", "bind_profile", "review_provider")
LEGACY_TIMEOUT_SECONDS = 450


@dataclass(frozen=True)
class Settings:
    data_root: str = "./mdf-data"
    model_root: str = "./mdf-models"
    host: str = "127.0.0.1"
    port: int = 8765
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
    bind_profile: str = "marker-cpu-v1"
    review_provider: str = "disabled"

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
        if self.bind_profile != "marker-cpu-v1" or self.review_provider != "disabled":
            raise ConfigurationError("Unsupported provider/profile")
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
                if not key.startswith("MDF_"):
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

    def public(self):
        return {k: v for k, v in asdict(self).items() if k not in {"data_root", "model_root"}}

    def identity(self):
        """Storage identity schema 2: every setting except the operational time limits."""
        value = {k: v for k, v in asdict(self).items() if k not in TIME_LIMITS}
        value["identity_schema"] = 2
        return digest(value)

    def legacy_identity(self):
        """Storage identity schema 1, with the timeout an unset variable used to mean."""
        value = {name: getattr(self, name) for name in LEGACY_FIELDS}
        value["timeout_seconds"] = self.timeout_seconds or LEGACY_TIMEOUT_SECONDS
        return digest(value)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def options(overrides=None):
    result = dict(use_llm=False, force_ocr=False, strip_existing_ocr=False,
                  disable_image_extraction=False, paginate_output=True,
                  pdftext_workers=1, layout_batch_size=1, recognition_batch_size=1,
                  detection_batch_size=1, table_rec_batch_size=1, ocr_error_batch_size=1,
                  lowres_image_dpi=96, highres_image_dpi=192)
    if overrides:
        if set(overrides) - {"force_ocr", "strip_existing_ocr"} or any(type(v) is not bool for v in overrides.values()):
            raise ConfigurationError("Unsupported job option or non-boolean value")
        result.update(overrides)
    return result
