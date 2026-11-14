"""Settings of `mdf bench`: `MDF_BENCH_*` values from `--env-file`, the environment and options.

Put `MDF_BENCH_*` variables in the file given to `--env-file`; do not export them:
`scripts/full-check.sh` stops when any `MDF_*` variable is in the environment. The product
(`markdown_for_llms.settings`) ignores the `MDF_BENCH_` prefix.
"""
from dataclasses import dataclass
import os
from pathlib import Path

from .codes import BenchError, Code
from .tools import DEFAULT_MEMORY_MB as DEFAULT_TOOL_MEMORY_MB, MEMORY_MB_RANGE

PREFIX = "MDF_BENCH_"
PORT_BLOCK = range(23460, 23480)   # the project's block of host ports (AGENTS.md)
DEFAULT_RELAY_PORT = 23461
DEFAULT_INFERENCE_PORT = 23463


def default_cache_dir() -> Path:
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "markdown-for-llms" / "bench"


def default_runs_dir() -> Path:
    base = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
    return Path(base) / "markdown-for-llms" / "bench" / "runs"


@dataclass(frozen=True)
class BenchSettings:
    cache_dir: Path
    runs_dir: Path
    relay_port: int = DEFAULT_RELAY_PORT
    inference_port: int = DEFAULT_INFERENCE_PORT
    cap_datalab_usd: float | None = None
    tool_memory_mb: int = DEFAULT_TOOL_MEMORY_MB     # cap of every external tool process (tools.py)

    def __post_init__(self):
        for name in ("relay_port", "inference_port"):
            if getattr(self, name) not in PORT_BLOCK:
                raise BenchError(Code.PORT_OUTSIDE_BLOCK, name)
        if self.relay_port == self.inference_port:
            raise BenchError(Code.INVALID_SETTING, "relay_port equals inference_port")
        if self.cap_datalab_usd is not None and not 0 <= self.cap_datalab_usd <= 1000:
            raise BenchError(Code.INVALID_SETTING, "cap_datalab_usd")
        if self.tool_memory_mb not in MEMORY_MB_RANGE:
            raise BenchError(Code.INVALID_SETTING, "tool_memory_mb")
        for name in ("cache_dir", "runs_dir"):
            object.__setattr__(self, name, Path(getattr(self, name)).expanduser().absolute())

    @property
    def data_dir(self) -> Path:
        """Directory of the spend log: the parent of the runs directory."""
        return self.runs_dir.parent

    @classmethod
    def load(cls, env_file=None, environ=None, cache=None, runs=None):
        from dotenv import dotenv_values

        known = {"MDF_BENCH_CACHE_DIR": ("cache_dir", Path), "MDF_BENCH_RUNS_DIR": ("runs_dir", Path),
                 "MDF_BENCH_RELAY_PORT": ("relay_port", int), "MDF_BENCH_INFERENCE_PORT": ("inference_port", int),
                 "MDF_BENCH_CAP_DATALAB_USD": ("cap_datalab_usd", float),
                 "MDF_BENCH_TOOL_MEMORY_MB": ("tool_memory_mb", int)}
        sources = []
        if env_file is not None:
            if not Path(env_file).is_file():
                raise BenchError(Code.ENV_FILE_UNAVAILABLE)
            sources.append(dotenv_values(env_file, interpolate=False))
        sources.append(os.environ if environ is None else environ)
        values = {}
        for source in sources:
            for key, value in source.items():
                if not key.startswith(PREFIX):
                    continue
                if key not in known:
                    raise BenchError(Code.UNKNOWN_SETTING, key)
                name, kind = known[key]
                try:   # as in the product: the environment wins over the file
                    values[name] = kind(value)
                except (TypeError, ValueError):
                    raise BenchError(Code.INVALID_SETTING, key) from None
        if cache is not None:
            values["cache_dir"] = Path(cache)
        if runs is not None:
            values["runs_dir"] = Path(runs)
        values.setdefault("cache_dir", default_cache_dir())
        values.setdefault("runs_dir", default_runs_dir())
        return cls(**values)
