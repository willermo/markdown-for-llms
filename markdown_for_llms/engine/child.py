"""Entry of the light child (`isolation.py`): limits first, then the module.

argv: limits.json (in /scratch), the module, its arguments. The module's `main(args)` returns
the exit status; a `MemoryError` is exit 3 with a marker line, so the caller reports
`memory_limit` instead of a generic failure.
"""
import importlib
import json
from pathlib import Path
import resource
import sys

MEMORY_EXIT = 3
MEMORY_MARKER = "MDF-ENGINE-CHILD-MEMORY-LIMIT"


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    limits = json.loads((Path("/scratch") / argv[0]).read_text())
    resource.setrlimit(resource.RLIMIT_AS, (limits["memory_mib"] * 1048576,) * 2)
    resource.setrlimit(resource.RLIMIT_FSIZE, (limits["output_mib"] * 1048576,) * 2)
    resource.setrlimit(resource.RLIMIT_CPU, (limits["cpu_seconds"],) * 2)
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    files = limits.get("files_limit", 64)
    resource.setrlimit(resource.RLIMIT_NOFILE, (files, files))
    resource.setrlimit(resource.RLIMIT_NPROC, (limits["process_limit"],) * 2)
    try:
        return importlib.import_module(argv[1]).main(argv[2:])
    except MemoryError:
        print(MEMORY_MARKER, flush=True)
        return MEMORY_EXIT


if __name__ == "__main__":
    raise SystemExit(main())
