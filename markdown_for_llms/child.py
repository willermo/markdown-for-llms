"""Set limits before importing any conversion or raster library."""
import importlib
import json
import os
from pathlib import Path
import resource
import sys


def main():
    limits = json.loads((Path("/scratch") / sys.argv[1]).read_text())
    resource.setrlimit(resource.RLIMIT_AS, (limits["memory_mib"] * 1048576,) * 2)
    resource.setrlimit(resource.RLIMIT_FSIZE, (limits["output_mib"] * 1048576,) * 2)
    resource.setrlimit(resource.RLIMIT_CPU, (limits["timeout_seconds"] * limits["threads"],) * 2)
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_NOFILE, (256, 256))
    resource.setrlimit(resource.RLIMIT_NPROC, (limits["process_limit"],) * 2)
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:limits["threads"]])
    return importlib.import_module(sys.argv[2]).main(sys.argv[3:])


if __name__ == "__main__":
    raise SystemExit(main())
