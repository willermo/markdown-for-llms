"""Driver of the tests of the CPU server (MFL033, plan r2 §7), run as a subprocess (the relay connects to
the socket of the server, which the suite forbids in its own process): one conversion on the `cpu`
endpoint, pinned to llama.cpp b12040, with SIGTERM handled as SIGINT, as `mdf convert` does.

usage: python -I -B cpu_sigterm_driver.py SOURCE OUT WORK MODEL_ROOT CPU_ROOT MANIFEST ATTEMPT
"""
import json
from pathlib import Path
import signal
import sys


def main(argv):
    signal.signal(signal.SIGTERM, signal.default_int_handler)
    from markdown_for_llms.engine import servers
    from markdown_for_llms.engine.core import convert
    src, out, work, model_root, root, manifest, attempt = argv
    ep = servers.endpoint("cpu", {"kind": "cpu", "pins": {"llama.cpp": "b12040"}})
    try:
        result = convert(src, out, {"route": "marker-server", "attempt": attempt,
                                    "server": {"name": "cpu", "mode": "balanced"}},
                         {"memory_mib": 4096, "threads": 2}, work=work, model_root=model_root, server_endpoint=ep,
                         server_keys={}, server_cpu={"root": root, "manifest": json.loads(Path(manifest).read_text())})
    except KeyboardInterrupt:
        return 130
    return 0 if result["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
