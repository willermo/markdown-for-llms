"""A caller of the launcher as a process of its own, for the component tests that kill the caller (MFL051, A1).

usage: python -B stop_driver.py NAME STEPS_JSON LIMITS_JSON

Runs the probe (`selfprobe container`) with the steps in a `light` or `marker` child named NAME and waits for it
like any caller: it renews the lease while it lives. The test kills this process and looks at what the child does
without it. Exit 0 when the child ended and its end was attested.
"""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))       # the tree under test, as pytest runs it

import container_support as support                                # noqa: E402
from markdown_for_llms.engine import containers             # noqa: E402


def main(argv):
    name, steps, limits = argv[0], json.loads(argv[1]), json.loads(argv[2])
    kind = containers.split_name(name)[1]
    with containers.Child(containers.Launcher(), containers.WORK_ROOT, name) as child:
        scratch = child.call.scratch
        (scratch / containers.LIMITS_FILE).write_text(json.dumps(support.CHILD_LIMITS))
        (scratch / "steps.json").write_text(json.dumps(steps))
        outcome = child.run("app", containers.python_command(containers.PROBE_MODULE, ["container"]), support.VARIABLES,
                            dict(support.LIMITS, **limits), timeout=600)
        print(json.dumps({"cause": outcome.cause, "exit": outcome.exit, "kind": kind}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
