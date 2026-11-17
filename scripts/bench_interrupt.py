#!/usr/bin/env python3
"""Interrupt a command at a proven phase: the harness of the interruption proofs of `mdf bench run`.

    bench_interrupt.py --evidence GLOB [--timeout SECONDS] -- COMMAND [ARG...]

It starts COMMAND in the foreground without a shell (`subprocess.Popen`), keeps its PID, waits at
most SECONDS for a path matching GLOB while checking that the child is alive, checks that the
evidence is still there, then sends SIGINT to the child and waits for it. The evidence of a Marker
conversion is the work directory of its attempt (`RUN/work/attempt-*-D04-marker-*`, which the
engine removes when the conversion ends); of an accepted Datalab request, the
`datalab-request.json` of its attempt. No `pkill`, no fixed sleep as the proof of the phase.
Run it under `scripts/run-capped.sh` with the command it interrupts.

Exit: the exit status of the child after the signal; 3 when the child ended before the evidence
(no signal sent); 4 when the evidence did not come in time (the child is interrupted anyway and
the proof does not count). Standard library only.
"""
import argparse
import glob
import signal
import subprocess
import sys
import time


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--evidence", required=True, help="glob of the path that proves the phase")
    parser.add_argument("--timeout", type=float, default=3600, help="seconds to wait for the evidence")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a command after --")
    child = subprocess.Popen(command)
    print(f"bench_interrupt: child pid {child.pid}", flush=True)
    end = time.monotonic() + args.timeout
    seen = False
    while time.monotonic() < end:
        if child.poll() is not None:
            print(f"bench_interrupt: the child ended before the evidence, exit {child.returncode}", flush=True)
            return 3
        if glob.glob(args.evidence):
            seen = True
            break
        time.sleep(0.2)
    if child.poll() is not None:
        print(f"bench_interrupt: the child ended before the signal, exit {child.returncode}", flush=True)
        return 3
    valid = seen and bool(glob.glob(args.evidence))     # the phase is still running at the signal
    print("bench_interrupt: " + ("evidence seen, the child is alive: SIGINT" if valid else
                                 "no evidence at the signal: SIGINT, the proof does not count"), flush=True)
    child.send_signal(signal.SIGINT)
    code = child.wait()
    print(f"bench_interrupt: child exit {code}", flush=True)
    return code if valid else 4


if __name__ == "__main__":
    sys.exit(main())
