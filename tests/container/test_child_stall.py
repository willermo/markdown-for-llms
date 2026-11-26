"""The stall of a child and what a command leaves locked, on real containers (MFL051 A2: E-7, A2-d).

E-7. The supervisor of a child runs in the cgroup it samples, so the CPU time of a command that is blocked still
grows: about two milliseconds a second, measured here over twenty seconds and written to the observations. "The
counter did not change" would never hold. The stall check of the engine (`isolation.Stall`) counts the CPU time
as activity only at `isolation.STALL_CPU_USEC_PER_SECOND` or more over a window of the stall period: well above
that noise, well below one busy core. A command that is blocked, with its supervisor running, is `stalled`; one
that is slow and alive is not.

A2-d. A command can take away the permissions of what it wrote: the folder of its call cannot be removed as it
is. The sweep gives the directories of that folder their permissions back and removes it, once the owner of the
call is gone and Docker attests that the container does not exist.
"""
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
import unittest

import container_support as support
from markdown_for_llms.engine import containers, isolation
from markdown_for_llms.engine.codes import Code, EngineError


def observe(what, **facts):
    path = support.OUT / "observations.jsonl"
    if path.parent.is_dir():
        with open(path, "a", encoding="utf-8") as stream:
            stream.write(json.dumps(dict(facts, observation=what), sort_keys=True) + "\n")


class EngineCase(support.ComponentCase):
    """A child started as the engine starts it (`isolation.run`), on the work volume of the suite."""

    def setUp(self):
        super().setUp()
        tmp = tempfile.mkdtemp(prefix="mdf-stall-")
        self.addCleanup(shutil.rmtree, tmp, True)
        self.scratch = Path(tmp) / "scratch"
        self.scratch.mkdir()

    def run_steps(self, steps, **kw):
        (self.scratch / "steps.json").write_text(json.dumps(steps))
        kw.setdefault("timeout", 120)
        return isolation.run(self.scratch, containers.PROBE_MODULE, ["container"], attempt=self.attempt,
                             results=("probe.json",), **kw)


@unittest.skipUnless(support.available(), support.REASON)
class TestStall(EngineCase):
    def test_a_blocked_command_shows_only_the_cpu_time_of_its_supervisor(self):
        """The measure the threshold comes from: the CPU time of the cgroup of a command that sleeps."""
        seen = []

        def on_poll(child, metrics):
            if metrics and metrics.get("cpu_usec") is not None and (not seen or metrics["tick"] != seen[-1][1]):
                seen.append((time.monotonic(), metrics["tick"], metrics["cpu_usec"]))
            return None
        outcome, _ = self.probe(steps=[{"do": "sleep", "seconds": 24}], on_poll=on_poll, timeout=90)
        self.assertEqual(outcome.exit, 0)
        resting = [s for s in seen if s[0] - seen[0][0] >= 3.0]         # past the start of the interpreter and the probe
        self.assertGreater(len(resting), 20)
        seconds = resting[-1][0] - resting[0][0]
        rate = (resting[-1][2] - resting[0][2]) / seconds
        worst = max((b[2] - a[2]) / (b[0] - a[0]) for a, b in zip(resting, resting[4:]))       # any two seconds of it
        observe("CPU time of the cgroup of a blocked command: microseconds a second (the supervisor alone)",
                seconds=round(seconds, 1), usec_per_second=round(rate), worst_two_seconds_usec_per_second=round(worst),
                threshold_usec_per_second=isolation.STALL_CPU_USEC_PER_SECOND)
        self.assertGreater(rate, 0)                                     # never still: the old rule would not fire
        self.assertLess(rate, isolation.STALL_CPU_USEC_PER_SECOND / 4)  # and far below the threshold
        self.assertLess(worst, isolation.STALL_CPU_USEC_PER_SECOND)

    def test_a_blocked_command_is_stalled_with_its_supervisor_running(self):
        started = time.monotonic()
        with self.assertRaises(EngineError) as caught:
            self.run_steps([{"do": "sleep", "seconds": 100}], stall_seconds=6, poll_seconds=0.25)
        elapsed = time.monotonic() - started
        self.assertEqual(caught.exception.code, Code.STALLED)
        self.assertLess(elapsed, 30)                                    # the stall period and a window of it, not the sleep
        self.assertGreater(elapsed, 6)
        self.assertEqual(self.launcher.list(self.prefix), [])

    def test_a_slow_command_that_is_alive_is_not_stalled(self):
        """A tenth of a core, no event: slow, and alive. Five times the threshold, a hundredth of a busy machine."""
        status, log = self.run_steps([{"do": "pulse", "seconds": 16, "busy": 0.1}], stall_seconds=6, poll_seconds=0.25)
        self.assertEqual(status, 0, log)
        report = json.loads((self.scratch / "probe.json").read_text())
        self.assertEqual([step["result"] for step in report["steps"]], ["pulsed"])

    def test_events_keep_a_command_that_uses_no_cpu_from_stalled(self):
        events = []
        status, log = self.run_steps([{"do": "events", "seconds": 14, "every": 2}], stall_seconds=6, poll_seconds=0.25,
                                     on_events=events.extend)
        self.assertEqual(status, 0, log)
        self.assertGreaterEqual(len(events), 5)


@unittest.skipUnless(support.available(), support.REASON)
class TestLockedFolder(EngineCase):
    def test_what_a_command_locked_is_removed_by_the_sweep_once_its_container_is_attested_absent(self):
        status, log = self.run_steps([{"do": "lock", "path": "/scratch/locked"}])
        self.assertEqual(status, 0, log)
        calls = Path(containers.WORK_ROOT) / "call"
        (name,) = [n for n in os.listdir(calls) if n.startswith(self.prefix + "-")]
        locked = calls / name / "scratch" / "locked"
        # the call ended and was attested; its folder could not go: the command took its permissions away
        self.assertEqual(os.lstat(locked).st_mode & 0o777, 0)
        self.assertEqual(self.launcher.list(self.prefix), [])
        with self.assertRaises(PermissionError):
            os.listdir(locked)
        removed, kept = containers.sweep(self.launcher, containers.WORK_ROOT, prefix=self.prefix, min_age=0)
        self.assertEqual((removed, kept), ([name], {}))
        self.assertFalse((calls / name).exists())


if __name__ == "__main__":
    unittest.main()
