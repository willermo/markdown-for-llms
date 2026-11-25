"""Peak, CPU, OOM and stall of a child as a real container (MFL051, plan r2.1 §4.3, J.4.3; finding A of the first
review).

The supervisor samples its own cgroup from inside the container and the caller reads the last complete line of
`stat/metrics`; `memory_limit` comes from Docker (`OOMKilled`, or 137 without a stop of ours), with or without
metrics. What Docker answers for the cases the plan left as assumptions is written to `observations.jsonl` of the
evidence of the run.
"""
import json
import unittest

import container_support as support
from markdown_for_llms.engine import containers
from markdown_for_llms.engine.codes import Code

MIB = 1048576


def observe(what, **facts):
    path = support.OUT / "observations.jsonl"
    if path.parent.is_dir():
        with open(path, "a", encoding="utf-8") as stream:
            stream.write(json.dumps(dict(facts, observation=what), sort_keys=True) + "\n")


@unittest.skipUnless(support.available(), support.REASON)
class TestMetrics(support.ComponentCase):
    def test_the_peak_of_memory_already_released_is_reported(self):
        steps = [{"do": "allocate", "mib": 200, "release": True}, {"do": "sleep", "seconds": 1.2}]
        outcome, report = self.probe(steps=steps, limits={"memory_mib": 512})
        self.assertEqual((outcome.cause, outcome.exit), ("exited", 0))
        self.assertEqual(self.step_result(report, "allocate"), "released")
        self.assertGreaterEqual(outcome.metrics["memory_peak_bytes"], 200 * MIB)
        self.assertLess(outcome.metrics["memory_peak_bytes"], 512 * MIB)
        self.assertEqual(outcome.metrics["oom_kill"], 0)
        self.assertGreater(outcome.metrics["tick"], 10)
        step = outcome.step()
        self.assertEqual((step["container"], step["exit"], step["oom_killed"], step["oom_kill"]), (outcome.name, 0, False, 0))
        self.assertGreaterEqual(step["memory_peak_bytes"], 200 * MIB)
        self.assertGreater(step["cpu_seconds"], 0)
        observe("the child reads memory.peak, cpu.stat and memory.events of its own cgroup as a non-root user",
                metrics=outcome.metrics)

    def test_a_command_that_exits_at_once_has_its_status_and_a_final_sample(self):
        outcome, _ = self.probe(steps=[{"do": "exit", "status": 7}])
        self.assertEqual((outcome.cause, outcome.exit), ("exited", 7))
        self.assertGreater(outcome.metrics["memory_peak_bytes"], MIB)
        self.assertIsNotNone(outcome.metrics["cpu_usec"])

    def test_cpu_time_grows_with_work_and_not_with_sleep(self):
        """What the stall check of the caller reads: CPU time that grows while the child works and stops while it
        sleeps (the events of the child are the caller's to read from the scratch)."""
        seen = []

        def on_poll(child, metrics):
            if metrics and (not seen or metrics["tick"] != seen[-1]["tick"]):
                seen.append(metrics)
            return None
        outcome, report = self.probe(steps=[{"do": "spin", "seconds": 2}, {"do": "sleep", "seconds": 3}], on_poll=on_poll)
        self.assertEqual(outcome.exit, 0)
        usec = [m["cpu_usec"] for m in seen]
        self.assertGreater(len(usec), 6)
        self.assertEqual(usec, sorted(usec))
        working = max(usec[i + 2] - usec[i] for i in range(len(usec) - 2))          # one second of samples
        resting = min(usec[i + 2] - usec[i] for i in range(len(usec) - 2))
        observe("CPU time of the cgroup in one second: the command at work, and at rest (the supervisor alone)",
                working_usec=working, resting_usec=resting)
        self.assertGreater(working, 500000)
        self.assertLess(resting, 100000)
        self.assertGreater(resting, 0)                      # the ticks of the supervisor are CPU time of the same cgroup
        self.assertGreaterEqual(outcome.metrics["cpu_usec"], 1500000)

    def test_the_caller_stops_a_child_that_stalls(self):
        """The supervisor runs in the cgroup it samples, so the CPU time of a child at rest still grows by the little
        its ticks cost: "no CPU" is growth under a threshold (here a tenth of a CPU), not an unchanged counter."""
        window = []

        def stalled(child, metrics):
            if metrics is None:
                return None
            window.append((child.clock(), metrics["cpu_usec"]))
            recent = [(t, u) for t, u in window if t >= child.clock() - 3.0]
            if recent[-1][0] - recent[0][0] < 2.5 or window[0][0] > child.clock() - 3.0:
                return None
            return "stalled" if (recent[-1][1] - recent[0][1]) / 1e6 < 0.1 * (recent[-1][0] - recent[0][0]) else None
        outcome, _ = self.probe(steps=[{"do": "spin", "seconds": 4}, {"do": "sleep", "seconds": 60}], on_poll=stalled)
        self.assertEqual((outcome.cause, outcome.reason, outcome.exit), ("stopped", "stalled", 202))
        self.assertGreater(outcome.seconds, 6)              # not while it worked
        self.assertLess(outcome.seconds, 20)


@unittest.skipUnless(support.available(), support.REASON)
class TestMemoryLimit(support.ComponentCase):
    def test_a_child_over_its_cap_is_memory_limit(self):
        outcome, _ = self.probe(steps=[{"do": "allocate", "mib": 400}], limits={"memory_mib": 128})
        self.assertEqual(outcome.cause, "memory")
        self.assertEqual(containers.error_of(outcome, "probe").code, Code.MEMORY_LIMIT)
        self.assertEqual(containers.error_of(outcome, "probe", tool=True).code, Code.TOOL_MEMORY_LIMIT)
        observe("the command itself is killed for memory", exit=outcome.exit, oom_killed=outcome.oom_killed,
                oom_kill_metric=outcome.metrics.get("oom_kill"))
        self.assertTrue(outcome.oom_killed or outcome.exit == 137)

    def test_memory_limit_does_not_need_the_metrics(self):
        """The command empties `stat/` first: no metrics can be read, and Docker still says why the child ended."""
        steps = [{"do": "remove", "path": "/stat"}, {"do": "mkfifo", "path": "/stat/metrics"}, {"do": "allocate", "mib": 400}]
        outcome, _ = self.probe(steps=steps, limits={"memory_mib": 128})
        self.assertEqual(outcome.metrics, {})
        self.assertEqual((outcome.stdout, outcome.stderr), (b"", b""))
        self.assertEqual(outcome.cause, "memory")
        self.assertTrue(outcome.oom_killed or outcome.exit == 137)

    def test_a_subprocess_killed_for_memory_while_pid_1_and_the_command_live(self):
        """The assumption of the plan (risk 4): what Docker reports when the kernel kills a subprocess and the
        command goes on and exits 0. Whatever `OOMKilled` says, the OOM kill counted by the supervisor makes the
        outcome `memory_limit`."""
        outcome, report = self.probe(steps=[{"do": "child_allocate", "mib": 400}, {"do": "sleep", "seconds": 0.8}],
                                     limits={"memory_mib": 128})
        self.assertEqual(self.step_result(report, "child_allocate"), {"signal": 9, "exit": None})
        self.assertEqual(outcome.exit, 0)
        self.assertGreaterEqual(outcome.metrics["oom_kill"], 1)
        self.assertEqual(outcome.cause, "memory")
        observe("a subprocess is killed for memory, the command exits 0 (risk 4)", exit=outcome.exit,
                oom_killed=outcome.oom_killed, oom_kill_metric=outcome.metrics["oom_kill"])

    def test_the_peak_of_a_child_killed_with_its_container_may_be_missing_and_the_code_is_there(self):
        """A child the launcher kills from outside: the supervisor dies with it, so the last sample may be missing;
        the exit status and the cause are Docker's."""
        def kill(child, metrics):
            if self.launcher.inspect(child.name)["state"] == "running" and metrics:
                self.launcher.kill(child.name)
            return None
        outcome, _ = self.probe(steps=[{"do": "sleep", "seconds": 60}], on_poll=kill)
        self.assertEqual((outcome.exit, outcome.reason), (137, None))
        self.assertEqual(outcome.cause, "memory")           # 137 without a stop of ours: Docker cannot tell it from an OOM kill
        observe("a child killed from outside (SIGKILL of the container)", exit=outcome.exit, oom_killed=outcome.oom_killed,
                metrics=outcome.metrics)


if __name__ == "__main__":
    unittest.main()
