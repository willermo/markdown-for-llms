"""How a child ends when something around it goes away (MFL051, plan r2.1 §4.2, J.4.2, J.4.4, H.5, H.6).

Three layers, each proven without the next: the supervisor inside the child (the caller is killed; the launcher is
killed and held down; both), the launcher from outside (a child whose supervisor does nothing, here a frozen one),
and the end that is NOT ATTESTED: with the launcher killed and held down the call ends with
`isolation_unavailable`, ITS FOLDER STAYS, `stop_attempt_children` raises `Unattested`, and only after the launcher
is back and attests the absence does `sweep` take the folder away. Also the term, the stop of the launcher itself,
and a launcher that has no access to Docker (the access check with the real identity, E-3).
"""
import calendar
import fcntl
import json
from pathlib import Path
import signal
import socket
import subprocess
import sys
import threading
import time
import unittest

import container_support as support
from markdown_for_llms import launcher as launcher_module
from markdown_for_llms.engine import containers, supervisor
from markdown_for_llms.engine.codes import Code, EngineError

from test_child_metrics import observe

DRIVER = Path(__file__).with_name("stop_driver.py")
LEASE = 3                               # seconds: the lease of the children of these tests
GRACE = launcher_module.GRACE_SECONDS   # how long after the supervisor the watch of the launcher acts
TICK = 0.5                              # the margin of a tick, of Docker's own stop and of a poll of the test
# Two times are measured for the end of a child, and they are not the same thing (MFL051 A2, the red check of
# fbf744b: Docker reported a container exited 4.5 s after it had ended, on a loaded machine).
# - The REAL end is `State.FinishedAt` of the container: when its PID 1, the supervisor, left. The tight bounds of
#   these tests are on it: that is what the supervisor guarantees.
# - The REPORT is the moment an inspect of Docker says `exited`: it adds the bookkeeping of the daemon, which the
#   supervisor does not control. It is written to the observations, within a wide bound.
REPORT_SECONDS = 15                     # the wide bound: how long Docker may take to report an end that happened
CLOCK_MARGIN = 0.1                      # two readings of the wall clock (a file time, Docker's time) are compared


def finished_at(found):
    """`State.FinishedAt` of an inspect, in seconds since the epoch (`2026-11-26T10:34:49.193595915Z`)."""
    whole, _, fraction = found["State"]["FinishedAt"].rstrip("Z").partition(".")
    return calendar.timegm(time.strptime(whole, "%Y-%m-%dT%H:%M:%S")) + float("0." + (fraction or "0"))


def wall(instant):
    """The wall-clock time of an instant of `time.monotonic()`."""
    return time.time() - (time.monotonic() - instant)


class StopCase(support.ComponentCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        support.prepare_volumes()               # a `marker` child needs its folder in the volume of the weights

    def real_end(self, what, found, since, reported):
        """Seconds from `since` (wall clock) to the real end of the container. `reported`: the seconds after which
        Docker said `exited`, written to the observations and held within the wide bound."""
        seconds = finished_at(found) - since
        observe(what, real_end_seconds=round(seconds, 3), reported_after_seconds=round(reported, 3),
                report_bound=REPORT_SECONDS, exit=found["State"]["ExitCode"])
        self.assertLess(reported, REPORT_SECONDS, what)
        return seconds

    def wait_state(self, name, wanted, seconds, since=None):
        """Poll Docker until the container is in a state of `wanted` (`absent`: no container); returns (state,
        seconds since `since` or the call)."""
        started = time.monotonic() if since is None else since
        while True:
            found = self.raw_inspect(name)
            state = "absent" if found is None else found["State"]["Status"]
            if state in wanted:
                return found, time.monotonic() - started
            self.assertLess(time.monotonic() - started, seconds, "%s still %s" % (name, state))
            time.sleep(0.05)

    def caller(self, name, steps, **limits):
        """A caller in a process of its own, running the probe with `steps`; returns it once its child runs."""
        proc = subprocess.Popen([sys.executable, "-B", str(DRIVER), name, json.dumps(steps), json.dumps(dict({"lease_seconds": LEASE}, **limits))],
                                stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=str(Path.cwd()))
        self.addCleanup(self._reap, proc)
        try:
            self.wait_state(name, ("running",), 30)
        except AssertionError:
            proc.kill()
            raise AssertionError("the driver did not start its child: %r" % (proc.communicate(),)) from None
        return proc

    def _reap(self, proc):
        if proc.poll() is None:
            proc.kill()
        proc.communicate()

    def kill_caller(self, proc):
        proc.send_signal(signal.SIGKILL)
        proc.wait()
        return time.monotonic()

    def folder(self, name):
        return Path(containers.WORK_ROOT) / "call" / name

    def owner_is_free(self, name):
        with open(self.folder(name) / "owner", "rb") as owner:
            try:
                fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return False
        return True


@unittest.skipUnless(support.available(), support.REASON)
class TestCallerKilled(StopCase):
    def test_the_child_of_a_killed_caller_ends_within_its_lease_and_a_tick(self):
        name = self.name("light")
        proc = self.caller(name, [{"do": "sleep", "seconds": 300}])
        self.assertFalse(self.owner_is_free(name))
        killed = self.kill_caller(proc)
        lease_at_kill = (self.folder(name) / "ctl" / "lease").read_text().strip()
        found, seconds = self.wait_state(name, ("exited", "absent"), LEASE + REPORT_SECONDS, since=killed)
        metrics = supervisor.last_metrics(containers.read_regular(self.folder(name), "stat/metrics", containers.STAT_LIMIT, tail=True))
        observe("caller killed: the child ends at its lease", lease_seconds=LEASE, seconds_after_the_kill=round(seconds, 2),
                exit=found and found["State"]["ExitCode"], ticks=metrics and metrics["tick"], started=found and found["State"]["StartedAt"],
                finished=found and found["State"]["FinishedAt"], lease_at_kill=lease_at_kill,
                lease_at_end=(self.folder(name) / "ctl" / "lease").read_text().strip(),
                exit_file=(containers.read_regular(self.folder(name), "stat/exit", 64) or b"").decode().strip())
        self.assertIsNotNone(found)
        self.assertEqual(found["State"]["ExitCode"], supervisor.EXIT_LEASE)
        # the lease in ticks, counted from the last renewal (at most a second before the kill), and a tick
        self.assertLessEqual(metrics["tick"], (LEASE + 1) / supervisor.TICK_SECONDS + 3)
        real = self.real_end("caller killed: the real end at the lease", found, wall(killed), seconds)
        self.assertLessEqual(real, LEASE + 1 + supervisor.TICK_SECONDS + 2 * TICK + CLOCK_MARGIN)
        self.assertGreaterEqual(seconds, LEASE - 1.5)
        self.assertEqual(self.launcher.inspect(name)["exit"], supervisor.EXIT_LEASE)
        # what the dead caller left is collected by whoever attests the attempt
        self.assertTrue(self.owner_is_free(name))
        self.assertEqual(containers.sweep(self.launcher, containers.WORK_ROOT, prefix=self.prefix, min_age=0), ([], {name: "container"}))
        self.assertEqual(containers.stop_attempt_children(self.launcher, self.prefix), [name])
        self.assertEqual(containers.sweep(self.launcher, containers.WORK_ROOT, prefix=self.prefix, min_age=0), ([name], {}))
        self.assertFalse(self.folder(name).exists())

    def test_every_process_of_the_child_ends_with_its_supervisor(self):
        """The end of the PID namespace when PID 1 exits: processes that left the session of the command and would
        outlive it are gone with the container, at once."""
        steps = [{"do": "detach", "count": 5, "seconds": 600}, {"do": "sleep", "seconds": 0.5}]
        started = time.monotonic()
        outcome, report = self.probe(steps=steps)
        self.assertEqual((outcome.cause, outcome.exit), ("exited", 0))
        self.assertEqual(self.step_result(report, "detach"), "detached")
        self.assertLess(time.monotonic() - started, 20)
        self.assertIsNone(self.raw_inspect(outcome.name))


@unittest.skipUnless(support.available(), support.REASON)
class TestSupervisorAsPid1(StopCase):
    """The cases of the supervisor as an ordinary process (`tests/unit/engine/test_supervisor.py`), repeated where it
    is PID 1 of a real container with its real limits. What a command can do to its supervisor only ever ends the
    child EARLIER (N-1); a case in which it made the child live longer would be a stop for the maintainer."""

    def test_a_file_already_in_stat_is_203(self):
        outcome, report = self.probe(prepare=lambda child: (child.call.stat / "metrics").write_text("x"))
        self.assertEqual((outcome.cause, outcome.exit, report), ("stat", supervisor.EXIT_STAT, None))    # the command never ran
        self.assertEqual(containers.error_of(outcome, "probe").code, Code.ENGINE_FAILED)
        self.assertLess(outcome.seconds, containers.INSPECT_SECONDS + 3)

    def test_a_renewed_lease_lets_the_child_live_past_it(self):
        outcome, _ = self.probe(steps=[{"do": "sleep", "seconds": 5}], limits={"lease_seconds": 2})
        self.assertEqual((outcome.cause, outcome.exit), ("exited", 0))
        self.assertGreater(outcome.seconds, 5)

    def test_the_command_killed_is_137(self):
        outcome, _ = self.probe(steps=[{"do": "kill", "pid": "self", "signal": "KILL"}])
        self.assertEqual((outcome.exit, outcome.oom_killed), (137, False))

    def test_the_signals_bash_handles_end_the_supervisor_at_once(self):
        """As PID 1, bash handles these whether a script traps them or not (seen on Docker 29.10: without a trap it
        ended with 128 plus the signal). The supervisor traps them all: it ends with 143 and the container with it."""
        for signal_name in ("TERM", "INT", "HUP", "USR1", "USR2", "PIPE", "ALRM"):
            started = time.monotonic()
            outcome, _ = self.probe(steps=[{"do": "kill", "pid": 1, "signal": signal_name}, {"do": "sleep", "seconds": 60}])
            self.assertEqual((outcome.cause, outcome.exit), ("exited", supervisor.EXIT_SIGNAL), signal_name)
            self.assertLess(time.monotonic() - started, 6, signal_name)

    def test_kill_and_stop_from_the_command_do_not_reach_pid_1(self):
        """The kernel drops, for the init of a PID namespace, the signals it has no handler for, KILL and STOP
        included when they come from inside: the supervisor goes on, and so do its lease and term."""
        steps = [{"do": "kill", "pid": 1, "signal": "KILL"}, {"do": "kill", "pid": 1, "signal": "STOP"},
                 {"do": "kill", "pid": 1, "signal": "CONT"}, {"do": "kill", "pid": 1, "signal": "WINCH"}, {"do": "sleep", "seconds": 60}]
        outcome, report = self.probe(steps=steps, limits={"max_seconds": 4})
        self.assertEqual([s["result"] for s in report["steps"][:4]], ["ok"] * 4)       # sent, and dropped
        self.assertEqual((outcome.cause, outcome.exit), ("term", supervisor.EXIT_TERM))    # the supervisor was still at work
        self.assertLess(outcome.seconds, 4 + containers.INSPECT_SECONDS + 2)

    def test_every_other_signal_from_the_command_can_only_end_the_child_earlier(self):
        """Whatever a signal does to the supervisor, the child never outlives its term: either the signal does
        nothing (201 at the term: `QUIT`, which bash ignores) or the supervisor ends at once (`ABRT`, `SEGV`)."""
        seen = {}
        for name in ("QUIT", "ABRT", "SEGV"):
            outcome, _ = self.probe(steps=[{"do": "kill", "pid": 1, "signal": name}, {"do": "sleep", "seconds": 60}],
                                    limits={"max_seconds": 4})
            seen[name] = (outcome.cause, outcome.exit, round(outcome.seconds, 1))
            self.assertLess(outcome.seconds, 4 + containers.INSPECT_SECONDS + 2, name)
            self.assertNotEqual(outcome.exit, 0, name)
        observe("signals from the command to PID 1 (bash) that have no trap, with a term of 4 s", **{k: list(v) for k, v in seen.items()})
        self.assertEqual(seen["QUIT"][:2], ("term", supervisor.EXIT_TERM))

    def test_a_writer_on_the_pipe_of_pid_1_only_ends_the_child_earlier(self):
        """The command reopens the supervisor's own pipe through /proc/1/fd and floods it: every line is a tick that
        comes sooner. With a term of 60 seconds and a caller that renews, the child ends long before, never after."""
        started = time.monotonic()
        outcome, report = self.probe(steps=[{"do": "flood", "path": "/proc/1/fd/9", "seconds": 40}], limits={"max_seconds": 60})
        seconds = time.monotonic() - started
        observe("a flood of lines on /proc/1/fd/9 (the supervisor's pipe) with a term of 60 s", exit=outcome.exit,
                cause=outcome.cause, seconds=round(seconds, 2), ticks=outcome.metrics.get("tick"))
        self.assertIn(outcome.exit, (supervisor.EXIT_LEASE, supervisor.EXIT_TERM))
        self.assertLess(seconds, 30)

    def test_bytes_without_a_newline_on_the_pipe_do_not_make_the_lease_longer(self):
        """The same flood without newlines, and the caller killed: the child still ends at its lease, not later."""
        name = self.name("light")
        proc = self.caller(name, [{"do": "flood", "path": "/proc/1/fd/9", "seconds": 60, "newline": False}])
        time.sleep(1.0)
        killed = self.kill_caller(proc)
        found, seconds = self.wait_state(name, ("exited",), LEASE + REPORT_SECONDS, since=killed)
        observe("a flood of bytes without a newline on /proc/1/fd/9, the caller killed", exit=found["State"]["ExitCode"],
                seconds_after_the_kill=round(seconds, 2), lease_seconds=LEASE)
        self.assertIn(found["State"]["ExitCode"], (supervisor.EXIT_LEASE, supervisor.EXIT_TERM))
        real = self.real_end("a flood without a newline: the real end at the lease", found, wall(killed), seconds)
        self.assertLessEqual(real, LEASE + 1 + supervisor.TICK_SECONDS + 2 * TICK + CLOCK_MARGIN)

    def test_a_fifo_in_place_of_the_exit_file_does_not_block_the_stop(self):
        seen = {}

        def on_poll(child, metrics):
            if metrics and metrics["tick"] > 15:
                seen["mode"] = oct((child.call.stat / "exit").lstat().st_mode)[:4]
                return "enough"
            return None
        outcome, _ = self.probe(steps=[{"do": "mkfifo", "path": "/stat/exit"}, {"do": "sleep", "seconds": 60}], on_poll=on_poll)
        self.assertEqual(seen["mode"], "0o10")                  # a FIFO where the supervisor's file was
        self.assertEqual((outcome.cause, outcome.reason, outcome.exit), ("stopped", "enough", supervisor.EXIT_STOP))
        self.assertLess(outcome.seconds, 15)

    def test_an_emptied_stat_directory_does_not_block_the_term(self):
        outcome, _ = self.probe(steps=[{"do": "remove", "path": "/stat"}, {"do": "sleep", "seconds": 60}], limits={"max_seconds": 3})
        self.assertEqual((outcome.cause, outcome.exit, outcome.metrics), ("term", supervisor.EXIT_TERM, {}))
        self.assertLess(outcome.seconds, 3 + containers.INSPECT_SECONDS + 2)


@unittest.skipUnless(support.available(), support.REASON)
class TestLauncherKilledAndHeldDown(StopCase):
    def tearDown(self):
        self.ensure_launcher()

    def test_caller_alive_the_end_is_not_attested_and_the_folder_stays(self):
        name = self.name("light")
        child = containers.Child(self.launcher, containers.WORK_ROOT, name)
        seen = {}
        stops = []                              # the wall-clock instants at which the caller writes `ctl/stop`
        write_stop = child.call.stop
        child.call.stop = lambda: (stops.append(time.time()), write_stop())[1]

        def on_poll(child, metrics):
            if "killed" not in seen and self.raw_inspect(name)["State"]["Running"]:
                self.kill_launcher()
                seen["killed"] = time.monotonic()
            elif "killed" in seen and "alive" not in seen and time.monotonic() - seen["killed"] > 10:
                seen["alive"] = self.raw_inspect(name)["State"]["Status"]           # ten seconds without a launcher
            return None
        (child.call.scratch / containers.LIMITS_FILE).write_text(json.dumps(support.CHILD_LIMITS))
        (child.call.scratch / "steps.json").write_text(json.dumps([{"do": "sleep", "seconds": 300}]))
        with self.assertRaises(EngineError) as raised, child:
            child.run("app", containers.PROBE_PYTHON, support.VARIABLES, support.LIMITS, timeout=600, on_poll=on_poll)
        ended = time.monotonic()
        self.assertEqual(raised.exception.code, Code.ISOLATION_UNAVAILABLE)
        self.assertIn("end not attested, call folder kept", raised.exception.detail)
        self.assertEqual(seen["alive"], "running")                                  # the child went on without the launcher
        waited = ended - seen["killed"]
        self.assertTrue(containers.UNREACHABLE_SECONDS <= waited <= containers.UNREACHABLE_SECONDS + containers.INSPECT_SECONDS + 3, waited)
        # the stop through `ctl/stop` works without the launcher and without Docker's help
        found, seconds = self.wait_state(name, ("exited",), REPORT_SECONDS, since=ended)
        self.assertEqual(found["State"]["ExitCode"], supervisor.EXIT_STOP)
        # the tight bound is on the real end of the supervisor, counted from the moment `ctl/stop` was written for
        # the first time: a tick, and Docker's own stop. (Not the time of the file: the caller writes it again when
        # it leaves the block, and a rewritten file would make the measure shorter than the truth.)
        self.assertGreaterEqual(len(stops), 1)
        real = self.real_end("stop without the launcher: the real end after ctl/stop", found, stops[0], seconds)
        self.assertLess(real, 1 + TICK + CLOCK_MARGIN)
        # the folder stays, with its stop, and nobody can say the attempt is over
        folder = self.folder(name)
        self.assertEqual(sorted(p.name for p in folder.iterdir()), ["ctl", "in", "owner", "scratch", "stat"])
        self.assertTrue((folder / "ctl" / "stop").exists())
        self.assertTrue(self.owner_is_free(name))
        with self.assertRaises(containers.Unattested):
            containers.stop_attempt_children(self.launcher, self.prefix)
        self.assertEqual(containers.sweep(self.launcher, containers.WORK_ROOT, prefix=self.prefix, min_age=0), ([], {name: "unattested"}))
        self.assertTrue(folder.is_dir())
        # the launcher is back: the absence is attested, then the folder goes
        self.start_launcher()
        self.assertEqual(containers.sweep(self.launcher, containers.WORK_ROOT, prefix=self.prefix, min_age=0), ([], {name: "container"}))
        self.assertEqual(containers.stop_attempt_children(self.launcher, self.prefix), [name])
        self.assertIsNone(self.raw_inspect(name))
        self.assertEqual(containers.sweep(self.launcher, containers.WORK_ROOT, prefix=self.prefix, min_age=0), ([name], {}))
        self.assertFalse(folder.exists())

    def test_caller_killed_too_the_supervisor_alone_ends_the_children(self):
        """Launcher and caller both gone: a child that writes into its volume, one that talks to a live relay, and
        one that put FIFOs in place of the files of `stat/` all end at their lease, by their supervisor alone."""
        relay = containers.socket_dirs(containers.RUN_ROOT, self.prefix) / "relay"
        self.addCleanup(self._remove_sockets)
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server.bind(str(relay / "relay.sock"))
        server.listen(8)
        self.addCleanup(server.close)
        answered = []

        def serve():
            while True:
                try:
                    conn, _ = server.accept()
                except OSError:
                    return
                with conn:
                    conn.recv(100)
                    conn.sendall(b"pong\n")
                    answered.append(time.monotonic())
        threading.Thread(target=serve, daemon=True).start()
        writer = [step for _ in range(200) for step in ({"do": "append", "path": "/scratch/log", "text": "x" * 64},
                                                        {"do": "sleep", "seconds": 0.05})]
        cases = {
            self.name("light"): writer,
            self.name("marker"): [{"do": "connect", "path": "/relay/relay.sock", "repeat": 2000, "interval": 0.05}],
            self.name("light"): [{"do": "mkfifo", "path": "/stat/metrics"}, {"do": "mkfifo", "path": "/stat/exit"},
                                 {"do": "mkfifo", "path": "/stat/stdout"}, {"do": "sleep", "seconds": 300}],
        }
        callers = {name: self.caller(name, steps) for name, steps in cases.items()}
        time.sleep(1.0)                                         # the children are at work
        self.kill_launcher()
        killed = max(self.kill_caller(proc) for proc in callers.values())
        for name in cases:
            found, seconds = self.wait_state(name, ("exited",), LEASE + REPORT_SECONDS, since=killed)
            self.assertEqual(found["State"]["ExitCode"], supervisor.EXIT_LEASE, name)
            real = self.real_end("launcher and caller killed: the real end at the lease", found, wall(killed), seconds)
            self.assertLessEqual(real, LEASE + 1 + supervisor.TICK_SECONDS + 2 * TICK + CLOCK_MARGIN, name)
        self.assertGreater(len(answered), 10)                   # the relay was live until the child ended
        self.assertGreater((self.folder(list(cases)[0]) / "scratch" / "log").stat().st_size, 20 * 64)
        with self.assertRaises(containers.Unattested):
            containers.stop_attempt_children(self.launcher, self.prefix)
        self.start_launcher()
        self.assertEqual(sorted(containers.stop_attempt_children(self.launcher, self.prefix)), sorted(cases))


@unittest.skipUnless(support.available(), support.REASON)
class TestLauncherFromOutside(StopCase):
    """The third layer: a child whose supervisor does nothing (frozen by the driver with Docker's pause) is killed
    and removed by the launcher, on its own monotonic clock."""

    def tearDown(self):
        self.ensure_launcher()

    def pause(self, name):
        status, _ = self.raw("POST", "/containers/%s/pause" % name)
        self.assertEqual(status, 204)

    def test_a_frozen_child_without_renewals_is_removed_at_its_lease(self):
        name = self.name("light")
        proc = self.caller(name, [{"do": "sleep", "seconds": 300}])
        self.pause(name)
        killed = self.kill_caller(proc)
        _, seconds = self.wait_state(name, ("absent",), LEASE + GRACE + 8, since=killed)
        self.assertGreaterEqual(seconds, LEASE + GRACE - 1.5)   # after the time it leaves to the supervisor
        self.assertEqual(self.launcher.inspect(name), {"ok": True, "state": "absent", "reaped": "lease"})   # and it says why

    def test_after_a_restart_the_launcher_gives_a_whole_lease_then_removes(self):
        name = self.name("light")
        proc = self.caller(name, [{"do": "sleep", "seconds": 300}])
        self.kill_launcher()
        self.pause(name)
        self.kill_caller(proc)
        time.sleep(LEASE + 2)                                   # past the lease, and nobody is there to enforce it
        self.assertEqual(self.raw_inspect(name)["State"]["Status"], "paused")
        self.start_launcher()
        restarted = time.monotonic()
        _, seconds = self.wait_state(name, ("absent",), LEASE + GRACE + 8, since=restarted)
        self.assertGreaterEqual(seconds, LEASE + GRACE - 1.0)   # the lease starts at the first observation of the new launcher

    def test_a_frozen_child_whose_caller_renews_is_removed_at_its_term(self):
        seen = {}

        def on_poll(child, metrics):
            if "paused" not in seen and self.raw_inspect(child.name)["State"]["Running"]:
                self.pause(child.name)
                seen["paused"] = time.monotonic()
            return None
        started = time.monotonic()
        outcome, _ = self.probe(steps=[{"do": "sleep", "seconds": 300}], limits={"max_seconds": 6}, on_poll=on_poll)
        # killed and removed by the launcher: no status of its own, and the launcher's reason, never a memory limit
        self.assertEqual((outcome.cause, outcome.exit), ("term", None))
        self.assertEqual(containers.error_of(outcome, "probe").code, Code.TIMEOUT)
        self.assertEqual(containers.error_of(outcome, "probe", tool=True).code, Code.TOOL_TIMEOUT)
        self.assertTrue(6 + GRACE - 1 <= time.monotonic() - started <= 6 + GRACE + containers.INSPECT_SECONDS + 4)


@unittest.skipUnless(support.available(), support.REASON)
class TestTermAndStopOfTheLauncher(StopCase):
    def tearDown(self):
        self.ensure_launcher()

    def test_the_term_ends_a_child_whose_caller_renews(self):
        outcome, _ = self.probe(steps=[{"do": "sleep", "seconds": 300}], limits={"max_seconds": 3})
        self.assertEqual((outcome.cause, outcome.exit), ("term", supervisor.EXIT_TERM))
        self.assertEqual(containers.error_of(outcome, "probe").code, Code.TIMEOUT)
        self.assertEqual(containers.error_of(outcome, "probe", tool=True).code, Code.TOOL_TIMEOUT)
        self.assertTrue(2.5 <= outcome.seconds <= 3 + containers.INSPECT_SECONDS + 2, outcome.seconds)

    def test_sigterm_makes_the_launcher_remove_its_children_and_exit(self):
        name = self.name("light")
        proc = self.caller(name, [{"do": "sleep", "seconds": 300}], lease_seconds=30)
        ident = self.service("launcher")
        started = time.monotonic()
        status, _ = self.raw("POST", "/containers/%s/stop" % ident, query={"t": "30"}, timeout=60)
        self.assertIn(status, (204, 304))
        self.assertLess(time.monotonic() - started, 30)
        state = self.raw_inspect(ident)["State"]
        self.assertEqual((state["Status"], state["ExitCode"]), ("exited", 0))
        self.assertIsNone(self.raw_inspect(name))               # removed by the launcher on its way out
        status, logs = self.docker.call("GET", "/containers/%s/logs" % ident, query={"stdout": "1", "stderr": "1", "tail": "5"})
        self.assertIn(b"launcher stopped: children removed", launcher_module.demux(logs, 2) + launcher_module.demux(logs, 1))
        self.kill_caller(proc)
        self.start_launcher()
        self.assertEqual(self.launcher.list(self.prefix), [])


@unittest.skipUnless(support.available(), support.REASON)
class TestLauncherWithoutDocker(StopCase):
    """E-3 with the real identity: the launcher's access check, run in containers of its own image by the two
    fault services of the development project. No fallback, no capability: it says why and exits."""

    def run_service(self, service):
        ident = self.service(service)
        status, _ = self.raw("POST", "/containers/%s/start" % ident)
        self.assertIn(status, (204, 304))
        found, _ = self.wait_state(ident, ("exited",), 60)
        status, logs = self.docker.call("GET", "/containers/%s/logs" % ident, query={"stdout": "1", "stderr": "1"})
        self.assertEqual(status, 200)
        return found, launcher_module.demux(logs, 1).decode(), launcher_module.demux(logs, 2).decode().strip()

    def test_a_socket_that_is_not_there(self):
        found, out, err = self.run_service("no-socket")
        self.assertEqual((found["State"]["ExitCode"], out), (launcher_module.EXIT_ACCESS, ""))
        self.assertRegex(err.splitlines()[-1], r"^docker endpoint unix:///run/absent/docker\.sock: ENOENT; effective uid %d, "
                                               r"gids [0-9,]+; no fallback$" % self.hello["uid"])

    def test_the_real_socket_without_its_group(self):
        """The same socket the working launcher uses, the same user, and no supplementary group: EACCES. The group of
        the socket is what gives the daemon to the launcher, and nothing else does."""
        found, out, err = self.run_service("no-group")
        self.assertEqual((found["State"]["ExitCode"], out), (launcher_module.EXIT_ACCESS, ""))
        line = err.splitlines()[-1]
        self.assertRegex(line, r"^docker endpoint unix:///run/docker\.sock: EACCES; effective uid %d, gids [0-9,]+; no fallback$"
                         % self.hello["uid"])
        gids = line.split("gids ")[1].split(";")[0].split(",")
        working = self.raw_inspect(self.service("launcher"))["HostConfig"]["GroupAdd"]
        self.assertEqual(len(working), 1)
        self.assertNotIn(working[0], gids)
        self.assertEqual(found["HostConfig"]["GroupAdd"] or [], [])
        self.assertEqual((found["HostConfig"]["CapAdd"], found["HostConfig"]["CapDrop"], found["Config"]["User"]),
                         (None, ["ALL"], "%d:%d" % (self.hello["uid"], int(found["Config"]["User"].split(":")[1]))))

    def test_the_working_launcher_has_the_group_of_the_socket_and_no_capability(self):
        found = self.raw_inspect(self.service("launcher"))
        host = found["HostConfig"]
        self.assertEqual((host["CapAdd"], host["CapDrop"], host["Privileged"], host["ReadonlyRootfs"], host["NetworkMode"]),
                         (None, ["ALL"], False, True, "none"))
        self.assertEqual(found["Config"]["User"].split(":")[0], str(self.hello["uid"]))
        self.assertNotEqual(found["Config"]["User"].split(":")[0], "0")
        self.assertEqual(len(host["GroupAdd"]), 1)
        self.assertEqual(self.hello["images"]["app"]["id"], found["Image"])        # its children run its own image, by ID
        self.assertEqual(found["Config"]["Hostname"], found["Id"][:12])


if __name__ == "__main__":
    unittest.main()
