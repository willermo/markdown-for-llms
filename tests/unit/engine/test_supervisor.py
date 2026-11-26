"""The supervisor of a child container as an ordinary process on synthetic files (MFL051, plan r2.1 App. H.3).

Black-box cases of `supervisor.SUPERVISOR`, run with `/bin/bash -c`, an empty PATH (so the loop uses builtins only)
and a wall-clock bound per case: a supervisor still alive after it is a failure, and its process group is killed.
Here the cgroup is a directory of three synthetic files; as PID 1 of a real container, with the real limits, the
same cases run in `tests/container/` (`make test-components`).

A shell cannot trap a signal that was ignored when it started, and whoever runs these tests may ignore some
(`nohup`, the host of an editor: SIGHUP). PID 1 of a container starts with none ignored, so every supervisor of
these tests starts with the seven signals of its trap at their default action, whatever this process inherited.
"""
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import threading
import time
import unittest

from markdown_for_llms.engine import containers, supervisor

SLEEP = "/usr/bin/sleep"
PEAK, CPU, OOM = 123456789, 4242, 1
TRAPPED = ("TERM", "INT", "HUP", "USR1", "USR2", "PIPE", "ALRM")


def default_signals():
    """In the child, before bash: the signals of the supervisor's trap as PID 1 of a container finds them."""
    for name in TRAPPED:
        signal.signal(getattr(signal, "SIG" + name), signal.SIG_DFL)


class SupervisorCase(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.cgroup = self.root / "cgroup"
        self.cgroup.mkdir()
        (self.cgroup / "memory.peak").write_text("%d\n" % PEAK)
        (self.cgroup / "cpu.stat").write_text("usage_usec %d\nuser_usec 1\n" % CPU)
        (self.cgroup / "memory.events").write_text("low 0\noom 0\noom_kill %d\n" % OOM)
        self.ctl, self.stat = self.root / "ctl", self.root / "stat"
        self.ctl.mkdir()
        self.stat.mkdir()
        (self.ctl / "lease").write_text("0\n")

    def spawn(self, command, max_seconds=60.0, lease_seconds=3.0, use_stat=True):
        """The supervisor around `command` (`@D` is the root of the case), as a process of its own session."""
        argv = supervisor.entrypoint(max_seconds, lease_seconds, stat=use_stat, ctl=str(self.ctl), stat_dir=str(self.stat),
                                     cgroup=str(self.cgroup)) + [c.replace("@D", str(self.root)) for c in command]
        with open(self.root / "supervisor.out", "wb") as out:
            return subprocess.Popen(argv, env={"PATH": ""}, stdin=subprocess.DEVNULL, stdout=out, stderr=out,
                                    start_new_session=True, preexec_fn=default_signals)

    def reap(self, proc):
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait()

    def run_case(self, command, renew=True, stop_after=None, max_seconds=60.0, lease_seconds=3.0, bound=6.0, use_stat=True):
        """(exit status or "ALIVE", seconds) of the supervisor around `command` (`@D` is the root of the case)."""
        started = time.monotonic()
        proc = self.spawn(command, max_seconds, lease_seconds, use_stat)
        done = threading.Event()

        def renewer():
            n = 0
            while not done.wait(0.2):
                n += 1
                (self.ctl / "lease.new").write_text("%d\n" % n)
                os.replace(self.ctl / "lease.new", self.ctl / "lease")
        if renew:
            threading.Thread(target=renewer, daemon=True).start()
        timer = None
        if stop_after is not None:
            timer = threading.Timer(stop_after, lambda: (self.ctl / "stop").write_text(""))
            timer.start()
        try:
            status = proc.wait(timeout=bound)
        except subprocess.TimeoutExpired:
            status = "ALIVE"
        seconds = time.monotonic() - started
        done.set()
        if timer is not None:
            timer.cancel()
        self.reap(proc)
        return status, seconds

    def exit_file(self):
        data = containers.read_regular(self.stat, "exit", 64)
        return None if data is None else data.decode().strip()

    def metrics(self):
        return supervisor.last_metrics(containers.read_regular(self.stat, "metrics", containers.STAT_LIMIT, tail=True))

    def stdout(self):
        return containers.read_regular(self.stat, "stdout", 4096)


class TestEntrypoint(unittest.TestCase):
    def test_the_command_follows_the_fixed_arguments(self):
        argv = supervisor.entrypoint(600, 30)
        self.assertEqual(argv[:4], ["/bin/bash", "-c", supervisor.SUPERVISOR, "mdf-supervise"])
        self.assertEqual(argv[4:], ["6000", "300", "0.1", "/ctl", "/stat", "/sys/fs/cgroup", "--"])
        self.assertEqual(supervisor.entrypoint(1, 1, stat=False)[8], "-")
        self.assertEqual(supervisor.ticks(0.01), 1)

    def test_the_script_names_no_external_program(self):
        """The loop runs with an empty PATH: nothing but builtins and the command."""
        for word in ("cat ", "sleep ", "date ", "grep ", "sed ", "awk "):
            self.assertNotIn(word, supervisor.SUPERVISOR)

    def test_last_metrics_takes_the_last_complete_line(self):
        data = b"mdf-metrics/1 10 20 0 5 end\nmdf-metrics/1 30 - 1 10 end\nmdf-metrics/1 99 99 9"
        self.assertEqual(supervisor.last_metrics(data), {"memory_peak_bytes": 30, "cpu_usec": None, "oom_kill": 1, "tick": 10})
        self.assertIsNone(supervisor.last_metrics(b"mdf-metrics/1 1 2 3 4\n"))
        self.assertIsNone(supervisor.last_metrics(b"mdf-metrics/2 1 2 3 4 end\n"))
        self.assertIsNone(supervisor.last_metrics(b"mdf-metrics/1 1 2 3 4 end trailing\n"))
        self.assertIsNone(supervisor.last_metrics(None))


class TestExits(SupervisorCase):
    def test_exit_status_and_final_metrics_of_a_command_that_ends_at_once(self):
        status, seconds = self.run_case(["/usr/bin/true"])
        self.assertEqual(status, 0)
        self.assertLess(seconds, 2)
        self.assertEqual(self.exit_file(), "0")
        self.assertEqual(self.metrics(), {"memory_peak_bytes": PEAK, "cpu_usec": CPU, "oom_kill": OOM, "tick": 1})

    def test_exit_status_of_the_command_is_reported(self):
        status, _ = self.run_case(["/bin/sh", "-c", "exit 7"])
        self.assertEqual((status, self.exit_file()), (7, "7"))

    def test_a_renewed_lease_lets_the_command_run_past_it(self):
        status, seconds = self.run_case([SLEEP, "3"], lease_seconds=1.0)
        self.assertEqual(status, 0)
        self.assertGreaterEqual(seconds, 3)
        self.assertGreaterEqual(self.metrics()["tick"], 25)

    def test_a_lease_that_is_not_renewed_ends_the_child(self):
        status, seconds = self.run_case([SLEEP, "31.7"], renew=False, lease_seconds=1.0)
        self.assertEqual((status, self.exit_file()), (supervisor.EXIT_LEASE, "200"))
        self.assertLess(seconds, 3)
        self.assertEqual(self.metrics()["tick"], 12)    # the counter is first read at tick 1; 10 ticks of lease after it

    def test_the_stop_file_ends_the_child_within_a_tick(self):
        status, seconds = self.run_case([SLEEP, "31.7"], stop_after=0.5)
        self.assertEqual((status, self.exit_file()), (supervisor.EXIT_STOP, "202"))
        self.assertLess(seconds, 1.5)

    def test_the_term_ends_the_child(self):
        status, seconds = self.run_case([SLEEP, "31.7"], max_seconds=1.5)
        self.assertEqual((status, self.exit_file()), (supervisor.EXIT_TERM, "201"))
        self.assertLess(seconds, 3.5)

    def test_a_killed_command_is_137(self):
        status, _ = self.run_case(["/bin/sh", "-c", "kill -KILL $$"])
        self.assertEqual((status, self.exit_file()), (137, "137"))

    def test_term_from_the_command_ends_the_supervisor_early(self):
        """N-1: TERM has a trap. The command can end its supervisor, which is the earlier exit, the safe direction."""
        status, seconds = self.run_case(["/bin/bash", "-c", 'kill -TERM "$PPID"; exec /usr/bin/sleep 5'], renew=False,
                                        max_seconds=0.9, lease_seconds=0.6, bound=1.6)
        self.assertEqual(status, supervisor.EXIT_SIGNAL)
        self.assertLess(seconds, 1)

    def signals_end_it(self):
        for name in TRAPPED:
            with self.subTest(signal=name):
                for path in self.stat.iterdir():
                    path.unlink()
                status, seconds = self.run_case(["/bin/bash", "-c", 'kill -%s "$PPID"; exec /usr/bin/sleep 31.7' % name], bound=3.0)
                self.assertEqual((status, self.exit_file()), (supervisor.EXIT_SIGNAL, "143"))
                self.assertLess(seconds, 1)

    def test_the_signals_bash_handles_end_the_supervisor_with_one_status(self):
        self.signals_end_it()

    def test_the_same_when_whoever_runs_the_tests_ignores_those_signals(self):
        """CAR-A1-N4. A shell cannot trap a signal ignored at its start: under `nohup`, or from a process that ignores
        SIGHUP, the supervisor of a test would never see it. The tests give it the signals as a container does."""
        for name in TRAPPED:
            number = getattr(signal, "SIG" + name)
            self.addCleanup(signal.signal, number, signal.signal(number, signal.SIG_IGN))
        self.signals_end_it()


class TestBeforeTheCommand(SupervisorCase):
    """CAR-A1-N1: what the supervisor needs to end is in place before its trap, and its trap before the command."""

    def test_variables_and_functions_come_before_the_trap_and_the_trap_before_the_command(self):
        script = supervisor.SUPERVISOR
        at = {mark: script.index(mark) for mark in ("n=0 seen=0 last= peak= cpu= oom= child=\n", "sample() {", "finish() {",
                                                    'exec 7> "$stat/metrics" 8> "$stat/exit"',
                                                    "trap 'finish 143' " + " ".join(TRAPPED) + "\n",
                                                    'exec 5> "$stat/stdout" 6> "$stat/stderr"', '"$@" <&4', "child=$!")}
        self.assertEqual(sorted(at, key=at.get), list(at))
        self.assertEqual(script.count("trap 'finish 143'"), 1)
        body = script[at["finish() {"]:script.index("}", at["finish() {"])]
        self.assertEqual(body.split("\n")[1].strip(), "trap '' " + " ".join(TRAPPED))       # disarmed first
        self.assertIn('[ -n "$child" ] && kill -KILL "$child"', body)                       # nothing to kill before the command

    def held_before_the_command(self):
        """A supervisor that has set its trap and cannot reach its command, with no pause of the test: a FIFO in
        place of `stat/stdout`, which the script opens after the trap, keeps it in that `open` until somebody
        reads the FIFO, and the kernel says when the trap is set (`SigCgt` of the process holds its seven
        signals). Returns the process."""
        for path in self.stat.iterdir():
            path.unlink()
        os.mkfifo(self.stat / "stdout")
        proc = self.spawn(["/usr/bin/touch", "@D/the-command-ran"])
        self.addCleanup(self.reap, proc)
        wanted = sum(1 << (getattr(signal, "SIG" + name) - 1) for name in TRAPPED)
        until = time.monotonic() + 3.0
        while True:
            status = Path("/proc/%d/status" % proc.pid).read_text()
            caught = int(next(line for line in status.splitlines() if line.startswith("SigCgt:")).split()[1], 16)
            if caught & wanted == wanted:
                return proc
            self.assertIsNone(proc.poll(), "the supervisor ended before its command")
            if time.monotonic() > until:
                self.fail("the supervisor is held before its command and has no trap yet: it catches %#x, not %#x" % (caught, wanted))

    def release(self):
        """Read the FIFO: the supervisor may go on from the `open` that held it."""
        reader = os.open(self.stat / "stdout", os.O_RDONLY | os.O_NONBLOCK)
        self.addCleanup(os.close, reader)

    def test_a_signal_before_the_command_starts_ends_with_143_and_writes_the_exit_file(self):
        """The signal is sent while the supervisor is held, then the supervisor is released: whether the signal
        found it in the `open` or an instant before, it is handled before any command exists."""
        for name in TRAPPED:
            with self.subTest(signal=name):
                proc = self.held_before_the_command()
                os.kill(proc.pid, getattr(signal, "SIG" + name))
                self.release()
                try:
                    status = proc.wait(timeout=3.0)
                except subprocess.TimeoutExpired:
                    status = "ALIVE"
                self.assertEqual((status, self.exit_file()), (supervisor.EXIT_SIGNAL, "143"))
                self.assertEqual(self.metrics(), {"memory_peak_bytes": PEAK, "cpu_usec": CPU, "oom_kill": OOM, "tick": 0})
                self.assertFalse((self.root / "the-command-ran").exists())
                self.assertEqual((self.root / "supervisor.out").read_bytes(), b"")      # no `finish: command not found`, no unbound `child`

    def test_without_a_signal_the_same_supervisor_starts_its_command(self):
        """The control of the case above: held in the same way and released, the supervisor runs its command."""
        proc = self.held_before_the_command()
        self.assertFalse((self.root / "the-command-ran").exists())
        self.release()
        self.assertEqual(proc.wait(timeout=5.0), 0)
        self.assertTrue((self.root / "the-command-ran").exists())
        self.assertEqual(self.exit_file(), "0")


class TestStatFiles(SupervisorCase):
    """SOL-R2-RR1: nothing the command does to the paths of `stat/` blocks the supervisor."""

    def test_a_fifo_in_place_of_metrics_does_not_block_lease_or_term(self):
        status, seconds = self.run_case(
            ["/bin/bash", "-c", '/usr/bin/rm -f "$1/metrics"; /usr/bin/mkfifo "$1/metrics"; exec /usr/bin/sleep 5', "_", "@D/stat"],
            renew=False, max_seconds=0.9, lease_seconds=0.6, bound=1.6)
        self.assertIn(status, (supervisor.EXIT_LEASE, supervisor.EXIT_TERM))
        self.assertLess(seconds, 1.5)
        self.assertIsNone(self.metrics())               # the path is a FIFO now: it reads as absent, without blocking
        self.assertEqual(self.exit_file(), str(status))

    def test_a_fifo_in_place_of_exit_does_not_block_the_stop(self):
        status, seconds = self.run_case(
            ["/bin/bash", "-c", '/usr/bin/rm -f "$1/exit"; /usr/bin/mkfifo "$1/exit"; exec /usr/bin/sleep 31.7', "_", "@D/stat"],
            stop_after=0.5)
        self.assertEqual(status, supervisor.EXIT_STOP)
        self.assertLess(seconds, 1.5)
        self.assertIsNone(self.exit_file())
        self.assertIsNotNone(self.metrics())

    def test_fifos_in_place_of_every_file_do_not_block_the_lease(self):
        script = ('for f in stdout stderr metrics exit; do /usr/bin/rm -f "$1/$f"; /usr/bin/mkfifo "$1/$f"; done; '
                  'exec /usr/bin/sleep 31.7')
        status, seconds = self.run_case(["/bin/bash", "-c", script, "_", "@D/stat"], renew=False, lease_seconds=1.0)
        self.assertEqual(status, supervisor.EXIT_LEASE)
        self.assertLess(seconds, 3)
        self.assertEqual((self.exit_file(), self.metrics(), self.stdout()), (None, None, None))

    def test_an_emptied_stat_directory_does_not_block_the_term(self):
        status, seconds = self.run_case(["/bin/bash", "-c", '/usr/bin/rm -rf "$1"; exec /usr/bin/sleep 31.7', "_", "@D/stat"],
                                        max_seconds=1.5)
        self.assertEqual(status, supervisor.EXIT_TERM)
        self.assertLess(seconds, 3.5)

    def test_a_file_already_in_stat_is_a_refusal(self):
        for name in supervisor.STREAMS:
            with self.subTest(file=name):
                for other in supervisor.STREAMS:
                    (self.stat / other).unlink(missing_ok=True)
                (self.stat / name).write_text("x")
                status, seconds = self.run_case(["/usr/bin/true"])
                self.assertEqual(status, supervisor.EXIT_STAT)
                self.assertLess(seconds, 1)
                self.assertEqual((self.stat / name).read_text(), "x")

    def test_a_link_already_in_stat_is_a_refusal(self):
        os.symlink(self.root / "elsewhere", self.stat / "metrics")
        status, _ = self.run_case(["/usr/bin/true"])
        self.assertEqual(status, supervisor.EXIT_STAT)
        self.assertFalse((self.root / "elsewhere").exists())

    def test_without_a_stat_directory_the_streams_go_to_the_log(self):
        status, _ = self.run_case(["/bin/sh", "-c", "echo to-the-log"], use_stat=False)
        self.assertEqual(status, 0)
        self.assertEqual((self.root / "supervisor.out").read_bytes(), b"to-the-log\n")
        self.assertEqual(os.listdir(self.stat), [])

    def test_the_command_inherits_no_descriptor_of_the_supervisor(self):
        status, _ = self.run_case(["/bin/sh", "-c", "for f in /proc/self/fd/*; do echo ${f##*/}; done"])
        self.assertEqual(status, 0)
        numbers = sorted(int(n) for n in self.stdout().split())
        self.assertEqual([n for n in numbers if n in (4, 5, 6, 7, 8, 9)], [])
        self.assertEqual(numbers[:3], [0, 1, 2])


class TestStdin(SupervisorCase):
    def test_stdin_is_ctl_stdin_when_the_caller_wrote_one(self):
        (self.ctl / "stdin").write_bytes(b"from ctl/stdin\n\x00\xff")
        status, _ = self.run_case(["/usr/bin/cat"])
        self.assertEqual((status, self.stdout()), (0, b"from ctl/stdin\n\x00\xff"))

    def test_stdin_is_dev_null_otherwise(self):
        status, _ = self.run_case(["/usr/bin/cat"])
        self.assertEqual((status, self.stdout()), (0, b""))

    def test_a_link_as_ctl_stdin_is_not_followed(self):
        (self.root / "secret").write_text("secret")
        os.symlink(self.root / "secret", self.ctl / "stdin")
        status, _ = self.run_case(["/usr/bin/cat"])
        self.assertEqual((status, self.stdout()), (0, b""))


class TestPipe(SupervisorCase):
    def test_a_writer_on_the_pipe_of_the_supervisor_only_makes_it_end_earlier(self):
        """The command can reopen the supervisor's own pipe through /proc: every line it writes is a tick that comes
        sooner, so lease and term arrive sooner. It never makes them later."""
        status, seconds = self.run_case(
            ["/bin/bash", "-c", 'exec 3> /proc/$PPID/fd/9 || exit 9; while :; do echo x >&3; done'],
            renew=False, lease_seconds=1.0, bound=1.6)
        self.assertIn(status, (supervisor.EXIT_LEASE, supervisor.EXIT_TERM))
        self.assertLess(seconds, 1.2)

    def test_bytes_without_a_newline_on_the_pipe_do_not_extend_the_lease(self):
        status, seconds = self.run_case(
            ["/bin/bash", "-c", 'exec 3> /proc/$PPID/fd/9 || exit 9; while :; do printf x >&3; done'],
            renew=False, lease_seconds=1.0, bound=2.5)
        self.assertIn(status, (supervisor.EXIT_LEASE, supervisor.EXIT_TERM))
        self.assertLess(seconds, 2.2)


class TestNoSurvivor(SupervisorCase):
    def test_the_command_is_killed_with_the_supervisor(self):
        marker = "31.%d" % (os.getpid() % 100000)
        status, _ = self.run_case([SLEEP, marker], stop_after=0.3)
        self.assertEqual(status, supervisor.EXIT_STOP)
        wanted = ("%s\0%s\0" % (SLEEP, marker)).encode()
        left = []
        for entry in os.listdir("/proc"):               # the image has no pgrep: the command lines of /proc
            if entry.isdigit():
                try:
                    with open("/proc/%s/cmdline" % entry, "rb") as stream:
                        if stream.read() == wanted:
                            left.append(entry)
                except OSError:
                    pass
        self.assertEqual(left, [])


if __name__ == "__main__":
    unittest.main()
