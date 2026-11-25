"""The client of the launcher reads what a child may still be writing without ever trusting or waiting for it
(MFL051, plan r2.1 App. H.4, J.4.3; SOL-R2-RR1).

`read_regular`: a FIFO, a link, a directory, a file of another user and a missing file all read as absent, without
blocking; it is rooted, so a link in place of a directory above the file reads as absent too, whether it points
inside or outside the root (SOL-A1-N1); a cap is respected; the metrics are the last complete line. The folder of a call (`Call`): four sister
directories, the lock, the lease, the stdin. `Child`: the wait, where `stat/exit` is only a hint and the end is
what the launcher says of Docker.
"""
import fcntl
import json
import os
from pathlib import Path
import socket
import stat
import tempfile
import threading
import time
import unittest
from unittest import mock

import containers_test_support as support
from markdown_for_llms.engine import containers, supervisor
from markdown_for_llms.engine.codes import Code, EngineError


class Base(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.work = Path(tmp.name) / "work"
        self.work.mkdir()
        self.launcher, self.clock = support.FakeLauncher(), support.Clock()

    def child(self, kind="light", n=1):
        return containers.Child(self.launcher, self.work, support.name(kind, n), clock=self.clock, sleep=self.clock.sleep)

    def supervisor_writes(self, child, status=0, stdout=b"", stderr=b"", metrics=b"mdf-metrics/1 1048576 2500000 0 7 end\n"):
        """What the supervisor of a child leaves in `stat/` when its command ends."""
        stat_dir = child.call.stat
        (stat_dir / "stdout").write_bytes(stdout)
        (stat_dir / "stderr").write_bytes(stderr)
        (stat_dir / "metrics").write_bytes(metrics)
        (stat_dir / "exit").write_text("%d\n" % status)
        self.launcher.exits(child.name, status)


class TestReadRegular(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)

    def test_a_regular_file_of_this_user_within_the_cap(self):
        (self.root / "file").write_bytes(b"0123456789")
        self.assertEqual(containers.read_regular(self.root, "file", 100), b"0123456789")
        self.assertEqual(containers.read_regular(self.root, "file", 4), b"0123")
        self.assertEqual(containers.read_regular(self.root, "file", 4, tail=True), b"6789")
        self.assertEqual(containers.regular_size(self.root, "file"), 10)
        (self.root / "big").write_bytes(b"x" * (containers.STAT_LIMIT + 10))
        self.assertEqual(len(containers.read_regular(self.root, "big", containers.STAT_LIMIT)), containers.STAT_LIMIT)
        (self.root / "empty").write_bytes(b"")
        self.assertEqual(containers.read_regular(self.root, "empty", 10), b"")

    def test_a_fifo_a_link_a_directory_and_nothing_read_as_absent_without_blocking(self):
        os.mkfifo(self.root / "fifo")
        (self.root / "target").write_text("secret")
        os.symlink(self.root / "target", self.root / "link")
        os.symlink(self.root / "nowhere", self.root / "dangling")
        (self.root / "dir").mkdir()
        server = socket.socket(socket.AF_UNIX)
        server.bind(str(self.root / "sock"))
        self.addCleanup(server.close)
        started = time.monotonic()
        for name in ("fifo", "link", "dangling", "dir", "absent", "sock", "dir/absent", "target/x"):
            self.assertIsNone(containers.read_regular(self.root, name, 100), name)
            self.assertEqual(containers.regular_size(self.root, name), 0, name)
        self.assertLess(time.monotonic() - started, 1.0)
        self.assertIsNone(containers.read_regular("/dev", "zero", 10))
        self.assertIsNone(containers.read_regular("/dev", "tty", 10))
        self.assertIsNone(containers.read_regular(self.root / "absent-root", "file", 10))
        self.assertIsNone(containers.read_regular(self.root / "target", "file", 10))       # a root that is not a directory

    def test_a_fifo_with_a_writer_does_not_block_either(self):
        os.mkfifo(self.root / "fifo")
        writer = os.open(self.root / "fifo", os.O_RDWR)
        self.addCleanup(os.close, writer)
        os.write(writer, b"from a fifo")
        started = time.monotonic()
        self.assertIsNone(containers.read_regular(self.root, "fifo", 100))
        self.assertLess(time.monotonic() - started, 1.0)

    def test_a_file_of_another_user_reads_as_absent(self):
        (self.root / "file").write_bytes(b"data")
        with mock.patch.object(containers.os, "getuid", return_value=os.getuid() + 1):
            self.assertIsNone(containers.read_regular(self.root, "file", 100))
            self.assertEqual(containers.regular_size(self.root, "file"), 0)
        self.assertIsNone(containers.read_regular("/etc", "hostname", 100) if os.stat("/etc/hostname").st_uid != os.getuid() else None)

    def rooted(self):
        """A trusted root with a result two levels down, and a tree outside it that nothing may read."""
        root, outside = self.root / "call", self.root / "outside"
        (root / "scratch" / "out").mkdir(parents=True)
        (root / "scratch" / "out" / "result.md").write_text("RESULT")
        (root / "other").mkdir()
        (root / "other" / "result.md").write_text("OF ANOTHER DIRECTORY")
        outside.mkdir()
        (outside / "result.md").write_text("SYNTHETIC SECRET")
        return root, outside

    def test_a_file_below_real_directories_is_read(self):
        root, _ = self.rooted()
        self.assertEqual(containers.read_regular(root, "scratch/out/result.md", 64), b"RESULT")
        self.assertEqual(containers.read_regular(root, Path("scratch") / "out" / "result.md", 64), b"RESULT")
        self.assertEqual(containers.regular_size(root, "scratch/out/result.md"), 6)

    def test_a_link_in_place_of_a_parent_reads_as_absent_inside_and_outside_the_root(self):
        """SOL-A1-N1. The command writes `scratch`: it can put a link where a directory of its results should be.
        The link is not followed, whether it leaves the root (a file of the caller, its keys) or stays in it
        (the inputs or the control files of the call)."""
        for target in ("outside", "inside", "relative", "dangling"):
            with self.subTest(link=target):
                root, outside = self.rooted()
                out = root / "scratch" / "out"
                (out / "result.md").unlink()
                out.rmdir()
                os.symlink({"outside": outside, "inside": root / "other", "relative": "../other", "dangling": root / "nowhere"}[target], out)
                if target in ("outside", "inside", "relative"):
                    self.assertTrue((out / "result.md").is_file())                  # an ordinary open would follow it
                self.assertEqual(containers.read_regular(root, "other/result.md", 64), b"OF ANOTHER DIRECTORY")   # a real path reads
                self.assertIsNone(containers.read_regular(root, "scratch/out/result.md", 64))
                self.assertEqual(containers.regular_size(root, "scratch/out/result.md"), 0)
                import shutil
                shutil.rmtree(self.root / "call")
                shutil.rmtree(self.root / "outside")

    def test_a_link_at_the_first_level_or_at_any_other_reads_as_absent(self):
        root, outside = self.rooted()
        (outside / "out").mkdir()
        (outside / "out" / "result.md").write_text("SYNTHETIC SECRET")
        os.rename(root / "scratch", root / "real")
        os.symlink(outside, root / "scratch")                                       # the first component
        self.assertIsNone(containers.read_regular(root, "scratch/out/result.md", 64))
        self.assertEqual(containers.read_regular(root, "real/out/result.md", 64), b"RESULT")
        os.symlink(root / "real" / "out", root / "real" / "deep")
        self.assertIsNone(containers.read_regular(root, "real/deep/result.md", 64))

    def test_dot_dot_an_absolute_path_and_an_empty_component_read_as_absent(self):
        root, outside = self.rooted()
        self.assertEqual(containers.read_regular(root, "scratch/out/result.md", 64), b"RESULT")
        for name in ("../outside/result.md", "scratch/../other/result.md", "scratch/out/../out/result.md", str(outside / "result.md"),
                     "/etc/hostname", "", ".", "scratch//out/result.md", "scratch/./out/result.md", "scratch/out/result.md/",
                     "scratch/out/", "..", b"scratch/out/result.md", None, 7):
            with self.subTest(name=name):
                self.assertIsNone(containers.read_regular(root, name, 64))
                self.assertEqual(containers.regular_size(root, name), 0)
        self.assertEqual((root / "scratch" / ".." / "other" / "result.md").read_text(), "OF ANOTHER DIRECTORY")   # what a plain path gives

    def test_a_file_in_place_of_a_parent_reads_as_absent(self):
        root, _ = self.rooted()
        (root / "scratch" / "file").write_text("x")
        self.assertEqual(containers.read_regular(root, "scratch/file", 64), b"x")
        self.assertIsNone(containers.read_regular(root, "scratch/file/result.md", 64))
        os.mkfifo(root / "scratch" / "fifo")
        started = time.monotonic()
        self.assertIsNone(containers.read_regular(root, "scratch/fifo/result.md", 64))
        self.assertLess(time.monotonic() - started, 1.0)

    def test_the_root_may_be_a_descriptor_which_is_left_open(self):
        root, _ = self.rooted()
        fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
        self.addCleanup(os.close, fd)
        before = len(os.listdir("/proc/self/fd"))
        self.assertEqual(containers.read_regular(fd, "scratch/out/result.md", 64), b"RESULT")
        self.assertEqual(containers.read_regular(fd, "other/result.md", 64), b"OF ANOTHER DIRECTORY")
        self.assertIsNone(containers.read_regular(fd, "scratch/absent/result.md", 64))
        os.rename(root, self.root / "moved")                    # the descriptor is the root, wherever its name goes
        self.assertEqual(containers.read_regular(fd, "scratch/out/result.md", 64), b"RESULT")
        self.assertEqual(os.fstat(fd).st_ino, os.stat(self.root / "moved").st_ino)
        self.assertEqual(len(os.listdir("/proc/self/fd")), before)                  # and no descriptor is left behind

    def test_the_reads_of_a_call_do_not_follow_a_link_in_place_of_stat(self):
        """`stat/` and `ctl/lease` are read from the folder of the call, which the child cannot replace: a link in
        place of `stat` (which only a defect of the caller could put there) is not followed either."""
        call = containers.Call(self.root, support.name())
        self.addCleanup(call.close, True)
        elsewhere = self.root / "elsewhere"
        elsewhere.mkdir()
        (elsewhere / "exit").write_text("0\n")
        (elsewhere / "metrics").write_bytes(b"mdf-metrics/1 1 2 0 3 end\n")
        call.stat.rmdir()
        os.symlink(elsewhere, call.root / "stat")
        self.assertEqual(((call.root / "stat" / "exit").read_text(), call.exit_hint(), call.metrics()), ("0\n", False, None))

    def test_the_metrics_are_the_last_complete_line(self):
        call = containers.Call(self.root, support.name())
        self.addCleanup(call.close, True)
        self.assertIsNone(call.metrics())
        metrics = call.stat / "metrics"
        metrics.write_bytes(b"mdf-metrics/1 100 10 0 5 end\nmdf-metrics/1 200 20 1 10 end\nmdf-metrics/1 999 99")
        self.assertEqual(call.metrics(), {"memory_peak_bytes": 200, "cpu_usec": 20, "oom_kill": 1, "tick": 10})
        with open(metrics, "ab") as stream:             # a long file: only its tail is read
            stream.write(b" 9 15 end\n" + b"mdf-metrics/1 300 30 1 20 end\n" * 5000 + b"mdf-metrics/1 - - - 100020 end\n")
        self.assertGreater(metrics.stat().st_size, containers.STAT_LIMIT)
        self.assertEqual(call.metrics(), {"memory_peak_bytes": None, "cpu_usec": None, "oom_kill": None, "tick": 100020})
        metrics.unlink()
        os.mkfifo(metrics)
        self.assertIsNone(call.metrics())
        metrics.unlink()
        metrics.write_bytes(b"garbage\n" * 10)
        self.assertIsNone(call.metrics())

    def test_the_exit_hint(self):
        call = containers.Call(self.root, support.name())
        self.addCleanup(call.close, True)
        self.assertFalse(call.exit_hint())
        (call.stat / "exit").write_text("")
        self.assertFalse(call.exit_hint())
        (call.stat / "exit").write_text("0\n")
        self.assertTrue(call.exit_hint())
        (call.stat / "exit").unlink()
        os.mkfifo(call.stat / "exit")
        self.assertFalse(call.exit_hint())


class TestCall(Base):
    def test_four_sister_directories_a_lock_and_a_lease(self):
        call = containers.Call(self.work, support.name())
        self.addCleanup(call.close, True)
        self.assertEqual(sorted(p.name for p in call.root.iterdir()), ["ctl", "in", "owner", "scratch", "stat"])
        for path in (self.work / "call", call.root, call.root / "ctl", call.root / "in", call.scratch, call.stat):
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o700, path)
        self.assertEqual(stat.S_IMODE((call.root / "ctl" / "lease").stat().st_mode), 0o600)
        self.assertEqual((call.root / "ctl" / "lease").read_text(), "0\n")
        self.assertEqual(os.listdir(call.stat) + os.listdir(call.scratch) + os.listdir(call.root / "in"), [])
        with open(call.root / "owner", "rb") as other:      # held: another open of the file cannot take the lock
            with self.assertRaises(BlockingIOError):
                fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
        call.close(remove=False)
        with open(call.root / "owner", "rb") as other:
            fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def test_the_roles_of_each_kind(self):
        expected = {"tool": ["ctl", "in", "scratch", "stat"], "light": ["ctl", "in", "scratch", "stat"],
                    "marker": ["ctl", "in", "scratch", "stat"], "cpuserver": ["ctl", "stat"], "probe": ["ctl", "scratch", "stat"],
                    "manifest": ["ctl"]}
        for n, (kind, roles) in enumerate(expected.items()):
            call = containers.Call(self.work, support.name(kind, n + 1))
            self.addCleanup(call.close, True)
            self.assertEqual(sorted(p.name for p in call.root.iterdir() if p.is_dir()), roles)
        with self.assertRaises(ValueError):
            containers.Call(self.work, support.name("manifest", 9)).path("scratch")

    def test_the_ctl_of_a_manifest_call_can_be_read_by_a_root_without_capabilities(self):
        """Its supervisor may be root without capabilities: `ctl` 0755 and its two files 0644, a counter and an empty
        file; the folder of the call above them stays 0700."""
        call = containers.Call(self.work, support.name("manifest"))
        self.addCleanup(call.close, True)
        call.stop()
        self.assertEqual(stat.S_IMODE(call.root.stat().st_mode), 0o700)
        self.assertEqual(stat.S_IMODE((call.root / "ctl").stat().st_mode), 0o755)
        self.assertEqual([stat.S_IMODE((call.root / "ctl" / n).stat().st_mode) for n in ("lease", "stop")], [0o644, 0o644])

    def test_a_folder_already_there_belongs_to_another_call(self):
        first = containers.Call(self.work, support.name())
        self.addCleanup(first.close, True)
        with self.assertRaises(FileExistsError):
            containers.Call(self.work, support.name())
        self.assertTrue(first.root.is_dir())

    def test_a_name_out_of_the_schema_makes_no_folder(self):
        for name in ("x", support.PREFIX + "-shell-1", "../" + support.name(), support.name() + "/x", "mdf-eng-" + "a" * 17 + "-light-1"):
            with self.assertRaises(ValueError):
                containers.Call(self.work, name)
        self.assertEqual(os.listdir(self.work), [])
        self.assertEqual(containers.child_name("0123456789abcdefTAIL", "tool", 3), support.PREFIX + "-tool-3")
        self.assertEqual(containers.child_name("mdf-bench-4242-0a1b2c3d", "tool", 7, bench=True), "mdf-bench-4242-0a1b2c3d-tool-7")
        self.assertEqual(containers.split_name("mdf-eng-a-tool-1-tool-2"), ("mdf-eng-a-tool-1", "tool", 2))

    def test_the_lease_is_renewed_atomically_and_at_most_once_a_second(self):
        call = containers.Call(self.work, support.name())
        self.addCleanup(call.close, True)
        lease = call.root / "ctl" / "lease"
        first = lease.stat().st_ino
        call.renew(now=10.0)
        self.assertEqual(lease.read_text(), "1\n")
        self.assertNotEqual(lease.stat().st_ino, first)             # a rename, never a file written in place
        for now in (10.2, 10.6, 10.99):
            call.renew(now=now)
        self.assertEqual(lease.read_text(), "1\n")
        call.renew(now=11.0)
        self.assertEqual(lease.read_text(), "2\n")
        self.assertEqual(sorted(os.listdir(call.root / "ctl")), ["lease"])

    def test_stdin_and_stop_are_files_of_ctl(self):
        call = containers.Call(self.work, support.name("tool"))
        self.addCleanup(call.close, True)
        call.stdin(b"\x00binary\xff")
        call.stop()
        self.assertEqual((call.root / "ctl" / "stdin").read_bytes(), b"\x00binary\xff")
        self.assertEqual((call.root / "ctl" / "stop").read_bytes(), b"")
        self.assertEqual(sorted(os.listdir(call.root / "ctl")), ["lease", "stdin", "stop"])

    def test_inputs_are_copied_never_linked_and_only_regular_files_and_directories(self):
        source = self.work.parent / "source"
        (source / "sub").mkdir(parents=True)
        (source / "a.txt").write_text("a")
        (source / "sub" / "b.txt").write_text("b")
        call = containers.Call(self.work, support.name())
        self.addCleanup(call.close, True)
        self.assertEqual(call.add_input(source / "a.txt", "source.txt"), "/in/source.txt")
        call.add_input(source, "tree")
        self.assertEqual((call.root / "in" / "source.txt").read_text(), "a")
        self.assertEqual((call.root / "in" / "tree" / "sub" / "b.txt").read_text(), "b")
        self.assertNotEqual((call.root / "in" / "source.txt").stat().st_ino, (source / "a.txt").stat().st_ino)
        os.symlink(source / "a.txt", source / "link")
        os.mkfifo(self.work.parent / "fifo")
        for bad, name in ((source / "link", "x"), (source, "tree2"), (self.work.parent / "fifo", "f"), (source / "absent", "y"),
                          (source / "a.txt", "../x"), (source / "a.txt", "a/b"), (source / "a.txt", ""), (source / "a.txt", "..")):
            with self.assertRaises(ValueError):
                call.add_input(bad, name)
        self.assertEqual(sorted(os.listdir(call.root / "in")), ["source.txt", "tree"])

    def test_the_socket_directories_of_an_attempt(self):
        run = self.work.parent / "run"
        run.mkdir()
        root = containers.socket_dirs(run, support.PREFIX)
        self.assertEqual(root, run / "sock" / support.PREFIX)
        self.assertEqual(sorted(os.listdir(root)), ["llama", "relay"])
        for path in (run / "sock", root, root / "relay", root / "llama"):
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o700)
        self.assertEqual(containers.socket_dirs(run, support.PREFIX), root)
        for prefix in ("x", "../x", support.name(), ""):
            with self.assertRaises(ValueError):
                containers.socket_dirs(run, prefix)
        self.assertLess(len(os.fsencode(str(Path(containers.RUN_ROOT) / "sock" / ("mdf-eng-" + "x" * 16) / "relay" / "relay.sock"))), 108)


class TestChild(Base):
    def run_child(self, child, **kw):
        kw.setdefault("timeout", 600)
        return child.run("app", containers.PROBE_PYTHON, {"HOME": "/scratch"}, support.LIMITS, **kw)

    def test_the_request_names_no_mount(self):
        with self.child() as child:
            self.clock.at(0.3, lambda: self.supervisor_writes(child))
            self.run_child(child)
        self.assertEqual(self.launcher.request, {"kind": "light", "name": support.name(), "image": "app",
                                                 "argv": containers.PROBE_PYTHON, "env": {"HOME": "/scratch"}, "limits": support.LIMITS})

    def test_an_attested_end(self):
        with self.child() as child:
            folder = child.call.root
            self.clock.at(0.35, lambda: self.supervisor_writes(child, status=3, stdout=b"out", stderr=b"err"))
            outcome = self.run_child(child)
            (child.call.scratch / "result.json").write_text("{}")          # the caller takes its results before the folder goes
            self.assertTrue(folder.is_dir())
        self.assertEqual((outcome.name, outcome.cause, outcome.exit, outcome.oom_killed, outcome.reason),
                         (support.name(), "exited", 3, False, None))
        self.assertEqual((outcome.stdout, outcome.stderr), (b"out", b"err"))
        self.assertEqual(outcome.step(), {"container": support.name(), "exit": 3, "seconds": outcome.seconds,
                                          "memory_peak_bytes": 1048576, "cpu_seconds": 2.5, "oom_kill": 0, "oom_killed": False})
        self.assertFalse(folder.exists())
        self.assertEqual([c[0] for c in self.launcher.calls], ["run", "inspect", "remove"])
        self.assertIsNone(containers.error_of(outcome, "probe"))

    def test_the_name_goes_to_the_observer_before_the_request(self):
        seen = []
        with self.child() as child:
            self.clock.at(0.3, lambda: self.supervisor_writes(child))
            self.run_child(child, before=lambda name: seen.append((name, list(self.launcher.calls))))
        self.assertEqual(seen, [(support.name(), [])])

        def stop(name):
            raise EngineError(Code.CANCELLED, "before start")
        with self.assertRaises(EngineError), self.child(n=2) as child:
            folder = child.call.root
            self.run_child(child, before=stop)
        self.assertEqual(self.launcher.ops("run"), [("run", support.name())])       # the second one never reached the launcher
        self.assertFalse(folder.exists())

    def test_a_forged_exit_file_is_not_the_end(self):
        """`stat/` is written by the command too: an `exit` file while Docker says the container runs changes nothing."""
        polls = []

        def on_poll(child, metrics):
            polls.append(len(self.launcher.ops("inspect")))
            return "enough" if len(polls) == 30 else None
        with self.child() as child:
            (child.call.stat / "exit").write_text("0\n")
            (child.call.stat / "stdout").write_bytes(b"forged")
            self.clock.at(100, lambda: None)
            outcome = self.run_child(child, on_poll=on_poll)
        self.assertEqual((outcome.cause, outcome.reason), ("stopped", "enough"))
        self.assertEqual(len(polls), 30)
        self.assertEqual(polls[0], 1)                       # the hint made the caller ask at once
        self.assertTrue(3 <= polls[-1] <= 5, polls[-1])     # and then once a second, in three seconds of polls

    def test_an_end_without_a_hint_is_seen_within_five_seconds(self):
        with self.child() as child:
            self.clock.at(2.0, lambda: self.launcher.exits(child.name, 9))       # no `stat/exit` at all
            outcome = self.run_child(child)
        self.assertEqual((outcome.cause, outcome.exit), ("exited", 9))
        self.assertLessEqual(outcome.seconds, 2.0 + containers.INSPECT_SECONDS + 0.2)
        self.assertEqual(outcome.metrics, {})
        self.assertIsNone(outcome.step()["memory_peak_bytes"])

    def test_the_lease_is_renewed_while_the_caller_waits(self):
        leases = []
        with self.child() as child:
            lease = child.call.root / "ctl" / "lease"
            for second in range(1, 8):
                self.clock.at(second, lambda: leases.append(int(lease.read_text())))
            self.clock.at(7.5, lambda: self.supervisor_writes(child))
            self.run_child(child)
        self.assertEqual(leases, sorted(leases))
        self.assertGreaterEqual(leases[-1] - leases[0], 5)
        self.assertLessEqual(leases[-1], 9)                 # at most once a second

    def test_the_outcomes_of_the_supervisor_and_of_docker(self):
        cases = ((supervisor.EXIT_LEASE, False, None, "lease", Code.INTERRUPTED, Code.INTERRUPTED),
                 (supervisor.EXIT_TERM, False, None, "term", Code.TIMEOUT, Code.TOOL_TIMEOUT),
                 (supervisor.EXIT_STAT, False, None, "stat", Code.ENGINE_FAILED, Code.ENGINE_FAILED),
                 (137, True, None, "memory", Code.MEMORY_LIMIT, Code.TOOL_MEMORY_LIMIT),
                 (137, False, None, "memory", Code.MEMORY_LIMIT, Code.TOOL_MEMORY_LIMIT),
                 (0, True, None, "memory", Code.MEMORY_LIMIT, Code.TOOL_MEMORY_LIMIT),
                 (1, False, b"mdf-metrics/1 5 5 2 9 end\n", "memory", Code.MEMORY_LIMIT, Code.TOOL_MEMORY_LIMIT),
                 (supervisor.EXIT_STOP, False, None, "stopped", None, None), (0, False, None, "exited", None, None),
                 (251, False, None, "exited", None, None))
        for n, (status, oom_killed, metrics, cause, code, tool_code) in enumerate(cases):
            with self.subTest(status=status, oom_killed=oom_killed):
                with self.child(n=n + 1) as child:
                    def end(child=child):
                        self.supervisor_writes(child, status, **({} if metrics is None else {"metrics": metrics}))
                        self.launcher.children[child.name]["oom_killed"] = oom_killed
                    self.clock.at(0.3, end)
                    outcome = self.run_child(child)
                self.assertEqual((outcome.cause, outcome.exit, outcome.oom_killed), (cause, status, oom_killed))
                for tool, wanted in ((False, code), (True, tool_code)):
                    error = containers.error_of(outcome, "what", tool=tool)
                    self.assertEqual(None if error is None else error.code, wanted)

    def test_memory_limit_without_any_metrics(self):
        """`memory_limit` comes from Docker (`OOMKilled`, or 137 without a stop of ours), with `stat/` empty."""
        with self.child() as child:
            self.clock.at(0.3, lambda: self.launcher.exits(child.name, 137, oom_killed=True))
            outcome = self.run_child(child)
        self.assertEqual((outcome.cause, outcome.metrics, outcome.stdout), ("memory", {}, b""))

    def test_a_child_removed_by_the_launcher_itself_has_vanished(self):
        with self.child() as child:
            self.clock.at(1.0, lambda: self.launcher.children.pop(child.name))
            outcome = self.run_child(child)
        self.assertEqual((outcome.cause, outcome.exit), ("vanished", None))
        self.assertEqual(containers.error_of(outcome, "what").code, Code.ENGINE_FAILED)

    def test_a_child_the_launcher_killed_has_the_launchers_reason_never_memory(self):
        """SOL-A1-B2. The launcher's own watch kills a child at its term or at its lease when the supervisor did
        not: Docker then says 137, or nothing at all once the child is removed. The launcher says why (`reaped`),
        and that is the cause, with its closed code: `timeout` for the term, `interrupted` for the lease."""
        cases = (("term", "removed", "term", Code.TIMEOUT, Code.TOOL_TIMEOUT), ("term", 137, "term", Code.TIMEOUT, Code.TOOL_TIMEOUT),
                 ("lease", "removed", "lease", Code.INTERRUPTED, Code.INTERRUPTED), ("lease", 137, "lease", Code.INTERRUPTED, Code.INTERRUPTED),
                 ("stop", 137, "vanished", Code.ENGINE_FAILED, Code.ENGINE_FAILED),            # the stop of the launcher itself
                 ("stop", "removed", "vanished", Code.ENGINE_FAILED, Code.ENGINE_FAILED),
                 ("ended", "removed", "vanished", Code.ENGINE_FAILED, Code.ENGINE_FAILED))
        for n, (reaped, seen, cause, code, tool_code) in enumerate(cases):
            with self.subTest(reaped=reaped, seen=seen):
                with self.child(n=n + 1) as child:
                    def end(child=child):
                        self.launcher.reaped[child.name] = reaped
                        if seen == "removed":
                            self.launcher.children.pop(child.name)
                        else:                               # between the kill and the removal: 137, as an OOM kill reads
                            self.launcher.exits(child.name, seen)
                    self.clock.at(1.0, end)
                    outcome = self.run_child(child)
                self.assertEqual((outcome.cause, outcome.exit), (cause, None if seen == "removed" else seen))
                self.assertNotEqual(outcome.cause, "memory")
                for tool, wanted in ((False, code), (True, tool_code)):
                    self.assertEqual(containers.error_of(outcome, "what", tool=tool).code, wanted)
        self.assertEqual(containers.REAPED, ("lease", "term", "ended", "stop"))

    def test_an_exited_child_the_launcher_removed_later_keeps_its_own_status(self):
        """`ended`: the launcher removes a child that exited long ago. If the caller still sees it, its status is its own."""
        with self.child() as child:
            self.launcher.reaped[child.name] = "ended"
            self.clock.at(0.3, lambda: self.supervisor_writes(child, status=3))
            outcome = self.run_child(child)
        self.assertEqual((outcome.cause, outcome.exit), ("exited", 3))

    def test_the_callers_own_stop_stays_the_callers_reason(self):
        with self.child() as child:
            self.launcher.reaped[child.name] = "term"
            outcome = self.run_child(child, timeout=5)
        self.assertEqual((outcome.cause, outcome.reason), ("stopped", "timeout"))


class TestStops(Base):
    def run_child(self, child, **kw):
        return child.run("app", containers.PROBE_PYTHON, {}, support.LIMITS, **kw)

    def test_the_callers_reason_stops_the_child_with_ctl_stop_first(self):
        with self.child() as child:
            stop = child.call.root / "ctl" / "stop"
            seen = {}

            def on_poll(child, metrics):
                if self.clock.now >= 1003:
                    seen["before"] = stop.exists()
                    self.clock.at(0.2, lambda: self.supervisor_writes(child, supervisor.EXIT_STOP))   # the supervisor sees the file
                    return EngineError(Code.STALLED, "no event")
                return None
            outcome = self.run_child(child, on_poll=on_poll, timeout=600)
            self.assertTrue(stop.exists())
        self.assertEqual((seen["before"], outcome.cause, outcome.exit), (False, "stopped", supervisor.EXIT_STOP))
        self.assertEqual(outcome.reason.code, Code.STALLED)
        self.assertEqual(self.launcher.ops("kill"), [])
        self.assertIsNone(containers.error_of(outcome, "what"))        # the caller's reason is the caller's to raise

    def test_a_child_that_ignores_the_stop_is_killed_after_ten_seconds(self):
        with self.child() as child:
            outcome = self.run_child(child, timeout=5)
        self.assertEqual((outcome.cause, outcome.reason, outcome.exit), ("stopped", "timeout", 137))
        self.assertEqual(len(self.launcher.ops("kill")), 1)
        self.assertGreaterEqual(outcome.seconds, 5 + containers.STOP_WAIT_SECONDS)
        self.assertLess(outcome.seconds, 5 + containers.STOP_WAIT_SECONDS + 2)

    def test_scratch_alone_is_measured_against_its_cap(self):
        with self.child() as child:
            (child.call.root / "in" / "big-input").write_bytes(b"x" * (3 << 20))      # inputs, streams and ctl do not count
            (child.call.stat / "stdout").write_bytes(b"x" * (3 << 20))
            self.clock.at(2.0, lambda: (child.call.scratch / "out").write_bytes(b"x" * (3 << 20)))
            self.clock.at(2.5, lambda: self.supervisor_writes(child, supervisor.EXIT_STOP, stdout=b"kept"))
            outcome = self.run_child(child, scratch_mib=2, timeout=600)
        self.assertEqual((outcome.cause, outcome.reason), ("stopped", "output_limit"))
        self.assertGreaterEqual(outcome.seconds, 2.0)

    def test_the_streams_have_their_own_cap(self):
        with self.child() as child:
            self.clock.at(1.0, lambda: (child.call.stat / "stderr").write_bytes(b"x" * 2000))
            self.clock.at(1.5, lambda: self.supervisor_writes(child, supervisor.EXIT_STOP, stderr=b"x" * 2000))
            outcome = self.run_child(child, log_limit=1000, timeout=600)
        self.assertEqual((outcome.cause, outcome.reason), ("stopped", "output_limit"))
        self.assertEqual(len(outcome.stderr), 1000)

    def test_a_fifo_in_place_of_a_stream_reads_as_empty(self):
        with self.child() as child:
            def end():
                self.supervisor_writes(child, 0, stdout=b"real")
                (child.call.stat / "stdout").unlink()
                os.mkfifo(child.call.stat / "stdout")
                (child.call.stat / "stderr").unlink()
                os.symlink("/etc/hostname", child.call.stat / "stderr")
            self.clock.at(0.3, end)
            started = time.monotonic()
            outcome = self.run_child(child, timeout=600)
        self.assertEqual((outcome.exit, outcome.stdout, outcome.stderr), (0, b"", b""))
        self.assertLess(time.monotonic() - started, 2)


class TestNotAttested(Base):
    """Plan r2.1 §4.2, H.5: the launcher unreachable, or a removal that is not confirmed."""

    def run_child(self, child, **kw):
        kw.setdefault("timeout", 600)
        return child.run("app", containers.PROBE_PYTHON, {}, support.LIMITS, **kw)

    def lock_is_free(self, folder):
        with open(folder / "owner", "rb") as other:
            fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True

    def test_a_launcher_that_comes_back_within_twenty_seconds_changes_nothing(self):
        with self.child() as child:
            self.clock.at(2.0, lambda: setattr(self.launcher, "down", True))
            self.clock.at(17.0, lambda: setattr(self.launcher, "down", False))
            self.clock.at(25.0, lambda: self.supervisor_writes(child, 0, stdout=b"done"))
            outcome = self.run_child(child)
            self.assertFalse((child.call.root / "ctl" / "stop").exists())
        self.assertEqual((outcome.cause, outcome.exit, outcome.stdout), ("exited", 0, b"done"))

    def test_a_launcher_unreachable_for_twenty_seconds_stops_the_child_and_keeps_the_folder(self):
        child = self.child()
        folder = child.call.root
        with self.assertRaises(EngineError) as raised, child:
            self.clock.at(2.0, lambda: setattr(self.launcher, "down", True))
            self.run_child(child)
        self.assertEqual(raised.exception.code, Code.ISOLATION_UNAVAILABLE)
        self.assertIn("end not attested, call folder kept", raised.exception.detail)
        self.assertTrue((folder / "ctl" / "stop").exists())            # the supervisor ends the container without Docker
        self.assertEqual(sorted(p.name for p in folder.iterdir()), ["ctl", "in", "owner", "scratch", "stat"])
        self.assertTrue(self.lock_is_free(folder))
        self.assertFalse(child.attested)
        waited = self.clock.now - 1002.0
        self.assertTrue(containers.UNREACHABLE_SECONDS <= waited <= containers.UNREACHABLE_SECONDS + containers.INSPECT_SECONDS + 1, waited)
        with self.assertRaises(containers.Unattested):
            containers.stop_attempt_children(self.launcher, support.PREFIX)

    def test_a_launcher_that_does_not_answer_the_request_leaves_the_folder_with_a_stop(self):
        """Whether the container exists is not known: if it does, its supervisor finds `ctl/stop`."""
        self.launcher.down = True
        child = self.child()
        with self.assertRaises(EngineError) as raised, child:
            self.run_child(child)
        self.assertEqual(raised.exception.code, Code.ISOLATION_UNAVAILABLE)
        self.assertTrue((child.call.root / "ctl" / "stop").exists())
        self.assertTrue(self.lock_is_free(child.call.root))

    def test_a_refusal_of_the_launcher_is_its_closed_code_and_leaves_nothing(self):
        for n, (code, wanted) in enumerate((("invalid_setting", Code.INVALID_SETTING), ("toolchain_missing", Code.TOOLCHAIN_MISSING),
                                            ("toolchain_mismatch", Code.TOOLCHAIN_MISMATCH), ("unknown_code", Code.ENGINE_FAILED))):
            self.launcher.refuse = (code, "why")
            child = self.child(n=n + 1)
            with self.assertRaises(EngineError) as raised, child:
                self.run_child(child)
            self.assertEqual(raised.exception.code, wanted)
            self.assertFalse(child.call.root.exists())

    def test_a_removal_that_is_not_confirmed_keeps_the_folder(self):
        child = self.child()
        self.launcher.unconfirmed.add(child.name)
        with self.assertRaises(EngineError) as raised, child:
            self.clock.at(0.3, lambda: self.supervisor_writes(child, 0, stdout=b"done"))
            self.run_child(child)
        self.assertEqual(raised.exception.code, Code.ISOLATION_UNAVAILABLE)
        self.assertTrue((child.call.root / "ctl" / "stop").exists())
        self.assertTrue(self.lock_is_free(child.call.root))
        self.assertGreaterEqual(self.clock.now - 1000.0, containers.UNREACHABLE_SECONDS)

    def test_a_stop_of_the_caller_with_the_launcher_down_is_not_attested_either(self):
        child = self.child()
        with self.assertRaises(EngineError) as raised, child:
            self.clock.at(3.0, lambda: setattr(self.launcher, "down", True))
            self.run_child(child, timeout=4)
        self.assertEqual(raised.exception.code, Code.ISOLATION_UNAVAILABLE)
        self.assertTrue((child.call.root / "ctl" / "stop").exists())
        self.assertTrue(child.call.root.is_dir())


class TestWire(unittest.TestCase):
    """`containers.Launcher.call` on a socket pair (the suite may not `connect`): one request, one line, maybe bytes."""

    def launcher(self, answer, payload=b"", seen=None):
        def connect(path, timeout):
            ours, theirs = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)

            def serve():
                data = b""
                while True:
                    chunk = theirs.recv(65536)
                    if not chunk:
                        break
                    data += chunk
                if seen is not None:
                    seen.append(json.loads(data))
                theirs.sendall(answer + payload)
                theirs.close()
            threading.Thread(target=serve, daemon=True).start()
            return ours
        return containers.Launcher("/unused", connect=connect)

    def test_a_request_is_one_json_object_and_the_answer_one_line(self):
        seen = []
        launcher = self.launcher(b'{"ok": true, "children": [{"name": "x", "state": "running"}]}\n', seen=seen)
        self.assertEqual(launcher.list(support.PREFIX), [{"name": "x", "state": "running"}])
        self.assertEqual(seen, [{"op": "list", "prefix": support.PREFIX}])
        seen.clear()
        self.launcher(b'{"ok": true, "name": "n", "id": "i"}\n', seen=seen).run("light", "n", "app", ["a"], {"HOME": "/"}, {"pids": 1})
        self.assertEqual(seen, [{"op": "run", "kind": "light", "name": "n", "image": "app", "argv": ["a"], "env": {"HOME": "/"},
                                 "limits": {"pids": 1}}])

    def test_the_bytes_of_logs_follow_the_line(self):
        data = bytes(range(256)) * 1000
        launcher = self.launcher(b'{"ok": true, "bytes": %d}\n' % len(data), payload=data)
        self.assertEqual(launcher.logs("n"), data)
        with self.assertRaises(containers.Unreachable):
            self.launcher(b'{"ok": true, "bytes": %d}\n' % (len(data) + 1), payload=data).logs("n")
        with self.assertRaises(containers.Unreachable):
            self.launcher(b'{"ok": true, "bytes": %d}\n' % (containers.MANIFEST_LIMIT + 1)).logs("n")

    def test_isolation_unavailable_is_unreachable_and_the_other_codes_are_refusals(self):
        with self.assertRaises(containers.Unreachable) as raised:
            self.launcher(b'{"ok": false, "code": "isolation_unavailable", "detail": "docker unreachable: ENOENT"}\n').hello()
        self.assertIn("ENOENT", str(raised.exception))
        for code in ("invalid_setting", "toolchain_missing", "toolchain_mismatch"):
            with self.assertRaises(containers.Refused) as refused:
                self.launcher(b'{"ok": false, "code": "%s", "detail": "why"}\n' % code.encode()).hello()
            self.assertEqual((refused.exception.code, refused.exception.detail), (code, "why"))

    def test_anything_else_is_unreachable(self):
        for answer in (b"", b"not json\n", b"[]\n", b'{"ok": "yes"}\n', b'{"ok": true', b"x" * (containers.ANSWER_LIMIT + 10)):
            with self.assertRaises(containers.Unreachable):
                self.launcher(answer).hello()

        def refuse(path, timeout):
            raise ConnectionRefusedError(111, "refused")
        with self.assertRaises(containers.Unreachable) as raised:
            containers.Launcher("/unused", connect=refuse).hello()
        self.assertIn("ECONNREFUSED", str(raised.exception))

        def absent(path, timeout):
            raise FileNotFoundError(2, "no socket")
        with self.assertRaises(containers.Unreachable) as raised:
            containers.Launcher("/unused", connect=absent).inspect("n")
        self.assertIn("ENOENT", str(raised.exception))

    def test_a_request_over_the_limit_is_not_sent(self):
        def never(path, timeout):
            raise AssertionError("connected")
        with self.assertRaises(containers.Refused):
            containers.Launcher("/unused", connect=never).run("tool", "n", "toolchain", ["x" * containers.REQUEST_LIMIT], {}, {})

    def test_the_socket_and_the_roots_are_constants(self):
        self.assertEqual((containers.LAUNCHER_SOCKET, containers.WORK_ROOT, containers.RUN_ROOT),
                         ("/run/mdf/launcher/launcher.sock", "/work", "/run/mdf/run"))
        self.assertEqual(containers.Launcher().path, containers.LAUNCHER_SOCKET)


if __name__ == "__main__":
    unittest.main()
