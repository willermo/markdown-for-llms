"""The socket directories of the attempts on the run volume (MFL033, P1 of plan r2, CNB2 of r3; MFL051)."""
import fcntl
import os
from pathlib import Path
import shutil
import stat
import tempfile
import time
import unittest
from unittest import mock

from markdown_for_llms.engine import containers, sockets
from markdown_for_llms.engine.codes import Code, EngineError

PREFIX = "mdf-eng-0123456789abcdef"


def fds():
    return set(os.listdir("/proc/self/fd"))


class TestPrivate(unittest.TestCase):
    def volume(self, length=None):
        """A run volume of the test, padded to `length` bytes when given."""
        root = Path(tempfile.mkdtemp(prefix="rt-"))
        self.addCleanup(shutil.rmtree, root, True)
        path = root
        if length is not None:
            path = root / ("d" * max(1, length - len(str(root)) - 1))
            path.mkdir(mode=0o700)
        return path

    def test_under_the_run_volume_by_the_prefix_of_the_attempt_short_and_private(self):
        run = self.volume()
        with sockets.Private(PREFIX, run) as p:
            self.assertEqual(p.path, run / "sock" / PREFIX)             # where the launcher derives its mounts from
            for d in (p.path, p.relay_dir, p.llama_dir):
                info = os.lstat(d)
                self.assertEqual((stat.S_IMODE(info.st_mode), info.st_uid), (0o700, os.getuid()))
            self.assertEqual((p.relay_dir.name, p.llama_dir.name), containers.SOCKET_ROLES)
            self.assertLessEqual(sockets.check(p.relay_socket), sockets.SUN_PATH_MAX)
            self.assertLessEqual(sockets.check(p.llama_socket), sockets.SUN_PATH_MAX)
            made = p.path
        self.assertFalse(made.exists())

    def test_the_fixed_base_of_the_containers_holds_the_longest_names(self):
        """The base is a constant of the containers: with the longest prefix an attempt or the bench can have,
        both socket paths fit `sun_path`."""
        for prefix in ("mdf-eng-" + "a" * 16, "mdf-bench-4194304-0123abcd"):
            for name in (sockets.RELAY_SOCKET, sockets.LLAMA_SOCKET):
                self.assertLessEqual(sockets.check(sockets.base() / prefix / name), sockets.SUN_PATH_MAX)
        self.assertEqual(sockets.base(), Path(containers.RUN_ROOT) / "sock")

    def test_without_the_run_volume_or_in_use_it_is_isolation_unavailable(self):
        with self.assertRaises(EngineError) as caught:
            sockets.Private(PREFIX, self.volume() / "absent")
        self.assertEqual(caught.exception.code, Code.ISOLATION_UNAVAILABLE)
        run = self.volume()
        with sockets.Private(PREFIX, run) as p:
            with self.assertRaises(EngineError) as caught:
                sockets.Private(PREFIX, run)                # the directory of an attempt has one owner
            self.assertEqual(caught.exception.code, Code.ISOLATION_UNAVAILABLE)
            self.assertTrue(p.relay_dir.is_dir())           # and the second one took nothing away
        with self.assertRaises(ValueError):
            sockets.Private("../elsewhere", run)

    def test_a_directory_that_is_not_private_is_refused(self):
        run = self.volume()
        (run / "sock" / PREFIX).mkdir(parents=True)
        os.chmod(run / "sock" / PREFIX, 0o755)
        with self.assertRaises(EngineError) as caught:
            sockets.Private(PREFIX, run)
        self.assertEqual(caught.exception.code, Code.ISOLATION_UNAVAILABLE)

    def test_too_long_leaves_nothing_behind(self):
        """CNB2 of the review of r3: every failure closes the lock and removes the directory; the error is
        `socket_path_too_long` before any bind."""
        run = self.volume(length=70)
        before = fds()
        with self.assertRaises(EngineError) as caught:
            sockets.Private(PREFIX, run)
        self.assertEqual(caught.exception.code, Code.SOCKET_PATH_TOO_LONG)
        self.assertEqual(os.listdir(run / "sock"), [])
        self.assertEqual(fds(), before)

    def test_check(self):
        self.assertEqual(sockets.check("/" + "a" * 106), 107)
        with self.assertRaises(EngineError) as caught:
            sockets.check("/" + "a" * 107)
        self.assertEqual(caught.exception.code, Code.SOCKET_PATH_TOO_LONG)


class TestSweep(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="sweep-"))
        self.addCleanup(shutil.rmtree, self.root, True)

    def make(self, name, age, lock=True):
        d = self.root / name
        d.mkdir(mode=0o700)
        os.chmod(d, 0o700)
        then = time.time() - age
        if lock:
            (d / "lock").write_text("")
            os.utime(d / "lock", (then, then))
        os.utime(d, (then, then))
        return d

    def test_orphans_go_live_and_young_stay(self):
        self.make("mdf-eng-orphan", 120)
        self.make("mdf-eng-nolock", 120, lock=False)
        self.make("mdf-eng-young", 5)
        self.make("mdf-eng-young-nolock", 5, lock=False)
        live = self.make("mdf-eng-live", 120)
        fd = os.open(live / "lock", os.O_RDWR)
        self.addCleanup(os.close, fd)
        fcntl.flock(fd, fcntl.LOCK_EX)
        other = self.make("not-of-an-attempt", 120)
        removed, kept = sockets.sweep(self.root)
        self.assertEqual(sorted(removed), ["mdf-eng-nolock", "mdf-eng-orphan"])
        self.assertEqual(sorted(kept), ["mdf-eng-live", "mdf-eng-young", "mdf-eng-young-nolock"])
        self.assertTrue(other.exists())

    def test_a_live_private_directory_is_kept(self):
        with sockets.Private(PREFIX, self.root) as p:
            removed, kept = sockets.sweep(self.root / "sock", min_age=0)
            self.assertEqual((removed, kept), ([], [p.path.name]))

    def test_the_default_is_the_run_volume_and_its_absence_is_nothing_to_sweep(self):
        with mock.patch.object(containers, "RUN_ROOT", str(self.root / "absent")):
            self.assertEqual(sockets.sweep(), ([], []))


if __name__ == "__main__":
    unittest.main()
