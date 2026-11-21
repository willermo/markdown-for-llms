"""Private socket directories of the `marker-server` route (MFL033, P1 of plan r2, CNB2 of r3)."""
import fcntl
import os
from pathlib import Path
import shutil
import stat
import tempfile
import time
import unittest

from markdown_for_llms.engine import sockets
from markdown_for_llms.engine.codes import Code, EngineError


def fds():
    return set(os.listdir("/proc/self/fd"))


class TestPrivate(unittest.TestCase):
    def runtime(self, length=None, mode=0o700):
        """A private runtime directory, padded to `length` bytes when given."""
        root = Path(tempfile.mkdtemp(prefix="rt-"))
        self.addCleanup(shutil.rmtree, root, True)
        path = root
        if length is not None:
            path = root / ("d" * max(1, length - len(str(root)) - 1))
            path.mkdir(mode=0o700)
        os.chmod(path, mode)
        return path

    def test_under_the_runtime_directory_short_and_private(self):
        rt = self.runtime()
        with sockets.Private({"XDG_RUNTIME_DIR": str(rt)}) as p:
            self.assertEqual(p.path.parent, rt)
            self.assertEqual(p.base_kind, "XDG_RUNTIME_DIR")
            for d in (p.path, p.relay_dir, p.llama_dir):
                info = os.lstat(d)
                self.assertEqual((stat.S_IMODE(info.st_mode), info.st_uid), (0o700, os.getuid()))
            self.assertLessEqual(sockets.check(p.relay_socket), sockets.SUN_PATH_MAX)
            self.assertLessEqual(sockets.check(p.llama_socket), sockets.SUN_PATH_MAX)
            made = p.path
        self.assertFalse(made.exists())

    def test_a_runtime_directory_that_is_not_private_falls_back_to_tmp(self):
        rt = self.runtime(mode=0o755)
        with sockets.Private({"XDG_RUNTIME_DIR": str(rt)}) as p:
            self.assertEqual((p.path.parent, p.base_kind), (Path("/tmp"), "/tmp"))
        with sockets.Private({}) as p:
            self.assertEqual(p.base_kind, "/tmp")

    def test_too_long_leaves_nothing_behind(self):
        """CNB2 of the review of r3: past `mkdtemp` every failure closes the lock and removes the
        directory; the error is `socket_path_too_long` before any bind."""
        rt = self.runtime(length=90)
        before = fds()
        with self.assertRaises(EngineError) as caught:
            sockets.Private({"XDG_RUNTIME_DIR": str(rt)})
        self.assertEqual(caught.exception.code, Code.SOCKET_PATH_TOO_LONG)
        self.assertEqual(list(rt.glob("mdf-*")), [])
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
        self.make("mdf-orphan", 120)
        self.make("mdf-nolock", 120, lock=False)
        self.make("mdf-young", 5)
        self.make("mdf-young-nolock", 5, lock=False)
        live = self.make("mdf-live", 120)
        fd = os.open(live / "lock", os.O_RDWR)
        self.addCleanup(os.close, fd)
        fcntl.flock(fd, fcntl.LOCK_EX)
        other = self.root / "not-ours"
        other.mkdir()
        removed, kept = sockets.sweep(self.root)
        self.assertEqual(sorted(removed), ["mdf-nolock", "mdf-orphan"])
        self.assertEqual(sorted(kept), ["mdf-live", "mdf-young", "mdf-young-nolock"])
        self.assertTrue(other.exists())

    def test_a_live_private_directory_is_kept(self):
        rt = Path(tempfile.mkdtemp(prefix="rt-"))
        self.addCleanup(shutil.rmtree, rt, True)
        with sockets.Private({"XDG_RUNTIME_DIR": str(rt)}) as p:
            removed, kept = sockets.sweep(rt, min_age=0)
            self.assertEqual((removed, kept), ([], [p.path.name]))


if __name__ == "__main__":
    unittest.main()
