"""`engine/envfile.read`: the file of the keys must still have a name (MFL051, E-1; plan r2.1 J.5.2).

In a container the env file of the user is ONE file bound from the host, and the mount keeps the inode it had
when the container started. After the atomic save of an editor (`os.replace`), a `sed -i` or an `rm` on the host,
the path in the container still opens the old file: regular, of the user, 0600, with the old keys. The reader
refuses an inode left without names (`st_nlink < 1`): a replacement always needs `make restart`.

The declared limit: when the old inode still has a name (a backup made with a hard link, or by renaming the file
before the new one is written), it can still be read until the restart. The tests keep it visible.

The mount is stood for by a descriptor held open on the file, given to the reader in place of the path: that is
what a bind mount of one file is, a reference to the inode.
"""
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from markdown_for_llms.engine import envfile

OLD, NEW = "old-synthetic-value", "new-synthetic-value"
MOUNT = "/run/mdf/env"                  # where the containers of the product see the env file


class Mounted:
    """The env file as a container sees it: the inode the file had when it was `mounted`, whatever the path
    holds afterwards."""

    def __init__(self, test, path):
        self.fd = os.open(path, os.O_RDONLY)
        test.addCleanup(os.close, self.fd)
        real = envfile._open
        patcher = mock.patch.object(envfile, "_open", lambda env_file: (
            os.open("/proc/self/fd/%d" % self.fd, os.O_RDONLY) if env_file == MOUNT else real(env_file)))
        patcher.start()
        test.addCleanup(patcher.stop)


class NameTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "env"
        self.write(self.path, OLD)

    def write(self, path, value):
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as stream:
            stream.write("SERVICE_API_KEY=%s\n" % value)

    def read(self):
        return envfile.read(MOUNT, ["SERVICE_API_KEY"])

    def test_a_file_with_its_name_is_read_at_every_call(self):
        self.assertEqual(envfile.read(self.path, ["SERVICE_API_KEY"]), {"SERVICE_API_KEY": OLD})
        Mounted(self, self.path)
        self.assertEqual(self.read(), {"SERVICE_API_KEY": OLD})
        os.link(self.path, self.path.with_name("env.second"))                   # two names: still a file with a name
        self.assertEqual(os.stat(self.path).st_nlink, 2)
        self.assertEqual(self.read(), {"SERVICE_API_KEY": OLD})

    def test_a_file_removed_after_the_mount_is_refused(self):
        Mounted(self, self.path)
        self.path.unlink()
        with self.assertRaises(envfile.Replaced) as ctx:
            self.read()
        self.assertIn("make restart", str(ctx.exception))
        self.assertNotIn(OLD, str(ctx.exception))

    def test_a_file_replaced_atomically_after_the_mount_is_refused(self):
        Mounted(self, self.path)
        fresh = self.path.with_name("env.tmp")
        self.write(fresh, NEW)
        os.replace(fresh, self.path)                                            # what an editor, or `sed -i`, does
        with self.assertRaises(envfile.Unsafe) as ctx:                          # `Replaced` is an `Unsafe`: every caller
            self.read()                                                         # already maps it to its closed code
        self.assertIsInstance(ctx.exception, envfile.Replaced)
        self.assertEqual(envfile.read(self.path, ["SERVICE_API_KEY"]), {"SERVICE_API_KEY": NEW})   # a new container

    def test_the_mode_is_read_at_every_call(self):
        Mounted(self, self.path)
        os.chmod(self.path, 0o644)
        with self.assertRaises(envfile.Unsafe) as ctx:
            self.read()
        self.assertNotIsInstance(ctx.exception, envfile.Replaced)
        os.chmod(self.path, 0o600)
        self.assertEqual(self.read(), {"SERVICE_API_KEY": OLD})

    def test_limit_an_old_file_kept_by_a_backup_is_still_read_until_the_restart(self):
        """The declared limit of E-1, in its two forms: the old inode keeps a name, so the reader cannot tell it
        from the file of the user. It goes away with `make restart`, which mounts the file of the path."""
        Mounted(self, self.path)
        os.link(self.path, self.path.with_name("env.bak"))                      # a backup made with a hard link
        fresh = self.path.with_name("env.tmp")
        self.write(fresh, NEW)
        os.replace(fresh, self.path)
        self.assertEqual(self.read(), {"SERVICE_API_KEY": OLD})

    def test_limit_a_file_renamed_before_the_new_one_is_written(self):
        Mounted(self, self.path)
        os.rename(self.path, self.path.with_name("env~"))                       # the backup of an editor
        self.write(self.path, NEW)
        self.assertEqual(self.read(), {"SERVICE_API_KEY": OLD})


if __name__ == "__main__":
    unittest.main()
