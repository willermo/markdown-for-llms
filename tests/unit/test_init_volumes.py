"""`docker/app/init-volumes.sh`, the `init` service of compose (MFL051, plan r2.1 J.3.3; SOL-R2-N1, CAR2-N1).

A named volume is born of root. The service gives the root of each volume to the user, mode 0700, as UID 0 with
the capability CHOWN alone. Without CAP_FOWNER, UID 0 cannot `chmod` a directory that is no longer its own: so a
root that is already right is not touched at all (every start after the first), and one that is not is taken
first (`chown 0:0`), given its mode by its owner, and handed over. Never below the root, never through a link.

An ordinary user cannot run the three commands: recorders stand for them (`CHOWN`, `CHMOD`), and `STAT` for the
state the service reads. The real run, as UID 0 with CHOWN alone, is `make up` twice in a row.
"""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "docker" / "app" / "init-volumes.sh"
RECORDER = '#!/bin/sh\nprintf \'%s %s\\n\' "$(basename "$0")" "$*" >> "$CALLS"\n'
STAT = '#!/bin/sh\n# stat -c FORMAT DIR: the state the test gives, per directory\ncat "$3/.state"\n'


class InitVolumesTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        bin_dir = self.base / "bin"
        bin_dir.mkdir()
        for name, text in (("chown", RECORDER), ("chmod", RECORDER), ("stat", STAT)):
            (bin_dir / name).write_text(text)
            (bin_dir / name).chmod(0o755)
        self.calls = self.base / "calls"
        self.env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "CALLS": str(self.calls),
                    "CHOWN": str(bin_dir / "chown"), "CHMOD": str(bin_dir / "chmod"), "STAT": str(bin_dir / "stat")}

    def volume(self, name, state):
        path = self.base / name
        path.mkdir()
        (path / ".state").write_text(state + "\n")
        return path

    def run_init(self, *args, env=None):
        proc = subprocess.run(["/bin/sh", str(SCRIPT), *map(str, args)], capture_output=True, text=True,
                              env=self.env if env is None else env, timeout=60)
        calls = self.calls.read_text().splitlines() if self.calls.exists() else []
        return proc.returncode, proc.stdout, proc.stderr, calls

    def test_a_root_that_is_already_right_is_not_touched(self):
        work, run = self.volume("work", "1000 1000 700"), self.volume("run", "1000 1000 700")
        status, out, err, calls = self.run_init(1000, 1000, work, run)
        self.assertEqual((status, calls, err), (0, [], ""))
        self.assertEqual(out.splitlines(), ["init: %s ok" % work, "init: %s ok" % run])

    def test_a_root_of_root_is_taken_given_its_mode_and_handed_over_in_this_order(self):
        fresh = self.volume("fresh", "0 0 755")
        status, out, _, calls = self.run_init(1000, 1001, fresh)
        self.assertEqual(status, 0)
        self.assertEqual(calls, ["chown 0:0 %s" % fresh, "chmod 0700 %s" % fresh, "chown 1000:1001 %s" % fresh])
        self.assertIn("prepared (was 0 0 755)", out)
        self.assertFalse(any("-R" in call for call in calls))                    # never below the root

    def test_a_root_of_the_user_with_another_mode_goes_the_same_way(self):
        """The order matters here: `chmod` first would be refused, without CAP_FOWNER, on a directory of the user."""
        loose = self.volume("loose", "1000 1000 755")
        other = self.volume("other", "1000 2000 700")
        right = self.volume("right", "1000 1000 700")
        status, _, _, calls = self.run_init(1000, 1000, loose, right, other)
        self.assertEqual(status, 0)
        self.assertEqual(calls, ["chown 0:0 %s" % loose, "chmod 0700 %s" % loose, "chown 1000:1000 %s" % loose,
                                 "chown 0:0 %s" % other, "chmod 0700 %s" % other, "chown 1000:1000 %s" % other])

    def test_a_link_or_a_file_in_place_of_a_root_is_refused_before_any_command(self):
        target = self.volume("target", "0 0 755")
        link = self.base / "link"
        link.symlink_to(target)
        plain = self.base / "plain"
        plain.write_text("x")
        for bad in (link, plain, self.base / "absent"):
            with self.subTest(bad=bad.name):
                status, _, err, calls = self.run_init(1000, 1000, bad)
                self.assertEqual((status, calls), (1, []))
                self.assertIn("not a directory", err)

    def test_the_first_root_that_is_wrong_stops_the_rest(self):
        link = self.base / "link"
        link.symlink_to(self.volume("target", "0 0 755"))
        after = self.volume("after", "0 0 755")
        status, _, _, calls = self.run_init(1000, 1000, link, after)
        self.assertEqual((status, calls), (1, []))

    def test_the_identity_is_two_numbers_and_never_root(self):
        work = self.volume("work", "0 0 755")
        for uid, gid in (("0", "0"), ("user", "1000"), ("1000", ""), ("1000", "10x")):
            with self.subTest(uid=uid, gid=gid):
                self.calls.unlink(missing_ok=True)
                status, _, _, calls = self.run_init(uid, gid, work)
                self.assertEqual((status, calls), (2, []))
        self.assertEqual(self.run_init()[0], 2)

    def test_the_real_commands_on_a_root_that_is_right_run_without_any_privilege(self):
        """With the real `stat`, on a directory of this user with mode 0700: nothing to do, so nothing can fail."""
        real = self.base / "real"
        real.mkdir(mode=0o700)
        status, out, err, _ = self.run_init(os.getuid(), os.getgid(), real,
                                            env={"PATH": os.environ.get("PATH", "/usr/bin:/bin")})
        self.assertEqual((status, err, out.strip()), (0, "", "init: %s ok" % real))


if __name__ == "__main__":
    unittest.main()
