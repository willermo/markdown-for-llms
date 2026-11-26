"""Attestation and collection of what an attempt left (MFL051, plan r2.1 App. H.4, J.4.4; SOL-R2-B1 on the side of
the engine).

`stop_attempt_children`: no child of the attempt is left only when the launcher says so after kill and remove; the
launcher unreachable, or a removal not confirmed, is `Unattested`. `sweep`: a folder of `call/` goes away only when
its `owner` is free AND the launcher attests that the container does not exist; held, still listed, or with the
launcher unreachable, it stays, with its reason. A name is among the removed only when its folder is gone
(SOL-A1-N2): a removal that failed keeps it, as `not_removed`, and so does `Call.close`.
"""
import fcntl
import os
from pathlib import Path
import stat
import tempfile
import time
import unittest
from unittest import mock

import containers_test_support as support
from markdown_for_llms.engine import containers

OTHER = "mdf-eng-fedcba9876543210"


class Base(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.work = Path(tmp.name)
        self.launcher = support.FakeLauncher()

    def folder(self, name, held=False):
        """The folder of a call whose caller ended (`held`: is still alive)."""
        call = containers.Call(self.work, name)
        if held:
            self.addCleanup(call.close, False)
        else:
            call.close(remove=False)
        return call.root

    def running(self, name, state="running"):
        self.launcher.children[name] = {"state": state, "exit": 0, "oom_killed": False}


class TestStopAttemptChildren(Base):
    def test_no_child_is_the_normal_case(self):
        self.assertEqual(containers.stop_attempt_children(self.launcher, support.PREFIX), [])
        self.assertEqual([c[0] for c in self.launcher.calls], ["list", "list"])

    def test_children_are_killed_removed_and_the_list_is_read_again(self):
        for n in (1, 2):
            self.running(support.name("light", n))
        self.running(support.name("tool", 3), state="exited")
        self.running(support.name("light", 1, OTHER))
        removed = containers.stop_attempt_children(self.launcher, support.PREFIX)
        self.assertEqual(removed, [support.name("light", 1), support.name("light", 2), support.name("tool", 3)])
        self.assertEqual(list(self.launcher.children), [support.name("light", 1, OTHER)])     # another attempt is not touched
        self.assertEqual([c[0] for c in self.launcher.calls], ["list", "kill", "remove", "kill", "remove", "kill", "remove", "list"])

    def test_the_launcher_unreachable_is_unattested(self):
        self.running(support.name())
        self.launcher.down = True
        with self.assertRaises(containers.Unattested) as raised:
            containers.stop_attempt_children(self.launcher, support.PREFIX)
        self.assertIn("launcher", str(raised.exception))
        self.assertIn(support.name(), self.launcher.children)

    def test_a_removal_that_is_not_confirmed_is_unattested(self):
        self.running(support.name())
        self.launcher.unconfirmed.add(support.name())
        with self.assertRaises(containers.Unattested):
            containers.stop_attempt_children(self.launcher, support.PREFIX)

        class SaysNotAbsent(support.FakeLauncher):
            def remove(self, name):
                self._answering("remove", name)
                return {"ok": True, "absent": False}
        launcher = SaysNotAbsent()
        launcher.children[support.name()] = {"state": "running", "exit": 0, "oom_killed": False}
        with self.assertRaises(containers.Unattested) as raised:
            containers.stop_attempt_children(launcher, support.PREFIX)
        self.assertIn("removal not confirmed", str(raised.exception))

    def test_a_child_still_listed_after_the_removals_is_unattested(self):
        class ComesBack(support.FakeLauncher):
            def remove(self, name):
                self._answering("remove", name)
                return {"ok": True, "absent": True}             # says absent, and the list still has it
        launcher = ComesBack()
        launcher.children[support.name()] = {"state": "running", "exit": 0, "oom_killed": False}
        with self.assertRaises(containers.Unattested) as raised:
            containers.stop_attempt_children(launcher, support.PREFIX)
        self.assertIn("still listed", str(raised.exception))

    def test_a_prefix_out_of_the_schema_is_an_error_of_the_caller(self):
        for prefix in ("", "mdf-eng-", support.name(), "x", "mdf-eng-" + "a" * 17):
            with self.assertRaises(ValueError):
                containers.stop_attempt_children(self.launcher, prefix)
        self.assertEqual(self.launcher.calls, [])


class TestSweep(Base):
    def sweep(self, **kw):
        kw.setdefault("min_age", 0)
        return containers.sweep(self.launcher, self.work, **kw)

    def test_free_and_attested_absent_goes_away(self):
        folder = self.folder(support.name())
        (folder / "scratch" / "left.bin").write_bytes(b"x" * 1000)
        (folder / "ctl" / "stop").write_text("")
        self.assertEqual(self.sweep(), ([support.name()], {}))
        self.assertFalse(folder.exists())
        self.assertEqual(self.launcher.ops("inspect"), [("inspect", support.name())])

    def test_a_folder_whose_owner_is_held_stays(self):
        folder = self.folder(support.name(), held=True)
        self.assertEqual(self.sweep(), ([], {support.name(): "held"}))
        self.assertTrue(folder.is_dir())
        self.assertEqual(self.launcher.calls, [])               # a live call: the launcher is not even asked

    def test_a_folder_whose_container_is_still_there_stays(self):
        for n, state in enumerate(("running", "exited", "created")):
            name = support.name("light", n + 1)
            folder = self.folder(name)
            self.running(name, state)
            self.assertEqual(self.sweep(prefix=None)[1], {support.name("light", k + 1): "container" for k in range(n + 1)})
            self.assertTrue(folder.is_dir())

    def test_with_the_launcher_unreachable_nothing_goes_away(self):
        folders = [self.folder(support.name("light", n)) for n in (1, 2, 3)]
        self.launcher.down = True
        removed, kept = self.sweep()
        self.assertEqual((removed, sorted(kept.values())), ([], ["unattested"] * 3))
        self.assertTrue(all(f.is_dir() for f in folders))
        self.launcher.down = False
        self.assertEqual(len(self.sweep()[0]), 3)

    def test_a_refusal_of_the_launcher_keeps_the_folder(self):
        folder = self.folder(support.name())

        class Refuses(support.FakeLauncher):
            def inspect(self, name):
                raise containers.Refused("invalid_setting", "not a child of this project")
        self.assertEqual(containers.sweep(Refuses(), self.work, min_age=0), ([], {support.name(): "unattested"}))
        self.assertTrue(folder.is_dir())

    def test_the_sweep_of_one_attempt_leaves_the_others(self):
        mine, other = self.folder(support.name()), self.folder(support.name("light", 1, OTHER))
        self.assertEqual(self.sweep(prefix=support.PREFIX), ([support.name()], {}))
        self.assertTrue(other.is_dir())
        self.assertFalse(mine.exists())

    def test_what_is_not_the_folder_of_a_call_is_never_touched(self):
        calls = self.work / "call"
        calls.mkdir()
        (calls / "notes.txt").write_text("x")
        (calls / "not-a-call").mkdir()
        (calls / (support.PREFIX + "-shell-1")).mkdir()
        outside = self.work / "outside"
        outside.mkdir()
        (outside / "keep").write_text("x")
        os.symlink(outside, calls / support.name("light", 5))           # a link with the name of a call
        (calls / support.name("light", 6)).write_text("a file")
        self.assertEqual(self.sweep(), ([], {}))
        self.assertEqual(sorted(os.listdir(calls)), sorted(["notes.txt", "not-a-call", support.PREFIX + "-shell-1",
                                                            support.name("light", 5), support.name("light", 6)]))
        self.assertTrue((outside / "keep").exists())
        self.assertEqual(self.launcher.calls, [])

    def test_a_folder_without_its_lock_waits_until_it_is_old(self):
        """Killed between the folder and its lock: no owner. It may also be a folder that is being made right now."""
        folder = self.work / "call" / support.name()
        folder.mkdir(parents=True, mode=0o700)
        self.assertEqual(containers.sweep(self.launcher, self.work), ([], {support.name(): "young"}))
        old = time.time() - containers.ORPHAN_AGE - 5
        os.utime(folder, (old, old))
        self.assertEqual(containers.sweep(self.launcher, self.work), ([support.name()], {}))

    def test_no_call_folder_at_all(self):
        self.assertEqual(self.sweep(), ([], {}))

    def test_an_owner_that_cannot_be_opened_keeps_the_folder(self):
        folder = self.folder(support.name())
        (folder / "owner").unlink()
        os.symlink(self.work / "elsewhere", folder / "owner")           # a link in place of the lock is not followed
        self.assertEqual(self.sweep(), ([], {support.name(): "owner"}))
        self.assertTrue(folder.is_dir())

    def test_the_reasons_are_a_closed_list(self):
        self.assertEqual(containers.SWEEP_KEPT, ("held", "young", "owner", "unattested", "container", "not_removed"))

    def locked_folder(self):
        folder = self.folder(support.name())
        locked = folder / "scratch" / "closed-directory"
        (locked / "deeper").mkdir(parents=True)
        (locked / "deeper" / "data").write_text("synthetic output")
        (locked / "deeper").chmod(0)
        locked.chmod(0)
        self.addCleanup(lambda: locked.exists() and (locked.chmod(0o700), (locked / "deeper").chmod(0o700)))
        return folder, locked

    def test_a_folder_that_cannot_be_removed_is_kept_and_never_reported_as_removed(self):
        """SOL-A1-N2. What a command left in `scratch` may not be removable (here directories that cannot be
        entered): when its permissions cannot be given back either, the folder stays on the volume, so its name
        stays among the kept, with the reason, at every pass."""
        folder, locked = self.locked_folder()
        with mock.patch.object(containers, "restore_permissions", lambda root: None):
            self.assertEqual(self.sweep(), ([], {support.name(): "not_removed"}))
            self.assertTrue(folder.is_dir())
            self.assertTrue(locked.is_dir())
            self.assertEqual(self.sweep(), ([], {support.name(): "not_removed"}))      # and again: the owner is still free

    def test_the_permissions_a_command_took_away_are_given_back_and_the_folder_goes(self):
        """A2-d. With the owner free and the absence of the container attested, the sweep gives the directories
        of the folder their permissions back and tries again: the folder goes at that pass."""
        folder, locked = self.locked_folder()
        self.assertEqual(self.sweep(), ([support.name()], {}))
        self.assertFalse(folder.exists())

    def test_the_permissions_are_given_back_only_inside_the_folder_and_through_no_link(self):
        """A2-d. What the folder points to is not touched: a link to a directory outside it, whose permissions were
        taken away, is removed as a link, and that directory keeps its mode."""
        outside = self.work / "outside"
        (outside / "inner").mkdir(parents=True)
        (outside / "inner").chmod(0)
        outside.chmod(0o500)
        self.addCleanup(lambda: (outside.chmod(0o700), (outside / "inner").chmod(0o700)))
        folder, locked = self.locked_folder()
        os.symlink(outside, folder / "scratch" / "to-outside")
        os.symlink(outside / "inner", folder / "scratch" / "to-inner")
        self.assertEqual(self.sweep(), ([support.name()], {}))
        self.assertFalse(folder.exists())
        self.assertEqual((stat.S_IMODE(outside.stat().st_mode), stat.S_IMODE(os.lstat(outside / "inner").st_mode)), (0o500, 0))

    def test_nothing_is_given_back_while_the_call_lives_or_its_container_is_not_attested_absent(self):
        """A2-d: only with the owner free and the absence attested."""
        folder, locked = self.locked_folder()
        calls = []
        with mock.patch.object(containers, "restore_permissions", lambda root: calls.append(root)):
            self.launcher.children[support.name()] = {"state": "exited", "exit": 0, "oom_killed": False}
            self.assertEqual(self.sweep(), ([], {support.name(): "container"}))
            del self.launcher.children[support.name()]
            self.launcher.down = True
            self.assertEqual(self.sweep(), ([], {support.name(): "unattested"}))
            self.launcher.down = False
            fd = os.open(folder / "owner", os.O_RDWR)
            fcntl.flock(fd, fcntl.LOCK_EX)
            self.addCleanup(os.close, fd)
            self.assertEqual(self.sweep(), ([], {support.name(): "held"}))
        self.assertEqual(calls, [])
        self.assertEqual(stat.S_IMODE(os.lstat(locked).st_mode), 0)

    def test_close_says_whether_the_folder_is_gone(self):
        call = containers.Call(self.work, support.name())
        locked = call.scratch / "closed-directory"
        locked.mkdir()
        (locked / "data").write_text("synthetic output")
        locked.chmod(0)
        self.addCleanup(lambda: locked.exists() and locked.chmod(0o700))
        self.assertIs(call.close(remove=True), False)
        self.assertTrue(call.root.is_dir())
        # what could go went, the lock too: the folder is now one without an owner, which `sweep` takes when it is old
        self.assertEqual([str(p.relative_to(call.root)) for p in sorted(call.root.rglob("*"))], ["scratch", "scratch/closed-directory"])
        self.assertIsNone(call.owner_fd)
        self.assertEqual(containers.sweep(self.launcher, self.work), ([], {support.name(): "young"}))
        self.assertTrue(locked.is_dir())                    # the call itself gives no permission back: the sweep does
        self.assertIs(containers.Call(self.work, support.name("light", 2)).close(remove=True), True)
        self.assertIs(containers.Call(self.work, support.name("light", 3)).close(remove=False), True)   # nothing was to go
        self.assertEqual(sorted(os.listdir(self.work / "call")), [support.name(), support.name("light", 3)])
        self.assertEqual(containers.sweep(self.launcher, self.work, min_age=0), ([support.name(), support.name("light", 3)], {}))

    def test_a_child_whose_folder_cannot_be_removed_says_so(self):
        clock = support.Clock()
        with containers.Child(self.launcher, self.work, support.name(), clock=clock, sleep=clock.sleep) as child:
            locked = child.call.scratch / "closed-directory"
            clock.at(0.3, lambda: (locked.mkdir(), (locked / "data").write_text("x"), locked.chmod(0), self.launcher.exits(child.name, 0)))
            self.addCleanup(lambda: locked.exists() and locked.chmod(0o700))
            outcome = child.run("app", containers.PROBE_PYTHON, {}, support.LIMITS, timeout=600)
        self.assertEqual((outcome.cause, child.attested, child.removed), ("exited", True, False))
        self.assertTrue(child.call.root.is_dir())
        with containers.Child(self.launcher, self.work, support.name("light", 2), clock=clock, sleep=clock.sleep) as other:
            clock.at(0.3, lambda: self.launcher.exits(other.name, 0))
            other.run("app", containers.PROBE_PYTHON, {}, support.LIMITS, timeout=600)
        self.assertEqual((other.removed, other.call.root.exists()), (True, False))

    def test_the_folder_kept_after_an_end_that_was_not_attested_is_collected_after_the_attestation(self):
        """The whole path on the side of the engine: the launcher goes away, the call ends without an attestation and
        keeps its folder; the attempt cannot be attested; the launcher comes back, the child is removed, the folder
        goes."""
        clock = support.Clock()
        child = containers.Child(self.launcher, self.work, support.name(), clock=clock, sleep=clock.sleep)
        folder = child.call.root
        with self.assertRaises(Exception), child:
            clock.at(1.0, lambda: setattr(self.launcher, "down", True))
            child.run("app", containers.PROBE_PYTHON, {}, support.LIMITS, timeout=600)
        self.assertTrue((folder / "ctl" / "stop").exists())
        with self.assertRaises(containers.Unattested):
            containers.stop_attempt_children(self.launcher, support.PREFIX)
        self.assertEqual(self.sweep(), ([], {support.name(): "unattested"}))
        self.launcher.down = False
        self.assertEqual(self.sweep(), ([], {support.name(): "container"}))     # the container is still listed: it stays
        self.assertEqual(containers.stop_attempt_children(self.launcher, support.PREFIX), [support.name()])
        self.assertEqual(self.sweep(prefix=support.PREFIX), ([support.name()], {}))
        self.assertFalse(folder.exists())


if __name__ == "__main__":
    unittest.main()
