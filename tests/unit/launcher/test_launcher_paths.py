"""The mounts of a call are derived by the launcher, never named by a request (MFL051, B2: SOL-R2-B2, CAR2-N5.2).

Three defences, each tested alone: the table (`mounts_of`: for every kind the four sister directories of the call,
each with its target and its `ro` or `rw`, and the volumes of the kind); the check of the finished list against a
wrong table (`check_mounts`: never the root of a call, an ancestor, another call, a level below a role, an overlap,
a target twice, a `..`); and the directories themselves on the volume (`check_folders`: `lstat` of every component,
real directories of the configured user, no link, distinct inodes), here on a synthetic volume.
"""
import os
from pathlib import Path
import tempfile
import unittest

import launcher_support as support
from markdown_for_llms import launcher
from markdown_for_llms.engine import containers

RO, RW = launcher.RO, launcher.RW
A, B = "mdf-eng-0123456789abcdef", "mdf-eng-fedcba9876543210"
EXPECTED = {
    ("tool", "toolchain"): [("work", "scratch", "/work", RW), ("work", "in", "/in", RO), ("work", "ctl", "/ctl", RO),
                            ("work", "stat", "/stat", RW)],
    ("light", "app"): [("work", "scratch", "/scratch", RW), ("work", "in", "/in", RO), ("work", "ctl", "/ctl", RO),
                       ("work", "stat", "/stat", RW)],
    ("marker", "app"): [("work", "scratch", "/scratch", RW), ("work", "in", "/in", RO), ("work", "ctl", "/ctl", RO),
                        ("work", "stat", "/stat", RW), ("models", "marker", "/models", RO),
                        ("run", "sock/%s/relay" % A, "/relay", RO)],
    ("cpuserver", "app"): [("work", "ctl", "/ctl", RO), ("work", "stat", "/stat", RW), ("cpuserver", "llama", "/opt/llama", RO),
                           ("cpuserver", "gguf", "/models", RO), ("run", "sock/%s/llama" % A, "/relay", RW)],
    ("probe", "app"): [("work", "scratch", "/scratch", RW), ("work", "ctl", "/ctl", RO), ("work", "stat", "/stat", RW)],
    ("probe", "toolchain"): [("work", "scratch", "/work", RW), ("work", "ctl", "/ctl", RO), ("work", "stat", "/stat", RW)],
    ("manifest", "toolchain"): [("work", "ctl", "/ctl", RO)],
}


def name_of(kind, prefix=A, n=1):
    return "%s-%s-%d" % (prefix, kind, n)


class TestDerivedMounts(unittest.TestCase):
    def test_every_kind_has_its_mounts_with_ro_and_rw(self):
        for (kind, image), expected in EXPECTED.items():
            with self.subTest(kind=kind, image=image):
                name = name_of(kind)
                mounts = launcher.mounts_of(kind, name, A, image)
                wanted = [(v, "call/%s/%s" % (name, s) if v == "work" else s, t, r) for v, s, t, r in expected]
                self.assertEqual(mounts, wanted)
                launcher.check_mounts(mounts, name, A)

    def test_the_candidate_image_has_the_mounts_of_the_toolchain(self):
        for kind in ("tool", "probe", "manifest"):
            self.assertEqual(launcher.mounts_of(kind, name_of(kind), A, "candidate"), launcher.mounts_of(kind, name_of(kind), A, "toolchain"))

    def test_nothing_writable_contains_or_is_ctl_or_in(self):
        for (kind, image) in EXPECTED:
            mounts = launcher.mounts_of(kind, name_of(kind), A, image)
            writable = [s for v, s, t, r in mounts if v == "work" and r is RW]
            protected = [s for v, s, t, r in mounts if v == "work" and r is RO]
            for w in writable:
                self.assertIn(w.rsplit("/", 1)[-1], ("scratch", "stat"))
                for p in protected:
                    self.assertFalse(w == p or w.startswith(p + "/") or p.startswith(w + "/"), (w, p))
            self.assertEqual([s.rsplit("/", 1)[-1] for s in protected], [r for r in containers.CALL_ROLES[kind] if r in ("in", "ctl")])

    def test_the_socket_directory_is_the_one_of_the_attempt_in_the_name(self):
        for prefix in (A, B, "mdf-eng-run.a_b-1", "mdf-bench-4242-0a1b2c3d"):
            mounts = launcher.mounts_of("marker", name_of("marker", prefix, 2), prefix, "app")
            self.assertEqual(mounts[-1], ("run", "sock/%s/relay" % prefix, "/relay", RO))

    def test_the_roles_of_the_volumes_become_the_exact_names_of_the_project(self):
        body = support.body_for(support.request("marker"))
        self.assertEqual([(m["Source"], m["Target"], m["ReadOnly"], m["VolumeOptions"]["Subpath"]) for m in body["HostConfig"]["Mounts"]], [
            ("mdf-unit_mdf-work", "/scratch", False, "call/%s/scratch" % name_of("marker")),
            ("mdf-unit_mdf-work", "/in", True, "call/%s/in" % name_of("marker")),
            ("mdf-unit_mdf-work", "/ctl", True, "call/%s/ctl" % name_of("marker")),
            ("mdf-unit_mdf-work", "/stat", False, "call/%s/stat" % name_of("marker")),
            ("mdf-unit_mdf-models", "/models", True, "marker"),
            ("mdf-unit_mdf-run", "/relay", True, "sock/%s/relay" % A)])
        body = support.body_for(support.request("cpuserver"))
        self.assertEqual([(m["Source"], m["Target"], m["ReadOnly"]) for m in body["HostConfig"]["Mounts"]][2:], [
            ("mdf-unit_mdf-cpu-server", "/opt/llama", True), ("mdf-unit_mdf-cpu-server", "/models", True),
            ("mdf-unit_mdf-run", "/relay", False)])


class TestFinishedList(unittest.TestCase):
    """`check_mounts` on a list a wrong table could give."""

    def setUp(self):
        self.name = name_of("tool")
        self.good = launcher.mounts_of("tool", self.name, A, "toolchain")

    def refused(self, mounts, contains, name=None, prefix=A):
        with self.assertRaises(launcher.Refusal) as refused:
            launcher.check_mounts(mounts, name or self.name, prefix)
        self.assertEqual(refused.exception.code, "invalid_setting")
        self.assertIn(contains, refused.exception.detail)

    def test_the_good_list_passes(self):
        launcher.check_mounts(self.good, self.name, A)

    def test_the_root_of_the_call_an_ancestor_another_call_a_level_below(self):
        n = self.name
        for what, bad, contains in (
                ("the root of the call rw", ("work", "call/" + n, "/work", RW), "root of the call"),
                ("the root of the call ro", ("work", "call/" + n, "/work", RO), "root of the call"),
                ("the ancestor call", ("work", "call", "/work", RW), "ancestor"),
                ("the ctl of the call as scratch", ("work", "call/%s/ctl" % n, "/work", RW), "writable"),
                ("the in of the call as scratch", ("work", "call/%s/in" % n, "/work", RW), "writable"),
                ("the scratch of another call", ("work", "call/%s/scratch" % name_of("tool", A, 2), "/work", RW), "outside the call"),
                ("the scratch of another attempt", ("work", "call/%s/scratch" % name_of("tool", B), "/work", RW), "outside the call"),
                ("a level below a role", ("work", "call/%s/in/x" % n, "/work", RW), "outside the call"),
                ("a role that is not of the kind", ("work", "call/%s/owner" % n, "/work", RW), "outside the call"),
                ("dot-dot", ("work", "call/%s/scratch/../ctl" % n, "/work", RW), "mount source"),
                ("a dot", ("work", "call/%s/./scratch" % n, "/work", RW), "mount source"),
                ("an absolute path", ("work", "/call/%s/scratch" % n, "/work", RW), "mount source"),
                ("an empty subpath", ("work", "", "/work", RW), "mount source"),
                ("a doubled slash", ("work", "call//%s/scratch" % n, "/work", RW), "mount source"),
                ("an unknown volume", ("host", "etc", "/work", RW), "mount source"),
                ("a subpath that is not a string", ("work", None, "/work", RW), "mount source")):
            with self.subTest(case=what):
                self.refused([bad] + self.good[1:], contains)

    def test_overlaps_and_targets(self):
        n = self.name
        self.refused(self.good + [self.good[0]], "overlapping sources")
        self.refused(self.good[:1] + [("work", "call/%s/scratch" % n, "/in", RO)] + self.good[2:], "overlapping sources")
        self.refused([self.good[0], ("work", "call/%s/in" % n, "/work", RO)] + self.good[2:], "the same target twice")
        self.refused([self.good[0], ("work", "call/%s/in" % n, "/work/in", RO)] + self.good[2:], "a target inside another")
        for target in ("in", "/", "/in/../etc", "/in/", "//in", None):
            self.refused([self.good[0], ("work", "call/%s/in" % n, target, RO)] + self.good[2:], "mount target")

    def test_the_other_volumes(self):
        name = name_of("marker")
        good = launcher.mounts_of("marker", name, A, "app")
        launcher.check_mounts(good, name, A)
        for bad, contains in ((("run", "sock/%s/relay" % B, "/relay", RO), "outside the socket directory"),
                              (("run", "sock", "/relay", RO), "outside the socket directory"),
                              (("run", "sock/%s" % A, "/relay", RO), "outside the socket directory"),
                              (("run", "sock/%s/relay/x" % A, "/relay", RO), "outside the socket directory"),
                              (("run", "sock/%s/other" % A, "/relay", RO), "outside the socket directory"),
                              (("run", "sock/%s/relay" % A, "/relay", RW), "writable"),
                              (("models", "other", "/relay", RW), "writable"),
                              (("cpuserver", "llama", "/relay", RW), "writable")):
            self.refused(good[:-1] + [bad], contains, name=name)

    def test_a_request_cannot_reach_any_of_this(self):
        """The request has no field that could carry a mount: the cases above are reachable only through a wrong
        table, which is why they are checked on the finished list."""
        self.assertEqual(launcher.REQUEST_KEYS["run"], {"op", "kind", "name", "image", "argv", "env", "limits"})
        sut, fake, _ = support.make()
        base = support.request("tool", "toolchain")
        parent = [{"volume": "work", "subpath": "call/" + base["name"], "target": "/work", "readonly": False}]
        answer, _ = sut.handle(dict(base, mounts=parent))
        self.assertEqual((answer["code"], answer["detail"]), ("invalid_setting", "unknown field: mounts"))
        self.assertEqual(fake.calls, [])


class TestFolders(unittest.TestCase):
    """`check_folders` on a synthetic work volume (the cases of the proof of the plan, `b2-mounts-proof`)."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.work, self.run = Path(tmp.name) / "work", Path(tmp.name) / "run"
        self.outside = Path(tmp.name) / "outside"
        self.outside.mkdir()
        self.name = name_of("marker")
        self.mounts = launcher.mounts_of("marker", self.name, A, "app")
        for role in ("scratch", "in", "ctl", "stat"):
            (self.work / "call" / self.name / role).mkdir(parents=True)
        (self.run / "sock" / A / "relay").mkdir(parents=True)
        self.roots = {"work": str(self.work), "run": str(self.run)}
        self.uid = os.getuid()

    def refused(self, contains, uid=None, mounts=None):
        with self.assertRaises(launcher.Refusal) as refused:
            launcher.check_folders(self.roots, mounts or self.mounts, self.uid if uid is None else uid)
        self.assertEqual(refused.exception.code, "invalid_setting")
        self.assertIn(contains, refused.exception.detail)

    def test_the_four_sister_directories_and_the_socket_directory_pass(self):
        launcher.check_folders(self.roots, self.mounts, self.uid)

    def test_a_volume_the_launcher_does_not_see_is_left_to_docker(self):
        launcher.check_folders({"work": str(self.work)}, self.mounts, self.uid)       # run, models: not seen here

    def test_a_missing_directory(self):
        (self.work / "call" / self.name / "stat").rmdir()
        self.refused("ENOENT")

    def test_a_component_that_is_a_link_inside_the_volume(self):
        other = self.work / "call" / name_of("marker", A, 9)
        (other / "ctl").mkdir(parents=True)
        ctl = self.work / "call" / self.name / "ctl"
        ctl.rmdir()
        os.symlink(other / "ctl", ctl)
        self.refused("not a directory of the user: work:call/%s/ctl" % self.name)

    def test_a_component_that_is_a_link_outside_the_volume(self):
        scratch = self.work / "call" / self.name / "scratch"
        scratch.rmdir()
        os.symlink(self.outside, scratch)
        self.refused("not a directory of the user")

    def test_the_folder_of_the_call_or_an_ancestor_is_a_link(self):
        call = self.work / "call" / self.name
        moved = self.work / "elsewhere"
        call.rename(moved)
        os.symlink(moved, call)
        self.refused("not a directory of the user: work:call/%s" % self.name)
        os.unlink(call)
        moved.rename(call)
        launcher.check_folders(self.roots, self.mounts, self.uid)
        calls = self.work / "call"
        calls.rename(self.work / "calls-moved")
        os.symlink(self.work / "calls-moved", calls)
        self.refused("not a directory of the user: work:call")

    def test_a_file_in_place_of_a_directory(self):
        stat_dir = self.work / "call" / self.name / "stat"
        stat_dir.rmdir()
        stat_dir.write_text("x")
        self.refused("not a directory of the user")

    def test_directories_of_another_user(self):
        self.refused("not a directory of the user: work:call", uid=self.uid + 1)

    def test_the_socket_directory_is_checked_like_the_call(self):
        relay = self.run / "sock" / A / "relay"
        relay.rmdir()
        os.symlink(self.outside, relay)
        self.refused("not a directory of the user: run:sock/%s/relay" % A)

    def test_two_roles_on_the_same_directory(self):
        name = self.name
        same = [("work", "call/%s/scratch" % name, "/scratch", RW), ("work", "call/%s/scratch" % name, "/stat", RW)]
        self.refused("two sources on the same directory", mounts=same)

    def test_the_launcher_refuses_the_run_and_docker_never_hears(self):
        sut, fake, _ = support.make(roots=self.roots)
        sut.config.uid = self.uid
        request = support.request("marker", name=self.name)
        self.assertTrue(sut.answer(__import__("json").dumps(request).encode(), self.uid)[0]["ok"])
        ctl = self.work / "call" / self.name / "ctl"
        ctl.rmdir()
        os.symlink(self.outside, ctl)
        answer, _ = sut.handle(dict(request, name=self.name))
        self.assertEqual(answer["code"], "invalid_setting")
        self.assertIn("not a directory of the user", answer["detail"])
        self.assertEqual(len(fake.did("POST", "/containers/create")), 1)


if __name__ == "__main__":
    unittest.main()
