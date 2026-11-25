"""The verification after the creation (MFL051, plan r2.1 App. H.2), against a Docker that is not there.

Positive: for every kind, the inspect of a REAL container created on Docker 29.10 from the launcher's body
(`tests/fixtures/launcher/inspect-*.json`, recorded by `tests/container/test_child_isolation.py`, see the `README.txt`
of the fixtures) passes, with its `Hostname`, `Image`, `Cmd`, `Entrypoint` and the defaults of the daemon; the child
is started. Negative: one field at a time of that inspect is altered; the launcher names it, removes the container
without starting it and answers `isolation_unavailable`. A field it does not know, with a value, is refused too.
"""
import copy
import unittest

import launcher_support as support
from markdown_for_llms import launcher
from markdown_for_llms.engine import toolchain

IGNORED = "ignored"


def replay(fixture, alter=None):
    """Send the request of a fixture to a launcher whose Docker reads the created container back as the recorded
    inspect (altered by `alter`); returns (answer, fake Docker)."""
    context = fixture["context"]
    entry = dict(context["image"])
    fake = support.FakeDocker({entry["id"]: entry["inspect"]})
    images = {}
    if fixture["request"]["image"] == "app":
        images["app"] = entry
    else:
        fake.images[toolchain.tag()] = entry["inspect"]
        images["app"] = {"id": support.APP_ID, "architecture": "amd64", "inspect": support.image_inspect(support.APP_ID, support.APP_ENV)}
    config = launcher.Config(context["project"], context["uid"], context["gid"], context["volumes"], {}, images, fixture["docker"])
    clock = support.Clock(context["t0"])
    sut = launcher.Launcher(fake, config, clock=clock, sleep=clock.sleep, identity={})
    sut.verified.add(entry["id"])                   # the verification of the toolchain image has its own tests

    def read_back(synthesized):
        found = copy.deepcopy(fixture["inspect"])
        if alter is not None:
            alter(found)
        return found
    fake.alter = read_back
    answer, _ = sut.handle(fixture["request"])
    return answer, fake


def setter(section, key, value):
    def alter(found):
        target = found if section is None else found[section]
        target[key] = value(target.get(key)) if callable(value) else value
    return alter


# (section, field, altered value or function of the recorded one, the name the launcher must give)
ALTERED = (
    (None, "Image", "sha256:" + "b" * 64, "Image"),
    (None, "Path", "/bin/sh", "Path"),
    (None, "Args", lambda v: v[:-1], "Args"),
    (None, "AppArmorProfile", "unconfined", "AppArmorProfile"),
    ("Config", "Image", "sha256:" + "b" * 64, "Config.Image"),
    ("Config", "Entrypoint", ["/bin/sh", "-c", 'exec "$@"', "sh"], "Config.Entrypoint"),
    ("Config", "Entrypoint", lambda v: v[:4] + ["999999"] + v[5:], "Config.Entrypoint"),
    ("Config", "Cmd", ["/bin/sh"], "Config.Cmd"),
    ("Config", "User", "0:0", "Config.User"),
    ("Config", "User", "", "Config.User"),
    ("Config", "Env", lambda v: v + ["LD_PRELOAD=/scratch/x.so"], "Config.Env"),
    ("Config", "Env", lambda v: v[1:], "Config.Env"),
    ("Config", "Tty", True, "Config.Tty"),
    ("Config", "OpenStdin", True, "Config.OpenStdin"),
    ("Config", "StdinOnce", True, "Config.StdinOnce"),
    ("Config", "AttachStdin", True, "Config.AttachStdin"),
    ("Config", "AttachStdout", True, "Config.AttachStdout"),
    ("Config", "AttachStderr", True, "Config.AttachStderr"),
    ("Config", "WorkingDir", "/", "Config.WorkingDir"),
    ("Config", "Labels", lambda v: dict(v, **{"mdf.project": "another"}), "Config.Labels"),
    ("Config", "Labels", lambda v: dict(v, **{"mdf.max": "999999"}), "Config.Labels"),
    ("Config", "Volumes", {"/data": {}}, "Config.Volumes"),
    ("Config", "Hostname", "elsewhere", "Config.Hostname"),
    ("Config", "Domainname", "example.org", "Config.Domainname"),
    ("Config", "NetworkDisabled", False, "Config.NetworkDisabled"),
    ("Config", "ExposedPorts", {"80/tcp": {}}, "Config.ExposedPorts"),
    ("Config", "Healthcheck", {"Test": ["CMD", "true"]}, "Config.Healthcheck"),
    ("Config", "MacAddress", "02:42:ac:11:00:02", "Config.MacAddress"),
    ("Config", "StopSignal", "SIGUSR1", "Config.StopSignal"),
    ("Config", "Shell", ["/bin/sh"], "Config.Shell"),
    ("Config", "NotAFieldTheLauncherKnows", "value", "Config.NotAFieldTheLauncherKnows"),
    ("HostConfig", "NetworkMode", "bridge", "HostConfig.NetworkMode"),
    ("HostConfig", "NetworkMode", "host", "HostConfig.NetworkMode"),
    ("HostConfig", "ReadonlyRootfs", False, "HostConfig.ReadonlyRootfs"),
    ("HostConfig", "CapAdd", ["SYS_ADMIN"], "HostConfig.CapAdd"),
    ("HostConfig", "CapDrop", None, "HostConfig.CapDrop"),
    ("HostConfig", "CapDrop", ["NET_RAW"], "HostConfig.CapDrop"),
    ("HostConfig", "SecurityOpt", None, "HostConfig.SecurityOpt"),
    ("HostConfig", "SecurityOpt", ["no-new-privileges", "seccomp=unconfined"], "HostConfig.SecurityOpt"),
    ("HostConfig", "Privileged", True, "HostConfig.Privileged"),
    ("HostConfig", "Init", True, "HostConfig.Init"),
    ("HostConfig", "Memory", lambda v: v * 2, "HostConfig.Memory"),
    ("HostConfig", "Memory", 0, "HostConfig.Memory"),
    ("HostConfig", "MemorySwap", -1, "HostConfig.MemorySwap"),
    ("HostConfig", "PidsLimit", None, "HostConfig.PidsLimit"),
    ("HostConfig", "PidsLimit", -1, "HostConfig.PidsLimit"),
    ("HostConfig", "ShmSize", 1 << 30, "HostConfig.ShmSize"),
    ("HostConfig", "Ulimits", lambda v: v[1:], "HostConfig.Ulimits"),
    ("HostConfig", "Ulimits", lambda v: v + [{"Name": "memlock", "Soft": -1, "Hard": -1}], "HostConfig.Ulimits"),
    ("HostConfig", "Tmpfs", {"/tmp": "rw,exec,size=16m"}, "HostConfig.Tmpfs"),
    ("HostConfig", "Tmpfs", lambda v: dict(v, **{"/run": ""}), "HostConfig.Tmpfs"),
    ("HostConfig", "Binds", ["/:/host"], "HostConfig.Binds"),
    ("HostConfig", "Mounts", lambda v: [dict(m, ReadOnly=False) for m in v], "HostConfig.Mounts"),
    ("HostConfig", "Mounts", lambda v: [dict(v[0], VolumeOptions=dict(v[0]["VolumeOptions"], Subpath="call"))] + v[1:], "HostConfig.Mounts"),
    ("HostConfig", "Mounts", lambda v: [dict(v[0], VolumeOptions={"NoCopy": True})] + v[1:], "HostConfig.Mounts"),
    ("HostConfig", "Mounts", lambda v: [dict(v[0], VolumeOptions=dict(v[0]["VolumeOptions"], NoCopy=False))] + v[1:], "HostConfig.Mounts"),
    ("HostConfig", "Mounts", lambda v: v + [{"Type": "bind", "Source": "/", "Target": "/host"}], "HostConfig.Mounts"),
    ("HostConfig", "Mounts", lambda v: v[:-1], "HostConfig.Mounts"),
    ("HostConfig", "Mounts", lambda v: [dict(v[0], Source="another_volume")] + v[1:], "HostConfig.Mounts"),
    ("HostConfig", "Mounts", lambda v: [dict(v[0], BindOptions={"Propagation": "shared"})] + v[1:], "HostConfig.Mounts"),
    ("HostConfig", "Devices", [{"PathOnHost": "/dev/kvm", "PathInContainer": "/dev/kvm", "CgroupPermissions": "rwm"}], "HostConfig.Devices"),
    ("HostConfig", "DeviceRequests", [{"Driver": "nvidia", "Count": -1}], "HostConfig.DeviceRequests"),
    ("HostConfig", "DeviceCgroupRules", ["c 1:3 mr"], "HostConfig.DeviceCgroupRules"),
    ("HostConfig", "PortBindings", {"80/tcp": [{"HostPort": "8080"}]}, "HostConfig.PortBindings"),
    ("HostConfig", "PublishAllPorts", True, "HostConfig.PublishAllPorts"),
    ("HostConfig", "RestartPolicy", {"Name": "always", "MaximumRetryCount": 0}, "HostConfig.RestartPolicy"),
    ("HostConfig", "RestartPolicy", {"Name": "no", "MaximumRetryCount": 3}, "HostConfig.RestartPolicy"),
    ("HostConfig", "AutoRemove", True, "HostConfig.AutoRemove"),
    ("HostConfig", "IpcMode", "host", "HostConfig.IpcMode"),
    ("HostConfig", "PidMode", "host", "HostConfig.PidMode"),
    ("HostConfig", "UTSMode", "host", "HostConfig.UTSMode"),
    ("HostConfig", "UsernsMode", "host", "HostConfig.UsernsMode"),
    ("HostConfig", "CgroupnsMode", "host", "HostConfig.CgroupnsMode"),
    ("HostConfig", "CgroupParent", "/system.slice", "HostConfig.CgroupParent"),
    ("HostConfig", "GroupAdd", ["126"], "HostConfig.GroupAdd"),
    ("HostConfig", "ExtraHosts", ["host.docker.internal:host-gateway"], "HostConfig.ExtraHosts"),
    ("HostConfig", "Dns", ["198.51.100.1"], "HostConfig.Dns"),
    ("HostConfig", "Links", ["db:db"], "HostConfig.Links"),
    ("HostConfig", "VolumesFrom", ["other"], "HostConfig.VolumesFrom"),
    ("HostConfig", "OomKillDisable", True, "HostConfig.OomKillDisable"),
    ("HostConfig", "Sysctls", {"net.ipv4.ip_forward": "1"}, "HostConfig.Sysctls"),
    ("HostConfig", "LogConfig", {"Type": "json-file", "Config": {}}, "HostConfig.LogConfig"),
    ("HostConfig", "LogConfig", lambda v: dict(v, Config=dict(v["Config"], tag="x")), "HostConfig.LogConfig"),
    ("HostConfig", "Runtime", "another-runtime", "HostConfig.Runtime"),
    ("HostConfig", "MaskedPaths", [], "HostConfig.MaskedPaths"),
    ("HostConfig", "MaskedPaths", lambda v: v + ["/etc/shadow"], "HostConfig.MaskedPaths"),
    ("HostConfig", "ReadonlyPaths", None, "HostConfig.ReadonlyPaths"),
    ("HostConfig", "ConsoleSize", [24, 80], "HostConfig.ConsoleSize"),
    ("HostConfig", "Isolation", "hyperv", "HostConfig.Isolation"),
    ("HostConfig", "OomScoreAdj", -500, "HostConfig.OomScoreAdj"),
    ("HostConfig", "VolumeDriver", "other", "HostConfig.VolumeDriver"),
    ("HostConfig", "Cgroup", "host", "HostConfig.Cgroup"),
    ("HostConfig", "StorageOpt", {"size": "100G"}, "HostConfig.StorageOpt"),
    ("HostConfig", "CpusetCpus", "0", "HostConfig.CpusetCpus"),
    ("HostConfig", "NanoCpus", 1000000000, "HostConfig.NanoCpus"),
    ("HostConfig", "MemoryReservation", 1 << 20, "HostConfig.MemoryReservation"),
    ("HostConfig", "NotAFieldTheLauncherKnows", ["value"], "HostConfig.NotAFieldTheLauncherKnows"),
    (None, "Mounts", lambda v: v + [{"Type": "volume", "Name": "0f" * 32, "Destination": "/data", "RW": True}], "Mounts"),
    (None, "Mounts", lambda v: [dict(m, RW=True) for m in v], "Mounts"),
    (None, "Mounts", lambda v: v[:-1], "Mounts"),
)


class TestPositiveFixtures(unittest.TestCase):
    def test_there_is_a_recorded_inspect_for_every_kind(self):
        kinds = {name[len("inspect-"):].split("-")[0] for name in support.fixture_names()}
        self.assertEqual(kinds, set(launcher.KINDS))
        self.assertIn("inspect-manifest-toolchain-root.json", support.fixture_names())
        self.assertIn("inspect-probe-toolchain.json", support.fixture_names())

    def test_every_recorded_inspect_passes_and_its_child_is_started(self):
        for name in support.fixture_names():
            with self.subTest(fixture=name):
                fixture = support.load_fixture(name)
                answer, fake = replay(fixture)
                self.assertTrue(answer["ok"], answer)
                self.assertEqual(len(fake.did("POST", "/start")), 1)
                self.assertEqual(fake.did("DELETE", "/containers/"), [])
                self.assertEqual(fake.state(fixture["request"]["name"]), "running")

    def test_the_recorded_inspects_are_of_a_real_daemon(self):
        """What makes them fixtures of a real container and not of the fake: the fields the daemon writes itself."""
        for name in support.fixture_names():
            with self.subTest(fixture=name):
                found = support.load_fixture(name)["inspect"]
                self.assertRegex(found["Id"], r"^[0-9a-f]{64}$")
                self.assertEqual(found["Config"]["Hostname"], found["Id"][:12])
                self.assertEqual((found["State"]["Status"], found["Driver"], found["HostConfig"]["Runtime"]), ("created", "overlay2", "runc"))
                self.assertGreater(len(found["HostConfig"]["MaskedPaths"]), 5)
                self.assertEqual(found["HostConfig"]["ConsoleSize"], [0, 0])
                self.assertIn("GraphDriver", found)
                self.assertEqual(found["Path"], "/bin/bash")

    def test_the_fake_synthesizes_what_the_real_daemon_answered(self):
        """The inspect the fake makes of a body has the keys of the recorded ones, so the other unit tests of the
        launcher see a Docker like the real one."""
        for name in support.fixture_names():
            with self.subTest(fixture=name):
                fixture = support.load_fixture(name)
                found = fixture["inspect"]
                body = support.body_for(fixture["request"], launcher.Config(
                    fixture["context"]["project"], 1000, 1000, fixture["context"]["volumes"], {},
                    {"app": fixture["context"]["image"]}, fixture["docker"]), t0=fixture["context"]["t0"])
                made = support.synth_inspect(body, found["Id"], found["Name"][1:], fixture["context"]["image"]["inspect"]["Config"]["Env"])
                self.assertEqual(sorted(made["Config"]), sorted(found["Config"]))
                self.assertEqual(sorted(made["HostConfig"]), sorted(found["HostConfig"]))


class TestAlteredFields(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = support.load_fixture("inspect-light-app.json")

    def test_every_altered_field_is_named_and_the_container_is_removed_without_a_start(self):
        for section, key, value, expected in ALTERED:
            with self.subTest(field=expected, value=str(value)[:40]):
                answer, fake = replay(self.fixture, setter(section, key, value))
                self.assertEqual((answer["ok"], answer.get("code")), (False, "isolation_unavailable"), answer)
                self.assertIn("differs from the request: ", answer["detail"])
                self.assertIn(expected, answer["detail"].split(": ", 1)[1].split(", "))
                self.assertEqual(fake.did("POST", "/start"), [])
                self.assertEqual(len(fake.did("DELETE", "/containers/")), 1)
                self.assertEqual(fake.containers, {})

    def test_the_list_covers_every_field_of_h2(self):
        named = {expected for _, _, _, expected in ALTERED}
        for field in ("Image", "Path", "Args", "Config.Image", "Config.Entrypoint", "Config.Cmd", "Config.User", "Config.Env",
                      "Config.Tty", "Config.OpenStdin", "Config.StdinOnce", "Config.AttachStdin", "Config.AttachStdout",
                      "Config.AttachStderr", "Config.WorkingDir", "Config.Labels", "Config.Volumes", "Config.Hostname",
                      "Config.Domainname", "HostConfig.NetworkMode", "HostConfig.ReadonlyRootfs", "HostConfig.CapAdd",
                      "HostConfig.CapDrop", "HostConfig.SecurityOpt", "HostConfig.Privileged", "HostConfig.Init", "HostConfig.Memory",
                      "HostConfig.MemorySwap", "HostConfig.PidsLimit", "HostConfig.ShmSize", "HostConfig.Ulimits", "HostConfig.Tmpfs",
                      "HostConfig.Binds", "HostConfig.Mounts", "HostConfig.Devices", "HostConfig.DeviceRequests",
                      "HostConfig.DeviceCgroupRules", "HostConfig.PortBindings", "HostConfig.PublishAllPorts", "HostConfig.RestartPolicy",
                      "HostConfig.AutoRemove", "HostConfig.IpcMode", "HostConfig.PidMode", "HostConfig.UTSMode", "HostConfig.UsernsMode",
                      "HostConfig.CgroupnsMode", "HostConfig.CgroupParent", "HostConfig.GroupAdd", "HostConfig.ExtraHosts", "HostConfig.Dns",
                      "HostConfig.Links", "HostConfig.VolumesFrom", "HostConfig.OomKillDisable", "HostConfig.Sysctls", "HostConfig.LogConfig",
                      "HostConfig.Runtime", "AppArmorProfile", "HostConfig.ConsoleSize", "HostConfig.Isolation", "HostConfig.MaskedPaths",
                      "HostConfig.ReadonlyPaths", "Mounts"):
            self.assertIn(field, named)

    def test_every_other_field_of_the_recorded_inspect_is_empty_and_any_value_in_it_is_refused(self):
        """The closing rule: each field of `Config` and `HostConfig` that the launcher does not name is empty in the
        real inspect, and a value in any of them is a refusal that names it."""
        found = self.fixture["inspect"]
        named = {expected for _, _, _, expected in ALTERED}
        others = [(section, key) for section in ("Config", "HostConfig") for key in found[section]
                  if "%s.%s" % (section, key) not in named]
        self.assertGreater(len(others), 15)
        for section, key in others:
            with self.subTest(field=section + "." + key):
                self.assertTrue(launcher._empty(found[section][key]), found[section][key])
                answer, fake = replay(self.fixture, setter(section, key, ["a value"]))
                self.assertEqual(answer.get("code"), "isolation_unavailable")
                self.assertIn("%s.%s" % (section, key), answer["detail"])
                self.assertEqual(fake.containers, {})

    def test_several_differences_are_all_named(self):
        def alter(found):
            found["HostConfig"]["Privileged"] = True
            found["HostConfig"]["NetworkMode"] = "host"
            found["Config"]["User"] = "0:0"
        answer, _ = replay(self.fixture, alter)
        for field in ("HostConfig.Privileged", "HostConfig.NetworkMode", "Config.User"):
            self.assertIn(field, answer["detail"])

    def test_a_container_that_vanished_or_a_docker_that_stops_answering_is_unavailable_and_removed(self):
        for alter in (lambda found: found.clear(), lambda found: found.update(Config="x"), lambda found: found.update(HostConfig=[]),
                      lambda found: found["HostConfig"].update(Mounts="x"), lambda found: found.update(Mounts=[1]),
                      lambda found: found["HostConfig"].update(Ulimits=[None])):
            answer, fake = replay(self.fixture, alter)
            self.assertEqual(answer.get("code"), "isolation_unavailable")
            self.assertIn("differs from the request", answer["detail"])
            self.assertEqual(fake.did("POST", "/start"), [])
            self.assertEqual(fake.containers, {})

    def test_a_start_that_fails_removes_the_container(self):
        fixture = self.fixture
        context = fixture["context"]
        fake = support.FakeDocker({context["image"]["id"]: context["image"]["inspect"]})
        route = fake._route

        def no_start(method, path, query, body):
            if method == "POST" and path.endswith("/start"):
                return 500, {"message": "failed to populate volume: subpath does not exist"}
            return route(method, path, query, body)
        fake._route = no_start
        fake.alter = lambda synthesized: copy.deepcopy(fixture["inspect"])
        config = launcher.Config(context["project"], context["uid"], context["gid"], context["volumes"], {},
                                 {"app": context["image"]}, fixture["docker"])
        clock = support.Clock(context["t0"])
        answer, _ = launcher.Launcher(fake, config, clock=clock, sleep=clock.sleep, identity={}).handle(fixture["request"])
        self.assertEqual(answer.get("code"), "isolation_unavailable")
        self.assertIn("subpath does not exist", answer["detail"])
        self.assertEqual(fake.containers, {})


class TestOtherKinds(unittest.TestCase):
    def test_what_matters_most_for_each_kind_is_named_too(self):
        cases = (("inspect-manifest-toolchain.json", "Config", "User", "0:0", "Config.User"),
                 ("inspect-manifest-toolchain.json", "HostConfig", "Mounts", lambda v: [dict(m, ReadOnly=False) for m in v], "HostConfig.Mounts"),
                 ("inspect-manifest-toolchain-root.json", "HostConfig", "CapAdd", ["DAC_OVERRIDE"], "HostConfig.CapAdd"),
                 ("inspect-manifest-toolchain-root.json", None, "Mounts", lambda v: [dict(m, RW=True) for m in v], "Mounts"),
                 ("inspect-tool-toolchain.json", "HostConfig", "NetworkMode", "bridge", "HostConfig.NetworkMode"),
                 ("inspect-tool-toolchain.json", "HostConfig", "Ulimits", lambda v: [u for u in v if u["Name"] != "data"], "HostConfig.Ulimits"),
                 ("inspect-marker-app.json", "HostConfig", "Mounts",
                  lambda v: [dict(m, ReadOnly=False) if m["Target"] == "/relay" else m for m in v], "HostConfig.Mounts"),
                 ("inspect-cpuserver-app.json", "HostConfig", "Mounts",
                  lambda v: [dict(m, ReadOnly=False) if m["Target"] == "/models" else m for m in v], "HostConfig.Mounts"),
                 ("inspect-probe-toolchain.json", "Config", "Cmd", ["sh", "-c", "id"], "Config.Cmd"))
        for name, section, key, value, expected in cases:
            with self.subTest(fixture=name, field=expected):
                answer, fake = replay(support.load_fixture(name), setter(section, key, value))
                self.assertEqual(answer.get("code"), "isolation_unavailable", answer)
                self.assertIn(expected, answer["detail"])
                self.assertEqual(fake.containers, {})


if __name__ == "__main__":
    unittest.main()
