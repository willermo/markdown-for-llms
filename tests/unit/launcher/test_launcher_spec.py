"""The launcher builds the body of every child itself and refuses everything else (MFL051, plan r2.1 App. H.1, H.2).

Golden bodies: for every kind and image the body of `POST /containers/create` is the recorded one
(`tests/fixtures/launcher/body-*.json`, where `<supervisor>` stands for the script of the day). Refusals: a request
is closed, so each thing it could try to widen is asserted to be refused before Docker hears of it. The Docker of
these tests is `launcher_support.FakeDocker`.
"""
import copy
import json
import os
from pathlib import Path
import socket
import threading
import unittest
from unittest import mock

import launcher_support as support
from markdown_for_llms import launcher
from markdown_for_llms.engine import containers, pages, supervisor, toolchain, tools


class TestGoldenBodies(unittest.TestCase):
    def test_the_body_of_every_kind_and_image_is_the_recorded_one(self):
        cases = support.golden_cases()
        self.assertEqual(sorted(label for label, _ in cases),
                         sorted(p.name[len("body-"):-len(".json")] for p in support.FIXTURES.glob("body-*.json")))
        for label, request in cases:
            with self.subTest(case=label):
                body = support.body_for(request)
                self.assertEqual(support.without_supervisor(body), json.loads((support.FIXTURES / ("body-%s.json" % label)).read_text()))
                self.assertEqual(body["Entrypoint"][2], supervisor.SUPERVISOR)

    def test_what_every_body_has_whatever_the_kind(self):
        for label, request in support.golden_cases():
            with self.subTest(case=label):
                body = support.body_for(request)
                host = body["HostConfig"]
                self.assertEqual((host["NetworkMode"], host["ReadonlyRootfs"], host["CapDrop"], host["SecurityOpt"], host["Privileged"],
                                  host["Init"], host["IpcMode"], host["CgroupnsMode"], host["AutoRemove"], host["RestartPolicy"]["Name"]),
                                 ("none", True, ["ALL"], ["no-new-privileges"], False, False, "private", "private", False, "no"))
                self.assertEqual(host["Memory"], host["MemorySwap"])
                self.assertEqual((body["Tty"], body["OpenStdin"], body["StdinOnce"], body["AttachStdin"], body["AttachStdout"],
                                  body["AttachStderr"], body["NetworkDisabled"]), (False,) * 6 + (True,))
                self.assertEqual(body["Entrypoint"][:2] + body["Entrypoint"][3:4], ["/bin/bash", "-c", "mdf-supervise"])
                self.assertEqual(body["Cmd"], request["argv"])                      # the command is an argument of the supervisor
                self.assertEqual(sorted(host), sorted(["NetworkMode", "ReadonlyRootfs", "CapDrop", "SecurityOpt", "Privileged", "Init",
                                                       "Memory", "MemorySwap", "PidsLimit", "ShmSize", "Ulimits", "Tmpfs", "Mounts",
                                                       "IpcMode", "CgroupnsMode", "RestartPolicy", "AutoRemove", "LogConfig"]))
                self.assertTrue(all(m["Type"] == "volume" and m["VolumeOptions"]["NoCopy"] and m["VolumeOptions"]["Subpath"]
                                    for m in host["Mounts"]))
                self.assertEqual(host["LogConfig"]["Type"], "local" if request["kind"] == "manifest" else "none")
                self.assertEqual(set(body["Labels"]), {"mdf.project", "mdf.kind", "mdf.attempt", "mdf.max", "mdf.lease", "mdf.t0"})
                self.assertEqual(body["User"], "0:0" if request["argv"] == containers.MANIFEST_REGENERATE else "1000:1000")
                self.assertEqual(any(u["Name"] == "data" for u in host["Ulimits"]), "data_mib" in request["limits"])

    def test_the_names_of_the_marker_child_are_those_of_the_engine(self):
        names = set(pages.child_env("/models", 4))
        self.assertTrue(names <= set(launcher.MARKER_VARIABLES), names - set(launcher.MARKER_VARIABLES))
        self.assertEqual(containers.MANIFEST_PATH, toolchain.MANIFEST_PATH)
        self.assertEqual(containers.CONTENT_SCRIPT, toolchain.CONTENT_SCRIPT)
        from markdown_for_llms.engine import cpuserver
        self.assertEqual(containers.CPU_SERVER_TREE, cpuserver.TREE)
        self.assertEqual(containers.LOG_LIMIT, tools.LOG_LIMIT)


class Refusals(unittest.TestCase):
    def setUp(self):
        self.launcher, self.fake, self.clock = support.make()

    def refused(self, request, code="invalid_setting", contains=None, sut=None):
        answer, payload = (sut or self.launcher).handle(request)
        self.assertEqual((answer["ok"], answer.get("code"), payload), (False, code, None), answer)
        if contains:
            self.assertIn(contains, answer["detail"])
        self.assertEqual(self.fake.did("POST", "/containers/create"), [], "Docker heard of a refused request")
        return answer["detail"]

    def accepted(self, request, sut=None):
        answer, _ = (sut or self.launcher).handle(request)
        self.assertTrue(answer["ok"], answer)
        return answer


class TestRequestIsClosed(Refusals):
    def test_a_good_request_runs(self):
        answer = self.accepted(support.request())
        self.assertEqual(answer["name"], support.ATTEMPT + "-light-1")
        self.assertEqual(self.fake.state(answer["name"]), "running")
        self.assertEqual([c[1] for c in self.fake.calls if c[0] == "POST"],
                         ["/containers/create", "/containers/%s/start" % answer["id"]])
        self.assertEqual(self.fake.did("POST", "/containers/create")[0][2], {"name": answer["name"]})

    def test_every_field_that_names_a_mount_or_a_subpath_is_refused(self):
        name = support.ATTEMPT + "-light-1"
        for field, value in (("mounts", [{"volume": "work", "subpath": "call/" + name, "target": "/scratch", "readonly": False}]),
                             ("subpath", "call"), ("volumes", {"work": "/"}), ("binds", ["/:/host"]), ("relay", "sock/other"),
                             ("scratch", "call/%s" % name), ("workdir", "/"), ("user", "0:0"), ("platform", "linux/arm64"),
                             ("entrypoint", ["/bin/sh"]), ("privileged", True), ("network", "host"), ("labels", {}),
                             ("HostConfig", {"Privileged": True}), ("devices", []), ("cap_add", ["SYS_ADMIN"]), ("stdin", "x")):
            with self.subTest(field=field):
                self.refused(dict(support.request(), **{field: value}), contains="unknown field: " + field)

    def test_a_missing_field_is_refused(self):
        for field in ("kind", "name", "image", "argv", "env", "limits"):
            request = support.request()
            del request[field]
            self.refused(request, contains="missing field: " + field)

    def test_an_image_that_is_not_a_role_of_the_kind_is_refused(self):
        for kind, image in (("light", "toolchain"), ("light", "candidate"), ("tool", "app"), ("marker", "toolchain"),
                            ("cpuserver", "toolchain"), ("manifest", "app"), ("light", "debian:trixie"), ("light", support.APP_ID),
                            ("light", "mdf-app:engine"), ("light", ""), ("light", None), ("light", ["app"])):
            with self.subTest(kind=kind, image=image):
                self.refused(dict(support.request(kind, launcher.KINDS[kind][0][0]), image=image), contains="image role")

    def test_a_command_the_kind_does_not_admit_is_refused(self):
        python = containers.python_command
        cases = (
            ("tool", "toolchain", ["sh", "-c", "id"]), ("tool", "toolchain", ["/usr/bin/qpdf"]), ("tool", "toolchain", ["bash"]),
            ("tool", "toolchain", ["env", "pandoc"]), ("tool", "toolchain", ["pandoc", "-f", "docx"]),
            ("tool", "toolchain", ["pandoc", "--sandbox", "+RTS", "-M512m", "-RTS"]),
            ("tool", "toolchain", ["pandoc", "+RTS", "-M512m", "-RTS", "--sandbox", "--sandbox=false"]),
            ("tool", "toolchain", ["pandoc", "+RTS", "-M512m", "-RTS", "--sandbox", "+RTS", "-M99999m", "-RTS"]),
            ("tool", "toolchain", ["pandoc", "+RTS", "-Mxm", "-RTS", "--sandbox"]),
            ("light", "app", python("markdown_for_llms.engine.marker_child")), ("light", "app", python("os")),
            ("light", "app", python("markdown_for_llms.worker")), ("light", "app", ["/bin/sh"]),
            ("light", "app", [containers.PYTHON, "-c", "import os"]),
            ("light", "app", [containers.PYTHON, "-I", "-B", "-c", "import os", "limits.json", containers.PROBE_MODULE]),
            ("light", "app", ["/usr/bin/python3"] + python(containers.PROBE_MODULE)[1:]),
            ("light", "app", python(containers.PROBE_MODULE)[:6]),
            ("marker", "app", python("markdown_for_llms.engine.pkgaudit")),
            ("cpuserver", "app", ["/bin/sh"]), ("cpuserver", "app", ["/opt/llama/../../bin/sh"]), ("cpuserver", "app", ["/opt/llama"]),
            ("cpuserver", "app", ["/opt/llamax/server"]), ("cpuserver", "app", ["/opt/llama//server"]),
            ("probe", "app", python(containers.PROBE_MODULE, ["allocate", "9999"])), ("probe", "app", containers.PROBE_SHELL),
            ("probe", "toolchain", ["sh", "-c", "id"]), ("probe", "toolchain", containers.PROBE_PYTHON),
            ("manifest", "toolchain", ["cat", "/etc/shadow"]), ("manifest", "toolchain", ["sh"]),
            ("manifest", "toolchain", [containers.CONTENT_SCRIPT, "x"]), ("manifest", "toolchain", ["cat"]),
        )
        for kind, image, argv in cases:
            with self.subTest(kind=kind, argv=argv[:4]):
                self.refused(support.request(kind, image, argv=argv))
        for argv in ([], "qpdf", ["qpdf", 1], ["qpdf", "a\0b"], None, ["qpdf"] * (launcher.ARGV_LIMIT + 1)):
            with self.subTest(argv=str(argv)[:30]):
                self.refused(dict(support.request("tool", "toolchain"), argv=argv), contains="argv")

    def test_every_tool_of_the_toolchain_is_admitted_and_nothing_else(self):
        for name in tools.TOOLS:
            if name != "pandoc":
                self.assertEqual(launcher.check_command("tool", "toolchain", [name, "--version"]), "user")
        self.assertEqual(launcher.check_command("tool", "toolchain", ["pandoc", "+RTS", "-M8192m", "-RTS", "--sandbox", "-t", "json"]), "user")

    def test_a_name_out_of_the_schema_is_refused(self):
        attempt = support.ATTEMPT
        for name in ("light-1", attempt + "-light", attempt + "-light-x", attempt + "-shell-1", "mdf-eng--light-1",
                     attempt + "x-light-1", attempt + "-light-1234567", "mdf-bench-1-light-1", attempt + "-light-1/../x",
                     "../" + attempt + "-light-1", attempt + "-light-1\n", " " + attempt + "-light-1", "mdf-full-check",
                     "mdf-eng-" + "a" * 16 + "-tool-" + "1" * 6 + "x" * 30, 7, None):
            with self.subTest(name=name):
                self.refused(support.request(name=name), contains="name")
        self.refused(support.request(name=attempt + "-tool-1"), contains="another kind")
        for good in ("mdf-eng-run.a_b-1-light-2", "mdf-bench-4242-0a1b2c3d-light-7", "mdf-eng-a-light-1"):
            self.accepted(support.request(name=good))

    def test_a_variable_the_kind_does_not_admit_is_refused(self):
        for kind, image, name in (("light", "app", "LD_PRELOAD"), ("light", "app", "PYTHONPATH"), ("light", "app", "TORCH_DEVICE"),
                                  ("tool", "toolchain", "PATH"), ("tool", "toolchain", "LD_LIBRARY_PATH"),
                                  ("manifest", "toolchain", "HOME"), ("cpuserver", "app", "LD_PRELOAD"),
                                  ("marker", "app", "DATALAB_API_KEY"), ("probe", "toolchain", "PATH"), ("probe", "app", "TORCH_DEVICE")):
            with self.subTest(kind=kind, name=name):
                request = support.request(kind, image)
                request["env"][name] = "x"
                self.refused(request, contains="variable")
        for value in ("a\nB=c", "a\0", 5, None, "x" * (launcher.VALUE_LIMIT + 1)):
            request = support.request()
            request["env"]["HOME"] = value
            self.refused(request, contains="value of HOME")
        self.refused(support.request(env=["HOME=/"]), contains="env")
        request = support.request("marker")
        request["env"].update({name: "1" for name in launcher.MARKER_VARIABLES})
        self.accepted(request)

    def test_limits_over_the_maxima_or_malformed_are_refused(self):
        for key in launcher.LIMIT_KEYS:
            for value in (launcher.MAXIMA[key] + 1, 0, -1, True, "64", 1.5, None):
                with self.subTest(key=key, value=value):
                    self.refused(support.request(limits=dict(support.LIMITS, **{key: value})), contains="limit " + key)
            missing = dict(support.LIMITS)
            del missing[key]
            self.refused(support.request(limits=missing), contains="limits")
        self.refused(support.request(limits=dict(support.LIMITS, cpus=2)), contains="limits")
        self.refused(support.request(limits=dict(support.LIMITS, data_mib=512)), contains="limits")       # only a tool has it
        self.refused(support.request(limits=[]), contains="limits")
        launcher.check_request(support.request("tool", "toolchain", limits=dict(support.LIMITS, data_mib=512)))

    def test_the_maxima_of_the_launcher_can_only_be_lower(self):
        sut, fake, _ = support.make(maxima=dict(launcher.MAXIMA, memory_mib=128))
        answer, _ = sut.handle(support.request())
        self.assertEqual((answer["ok"], answer["code"]), (False, "invalid_setting"))
        self.assertIn("memory_mib outside 1..128", answer["detail"])
        self.assertEqual(sut.op_hello({})["max"]["memory_mib"], 128)

    def test_no_more_children_than_the_maximum(self):
        sut, fake, _ = support.make(max_children=2)
        for n in (1, 2):
            self.assertTrue(sut.handle(support.request(n=n))[0]["ok"])
        answer, _ = sut.handle(support.request(n=3))
        self.assertEqual((answer["ok"], answer["code"]), (False, "isolation_unavailable"))
        self.assertIn("at most 2", answer["detail"])
        fake.containers[fake._find(support.ATTEMPT + "-light-1")]["inspect"]["State"]["Status"] = "exited"
        self.assertTrue(sut.handle(support.request(n=3))[0]["ok"])

    def test_a_name_in_use_is_refused(self):
        self.accepted(support.request())
        answer, _ = self.launcher.handle(support.request())
        self.assertEqual((answer["ok"], answer["code"]), (False, "invalid_setting"))
        self.assertIn("exists", answer["detail"])

    def test_a_stopping_launcher_refuses_run(self):
        self.launcher.stopping.set()
        self.refused(support.request(), code="isolation_unavailable", contains="stopping")

    def test_a_volume_role_the_launcher_was_not_given_is_unavailable(self):
        sut, fake, _ = support.make(volumes={k: v for k, v in support.VOLUMES.items() if k in ("work", "run")})
        for kind in ("marker", "cpuserver"):
            answer, _ = sut.handle(support.request(kind))
            self.assertEqual((answer["ok"], answer["code"]), (False, "isolation_unavailable"), kind)
        self.assertTrue(sut.handle(support.request("light"))[0]["ok"])
        self.assertEqual(len(fake.did("POST", "/containers/create")), 1)

    def test_the_cpu_server_is_refused_by_a_launcher_whose_image_is_not_amd64(self):
        sut, fake, _ = support.make(architecture="arm64")
        answer, _ = sut.handle(support.request("cpuserver"))
        self.assertEqual((answer["ok"], answer["code"]), (False, "isolation_unavailable"))
        self.assertIn("amd64 only", answer["detail"])
        self.assertEqual(fake.did("POST", "/containers/create"), [])
        self.assertTrue(sut.handle(support.request("light"))[0]["ok"])


class TestTransport(Refusals):
    def test_a_peer_of_another_user_is_refused_before_its_request_is_read(self):
        data = json.dumps(support.request()).encode()
        for uid in (0, support.UID + 1, 65534):
            answer, _ = self.launcher.answer(data, uid)
            self.assertEqual((answer["ok"], answer["code"], answer["detail"]), (False, "invalid_setting", "peer of another user"))
        self.assertEqual(self.fake.calls, [])
        self.assertTrue(self.launcher.answer(data, support.UID)[0]["ok"])

    def test_a_request_over_64_kib_is_refused(self):
        request = support.request("tool", "toolchain", argv=["qpdf", "x" * containers.REQUEST_LIMIT])
        answer, _ = self.launcher.answer(json.dumps(request).encode(), support.UID)
        self.assertEqual((answer["ok"], answer["code"]), (False, "invalid_setting"))
        self.assertIn("65536", answer["detail"])
        self.assertEqual(self.fake.calls, [])

    NESTED = b"[" * 30000 + b"]" * 30000            # valid JSON, within the 64 KiB, too deep for the decoder (CAR-A1-N2)

    def test_broken_json_and_other_shapes_are_refused(self):
        self.assertLess(len(self.NESTED), containers.REQUEST_LIMIT)
        for data in (b"", b"{", b"\xff\xfe", b"[]", b"null", b"7", b'{"op": 1}', b'{"op": "exec"}', b'{"op": "stats"}', b'{"op": "touch"}',
                     b'{"kind": "light"}', b'{"op": "hello", "extra": 1}', b'{"op": "inspect", "name": "x", "id": "y"}',
                     self.NESTED, b'{"op": ' + self.NESTED + b"}", b'{"op": "hello", "x": ' + b'{"a": ' * 9000 + b"1" + b"}" * 9000 + b"}"):
            with self.subTest(data=data[:40]):
                answer, payload = self.launcher.answer(data, support.UID)
                self.assertEqual((answer["ok"], answer["code"], payload), (False, "invalid_setting", None))
        self.assertEqual(self.launcher.answer(self.NESTED, support.UID)[0]["detail"], "request is not JSON")
        self.assertEqual(self.fake.calls, [])

    def over_a_socket(self, data):
        """The bytes of the answer to `data` sent over a real socket, served as the launcher serves a connection."""
        sut, fake, _ = support.make(uid=os.getuid())            # the peer of a socket pair is this very process
        ours, theirs = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
        self.addCleanup(ours.close)
        served = threading.Thread(target=launcher.serve_connection, args=(sut, theirs), daemon=True)
        served.start()
        ours.settimeout(30)
        ours.sendall(data)
        ours.shutdown(socket.SHUT_WR)
        answer = b""
        while True:
            chunk = ours.recv(65536)
            if not chunk:
                break
            answer += chunk
        served.join(30)
        self.assertFalse(served.is_alive())
        return answer, fake

    def test_json_nested_too_deep_is_a_closed_refusal_over_a_socket_too(self):
        """Before, the thread of the connection died of `RecursionError` and the client read an answer cut short,
        which it takes for a launcher that is unreachable."""
        answer, fake = self.over_a_socket(self.NESTED)
        self.assertEqual(json.loads(answer), {"ok": False, "code": "invalid_setting", "detail": "request is not JSON"})
        self.assertTrue(answer.endswith(b"\n"))
        self.assertEqual(fake.calls, [])
        answer, _ = self.over_a_socket(json.dumps({"op": "hello"}).encode())     # and the same socket path answers a request
        self.assertEqual((json.loads(answer)["ok"], json.loads(answer)["project"]), (True, support.PROJECT))

    def test_a_defect_of_the_launcher_is_an_answer_never_a_crash(self):
        with mock.patch.object(launcher, "check_request", side_effect=RuntimeError("boom")):
            answer, _ = self.launcher.handle(support.request())
        self.assertEqual(answer, {"ok": False, "code": "isolation_unavailable", "detail": "launcher error: RuntimeError"})

    def test_docker_unreachable_is_isolation_unavailable(self):
        self.fake.fail = "ECONNREFUSED"
        for request in (support.request(), {"op": "inspect", "name": support.ATTEMPT + "-light-1"},
                        {"op": "list", "prefix": support.ATTEMPT}, {"op": "remove", "name": support.ATTEMPT + "-light-1"}):
            answer, _ = self.launcher.handle(request)
            self.assertEqual((answer["ok"], answer["code"]), (False, "isolation_unavailable"), request["op"])
            self.assertIn("ECONNREFUSED", answer["detail"])


class TestOtherProjects(Refusals):
    """inspect, kill, remove, list and logs only see the children of this project."""

    def test_a_container_of_another_project_is_not_touched(self):
        name = support.ATTEMPT + "-light-1"
        for labels in ({"mdf.project": "other", "mdf.kind": "light"}, {"mdf.kind": "light"}, {"mdf.project": support.PROJECT},
                       {"com.docker.compose.project": support.PROJECT}, {}):
            fake = support.FakeDocker()
            sut, _, _ = support.make(fake)
            fake.add(name, labels)
            for op in ("inspect", "kill", "remove", "logs"):
                answer, _ = sut.handle({"op": op, "name": name})
                self.assertEqual((answer["ok"], answer["code"], answer["detail"]),
                                 (False, "invalid_setting", "not a child of this project"), (op, labels))
            self.assertEqual(sut.handle({"op": "list", "prefix": support.ATTEMPT})[0]["children"], [])
            self.assertEqual(fake.state(name), "running")
            self.assertEqual(fake.did("POST", "/kill") + fake.did("DELETE", "/containers/"), [])

    def test_a_child_of_this_project_is_inspected_killed_removed_and_listed(self):
        self.accepted(support.request())
        name = support.ATTEMPT + "-light-1"
        self.assertEqual(self.accepted({"op": "inspect", "name": name})["state"], "running")
        self.assertEqual(self.accepted({"op": "list", "prefix": support.ATTEMPT})["children"], [{"name": name, "state": "running"}])
        self.assertEqual(self.accepted({"op": "list", "prefix": "mdf-eng-other"})["children"], [])
        self.assertEqual(self.accepted({"op": "kill", "name": name})["state"], "killed")
        inspected = self.accepted({"op": "inspect", "name": name})
        self.assertEqual((inspected["state"], inspected["exit"], inspected["oom_killed"]), ("exited", 137, False))
        self.assertEqual(self.accepted({"op": "remove", "name": name}), {"ok": True, "absent": True})
        self.assertEqual(self.accepted({"op": "inspect", "name": name}), {"ok": True, "state": "absent"})
        self.assertEqual(self.accepted({"op": "kill", "name": name}), {"ok": True, "state": "absent"})
        self.assertEqual(self.accepted({"op": "remove", "name": name}), {"ok": True, "absent": True})

    def test_remove_answers_absent_only_when_docker_confirms(self):
        self.accepted(support.request())
        name = support.ATTEMPT + "-light-1"
        route = self.fake._route

        def keeps(method, path, query, body):                   # Docker answers 204 and the container stays
            if method == "DELETE":
                return 204, None
            return route(method, path, query, body)
        with mock.patch.object(self.fake, "_route", side_effect=keeps):
            answer, _ = self.launcher.handle({"op": "remove", "name": name})
        self.assertEqual((answer["ok"], answer["code"]), (False, "isolation_unavailable"))
        self.assertIn("not confirmed", answer["detail"])
        self.assertGreaterEqual(self.clock.now - 1000.0, launcher.REMOVE_SECONDS)

    def test_logs_only_of_an_exited_manifest_child(self):
        self.accepted(support.request())
        answer, _ = self.launcher.handle({"op": "logs", "name": support.ATTEMPT + "-light-1"})
        self.assertEqual(answer["detail"], "logs are of a manifest child only")
        self.accepted(support.request("manifest", "toolchain"))
        name = support.ATTEMPT + "-manifest-1"
        self.assertEqual(self.launcher.handle({"op": "logs", "name": name})[0]["detail"], "the child has not exited")
        container = self.fake.containers[self.fake._find(name)]
        container["inspect"]["State"]["Status"] = "exited"
        container["logs"] = support.frames(b"f abc  /usr/x\n", b"noise on stderr")
        answer, payload = self.launcher.handle({"op": "logs", "name": name})
        self.assertEqual((answer, payload), ({"ok": True, "bytes": 14}, b"f abc  /usr/x\n"))

    def test_a_prefix_out_of_the_schema_is_refused(self):
        for prefix in ("", "mdf-eng-", "mdf-", "other", support.ATTEMPT + "-light-1", "mdf-eng-" + "a" * 17, None, 5):
            self.assertEqual(self.launcher.handle({"op": "list", "prefix": prefix})[0]["code"], "invalid_setting")


class TestToolchainRole(Refusals):
    """`toolchain` is a role: the local image with the tag of `toolchain.json`, verified by the launcher before a
    tool or a probe may use it."""

    def test_no_image_is_toolchain_missing(self):
        sut, fake, _ = support.make(support.FakeDocker({support.APP_ID: support.images()[support.APP_ID]}))
        for request in (support.request("tool", "toolchain"), support.request("manifest", "toolchain"), {"op": "toolchain"}):
            answer, _ = sut.handle(request)
            self.assertEqual((answer["ok"], answer["code"]), (False, "toolchain_missing"))
        self.assertEqual(fake.did("POST", "/containers/create"), [])

    def test_another_config_is_toolchain_mismatch(self):
        table = support.images()
        table[toolchain.tag()]["Config"]["Entrypoint"] = ["/bin/sh"]
        sut, fake, _ = support.make(support.FakeDocker(table))
        answer, _ = sut.handle(support.request("tool", "toolchain"))
        self.assertEqual((answer["code"], answer["detail"]), ("toolchain_mismatch", "image config"))
        self.assertEqual(fake.did("POST", "/containers/create"), [])

    def manifest(self):
        return b"f 00  /usr/bin/x\n"

    def identity_is(self, data):
        """`toolchain.json` as if its content manifest were `data` (the tag of the image follows the identity)."""
        import contextlib
        import hashlib

        @contextlib.contextmanager
        def patched():
            with mock.patch.object(toolchain, "identity", return_value=hashlib.sha256(data).hexdigest()):
                self.fake.images[toolchain.tag()] = support.image_inspect(support.toolchain_id(), support.TOOLCHAIN_ENV)
                yield
        return patched()

    def test_the_image_is_verified_once_by_a_manifest_child_of_the_launcher_itself(self):
        support.manifest_exits(self.fake, self.manifest())
        with self.identity_is(self.manifest()):
            first = self.accepted(support.request("tool", "toolchain", n=1))
            created = self.fake.did("POST", "/containers/create")
            self.assertEqual(len(created), 2)                       # the manifest child of the launcher, then the tool
            self.assertRegex(created[0][2]["name"], r"^mdf-launcher-[0-9a-f]{8}-[0-9a-f]{8}-manifest-1$")
            self.assertEqual(created[0][2]["name"], "%s%s-manifest-1" % (self.launcher.internal_prefix(), self.launcher.token))
            self.assertEqual(created[0][2]["platform"], "linux/amd64")
            self.assertEqual(created[1][2], {"name": first["name"], "platform": "linux/amd64"})
            self.assertIsNone(self.fake._find(created[0][2]["name"]))    # removed
            self.accepted(support.request("tool", "toolchain", n=2))
            self.accepted(support.request("probe", "toolchain"))
            self.assertEqual(len(self.fake.did("POST", "/containers/create")), 4)
            answer = self.accepted({"op": "toolchain"})
            self.assertEqual((answer["verified"], answer["image"]), (True, support.toolchain_id()))

    def test_the_manifest_child_of_the_launcher_has_no_mount_and_no_root(self):
        bodies = []
        route = self.fake._route

        def watch(method, path, query, body):
            if (method, path) == ("POST", "/containers/create"):
                bodies.append(body)
            return route(method, path, query, body)
        support.manifest_exits(self.fake, self.manifest())
        with mock.patch.object(self.fake, "_route", side_effect=watch), self.identity_is(self.manifest()):
            self.accepted({"op": "toolchain"})
        body = bodies[0]
        self.assertEqual((body["HostConfig"]["Mounts"], body["User"], body["Cmd"]), ([], "1000:1000", containers.MANIFEST_READ))
        self.assertEqual((body["HostConfig"]["NetworkMode"], body["HostConfig"]["ReadonlyRootfs"], body["HostConfig"]["CapDrop"]),
                         ("none", True, ["ALL"]))
        self.assertEqual(body["Labels"]["mdf.project"], support.PROJECT)

    def test_another_manifest_is_toolchain_mismatch_and_nothing_runs(self):
        support.manifest_exits(self.fake, b"f ff  /usr/bin/other\n")
        answer, _ = self.launcher.handle(support.request("tool", "toolchain"))
        self.assertEqual((answer["code"], answer["detail"]), ("toolchain_mismatch", "content manifest"))
        self.assertEqual(self.fake.containers, {})
        support.manifest_exits(self.fake, b"", exit_code=1)
        answer, _ = self.launcher.handle(support.request("probe", "toolchain"))
        self.assertEqual((answer["code"], answer["detail"]), ("toolchain_mismatch", "content manifest: exit 1"))
        self.assertEqual(self.fake.containers, {})

    def dies_in_the_middle_of_a_verification(self, sut):
        """What a launcher killed while its manifest child runs leaves on the daemon; returns the name of the child."""
        class Killed(BaseException):
            pass

        def dies(container):
            raise Killed()
        self.fake.on_start = dies
        with mock.patch.object(sut, "_force_remove"), self.assertRaises(Killed):       # a dead process removes nothing
            sut.handle({"op": "toolchain"})
        left = [c["name"] for c in self.fake.containers.values()]
        self.assertEqual(len(left), 1)
        return left[0]

    def test_after_a_death_in_the_middle_of_a_verification_the_first_toolchain_of_the_next_launcher_succeeds(self):
        """CAR-A1-N3. The names of the launcher's own manifest children counted from 1 in every process: the child
        left by a launcher that died took the first name of the next one, whose first verification failed (409)."""
        with self.identity_is(self.manifest()):
            stale = self.dies_in_the_middle_of_a_verification(self.launcher)
            self.assertEqual(self.fake.state(stale), "running")
            restarted, _, _ = support.make(self.fake)
            support.manifest_exits(self.fake, self.manifest())
            answer, _ = restarted.handle({"op": "toolchain"})
            self.assertEqual((answer["ok"], answer.get("verified")), (True, True), answer)
            names = [c[2]["name"] for c in self.fake.did("POST", "/containers/create")]
            self.assertEqual(len(set(names)), 2)
            self.assertEqual([n.rsplit("-manifest-", 1)[1] for n in names], ["1", "1"])    # the same number, another process
            self.assertNotEqual(self.launcher.token, restarted.token)
            self.assertEqual(list(c["name"] for c in self.fake.containers.values()), [stale])

    def test_at_its_start_the_launcher_removes_its_own_manifest_children_that_exited(self):
        with self.identity_is(self.manifest()):
            stale = self.dies_in_the_middle_of_a_verification(self.launcher)
        running = self.fake.add(self.launcher.internal_prefix() + "0a0a0a0a-manifest-1",
                                {"mdf.project": support.PROJECT, "mdf.kind": "manifest", "mdf.attempt": "-"})
        of_a_client = self.fake.add(support.ATTEMPT + "-manifest-1", {"mdf.project": support.PROJECT, "mdf.kind": "manifest",
                                                                     "mdf.attempt": support.ATTEMPT}, state="exited")
        other_project = self.fake.add("mdf-launcher-ffffffff-0b0b0b0b-manifest-1",
                                      {"mdf.project": "other", "mdf.kind": "manifest", "mdf.attempt": "-"}, state="exited")
        self.fake.containers[self.fake._find(stale)]["inspect"]["State"].update(Status="exited", Running=False)
        restarted, _, _ = support.make(self.fake)
        self.assertEqual(restarted.forget_internal(), [stale])
        self.assertEqual(sorted(self.fake.containers), sorted([running, of_a_client, other_project]))
        self.assertEqual(restarted.forget_internal(), [])
        self.fake.fail = "ECONNREFUSED"                         # without Docker it is a refusal, which `serve` lets pass
        with self.assertRaises(launcher.Refusal):
            restarted.forget_internal()

    def test_the_names_of_the_manifest_children_are_of_the_process(self):
        tokens = {support.make()[0].token for _ in range(50)}
        self.assertEqual(len(tokens), 50)
        for token in tokens:
            self.assertRegex(token, r"^[0-9a-f]{8}$")
        source = Path(launcher.__file__).read_text(encoding="utf-8")
        self.assertIn("launcher.forget_internal()", source[source.index("def serve(args"):])     # called when the service starts

    def test_a_manifest_child_of_a_client_needs_no_verified_image(self):
        answer = self.accepted(support.request("manifest", "toolchain", argv=containers.MANIFEST_REGENERATE))
        body = self.fake.containers[answer["id"]]["body"]
        self.assertEqual((body["User"], [m["ReadOnly"] for m in body["HostConfig"]["Mounts"]]), ("0:0", [True]))

    def test_the_candidate_is_the_local_tag_and_is_never_marked_verified(self):
        table = support.images()
        candidate = "sha256:" + "c" * 64
        table[launcher.CANDIDATE_TAG] = support.image_inspect(candidate, support.TOOLCHAIN_ENV)
        sut, fake, _ = support.make(support.FakeDocker(table))
        answer, _ = sut.handle({"op": "toolchain", "candidate": True})
        self.assertEqual((answer["ok"], answer["image"], answer["verified"]), (True, candidate, False))
        run, _ = sut.handle(support.request("tool", "candidate"))
        self.assertTrue(run["ok"], run)
        self.assertEqual(fake.containers[run["id"]]["body"]["Image"], candidate)
        self.assertEqual(sut.handle({"op": "toolchain", "candidate": 1})[0]["code"], "invalid_setting")
        self.assertEqual(sut.handle({"op": "toolchain", "name": "x"})[0]["code"], "invalid_setting")


class TestHello(unittest.TestCase):
    def test_hello_says_what_the_launcher_is(self):
        sut, _, _ = support.make()
        answer, _ = sut.handle({"op": "hello"})
        self.assertEqual(answer["images"], {"app": {"id": support.APP_ID, "architecture": "amd64"}})
        self.assertEqual((answer["project"], answer["uid"], answer["docker"]["runtime"], answer["max"]["children"]),
                         (support.PROJECT, support.UID, "runc", launcher.MAX_CHILDREN))
        self.assertEqual(set(answer) >= {"payload_sha256", "inputs_sha256", "environment"}, True)
        self.assertNotIn("inspect", json.dumps(answer))


class TestBodyInvariants(unittest.TestCase):
    """`check_body`: what holds whatever a table says (a wrong table is a refusal, not a container)."""

    def parts(self, request):
        kind, name, prefix, image, argv, variables, limits = launcher.check_request(request)
        user = launcher.check_command(kind, image, argv)
        mounts = launcher.mounts_of(kind, name, prefix, image)
        cfg = support.config()
        cfg.images["toolchain"] = {"id": support.toolchain_id(), "architecture": "amd64", "inspect": {}}
        body = launcher.body_of(cfg, kind, name, prefix, image, argv, variables, limits, user, mounts, 1.0)
        return body, kind, name, prefix, user, mounts, cfg

    def test_uid_0_outside_the_regeneration_of_the_manifest(self):
        for request in (support.request("light"), support.request("tool", "toolchain"), support.request("manifest", "toolchain")):
            body, kind, name, prefix, user, mounts, cfg = self.parts(request)
            for value in ("0:0", "0", "root", "0:1000", ""):
                with self.subTest(kind=kind, user=value), self.assertRaises(launcher.Refusal) as refused:
                    launcher.check_body(dict(body, User=value), kind, name, prefix, user, mounts, cfg)
                self.assertIn("UID 0", refused.exception.detail)
        body, kind, name, prefix, user, mounts, cfg = self.parts(support.request("manifest", "toolchain", argv=containers.MANIFEST_REGENERATE))
        self.assertEqual((user, body["User"]), ("root", "0:0"))
        launcher.check_body(body, kind, name, prefix, user, mounts, cfg)
        with self.assertRaises(launcher.Refusal):               # a table that makes a light child root
            with mock.patch.object(launcher, "check_command", return_value="root"):
                self.parts(support.request("light"))
                body, kind, name, prefix, user, mounts, cfg = self.parts(support.request("light"))
                launcher.check_body(body, kind, name, prefix, user, mounts, cfg)

    def test_a_writable_volume_for_manifest(self):
        body, kind, name, prefix, user, mounts, cfg = self.parts(support.request("manifest", "toolchain"))
        tampered = copy.deepcopy(body)
        tampered["HostConfig"]["Mounts"][0]["ReadOnly"] = False
        with self.assertRaises(launcher.Refusal) as refused:
            launcher.check_body(tampered, kind, name, prefix, user, mounts, cfg)
        self.assertIn("writable volume for manifest", refused.exception.detail)
        table = dict(launcher.KINDS, manifest=(("toolchain",), "/work", "/", launcher.MANIFEST_VARIABLES, ()))
        with mock.patch.dict(containers.CALL_ROLES, {"manifest": ("ctl", "scratch")}):
            wrong = launcher.mounts_of("manifest", name, prefix, "toolchain", table)
            wrong_body = launcher.body_of(cfg, kind, name, prefix, "toolchain", containers.MANIFEST_READ, {}, support.LIMITS, user, wrong, 1.0)
            with self.assertRaises(launcher.Refusal):
                launcher.check_body(wrong_body, kind, name, prefix, user, wrong, cfg)

    def test_the_isolation_of_the_body_cannot_be_dropped(self):
        body, kind, name, prefix, user, mounts, cfg = self.parts(support.request("light"))
        for key, value in (("NetworkMode", "host"), ("NetworkMode", "bridge"), ("ReadonlyRootfs", False), ("CapDrop", []),
                           ("SecurityOpt", []), ("Privileged", True), ("Init", True), ("IpcMode", "host"), ("CgroupnsMode", "host"),
                           ("AutoRemove", True), ("Binds", ["/:/host"]), ("Devices", [{"PathOnHost": "/dev/kvm"}]),
                           ("DeviceRequests", [{"Count": -1}]), ("PortBindings", {"80/tcp": [{}]}), ("CapAdd", ["SYS_ADMIN"]),
                           ("VolumesFrom", ["x"]), ("MemorySwap", -1), ("Memory", 0), ("PidsLimit", 0)):
            tampered = copy.deepcopy(body)
            tampered["HostConfig"][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(launcher.Refusal):
                launcher.check_body(tampered, kind, name, prefix, user, mounts, cfg)
        for mount in ({"Type": "bind", "Source": "/", "Target": "/host", "ReadOnly": False},
                      {"Type": "volume", "Source": "another-volume", "Target": "/x", "ReadOnly": True, "VolumeOptions": {}}):
            tampered = copy.deepcopy(body)
            tampered["HostConfig"]["Mounts"].append(mount)
            with self.assertRaises(launcher.Refusal):
                launcher.check_body(tampered, kind, name, prefix, user, mounts, cfg)
        tampered = copy.deepcopy(body)
        tampered["Entrypoint"] = ["/bin/sh", "-c", "exec \"$@\"", "sh"]
        with self.assertRaises(launcher.Refusal):
            launcher.check_body(tampered, kind, name, prefix, user, mounts, cfg)


if __name__ == "__main__":
    unittest.main()
