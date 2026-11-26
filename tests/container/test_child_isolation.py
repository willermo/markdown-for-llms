"""Isolation of the children as real containers (MFL051, plan r2.1 J.4.1), kind by kind.

`TestCreatedAsAsked` creates, for every kind and image, the container of the launcher's own body and reads it
back from Docker: the verification after the creation passes on the real thing, and the inspect is recorded (made
anonymous) as the positive fixture of `tests/unit/launcher/test_launcher_verify.py`. The other classes run
children through the launcher and look from inside: the probe (`selfprobe container`) for `light`, `marker` and
`probe`, a fixed shell probe for the toolchain image and for the program of the CPU server, real tools for
`tool`, the two fixed commands for `manifest`. What the tests assert on mounts and users they also read from
Docker itself, not only from what the launcher answers.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import socket
import tempfile
import threading
import time
import unittest

import container_support as support
from markdown_for_llms import launcher as launcher_module
from markdown_for_llms.engine import containers, supervisor, toolchain
from markdown_for_llms.engine.codes import Code, EngineError

FAKE_ID = "c0ffee" + "0123456789abcdef" * 3 + "c0ffee0123"       # 64 hex characters
FIXTURE_PROJECT = "mdf-fixture"
ROFS = ("EROFS",)
CASES = (("tool", "toolchain", ["qpdf", "--version"]), ("light", "app", containers.PROBE_PYTHON),
         ("marker", "app", containers.python_command(containers.PROBE_MODULE, ["container"])),
         ("cpuserver", "app", ["/opt/llama/probe.sh"]), ("probe", "app", containers.PROBE_PYTHON),
         ("probe", "toolchain", containers.PROBE_SHELL), ("manifest", "toolchain", containers.MANIFEST_READ),
         ("manifest", "toolchain", containers.MANIFEST_REGENERATE))


def variables_for(kind, image):
    if kind == "manifest":
        return {"LC_ALL": "C.UTF-8", "LANG": "C.UTF-8"}
    if kind == "cpuserver":
        return {"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8", "HOME": "/tmp", "LD_LIBRARY_PATH": "/opt/llama"}
    if image == "app":
        return dict(support.VARIABLES)
    return {"LC_ALL": "C.UTF-8", "LANG": "C.UTF-8", "HOME": "/work", "TMPDIR": "/work"}


class Base(support.ComponentCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        support.prepare_volumes()
        cls.launcher.toolchain()                # the toolchain image is verified by the launcher once

    def shell(self, kind, image, argv=None, limits=None, **run):
        """Run the fixed shell probe (`probe` on the toolchain image, or the program of the CPU server); returns
        (outcome, parsed report)."""
        with self.child(kind) as child:
            run.setdefault("timeout", 60)
            outcome = child.run(image, argv or containers.PROBE_SHELL, variables_for(kind, image),
                                dict(support.LIMITS, **(limits or {})), **run)
        return outcome, support.parse_shell_probe(outcome.stdout.decode("utf-8", "replace"))


def anonymous(text, ident, project, uid, gid):
    """A recorded inspect without what identifies the host or the run: the ID of the container (and so its
    hostname and the paths of Docker that hold it), the name of the compose project, the user."""
    text = text.replace(ident, FAKE_ID).replace(ident[:12], FAKE_ID[:12]).replace(project, FIXTURE_PROJECT)
    text = re.sub(r"/var/lib/docker/overlay2/[0-9a-f-]+", "/var/lib/docker/overlay2/" + "0" * 64, text)
    text = re.sub(r'"(Created|StartedAt|FinishedAt)": "[^"]*"', r'"\1": "2026-01-01T00:00:00Z"', text)
    return text.replace('"%d:%d"' % (uid, gid), '"1000:1000"')


@unittest.skipUnless(support.available(), support.REASON)
class TestCreatedAsAsked(Base):
    """What Docker really makes of the launcher's body: the verification passes, field by field, and the inspect
    becomes the fixture of the unit tests. The driver creates these containers itself, with the functions of the
    launcher, to read them before a start; it never starts them."""

    def config(self):
        images = {}
        for role, ref in (("app", self.hello["images"]["app"]["id"]), ("toolchain", toolchain.tag())):
            status, found = self.raw("GET", "/images/%s/json" % ref)
            self.assertEqual(status, 200, ref)
            images[role] = {"id": found["Id"], "architecture": found["Architecture"], "inspect": found}
        volumes = {role: "%s_mdf-%s" % (self.project, name) for role, name in
                   (("work", "work"), ("run", "run"), ("models", "models"), ("cpuserver", "cpu-server"))}
        return launcher_module.Config(self.project, os.getuid(), os.getgid(), volumes,
                                      {"work": containers.WORK_ROOT, "run": containers.RUN_ROOT}, images, self.hello["docker"])

    def test_every_kind_is_created_as_asked_and_recorded(self):
        config = self.config()
        for kind, image, argv in CASES:
            label = "%s-%s%s" % (kind, image, "-root" if argv == containers.MANIFEST_REGENERATE else "")
            with self.subTest(case=label):
                name = self.name(kind)
                limits = dict(support.LIMITS, **({"data_mib": 512} if kind == "tool" else {}))
                request = {"op": "run", "kind": kind, "name": name, "image": image, "argv": argv,
                           "env": variables_for(kind, image), "limits": limits}
                call = containers.Call(containers.WORK_ROOT, name)
                self.addCleanup(call.close, True)
                if kind in ("marker", "cpuserver"):
                    containers.socket_dirs(containers.RUN_ROOT, self.prefix)
                    self.addCleanup(self._remove_sockets)
                kind_, name_, prefix, image_, argv_, variables, limits_ = launcher_module.check_request(request)
                user = launcher_module.check_command(kind, image, argv)
                mounts = launcher_module.mounts_of(kind, name, prefix, image)
                launcher_module.check_folders(config.roots, mounts, config.uid)
                body = launcher_module.body_of(config, kind, name, prefix, image, argv, variables, limits, user, mounts, 1000.0)
                launcher_module.check_body(body, kind, name, prefix, user, mounts, config)
                query = {"name": name, **({} if image == "app" else {"platform": "linux/amd64"})}
                status, created = self.raw("POST", "/containers/create", query=query, body=body)
                self.assertEqual(status, 201, created)
                try:
                    found = self.raw_inspect(created["Id"])
                    self.assertEqual(launcher_module.verify(found, body, config.images[image]["inspect"], config.daemon), [])
                    # the same container, read by Docker: nothing writable but scratch, stat and the socket of the CPU server
                    writable = sorted(m["Destination"] for m in found["Mounts"] if m["RW"])
                    expected = {"tool": ["/stat", "/work"], "light": ["/scratch", "/stat"], "marker": ["/scratch", "/stat"],
                                "cpuserver": ["/relay", "/stat"], "manifest": [],
                                "probe": ["/scratch", "/stat"] if image == "app" else ["/stat", "/work"]}[kind]
                    self.assertEqual(writable, expected)
                    self.assertEqual(found["Config"]["User"], "0:0" if user == "root" else "%d:%d" % (os.getuid(), os.getgid()))
                    self.assertEqual(found["HostConfig"]["GroupAdd"] or [], [])
                    self.record(label, request, body, found, config, image)
                finally:
                    self.raw("DELETE", "/containers/%s" % created["Id"], query={"force": "1", "v": "1"})

    def test_docker_keeps_a_subpath_inside_its_volume_and_follows_a_link_that_stays_in_it(self):
        """The assumption of the plan on `Subpath` and links (risk 1). Docker resolves the subpath inside the volume:
        a link that points out of it never mounts a path of the host; a link that stays inside the volume IS
        followed, to wherever it points. So Docker guards the boundary of the volume, and the boundary of the call
        is the launcher's (`check_folders` refuses every link, before Docker sees the request)."""
        base = Path(containers.WORK_ROOT) / ("linktest-" + self.attempt)
        (base / "real").mkdir(parents=True, mode=0o700)
        self.addCleanup(__import__("shutil").rmtree, base, True)
        (base / "real" / "marker-of-the-volume.txt").write_text("x")
        os.symlink("real", base / "inside")
        os.symlink("/etc", base / "absolute")
        os.symlink("../../../../../../../../etc", base / "upwards")
        volume = "%s_mdf-work" % self.project
        seen = {}
        for case in ("real", "inside", "absolute", "upwards"):
            body = {"Image": self.hello["images"]["app"]["id"], "Cmd": ["/bin/ls", "-A", "/mnt"], "User": "%d:%d" % (os.getuid(), os.getgid()),
                    "Labels": {"mdf.project": self.project},
                    "HostConfig": {"NetworkMode": "none", "ReadonlyRootfs": True, "CapDrop": ["ALL"],
                                   "Mounts": [{"Type": "volume", "Source": volume, "Target": "/mnt", "ReadOnly": True,
                                               "VolumeOptions": {"NoCopy": True, "Subpath": "%s/%s" % (base.name, case)}}]}}
            status, created = self.raw("POST", "/containers/create", query={"name": "%s-link-%s" % (self.prefix, case)}, body=body)
            if status != 201:
                seen[case] = "create refused: HTTP %d %s" % (status, str(created)[:160])
                continue
            try:
                status, answer = self.raw("POST", "/containers/%s/start" % created["Id"])
                if status != 204:
                    seen[case] = "start refused: HTTP %d %s" % (status, str(answer)[:160])
                    continue
                until = time.monotonic() + 30
                while self.raw_inspect(created["Id"])["State"]["Running"]:
                    self.assertLess(time.monotonic(), until)
                    time.sleep(0.05)
                _, logs = self.docker.call("GET", "/containers/%s/logs" % created["Id"], query={"stdout": "1", "stderr": "1"})
                seen[case] = "listed: " + " ".join(sorted(launcher_module.demux(logs, 1).decode().split()))
            finally:
                self.raw("DELETE", "/containers/%s" % created["Id"], query={"force": "1", "v": "1"})
        if (support.OUT).is_dir():
            with open(support.OUT / "observations.jsonl", "a", encoding="utf-8") as stream:
                stream.write(json.dumps({"observation": "Docker and a Subpath that is a link (risk 1)", **seen}, sort_keys=True) + "\n")
        self.assertEqual(seen["real"], "listed: marker-of-the-volume.txt")
        self.assertEqual(seen["inside"], "listed: marker-of-the-volume.txt")       # followed: it stays inside the volume
        for case in ("absolute", "upwards"):
            self.assertNotIn("passwd", seen[case], case)                             # never the /etc of the host or of the image
            self.assertNotIn("hostname", seen[case], case)
            self.assertTrue(seen[case].startswith(("create refused", "start refused")), seen[case])
        # the launcher refuses all three links itself
        for case in ("inside", "absolute", "upwards"):
            with self.assertRaises(launcher_module.Refusal):
                launcher_module.check_folders({"work": containers.WORK_ROOT}, [("work", "%s/%s" % (base.name, case), "/mnt", True)], os.getuid())
        launcher_module.check_folders({"work": containers.WORK_ROOT}, [("work", "%s/real" % base.name, "/mnt", True)], os.getuid())

    def record(self, label, request, body, found, config, image):
        if not (support.OUT / "fixtures").is_dir():
            return
        uid, gid = os.getuid(), os.getgid()
        entry = config.images[image]
        fixture = {
            "docker": config.daemon,
            "context": {"project": config.project, "uid": uid, "gid": gid, "volumes": config.volumes, "t0": 1000.0,
                        "image": {"id": entry["id"], "architecture": entry["architecture"],
                                  "inspect": {"Id": entry["id"], "Architecture": entry["architecture"],
                                              "Config": {key: entry["inspect"]["Config"].get(key) for key in ("Env", "Entrypoint", "User")}}}},
            "request": request,
            "inspect": found,
        }
        text = json.dumps(fixture, indent=1, sort_keys=True)
        text = anonymous(text, found["Id"], config.project, uid, gid)
        text = text.replace('"uid": %d' % uid, '"uid": 1000').replace('"gid": %d' % gid, '"gid": 1000')
        text = text.replace(self.attempt, "0123456789abcdef")
        text = text.replace(json.dumps(supervisor.SUPERVISOR), '"<supervisor>"')      # the scripts of the day, not recorded
        text = text.replace(json.dumps(containers.SHELL_PROBE), '"<shell-probe>"')
        (support.OUT / "fixtures" / ("inspect-%s.json" % label)).write_text(text + "\n")


@unittest.skipUnless(support.available(), support.REASON)
class TestPythonChildren(Base):
    """`light`, `marker` and the fixed Python `probe`: every row of J.4.1 that a process sees from inside."""

    def check_report(self, report, kind):
        scratch_only = sorted(["/scratch", "/stat", "/tmp"])
        self.assertEqual((report["loopback_refused"], report["dns_failed"], report["root_read_only"],
                          report["docker_socket_absent"], report["scratch_writable"], report["host_path_hidden"]),
                         (True, True, True, True, True, True))
        self.assertEqual(report["interfaces"], ["lo"])
        self.assertNotEqual(report["outside"], "ok")
        self.assertEqual((report["uid"], report["gid"]), (os.getuid(), os.getgid()))
        self.assertIn(report["groups"], ([], [os.getgid()]))              # never the group of the Docker socket
        self.assertEqual((int(report["cap_eff"], 16), int(report["cap_prm"], 16), int(report["cap_bnd"], 16)), (0, 0, 0))
        self.assertEqual((report["no_new_privs"], report["seccomp"]), ("1", "2"))
        self.assertEqual(report["cgroup"]["memory.max"], str(support.LIMITS["memory_mib"] * 1048576))
        self.assertEqual(report["cgroup"]["memory.swap.max"], "0")
        self.assertEqual(report["cgroup"]["pids.max"], str(support.LIMITS["pids"]))
        self.assertEqual(report["pid1"], [supervisor.BASH, supervisor.ARG0])
        self.assertEqual(report["ppid"], 1)
        self.assertEqual((report["tty"], report["stdin"]), (False, "/dev/null"))
        self.assertNotEqual(report["controlling_tty"], "ok")
        self.assertEqual(report["descriptors"], [0, 1, 2])
        self.assertEqual(report["limits"]["FSIZE"], [support.LIMITS["fsize_mib"] * 1048576] * 2)
        self.assertEqual(report["limits"]["CORE"], [0, 0])
        self.assertEqual(report["cwd"], "/scratch")
        self.assertEqual(report["writable_mounts"], scratch_only)
        self.assertEqual(report["reach"], {root: [] for root in scratch_only})
        for path, answer in report["writes"].items():
            self.assertIn(answer, ("EROFS", "ENOENT", "EACCES", "ENOTDIR"), path)
        for path in ("/ctl/mdf-probe", "/ctl/lease", "/ctl/stop", "/proc/self/root/ctl/lease", "/proc/1/root/ctl/lease",
                     "/proc/self/root/ctl/stop", "/usr/mdf-probe", "/ctl/lease (remove)", "/ctl/lease (link)"):
            self.assertEqual(report["writes"][path], "EROFS", path)
        inodes = report["inodes"]
        self.assertEqual(inodes["/scratch/.."], inodes["/"])              # the parent of a mount is the root of the container
        self.assertEqual(inodes["/stat/.."], inodes["/"])
        self.assertEqual(inodes["/proc/self/root/ctl"], inodes["/ctl"])
        self.assertEqual(inodes["/proc/1/root/ctl"], inodes["/ctl"])
        distinct = [tuple(inodes[p]) for p in ("/scratch", "/stat", "/ctl") + (("/in",) if kind != "probe" else ())]
        self.assertEqual(len(set(distinct)), len(distinct))
        targets = {m["target"]: m for m in report["mounts"]}
        self.assertTrue(targets["/"]["read_only"])
        for target, read_only in (("/scratch", False), ("/stat", False), ("/ctl", True)):
            self.assertEqual((targets[target]["read_only"], targets[target]["root"].rsplit("/", 1)[-1]),
                             (read_only, target[1:]), target)
        self.assertEqual(targets["/tmp"]["type"], "tmpfs")

    def test_light(self):
        outcome, report = self.probe("light")
        self.assertEqual((outcome.cause, outcome.exit, outcome.oom_killed), ("exited", 0, False))
        self.check_report(report, "light")
        # the image's (PATH, LANG, LC_ALL), the request's (HOME, TMPDIR), Docker's HOSTNAME, bash's SHLVL and _
        self.assertEqual(report["environment"], ["HOME", "HOSTNAME", "LANG", "LC_ALL", "PATH", "SHLVL", "TMPDIR", "_"])
        self.assertTrue({m["target"]: m for m in report["mounts"]}["/in"]["read_only"])
        self.assertEqual(report["writes"]["/in/mdf-probe"], "EROFS")
        self.assertEqual(report["writes"]["/proc/1/root/in/mdf-probe"], "EROFS")
        self.assertEqual(sorted(m["target"] for m in report["mounts"] if "/call/" in m["root"]), ["/ctl", "/in", "/scratch", "/stat"])

    def test_marker_with_its_weights_read_only_and_a_socket_through_a_read_only_mount(self):
        """Also the assumption of the plan on `/relay`: a child connects to a Unix socket that lives in a directory
        mounted read-only, and the server on the other side is the caller."""
        relay = Path(containers.RUN_ROOT) / "sock" / self.prefix / "relay"
        containers.socket_dirs(containers.RUN_ROOT, self.prefix)
        (support.MODELS / "marker" / "weights.txt").write_text("weights")
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server.bind(str(relay / "relay.sock"))
        server.listen(4)
        server.settimeout(30)
        served = []

        def serve():
            try:
                while len(served) < 3:
                    conn, _ = server.accept()
                    with conn:
                        served.append(conn.recv(100))
                        conn.sendall(b"pong %d\n" % len(served))
            except OSError:
                pass
        thread = threading.Thread(target=serve, daemon=True)
        thread.start()
        steps = [{"do": "connect", "path": "/relay/relay.sock", "send": "ping", "repeat": 3},
                 {"do": "try_write", "paths": ["/relay/mdf-probe", "/models/mdf-probe", "/models/weights.txt"]},
                 {"do": "stat", "paths": ["/models/weights.txt"]}]
        try:
            outcome, report = self.probe("marker", steps=steps, variables={"TORCH_DEVICE": "cpu", "HF_HUB_OFFLINE": "1"})
        finally:
            server.close()
            thread.join(5)
        self.assertEqual((outcome.cause, outcome.exit), ("exited", 0))
        self.check_report(report, "marker")
        self.assertEqual(self.step_result(report, "connect"), ["pong 1", "pong 2", "pong 3"])
        self.assertEqual(served, [b"ping\n"] * 3)
        self.assertEqual(self.step_result(report, "try_write"),
                         {"/relay/mdf-probe": "EROFS", "/models/mdf-probe": "EROFS", "/models/weights.txt": "EROFS"})
        self.assertIsNotNone(self.step_result(report, "stat")["/models/weights.txt"])
        targets = {m["target"]: m for m in report["mounts"]}
        self.assertEqual((targets["/models"]["read_only"], targets["/relay"]["read_only"]), (True, True))
        self.assertEqual(report["environment"], ["HF_HUB_OFFLINE", "HOME", "HOSTNAME", "LANG", "LC_ALL", "PATH", "SHLVL",
                                                 "TMPDIR", "TORCH_DEVICE", "_"])

    def test_the_fixed_python_probe_has_no_inputs(self):
        with self.child("probe") as child:
            (child.call.scratch / containers.LIMITS_FILE).write_text(json.dumps(support.CHILD_LIMITS))
            outcome = child.run("app", containers.PROBE_PYTHON, support.VARIABLES, support.LIMITS, timeout=60)
            report = json.loads((child.call.scratch / "probe.json").read_text())
        self.assertEqual((outcome.cause, outcome.exit), ("exited", 0))
        self.check_report(report, "probe")
        self.assertNotIn("/in", {m["target"] for m in report["mounts"]})

    def test_stdin_arrives_byte_for_byte_and_only_from_the_caller(self):
        data = bytes(range(256)) * 300 + b"\x00tail"
        outcome, report = self.probe(steps=[{"do": "stdin"}, {"do": "try_write", "paths": ["/ctl/stdin"]}], stdin=data)
        self.assertEqual(outcome.exit, 0)
        self.assertEqual(self.step_result(report, "stdin"), {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
        self.assertEqual(report["stdin"], "/ctl/stdin")
        self.assertEqual(self.step_result(report, "try_write"), {"/ctl/stdin": "EROFS"})

    def test_write_denied_through_a_link_and_between_calls(self):
        """A link in a writable mount that points to `/ctl` or `/in` does not make them writable; and nothing a
        call can reach is a directory of another call."""
        with self.child("light") as other:               # another call of the same attempt, with its folder
            (other.call.scratch / "secret").write_text("of the other call")
            other_inodes = [[info.st_dev, info.st_ino] for info in
                            (os.stat(other.call.root / role) for role in ("scratch", "in", "ctl", "stat"))]
            steps = [{"do": "symlink", "target": "/ctl/lease", "path": "/scratch/to-lease"},
                     {"do": "symlink", "target": "/in", "path": "/scratch/to-in"},
                     {"do": "symlink", "target": "/work/call/%s/ctl/lease" % other.name, "path": "/scratch/to-other"},
                     {"do": "symlink", "target": "../%s/scratch/secret" % other.name, "path": "/scratch/to-sibling"},
                     {"do": "try_write", "paths": ["/scratch/to-lease", "/scratch/to-in/x", "/scratch/to-other", "/scratch/to-sibling",
                                                   "/scratch/../ctl/lease", "/scratch/../../call/%s/ctl/lease" % other.name]},
                     {"do": "walk", "path": "/scratch"}, {"do": "walk", "path": "/stat"}, {"do": "walk", "path": "/tmp"},
                     {"do": "stat", "paths": ["/ctl", "/in"]}]
            outcome, report = self.probe(steps=steps)
            self.assertEqual(other.call.root.joinpath("ctl", "lease").read_text(), "0\n")
            self.assertEqual((other.call.scratch / "secret").read_text(), "of the other call")
        self.assertEqual(outcome.exit, 0)
        answers = self.step_result(report, "try_write")
        self.assertEqual((answers["/scratch/to-lease"], answers["/scratch/to-in/x"], answers["/scratch/../ctl/lease"]),
                         ("EROFS", "EROFS", "EROFS"))
        for path in ("/scratch/to-other", "/scratch/to-sibling", "/scratch/../../call/%s/ctl/lease" % other.name):
            self.assertEqual(answers[path], "ENOENT", path)
        reached = [tuple(i) for step in report["steps"] if step["do"] == "walk" for i in step["result"]]
        own = [tuple(i) for i in self.step_result(report, "stat").values()]
        self.assertGreater(len(reached), 6)
        for inode in [tuple(i) for i in other_inodes] + own:
            self.assertNotIn(inode, reached)

    def test_links_a_command_leaves_in_place_of_directories_of_its_results_are_not_followed(self):
        """SOL-A1-N1, where it comes from: the command writes `scratch`, so it can leave a link where a directory of
        its results should be. An ordinary open of the caller follows it, out of the call (a file of the caller:
        its keys, in the worker) or into another directory of it. The rooted read of the client follows neither."""
        secret = Path(tempfile.mkdtemp(prefix="mdf-outside-"))
        self.addCleanup(shutil.rmtree, secret, True)
        (secret / "keys.txt").write_text("SYNTHETIC SECRET OF THE CALLER")
        with self.child("light") as child:
            scratch, root = child.call.scratch, child.call.root
            (scratch / containers.LIMITS_FILE).write_text(json.dumps(support.CHILD_LIMITS))
            (root / "in" / "source.txt").write_text("OF THE INPUTS")
            steps = [{"do": "symlink", "target": str(secret), "path": "/scratch/out"},      # dangling for the child, real for the caller
                     {"do": "symlink", "target": "../in", "path": "/scratch/inside"},
                     {"do": "symlink", "target": str(secret / "keys.txt"), "path": "/scratch/result.md"}]
            (scratch / "steps.json").write_text(json.dumps(steps))
            outcome = child.run("app", containers.python_command(containers.PROBE_MODULE, ["container"]), support.VARIABLES,
                                support.LIMITS, timeout=60)
            self.assertEqual((outcome.cause, outcome.exit), ("exited", 0))
            report = json.loads(containers.read_regular(root, "scratch/probe.json", 1 << 20))
            self.assertEqual([step["result"] for step in report["steps"]], ["ok"] * 3)
            # what a caller that opens the paths gets
            self.assertEqual((scratch / "out" / "keys.txt").read_text(), "SYNTHETIC SECRET OF THE CALLER")
            self.assertEqual((scratch / "inside" / "source.txt").read_text(), "OF THE INPUTS")
            self.assertEqual((scratch / "result.md").read_text(), "SYNTHETIC SECRET OF THE CALLER")
            # and what the rooted read gets
            for name in ("scratch/out/keys.txt", "scratch/inside/source.txt", "scratch/result.md"):
                self.assertIsNone(containers.read_regular(root, name, 4096), name)
                self.assertEqual(containers.regular_size(root, name), 0, name)
            self.assertEqual(containers.read_regular(root, "in/source.txt", 4096), b"OF THE INPUTS")

    def test_the_memory_cap_is_of_the_whole_container(self):
        outcome, _ = self.probe(steps=[{"do": "allocate", "mib": 400}], limits={"memory_mib": 128})
        self.assertEqual(outcome.cause, "memory")
        error = containers.error_of(outcome, "selfprobe")
        self.assertEqual(error.code, Code.MEMORY_LIMIT)

    def test_processes_and_threads_stop_at_the_limit_of_the_container(self):
        steps = [{"do": "fork", "count": 60}, {"do": "threads", "count": 60}]
        outcome, report = self.probe(steps=steps, limits={"pids": 20})
        self.assertEqual(outcome.exit, 0)
        forked, threads = self.step_result(report, "fork"), self.step_result(report, "threads")
        self.assertEqual(forked["error"], "EAGAIN")
        self.assertTrue(10 <= forked["started"] <= 18, forked)            # 20 with the supervisor and the probe itself
        self.assertTrue(10 <= threads["started"] <= 18, threads)
        self.assertIsNotNone(threads["error"])

    def test_rlimit_nproc_counts_the_processes_of_the_user_on_the_whole_host(self):
        """Why the process limit of a child is `PidsLimit` and `child.py` sets no `RLIMIT_NPROC` any more (tranche
        A2): without a user namespace that limit counts every process of this UID on the host, so a small one
        stops a child that has two processes, while a large one stops nothing that the container does not."""
        outcome, report = self.probe(steps=[{"do": "fork", "count": 3}], child_limits={"process_limit": 4})
        self.assertEqual(outcome.exit, 0)
        self.assertNotEqual(report["limits"]["NPROC"], [4, 4])          # `child.py` no longer reads `process_limit`
        self.assertEqual(self.step_result(report, "fork"), {"started": 3, "error": None})
        # the fact itself, with the limit set as `child.py` used to: not one fork, in a container with two processes
        outcome, report = self.probe(steps=[{"do": "nproc", "limit": 4}, {"do": "fork", "count": 3}])
        self.assertEqual(outcome.exit, 0)
        self.assertEqual(self.step_result(report, "fork"), {"started": 0, "error": "EAGAIN"})

    def test_a_file_over_the_cap_is_refused(self):
        """The limit of `child.py` cannot be over the one of the container: here both are 4 MiB."""
        outcome, report = self.probe(steps=[{"do": "write", "path": "/scratch/big", "mib": 8}], limits={"fsize_mib": 4},
                                     child_limits={"output_mib": 4})
        self.assertEqual(outcome.exit, 0)
        self.assertEqual(self.step_result(report, "write"), "EFBIG")
        self.assertEqual(report["limits"]["FSIZE"], [4 << 20, 4 << 20])
        outcome, _ = self.probe(limits={"fsize_mib": 4}, child_limits={"output_mib": 8})      # over the hard limit: refused
        self.assertEqual((outcome.cause, outcome.exit), ("exited", 1))
        self.assertIn(b"ValueError", outcome.stderr)

    def test_the_scratch_cap_is_measured_by_the_caller_on_scratch_alone(self):
        outcome, _ = self.probe(steps=[{"do": "write", "path": "/scratch/big", "mib": 6}, {"do": "sleep", "seconds": 30}],
                                scratch_mib=4)
        self.assertEqual((outcome.cause, outcome.reason, outcome.exit), ("stopped", "output_limit", 202))

    def test_the_tmpfs_has_its_size(self):
        outcome, report = self.probe(steps=[{"do": "write", "path": "/tmp/big", "mib": 32}])
        self.assertEqual(self.step_result(report, "write"), "ENOSPC")


@unittest.skipUnless(support.available(), support.REASON)
class TestToolchainChildren(Base):
    def test_the_shell_probe_of_the_toolchain_image(self):
        outcome, report = self.shell("probe", "toolchain")
        self.assertEqual((outcome.cause, outcome.exit), ("exited", 0))
        self.assertEqual((report["uid"], report["gid"]), (str(os.getuid()), str(os.getgid())))
        self.assertEqual((int(report["CapEff"], 16), report["NoNewPrivs"], report["Seccomp"]), (0, "1", "2"))
        self.assertEqual(report["interface"], ["lo"])
        self.assertEqual(report["cgroup"], {"memory.max": str(support.LIMITS["memory_mib"] * 1048576), "memory.swap.max": "0",
                                            "pids.max": str(support.LIMITS["pids"])})
        self.assertTrue(report["pid1"].startswith("/bin/bash -c "))
        self.assertRegex(report["bash"], r"^5\.")
        # the image's, the request's, Docker's HOSTNAME, bash's SHLVL and _, and the PWD that `sh` exports itself
        self.assertEqual(sorted(report["variable"]), ["HOME", "HOSTNAME", "LANG", "LC_ALL", "LD_LIBRARY_PATH", "PATH", "PWD",
                                                      "SHLVL", "TMPDIR", "_"])
        self.assertEqual((report["ulimit_fsize"], report["ulimit_nofile"], report["ulimit_core"]),
                         (str(support.LIMITS["fsize_mib"] * 2048), str(support.LIMITS["nofile"]), "0"))
        self.assertEqual(report["write"], {"/usr/mdf-probe": "no", "/ctl/lease": "no", "/ctl/mdf-probe": "no",
                                           "/work/mdf-probe": "yes", "/stat/mdf-probe": "yes", "/tmp/mdf-probe": "yes",
                                           "/proc/self/root/ctl/lease": "no", "/proc/1/root/ctl/lease": "no",
                                           "/relay/mdf-probe": "no", "/opt/llama/mdf-probe": "no", "/models/mdf-probe": "no"})
        self.assertEqual((report["docker_socket"], report["tty"], report["stdin"]), ("no", "no", "/dev/null"))
        targets = {m["target"]: m for m in report["mount"]}
        self.assertEqual((targets["/"]["read_only"], targets["/ctl"]["read_only"], targets["/work"]["read_only"],
                          targets["/stat"]["read_only"]), (True, True, False, False))
        self.assertNotIn("/in", targets)
        self.assertEqual(sorted(t for t, m in targets.items() if "/call/" in m["root"]), ["/ctl", "/stat", "/work"])

    def run_tool(self, argv, inputs=(), stdin=None, limits=None, **run):
        with self.child("tool") as child:
            for sub in ("home", "tmp"):
                (child.call.scratch / sub).mkdir()
            for source, name in inputs:
                child.call.add_input(source, name)
            if stdin is not None:
                child.call.stdin(stdin)
            inspected = {}

            def before(name):
                inspected["before"] = self.raw_inspect(name)
            run.setdefault("timeout", 120)
            outcome = child.run("toolchain", argv, support.TOOL_VARIABLES,
                                dict(support.LIMITS, memory_mib=1024, pids=256, **(limits or {})), before=before, **run)
            files = {p.name: p.read_bytes() for p in child.call.scratch.iterdir() if p.is_file()}
        return outcome, files

    def test_a_real_tool_reads_in_and_stdin_and_writes_work_only(self):
        source = Path(containers.WORK_ROOT) / ("%s-source.md" % self.attempt)
        source.write_text("# Title\n\nAlpha *beta*.\n")
        self.addCleanup(source.unlink)
        pandoc = ["pandoc", "+RTS", "-M512m", "-RTS", "--sandbox", "-f", "markdown", "-t", "plain"]
        outcome, files = self.run_tool(pandoc + ["-o", "/work/out.txt", "/in/source.md"], inputs=[(source, "source.md")])
        self.assertEqual((outcome.cause, outcome.exit, outcome.stderr), ("exited", 0, b""))
        self.assertEqual(files["out.txt"], b"Title\n\nAlpha beta.\n")
        self.assertGreater(outcome.metrics["memory_peak_bytes"], 1 << 20)
        outcome, _ = self.run_tool(pandoc, stdin=b"From *stdin*.\n")
        self.assertEqual((outcome.exit, outcome.stdout), (0, b"From stdin.\n"))
        for target in ("/in/out.txt", "/in/source.md", "/ctl/out.txt", "/ctl/lease", "/proc/self/root/ctl/lease",
                       "/proc/1/root/in/source.md", "/usr/out.txt"):
            with self.subTest(target=target):
                outcome, files = self.run_tool(pandoc + ["-o", target, "/in/source.md"], inputs=[(source, "source.md")])
                self.assertEqual(outcome.cause, "exited")
                self.assertNotEqual(outcome.exit, 0)
                self.assertRegex(outcome.stderr.decode("utf-8", "replace"), r"(?i)read-only|permission denied|sandbox|not permitted")

    def test_a_tool_outside_the_list_and_pandoc_without_its_sandbox_never_start(self):
        for argv in (["sh", "-c", "id"], ["bash"], ["/bin/sh"], ["pandoc", "-f", "markdown"],
                     ["pandoc", "+RTS", "-M512m", "-RTS", "--sandbox", "--sandbox=false"]):
            with self.child("tool") as child:
                with self.assertRaises(EngineError) as refused:
                    child.run("toolchain", argv, support.TOOL_VARIABLES, support.LIMITS, timeout=30)
                self.assertEqual(refused.exception.code, Code.INVALID_SETTING)
                self.assertIsNone(self.raw_inspect(child.name))

    def test_the_program_of_the_cpu_server_sees_its_socket_directory_and_nothing_of_a_call_but_ctl_and_stat(self):
        outcome, report = self.shell("cpuserver", "app", argv=["/opt/llama/probe.sh"])
        self.assertEqual((outcome.cause, outcome.exit), ("exited", 0))
        self.assertEqual((int(report["CapEff"], 16), report["NoNewPrivs"], report["Seccomp"], report["interface"]), (0, "1", "2", ["lo"]))
        self.assertEqual(report["write"]["/relay/mdf-probe"], "yes")
        self.assertRegex(report["bash"], r"^5\.")              # the supervisor of the image of the application
        for path in ("/usr/mdf-probe", "/ctl/lease", "/opt/llama/mdf-probe", "/models/mdf-probe", "/work/mdf-probe",
                     "/proc/1/root/ctl/lease"):
            self.assertEqual(report["write"][path], "no", path)
        targets = {m["target"]: m for m in report["mount"]}
        self.assertEqual({t: m["read_only"] for t, m in targets.items() if t in ("/ctl", "/stat", "/opt/llama", "/models", "/relay")},
                         {"/ctl": True, "/stat": False, "/opt/llama": True, "/models": True, "/relay": False})
        self.assertEqual(sorted(report["variable"]), ["HOME", "HOSTNAME", "LANG", "LC_ALL", "LD_LIBRARY_PATH", "PATH", "PWD",
                                                      "SHLVL", "_"])
        self.assertTrue((Path(containers.RUN_ROOT) / "sock" / self.prefix / "llama" / "mdf-probe").exists())
        with self.child("cpuserver") as child:
            with self.assertRaises(EngineError) as refused:
                child.run("app", ["/opt/llama/../../bin/sh"], variables_for("cpuserver", "app"), support.LIMITS, timeout=30)
            self.assertEqual(refused.exception.code, Code.INVALID_SETTING)


@unittest.skipUnless(support.available(), support.REASON)
class TestManifest(Base):
    def run_manifest(self, argv, on_poll=None, **limits):
        seen = {}

        def watch(child, metrics):
            if "inspect" not in seen:
                found = self.raw_inspect(child.name)
                if found and found["State"]["Running"]:
                    seen["inspect"] = found
            return on_poll(child, seen) if on_poll else None
        with self.child("manifest") as child:
            seen["ctl_mode"] = oct(os.stat(child.call.root / "ctl").st_mode & 0o777)
            seen["root_mode"] = oct(os.stat(child.call.root).st_mode & 0o777)
            seen["roles"] = sorted(p.name for p in child.call.root.iterdir())
            outcome = child.run("toolchain", argv, {"LC_ALL": "C.UTF-8", "LANG": "C.UTF-8"},
                                dict(support.LIMITS, memory_mib=512, **limits), timeout=300, on_poll=watch, poll_seconds=0.05)
            seen["launcher"] = self.launcher
        return outcome, seen

    def test_the_stored_manifest_is_read_with_logs_and_is_the_one_of_toolchain_json(self):
        name = self.name("manifest")
        with containers.Child(self.launcher, containers.WORK_ROOT, name) as child:
            child.launcher.run("manifest", name, "toolchain", containers.MANIFEST_READ, {"LC_ALL": "C.UTF-8"}, support.LIMITS)
            until = time.monotonic() + 60
            while self.launcher.inspect(name)["state"] != "exited":
                child.call.renew()
                self.assertLess(time.monotonic(), until)
                time.sleep(0.05)
            found = self.raw_inspect(name)
            data = self.launcher.logs(name)
            self.assertEqual(self.launcher.remove(name), {"ok": True, "absent": True})
        self.assertEqual(found["State"]["ExitCode"], 0)
        self.assertEqual(hashlib.sha256(data).hexdigest(), toolchain.identity())
        self.assertEqual([(m["Destination"], m["RW"]) for m in found["Mounts"]], [("/ctl", False)])
        self.assertEqual(found["Config"]["User"], "%d:%d" % (os.getuid(), os.getgid()))
        answer = self.launcher.toolchain()
        self.assertEqual((answer["verified"], answer["content_sha256"]), (True, toolchain.identity()))

    def test_the_regeneration_runs_as_root_without_a_writable_volume_and_sees_the_stop_of_the_caller(self):
        """The supervisor of a root child without capabilities reads `ctl/stop` and `ctl/lease` of a 0755 `ctl`."""
        def stop_when_running(child, seen):
            return "seen" if "inspect" in seen else None
        started = time.monotonic()
        outcome, seen = self.run_manifest(containers.MANIFEST_REGENERATE, on_poll=stop_when_running)
        found = seen["inspect"]
        self.assertEqual(found["Config"]["User"], "0:0")
        self.assertEqual([(m["Destination"], m["RW"]) for m in found["Mounts"]], [("/ctl", False)])
        self.assertEqual((found["HostConfig"]["CapDrop"], found["HostConfig"]["CapAdd"], found["HostConfig"]["ReadonlyRootfs"],
                          found["HostConfig"]["NetworkMode"], found["HostConfig"]["SecurityOpt"]),
                         (["ALL"], None, True, "none", ["no-new-privileges"]))
        self.assertEqual((seen["ctl_mode"], seen["root_mode"], seen["roles"]), ("0o755", "0o700", ["ctl", "owner"]))
        self.assertEqual((outcome.cause, outcome.reason, outcome.exit), ("stopped", "seen", supervisor.EXIT_STOP))
        self.assertLess(time.monotonic() - started, 30)

    def test_a_root_supervisor_reads_the_lease_of_the_caller(self):
        """With a lease of one second the regeneration, which takes several, ends well only because its supervisor,
        root without capabilities, reads the renewals of the caller: without them it would end with 200 at once."""
        outcome, seen = self.run_manifest(containers.MANIFEST_REGENERATE, lease_seconds=1)
        self.assertEqual((outcome.cause, outcome.exit), ("exited", 0))
        self.assertGreater(outcome.seconds, 2.5)

    def test_logs_are_of_an_exited_manifest_child_only(self):
        outcome_name = {}

        def on_poll(child, metrics):
            if self.launcher.inspect(child.name)["state"] == "running":
                with self.assertRaises(containers.Refused) as refused:
                    self.launcher.logs(child.name)
                outcome_name["refused"] = refused.exception.detail
                return "seen"
            return None
        self.probe(steps=[{"do": "sleep", "seconds": 30}], on_poll=on_poll)
        self.assertEqual(outcome_name["refused"], "logs are of a manifest child only")


if __name__ == "__main__":
    unittest.main()
