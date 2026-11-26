"""What the launcher verifies before it serves anything (MFL051, plan r2.1 App. H.6; N-7 with its condition E-3).

- Access: the EFFECTIVE access to the Docker endpoint it will use, from its own identity, with a real connection
  and a real request (`version`, API 1.45 or newer). One test per outcome: here with the `Docker` of the fake, and,
  since the outcome that matters is the one of a real `connect`, with a real Unix socket served by this test and the
  check run as a separate process (`python -m markdown_for_llms.launcher check`): the suite forbids `connect` in its
  own process, and the launcher is a process of its own anyway. Without access: a closed diagnosis that names the
  endpoint, the error and the effective UID and GIDs; no fallback.
- Identity: its own container is the one container with the compose labels of its project and service, running,
  whose ID starts with its hostname; zero, two, or another hostname: it does not start.
- Host: cgroup v2, not rootless.
"""
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest

import launcher_support as support
from markdown_for_llms import launcher

PROJECT, SERVICE = "mdf-unit", "launcher"
WHO = r"effective uid %d, gids [0-9,]+" % os.geteuid()


def own(fake, name="mdf-unit-launcher-1", project=PROJECT, service=SERVICE, state="running", image=support.APP_ID):
    ident = fake.add(name, {"com.docker.compose.project": project, "com.docker.compose.service": service}, state=state)
    fake.containers[ident]["inspect"]["Image"] = image
    return ident


class TestAccessOutcomes(unittest.TestCase):
    """`check_access` on every answer a daemon, or its absence, can give."""

    def check(self, fake):
        return launcher.check_access(fake, identity=(1000, [126, 1000]))

    def test_access(self):
        access = self.check(support.FakeDocker())
        self.assertEqual((access.ok, access.error, access.version["ApiVersion"]), (True, None, "1.57"))
        self.assertEqual(access.detail, "docker endpoint unix:///fake/docker.sock: API 1.57; effective uid 1000, gids 126,1000")

    def test_each_failure_names_the_endpoint_the_error_and_the_identity(self):
        for kind in ("ENOENT", "EACCES", "ECONNREFUSED", "no answer"):
            fake = support.FakeDocker()
            fake.fail = kind
            access = self.check(fake)
            self.assertEqual((access.ok, access.error, access.version), (False, kind, None))
            self.assertEqual(access.detail, "docker endpoint unix:///fake/docker.sock: %s; effective uid 1000, gids 126,1000; "
                                            "no fallback" % kind)

    def test_an_api_too_old_or_too_new(self):
        for version, error in (({"ApiVersion": "1.44", "MinAPIVersion": "1.24"}, "API too old"),
                               ({"ApiVersion": "1.9"}, "API too old"), ({}, "API too old"), ({"ApiVersion": "latest"}, "API too old"),
                               ({"ApiVersion": "1.60", "MinAPIVersion": "1.50"}, "API too new")):
            fake = support.FakeDocker()
            fake.version = version
            access = self.check(fake)
            self.assertEqual((access.ok, access.error), (False, error), version)
            self.assertIn("1.45", access.detail)
            self.assertIn("effective uid 1000, gids 126,1000; no fallback", access.detail)
        for version in ({"ApiVersion": "1.45"}, {"ApiVersion": "1.45", "MinAPIVersion": "1.45"}, {"ApiVersion": "2.0", "MinAPIVersion": "1.40"}):
            fake = support.FakeDocker()
            fake.version = version
            self.assertTrue(self.check(fake).ok, version)

    def test_an_answer_that_is_not_the_version_is_no_answer(self):
        fake = support.FakeDocker()
        fake._route = lambda method, path, query, body: (500, {"message": "x"})
        self.assertEqual(self.check(fake).error, "no answer")
        fake._route = lambda method, path, query, body: (200, b"<html>")
        self.assertEqual(self.check(fake).error, "no answer")

    def test_the_identity_is_the_effective_one_of_the_process(self):
        access = launcher.check_access(support.FakeDocker())
        self.assertRegex(access.detail, WHO + "$")
        self.assertIn(str(os.getegid()), access.detail.rsplit("gids ", 1)[1].split(","))


class TestTheStartOfTheService(unittest.TestCase):
    """A2-e (H.6). The cleanup of the start is a pass of the watch made before the first request is served, not a
    second later: what a launcher that died left past its term is gone before any client is answered."""

    def serve(self, tick):
        import argparse
        import io
        import signal
        from unittest import mock
        order, made = [], []
        fake = support.FakeDocker()
        past = fake.add(support.ATTEMPT + "-light-9", {"mdf.project": support.PROJECT, "mdf.kind": "light",
                                                       "mdf.attempt": support.ATTEMPT, "mdf.max": "60", "mdf.lease": "30",
                                                       "mdf.t0": repr(1e12)})          # a label in the future: an expired term

        class Recording(launcher.Launcher):
            def __init__(self, *args, **kw):
                super().__init__(*args, **kw)
                made.append(self)

        class Server:
            def accept(self):
                order.append("accept")
                made[0].stopping.set()
                raise OSError("closed")

            def close(self):
                pass

        def first_tick(watch):
            order.append("tick")
            return tick(watch)
        for name in ("SIGTERM", "SIGINT"):
            self.addCleanup(signal.signal, getattr(signal, name), signal.getsignal(getattr(signal, name)))
        args = argparse.Namespace(docker="/fake/docker.sock", socket="/nonexistent/launcher.sock", project=support.PROJECT,
                                  service="launcher", work="/nonexistent/work", run="/nonexistent/run",
                                  volume=["work=w", "run=r"], max=[], max_children=8)
        real_tick = launcher.Watch.tick
        with mock.patch.object(launcher, "Docker", lambda path: fake), mock.patch.object(launcher, "Launcher", Recording), \
                mock.patch.object(launcher, "check_access", lambda docker: launcher.Access(True, None, "ok", dict(fake.version))), \
                mock.patch.object(launcher, "find_self", lambda *a: {"Image": "sha256:" + "a" * 64}), \
                mock.patch.object(Recording, "_image_inspect", lambda self, ref: {"Architecture": "amd64"}), \
                mock.patch.object(Recording, "identity", lambda self: {}), \
                mock.patch.object(launcher, "bind", lambda path, uid: Server()), \
                mock.patch.object(launcher.os, "geteuid", lambda: 1000), \
                mock.patch.object(launcher.Watch, "tick", lambda watch: first_tick(watch)):
            self.real_tick = real_tick
            status = launcher.serve(args, out=io.StringIO())
        return status, order, fake, past

    def test_the_first_pass_of_the_watch_comes_before_the_first_request(self):
        status, order, fake, past = self.serve(lambda watch: self.real_tick(watch))
        self.assertEqual((status, order[:2]), (0, ["tick", "accept"]))
        self.assertNotIn(past, fake.containers)                      # killed and removed by that pass
        self.assertEqual(len(fake.did("POST", "/kill")), 1)

    def test_a_first_pass_that_fails_does_not_stop_the_service(self):
        def broken(watch):
            raise RuntimeError("a bad pass")
        status, order, fake, past = self.serve(broken)
        self.assertEqual((status, order[:2]), (0, ["tick", "accept"]))


DRIBBLE_DRIVER = r"""
import json, sys, time
from markdown_for_llms import launcher
path, budget, mode = sys.argv[1], float(sys.argv[2]), sys.argv[3]
docker = launcher.Docker(path)
started = time.monotonic()
if mode == "call":
    try:
        status, data = docker.call("GET", "/containers/json", timeout=budget)
        outcome = [status, len(data)]
    except launcher.DockerUnreachable as exc:
        outcome = exc.kind
else:
    sut = launcher.Launcher(docker, launcher.Config("mdf-fixture", 1000, 1000, {}, {}))
    outcome = sut.stop(seconds=budget)
print(json.dumps({"outcome": outcome, "seconds": time.monotonic() - started}))
"""
MARGIN = 0.25                           # what a deadline may be exceeded by here: the start of the last wait and the scheduling


class TestTheWholeCallHasOneDeadline(unittest.TestCase):
    """SOL-FIX-N1 (A2-a). The timeout of a call to Docker bounded each wait on the socket, not their sum: a peer that
    sends its answer in pieces, each within the timeout, made the call last as long as it liked, and the stop of
    the launcher outlast its budget. The deadline is of the whole call, the body included. A real Unix peer and a
    real `connect`, so in a process of its own."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.path = os.path.join(tmp.name, "docker.sock")
        self.server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.server.bind(self.path)
        self.server.listen(8)
        self.addCleanup(self.server.close)

    def peer(self, pieces, pause, head_pause=0.0):
        """Answers every request with the JSON list `[]` padded to `pieces` pieces, one every `pause` seconds, after
        `head_pause` seconds for the head."""
        body = b"[" + b" " * (pieces - 2) + b"]" if pieces > 1 else b"[]"

        def serve():
            while True:
                try:
                    conn, _ = self.server.accept()
                except OSError:
                    return
                threading.Thread(target=answer, args=(conn,), daemon=True).start()

        def answer(conn):
            try:
                data = b""
                while b"\r\n\r\n" not in data:
                    chunk = conn.recv(4096)
                    if not chunk:
                        return
                    data += chunk
                time.sleep(head_pause)
                conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: %d\r\n\r\n" % len(body))
                for i in range(len(body)):
                    time.sleep(pause)
                    conn.sendall(body[i:i + 1])
            except OSError:
                pass
            finally:
                conn.close()
        threading.Thread(target=serve, daemon=True).start()

    def drive(self, budget, mode):
        done = subprocess.run([sys.executable, "-I", "-B", "-c", DRIBBLE_DRIVER, self.path, str(budget), mode],
                              capture_output=True, text=True, timeout=60)
        self.assertEqual(done.returncode, 0, done.stderr)
        return json.loads(done.stdout)

    def test_an_answer_in_pieces_within_the_budget_is_read_whole(self):
        self.peer(pieces=6, pause=0.02)
        got = self.drive(2.0, "call")
        self.assertEqual(got["outcome"], [200, 6])
        self.assertLess(got["seconds"], 2.0)

    def test_a_body_in_pieces_ends_at_the_deadline_of_the_call_not_at_the_sum_of_its_waits(self):
        """Ten pieces, one every 0.15 s: every wait is within the 0.4 s of the call, their sum is 1.5 s."""
        self.peer(pieces=10, pause=0.15)
        got = self.drive(0.4, "call")
        self.assertEqual(got["outcome"], "no answer")
        self.assertLessEqual(got["seconds"], 0.4 + MARGIN)

    def test_a_head_that_comes_late_and_a_body_after_it_share_the_same_deadline(self):
        self.peer(pieces=4, pause=0.15, head_pause=0.3)
        got = self.drive(0.4, "call")
        self.assertEqual(got["outcome"], "no answer")
        self.assertLessEqual(got["seconds"], 0.4 + MARGIN)

    def test_the_stop_of_the_launcher_ends_within_its_budget_with_a_peer_that_answers_in_pieces(self):
        """The reproduction of the review: `stop(seconds=0.4)` lasted about 0.5 s and said True. It ends within the
        budget and the declared margin, and says it did not see every child gone."""
        self.peer(pieces=10, pause=0.15)
        got = self.drive(0.4, "stop")
        self.assertIs(got["outcome"], False)
        self.assertLessEqual(got["seconds"], 0.4 + MARGIN)


class TestAccessOnARealSocket(unittest.TestCase):
    """The check as the launcher runs it: a separate process and a real `connect` on a socket of this test."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.path = os.path.join(tmp.name, "docker.sock")
        self.servers = []

    def tearDown(self):
        for server in self.servers:
            server.close()

    def listen(self, answer=None):
        """A Unix socket that answers every request with `answer` (a dict as JSON), or never answers (None)."""
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server.bind(self.path)
        server.listen(4)
        self.servers.append(server)
        seen = []

        def serve():
            held = []
            while True:
                try:
                    conn, _ = server.accept()
                except OSError:
                    return
                data = b""
                while b"\r\n\r\n" not in data:
                    chunk = conn.recv(4096)
                    if not chunk:
                        break
                    data += chunk
                seen.append(data.split(b"\r\n", 1)[0].decode())
                if answer is None:
                    held.append(conn)                   # keep it open and say nothing
                    continue
                body = json.dumps(answer).encode()
                conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: %d\r\n\r\n" % len(body) + body)
                conn.close()
        threading.Thread(target=serve, daemon=True).start()
        return seen

    def check(self, path=None, timeout="1"):
        done = subprocess.run([sys.executable, "-I", "-B", "-m", "markdown_for_llms.launcher", "check", "--docker", path or self.path,
                               "--timeout", timeout], stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=60, env={})
        return done.returncode, done.stdout.strip(), done.stderr.strip()

    def diagnosis(self, error, path=None):
        return r"^docker endpoint unix://%s: %s(?: \(.*\))?; %s; no fallback$" % (path or self.path, error, WHO)

    def test_access_with_a_real_request(self):
        seen = self.listen({"Version": "29.10.0", "ApiVersion": "1.57", "MinAPIVersion": "1.40"})
        status, out, err = self.check()
        self.assertEqual((status, err), (0, ""))
        self.assertRegex(out, r"^docker endpoint unix://%s: API 1\.57; %s$" % (self.path, WHO))
        self.assertEqual(seen, ["GET /version HTTP/1.1"])

    def test_no_socket_at_the_path_is_enoent(self):
        status, out, err = self.check()
        self.assertEqual((status, out), (launcher.EXIT_ACCESS, ""))
        self.assertRegex(err, self.diagnosis("ENOENT"))

    def test_a_socket_the_identity_may_not_use_is_eacces(self):
        """The mode of the socket decides for this very process: no group, no capability, no fallback."""
        self.listen({"ApiVersion": "1.57"})
        os.chmod(self.path, 0o000)
        status, _, err = self.check()
        self.assertEqual(status, launcher.EXIT_ACCESS)
        self.assertRegex(err, self.diagnosis("EACCES"))
        os.chmod(self.path, 0o600)
        self.assertEqual(self.check()[0], 0)

    def test_a_directory_that_cannot_be_entered_is_eacces(self):
        self.listen({"ApiVersion": "1.57"})
        os.chmod(os.path.dirname(self.path), 0o000)
        try:
            status, _, err = self.check()
        finally:
            os.chmod(os.path.dirname(self.path), 0o700)
        self.assertEqual(status, launcher.EXIT_ACCESS)
        self.assertRegex(err, self.diagnosis("EACCES"))

    def test_a_daemon_that_does_not_answer_is_no_answer(self):
        self.listen(None)
        status, _, err = self.check(timeout="0.5")
        self.assertEqual(status, launcher.EXIT_ACCESS)
        self.assertRegex(err, self.diagnosis("no answer"))

    def test_a_socket_nobody_listens_on_is_refused(self):
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server.bind(self.path)
        server.close()                                  # the file stays, nobody listens
        status, _, err = self.check()
        self.assertEqual(status, launcher.EXIT_ACCESS)
        self.assertRegex(err, self.diagnosis("ECONNREFUSED"))

    def test_an_old_api_is_named_with_the_version_needed(self):
        self.listen({"Version": "25.0.0", "ApiVersion": "1.44", "MinAPIVersion": "1.24"})
        status, _, err = self.check()
        self.assertEqual(status, launcher.EXIT_ACCESS)
        self.assertRegex(err, self.diagnosis("API too old"))
        self.assertIn("daemon API 1.44, the launcher needs 1.45", err)

    def test_the_service_does_not_start_without_access(self):
        """`serve` runs the same check first and exits with the same diagnosis: nothing is bound, nothing is served."""
        api = os.path.join(os.path.dirname(self.path), "launcher.sock")
        done = subprocess.run([sys.executable, "-I", "-B", "-m", "markdown_for_llms.launcher", "serve", "--docker", self.path,
                               "--socket", api, "--project", PROJECT, "--service", SERVICE, "--work", "/work", "--run", "/run",
                               "--volume", "work=w", "--volume", "run=r"], stdin=subprocess.DEVNULL, capture_output=True, text=True,
                              timeout=60, env={})
        self.assertEqual((done.returncode, done.stdout), (launcher.EXIT_ACCESS, ""))
        self.assertRegex(done.stderr.strip(), r"^launcher: " + self.diagnosis("ENOENT")[1:])
        self.assertFalse(os.path.exists(api))

    def test_the_command_line_is_closed(self):
        for argv in ([], ["serve"], ["check"], ["exec", "sh"], ["check", "--docker", self.path, "--fallback"]):
            done = subprocess.run([sys.executable, "-I", "-B", "-m", "markdown_for_llms.launcher", *argv], stdin=subprocess.DEVNULL,
                                  capture_output=True, text=True, timeout=60, env={})
            self.assertEqual(done.returncode, launcher.EXIT_USAGE, argv)


class TestOwnContainer(unittest.TestCase):
    def find(self, fake, hostname):
        return launcher.find_self(fake, PROJECT, SERVICE, hostname)

    def refused(self, fake, hostname, contains, status=launcher.EXIT_SELF):
        with self.assertRaises(launcher.Startup) as refused:
            self.find(fake, hostname)
        self.assertEqual(refused.exception.status, status)
        self.assertIn(contains, refused.exception.detail)

    def test_exactly_one_running_with_the_hostname_of_this_process(self):
        fake = support.FakeDocker()
        ident = own(fake)
        own(fake, "other-project-launcher-1", project="other")
        own(fake, "mdf-unit-worker-1", service="worker")
        found = self.find(fake, ident[:12])
        self.assertEqual((found["Id"], found["Image"]), (ident, support.APP_ID))
        query = fake.did("GET", "/containers/json")[0][2]
        self.assertEqual(json.loads(query["filters"]), {"label": ["com.docker.compose.project=mdf-unit",
                                                                   "com.docker.compose.service=launcher"]})

    def test_no_container(self):
        fake = support.FakeDocker()
        own(fake, "other-launcher-1", project="other")
        self.refused(fake, "0123456789ab", "0 containers of mdf-unit/launcher")

    def test_two_containers(self):
        fake = support.FakeDocker()
        ident = own(fake)
        own(fake, "mdf-unit-launcher-2")
        self.refused(fake, ident[:12], "2 containers of mdf-unit/launcher")

    def test_another_hostname(self):
        fake = support.FakeDocker()
        ident = own(fake)
        other = hashlib.sha256(b"another").hexdigest()[:12]
        for hostname in (other, "", ident[:11], "launcher", ident[1:13]):
            self.refused(fake, hostname, "is not this process")

    def test_not_running(self):
        fake = support.FakeDocker()
        ident = own(fake, state="exited")
        self.refused(fake, ident[:12], "is not this process (state exited)")

    def test_an_image_that_is_not_an_id(self):
        fake = support.FakeDocker()
        ident = own(fake, image="mdf-app:engine")
        self.refused(fake, ident[:12], "own container unreadable")

    def test_docker_unreachable(self):
        fake = support.FakeDocker()
        fake.fail = "EACCES"
        self.refused(fake, "0123456789ab", "EACCES", status=launcher.EXIT_ACCESS)


class TestHost(unittest.TestCase):
    def test_what_the_launcher_keeps_of_the_daemon(self):
        fake = support.FakeDocker()
        self.assertEqual(launcher.daemon_of(fake, fake.version), support.DAEMON)

    def test_cgroup_v1_and_rootless_are_refused(self):
        for change, contains in (({"CgroupVersion": "1"}, "cgroup v1"), ({"CgroupVersion": None}, "cgroup v"),
                                 ({"SecurityOptions": ["name=seccomp,profile=builtin", "name=rootless"]}, "rootless")):
            fake = support.FakeDocker()
            fake.info.update(change)
            with self.assertRaises(launcher.Startup) as refused:
                launcher.daemon_of(fake, fake.version)
            self.assertEqual(refused.exception.status, launcher.EXIT_HOST)
            self.assertIn(contains, refused.exception.detail)

    def test_without_apparmor_no_profile_is_expected(self):
        fake = support.FakeDocker()
        fake.info["SecurityOptions"] = ["name=seccomp,profile=builtin"]
        self.assertEqual(launcher.daemon_of(fake, fake.version)["apparmor"], "")

    def test_the_socket_of_the_api_needs_a_private_directory_of_the_user(self):
        with tempfile.TemporaryDirectory() as tmp:
            good = Path(tmp) / "good"
            good.mkdir(mode=0o700)
            server = launcher.bind(str(good / "launcher.sock"), os.getuid())
            server.close()
            self.assertEqual(os.stat(good / "launcher.sock").st_mode & 0o777, 0o600)
            launcher.bind(str(good / "launcher.sock"), os.getuid()).close()      # a socket left by a dead launcher is replaced
            (good / "file").write_text("x")
            for directory, uid in ((Path(tmp) / "open", os.getuid()), (good, os.getuid() + 1)):
                directory.mkdir(mode=0o755, exist_ok=True)
                with self.assertRaises(launcher.Startup) as refused:
                    launcher.bind(str(directory / "launcher.sock"), uid)
                self.assertEqual(refused.exception.status, launcher.EXIT_HOST)
            os.symlink(good, Path(tmp) / "link")
            with self.assertRaises(launcher.Startup):
                launcher.bind(str(Path(tmp) / "link" / "launcher.sock"), os.getuid())


if __name__ == "__main__":
    unittest.main()
