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
