"""The socket guard of the suite (tests/conftest.py) and its one exception (MFL051, plan r2.1 App. I.4).

The suite forbids every connection: no test reaches the network or a daemon. The engine starts its children only
through the launcher, so `connect` of an `AF_UNIX` socket to the exact path of the launcher's socket is let
through, and nothing else: any other Unix path (the Docker socket included), every `AF_INET` address, `connect_ex`
and `sendto` stay forbidden.
"""
import importlib.util
from pathlib import Path
import socket
import tempfile
import unittest

from markdown_for_llms.engine import containers

ROOT = Path(__file__).resolve().parents[2]


def load():
    """tests/conftest.py as a module of its own: its functions, whether the suite installed them or not."""
    spec = importlib.util.spec_from_file_location("mdf_conftest_under_test", ROOT / "tests" / "conftest.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class GuardTests(unittest.TestCase):
    def setUp(self):
        self.guard = load()
        self.reached = []
        self.guard._guard_originals["connect"] = lambda sock, address, *a, **kw: self.reached.append(address)

    def connect(self, family, address):
        sock = socket.socket(family, socket.SOCK_STREAM)
        self.addCleanup(sock.close)
        return self.guard._connect_launcher_only(sock, address)

    def test_the_constant_is_the_socket_of_the_launcher(self):
        self.assertEqual(self.guard.LAUNCHER_SOCKET, containers.LAUNCHER_SOCKET)

    def test_the_socket_of_the_launcher_is_let_through(self):
        self.connect(socket.AF_UNIX, containers.LAUNCHER_SOCKET)
        self.assertEqual(self.reached, [containers.LAUNCHER_SOCKET])

    def test_every_other_unix_address_is_forbidden_the_docker_socket_included(self):
        with tempfile.TemporaryDirectory() as tmp:
            link = Path(tmp) / "launcher.sock"
            link.symlink_to(containers.LAUNCHER_SOCKET)
            others = ["/var/run/docker.sock", "/run/docker.sock", str(link), "/run/mdf/launcher/../launcher/launcher.sock",
                      containers.LAUNCHER_SOCKET + ".other", "launcher.sock", "", containers.LAUNCHER_SOCKET.encode(),
                      "\0" + containers.LAUNCHER_SOCKET, "/run/mdf/run/sock/mdf-eng-x/relay/relay.sock"]
            for address in others:
                with self.subTest(address=address), self.assertRaisesRegex(RuntimeError, "vietati"):
                    self.connect(socket.AF_UNIX, address)
        self.assertEqual(self.reached, [])

    def test_every_inet_address_is_forbidden(self):
        for family, address in ((socket.AF_INET, ("127.0.0.1", 80)), (socket.AF_INET, ("198.51.100.1", 443)),
                                (socket.AF_INET6, ("::1", 80)), (socket.AF_INET, containers.LAUNCHER_SOCKET)):
            with self.subTest(address=address), self.assertRaisesRegex(RuntimeError, "vietati"):
                self.connect(family, address)
        self.assertEqual(self.reached, [])

    def test_connect_ex_and_sendto_have_no_exception(self):
        for name in ("connect_ex", "sendto"):
            with self.subTest(name=name), self.assertRaisesRegex(RuntimeError, "vietati"):
                self.guard._deny_network(None, containers.LAUNCHER_SOCKET)
        source = (ROOT / "tests" / "conftest.py").read_text(encoding="utf-8")
        self.assertIn("setattr(socket.socket, name, _connect_launcher_only if name == 'connect' else _deny_network)", source)

    def test_as_the_suite_installed_it(self):
        """With the guard of this very run (absent with `--noconftest`): the launcher's socket is reached, or is
        simply not there; the Docker socket and the network are refused before any system call."""
        installed = socket.socket.connect
        if getattr(installed, "__name__", "") != "_connect_launcher_only":
            self.skipTest("the suite runs without its conftest")
        for family, address in ((socket.AF_UNIX, "/var/run/docker.sock"), (socket.AF_INET, ("127.0.0.1", 9))):
            sock = socket.socket(family, socket.SOCK_STREAM)
            self.addCleanup(sock.close)
            with self.assertRaisesRegex(RuntimeError, "vietati"):
                sock.connect(address)
            with self.assertRaisesRegex(RuntimeError, "vietati"):
                sock.connect_ex(address)
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.addCleanup(sock.close)
        try:
            sock.connect(containers.LAUNCHER_SOCKET)
        except OSError:                     # no launcher here: the call itself was made
            pass


if __name__ == "__main__":
    unittest.main()
