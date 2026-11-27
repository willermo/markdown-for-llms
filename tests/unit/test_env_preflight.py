"""`scripts/env-preflight.sh`: what `make` and `bin/mdf` know on the host before a container starts (MFL051, plan
r2.1 J.5.3; rilievo G; E-3).

- `env_state` looks at the ORIGINAL path of the env file: inside a container the mounted file is already
  resolved, so a link can be seen only here. One word: ok, missing, link, type, owner, mode.
- `env_value` reads one name of the file, for the few settings `make` and compose need; it is how compose is kept
  from ever opening the env file of the user.
- `compose_environment` exports what compose.yaml interpolates, from the Docker context in use (E-3: the socket
  of the context, not a conventional path) and from the file.

The functions run for real, in `sh`, against synthetic files and a `docker` that answers what they ask.
"""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "env-preflight.sh"
DOCKER = r'''#!/bin/sh
[ -z "${FAKE_LOG:-}" ] || printf '%s\n' "$*" >> "$FAKE_LOG"
case "$*" in
  "ps -a -q --filter label=com.docker.compose.project="*) cat "$FAKE_STATE/containers" 2> /dev/null ;;
  "inspect --format"*) for last; do :; done; cat "$FAKE_STATE/$last.inspect" ;;
  "logs --tail 200 "*) for last; do :; done; cat "$FAKE_STATE/$last.log" 2> /dev/null ;;
  "context inspect"*) echo "${FAKE_ENDPOINT:-unix://$FAKE_SOCKET}" ;;
  "version --format"*) [ -n "${FAKE_API:-}" ] || exit 1; echo "$FAKE_API" ;;
  "info --format"*) echo "${FAKE_INFO:-2 [name=seccomp,profile=builtin name=cgroupns]}" ;;
  *) echo "unexpected: $*" >&2; exit 9 ;;
esac
'''


class PreflightTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(os.path.realpath(self.tmp.name))
        bin_dir = self.base / "bin"
        bin_dir.mkdir()
        (bin_dir / "docker").write_text(DOCKER)
        (bin_dir / "docker").chmod(0o755)
        self.checkout = self.base / "My Checkout_1"
        self.checkout.mkdir()
        import socket
        self.socket = self.base / "docker.sock"
        server = socket.socket(socket.AF_UNIX)
        server.bind(str(self.socket))
        self.addCleanup(server.close)
        self.env = {"PATH": "%s:%s" % (bin_dir, os.environ.get("PATH", "/usr/bin:/bin")), "FAKE_SOCKET": str(self.socket),
                    "FAKE_API": "1.45"}

    def sh(self, code, env=None, shell="sh"):
        proc = subprocess.run([shell, "-euc", '. "%s"; %s' % (SCRIPT, code)], capture_output=True, text=True,
                              cwd=self.checkout, env={**self.env, **(env or {})}, timeout=60)
        return proc.returncode, proc.stdout.strip(), proc.stderr.strip()

    def env_file(self, text, mode=0o600, name="env"):
        path = self.base / name
        path.write_text(text)
        path.chmod(mode)
        return path

    def test_the_state_of_an_env_file_on_its_original_path(self):
        good = self.env_file("A_KEY=synthetic\n")
        loose = self.env_file("A_KEY=synthetic\n", 0o644, "loose")
        link = self.base / "link"
        link.symlink_to(good)
        dangling = self.base / "dangling"
        dangling.symlink_to(self.base / "absent")
        fifo = self.base / "fifo"
        os.mkfifo(fifo)
        cases = ((good, "ok"), (loose, "mode"), (link, "link"), (dangling, "link"), (self.base / "absent", "missing"),
                 (self.base, "type"), (fifo, "type"), (Path("/etc/hostname"), "owner"))
        for path, expected in cases:
            with self.subTest(path=path.name):
                if expected == "owner" and (not path.is_file() or path.stat().st_uid == os.getuid()):
                    self.skipTest("no regular file of another user to look at")
                self.assertEqual(self.sh('env_state "%s"' % path), (0, expected, ""))

    def test_a_link_is_seen_before_any_chmod_could_follow_it(self):
        """The reason the state is read first: `chmod` follows a link. `make up` gives a file its mode only in the
        state `mode`, which a link never has."""
        target = self.env_file("A_KEY=synthetic\n", 0o644, "target")
        link = self.base / "link"
        link.symlink_to(target)
        self.assertEqual(self.sh('env_state "%s"' % link)[1], "link")
        self.assertEqual(target.stat().st_mode & 0o777, 0o644)

    def test_the_same_answers_in_dash_and_bash(self):
        good = self.env_file("A_KEY=synthetic\n")
        import shutil
        for shell in ("dash", "bash"):
            if shutil.which(shell) is None:
                continue
            with self.subTest(shell=shell):
                self.assertEqual(self.sh('env_state "%s"; env_state "%s"' % (good, self.base), shell=shell)[1], "ok\ntype")

    def test_one_value_of_the_file_the_last_line_that_sets_it(self):
        path = self.env_file("# MDF_PORT=1\nMDF_PORT=23465\n  export MDF_COMPOSE_DATA = './my data'  \n"
                             'MDF_COMPOSE_STATE="/srv/state" # where the state goes\nMDF_PORT=23467\n'
                             "MDF_PORTAL=9\nA_KEY=synthetic-value\n")
        for name, expected in (("MDF_PORT", "23467"), ("MDF_COMPOSE_DATA", "./my data"), ("MDF_COMPOSE_STATE", "/srv/state"),
                               ("MDF_UPLOAD_MIB", ""), ("MDF_POR", "")):
            with self.subTest(name=name):
                self.assertEqual(self.sh('env_value "%s" %s' % (path, name)), (0, expected, ""))
        self.assertEqual(self.sh('env_value "%s" MDF_PORT' % (self.base / "absent")), (0, "", ""))

    def test_a_comment_is_one_only_outside_quotes(self):
        """SOL-A2-N1, CARB-R2: a folder named `Documents # keep` is an ordinary path, and the archive must be the
        one written in the file. The same lines are read by the parser of the product for the settings."""
        cases = (
            ("MDF_COMPOSE_DATA='./Documents # keep'", "./Documents # keep"),            # ` #` between single quotes
            ('MDF_COMPOSE_DATA="./State # keep"', "./State # keep"),                    # and between double quotes
            ("MDF_COMPOSE_DATA='/srv/my #data' # the archive", "/srv/my #data"),        # a comment after a quoted value
            ('MDF_COMPOSE_DATA="/srv/my #data"   # the archive', "/srv/my #data"),
            ("MDF_COMPOSE_DATA='/srv/a'# no space before it", "/srv/a"),
            ('MDF_COMPOSE_DATA="/srv/my#data" # the archive', "/srv/my#data"),
            ("MDF_COMPOSE_DATA=/srv/archive # the archive", "/srv/archive"),            # no quotes: ` #` starts a comment
            ("MDF_COMPOSE_DATA=/srv/arch#ive", "/srv/arch#ive"),                        # and `#` without a space does not
            ("MDF_COMPOSE_DATA=/srv/my data   ", "/srv/my data"),
            ("export MDF_COMPOSE_DATA='./Documents # keep' # exported", "./Documents # keep"),
            ('  export   MDF_COMPOSE_DATA = "./State # keep"', "./State # keep"),
            ("export MDF_COMPOSE_DATA=/srv/archive # exported", "/srv/archive"),
            ("MDF_COMPOSE_DATA=''", ""),
            ("MDF_COMPOSE_DATA='it''s here'", "it''s here"),                            # a quote inside: as before
            ('MDF_COMPOSE_DATA="/srv/a b"', "/srv/a b"),
        )
        from dotenv import dotenv_values
        import io
        for line, expected in cases:
            with self.subTest(line=line):
                path = self.env_file("A_KEY=synthetic-value\n%s\nMDF_PORT=23465 # the port\n" % line)
                self.assertEqual(self.sh('env_value "%s" MDF_COMPOSE_DATA' % path), (0, expected, ""))
                self.assertEqual(self.sh('env_value "%s" MDF_PORT' % path), (0, "23465", ""))
                if "''s" not in line:                                   # the parser of the product reads the same value
                    self.assertEqual(dotenv_values(stream=io.StringIO(line + "\n"), interpolate=False)["MDF_COMPOSE_DATA"],
                                     expected)
        # what compose gets is the folder of the file, whole
        path = self.env_file("MDF_COMPOSE_DATA='./Documents # keep'\nMDF_COMPOSE_STATE=\"/srv/State # keep\" # state\n")
        status, values, err = self.exported(path)
        self.assertEqual((status, err), (0, ""))
        self.assertEqual((values["MDF_COMPOSE_DATA"], values["MDF_COMPOSE_STATE"]),
                         (str(self.checkout / "Documents # keep"), "/srv/State # keep"))

    def test_a_value_is_never_run_and_never_shown(self):
        made = self.base / "made"
        path = self.env_file("MDF_COMPOSE_DATA='/srv/$(touch %s) `touch %s` ; touch %s # x'\n"
                             "MDF_COMPOSE_STATE=\"/srv/$HOME ${PATH}\" # state\nA_SERVICE_KEY=synthetic-value\n"
                             % (made, made, made))
        status, out, err = self.sh('compose_environment "%s"; printf "%%s\\n%%s\\n" "$MDF_COMPOSE_DATA" "$MDF_COMPOSE_STATE" > "%s"'
                                   % (path, self.base / "seen"))
        self.assertEqual((status, out, err), (0, "", ""))               # nothing on the terminal: a value is never shown
        self.assertFalse(made.exists())                                 # and never run
        self.assertEqual((self.base / "seen").read_text().splitlines(),
                         ["/srv/$(touch %s) `touch %s` ; touch %s # x" % (made, made, made), "/srv/$HOME ${PATH}"])

    def exported(self, path, env=None):
        names = ("COMPOSE_PROJECT_NAME MDF_COMPOSE_UID MDF_COMPOSE_GID MDF_COMPOSE_DOCKER_SOCKET MDF_COMPOSE_DOCKER_GID "
                 "MDF_COMPOSE_ENV_FILE MDF_COMPOSE_DATA MDF_COMPOSE_STATE MDF_PORT MDF_COMPOSE_BODY_MIB "
                 "MDF_COMPOSE_MAX_CHILDREN").split()
        status, out, err = self.sh('compose_environment "%s"; for n in %s; do eval "v=\\${$n:-}"; echo "$n=$v"; done'
                                   % (path, " ".join(names)), env)
        return status, dict(line.split("=", 1) for line in out.splitlines() if "=" in line), err

    def test_what_compose_interpolates_with_the_defaults(self):
        status, values, err = self.exported("./.env")                   # no file yet: the defaults
        self.assertEqual((status, err), (0, ""))
        self.assertEqual(values, {
            "COMPOSE_PROJECT_NAME": "mycheckout_1",                     # the name compose derives from the directory
            "MDF_COMPOSE_UID": str(os.getuid()), "MDF_COMPOSE_GID": str(os.getgid()),
            "MDF_COMPOSE_DOCKER_SOCKET": str(self.socket), "MDF_COMPOSE_DOCKER_GID": str(self.socket.stat().st_gid),
            "MDF_COMPOSE_ENV_FILE": str(self.checkout / ".env"), "MDF_COMPOSE_DATA": str(self.checkout / "data"),
            "MDF_COMPOSE_STATE": str(self.checkout / "state"), "MDF_PORT": "23465", "MDF_COMPOSE_BODY_MIB": "58",
            "MDF_COMPOSE_MAX_CHILDREN": ""})

    def test_what_compose_interpolates_from_the_file_and_from_make(self):
        path = self.env_file("MDF_PORT=23467\nMDF_UPLOAD_MIB=200\nMDF_COMPOSE_DATA=/srv/archive\n"
                             "MDF_COMPOSE_STATE=./var/state\nMDF_COMPOSE_MAX_CHILDREN=4\nA_SERVICE_KEY=synthetic-value\n")
        status, values, _ = self.exported(path)
        self.assertEqual(status, 0)
        self.assertEqual((values["MDF_PORT"], values["MDF_COMPOSE_BODY_MIB"], values["MDF_COMPOSE_DATA"],
                          values["MDF_COMPOSE_STATE"], values["MDF_COMPOSE_MAX_CHILDREN"], values["MDF_COMPOSE_ENV_FILE"]),
                         ("23467", "208", "/srv/archive", str(self.checkout / "var/state"), "4", str(path)))
        status, values, _ = self.exported(path, {"MDF_COMPOSE_DATA": "/elsewhere"})      # `make up MDF_COMPOSE_DATA=...`
        self.assertEqual(values["MDF_COMPOSE_DATA"], "/elsewhere")
        # of the env file nothing else is exported: a key never reaches the environment of `make` or of compose
        status, out, _ = self.sh('compose_environment "%s"; env' % path)
        self.assertEqual(status, 0)
        self.assertNotIn("synthetic-value", out)
        self.assertNotIn("A_SERVICE_KEY", out)

    def test_a_port_that_is_not_a_number_stops_before_compose(self):
        path = self.env_file("MDF_PORT=23465; touch %s\n" % (self.base / "made"))
        status, _, err = self.sh('compose_environment "%s"' % path)
        self.assertEqual(status, 1)
        self.assertIn("MDF_PORT", err)
        self.assertFalse((self.base / "made").exists())                 # a value is never run

    def test_the_socket_is_the_one_of_the_docker_context(self):
        """E-3: no conventional path. A context on another socket is used as it is; one that is not a local
        socket, or whose socket is not there, stops with the reason."""
        status, out, _ = self.sh('docker_endpoint; echo "$mdf_socket $mdf_socket_gid"')
        self.assertEqual((status, out), (0, "%s %d" % (self.socket, self.socket.stat().st_gid)))
        link = self.base / "desktop.sock"                               # Docker Desktop: a link to the real socket
        link.symlink_to(self.socket)
        status, out, _ = self.sh('docker_endpoint; echo "$mdf_socket_gid"', {"FAKE_ENDPOINT": "unix://%s" % link})
        self.assertEqual((status, out), (0, str(self.socket.stat().st_gid)))
        for endpoint, reason in (("tcp://10.0.0.1:2376", "not a local socket"), ("ssh://host", "not a local socket"),
                                 ("unix://%s" % (self.base / "absent.sock"), "does not exist")):
            with self.subTest(endpoint=endpoint):
                status, _, err = self.sh("docker_endpoint", {"FAKE_ENDPOINT": endpoint})
                self.assertEqual(status, 1)
                self.assertIn(reason, err)

    def test_the_requirements_of_the_host(self):
        self.assertEqual(self.sh("docker_host_check"), (0, "", ""))
        for env, reason in (({"FAKE_API": "1.44"}, "Docker Engine 26 or newer"), ({"FAKE_API": ""}, "does not answer"),
                            ({"FAKE_INFO": "1 [name=seccomp,profile=builtin]"}, "cgroup v2"),
                            ({"FAKE_INFO": "2 [name=seccomp,profile=builtin name=rootless]"}, "rootless")):
            with self.subTest(env=env):
                status, _, err = self.sh("docker_host_check", env)
                self.assertEqual(status, 1)
                self.assertIn(reason, err)
        self.assertEqual(self.sh("docker_host_check", {"FAKE_API": "1.57"})[0], 0)
        self.assertEqual(self.sh("docker_host_check", {"FAKE_API": "2.0"})[0], 0)


class GpuOfTheUserTests(PreflightTests):
    """MFL051 B: `gpu_host_check` (an amd64 daemon with the runtime of the NVIDIA Container Toolkit) and
    `server_environment` (the remote server: an address that is not the loopback, its port, two private files)."""

    def test_the_host_of_a_gpu(self):
        nvidia = '{"io.containerd.runc.v2":{},"nvidia":{"path":"nvidia-container-runtime"}}'
        self.assertEqual(self.sh("gpu_host_check", {"FAKE_API": "amd64", "FAKE_INFO": nvidia})[0], 0)
        code, _, err = self.sh("gpu_host_check", {"FAKE_API": "arm64", "FAKE_INFO": nvidia})
        self.assertEqual(code, 1)
        self.assertIn("amd64 only", err)
        code, _, err = self.sh("gpu_host_check", {"FAKE_API": "amd64", "FAKE_INFO": '{"runc":{}}'})
        self.assertEqual(code, 1)
        self.assertIn("NVIDIA Container Toolkit", err)

    def server(self, lines, key_mode=0o600, tls_key_mode=0o600):
        (self.checkout / "srv" / "tls").mkdir(parents=True, exist_ok=True)
        for path, mode in ((self.checkout / "srv" / "key", key_mode), (self.checkout / "srv" / "tls" / "server.key",
                                                                        tls_key_mode),
                           (self.checkout / "srv" / "tls" / "server.crt", 0o644)):
            path.write_text("synthetic\n")
            path.chmod(mode)
        env = self.env_file(lines)
        return self.sh('compose_environment "%s" > /dev/null && server_environment "%s" && echo '
                       '"$MDF_COMPOSE_SERVER_ADDRESS $MDF_COMPOSE_SERVER_PORT $MDF_COMPOSE_SERVER_TLS '
                       '$MDF_COMPOSE_SERVER_KEY"' % (env, env))

    def test_the_environment_of_the_remote_server(self):
        files = "MDF_COMPOSE_SERVER_TLS=./srv/tls\nMDF_COMPOSE_SERVER_KEY=./srv/key\n"
        code, out, err = self.server("MDF_COMPOSE_SERVER_ADDRESS=192.0.2.10\n" + files)
        self.assertEqual(code, 0, err)
        self.assertEqual(out, "192.0.2.10 23467 %s/srv/tls %s/srv/key" % (self.checkout, self.checkout))
        for address in ("", "127.0.0.1", "::1", "localhost", "0.0.0.0"):
            with self.subTest(address=address):
                code, _, err = self.server("MDF_COMPOSE_SERVER_ADDRESS=%s\n" % address + files)
                self.assertEqual(code, 1)
                self.assertIn("MDF_COMPOSE_SERVER_ADDRESS", err)
        code, _, err = self.server("MDF_COMPOSE_SERVER_ADDRESS=192.0.2.10\n" + files, key_mode=0o644)
        self.assertEqual(code, 1)
        self.assertIn("MDF_COMPOSE_SERVER_KEY", err)
        code, _, err = self.server("MDF_COMPOSE_SERVER_ADDRESS=192.0.2.10\n" + files, tls_key_mode=0o640)
        self.assertEqual(code, 1)
        self.assertIn("MDF_COMPOSE_SERVER_TLS", err)
        code, _, err = self.server("MDF_COMPOSE_SERVER_ADDRESS=192.0.2.10\nMDF_COMPOSE_SERVER_PORT=8443x\n" + files)
        self.assertEqual(code, 1)
        self.assertIn("not a number", err)

    def test_the_remote_server_is_never_on_every_address_nor_the_loopback_in_ipv6(self):
        # SOL-B-N1: compose takes an IPv6 address between brackets ([::] and [::1] were published as :: and ::1);
        # every form of the unspecified address and of the loopback is refused, with or without the brackets
        files = "MDF_COMPOSE_SERVER_TLS=./srv/tls\nMDF_COMPOSE_SERVER_KEY=./srv/key\n"
        # an IPv4 address mapped is written as two parts here, so that the guard of repository hygiene does not read
        # the text as an address of the internet (tests/unit/test_repository_hygiene.py)
        mapped = ":".join(("", "", "ffff", ""))
        refused = ("::", "::1", "0:0:0:0:0:0:0:0", "0::0", "::0", "0:0:0:0:0:0:0:1", "0000:0000::0001", "::0:1",
                   mapped + "127.0.0.1", mapped + "127.9.8.7", mapped + "0.0.0.0", "::FFFF:7f00:1", mapped + "0:0",
                   "::1%lo")
        for address in refused + tuple("[%s]" % a for a in refused) + ("[]", "LocalHost", "127.1.2.3"):
            with self.subTest(address=address):
                code, out, err = self.server("MDF_COMPOSE_SERVER_ADDRESS=%s\n" % address + files)
                self.assertEqual((code, out), (1, ""))
                self.assertIn("MDF_COMPOSE_SERVER_ADDRESS is not the loopback nor every address", err)
        for address in ("[2001:db8::10]", "2001:db8::10", "[2001:db8:0:0:0:0:0:1]", "[fe80::1]",
                        "[%s192.0.2.10]" % mapped, "[::2]", "192.0.2.10"):
            with self.subTest(address=address):
                code, out, err = self.server("MDF_COMPOSE_SERVER_ADDRESS=%s\n" % address + files)
                self.assertEqual(code, 0, err)
                self.assertEqual(out.split(" ")[0], address)


class RestartReportTests(PreflightTests):
    """The note of the supervision on A2-M2: the restart policy stays (`unless-stopped`), so a service that leaves
    with an error is started again for ever; `make status`, and `make up` after its wait, name it with its count of
    restarts (`RestartCount` of `docker inspect`) and the last error line of its log."""

    def setUp(self):
        super().setUp()
        self.state = self.base / "docker-state"
        self.state.mkdir()
        self.log = self.base / "docker.log"
        self.env.update(FAKE_STATE=str(self.state), FAKE_LOG=str(self.log), COMPOSE_PROJECT_NAME="synthetic")

    def container(self, ident, service, restarts, status, log=""):
        with open(self.state / "containers", "a") as stream:
            stream.write(ident + "\n")
        (self.state / (ident + ".inspect")).write_text("%s %s %s\n" % (service, restarts, status))
        (self.state / (ident + ".log")).write_text(log)

    def report(self):
        status, out, err = self.sh("restart_report")
        self.assertEqual(err, "")
        return status, out.splitlines()

    def test_no_service_was_started_again(self):
        self.assertEqual(self.report(), (0, []))                        # the product is down: no container
        self.container("aaa", "worker", 0, "running", '{\n  "error": "an error of an earlier life"\n}\n')
        self.container("bbb", "init", 0, "exited", "init: /volumes/work ok\n")
        self.container("ccc", "config", 0, "exited")
        self.assertEqual(self.report(), (0, []))
        calls = self.log.read_text().splitlines()
        self.assertEqual(calls[-4], "ps -a -q --filter label=com.docker.compose.project=synthetic")   # this project only
        self.assertFalse([call for call in calls if call.startswith("logs")])                         # no log is read

    def test_a_service_that_keeps_leaving_is_named_with_the_last_error_line_of_its_log(self):
        self.container("aaa", "launcher", 0, "running")
        self.container("bbb", "worker", 7, "restarting",
                       '{\n  "error": "Unsupported database schema"\n}\n{\n  "error": "Storage upgrade incomplete"\n}\n\n')
        self.container("ccc", "api", 3, "running", "INFO: started\r\nTraceback (most recent call last):\nValueError: boom\n\n")
        self.container("ddd", "nginx", 2, "running")
        status, lines = self.report()
        self.assertEqual(status, 1)
        self.assertEqual(lines, [
            # the errors of the services are JSON over several lines: the line that names the error, not the brace
            'worker: started again by Docker 7 times (now restarting); the last error line of its log: '
            '"error": "Storage upgrade incomplete"',
            # no line names an error: the last line that is not empty
            "api: started again by Docker 3 times (now running); the last error line of its log: ValueError: boom",
            "nginx: started again by Docker 2 times (now running); the last error line of its log: (no log)"])

    def test_a_first_restart_in_progress_counts_and_a_long_line_is_bounded(self):
        self.container("aaa", "api", 0, "restarting", "error: " + "x" * 1000 + "\n")
        status, lines = self.report()
        self.assertEqual(status, 1)
        self.assertTrue(lines[0].startswith("api: started again by Docker 0 times (now restarting); the last error line"))
        self.assertLessEqual(len(lines[0]), 400)

    def test_an_inspect_that_fails_or_answers_something_else_is_skipped(self):
        with open(self.state / "containers", "w") as stream:
            stream.write("gone\n")                                      # removed between `ps` and `inspect`
        self.container("odd", "worker", "<no value>", "running")
        self.assertEqual(self.report(), (0, []))

    def test_the_recipes_that_report_it(self):
        import re
        text = (ROOT / "Makefile").read_text()
        bodies = {target: re.search(r"^%s:[^\n]*\n((?:\t[^\n]*\n)+)" % target, text, re.M).group(1) for target in ("up", "status")}
        for target, body in bodies.items():
            self.assertIn("restart_report ||", body, target)
        up = bodies["up"]
        self.assertEqual(up.count("restart_report"), 2)                 # when the wait ends well, and when it times out
        self.assertLess(up.index("$(COMPOSE) up -d --no-build"), up.index("restart_report"))
        # the restart policy of the product itself is not changed; the local GPU of the profile `gpu` (MFL051 B) alone
        # has a bounded one, so that a vLLM that cannot start does not hold a GPU for ever
        compose = (ROOT / "compose.yaml").read_text()
        self.assertEqual(compose.count("restart: unless-stopped"), 4)
        self.assertEqual(compose.count("on-failure"), 1)
        self.assertIn("restart: on-failure:3", compose[compose.index("\n  gpu:\n"):])


if __name__ == "__main__":
    unittest.main()
