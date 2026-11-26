"""`bin/mdf`, the CLI of the product on the host, and the file `make install-cli` writes (MFL051, plan r2.1 §6,
J.5.3, J.6; E-2), against a `docker` that is not there.

- `bin/mdf` runs `mdf ...` in the one-off `cli` service: as the user, with the current directory mounted at its
  own path, with the settings of the product and the keys file of the checkout; without a terminal, `-T`.
- The env file is looked at on its original path BEFORE anything is mounted: a link, another type or another
  owner are refused; another mode is mounted as it is (never a `chmod` here). A file of `--env-file` that is not
  there is not mounted at all: Docker would make a directory in its place.
- When the launcher of the product does not run: exit 1 with `isolation_unavailable`, and neither `compose run`
  nor `up`. Nothing is started for the user, nothing falls back to the host. `mdf configure` and
  `mdf downgrade-storage` alone run without it (SOL-A2-B2): they change the archive, which needs the product
  stopped, and they convert nothing.
- Exit codes are those of `mdf`; 127 when Docker or the checkout is missing.
- The installed file runs `bin/mdf` of the checkout it was installed from, and says so when that checkout was
  moved or removed (E-2).

The script runs for real, in a synthetic checkout, with `docker` replaced by a recorder.
"""
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
DOCKER = r'''#!/bin/sh
printf '%s\n' "$*" >> "$FAKE_LOG"
case "$*" in
  "context inspect"*) echo "unix://$FAKE_SOCKET" ;;
  *" ps --status running -q launcher") [ "${FAKE_NO_LAUNCHER:-}" = 1 ] || echo "launcher-container" ;;
  *" run --rm "*) echo "ran in $(pwd)"; exit "${FAKE_EXIT:-0}" ;;
esac
exit 0
'''
SETTINGS = "--env-file=/run/mdf/settings/settings.env"
UID, GID = os.getuid(), os.getgid()


class Base(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.base = Path(os.path.realpath(tmp.name))
        self.top = self.base / "checkout"
        for directory in ("bin", "scripts"):
            (self.top / directory).mkdir(parents=True)
        shutil.copy(ROOT / "bin" / "mdf", self.top / "bin" / "mdf")
        shutil.copy(ROOT / "scripts" / "env-preflight.sh", self.top / "scripts" / "env-preflight.sh")
        (self.top / "bin" / "mdf").chmod(0o755)
        (self.top / "compose.yaml").write_text("services: {}\n")
        self.env_file = self.top / ".env"
        self.env_file.write_text("MDF_PORT=23465\nA_SERVICE_KEY=synthetic-value\n")
        self.env_file.chmod(0o600)
        self.tools = self.base / "tools"
        self.tools.mkdir()
        (self.tools / "docker").write_text(DOCKER)
        (self.tools / "docker").chmod(0o755)
        self.log = self.base / "docker.log"
        self.socket = self.base / "docker.sock"
        server = socket.socket(socket.AF_UNIX)
        server.bind(str(self.socket))
        self.addCleanup(server.close)
        self.work = self.base / "my documents"
        self.work.mkdir()
        self.env = {"PATH": "%s:%s" % (self.tools, os.environ.get("PATH", "/usr/bin:/bin")), "HOME": str(self.base),
                    "FAKE_LOG": str(self.log), "FAKE_SOCKET": str(self.socket)}

    def mdf(self, *args, program=None, cwd=None, **env):
        done = subprocess.run([str(program or self.top / "bin" / "mdf"), *args], cwd=cwd or self.work,
                              env={**self.env, **env}, capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=60)
        calls = self.log.read_text().splitlines() if self.log.exists() else []
        self.log.write_text("")
        self.assertNotIn("synthetic-value", done.stdout + done.stderr + "\n".join(calls))
        return done, calls

    def run_call(self, calls):
        runs = [call for call in calls if " run --rm " in call]
        self.assertEqual(len(runs), 1, calls)
        return runs[0]


class WrapperTests(Base):
    def test_a_command_runs_in_the_cli_service_as_the_user_in_the_current_directory(self):
        done, calls = self.mdf("convert", "paper.pdf", "--out", "out dir")
        self.assertEqual((done.returncode, done.stderr), (0, ""))
        self.assertEqual(done.stdout.strip(), "ran in %s" % self.top)           # compose runs in the checkout
        prefix = "compose --progress quiet --env-file /dev/null --profile tools"
        self.assertEqual(self.run_call(calls),
                         "%s run --rm -T --no-deps --user %d:%d -v %s:%s -w %s cli mdf %s --keys-file=/run/mdf/env "
                         "convert paper.pdf --out out dir" % (prefix, UID, GID, self.work, self.work, self.work, SETTINGS))
        # the launcher was asked for first, and nothing else was started
        self.assertEqual(calls[-2], prefix + " ps --status running -q launcher")
        self.assertFalse([call for call in calls if " up " in call or call.endswith(" up")])
        self.assertEqual(self.env_file.stat().st_mode & 0o777, 0o600)

    def test_the_exit_code_is_the_one_of_mdf(self):
        for code in (0, 1, 2, 130):
            with self.subTest(code=code):
                done, _ = self.mdf("convert", "a.pdf", "--out", "o", FAKE_EXIT=str(code))
                self.assertEqual(done.returncode, code)

    def test_without_the_launcher_nothing_runs_and_nothing_is_started(self):
        done, calls = self.mdf("convert", "paper.pdf", "--out", "o", FAKE_NO_LAUNCHER="1")
        self.assertEqual(done.returncode, 1)
        self.assertEqual(json.loads(done.stderr), {"error": "the product is not running: run make up",
                                                   "code": "isolation_unavailable"})
        self.assertEqual(done.stdout, "")
        self.assertFalse([call for call in calls if " run " in call or " up" in call or " start" in call])

    def test_every_command_but_the_two_of_the_archive_is_refused_without_the_launcher(self):
        for command in (("convert", "a.pdf", "--out", "o"), ("convert", "a.pdf", "--out", "o", "--route", "datalab"),
                        ("submit", "--wait", "a.pdf"), ("retry", "ID"), ("routes",), ("jobs",), ("show", "ID"),
                        ("presets", "list"), ("gpu", "session"), ("prepare", "--weights"),
                        ("bench", "run", "--level", "quick"), ("worker", "--once"), ("serve",),
                        ("settings-env", "--out", "x"), ("configured",), ("downgrade",)):
            with self.subTest(command=" ".join(command)):
                done, calls = self.mdf(*command, FAKE_NO_LAUNCHER="1")
                self.assertEqual((done.returncode, json.loads(done.stderr)["code"]), (1, "isolation_unavailable"))
                self.assertFalse([call for call in calls if " run " in call or " up" in call or " start" in call])

    def test_the_two_commands_of_the_archive_run_with_the_product_stopped(self):
        """SOL-A2-B2: the archive refuses them while a service holds it, so the only moment they can run is after
        `make down`, when no launcher is there to ask for. They reach the one-off container as any command does,
        with the settings of the last start; nothing else is started."""
        identity = "0f" * 32
        for command in (("configure", "--expected-config", identity, "--storage-mib", "30000"),
                        ("downgrade-storage", "--expected-config", identity, "--to-schema", "4")):
            with self.subTest(command=command[0]):
                done, calls = self.mdf(*command, FAKE_NO_LAUNCHER="1")
                self.assertEqual((done.returncode, done.stderr), (0, ""))
                prefix = "compose --progress quiet --env-file /dev/null --profile tools"
                self.assertEqual(self.run_call(calls),
                                 "%s run --rm -T --no-deps --user %d:%d -v %s:%s -w %s cli mdf %s --keys-file=/run/mdf/env %s"
                                 % (prefix, UID, GID, self.work, self.work, self.work, SETTINGS, " ".join(command)))
                self.assertFalse([call for call in calls if " ps " in call or " up" in call or " start" in call])
                # and with the launcher running they are not stopped by the wrapper either: the archive answers
                done, calls = self.mdf(*command)
                self.assertEqual(done.returncode, 0)
                self.run_call(calls)

    def test_help_and_version_need_no_launcher(self):
        for argument in ("--help", "--version", "-h"):
            with self.subTest(argument=argument):
                done, calls = self.mdf(argument, FAKE_NO_LAUNCHER="1")
                self.assertEqual(done.returncode, 0)
                self.assertTrue(self.run_call(calls).endswith(" cli mdf %s --keys-file=/run/mdf/env %s" % (SETTINGS, argument)))
                self.assertFalse([call for call in calls if " ps " in call])

    def test_prepare_and_bench_have_their_own_command_line(self):
        for command in (("prepare", "--weights"), ("bench", "run", "--level", "quick")):
            with self.subTest(command=command[0]):
                _, calls = self.mdf(*command)
                self.assertTrue(self.run_call(calls).endswith(" cli mdf " + " ".join(command)))

    def test_an_env_file_that_is_not_safe_is_refused_before_any_mount(self):
        real = self.base / "real.env"
        self.env_file.rename(real)
        cases = {"link": lambda: self.env_file.symlink_to(real), "type": lambda: self.env_file.mkdir()}
        for name, make in cases.items():
            with self.subTest(state=name):
                make()
                done, calls = self.mdf("convert", "a.pdf", "--out", "o")
                self.assertEqual(done.returncode, 1)
                self.assertEqual(json.loads(done.stderr)["code"], "server_key_unsafe")
                self.assertFalse([call for call in calls if " run " in call or " -v " in call])
                if self.env_file.is_symlink():
                    self.env_file.unlink()
                else:
                    self.env_file.rmdir()
        done, calls = self.mdf("convert", "a.pdf", "--out", "o")              # no .env at all
        self.assertEqual((done.returncode, json.loads(done.stderr)["code"]), (1, "invalid_setting"))
        self.assertFalse([call for call in calls if " run " in call])

    def test_another_mode_is_mounted_as_it_is_and_never_changed(self):
        self.env_file.chmod(0o644)
        done, calls = self.mdf("routes")
        self.assertEqual(done.returncode, 0)
        self.run_call(calls)                                                    # the CLI answers with the code of the route
        self.assertEqual(self.env_file.stat().st_mode & 0o777, 0o644)

    def test_the_keys_file_of_env_file(self):
        other = self.work / "other.env"
        other.write_text("B_SERVICE_KEY=synthetic-value\n")
        other.chmod(0o600)
        for given in (["--env-file", "other.env"], ["--env-file=%s" % other]):
            with self.subTest(given=given[0]):
                _, calls = self.mdf(*given, "convert", "a.pdf", "--out", "o")
                call = self.run_call(calls)
                self.assertIn(" -v %s:/run/mdf/keys:ro cli mdf %s --keys-file=/run/mdf/keys convert a.pdf" % (other, SETTINGS), call)
        # a file that is not there is not mounted: the CLI gets a path that does not exist
        _, calls = self.mdf("--env-file", "absent.env", "convert", "a.pdf", "--out", "o")
        call = self.run_call(calls)
        self.assertIn(" cli mdf %s --keys-file=/run/mdf/absent convert a.pdf" % SETTINGS, call)
        self.assertNotIn("absent.env", call)
        self.assertFalse((self.work / "absent.env").exists())
        link = self.work / "link.env"
        link.symlink_to(other)
        done, calls = self.mdf("--env-file", "link.env", "convert", "a.pdf", "--out", "o")
        self.assertEqual((done.returncode, json.loads(done.stderr)["code"]), (1, "server_key_unsafe"))
        self.assertFalse([call for call in calls if " run " in call])
        done, _ = self.mdf("--env-file")
        self.assertEqual(done.returncode, 2)

    def test_without_docker_127(self):
        empty = self.base / "empty"
        empty.mkdir()
        for tool in ("sh", "dirname", "readlink", "pwd"):
            found = shutil.which(tool)
            if found:
                (empty / tool).symlink_to(found)
        done, _ = self.mdf("routes", PATH=str(empty))
        self.assertEqual(done.returncode, 127)
        self.assertIn("docker not found", done.stderr)

    def test_through_a_link_the_checkout_is_still_found(self):
        elsewhere = self.base / "links"
        elsewhere.mkdir()
        (elsewhere / "mdf").symlink_to(self.top / "bin" / "mdf")
        done, calls = self.mdf("routes", program=elsewhere / "mdf")
        self.assertEqual((done.returncode, done.stdout.strip()), (0, "ran in %s" % self.top))
        self.run_call(calls)


class InstallTests(Base):
    def install(self, checkout=None, target=None):
        target = target or self.base / "home bin"
        done = subprocess.run(["sh", "-euc", '. "$1/scripts/env-preflight.sh"; install_cli "$2" "$3"', "sh",
                               str(ROOT), str(checkout or self.top), str(target)],
                              env=self.env, capture_output=True, text=True, timeout=60)
        return done, target / "mdf"

    def test_the_installed_file_runs_the_cli_of_its_checkout(self):
        done, installed = self.install()
        self.assertEqual(done.returncode, 0, done.stderr)
        lines = installed.read_text().splitlines()
        self.assertEqual(lines, ["#!/bin/sh",
                                 "[ -x '%s/bin/mdf' ] || { echo \"checkout not found: run make install-cli again\" >&2; exit 127; }"
                                 % self.top,
                                 "exec '%s/bin/mdf' \"$@\"" % self.top])
        self.assertEqual(installed.stat().st_mode & 0o777, 0o755)
        self.assertEqual(sorted(p.name for p in installed.parent.iterdir()), ["mdf"])
        ran, calls = self.mdf("convert", "my paper.pdf", "--out", "o", program=installed)
        self.assertEqual(ran.returncode, 0)
        self.assertTrue(self.run_call(calls).endswith(" convert my paper.pdf --out o"))
        self.assertIn("is not on your PATH", done.stdout)                       # said, not fixed

    def test_a_checkout_that_was_removed_or_moved_is_said_with_127(self):
        """E-2: the second line of the installed file. Without it the shell would answer only "not found"."""
        _, installed = self.install()
        moved = self.base / "moved"
        self.top.rename(moved)
        ran, calls = self.mdf("routes", program=installed)
        self.assertEqual((ran.returncode, ran.stderr.strip(), ran.stdout), (127, "checkout not found: run make install-cli again", ""))
        self.assertEqual(calls, [])
        again, installed = self.install(checkout=moved)                         # installed again from the new place
        self.assertEqual(again.returncode, 0)
        ran, calls = self.mdf("routes", program=installed)
        self.assertEqual((ran.returncode, ran.stdout.strip()), (0, "ran in %s" % moved))
        shutil.rmtree(moved)                                                    # the checkout is gone
        ran, _ = self.mdf("routes", program=installed)
        self.assertEqual((ran.returncode, ran.stderr.strip()), (127, "checkout not found: run make install-cli again"))

    def test_a_path_that_cannot_be_quoted_is_refused_and_nothing_is_written(self):
        for name in ("it's here", "two\nlines"):
            with self.subTest(name=name):
                odd = self.base / name
                shutil.copytree(self.top, odd)
                done, installed = self.install(checkout=odd, target=self.base / ("bin-" + name[:3]))
                self.assertEqual(done.returncode, 1)
                self.assertFalse(installed.exists())
        done, installed = self.install(checkout=self.base / "absent")
        self.assertEqual(done.returncode, 1)
        self.assertFalse(installed.exists())

    def test_the_recipe_installs_in_bin_of_the_home_by_default_and_wherever_asked(self):
        text = (ROOT / "Makefile").read_text()
        self.assertRegex(text, r"(?m)^BIN \?= \$\(HOME\)/\.local/bin$")
        self.assertIn('install_cli "$(CURDIR)" "$(BIN)"', text)
        self.assertRegex(text, r"(?m)^install-cli:.*## \S")
        self.assertNotRegex(text, r"(?m)^cli:")                                 # there is no `make cli`


if __name__ == "__main__":
    unittest.main()
