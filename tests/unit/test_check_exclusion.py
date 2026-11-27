"""`scripts/check.sh`, the host side of `make check` and `make test`, against a `docker` that is not there
(MFL051, plan r2.1 App. I.3; SOL-R2-N3; SOL-A2-B1).

One full check runs at a time on a Docker daemon: its container has the fixed name `mdf-full-check`, and Docker
admits one container of a name, decided when it is created. So:
- the container of the check is CREATED before the launcher of the check is started, in a compose project named
  after the invocation;
- an invocation that does not get the name ends nonzero, removes only its own project, and never removes
  `mdf-full-check`, which belongs to the check that is running;
- `make test` has no fixed name: its project and the tags of its images carry the identifier of the invocation;
- the full check is green only when the chain in the container AND the tests of the components are (E-6);
- the tests of the components are part of the SAME gate (SOL-A2-B1): they run from the clone of `HEAD`, never from
  the working copy, and under the reservation, which is held until they end and freed whatever the outcome (a red
  step, a signal). Before the verdict the clone must still be at the commit under test, with a clean tree.

The script runs for real, in a synthetic checkout, with `docker` and `make` replaced by recorders. The real double
start is among the evidence of the run.
"""
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import tempfile
import time
import unittest

ROOT = Path(__file__).resolve().parents[2]
GIT = ["git", "-c", "user.name=test", "-c", "user.email=test@example.invalid", "-c", "init.defaultBranch=main",
       "-c", "commit.gpgsign=false"]
# A `docker` that records its arguments, one invocation per line, and answers what the script reads. It keeps the
# one fact of Docker the exclusion rests on: a container that exists keeps its name. `$FAKE_STATE/reserved` is that
# container, with the project it belongs to: made by the `up --no-start` of a check, refused while it is there,
# removed only by the `down` of the project that made it.
DOCKER = r'''#!/bin/sh
printf '%s\n' "$*" >> "$FAKE_LOG"
project=
if [ "${1:-}" = compose ] && [ "${2:-}" = -p ]; then project=$3; fi
case "$*" in
  "version --format {{.Server.Arch}}") echo amd64 ;;
  "version --format"*) echo "docker 0.0.test api 1.45 linux/amd64" ;;
  "context inspect"*) echo "unix://$FAKE_SOCKET" ;;
  "volume inspect"*) [ "${FAKE_NO_VOLUME:-}" = 1 ] && exit 1; echo "[]" ;;
  "image inspect"*) echo "mdf-app:x sha256:0000" ;;
  *" config --format json") echo "{}" ;;
  *" up --no-start --no-deps check")
    if [ "${FAKE_NAME_TAKEN:-}" = 1 ] || [ -e "$FAKE_STATE/reserved" ]; then
      echo 'Error response from daemon: Conflict. The container name "/mdf-full-check" is already in use by container "abc".' >&2
      exit 1
    fi
    printf '%s\n' "$project" > "$FAKE_STATE/reserved" ;;
  *" down -v --remove-orphans")
    if [ "$(cat "$FAKE_STATE/reserved" 2> /dev/null)" = "$project" ]; then rm -f "$FAKE_STATE/reserved"; fi ;;
  *" ps -a -q check"|*" ps -a -q test") echo "container-of-the-invocation" ;;
  "container inspect --format {{.Name}} {{.State.Status}} container-of-the-invocation")
    [ -e "$FAKE_STATE/reserved" ] || exit 1
    echo "/mdf-full-check exited" ;;
  "start --attach container-of-the-invocation")
    # the working copy changes while the chain runs on the clone
    [ -z "${FAKE_EDIT:-}" ] || echo "edited while the check runs" > "$FAKE_EDIT"
    echo "FULL_CHECK_PASS"; exit "${FAKE_CHECK_EXIT:-0}" ;;
esac
exit 0
'''
# A `make` that records its arguments and, of the step of the components, the directory it runs in, the tracked
# file it finds there and whether the name of the check is taken at that moment.
MAKE = r'''#!/bin/sh
printf 'make %s\n' "$*" >> "$FAKE_LOG"
if [ "${1:-}" = -C ]; then cd "$2" || exit 9; fi
printf 'cwd=%s\ttracked=%s\treservation=%s\n' "$(pwd -P)" "$(cat tracked.txt 2> /dev/null)" \
  "$([ -e "$FAKE_STATE/reserved" ] && echo held || echo free)" >> "$FAKE_STATE/components"
if [ "${FAKE_COMPONENTS_HOLD:-}" = 1 ]; then
  : > "$FAKE_STATE/components-started"
  while [ ! -e "$FAKE_STATE/release" ]; do sleep 0.05; done
fi
case "${FAKE_COMPONENTS_CLONE:-}" in
  dirty) echo "left behind" > left-behind.txt ;;
  moved) git -c user.name=test -c user.email=test@example.invalid -c commit.gpgsign=false \
           commit --quiet --allow-empty -m moved ;;
  evidence) mkdir -p .cache/test-components/mdf-test-0000000-1
            echo "73 passed" > .cache/test-components/mdf-test-0000000-1/pytest.out ;;
esac
exit "${FAKE_COMPONENTS_EXIT:-0}"
'''
COMMITTED = "the committed bytes\n"


class CheckScript(unittest.TestCase):
    def setUp(self):
        for tool in ("git", "sh"):
            if shutil.which(tool) is None:
                self.fail(tool + " is needed: the image `dev` has it")
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.base = Path(os.path.realpath(tmp.name))
        self.top = self.base / "Checkout-Of.The_Tests"          # its compose project: checkout-ofthe_tests
        (self.top / "scripts").mkdir(parents=True)
        (self.top / "docker" / "dev").mkdir(parents=True)
        (self.top / "docker" / "app").mkdir()
        shutil.copy(ROOT / "scripts" / "check.sh", self.top / "scripts" / "check.sh")
        shutil.copy(ROOT / "scripts" / "env-preflight.sh", self.top / "scripts" / "env-preflight.sh")
        for name in ("compose.yaml", "compose.public.yaml"):
            (self.top / name).write_text("services: {}\n")
        (self.top / "docker" / "dev" / "compose.yaml").write_text("name: synthetic\n")
        (self.top / "docker" / "app" / "Dockerfile").write_text("FROM scratch\n")
        (self.top / ".gitignore").write_text(".cache/\n")
        (self.top / "tracked.txt").write_text(COMMITTED)
        self.git("init", "--quiet")
        self.git("add", "-A")
        self.git("commit", "--quiet", "-m", "synthetic")
        self.git("branch", "--quiet", "dev")
        self.bin = self.base / "bin"
        self.bin.mkdir()
        for name, body in (("docker", DOCKER), ("make", MAKE)):
            (self.bin / name).write_text(body)
            (self.bin / name).chmod(0o755)
        self.log = self.base / "docker.log"
        self.state = self.base / "state"                        # what the two recorders share: the name of the check
        self.state.mkdir()
        self.socket = self.base / "docker.sock"
        import socket
        server = socket.socket(socket.AF_UNIX)                  # a socket, as the preflight of the product asks
        server.bind(str(self.socket))
        self.addCleanup(server.close)
        self.env = {"PATH": "%s:%s" % (self.bin, os.environ.get("PATH", "/usr/bin:/bin")), "HOME": str(self.base),
                    "FAKE_LOG": str(self.log), "FAKE_SOCKET": str(self.socket), "FAKE_STATE": str(self.state),
                    "GIT_CONFIG_GLOBAL": "/dev/null",
                    "GIT_AUTHOR_NAME": "test", "GIT_AUTHOR_EMAIL": "test@example.invalid",
                    "GIT_COMMITTER_NAME": "test", "GIT_COMMITTER_EMAIL": "test@example.invalid"}

    def git(self, *args):
        subprocess.run([*GIT, *args], cwd=self.top, check=True, capture_output=True, timeout=60)

    def run_script(self, *args, **env):
        done = subprocess.run(["sh", "scripts/check.sh", *args], cwd=self.top, env={**self.env, **env}, capture_output=True,
                              text=True, timeout=120)
        calls = self.log.read_text().splitlines() if self.log.exists() else []
        self.log.write_text("")
        return done, calls

    def components(self):
        """What the recorder of `make` saw at each step of the components, then forgotten."""
        path = self.state / "components"
        seen = [dict(item.split("=", 1) for item in line.split("\t")) for line in path.read_text().splitlines()] \
            if path.exists() else []
        path.unlink(missing_ok=True)
        return seen

    def clone(self):
        (evidence,) = (self.top / ".cache" / "full-check").iterdir()
        return evidence / "src"

    @staticmethod
    def at(calls, pattern):
        found = [i for i, call in enumerate(calls) if re.search(pattern, call)]
        return found[0] if found else None

    def project(self, calls, mode):
        (project,) = {m.group(1) for call in calls for m in [re.search(r"compose -p (\S+)", call)] if m}
        self.assertRegex(project, r"^mdf-%s-[0-9a-f]{7,12}-[0-9]{8}t[0-9]{6}z-[0-9]+$" % mode)
        self.assertEqual(project, project.lower())              # a compose project name
        return project

    def test_the_container_of_the_check_is_created_before_its_launcher_and_its_volumes_are_used(self):
        done, calls = self.run_script("check")
        self.assertEqual(done.returncode, 0, done.stderr)
        project = self.project(calls, "check")
        reserve = self.at(calls, r"up --no-start --no-deps check$")
        launcher = self.at(calls, r"up -d launcher$")
        started = self.at(calls, r"^start --attach container-of-the-invocation$")
        builds = [i for i, call in enumerate(calls) if call.startswith("buildx build")]
        self.assertIsNotNone(reserve)
        self.assertEqual(len(builds), 2)
        self.assertLess(max(builds), reserve)                   # building touches no container
        self.assertLess(reserve, launcher)
        self.assertLess(launcher, started)
        self.assertFalse([c for c in calls[:reserve] if re.search(r" up |^run |^start |^create ", c)])
        # what it made is removed: its project and its two tags, by their identifier
        ident = project.removeprefix("mdf-check-")
        down = self.at(calls, r"compose -p %s .* down -v --remove-orphans$" % re.escape(project))
        self.assertIsNotNone(down)
        self.assertIn("image rm mdf-app:engine-%s mdf-app:dev-%s" % (ident, ident), calls)
        # E-6: the tests of the components are a step of the same gate, after the chain and before the project of
        # the check goes (SOL-A2-B1)
        components = self.at(calls, r"^make ")
        self.assertEqual(calls[components], "make -C %s --no-print-directory test-components" % self.clone())
        self.assertLess(started, components)
        self.assertLess(components, down)
        self.assertEqual(len([c for c in calls if c.endswith(" down -v --remove-orphans")]), 1)
        self.assertIn("CHECK_PASS", done.stdout)

    def test_the_chain_runs_on_a_clone_of_head_with_the_evidence_inside_it(self):
        done, calls = self.run_script("check")
        self.assertEqual(done.returncode, 0, done.stderr)
        (evidence,) = (self.top / ".cache" / "full-check").iterdir()
        clone = evidence / "src"
        self.assertTrue((clone / ".git").is_dir())                         # a real git directory, of its own
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=self.top, capture_output=True, text=True).stdout
        self.assertEqual(subprocess.run(["git", "rev-parse", "HEAD"], cwd=clone, capture_output=True, text=True).stdout, head)
        self.assertEqual(subprocess.run(["git", "rev-parse", "dev"], cwd=clone, capture_output=True, text=True).stdout, head)
        self.assertTrue((clone / ".cache" / "full-check" / "proof").is_dir())          # the mount point, made by the user
        self.assertEqual(os.stat(clone / ".cache" / "full-check" / "proof").st_uid, os.getuid())
        self.assertEqual(sorted(p.name for p in (evidence / "proof").iterdir()),
                         ["compose-dev-check.json", "compose-dev-faults.json", "compose-dev-suite.json",
                          "compose-dev-test.json", "compose-gpu.json", "compose-product.json", "compose-public.json",
                          "compose-server.json", "images.txt"])
        # the configuration of the product is asked of the files of the clone, and compose never opens an env file
        product = [call for call in calls if "--profile tools config --format json" in call]
        self.assertEqual(product, ["compose --env-file /dev/null --profile tools config --format json",
                                   "compose --env-file /dev/null -f compose.yaml -f compose.public.yaml --profile tools "
                                   "config --format json"])
        # MFL051 B: the profile gpu and the remote server, read and never started; the server as a project of its own
        self.assertIn("compose --env-file /dev/null --profile tools --profile gpu config --format json", calls)
        server = [call for call in calls if "compose.server.yaml" in call]
        self.assertEqual(len(server), 1)
        self.assertRegex(server[0], r"^compose --env-file /dev/null -p \S+-server -f compose.server.yaml --profile prepare "
                                    r"config --format json$")
        # the images of the evidence carry the tags of THIS invocation (an image can have the tags of another check too)
        ident = evidence.name
        images = (evidence / "proof" / "images.txt").read_text().splitlines()[1:]
        self.assertEqual([line.split()[0] for line in images], ["mdf-app:engine-" + ident, "mdf-app:dev-" + ident])
        build = calls[self.at(calls, r"^buildx build")]
        self.assertIn(str(clone / "docker" / "app" / "Dockerfile"), build)   # built from the clone, never the working copy
        self.assertTrue(build.endswith(" " + str(clone)))
        self.assertIn("--builder mdf-bench-toolchain", build)               # the capped builder

    def test_a_name_that_is_taken_stops_there_and_removes_only_its_own_project(self):
        done, calls = self.run_script("check", FAKE_NAME_TAKEN="1")
        self.assertEqual(done.returncode, 1)
        self.assertIn("another full check is running", done.stderr)
        self.assertIn("docker ps -a --filter name=mdf-full-check", done.stderr)      # how to look at one left behind
        project = self.project(calls, "check")
        reserve = self.at(calls, r"up --no-start --no-deps check$")
        after = calls[reserve + 1:]
        self.assertIsNone(self.at(after, r"up -d launcher$"))               # nothing of the check was started
        self.assertIsNone(self.at(after, r"^start "))
        self.assertIsNotNone(self.at(after, r"compose -p %s .* down -v --remove-orphans$" % re.escape(project)))
        # no command names the container of the other check, and none removes a container but by this project's label
        self.assertFalse([c for c in calls if "mdf-full-check" in c])
        self.assertFalse([c for c in after if re.match(r"(rm|container rm|stop|kill)\b", c)])
        for call in after:
            if call.startswith("ps "):
                self.assertIn("label=mdf.project=" + project, call)
        self.assertFalse([c for c in calls if c.startswith("make")])        # and the gate is not run in part
        self.assertNotRegex(done.stdout, r"(?m)^CHECK_PASS")

    def test_the_gate_is_red_when_the_chain_or_the_components_are(self):
        for env, text in (({"FAKE_CHECK_EXIT": "1"}, "full check exit 1, component tests exit 0"),
                          ({"FAKE_COMPONENTS_EXIT": "2"}, "full check exit 0, component tests exit 2")):
            with self.subTest(env=env):
                done, calls = self.run_script("check", **env)
                self.assertEqual(done.returncode, 1)
                self.assertIn("CHECK_FAIL", done.stdout)
                self.assertNotRegex(done.stdout, r"(?m)^CHECK_PASS")
                self.assertIn(text, done.stdout)
                self.assertEqual(len([c for c in calls if c.startswith("make ")]), 1)          # both ran
                # the name of the check is free again, also after a red step (SOL-A2-B1)
                self.assertEqual([seen["reservation"] for seen in self.components()], ["held"])
                self.assertFalse((self.state / "reserved").exists())
                shutil.rmtree(self.top / ".cache")

    def test_a_second_check_is_refused_while_the_first_is_in_the_components(self):
        """SOL-A2-B1: the reservation covers the whole gate, not only the chain in the container."""
        first_log = self.base / "first.log"
        with (self.base / "first.out").open("w") as output:
            first = subprocess.Popen(["sh", "scripts/check.sh", "check"], cwd=self.top, stdout=output,
                                     stderr=subprocess.STDOUT,
                                     env={**self.env, "FAKE_LOG": str(first_log), "FAKE_COMPONENTS_HOLD": "1"})
            try:
                until = time.monotonic() + 60
                while not (self.state / "components-started").exists():
                    self.assertIsNone(first.poll(), "the first check ended before its components")
                    self.assertLess(time.monotonic(), until, "the first check did not reach its components")
                    time.sleep(0.05)
                second, calls = self.run_script("check")
                self.assertIsNone(first.poll())                 # the first is still in its components
            finally:
                (self.state / "release").touch()
                first.wait(timeout=60)
        self.assertEqual(second.returncode, 1)
        self.assertIn("another full check is running", second.stderr)
        self.assertNotRegex(second.stdout, r"(?m)^CHECK_PASS")
        self.assertFalse([c for c in calls if c.startswith("make") or c.startswith("start ")])
        self.assertFalse([c for c in calls if "mdf-full-check" in c])       # it never names the container of the first
        # the first went on under its reservation, ended green and freed the name
        self.assertEqual(first.returncode, 0, (self.base / "first.out").read_text())
        self.assertIn("CHECK_PASS", (self.base / "first.out").read_text())
        self.assertEqual([seen["reservation"] for seen in self.components()], ["held"])
        self.assertFalse((self.state / "reserved").exists())
        shutil.rmtree(self.top / ".cache")
        done, _ = self.run_script("check")                      # and a check after it gets the name
        self.assertEqual(done.returncode, 0, done.stderr)

    def test_the_components_run_from_the_clone_and_a_change_of_the_working_copy_does_not_enter_them(self):
        """SOL-A2-B1: the gate names one commit, so both its steps run on the bytes of that commit."""
        done, calls = self.run_script("check", FAKE_EDIT=str(self.top / "tracked.txt"), FAKE_COMPONENTS_CLONE="evidence")
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertEqual((self.top / "tracked.txt").read_text(), "edited while the check runs\n")   # the copy did change
        clone = self.clone()
        (seen,) = self.components()
        self.assertEqual(seen, {"cwd": str(clone), "tracked": COMMITTED.strip(), "reservation": "held"})
        self.assertIn("make -C %s --no-print-directory test-components" % clone, calls)
        # said in the output of the gate: where the components run, and under which name
        self.assertIn("== component tests from the clone of HEAD: %s" % clone, done.stdout)
        self.assertIn("reservation held until they end: mdf-full-check exited (container container-of-the-invocation)",
                      done.stdout)
        # their evidence is in the clone, and still found where `make test-components` puts it
        link = self.top / ".cache" / "test-components" / "mdf-test-0000000-1"
        self.assertTrue(link.is_symlink())
        self.assertEqual(link.resolve(), clone / ".cache" / "test-components" / "mdf-test-0000000-1")
        self.assertEqual((link / "pytest.out").read_text(), "73 passed\n")
        self.assertIn("evidence of the component tests: %s" % link, done.stdout)

    def test_a_clone_that_is_not_the_commit_any_more_is_never_green(self):
        for how in ("dirty", "moved"):
            with self.subTest(how=how):
                done, _ = self.run_script("check", FAKE_COMPONENTS_CLONE=how)
                self.assertEqual(done.returncode, 1)
                self.assertNotRegex(done.stdout, r"(?m)^CHECK_PASS")
                self.assertIn("CHECK_FAIL", done.stdout)
                self.assertIn("the clone changed during the component tests", done.stderr)
                self.assertFalse((self.state / "reserved").exists())
                shutil.rmtree(self.top / ".cache")

    def test_a_reservation_that_was_lost_is_never_green(self):
        # someone removed the container of the check while the chain ran: the components do not run without it
        docker = self.bin / "docker"
        docker.write_text(DOCKER.replace('echo "FULL_CHECK_PASS"', 'rm -f "$FAKE_STATE/reserved"; echo "FULL_CHECK_PASS"'))
        done, calls = self.run_script("check")
        self.assertEqual(done.returncode, 1)
        self.assertIn("the reservation was lost", done.stderr)
        self.assertIn("component tests not run", done.stdout)
        self.assertNotRegex(done.stdout, r"(?m)^CHECK_PASS")
        self.assertFalse([c for c in calls if c.startswith("make")])

    def test_a_signal_during_the_components_frees_the_reservation_and_removes_the_project(self):
        """As Ctrl-C does, the signal reaches the script and the step it waits for."""
        with (self.base / "first.out").open("w") as output:
            first = subprocess.Popen(["sh", "scripts/check.sh", "check"], cwd=self.top, stdout=output,
                                     stderr=subprocess.STDOUT, start_new_session=True,
                                     env={**self.env, "FAKE_COMPONENTS_HOLD": "1"})
            try:
                until = time.monotonic() + 60
                while not (self.state / "components-started").exists():
                    self.assertIsNone(first.poll(), "the check ended before its components")
                    self.assertLess(time.monotonic(), until, "the check did not reach its components")
                    time.sleep(0.05)
                self.assertTrue((self.state / "reserved").exists())
                os.killpg(first.pid, signal.SIGTERM)
                first.wait(timeout=60)
            finally:
                (self.state / "release").touch()
                if first.poll() is None:
                    os.killpg(first.pid, signal.SIGKILL)
                    first.wait(timeout=60)
        calls = self.log.read_text().splitlines()
        self.assertEqual(first.returncode, 130)
        self.assertNotRegex((self.base / "first.out").read_text(), r"(?m)^CHECK_PASS")
        project = self.project(calls, "check")
        self.assertIsNotNone(self.at(calls, r"compose -p %s .* down -v --remove-orphans$" % re.escape(project)))
        ident = project.removeprefix("mdf-check-")
        self.assertIn("image rm mdf-app:engine-%s mdf-app:dev-%s" % (ident, ident), calls)
        self.assertFalse((self.state / "reserved").exists())

    def test_a_dirty_tree_stops_the_check_before_anything_and_not_the_test(self):
        (self.top / "untracked.txt").write_text("x")
        done, calls = self.run_script("check")
        self.assertEqual((done.returncode, done.stderr.strip()), (1, "STOP tree not clean"))
        self.assertEqual(calls, [])
        done, calls = self.run_script("test")
        self.assertEqual(done.returncode, 0, done.stderr)

    def test_make_test_has_a_project_and_tags_of_the_invocation_and_no_fixed_name(self):
        done, calls = self.run_script("test", "tests/unit/test_x.py", "-k", "one or two")
        self.assertEqual(done.returncode, 0, done.stderr)
        project = self.project(calls, "test")
        ident = project.removeprefix("mdf-test-")
        builds = [c for c in calls if c.startswith("buildx build")]
        self.assertEqual([re.search(r"-t (\S+)", b).group(1) for b in builds], ["mdf-app:engine-" + ident, "mdf-app:dev-" + ident])
        self.assertTrue(all(b.endswith(" " + str(self.top)) for b in builds))          # the working copy itself
        self.assertIsNotNone(self.at(calls, r"up --no-start --no-deps test$"))
        self.assertFalse([c for c in calls if "mdf-full-check" in c or c.startswith("make")])
        (evidence,) = (self.top / ".cache" / "full-check").iterdir()
        self.assertFalse((evidence / "src").exists())                                   # no clone
        self.assertEqual((evidence / "selection").read_text(), "tests/unit/test_x.py\n-k\none or two\n")
        # two invocations share nothing: another identifier, another project
        done, again = self.run_script("test")
        self.assertNotEqual(self.project(again, "test"), project)

    def test_the_preconditions_of_the_host(self):
        done, calls = self.run_script("check", FAKE_NO_VOLUME="1")
        self.assertEqual(done.returncode, 1)
        self.assertIn("no models volume checkout-ofthe_tests_mdf-models", done.stderr)   # the product of this checkout
        self.assertIn("make prepare", done.stderr)
        self.assertIsNone(self.at(calls, r"buildx build"))
        done, calls = self.run_script("check", MDF_COMPOSE_MODELS_VOLUME="another_volume")
        self.assertIn("volume inspect another_volume", calls)
        done, calls = self.run_script("unknown")
        self.assertEqual((done.returncode, calls), (2, []))


if __name__ == "__main__":
    unittest.main()
