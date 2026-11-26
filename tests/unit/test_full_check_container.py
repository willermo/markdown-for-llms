"""`scripts/full-check.sh`, the chain of the full check, as it runs in a container (MFL051, plan r2.1 App. I.2).

The chain itself is proved by running it (`make check`, `make test`). Here what changed around it, with the script
run only as far as its preconditions, on a copy of it in a synthetic checkout, and read for the order of its steps:
- `MDF_` and `PYTEST_` variables are still refused;
- `--working-tree` (`make test`) skips the clean tree, and nothing else;
- the evidence may be given a directory (`FULL_CHECK_PROOF_DIR`), which must be under the checkout: step S reads
  its inventory below `--repo`;
- step C, the payload gate, comes after F and before X, W and the suite; X asks the launcher;
- the cache of pytest is outside the checkout.
"""
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "full-check.sh"
GIT = ["git", "-c", "user.name=test", "-c", "user.email=test@example.invalid", "-c", "init.defaultBranch=main",
       "-c", "commit.gpgsign=false"]
VARIABLES = ("FULL_CHECK_PYTHON_INSTALL_DIR", "FULL_CHECK_DEV_PYTHON", "FULL_CHECK_UV", "FULL_CHECK_UV_CACHE_DIR",
             "FULL_CHECK_BACKEND_WHEEL", "FULL_CHECK_MODEL_ROOT")


class ChainTests(unittest.TestCase):
    def setUp(self):
        if shutil.which("git") is None or shutil.which("bash") is None:
            self.fail("git and bash are needed: the image `dev` has both")
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.top = Path(os.path.realpath(tmp.name)) / "checkout"
        (self.top / "scripts").mkdir(parents=True)
        shutil.copy(SCRIPT, self.top / "scripts" / "full-check.sh")
        (self.top / "pyproject.toml").write_text('[tool.uv]\nrequired-version = "==0.0.1"\n')
        self.git("init", "--quiet")
        self.git("add", "-A")
        self.git("commit", "--quiet", "-m", "synthetic")
        self.base = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": tmp.name, "GIT_OPTIONAL_LOCKS": "0"}

    def git(self, *args):
        subprocess.run([*GIT, *args], cwd=self.top, check=True, capture_output=True, timeout=60)

    def run_script(self, *args, env=None):
        done = subprocess.run(["bash", "scripts/full-check.sh", *args], cwd=self.top, env={**self.base, **(env or {})},
                              capture_output=True, text=True, timeout=60)
        return done.returncode, done.stderr.strip().splitlines()[-1] if done.stderr.strip() else done.stdout.strip()

    def paths(self):
        """The six paths, valid as far as the preconditions look: a uv that reports the version of the checkout."""
        tools = self.top.parent / "tools"
        (tools / "python" / "cpython-3.12.13-linux-x86_64-gnu" / "bin").mkdir(parents=True)
        (tools / "cache").mkdir()
        (tools / "models").mkdir()
        (tools / "venv" / "bin").mkdir(parents=True)
        for name, body in (("uv", "#!/bin/sh\necho 'uv 0.0.1'\n"), ("venv/bin/python", "#!/bin/sh\nexit 0\n"),
                           ("python/cpython-3.12.13-linux-x86_64-gnu/bin/python3.12", "#!/bin/sh\nexit 0\n")):
            (tools / name).write_text(body)
            (tools / name).chmod(0o755)
        (tools / "backend.whl").write_text("")
        return dict(zip(VARIABLES, (str(tools / "python"), str(tools / "venv/bin/python"), str(tools / "uv"),
                                    str(tools / "cache"), str(tools / "backend.whl"), str(tools / "models"))))

    def test_mdf_and_pytest_variables_are_still_refused(self):
        for name in ("MDF_DATA_ROOT", "MDF_COMPOSE_UID", "PYTEST_ADDOPTS"):
            for args in ((), ("--working-tree",)):
                with self.subTest(name=name, args=args):
                    status, last = self.run_script(*args, env={name: "x"})
                    self.assertEqual(status, 1)
                    self.assertIn("variables in the environment change the suites: " + name, last)

    def test_a_dirty_tree_stops_the_full_check_and_not_the_test_of_the_working_tree(self):
        (self.top / "untracked.txt").write_text("x")
        self.assertEqual(self.run_script(), (1, "STOP tree not clean"))
        status, last = self.run_script("--working-tree")
        self.assertEqual((status, last), (1, "STOP FULL_CHECK_PYTHON_INSTALL_DIR is not set (environment or .full-check.env)"))
        status, last = self.run_script("--working-tree", "tests/unit/test_x.py", "-k", "one")
        self.assertIn("FULL_CHECK_PYTHON_INSTALL_DIR is not set", last)          # a selection is accepted after it

    def test_any_other_argument_is_a_usage_error(self):
        for args in (("tests/",), ("--working", ), ("--help",)):
            with self.subTest(args=args):
                status, last = self.run_script(*args)
                self.assertEqual(status, 1)
                self.assertIn("usage: scripts/full-check.sh [--working-tree [PYTEST SELECTION...]]", last)

    def test_the_evidence_directory_must_be_under_the_checkout(self):
        """CAR2-N2. Outside the checkout step S could not read its inventory: refused before anything is written."""
        paths = self.paths()
        outside = self.top.parent / "proof"
        outside.mkdir()
        for given in (str(outside), "/tmp", str(self.top) + "-other/proof", str(self.top.parent)):
            with self.subTest(given=given):
                status, last = self.run_script(env={**paths, "FULL_CHECK_PROOF_DIR": given})
                self.assertEqual((status, last), (1, "STOP FULL_CHECK_PROOF_DIR is not under the checkout"))
        self.assertEqual(list(outside.iterdir()), [])
        # under the checkout it is accepted as far as being looked for: this one is not there
        inside = self.top / ".cache" / "full-check" / "proof"
        status, last = self.run_script(env={**paths, "FULL_CHECK_PROOF_DIR": str(inside)})
        self.assertEqual((status, last), (1, "STOP FULL_CHECK_PROOF_DIR is not a directory"))

    def test_the_environment_prevails_over_the_file_of_the_developer(self):
        """In the container all six variables come from the environment: a `.full-check.env` of the developer,
        with its paths of the host, is read and changes nothing."""
        paths = self.paths()
        (self.top / ".full-check.env").write_text("".join("%s=/host/of/the/developer\n" % name for name in VARIABLES))
        self.git("add", "-A")
        self.git("commit", "--quiet", "-m", "env")
        status, last = self.run_script(env={**paths, "FULL_CHECK_PROOF_DIR": "/outside"})
        self.assertEqual((status, last), (1, "STOP FULL_CHECK_PROOF_DIR is not under the checkout"))     # the paths passed
        status, last = self.run_script(env={"FULL_CHECK_PROOF_DIR": "/outside"})
        self.assertIn("is not a directory", last)                              # without them, the file's paths are used


class OrderTests(unittest.TestCase):
    def setUp(self):
        self.text = SCRIPT.read_text(encoding="utf-8")

    def at(self, needle):
        self.assertEqual(self.text.count(needle), 1, needle)
        return self.text.index(needle)

    def test_the_payload_gate_is_a_step_after_f_and_before_x_w_and_the_suite(self):
        order = [self.at("# --- F: preflight"), self.at("# --- C: payload gate"),
                 self.at("-m markdown_for_llms.engine.containers gate --repo \"$top\" --wheel \"$WHEEL\""),
                 self.at("# --- X: toolchain image"), self.at("# --- W: weights"), self.at("run_suite fast-suite")]
        self.assertEqual(order, sorted(order))
        self.assertIn('|| stop "C: $(tail -n 1 "$PROOF/C.out")"', self.text)

    def test_x_asks_the_launcher_and_no_step_drives_docker(self):
        self.assertIn('-m markdown_for_llms.engine.toolchain verify > "$PROOF/X.out"', self.text)
        code = "\n".join(line for line in self.text.splitlines() if not line.lstrip().startswith("#"))
        self.assertNotRegex(code, r"\bdocker\b")
        self.assertNotIn("docker.sock", self.text)

    def test_the_cache_of_pytest_is_outside_the_checkout(self):
        (line,) = [l for l in self.text.splitlines() if "-m pytest" in l]
        self.assertIn('-o cache_dir="${TMPDIR:-/tmp}/mdf-pytest-cache"', line)
        self.assertNotIn("$top", re.search(r"cache_dir=\S+", line).group(0))

    def test_the_guard_of_the_bench_and_engine_tests_judges_the_whole_suite_only(self):
        self.assertIn('if [ "${selection[*]}" = "tests/" ]; then', self.text)
        self.assertIn('echo "bench guard: not run (a selection)"', self.text)
        self.assertIn("pass=TEST_PASS failed=TEST_FAIL; else pass=FULL_CHECK_PASS failed=FULL_CHECK_FAIL", self.text)


if __name__ == "__main__":
    unittest.main()
