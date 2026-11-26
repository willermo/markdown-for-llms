"""The build context of the images of the product and the pins of their Dockerfile (MFL051).

`.dockerignore` is a positive list: `.env`, `temp/`, `.cache/`, `data/`, `state/`, the virtual environments, git
and the tests never enter the context of `docker/app/Dockerfile`; what enters is exactly what the images are built
from. The patterns are read here with the rules of Docker (the last pattern that matches a path or one of its
parents decides; `!` brings a path back), without a daemon.

The second part keeps the values the Dockerfile and the Makefile repeat equal to their owners: the interpreter
and the Setuptools wheel of the guard of the full check, the version of uv of `pyproject.toml`, the base image of
the toolchain image, the builder of `toolchain.json`, the build inputs of the payload gate.
"""
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import tomllib
import unittest

ROOT = Path(__file__).resolve().parents[2]
DOCKERFILE = ROOT / "docker" / "app" / "Dockerfile"
GUARD = ROOT / "scripts" / "diagnostics" / "run-a001-fase0-uv" / "guard_product_inputs.py"
ROOT_FILES = {"pyproject.toml", "uv.lock", "build-constraints.txt", "MANIFEST.in", "README.md", "LICENSE", ".python-version"}
EXTRA_FILES = {"docker/app/Dockerfile", "docker/app/modal-requirements.txt",
               "scripts/diagnostics/run-a001-fase0-uv/guard_product_inputs.py"}
OUTSIDE = (".env", ".env.local", ".full-check.env", "temp/run/stato.md", ".cache/full-check/x/S.json", "data/archive.sqlite3",
           "state/config/servers.json", ".git/config", ".venv-dev/bin/python", ".venv-python/x", "tests/conftest.py",
           "markdown_for_llms.egg-info/PKG-INFO", "markdown_for_llms/__pycache__/api.cpython-312.pyc",
           "markdown_for_llms/engine/__pycache__/x.pyc", "markdown_for_llms/engine/stale.pyc", "docker/dev/compose.yaml",
           "docker/nginx/local.conf.template", "docker/app/init-volumes.sh", "scripts/full-check.sh",
           "scripts/diagnostics/run-a001-fase0-uv/check_preflight.py", "Makefile", "AGENTS.md", "CLAUDE.local.md",
           "build/lib/markdown_for_llms/api.py", "documentation/roadmap.md", "compose.yaml",
           # below a path that comes back: a file of settings or keys copied there by mistake stays outside all the same
           "markdown_for_llms/.env", "markdown_for_llms/engine/.env.local")


def _regex(pattern):
    out, at = "", 0
    while at < len(pattern):
        c = pattern[at]
        if pattern.startswith("**/", at):
            out, at = out + "(?:.*/)?", at + 3
            continue
        if pattern.startswith("**", at):
            out, at = out + ".*", at + 2
            continue
        if c == "*":
            out += "[^/]*"
        elif c == "?":
            out += "[^/]"
        elif c == "[":
            end = pattern.index("]", at)
            out += pattern[at:end + 1]
            at = end
        else:
            out += re.escape(c)
        at += 1
    return re.compile(out)


def patterns(text):
    rules = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        negated = line.startswith("!")
        rules.append((negated, _regex(line[1:] if negated else line).fullmatch))
    return rules


def in_context(path, rules):
    """True when Docker sends `path` (relative, `/`-separated): the last rule that matches it or a parent decides."""
    parts = path.split("/")
    prefixes = ["/".join(parts[:n]) for n in range(1, len(parts) + 1)]
    excluded = False
    for negated, match in rules:
        if any(match(p) for p in prefixes):
            excluded = not negated
    return not excluded


class TestMatcher(unittest.TestCase):
    def test_the_rules_of_docker(self):
        rules = patterns("*\n!keep\n!a\na/*\n!a/b\n**/__pycache__\n**/*.py[cod]\n# comment\n")
        self.assertTrue(in_context("keep/x/y.txt", rules))
        self.assertFalse(in_context("other/x", rules))
        self.assertFalse(in_context("a/c/file", rules))
        self.assertTrue(in_context("a/b/file", rules))
        self.assertFalse(in_context("keep/__pycache__/x", rules))
        self.assertFalse(in_context("keep/deep/er/m.pyc", rules))
        self.assertTrue(in_context("keep/m.py", rules))


class TestDockerignore(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = (ROOT / ".dockerignore").read_text(encoding="utf-8")
        cls.rules = patterns(cls.text)

    def test_everything_is_left_out_first(self):
        first = next(line.strip() for line in self.text.splitlines() if line.strip() and not line.startswith("#"))
        self.assertEqual(first, "*")

    def test_secrets_data_and_run_material_stay_outside(self):
        for path in OUTSIDE:
            self.assertFalse(in_context(path, self.rules), path)

    def test_the_exclusion_of_the_settings_files_comes_after_every_line_that_brings_a_path_back(self):
        lines = [line.strip() for line in self.text.splitlines() if line.strip() and not line.startswith("#")]
        self.assertIn("**/.env*", lines)
        self.assertGreater(lines.index("**/.env*"), max(n for n, line in enumerate(lines) if line.startswith("!")))

    def test_the_context_is_exactly_what_the_images_are_built_from(self):
        out = subprocess.run(["git", "-C", str(ROOT), "ls-files", "-z"], check=True, capture_output=True).stdout
        tracked = [name for name in out.decode("utf-8").split("\0") if name]
        sent = {name for name in tracked if in_context(name, self.rules)}
        package = {name for name in tracked if name.startswith("markdown_for_llms/")}
        self.assertTrue(package)
        self.assertEqual(sent, package | ROOT_FILES | EXTRA_FILES)

    def test_every_file_the_dockerfile_reads_from_the_context_is_in_it(self):
        text = DOCKERFILE.read_text(encoding="utf-8")
        copied = set()
        for line in text.splitlines():
            found = re.match(r"COPY (?!--from|<<)(.+)", line)
            if found:
                copied.update(found.group(1).split()[:-1])
        copied.update(re.findall(r"--mount=type=bind,source=([^,\s]+),target=", text))
        copied.discard(".")
        self.assertIn("uv.lock", copied)
        for name in copied:
            self.assertTrue(in_context(name, self.rules), name)
            self.assertTrue((ROOT / name).exists(), name)


def _guard_source():
    return GUARD.read_text(encoding="utf-8")


class TestPins(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dockerfile = DOCKERFILE.read_text(encoding="utf-8")

    def test_the_interpreter_and_the_backend_are_those_of_the_guard(self):
        guard = _guard_source()
        interpreter = re.search(r"managed_binary\)\['sha256'\]!='([0-9a-f]{64})'", guard).group(1)
        hook = re.search(r"virtual\['sha256'\]!='([0-9a-f]{64})'", guard).group(1)
        backend = re.search(r"descriptor\(backend_wheel\)\['sha256'\]!='([0-9a-f]{64})'", guard).group(1)
        self.assertIn(interpreter + "  /opt/mdf/python/cpython-3.12.13-linux-x86_64-gnu/bin/python3.12", self.dockerfile)
        self.assertIn(hook + "  /opt/mdf/venv/lib/python3.12/site-packages/_virtualenv.py", self.dockerfile)
        self.assertEqual(self.dockerfile.count("sha256:" + backend), 2)       # the download and the build constraint
        self.assertIn("guard_product_inputs.py --python /opt/mdf/python/cpython-3.12.13-linux-x86_64-gnu/bin/python3.12 "
                      "--managed /opt/mdf/python", " ".join(self.dockerfile.replace("\\\n", " ").split()).replace("/tmp/", ""))

    def test_the_backend_wheel_is_the_one_of_the_lock(self):
        lock = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))
        wheel = next(p for p in lock["package"] if p["name"] == "setuptools")["wheels"][0]
        self.assertIn(wheel["url"], self.dockerfile)
        self.assertIn("--checksum=" + wheel["hash"], self.dockerfile)
        self.assertEqual((ROOT / "build-constraints.txt").read_text().strip(), "setuptools==84.0.0")

    def test_uv_and_python_are_those_of_the_project(self):
        project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        version = project["tool"]["uv"]["required-version"].removeprefix("==")
        self.assertEqual(len(re.findall(r"/uv/releases/download/%s/uv-(?:x86_64|aarch64)-unknown-linux-gnu\.tar\.gz" % re.escape(version),
                                        self.dockerfile)), 2)
        self.assertIn('= %s' % version, self.dockerfile)
        python = (ROOT / ".python-version").read_text().strip()
        self.assertIn("python install %s " % python, self.dockerfile)
        self.assertNotIn("python install %s --no-bin \\\n &&" % python, self.dockerfile)   # never the newest build uv knows
        self.assertEqual(len(re.findall(r'"build": "20260310"', self.dockerfile)), 2)

    def test_the_base_image_is_the_one_of_the_toolchain(self):
        ours = re.search(r"^ARG BASE=(\S+)$", self.dockerfile, re.M).group(1)
        theirs = re.search(r"^ARG BASE=(\S+)$", (ROOT / "scripts/bench-toolchain/Dockerfile").read_text(), re.M).group(1)
        self.assertEqual(ours, theirs)
        self.assertRegex(ours, r"^debian:trixie-slim@sha256:[0-9a-f]{64}$")
        syntax = self.dockerfile.splitlines()[0]
        self.assertEqual(syntax, (ROOT / "scripts/bench-toolchain/Dockerfile").read_text().splitlines()[0])

    # The Debian packages of the images (MFL051 A2): nothing outside these two lists is installed
    ENGINE_PACKAGES = ("ca-certificates", "curl",                               # `weights.prepare`, `cpuserver.prepare`
                       "libglib2.0-0t64", "libpango-1.0-0", "libpangoft2-1.0-0", "libharfbuzz0b", "libharfbuzz-subset0",
                       "libfontconfig1", "fontconfig",                          # what WeasyPrint 63.1 opens, and /etc/fonts
                       "fonts-noto-core", "fonts-dejavu-mono")                  # sans-serif and serif, monospace
    DEV_PACKAGES = ("git",)                                                     # steps O and S of the full check

    def test_the_debian_packages_are_the_two_lists_from_the_snapshot_of_the_toolchain(self):
        theirs = (ROOT / "scripts/bench-toolchain/Dockerfile").read_text()
        snapshot = re.search(r"^ARG SNAPSHOT=(\S+)$", self.dockerfile, re.M).group(1)
        self.assertEqual(snapshot, re.search(r"^ARG SNAPSHOT=(\S+)$", theirs, re.M).group(1))
        self.assertRegex(snapshot, r"^[0-9]{8}T[0-9]{6}Z$")
        lists = dict(re.findall(r'^ARG (ENGINE_PACKAGES|DEV_PACKAGES)="([^"]*)"$', self.dockerfile, re.M))
        self.assertEqual(sorted(lists["ENGINE_PACKAGES"].split()), sorted(self.ENGINE_PACKAGES))
        self.assertEqual(sorted(lists["DEV_PACKAGES"].split()), sorted(self.DEV_PACKAGES))
        # every installation names one of the two lists and nothing else, without recommended packages, and the
        # lists of apt go away in the same layer
        stages = re.split(r"^FROM ", self.dockerfile, flags=re.M)
        installs = {}
        for stage in stages:
            name = stage.split("\n", 1)[0].rsplit(" AS ", 1)[-1].strip()
            for run in re.findall(r"^RUN ((?:.*\\\n)*.*)$", stage, re.M):
                if "apt-get" in run or "apt " in run or "dpkg" in run:
                    installs[name] = " ".join(run.replace("\\\n", " ").split())
        self.assertEqual(sorted(installs), ["dev", "engine"])
        for name, variable in (("engine", "ENGINE_PACKAGES"), ("dev", "DEV_PACKAGES")):
            run = installs[name]
            self.assertEqual(re.findall(r"apt-get install (.*?) &&", run), ["-y --no-install-recommends ${%s}" % variable], name)
            self.assertEqual(run.count("apt-get"), 2, name)                    # `update` and that one `install`
            self.assertIn("&& rm -rf /var/lib/apt/lists/*", run, name)
            self.assertNotIn("dpkg", run, name)
            for suite, archive in (("trixie", "debian"), ("trixie-security", "debian-security")):
                self.assertIn('echo "deb [check-valid-until=no] http://snapshot.debian.org/archive/%s/${SNAPSHOT} %s main"'
                              % (archive, suite), run, name)
            self.assertIn("rm -f /etc/apt/sources.list.d/debian.sources", run, name)
        self.assertEqual(len(re.findall(r"snapshot\.debian\.org", self.dockerfile)), 5)     # the header and the two stages

    def test_no_unpinned_download(self):
        self.assertNotRegex(self.dockerfile, r"apt install|pip install (?!--python)")
        adds = re.findall(r"^ADD (.*)$", self.dockerfile, re.M)
        self.assertEqual(len(adds), 3)
        self.assertTrue(all(a.startswith("--checksum=sha256:") for a in adds))

    def test_the_builder_of_the_makefile_is_the_one_of_toolchain_json(self):
        spec = json.loads((ROOT / "markdown_for_llms/engine/data/toolchain.json").read_text())["builder"]
        makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
        self.assertIn("BUILDKIT := " + spec["image"], makefile)
        self.assertIn("BUILDER_MEMORY := " + spec["memory"], makefile)
        spec_module = importlib.util.spec_from_file_location("mdf_toolchain", ROOT / "markdown_for_llms/engine/toolchain.py")
        self.assertIn("BUILDER := mdf-bench-toolchain", makefile)
        self.assertIn('BUILDER = "mdf-bench-toolchain"', Path(spec_module.origin).read_text())

    def test_the_build_inputs_of_the_gate_are_in_the_context(self):
        from markdown_for_llms.engine import containers
        rules = patterns((ROOT / ".dockerignore").read_text(encoding="utf-8"))
        for name in containers.INPUTS:
            self.assertTrue((ROOT / name).is_file(), name)
            self.assertTrue(in_context(name, rules), name)
        self.assertIn("containers.inputs_sha256(\"/context\")", self.dockerfile)


if __name__ == "__main__":
    unittest.main()
