"""`uv.lock` closes on Linux for the images of the product (MFL051, docker/app/Dockerfile).

The images install the extras `app` and `engine` and the group `dev` with `uv sync --frozen` on linux/amd64 and
linux/arm64. This guard reads the lock alone, without a network and without uv: for each of the two machines it
walks the dependencies of the project with the markers of that machine and asserts that every package it reaches
has a wheel that CPython 3.12 on glibc Linux can install. The project itself is not walked into a wheel (the
images build it), the records that hold only macOS wheels (torch and torchvision, behind `sys_platform ==
'darwin'`) are never reached, and a package with a source archive only is admitted by name: the Dockerfile
builds it and checks that its wheel is pure Python.
"""
from pathlib import Path
import re
import tomllib
import unittest

ROOT = Path(__file__).resolve().parents[2]
PROJECT = "markdown-for-llms"
EXTRAS = ("app", "engine")
GROUPS = ("dev",)
MACHINES = ("x86_64", "aarch64")
SDIST_ONLY = {"ebooklib"}                     # built in the image; its wheel must be `py3-none-any`
TOKEN = re.compile(r"\s*(\(|\)|==|!=|<=|>=|~=|<|>|'[^']*'|\"[^\"]*\"|[A-Za-z_][A-Za-z0-9_.]*)")
VERSIONED = ("python_version", "python_full_version", "implementation_version")


def environment(machine):
    return {"sys_platform": "linux", "platform_machine": machine, "platform_system": "Linux", "os_name": "posix",
            "python_version": "3.12", "python_full_version": "3.12.13", "implementation_name": "cpython",
            "platform_python_implementation": "CPython", "implementation_version": "3.12.13"}


def _version(text):
    return tuple(int(p) if p.isdigit() else p for p in re.split(r"[.+]", text))


class Marker:
    """A marker of the lock, evaluated for one environment and a set of active extras (`extra == 'x'` is true
    when `x` is active, which a single value of `extra` cannot say)."""

    def __init__(self, text):
        self.tokens, at = [], 0
        while at < len(text):
            found = TOKEN.match(text, at)
            if not found:
                raise ValueError("marker: " + text)
            self.tokens.append(found.group(1))
            at = found.end()
        self.text = text

    def evaluate(self, env, extras):
        self.at, self.env, self.extras = 0, env, extras
        value = self._or()
        if self.at != len(self.tokens):
            raise ValueError("marker: " + self.text)
        return value

    def _peek(self):
        return self.tokens[self.at] if self.at < len(self.tokens) else None

    def _take(self):
        self.at += 1
        return self.tokens[self.at - 1]

    def _or(self):
        value = self._and()
        while self._peek() == "or":
            self._take()
            value = self._and() or value
        return value

    def _and(self):
        value = self._atom()
        while self._peek() == "and":
            self._take()
            value = self._atom() and value
        return value

    def _atom(self):
        if self._peek() == "(":
            self._take()
            value = self._or()
            if self._take() != ")":
                raise ValueError("marker: " + self.text)
            return value
        left, op = self._take(), self._take()
        if op == "not":
            op += " " + self._take()
        right = self._take()
        name = left if left[0] not in "'\"" else right
        quoted = (right if name is left else left)[1:-1]
        if name == "extra":
            if op not in ("==", "!="):
                raise ValueError("marker: " + self.text)
            return (quoted in self.extras) == (op == "==")
        actual = self.env[name]
        a, b = (actual, quoted) if name is left else (quoted, actual)
        if op in ("in", "not in"):
            return (a in b) == (op == "in")
        if name in VERSIONED:
            a, b = _version(a), _version(b)
        return {"==": a == b, "!=": a != b, "<": a < b, "<=": a <= b, ">": a > b, ">=": a >= b}[op]


def wheel_fits(filename, machine):
    """True when CPython 3.12 on glibc Linux for `machine` installs the wheel `filename`."""
    parts = filename[:-len(".whl")].split("-")
    if not filename.endswith(".whl") or len(parts) < 5:
        return False
    pythons, abis, platforms = parts[-3].split("."), parts[-2].split("."), parts[-1].split(".")
    for python in pythons:
        for abi in abis:
            if python in ("py3", "py2", "py312") and abi == "none":
                interpreter = True
            elif python == "cp312" and abi in ("cp312", "abi3", "none"):
                interpreter = True
            elif re.fullmatch(r"cp3\d{1,2}", python) and abi == "abi3" and int(python[3:]) <= 12:
                interpreter = True
            else:
                interpreter = False
            if interpreter and any(p == "any" or re.fullmatch(r"(manylinux[0-9_]*|manylinux\d+|linux)_" + machine, p)
                                   for p in platforms):
                return True
    return False


def load_lock(path=None):
    return tomllib.loads((path or ROOT / "uv.lock").read_text(encoding="utf-8"))


def closure(lock, machine, extras=EXTRAS, groups=GROUPS):
    """The packages `uv sync` installs for the project with `extras` and `groups` on Linux for `machine`:
    {(name, version, source): record}, the project excluded."""
    env = environment(machine)
    active = {"extra-%d-%s-%s" % (len(PROJECT), PROJECT, e) for e in extras}
    records = {}
    for package in lock["package"]:
        records.setdefault(package["name"], []).append(package)
    project = next(p for p in lock["package"] if p["name"] == PROJECT)

    def find(dep):
        candidates = [p for p in records[dep["name"]]
                      if ("version" not in dep or p["version"] == dep["version"])
                      and ("source" not in dep or p["source"] == dep["source"])]
        if len(candidates) != 1:
            raise AssertionError("ambiguous dependency %s: %d records" % (dep, len(candidates)))
        return candidates[0]

    def wanted(dep):
        return "marker" not in dep or Marker(dep["marker"]).evaluate(env, active)
    queue = list(project.get("dependencies", []))
    for extra in extras:
        queue += project["optional-dependencies"][extra]
    for group in groups:
        queue += project["dev-dependencies"][group]
    reached = {}
    while queue:
        dep = queue.pop()
        if not wanted(dep):
            continue
        record = find(dep)
        key = (record["name"], record["version"], str(record["source"]))
        fresh = key not in reached
        reached[key] = record
        todo = list(record.get("dependencies", [])) if fresh else []
        for extra in dep.get("extra", []):        # the extras of a package are walked every time they are asked
            todo += record.get("optional-dependencies", {}).get(extra, [])
        queue += todo
    reached.pop((PROJECT, project["version"], str(project["source"])), None)
    return reached


def _wheels(record):
    return [w["url"].rsplit("/", 1)[-1].replace("%2B", "+") for w in record.get("wheels", [])]


class TestMarker(unittest.TestCase):
    def test_extras_are_a_set_and_the_machine_decides(self):
        env = environment("aarch64")
        both = {"extra-17-markdown-for-llms-app", "extra-17-markdown-for-llms-engine"}
        marker = Marker("(sys_platform != 'darwin' and extra == 'extra-17-markdown-for-llms-engine') or "
                        "(extra == 'extra-17-markdown-for-llms-bench' and extra == 'extra-17-markdown-for-llms-bench-cu132')")
        self.assertTrue(marker.evaluate(env, both))
        self.assertFalse(marker.evaluate(env, {"extra-17-markdown-for-llms-app"}))
        self.assertFalse(Marker("sys_platform == 'darwin'").evaluate(env, both))
        self.assertTrue(Marker("platform_machine == 'aarch64' and sys_platform == 'linux'").evaluate(env, both))
        self.assertTrue(Marker("python_full_version < '3.13'").evaluate(env, both))
        self.assertFalse(Marker("python_full_version >= '3.13' and extra != 'x'").evaluate(env, both))
        self.assertTrue(Marker("extra != 'extra-17-markdown-for-llms-bench-cu132'").evaluate(env, both))
        with self.assertRaises(ValueError):
            Marker("sys_platform === 'linux'").evaluate(env, both)

    def test_the_prefix_of_the_extras_is_the_one_of_the_lock(self):
        text = (ROOT / "uv.lock").read_text(encoding="utf-8")
        self.assertIn("extra-%d-%s-engine" % (len(PROJECT), PROJECT), text)


class TestWheelFits(unittest.TestCase):
    def test_tags(self):
        yes = [("six-1.17.0-py2.py3-none-any.whl", "aarch64"), ("x-1-cp312-cp312-manylinux_2_28_aarch64.whl", "aarch64"),
               ("x-1-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl", "x86_64"),
               ("x-1-cp38-abi3-manylinux_2_28_x86_64.whl", "x86_64"), ("x-1+cpu-cp312-cp312-linux_x86_64.whl", "x86_64"),
               ("x-1-1build-py3-none-any.whl", "x86_64")]
        no = [("x-1-cp312-cp312-macosx_14_0_arm64.whl", "aarch64"), ("x-1-cp312-cp312-manylinux_2_28_x86_64.whl", "aarch64"),
              ("x-1-cp313-cp313-manylinux_2_28_x86_64.whl", "x86_64"), ("x-1-cp312-cp312-musllinux_1_2_x86_64.whl", "x86_64"),
              ("x-1-cp312-cp312-win_amd64.whl", "x86_64"), ("x-1-cp313-abi3-manylinux_2_28_x86_64.whl", "x86_64"),
              ("x-1.tar.gz", "x86_64")]
        for name, machine in yes:
            self.assertTrue(wheel_fits(name, machine), name)
        for name, machine in no:
            self.assertFalse(wheel_fits(name, machine), name)


class TestLockPlatforms(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lock = load_lock()

    def test_every_package_of_the_images_has_a_linux_wheel(self):
        for machine in MACHINES:
            reached = closure(self.lock, machine)
            self.assertGreater(len(reached), 100)
            missing = sorted(name for (name, _, _), record in reached.items()
                             if not any(wheel_fits(w, machine) for w in _wheels(record)) and name not in SDIST_ONLY)
            self.assertEqual(missing, [], "no wheel for linux %s" % machine)

    def test_only_the_listed_packages_come_from_a_source_archive(self):
        for machine in MACHINES:
            without = sorted(name for (name, _, _), record in closure(self.lock, machine).items() if not record.get("wheels"))
            self.assertEqual(without, sorted(SDIST_ONLY), machine)
        for name in SDIST_ONLY:
            record = next(p for p in self.lock["package"] if p["name"] == name)
            self.assertIn("sdist", record)

    def test_the_macos_records_and_the_project_are_never_installed(self):
        for machine in MACHINES:
            reached = closure(self.lock, machine)
            self.assertNotIn(PROJECT, {name for name, _, _ in reached})
            for (name, version, _), record in reached.items():
                wheels = _wheels(record)
                self.assertFalse(wheels and all("macosx" in w for w in wheels), "%s %s is a macOS record" % (name, version))
        macos_only = sorted(p["name"] for p in self.lock["package"]
                            if _wheels(p) and all("macosx" in w for w in _wheels(p)))
        self.assertEqual(macos_only, ["torch", "torchvision"])

    def test_torch_is_the_cpu_build_on_both_machines(self):
        for machine in MACHINES:
            versions = {name: version for (name, version, _), _ in closure(self.lock, machine).items()
                        if name in ("torch", "torchvision")}
            self.assertEqual(versions, {"torch": "2.15.0+cpu", "torchvision": "0.29.1+cpu"}, machine)

    def test_the_guard_fails_on_a_lock_without_the_wheel(self):
        """The guard is not vacuous: a lock whose torch has no aarch64 wheel is found."""
        lock = load_lock()
        for package in lock["package"]:
            if package["name"] == "torch" and package["version"] == "2.15.0+cpu":
                package["wheels"] = [w for w in package["wheels"] if "aarch64" not in w["url"]]
        reached = closure(lock, "aarch64")
        missing = [name for (name, _, _), record in reached.items()
                   if not any(wheel_fits(w, "aarch64") for w in _wheels(record)) and name not in SDIST_ONLY]
        self.assertEqual(missing, ["torch"])


if __name__ == "__main__":
    unittest.main()
