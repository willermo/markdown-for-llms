"""The payload gate (MFL051, plan r2.1 J.8.2; SOL-R2-N4): do the children run what is under test, or an old image?

Three comparisons between the copy under test (this working copy and the environment of this process), the launcher
(`hello`) and a child (the probe, which reads the files it runs): the code, the build inputs the image recorded, the
interpreter and the distributions. `TestGateFunction` needs no Docker and runs everywhere. The other classes need
the launcher: the real one must pass, and two images derived here, without network, from the image under test must
fail on exactly one comparison each: one with a file of the project changed (code), one with the same code and one
more distribution, installed from a synthetic wheel (environment). A launcher is started on each of them, because
the image of the children is the launcher's own.
"""
import copy
import io
import json
import os
from pathlib import Path
import tarfile
import tempfile
import time
import unittest
import zipfile

import container_support as support
from markdown_for_llms import launcher as launcher_module
from markdown_for_llms.engine import containers

ROOT = Path(__file__).resolve().parents[2]
SITE = "/opt/mdf/venv/lib/python3.12/site-packages"


def reference():
    """What the copy under test says of itself: the package of this working copy, the build inputs computed on its
    files, the environment of this process (the image `dev`)."""
    return {"payload_sha256": containers.payload_sha256(ROOT / "markdown_for_llms"),
            "inputs_sha256": containers.inputs_sha256(ROOT), "environment": containers.environment()}


class TestGateFunction(unittest.TestCase):
    """The function alone, on synthetic trees: no launcher, no Docker."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        for name in containers.INPUTS:
            (self.root / name).parent.mkdir(parents=True, exist_ok=True)
            (self.root / name).write_text("content of " + name)
        package = self.root / "site" / "markdown_for_llms"
        (package / "engine" / "__pycache__").mkdir(parents=True)
        (package / "__init__.py").write_text("")
        (package / "engine" / "core.py").write_text("x = 1\n")
        (package / "engine" / "__pycache__" / "core.cpython-312.pyc").write_bytes(b"compiled")
        self.package = package
        self.same = {"payload_sha256": containers.payload_sha256(package), "inputs_sha256": containers.inputs_sha256(self.root),
                     "environment": {"python_sha256": "a" * 64, "distributions": ["numpy==2.0", "torch==2.15.0+cpu"]}}

    def parties(self):
        ref = copy.deepcopy(self.same)
        ref["environment"]["distributions"] += ["pytest==9.1.1"]            # the copy under test has the dev group on top
        return ref, copy.deepcopy(self.same), copy.deepcopy(self.same)

    def test_three_equal_parties_pass(self):
        self.assertEqual(containers.gate(*self.parties()), [])

    def test_the_payload_is_the_files_of_the_package_without_compiled_ones(self):
        before = containers.payload_sha256(self.package)
        (self.package / "engine" / "__pycache__" / "other.cpython-312.pyc").write_bytes(b"x")
        (self.package / "stale.pyc").write_bytes(b"x")
        self.assertEqual(containers.payload_sha256(self.package), before)
        (self.package / "engine" / "core.py").write_text("x = 2\n")
        changed = containers.payload_sha256(self.package)
        self.assertNotEqual(changed, before)
        (self.package / "engine" / "core.py").write_text("x = 1\n")
        self.assertEqual(containers.payload_sha256(self.package), before)
        (self.package / "engine" / "data.json").write_text("{}")
        self.assertNotEqual(containers.payload_sha256(self.package), before)       # a file more is another payload
        (self.package / "engine" / "data.json").unlink()
        (self.package / "engine" / "core.py").rename(self.package / "engine" / "moved.py")
        self.assertNotEqual(containers.payload_sha256(self.package), before)       # the same bytes under another name too

    def test_only_uv_lock_changed_fails_on_the_inputs(self):
        ref, lau, chi = self.parties()
        (self.root / "uv.lock").write_text("another lock")
        ref["inputs_sha256"] = containers.inputs_sha256(self.root)
        self.assertEqual(containers.gate(ref, lau, chi), ["inputs"])

    def test_each_build_input_changed_alone_fails_on_the_inputs(self):
        for name in containers.INPUTS:
            ref, lau, chi = self.parties()
            original = (self.root / name).read_text()
            (self.root / name).write_text(original + "\n# changed")
            ref["inputs_sha256"] = containers.inputs_sha256(self.root)
            (self.root / name).write_text(original)
            self.assertEqual(containers.gate(ref, lau, chi), ["inputs"], name)

    def test_only_the_lock_of_modal_changed_fails_on_the_inputs(self):
        """The lock of the Modal CLI joins the inputs with tranche A2: the function takes the list, so a list with it
        behaves like the list of today."""
        names = containers.INPUTS + ("docker/app/modal-requirements.txt",)
        (self.root / "docker/app/modal-requirements.txt").write_text("modal==1.6.3 --hash=sha256:" + "0" * 64)
        ref, lau, chi = self.parties()
        for party in (ref, lau, chi):
            party["inputs_sha256"] = containers.inputs_sha256(self.root, names)
        self.assertEqual(containers.gate(ref, lau, chi), [])
        (self.root / "docker/app/modal-requirements.txt").write_text("modal==1.6.4 --hash=sha256:" + "1" * 64)
        ref["inputs_sha256"] = containers.inputs_sha256(self.root, names)
        self.assertEqual(containers.gate(ref, lau, chi), ["inputs"])

    def test_an_image_that_recorded_no_inputs_fails(self):
        ref, lau, chi = self.parties()
        chi["inputs_sha256"] = None
        self.assertEqual(containers.gate(ref, lau, chi), ["inputs"])
        for party in (ref, lau, chi):
            party["inputs_sha256"] = None
        self.assertEqual(containers.gate(ref, lau, chi), ["inputs"])

    def test_code_and_environment(self):
        ref, lau, chi = self.parties()
        chi["payload_sha256"] = "0" * 64
        self.assertEqual(containers.gate(ref, lau, chi), ["code"])
        ref, lau, chi = self.parties()
        lau["payload_sha256"] = "0" * 64
        self.assertEqual(containers.gate(ref, lau, chi), ["code"])
        for change in (lambda e: e.update(python_sha256="b" * 64), lambda e: e["distributions"].append("extra==1.0"),
                       lambda e: e.update(distributions=["numpy==2.1", "torch==2.15.0+cpu"])):
            ref, lau, chi = self.parties()
            change(chi["environment"])
            change(lau["environment"])
            self.assertEqual(containers.gate(ref, lau, chi), ["environment"])
        ref, lau, chi = self.parties()
        chi["environment"]["distributions"] = ["numpy==2.0"]                 # launcher and child disagree
        self.assertEqual(containers.gate(ref, lau, chi), ["environment"])
        ref, lau, chi = self.parties()
        chi["payload_sha256"], chi["inputs_sha256"] = "0" * 64, "1" * 64
        chi["environment"]["python_sha256"] = "c" * 64
        self.assertEqual(containers.gate(ref, lau, chi), ["code", "inputs", "environment"])

    def test_what_a_process_reports_of_itself(self):
        identity = containers.identity(self.root / "absent.json")
        self.assertIsNone(identity["inputs_sha256"])
        (self.root / "inputs.json").write_text(json.dumps({"inputs_sha256": "f" * 64}))
        identity = containers.identity(self.root / "inputs.json")
        self.assertEqual(identity["inputs_sha256"], "f" * 64)
        self.assertRegex(identity["payload_sha256"], "^[0-9a-f]{64}$")
        self.assertRegex(identity["environment"]["python_sha256"], "^[0-9a-f]{64}$")
        names = [d.split("==")[0] for d in identity["environment"]["distributions"]]
        self.assertIn("pytest", names)
        self.assertNotIn("markdown-for-llms", names)
        self.assertEqual(identity["environment"]["distributions"], sorted(identity["environment"]["distributions"]))


@unittest.skipUnless(support.available(), support.REASON)
class TestGateOnTheImagesUnderTest(support.ComponentCase):
    def child_identity(self, launcher=None, attempt=None):
        """What a child of `launcher` reports of itself: the probe computes it from the files it runs."""
        name = containers.child_name(attempt or self.attempt, "probe", 1)
        with containers.Child(launcher or self.launcher, containers.WORK_ROOT, name) as child:
            (child.call.scratch / containers.LIMITS_FILE).write_text(json.dumps(support.CHILD_LIMITS))
            outcome = child.run("app", containers.PROBE_PYTHON, support.VARIABLES, support.LIMITS, timeout=120)
            self.assertEqual(outcome.exit, 0, outcome.stderr)
            return json.loads((child.call.scratch / "probe.json").read_text())["identity"]

    def test_the_launcher_and_its_children_run_the_copy_under_test(self):
        ref, hello, child = reference(), self.launcher.hello(), self.child_identity()
        self.assertEqual(containers.gate(ref, hello, child), [])
        self.assertEqual(ref["payload_sha256"], child["payload_sha256"])
        self.assertEqual(hello["inputs_sha256"], child["inputs_sha256"])
        self.assertEqual(child["environment"], hello["environment"])
        self.assertTrue(set(child["environment"]["distributions"]) < set(ref["environment"]["distributions"]))
        guard = (ROOT / "scripts/diagnostics/run-a001-fase0-uv/guard_product_inputs.py").read_text()
        self.assertIn(child["environment"]["python_sha256"], guard)             # the interpreter the guard pins
        if (support.OUT).is_dir():
            (support.OUT / "payload-gate.json").write_text(json.dumps(
                {"reference": {k: ref[k] for k in ("payload_sha256", "inputs_sha256")}, "python_sha256": ref["environment"]["python_sha256"],
                 "launcher": {k: hello[k] for k in ("payload_sha256", "inputs_sha256", "images", "docker")},
                 "child": {k: child[k] for k in ("payload_sha256", "inputs_sha256")},
                 "distributions": {"reference": len(ref["environment"]["distributions"]), "child": len(child["environment"]["distributions"])},
                 "failed": containers.gate(ref, hello, child)}, indent=1, sort_keys=True) + "\n")

    # -- images derived from the one under test, without network ----------------------------------------------
    def derive(self, label, members):
        """An image with `members` ({path under site-packages: bytes}) added to the image under test; returns its
        reference. The driver copies the files into a container of the image and commits it: nothing is downloaded."""
        own = self.hello["images"]["app"]["id"]
        reference_ = "mdf-app:%s-%s" % (label, self.project)
        status, created = self.raw("POST", "/containers/create", query={"name": "%s-derive-%s" % (self.project, label)},
                                   body={"Image": own, "Cmd": ["/bin/true"], "Labels": {"mdf.project": self.project},
                                         "HostConfig": {"NetworkMode": "none"}})
        self.assertEqual(status, 201, created)
        try:
            archive = io.BytesIO()
            with tarfile.open(fileobj=archive, mode="w") as tar:
                for path, data in sorted(members.items()):
                    info = tarfile.TarInfo(path)
                    info.size, info.mode, info.mtime = len(data), 0o644, int(time.time())
                    tar.addfile(info, io.BytesIO(data))
            conn = launcher_module._UnixConnection(support.DOCKER_SOCKET, 120)
            try:
                conn.request("PUT", "/v1.45/containers/%s/archive?path=%s" % (created["Id"], SITE), body=archive.getvalue(),
                             headers={"Host": "docker", "Content-Type": "application/x-tar"})
                answer = conn.getresponse()
                self.assertEqual(answer.status, 200, answer.read())
            finally:
                conn.close()
            repo, tag = reference_.split(":")
            status, image = self.raw("POST", "/commit", query={"container": created["Id"], "repo": repo, "tag": tag}, body={})
            self.assertEqual(status, 201, image)
        finally:
            self.raw("DELETE", "/containers/%s" % created["Id"], query={"force": "1"})
        self.addCleanup(self.raw, "DELETE", "/images/%s" % reference_, query={"force": "1"})
        return reference_

    def launcher_on(self, image, label):
        """A second launcher, started by the driver from `image` with the mounts, user and group of the real one,
        its own project and its own socket; returns its client."""
        real = self.raw_inspect(self.service("launcher"))
        project = "%s-%s" % (self.project, label)
        path = "/run/mdf/launcher/%s.sock" % label
        command = [a for a in real["Config"]["Cmd"] if not a.startswith(("--socket=", "--project="))]
        command += ["--socket=" + path, "--project=" + project]
        body = {"Image": image, "Cmd": command, "User": real["Config"]["User"],
                "Labels": {"com.docker.compose.project": project, "com.docker.compose.service": "launcher", "mdf.project": self.project},
                "HostConfig": {"NetworkMode": "none", "ReadonlyRootfs": True, "CapDrop": ["ALL"], "SecurityOpt": ["no-new-privileges"],
                               "GroupAdd": real["HostConfig"]["GroupAdd"], "Binds": real["HostConfig"].get("Binds") or [],
                               "Mounts": real["HostConfig"].get("Mounts") or [], "Memory": 512 << 20, "MemorySwap": 512 << 20,
                               "PidsLimit": 256}}
        status, created = self.raw("POST", "/containers/create", query={"name": "%s-launcher" % project}, body=body)
        self.assertEqual(status, 201, created)
        self.addCleanup(self.stop_launcher_on, created["Id"], path)
        status, _ = self.raw("POST", "/containers/%s/start" % created["Id"])
        self.assertEqual(status, 204)
        client = containers.Launcher(path)
        until = time.monotonic() + 60
        while True:
            try:
                client.hello()
                return client
            except containers.Unreachable:
                state = self.raw_inspect(created["Id"])["State"]
                if not state["Running"] or time.monotonic() > until:
                    _, logs = self.docker.call("GET", "/containers/%s/logs" % created["Id"], query={"stdout": "1", "stderr": "1"})
                    raise AssertionError("the launcher on %s did not start: %r" % (image, logs[-600:]))
                time.sleep(0.2)

    def stop_launcher_on(self, ident, path):
        self.raw("POST", "/containers/%s/stop" % ident, query={"t": "30"}, timeout=60)      # it removes its own children
        self.raw("DELETE", "/containers/%s" % ident, query={"force": "1"})
        try:
            os.unlink(path)
        except OSError:
            pass

    def test_an_image_with_a_file_of_the_project_changed_fails_on_the_code(self):
        source = (ROOT / "markdown_for_llms" / "__init__.py").read_bytes()
        image = self.derive("stale-code", {"markdown_for_llms/__init__.py": source + b"\n# a line the tree under test does not have\n"})
        stale = self.launcher_on(image, "stale-code")
        ref, hello, child = reference(), stale.hello(), self.child_identity(stale, self.attempt[:14] + "c1")
        self.assertEqual(containers.gate(ref, hello, child), ["code"])
        self.assertNotEqual(hello["images"]["app"]["id"], self.hello["images"]["app"]["id"])
        self.assertEqual(hello["payload_sha256"], child["payload_sha256"])       # launcher and child agree, and both are stale
        self.assertNotEqual(child["payload_sha256"], ref["payload_sha256"])
        self.assertEqual(child["inputs_sha256"], ref["inputs_sha256"])
        self.assertEqual(containers.gate(ref, self.launcher.hello(), child), ["code"])      # a stale child under the right launcher too

    def test_an_image_with_the_same_code_and_one_more_distribution_fails_on_the_environment(self):
        """The case of the dependencies alone: a wheel made here is installed into the image without network."""
        wheel = io.BytesIO()
        info = "mdf_synthetic_wheel-1.0.dist-info"
        files = {"mdf_synthetic_wheel/__init__.py": b"VALUE = 1\n",
                 info + "/METADATA": b"Metadata-Version: 2.1\nName: mdf-synthetic-wheel\nVersion: 1.0\n",
                 info + "/WHEEL": b"Wheel-Version: 1.0\nGenerator: tests\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
                 info + "/RECORD": b"", info + "/INSTALLER": b"tests\n"}
        with zipfile.ZipFile(wheel, "w") as archive:
            for name, data in files.items():
                archive.writestr(name, data)
        with zipfile.ZipFile(io.BytesIO(wheel.getvalue())) as archive:          # installed as a pure wheel is: its files, unpacked
            members = {name: archive.read(name) for name in archive.namelist()}
        image = self.derive("stale-deps", members)
        stale = self.launcher_on(image, "stale-deps")
        ref, hello, child = reference(), stale.hello(), self.child_identity(stale, self.attempt[:14] + "d1")
        self.assertEqual(containers.gate(ref, hello, child), ["environment"])
        self.assertEqual(child["payload_sha256"], ref["payload_sha256"])         # the same code
        self.assertEqual(child["inputs_sha256"], ref["inputs_sha256"])           # and the same recorded inputs
        extra = set(child["environment"]["distributions"]) - set(ref["environment"]["distributions"])
        self.assertEqual(extra, {"mdf-synthetic-wheel==1.0"})


if __name__ == "__main__":
    unittest.main()
