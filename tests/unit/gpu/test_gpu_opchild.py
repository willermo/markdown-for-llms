"""Every call of the product to a provider in a child with its own process group (MFL023 B1; plan r2 §4.3, r3 §4.3):
the limits `until` (the wall clock, the term T of an arm) and `budget_s` (the monotonic clock, the readings and the
cleanup), the stop of the caller, the death of the whole group at a limit, the child that dies with its parent, and an
answer in closed fields only. No network: a fake provider in a JSON file (`opchild_driver.py`), `httpx.MockTransport`
and a fake CLI of Modal in process."""
import io
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import types
import unittest

import httpx

from markdown_for_llms.engine import envfile
from markdown_for_llms.engine.relay import credential_forms
from markdown_for_llms.gpu import errors, opchild, presets
from markdown_for_llms.gpu import providers as pv

HERE = Path(__file__).resolve().parent
DRIVER = HERE / "opchild_driver.py"
KEY = "b1-synthetic-key/0123+456789=="
FORMS = sorted({f.decode() for f in credential_forms([KEY])})
LEAK = " ".join(FORMS)


def gone(pid, within=3.0):
    """True when `pid` is no longer a live process (a zombie counts as gone) within `within` seconds."""
    end = time.monotonic() + within
    while time.monotonic() < end:
        try:
            state = Path("/proc/%d/stat" % pid).read_text().rsplit(")", 1)[1].split()[0]
        except (FileNotFoundError, ProcessLookupError, IndexError):
            return True
        if state == "Z":
            return True
        time.sleep(0.05)
    return False


class Base(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp(prefix="mdf-opchild-"))
        self.addCleanup(shutil.rmtree, self.dir, True)
        self.state = self.dir / "state.json"

    def fake(self, provider="runpod", **behave):
        self.state.write_text(json.dumps({"provider": provider, "behave": behave}))

    def read(self):
        return json.loads(self.state.read_text())

    def call(self, op, provider="runpod", **kw):
        argv = [sys.executable, "-I", "-B", str(DRIVER), str(self.state)]
        return opchild.call(op, provider, "mdf-surya", str(self.dir / ".env"), kw.pop("args", None), argv=argv, **kw)

    def tearDown(self):
        try:
            s = self.read()
        except (OSError, ValueError):
            return
        for pid in (s.get("grandchild"), s.get("pid")):
            if pid:
                try:
                    os.kill(pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass


class TestAnswers(Base):
    def test_an_answer_in_closed_fields(self):
        self.fake(state={"result": {"armed": False, "busy": False, "workers": []}})
        got = self.call("state")
        self.assertTrue(got["ok"], got)
        self.assertEqual(got["result"], {"state": {"armed": False, "busy": False, "workers": []}})
        self.assertEqual((got["error"], got["child"]["exit"], got["child"]["killed"]), (None, 0, False))
        self.assertEqual([c[0] for c in self.read()["calls"]], ["state"])

    def test_no_key_in_the_environment_of_the_child(self):
        self.fake()
        env = dict(os.environ, **{k: "x" for k in pv.KEYS})
        self.assertTrue(self.call("state", environ=env)["ok"])
        self.assertFalse(set(pv.KEYS) & set(self.read()["env"]))

    def test_no_name_of_a_key_in_the_environment_of_the_child(self):
        """C-DB-NB5 (MFL023 B10, §5 of the plan r3): `child_env(environ, also)` drops the names of `KEYS`, those of
        `also` (the `key_env` of an endpoint) and every name of a key by the rule of `engine.servers` (ending in
        `envfile.KEY_SUFFIXES`, never an `MDF_` setting); `opchild.call` passes `also` on."""
        env = {"PATH": "/usr/bin", "HOME": "/tmp", "CUSTOM_ENDPOINT_KEY": "1", "TEAM_SECRET": "2", "A_TOKEN": "3",
               "B_TOKEN_ID": "4", "C_BEARER": "5", "MDF_DATALAB_CAP_USD": "6", "MDF_X_KEY": "7", "PLAIN": "8",
               **{k: "x" for k in pv.KEYS}}
        self.assertEqual(pv.child_env(env), {"PATH": "/usr/bin", "HOME": "/tmp", "MDF_DATALAB_CAP_USD": "6",
                                             "MDF_X_KEY": "7", "PLAIN": "8"})
        self.assertEqual(pv.child_env(env, also=("PLAIN",)), {"PATH": "/usr/bin", "HOME": "/tmp",
                                                              "MDF_DATALAB_CAP_USD": "6", "MDF_X_KEY": "7"})
        self.fake()
        self.assertTrue(self.call("state", environ=dict(os.environ, **env), also=("PLAIN",))["ok"])
        seen = set(self.read()["env"])
        self.assertFalse(seen & ({"CUSTOM_ENDPOINT_KEY", "TEAM_SECRET", "A_TOKEN", "B_TOKEN_ID", "C_BEARER", "PLAIN"}
                                 | set(pv.KEYS)))
        self.assertIn("MDF_DATALAB_CAP_USD", seen)

    def test_the_error_of_the_provider_leaves_in_closed_fields(self):
        """The text of the provider (here the stderr of the CLI of Modal, in memory as `remote`) with a synthetic
        key as it is and in every form of `relay.credential_forms`: on the stdout of the child, nowhere."""
        self.fake("modal", zero={"raise": {"local": "modal app list: exit 1", "op": "modal.app_list",
                                           "remote": "Error: token=" + LEAK}})
        request = {"op": "zero", "provider": "modal", "resource": "mdf-surya", "env_file": "x", "args": {}}
        raw = subprocess.run([sys.executable, "-I", "-B", str(DRIVER), str(self.state)], capture_output=True,
                             input=json.dumps(dict(request, parent=os.getpid())).encode(), timeout=60)
        self.assertEqual(raw.returncode, 1)
        for form in FORMS:
            self.assertNotIn(form.encode(), raw.stdout)
        self.assertEqual(json.loads(raw.stdout)["error"], {"class": "ProviderError", "op": "modal.app_list",
                                                           "http": None, "detail": "modal app list: exit 1"})
        got = self.call("zero", "modal")
        self.assertFalse(got["ok"])
        self.assertEqual(got["error"], {"class": "ProviderError", "op": "modal.app_list", "http": None,
                                        "detail": "modal app list: exit 1"})
        self.assertEqual(errors.public(got["error"]), {"class": "ProviderError", "op": "modal.app_list", "http": None})

    def test_a_child_without_a_valid_answer(self):
        for code in ("print('not json')", "print('[1]')", "print(json.dumps({'ok': True, 'result': 3}))",
                     "import sys; sys.exit(9)"):
            with self.subTest(code=code):
                got = opchild.call("state", "runpod", "r", "e", argv=[sys.executable, "-I", "-B", "-c",
                                                                      "import json, sys; sys.stdin.read(); " + code])
                self.assertFalse(got["ok"])
                self.assertEqual(got["error"]["class"], "ChildFailed")

    def test_a_child_of_another_parent_calls_nothing(self):
        self.fake()
        request = {"op": "disarm", "provider": "runpod", "resource": "mdf-surya", "env_file": "x", "args": {},
                   "parent": 1}
        raw = subprocess.run([sys.executable, "-I", "-B", str(DRIVER), str(self.state)], capture_output=True,
                             input=json.dumps(request).encode(), timeout=60)
        self.assertEqual((raw.returncode, raw.stdout), (125, b""))
        self.assertNotIn("calls", self.read())

    def test_a_request_out_of_its_form(self):
        for request in ({}, {"op": "deploy", "provider": "runpod", "resource": "r", "parent": 1},
                        {"op": "state", "provider": "aws", "resource": "r", "parent": 1}, []):
            out = io.StringIO()
            code = opchild.main(stdin=io.BytesIO(json.dumps(request).encode()), stdout=out,
                                factory=lambda r: self.fail("called"), watch=lambda p: True)
            self.assertEqual(code, 2)
            self.assertEqual(json.loads(out.getvalue())["error"]["class"], "RequestInvalid")

    def test_the_module_of_the_package(self):
        """`python -I -B -m markdown_for_llms.gpu.opchild`, the command of the worker: the installed wheel (the full
        check installs the canonical one)."""
        raw = subprocess.run(opchild.command(), capture_output=True, input=b"{}", timeout=60)
        self.assertEqual(raw.returncode, 2)
        self.assertEqual(json.loads(raw.stdout)["error"]["class"], "RequestInvalid")


class TestLimits(Base):
    def test_the_term_kills_a_call_that_drips_past_it(self):
        """An arm still receiving its answer at `until`: at `until` + at most 1 s the child is gone, the answer is
        `TermReached`, and the fake had the call open (it was still ticking)."""
        self.fake(arm={"sleep": 30})
        until = time.time() + 1.5
        got = self.call("arm", until=until)
        self.assertLess(time.time(), until + 1.0)
        self.assertEqual((got["ok"], got["error"]["class"], got["child"]["killed"]), (False, "TermReached", True))
        s = self.read()
        self.assertGreater(s.get("ticks", 0), 0)
        self.assertTrue(gone(s["pid"]))
        ticks = self.read()["ticks"]
        time.sleep(0.4)
        self.assertEqual(self.read()["ticks"], ticks)          # nothing more after the kill
        self.assertEqual(s["calls"][0][1]["deadline"], until)   # the provider got the term too

    def test_the_group_dies_with_the_child(self):
        """`modal deploy` runs `uv` and the CLI in the group of the child: at the term they die with it."""
        self.fake("modal", arm={"spawn": True, "sleep": 30})
        got = self.call("arm", "modal", until=time.time() + 1.5)
        self.assertEqual(got["error"]["class"], "TermReached")
        s = self.read()
        self.assertTrue(gone(s["grandchild"]))
        self.assertTrue(gone(s["pid"]))

    def test_a_budget_on_the_monotonic_clock(self):
        """`budget_s` counts the monotonic clock: a jump of the wall clock spends none of it."""
        self.fake(state={"sleep": 30})
        wall = iter(range(10 ** 9, 10 ** 10, 3600))               # a jump of one hour at every reading
        t0 = time.monotonic()
        got = self.call("state", budget_s=1.0, wall=lambda: next(wall))
        self.assertGreaterEqual(time.monotonic() - t0, 1.0)
        self.assertLess(time.monotonic() - t0, 2.0)
        self.assertEqual(got["error"]["class"], "BudgetSpent")
        self.assertTrue(gone(self.read()["pid"]))

    def test_the_stop_of_the_caller(self):
        self.fake(zero={"sleep": 30})
        t0 = time.monotonic()
        got = self.call("zero", stop=lambda: time.monotonic() - t0 > 0.6)
        self.assertEqual(got["error"]["class"], "Stopped")
        self.assertLess(time.monotonic() - t0, 1.5)
        self.assertTrue(gone(self.read()["pid"]))

    def test_a_limit_already_reached_starts_no_child(self):
        self.fake()
        for kw, cls in (({"until": time.time() - 1}, "TermReached"), ({"stop": lambda: True}, "Stopped"),
                        ({"budget_s": 0}, "BudgetSpent")):
            with self.subTest(cls=cls):
                got = self.call("arm", **kw)
                self.assertEqual((got["error"]["class"], got["child"]), (cls, None))
        self.assertNotIn("calls", self.read())

    def test_the_child_dies_with_its_parent(self):
        """A SIGKILL of the caller (the worker): the child goes too (`PR_SET_PDEATHSIG`)."""
        self.fake(arm={"sleep": 30})
        root = str(HERE.parents[2])
        code = ("import sys; sys.path.insert(0, %r); from markdown_for_llms.gpu import opchild; "
                "opchild.call('arm', 'runpod', 'mdf-surya', 'x', argv=[sys.executable, '-I', '-B', %r, %r])"
                % (root, str(DRIVER), str(self.state)))
        parent = subprocess.Popen([sys.executable, "-I", "-B", "-c", code])
        try:
            end = time.monotonic() + 30
            while "pid" not in self.read() or self.read().get("ticks", 0) < 2:
                self.assertLess(time.monotonic(), end)
                time.sleep(0.05)
            child = self.read()["pid"]
            parent.kill()
            parent.wait(10)
            self.assertTrue(gone(child))
        finally:
            if parent.poll() is None:
                parent.kill()
                parent.wait(10)


class FakeRunPod:
    """The REST v2 of one endpoint at rest in CA-MTL-1 with its volume (as `test_gpu_preset_pins.py`)."""

    def __init__(self, volume=None):
        self.ep = {"id": "ep1", "name": "mdf-surya", "type": "LOAD_BALANCER", "image": pv.IMAGE,
                   "entrypoint": pv.ENTRYPOINT, "env": pv.base_env(), "workers": pv.workers_block(0),
                   "scaling": {"type": "REQUEST_COUNT", "requestCount": 32}, "flashboot": "OFF",
                   "ports": ["8000/http"], "dataCenterIds": ["CA-MTL-1"], "mounts": {"network": [{"volumeId": "vol1"}]},
                   "gpu": {"pools": [pv.POOL], "minCudaVersion": pv.CUDA_MIN, "count": 1}}
        self.volume = dict({"id": "vol1", "name": pv.VOLUME, "dataCenter": "CA-MTL-1", "type": "HIGH_PERFORMANCE",
                            "size": 10}, **(volume or {}))
        self.calls = []

    def __call__(self, request):
        path, method = request.url.path.split("/v2", 1)[1], request.method
        self.calls.append((method, path))
        if path == "/serverless" and method == "GET":
            return httpx.Response(200, json={"endpoints": [self.ep], "pagination": {}})
        if path == "/serverless/ep1" and method == "GET":
            return httpx.Response(200, json=self.ep)
        if path == "/serverless/ep1" and method == "PATCH":
            body = json.loads(request.content)
            self.ep.update({k: v for k, v in body.items() if k != "gpu"})
            return httpx.Response(200, json=self.ep)
        if path == "/serverless/ep1/workers":
            return httpx.Response(200, json={"summary": {"total": 0}, "workers": []})
        if path == "/network-volumes":
            return httpx.Response(200, json={"networkVolumes": [self.volume]})
        if path == "/catalog/gpus":
            return httpx.Response(200, json={"gpus": [{"id": "NVIDIA RTX A5000", "pool": pv.POOL, "secure": True,
                                                       "price": {"serverless": 0.75}}]})
        if path == "/catalog/datacenters":
            return httpx.Response(200, json={"dataCenters": [{"id": "CA-MTL-1", "networkVolumeTypes": ["HIGH_PERFORMANCE"]}]})
        return httpx.Response(404, json={"error": "not found " + KEY})


class TestOperations(unittest.TestCase):
    """The operations of the child, in process (`main` with a factory)."""

    def run_op(self, op, prov, args=None, until=None, provider="runpod"):
        out = io.StringIO()
        request = {"op": op, "provider": provider, "resource": "mdf-surya", "env_file": "x", "args": args or {},
                   "parent": 1, "until": until}
        code = opchild.main(stdin=io.BytesIO(json.dumps(request).encode()), stdout=out, factory=lambda r: prov,
                            watch=lambda p: True)
        return code, json.loads(out.getvalue())

    def runpod(self, volume=None):
        fake = FakeRunPod(volume)
        return fake, pv.RunPod(envfile.Secret("rk-b1"), "mdf-surya", transport=httpx.MockTransport(fake))

    def test_the_reading_at_rest_of_runpod(self):
        preset = presets.load()["runpod-serverless-volume-dc"]
        fake, prov = self.runpod()
        code, got = self.run_op("rest", prov, {"preset": preset})
        self.assertEqual(code, 0)
        self.assertEqual((got["result"]["pins_failed"], got["result"]["url"]), ([], "https://ep1.api.runpod.ai"))
        self.assertFalse(got["result"]["state"]["armed"])
        fake, prov = self.runpod({"dataCenter": "EU-SE-1"})
        code, got = self.run_op("rest", prov, {"preset": preset})
        self.assertIn("preset_data_center", got["result"]["pins_failed"])

    def test_the_arm_of_runpod_with_its_preset_and_term(self):
        preset = presets.load()["runpod-serverless-volume-dc"]
        fake, prov = self.runpod()
        code, got = self.run_op("arm", prov, {"preset": preset, "levers": pv.DEFAULT_LEVERS}, until=time.time() + 60)
        self.assertEqual(code, 0, got)
        self.assertEqual(got["result"]["url"], "https://ep1.api.runpod.ai")
        self.assertIn(("PATCH", "/serverless/ep1"), fake.calls)
        fake, prov = self.runpod({"type": "STANDARD"})
        code, got = self.run_op("arm", prov, {"preset": preset}, until=time.time() + 60)
        self.assertEqual((code, got["error"]["class"]), (1, "ProviderError"))
        self.assertIn("preset_volume_class", got["error"]["detail"])
        self.assertNotIn(("PATCH", "/serverless/ep1"), fake.calls)          # refused before its PATCH
        fake, prov = self.runpod()
        code, got = self.run_op("arm", prov, {"preset": preset}, until=time.time() - 1)
        self.assertEqual(got["error"]["class"], "TermReached")
        self.assertEqual(fake.calls, [])

    def test_a_body_of_runpod_never_in_the_answer(self):
        fake, prov = self.runpod()
        fake.ep["id"] = "other"                                              # every reading of ep1 answers 404
        code, got = self.run_op("state", prov)
        text = json.dumps(got)
        for form in FORMS:
            self.assertNotIn(form, text)
        self.assertEqual((got["error"]["op"], got["error"]["http"]), ("runpod.get", 404))

    def test_the_arm_of_modal_gives_its_url_canonical(self):
        def runner(argv, **kw):
            args = argv[len(pv.MODAL_SDK):]
            word = args[0] if args[0] == "deploy" else " ".join(args[:2])
            out = {"volume list": json.dumps([{"name": pv.VOLUME}]),
                   "workspace proxy-tokens": json.dumps([{"token_id": "tid", "scoped": False}]),
                   "deploy": "Created web function serve => https://ws--mdf-surya-serve.modal.run\n"}.get(word, "[]")
            return types.SimpleNamespace(returncode=0, stdout=out, stderr="")
        env = Path(tempfile.mkdtemp()) / ".env"
        self.addCleanup(shutil.rmtree, env.parent, True)
        env.write_text("MODAL_PROXY_TOKEN_ID=tid\n")
        env.chmod(0o600)
        prov = pv.Modal("mdf-surya", runner=runner, environ={}, env_file=str(env))
        code, got = self.run_op("arm", prov, until=time.time() + 60, provider="modal")
        self.assertEqual(code, 0, got)
        self.assertEqual(got["result"]["url"], "https://ws--mdf-surya-serve.modal.run")
        self.assertEqual(got["result"]["info"]["proxy_token"]["in_workspace"], True)


class TestBounded(unittest.TestCase):
    def test_only_the_closed_fields(self):
        self.assertEqual(errors.bounded({"class": "ProviderError", "op": "runpod.patch", "http": 500, "detail": "x" * 900,
                                         "remote": LEAK, "body": KEY}),
                         {"class": "ProviderError", "op": "runpod.patch", "http": 500, "detail": "x" * 300})
        self.assertEqual(errors.bounded({"class": "a b; " + KEY, "op": 3, "http": "500"}),
                         {"class": "Error", "op": None, "http": None, "detail": None})
        self.assertEqual(errors.bounded(None)["class"], "Error")
        self.assertEqual(errors.public({"class": "TermReached", "op": "opchild.arm", "http": None, "detail": "d"}),
                         {"class": "TermReached", "op": "opchild.arm", "http": None})

    def test_closed_of_a_local_error_is_its_class_only(self):
        self.assertEqual(pv.closed(ValueError("a local text " + KEY)), {"class": "ValueError"})
        self.assertEqual(pv.closed(pv.TermReached("the term")),
                         {"class": "TermReached", "op": None, "http": None, "detail": "the term"})


if __name__ == "__main__":
    unittest.main()
