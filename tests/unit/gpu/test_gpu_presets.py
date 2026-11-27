"""The GPU presets (MFL023 A3): the versioned file and its closed schema, the values equal to the constants of the
session code, the presets an instance may enable, and the file in the wheel. MFL051 B: the presets of a GPU of the
user (`gpu-local`, `gpu-remote`), without rate, volume, data center or lever, and the pure table of the traits."""
import ast
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from markdown_for_llms.gpu import codes, presets
from markdown_for_llms.gpu import providers as pv
from markdown_for_llms.gpu import watchdog

ROOT = Path(__file__).resolve().parents[3]
GPU = ROOT / "markdown_for_llms" / "gpu"


def constant(module, name):
    """A constant of a module of the package read from its source, never imported (`modal_app` imports the SDK of
    Modal, `drivers` the measurement tools)."""
    tree = ast.parse((GPU / (module + ".py")).read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError("%s.%s not found" % (module, name))


class Base(unittest.TestCase):
    def setUp(self):
        self.doc = json.loads(presets.PATH.read_text(encoding="utf-8"))

    def refused(self, doc, code=codes.GPU_PRESET_INVALID):
        with self.assertRaises(codes.GpuError) as ctx:
            presets.check(doc)
        self.assertEqual(ctx.exception.code, code)
        return ctx.exception

    def changed(self, name, change):
        doc = copy.deepcopy(self.doc)
        change(doc["presets"][name])
        return doc

    def rented(self):
        return [n for n, p in self.doc["presets"].items() if p["provider"] in ("runpod", "modal")]


class TheFileTests(Base):
    def test_the_presets_of_mfl023(self):
        found = presets.load()
        self.assertEqual(sorted(found), ["gpu-local", "gpu-remote", "modal-base", "runpod-serverless-volume-dc",
                                         "runpod-serverless-volume-eu-se-1"])
        self.assertEqual({n: p["status"] for n, p in found.items()}, {
            "modal-base": "supported", "runpod-serverless-volume-dc": "supported",
            "runpod-serverless-volume-eu-se-1": "superseded", "gpu-local": "supported", "gpu-remote": "supported"})
        old = found["runpod-serverless-volume-eu-se-1"]
        self.assertEqual(old["superseded_by"], "runpod-serverless-volume-dc")
        self.assertEqual((old["data_center"], old["volume"]["class"]), ("EU-SE-1", "STANDARD"))
        dc = found["runpod-serverless-volume-dc"]
        self.assertEqual((dc["data_center"], dc["volume"]["class"], dc["volume"]["size_gb"]),
                         ("CA-MTL-1", "HIGH_PERFORMANCE", 10))
        self.assertEqual((dc["levers"]["arm_ahead_s"], found["modal-base"]["levers"]["arm_ahead_s"]), (420, None))
        self.assertEqual((found["modal-base"]["seconds"]["ready_within"], found["modal-base"]["seconds"]["cold_start_s"]),
                         (900, 480))

    def test_the_values_equal_the_constants_of_the_session_code(self):
        image = constant("modal_app", "IMAGE")
        for name, p in presets.load().items():
            with self.subTest(preset=name):
                self.assertEqual(p["image"], pv.IMAGE)
                self.assertEqual(p["image"], image)
                self.assertEqual(p["model"]["id"], watchdog.MODEL)
                self.assertEqual(p["model"]["revision"], watchdog.REVISION)
                self.assertEqual(p["model"]["vllm"], constant("drivers", "VLLM_VERSION"))
                if p["provider"] in ("local", "remote"):
                    # MFL051 B: the same image and model as the rented GPUs, nothing rented
                    self.assertEqual((p["volume"], p["rate"], p["data_center"], p["mode"]), (None, None, None, "server"))
                    continue
                self.assertEqual(p["volume"]["name"], pv.VOLUME)
                self.assertEqual(p["rate"]["usd_per_s"], str(pv.RATES[p["provider"]]))
                if p["provider"] == "runpod":
                    self.assertEqual((p["gpu"]["group"], p["volume"]["size_gb"]), (pv.POOL, pv.VOLUME_GB))
                else:
                    self.assertEqual(p["gpu"]["group"], list(pv.MODAL_GPUS))

    def test_the_compile_caches_of_the_supported_presets_are_the_levers_adopted(self):
        """MFL049 (d): the SHA-256 of the caches live in the presets; the levers of an arm without `--levers` are those
        of the supported presets, 6222ea36 for RunPod since the move to CA-MTL-1 (d3818054 stays in the superseded
        preset, whose volume was deleted)."""
        self.assertEqual(watchdog.cache_shas(watchdog.levers(pv.DEFAULT_LEVERS)), sorted(presets.compile_caches()))
        self.assertIn("6222ea36b0e660ccbd89b25386a30aab3dbaec14231bd36e7aae4f3b9751e7f4", pv.DEFAULT_LEVERS)
        self.assertNotIn("d38180544831bf3c0d1a45712e25f125ff4d18a9929012376f844a2224e0ff73", pv.DEFAULT_LEVERS)
        self.assertEqual(pv.levers_text(pv.DEFAULT_LEVERS), pv.DEFAULT_LEVERS)

    def test_the_endpoint_fields_of_a_preset(self):
        fields = presets.endpoint_fields(presets.load()["modal-base"])
        self.assertEqual(fields, {
            "server": "vllm", "model": "datalab-to/surya-ocr-2.5", "revision": watchdog.REVISION,
            "pins": {"vllm": "0.25.1"}, "parallel": 32, "gpu": "L4,A10G", "rate_usd_per_s": "0.00036772",
            "rate_source": "https://modal.com/pricing (A10G, 2 cores, 16 GiB)", "rate_date": "2026-11-20",
            "ready_within": 900, "read_timeout": 140, "page_seconds": 30, "cold_start_s": 480, "w1_s": 900,
            "tail_idle_s": 90, "tail_fault_s": 225})
        table = presets.endpoint_table()
        self.assertEqual(table["runpod-serverless-volume-dc"]["fields"]["gpu"], "AMPERE_24")
        self.assertEqual({n: e["refusal"] for n, e in table.items()}, {
            "modal-base": None, "runpod-serverless-volume-dc": None, "gpu-local": None, "gpu-remote": None,
            "runpod-serverless-volume-eu-se-1": codes.GPU_PRESET_SUPERSEDED})
        self.assertEqual({n: (e["kind"], e["key_env"]) for n, e in table.items()}, {
            "modal-base": ("modal", ["MODAL_PROXY_TOKEN_ID", "MODAL_PROXY_TOKEN_SECRET"]),
            "runpod-serverless-volume-dc": ("runpod", ["RUNPOD_API_KEY"]),
            "runpod-serverless-volume-eu-se-1": ("runpod", ["RUNPOD_API_KEY"]),
            "gpu-local": ("gpu", []), "gpu-remote": ("user", ["GPU_SERVER_KEY"])})

    def test_the_endpoint_fields_of_a_gpu_of_the_user(self):
        """MFL051 B: no rate and none of the seconds that bound a spend; the seconds of a conversion stay."""
        fields = presets.endpoint_fields(presets.load()["gpu-local"])
        self.assertEqual(fields, {
            "server": "vllm", "model": "datalab-to/surya-ocr-2.5", "revision": watchdog.REVISION,
            "pins": {"vllm": "0.25.1"}, "parallel": 32, "gpu": "NVIDIA 24 GB, CUDA 13", "rate_usd_per_s": None,
            "rate_source": None, "rate_date": None, "ready_within": 900, "read_timeout": 300, "page_seconds": 30,
            "cold_start_s": None, "w1_s": None, "tail_idle_s": None, "tail_fault_s": None})
        self.assertEqual(presets.endpoint_fields(presets.load()["gpu-remote"]), fields)

    def test_the_traits_of_the_providers(self):
        """MFL051 B (App. A, rilievo H): a pure table; only a rented GPU costs and arms; only the local GPU keeps the
        page images on this machine."""
        found = presets.load()
        self.assertEqual({n: presets.traits(p) for n, p in found.items() if p["status"] == "supported"}, {
            "modal-base": {"costs": True, "arms": True, "leaves_machine": True},
            "runpod-serverless-volume-dc": {"costs": True, "arms": True, "leaves_machine": True},
            "gpu-local": {"costs": False, "arms": False, "leaves_machine": False},
            "gpu-remote": {"costs": False, "arms": False, "leaves_machine": True}})
        self.assertEqual(set(presets.TRAITS), set(presets.PROVIDERS))
        self.assertEqual((presets.costs(found["gpu-remote"]), presets.leaves_machine(found["gpu-remote"])),
                         (False, True))


class TheSchemaTests(Base):
    def test_a_field_more_or_less_refuses_the_file(self):
        for name in self.rented():
            for change in (lambda p: p.update(extra=1), lambda p: p.pop("measures"),
                           lambda p: p["volume"].update(extra=1), lambda p: p["seconds"].pop("idle_s"),
                           lambda p: p["model"].pop("vllm"), lambda p: p["rate"].update(currency="USD")):
                with self.subTest(preset=name, change=change):
                    self.refused(self.changed(name, change))
        self.refused(dict(self.doc, schema=2))
        self.refused(dict(self.doc, extra=1))
        self.refused({"schema": 1, "presets": {}})
        for name in ("gpu-local", "gpu-remote"):
            for change in (lambda p: p.update(extra=1), lambda p: p.pop("measures"),
                           lambda p: p["seconds"].pop("idle_s"), lambda p: p["model"].pop("vllm")):
                with self.subTest(preset=name, change=change):
                    self.refused(self.changed(name, change))

    def test_a_gpu_of_the_user_rents_nothing(self):
        """MFL051 B: a rate, a volume, a data center, a lever or a second that bounds a spend refuses the file, and so
        does the mode of a rented GPU."""
        rp = self.doc["presets"]["runpod-serverless-volume-dc"]
        for name in ("gpu-local", "gpu-remote"):
            for change in (lambda p: p.update(rate=copy.deepcopy(rp["rate"])),
                           lambda p: p.update(volume=copy.deepcopy(rp["volume"])),
                           lambda p: p.update(data_center="CA-MTL-1"), lambda p: p.update(mode="serverless"),
                           lambda p: p["levers"].update(arm_ahead_s=420),
                           lambda p: p["levers"].update(compile_cache=list(rp["levers"]["compile_cache"])),
                           lambda p: p["seconds"].update(idle_s=60), lambda p: p["seconds"].update(w1_s=900),
                           lambda p: p["seconds"].update(ready_within=None)):
                with self.subTest(preset=name, change=change):
                    self.refused(self.changed(name, change))
        self.refused(self.changed("modal-base", lambda p: p.update(mode="server")))
        self.refused(self.changed("modal-base", lambda p: p.update(rate=None)))
        self.refused(self.changed("runpod-serverless-volume-dc", lambda p: p.update(volume=None)))

    def test_the_nullable_fields(self):
        modal = "modal-base"
        self.refused(self.changed(modal, lambda p: p["volume"].update({"class": "STANDARD"})))
        self.refused(self.changed(modal, lambda p: p["volume"].update(size_gb=10)))
        self.refused(self.changed(modal, lambda p: p.update(data_center="EU-SE-1")))
        rp = "runpod-serverless-volume-dc"
        for change in (lambda p: p["volume"].update({"class": None}), lambda p: p["volume"].update(size_gb=None),
                       lambda p: p.update(data_center=None), lambda p: p["volume"].update({"class": "FAST"})):
            with self.subTest(change=change):
                self.refused(self.changed(rp, change))
        self.refused(self.changed(rp, lambda p: p.update(superseded_by="modal-base")))
        self.refused(self.changed(rp, lambda p: p.update(reason="old")))
        old = "runpod-serverless-volume-eu-se-1"
        self.refused(self.changed(old, lambda p: p.update(superseded_by=None)))
        self.refused(self.changed(old, lambda p: p.update(superseded_by="nope")))
        self.refused(self.changed(old, lambda p: p.update(reason=None)))
        self.assertEqual(len(presets.check(self.changed(rp, lambda p: p["levers"].update(arm_ahead_s=None)))), 5)

    def test_no_experimental_lever_in_a_preset(self):
        for lever in ("snapshot", "flashboot", "skip_mm_profiling"):
            with self.subTest(lever=lever):
                self.refused(self.changed("modal-base", lambda p: p["levers"].update({lever: True})))
        self.refused(self.changed("modal-base", lambda p: p["levers"].update(compile_cache=["nope"])))
        self.refused(self.changed("modal-base", lambda p: p["levers"].update(compile_cache="29012f5b")))

    def test_values_outside_their_form(self):
        rp = "runpod-serverless-volume-dc"
        for change in (lambda p: p.update(status="beta"), lambda p: p.update(provider="aws"),
                       lambda p: p.update(mode="pod"), lambda p: p.update(server="llamacpp"),
                       lambda p: p.update(parallel=0), lambda p: p.update(image="vllm/vllm-openai:latest"),
                       lambda p: p["model"].update(revision="main"), lambda p: p["rate"].update(usd_per_s=0.0002),
                       lambda p: p["rate"].update(usd_per_s="1.5"), lambda p: p["rate"].update(date="22/11/2026"),
                       lambda p: p["seconds"].update(ready_within=600), lambda p: p["seconds"].update(w1_s=True),
                       lambda p: p["gpu"].update(group=[]), lambda p: p["volume"].update(type="local"),
                       lambda p: p["measures"].append({"what": "x"})):
            with self.subTest(change=change):
                self.refused(self.changed(rp, change))

    def test_a_global_volume_is_recognized(self):
        doc = self.changed("runpod-serverless-volume-dc", lambda p: p["volume"].update(type="global"))
        found = presets.check(doc)
        self.assertEqual(presets.refusal(found["runpod-serverless-volume-dc"]), codes.GPU_VOLUME_UNSUPPORTED)

    def test_the_file_unreadable(self):
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "presets.json"
            bad.write_text("{not json")
            with self.assertRaises(codes.GpuError) as ctx:
                presets.load(bad)
            self.assertEqual(ctx.exception.code, codes.GPU_PRESET_INVALID)
            with self.assertRaises(codes.GpuError):
                presets.load(Path(tmp) / "missing.json")


class EnableTests(Base):
    def test_only_supported_presets_on_a_network_volume(self):
        found = presets.load()
        self.assertEqual(presets.enable(["modal-base", "runpod-serverless-volume-dc"], found),
                         ["modal-base", "runpod-serverless-volume-dc"])
        self.assertEqual(presets.enable(["gpu-local", "gpu-remote"], found), ["gpu-local", "gpu-remote"])
        self.assertEqual(presets.enable([], found), [])
        experimental = presets.check(self.changed("modal-base", lambda p: p.update(status="experimental")))
        glob = presets.check(self.changed("runpod-serverless-volume-dc", lambda p: p["volume"].update(type="global")))
        for names, table, code in (
                (["runpod-serverless-volume-eu-se-1"], found, codes.GPU_PRESET_SUPERSEDED),
                (["nope"], found, codes.GPU_PRESET_UNKNOWN),
                (["modal-base", "modal-base"], found, codes.GPU_PRESET_INVALID),
                (["modal-base"], experimental, codes.GPU_PRESET_NOT_ENABLED),
                (["runpod-serverless-volume-dc"], glob, codes.GPU_VOLUME_UNSUPPORTED)):
            with self.subTest(names=names, code=code):
                with self.assertRaises(codes.GpuError) as ctx:
                    presets.enable(names, table)
                self.assertEqual(ctx.exception.code, code)

    def test_the_codes_are_closed(self):
        self.assertEqual(len(set(codes.CODES)), 10)
        with self.assertRaises(ValueError):
            codes.GpuError("something_else")
        from markdown_for_llms.engine.codes import Code
        self.assertFalse((set(codes.CODES) - {codes.PROVIDER_NOT_IDLE}) & {c.value for c in Code})


class TheWheelTests(unittest.TestCase):
    def test_the_catalog_loads_from_the_installed_package(self):
        """`package-data` of the wheel (SOL-R6 = C-NB1 of the plan r1): an isolated process started outside the checkout
        loads the presets from the installed package. Green once the full check has installed the canonical wheel."""
        code = ("import json; from markdown_for_llms.gpu import presets; "
                "print(json.dumps([str(presets.PATH), sorted(presets.load())]))")
        with tempfile.TemporaryDirectory() as tmp:
            out = subprocess.run([sys.executable, "-I", "-B", "-c", code], capture_output=True, text=True, cwd=tmp,
                                 timeout=60)
        self.assertEqual(out.returncode, 0, out.stderr)
        path, names = json.loads(out.stdout)
        self.assertIn("site-packages", Path(path).parts)
        self.assertEqual(names, sorted(presets.load()))


if __name__ == "__main__":
    unittest.main()
