"""The pins of a GPU preset on RunPod (MFL023 A3, C-B2 of the plan r1): with a preset `pin_checks` adds the data center,
class and size of its volume, on the facts already read (no call more); they are invariant, so an arm refuses before
its PATCH an endpoint whose volume differs from the preset, and checks them again on its read-back. Without a preset
(the command line) the pins are those of MFL033. No network: `httpx.MockTransport`."""
import json
import unittest

import httpx

from markdown_for_llms.engine import envfile
from markdown_for_llms.gpu import presets
from markdown_for_llms.gpu import providers as pv

PRESET = presets.load()["runpod-serverless-volume-dc"]
NEW = {"preset_data_center", "preset_volume_class", "preset_volume_size"}


class FakeRunPod:
    """The REST v2 of one endpoint at rest in CA-MTL-1 with its volume, as read on 23/11 (plan r1 evidence)."""

    def __init__(self, volume=None):
        self.ep = {"id": "ep1", "name": "mdf-surya", "type": "LOAD_BALANCER", "image": pv.IMAGE,
                   "entrypoint": pv.ENTRYPOINT, "env": pv.base_env(), "workers": pv.workers_block(0),
                   "scaling": {"type": "REQUEST_COUNT", "requestCount": 32}, "flashboot": "OFF",
                   "ports": ["8000/http"], "dataCenterIds": ["CA-MTL-1"], "mounts": {"network": [{"volumeId": "vol1"}]},
                   "gpu": {"pools": [pv.POOL], "minCudaVersion": pv.CUDA_MIN, "count": 1}}
        self.volume = dict({"id": "vol1", "name": pv.VOLUME, "dataCenter": "CA-MTL-1", "type": "HIGH_PERFORMANCE",
                            "size": 10}, **(volume or {}))
        self.calls = []

    def facts(self):
        return ([self.volume], [{"id": "NVIDIA RTX A5000", "pool": pv.POOL, "secure": True,
                                 "price": {"serverless": 0.75}}],
                [{"id": "CA-MTL-1", "networkVolumeTypes": ["HIGH_PERFORMANCE"]}])

    def __call__(self, request):
        path, method = request.url.path.split("/v2", 1)[1], request.method
        self.calls.append((method, path))
        if path == "/serverless" and method == "GET":
            return httpx.Response(200, json={"endpoints": [self.ep], "pagination": {}})
        if path == "/serverless/ep1" and method == "GET":
            return httpx.Response(200, json=self.ep)
        if path == "/serverless/ep1" and method == "PATCH":
            body = json.loads(request.content)
            if "workers" in body:
                body["workers"] = dict(self.ep["workers"], **body["workers"])
            self.ep.update(body)
            return httpx.Response(200, json=self.ep)
        volumes, gpus, dcs = self.facts()
        if path == "/network-volumes":
            return httpx.Response(200, json={"networkVolumes": volumes})
        if path == "/catalog/gpus":
            return httpx.Response(200, json={"gpus": gpus})
        if path == "/catalog/datacenters":
            return httpx.Response(200, json={"dataCenters": dcs})
        return httpx.Response(404, json={})


def runpod(fake):
    return pv.RunPod(envfile.Secret("synthetic-key-0000"), "mdf-surya", transport=httpx.MockTransport(fake))


class PinTests(unittest.TestCase):
    def test_the_pins_of_the_preset_on_the_facts_read(self):
        fake = FakeRunPod()
        checks = pv.pin_checks(fake.ep, *fake.facts(), preset=PRESET)
        self.assertTrue(all(checks.values()), checks)
        self.assertLessEqual(NEW, set(checks))
        self.assertFalse(NEW & set(pv.pin_checks(fake.ep, *fake.facts())))       # the command line: as in MFL033
        for change, pin in (({"dataCenter": "EU-SE-1"}, "preset_data_center"),
                            ({"type": "STANDARD"}, "preset_volume_class"), ({"size": 20}, "preset_volume_size")):
            with self.subTest(change=change):
                bad = FakeRunPod(change)
                checks = pv.pin_checks(bad.ep, *bad.facts(), preset=PRESET)
                self.assertEqual(NEW - {k for k, ok in checks.items() if ok}, {pin})
                self.assertIn(pin, pv.invariant_failures(checks))
                self.assertNotIn(pin, pv.PIN_ALIGNABLE)

    def test_no_volume_of_the_name(self):
        fake = FakeRunPod({"name": "another"})
        failed = pv.invariant_failures(pv.pin_checks(fake.ep, *fake.facts(), preset=PRESET))
        self.assertLessEqual(NEW, set(failed))

    def test_the_superseded_preset_differs_from_the_endpoint_of_ca_mtl_1(self):
        fake = FakeRunPod()
        old = presets.load()["runpod-serverless-volume-eu-se-1"]
        self.assertEqual(sorted(pv.invariant_failures(pv.pin_checks(fake.ep, *fake.facts(), preset=old))),
                         ["preset_data_center", "preset_volume_class"])


class ArmTests(unittest.TestCase):
    def test_an_arm_with_the_preset(self):
        fake = FakeRunPod()
        rp = runpod(fake)
        rp.arm("", False, pv.DEFAULT_LEVERS, preset=PRESET)
        self.assertEqual(fake.ep["workers"]["max"], 1)
        self.assertEqual(len([c for c in fake.calls if c[0] == "PATCH"]), 1)

    def test_an_endpoint_outside_the_preset_is_never_armed(self):
        for change in ({"dataCenter": "EU-SE-1"}, {"type": "STANDARD"}, {"size": 20}):
            with self.subTest(change=change):
                fake = FakeRunPod(change)
                with self.assertRaises(pv.ProviderError) as ctx:
                    runpod(fake).arm("", False, pv.DEFAULT_LEVERS, preset=PRESET)
                self.assertIn("preset_", str(ctx.exception))
                self.assertFalse([c for c in fake.calls if c[0] == "PATCH"])
                if "dataCenter" not in change:                  # without a preset: the pins of MFL033 only
                    runpod(FakeRunPod(change)).arm("", False, pv.DEFAULT_LEVERS)
