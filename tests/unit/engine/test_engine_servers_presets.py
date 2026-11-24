"""Endpoints of servers.json that name a GPU preset (MFL023 A3, C-B2 and D3): only `kind`, `preset`, `resource`, `url`
and `key_env`; the preset fixes every other value, a value written by hand is refused, as are an unknown, superseded
or foreign preset, a missing table and two endpoints with one preset. The identity is computed on the expanded values,
and the engine imports nothing of the GPU package."""
import json
from pathlib import Path
import tempfile
import unittest

from markdown_for_llms.engine import servers
from markdown_for_llms.engine.codes import Code, EngineError
from markdown_for_llms.gpu import presets

ROOT = Path(__file__).resolve().parents[3]
TABLE = presets.endpoint_table()
RUNPOD = {"kind": "runpod", "preset": "runpod-serverless-volume-dc", "resource": "mdf-surya",
          "url": "https://ep123.api.runpod.ai"}
MODAL = {"kind": "modal", "preset": "modal-base", "resource": "mdf-surya",
         "url": "https://ws--mdf-surya-serve.modal.run"}


class Base(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)

    def refused(self, fn):
        with self.assertRaises(EngineError) as ctx:
            fn()
        self.assertEqual(ctx.exception.code, Code.SERVER_NOT_CONFIGURED)
        return ctx.exception

    def file(self, doc):
        path = self.root / "servers.json"
        path.write_text(json.dumps(doc))
        return path


class ExpansionTests(Base):
    def test_the_values_of_the_preset(self):
        ep = servers.endpoint("rp", RUNPOD, TABLE)
        fields = TABLE["runpod-serverless-volume-dc"]["fields"]
        for key, value in fields.items():
            with self.subTest(field=key):
                self.assertEqual(getattr(ep, key), value)
        self.assertEqual((ep.kind, ep.provider, ep.preset, ep.resource), ("runpod", "runpod",
                                                                           "runpod-serverless-volume-dc", "mdf-surya"))
        self.assertEqual(ep.key_env, ["RUNPOD_API_KEY"])
        modal = servers.endpoint("md", MODAL, TABLE)
        self.assertEqual((modal.gpu, modal.ready_within, modal.cold_start_s, modal.key_env),
                         ("L4,A10G", 900, 480, ["MODAL_PROXY_TOKEN_ID", "MODAL_PROXY_TOKEN_SECRET"]))
        same = servers.endpoint("rp", dict(RUNPOD, key_env=["RUNPOD_API_KEY"]), TABLE)
        self.assertEqual(same.key_env, ["RUNPOD_API_KEY"])

    def test_the_keys_of_a_preset_are_those_its_provider_reads(self):
        """C-DB-NB5 (MFL023 B10, a reversible choice of the supervision): next to `preset`, `key_env` is absent or the
        names the provider of the preset reads (the worker, its children and the relay read those), in their order;
        any other is `server_not_configured`."""
        for raw, names in ((RUNPOD, ["RUNPOD_API_KEY"]), (dict(RUNPOD, key_env="RUNPOD_API_KEY"), ["RUNPOD_API_KEY"]),
                           (dict(MODAL, key_env=["MODAL_PROXY_TOKEN_ID", "MODAL_PROXY_TOKEN_SECRET"]),
                            ["MODAL_PROXY_TOKEN_ID", "MODAL_PROXY_TOKEN_SECRET"])):
            with self.subTest(raw=raw):
                self.assertEqual(servers.endpoint("e", raw, TABLE).key_env, names)
        for raw in (dict(RUNPOD, key_env=["TEAM_RUNPOD_API_KEY"]), dict(RUNPOD, key_env=[]),
                    dict(MODAL, key_env=["MODAL_PROXY_TOKEN_SECRET", "MODAL_PROXY_TOKEN_ID"]),
                    dict(MODAL, key_env=["TEAM_MODAL_TOKEN_ID", "TEAM_MODAL_TOKEN_SECRET"])):
            with self.subTest(raw=raw):
                self.assertIn("key_env", str(self.refused(lambda: servers.endpoint("e", raw, TABLE))))

    def test_the_identity_is_that_of_the_values_written_by_hand(self):
        by_hand = dict(TABLE["modal-base"]["fields"], kind="modal", url=MODAL["url"])
        self.assertEqual(servers.endpoint("md", MODAL, TABLE).identity_sha256(),
                         servers.endpoint("md", by_hand).identity_sha256())

    def test_the_record_names_the_preset(self):
        record = servers.endpoint("rp", RUNPOD, TABLE).record()
        self.assertEqual((record["preset"], record["resource"]), ("runpod-serverless-volume-dc", "mdf-surya"))
        plain = servers.endpoint("rp", dict(TABLE["runpod-serverless-volume-dc"]["fields"], kind="runpod",
                                            url=RUNPOD["url"])).record()
        self.assertNotIn("preset", plain)
        self.assertNotIn("resource", plain)

    def test_a_value_fixed_by_the_preset_written_by_hand(self):
        for key, value in dict(TABLE["runpod-serverless-volume-dc"]["fields"], provider="runpod").items():
            with self.subTest(field=key):
                self.refused(lambda: servers.endpoint("rp", dict(RUNPOD, **{key: value}), TABLE))

    def test_refused_presets(self):
        cases = {"unknown": dict(RUNPOD, preset="nope"), "superseded": dict(RUNPOD, preset="runpod-serverless-volume-eu-se-1"),
                 "another provider": dict(RUNPOD, kind="modal"), "user kind": dict(RUNPOD, kind="user"),
                 "no resource": {k: v for k, v in RUNPOD.items() if k != "resource"},
                 "bad resource": dict(RUNPOD, resource="../x"), "no url": {k: v for k, v in RUNPOD.items() if k != "url"},
                 "a list": dict(RUNPOD, preset=["modal-base"])}
        for what, raw in cases.items():
            with self.subTest(case=what):
                self.refused(lambda: servers.endpoint("rp", raw, TABLE))
        err = self.refused(lambda: servers.endpoint("rp", RUNPOD))                 # without the table
        self.assertIn("table", err.detail)
        err = self.refused(lambda: servers.endpoint("rp", cases["superseded"], TABLE))
        self.assertIn("gpu_preset_superseded", err.detail)

    def test_the_file(self):
        path = self.file({"default": "rp", "servers": {"rp": RUNPOD, "md": MODAL, "cpu": {"kind": "cpu"}}})
        self.assertEqual(servers.load(path, presets=TABLE).preset, "runpod-serverless-volume-dc")
        self.assertEqual(servers.load(path, "cpu", presets=TABLE).kind, "cpu")
        self.assertEqual(servers.load(path, "cpu").kind, "cpu")                    # the table only for a preset
        self.refused(lambda: servers.load(path))
        twice = self.file({"default": "cpu", "servers": {"a": RUNPOD, "b": dict(RUNPOD, resource="other"),
                                                         "cpu": {"kind": "cpu"}}})
        self.refused(lambda: servers.load(twice, "cpu", presets=TABLE))


class BoundaryTests(unittest.TestCase):
    def test_the_engine_imports_nothing_of_the_gpu_package(self):
        for path in sorted((ROOT / "markdown_for_llms" / "engine").rglob("*.py")):
            text = path.read_text(encoding="utf-8")
            with self.subTest(module=path.name):
                self.assertNotIn("from ..gpu", text)
                self.assertNotIn("markdown_for_llms.gpu", text)


if __name__ == "__main__":
    unittest.main()
