"""`servers.load(..., host_loopback=False)`: an endpoint of kind `local` from a container (MFL051, plan r2.1 J.7.4).

A server of kind `local` listens on the loopback of the host, which a container of the product does not reach.
The refusal is a policy of the product, decided by the caller: `engine/servers.py` only takes the parameter. The
callers of the product pass what `containers.in_product_image()` says.
"""
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from markdown_for_llms.engine import containers, servers
from markdown_for_llms.engine.codes import EngineError


class HostLoopbackTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.path = self.root / "servers.json"
        self.path.write_text(json.dumps({"default": "fake", "servers": {
            "fake": {"kind": "local", "url": "http://127.0.0.1:23463"},
            "mine": {"kind": "user", "url": "https://gpu.example.org", "key_env": "MY_GPU_KEY"}}}))

    def test_a_local_endpoint_is_refused_only_when_the_caller_says_it_cannot_reach_the_host(self):
        self.assertEqual(servers.load(self.path).kind, "local")                  # the default: as before
        self.assertEqual(servers.load(self.path, host_loopback=True).kind, "local")
        with self.assertRaises(EngineError) as ctx:
            servers.load(self.path, host_loopback=False)
        self.assertEqual(ctx.exception.code.value, "server_not_configured")
        self.assertIn("loopback of the host", ctx.exception.detail)
        self.assertNotIn("23463", ctx.exception.detail)                          # the name of the entry, not its URL
        # the other kinds are the same either way
        self.assertEqual(servers.load(self.path, "mine", host_loopback=False).kind, "user")

    def test_the_module_holds_no_fact_of_a_container(self):
        text = Path(servers.__file__).read_text()
        for word in ("/run/mdf", "/opt/mdf", "in_product_image", "inputs.json"):
            self.assertNotIn(word, text)

    def test_the_callers_of_the_product_pass_what_their_image_says(self):
        self.assertTrue(containers.in_product_image(self.path))                  # a recorded file: an image of the product
        self.assertFalse(containers.in_product_image(self.root / "absent"))
        from markdown_for_llms import cli
        from types import SimpleNamespace
        args = SimpleNamespace(servers=str(self.path), server="fake", env_file=None, keys_file=None, server_mode="balanced",
                               confirm_transit=False, server_cap_usd=None, spend_ledger=None, spend_cap_usd=None)
        with mock.patch.object(containers, "in_product_image", return_value=False):
            self.assertEqual(cli.prepare_server(args)[0].kind, "local")
        with mock.patch.object(containers, "in_product_image", return_value=True), self.assertRaises(EngineError) as ctx:
            cli.prepare_server(args)
        self.assertEqual(ctx.exception.code.value, "server_not_configured")

    def test_the_bench_passes_it_at_the_start_of_a_run_and_on_resumption(self):
        """CARB-NB2: `mdf bench run --resume` loads the endpoint of the run again (`load_server`). In a container of
        the product a `local` endpoint is refused there as at the start, before anything waits for a server that the
        container cannot reach."""
        from markdown_for_llms.bench import run
        state = {"server": {"servers_file": str(self.path), "name": "fake",
                            "identity_sha256": servers.load(self.path, "fake").identity_sha256()}}
        with mock.patch.object(containers, "in_product_image", return_value=False):
            endpoint, keys = run.load_server(state, None)                        # on the host: the run goes on
        self.assertEqual((endpoint.kind, keys), ("local", {}))
        with mock.patch.object(containers, "in_product_image", return_value=True), self.assertRaises(EngineError) as ctx:
            run.load_server(state, None)
        self.assertEqual(ctx.exception.code.value, "server_not_configured")
        self.assertIn("loopback of the host", ctx.exception.detail)
        # the two callers of the bench are the only ones, and both say what their image is
        text = Path(run.__file__).read_text()
        self.assertEqual(text.count("servers.load("), 2)
        self.assertEqual(text.count("host_loopback=not containers.in_product_image()"), 2)


class ExampleTests(unittest.TestCase):
    """servers.example.json, next to .env.example: what a user copies to the state folder
    (`config/servers.json`). Every entry is one the product accepts, and none holds a secret."""

    def test_every_entry_of_the_example_is_accepted(self):
        from markdown_for_llms.gpu import presets
        example = Path(__file__).resolve().parents[3] / "servers.example.json"
        doc = json.loads(example.read_text())
        self.assertIn(doc["default"], doc["servers"])
        table = presets.endpoint_table()
        for name, entry in doc["servers"].items():
            with self.subTest(endpoint=name):
                ep = servers.load(example, name, presets=table, host_loopback=False)
                self.assertEqual(ep.provider, entry["kind"])
                self.assertEqual(set(entry), {"kind", "preset", "resource", "url", "key_env"})      # names of keys only
                for key in ep.key_env:
                    self.assertRegex(key, r"_(KEY|SECRET|TOKEN|TOKEN_ID|BEARER)$")
        self.assertEqual(sorted(e["preset"] for e in doc["servers"].values()), ["modal-base", "runpod-serverless-volume-dc"])


if __name__ == "__main__":
    unittest.main()
