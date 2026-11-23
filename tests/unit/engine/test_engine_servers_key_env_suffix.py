"""MFL046 (c): the names of the keys of an endpoint (`key_env`) end with a suffix of `envfile.KEY_SUFFIXES`, the
suffixes `mdf bench secret-scan` searches, so a key with another name can no longer escape the scan. The suffixes
live in the engine (which imports nothing of the bench) and the bench reads them from there."""
import os
from pathlib import Path
import tempfile
import unittest

from markdown_for_llms.bench import secretscan
from markdown_for_llms.engine import envfile, servers
from markdown_for_llms.engine.codes import Code, EngineError


def endpoint(**raw):
    return servers.endpoint("t", raw)


class KeyEnvSuffixTests(unittest.TestCase):
    def test_one_tuple_of_suffixes(self):
        self.assertIs(secretscan.KEY_SUFFIXES, envfile.KEY_SUFFIXES)
        self.assertEqual(envfile.KEY_SUFFIXES, ("_KEY", "_SECRET", "_TOKEN", "_TOKEN_ID", "_BEARER"))

    def test_names_without_a_suffix_are_refused(self):
        cases = ({"kind": "user", "url": "https://gpu.example", "key_env": "MY_GPU"},
                 {"kind": "user", "url": "https://gpu.example", "key_env": ["GPU_PASSWORD"]},
                 {"kind": "runpod", "url": "https://x.api.runpod.ai", "rate_usd_per_s": "0.0002", "key_env": "RUNPOD"},
                 {"kind": "modal", "url": "https://x.modal.run", "rate_usd_per_s": "0.0003",
                  "key_env": ["MODAL_PROXY_TOKEN_ID", "MODAL_PROXY"]})
        for raw in cases:
            with self.subTest(key_env=raw["key_env"]):
                with self.assertRaises(EngineError) as ctx:
                    endpoint(**raw)
                self.assertEqual(ctx.exception.code, Code.SERVER_NOT_CONFIGURED)
                self.assertIn("_KEY", ctx.exception.detail)

    def test_names_with_a_suffix_pass(self):
        for name in ("MY_GPU_KEY", "TEAM_SECRET", "GPU_TOKEN", "X_TOKEN_ID", "MODAL_PROXY_BEARER"):
            with self.subTest(name=name):
                self.assertEqual(endpoint(kind="user", url="https://gpu.example", key_env=name).key_env, [name])
        self.assertEqual(endpoint(kind="runpod", url="https://x.api.runpod.ai", rate_usd_per_s="0.0002").key_env,
                         ["RUNPOD_API_KEY"])

    def test_secret_scan_reads_the_key_of_such_an_endpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = Path(tmp) / "keys.env"
            env.write_text("MY_GPU_KEY=synthetic-gpu-key-0000\nPLAIN_SETTING=synthetic-other-0000\n")
            os.chmod(env, 0o600)
            self.assertEqual(secretscan.keys_bytes(env), [b"synthetic-gpu-key-0000"])


if __name__ == "__main__":
    unittest.main()
