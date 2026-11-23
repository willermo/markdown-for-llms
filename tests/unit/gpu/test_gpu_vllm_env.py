"""MFL049 (c): the OIDC token of Modal's runtime (`MODAL_IDENTITY_TOKEN`, put in every container, not a key of the user)
never reaches the environment of vLLM or of the download of the weights: one builder, `watchdog.vllm_env`, used by
the download, by `prepare` and by the boot of a memory snapshot."""
import ast
from contextlib import redirect_stdout
import io
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from markdown_for_llms.gpu import watchdog

GPU = Path(__file__).resolve().parents[3] / "markdown_for_llms" / "gpu"


class VllmEnvTests(unittest.TestCase):
    def test_the_builder(self):
        base = {"PATH": "/usr/bin", "MODAL_IDENTITY_TOKEN": "an-oidc-token", "MODAL_TASK_ID": "ta-1"}
        env = watchdog.vllm_env(base, HF_HOME="/w", HF_HUB_OFFLINE="1")
        self.assertEqual(env, {"PATH": "/usr/bin", "MODAL_TASK_ID": "ta-1", "HF_HOME": "/w", "HF_HUB_OFFLINE": "1"})
        self.assertIn("MODAL_IDENTITY_TOKEN", base)                    # the base is never changed
        self.assertNotIn("MODAL_IDENTITY_TOKEN", watchdog.vllm_env(base, MODAL_IDENTITY_TOKEN="again"))

    def test_the_child_of_the_download_has_no_token(self):
        probe = [sys.executable, "-I", "-c", "import os, sys; sys.exit(3 if 'MODAL_IDENTITY_TOKEN' in os.environ else 0)"]
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"MODAL_IDENTITY_TOKEN": "an-oidc-token"}), \
                redirect_stdout(io.StringIO()) as out:
            code = watchdog.download(watchdog.Standalone(), tmp, 60, tick_s=5, argv=probe)
        self.assertEqual(code, 0)
        self.assertNotIn("an-oidc-token", out.getvalue())

    def test_every_environment_of_vllm_is_built_by_it(self):
        """No `dict(os.environ, ...)` or `dict(env, ...)` with HF_HOME left in the watchdog or the snapshot: the three
        environments of vLLM and of the download go through `vllm_env`."""
        for name, expected in (("watchdog", 2), ("snapshot", 1)):
            tree = ast.parse((GPU / (name + ".py")).read_text(encoding="utf-8"))
            built, direct = 0, 0
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call) or not any(k.arg == "HF_HOME" for k in node.keywords):
                    continue
                func = node.func
                called = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
                built += called == "vllm_env"
                direct += called == "dict"
            with self.subTest(module=name):
                self.assertEqual((built, direct), (expected, 0))


if __name__ == "__main__":
    unittest.main()
