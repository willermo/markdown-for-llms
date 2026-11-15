"""Stdlib checks of the source manifest gate; synthetic data only."""

import importlib.util
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]


def diagnostic(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts/diagnostics/run-a001-fase0-uv" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestDiagnosticGates(unittest.TestCase):
    def test_source_manifest_rejects_symlink_and_snapshot_hash_mismatch(self):
        tool = diagnostic("make_source_manifest")
        with tempfile.TemporaryDirectory(prefix="a001-unit-") as directory:
            root = Path(directory)
            file = root / "module.py"
            file.write_bytes(b"synthetic = 1\n")
            link = root / "link.py"
            link.symlink_to(file)
            with self.assertRaises(ValueError):
                tool.entry(link, root)
            item = tool.entry(file, root)
            with self.assertRaises(ValueError):
                tool.match_snapshot([item], {item["path"]: {"kind": "file", "sha256": "0" * 64}})
            with self.assertRaises(ValueError):
                tool.match_snapshot([item], {})

    def test_source_id_uses_canonical_payload_independent_of_key_order(self):
        tool = diagnostic("make_source_manifest")
        self.assertEqual(tool.canonical({"b": 2, "a": "città"}), tool.canonical({"a": "città", "b": 2}))
        self.assertNotEqual(tool.canonical({"a": 1}), tool.canonical({"a": 2}))


if __name__ == "__main__":
    unittest.main()
