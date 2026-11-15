"""The weights of the Marker route (`engine/weights.py`, criterion 12.12 and PB11 of MFL032).

The checks run on a small synthetic manifest; one test verifies the real manifest against the
model root of the full check (`FULL_CHECK_MODEL_ROOT`, precondition W), and fails, saying how to
prepare it, when the variable is missing. No test downloads anything: `prepare` gets a fake
transfer in place of curl.
"""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from markdown_for_llms.engine import weights
from markdown_for_llms.engine.codes import Code, EngineError

BLOB = b"synthetic onnx weights\n" * 100
META = b'{"synthetic": true}\n'


def git_sha1(data):
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def fake_manifest():
    return {"schema": "mdf-engine-weights/1", "indra_ocr": {"commit": "c0ffee", "directory": "indra-ocr/c0ffee"},
            "font": "fonts/font.ttf",
            "files": [
                {"path": "indra-ocr/c0ffee/det.onnx", "bytes": len(BLOB), "sha256": hashlib.sha256(BLOB).hexdigest(),
                 "url": "https://example.invalid/det.onnx", "set": "base"},
                {"path": "indra-ocr/c0ffee/meta.json", "bytes": len(META), "sha256": hashlib.sha256(META).hexdigest(),
                 "git_sha1": git_sha1(META), "url": "https://example.invalid/meta.json", "set": "base"},
                {"path": "fonts/font.ttf", "bytes": 4, "sha256": hashlib.sha256(b"font").hexdigest(),
                 "url": "https://example.invalid/font.ttf", "set": "font"}]}


CONTENT = {"https://example.invalid/det.onnx": BLOB, "https://example.invalid/meta.json": META,
           "https://example.invalid/font.ttf": b"font"}


class Done:
    def __init__(self, code):
        self.returncode = code


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "models"
        patcher = mock.patch.object(weights, "manifest", fake_manifest)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(self.tmp.cleanup)

    def write_all(self):
        for entry in fake_manifest()["files"]:
            path = self.root / entry["path"]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(CONTENT[entry["url"]])


class VerifyTests(Base):
    def test_complete_root_verifies(self):
        self.write_all()
        report = weights.verify(self.root)
        self.assertEqual(report["files"], 3)
        self.assertEqual(report["bytes"], len(BLOB) + len(META) + 4)

    def test_missing_root_and_missing_file(self):
        with self.assertRaises(EngineError) as e:
            weights.verify(self.root)
        self.assertEqual(e.exception.code, Code.WEIGHTS_MISSING)
        self.write_all()
        (self.root / "fonts/font.ttf").unlink()
        with self.assertRaises(EngineError) as e:
            weights.verify(self.root)
        self.assertEqual(e.exception.code, Code.WEIGHTS_MISSING)

    def test_altered_truncated_and_same_size_files_are_mismatches(self):
        for change in (lambda p: p.write_bytes(BLOB[:-1]),                     # shorter
                       lambda p: p.write_bytes(BLOB[:-1] + b"?"),              # same size, other bytes
                       lambda p: p.write_bytes(BLOB + b"x")):                  # longer
            self.write_all()
            change(self.root / "indra-ocr/c0ffee/det.onnx")
            with self.subTest(), self.assertRaises(EngineError) as e:
                weights.verify(self.root)
            self.assertEqual(e.exception.code, Code.WEIGHTS_MISMATCH)

    def test_git_sha1_is_checked_for_the_json_files(self):
        self.write_all()
        m = fake_manifest()
        m["files"][1]["git_sha1"] = "0" * 40
        with mock.patch.object(weights, "manifest", lambda: m), self.assertRaises(EngineError) as e:
            weights.verify(self.root)
        self.assertEqual(e.exception.code, Code.WEIGHTS_MISMATCH)

    def test_links_are_refused_for_files_directories_and_the_root(self):
        self.write_all()
        elsewhere = Path(self.tmp.name) / "elsewhere.onnx"
        elsewhere.write_bytes(BLOB)
        target = self.root / "indra-ocr/c0ffee/det.onnx"
        target.unlink()
        target.symlink_to(elsewhere)                     # same bytes, through a link
        with self.assertRaises(EngineError) as e:
            weights.verify(self.root)
        self.assertEqual(e.exception.code, Code.WEIGHTS_MISMATCH)
        target.unlink()
        self.write_all()
        moved = Path(self.tmp.name) / "fonts-real"
        (self.root / "fonts").rename(moved)
        (self.root / "fonts").symlink_to(moved)          # a linked directory
        with self.assertRaises(EngineError) as e:
            weights.verify(self.root)
        self.assertEqual(e.exception.code, Code.WEIGHTS_MISMATCH)
        link = Path(self.tmp.name) / "link-root"
        link.symlink_to(self.root)
        with self.assertRaises(EngineError) as e:
            weights.verify(link)
        self.assertEqual(e.exception.code, Code.WEIGHTS_MISSING)

    def test_a_directory_in_place_of_a_file(self):
        self.write_all()
        target = self.root / "fonts/font.ttf"
        target.unlink()
        target.mkdir()
        with self.assertRaises(EngineError) as e:
            weights.verify(self.root)
        self.assertEqual(e.exception.code, Code.WEIGHTS_MISMATCH)

    def test_command_line_exit_status(self):
        self.assertEqual(weights.main(["verify", str(self.root)]), 1)
        self.write_all()
        self.assertEqual(weights.main(["verify", str(self.root)]), 0)
        self.assertEqual(weights.main(["verify"]), 2)


class PrepareTests(Base):
    def fetcher(self, plan=None):
        """A fake transfer: `plan[url]` is a list of steps, each 'ok', 'fail' or 'half' (half the
        bytes, exit 18 as curl after a cut connection), consumed one per call; resumes append."""
        calls, plan = [], plan or {}

        def fetch(url, partial):
            calls.append(url)
            step = plan[url].pop(0) if plan.get(url) else "ok"
            data = CONTENT[url]
            have = partial.read_bytes() if partial.exists() else b""
            if step == "fail":
                return Done(7)
            if step == "half":
                partial.write_bytes(have + data[len(have):len(have) + len(data) // 2])
                return Done(18)
            if step == "bad":
                partial.write_bytes(b"\0" * len(data))
                return Done(0)
            partial.write_bytes(have + data[len(have):])
            return Done(0)
        return fetch, calls

    def test_downloads_into_partial_files_and_renames(self):
        fetch, calls = self.fetcher()
        report = weights.prepare(self.root, fetch=fetch, log=lambda line: None)
        self.assertEqual((report["downloaded"], report["kept"]), (3, 0))
        self.assertEqual(len(calls), 3)
        weights.verify(self.root)
        self.assertEqual(list(self.root.rglob("*.partial")), [])

    def test_verified_files_are_kept_and_not_fetched_again(self):
        self.write_all()
        (self.root / "fonts/font.ttf").unlink()
        fetch, calls = self.fetcher()
        report = weights.prepare(self.root, fetch=fetch, log=lambda line: None)
        self.assertEqual((report["downloaded"], report["kept"]), (1, 2))
        self.assertEqual(calls, ["https://example.invalid/font.ttf"])

    def test_bounded_retries_resume_the_partial_file(self):
        url = "https://example.invalid/det.onnx"
        fetch, calls = self.fetcher({url: ["half", "ok"]})
        weights.prepare(self.root, fetch=fetch, log=lambda line: None)
        self.assertEqual(calls.count(url), 2)
        self.assertEqual((self.root / "indra-ocr/c0ffee/det.onnx").read_bytes(), BLOB)

    def test_stops_after_the_last_transfer(self):
        url = "https://example.invalid/det.onnx"
        fetch, calls = self.fetcher({url: ["fail"] * weights.ATTEMPTS})
        with self.assertRaises(EngineError) as e:
            weights.prepare(self.root, fetch=fetch, log=lambda line: None)
        self.assertEqual(e.exception.code, Code.WEIGHTS_MISSING)
        self.assertEqual(calls.count(url), weights.ATTEMPTS)
        self.assertFalse((self.root / "indra-ocr/c0ffee/det.onnx").exists())

    def test_a_present_file_that_differs_is_left_untouched(self):
        self.write_all()
        target = self.root / "indra-ocr/c0ffee/det.onnx"
        target.write_bytes(b"local file")
        fetch, calls = self.fetcher()
        with self.assertRaises(EngineError) as e:
            weights.prepare(self.root, fetch=fetch, log=lambda line: None)
        self.assertEqual(e.exception.code, Code.WEIGHTS_MISMATCH)
        self.assertEqual(target.read_bytes(), b"local file")
        self.assertEqual(calls, [])

    def test_a_downloaded_file_that_differs_is_refused_and_removed(self):
        url = "https://example.invalid/meta.json"
        fetch, _ = self.fetcher({url: ["bad"]})
        with self.assertRaises(EngineError) as e:
            weights.prepare(self.root, fetch=fetch, log=lambda line: None)
        self.assertEqual(e.exception.code, Code.WEIGHTS_MISMATCH)
        self.assertFalse((self.root / "indra-ocr/c0ffee/meta.json").exists())
        self.assertFalse((self.root / "indra-ocr/c0ffee/meta.json.partial").exists())

    def test_a_partial_file_longer_than_the_file_starts_again(self):
        partial = self.root / "fonts/font.ttf.partial"
        partial.parent.mkdir(parents=True)
        partial.write_bytes(b"too long for the font")
        fetch, _ = self.fetcher()
        weights.prepare(self.root, fetch=fetch, log=lambda line: None)
        self.assertEqual((self.root / "fonts/font.ttf").read_bytes(), b"font")

    def test_curl_is_https_only_with_bounded_retries_and_resume(self):
        with mock.patch.object(weights.subprocess, "run", return_value=Done(0)) as run:
            weights._curl("https://example.invalid/x", Path("/nonexistent/x.partial"))
        argv = run.call_args.args[0]
        self.assertEqual(argv[0], "curl")
        for flag, value in (("--proto", "=https"), ("--proto-redir", "=https"), ("--retry", str(weights.CURL_RETRIES)),
                            ("-C", "-")):
            self.assertEqual(argv[argv.index(flag) + 1], value)


class RealManifestTests(unittest.TestCase):
    def test_manifest_lists_the_18_files_of_indra_v110_and_gonoto(self):
        m = weights.manifest()
        self.assertEqual(len(m["files"]), 18)
        self.assertEqual(sum(e["bytes"] for e in m["files"]), 110663817)
        self.assertEqual(m["indra_ocr"]["commit"], "bd02167fdf2005a19962cc15188f8496a2c87d3b")
        for entry in m["files"]:
            self.assertTrue(entry["url"].startswith("https://"), entry["path"])
            self.assertRegex(entry["sha256"], "^[0-9a-f]{64}$")
        self.assertEqual({e["path"].rsplit("/", 1)[1] for e in m["files"] if "git_sha1" in e},
                         {"meta.json", "math_meta.json"})

    def test_model_root_of_the_full_check_verifies(self):
        root = os.environ.get("FULL_CHECK_MODEL_ROOT")
        if not root:
            self.fail("FULL_CHECK_MODEL_ROOT is not set: run the suite from scripts/full-check.sh, with the weights "
                      "prepared by `mdf prepare --weights --model-root DIR` (docs/how-to/uv-environment.md)")
        self.assertEqual(weights.verify(root)["files"], 18)


if __name__ == "__main__":
    unittest.main()
