"""`mdf convert --preset NAME` (MFL051, plan r2.1 §6; D-4).

A conversion preset in `mdf convert` is the function of `mdf submit --preset` and of the API: its route and its
options, with only `--lang` and `--confirm-transit` beside it. One preset more is refused: a preset on a rented
GPU needs the session, the ledger and the caps of the worker, so it runs through the archive; the answer is the
closed code `preset_needs_archive`, the only new code of the tranche, with the command to use. Exit codes are
those of today: 0, 1, 2, 130.
"""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from markdown_for_llms import cli
from markdown_for_llms.engine.codes import Code, EngineError
from markdown_for_llms.services import REFUSALS

SECRET = "dl-synthetic-0123456789"


class PresetConvertTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.base = ["--data-root", str(self.root / "data"), "--model-root", str(self.root / "models")]
        self.text = self.root / "note.txt"
        self.text.write_text("Una riga.\n", encoding="utf-8")
        self.pdf = Path(__file__).resolve().parents[1] / "fixtures" / "conversion-corpus" / "inputs" / "C06-pdf-text-it.pdf"
        self.count = 0

    def mdf(self, *argv, options=()):
        out = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
            try:
                code = cli.main([*options, *self.base, *argv])
            except SystemExit as exc:
                code = exc.code
        text = out.getvalue()
        self.assertNotIn(SECRET, text)
        return code, json.loads(text) if text.strip() else None

    def out(self):
        self.count += 1
        return str(self.root / ("out-%d" % self.count))

    def captured(self):
        """The engine replaced by a recorder: (the calls, the patch)."""
        calls = []

        def convert(source, out, request, limits, **kw):
            calls.append({"request": request, "key": kw.get("datalab_key")})
            raise EngineError(Code.OUTPUT_INVALID, "the recorder stops here")
        return calls, mock.patch("markdown_for_llms.engine.core.convert", convert)

    def test_a_local_preset_converts_as_its_options_given_one_by_one(self):
        out = self.out()
        code, summary = self.mdf("convert", str(self.text), "--preset", "local-fast", "--out", out)
        self.assertEqual((code, summary["status"], summary["route"]["converter"]), (0, "complete", "text"))
        self.assertEqual((Path(out) / "document.md").read_text(), "Una riga.\n")
        plain = self.out()
        code, other = self.mdf("convert", str(self.text), "--out", plain)
        self.assertEqual((code, other["route"]), (0, summary["route"]))

    def test_only_lang_and_the_transit_beside_a_preset(self):
        code, summary = self.mdf("convert", str(self.text), "--preset", "local-fast", "--lang", "it", "--out", self.out())
        self.assertEqual(code, 0)
        for extra in (["--route", "marker"], ["--ocr", "force"], ["--datalab-mode", "lite"], ["--server-mode", "balanced"],
                      ["--server", "mine"], ["--spend-cap-usd", "1"]):
            with self.subTest(extra=extra[0]):
                out = self.out()
                code, answer = self.mdf("convert", str(self.text), "--preset", "local-fast", *extra, "--out", out)
                self.assertEqual((code, answer["status"], answer["error"]), (2, "refused", "invalid_setting"))
                self.assertIn("--preset takes only --lang and --confirm-transit", answer["detail"])
                self.assertFalse(Path(out).exists())

    def test_a_preset_on_a_rented_gpu_needs_the_archive(self):
        for name in ("quality-now", "batch"):
            with self.subTest(preset=name):
                out = self.out()
                code, answer = self.mdf("convert", str(self.pdf), "--preset", name, "--confirm-transit", "--out", out)
                self.assertEqual((code, answer["status"], answer["error"]), (1, "refused", "preset_needs_archive"))
                self.assertIn("mdf submit --preset NAME --wait FILE", answer["detail"])
                self.assertFalse(Path(out).exists())                    # nothing was converted, nothing was sent
        self.assertIn("preset_needs_archive", REFUSALS)

    def test_a_preset_nobody_defined(self):
        code, answer = self.mdf("convert", str(self.text), "--preset", "no-such-preset", "--out", self.out())
        self.assertEqual((code, answer["error"]), (1, "preset_unknown"))

    def test_a_preset_of_the_user(self):
        code, _ = self.mdf("presets", "add", "scans", "--route", "marker", "--ocr", "force")
        self.assertEqual(code, 0)
        calls, patch = self.captured()
        with patch:
            code, answer = self.mdf("convert", str(self.pdf), "--preset", "scans", "--lang", "it", "--out", self.out())
        self.assertEqual((code, answer["status"]), (2, "refused"))      # the recorder stops the conversion
        self.assertEqual({k: calls[0]["request"][k] for k in ("route", "ocr", "lang")}, {"route": "marker", "ocr": "force", "lang": "it"})

    def test_the_datalab_preset_takes_its_mode_and_the_key_of_the_keys_file(self):
        code, answer = self.mdf("convert", str(self.pdf), "--preset", "datalab", "--out", self.out())
        self.assertEqual((code, answer["error"]), (1, "datalab_key_missing"))            # no keys file: nothing leaves
        keys = self.root / "keys"
        keys.write_text("DATALAB_API_KEY=%s\n" % SECRET)
        keys.chmod(0o600)
        calls, patch = self.captured()
        with patch:
            self.mdf("convert", str(self.pdf), "--preset", "datalab", "--confirm-transit", "--datalab-cap-usd", "0.05",
                     "--out", self.out(), options=["--keys-file", str(keys)])
        request = calls[0]["request"]
        code, (shown) = self.mdf("presets", "show", "datalab")
        self.assertEqual(request["route"], "datalab")
        self.assertEqual(request["datalab"], {"mode": shown["options"]["datalab_mode"], "transit_confirmed": True, "cap_usd": "0.05"})
        self.assertEqual(calls[0]["key"].header(), {"X-API-Key": SECRET})

    def test_an_interrupted_conversion_is_130(self):
        def interrupted(*args, **kw):
            raise KeyboardInterrupt
        with mock.patch("markdown_for_llms.engine.core.convert", interrupted):
            code, answer = self.mdf("convert", str(self.text), "--preset", "local-fast", "--out", self.out())
        self.assertEqual((code, answer["status"]), (130, "interrupted"))


if __name__ == "__main__":
    unittest.main()
