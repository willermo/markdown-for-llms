"""txt with a UTF-16 or UTF-32 byte order mark (M5 of MFL026, Δ3 of the plan): recognized by
`formats.signature` and decoded by the `text` route, the 4-byte marks before the 2-byte ones, a
binary or a text with C0 controls behind a mark refused, the UTF-8 form capped before it is
written. Synthetic inputs; the engine and the upload of the product as they run."""
import codecs
import io
import math
import os
from pathlib import Path
import random
import struct
import tempfile
import unittest

from markdown_for_llms.engine import formats, text
from markdown_for_llms.engine.codes import Code, EngineError
from markdown_for_llms.engine.core import convert
from markdown_for_llms.services import Refused, Service
from markdown_for_llms.settings import Settings

# Latin, Cyrillic, Greek, a character outside the BMP (a surrogate pair in UTF-16), CRLF endings
SAMPLE = ("Ліс шумить, і вітер віє.\r\nThe forest — ελληνικά: αβγ, Čapek ř ů.\r\n"
          "Tab\there, form feed\f and vertical tab\v stay.\r\nEmoji \U0001F600 at the end") * 40
ORACLE = SAMPLE.replace("\r\n", "\n") + "\n"
MARKS = {"utf-16-le": (b"\xff\xfe", "utf_16"), "utf-16-be": (b"\xfe\xff", "utf_16"),
         "utf-32-le": (b"\xff\xfe\x00\x00", "utf_32"), "utf-32-be": (b"\x00\x00\xfe\xff", "utf_32")}


def encoded(codec, body=SAMPLE):
    mark, _ = MARKS[codec]
    return mark + body.encode(codec)


def pcm(amplitude, samples=8000):
    """16-bit little-endian PCM of a sine at low amplitude: valid UTF-16 code units, no NUL."""
    return b"".join(struct.pack("<h", int(amplitude * math.sin(i / 7)) or 1) for i in range(samples))


class Base(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.n = 0

    def file(self, data, name="t.txt"):
        self.n += 1
        path = self.root / f"{self.n}-{name}"
        path.write_bytes(data)
        return path

    def convert(self, data, **limits):
        self.n += 1
        out = self.root / f"out{self.n}"
        result = convert(self.file(data), out, {}, {"output_mib": 64, **limits}, work=self.root / "work")
        return result, out

    def refused(self, data):
        with self.assertRaises(EngineError) as caught:
            formats.detect(self.file(data))
        self.assertEqual(caught.exception.code, Code.FORMAT_MISMATCH)
        with self.assertRaises(EngineError) as caught:
            text.decode(data)
        self.assertEqual(caught.exception.code, Code.FORMAT_MISMATCH)


class EngineTests(Base):
    def test_utf16_and_utf32_through_the_engine(self):
        for codec, (_, name) in MARKS.items():
            with self.subTest(codec=codec):
                data = encoded(codec)
                self.assertEqual(formats.detect(self.file(data))["signature"], f"text ({name} BOM)")
                result, out = self.convert(data)
                self.assertEqual(result["status"], "complete", result.get("error"))
                self.assertEqual((result["text"]["encoding"], result["text"]["bom"]), (name, True))
                self.assertEqual((out / "document.md").read_bytes(), ORACLE.encode("utf-8"))

    def test_utf32_le_is_not_read_as_utf16(self):
        """FF FE 00 00 starts with the UTF-16 LE mark: the 4-byte mark is looked for first."""
        data = encoded("utf-32-le")
        self.assertEqual(text.decode(data)[1]["encoding"], "utf_32")

    def test_a_surrogate_pair_cut_by_the_probe_is_no_mismatch(self):
        body = "a" * (formats.TEXT_PROBE // 2 - 2) + "\U0001F600" + "tail"
        data = encoded("utf-16-le", body)
        head = data[:formats.TEXT_PROBE]
        with self.assertRaises(UnicodeDecodeError):         # a strict decoding of the probe alone fails
            head[2:].decode("utf-16-le", "strict")
        self.assertEqual(formats.detect(self.file(data))["signature"], "text (utf_16 BOM)")

    def test_binaries_behind_a_mark_are_refused(self):
        rng = random.Random(7)
        noise = bytes(rng.randrange(256) for _ in range(20000))
        for data in (b"\xff\xfe" + noise, b"\xfe\xff" + noise, b"\xff\xfe" + pcm(30000),
                     b"\xff\xfe" + pcm(200)):                 # low amplitude: valid units, stopped by C0
            with self.subTest(data=data[:8]):
                self.refused(data)

    def test_control_characters_behind_a_mark(self):
        for bad in ("\x01", "\x1b[31m", "\x1a"):              # an ANSI escape, a DOS end of file
            with self.subTest(bad=bad):
                self.refused(encoded("utf-16-le", "Riga di log " + bad + " fine\n"))
        self.assertEqual(text.decode(encoded("utf-16-be", "a\tb\fc\vd\r\n"))[0], "a\tb\fc\vd\n")

    def test_a_cp1252_text_ending_with_a_dos_end_of_file_keeps_the_old_path(self):
        data = ("Città è più bella: perché sì. " * 20).encode("cp1252") + b"\x1a"
        result, out = self.convert(data)
        self.assertEqual(result["status"], "complete", result.get("error"))
        self.assertFalse(result["text"]["bom"])

    def test_the_utf8_form_over_the_output_cap_is_not_written(self):
        # 400 000 characters of the BMP above U+07FF: 800 000 bytes in UTF-16, 1 200 000 in UTF-8
        data = encoded("utf-16-le", "中" * 400000)
        self.assertLess(len(data), 1048576)
        result, out = self.convert(data, output_mib=1)
        self.assertEqual((result["status"], result["error"]["code"]), ("failed", "output_limit"))
        self.assertFalse((out / "document.md").exists())
        stage = self.root / "stage"                       # the route itself writes nothing
        stage.mkdir()
        with self.assertRaises(EngineError) as caught:
            text.convert(self.file(data), stage, 1)
        self.assertEqual(caught.exception.code, Code.OUTPUT_LIMIT)
        self.assertEqual(list(stage.iterdir()), [])
        self.assertTrue(text.convert(self.file(data), stage, 2)["bom"])


class UploadTests(Base):
    """The upload of the product detects the format with the engine (`services.py`)."""

    def setUp(self):
        super().setUp()
        settings = Settings(data_root=str(self.root / "data"), model_root=str(self.root / "models"))
        self.service = Service(settings)
        self.addCleanup(self.service.storage.close)

    def test_a_utf16_text_is_admitted(self):
        job = self.service.submit(io.BytesIO(encoded("utf-16-le")), "notes.txt")
        self.assertEqual((job["format"], job["state"]), ("txt", "queued"))

    def test_low_amplitude_pcm_named_txt_is_refused(self):
        with self.assertRaises(Refused) as raised:
            self.service.submit(io.BytesIO(b"\xff\xfe" + pcm(200)), "sound.txt")
        self.assertEqual(raised.exception.code, "format_mismatch")


if __name__ == "__main__":
    unittest.main()
