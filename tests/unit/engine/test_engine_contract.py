"""The output contract (`engine/contract.py`) and `convert` without external tools (txt)."""
import json
import os
from pathlib import Path
import tempfile
import unittest

from markdown_for_llms.engine import contract
from markdown_for_llms.engine.codes import Code, EngineError
from markdown_for_llms.engine.core import availability, convert


class Base(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.stage = self.root / "stage"
        (self.stage / "assets").mkdir(parents=True)

    def code(self, call):
        with self.assertRaises(EngineError) as ctx:
            call()
        return ctx.exception.code


class ScanTests(Base):
    def test_checks_of_the_staged_output(self):
        doc = self.stage / "document.md"
        doc.write_bytes(b"ok\n")
        (self.stage / "assets" / "asset-0001.png").write_bytes(b"png")
        self.assertEqual([r for r, _ in contract.scan(self.stage, 1)], ["assets/asset-0001.png", "document.md"])
        doc.write_bytes(b"\xff\xfe")
        self.assertEqual(self.code(lambda: contract.scan(self.stage, 1)), Code.OUTPUT_INVALID)
        doc.write_bytes(b"a\x00b")
        self.assertEqual(self.code(lambda: contract.scan(self.stage, 1)), Code.OUTPUT_INVALID)
        doc.write_bytes(b" \n\n")
        self.assertEqual(self.code(lambda: contract.scan(self.stage, 1)), Code.OUTPUT_EMPTY)
        doc.write_bytes(b"ok\n")
        os.symlink("/etc/passwd", self.stage / "assets" / "link.png")
        self.assertEqual(self.code(lambda: contract.scan(self.stage, 1)), Code.OUTPUT_INVALID)
        (self.stage / "assets" / "link.png").unlink()
        (self.stage / "other.txt").write_text("x")
        self.assertEqual(self.code(lambda: contract.scan(self.stage, 1)), Code.OUTPUT_INVALID)
        (self.stage / "other.txt").unlink()
        (self.stage / "assets" / "asset-0002.bin").write_bytes(b"x" * (2 << 20))
        self.assertEqual(self.code(lambda: contract.scan(self.stage, 1)), Code.OUTPUT_LIMIT)

    def test_assets_have_the_names_of_the_contract(self):
        """FA6 (Carbon R3): under assets/ only `asset-NNNN.<ext>`, as the filter names them."""
        (self.stage / "document.md").write_bytes(b"ok\n")
        for name in ("asset-0001.png", "asset-12345.jpeg"):
            (self.stage / "assets" / name).write_bytes(b"x")
        self.assertEqual(len(contract.scan(self.stage, 1)), 3)
        for name in ("f5ae993f4c638efd0b471ff1ffc4cb5712202ab1.png", "asset-1.png", "asset-0003", "asset-0003.PNG",
                     "sub/asset-0004.png"):
            with self.subTest(name=name):
                bad = self.stage / "assets" / name
                bad.parent.mkdir(exist_ok=True)
                bad.write_bytes(b"x")
                self.assertEqual(self.code(lambda: contract.scan(self.stage, 1)), Code.OUTPUT_INVALID)
                bad.unlink()

    def test_publish_and_failure(self):
        (self.stage / "document.md").write_text("# T\n")
        out = contract.prepare_out(self.root / "out")
        result = contract.publish(self.stage, out, {"schema": contract.SCHEMA}, 1)
        self.assertEqual(result["status"], "complete")
        self.assertEqual(json.loads((out / "engine.json").read_text()), result)
        self.assertEqual(sorted(p.name for p in out.iterdir()), ["document.md", "engine.json"])
        self.assertEqual(self.code(lambda: contract.prepare_out(out)), Code.OUTPUT_INVALID)        # not empty
        out2 = contract.prepare_out(self.root / "out2")
        failed = contract.publish_failure(out2, {"schema": contract.SCHEMA, "inventory": []}, EngineError(Code.TOOL_FAILED, "x"),
                                          {"tool.log": b"y" * (200 * 1024)})
        self.assertEqual((failed["status"], failed["error"], failed["inventory_complete"]),
                         ("failed", {"code": "tool_failed", "fatal": False}, False))
        self.assertNotIn("inventory", failed)
        self.assertEqual(len((out2 / "diagnostics" / "tool.log").read_bytes()), contract.LOG_TAIL)
        self.assertFalse((out2 / "document.md").exists())

    def test_problems_have_a_closed_code_and_a_severity(self):
        self.assertEqual(contract.problem("media_lost", path="a")["severity"], "incomplete")
        self.assertEqual(contract.problem("media_furniture")["severity"], "warning")
        with self.assertRaises(ValueError):
            contract.problem("something")
        items, counts = contract.bounded([{"code": "a"}] * 5 + [{"code": "b"}], 2)
        self.assertEqual((len(items), counts), (2, {"a": 5, "b": 1}))


class TextConvertTests(Base):
    def test_txt_end_to_end_and_engine_json(self):
        src = self.root / "note.txt"
        src.write_bytes("Riga uno è\r\nRiga *due*\r\n".encode("utf-8"))
        out = self.root / "out"
        result = convert(src, out, {"lang": "it"}, work=self.root / "work")
        self.assertEqual(result["status"], "complete")
        self.assertEqual((out / "document.md").read_text(encoding="utf-8"), "Riga uno è\nRiga *due*\n")
        self.assertEqual(result["schema"], "mdf-engine-result/1")
        self.assertEqual(result["source"]["format"], "txt")
        self.assertEqual(result["route"], {"converter": "text", "family": "native", "chosen_by": "matrix",
                                           "reason": "matrix-default", "steps": []})
        self.assertEqual(result["options"]["effective"], {"ocr": None, "lang": "it", "lang_passed_to": []})
        self.assertEqual(result["transit"], {"left_machine": False, "service": None})
        self.assertEqual(result["text"]["encoding"], "utf_8")
        self.assertEqual([i["path"] for i in result["inventory"]], ["document.md"])
        self.assertEqual(os.listdir(self.root / "work"), [])            # the private work of the attempt is gone

    def test_failures_of_the_text_route(self):
        for name, data, code in (("empty.txt", b"  \n", "output_empty"), ("nul.txt", b"a\x00b", "format_mismatch")):
            with self.subTest(name=name):
                src = self.root / name
                src.write_bytes(data)
                result = convert(src, self.root / ("out-" + name), {})
                self.assertEqual(result["error"]["code"], code)

    def test_work_inside_out_is_refused_and_a_defect_is_engine_failed(self):
        src = self.root / "a.txt"
        src.write_text("x\n")
        with self.assertRaises(EngineError):
            convert(src, self.root / "o", {}, work=self.root / "o" / "w")
        from unittest import mock
        with mock.patch("markdown_for_llms.engine.text.convert", side_effect=RuntimeError("boom")):
            result = convert(src, self.root / "o2", {})
        self.assertEqual(result["error"]["code"], "engine_failed")
        self.assertIn("RuntimeError", (self.root / "o2" / "diagnostics" / "engine-traceback.txt").read_text())

    def test_engine_json_is_inside_the_output_cap(self):
        """HA2 (Sol H2): the cap counts `engine.json` too."""
        src = self.root / "full.txt"
        src.write_bytes(b"a" * 1048575 + b"\n")                      # document.md alone is the whole MiB
        out = self.root / "o-full"
        result = convert(src, out, {}, {"output_mib": 1})
        self.assertEqual((result["status"], result["error"]["code"]), ("failed", "output_limit"))
        self.assertFalse((out / "document.md").exists())
        src.write_bytes(b"a" * (1048576 - 16384) + b"\n")
        out = self.root / "o-fits"
        result = convert(src, out, {}, {"output_mib": 1})
        self.assertEqual(result["status"], "complete")
        self.assertLessEqual(sum(p.stat().st_size for p in out.rglob("*") if p.is_file()), 1048576)

    def test_attempt_is_checked_before_any_directory(self):
        """HA4 (Carbon I1): the attempt names directories and containers: a closed error first."""
        src = self.root / "a.txt"
        src.write_text("x\n")
        for i, attempt in enumerate(("x/../../outside", "bad name", "-x", "a" * 65, "é", "a\n")):
            with self.subTest(attempt=attempt):
                out, work = self.root / f"o-attempt{i}", self.root / f"w-attempt{i}"
                self.assertEqual(self.code(lambda: convert(src, out, {"attempt": attempt}, work=work)), Code.INVALID_SETTING)
                self.assertFalse(out.exists())
                self.assertFalse(work.exists())
                self.assertFalse((self.root / "outside").exists())
        result = convert(src, self.root / "o-ok", {"attempt": "job-12.a_b"}, work=self.root / "w-ok")
        self.assertEqual((result["status"], result["attempt"]), ("complete", "job-12.a_b"))

    def test_attempt_is_random_only_when_absent(self):
        """FB3 (Sol H1, Carbon RR1 c): absent or None gives a random attempt; every other value is
        checked, the empty string, 0, False and containers included."""
        src = self.root / "a.txt"
        src.write_text("x\n")
        for i, attempt in enumerate(("", 0, False, [], {}, 12, ["job"])):
            with self.subTest(attempt=attempt):
                out, work = self.root / f"o-falsy{i}", self.root / f"w-falsy{i}"
                self.assertEqual(self.code(lambda: convert(src, out, {"attempt": attempt}, work=work)), Code.INVALID_SETTING)
                self.assertFalse(out.exists())
        for i, request in enumerate(({}, {"attempt": None})):
            with self.subTest(request=request):
                result = convert(src, self.root / f"o-random{i}", request, work=self.root / f"w-random{i}")
                self.assertEqual(result["status"], "complete")
                self.assertRegex(result["attempt"], "^[0-9a-f]{16}$")

    def test_observer_can_cancel(self):
        src = self.root / "a.txt"
        src.write_text("x\n")
        result = convert(src, self.root / "o", {}, observer=lambda event: "cancelled")
        self.assertEqual(result["error"]["code"], "cancelled")

    def test_availability_reports_every_converter(self):
        found = availability()
        self.assertEqual(set(found), {"text", "pandoc", "djvutxt", "marker", "datalab"})
        self.assertEqual(found["marker"], {"available": False, "code": "route_unavailable"})
        self.assertTrue(found["text"]["available"])

    def test_availability_without_bubblewrap(self):
        """HA5 (Carbon I2): the pandoc route audits its packages in the light child, so without
        Bubblewrap it is not available; djvutxt does not use the child."""
        from unittest import mock
        from markdown_for_llms.engine import isolation, toolchain
        with mock.patch.object(toolchain, "image", return_value="sha256:" + "0" * 64), \
                mock.patch.object(isolation, "BWRAP", str(self.root / "no-bwrap")):
            found = availability()
        self.assertEqual(found["pandoc"], {"available": False, "code": "isolation_unavailable"})
        self.assertEqual(found["djvutxt"], {"available": True, "code": None})


if __name__ == "__main__":
    unittest.main()
