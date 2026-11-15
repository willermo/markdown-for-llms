"""The Datalab route (MFL032 B4, K8) with a fake transport: no network, no cost. The PDF
preparation runs for real in the light child. Criteria: transit confirmed before any byte
leaves, cap from the estimate, one POST at most, DELETE always, the key never in the result."""
import base64
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest import mock

from markdown_for_llms.engine import datalab
from markdown_for_llms.engine.core import convert

from engine_test_support import GLYPHLESS_MODE0, INVISIBLE, NATIVE_TEXT, SCAN, checker_rgb, make_pdf, png

SECRET = "dl-SECRET-VALUE-0123456789"
IMG = (100, 100, checker_rgb(100, 100))
SEP = "-" * 48


class FakeTransport:
    def __init__(self, polls=1, post=None, get_error=None, delete_ok=True, markdown=None, images=None, breakdown=None):
        self.calls, self.uploads = [], []
        self.polls, self.post_answer, self.get_error, self.delete_ok = polls, post, get_error, delete_ok
        self.markdown = markdown if markdown is not None else "{0}%s\n\nText from Datalab.\n\n![](_page_0_Picture_1.png)\n\n{1}%s\n\nPage two.\n" % (SEP, SEP)
        self.images = {"_page_0_Picture_1.png": base64.b64encode(png()).decode()} if images is None else images
        self.breakdown = {"list_cost_cents": 0.8, "final_cost_cents": 0.6} if breakdown is None else breakdown

    def network_errors(self):
        return (OSError,)

    def post(self, url, headers, files, data):
        self.calls.append(("POST", url, dict(headers), dict(data)))
        self.uploads.append(files["file"][1].read())
        if isinstance(self.post_answer, Exception):
            raise self.post_answer
        return self.post_answer or (200, {"success": True, "request_id": "req-1", "request_check_url": url + "/req-1"})

    def get(self, url, headers):
        self.calls.append(("GET", url, dict(headers), None))
        if self.get_error:
            raise self.get_error
        self.polls -= 1
        if self.polls > 0:
            return 200, {"status": "processing"}
        return 200, {"status": "complete", "success": True, "markdown": self.markdown, "images": self.images,
                     "page_count": 2, "cost_breakdown": self.breakdown, "versions": {"marker": "x"}}

    def delete(self, url, headers):
        self.calls.append(("DELETE", url, dict(headers), None))
        return (200, {"success": True, "message": "deleted"}) if self.delete_ok else (500, None)

    def methods(self):
        return [c[0] for c in self.calls]


class Base(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.n = 0
        datalab.POLL_SECONDS = 0.01
        self.addCleanup(setattr, datalab, "POLL_SECONDS", 2.0)

    def pdf(self, spec=None):
        path = self.root / "s.pdf"
        path.write_bytes(make_pdf(spec or [{"content": NATIVE_TEXT}, {"content": NATIVE_TEXT}]))
        return path

    def run_route(self, transport, path=None, request=None, observer=None, **datalab_options):
        self.n += 1
        options = {"mode": "balanced", "transit_confirmed": True, "cap_usd": "0.05", **datalab_options}
        req = {"route": "datalab", "datalab": options, **(request or {})}
        out = self.root / ("out%d" % self.n)
        with mock.patch.object(datalab, "Transport", return_value=transport):
            result = convert(path or self.pdf(), out, req, {"memory_mib": 2048}, work=self.root / "work",
                             observer=observer, datalab_key=datalab.Key(SECRET))
        return result, out


class KeyTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)

    def env(self, mode=0o600, text="MDF_PORT=23465\n# comment\nexport DATALAB_API_KEY='%s'\n" % SECRET):
        path = self.root / ".env"
        path.write_text(text)
        os.chmod(path, mode)
        return path

    def test_key_from_a_private_env_file_and_never_shown(self):
        key = datalab.read_key(self.env(), environ={})
        self.assertEqual(key.header(), {"X-API-Key": SECRET})
        self.assertNotIn(SECRET, repr(key) + str(key))

    def test_unsafe_files_are_refused(self):
        link = self.root / "link"
        link.symlink_to(self.env())
        for path in (self.env(0o644), link, self.root):
            with self.subTest(path=path.name), self.assertRaises(datalab.EngineError) as e:
                datalab.read_key(path, environ={})
            self.assertEqual(e.exception.code.value, "datalab_key_unsafe")

    def test_environment_and_missing_key(self):
        self.assertEqual(datalab.read_key(None, environ={"DATALAB_API_KEY": "k"}).header(), {"X-API-Key": "k"})
        with self.assertRaises(datalab.EngineError) as e:
            datalab.read_key(self.env(text="OTHER=1\n"), environ={})
        self.assertEqual(e.exception.code.value, "datalab_key_missing")

    def test_estimate_and_cap(self):
        self.assertEqual(datalab.estimate_cents(4, "balanced"), datalab.Decimal("2.60"))     # 4 x 0.4 + 1
        for bad in ("-1", "nan", "inf", "x", None):
            with self.subTest(bad=bad), self.assertRaises(datalab.EngineError):
                datalab.parse_cap(bad)


class RouteTests(Base):
    def test_complete_conversion_deletes_and_records_cost_and_transit(self):
        t = FakeTransport(polls=2)
        with redirect_stdout(io.StringIO()) as printed:
            result, out = self.run_route(t)
        self.assertEqual(result["status"], "complete", result.get("error"))
        self.assertEqual(t.methods(), ["POST", "GET", "GET", "DELETE"])
        self.assertEqual(t.calls[0][3], {"mode": "balanced", "output_format": "markdown", "paginate": "true",
                                         "disable_image_extraction": "false"})
        self.assertEqual(t.calls[-1][1], datalab.API + "/results/req-1")
        self.assertEqual(result["transit"], {"left_machine": True, "service": "datalab", "endpoint": datalab.API,
                                             "request_id": "req-1", "deleted": True})
        self.assertEqual((result["cost"]["estimated_cents"], result["cost"]["billed_cents"]), ("1.80", "0.6"))
        text = (out / "document.md").read_text()
        self.assertIn("![](assets/asset-0001.png)", text)
        self.assertEqual((out / "assets" / "asset-0001.png").read_bytes(), png())
        everything = printed.getvalue() + "".join(p.read_text(errors="replace") for p in out.rglob("*") if p.is_file())
        self.assertNotIn(SECRET, everything)                     # the key is in no output and no log
        self.assertEqual(result["components"]["datalab"]["versions"], {"marker": "x"})

    def test_without_confirmation_nothing_leaves(self):
        t = FakeTransport()
        result, _ = self.run_route(t, transit_confirmed=False)
        self.assertEqual(result["error"]["code"], "transit_not_confirmed")
        self.assertEqual(t.calls, [])
        self.assertEqual((result["cost"]["pages"], result["transit"]["left_machine"]), (2, False))

    def test_over_the_cap_nothing_leaves(self):
        t = FakeTransport()
        result, _ = self.run_route(t, cap_usd="0.01")
        self.assertEqual((result["error"]["code"], t.calls), ("datalab_cap", []))

    def test_refused_options(self):
        for request, options, code in (({"ocr": "force"}, {}, "route_option_unsupported"),
                                       ({}, {"mode": "premium"}, "route_option_unsupported")):
            with self.subTest(code=code, request=request):
                t = FakeTransport()
                result, _ = self.run_route(t, request=request, **options)
                self.assertEqual((result["error"]["code"], t.calls), (code, []))

    def test_ocr_layer_removed_before_sending_or_refused(self):
        t = FakeTransport()
        layer = self.pdf([{"content": NATIVE_TEXT}, {"content": SCAN + b" " + INVISIBLE, "image": IMG}])
        result, _ = self.run_route(t, path=layer)
        self.assertEqual(result["status"], "complete", result.get("error"))
        self.assertNotIn(b"INVISIBLE OCR LAYER", t.uploads[0])
        t = FakeTransport()
        path = self.root / "glyphless.pdf"
        path.write_bytes(make_pdf([{"content": NATIVE_TEXT}, {"content": SCAN + b" " + GLYPHLESS_MODE0, "image": IMG}],
                                  font="GlyphlessFont"))
        result, _ = self.run_route(t, path=path)
        self.assertEqual((result["error"]["code"], t.calls), ("ocr_layer_not_removable", []))

    def test_post_without_answer_is_never_repeated(self):
        t = FakeTransport(post=OSError("reset"))
        result, _ = self.run_route(t)
        self.assertEqual((result["error"]["code"], t.methods()), ("datalab_unavailable", ["POST"]))
        self.assertTrue(result["transit"]["left_machine"])
        t = FakeTransport(post=(401, {"detail": "bad key"}))
        result, _ = self.run_route(t)
        self.assertEqual((result["error"]["code"], t.methods()), ("datalab_rejected", ["POST"]))

    def test_delete_after_a_failure_and_a_cancellation(self):
        t = FakeTransport(get_error=OSError("down"))
        result, _ = self.run_route(t)
        self.assertEqual(result["error"]["code"], "datalab_unavailable")
        self.assertEqual(t.methods()[-1], "DELETE")
        t = FakeTransport(polls=5)
        result, _ = self.run_route(t, observer=lambda e: "cancelled" if e.get("service") == "datalab" else None)
        self.assertEqual((result["error"]["code"], t.methods()), ("cancelled", ["POST", "DELETE"]))

    def test_a_local_error_after_the_acceptance_still_deletes(self):                      # FC3
        import errno
        real = Path.write_text

        def full(path, *args, **kwargs):
            if path.name == "datalab-request.json":
                raise OSError(errno.ENOSPC, "No space left on device")
            return real(path, *args, **kwargs)
        for delete_ok, problems in ((True, []), (False, ["datalab_delete_failed"])):
            with self.subTest(delete_ok=delete_ok):
                t = FakeTransport(delete_ok=delete_ok)
                with mock.patch.object(Path, "write_text", full):
                    result, _ = self.run_route(t)
                self.assertEqual(result["status"], "failed")
                self.assertEqual(t.methods(), ["POST", "DELETE"])
                self.assertEqual((result["transit"]["request_id"], result["transit"]["deleted"]), ("req-1", delete_ok))
                self.assertEqual([p["code"] for p in result["problems"]], problems)
                if problems:
                    self.assertIn("req-1", result["problems"][0]["detail"])

    def test_a_request_id_with_another_status_is_deleted(self):
        t = FakeTransport(post=(202, {"request_id": "req-2"}))
        result, _ = self.run_route(t)
        self.assertEqual((result["error"]["code"], t.methods()), ("datalab_failed", ["POST", "DELETE"]))

    def test_failed_delete_is_a_warning(self):
        t = FakeTransport(delete_ok=False, breakdown={})
        result, _ = self.run_route(t)
        self.assertEqual(result["status"], "complete")
        self.assertEqual(sorted(p["code"] for p in result["problems"]), ["datalab_cost_unknown", "datalab_delete_failed"])
        self.assertFalse(result["transit"]["deleted"])

    def test_answers_that_are_not_a_document(self):
        for t, code in ((FakeTransport(markdown="{0}%s\n\n{1}%s\n" % (SEP, SEP), images={}), "output_empty"),
                        (FakeTransport(images={"../x.png": base64.b64encode(png()).decode()}), "output_invalid")):
            with self.subTest(code=code):
                result, _ = self.run_route(t)
                self.assertEqual((result["error"]["code"], t.methods()[-1]), (code, "DELETE"))


if __name__ == "__main__":
    unittest.main()
