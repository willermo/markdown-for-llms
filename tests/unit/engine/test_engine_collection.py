"""E-8 of MFL051: every place where the engine takes the results of a child, with a link in place of each.

A child writes its scratch, and nothing else it can write is ever read. So a command that was led astray (a
hostile document, a defect) can leave, where a result should be, a link to a file of the process that started it:
in the worker in a container, the key file mounted from the host, or the secrets it keeps for the providers. An
ordinary open would follow it, and the bytes would go to the archive or to the diagnostics of the job.

The engine reads no result from the scratch of a child. Its results are copied to a directory of the caller with
the rooted copy of `containers` (each component opened from the one before, without following a link, regular
files and directories of this user only), and the callers read the copy. Here, for each collection point of the
engine, a scripted child leaves a link to a file with a synthetic value (and to a directory that holds one) in
place of every result that point takes; the value must reach neither the caller's directories, nor the
diagnostics, nor the output.

The tools of the toolchain image: `tests/unit/bench/test_bench_tools.py`
(`test_a_link_in_place_of_an_output_or_above_it_is_never_followed`). On a real container:
`tests/container/test_child_isolation.py`.
"""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import containers_test_support as launcher_fake
from engine_test_support import make_docx, para
from markdown_for_llms.engine import containers, core, native, pages, toolchain
from markdown_for_llms.engine.codes import Code, EngineError

VALUE = b"synthetic-value-of-the-caller"
SECRET = b"SYNTHETIC_API_KEY=" + VALUE + b"\n"


class CollectionCase(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.top = Path(tmp.name)
        # what the process that starts the children holds, outside every call: its key file and its secrets
        self.key_file = self.top / "run" / "mdf" / "env"
        self.secrets = self.top / "run" / "mdf" / "secrets"
        self.secrets.mkdir(parents=True)
        self.key_file.write_bytes(SECRET)
        (self.secrets / "modal.toml").write_bytes(SECRET)
        (self.secrets / "document.md").write_bytes(SECRET)
        self.caller = self.top / "caller"           # everything the engine writes for itself: work, stage, output
        self.caller.mkdir()
        saved = dict(toolchain._state)
        toolchain.reset()
        self.addCleanup(toolchain._state.update, saved)
        self.links = {}                             # module: {path in the scratch: target}
        self.files = {}                             # module: {path in the scratch: bytes}, written before the links
        self.launcher = launcher_fake.patched(self, self.top / "volume", self.script)

    def script(self, request, call):
        """The child: the files of `files` and the links of `links` for its module, then exit 0."""
        if request["kind"] == "tool":
            return 0
        module = request["argv"][6].rsplit(".", 1)[-1]
        scratch = call / "scratch"
        for name, data in self.files.get(module, {}).items():
            (scratch / name).parent.mkdir(parents=True, exist_ok=True)
            (scratch / name).write_bytes(data)
        for name, target in self.links.get(module, {}).items():
            path = scratch / name
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.is_dir() and not path.is_symlink():
                path.rmdir()
            elif path.exists():
                path.unlink()
            os.symlink(target, path)
        return 0

    def context(self, attempt="collect1"):
        work = self.caller / ("work-" + attempt)
        stage = work / "stage"
        (stage / "diagnostics").mkdir(parents=True)
        ctx = core.Context(attempt, work, stage, dict(core.DEFAULT_LIMITS), None, self.top / "models")
        self.addCleanup(ctx.close)
        return ctx

    def leaked(self, *more):
        """Where the synthetic value is, outside the two files that hold it by design: nowhere."""
        found = []
        for root in (self.caller, self.top / "volume"):
            for path in root.rglob("*"):
                if path.is_symlink():
                    continue
                if path.is_file() and VALUE in path.read_bytes():
                    found.append(str(path.relative_to(self.top)))
        for name, data in more:
            if VALUE in (data if isinstance(data, bytes) else json.dumps(data, default=str).encode()):
                found.append(name)
        return found


class AuditTests(CollectionCase):
    """`native.audit`: `audit.json`, the normalized copy of the package, the media of the sheets."""

    def audit(self, ctx):
        src = make_docx(self.top / "letter.docx", para("Text of the synthetic letter"))
        return native.audit(ctx, src, "docx")

    def test_a_link_in_place_of_the_audit(self):
        for target in (self.key_file, self.secrets / "modal.toml", self.secrets):
            with self.subTest(target=target.name):
                self.links = {"pkgaudit": {"audit.json": str(target)}}
                ctx = self.context("audit%d" % len(str(target)))
                with self.assertRaises(EngineError) as caught:
                    self.audit(ctx)
                self.assertEqual(caught.exception.code, Code.ENGINE_FAILED)        # no audit: the package is not converted
                self.assertEqual(self.leaked(("diagnostics", dict(ctx.diagnostics))), [])

    def test_a_link_in_place_of_the_normalized_copy_and_of_the_media(self):
        audit = {"format": "docx", "normalized": {"w": "x"}, "media": [], "sheets": [{"images": ["a.png"]}]}
        self.files = {"pkgaudit": {"audit.json": json.dumps(audit).encode()}}
        self.links = {"pkgaudit": {"normalized.docx": str(self.key_file), "media": str(self.secrets)}}
        ctx = self.context()
        package, directory = self.audit(ctx)
        self.assertEqual(package["normalized"], {"w": "x"})
        self.assertEqual(sorted(p.name for p in directory.iterdir()), ["audit.json"])      # neither link came back
        self.assertFalse((directory / "normalized.docx").exists())
        self.assertFalse((directory / "media").exists())
        self.assertEqual(self.leaked(("diagnostics", dict(ctx.diagnostics))), [])

    def test_a_link_inside_the_media(self):
        audit = {"format": "docx", "normalized": {}, "media": []}
        self.files = {"pkgaudit": {"audit.json": json.dumps(audit).encode(), "media/real.png": b"png"}}
        self.links = {"pkgaudit": {"media/image1.png": str(self.key_file), "media/more": str(self.secrets)}}
        ctx = self.context()
        package, directory = self.audit(ctx)
        self.assertEqual(sorted(p.name for p in (directory / "media").iterdir()), ["real.png"])
        self.assertEqual(self.leaked(), [])


class PrepTests(CollectionCase):
    """`pages._prep`: the JSON of each step of `pdfprep` and the PDF it leaves."""

    def test_a_link_in_place_of_each_result_of_the_preparation(self):
        cases = (("image", "image.json", ["image", "png", 10, 100]), ("image", "source.pdf", ["image", "png", 10, 100]),
                 ("classify", "pdfprep.json", ["classify", 10]), ("strip", "strip.json", ["strip", "1"]),
                 ("strip", "stripped.pdf", ["strip", "1"]))
        src = self.top / "source.bin"
        src.write_bytes(b"synthetic input")
        for n, (name, result, args) in enumerate(cases):
            for target in (self.key_file, self.secrets):
                with self.subTest(step=name, result=result, target=target.name):
                    others = {r: b"{}" for r in pages.PREP_RESULTS[name] if r != result}
                    self.files = {"pdfprep": others}
                    self.links = {"pdfprep": {result: str(target)}}
                    ctx = self.context("prep%d%d" % (n, len(target.name)))
                    scratch = pages._prep(ctx, name, args, [(src, "source.bin")])
                    self.assertEqual(sorted(p.name for p in scratch.iterdir()), sorted(others))
                    self.assertEqual(self.leaked(("diagnostics", dict(ctx.diagnostics))), [])

    def test_the_caller_that_reads_a_missing_result_fails_closed(self):
        self.links = {"pdfprep": {"pdfprep.json": str(self.key_file)}}
        src = self.top / "source.pdf"
        src.write_bytes(b"%PDF synthetic")
        with self.assertRaises(FileNotFoundError):
            pages.classify(self.context(), src)
        self.assertEqual(self.leaked(), [])


class MarkerChildTests(CollectionCase):
    """`pages.run_child` and `pages.collect`: `result.json`, `out/document.md`, `out/images/`, `out/source.pdf`."""

    def child(self, ctx, args=("convert",)):
        src = self.top / "source.pdf"
        src.write_bytes(b"%PDF synthetic")
        with mock.patch.object(pages.weights, "verify", return_value={"files": 18}), \
                mock.patch.object(pages.containers, "MODEL_ROOT", str(self.top / "models")):
            return pages.run_child(ctx, list(args), [(src, "source.pdf")], {"threads": 1})

    def test_a_link_in_place_of_each_result_of_the_marker_child(self):
        results = {"result.json": b'{"pages": 1}', "out/document.md": b"# the document\n", "out/images/a.png": b"png",
                   "out/source.pdf": b"%PDF"}
        cases = [(name, target) for name in results for target in (self.key_file, self.secrets / "document.md", self.secrets)]
        cases += [("out", self.secrets), ("out/images", self.secrets)]
        for n, (name, target) in enumerate(cases):
            with self.subTest(result=name, target=target.name):
                self.files = {"marker_child": {k: v for k, v in results.items() if not (k == name or k.startswith(name + "/"))}}
                self.links = {"marker_child": {name: str(target)}}
                ctx = self.context("marker%d" % n)
                status, scratch = self.child(ctx)
                self.assertEqual(status, 0)
                self.assertFalse(any(p.is_symlink() for p in scratch.rglob("*")))
                self.assertFalse((scratch / name).is_file() and VALUE in (scratch / name).read_bytes())
                self.assertEqual(self.leaked(("diagnostics", dict(ctx.diagnostics))), [])

    def test_the_document_and_its_images_never_come_from_a_link(self):
        """`pages.collect` reads `out/document.md` and copies `out/images/*` to the output: from the copy only."""
        result = {"pages": 1, "provenance": {"1": {"indra_source": "text", "note": None, "blocks": 1, "text_blocks": 1,
                                                    "pictures_added": 0, "pictures_contained": 0}},
                  "methods": {"1": "pdftext"}, "page_problems": [], "images": ["a.png"], "versions": {}, "seconds": {},
                  "network_calls": 0}
        self.files = {"marker_child": {"result.json": json.dumps(result).encode(), "out/images/a.png": b"png"}}
        self.links = {"marker_child": {"out/document.md": str(self.secrets / "document.md")}}
        ctx = self.context()
        status, scratch = self.child(ctx)
        page = {"page": 1, "class": "native", "invisible_chars": 0}
        with self.assertRaises(FileNotFoundError):
            pages.collect(ctx, scratch, result, [page], {}, {0: {"total": 0}}, False, "native_pages", None, None)
        self.assertEqual(self.leaked(("diagnostics", dict(ctx.diagnostics))), [])
        # and with the document in place and a link among the images: the image is not there to copy
        self.files = {"marker_child": {"result.json": json.dumps(result).encode(),
                                       "out/document.md": b"Text of the page.\n\n![](a.png)\n"}}
        self.links = {"marker_child": {"out/images/a.png": str(self.key_file)}}
        ctx = self.context("collect2")
        status, scratch = self.child(ctx)
        with self.assertRaises(FileNotFoundError):
            pages.collect(ctx, scratch, result, [page], {}, {0: {"total": 0}}, False, "native_pages", None, None)
        self.assertEqual(self.leaked(("diagnostics", dict(ctx.diagnostics))), [])


class ConvertTests(CollectionCase):
    """The whole engine: a conversion whose child leaves a link fails with a closed code, and neither its output
    nor its diagnostics hold the value."""

    def test_a_conversion_whose_audit_is_a_link_fails_closed(self):
        self.links = {"pkgaudit": {"audit.json": str(self.key_file)}}
        src = make_docx(self.top / "letter.docx", para("Text of the synthetic letter"))
        out = self.caller / "out"
        result = core.convert(src, out, {"attempt": "convert1"}, None, work=self.caller / "engine-work")
        self.assertEqual((result["status"], result["error"]["code"]), ("failed", "engine_failed"))
        self.assertEqual(self.leaked(("result", result)), [])
        self.assertEqual(os.listdir(self.top / "volume" / "call"), [])              # the call was attested and removed


class WorkerTests(CollectionCase):
    """`worker.execute`: the result of the raster check and the report of the probe."""

    def execute(self, module, results, label):
        from markdown_for_llms import worker
        from types import SimpleNamespace
        return worker.execute(SimpleNamespace(output_mib=64), self.caller / "attempt", module, label=label,
                              attempt="worker1", results=results)

    def test_a_link_in_place_of_the_raster_result_and_of_the_probe_report(self):
        cases = (("markdown_for_llms.engines.raster", "raster", "raster-result.json"),
                 (containers.PROBE_MODULE, "selfprobe", "probe.json"))
        for module, name, result in cases:
            for target in (self.key_file, self.secrets):
                with self.subTest(result=result, target=target.name):
                    self.links = {name: {result: str(target)}}
                    receipt = self.execute(module, (result,), name + target.name)
                    self.assertEqual((receipt["exit"], receipt["reason"]), (0, None))
                    self.assertFalse((receipt["dir"] / result).exists())
                    self.assertEqual(self.leaked(), [])

    def test_a_probe_whose_report_is_a_link_does_not_pass(self):
        from markdown_for_llms import worker
        from markdown_for_llms.settings import Settings
        self.links = {"selfprobe": {"probe.json": str(self.key_file)}}
        self.launcher.answer.update(payload_sha256=containers.payload_sha256(), inputs_sha256=None)
        self.launcher.hello = lambda: dict(self.launcher.answer)
        w = worker.Worker(Settings(data_root=str(self.caller / "data"), model_root=str(self.top / "models")))
        self.addCleanup(w.storage.close)
        self.assertEqual(w.probe(), ["probe: no report"])
        self.assertEqual(self.leaked(), [])


if __name__ == "__main__":
    unittest.main()
