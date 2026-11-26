"""External tools of the bench: the runner, its request to the launcher, caps, timeouts and LibreOffice.

`RealToolchainTests` run the tools for real in the reference image, through the launcher (E1 of the toolchain
run); the scripted launcher covers the request of each call (what the launcher is asked: it names no mount) and
the exits a real tool cannot be made to give on demand. What the container of a tool is from inside is proved on
real containers in `tests/container/test_child_isolation.py`, the body of its creation in `tests/unit/launcher/`.
"""
import os
from pathlib import Path
import tempfile
import unittest
import unittest.mock

from bench_test_support import FAKE_IMAGE, growing_html, left_by_this_process, make_pdf, require_toolchain, scripted
from markdown_for_llms.bench import toolchain, tools
from markdown_for_llms.bench.codes import BenchError, Code
from markdown_for_llms.engine import containers


class Base(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.work_root = self.root / "cache" / "tool-work"


class RealToolchainTests(Base):
    def setUp(self):
        super().setUp()
        require_toolchain()

    def assert_work_root_empty(self):
        """Nothing of the calls of this process is left: no container, no folder of a call on the work volume."""
        self.assertEqual(left_by_this_process(), ([], []))
        self.assertFalse(self.work_root.exists())                      # the calls no longer live under the cache


class ScriptedTests(Base):
    """The runner against a scripted launcher (no container), with the image taken as verified."""

    def launcher(self, script=None, **kw):
        self.fake = scripted(self, self.root / "volume", script, **kw)
        return self.fake

    def calls_left(self):
        calls = self.root / "volume" / "call"
        return sorted(os.listdir(calls)) if calls.is_dir() else []


class RunnerArgumentTests(ScriptedTests):
    def test_the_request_names_the_tool_its_environment_and_its_caps_and_no_mount(self):
        seen = []

        def script(request, call):
            seen.append({"in": {p.name: p.read_bytes() for p in (call / "in").iterdir()},
                         "stdin": (call / "ctl" / "stdin").read_bytes() if (call / "ctl" / "stdin").exists() else None})
            return 0
        fake = self.launcher(script)
        src = self.root / "runs" / "r1" / "in,put file.pdf"
        src.parent.mkdir(parents=True)
        src.write_bytes(b"%PDF")
        (src.parent / "inference.key").write_text("not for tools")
        before = tools.memory_limit()
        self.addCleanup(tools.set_memory_limit, before)
        tools.set_memory_limit(1234)
        tools.run("pdftotext", ["-raw", tools.In(src), "-"], self.work_root)
        tools.run("pandoc", ["-f", "latex", "-t", "json"], self.work_root, stdin=b"x")
        (poppler, pandoc) = fake.requests
        for request in (poppler, pandoc):
            self.assertEqual(set(request), {"kind", "name", "image", "argv", "env", "limits"})    # no mount, no path
            self.assertEqual((request["kind"], request["image"]), ("tool", "toolchain"))           # a role, never a reference
            self.assertRegex(request["name"], r"^mdf-bench-\d+-[0-9a-f]{8}-tool-\d+$")          # a name per call (E6)
            self.assertEqual(request["env"], {"LC_ALL": "C.UTF-8", "LANG": "C.UTF-8", "HOME": "/work/home",
                                              "TMPDIR": "/work/tmp"})
            self.assertEqual(request["limits"]["memory_mib"], 1746)             # cap + 512 MiB; the launcher allows no swap
            self.assertFalse([a for a in request["argv"] if "docker.sock" in a or str(self.root) in a])
        self.assertEqual(poppler["argv"], ["pdftotext", "-raw", "/in/in_put_file.pdf", "-"])
        self.assertEqual(poppler["limits"]["data_mib"], 1234)
        self.assertEqual(pandoc["argv"], ["pandoc", "+RTS", "-M1234m", "-RTS", "--sandbox", "-f", "latex", "-t", "json"])
        self.assertNotIn("data_mib", pandoc["limits"])                          # the runtime's own cap
        # only the input of the call was copied for it, and the stdin is a file of the call
        self.assertEqual(seen, [{"in": {"in_put_file.pdf": b"%PDF"}, "stdin": None}, {"in": {}, "stdin": b"x"}])
        self.assertEqual(fake.removed, [poppler["name"], pandoc["name"]])
        self.assertEqual(self.calls_left(), [])
        self.assertEqual(toolchain.report()["image_id"], FAKE_IMAGE)             # by ID, as the launcher verified it

    def test_the_work_directory_is_private_and_removed(self):
        seen = {}

        def script(request, call):
            seen["mode"] = os.stat(call / "scratch").st_mode & 0o777
            seen["dirs"] = sorted(os.listdir(call / "scratch"))
            seen["root"] = os.stat(call).st_mode & 0o777
            return 0
        self.launcher(script)
        tools.run("qpdf", ["--version"], os.path.relpath(self.work_root))
        self.assertEqual(seen, {"mode": 0o700, "dirs": ["home", "tmp"], "root": 0o700})
        self.assertEqual(self.calls_left(), [])

    def test_outputs_are_copied_out_and_must_exist(self):
        def script(request, call):
            if "make" in request["argv"]:
                (call / "scratch" / "page.png").write_bytes(b"png")
            if "empty" in request["argv"]:
                (call / "scratch" / "page.png").write_bytes(b"")
            return 0
        self.launcher(script)
        dest = self.root / "out" / "page.png"
        tools.run("pdftoppm", ["make", tools.Out(dest, arg="page", produced="page.png")], self.work_root)
        self.assertEqual(dest.read_bytes(), b"png")
        for args in (["none"], ["empty"]):
            with self.subTest(args=args), self.assertRaises(BenchError) as ctx:
                tools.run("pdftoppm", [*args, tools.Out(dest, arg="page", produced="page.png")], self.work_root)
            self.assertEqual(ctx.exception.code, Code.TOOL_FAILED)
        tools.run("gs", ["empty", tools.Out(self.root / "out" / "page.png", arg="page.png", allow_empty=True)],
                  self.work_root)
        with self.assertRaises(ValueError):
            tools.Out(dest, arg="../escape")

    def test_a_link_in_place_of_an_output_or_above_it_is_never_followed(self):
        """E-8. A tool writes /work, so it can leave a link where its output should be: to a file of the process
        that started it (its key file), or a directory in place of the directory of the output. The output is
        `tool_failed` and no byte of the target reaches the destination."""
        secret = self.root / "keys.env"
        secret.write_text("SYNTHETIC_KEY=synthetic-value-of-the-caller\n")
        elsewhere = self.root / "elsewhere"
        elsewhere.mkdir()
        (elsewhere / "source.pdf").write_text("SYNTHETIC_KEY=synthetic-value-of-the-caller\n")

        def script(request, call):
            scratch = call / "scratch"
            how = request["argv"][2]
            if how == "leaf":
                os.symlink(secret, scratch / "page.png")
            elif how == "parent":
                os.symlink(elsewhere, scratch / "out")
            elif how == "inside":
                (scratch / "media").mkdir()
                (scratch / "media" / "real.png").write_bytes(b"png")
                os.symlink(secret, scratch / "media" / "image.png")
            elif how == "fifo":
                os.mkfifo(scratch / "page.png")
            return 0
        self.launcher(script)
        cases = (("leaf", tools.Out(self.root / "o1" / "page.png", arg="page.png"), "no output"),
                 ("fifo", tools.Out(self.root / "o2" / "page.png", arg="page.png"), "no output"),
                 ("parent", tools.Out(self.root / "o3" / "source.pdf", arg="out", produced="out/source.pdf"), "no output"),
                 ("parent", tools.Out(self.root / "o4", arg="out", directory=True), "not a directory"),
                 ("inside", tools.Out(self.root / "o5", arg="media", directory=True), "a link or a special file"))
        for how, out, why in cases:
            with self.subTest(how=how, why=why):
                with self.assertRaises(BenchError) as ctx:
                    tools.run("pdftoppm", ["x", how, out], self.work_root)
                self.assertEqual(ctx.exception.code, Code.TOOL_FAILED)
                self.assertIn(why, ctx.exception.detail)
                self.assertFalse(out.dest.exists())
        found = [p for p in self.root.rglob("*") if p.is_file() and p not in (secret, elsewhere / "source.pdf")
                 and b"synthetic-value-of-the-caller" in p.read_bytes()]
        self.assertEqual(found, [])
        self.assertEqual(self.calls_left(), [])


class RunnerExitTests(ScriptedTests):
    def run_tool(self, name, rc, stderr=b""):
        self.launcher(lambda request, call: (rc, b"", stderr))
        with self.assertRaises(BenchError) as ctx:
            tools.run(name, ["x"], self.work_root)
        return ctx.exception

    def test_a_killed_container_is_a_memory_limit_for_every_tool(self):
        for name in ("pandoc", "soffice", "pdftoppm"):
            with self.subTest(tool=name):
                exc = self.run_tool(name, 137)
                self.assertEqual((exc.code, exc.detail.endswith("exit 137")), (Code.TOOL_MEMORY_LIMIT, True))

    def test_a_process_of_the_tool_killed_for_memory_is_a_memory_limit_also_with_exit_0(self):
        self.launcher(lambda request, call: {"exit": 0, "oom_killed": True, "oom_kill": 1})
        with self.assertRaises(BenchError) as ctx:
            tools.run("soffice", ["x"], self.work_root)
        self.assertEqual((ctx.exception.code, ctx.exception.detail.endswith("exit 0")), (Code.TOOL_MEMORY_LIMIT, True))

    def test_runtime_and_allocation_failures(self):
        self.assertEqual(self.run_tool("pandoc", 251, b"pandoc: Heap exhausted;\n").code, Code.TOOL_MEMORY_LIMIT)
        self.assertEqual(self.run_tool("pdftoppm", 0, b"Out of memory\n").code, Code.TOOL_MEMORY_LIMIT)
        self.assertEqual(self.run_tool("qpdf", 2, b"qpdf: not a PDF file\n").code, Code.TOOL_FAILED)

    def test_engine_failures_are_not_tool_failures(self):
        for rc in (126, 127):               # the supervisor could not start the command
            with self.subTest(rc=rc):
                exc = self.run_tool("qpdf", rc)
                self.assertEqual((exc.code, exc.fatal), (Code.ENGINE_FAILED, False))

    def test_timeout_stops_the_container_and_attests_its_removal(self):
        fake = self.launcher(lambda request, call: None)            # the tool never ends by itself
        with self.assertRaises(BenchError) as ctx:
            tools.run("gs", ["x"], self.work_root, timeout=1)
        self.assertEqual((ctx.exception.code, ctx.exception.fatal, ctx.exception.detail), (Code.TOOL_FAILED, False, "gs: timeout"))
        name = fake.requests[0]["name"]
        self.assertEqual(fake.removed, [name])
        self.assertEqual(self.calls_left(), [])

    def test_a_container_whose_removal_is_not_confirmed_after_a_timeout_is_fatal(self):
        fake = self.launcher(lambda request, call: None)

        class Every:
            def __contains__(self, name):
                return True
        fake.unconfirmed = Every()
        with unittest.mock.patch.object(containers, "UNREACHABLE_SECONDS", 0.3), self.assertRaises(BenchError) as ctx:
            tools.run("gs", ["x"], self.work_root, timeout=1)
        self.assertEqual((ctx.exception.code, ctx.exception.fatal), (Code.TOOL_FAILED, True))
        # the end is not attested: the folder of the call stays, with the stop its supervisor reads
        name = fake.requests[0]["name"]
        self.assertEqual(self.calls_left(), [name])
        self.assertTrue((self.root / "volume" / "call" / name / "ctl" / "stop").exists())

    def test_an_interrupted_client_removes_its_container(self):
        def interrupt(request, call):
            raise KeyboardInterrupt
        fake = self.launcher(interrupt)
        with self.assertRaises(KeyboardInterrupt):
            tools.run("gs", ["x"], self.work_root)
        name = fake.ops("run")[0][1]
        self.assertEqual((fake.ops("kill"), fake.removed), ([("kill", name)], [name]))
        self.assertEqual(self.calls_left(), [])

    def test_an_interrupted_client_without_the_launcher_leaves_the_stop_for_the_supervisor(self):
        def interrupt(request, call):
            fake.down = True
            raise KeyboardInterrupt
        fake = self.launcher(interrupt)
        with self.assertRaises(KeyboardInterrupt):
            tools.run("gs", ["x"], self.work_root)
        name = fake.ops("run")[0][1]
        # nothing attested the end: `ctl/stop` ends the container within a tick, without Docker, and the folder
        # stays for the sweep
        self.assertTrue((self.root / "volume" / "call" / name / "ctl" / "stop").exists())

    def test_memory_markers(self):
        self.assertTrue(tools.out_of_memory("pandoc", 251, b"pandoc: Heap exhausted;\n"))
        self.assertTrue(tools.out_of_memory("pandoc", 137, b""))
        self.assertFalse(tools.out_of_memory("pandoc", 1, b"[WARNING] out of memory in a title\n"))
        self.assertTrue(tools.out_of_memory("pdftoppm", 0, b"Out of memory\n"))
        self.assertTrue(tools.out_of_memory("qpdf", 2, b"terminate called after throwing std::bad_alloc\n"))
        self.assertFalse(tools.out_of_memory("qpdf", 2, b"qpdf: in.pdf: not a PDF file\n"))

    def test_invalid_caps_are_rejected(self):
        for value in (0, 63, True, "512", 2 ** 21):
            with self.assertRaises(BenchError) as ctx:
                tools.set_memory_limit(value)
            self.assertEqual(ctx.exception.code, Code.INVALID_SETTING)

    def test_versions_come_from_toolchain_json_without_a_container(self):
        fake = self.launcher()
        self.assertEqual(tools.version("pandoc"), toolchain.spec()["versions"]["pandoc"])
        self.assertEqual(fake.calls, [])
        with self.assertRaises(ValueError):
            tools.version("curl")

    def test_without_the_launcher_or_with_an_image_it_refuses_no_tool_runs(self):
        fake = self.launcher(toolchain={"refuse": ("toolchain_mismatch", "content manifest")})
        with self.assertRaises(BenchError) as ctx:
            tools.run("qpdf", ["--version"], self.work_root)
        self.assertEqual(ctx.exception.code, Code.TOOLCHAIN_MISMATCH)
        fake.down = True
        with self.assertRaises(BenchError) as ctx:
            tools.run("qpdf", ["--version"], self.work_root)
        self.assertEqual(ctx.exception.code, Code.ISOLATION_UNAVAILABLE)
        self.assertEqual(fake.requests, [])


class SofficeIsolationTests(ScriptedTests):
    """LibreOffice runs in the private /work of its call: with a relative TMPDIR its PDF export
    failed with exit 0 and NSS wrote `pkcs11.txt` in the current directory (the repository root)."""

    def test_profile_home_output_and_input_are_in_the_private_work(self):
        seen = {}

        def script(request, call):
            seen["in"] = {p.name: p.read_bytes() for p in (call / "in").iterdir()}
            out = call / "scratch" / "out"
            out.mkdir()
            (out / "twin.pdf").write_bytes(b"%PDF converted")
            return 0
        fake = self.launcher(script)
        src = self.root / "cache" / "sources" / "abc"
        src.parent.mkdir(parents=True)
        src.write_bytes(b"pptx bytes")
        dest = self.root / "cache" / "work" / "selfcheck-D09"
        dest.mkdir(parents=True)
        out = tools.soffice_convert(src, "twin.pptx", "pdf", dest, self.work_root)
        self.assertEqual((out, out.read_bytes()), (dest / "twin.pdf", b"%PDF converted"))
        (request,) = fake.requests
        self.assertEqual(request["argv"], ["soffice", "-env:UserInstallation=file:///work/profile", "--headless", "--norestore",
                                           "--convert-to", "pdf", "--outdir", "/work/out", "/in/twin.pptx"])
        self.assertEqual(seen["in"], {"twin.pptx": b"pptx bytes"})
        self.assertEqual(self.calls_left(), [])

    def test_an_export_without_output_fails_even_with_exit_0(self):
        self.launcher()
        src = self.root / "in.xlsx"
        src.write_bytes(b"x")
        with self.assertRaises(BenchError) as ctx:
            tools.soffice_convert(src, "twin.xlsx", "html", self.root, self.work_root)
        self.assertEqual(ctx.exception.code, Code.TOOL_FAILED)


class RealRunnerTests(RealToolchainTests):
    def test_real_conversion_leaves_the_current_directory_untouched(self):
        src = self.root / "note.txt"
        src.write_text("hello\n")
        cwd = self.root / "cwd"
        cwd.mkdir()
        before = os.getcwd()
        os.chdir(cwd)
        self.addCleanup(os.chdir, before)
        out = tools.soffice_convert("../note.txt", "note.txt", "pdf", ".", "../cache/tool-work")
        self.assertEqual(sorted(os.listdir(cwd)), ["note.pdf"])        # no pkcs11.txt, no profile
        self.assertTrue(Path(out).read_bytes().startswith(b"%PDF"))
        self.assert_work_root_empty()

    def test_only_the_input_file_of_a_call_is_visible(self):
        """E3: the container sees /work and the input file, nothing of the cache or of a run."""
        run = self.root / "runs" / "r1"
        run.mkdir(parents=True)
        key = run / "inference.key"
        key.write_text("sk-local-test-key-not-for-tools\n")
        twin = run / "twin.md"
        twin.write_text("visible text\n")
        pdf = self.root / "cache" / "sources" / "page.pdf"
        pdf.parent.mkdir(parents=True)
        pdf.write_bytes(make_pdf(["one page"]))
        self.assertIn(b"visible text", tools.run("pandoc", ["-f", "markdown", "-t", "plain", tools.In(twin)], self.work_root))
        self.assertIn(b"Pages:", tools.run("pdfinfo", [tools.In(pdf)], self.work_root))
        cases = [("pandoc", ["-f", "markdown", "-t", "plain", str(key)]),                  # the key, by host path
                 ("pandoc", ["-f", "markdown", "-t", "plain", tools.In(twin), "/in/inference.key"]),   # beside the input
                 ("pandoc", ["-f", "markdown", "-t", "plain", str(twin)]),                 # the input, unmounted
                 ("pdfinfo", [str(pdf)])]                                                  # the cache
        for name, args in cases:
            with self.subTest(args=[str(a) for a in args]), self.assertRaises(BenchError) as ctx:
                tools.run(name, args, self.work_root)
            self.assertEqual(ctx.exception.code, Code.TOOL_FAILED)
        self.assert_work_root_empty()

    def test_pandoc_sandbox_blocks_input(self):
        sentinel = self.root / "sentinel.tex"
        sentinel.write_text("SENTINELTEXT\n")
        # the file is mounted (as a metadata value, not an input of the command line): only the
        # sandbox keeps \input from reading it
        out = tools.run("pandoc", ["-f", "latex", "-t", "plain", tools.In(sentinel, prefix="--metadata=x:")],
                        self.work_root, stdin=b"Before \\input{/in/sentinel.tex} after.\n")
        self.assertIn(b"Before", out)
        self.assertNotIn(b"SENTINELTEXT", out)

    def test_pandoc_stops_at_the_cap_of_its_runtime(self):
        with self.assertRaises(BenchError) as ctx:
            tools.run("pandoc", ["-f", "html", "-t", "json"], self.work_root, stdin=growing_html(60000), memory_mb=64)
        self.assertEqual(ctx.exception.code, Code.TOOL_MEMORY_LIMIT)
        self.assertTrue(ctx.exception.detail.endswith("exit 251"))
        # the same cap leaves an ordinary input alone
        out = tools.run("pandoc", ["-f", "html", "-t", "json"], self.work_root, stdin=growing_html(10), memory_mb=64)
        self.assertIn(b'"Para"', out)

    def test_poppler_allocation_failure_with_exit_0_is_a_memory_limit(self):
        pdf = self.root / "page.pdf"
        pdf.write_bytes(make_pdf(["a page rendered at 1200 dpi"]))
        # a Letter page at 1200 dpi is a 400 MB bitmap, above a 128 MiB data limit: poppler prints
        # "Out of memory" and exits 0 with an incomplete page
        with self.assertRaises(BenchError) as ctx:
            tools.run("pdftoppm", ["-r", "1200", "-png", "-singlefile", tools.In(pdf),
                                   tools.Out(self.root / "big.png", arg="big", produced="big.png")],
                      self.work_root, memory_mb=128)
        self.assertEqual(ctx.exception.code, Code.TOOL_MEMORY_LIMIT)
        self.assertTrue(ctx.exception.detail.endswith("exit 0"))
        tools.run("pdftoppm", ["-r", "72", "-png", "-singlefile", tools.In(pdf),
                               tools.Out(self.root / "small.png", arg="small", produced="small.png")],
                  self.work_root, memory_mb=128)
        self.assertTrue((self.root / "small.png").read_bytes().startswith(b"\x89PNG"))

    def test_a_timed_out_container_is_removed(self):
        """E6: the container is stopped at the timeout and its removal attested by the launcher."""
        ps = self.root / "endless.ps"
        ps.write_bytes(b"%!PS\n{ 1 pop } loop\n")
        with self.assertRaises(BenchError) as ctx:
            tools.run("gs", ["-q", "-dNOPAUSE", "-dBATCH", "-dSAFER", "-sDEVICE=nullpage", tools.In(ps)],
                      self.work_root, timeout=3)
        self.assertEqual((ctx.exception.code, ctx.exception.fatal, ctx.exception.detail), (Code.TOOL_FAILED, False, "gs: timeout"))
        self.assert_work_root_empty()


if __name__ == "__main__":
    unittest.main()
