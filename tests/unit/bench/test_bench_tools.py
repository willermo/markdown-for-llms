"""External tools of the bench: the container runner, its mounts, caps, timeouts and LibreOffice.

`RealToolchainTests` run the tools for real in the reference image (E1 of the toolchain run); the
fake engine covers only the exits a real tool cannot be made to give on demand.
"""
import os
from pathlib import Path
import re
import subprocess
import tempfile
import unittest
import unittest.mock

from bench_test_support import FAKE_IMAGE, FakeEngine, growing_html, make_pdf, require_toolchain
from markdown_for_llms.bench import toolchain, tools
from markdown_for_llms.bench.codes import BenchError, Code


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
        self.assertEqual(sorted(os.listdir(self.work_root)), [])       # every /work removed after its call


class FakeEngineTests(Base):
    """The runner against a fake Docker client, with the image taken as verified."""

    def setUp(self):
        super().setUp()
        saved = dict(toolchain._state)
        self.addCleanup(toolchain._state.update, saved)
        toolchain._state.update(image=FAKE_IMAGE, engine="test")

    def engine(self, **kw):
        fake = FakeEngine(**kw)
        patcher = unittest.mock.patch.object(toolchain, "engine_run", fake)
        patcher.start()
        self.addCleanup(patcher.stop)
        return fake


def option_values(argv, flag):
    return [argv[i + 1] for i, a in enumerate(argv) if a == flag]


class RunnerArgumentTests(FakeEngineTests):
    def test_isolation_mounts_environment_and_caps(self):
        fake = self.engine()
        src = self.root / "runs" / "r1" / "in,put file.pdf"
        src.parent.mkdir(parents=True)
        src.write_bytes(b"%PDF")
        (src.parent / "inference.key").write_text("not for tools")
        before = tools.memory_limit()
        self.addCleanup(tools.set_memory_limit, before)
        tools.set_memory_limit(1234)
        tools.run("pdftotext", ["-raw", tools.In(src), "-"], self.work_root)
        tools.run("pandoc", ["-f", "latex", "-t", "json"], self.work_root, stdin=b"x")
        (poppler, pandoc) = fake.runs()
        for argv in (poppler, pandoc):
            self.assertEqual(argv[:3], ["run", "--rm", "-i"])
            self.assertRegex(argv[4], r"^mdf-bench-\d+-[0-9a-f]{8}-\d+$")       # a name per call (E6)
            for flag in ("--network none", "--cap-drop ALL", "--security-opt no-new-privileges"):
                self.assertIn(flag, " ".join(argv))
            self.assertIn("--read-only", argv)
            self.assertEqual(option_values(argv, "--user"), [f"{os.getuid()}:{os.getgid()}"])
            self.assertEqual(option_values(argv, "-w"), ["/work"])
            self.assertEqual(option_values(argv, "-e"), ["LC_ALL=C.UTF-8", "LANG=C.UTF-8", "HOME=/work/home",
                                                         "TMPDIR=/work/tmp"])
            self.assertNotIn("-v", argv)
            self.assertNotIn("--volume", argv)
            self.assertFalse([a for a in argv if "docker.sock" in a])
            self.assertIn(FAKE_IMAGE, argv)                                    # by ID, never by tag
            self.assertFalse([a for a in argv if a.startswith(toolchain.REPOSITORY + ":")])
            self.assertEqual(option_values(argv, "--memory"), ["1746m"])        # cap + 512 MiB, no swap
            self.assertEqual(option_values(argv, "--memory-swap"), ["1746m"])
        mounts = option_values(poppler, "--mount")
        self.assertEqual(len(mounts), 2)
        work = re.fullmatch(r"type=bind,src=(.+),dst=/work", mounts[0])
        self.assertIsNotNone(work)
        self.assertTrue(Path(work.group(1)).parent == self.work_root.resolve())   # private, under the cache
        self.assertEqual(mounts[1], f'type=bind,"src={src.resolve()}",dst=/in/in_put_file.pdf,readonly')
        self.assertEqual(poppler[poppler.index(FAKE_IMAGE) + 1:], ["pdftotext", "-raw", "/in/in_put_file.pdf", "-"])
        self.assertEqual(option_values(poppler, "--ulimit"), [f"data={1234 * 1024 * 1024}:{1234 * 1024 * 1024}"])
        self.assertEqual(pandoc[pandoc.index(FAKE_IMAGE) + 1:],
                         ["pandoc", "+RTS", "-M1234m", "-RTS", "--sandbox", "-f", "latex", "-t", "json"])
        self.assertEqual(option_values(pandoc, "--ulimit"), [])               # the runtime's own cap
        self.assertEqual(len(option_values(pandoc, "--mount")), 1)            # no input file: /work only
        self.assertEqual(os.listdir(self.work_root), [])

    def test_the_work_directory_is_private_and_removed(self):
        seen = {}

        def result(argv, stdin):
            src = re.fullmatch(r"type=bind,src=(.+),dst=/work", option_values(argv, "--mount")[0]).group(1)
            seen["mode"] = os.stat(src).st_mode & 0o777
            seen["dirs"] = sorted(os.listdir(src))
            return 0, b"", b""
        self.engine(result=result)
        tools.run("qpdf", ["--version"], os.path.relpath(self.work_root))      # a relative cache still works
        self.assertEqual(seen, {"mode": 0o700, "dirs": ["home", "tmp"]})
        self.assertEqual(os.listdir(self.work_root), [])

    def test_outputs_are_moved_and_must_exist(self):
        def result(argv, stdin):
            work = re.fullmatch(r"type=bind,src=(.+),dst=/work", option_values(argv, "--mount")[0]).group(1)
            if "make" in argv:
                Path(work, "page.png").write_bytes(b"png")
            if "empty" in argv:
                Path(work, "page.png").write_bytes(b"")
            return 0, b"", b""
        self.engine(result=result)
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


class RunnerExitTests(FakeEngineTests):
    def run_tool(self, name, rc, stderr=b""):
        self.engine(result=lambda argv, stdin: (rc, b"", stderr))
        with self.assertRaises(BenchError) as ctx:
            tools.run(name, ["x"], self.work_root)
        return ctx.exception

    def test_a_killed_container_is_a_memory_limit_for_every_tool(self):
        for name in ("pandoc", "soffice", "pdftoppm"):
            with self.subTest(tool=name):
                exc = self.run_tool(name, 137)
                self.assertEqual((exc.code, exc.detail.endswith("exit 137")), (Code.TOOL_MEMORY_LIMIT, True))

    def test_runtime_and_allocation_failures(self):
        self.assertEqual(self.run_tool("pandoc", 251, b"pandoc: Heap exhausted;\n").code, Code.TOOL_MEMORY_LIMIT)
        self.assertEqual(self.run_tool("pdftoppm", 0, b"Out of memory\n").code, Code.TOOL_MEMORY_LIMIT)
        self.assertEqual(self.run_tool("qpdf", 2, b"qpdf: not a PDF file\n").code, Code.TOOL_FAILED)

    def test_engine_failures_are_not_tool_failures(self):
        for rc in (125, 126, 127):
            with self.subTest(rc=rc):
                exc = self.run_tool("qpdf", rc)
                self.assertEqual((exc.code, exc.fatal), (Code.ENGINE_FAILED, False))

    def timeout(self, argv, stdin):
        raise subprocess.TimeoutExpired(argv, 1)

    def test_timeout_removes_the_container_by_name(self):
        fake = self.engine(result=self.timeout)
        with self.assertRaises(BenchError) as ctx:
            tools.run("gs", ["x"], self.work_root, timeout=1)
        self.assertEqual((ctx.exception.code, ctx.exception.fatal), (Code.TOOL_FAILED, False))
        name = fake.runs()[0][4]
        self.assertEqual(fake.calls[1:], [["rm", "-f", name], ["container", "inspect", name]])
        self.assertEqual(os.listdir(self.work_root), [])

    def test_a_container_left_after_a_timeout_is_fatal(self):
        self.engine(result=self.timeout, remove_ok=False)
        with self.assertRaises(BenchError) as ctx:
            tools.run("gs", ["x"], self.work_root, timeout=1)
        self.assertEqual((ctx.exception.code, ctx.exception.fatal), (Code.TOOL_FAILED, True))

    def test_an_interrupted_client_removes_its_container(self):
        def interrupt(argv, stdin):
            raise KeyboardInterrupt
        fake = self.engine(result=interrupt)
        with self.assertRaises(KeyboardInterrupt):
            tools.run("gs", ["x"], self.work_root)
        name = fake.runs()[0][4]
        self.assertIn(["rm", "-f", name], fake.calls)

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
        fake = self.engine()
        self.assertEqual(tools.version("pandoc"), toolchain.spec()["versions"]["pandoc"])
        self.assertEqual(fake.calls, [])
        with self.assertRaises(ValueError):
            tools.version("curl")


class SofficeIsolationTests(FakeEngineTests):
    """LibreOffice runs in the private /work of its call: with a relative TMPDIR its PDF export
    failed with exit 0 and NSS wrote `pkcs11.txt` in the current directory (the repository root)."""

    def test_profile_home_output_and_input_are_in_the_private_work(self):
        def result(argv, stdin):
            work = re.fullmatch(r"type=bind,src=(.+),dst=/work", option_values(argv, "--mount")[0]).group(1)
            out = Path(work, "out")
            out.mkdir()
            (out / "twin.pdf").write_bytes(b"%PDF converted")
            return 0, b"", b""
        fake = self.engine(result=result)
        src = self.root / "cache" / "sources" / "abc"
        src.parent.mkdir(parents=True)
        src.write_bytes(b"pptx bytes")
        dest = self.root / "cache" / "work" / "selfcheck-D09"
        dest.mkdir(parents=True)
        out = tools.soffice_convert(src, "twin.pptx", "pdf", dest, self.work_root)
        self.assertEqual((out, out.read_bytes()), (dest / "twin.pdf", b"%PDF converted"))
        (argv,) = fake.runs()
        command = argv[argv.index(FAKE_IMAGE) + 1:]
        self.assertEqual(command, ["soffice", "-env:UserInstallation=file:///work/profile", "--headless", "--norestore",
                                   "--convert-to", "pdf", "--outdir", "/work/out", "/in/twin.pptx"])
        self.assertEqual(option_values(argv, "--mount")[1], f"type=bind,src={src.resolve()},dst=/in/twin.pptx,readonly")
        self.assertEqual(os.listdir(self.work_root), [])

    def test_an_export_without_output_fails_even_with_exit_0(self):
        self.engine()
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
        """E6: the client is killed at the timeout and the container removed by name."""
        ps = self.root / "endless.ps"
        ps.write_bytes(b"%!PS\n{ 1 pop } loop\n")
        with self.assertRaises(BenchError) as ctx:
            tools.run("gs", ["-q", "-dNOPAUSE", "-dBATCH", "-dSAFER", "-sDEVICE=nullpage", tools.In(ps)],
                      self.work_root, timeout=3)
        self.assertEqual((ctx.exception.code, ctx.exception.fatal, ctx.exception.detail), (Code.TOOL_FAILED, False, "gs: timeout"))
        left = toolchain.engine_run(["ps", "-a", "--filter", f"name=mdf-bench-{os.getpid()}-", "--format", "{{.Names}}"], 60)
        self.assertEqual((left.returncode, left.stdout.strip()), (0, b""))
        self.assert_work_root_empty()


if __name__ == "__main__":
    unittest.main()
