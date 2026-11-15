"""Isolation and caps of the engine for real (criterion 12.3 of MFL032, native part).

The container of a conversion: no network, read-only root, the caller's user, only its inputs
visible (the runner, shared with the bench, is tested in `tests/unit/bench/test_bench_tools.py`);
here the caps that the engine adds: /work as a whole, one file, the observer that stops a call,
the deterministic container name and the memory peak. The light child: no network, no host
paths, a read-only root, only its scratch writable.
"""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from engine_test_support import require_toolchain
from markdown_for_llms.engine import isolation, toolchain, tools
from markdown_for_llms.engine.codes import Code, EngineError


class Base(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)

    def left(self, attempt):
        proc = toolchain.engine_run(["ps", "-a", "--filter", f"name=mdf-eng-{attempt}-", "--format", "{{.Names}}"], 60)
        return proc.returncode, proc.stdout.strip()


class ContainerCapTests(Base):
    def setUp(self):
        super().setUp()
        require_toolchain()

    def big_markdown(self, mib):
        src = self.root / "big.md"
        src.write_text(("word " * 200 + "\n\n") * (mib * 1024), encoding="utf-8")
        return src

    def test_work_cap_stops_the_call_and_removes_the_container(self):
        src = self.big_markdown(3)
        steps = []
        watch = tools.Watch(1, observer=None, attempt="captest1", steps=steps)
        with self.assertRaises(EngineError) as ctx:
            tools.run("pandoc", ["-f", "markdown", "-t", "plain", "-o", tools.Out(self.root / "o.txt", arg="o.txt"),
                                 tools.In(src)], self.root / "tools", memory_mb=1024, watch=watch)
        self.assertEqual(ctx.exception.code, Code.OUTPUT_LIMIT)
        self.assertEqual(self.left("captest1"), (0, b""))
        self.assertEqual(os.listdir(self.root / "tools"), [])

    def test_work_cap_is_checked_when_the_tool_ends_before_a_poll(self):
        """HA6 (Carbon I3): a tool that fills /work and exits between two polls is still over the cap."""
        src = self.big_markdown(3)
        watch = tools.Watch(1, fsize_mib=64, observer=None, attempt="captest4", poll_seconds=120)
        with self.assertRaises(EngineError) as ctx:
            tools.run("pandoc", ["-f", "markdown", "-t", "plain", "-o", tools.Out(self.root / "o.txt", arg="o.txt"),
                                 tools.In(src)], self.root / "tools", memory_mb=1024, watch=watch)
        self.assertEqual(ctx.exception.code, Code.OUTPUT_LIMIT)
        self.assertEqual(self.left("captest4"), (0, b""))
        self.assertFalse((self.root / "o.txt").exists())

    def test_observer_cancels_and_names_the_container_before_the_start(self):
        ps = self.root / "endless.ps"
        ps.write_bytes(b"%!PS\n{ 1 pop } loop\n")
        events = []

        def observer(event):
            events.append(event)
            return "cancelled" if event["event"] == "tick" and len(events) > 3 else None
        watch = tools.Watch(64, observer=observer, attempt="captest2", poll_seconds=0.1)
        with self.assertRaises(EngineError) as ctx:
            tools.run("gs", ["-q", "-dNOPAUSE", "-dBATCH", "-dSAFER", "-sDEVICE=nullpage", tools.In(ps)],
                      self.root / "tools", timeout=60, watch=watch)
        self.assertEqual(ctx.exception.code, Code.CANCELLED)
        self.assertEqual(events[0]["event"], "tool")
        self.assertRegex(events[0]["container"], r"^mdf-eng-captest2-\d+$")
        self.assertEqual(self.left("captest2"), (0, b""))

    def test_deadline_is_a_tool_timeout(self):
        ps = self.root / "endless.ps"
        ps.write_bytes(b"%!PS\n{ 1 pop } loop\n")
        import time
        watch = tools.Watch(64, attempt="captest3", deadline=time.monotonic() + 2)
        with self.assertRaises(EngineError) as ctx:
            tools.run("gs", ["-q", "-dNOPAUSE", "-dBATCH", "-dSAFER", "-sDEVICE=nullpage", tools.In(ps)],
                      self.root / "tools", timeout=60, watch=watch)
        self.assertEqual(ctx.exception.code, Code.TOOL_TIMEOUT)
        self.assertEqual(self.left("captest3"), (0, b""))

    def test_steps_record_arguments_without_host_paths_and_the_memory_peak(self):
        src = self.root / "in.md"
        src.write_text("# T\n\ntext\n", encoding="utf-8")
        steps = []
        out = tools.run("pandoc", ["-f", "markdown", "-t", "plain", tools.In(src)], self.root / "tools",
                        memory_mb=512, watch=tools.Watch(64, steps=steps))
        self.assertIn(b"text", out)
        (step,) = steps
        self.assertEqual(step["tool"], "pandoc")
        self.assertEqual(step["args"], ["pandoc", "+RTS", "-M512m", "-RTS", "--sandbox", "-f", "markdown", "-t", "plain",
                                        "/in/in.md"])
        self.assertGreater(step["memory_peak_bytes"], 0)
        self.assertFalse([a for a in step["args"] if str(self.root) in a])


class LightChildTests(Base):
    def test_isolation_probe(self):
        """No network (loopback refused, no DNS), host paths hidden, read-only root, scratch writable."""
        secret = self.root / "host-secret.txt"
        secret.write_text("not for the child")
        scratch = self.root / "child"
        rc, log = isolation.run(scratch, "markdown_for_llms.engine.selfprobe", [str(secret)], timeout=60)
        self.assertEqual(rc, 0, log)
        probe = json.loads((scratch / "probe.json").read_text())
        self.assertEqual(probe, {"loopback_refused": True, "dns_failed": True, "host_path_hidden": True,
                                 "root_read_only": True, "docker_socket_absent": True, "scratch_writable": True})

    def test_inputs_are_read_only_and_alone(self):
        argv = isolation.command(self.root / "s", "markdown_for_llms.engine.pkgaudit", ["docx", "/in/source.docx"],
                                 inputs=[(self.root / "a.docx", "source.docx")])
        self.assertEqual(argv[:7], ["/usr/bin/bwrap", "--unshare-all", "--die-with-parent", "--new-session",
                                    "--cap-drop", "ALL", "--clearenv"])
        i = argv.index("--ro-bind", argv.index("/mdf/markdown_for_llms") - 2)
        self.assertIn(["--ro-bind", os.path.realpath(self.root / "a.docx"), "/in/source.docx"],
                      [argv[k:k + 3] for k in range(len(argv) - 2)])
        self.assertIn("--remount-ro", argv)
        self.assertEqual(argv[argv.index("--bind") + 2], "/scratch")
        self.assertGreater(i, 0)
        with self.assertRaises(ValueError):
            isolation.command(self.root / "s", "m", inputs=[(self.root / "a", "../x")])

    def test_without_bubblewrap_there_is_no_fallback(self):
        with mock.patch.object(isolation, "BWRAP", str(self.root / "no-bwrap")), self.assertRaises(EngineError) as ctx:
            isolation.run(self.root / "child", "markdown_for_llms.engine.selfprobe", timeout=5)
        self.assertEqual(ctx.exception.code, Code.ISOLATION_UNAVAILABLE)

    def test_scratch_cap_is_checked_when_the_child_ends_before_a_poll(self):
        """HA6 (Carbon I3): the scratch and the log are checked once more after a normal exit."""
        with mock.patch.object(isolation, "scratch_bytes", return_value=10 << 30), self.assertRaises(EngineError) as ctx:
            isolation.run(self.root / "child", "markdown_for_llms.engine.selfprobe", [str(self.root / "x")], timeout=60,
                          output_mib=1, poll_seconds=120)
        self.assertEqual(ctx.exception.code, Code.OUTPUT_LIMIT)

    def test_audit_memory_follows_a_lower_cap_of_the_conversion(self):
        """HA1 (Sol H1): the child of the audit never gets more memory than the conversion allows."""
        from markdown_for_llms.engine import native
        from markdown_for_llms.engine.core import convert
        from engine_test_support import make_docx, para
        src = make_docx(self.root / "d.docx", para("x"))
        seen = []

        def fake_run(*args, **kwargs):
            seen.append(kwargs["memory_mib"])
            raise EngineError(Code.ISOLATION_UNAVAILABLE, "stop here")
        with mock.patch.object(isolation, "run", fake_run):
            for cap in (512, 4096):
                convert(src, self.root / f"o{cap}", {}, {"memory_mib": cap}, work=self.root / "w")
        self.assertEqual(seen, [512, native.AUDIT_MEMORY_MIB])

    def test_observer_can_stop_the_child(self):
        with self.assertRaises(EngineError) as ctx:
            isolation.run(self.root / "child", "markdown_for_llms.engine.selfprobe", timeout=60,
                          observer=lambda event: "interrupted")
        self.assertEqual(ctx.exception.code, Code.INTERRUPTED)


if __name__ == "__main__":
    unittest.main()
