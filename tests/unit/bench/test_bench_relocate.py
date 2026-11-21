"""MFL044 (d): a run that moves, and its report read-only (`mdf bench report --report-dir`).

The state of schema 2 records the paths of the run so that it can move: a manifest and
exclusions given by hand are copied into `RUN/inputs/`, the runs directory is relative to the run,
the cache and the model root are hints replaced by options and verified by what the state
recorded; a missing path without its option is `run_path_missing`; a state of schema 1 stays
readable. `--report-dir` takes no lock and writes nothing in the run, and the state read again at
the end must be the same (`report_run_changed`). The engine is the real one on its `text` route."""
import contextlib
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock

from bench_run_support import Engine, Died, build_set, patches
from markdown_for_llms.bench import cli, report as report_mod, run
from markdown_for_llms.bench.codes import BenchError, Code
from markdown_for_llms.bench.settings import BenchSettings

PACKAGED_EXCLUSIONS = Path(run.__file__).parent / "data" / "exclusions.json"


def fingerprint(root):
    """Every file and directory under `root`: path, size, SHA-256 and modification time."""
    out = {}
    for p in sorted(Path(root).rglob("*")):
        info = os.lstat(p)
        digest = hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() and not p.is_symlink() else None
        out[str(p.relative_to(root))] = (info.st_mode, info.st_size, info.st_mtime_ns, digest)
    return out


class RelocateCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.s = build_set(self.base)
        self.exclusions = self.base / "my-exclusions.json"
        shutil.copyfile(PACKAGED_EXCLUSIONS, self.exclusions)
        self.out = self.s["runs"] / "r1"

    @contextlib.contextmanager
    def patched(self, engine=None):
        with contextlib.ExitStack() as stack:
            for obj, name, value in patches(self.s["venv"], engine or Engine()):
                stack.enter_context(mock.patch.object(obj, name, value))
            self.printed = stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            yield

    def start(self, engine=None):
        settings = BenchSettings.load(None, environ={}, cache=self.s["cache"], runs=self.s["runs"])
        with self.patched(engine):
            return run.start(settings, level="quick", manifest=str(self.s["manifest"]),
                             exclusions=str(self.exclusions), out=str(self.out))

    def cli(self, *argv, engine=None):
        err = io.StringIO()
        with self.patched(engine), contextlib.redirect_stderr(err):
            code = cli.main([str(a) for a in argv])
        return code, err.getvalue().strip()

    def state(self, where=None):
        return json.loads(((where or self.out) / "state.json").read_bytes())


class StateTests(RelocateCase):
    def test_schema_2_copies_the_inputs_and_records_relative_paths(self):
        self.assertEqual(self.start(), 0)
        st = self.state()
        self.assertEqual(st["schema"], "mdf-bench-run/2")
        self.assertEqual((st["manifest"]["path"], st["exclusions"]["path"], st["runs_dir"]),
                         ("inputs/manifest.json", "inputs/exclusions.json", ".."))
        self.assertEqual((self.out / "inputs" / "manifest.json").read_bytes(), self.s["manifest"].read_bytes())
        self.assertEqual((self.out / "inputs" / "exclusions.json").read_bytes(), self.exclusions.read_bytes())
        self.assertEqual(st["origin"], {"run_dir": str(self.out), "runs_dir": str(self.s["runs"])})
        self.assertEqual(st["cache"], str(self.s["cache"]))

    def test_model_root_goes_with_resume_only(self):
        code, err = self.cli("run", "--level", "quick", "--manifest", self.s["manifest"], "--cache", self.s["cache"],
                             "--runs-dir", self.s["runs"], "--out", self.out, "--model-root", self.base)
        self.assertEqual((code, err), (1, "error: invalid_setting"))
        self.assertFalse(self.out.exists())


class MovedRunTests(RelocateCase):
    def test_an_interrupted_run_moved_is_reported_resumed_and_reported(self):
        with self.assertRaises(Died):
            self.start(Engine("raise"))                     # the first attempt published, never recorded
        moved = self.base / "elsewhere" / "deeper" / "r1"
        moved.parent.mkdir(parents=True)
        shutil.move(self.out, moved)
        cache = self.base / "moved-cache"
        shutil.move(self.s["cache"], cache)
        self.s["manifest"].unlink()                         # the run has its own copies
        self.exclusions.unlink()
        code, err = self.cli("report", moved, "--report-dir", self.base / "rep0")
        self.assertEqual((code, err), (1, "error: report_run_not_finished"))
        code, err = self.cli("run", "--resume", moved)
        self.assertEqual((code, err), (1, "error: run_path_missing"))      # the cache hint is gone
        engine = Engine()
        code, err = self.cli("run", "--resume", moved, "--cache", cache, engine=engine)
        self.assertEqual((code, err), (0, ""))
        st = self.state(moved)
        self.assertEqual(st["status"], "complete")
        self.assertEqual(st["generations"], {"adopted": 1, "abandoned": 0})
        self.assertEqual(len(engine.calls), 2)
        before = fingerprint(moved)
        rep = self.base / "rep1"
        code, err = self.cli("report", moved, "--report-dir", rep, "--cache", cache)
        self.assertEqual((code, err), (0, ""))
        self.assertEqual(fingerprint(moved), before)                      # nothing written in the run
        self.assertEqual(json.loads((rep / "report.json").read_bytes())["documents"],
                         json.loads((moved / "report.json").read_bytes())["documents"])
        self.assertEqual(sorted(p.name for p in (rep / "attempts").iterdir()),
                         sorted(a for a in st["attempts"] if (moved / "attempts" / a / "results.json").exists()))

    def test_a_missing_path_without_its_option(self):
        self.assertEqual(self.start(), 0)
        (self.out / "inputs" / "exclusions.json").unlink()
        code, err = self.cli("report", self.out, "--report-dir", self.base / "rep")
        self.assertEqual((code, err), (1, "error: run_path_missing"))
        code, err = self.cli("report", self.out, "--report-dir", self.base / "rep", "--exclusions", PACKAGED_EXCLUSIONS)
        self.assertEqual((code, err), (0, ""))                            # the same bytes: verified by SHA-256
        other = self.base / "other.json"
        other.write_bytes(PACKAGED_EXCLUSIONS.read_bytes() + b"\n")
        code, err = self.cli("report", self.out, "--report-dir", self.base / "rep2", "--exclusions", other)
        self.assertEqual((code, err), (1, "error: report_identity_changed"))
        self.assertFalse((self.base / "rep2").exists())


class ReportDirTests(RelocateCase):
    def setUp(self):
        super().setUp()
        self.assertEqual(self.start(), 0)

    def test_read_only_without_the_lock_of_the_run(self):
        before = fingerprint(self.out)
        ledger_dir = fingerprint(self.s["runs"])
        rep = self.base / "rep"
        fd = os.open(self.out / "run.lock", os.O_RDWR)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)                  # another process holds the run
            code, err = self.cli("report", self.out, "--report-dir", rep)
        finally:
            os.close(fd)
        self.assertEqual((code, err), (0, ""))
        self.assertEqual(fingerprint(self.out), before)
        self.assertEqual(fingerprint(self.s["runs"]), ledger_dir)
        self.assertTrue((rep / "report.json").is_file() and (rep / "report.md").is_file())
        written = sorted(p.relative_to(rep).as_posix() for p in rep.rglob("results.json"))
        self.assertEqual(len(written), 2)                                 # D01 and D02 have checks

    def test_the_report_directory_must_be_new_or_empty_and_outside(self):
        busy = self.base / "busy"
        busy.mkdir()
        (busy / "x").write_text("x")
        for target in (busy, self.out / "rep", self.out):
            with self.subTest(target=target):
                code, err = self.cli("report", self.out, "--report-dir", target)
                self.assertEqual((code, err), (1, "error: invalid_setting"))
        self.assertFalse((self.out / "rep").exists())

    def test_a_state_that_changes_during_the_report(self):
        real = report_mod.build

        def build(run_obj, **kw):
            result = real(run_obj, **kw)
            data = (self.out / "state.json").read_bytes()
            (self.out / "state.json").write_bytes(data + b"\n")          # a resumption wrote meanwhile
            return result
        rep = self.base / "rep"
        with mock.patch.object(report_mod, "build", build):
            code, err = self.cli("report", self.out, "--report-dir", rep)
        self.assertEqual((code, err), (1, "error: report_run_changed"))
        self.assertFalse(rep.exists())

    def test_schema_1_is_read_with_its_absolute_paths(self):
        st = self.state()
        old = dict(st, schema="mdf-bench-run/1", runs_dir=str(self.base / "gone-runs"),
                   manifest={"path": str(self.s["manifest"]), "sha256": st["manifest"]["sha256"]},
                   exclusions={"path": str(self.exclusions), "sha256": st["exclusions"]["sha256"]})
        del old["origin"]
        shutil.rmtree(self.out / "inputs")
        (self.out / "state.json").write_text(json.dumps(old))
        code, err = self.cli("report", self.out, "--report-dir", self.base / "rep")
        self.assertEqual((code, err), (0, ""))
        paths = run.run_paths(self.out, old)
        self.assertEqual(paths["runs_dir"], self.s["runs"])               # the recorded one is gone: the parent
        self.assertFalse(paths["in_place"])
        self.s["manifest"].unlink()
        code, err = self.cli("report", self.out, "--report-dir", self.base / "rep2")
        self.assertEqual((code, err), (1, "error: run_path_missing"))


if __name__ == "__main__":
    unittest.main()
