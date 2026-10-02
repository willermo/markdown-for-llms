"""Verify preservation of context and detection of stale review inputs, without AI dependencies."""

import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
from types import SimpleNamespace
import unittest


PROJECT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("run_context", PROJECT / "scripts/run_context.py")
CONTEXT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CONTEXT)


class RunContextTests(unittest.TestCase):
    def setUp(self):
        self.workspace = tempfile.TemporaryDirectory()
        self.addCleanup(self.workspace.cleanup)
        self.root = Path(self.workspace.name)
        self.git("init", "-q", "-b", "fixture-base")
        (self.root / ".gitignore").write_text("temp/\n", encoding="utf-8")
        (self.root / "app.py").write_text("value = 1\n", encoding="utf-8")
        template = Path("documentation/development/templates/run-state.md")
        (self.root / template).parent.mkdir(parents=True)
        shutil.copyfile(PROJECT / template, self.root / template)
        self.git("add", ".")
        self.git("-c", "user.name=Context test", "-c", "user.email=test@example.invalid", "commit", "-qm", "baseline")
        self.git("branch", "dev")
        self.args = SimpleNamespace(
            run_id="run-a001-fase0-test", phase="0.1", title="Test run",
            branch="feature/test", base="dev", label="plan-r001", artifact=[],
        )
        CONTEXT.initialize(self.root, self.args)
        self.run = self.root / "temp" / self.args.run_id

    def git(self, *args):
        return subprocess.check_output(
            ["git", "-C", str(self.root), "-c", "commit.gpgsign=false",
             "-c", "core.hooksPath=" + str(self.root / ".git/test-hooks"), *args],
            stderr=subprocess.PIPE,
        )

    def test_bootstrap_and_reinitialization_preserve_existing_context(self):
        handover = self.root / "temp/HANDOVER.md"
        handover.write_text("Important unfinished work\n", encoding="utf-8")
        CONTEXT.bootstrap(self.root)
        with self.assertRaises(ValueError):
            CONTEXT.initialize(self.root, self.args)
        self.assertEqual(handover.read_text(), "Important unfinished work\n")
        state = (self.run / "STATE.md").read_text()
        self.assertNotIn("{{", state)
        self.assertIn("PREPARATION", state)

    def test_snapshot_tracks_plan_tracked_untracked_and_deleted_files(self):
        plan = self.run / "plans/plan-r001.md"
        plan.write_text("Plan A\n", encoding="utf-8")
        self.args.artifact = [plan.relative_to(self.root).as_posix()]
        CONTEXT.snapshot(self.root, self.args)
        self.assertEqual(CONTEXT.verify(self.root, self.args), 0)
        plan.write_text("Plan B\n", encoding="utf-8")
        self.assertEqual(CONTEXT.verify(self.root, self.args), 2)
        plan.write_text("Plan A\n", encoding="utf-8")
        (self.root / "new.py").write_text("unreviewed = True\n", encoding="utf-8")
        self.assertEqual(CONTEXT.verify(self.root, self.args), 2)
        (self.root / "new.py").unlink()
        (self.root / "app.py").write_text("value = 2\n", encoding="utf-8")
        self.assertEqual(CONTEXT.verify(self.root, self.args), 2)
        (self.root / "app.py").unlink()
        self.assertEqual(CONTEXT.verify(self.root, self.args), 2)
        with self.assertRaises(ValueError):
            CONTEXT.snapshot(self.root, self.args)

    def test_other_reports_and_archiving_do_not_change_review_input(self):
        CONTEXT.snapshot(self.root, self.args)
        (self.run / "reviews/review-chatgpt.md").write_text("Review in progress\n", encoding="utf-8")
        archive = self.root / "documentation/runs/example"
        archive.mkdir(parents=True)
        (archive / "report.md").write_text("Archived evidence\n", encoding="utf-8")
        self.assertEqual(CONTEXT.verify(self.root, self.args), 0)

    def test_traversal_and_context_symlinks_are_rejected(self):
        with self.assertRaises(ValueError):
            CONTEXT.run_path(self.root, "../../outside", must_exist=False)
        self.args.artifact = ["../outside.md"]
        with self.assertRaises(ValueError):
            CONTEXT.snapshot(self.root, self.args)
        outside = self.root / "outside"
        outside.mkdir()
        shutil.rmtree(self.root / "temp")
        try:
            (self.root / "temp").symlink_to(outside, target_is_directory=True)
        except OSError:
            self.skipTest("Symlinks are not available on this host")
        with self.assertRaises(ValueError):
            CONTEXT.bootstrap(self.root)
        self.assertEqual(list(outside.iterdir()), [])

    def test_init_does_not_change_git_and_requires_ignored_context(self):
        before = self.git("status", "--porcelain")
        self.assertEqual(before, b"")
        self.assertNotEqual(self.git("branch", "--show-current").decode().strip(), self.args.branch)
        (self.root / ".gitignore").write_text("", encoding="utf-8")
        with self.assertRaises(ValueError):
            CONTEXT.bootstrap(self.root)

    def test_snapshot_contains_fingerprints_without_file_contents(self):
        CONTEXT.snapshot(self.root, self.args)
        saved = json.loads((self.run / "snapshots/plan-r001.json").read_text())
        self.assertEqual(saved["schema"], 1)
        app = next(item for item in saved["files"] if item["path"] == "app.py")
        self.assertEqual(len(app["sha256"]), 64)
        self.assertNotIn("value = 1", json.dumps(saved))


if __name__ == "__main__":
    unittest.main()
