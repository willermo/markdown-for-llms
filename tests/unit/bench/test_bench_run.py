"""`mdf bench run`: attempts, state, locks, interruption, resumption and adoption (plan r2 §7, delta r3
Δ1, Δ2, Δ7, Δ8; C-COMANDI, C-ADOZIONE and C-ORDINE of the review of the delta).

The engine is the real one (its `text` route needs no toolchain) driven by `bench_run_support`;
two processes are real processes where the plan asks for them."""
import contextlib
import errno
import fcntl
import io
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from bench_run_support import (FAKE_IMAGE, ROOT, Engine, Died, build_set, fake_identity, patches, spawn, tree_digest,
                               wait_for)
from markdown_for_llms.bench import cli, run, spend
from markdown_for_llms.bench.codes import BenchError, Code
from markdown_for_llms.bench.settings import BenchSettings
from markdown_for_llms.engine import contract, core
from markdown_for_llms.settings import Settings


@contextlib.contextmanager
def patched(venv, engine, identity=fake_identity):
    with contextlib.ExitStack() as stack:
        for obj, name, value in patches(venv, engine, identity):
            stack.enter_context(mock.patch.object(obj, name, value))
        stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
        yield


class RunCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.s = build_set(self.tmp.name)
        self.settings = BenchSettings.load(None, environ={}, cache=self.s["cache"], runs=self.s["runs"])
        self.out = self.s["runs"] / "r1"

    def tearDown(self):
        self.tmp.cleanup()

    def start(self, engine=None, identity=fake_identity, **options):
        options = {"level": "quick", "manifest": str(self.s["manifest"]), "out": str(self.out), **options}
        with patched(self.s["venv"], engine or Engine(), identity):
            return run.start(self.settings, **options)

    def resume(self, engine=None, identity=fake_identity):
        with patched(self.s["venv"], engine or Engine(), identity):
            return run.resume(self.out)

    def state(self):
        return json.loads((self.out / "state.json").read_bytes())

    def args(self, *extra):
        return ["--level", "quick", "--manifest", self.s["manifest"], "--cache", self.s["cache"],
                "--runs-dir", self.s["runs"], "--out", self.out, *extra]

    def died_after_publish(self):
        """A run whose first attempt published its result and was never recorded (state: started)."""
        with self.assertRaises(Died):
            self.start(Engine("raise"))
        first = next(iter(self.state()["attempts"].values()))
        self.assertEqual(first["status"], "started")
        return first, self.out / "attempts" / first["id"] / "out"

    def assert_not_adopted(self, first):
        engine = Engine()
        self.assertEqual(self.resume(engine), 0)
        st = self.state()
        self.assertEqual(st["attempts"][first["id"]]["status"], "abandoned")
        self.assertTrue((self.out / "attempts" / first["id"]).is_dir())          # kept
        name = run.key_name(first["document"], first["route"], first["v8"], first["k"])
        second = st["keys"][name][1]
        self.assertTrue(second.endswith("-g2"))
        self.assertEqual(st["attempts"][second]["status"], "done")
        self.assertEqual(engine.calls[0]["attempt"], second)
        self.assertEqual(st["generations"], {"adopted": 0, "abandoned": 1})


class AttemptIdTests(unittest.TestCase):
    def test_ids_unique_on_the_standard_campaign(self):
        from markdown_for_llms.bench import manifest
        docs = manifest.load().select(level="standard")
        keys = {}
        for ocr in ("auto", "force"):
            for key in run.plan_keys(docs, None, ocr, 3):
                for g in (1, 2, 3):
                    aid = run.attempt_id("0123abcd", key["document"], key["route"], key["v8"], key["k"], g)
                    self.assertTrue(core.ATTEMPT.fullmatch(aid))
                    self.assertLessEqual(len(aid), 40)
                    keys.setdefault(aid, set()).add((key["document"], key["route"], json.dumps(key["variant"]), key["k"], g))
        self.assertTrue(all(len(v) == 1 for v in keys.values()))        # one key per id
        self.assertGreater(len(keys), 600)
        marker = {run.variant_sha8(run.variant("marker", o)) for o in ("auto", "force")}
        self.assertEqual(len(marker), 2)

    def test_two_variants_with_one_v8_are_refused(self):
        import itertools
        from markdown_for_llms.bench import manifest
        docs = manifest.load().select(ids=["D04", "D05"])
        counter = itertools.count()
        with mock.patch.object(run, "variant_sha8", lambda v: "00000000"), \
                mock.patch.object(run, "variant", lambda route, ocr: {"n": next(counter)}):
            with self.assertRaises(BenchError) as ctx:
                run.plan_keys(docs, ["marker"], "auto", 1)
        self.assertEqual(ctx.exception.code, Code.ATTEMPT_ID_COLLISION)


class LedgerTests(unittest.TestCase):
    def test_reservations_and_completions(self):
        r = "run-1"
        entries = [{"kind": "reserve", "run_id": r, "attempt_id": "a-lite", "estimate_cents": "1.05"},
                   {"kind": "reserve", "run_id": r, "attempt_id": "a-fast", "estimate_cents": "1.14"},
                   {"kind": "complete", "run_id": r, "attempt_id": "a-lite", "billed_cents": "0.0375"},
                   {"kind": "reserve", "run_id": r, "attempt_id": "a-balanced", "estimate_cents": "1.40"},
                   {"kind": "complete", "run_id": r, "attempt_id": "a-fast", "billed_cents": None},
                   {"kind": "reserve", "run_id": r, "attempt_id": "a-accurate", "estimate_cents": "2.00"},
                   {"kind": "complete", "run_id": r, "attempt_id": "a-accurate", "billed_cents": 0},
                   {"kind": "reserve", "run_id": "run-2", "attempt_id": "a-lite", "estimate_cents": 1}]
        per, total = spend.reconcile(entries)
        self.assertEqual(str(per[(r, "a-lite")]), "0.0375")        # the completion replaces the estimate
        self.assertEqual(str(per[(r, "a-fast")]), "1.14")          # uncertain end: the estimate stays
        self.assertEqual(str(per[(r, "a-balanced")]), "1.40")      # no completion
        self.assertEqual(per[(r, "a-accurate")], 0)
        self.assertEqual(str(total), "3.5775")

    def test_inconsistent_ledgers(self):
        reserve = {"kind": "reserve", "run_id": "r", "attempt_id": "a", "estimate_cents": 1}
        complete = {"kind": "complete", "run_id": "r", "attempt_id": "a", "billed_cents": 1}
        for entries in ([complete], [reserve, reserve], [reserve, complete, complete],
                        [{"kind": "other", "run_id": "r", "attempt_id": "a"}], [dict(reserve, estimate_cents=-1)]):
            with self.assertRaises(BenchError) as ctx:
                spend.reconcile(entries)
            self.assertEqual(ctx.exception.code, Code.SPEND_LEDGER_INCONSISTENT)


class NewRunTests(RunCase):
    def test_refusals_leave_no_directory(self):
        cases = [({"docs": "D07"}, Code.UNKNOWN_DOCUMENT), ({"routes": "nothing"}, Code.ROUTE_INVALID),
                 ({"routes": "datalab"}, Code.INVALID_SETTING), ({"routes": "marker"}, Code.INVALID_SETTING),
                 ({"repeat": 11}, Code.INVALID_SETTING), ({"out": str(Path(self.tmp.name) / "elsewhere")}, Code.INVALID_SETTING),
                 ({"manifest": str(Path(self.tmp.name) / "missing.json")}, Code.MANIFEST_INVALID),
                 ({"exclusions": str(Path(self.tmp.name) / "missing.json")}, Code.MANIFEST_INVALID),
                 ({"docs": "D01", "level": "quick"}, Code.INVALID_SETTING)]
        for options, code in cases:
            options = {"level": None, **options} if "docs" in options and "level" not in options else options
            with self.assertRaises(BenchError) as ctx:
                self.start(**options)
            self.assertEqual(ctx.exception.code, code, options)
            self.assertFalse(self.s["runs"].exists(), options)
            self.assertFalse((Path(self.tmp.name) / "elsewhere").exists())

    def test_existing_directory_is_run_exists(self):
        self.out.mkdir(parents=True)
        with self.assertRaises(BenchError) as ctx:
            self.start()
        self.assertEqual(ctx.exception.code, Code.RUN_EXISTS)
        self.assertEqual(list(self.out.iterdir()), [])

    def test_caps_are_frozen_and_apart_from_engine_limits(self):
        engine = Engine()
        self.assertEqual(self.start(engine), 0)
        st = self.state()
        caps = Settings().engine_limits()
        self.assertEqual(st["caps"], {"D01": caps, "D02": caps, "D03": caps})
        for call in engine.calls:
            self.assertEqual(call["limits"], caps)
            self.assertEqual(st["attempts"][call["attempt"]]["caps"], caps)
            result = json.loads((self.out / "attempts" / call["attempt"] / "out" / "engine.json").read_bytes())
            self.assertEqual(result["limits"], [])     # the declared limitations of the route: a diagnostic
        self.assertEqual(len(engine.calls), 3)
        self.assertEqual({a["status"] for a in st["attempts"].values()}, {"done"})
        self.assertEqual(st["status"], "complete")
        self.assertTrue((self.out / "report.json").is_file() and (self.out / "report.md").is_file())
        self.assertEqual(os.stat(self.out).st_mode & 0o777, 0o700)

    def test_resume_of_a_missing_or_invalid_state(self):
        with self.assertRaises(BenchError) as ctx:
            self.resume()
        self.assertEqual(ctx.exception.code, Code.RESUME_STATE_MISSING)
        self.assertEqual(self.start(), 0)
        state = self.state()
        del state["selection"]
        (self.out / "state.json").write_text(json.dumps(state))
        with self.assertRaises(BenchError) as ctx:
            self.resume()
        self.assertEqual(ctx.exception.code, Code.RESUME_STATE_INVALID)

    def test_resume_takes_no_other_option(self):
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            code = cli.main(["run", "--resume", str(self.out), "--level", "quick"])
        self.assertEqual((code, err.getvalue().strip()), (1, "error: resume_option_not_allowed"))
        self.assertFalse(self.out.exists())

    def test_unavailable_document_exits_2(self):
        from bench_run_support import TEXT, sha
        source = self.s["cache"] / "sources" / sha(TEXT.encode("utf-8"))
        data = bytearray(source.read_bytes())
        data[0] ^= 1
        source.write_bytes(bytes(data))       # U6: a cached source altered
        self.assertEqual(self.start(), 2)
        report = json.loads((self.out / "report.json").read_bytes())
        self.assertTrue(all(d["status"] == "unavailable" for d in report["documents"]))


class ResumeTests(RunCase):
    def test_t1_killed_after_publish_is_adopted(self):
        proc = spawn("kill", self.s["base"], self.args())
        proc.communicate(timeout=120)
        self.assertEqual(proc.returncode, -signal.SIGKILL)
        first = next(iter(self.state()["attempts"].values()))
        self.assertEqual(first["status"], "started")
        engine = Engine()
        self.assertEqual(self.resume(engine), 0)
        st = self.state()
        self.assertEqual(st["attempts"][first["id"]]["status"], "done")
        self.assertTrue(st["attempts"][first["id"]]["adopted"])
        self.assertNotIn(first["id"], [c["attempt"] for c in engine.calls])
        self.assertEqual(len(engine.calls), 2)
        self.assertEqual(st["generations"], {"adopted": 1, "abandoned": 0})
        report = json.loads((self.out / "report.json").read_bytes())
        self.assertTrue(report["run"]["resumed"])

    def test_t2_sigint_during_a_conversion(self):
        proc = spawn("wait", self.s["base"], self.args())
        self.assertTrue(wait_for(self.s["base"] / "started", proc))
        proc.send_signal(signal.SIGINT)
        _, err = proc.communicate(timeout=60)
        self.assertEqual((proc.returncode, err.decode().strip()), (1, "error: interrupted"))
        first = next(iter(self.state()["attempts"].values()))
        self.assertEqual((first["status"], self.state()["status"]), ("interrupted", "interrupted"))
        result = json.loads((self.out / "attempts" / first["id"] / "out" / "engine.json").read_bytes())
        self.assertEqual(result["error"]["code"], "interrupted")
        self.assert_not_adopted(first)

    def test_t3_sigint_between_attempts_redoes_nothing_done(self):
        engine = Engine()
        real = engine.__call__

        def second_interrupted(*args, **kwargs):
            if len(engine.calls) == 1:
                raise KeyboardInterrupt
            return real(*args, **kwargs)
        with self.assertRaises(BenchError) as ctx:
            self.start(second_interrupted)
        self.assertEqual(ctx.exception.code, Code.INTERRUPTED)
        done = [a["id"] for a in self.state()["attempts"].values() if a["status"] == "done"]
        self.assertEqual(len(done), 1)
        engine2 = Engine()
        self.assertEqual(self.resume(engine2), 0)
        self.assertNotIn(done[0], [c["attempt"] for c in engine2.calls])
        self.assertEqual(len(engine2.calls), 2)

    def test_t4_changed_identity_is_refused(self):
        self.assertEqual(self.start(), 0)
        before = tree_digest(self.out)
        for identity in (lambda m: fake_identity(m, image="sha256:" + "e" * 64),
                         lambda m: dict(fake_identity(m), package={"version": "1.0.0", "record_sha256": "1" * 64})):
            engine = Engine()
            with self.assertRaises(BenchError) as ctx:
                self.resume(engine, identity)
            self.assertEqual(ctx.exception.code, Code.RESUME_IDENTITY_CHANGED)
            self.assertEqual(engine.calls, [])
            self.assertEqual(tree_digest(self.out), before)

    def test_t6_other_attempt_or_source_is_not_adopted(self):
        for field in ("attempt", "source"):
            with self.subTest(field=field):
                self.tearDown()
                self.setUp()
                first, out = self.died_after_publish()
                result = json.loads((out / "engine.json").read_bytes())
                if field == "attempt":
                    result["attempt"] = first["id"].replace("-g1", "-g9")
                else:
                    result["source"]["sha256"] = "0" * 64
                contract.write_result(out, result)
                self.assert_not_adopted(first)

    def test_t7_to_t11_unadoptable_outputs(self):
        def t7(out):
            (out / "engine.json").unlink()

        def t8(out):
            data = (out / "engine.json").read_bytes()
            (out / "engine.json").write_bytes(data[:len(data) // 2])

        def t9(out):
            os.rename(out / "engine.json", out / ".partial-engine.json")

        def t10(out):
            data = bytearray((out / "document.md").read_bytes())
            data[0] ^= 1
            (out / "document.md").write_bytes(bytes(data))

        def t11(out):
            result = json.loads((out / "engine.json").read_bytes())
            state = self.state()
            first = state["attempts"][result["attempt"]]
            first["request"]["datalab"] = {"mode": "lite", "transit_confirmed": True, "cap_usd": "0.02"}
            result["options"]["requested"]["datalab"] = {"mode": "lite", "transit_confirmed": True, "cap_usd": "0.03"}
            contract.write_result(out, result)
            run.save_state(self.out, state)
        for case in (t7, t8, t9, t10, t11):
            with self.subTest(case=case.__name__):
                self.tearDown()
                self.setUp()
                first, out = self.died_after_publish()
                case(out)
                self.assert_not_adopted(first)

    def test_adoption_ignores_engine_limits_and_checks_components(self):
        first, out = self.died_after_publish()
        result = json.loads((out / "engine.json").read_bytes())
        result["limits"] = ["a declared limitation of the route"]
        contract.write_result(out, result)
        engine = Engine()
        self.assertEqual(self.resume(engine), 0)
        self.assertTrue(self.state()["attempts"][first["id"]]["adopted"])
        self.assertEqual(len(engine.calls), 2)
        self.assertFalse(run.components_ok({"package": {"markdown-for-llms": "0.0.0"}}, fake_identity(None)))
        self.assertFalse(run.components_ok({"package": {"markdown-for-llms": fake_identity(None)["package"]["version"]},
                                            "toolchain": {"content_sha256": "x", "image_id": FAKE_IMAGE}},
                                           fake_identity(None)))

    def test_t12_unwritable_state_stops_without_a_new_attempt(self):
        engine = Engine()
        real, count = os.replace, []

        def replace(src, dst, *args, **kwargs):
            if Path(dst).name == "state.json":
                count.append(dst)
                if len(count) == 2:
                    raise OSError(errno.ENOSPC, "No space left on device")
            return real(src, dst, *args, **kwargs)
        with mock.patch("os.replace", replace):
            with self.assertRaises(BenchError) as ctx:
                self.start(engine)
        self.assertEqual(ctx.exception.code, Code.STATE_UNWRITABLE)
        self.assertEqual(engine.calls, [])
        self.assertEqual(self.resume(), 0)          # the next resumption reconciles

    def test_t13_adopted_failure_is_an_outcome_without_scores(self):
        first, out = self.died_after_publish()
        result = json.loads((out / "engine.json").read_bytes())
        for name in ("document.md", "assets"):
            target = out / name
            if target.is_dir():
                shutil.rmtree(target)
            elif target.exists():
                target.unlink()
        result = dict(result, status="failed", error={"code": "page_limit", "fatal": False}, diagnostics=[],
                      inventory_complete=False)
        result.pop("inventory", None)
        contract.write_result(out, result)
        self.assertEqual(self.resume(Engine()), 0)
        report = json.loads((self.out / "report.json").read_bytes())
        attempt = next(a for d in report["documents"] for a in d["attempts"] if a["attempt_id"] == first["id"])
        self.assertEqual((attempt["status"], attempt["error"], attempt["scores"], attempt["adopted"]),
                         ("failed", "page_limit", None, True))
        cell = next(c for c in report["summary"]["format_route"] if c["route"] == "text")
        self.assertEqual((cell["complete"], cell["failed"]), (2, 1))


class LockTests(RunCase):
    def test_run_against_resume_and_resume_against_resume(self):
        proc = spawn("wait", self.s["base"], self.args())
        try:
            self.assertTrue(wait_for(self.s["base"] / "started", proc))
            before = tree_digest(self.out)
            with self.assertRaises(BenchError) as ctx:
                self.resume()
            self.assertEqual(ctx.exception.code, Code.RUN_BUSY)
            self.assertEqual(tree_digest(self.out), before)
            proc.send_signal(signal.SIGINT)
            proc.communicate(timeout=60)
            self.assertEqual(proc.returncode, 1)
            (self.s["base"] / "started").unlink()
            proc = spawn("wait", self.s["base"], ["--resume", self.out])
            self.assertTrue(wait_for(self.s["base"] / "started", proc))
            before = tree_digest(self.out)
            with self.assertRaises(BenchError) as ctx:
                self.resume()
            self.assertEqual(ctx.exception.code, Code.RUN_BUSY)
            self.assertEqual(tree_digest(self.out), before)
        finally:
            proc.kill()                      # SIGKILL: the kernel releases the lock
            proc.communicate(timeout=60)
        self.assertEqual(self.resume(), 0)

    def test_sigint_from_a_non_interactive_shell_in_the_background(self):
        proc = spawn("wait", self.s["base"], self.args(), shell_background=True)
        try:
            self.assertTrue(wait_for(self.s["base"] / "started", proc))
            pid = int((self.s["base"] / "pid").read_text())
            with open(f"/proc/{pid}/status") as fh:
                ignored = int(next(x for x in fh if x.startswith("SigIgn:")).split()[1], 16)
            os.kill(pid, signal.SIGINT)
            proc.communicate(timeout=30)
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.communicate()
        self.assertEqual(ignored & (1 << (signal.SIGINT - 1)), 0)      # the bench restored the handler
        self.assertEqual(proc.returncode, 1)
        self.assertEqual(self.state()["status"], "interrupted")

    def test_venv_busy_while_the_full_check_holds_it(self):
        fd = os.open(self.s["venv"], os.O_RDONLY | os.O_DIRECTORY)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaises(BenchError) as ctx:
                self.start()
            self.assertEqual(ctx.exception.code, Code.VENV_BUSY)
        finally:
            os.close(fd)
        self.assertFalse(self.s["runs"].exists())

    def test_venv_in_use_by_another_program_but_not_by_a_bench(self):
        venv_bin = self.s["venv"] / "bin"
        for name in ("fakeapi", "fakebench"):
            (venv_bin / name).write_text("#!/bin/sh\nsleep 60\n")
            (venv_bin / name).chmod(0o755)

        def detached(command):
            out = subprocess.run(["bash", "-c", command + " >/dev/null 2>&1 & echo $!"], capture_output=True, text=True,
                                 cwd=self.s["base"], check=True)
            return int(out.stdout)
        bench = detached("venv/bin/fakebench -m markdown_for_llms.cli bench run")
        try:
            self.assertEqual(self.start(), 0)               # another bench is not a user of the venv
            api = detached("venv/bin/fakeapi serve")
            try:
                self.assertIn(api, run.venv_users(self.s["venv"]))
                self.out = self.s["runs"] / "r2"
                with self.assertRaises(BenchError) as ctx:
                    self.start()
                self.assertEqual(ctx.exception.code, Code.VENV_IN_USE)
                self.assertFalse(self.out.exists())
            finally:
                os.kill(api, signal.SIGKILL)
        finally:
            os.kill(bench, signal.SIGKILL)


class InterruptHarnessTests(unittest.TestCase):
    def harness(self, child, evidence, timeout="20"):
        return subprocess.run([sys.executable, "-I", "-B", str(ROOT / "scripts" / "bench_interrupt.py"), "--evidence",
                               evidence, "--timeout", timeout, "--", sys.executable, "-I", "-B", "-c", child],
                              capture_output=True, text=True, timeout=60)

    def test_signal_after_the_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            child = ("import pathlib, sys, time\n"
                     f"pathlib.Path({tmp!r}, 'attempt-x').mkdir()\n"
                     "try:\n    time.sleep(30)\nexcept KeyboardInterrupt:\n    sys.exit(1)\nsys.exit(0)\n")
            done = self.harness(child, str(Path(tmp) / "attempt-*"))
        self.assertEqual(done.returncode, 1, done.stdout + done.stderr)
        self.assertIn("evidence seen", done.stdout)
        self.assertIn("child exit 1", done.stdout)

    def test_child_that_ends_first_is_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            done = self.harness("import sys; sys.exit(0)", str(Path(tmp) / "never-*"))
        self.assertEqual(done.returncode, 3)
        self.assertIn("ended before", done.stdout)


if __name__ == "__main__":
    unittest.main()
