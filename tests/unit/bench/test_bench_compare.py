"""`compare`, `crosscheck`, feedback (U5), the personal set and `secret-scan` (plan r2 §10, §13; delta r3
Δ9; S20)."""
import contextlib
import copy
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from bench_run_support import PARAGRAPHS, Engine, build_set, make_docx, patches
from markdown_for_llms.bench import checks, cli, compare, crosscheck, exclusions, manifest, run, secretscan
from markdown_for_llms.bench.codes import BenchError, Code
from markdown_for_llms.bench.model import Builder
from markdown_for_llms.bench.settings import BenchSettings

SENTINEL = "zqsentinel9c1d"


def attempt(route, k=1, ratio=1.0, status="complete", failed=(), md="a" * 64, v8="44136fa3"):
    return {"route": route, "variant": {}, "variant_sha8": v8, "repeat": k, "status": status, "error": None,
            "timing": {"total_s": 1.0}, "markdown_sha256": md,
            "scores": None if status != "complete" else {c: {"ratio": ratio} for c in ("content", "structure", "form")},
            "failed": [{"id": i, "class": "content", "type": "paragraph", "outcome": o} for i, o in failed]}


def report(docs, **identity):
    base = {"schema": "mdf-bench-report/1", "package": {"version": "1.0.0", "record_sha256": "r"},
            "toolchain": {"content_sha256": "t", "image_id": "i"}, "weights": {"manifest_sha256": "w"},
            "manifest": {"sha256": "m"}, "bench": {"generator": "5", "matcher": "3", "normalizer": "2"},
            "exclusions_sha256": "e", "documents": docs}
    for key, value in identity.items():
        base[key] = value
    return base


def doc(doc_id, attempts, checks_sha="c" * 64, input_sha="s" * 64):
    return {"id": doc_id, "checks": {"sha256": checks_sha}, "input_sha256": input_sha, "attempts": attempts}


class CompareTests(unittest.TestCase):
    def test_identical_reports_have_no_regression(self):
        a = report([doc("D04", [attempt("marker", k) for k in (1, 2, 3)])])
        result = compare.compare(a, copy.deepcopy(a))
        self.assertEqual((result["regressions"], len(result["pairs"]), result["identity_differences"]), (0, 3, []))
        self.assertTrue(all(p["same_output"] for p in result["pairs"]))

    def test_a_controlled_pair_regresses_and_exits_1(self):
        a = report([doc("D04", [attempt("marker", failed=[("p1", "missing")])]),
                    doc("D23", [attempt("pandoc")], input_sha="1" * 64)])
        b = report([doc("D04", [attempt("marker", ratio=0.99, failed=[("p1", "missing"), ("p7", "partial")], md="b" * 64)]),
                    doc("D23", [attempt("pandoc", status="failed")], input_sha="2" * 64)],
                   toolchain={"content_sha256": "t", "image_id": "other"})
        result = compare.compare(a, b)
        pairs = {p["document"]: p for p in result["pairs"]}
        self.assertEqual(pairs["D04"]["newly_failed"], ["p7"])
        self.assertEqual(pairs["D04"]["fell"], ["content", "structure", "form"])
        self.assertFalse(pairs["D04"]["same_output"])
        self.assertTrue(pairs["D23"]["regression"])                 # complete -> failed
        self.assertEqual(result["identity_differences"], ["toolchain.image_id"])
        self.assertEqual(result["input_differs"], ["D23"])
        self.assertEqual(compare.compare(a, b, tolerance=0.02)["pairs"][0]["regression"], False)
        with tempfile.TemporaryDirectory() as tmp:
            pa, pb = Path(tmp, "a.json"), Path(tmp, "b.json")
            pa.write_text(json.dumps(a))
            pb.write_text(json.dumps(b))
            with contextlib.redirect_stdout(io.StringIO()) as out:
                self.assertEqual(cli.main(["compare", str(pa), str(pb)]), 0)
                self.assertEqual(cli.main(["compare", str(pa), str(pb), "--fail-on-regression"]), 1)
                self.assertEqual(cli.main(["compare", str(pa), str(pa), "--fail-on-regression"]), 0)
            self.assertIn("REGRESSION", out.getvalue())

    def test_variants_are_aligned_apart_and_other_check_sets_are_not_compared(self):
        a = report([doc("D01", [attempt("marker", v8="aaaaaaaa"), attempt("marker", v8="bbbbbbbb", ratio=0.5)])])
        b = report([doc("D01", [attempt("marker", v8="bbbbbbbb", ratio=0.5, failed=[("p2", "missing")])],
                        checks_sha="d" * 64)])
        result = compare.compare(a, b)
        self.assertEqual([(p["variant_sha8"], p["same_checks"], p["newly_failed"]) for p in result["pairs"]],
                         [("bbbbbbbb", False, [])])
        self.assertEqual(result["only_in_a"], [["D01", "marker", "aaaaaaaa", 1]])

    def test_an_invalid_report_is_a_closed_code(self):
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp, "r.json")
            bad.write_text('{"schema": "other"}')
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                self.assertEqual(cli.main(["compare", str(bad), str(bad)]), 1)
            self.assertEqual(err.getvalue().strip(), "error: report_invalid")


class CrosscheckTests(unittest.TestCase):
    def test_a_check_failed_alike_by_three_paths_is_suspect(self):
        same = [("pg3/p2", "missing"), ("p9", "missing")]
        r1 = report([doc("D14", [attempt("marker", k, failed=same + [("p4", "partial")]) for k in (1, 2)]
                         + [attempt("marker", v8="f464ef05", failed=same)])])
        r2 = report([doc("D14", [attempt("datalab", v8="11111111", failed=[("pg3/p2", "missing"), ("p9", "partial")])]),
                     doc("D15", [attempt("pandoc"), attempt("marker")])])
        result = crosscheck.crosscheck([r1, r2])
        d14 = result["documents"][0]
        self.assertEqual((d14["status"], len(d14["paths"])), ("checked", 3))     # the repetitions count once
        # C-CROSSCHECK: three paths, but two of them are variants of Marker: two families, correlated
        self.assertEqual(d14["suspects"], [{"id": "pg3/p2", "class": "content", "type": "paragraph",
                                            "outcome": "missing", "page": 3, "variants": 3, "families": 2,
                                            "correlated": True}])
        self.assertEqual((d14["families"], result["suspects"], result["suspects_by_families"]), (2, 1, {"2": 1}))
        self.assertEqual(result["min_paths"], 3)                                 # the threshold does not change
        self.assertEqual(result["documents"][1]["status"], "too_few_paths")
        r3 = report([doc("D14", [attempt("djvutxt", failed=same)], checks_sha="d" * 64)])
        self.assertEqual(len(crosscheck.crosscheck([r1, r2, r3])["documents"][0]["paths"]), 3)   # other check set
        r4 = report([doc("D14", [attempt("djvutxt", failed=same)])])
        result = crosscheck.crosscheck([r1, r2, r4])
        self.assertEqual([(s["id"], s["variants"], s["families"], s["correlated"]) for s in result["documents"][0]["suspects"]],
                         [("pg3/p2", 4, 3, True)])
        self.assertEqual(result["suspects_by_families"], {"3": 1})

    def test_the_output_names_variants_and_families(self):
        same = [("p9", "missing")]
        r = report([doc("D14", [attempt("marker", failed=same), attempt("marker", v8="f464ef05", failed=same),
                                attempt("pandoc", failed=same)])])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp, "report.json")
            path.write_text(json.dumps(r))
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                self.assertEqual(cli.main(["crosscheck", str(path), "--out", str(Path(tmp, "cc"))]), 0)
            saved = json.loads(Path(tmp, "cc", "crosscheck.json").read_text())
        self.assertIn("D14 paths 3 families 2 checked suspects 1", out.getvalue())
        self.assertIn("1 documents, 1 suspects (by route families: 2 families 1)", out.getvalue())
        self.assertIn("variants of one route are correlated", out.getvalue())
        self.assertEqual(saved["suspects_by_families"], {"2": 1})


class FeedbackTests(unittest.TestCase):
    def twin(self):
        b = Builder("docx")
        for text in PARAGRAPHS:
            b.add("paragraph", text)
        return b.twin

    def entry(self, position, type_="paragraph"):
        return {"document": "D99", "id": f"fb:{position}:{type_}", "class": checks.CLASS_OF[type_], "type": type_,
                "position": position, "origin": "feedback", "generator_version": "5"}

    def test_feedback_checks_enter_the_set_and_its_key(self):
        doc = {"id": "D99", "twin": {"kind": "docx"}}
        files = type("F", (), {"hashes": {"source": "s", "twin": "t"}})()
        plain = checks.checkset(doc, files, self.twin())
        with_fb = checks.checkset(doc, files, self.twin(), [self.entry("p2"), self.entry("p9"), self.entry("p1", "heading")])
        self.assertNotEqual(plain["key"], with_fb["key"])
        self.assertEqual(plain["key"], checks.cache_key(doc, files.hashes))           # no feedback: the key of before
        extra = [c for c in with_fb["checks"] if c["id"].startswith("fb:")]
        self.assertEqual([(c["id"], c["type"], c["origin"]) for c in extra], [("fb:p2:paragraph", "paragraph", "feedback")])
        self.assertEqual(with_fb["feedback"], {"entries": 3, "missing": 2})
        self.assertEqual(with_fb["checks"][:len(plain["checks"])], plain["checks"])

    def test_invalid_feedback_entries_are_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            for bad in (dict(self.entry("p2"), id="fb:p3:paragraph"), dict(self.entry("p2"), origin="self-check"),
                        dict(self.entry("p2"), **{"class": "form"})):
                path = Path(tmp, "x.json")
                path.write_text(json.dumps({"schema": exclusions.SCHEMA, "exclusions": [], "feedback": [bad]}))
                with self.assertRaises(BenchError):
                    exclusions.load(path)

    def test_a_feedback_check_is_red_on_the_output_and_stays_in_later_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            # the output lacks the second paragraph of the twin (the defect noted by the maintainer)
            s = build_set(tmp, text="\n\n".join([PARAGRAPHS[0], PARAGRAPHS[2]]) + "\n")
            excl = Path(tmp, "exclusions.json")
            entry = dict(self.entry("p2"), document="D01")
            excl.write_text(json.dumps({"schema": exclusions.SCHEMA, "exclusions": [], "feedback": [entry]}))
            settings = BenchSettings.load(None, environ={}, cache=s["cache"], runs=s["runs"])
            for name in ("r1", "r2"):
                with contextlib.ExitStack() as stack:
                    for obj, attr, value in patches(s["venv"], Engine()):
                        stack.enter_context(mock.patch.object(obj, attr, value))
                    stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                    self.assertEqual(run.start(settings, docs="D01", manifest=str(s["manifest"]), exclusions=str(excl),
                                               out=str(s["runs"] / name)), 0)
                rep = json.loads((s["runs"] / name / "report.json").read_bytes())
                d01 = rep["documents"][0]
                self.assertEqual(d01["checks"]["feedback"], {"entries": 1, "missing": 0})
                a = d01["attempts"][0]
                failed = {f["id"]: f["outcome"] for f in a["failed"]}
                self.assertIn(failed.get("fb:p2:paragraph"), ("partial", "missing"))
                self.assertEqual(failed["fb:p2:paragraph"], failed["p2"])
                self.assertEqual(a["scores"]["content"]["feedback"], 1)


class PersonalTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)
        self.s = build_set(self.tmp.name)       # for its venv and runs directories
        src = self.base / f"private-{SENTINEL}"
        src.mkdir()
        text = "\n\n".join(PARAGRAPHS[:2]) + f"\n\nUn paragrafo privato {SENTINEL} che resta locale per sempre.\n"
        self.input = src / f"{SENTINEL}.txt"
        self.input.write_text(text)
        self.twin = make_docx(src / f"{SENTINEL}.docx", PARAGRAPHS)
        self.manifest = self.base / "personal.json"
        self.manifest.write_text(json.dumps({"schema": "mdf-bench-manifest/2", "set": "personal", "documents": [
            {"id": "P01", "format": "txt", "input": {"path": str(self.input)}, "twin": None, "title": SENTINEL},
            {"id": "P02", "format": "txt", "input": {"path": str(self.input)},
             "twin": {"kind": "docx", "path": str(self.twin)}}]}))

    def tearDown(self):
        self.tmp.cleanup()

    def start(self, out="p1", docs="P01,P02"):
        settings = BenchSettings.load(None, environ={}, cache=self.base / "cache", runs=self.s["runs"])
        printed = io.StringIO()
        with contextlib.ExitStack() as stack:
            for obj, attr, value in patches(self.s["venv"], Engine()):
                stack.enter_context(mock.patch.object(obj, attr, value))
            stack.enter_context(contextlib.redirect_stdout(printed))
            code = run.start(settings, docs=docs, manifest=str(self.manifest), out=str(self.s["runs"] / out))
        return code, printed.getvalue()

    def test_manifest_rules(self):
        m = manifest.load(self.manifest)
        self.assertTrue(m.personal)
        self.assertEqual(m.document("P01")["profile"], "office")
        data = json.loads(self.manifest.read_text())
        for change in ({"checks": "encoding"}, {"input": {"path": "relative.txt"}}, {"id": "D01"},
                       {"twin": {"kind": "wikisource", "path": "/x"}}, {"levels": ["quick"]}):
            bad = copy.deepcopy(data)
            bad["documents"][0].update(change)
            with self.assertRaises(BenchError) as ctx:
                manifest.parse(json.dumps(bad).encode())
            self.assertEqual(ctx.exception.code, Code.MANIFEST_INVALID, change)

    def test_a_personal_run_reports_numbers_only(self):
        code, printed = self.start()
        self.assertEqual(code, 0)
        run_dir = self.s["runs"] / "p1"
        rep_json, rep_md = (run_dir / "report.json").read_text(), (run_dir / "report.md").read_text()
        for name, body in (("report.json", rep_json), ("report.md", rep_md), ("stdout", printed)):
            self.assertNotIn(SENTINEL, body, name)
            self.assertNotIn(str(self.base), body, name)
        rep = json.loads(rep_json)
        p01, p02 = rep["documents"]
        self.assertNotIn("input_sha256", p01)
        a1, a2 = p01["attempts"][0], p02["attempts"][0]
        self.assertEqual((a1["scores"], a1["quality"]), (None, "unverified"))      # no twin: operational measures only
        self.assertAlmostEqual(a2["scores"]["content"]["ratio"], 2 / 3)      # the third paragraph of the twin is missing
        self.assertIsInstance(a2["failed"], int)                               # a count, no check ids
        self.assertGreaterEqual(a2["failed"], 1)
        for a in (a1, a2):
            self.assertNotIn("markdown_sha256", a)
        self.assertNotIn("sha256", p02["checks"])
        state = json.loads((run_dir / "state.json").read_text())
        self.assertEqual(state["documents"]["P01"]["path"], str(self.input))      # paths stay in the state
        copied = run_dir / "inputs" / "P01.txt"
        self.assertEqual(os.stat(copied).st_mode & 0o777, 0o600)
        self.assertEqual(copied.read_bytes(), self.input.read_bytes())
        self.assertFalse(any((self.base / "cache").rglob("*.json")) if (self.base / "cache").exists() else False)
        self.assertTrue((run_dir / "checks" / "P02.json").is_file())

    def test_a_copy_cut_short_is_never_used(self):
        """B-COPIA: the copy is interrupted before the document is registered in the state; the resumption
        never takes the partial file as the input: it copies again, and the partial stays as evidence."""
        import hashlib
        from markdown_for_llms.bench import personal
        full = b"Ordinary notes from a long meeting.\n" * 100000          # 3.6 MB: several chunks
        self.input.write_bytes(full)
        run_dir = self.s["runs"] / "p4"
        real_read, calls = os.read, [0]

        def cut(fd, size):
            calls[0] += 1
            if calls[0] == 2:
                raise KeyboardInterrupt
            return real_read(fd, size)
        with mock.patch.object(personal.os, "read", cut), self.assertRaises(BenchError) as ctx:
            self.start(out="p4", docs="P01")
        self.assertEqual(ctx.exception.code, Code.INTERRUPTED)
        target = run_dir / "inputs" / "P01.txt"
        self.assertFalse(target.exists())                                    # never published
        partial = [p for p in (run_dir / "inputs").iterdir()                  # the run's manifest copy apart (MFL044 (d))
                   if p.name not in ("P01.txt", "manifest.json")]
        self.assertEqual([(p.name.startswith(".partial-P01.txt"), p.stat().st_size) for p in partial],
                         [(True, personal.CHUNK)])
        self.assertNotIn("P01", json.loads((run_dir / "state.json").read_text())["documents"])
        settings_patches = patches(self.s["venv"], Engine())
        with contextlib.ExitStack() as stack:
            for obj, attr, value in settings_patches:
                stack.enter_context(mock.patch.object(obj, attr, value))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            self.assertEqual(run.resume(run_dir), 0)
        self.assertEqual(target.read_bytes(), full)
        self.assertTrue(partial[0].exists())                                 # kept as evidence, not promoted
        state = json.loads((run_dir / "state.json").read_text())
        attempt = next(iter(state["attempts"].values()))
        result = json.loads((run_dir / "attempts" / attempt["id"] / "out" / "engine.json").read_text())
        self.assertEqual((state["documents"]["P01"]["input_sha256"], result["source"]["sha256"]),
                         (hashlib.sha256(full).hexdigest(),) * 2)
        target.write_bytes(full[:-1] + b"?")                                 # the registered copy changed
        with contextlib.ExitStack() as stack:
            for obj, attr, value in patches(self.s["venv"], Engine()):
                stack.enter_context(mock.patch.object(obj, attr, value))
            with self.assertRaises(BenchError) as ctx:
                run.resume(run_dir)
        self.assertEqual(ctx.exception.code, Code.RESUME_IDENTITY_CHANGED)

    def test_the_copy_of_a_twin_is_registered(self):
        code, _ = self.start(out="p5", docs="P02")
        self.assertEqual(code, 0)
        state = json.loads((self.s["runs"] / "p5" / "state.json").read_text())
        import hashlib
        self.assertEqual(state["documents"]["P02"]["twin_sha256"], hashlib.sha256(self.twin.read_bytes()).hexdigest())
        self.assertEqual(sorted(p.name for p in (self.s["runs"] / "p5" / "inputs").iterdir()),
                         ["P02-twin.docx", "P02.txt", "manifest.json"])                  # with the run's manifest copy

    def test_a_symbolic_link_or_a_level_is_refused(self):
        link = self.base / "link.txt"
        link.symlink_to(self.input)
        data = json.loads(self.manifest.read_text())
        data["documents"][0]["input"]["path"] = str(link)
        self.manifest.write_text(json.dumps(data))
        code, printed = self.start(docs="P01")
        self.assertEqual(code, 2)
        self.assertIn("P01 unavailable input_unreadable", printed)
        settings = BenchSettings.load(None, environ={}, cache=self.base / "cache", runs=self.s["runs"])
        with mock.patch.object(run, "venv_root", lambda: self.s["venv"]):
            with self.assertRaises(BenchError) as ctx:
                run.start(settings, level="quick", manifest=str(self.manifest), out=str(self.s["runs"] / "p3"))
        self.assertEqual(ctx.exception.code, Code.INVALID_SETTING)


class SecretScanTests(unittest.TestCase):
    def test_the_key_is_found_and_never_printed(self):
        key = "dl-" + "k3y" * 20
        with tempfile.TemporaryDirectory() as tmp:
            env = Path(tmp, "key.env")
            env.write_text(f"DATALAB_API_KEY={key}\n")
            os.chmod(env, 0o600)
            tree = Path(tmp, "tree")
            (tree / "a").mkdir(parents=True)
            (tree / "a" / "clean.txt").write_text("nothing here")
            big = (b"x" * ((1 << 20) - 5)) + key.encode() + b"y"           # across two chunks of the scanner
            (tree / "a" / "leak.bin").write_bytes(big + key.encode())
            (tree / "link").symlink_to(env)                                  # symbolic links are not followed
            out, err = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err), \
                    mock.patch.dict(os.environ, {"DATALAB_API_KEY": "from-the-environment"}):
                code = cli.main(["secret-scan", "--env-file", str(env), str(tree)])
            self.assertEqual(code, 1)
            self.assertNotIn(key, out.getvalue() + err.getvalue())
            self.assertIn("found 2 in", out.getvalue())
            self.assertEqual(err.getvalue().strip(), "error: secret_found")
            (tree / "a" / "leak.bin").unlink()
            with contextlib.redirect_stdout(io.StringIO()) as out:
                self.assertEqual(cli.main(["secret-scan", "--env-file", str(env), str(tree)]), 0)
            self.assertIn("secret-scan: 1 files read, 0 errors, 0 occurrences", out.getvalue())
            os.chmod(env, 0o644)                                             # read only as a 0600 file
            with contextlib.redirect_stderr(io.StringIO()) as err:
                self.assertEqual(cli.main(["secret-scan", "--env-file", str(env), str(tree)]), 1)
            self.assertEqual(err.getvalue().strip(), "error: datalab_key_unsafe")
        self.assertEqual(secretscan.scan([], b"x"), (0, [], []))

    def test_a_scan_that_cannot_read_everything_is_not_clean(self):
        """C-SCAN: an unreadable file, a missing path and a path given as a symbolic link are errors, counted
        apart from the files read; the command is not green then, even with no occurrence."""
        key = "dl-" + "k3y" * 20
        with tempfile.TemporaryDirectory() as tmp:
            env = Path(tmp, "key.env")
            env.write_text(f"DATALAB_API_KEY={key}\n")
            os.chmod(env, 0o600)
            tree = Path(tmp, "tree")
            tree.mkdir()
            (tree / "clean.txt").write_text("nothing here")
            hidden = tree / "hidden.bin"
            hidden.write_bytes(key.encode())
            os.chmod(hidden, 0)
            link = Path(tmp, "link")
            link.symlink_to(tree)
            try:
                for paths, errors in (([tree], 1), ([Path(tmp, "missing")], 1), ([link], 1),
                                      ([tree, Path(tmp, "missing")], 2)):
                    out, err = io.StringIO(), io.StringIO()
                    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                        code = cli.main(["secret-scan", "--env-file", str(env), *map(str, paths)])
                    self.assertEqual((code, err.getvalue().strip()), (1, "error: secret_scan_incomplete"), paths)
                    self.assertIn(f", {errors} errors, 0 occurrences", out.getvalue(), paths)
                    self.assertIn("not read: ", out.getvalue())
                    self.assertNotIn(key, out.getvalue() + err.getvalue())
                read, errors, found = secretscan.scan([tree], key.encode())
                self.assertEqual((read, [Path(e).name for e in errors], found), (1, ["hidden.bin"], []))
            finally:
                os.chmod(hidden, 0o600)


if __name__ == "__main__":
    unittest.main()
