"""`mdf bench run --route marker-server` and `mdf bench compare --align document` (MFL033: plan r2 §9,
r3 §9, r4 §11.10). The rented run uses the real Marker child through the real relay socket and the
fake server (`httpx.MockTransport`; the suite forbids connections in its process), inside a session
whose window is in the ledger of the runs folder."""
import contextlib
from decimal import Decimal
import fcntl
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest import mock

from bench_run_support import Died, build_datalab_set, fake_identity
from markdown_for_llms.bench import compare, manifest as manifest_mod, run, serverspend, spend
from markdown_for_llms.bench.codes import BenchError, Code
from markdown_for_llms.bench.settings import BenchSettings

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "engine"))
from fake_inference import Fake  # noqa: E402

KEY = "rp-bench-test-key-0123456789"
URL = "http://127.0.0.1:9"
RATE = "0.0002090948"


def respond(request):
    return '<div data-bbox="60 60 940 160" data-label="Text"><p>page</p></div>'


class ServerRunCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.s = build_datalab_set(self.base)
        self.servers = self.base / "servers.json"
        self.servers.write_text(json.dumps({"default": "local", "servers": {
            "local": {"kind": "local", "url": URL},
            "gpu": {"kind": "user", "url": "https://gpu.example"},
            "rp": {"kind": "runpod", "url": URL, "rate_usd_per_s": RATE}}}))
        self.env = self.base / ".env"
        self.env.write_text("RUNPOD_API_KEY=%s\n" % KEY)
        os.chmod(self.env, 0o600)
        self.ledger = self.s["runs"] / spend.LEDGER

    def options(self, **extra):
        return {"manifest": str(self.s["manifest"]), "docs": "D05,D06", "routes": "marker-server",
                "server_modes": "balanced", "servers_file": str(self.servers), **extra}


class TestOptions(ServerRunCase):
    def test_keys_of_the_selection(self):
        m = manifest_mod.load(str(self.s["manifest"]))
        docs = m.select(ids=["D05", "D06", "D01"])
        keys = run.plan_keys(docs, ["marker-server"], "auto", 1, server_modes=["balanced", "accurate"])
        self.assertEqual([(k["document"], k["variant"]["server_mode"]) for k in keys],
                         [("D05", "balanced"), ("D05", "accurate"), ("D06", "balanced"), ("D06", "accurate")])
        self.assertNotIn("marker-server", [k["route"] for k in run.plan_keys(docs, None, "auto", 1)])

    def test_refusals_before_anything_runs(self):
        m = manifest_mod.load(str(self.s["manifest"]))
        cases = [
            (dict(routes=["marker-server", "datalab"], modes="balanced"), Code.INVALID_SETTING),
            (dict(routes=["marker"], modes="balanced"), Code.INVALID_SETTING),
            (dict(routes=["marker-server"], modes="fast"), Code.ROUTE_OPTION_UNSUPPORTED),
            (dict(routes=["marker-server"], modes="balanced", server="gpu"), Code.TRANSIT_NOT_CONFIRMED),
            (dict(routes=["marker-server"], modes="balanced", server="rp"), Code.SERVER_LEDGER_REQUIRED),
            (dict(routes=["marker-server"], modes="balanced", server="rp", max_cost_usd="1", spend_cap_usd="4"),
             Code.SERVER_KEY_MISSING),
            (dict(routes=["marker-server"], modes="balanced", max_cost_usd="1"), Code.INVALID_SETTING),
        ]
        for kw, code in cases:
            with self.subTest(kw=kw):
                with self.assertRaises(BenchError) as caught:
                    run.server_options(m, kw.pop("routes"), servers_file=str(self.servers), **kw)
                self.assertEqual(caught.exception.code, code)
        srv, (ep, keys) = run.server_options(m, ["marker-server"], "balanced,accurate", "rp", str(self.servers),
                                             max_cost_usd="1", spend_cap_usd="4", key_file=str(self.env))
        self.assertEqual((srv["modes"], srv["max_cost_usd"], srv["identity_sha256"]),
                         (["balanced", "accurate"], "1", ep.identity_sha256()))
        self.assertNotIn(KEY, json.dumps(srv))

    def test_a_personal_set_never_leaves_the_loopback(self):
        from types import SimpleNamespace
        m = SimpleNamespace(personal=True)
        with self.assertRaises(BenchError) as caught:
            run.server_options(m, ["marker-server"], "balanced", "gpu", str(self.servers), confirm_transit=True)
        self.assertEqual(caught.exception.code, Code.SERVER_REFUSED_PERSONAL)
        self.assertIsNotNone(run.server_options(m, ["marker-server"], "balanced", "local", str(self.servers))[0])


class TestRentedRun(ServerRunCase):
    def arm(self):
        self.s["runs"].mkdir(parents=True, exist_ok=True, mode=0o700)
        with spend.Ledger(self.s["runs"]).locked() as locked:
            row = serverspend.arm(locked, self.ledger, "runpod", "rp", URL, 200, time.time() + 7200, 400, RATE, time.time())
        fd = os.open(serverspend.lock_path(self.ledger, row["arm_id"]), os.O_RDWR | os.O_CREAT | os.O_CLOEXEC, 0o600)
        fcntl.flock(fd, fcntl.LOCK_EX)
        self.addCleanup(os.close, fd)
        return row["arm_id"]

    def start(self, out, environ):
        from markdown_for_llms.bench.adapters import engine as adapter
        from markdown_for_llms.engine import core
        model_root = os.environ.get("FULL_CHECK_MODEL_ROOT")
        if not model_root:
            raise AssertionError("FULL_CHECK_MODEL_ROOT is not set: run the suite from scripts/full-check.sh")
        fake = Fake(respond=respond, mark="forwarded", key=KEY)
        self.addCleanup(fake.close)
        self.fake = fake

        def convert(source, out_dir, request, limits, work, **kw):
            kw["model_root"] = model_root
            return core.convert(source, out_dir, request, limits, work, server_transport=fake.factory(), **kw)
        settings = BenchSettings.load(None, environ={}, cache=self.s["cache"], runs=self.s["runs"])
        with mock.patch.object(adapter, "convert", convert), mock.patch.object(adapter, "identity", fake_identity), \
                mock.patch.object(run, "venv_root", lambda: Path(self.s["venv"])), \
                mock.patch.dict(os.environ, environ), contextlib.redirect_stdout(io.StringIO()) as printed:
            if serverspend.SESSION_ENV not in environ:
                os.environ.pop(serverspend.SESSION_ENV, None)
            code = run.start(settings, **self.options(server="rp", max_cost_usd="2", spend_cap_usd="4",
                                                      key_file=str(self.env), out=str(out)))
        return code, printed.getvalue()

    def test_inside_a_session_then_without(self):
        arm_id = self.arm()
        code, printed = self.start(self.s["runs"] / "r1", {serverspend.SESSION_ENV: arm_id})
        self.assertEqual(code, 0, printed)
        state = json.loads((self.s["runs"] / "r1" / "state.json").read_text())
        rows = [e for e in spend.read(self.ledger) if e.get("run_id") == state["run_id"]]
        self.assertEqual(sorted(r["kind"] for r in rows), ["complete", "complete", "reserve", "reserve"])
        self.assertTrue(all(r["arm_id"] == arm_id for r in rows if r["kind"] == "reserve"))
        report = json.loads((self.s["runs"] / "r1" / "report.json").read_text())
        attempts = [a for d in report["documents"] for a in d["attempts"]]
        self.assertEqual([a["status"] for a in attempts], ["complete", "complete"])
        for a in attempts:
            sv = a["server"]
            self.assertEqual((sv["provider"], sv["mode"], sv["clean"]), ("runpod", "balanced", True))
            self.assertLessEqual(Decimal(sv["bound_cents"]), Decimal(sv["reserve_cents"]))
        self.assertEqual(report["server"]["name"], "rp")
        self.assertIn("## Inference server", (self.s["runs"] / "r1" / "report.md").read_text())
        everything = "".join(p.read_text(errors="replace") for p in (self.s["runs"]).rglob("*") if p.is_file())
        self.assertNotIn(KEY, everything)
        # without the session the run stops before any wake
        with self.assertRaises(BenchError) as caught:
            self.start(self.s["runs"] / "r2", {})
        self.assertEqual(caught.exception.code, Code.SERVER_NOT_ARMED)
        state = json.loads((self.s["runs"] / "r2" / "state.json").read_text())
        self.assertEqual(state["stop"], "server_not_armed")
        self.assertEqual(self.fake.seen, [])                 # the fake of this run saw nothing


class TestReportOfAServerRun(ServerRunCase):
    """G-B1 of the final review of MFL033: `mdf bench report` builds the keys of the run with its modes of
    `marker-server`, in place and with `--report-dir`, for a run of that route alone (with the former text
    of its limits, `pages.FORMER_LIMITS`) and for a run with `marker` and `marker-server`, losing no
    attempt. The local endpoint, the real Marker child and the fake server."""

    @contextlib.contextmanager
    def patched(self, former=False):
        from markdown_for_llms.bench.adapters import engine as adapter
        from markdown_for_llms.engine import core, pages
        model_root = os.environ.get("FULL_CHECK_MODEL_ROOT")
        if not model_root:
            raise AssertionError("FULL_CHECK_MODEL_ROOT is not set: run the suite from scripts/full-check.sh")
        fake = Fake(respond=respond)
        self.addCleanup(fake.close)

        def convert(source, out_dir, request, limits, work, **kw):
            kw["model_root"] = model_root
            return core.convert(source, out_dir, request, limits, work, server_transport=fake.factory(), **kw)
        limits = [pages.FORMER_LIMITS[0]] + pages.SERVER_LIMITS[1:] if former else pages.SERVER_LIMITS

        def identity(root):
            """The fake identity with the manifest of the real weights, which the results of the real child
            name: the report verifies their components against it."""
            from markdown_for_llms.engine import weights
            ident = fake_identity(root)
            ident["weights"]["manifest_sha256"] = weights.manifest_sha256()
            return ident
        with mock.patch.object(adapter, "convert", convert), mock.patch.object(adapter, "identity", identity), \
                mock.patch.object(run, "venv_root", lambda: Path(self.s["venv"])), \
                mock.patch.object(pages, "SERVER_LIMITS", limits), contextlib.redirect_stdout(io.StringIO()) as printed:
            yield printed

    def start(self, out, routes, former=False):
        settings = BenchSettings.load(None, environ={}, cache=self.s["cache"], runs=self.s["runs"])
        with self.patched(former) as printed:
            code = run.start(settings, **self.options(docs="D05", routes=routes, out=str(out)))
        self.assertEqual(code, 0, printed.getvalue())

    def rebuild(self, out, report_dir=None):
        with self.patched() as printed:
            code = run.rebuild_report(out, report_dir=report_dir)
        self.assertEqual(code, 0)
        return printed.getvalue()

    def attempts(self, report_path):
        report = json.loads(Path(report_path).read_text())
        return sorted((d["id"], a["route"], a["variant"].get("server_mode"), a["status"])
                      for d in report["documents"] for a in d["attempts"]), report

    def check(self, out, expected):
        first, _ = self.attempts(out / "report.json")
        self.assertEqual(first, expected)
        printed = self.rebuild(out)
        self.assertIn("%d results verified" % len(expected), printed)
        again, report = self.attempts(out / "report.json")
        self.assertEqual(again, expected)                                  # in place, no attempt lost
        target = self.base / ("rep-" + out.name)
        self.rebuild(out, report_dir=target)
        apart, report = self.attempts(target / "report.json")
        self.assertEqual(apart, expected)                                  # read-only, the same attempts
        return report

    def test_a_marker_server_run_with_the_former_limits(self):
        from markdown_for_llms.engine import pages
        out = self.s["runs"] / "server-only"
        self.start(out, "marker-server", former=True)
        engine_json = next(out.glob("attempts/*/out/engine.json"))
        self.assertIn(pages.FORMER_LIMITS[0], json.loads(engine_json.read_text())["limits"])
        report = self.check(out, [("D05", "marker-server", "balanced", "complete")])
        limits = [x for d in report["documents"] for a in d["attempts"] for x in a.get("limits") or []]
        self.assertIn(pages.FORMER_LIMITS[0], limits)
        self.assertNotIn("unrecognized", limits)

    def test_a_run_of_marker_and_marker_server(self):
        out = self.s["runs"] / "mixed"
        self.start(out, "marker,marker-server")
        self.check(out, [("D05", "marker", None, "complete"), ("D05", "marker-server", "balanced", "complete")])


class TestResumeOnARentedServer(ServerRunCase):
    """G-B2 of the final review of MFL033: on resumption, an attempt stopped before the wake of the server
    (`SERVER_STOPS`) is converted again, never adopted as failed; an attempt completed before the death of
    the bench is adopted, because the state holds the request that engine.json records (budget and
    session); the line of an adoption names the mode."""

    def arm(self, a_cents):
        self.s["runs"].mkdir(parents=True, exist_ok=True, mode=0o700)
        with spend.Ledger(self.s["runs"]).locked() as locked:
            row = serverspend.arm(locked, self.ledger, "runpod", "rp", URL, a_cents, time.time() + 7200, 400, RATE,
                                  time.time())
        fd = os.open(serverspend.lock_path(self.ledger, row["arm_id"]), os.O_RDWR | os.O_CREAT | os.O_CLOEXEC, 0o600)
        fcntl.flock(fd, fcntl.LOCK_EX)
        return row["arm_id"], fd

    def close_window(self, arm_id, fd):
        os.close(fd)
        with spend.Ledger(self.s["runs"]).locked() as locked:
            serverspend.disarm(locked, "runpod", arm_id, time.time(), False)

    @contextlib.contextmanager
    def patched(self, environ, die_after=None):
        from markdown_for_llms.bench.adapters import engine as adapter
        from markdown_for_llms.engine import core, weights
        model_root = os.environ.get("FULL_CHECK_MODEL_ROOT")
        if not model_root:
            raise AssertionError("FULL_CHECK_MODEL_ROOT is not set: run the suite from scripts/full-check.sh")
        fake = Fake(respond=respond, mark="forwarded", key=KEY)
        self.addCleanup(fake.close)
        self.fake = fake

        def convert(source, out_dir, request, limits, work, **kw):
            kw["model_root"] = model_root
            result = core.convert(source, out_dir, request, limits, work, server_transport=fake.factory(), **kw)
            if die_after is not None and request["attempt"].endswith(die_after):
                raise Died()                       # engine.json published, the bench dies before its save
            return result

        def identity(root):
            ident = fake_identity(root)
            ident["weights"]["manifest_sha256"] = weights.manifest_sha256()
            return ident
        with mock.patch.object(adapter, "convert", convert), mock.patch.object(adapter, "identity", identity), \
                mock.patch.object(run, "venv_root", lambda: Path(self.s["venv"])), \
                mock.patch.dict(os.environ, environ), contextlib.redirect_stdout(io.StringIO()) as printed:
            yield printed

    def start(self, out, environ, docs="D05,D06", die_after=None):
        settings = BenchSettings.load(None, environ={}, cache=self.s["cache"], runs=self.s["runs"])
        with self.patched(environ, die_after) as printed:
            try:
                return run.start(settings, **self.options(docs=docs, server="rp", max_cost_usd="2", spend_cap_usd="4",
                                                          key_file=str(self.env), out=str(out))), printed.getvalue()
            except BenchError as exc:
                return exc.code, printed.getvalue()

    def resume(self, out, environ):
        with self.patched(environ) as printed:
            code = run.resume(out, key_file=str(self.env))
        return code, printed.getvalue()

    def state(self, out):
        return json.loads((out / "state.json").read_text())

    def test_an_attempt_stopped_before_the_wake_is_converted_again(self):
        """The case of Carbon: a window of 28 cents; D05 completes, D06 is stopped before the wake with
        `server_cap`; a resumption in a window of 200 cents converts D06 in a new generation."""
        out = self.s["runs"] / "r-stop"
        arm_id, fd = self.arm(28)
        code, printed = self.start(out, {serverspend.SESSION_ENV: arm_id})
        self.assertEqual(code, Code.SERVER_CAP, printed)
        st = self.state(out)
        d06, = [a for a in st["attempts"].values() if a["document"] == "D06"]
        self.assertEqual((d06["status"], d06["result"]["code"]), ("stopped", "server_cap"))
        self.close_window(arm_id, fd)
        arm2, fd2 = self.arm(200)
        self.addCleanup(os.close, fd2)
        code, printed = self.resume(out, {serverspend.SESSION_ENV: arm2})
        self.assertEqual(code, 0, printed)
        self.assertNotIn("adopted failed server_cap", printed)
        st = self.state(out)
        gens = {a["id"].rsplit("-", 1)[1]: a for a in st["attempts"].values() if a["document"] == "D06"}
        self.assertEqual(sorted(gens), ["g1", "g2"])
        self.assertEqual((gens["g1"]["status"], gens["g2"]["status"], gens["g2"]["result"]["status"]),
                         ("abandoned", "done", "complete"))
        report = json.loads((out / "report.json").read_text())
        self.assertEqual(sorted((d["id"], a["status"]) for d in report["documents"] for a in d["attempts"]),
                         [("D05", "complete"), ("D06", "complete")])

    def test_an_attempt_completed_before_the_death_of_the_bench_is_adopted(self):
        out = self.s["runs"] / "r-died"
        arm_id, fd = self.arm(200)
        self.addCleanup(os.close, fd)
        environ = {serverspend.SESSION_ENV: arm_id}
        with self.assertRaises(Died):
            self.start(out, environ, docs="D05", die_after="g1")
        st = self.state(out)
        a, = st["attempts"].values()
        self.assertEqual(a["status"], "started")
        r = json.loads((out / "attempts" / a["id"] / "out" / "engine.json").read_text())
        self.assertEqual(r["status"], "complete")
        self.assertEqual(r["options"]["requested"], run.expected_requested(a["request"]))   # budget and arm_id
        self.assertEqual(r["options"]["requested"]["server"]["arm_id"], arm_id)
        code, printed = self.resume(out, environ)
        self.assertEqual(code, 0, printed)
        self.assertIn("D05 marker-server:balanced %s k1 g1 adopted complete" % a["v8"], printed)
        st = self.state(out)
        self.assertEqual([(x["id"], x["status"], x["adopted"]) for x in st["attempts"].values()],
                         [(a["id"], "done", True)])
        rows = [e["kind"] for e in spend.read(self.ledger) if e.get("run_id") == st["run_id"]]
        self.assertEqual(sorted(rows), ["complete", "reserve"])            # converted and paid once

    def test_the_key_of_each_mode(self):
        m = manifest_mod.load(str(self.s["manifest"]))
        keys = run.plan_keys(m.select(ids=["D05"]), ["marker-server"], "auto", 1, server_modes=["balanced", "accurate"])
        names = [run.key_name(k["document"], k["route"], k["v8"], k["k"]) for k in keys]
        self.assertEqual(len(set(names)), 2)
        self.assertEqual([k["variant"]["server_mode"] for k in keys], ["balanced", "accurate"])
        self.assertNotEqual(keys[0]["v8"], keys[1]["v8"])


class TestCompareByDocument(unittest.TestCase):
    def report(self, route, mode, failed, sha="i", checks="c", status="complete"):
        return {"schema": "mdf-bench-report/1", "documents": [{
            "id": "D04", "input_sha256": sha, "checks": {"sha256": checks},
            "attempts": [{"route": route, "variant": {"datalab_mode" if route == "datalab" else "server_mode": mode,
                                                      **({} if route == "datalab" else {"ocr": "auto"})},
                          "variant_sha8": "x", "repeat": 1, "status": status, "error": None,
                          "scores": None, "failed": [{"id": f} for f in failed]}]}]}

    def test_r10_cells_by_id(self):
        a = self.report("datalab", "balanced", ["t1/r17/c3"])
        b = self.report("marker-server", "balanced", ["t1/r3/c4", "t1/r17/c3"])
        result = compare.compare_documents(a, b, "datalab:balanced", "marker-server:balanced")
        pair = result["pairs"][0]
        self.assertTrue(pair["comparable"])
        self.assertEqual((pair["passed_in_a_failed_in_b"], pair["passed_in_b_failed_in_a"]), (["t1/r3/c4"], []))
        for other, reason in ((self.report("marker-server", "balanced", [], sha="j"), "input differs"),
                              (self.report("marker-server", "balanced", [], checks="d"), "checks differ"),
                              (self.report("marker-server", "balanced", [], status="failed"), "not complete")):
            pair = compare.compare_documents(a, other, "datalab:balanced", "marker-server:balanced")["pairs"][0]
            self.assertEqual((pair["comparable"], pair["reason"]), (False, reason))
        self.assertEqual(compare.compare_documents(a, b, "datalab:fast", "marker-server:balanced")["pairs"], [])
        with self.assertRaises(BenchError):
            compare.parse_select("nonsense:x")


if __name__ == "__main__":
    unittest.main()


class TestSecretScanOfEveryKey(unittest.TestCase):
    def test_the_keys_of_the_servers_are_searched_too(self):
        from markdown_for_llms.bench import cli as bench_cli
        with tempfile.TemporaryDirectory() as tmp:
            env = Path(tmp, ".env")
            env.write_text("MDF_PORT=23465\nRUNPOD_API_KEY=rp-0123456789\nMODAL_PROXY_TOKEN_SECRET=ws-0123456789\n")
            os.chmod(env, 0o600)
            tree = Path(tmp, "tree")
            tree.mkdir()
            (tree / "a.json").write_text('{"x": "ws-0123456789"}')
            (tree / "b.txt").write_text("23465 is a port, not a key")
            out, err = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                self.assertEqual(bench_cli.main(["secret-scan", "--env-file", str(env), str(tree)]), 1)
            self.assertIn("found 1 in", out.getvalue())
            self.assertNotIn("ws-0123456789", out.getvalue() + err.getvalue())
            (tree / "a.json").unlink()
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(bench_cli.main(["secret-scan", "--env-file", str(env), str(tree)]), 0)
