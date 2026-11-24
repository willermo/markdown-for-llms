"""The GPU session through the API and the CLI (MFL023 B5; plan r2 §4.5, r3 §4.7-§4.8): `GET /api/v1/gpu/session` with
the state, the ledger of the provider from the worker and, for a window not certified, orphan or stopped, the texts of
the block and the commands; `POST /api/v1/gpu/arm` (only a preset that arms ahead, the transit confirmed, the checks of
the instance) and `POST /api/v1/gpu/disarm` (the jobs of the preset cancelled, in progress and queued, in one
transaction with the request); `GET /api/v1/gpu/estimate` with the minimum reserve, never a charge; CSRF on every change;
the folder of the ledger never in `Settings.public()`; the same services from `mdf gpu`. The API reads states and the
heartbeat of the worker, never a provider nor the ledger."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import time
import unittest

from fastapi.testclient import TestClient

from markdown_for_llms.api import create_app
from markdown_for_llms.cli import main
from markdown_for_llms.settings import Settings

PDF = b'%PDF-1.4\nsynthetic contract input; not a valid conversion fixture\n'
ENABLED = {"gpu_presets": "modal-base,runpod-serverless-volume-dc", "gpu_cap_usd_modal": "3", "gpu_cap_usd_runpod": "2",
           "gpu_session_cap_usd": "0.40"}
GPU = "route=marker-server&server_mode=balanced&gpu_preset=runpod-serverless-volume-dc&transit_confirmed=true"


class Base(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.ledger = root / "ledger"
        self.settings = Settings(data_root=str(root / "data"), model_root=str(root / "models"),
                                 gpu_ledger_dir=str(self.ledger), servers=str(root / "servers.json"), **ENABLED)
        self.client = TestClient(create_app(self.settings), base_url=f"http://127.0.0.1:{self.settings.port}")
        self.csrf = self.client.get("/api/v1/session").json()["csrf"]
        self.headers = {"origin": f"http://127.0.0.1:{self.settings.port}", "x-mdf-csrf": self.csrf}

    def db(self, statement, *args):
        with sqlite3.connect(Path(self.settings.data_root) / "archive.sqlite3") as conn:
            rows = conn.execute(statement, args).fetchall()
        conn.close()
        return rows

    def heartbeat(self, gpu):
        readiness = {"ready": True, "message": "Local worker ready", "converters": {}, "gpu": gpu}
        self.db("INSERT INTO worker VALUES (1,'t',?,?) ON CONFLICT(id) DO UPDATE SET heartbeat=excluded.heartbeat, "
                "readiness=excluded.readiness", time.time(), json.dumps(readiness))

    def post(self, path, body):
        return self.client.post(path, content=json.dumps(body), headers=self.headers)

    def session_row(self, state, reason, arm_id="20261124T100000Z-aaaa0001", **extra):
        values = dict({"t_arm": 1.8e9, "until": 1.8e9 + 1703, "t_zero": 1.8e9 + 11000, "limit_at": 1.8e9 + 1883,
                       "drift_s": 10800.0}, **extra)
        self.db("INSERT INTO gpu_sessions (arm_id, owner, gpu_preset, provider, resource, url, ledger_path, env_file, "
                "key_env, origin, a_cents, w_s, t_arm, until, state, reason, t_zero, limit_at, drift_s, created, updated) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                arm_id, "local", "runpod-serverless-volume-dc", "runpod", "mdf-surya", "https://ep1.api.runpod.ai",
                str(self.ledger / "spend.jsonl"), str(self.ledger.parent / ".env"), '["RUNPOD_API_KEY"]', "job", "40.00", 1703,
                values["t_arm"], values["until"], state, reason, values["t_zero"], values["limit_at"], values["drift_s"],
                time.time(), time.time())
        return arm_id


class SessionTests(Base):
    LEDGERS = {"runpod": {"total_cents": "200.00", "orphans": 1, "cap_cents": "200.00"},
               "modal": {"total_cents": "0.00", "orphans": 0, "cap_cents": "300.00"}}

    def test_no_session_yet(self):
        body = self.client.get("/api/v1/gpu/session").json()
        self.assertEqual((body["session"], body["requests"]), (None, []))

    def test_a_session_not_certified(self):
        self.heartbeat({"presets": {}, "ledgers": self.LEDGERS})
        arm_id = self.session_row("uncertain", "unobserved_interval")
        s = self.client.get("/api/v1/gpu/session").json()["session"]
        self.assertEqual((s["state"], s["reason"], s["arm_id"]), ("uncertain", "unobserved_interval", arm_id))
        self.assertEqual((s["ledger_total_cents"], s["cap_cents"]), ("200.00", "200.00"))
        self.assertEqual(s["texts"]["title"], "GPU session not certified")
        self.assertIn("suspended, or its clock jumped", s["texts"]["reason"])
        self.assertIn("about 180 min not observed", s["texts"]["reason"])
        self.assertIn("counts RunPod at its cap (2.00 USD)", s["texts"]["block"])
        self.assertEqual(s["commands"], ["mdf bench spend reconcile --runs-dir %s --provider runpod --orphan --arm %s "
                                         "--manual-cents N" % (self.ledger, arm_id)])
        self.assertNotIn("env_file", s)
        self.assertNotIn(str(self.ledger), json.dumps(self.settings.public()))

    def test_over_window_orphan_and_stopped(self):
        self.heartbeat({"presets": {}, "ledgers": self.LEDGERS})
        for state, reason, title, words, commands in (
                ("uncertain", "over_window", "GPU session not certified", "after the limit of this session's window", 1),
                ("orphan", "worker_died", "GPU window orphan", "The worker stopped while", 1),
                ("stopped", "zero_missing", "GPU window orphan", "could not see RunPod at zero", 2)):
            with self.subTest(state=state):
                self.db("DELETE FROM gpu_sessions")
                self.session_row(state, reason)
                s = self.client.get("/api/v1/gpu/session").json()["session"]
                self.assertEqual(s["texts"]["title"], title)
                self.assertIn(words, s["texts"]["reason"])
                self.assertEqual(len(s["commands"]), commands)
                if commands == 2:
                    self.assertIn("session.py disarm --provider runpod --name mdf-surya", s["commands"][0])

    def test_a_closed_session_has_no_commands(self):
        self.session_row("closed", "idle")
        s = self.client.get("/api/v1/gpu/session").json()["session"]
        self.assertEqual(s["state"], "closed")
        self.assertNotIn("commands", s)


class RequestTests(Base):
    def test_arm_ahead(self):
        refused = self.post("/api/v1/gpu/arm", {"gpu_preset": "runpod-serverless-volume-dc", "transit_confirmed": False})
        self.assertEqual((refused.status_code, refused.json()["code"]), (409, "transit_not_confirmed"))
        for preset, code in (("modal-base", "route_option_unsupported"), ("nope", "gpu_preset_unknown"),
                             ("runpod-serverless-volume-eu-se-1", "gpu_preset_superseded")):
            with self.subTest(preset=preset):
                answer = self.post("/api/v1/gpu/arm", {"gpu_preset": preset, "transit_confirmed": True})
                self.assertEqual((answer.status_code, answer.json()["code"]), (409, code))
        made = self.post("/api/v1/gpu/arm", {"gpu_preset": "runpod-serverless-volume-dc", "transit_confirmed": True})
        self.assertEqual(made.status_code, 202, made.text)
        self.assertEqual(self.db("SELECT gpu_preset, kind, state FROM gpu_requests"),
                         [("runpod-serverless-volume-dc", "arm", "pending")])
        no_csrf = self.client.post("/api/v1/gpu/arm", content=json.dumps({"gpu_preset": "runpod-serverless-volume-dc",
                                                                         "transit_confirmed": True}),
                                   headers={"origin": self.headers["origin"]})
        self.assertEqual(no_csrf.status_code, 403)
        bad = self.post("/api/v1/gpu/arm", {"gpu_preset": "runpod-serverless-volume-dc"})
        self.assertEqual((bad.status_code, bad.json()["code"]), (409, "invalid_setting"))

    def test_disarm_cancels_the_jobs_of_the_preset(self):
        headers = dict(self.headers, **{"x-filename": "a.pdf"})
        gpu = [self.client.post("/api/v1/runs?" + GPU, content=PDF, headers=headers).json()["id"] for _ in range(3)]
        local = self.client.post("/api/v1/runs?route=marker", content=PDF, headers=headers).json()["id"]
        self.db("UPDATE runs SET state='running' WHERE id=?", gpu[0])
        answer = self.post("/api/v1/gpu/disarm", {"gpu_preset": "runpod-serverless-volume-dc"})
        self.assertEqual(answer.status_code, 202, answer.text)
        self.assertEqual(answer.json()["cancelled"], {"running": 1, "queued": 2})
        states = dict(self.db("SELECT id, state FROM runs"))
        self.assertEqual((states[gpu[0]], states[gpu[1]], states[gpu[2]], states[local]),
                         ("running", "cancelled", "cancelled", "queued"))
        self.assertEqual(self.db("SELECT cancel FROM runs WHERE id=?", gpu[0]), [(1,)])
        self.assertEqual(self.db("SELECT error FROM runs WHERE id=?", gpu[1]), [("Session disarmed by the user",)])
        self.assertEqual(self.db("SELECT kind, state FROM gpu_requests"), [("disarm", "pending")])

    def test_the_estimate(self):
        self.heartbeat({"presets": {"runpod-serverless-volume-dc": {"available": True, "code": None,
                                                                   "minimum_reserve_cents": "28.76", "a_cents": "40.00",
                                                                   "w_s": 1703}}, "ledgers": {}})
        e = self.client.get("/api/v1/gpu/estimate?preset=runpod-serverless-volume-dc").json()
        self.assertEqual((e["provider"], e["region"], e["gpu"], e["available"]), ("runpod", "CA-MTL-1", ["AMPERE_24"], True))
        self.assertEqual((e["minimum_reserve_cents"], e["a_cents"], e["w_s"]), ("28.76", "40.00", 1703))
        self.assertEqual((e["provisional"], e["quality"], e["arm_ahead"]), (True, "unverified", True))
        self.assertEqual((e["start_cents"], e["page_cents"]), ("10.04", "0.63"))         # 480 s and 30 s at its rate
        self.assertIn("leave this computer", e["notice"])
        self.assertIn("not a bill", e["bound_text"])
        m = self.client.get("/api/v1/gpu/estimate?preset=modal-base").json()
        self.assertEqual((m["region"], m["arm_ahead"]), ("chosen by Modal", False))

    def test_a_gpu_preset_that_is_not_a_string(self):
        """C-DB-NB4 (MFL023 B10): a list or an object as `gpu_preset` is `invalid_setting` (409), before any other check,
        never an error of the server; nothing is written."""
        for path, extra in (("/api/v1/gpu/arm", {"transit_confirmed": True}), ("/api/v1/gpu/arm", {"transit_confirmed": False}),
                            ("/api/v1/gpu/disarm", {})):
            for value in ([], {}, ["runpod-serverless-volume-dc"], {"name": "runpod-serverless-volume-dc"}):
                with self.subTest(path=path, value=value, extra=extra):
                    answer = self.post(path, dict(extra, gpu_preset=value))
                    self.assertEqual((answer.status_code, answer.json()["code"]), (409, "invalid_setting"))
        self.assertEqual(self.db("SELECT * FROM gpu_requests"), [])

    def test_the_readiness_carries_the_gpu_of_the_worker(self):
        self.heartbeat({"presets": {"modal-base": {"available": False, "code": "session_orphaned"}}, "ledgers": {}})
        ready = self.client.get("/api/v1/readiness").json()
        self.assertEqual(ready["gpu"]["presets"]["modal-base"]["code"], "session_orphaned")


class WebUiTests(Base):
    """B6 (D8, in English): the conversion presets, the transit and the estimate of a GPU preset before anything is sent,
    the state of the session with a disarm confirmed in the page, the block of a window not certified, the facts of a
    GPU job as an upper bound. The browser check of the run (`b-evidence/browser_b6.py`) loads them in headless Chrome."""

    def test_the_page_and_its_script(self):
        page = self.client.get("/").text
        for ident in ("preset", "gpu-box", "gpu-estimate", "gpu-transit", "arm-ahead", "gpu-session", "gpu-state"):
            self.assertIn('id="%s"' % ident, page)
        script = self.client.get("/static/app.js").text
        for text in ("/gpu/disarm", "/gpu/arm", "/gpu/estimate?preset=", "/gpu/session", "Confirm disarm",
                     "Disarm: stops the GPU session now; the conversions of this preset in progress and in the queue are "
                     "cancelled.", "an upper bound, not a bill", "minimum reserve of a session"):
            self.assertIn(text, script)

    def test_the_presets_follow_the_worker(self):
        self.heartbeat({"presets": {"runpod-serverless-volume-dc": {"available": False, "code": "session_orphaned"},
                                    "modal-base": {"available": True, "code": None}}, "ledgers": {}})
        body = self.client.get("/api/v1/presets").json()
        presets = {p["name"]: (p["available"], p["code"]) for p in body["presets"]}
        self.assertEqual((presets["batch"], presets["quality-now"]), ((False, "session_orphaned"), (True, None)))
        gpus = {g["name"]: (g["available"], g["code"]) for g in body["gpu_presets"]}
        self.assertEqual(gpus["runpod-serverless-volume-dc"], (False, "session_orphaned"))
        headers = dict(self.headers, **{"x-filename": "a.pdf"})
        made = self.client.post("/api/v1/runs?" + GPU, content=PDF, headers=headers)
        self.assertEqual(made.status_code, 202)        # the validation of a job stays that of the instance (parity)

    def test_the_presets_follow_a_change_of_the_readiness(self):
        """SOL-B-N2 and C-DB-NB3 (MFL023 B10): when the GPU part of the readiness changes, the page reads the presets
        again (the selector keeps a choice still valid) and forgets its estimates, without a reload. The browser check of
        the run (`b-evidence/browser_b10.py`) changes the heartbeat under an open page."""
        script = self.client.get("/static/app.js").text
        start = script.index("function showReadiness(")
        body = script[start:script.index("\n}\n", start)]
        for text in ("ready.gpu", "readyGpu", "estimates = {}", "api('/presets')", "setupPresets", "updateForm"):
            self.assertIn(text, body)
        keep = script[script.index("function setupPresets("):]
        self.assertIn("!o.disabled", keep[:keep.index("\n}\n")])          # the choice kept only while still valid

    def test_a_run_shows_its_gpu_facts(self):
        headers = dict(self.headers, **{"x-filename": "a.pdf"})
        run = self.client.post("/api/v1/runs?" + GPU, content=PDF, headers=headers).json()
        self.assertIsNone(run["gpu"])
        self.db("UPDATE runs SET gpu=? WHERE id=?", json.dumps({"preset": "runpod-serverless-volume-dc", "provider": "runpod",
                                                               "arm_id": "a", "a_cents": "40.00", "until": 1.0,
                                                               "reserve_cents": "33.00", "bound_cents": "9.10",
                                                               "completed": True, "error": None, "extra": "dropped"}),
                run["id"])
        gpu = self.client.get("/api/v1/runs/" + run["id"]).json()["gpu"]
        self.assertEqual((gpu["reserve_cents"], gpu["bound_cents"], gpu["completed"]), ("33.00", "9.10", True))
        self.assertNotIn("extra", gpu)


class PresetParityTests(Base):
    """C-DB-NB2.2-3 (MFL023 B10): a job sent with `?preset=NAME` is validated as one sent with the same options, by the
    availability of the instance alone (`with_worker` is for the list and the estimate, which show what the worker
    knows); the texts of the refusals of the GPU presets."""

    UNAVAILABLE = {"presets": {"runpod-serverless-volume-dc": {"available": False, "code": "server_cap"},
                               "modal-base": {"available": False, "code": "server_key_missing"}}, "ledgers": {}}

    def test_a_preset_and_its_options_with_the_heartbeat_of_another_worker(self):
        self.heartbeat(self.UNAVAILABLE)            # e.g. a worker with other caps that just left
        listed = {p["name"]: (p["available"], p["code"]) for p in self.client.get("/api/v1/presets").json()["presets"]}
        self.assertEqual((listed["batch"], listed["quality-now"]), ((False, "server_cap"), (False, "server_key_missing")))
        headers = dict(self.headers, **{"x-filename": "a.pdf"})
        for query in ("preset=batch&transit_confirmed=true", GPU, "preset=quality-now&transit_confirmed=true",
                      "route=marker-server&server_mode=balanced&gpu_preset=modal-base&transit_confirmed=true"):
            with self.subTest(query=query):
                made = self.client.post("/api/v1/runs?" + query, content=PDF, headers=headers)
                self.assertEqual(made.status_code, 202, made.text)
        for query, code in (("preset=batch", "transit_not_confirmed"), ("preset=nope", "preset_unknown")):
            with self.subTest(query=query):
                refused = self.client.post("/api/v1/runs?" + query, content=PDF, headers=headers)
                self.assertEqual((refused.status_code, refused.json()["code"]), (409, code))

    def test_the_texts_of_the_refusals(self):
        import ast
        from markdown_for_llms import services
        tree = ast.parse(Path(services.__file__).read_text())
        literal, = [n.value for n in ast.walk(tree) if isinstance(n, ast.Assign)
                    and any(getattr(t, "id", None) == "REFUSALS" for t in n.targets)]
        keys = [k.value for k in literal.keys]
        self.assertEqual(len(keys), len(set(keys)))                           # one text per code
        for code in ("server_cap", "server_key_missing", "server_key_unsafe", "server_not_configured",
                     "server_billed_over_bound", "session_orphaned", "gpu_preset_not_enabled", "transit_not_confirmed"):
            with self.subTest(code=code):
                self.assertNotEqual(str(services.Refused(code)), "Request refused")
        self.assertIn("minimum reserve", services.REFUSALS["server_cap"])
        transit = services.REFUSALS["transit_not_confirmed"]
        self.assertFalse(transit.startswith("The Datalab route"))
        self.assertIn("Datalab", transit)
        self.assertIn("GPU", transit)
        refused = self.post("/api/v1/gpu/arm", {"gpu_preset": "runpod-serverless-volume-dc", "transit_confirmed": False})
        self.assertEqual(refused.json()["error"], transit)


class CliTests(Base):
    def test_a_preset_with_the_heartbeat_of_another_worker(self):
        """`mdf submit --preset` as `?preset=` (C-DB-NB2.2): the availability of the instance alone."""
        self.heartbeat(PresetParityTests.UNAVAILABLE)
        pdf = Path(self.temp.name) / "a.pdf"
        pdf.write_bytes(PDF)
        code, body = self.mdf("submit", str(pdf), "--preset", "batch", "--confirm-transit")
        self.assertEqual((code, body["state"], body["options"]["gpu_preset"]), (0, "queued", "runpod-serverless-volume-dc"))

    def test_the_help_of_the_transit_names_both_routes(self):
        """`--confirm-transit` of `mdf submit` and `mdf retry` is for Datalab and for the GPU presets (C-DB-NB2.3)."""
        for command in ("submit", "retry"):
            out = io.StringIO()
            with redirect_stdout(out), self.assertRaises(SystemExit):
                main([command, "--help"])
            text = " ".join(out.getvalue().split())
            with self.subTest(command=command):
                self.assertIn("Datalab", text)
                self.assertIn("GPU preset", text)

    def mdf(self, *argv):
        env = Path(self.temp.name) / "mdf.env"
        env.write_text("".join("MDF_%s=%s\n" % (k.upper(), v) for k, v in dict(
            ENABLED, gpu_ledger_dir=str(self.ledger), servers=self.settings.servers).items()))
        out = io.StringIO()
        with redirect_stdout(out):
            code = main(["--env-file", str(env), "--data-root", self.settings.data_root, "--model-root",
                         self.settings.model_root, *argv])
        return code, json.loads(out.getvalue())

    def test_the_same_services(self):
        code, body = self.mdf("gpu", "session")
        self.assertEqual((code, body["session"]), (0, None))
        code, body = self.mdf("gpu", "arm", "runpod-serverless-volume-dc")
        self.assertEqual((code, body.get("code")), (1, "transit_not_confirmed"))
        code, body = self.mdf("gpu", "arm", "runpod-serverless-volume-dc", "--confirm-transit")
        self.assertEqual((code, body["state"]), (0, "pending"))
        code, body = self.mdf("gpu", "disarm", "runpod-serverless-volume-dc")
        self.assertEqual((code, body["kind"]), (0, "disarm"))
        code, body = self.mdf("gpu", "estimate", "modal-base")
        self.assertEqual((code, body["provider"]), (0, "modal"))


if __name__ == "__main__":
    unittest.main()
