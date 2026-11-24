"""The GPU session of the product in the worker (MFL023 B4; plan r2 §4.1-§4.5, r3 §4.2-§4.3, §6 point 5): the opening
and its refusals (the floor before any lock, row or call), a batch of ten jobs in one window, the jobs of the open
preset first, LINGER, arming ahead and its end, a disarm between and during the jobs, the term T during the arm and
during a job, Modal without capacity, a URL or a rate that differ, and closed fields everywhere. The provider is a fake
in a JSON file served to the children of `opchild` (`provider_driver.py`); the engine is a fake of the route
(`gpu_harness.gpu_convert`); the clocks are given to the worker."""
from datetime import datetime, timezone
from decimal import Decimal
import json
import os
from pathlib import Path
import sqlite3
import sys
import threading
import time
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gpu_harness import JOBS, KEY, Instance, gpu_convert, wait_for  # noqa: E402

from markdown_for_llms.bench import serverspend, spend  # noqa: E402
from markdown_for_llms.engine.relay import credential_forms  # noqa: E402
from markdown_for_llms.gpu import lifecycle, providers  # noqa: E402

FORMS = sorted({f.decode() for f in credential_forms([KEY])})


def process(inst, w, convert=None):
    row = w.claim()
    with mock.patch("markdown_for_llms.worker.convert", convert or gpu_convert(inst.clocks)):
        w.process(row)
    return inst.run(row["id"])


def gpu_of(run):
    return json.loads(run["gpu"]) if run.get("gpu") else None


class Base(unittest.TestCase):
    def close_idle(self, inst, w):
        inst.clocks.advance(61)
        w.gpu.poll()
        inst.settle(w)

    def assertNoKey(self, inst, w):
        """No form of the synthetic key in the database (every table), the ledger, the diaries and the heartbeat."""
        texts = []
        with sqlite3.connect(Path(inst.settings.data_root) / "archive.sqlite3") as conn:
            for (table,) in conn.execute("SELECT name FROM sqlite_master WHERE type='table'"):
                texts.append(json.dumps(conn.execute("SELECT * FROM %s" % table).fetchall()))
        conn.close()
        for path in inst.ledger_dir.rglob("*"):
            if path.is_file() and not path.name.endswith(".lock"):
                texts.append(path.read_text(errors="replace"))
        texts.append(json.dumps(w.readiness))
        for text in texts:
            for form in FORMS:
                self.assertNotIn(form, text)


class TestOpening(Base):
    def test_a_job_on_runpod_and_its_window(self):
        inst = Instance(self)
        w = inst.worker()
        job = inst.submit()
        run = process(inst, w)
        self.assertEqual(run["state"], "completed", run["error"])
        self.assertEqual(inst.calls(), ["state", "arm"])
        arm, = inst.rows("arm")
        self.assertEqual((arm["a_cents"], arm["window_s"], arm["z_s"], arm["tail_s"]), ("40.00", 1703, 120, 30))
        self.assertEqual(arm["levers"], providers.levers_text("compile-cache:" + "6222ea36b0e660ccbd89b25386a30aab3dbaec14"
                                                              "231bd36e7aae4f3b9751e7f4", "runpod"))
        self.assertEqual(inst.provider_state()["levers"], arm["levers"])            # the levers of the preset only
        gpu = gpu_of(run)
        self.assertEqual((gpu["preset"], gpu["provider"], gpu["arm_id"], gpu["a_cents"], gpu["completed"]),
                         ("runpod-serverless-volume-dc", "runpod", arm["arm_id"], "40.00", True))
        self.assertEqual(inst.sessions()[0]["state"], "linger")
        self.close_idle(inst, w)
        self.assertEqual(inst.calls(), ["state", "arm", "disarm", "zero"])
        session, = inst.sessions()
        self.assertEqual((session["state"], session["reason"]), ("closed", "idle"))
        self.assertEqual([r["kind"] for r in inst.rows()], ["arm", "reserve", "complete", "disarm"])
        window, = inst.summary()["windows"]
        self.assertEqual(window["status"], "clean")
        self.assertLessEqual(Decimal(window["amount_cents"]), Decimal("40.00"))
        diary = json.loads((inst.ledger_dir / "worker-logs" / arm["arm_id"] / "session.json").read_text())
        self.assertEqual(diary["phase"], "closed")
        self.assertFalse(list(inst.ledger_dir.glob("*.lock")))
        self.assertNoKey(inst, w)
        self.assertEqual(job["id"], run["id"])

    def test_ten_jobs_in_one_window(self):
        inst = Instance(self, session_cap="0.70")
        w = inst.worker()
        ids = [inst.submit()["id"] for _ in range(10)]
        for _ in ids:
            run = process(inst, w, gpu_convert(inst.clocks, run_s=20))
            self.assertEqual(run["state"], "completed", run["error"])
            self.assertEqual(inst.sessions()[0]["state"], "open" if _ != ids[-1] else "linger")
        self.close_idle(inst, w)
        self.assertEqual(len(inst.rows("arm")), 1)
        self.assertEqual(len(inst.rows("reserve")), 10)
        self.assertEqual(inst.calls().count("arm"), 1)
        window, = inst.summary()["windows"]
        self.assertEqual((window["status"], window["attempts"]), ("clean", 10))

    def test_the_jobs_of_the_open_preset_first(self):
        inst = Instance(self)
        w = inst.worker()
        first = inst.submit()["id"]
        local = inst.submit({"route": "marker"})["id"]
        second = inst.submit()["id"]
        self.assertEqual(process(inst, w)["id"], first)
        row = w.claim()
        self.assertEqual(row["id"], second)                       # before the local job, submitted earlier
        with mock.patch("markdown_for_llms.worker.convert", gpu_convert(inst.clocks)):
            w.process(row)
        self.assertEqual(inst.run(local)["state"], "queued")
        w.gpu.poll()                                              # LINGER after a job, a local job queued: close now
        inst.settle(w)
        session, = inst.sessions()
        self.assertEqual((session["state"], session["reason"]), ("closed", "local_job"))

    def test_the_floor_refuses_before_any_lock_row_or_call(self):
        for provider, cap in (("runpod", "0.28"), ("modal", "0.45")):
            with self.subTest(provider=provider):
                inst = Instance(self, provider=provider, session_cap=cap)
                w = inst.worker()
                inst.submit()
                run = process(inst, w)
                self.assertEqual(run["state"], "failed")
                self.assertIn("server_cap", run["error"])
                self.assertEqual((inst.calls(), inst.sessions(), inst.rows()), ([], [], []))
                self.assertFalse(list(inst.ledger_dir.glob("*.lock")))
                self.assertEqual(gpu_of(run)["error"], {"class": "Refusal", "op": "server_cap", "http": None})

    def test_refusals_at_the_reading(self):
        for state, code in (({"armed": True}, "provider_not_idle"), ({"busy": True}, "provider_not_idle"),
                            ({"volume": {"type": "STANDARD"}}, "endpoint_out_of_pins"),
                            ({"volume": {"dataCenter": "EU-SE-1"}}, "endpoint_out_of_pins"),
                            ({"url": "https://ep9.api.runpod.ai"}, "server_not_configured"),
                            ({"fail": {"state": 1}}, "server_unavailable")):
            with self.subTest(state=state):
                inst = Instance(self, state=state)
                w = inst.worker()
                inst.submit()
                run = process(inst, w)
                self.assertEqual(run["state"], "failed")
                self.assertIn(code, run["error"])
                self.assertEqual((inst.calls(), inst.sessions(), inst.rows()), (["state"], [], []))
                self.assertNoKey(inst, w)

    def test_an_orphan_or_a_live_window_in_the_ledger(self):
        for live, code in ((False, "session_orphaned"), (True, "provider_not_idle")):
            with self.subTest(live=live):
                inst = Instance(self)
                inst.ledger_dir.mkdir()
                ledger = spend.Ledger.file(inst.ledger_dir / spend.LEDGER)
                with ledger.locked() as locked:
                    row = serverspend.arm(locked, ledger.path, "runpod", "mdf-surya", "https://ep1.api.runpod.ai", "10",
                                          2e9, "200", "0.0002090948", 1.7e9)
                if live:
                    import fcntl
                    fd = os.open(serverspend.lock_path(ledger.path, row["arm_id"]), os.O_RDWR | os.O_CREAT, 0o600)
                    fcntl.flock(fd, fcntl.LOCK_EX)
                    self.addCleanup(os.close, fd)
                w = inst.worker()
                w.gpu.recover_cli = lambda: None             # here the opening itself, not the recovery
                inst.submit()
                run = process(inst, w)
                self.assertIn(code, run["error"])
                self.assertEqual((inst.calls(), inst.sessions()), ([], []))

    def test_modal_and_the_url_of_its_deploy(self):
        inst = Instance(self, provider="modal", session_cap="0.50", state={"url": "https://other--x-serve.modal.run"})
        w = inst.worker()
        inst.submit()
        run = process(inst, w)
        self.assertIn("server_not_configured", run["error"])
        inst.settle(w)
        session, = inst.sessions()
        self.assertEqual((session["state"], session["reason"]), ("closed", "server_not_configured"))
        self.assertEqual(inst.calls(), ["state", "arm", "disarm", "zero"])
        window, = inst.summary()["windows"]
        self.assertEqual(window["status"], "unclean")                 # not an ordinary close, but certified within A
        self.assertLessEqual(Decimal(window["amount_cents"]), Decimal("50.00"))

    def test_a_rate_that_is_not_the_rate_of_the_window(self):
        inst = Instance(self)
        w = inst.worker()
        inst.submit()
        self.assertEqual(process(inst, w)["state"], "completed")
        w.gpu.session.ep.rate_usd_per_s = "0.00019"
        inst.submit()
        run = process(inst, w)
        self.assertIn("server_not_configured", run["error"])
        self.assertEqual(len(inst.rows("reserve")), 1)


class TestArmAhead(Base):
    def test_a_window_that_waits_then_closes(self):
        """R2 of the plan: RunPod at 0.30 USD, W 1224 s; nothing arrives, the session lingers 264 s and closes."""
        inst = Instance(self, session_cap="0.30")
        w = inst.worker()
        inst.request("arm")
        w.gpu.poll()
        session, = inst.sessions()
        self.assertEqual((session["state"], session["origin"], session["w_s"]), ("linger", "ahead", 1224))
        inst.clocks.advance(263)
        w.gpu.poll()
        self.assertEqual(inst.sessions()[0]["state"], "linger")
        inst.clocks.advance(2)
        w.gpu.poll()
        inst.settle(w)
        session, = inst.sessions()
        self.assertEqual((session["state"], session["reason"]), ("closed", "arm_ahead_expired"))
        with inst.service.storage.connect() as conn:
            self.assertEqual(conn.execute("SELECT state FROM gpu_requests").fetchone()[0], "done")

    def test_a_job_on_a_window_armed_ahead(self):
        inst = Instance(self)
        w = inst.worker()
        inst.request("arm")
        w.gpu.poll()
        inst.clocks.advance(120)
        inst.submit()
        self.assertEqual(process(inst, w)["state"], "completed")
        self.assertEqual(inst.calls().count("arm"), 1)
        self.close_idle(inst, w)
        self.assertEqual(inst.sessions()[0]["reason"], "idle")

    def test_requests_refused(self):
        inst = Instance(self, provider="modal", session_cap="0.50")
        w = inst.worker()
        inst.request("arm", "modal-base")                         # Modal: arming ahead gives nothing (S2)
        w.gpu.poll()
        with inst.service.storage.connect() as conn:
            conn.execute("INSERT INTO gpu_requests VALUES ('old','local','runpod-serverless-volume-dc','arm','pending',"
                         "NULL,?,?)", (time.time() - 700, time.time() - 700))
        w.gpu.poll()
        with inst.service.storage.connect() as conn:
            got = sorted(tuple(r) for r in conn.execute("SELECT gpu_preset, state, code FROM gpu_requests"))
        self.assertEqual(got, [("modal-base", "refused", "route_option_unsupported"),
                               ("runpod-serverless-volume-dc", "refused", "stale")])
        self.assertEqual((inst.calls(), inst.sessions()), ([], []))


class TestDisarm(Base):
    def test_a_disarm_between_two_jobs(self):
        inst = Instance(self)
        w = inst.worker()
        inst.submit()
        process(inst, w)
        inst.request("disarm")
        w.gpu.poll()
        inst.settle(w)
        session, = inst.sessions()
        self.assertEqual((session["state"], session["reason"]), ("closed", "user_disarm"))
        self.assertEqual(inst.summary()["windows"][0]["status"], "clean")

    def test_a_disarm_during_a_job(self):
        """The API (B5) cancels the jobs of the preset and asks for the disarm in one transaction: the job stops at its
        next tick, then the worker closes the session."""
        inst = Instance(self)
        w = inst.worker()
        job = inst.submit()

        def during(observer):
            with inst.service.storage.connect() as conn:
                conn.execute("UPDATE runs SET cancel=1 WHERE id=?", (job["id"],))
            inst.request("disarm")
        run = process(inst, w, gpu_convert(inst.clocks, during=during))
        self.assertEqual(run["state"], "cancelled")
        w.gpu.poll()
        inst.settle(w)
        session, = inst.sessions()
        self.assertEqual((session["state"], session["reason"]), ("closed", "user_disarm"))
        self.assertEqual(len(inst.rows("complete")), 1)


class TestTheServicesOfTheApi(Base):
    """B5: the disarm of the API (`Service.gpu_disarm`) during a job, arming ahead (`Service.gpu_arm`), and the readiness of
    the worker with the reserve, the window and the minimum reserve of each preset."""

    def test_the_disarm_of_the_api_during_a_job(self):
        inst = Instance(self)
        w = inst.worker()
        job = inst.submit()
        queued = inst.submit()["id"]

        def during(observer):
            inst.service.gpu_disarm("runpod-serverless-volume-dc")
        run = process(inst, w, gpu_convert(inst.clocks, during=during))
        self.assertEqual((run["id"], run["state"]), (job["id"], "cancelled"))
        self.assertEqual(inst.run(queued)["state"], "cancelled")
        w.gpu.poll()
        inst.settle(w)
        session, = inst.sessions()
        self.assertEqual((session["state"], session["reason"]), ("closed", "user_disarm"))
        self.assertIsNone(w.claim())

    def test_arm_ahead_from_the_api_and_the_readiness(self):
        inst = Instance(self)
        w = inst.worker()
        inst.service.gpu_arm("runpod-serverless-volume-dc", True)
        w.gpu.poll()
        self.assertEqual(inst.sessions()[0]["state"], "linger")
        w.ready()
        preset = w.readiness["gpu"]["presets"]["runpod-serverless-volume-dc"]
        # the jobs of the preset run on the session already open
        self.assertEqual((preset["available"], preset["minimum_reserve_cents"]), (True, "28.76"))
        modal = w.readiness["gpu"]["presets"]["modal-base"]
        self.assertEqual((modal["minimum_reserve_cents"], modal["a_cents"], modal["w_s"]), ("45.23", "40.00", 907))
        self.assertEqual((modal["available"], modal["code"]), (False, "server_cap"))   # 40 cents under 45.23


class TestTheTerm(Base):
    def test_the_term_kills_an_arm_still_in_progress(self):
        inst = Instance(self, state={"hang": {"arm": 30}})
        w = inst.worker()
        inst.submit()

        def at_the_arm():
            wait_for(lambda: inst.provider_state().get("ticks", 0) > 0, what="the arm in progress")
            inst.clocks.advance(1704)                               # past T = t_arm + 1703, both clocks
        threading.Thread(target=at_the_arm, daemon=True).start()
        t0 = time.monotonic()
        run = process(inst, w)
        self.assertLess(time.monotonic() - t0, 20)
        self.assertIn("server_deadline", run["error"])
        inst.settle(w)
        session, = inst.sessions()
        self.assertEqual((session["state"], session["reason"]), ("closed", "term"))
        ticks = inst.provider_state()["ticks"]
        time.sleep(0.4)
        self.assertEqual(inst.provider_state()["ticks"], ticks)    # the arm child is gone
        self.assertEqual(inst.calls()[-2:], ["disarm", "zero"])
        dis, = inst.rows("disarm")
        self.assertFalse(dis["orphan"])
        arm, = inst.rows("arm")
        self.assertIsNone(serverspend.closing(arm, dis["t_zero"]))

    def test_the_term_while_the_job_is_busy(self):
        """A local phase or a slow publication past T: the clock thread closes while the main thread is busy."""
        inst = Instance(self)
        w = inst.worker()
        inst.submit()

        def during(observer):
            inst.clocks.advance(1704)
            wait_for(lambda: inst.sessions()[0]["state"] == "closed", what="the close while the job is busy")
        run = process(inst, w, gpu_convert(inst.clocks, during=during))
        self.assertIn("server_deadline", run["error"])
        arm, = inst.rows("arm")
        dis, = inst.rows("disarm")
        self.assertLessEqual(dis["t_zero"], arm["until"] + 60 + 120)
        self.assertEqual(len(inst.rows("complete")), 1)


class TestModal(Base):
    def test_no_capacity(self):
        inst = Instance(self, provider="modal", session_cap="0.50")
        stamp = datetime.fromtimestamp(inst.clocks.wall() + 5, timezone.utc).isoformat()
        inst.change(system=["%s fu-abc Function 'serve' (fu-abc) is waiting to be scheduled on a GPU_L4 worker. "
                            "We are actively working on acquiring more capacity." % stamp], containers=[])
        w = inst.worker()
        inst.submit()

        def during(observer):
            for _ in range(12):
                inst.clocks.advance(30)
                time.sleep(0.4)
                if w.gpu.session.state.closing.is_set():
                    break
            wait_for(lambda: inst.sessions()[0]["state"] == "closed", what="the close for no capacity")
        run = process(inst, w, gpu_convert(inst.clocks, during=during))
        self.assertIn("modal_no_capacity", run["error"])
        self.assertIn("choose the preset Batch (RunPod)", run["error"])
        session, = inst.sessions()
        self.assertEqual(session["reason"], "modal_no_capacity")
        self.assertIn("system_logs", inst.calls())


class TestClosedFields(Base):
    def test_errors_of_the_provider_reach_nothing(self):
        """The provider fails the arm, then the disarm, with its key in its text: closed fields in the database, the
        ledger, the diary and the heartbeat; no form of the key anywhere."""
        inst = Instance(self, state={"fail": {"arm": 1}})
        w = inst.worker()
        inst.submit()
        run = process(inst, w)
        self.assertIn("server_unavailable", run["error"])
        inst.settle(w)
        inst.change(fail={"disarm": 1})
        inst.submit()
        process(inst, w)
        self.close_idle(inst, w)
        w.ready()
        diaries = [json.loads(p.read_text()) for p in inst.ledger_dir.glob("worker-logs/*/session.json")]
        errors = [e for d in diaries for e in d["events"] if e["event"].endswith("-error")]
        self.assertEqual(sorted({(e["event"], e["class"], e["op"]) for e in errors}),
                         [("arm-error", "ProviderError", "runpod.arm"), ("disarm-error", "ProviderError", "runpod.disarm")])
        self.assertNoKey(inst, w)


class TestStop(Base):
    def test_a_sigterm_during_the_opening(self):
        inst = Instance(self, state={"hang": {"arm": 30}})
        w = inst.worker()
        inst.submit()

        def stop():
            wait_for(lambda: inst.provider_state().get("ticks", 0) > 0, what="the arm in progress")
            w.stop = True
        threading.Thread(target=stop, daemon=True).start()
        run = process(inst, w)
        self.assertEqual(run["state"], "interrupted")
        w.gpu.shutdown()
        session, = inst.sessions()
        self.assertEqual((session["state"], session["reason"]), ("closed", "stop"))
        self.assertEqual(len(inst.rows("disarm")), 1)


if __name__ == "__main__":
    unittest.main()
