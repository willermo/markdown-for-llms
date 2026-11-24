"""The recovery of the GPU sessions when the worker starts (MFL023 B4; SOL-R3 of the plan r1, plan r3 §4.4, §6 point 7,
SOL3-N2 = C-R3-NB3 of the review of r3).

Before the first claim, every row of `gpu_sessions` that is not closed is closed from its own references (ledger,
provider, resource, env file), never from the settings of today: a session that died before its arm row never existed
(`never_armed`, no call); one armed and never disarmed is disarmed by its recorded resource and its window closed as an
orphan, held at the cap until the invoice; one whose `disarm` row was written takes its state from the ledger (an
orphan row stays `orphan`, with its command, until the reconciliation). A ledger recorded but missing, a key missing or
no zero is `gpu_unrecoverable`: the worker stops before any claim, with the commands. Then the orphan windows that a
session of the command line left on the endpoints of the enabled presets."""
import json
import os
from pathlib import Path
import sys
import time
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gpu_harness import RUNPOD_URL, Instance, wait_for  # noqa: E402

from markdown_for_llms.bench import serverspend, spend  # noqa: E402
from markdown_for_llms.gpu import lifecycle  # noqa: E402
from markdown_for_llms.settings import Settings  # noqa: E402
from markdown_for_llms.storage import StorageError  # noqa: E402

RATE = "0.0002090948"


class Base(unittest.TestCase):
    def left(self, inst, state, armed=True, disarm=None, arm_id=None, resource="mdf-surya", ledger=None):
        """What a dead worker left: the row of gpu_sessions in `state`, the arm row when `armed`, and a disarm row
        (`disarm`: its extra fields, `orphan` included) when given."""
        arm_id = arm_id or serverspend.new_arm_id()
        ledger = Path(ledger or inst.ledger_dir / spend.LEDGER)
        ledger.parent.mkdir(parents=True, exist_ok=True)
        lg = spend.Ledger.file(ledger)
        with lg.locked() as locked:
            if armed:
                serverspend.arm(locked, ledger, "runpod", resource, RUNPOD_URL, "40", 1.8e9 + 1703, "200", RATE, 1.8e9,
                                arm_id=arm_id, extra={"window_s": 1703, "z_s": 120, "tail_s": 30})
            if disarm is not None:
                d = dict(disarm)
                serverspend.disarm(locked, "runpod", arm_id, 1.8e9 + 100, d.pop("clean", False),
                                   orphan=d.pop("orphan", False), extra=d)
        serverspend.lock_path(ledger, arm_id).write_text("")
        now = time.time()
        with inst.service.storage.connect() as conn:
            conn.execute("INSERT INTO gpu_sessions (arm_id, owner, gpu_preset, provider, resource, url, ledger_path, "
                         "env_file, key_env, origin, a_cents, w_s, t_arm, until, state, created, updated) VALUES "
                         "(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                         (arm_id, "local", "runpod-serverless-volume-dc", "runpod", resource, RUNPOD_URL, str(ledger),
                          str(inst.env), '["RUNPOD_API_KEY"]', "job", "40.00", 1703, 1.8e9, 1.8e9 + 1703, state, now, now))
        return arm_id

    def session(self, inst, arm_id):
        return [s for s in inst.sessions() if s["arm_id"] == arm_id][0]

    def window(self, inst, arm_id, ledger=None):
        rows = spend.read(ledger or inst.ledger_dir / spend.LEDGER)
        return [r for r in rows if r.get("arm_id") == arm_id and r.get("kind") == "disarm"]


class TestRecovery(Base):
    def test_a_session_that_died_before_its_arm_row(self):
        inst = Instance(self)
        arm_id = self.left(inst, "opening", armed=False)
        w = inst.worker()
        w.gpu.recover()
        s = self.session(inst, arm_id)
        self.assertEqual((s["state"], s["reason"]), ("closed", "never_armed"))
        self.assertEqual(inst.calls(), [])                           # no call, no reconciliation asked
        self.assertFalse(serverspend.lock_path(inst.ledger_dir / spend.LEDGER, arm_id).exists())

    def test_a_window_armed_and_never_disarmed(self):
        for state in ("opening", "open", "linger", "closing"):
            with self.subTest(state=state):
                inst = Instance(self, state={"armed": True, "busy": True})
                arm_id = self.left(inst, state)
                w = inst.worker()
                w.gpu.recover()
                self.assertEqual(inst.calls(), ["disarm", "zero"])
                s = self.session(inst, arm_id)
                self.assertEqual((s["state"], s["reason"]), ("orphan", "worker_died"))
                dis, = self.window(inst, arm_id)
                self.assertTrue(dis["orphan"])
                summary = inst.summary()
                self.assertEqual((summary["total_cents"], summary["orphans"]), ("200.00", [arm_id]))
                ledger = spend.Ledger.file(inst.ledger_dir / spend.LEDGER)
                with ledger.locked() as locked:
                    serverspend.reconcile(locked, ledger.path, "runpod", "3.10", manual=True, arms=[arm_id], orphan=True)
                w.gpu.reconciled()
                self.assertEqual(self.session(inst, arm_id)["state"], "closed")

    def test_a_disarm_row_already_written(self):
        """SOL3-N2: an orphan row (a second death of the recovery between its row and its state) stays orphan, with no
        call, until the reconciliation; an ordinary row is closed; an uncertain row is uncertain."""
        for disarm, state, expected in (({"orphan": True}, "open", "orphan"), ({"orphan": True}, "closing", "orphan"),
                                        ({"orphan": False, "clean": True}, "closing", "closed"),
                                        ({"orphan": True, "uncertain": "over_window"}, "closing", "uncertain")):
            with self.subTest(disarm=disarm, state=state):
                inst = Instance(self)
                arm_id = self.left(inst, state, disarm=disarm)
                w = inst.worker()
                w.gpu.recover()
                self.assertEqual(inst.calls(), [])
                s = self.session(inst, arm_id)
                self.assertEqual(s["state"], expected)
                self.assertFalse(serverspend.lock_path(inst.ledger_dir / spend.LEDGER, arm_id).exists())
                if expected == "orphan":
                    self.assertIn("--orphan --arm %s" % arm_id,
                                  lifecycle.commands("runpod", "mdf-surya", s["ledger_path"], arm_id)[0])
                    ledger = spend.Ledger.file(inst.ledger_dir / spend.LEDGER)
                    with ledger.locked() as locked:
                        serverspend.reconcile(locked, ledger.path, "runpod", "1", manual=True, arms=[arm_id], orphan=True)
                    w.gpu.reconciled()
                    self.assertEqual(self.session(inst, arm_id)["state"], "closed")

    def test_a_stopped_session(self):
        inst = Instance(self, state={"armed": False, "busy": True, "zero_after": 1})
        arm_id = self.left(inst, "stopped")
        w = inst.worker()
        w.gpu.recover()
        s = self.session(inst, arm_id)
        self.assertEqual((s["state"], s["reason"]), ("orphan", "zero_missing"))
        self.assertEqual(inst.calls()[0], "disarm")

    def test_the_recovery_ignores_the_settings_of_today(self):
        """The preset disabled, the URL of servers.json changed, the ledger of the instance moved: the recorded
        references close the window."""
        inst = Instance(self, state={"armed": True})
        arm_id = self.left(inst, "open")
        inst.servers.write_text(json.dumps({"servers": {"rp": {"kind": "runpod", "preset": "runpod-serverless-volume-dc",
                                                               "resource": "other", "url": "https://ep7.api.runpod.ai"}}}))
        moved = inst.root / "moved-ledger"
        from dataclasses import replace
        inst.settings = replace(inst.settings, gpu_presets="modal-base", gpu_ledger_dir=str(moved))
        w = inst.worker()
        w.gpu.recover()
        self.assertEqual(inst.calls(), ["disarm", "zero"])
        self.assertEqual(self.session(inst, arm_id)["state"], "orphan")
        self.assertTrue(self.window(inst, arm_id))                   # in the ledger recorded, not the moved one
        self.assertFalse((moved / spend.LEDGER).exists())

    def test_what_stops_the_worker_before_any_claim(self):
        for case in ("ledger_removed", "ledger_link", "key_missing", "no_zero"):
            with self.subTest(case=case):
                inst = Instance(self, state={"armed": True, "never_zero": case == "no_zero"})
                arm_id = self.left(inst, "open")
                ledger = inst.ledger_dir / spend.LEDGER
                if case == "ledger_removed":
                    ledger.unlink()
                elif case == "ledger_link":
                    real = inst.root / "elsewhere.jsonl"
                    ledger.rename(real)
                    ledger.symlink_to(real)
                elif case == "key_missing":
                    inst.env.write_text("MODAL_PROXY_TOKEN_ID=x\n")
                    inst.env.chmod(0o600)
                w = inst.worker()
                if case == "no_zero":
                    real_sleep = inst.clocks.sleep
                    inst.clocks.sleep = lambda s: real_sleep(s * 30)          # the budget of the zero passes fast
                with self.assertRaises(lifecycle.Unrecoverable) as caught:
                    w.gpu.recover()
                self.assertTrue(any("reconcile" in c for c in caught.exception.commands))
                self.assertEqual(self.session(inst, arm_id)["state"], "open")
                if case != "no_zero":
                    self.assertEqual(inst.calls(), [])

    def test_the_worker_stops_with_its_heartbeat(self):
        inst = Instance(self)
        arm_id = self.left(inst, "open")
        (inst.ledger_dir / spend.LEDGER).unlink()
        w = inst.worker()
        with mock.patch.object(w, "claim", side_effect=AssertionError("claimed")), \
                mock.patch.object(w, "reconcile", side_effect=AssertionError("reconciled first")):
            with self.assertRaises(StorageError) as caught:
                w.run(once=True)
        self.assertIn("gpu_unrecoverable", str(caught.exception))
        with inst.service.storage.connect() as conn:
            beat = json.loads(conn.execute("SELECT readiness FROM worker").fetchone()[0])
        self.assertEqual((beat["ready"], beat["code"]), (False, "gpu_unrecoverable"))
        self.assertTrue(any("session.py disarm" in c for c in beat["commands"]))
        self.assertEqual(self.session(inst, arm_id)["state"], "open")

    def test_an_orphan_left_by_the_command_line(self):
        inst = Instance(self, state={"armed": True})
        ledger = spend.Ledger.file(inst.ledger_dir / spend.LEDGER)
        inst.ledger_dir.mkdir()
        with ledger.locked() as locked:
            row = serverspend.arm(locked, ledger.path, "runpod", "mdf-surya", RUNPOD_URL, "30", 1.8e9, "200", RATE, 1.7e9)
        w = inst.worker()
        w.gpu.recover()
        self.assertEqual(inst.calls(), ["disarm", "zero"])
        dis, = self.window(inst, row["arm_id"])
        self.assertTrue(dis["orphan"])
        self.assertEqual(inst.sessions(), [])


if __name__ == "__main__":
    unittest.main()
