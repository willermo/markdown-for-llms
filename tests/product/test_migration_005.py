"""Schema 5 of the archive (MFL023 B4; plan r3 §4.7): the GPU sessions (`gpu_sessions`), the requests to arm ahead and to
disarm (`gpu_requests`) and the facts of a run on its session (`runs.gpu`), with the return to schema 4 or 3 (the
identity unchanged byte for byte) refused, before anything changes, by a session that is not closed (`uncertain`,
`orphan` and `stopped` included), a pending request, the GPU facts of a run or a run on the route marker-server. The
return to 3 reads the options of a run against the five of schema 3 (`storage.V3_OPTIONS`), never against the routes of
today, which B3 extended."""
import json
from pathlib import Path
import sqlite3
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_migration_004 import Base, CURRENT, V3_OPTIONS, fresh  # noqa: E402

from markdown_for_llms import storage  # noqa: E402
from markdown_for_llms.storage import Storage, StorageError  # noqa: E402

GPU = {"route": "marker-server", "ocr": "auto", "lang": None, "datalab_mode": None, "transit_confirmed": True,
       "server_mode": "balanced", "gpu_preset": "modal-base"}


class SchemaFiveTests(Base):
    def session(self, state, arm_id="a1"):
        self.execute("INSERT INTO gpu_sessions (arm_id, owner, gpu_preset, provider, resource, url, ledger_path, env_file, "
                     "key_env, origin, a_cents, w_s, state, created, updated) VALUES "
                     "(?,'local','modal-base','modal','mdf-surya','https://x.modal.run','/l/spend.jsonl',NULL,'[]','job',"
                     "'50.00',1179,?,0,0)", arm_id, state)

    def test_a_new_archive_is_born_at_schema_5(self):
        self.open()
        snap = self.snapshot()
        self.assertEqual((storage.SCHEMA, snap[1], snap[2]), (5, 5, fresh(*CURRENT)))
        names = {row[1] for row in snap[2]}
        self.assertLessEqual({"gpu_sessions", "gpu_requests"}, names)
        with sqlite3.connect(self.root / "archive.sqlite3") as conn:
            self.assertIn("gpu", [r[1] for r in conn.execute("PRAGMA table_info(runs)")])
        conn.close()

    def test_an_archive_of_schema_3_goes_to_5_and_back(self):
        self.make(3)
        before = self.snapshot()
        self.open()
        up = self.snapshot()
        self.assertEqual((up[0], up[1], up[2], up[3]), (before[0], 5, fresh(*CURRENT), before[3]))
        self.assertEqual(Storage.downgrade(self.settings, self.settings.identity(), 4)["schema"], 4)
        mid = self.snapshot()
        self.assertEqual((mid[0], mid[1], mid[2]), (before[0], 4, fresh("001", "002", "003", "004")))
        self.open()
        code, out = self.downgrade_cli("--to-schema", "3")                  # two levels: 5 to 4 to 3
        self.assertEqual((code, out["schema"]), (0, 3))
        down = self.snapshot()
        self.assertEqual(down, (before[0], 3, fresh("001", "002", "003"), before[3]))

    def test_what_refuses_the_return_from_5(self):
        cases = [("session uncertain", lambda: self.session("uncertain")),
                 ("session orphan", lambda: self.session("orphan")),
                 ("session stopped", lambda: self.session("stopped")),
                 ("session open", lambda: self.session("open")),
                 ("request pending", lambda: self.execute("INSERT INTO gpu_requests VALUES ('q','local','modal-base','arm',"
                                                          "'pending',NULL,0,0)")),
                 ("runs.gpu", lambda: self.execute("UPDATE runs SET gpu=?", json.dumps({"preset": "modal-base"}))),
                 ("a GPU run", lambda: self.execute("UPDATE runs SET options=?", json.dumps(GPU)))]
        for label, change in cases:
            for to in (4, 3):
                with self.subTest(case=label, to=to):
                    self.tmp.cleanup()
                    __import__("os").makedirs(self.tmp.name)
                    self.make(3)
                    self.open()
                    change()
                    snap = self.snapshot()
                    with self.assertRaisesRegex(StorageError, "Downgrade unavailable"):
                        Storage.downgrade(self.settings, self.settings.identity(), to)
                    self.assertEqual(self.snapshot(), snap)                   # nothing changed

    def test_closed_sessions_and_done_requests_return(self):
        self.make(3)
        self.open()
        self.session("closed")
        self.execute("INSERT INTO gpu_requests VALUES ('q','local','modal-base','disarm','done',NULL,0,0)")
        self.assertEqual(Storage.downgrade(self.settings, self.settings.identity(), 4)["schema"], 4)

    def test_the_options_of_schema_3_are_its_own(self):
        """A run with the GPU options cannot go back to 3 by the check of 4 to 3 either (the defect of B3 corrected in
        B4: that check read the job options of the routes of today)."""
        self.make(3)
        self.open()
        self.execute("UPDATE runs SET options=?", json.dumps(dict(V3_OPTIONS, server_mode="balanced")))
        with sqlite3.connect(self.root / "archive.sqlite3") as conn:
            with self.assertRaisesRegex(StorageError, "job options of schema 4"):
                Storage._check_v3_content(conn)
        conn.close()
        self.assertEqual(storage.V3_OPTIONS, tuple(V3_OPTIONS))

    def test_an_interrupted_upgrade_to_5_is_completed_by_the_next_opening(self):
        from unittest.mock import patch
        self.make(3)
        before = self.snapshot()
        real = storage.migration

        def broken(name):
            if name == "005":
                return real(name).replace("PRAGMA user_version = 5;", "SELECT no_such_function();")
            return real(name)
        with patch.object(storage, "migration", side_effect=broken):
            with self.assertRaises(sqlite3.OperationalError):
                self.open()
        self.assertEqual(self.snapshot(), before)                      # one transaction: nothing of 004 and 005 stays
        self.open()
        self.assertEqual(self.snapshot()[1:3], (5, fresh(*CURRENT)))


if __name__ == "__main__":
    unittest.main()
