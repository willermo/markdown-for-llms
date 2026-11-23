"""Database schema 4 (MFL023 A4, C-B4 of the plan r1, C-R2-NB1 of the plan r2): the conversion presets of the users and
their default, with a return to schema 3 or 2 (`mdf downgrade-storage --to-schema N`, 2 by default). An archive of
schema 3 as fe6b952 leaves it goes up once, also with concurrent openers, and comes back to 3 with the same catalog as
a new archive of schema 3 and identity.json byte for byte; the checks of every level crossed run before any change; the
runs of schema 2 (the 13 options of Marker 1) pass the check of 4 to 3 through `stored_options`."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sqlite3
import tempfile
import threading
import unittest
from unittest.mock import patch
import uuid

from markdown_for_llms import storage
from markdown_for_llms.cli import main
from markdown_for_llms.services import Service
from markdown_for_llms.settings import Settings
from markdown_for_llms.storage import Storage, StorageError

MIGRATIONS = Path(__file__).parents[2] / "markdown_for_llms" / "migrations"
V2_OPTIONS = dict(use_llm=False, force_ocr=True, strip_existing_ocr=False, disable_image_extraction=False,
                  paginate_output=True, pdftext_workers=1, layout_batch_size=1, recognition_batch_size=1,
                  detection_batch_size=1, table_rec_batch_size=1, ocr_error_batch_size=1,
                  lowres_image_dpi=96, highres_image_dpi=192)
V3_OPTIONS = {"route": None, "ocr": "auto", "lang": None, "datalab_mode": None, "transit_confirmed": False}


def sql(*names):
    return "".join((MIGRATIONS / (n + ".sql")).read_text() for n in names)


def catalog(conn):
    return conn.execute("SELECT type,name,tbl_name,sql FROM sqlite_master ORDER BY name").fetchall()


def fresh(*names):
    conn = sqlite3.connect(":memory:")
    conn.executescript(sql(*names))
    return catalog(conn)


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.settings = Settings(data_root=self.tmp.name + "/data", model_root=self.tmp.name + "/models",
                                 upload_mib=1, output_mib=3, log_mib=1, storage_mib=16, reserve_mib=1)
        self.root = Path(self.settings.data_root)

    def make(self, schema, options=V3_OPTIONS):
        """An archive as an earlier version leaves it: schema 3 as fe6b952 creates it (migrations 001-003, an identity
        of schema 3 without history), or schema 2 (001-002, the identity of schema 2), one failed run."""
        from test_storage_identity import v2_identity
        self.root.mkdir(mode=0o700)
        for name in ("sources", "attempts", "results", "staging", "downloads", "operations"):
            (self.root / name).mkdir(mode=0o700)
        storage_id = uuid.uuid4().hex
        config = self.settings.identity() if schema == 3 else v2_identity(self.settings)
        (self.root / "identity.json").write_text(json.dumps({"id": storage_id, "config": config}))
        conn = sqlite3.connect(self.root / "archive.sqlite3")
        conn.executescript(sql(*["%03d" % n for n in range(1, schema + 1)]))
        conn.execute("INSERT INTO metadata VALUES ('storage_id',?)", (storage_id,))
        conn.execute("INSERT INTO documents (id,name,source_path,source_sha256,source_bytes,created) "
                     "VALUES ('d1','a.pdf','sources/a.pdf','00',1,0)")
        conn.execute("INSERT INTO runs (id,document_id,attempt_id,options,state,created,updated,error) "
                     "VALUES ('r1','d1','a1',?,'failed',0,0,'synthetic')", (json.dumps(options),))
        conn.commit()
        conn.close()

    def snapshot(self):
        conn = sqlite3.connect(self.root / "archive.sqlite3")
        try:
            return ((self.root / "identity.json").read_bytes(), conn.execute("PRAGMA user_version").fetchone()[0],
                    catalog(conn), conn.execute("SELECT id,options,state,error FROM runs").fetchall())
        finally:
            conn.close()

    def execute(self, statement, *args):
        with sqlite3.connect(self.root / "archive.sqlite3") as conn:
            conn.execute(statement, args)
        conn.close()

    def open(self):
        Service(self.settings).storage.close()

    def downgrade_cli(self, *extra):
        env = Path(self.tmp.name) / "mdf.env"
        env.write_text("MDF_UPLOAD_MIB=1\nMDF_OUTPUT_MIB=3\nMDF_LOG_MIB=1\nMDF_STORAGE_MIB=16\nMDF_RESERVE_MIB=1\n")
        out = io.StringIO()
        with redirect_stdout(out):
            code = main(["--env-file", str(env), "--data-root", self.settings.data_root, "--model-root",
                         self.settings.model_root, "downgrade-storage", "--expected-config", self.settings.identity(),
                         *extra])
        return code, json.loads(out.getvalue())


class SchemaFourTests(Base):
    def test_a_new_archive_is_born_at_schema_4(self):
        self.open()
        snap = self.snapshot()
        self.assertEqual((storage.SCHEMA, snap[1]), (4, 4))
        self.assertEqual(snap[2], fresh("001", "002", "003", "004"))

    def test_a_schema_3_archive_goes_up_once_and_back_to_3(self):
        self.make(3)
        before = self.snapshot()
        self.assertEqual(before[1], 3)
        barrier = threading.Barrier(4)

        def open_storage(_):
            barrier.wait()
            try:
                Storage(self.settings).close()
                return "opened"
            except StorageError as exc:
                return str(exc)
        with ThreadPoolExecutor(max_workers=4) as pool:
            outcomes = list(pool.map(open_storage, range(4)))
        self.assertIn("opened", outcomes)
        self.assertLessEqual(set(outcomes), {"opened", storage.UPGRADE_BUSY})
        up = self.snapshot()
        self.assertEqual((up[0], up[1], up[2], up[3]), (before[0], 4, fresh("001", "002", "003", "004"), before[3]))
        self.assertNotIn("history", json.loads(up[0]))
        code, out = self.downgrade_cli("--to-schema", "3")
        self.assertEqual((code, out), (0, {"previous_config": self.settings.identity(),
                                           "config": self.settings.identity(), "schema": 3}))
        down = self.snapshot()
        self.assertEqual((down[0], down[1], down[2], down[3]), (before[0], 3, fresh("001", "002", "003"), before[3]))
        # already at 3: nothing changes; under 3 is refused, and so is a target outside 2..SCHEMA-1
        Storage.downgrade(self.settings, self.settings.identity(), 3)
        self.assertEqual(self.snapshot(), down)
        for to in (1, 4, 5, True, "3"):
            with self.subTest(to=to):
                with self.assertRaisesRegex(StorageError, "Downgrade target"):
                    Storage.downgrade(self.settings, self.settings.identity(), to)
        self.assertEqual(self.snapshot(), down)
        # the archive born at schema 3 has no identity of schema 2 to return to
        with self.assertRaisesRegex(StorageError, "not upgraded from schema 2"):
            Storage.downgrade(self.settings, self.settings.identity())
        self.assertEqual(self.snapshot(), down)

    def test_the_refusals_of_4_to_3_change_nothing(self):
        self.make(3)
        self.open()
        cases = (
            ("INSERT INTO conversion_presets VALUES ('p1','local','mine','{}',0,0)", "conversion presets"),
            ("INSERT INTO user_settings VALUES ('local','quality-now',0)", "default conversion preset"),
            ("UPDATE runs SET options=?", "job options of schema 4"),
        )
        for statement, message in cases:
            with self.subTest(message=message):
                args = (json.dumps(dict(V3_OPTIONS, preset="quality-now")),) if "?" in statement else ()
                self.execute(statement, *args)
                before = self.snapshot()
                for to in (3, 2):
                    with self.assertRaisesRegex(StorageError, message if to == 3 else "Downgrade unavailable"):
                        Storage.downgrade(self.settings, self.settings.identity(), to)
                    self.assertEqual(self.snapshot(), before)
                self.execute("DELETE FROM conversion_presets")
                self.execute("DELETE FROM user_settings")
                self.execute("UPDATE runs SET options=?", json.dumps(V3_OPTIONS))
        self.execute("INSERT INTO user_settings VALUES ('local',NULL,0)")      # no default: nothing to lose
        self.assertEqual(Storage.downgrade(self.settings, self.settings.identity(), 3)["schema"], 3)

    def test_runs_of_schema_2_pass_the_check_of_4_to_3(self):
        from test_storage_identity import v2_identity
        self.make(2, V2_OPTIONS)
        before = self.snapshot()
        self.open()
        self.assertEqual(self.snapshot()[1], 4)
        self.assertEqual(Storage.downgrade(self.settings, self.settings.identity(), 3)["schema"], 3)
        mid = self.snapshot()
        self.assertEqual((mid[1], mid[2], mid[3]), (3, fresh("001", "002", "003"), before[3]))
        self.assertEqual(json.loads(mid[0])["config"], self.settings.identity())    # 4 to 3: the identity stays
        code, out = self.downgrade_cli()                                             # the default: 2
        self.assertEqual((code, out["schema"], out["config"]), (0, 2, v2_identity(self.settings)))
        self.assertEqual(self.snapshot()[1:], (2, fresh("001", "002"), before[3]))

    def test_from_4_to_2_in_one_command(self):
        from test_storage_identity import v2_identity
        self.make(2, V2_OPTIONS)
        before = self.snapshot()
        self.open()
        code, out = self.downgrade_cli("--to-schema", "2")
        self.assertEqual((code, out["schema"], out["config"]), (0, 2, v2_identity(self.settings)))
        after = self.snapshot()
        self.assertEqual(after[1:], (2, fresh("001", "002"), before[3]))
        self.assertEqual([e["reason"] for e in json.loads(after[0])["history"]], ["identity-v3", "downgrade-v2"])
        with self.assertRaisesRegex(StorageError, "already under schema 3"):
            Storage.downgrade(self.settings, v2_identity(self.settings), 3)
        self.assertEqual(self.snapshot(), after)

    def test_an_interrupted_upgrade_to_4_is_completed_by_the_next_opening(self):
        self.make(3)
        before = self.snapshot()
        real = storage.migration

        def broken(name):
            if name == "004":
                return real(name).replace("PRAGMA user_version = 4;", "SELECT no_such_function();")
            return real(name)
        with patch.object(storage, "migration", side_effect=broken):
            with self.assertRaises(sqlite3.OperationalError):
                self.open()
        self.assertEqual(self.snapshot(), before)                    # one transaction: nothing of 004 stays
        self.open()
        self.assertEqual(self.snapshot()[1:3], (4, fresh("001", "002", "003", "004")))

    def test_the_binary_of_schema_3_needs_the_return(self):
        """fe6b952 accepts the database schemas 0-3 only (`Unsupported database schema`); a database of schema 4 is
        refused by it, so the guide gives `--to-schema 3` to go back to that binary."""
        self.make(3)
        self.open()
        conn = sqlite3.connect(self.root / "archive.sqlite3")
        version = conn.execute("PRAGMA user_version").fetchone()[0]
        conn.close()
        self.assertNotIn(version, (0, 1, 2, 3))


if __name__ == "__main__":
    unittest.main()
