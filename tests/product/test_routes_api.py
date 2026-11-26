"""`GET /api/v1/routes`, `POST /api/v1/routes/refresh` and `mdf routes` (MFL051, plan r2.1 §5.4, J.5.4).

The worker computes the state of the routes (`routestate.py`) at every check of the routes and writes it in its
heartbeat; the API and the CLI read it through the same service and never compute it. The worker is the real one
(its launcher scripted, the availability of the converters given); the API is the real application.

- the shape `mdf-routes/1`; `transit` a requirement of the route, never a confirmation;
- the very first start, without a heartbeat: 200, `worker: absent`, no routes;
- a heartbeat older than the lease: the last routes, `stale`;
- `refresh`: CSRF like every write, 202, and the worker checks the routes at its next turn, not at the next minute;
- no key in any answer: only the names of what is missing;
- a worker that takes no job (its children would not run its code): every route broken, with its reason.
"""
import contextlib
import io
import json
import os
from pathlib import Path
import sqlite3
import tempfile
import time
import unittest
from unittest import mock

from fastapi.testclient import TestClient

from markdown_for_llms import cli
from markdown_for_llms.api import create_app
from markdown_for_llms.services import Service
from markdown_for_llms.settings import Settings
from markdown_for_llms.worker import Worker

from engine_fake import launcher_fake

SECRET = "dl-synthetic-0123456789"
LOCAL = {name: {"available": True, "code": None} for name in ("text", "pandoc", "libreoffice", "djvutxt")}
GPU_PRESET = "runpod-serverless-volume-dc"


class Base(unittest.TestCase):
    SETTINGS = {}

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.settings = Settings(data_root=str(self.root / "data"), model_root=str(self.root / "models"),
                                 gpu_ledger_dir=str(self.root / "ledger"), servers=str(self.root / "servers.json"),
                                 **self.SETTINGS)
        self.client = TestClient(create_app(self.settings), base_url=f"http://127.0.0.1:{self.settings.port}")
        self.headers = {"origin": f"http://127.0.0.1:{self.settings.port}",
                        "x-mdf-csrf": self.client.get("/api/v1/session").json()["csrf"]}
        self.launcher = launcher_fake.patched(self, self.root / "volume")
        self.marker = {"available": False, "code": "weights_missing"}
        self.checks = 0

        def availability(**kw):
            self.checks += 1
            return dict(LOCAL, marker=dict(self.marker), datalab={"available": False, "code": "datalab_key_missing"})
        patcher = mock.patch("markdown_for_llms.worker.availability", availability)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.keys = self.root / "env"
        self.keys.write_text("DATALAB_API_KEY=%s\nRUNPOD_API_KEY=rp-%s\n" % (SECRET, SECRET))
        self.keys.chmod(0o600)

    def worker(self, keys=None, settings=None):
        worker = Worker(settings or self.settings, env_file=keys)
        self.addCleanup(worker.storage.close)
        return worker

    def routes(self):
        response = self.client.get("/api/v1/routes")
        self.assertEqual(response.status_code, 200)
        self.assertNotIn(SECRET, response.text)
        body = response.json()
        self.assertEqual(set(body), {"schema", "worker", "computed_at", "stale", "routes"})
        return body, {item["id"]: item for item in body["routes"]}

    def db(self, statement, *args):
        with sqlite3.connect(Path(self.settings.data_root) / "archive.sqlite3") as conn:
            rows = conn.execute(statement, args).fetchall()
        conn.close()
        return rows


class RoutesTests(Base):
    def test_the_first_start_without_a_heartbeat(self):
        body, _ = self.routes()
        self.assertEqual(body, {"schema": "mdf-routes/1", "worker": "absent", "computed_at": None, "stale": True, "routes": []})

    def test_the_routes_the_worker_computed(self):
        before = time.time()
        self.worker().turn()
        body, by_id = self.routes()
        self.assertEqual((body["schema"], body["worker"], body["stale"]), ("mdf-routes/1", "alive", False))
        self.assertGreaterEqual(body["computed_at"], before)
        self.assertEqual(list(by_id), ["text", "pandoc", "libreoffice", "djvutxt", "marker", "datalab"])
        self.assertEqual({name: by_id[name]["state"] for name in LOCAL}, {name: "available" for name in LOCAL})
        self.assertEqual((by_id["marker"]["state"], by_id["marker"]["code"], by_id["marker"]["missing"]),
                         ("not_configured", "weights_missing",
                          [{"name": "the weights of the Marker route", "where": "make prepare", "kind": "command"}]))
        # no keys file: the key of Datalab is missing, by name
        self.assertEqual((by_id["datalab"]["state"], by_id["datalab"]["missing"]),
                         ("not_configured", [{"name": "DATALAB_API_KEY", "where": ".env", "kind": "key"}]))
        # the readiness of before is unchanged, and does not carry the routes twice
        readiness = self.client.get("/api/v1/readiness").json()
        self.assertEqual(set(readiness), {"ready", "worker", "converters"})

    def test_transit_is_a_requirement_of_the_route_never_a_confirmation(self):
        self.worker(self.keys).turn()
        body, by_id = self.routes()
        self.assertEqual(by_id["datalab"]["transit"], {"required": True, "what": "document"})
        self.assertEqual(by_id["text"]["transit"], {"required": False, "what": None})
        self.assertNotIn("confirmed", json.dumps(body))

    def test_datalab_from_the_keys_file_and_the_cap_and_no_key_in_the_answer(self):
        from dataclasses import replace
        self.worker(self.keys).turn()                                   # the key is there, the cap is 0
        _, by_id = self.routes()
        self.assertEqual((by_id["datalab"]["state"], by_id["datalab"]["code"], by_id["datalab"]["caps"]),
                         ("disabled", "datalab_cap", {"conversion_usd": "0"}))
        capped = self.worker(self.keys, replace(self.settings, datalab_cap_usd="0.05"))
        capped.turn()
        _, by_id = self.routes()
        self.assertEqual((by_id["datalab"]["state"], by_id["datalab"]["spend"]), ("available", "capped"))
        os.chmod(self.keys, 0o644)                                      # not safe any more: broken, with what to do
        capped._ready_at = None
        capped.turn()
        _, by_id = self.routes()
        self.assertEqual((by_id["datalab"]["state"], by_id["datalab"]["code"]), ("broken", "datalab_key_unsafe"))
        self.assertIn("0600", by_id["datalab"]["detail"])
        self.assertEqual(by_id["text"]["state"], "available")           # the local routes go on

    def test_a_heartbeat_older_than_the_lease_is_stale_with_the_last_routes(self):
        self.worker().turn()
        fresh, _ = self.routes()
        self.db("UPDATE worker SET heartbeat=?", time.time() - self.settings.lease_seconds - 1)
        body, by_id = self.routes()
        self.assertEqual((body["worker"], body["stale"], body["computed_at"]), ("absent", True, fresh["computed_at"]))
        self.assertEqual(len(by_id), 6)

    def test_refresh_needs_the_csrf_token_and_makes_the_worker_check_at_its_next_turn(self):
        worker = self.worker()
        worker.turn()
        first, by_id = self.routes()
        self.assertEqual((self.checks, by_id["marker"]["state"]), (1, "not_configured"))
        self.marker = {"available": True, "code": None}                 # `make prepare` in the meantime
        worker.turn()
        self.assertEqual(self.checks, 1)                                # held for READY_SECONDS
        self.assertEqual(self.client.post("/api/v1/routes/refresh").status_code, 403)
        self.assertEqual(self.checks, 1)
        response = self.client.post("/api/v1/routes/refresh", headers=self.headers)
        self.assertEqual((response.status_code, set(response.json())), (202, {"requested_at"}))
        worker.turn()
        self.assertEqual(self.checks, 2)
        body, by_id = self.routes()
        self.assertEqual(by_id["marker"]["state"], "available")
        self.assertGreater(body["computed_at"], first["computed_at"])
        worker.turn()                                                   # one request, one check
        self.assertEqual(self.checks, 2)
        # a request made before a worker started is not a request to that worker: it checks at its start anyway
        other = self.worker()
        other.turn()
        other.turn()
        self.assertEqual(self.checks, 3)

    def test_the_request_is_a_row_and_the_schema_does_not_change(self):
        version = self.db("PRAGMA user_version")
        tables = self.db("SELECT name, sql FROM sqlite_master ORDER BY name")
        self.client.post("/api/v1/routes/refresh", headers=self.headers)
        self.client.post("/api/v1/routes/refresh", headers=self.headers)
        self.assertEqual(len(self.db("SELECT value FROM metadata WHERE key='routes_refresh'")), 1)
        self.assertEqual((self.db("PRAGMA user_version"), self.db("SELECT name, sql FROM sqlite_master ORDER BY name")),
                         (version, tables))

    def test_a_worker_that_takes_no_job_has_every_route_broken_with_its_reason(self):
        worker = self.worker()
        self.launcher.code = {"payload_sha256": "0" * 64}               # a launcher of another build
        worker.turn()
        body, by_id = self.routes()
        self.assertEqual((body["worker"], body["stale"]), ("alive", False))
        self.assertEqual({item["state"] for item in by_id.values()}, {"broken"})
        self.assertEqual({item["code"] for item in by_id.values()}, {"isolation_unavailable"})
        self.assertIn("stale engine image (payload_sha256): run `make up`", by_id["text"]["detail"])
        self.assertIs(self.client.get("/api/v1/readiness").json()["ready"], False)


class GpuRoutesTests(Base):
    SETTINGS = {"gpu_presets": GPU_PRESET, "gpu_cap_usd_runpod": "2", "gpu_session_cap_usd": "1"}

    def test_a_gpu_preset_is_a_route_with_its_provider_and_what_is_missing(self):
        self.worker(self.keys).turn()                                   # no servers.json: the endpoint is missing
        _, by_id = self.routes()
        item = by_id["marker-server:" + GPU_PRESET]
        self.assertEqual((item["converter"], item["provider"], item["gpu_preset"]), ("marker-server", "runpod", GPU_PRESET))
        self.assertEqual(item["transit"], {"required": True, "what": "page_images"})
        self.assertEqual(item["spend"], "capped")
        self.assertEqual((item["state"], item["code"]), ("not_configured", "server_not_configured"))
        self.assertEqual(item["missing"], [{"name": "the endpoint of the GPU preset", "where": "state/config/servers.json",
                                            "kind": "servers.json"}])
        self.assertEqual(set(item["ledger"]), {"total_cents", "cap_cents", "orphans", "read_at"})
        self.assertEqual((item["ledger"]["cap_cents"], item["caps"]), ("200.00", {"ledger_cents": "200.00"}))
        self.assertEqual(set(item["live"]), {"open_session", "read_at"})

    def test_what_is_missing_prevails_over_a_cap_of_zero(self):
        """The order of the states: with a cap of 0 AND no endpoint, the route is `not_configured` (the worker
        reports the endpoint first, as before MFL051); `disabled` is a route that lacks nothing but a cap, and the
        table of test_routestate.py holds it (`server_cap`)."""
        from dataclasses import replace
        self.worker(self.keys, replace(self.settings, gpu_session_cap_usd="0")).turn()
        _, by_id = self.routes()
        item = by_id["marker-server:" + GPU_PRESET]
        self.assertEqual((item["state"], item["code"]), ("not_configured", "server_not_configured"))


class CliTests(Base):
    def run_cli(self, *argv):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = cli.main(["--data-root", self.settings.data_root, "--model-root", self.settings.model_root, *argv])
        self.assertNotIn(SECRET, out.getvalue())
        return code, out.getvalue()

    def test_mdf_routes_is_the_same_answer_one_line_per_route(self):
        code, out = self.run_cli("routes")
        self.assertEqual((code, out.strip()), (1, "worker absent (stale)"))            # nothing to read yet: exit 1
        self.worker(self.keys).turn()
        code, out = self.run_cli("routes")
        lines = out.splitlines()
        self.assertEqual((code, lines[0], len(lines)), (0, "worker alive", 7))
        self.assertRegex(lines[5], r"^marker\s+not_configured\s+weights_missing; missing the weights of the Marker route")
        code, out = self.run_cli("routes", "--json")
        body = json.loads(out)
        self.assertEqual((code, body["schema"], body["routes"]), (0, "mdf-routes/1", self.routes()[0]["routes"]))

    def test_refresh_writes_the_request_and_does_not_wait_for_a_worker_that_is_absent(self):
        started = time.monotonic()
        code, out = self.run_cli("routes", "--refresh")
        self.assertEqual((code, out.strip()), (1, "worker absent (stale)"))
        self.assertLess(time.monotonic() - started, 5)
        self.assertGreater(Service(self.settings).routes_refresh_asked(), 0)


if __name__ == "__main__":
    unittest.main()
