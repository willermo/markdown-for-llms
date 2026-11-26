"""`routestate.py`: the state of every route from what the worker checked (MFL051, plan r2.1 §5.4, J.5.4).

A pure module: the table below is the whole of it. For each route one state, in the order in which they
prevail: `broken` (a check fails), `not_configured` (something is missing: `missing` names it, says where to
write it and of which kind, never a value), `disabled` (everything is there and a cap is 0), `available`. The
codes are the closed ones of today. `transit` is a requirement of the route, never a confirmation.
"""
import json
import unittest

from markdown_for_llms import routestate
from markdown_for_llms.engine import routes
from markdown_for_llms.engine.codes import Code
from markdown_for_llms.gpu import codes as gpu_codes

OK = {"available": True, "code": None}
LOCAL = {name: OK for name in routestate.LOCAL}
SECRET = "synthetic-key-0123456789"


def no(code):
    return {"available": False, "code": code}


def by_id(computed):
    return {item["id"]: item for item in computed["routes"]}


class TableTests(unittest.TestCase):
    # code of the check -> the state of the route
    TABLE = {
        None: "available",
        # a check fails: something has to be repaired
        "isolation_unavailable": "broken", "toolchain_mismatch": "broken", "weights_mismatch": "broken",
        "engine_adapter_mismatch": "broken", "datalab_key_unsafe": "broken", "server_key_unsafe": "broken",
        "session_orphaned": "broken", "gpu_unrecoverable": "broken", "server_mismatch": "broken",
        "endpoint_out_of_pins": "broken", "provider_not_idle": "broken", "gpu_preset_unknown": "broken",
        "gpu_preset_superseded": "broken", "gpu_volume_unsupported": "broken", "a_code_of_tomorrow": "broken",
        # something is missing
        "toolchain_missing": "not_configured", "weights_missing": "not_configured", "datalab_key_missing": "not_configured",
        "server_key_missing": "not_configured", "server_not_configured": "not_configured",
        "gpu_preset_not_enabled": "not_configured",
        # everything is there and a cap is 0
        "datalab_cap": "disabled", "server_cap": "disabled",
    }

    def test_every_code_has_one_state(self):
        for code, state in self.TABLE.items():
            with self.subTest(code=code):
                check = OK if code is None else no(code)
                self.assertEqual(routestate.state_of(check), (state, code))
        self.assertEqual(set(self.TABLE.values()), set(routestate.STATES))

    def test_the_codes_are_the_closed_ones_of_today(self):
        known = {c.value for c in Code} | set(gpu_codes.CODES)
        for code in list(routestate.MISSING) + list(routestate.DISABLED_BY):
            self.assertIn(code, known)
        for code in self.TABLE:
            if code not in (None, "a_code_of_tomorrow"):
                self.assertIn(code, known, code)

    def test_a_check_that_is_not_one_is_broken(self):
        for bad in (None, "ready", [], {"available": "yes"}, {"available": False}, {"available": False, "code": 7}):
            with self.subTest(bad=bad):
                state, code = routestate.state_of(bad)
                self.assertEqual(state, "broken")
                self.assertIsInstance(code, str)

    def test_what_is_missing_is_named_with_where_and_kind_and_never_a_value(self):
        self.assertEqual(routestate.missing_of("weights_missing"),
                         [{"name": "the weights of the Marker route", "where": "make prepare", "kind": "command"}])
        self.assertEqual(routestate.missing_of("toolchain_missing")[0]["where"], "make prepare")
        self.assertEqual(routestate.missing_of("datalab_key_missing"), [{"name": "DATALAB_API_KEY", "where": ".env", "kind": "key"}])
        self.assertEqual(routestate.missing_of("server_not_configured")[0]["kind"], "servers.json")
        self.assertEqual(routestate.missing_of("gpu_preset_not_enabled"), [{"name": "MDF_GPU_PRESETS", "where": ".env", "kind": "setting"}])
        self.assertEqual(routestate.missing_of("server_key_missing", ["RUNPOD_API_KEY", "MY_GPU_KEY"]),
                         [{"name": "RUNPOD_API_KEY", "where": ".env", "kind": "key"}, {"name": "MY_GPU_KEY", "where": ".env", "kind": "key"}])
        self.assertEqual(len(routestate.missing_of("server_key_missing")), 1)       # the names not known: one generic entry
        self.assertEqual(routestate.missing_of("weights_mismatch"), [])
        for code in routestate.MISSING:
            for gap in routestate.missing_of(code, ["A_KEY"]):
                self.assertEqual(set(gap), {"name", "where", "kind"})
                self.assertIn(gap["kind"], ("key", "setting", "servers.json", "command"))


class ComputeTests(unittest.TestCase):
    GPU = {"presets": {"runpod-serverless-volume-dc": dict(OK, minimum_reserve_cents="12", a_cents="100", w_s=600, open_session=False),
                       "modal-l4": dict(no("server_key_missing"), minimum_reserve_cents=None, a_cents=None, w_s=None,
                                        open_session=False, missing_keys=["MODAL_PROXY_TOKEN_ID"])},
           "ledgers": {"runpod": {"total_cents": "30", "cap_cents": "200", "orphans": 0}, "modal": {"error": "ledger_unreadable"}}}
    PROVIDERS = {"runpod-serverless-volume-dc": "runpod", "modal-l4": "modal"}

    def compute(self, **changes):
        converters = dict(LOCAL, datalab=OK)
        converters.update(changes)
        return routestate.compute(converters, "0.05", self.GPU, self.PROVIDERS,
                                  key_names={"modal-l4": ["MODAL_PROXY_TOKEN_ID"]}, now=1795680000.5)

    def test_the_routes_of_an_instance_in_the_order_of_the_matrix(self):
        computed = self.compute()
        self.assertEqual((computed["schema"], computed["computed_at"]), ("mdf-routes/1", 1795680000.5))
        self.assertEqual([item["id"] for item in computed["routes"]],
                         ["text", "pandoc", "libreoffice", "djvutxt", "marker", "datalab", "marker-server:modal-l4",
                          "marker-server:runpod-serverless-volume-dc"])
        self.assertEqual([item["id"] for item in computed["routes"]][:6], [c for c in routes.CONVERTERS if c != routes.MARKER_SERVER])
        for item in computed["routes"]:
            self.assertEqual(set(item), {"id", "converter", "provider", "gpu_preset", "state", "code", "detail", "missing",
                                         "transit", "spend", "caps", "preset", "ledger", "live"})
        json.dumps(computed, allow_nan=False)                           # it goes in the heartbeat

    def test_a_local_route_sends_nothing_and_spends_nothing(self):
        item = by_id(self.compute())["marker"]
        self.assertEqual(item, {"id": "marker", "converter": "marker", "provider": None, "gpu_preset": None,
                                "state": "available", "code": None, "detail": None, "missing": [],
                                "transit": {"required": False, "what": None}, "spend": "none", "caps": None,
                                "preset": None, "ledger": None, "live": None})

    def test_datalab(self):
        item = by_id(self.compute())["datalab"]
        self.assertEqual((item["state"], item["provider"], item["transit"], item["spend"], item["caps"]),
                         ("available", "datalab", {"required": True, "what": "document"}, "capped", {"conversion_usd": "0.05"}))
        missing = by_id(self.compute(datalab=no("datalab_key_missing")))["datalab"]
        self.assertEqual((missing["state"], missing["missing"]),
                         ("not_configured", [{"name": "DATALAB_API_KEY", "where": ".env", "kind": "key"}]))
        self.assertEqual(by_id(self.compute(datalab=no("datalab_cap")))["datalab"]["state"], "disabled")
        self.assertEqual(by_id(self.compute(datalab=no("datalab_cap")))["datalab"]["missing"], [])

    def test_a_broken_route_says_what_to_do_and_only_a_broken_one(self):
        hint = "env file: replaced or removed since it was mounted; run `make restart`"
        converters = dict(LOCAL, datalab=no("datalab_key_unsafe"))
        item = by_id(routestate.compute(converters, "0.05", details={"datalab": hint, "marker": hint}))
        self.assertEqual((item["datalab"]["state"], item["datalab"]["detail"]), ("broken", hint))
        self.assertIsNone(item["marker"]["detail"])                     # available: nothing to do

    def test_a_gpu_preset_is_a_route_of_its_own_with_its_provider(self):
        item = by_id(self.compute())
        runpod, modal = item["marker-server:runpod-serverless-volume-dc"], item["marker-server:modal-l4"]
        self.assertEqual((runpod["converter"], runpod["provider"], runpod["gpu_preset"], runpod["state"]),
                         ("marker-server", "runpod", "runpod-serverless-volume-dc", "available"))
        self.assertEqual(runpod["transit"], {"required": True, "what": "page_images"})
        self.assertEqual((runpod["spend"], runpod["caps"]), ("capped", {"ledger_cents": "200"}))
        self.assertEqual(runpod["preset"], {"minimum_reserve_cents": "12", "a_cents": "100", "w_s": 600})
        self.assertEqual(runpod["ledger"], {"total_cents": "30", "cap_cents": "200", "orphans": 0, "read_at": 1795680000.5})
        self.assertEqual(runpod["live"], {"open_session": False, "read_at": 1795680000.5})
        self.assertEqual((modal["state"], modal["code"], modal["provider"]), ("not_configured", "server_key_missing", "modal"))
        self.assertEqual(modal["missing"], [{"name": "MODAL_PROXY_TOKEN_ID", "where": ".env", "kind": "key"}])
        self.assertEqual((modal["caps"], modal["ledger"]["error"]), (None, "ledger_unreadable"))

    def test_transit_is_a_requirement_never_a_confirmation(self):
        text = json.dumps(self.compute())
        for word in ("confirmed", "transit_confirmed"):
            self.assertNotIn(word, text)
        for item in self.compute()["routes"]:
            self.assertEqual(set(item["transit"]), {"required", "what"})
            self.assertIn(item["transit"]["what"], (None, "document", "page_images"))

    def test_without_gpu_presets_and_with_nothing_checked(self):
        computed = routestate.compute(None, "0")
        self.assertEqual(len(computed["routes"]), 6)
        self.assertEqual({item["state"] for item in computed["routes"]}, {"broken"})
        self.assertIsNone(computed["computed_at"])

    def test_a_worker_that_takes_no_job_has_every_route_broken_with_its_reason(self):
        why = "Isolation unavailable; no fallback (stale engine image (payload_sha256): run `make up`)"
        computed = routestate.blocked("isolation_unavailable", "0.05", ["modal-l4"], {"modal-l4": "modal"}, why, now=5.0)
        self.assertEqual([item["id"] for item in computed["routes"]],
                         ["text", "pandoc", "libreoffice", "djvutxt", "marker", "datalab", "marker-server:modal-l4"])
        for item in computed["routes"]:
            self.assertEqual((item["state"], item["code"], item["detail"], item["missing"]),
                             ("broken", "isolation_unavailable", why, []))
        self.assertEqual(by_id(computed)["marker-server:modal-l4"]["provider"], "modal")


class AnswerTests(unittest.TestCase):
    def test_the_first_start_without_a_heartbeat(self):
        self.assertEqual(routestate.answer(None, False),
                         {"schema": "mdf-routes/1", "worker": "absent", "computed_at": None, "stale": True, "routes": []})

    def test_alive_and_stale(self):
        computed = routestate.compute(dict(LOCAL, datalab=no("datalab_cap")), "0", now=7.0)
        alive = routestate.answer(computed, True)
        self.assertEqual((alive["worker"], alive["stale"], alive["computed_at"], len(alive["routes"])), ("alive", False, 7.0, 6))
        gone = routestate.answer(computed, False)                       # the last routes, marked stale
        self.assertEqual((gone["worker"], gone["stale"], gone["routes"]), ("absent", True, computed["routes"]))

    def test_a_heartbeat_of_another_version_or_without_routes(self):
        for recorded in ({}, {"schema": "mdf-routes/0", "routes": []}, {"schema": "mdf-routes/1", "routes": "none"}, "ready", 3):
            with self.subTest(recorded=recorded):
                body = routestate.answer(recorded, True)
                self.assertEqual((body["worker"], body["stale"], body["routes"], body["computed_at"]), ("alive", True, [], None))

    def test_one_line_per_route_for_a_terminal(self):
        hint = "run `make restart`"
        computed = routestate.compute(dict(LOCAL, marker=no("weights_missing"), datalab=no("datalab_key_unsafe")), "0",
                                      details={"datalab": hint}, now=1.0)
        lines = routestate.summary(routestate.answer(computed, True)).splitlines()
        self.assertEqual(lines[0], "worker alive")
        self.assertEqual(len(lines), 7)
        self.assertRegex(lines[1], r"^text\s+available$")
        self.assertRegex(lines[5], r"^marker\s+not_configured\s+weights_missing; missing the weights of the Marker route "
                                   r"\(command: make prepare\)$")
        self.assertRegex(lines[6], r"^datalab\s+broken\s+datalab_key_unsafe; run `make restart`$")
        self.assertEqual(routestate.summary(routestate.answer(None, False)), "worker absent (stale)")

    def test_the_module_is_pure(self):
        """Standard library only, nothing of the engine: the API process imports it."""
        import ast
        from pathlib import Path
        tree = ast.parse(Path(routestate.__file__).read_text())
        imported = [node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))]
        self.assertEqual(imported, [])
        self.assertNotIn(SECRET, json.dumps(routestate.compute(dict(LOCAL, datalab=OK), "0.05")))


if __name__ == "__main__":
    unittest.main()
