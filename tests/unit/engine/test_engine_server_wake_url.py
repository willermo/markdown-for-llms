"""The canonical URL of the endpoint in the event `server-wake` of the `marker-server` route (MFL023 B2, C-B2 of the
plan r1): the reservation of the caller compares it with the URL of the window it draws from. The route stops at the
event (the observer refuses, as `TestBeforeTheWake` of `test_engine_server_route.py` does), so no request reaches the
fake server; the weights come from `FULL_CHECK_MODEL_ROOT`, as for the whole route."""
import time

from test_engine_server_route import RouteCase, runpod, scanned
from markdown_for_llms.engine.codes import Code, EngineError


class TestTheEventOfTheWake(RouteCase):
    def test_the_event_names_the_canonical_url(self):
        seen = []

        def observer(event):
            if event.get("step") == "server-wake":
                seen.append(event["reserve"])
                raise EngineError(Code.SERVER_CAP, "stop at the wake")
        far = {"arm_id": "a", "armed_until": time.time() + 7200}
        result, _, fake = self.run_route([scanned()], "balanced", ep=runpod(url="http://127.0.0.1:9/"), observer=observer,
                                         server={"transit_confirmed": True, "budget_cents": "60", "session": far})
        self.assertEqual((result["status"], result["error"]["code"]), ("failed", "server_cap"))
        self.assertEqual(seen[0]["url"], "http://127.0.0.1:9")
        self.assertEqual(fake.seen, [])
