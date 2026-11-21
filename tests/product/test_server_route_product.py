"""The product refuses the `marker-server` route until MFL023 (MFL033, P9 of plan r2): it does not
list it, does not accept it, and its readiness never contacts an inference server."""
import socket
import sys
import tempfile
import unittest

from markdown_for_llms.engine import routes
from markdown_for_llms.engine.codes import Code, EngineError
from markdown_for_llms.settings import Settings
from markdown_for_llms.worker import Worker

CONNECTS = []
WATCH = [False]


def _audit(event, args):
    if WATCH[0] and event in ("socket.connect", "socket.getaddrinfo", "http.client.connect"):
        CONNECTS.append((event, repr(args[1:2])[:80]))


sys.addaudithook(_audit)


class TestServerRouteInTheProduct(unittest.TestCase):
    def test_the_product_view_of_the_matrix(self):
        for fmt in routes.MATRIX:
            self.assertNotIn(routes.MARKER_SERVER, routes.routes_for(fmt)["alternatives"])
        for fmt in ("pdf", "png", "jpeg", "tiff", "djvu"):
            self.assertIn(routes.MARKER_SERVER, routes.engine_routes_for(fmt)["alternatives"])
        self.assertEqual(routes.routes_for("pdf"), {"default": "marker", "alternatives": ["datalab"]})

    def test_job_options_refuse_it(self):
        for options in ({"route": "marker-server"}, {"route": "marker-server", "ocr": "force"}):
            with self.assertRaises(EngineError) as caught:
                routes.job_options("pdf", options)
            self.assertEqual(caught.exception.code, Code.ROUTE_NOT_IN_PRODUCT)

    def test_the_refusal_has_a_message_of_its_own(self):
        """G-CNB4 of the final review of MFL033: the API returns a text next to the code, not the generic one."""
        from markdown_for_llms import services
        message = str(services.Refused(Code.ROUTE_NOT_IN_PRODUCT.value))
        self.assertNotEqual(message, "Request refused")
        self.assertIn("mdf convert and mdf bench only", message)

    def test_the_readiness_of_the_worker_contacts_no_server(self):
        with tempfile.TemporaryDirectory() as tmp:
            worker = Worker(Settings(data_root=tmp + "/data", model_root=tmp + "/models"))
            self.addCleanup(worker.storage.close)
            CONNECTS.clear()
            WATCH[0] = True
            try:
                state = worker.ready()
            finally:
                WATCH[0] = False
        self.assertNotIn(routes.MARKER_SERVER, state)
        self.assertEqual([c for c in CONNECTS if "AF_UNIX" not in c[1]], [])


if __name__ == "__main__":
    unittest.main()
