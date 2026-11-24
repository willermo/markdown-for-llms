"""`runpod_endpoint.py delete` removes the template of the endpoint too (MFL023 B7, MFL049 (e)): on 23/11 an endpoint
created again with the same name was refused (HTTP 400) for hours while the template of the old one lived. A template the
key may not delete (403) is a declared outcome, exit 0; another failure is exit 4 with closed fields. Fakes only (D11:
no resource is created or deleted in this run)."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

import httpx

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts" / "servers"))
import providers as pv  # noqa: E402
import runpod_endpoint as rpe  # noqa: E402

KEY = "rk-b7-delete-0123"


class Fake:
    def __init__(self, template="tpl1", template_status=204):
        self.ep = {"id": "ep1", "name": "mdf-surya", "workers": pv.workers_block(0)}
        if template:
            self.ep["templateId"] = template
        self.template_status, self.calls, self.exists = template_status, [], True

    def __call__(self, request):
        path, method = request.url.path.split("/v2", 1)[1], request.method
        self.calls.append((method, path))
        if path == "/serverless" and method == "GET":
            return httpx.Response(200, json={"endpoints": [self.ep] if self.exists else [], "pagination": {}})
        if path == "/serverless/ep1" and method == "GET":
            return httpx.Response(200, json=self.ep)
        if path == "/serverless/ep1/workers":
            return httpx.Response(200, json={"workers": [], "summary": {"total": 0}})
        if path == "/serverless/ep1" and method == "DELETE":
            self.exists = False
            return httpx.Response(204)
        if path.startswith("/templates/") and method == "DELETE":
            return httpx.Response(self.template_status, json={"error": "refused " + KEY} if self.template_status >= 400 else None)
        return httpx.Response(404, json={})


class TestDelete(unittest.TestCase):
    def run_delete(self, fake):
        d = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, d, True)
        env = d / ".env"
        env.write_text("RUNPOD_API_KEY=%s\n" % KEY)
        env.chmod(0o600)
        out = io.StringIO()
        with redirect_stdout(out):
            code = rpe.main(["--env-file", str(env), "delete", "--name", "mdf-surya"], transport=httpx.MockTransport(fake))
        return code, json.loads(out.getvalue().splitlines()[-1]), out.getvalue()

    def test_the_endpoint_and_its_template(self):
        fake = Fake()
        code, result, _ = self.run_delete(fake)
        self.assertEqual((code, result["template"]), (0, "deleted"))
        self.assertEqual([c for c in fake.calls if c[0] == "DELETE"], [("DELETE", "/serverless/ep1"),
                                                                      ("DELETE", "/templates/tpl1")])

    def test_a_template_the_key_may_not_delete(self):
        code, result, text = self.run_delete(Fake(template_status=403))
        self.assertEqual((code, result["template"]), (0, "forbidden"))
        self.assertNotIn(KEY, text)

    def test_another_failure_of_the_template(self):
        code, result, text = self.run_delete(Fake(template_status=500))
        self.assertEqual(code, 4)
        self.assertEqual((result["template"]["outcome"], result["template"]["http"], result["template"]["op"]),
                         ("failed", 500, "runpod.delete"))
        self.assertNotIn(KEY, text)

    def test_an_endpoint_without_a_template(self):
        fake = Fake(template=None)
        code, result, _ = self.run_delete(fake)
        self.assertEqual((code, result["template"]), (0, None))
        self.assertEqual([c for c in fake.calls if c[0] == "DELETE"], [("DELETE", "/serverless/ep1")])


if __name__ == "__main__":
    unittest.main()
