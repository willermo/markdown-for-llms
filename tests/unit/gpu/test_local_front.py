"""The front of a GPU of the user (MFL051 B; J.7.2, J.7.4 of the plan): the real front of `gpu/local_front.py` with a
fake vLLM behind it (`fake_vllm.py`), in a driver subprocess (`local_front_driver.py`), because the guard of
`conftest.py` forbids the sockets in the process of pytest.

- local GPU: a Unix socket 0600; the closed list of `watchdog.FORWARD` reaches vLLM, nothing else does (404
  `mdf_not_forwarded`); a body without one Content-Length, with an ambiguous framing or over the limit is refused
  before vLLM; /health is served by the front; at SIGTERM vLLM is stopped and the socket removed;
- remote server: every request but /health needs the bearer key of a private file, compared in constant time; a key
  file readable by others, or too short, stops the front before it listens;
- the arguments of vLLM are those of `watchdog.vllm_argv` without a lever, and `watchdog.py` is imported, not
  copied; without verified weights the front does not start vLLM (exit 3)."""
import ast
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from markdown_for_llms.gpu import local_front, watchdog

HERE = Path(__file__).resolve().parent
DRIVER = HERE / "local_front_driver.py"
FRONT = HERE.parents[2] / "markdown_for_llms" / "gpu" / "local_front.py"
KEY = "sk-remote-gpu-synthetic-0123456789abcdef"
CHAT = json.dumps({"model": watchdog.MODEL, "messages": [{"role": "user", "content": "x"}]})


def short_dir(test):
    # a Unix socket path is at most 107 bytes: a short folder of its own under the temporary root
    root = Path(tempfile.mkdtemp(prefix="lf-"))
    test.addCleanup(shutil.rmtree, root, True)
    return root


class Driver(unittest.TestCase):
    def drive(self, mode, requests, key=None, key_mode=0o600):
        work = short_dir(self)
        cfg = work / "cfg.json"
        cfg.write_text(json.dumps({"mode": mode, "dir": str(work), "key": key, "key_mode": key_mode,
                                   "requests": requests}))
        r = subprocess.run([sys.executable, "-I", "-B", str(DRIVER), str(cfg)], capture_output=True, text=True,
                           timeout=120)
        self.assertEqual(r.returncode, 0, r.stderr[-3000:])
        out = json.loads(r.stdout.strip().splitlines()[-1])
        out["stderr"] = r.stderr
        return out


class LocalTests(Driver):
    def test_the_closed_list_reaches_vllm_and_nothing_else(self):
        allowed = [{"method": "GET", "path": "/v1/models"}, {"method": "GET", "path": "/version"},
                   {"method": "GET", "path": "/metrics"},
                   {"method": "POST", "path": "/v1/chat/completions", "body": CHAT,
                    "headers": {"Content-Type": "application/json", "Authorization": "Bearer not-for-vllm"}}]
        refused = [{"method": "GET", "path": "/docs"}, {"method": "POST", "path": "/v1/load_lora_adapter", "body": "{}"},
                   {"method": "POST", "path": "/sleep", "body": ""}, {"method": "POST", "path": "/collective_rpc",
                                                                       "body": "{}"},
                   {"method": "DELETE", "path": "/v1/models"}, {"method": "GET", "path": "/v1/models/../docs"}]
        out = self.drive("local", allowed + refused)
        self.assertTrue(out["ready"], out["stderr"][-2000:])
        self.assertEqual(out["socket_mode"], "0o600")
        answers = out["answers"]
        self.assertEqual([a["status"] for a in answers[:4]], [200] * 4)
        self.assertEqual({a["proxy"] for a in answers[:4]}, {"forwarded"})
        self.assertEqual(answers[0]["body"]["data"][0]["id"], watchdog.MODEL)
        for a in answers[4:]:
            self.assertEqual((a["status"], a["body"]), (404, {"error": "mdf_not_forwarded"}))
        reached = [(r["method"], r["path"]) for r in out["reached"]]
        self.assertNotIn(("GET", "/docs"), reached)
        self.assertEqual({p for _, p in reached} - {"/v1/models", "/version", "/metrics", "/v1/chat/completions"}, set())
        # the incoming headers are dropped: no Authorization reaches vLLM
        self.assertEqual({r["authorization"] for r in out["reached"]}, {None})
        self.assertEqual(out["exit"], 0)
        self.assertFalse(out["socket_left"])

    def test_the_framing_of_a_body(self):
        out = self.drive("local", [
            {"method": "POST", "path": "/v1/chat/completions"},                                   # no length
            {"method": "POST", "path": "/v1/chat/completions", "length": watchdog.BODY_LIMIT + 1},
            {"method": "POST", "path": "/v1/chat/completions", "headers": {"Transfer-Encoding": "chunked"}},
            {"method": "POST", "path": "/v1/chat/completions", "length": -1}])
        self.assertEqual([a["status"] for a in out["answers"]], [411, 413, 411, 400])
        self.assertEqual([r for r in out["reached"] if r["path"] == "/v1/chat/completions"], [])

    def test_health_is_served_by_the_front(self):
        out = self.drive("local", [{"method": "GET", "path": "/health"}, {"method": "GET", "path": "/ping"}])
        self.assertEqual(out["answers"][0]["status"], 200)
        self.assertEqual(out["answers"][0]["proxy"], "health")
        self.assertEqual(out["answers"][1]["status"], 404)
        self.assertNotIn("/health", [r["path"] for r in out["reached"]])


class RemoteTests(Driver):
    def test_every_request_needs_the_key(self):
        out = self.drive("remote", [
            {"method": "GET", "path": "/v1/models"},
            {"method": "GET", "path": "/v1/models", "headers": {"Authorization": "Bearer wrong-" + KEY}},
            {"method": "GET", "path": "/v1/models", "headers": {"Authorization": KEY}},
            {"method": "GET", "path": "/v1/models", "headers": {"Authorization": "Bearer " + KEY}},
            {"method": "GET", "path": "/docs", "headers": {"Authorization": "Bearer " + KEY}},
            {"method": "GET", "path": "/health"}], key=KEY)
        self.assertEqual([a["status"] for a in out["answers"]], [401, 401, 401, 200, 404, 200])
        self.assertEqual(out["answers"][0]["body"], {"error": "mdf_unauthorized"})
        self.assertEqual([r["path"] for r in out["reached"] if r["path"] != "/v1/models" or r["authorization"]], [])
        self.assertNotIn(KEY, out["stderr"])

    def test_a_key_file_that_is_not_private_or_too_short(self):
        out = self.drive("remote", [], key=KEY, key_mode=0o644)
        self.assertIn("SystemExit", str(out["exit"]))
        self.assertIn("not a private regular file", str(out["exit"]))
        out = self.drive("remote", [], key="short")
        self.assertIn("SystemExit", str(out["exit"]))
        self.assertNotIn("short", str(out["exit"]).replace("characters", ""))
        self.assertEqual(out["reached"], [])

    def test_the_key_is_compared_in_constant_time(self):
        tree = ast.parse(FRONT.read_text(encoding="utf-8"))
        calls = {node.func.attr for node in ast.walk(tree) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Attribute)}
        self.assertIn("compare_digest", calls)
        state = local_front.State(key=KEY)
        self.assertEqual([state.authorized(h) for h in (None, "", "Bearer", "Bearer " + KEY, "bearer " + KEY)],
                         [False, False, False, True, False])
        self.assertTrue(local_front.State().authorized(None))


class StartTests(unittest.TestCase):
    def test_the_arguments_of_vllm_are_those_of_the_watchdog(self):
        hf = short_dir(self)
        snap = watchdog.snapshot(hf)
        snap.mkdir(parents=True)
        (snap / "config.json").write_text(json.dumps({"model_type": "qwen2_5_vl"}))
        self.assertEqual(local_front.vllm_command(str(hf)),
                         watchdog.vllm_argv({"levers": (), "upstream": local_front.UPSTREAM_PORT}, False))
        self.assertIn("--host", local_front.vllm_command(str(hf)))
        self.assertEqual(local_front.vllm_command(str(hf))[local_front.vllm_command(str(hf)).index("--host") + 1],
                         "127.0.0.1")

    def test_without_the_verified_weights_vllm_does_not_start(self):
        work = short_dir(self)
        self.assertEqual(local_front.main(["serve", "--socket", str(work / "s.sock"), "--hf-home", str(work / "hf")]), 3)
        self.assertFalse((work / "s.sock").exists())

    def test_the_watchdog_is_imported_not_copied(self):
        tree = ast.parse(FRONT.read_text(encoding="utf-8"))
        names = {node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
        self.assertFalse(names & {"vllm_argv", "weights_ok", "mtp_head", "download", "forwarded"})
        self.assertIs(local_front.wd, watchdog)
        # a socket path too long for AF_UNIX is refused before anything listens
        with self.assertRaises(SystemExit):
            local_front.socket_path("/" + "x" * 120)


if __name__ == "__main__":
    unittest.main()
