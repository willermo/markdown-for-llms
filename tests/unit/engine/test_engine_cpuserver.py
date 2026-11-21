"""The local CPU server of the `marker-server` route (MFL033, plan r2 §7): preparation and checks of a
release, the route on the `cpu` endpoint, SIGTERM in the middle of a conversion. The suite has no
GGUF weights: a fake `llama-server` (a Python program of this file) is packed into a fake release
with its own manifest, and runs in the real sandbox and scope; the real build is proved in N3 with
`mdf prepare --cpu-server` and the measurements."""
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile
import time
import unittest
from types import SimpleNamespace

from engine_test_support import SCAN, make_pdf, scan_of
from markdown_for_llms.engine import cpuserver, isolation, servers
from markdown_for_llms.engine.codes import Code, EngineError

FAKE_LLAMA = r'''#!/usr/bin/python3 -I
import json, os, socketserver, sys, time
from http.server import BaseHTTPRequestHandler
args = sys.argv[1:]
if "--help" in args:
    print("--host HOST   IP addresses to listen on, comma-separated, or UNIX socket paths\n      ending in .sock")
    raise SystemExit(0)
host, alias = args[args.index("--host") + 1], args[args.index("--alias") + 1]
delay = float(args[args.index("--fake-delay") + 1])
FULL = '<div data-bbox="60 60 940 160" data-label="Text"><p>Page of the fake CPU server</p></div>'
class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    def log_message(self, *a):
        pass
    def send(self, status, body):
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)
    def do_GET(self):
        if self.path == "/v1/models":
            return self.send(200, {"data": [{"id": alias}]})
        if self.path == "/props":
            return self.send(200, {"build_info": "b12040-test"})
        return self.send(404, {})
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        time.sleep(delay)
        self.send(200, {"id": "f", "object": "chat.completion", "created": 0, "model": body.get("model"),
                        "choices": [{"index": 0, "message": {"role": "assistant", "content": FULL},
                                     "finish_reason": "stop", "logprobs": None}],
                        "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}})
class S(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = True
os.umask(0o077)
S(host, H).serve_forever()
'''


def sha(data):
    return hashlib.sha256(data).hexdigest()


def model_root():
    root = os.environ.get("FULL_CHECK_MODEL_ROOT")
    if not root:
        raise AssertionError("FULL_CHECK_MODEL_ROOT is not set: run the suite from scripts/full-check.sh")
    return root


class FakeRelease(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.sources = self.base / "sources"
        self.sources.mkdir()
        self.root = self.base / "cpu-server"

    def release(self, delay=0, extra_member=None, help_text=True):
        program = FAKE_LLAMA if help_text else FAKE_LLAMA.replace("UNIX socket paths", "addresses").replace(".sock", "")
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tar:
            for name, data, mode in (("llama-b12040/llama-server", program.encode(), 0o755),
                                     ("llama-b12040/LICENSE", b"MIT\n", 0o644)):
                info = tarfile.TarInfo(name)
                info.size, info.mode = len(data), mode
                tar.addfile(info, io.BytesIO(data))
            if extra_member is not None:
                tar.addfile(extra_member)
        files = {"fake.tar.gz": buf.getvalue(), "surya-2.5.gguf": b"GGUF weights" * 10, "surya-2.5-mmproj.gguf": b"GGUF mm"}
        for name, data in files.items():
            (self.sources / name).write_bytes(data)
        m = {"schema": "mdf-cpu-server/1",
             "llama_server": {"release": "b12040", "path": "fake.tar.gz", "url": "test:fake.tar.gz",
                              "bytes": len(files["fake.tar.gz"]), "sha256": sha(files["fake.tar.gz"])},
             "gguf": {"repository": "test/fake", "revision": "0" * 40, "files": [
                 {"path": n, "url": "test:" + n, "bytes": len(files[n]), "sha256": sha(files[n])}
                 for n in ("surya-2.5.gguf", "surya-2.5-mmproj.gguf")]},
             "arguments": ["--jinja", "--alias", "datalab-to/surya-ocr-2.5", "--threads", "1", "--fake-delay", str(delay)],
             "memory_mib": 512}
        (self.base / "manifest.json").write_text(json.dumps(m))
        return m

    def fetch(self, url, partial):
        shutil.copyfile(self.sources / url.removeprefix("test:"), partial)
        return SimpleNamespace(returncode=0)


class TestPrepare(FakeRelease):
    def test_prepare_verify_and_the_socket_of_the_build(self):
        m = self.release()
        record = cpuserver.prepare(self.root, fetch=self.fetch, log=lambda line: None, m=m)
        self.assertEqual((record["downloaded"], record["kept"], record["unix_socket"]), (3, 0, True))
        self.assertEqual(record["binary"], "llama-b12040/llama-server")
        again = cpuserver.prepare(self.root, fetch=self.fetch, log=lambda line: None, m=m)
        self.assertEqual((again["downloaded"], again["kept"]), (0, 3))
        tree = self.root / "llama-b12040" / "llama-b12040"
        (tree / "LICENSE").write_text("changed\n")
        with self.assertRaises(EngineError) as caught:
            cpuserver.verify(self.root, m)
        self.assertEqual(caught.exception.code, Code.SERVER_NOT_CONFIGURED)
        (tree / "LICENSE").write_text("MIT\n")
        cpuserver.verify(self.root, m)
        (self.root / "gguf" / "surya-2.5.gguf").write_bytes(b"other")
        with self.assertRaises(EngineError) as caught:           # present and different: left untouched
            cpuserver.prepare(self.root, fetch=self.fetch, log=lambda line: None, m=m)
        self.assertEqual(caught.exception.code, Code.SERVER_NOT_CONFIGURED)
        self.assertEqual((self.root / "gguf" / "surya-2.5.gguf").read_bytes(), b"other")

    def test_a_build_without_unix_sockets_and_an_archive_with_a_link_outside(self):
        m = self.release(help_text=False)
        self.assertFalse(cpuserver.prepare(self.root, fetch=self.fetch, log=lambda line: None, m=m)["unix_socket"])
        shutil.rmtree(self.root)
        link = tarfile.TarInfo("llama-b12040/escape")
        link.type, link.linkname = tarfile.SYMTYPE, "../../outside"
        m = self.release(extra_member=link)
        with self.assertRaises(EngineError):
            cpuserver.prepare(self.root, fetch=self.fetch, log=lambda line: None, m=m)
        self.assertFalse((self.root / "llama-b12040").exists())


DRIVER = Path(__file__).resolve().parent / "cpu_sigterm_driver.py"


class TestRoute(FakeRelease):
    """The relay connects to the socket of the CPU server: the conversion runs in a subprocess (the
    suite forbids connections in its own process)."""

    def driver(self, attempt):
        src = self.base / "scan.pdf"
        src.write_bytes(make_pdf([{"content": SCAN, "image": scan_of(["Page of the fake CPU server"], scale=1.5)}]))
        return [sys.executable, "-I", "-B", str(DRIVER), str(src), str(self.base / ("out-" + attempt)),
                str(self.base / ("work-" + attempt)), model_root(), str(self.root), str(self.base / "manifest.json"), attempt]

    def convert(self, attempt, m):
        subprocess.run(self.driver(attempt), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=600)
        return json.loads((self.base / ("out-" + attempt) / "engine.json").read_text())

    def test_the_cpu_endpoint_runs_its_server_in_a_scope_and_stops_it(self):
        m = self.release()
        cpuserver.prepare(self.root, fetch=self.fetch, log=lambda line: None, m=m)
        result = self.convert("cpusrv", m)
        self.assertEqual(result["status"], "complete", result.get("error"))
        self.assertIn("Page of the fake CPU server", (self.base / "out-cpusrv" / "document.md").read_text())
        cpu = result["components"]["server"]["cpu"]
        self.assertEqual((cpu["release"], cpu["run"]["scope"], cpu["run"]["scope_after"]),
                         ("b12040", "mdf-eng-cpusrv-llama.scope", "absent"))
        self.assertEqual(result["components"]["server"]["versions"], {"llamacpp": "b12040-test"})
        self.assertEqual(result["components"]["server"]["endpoint"]["pins"], {"llama.cpp": "b12040"})
        self.assertTrue(any(s["tool"] == "bwrap:llama-server" for s in result["route"]["steps"]))
        self.assertEqual(result["marker_server"]["requests"]["chat"], 1)
        self.assertEqual(isolation.list_scopes("mdf-eng-cpusrv-"), [])

    def test_a_server_not_prepared_is_refused_before_anything_starts(self):
        m = self.release()
        result = self.convert("cpunone", m)
        self.assertEqual((result["status"], result["error"]["code"]), ("failed", Code.SERVER_NOT_CONFIGURED.value))

    def test_sigterm_in_the_middle_leaves_no_scope_and_no_socket_directory(self):
        """Plan r2 §7: SIGTERM is SIGINT, so the `finally` of the engine stops the CPU server, the
        Marker child, their scopes and the private socket directory."""
        m = self.release(delay=120)
        cpuserver.prepare(self.root, fetch=self.fetch, log=lambda line: None, m=m)
        runtime = Path(os.environ.get("XDG_RUNTIME_DIR") or "/tmp")
        before = set(runtime.glob("mdf-*"))
        proc = subprocess.Popen(self.driver("cputerm"), stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        self.addCleanup(lambda: proc.poll() is None and proc.kill())
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline:
            scopes = isolation.list_scopes("mdf-eng-cputerm-")
            if any(s.endswith("-llama.scope") for s in scopes) and any("-marker-" in s for s in scopes):
                break
            self.assertIsNone(proc.poll(), proc.stderr.read().decode("utf-8", "replace")[-2000:] if proc.poll() else "")
            time.sleep(0.5)
        else:
            self.fail("the CPU server and the Marker child never ran together")
        time.sleep(3)
        proc.send_signal(signal.SIGTERM)
        self.assertEqual(proc.wait(timeout=90), 130)
        for _ in range(20):
            if not isolation.list_scopes("mdf-eng-cputerm-"):
                break
            time.sleep(0.5)
        self.assertEqual(isolation.list_scopes("mdf-eng-cputerm-"), [])
        self.assertEqual(set(runtime.glob("mdf-*")) - before, set())
        result = json.loads((self.base / "out-cputerm" / "engine.json").read_text())
        self.assertEqual(result["error"]["code"], Code.INTERRUPTED.value)


if __name__ == "__main__":
    unittest.main()
