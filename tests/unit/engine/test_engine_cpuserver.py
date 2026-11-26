"""The local CPU server of the `marker-server` route (MFL033, plan r2 §7): preparation and checks of a
release, the route on the `cpu` endpoint, SIGTERM in the middle of a conversion. The suite has no
GGUF weights: a fake `llama-server` (a Python program of this file) is packed into a fake release
with its own manifest, prepared in the volume of the CPU server of the tests, and runs as a real
`cpuserver` child of the launcher; the real build was proved in N3 of MFL033 with
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
from markdown_for_llms.engine import containers, cpuserver, servers
from markdown_for_llms.engine.codes import Code, EngineError

FAKE_LLAMA = r'''#!/opt/mdf/venv/bin/python -I
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


MARK = ".mdf-test-volume"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def model_root():
    root = os.environ.get("FULL_CHECK_MODEL_ROOT")
    if not root:
        raise AssertionError("FULL_CHECK_MODEL_ROOT is not set: run the suite with `make test`")
    return root


class FakeRelease(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.sources = self.base / "sources"
        self.sources.mkdir()
        # the launcher mounts `llama/` and `gguf/` of the volume of the CPU server in the child: the fake release is
        # prepared there. The volume of a check is of that check; one that holds anything else is not touched.
        self.root = Path(containers.CPU_SERVER_ROOT)
        if not self.root.is_dir():
            self.fail("no volume of the CPU server at %s: run the suite with `make test`" % self.root)
        if os.listdir(self.root) and not (self.root / MARK).exists():
            self.fail("%s holds a CPU server that is not of the tests: nothing is removed" % self.root)
        self.clear()
        (self.root / MARK).write_text("the fake release of tests/unit/engine/test_engine_cpuserver.py\n")
        self.addCleanup(self.clear)

    def clear(self):
        for entry in self.root.iterdir():
            if entry.is_dir() and not entry.is_symlink():
                shutil.rmtree(entry)
            else:
                entry.unlink()

    def children(self, attempt):
        return containers.launcher().list(containers.prefix_of(attempt))

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
        tree = self.root / "llama" / "llama-b12040"
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
        self.clear()
        link = tarfile.TarInfo("llama-b12040/escape")
        link.type, link.linkname = tarfile.SYMTYPE, "../../outside"
        m = self.release(extra_member=link)
        with self.assertRaises(EngineError):
            cpuserver.prepare(self.root, fetch=self.fetch, log=lambda line: None, m=m)
        self.assertFalse((self.root / "llama").exists())

    def test_a_root_that_is_not_the_volume_or_another_release_is_refused(self):
        """The child gets `llama/` and `gguf/` of the volume, whatever the caller names: a prepared copy elsewhere
        is refused, and so is a tree extracted from another release."""
        m = self.release()
        cpuserver.prepare(self.root, fetch=self.fetch, log=lambda line: None, m=m)
        copy = self.base / "copy"
        shutil.copytree(self.root, copy)
        for call in (lambda: cpuserver.verify(copy, m), lambda: cpuserver.prepare(copy, fetch=self.fetch, log=lambda line: None, m=m),
                     lambda: cpuserver.start(copy, "elsewhere", self.base / "log", m)):
            with self.assertRaises(EngineError) as caught:
                call()
            self.assertEqual(caught.exception.code, Code.SERVER_NOT_CONFIGURED)
        other = json.loads(json.dumps(m))
        other["llama_server"]["release"] = "b99999"
        with self.assertRaises(EngineError) as caught:
            cpuserver.verify(self.root, other)
        self.assertEqual(caught.exception.code, Code.SERVER_NOT_CONFIGURED)
        self.assertEqual(self.children("elsewhere"), [])


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

    def test_the_cpu_endpoint_runs_its_server_in_a_container_and_stops_it(self):
        m = self.release()
        cpuserver.prepare(self.root, fetch=self.fetch, log=lambda line: None, m=m)
        result = self.convert("cpusrv", m)
        self.assertEqual(result["status"], "complete", result.get("error"))
        self.assertIn("Page of the fake CPU server", (self.base / "out-cpusrv" / "document.md").read_text())
        cpu = result["components"]["server"]["cpu"]
        self.assertEqual((cpu["release"], cpu["run"]["cause"]), ("b12040", "stopped"))      # ended by its caller, attested
        self.assertRegex(cpu["run"]["container"], r"^mdf-eng-cpusrv-cpuserver-\d+$")
        self.assertEqual(result["components"]["server"]["versions"], {"llamacpp": "b12040-test"})
        self.assertEqual(result["components"]["server"]["endpoint"]["pins"], {"llama.cpp": "b12040"})
        self.assertTrue(any(s["tool"] == "child:llama-server" for s in result["route"]["steps"]))
        self.assertEqual(result["marker_server"]["requests"]["chat"], 1)
        self.assertEqual(self.children("cpusrv"), [])
        self.assertFalse((Path(containers.RUN_ROOT) / "sock" / containers.prefix_of("cpusrv")).exists())

    def test_a_server_not_prepared_is_refused_before_anything_starts(self):
        m = self.release()
        result = self.convert("cpunone", m)
        self.assertEqual((result["status"], result["error"]["code"]), ("failed", Code.SERVER_NOT_CONFIGURED.value))

    def test_sigterm_in_the_middle_leaves_no_container_and_no_socket_directory(self):
        """Plan r2 §7: SIGTERM is SIGINT, so the engine, on its way out, ends the CPU server and the Marker
        child, attests that their containers are gone and removes the socket directory of the attempt."""
        m = self.release(delay=120)
        cpuserver.prepare(self.root, fetch=self.fetch, log=lambda line: None, m=m)
        sockets = Path(containers.RUN_ROOT) / "sock" / containers.prefix_of("cputerm")
        proc = subprocess.Popen(self.driver("cputerm"), stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        self.addCleanup(lambda: proc.poll() is None and proc.kill())
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline:
            running = [c["name"] for c in self.children("cputerm") if c["state"] == "running"]
            if any("-cpuserver-" in n for n in running) and any("-marker-" in n for n in running):
                break
            self.assertIsNone(proc.poll(), proc.stderr.read().decode("utf-8", "replace")[-2000:] if proc.poll() else "")
            time.sleep(0.5)
        else:
            self.fail("the CPU server and the Marker child never ran together")
        time.sleep(3)
        proc.send_signal(signal.SIGTERM)
        self.assertEqual(proc.wait(timeout=90), 130)
        self.assertEqual(self.children("cputerm"), [])              # both removed before the process ended
        self.assertFalse(sockets.exists())
        result = json.loads((self.base / "out-cputerm" / "engine.json").read_text())
        self.assertEqual(result["error"]["code"], Code.INTERRUPTED.value)


if __name__ == "__main__":
    unittest.main()
