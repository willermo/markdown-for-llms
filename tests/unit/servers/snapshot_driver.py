"""Runs `snap_boot` and `snap_wake` of `scripts/servers/watchdog.py` for real, outside the pytest process (whose
guard refuses `connect`): the vLLM they start is a fake one, a child process on the loopback that writes access
lines and, as the scenario asks, an ERROR line of its own pid (`snapshot.py` with the watchdog it imports); the
watchdog writes its JSON lines on stderr
(MFL045 c2, the integrated proof of A9 and P-7).

    python -I -B snapshot_driver.py SCENARIO OUT

SCENARIO: `awake` (the wake succeeds), `fallback` (the wake fails with an ERROR of the adopted pid, the fallback
starts a vLLM that becomes ready) or `definitive` (the same, and the vLLM of the fallback fails at its start).
The fake vLLM prefixes each of its lines with `T<epoch>|`, the time Modal's CLI would print. OUT gets the
result: the lines of the fake vLLM with their times, the times of the snapshot and of the restore."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import threading
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts" / "servers"))
spec = importlib.util.spec_from_file_location("mdf_snapshot_sd", ROOT / "scripts" / "servers" / "snapshot.py")
snmod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(snmod)
wdmod = snmod.wd                                   # the watchdog that snapshot.py uses

FAKE = r'''
import http.server, json, os, sys, threading, time
port, mode = int(sys.argv[1]), sys.argv[2]
state = {"asleep": False}
pid = os.getpid()


def say(text):
    sys.stdout.write("T%.3f|%s\n" % (time.time(), text))
    sys.stdout.flush()


class H(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def answer(self, status, body=b""):
        say('(APIServer pid=%d) INFO:     127.0.0.1:41000 - "%s %s HTTP/1.1" %d' % (pid, self.command, self.path, status))
        self.send_response(status)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        p = self.path
        if p == "/health":
            return self.answer(200)
        if p == "/v1/models":
            return self.answer(200, json.dumps({"data": [{"id": "datalab-to/surya-ocr-2.5"}]}).encode())
        if p == "/is_sleeping":
            return self.answer(200, json.dumps({"is_sleeping": state["asleep"]}).encode())
        if p == "/metrics":
            return self.answer(200, b"vllm:num_requests_running 0\nvllm:generation_tokens_total 1\n")
        return self.answer(404, b"{}")

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        self.rfile.read(n)
        p = self.path
        if p == "/v1/chat/completions":
            return self.answer(200, b'{"choices": []}')
        if p in ("/reset_prefix_cache", "/reset_mm_cache", "/reset_encoder_cache"):
            return self.answer(200)
        if p.startswith("/sleep"):
            state["asleep"] = True
            return self.answer(200)
        if p == "/wake_up":
            if mode in ("fallback", "definitive"):
                say("(APIServer pid=%d) ERROR 11-22 14:00:01 [api_server.py:1] wake_up failed: CUDA error" % pid)
                return self.answer(500, b'{"error": "wake_up failed"}')
            state["asleep"] = False
            return self.answer(200)
        return self.answer(404, b"{}")


if mode == "broken":
    say("(APIServer pid=%d) ERROR 11-22 14:00:09 [api_server.py:1] the engine failed to start" % pid)
    sys.exit(1)
say("(APIServer pid=%d) INFO 11-22 14:00:00 [api_server.py:1] Starting vLLM API server on http://127.0.0.1:%d" % (pid, port))
srv = http.server.ThreadingHTTPServer(("127.0.0.1", port), H)
srv.daemon_threads = True
srv.serve_forever()
'''


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def main():
    scenario, out = sys.argv[1], Path(sys.argv[2])
    tmp = Path(tempfile.mkdtemp(prefix="mdf-snap-driver-"))
    fake = tmp / "fake_vllm.py"
    fake.write_text(FAKE)
    data = b"weights-of-the-snapshot-driver"
    wdmod.WEIGHT = ("model.safetensors", len(data), hashlib.sha256(data).hexdigest())
    hf = tmp / "vol" / "hf"
    snap_dir = wdmod.snapshot(str(hf))
    snap_dir.mkdir(parents=True)
    for name in wdmod.FILES:
        (snap_dir / name).write_text("{}")
    (snap_dir / "model.safetensors").write_bytes(data)
    up = free_port()
    env = dict(os.environ, MDF_PROVIDER="modal", HF_HOME=str(hf), MDF_LEVERS="snapshot", MDF_UPSTREAM_PORT=str(up),
               PORT="0", MDF_TICK_S="0.2", MODAL_TASK_ID="ta-01CREATE", MODAL_ENABLE_SNAP_RESTORE="1",
               VLLM_CACHE_ROOT=str(tmp / "vllm"))
    lines, procs = [], []

    class Piped(wdmod.VLLM):
        """vLLM with its stdout read here: each line with the time the fake wrote in it."""

        def __init__(self, argv, env=None):
            self.proc = subprocess.Popen(argv, start_new_session=True, stdin=subprocess.DEVNULL, env=env,
                                         stdout=subprocess.PIPE, text=True)
            procs.append(self.proc)
            threading.Thread(target=self.read, daemon=True).start()

        def read(self):
            for raw in self.proc.stdout:
                t, _, text = raw.rstrip("\n").partition("|")
                lines.append([float(t[1:]), text, self.proc.pid])

    wdmod.VLLM = Piped
    normal = "broken" if scenario == "definitive" else "awake"
    wdmod.vllm_argv = lambda cfg, mtp: [sys.executable, "-I", "-B", str(fake), str(up), normal]
    argv = [sys.executable, "-I", "-B", str(fake), str(up), scenario]
    t0 = time.time()
    snap = snmod.snap_boot(env=env, argv=argv)
    t_snap = time.time()
    time.sleep(1.2)                                       # the checkpoint and the restore take their time
    t_restore = time.time()
    w = snmod.snap_wake(snap, env=dict(env, MODAL_TASK_ID="ta-01RESTORE"))
    end = time.monotonic() + 30
    while time.monotonic() < end and not w.ready and w.inert is None and not w.seen["exit"]:
        time.sleep(0.1)
    time.sleep(1.0)
    out.write_text(json.dumps({"t0": t0, "t_snap": t_snap, "t_restore": t_restore, "ready": w.ready,
                               "inert": w.inert, "boot_id": w.boot_id, "lines": lines, "pids": snap["pids"]}))
    for p in procs:
        try:
            os.killpg(p.pid, 9)
        except OSError:
            pass
    os._exit(0)


if __name__ == "__main__":
    sys.exit(main())
