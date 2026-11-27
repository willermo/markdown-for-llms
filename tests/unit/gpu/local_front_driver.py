#!/usr/bin/env python3
"""The driver of the tests of the front of a GPU of the user (MFL051 B): a subprocess, because the guard of
`conftest.py` forbids the sockets in the process of pytest. It runs the real front (`local_front.main`, its `serve`)
with a fake vLLM (`fake_vllm.py`) as the process it starts, sends the requests of a test through the Unix socket (the
local GPU) or TCP (the remote server), then stops the front with SIGTERM and prints one JSON object: the answers, what
reached the fake vLLM, the mode of the socket, the exit status of the front.

    python3 -I -B local_front_driver.py CONFIG.json
CONFIG: {"mode": "local"|"remote", "dir": DIR, "key": KEY or null, "key_mode": 0o600, "requests": [REQUEST...]}
REQUEST: {"method", "path", "body" (text) or null, "headers": {...}, "length": a Content-Length to send instead}
"""
import http.client
import json
import os
from pathlib import Path
import signal
import socket
import stat
import sys
import threading
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
from markdown_for_llms.gpu import local_front  # noqa: E402


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def connect(cfg):
    if cfg["mode"] == "local":
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(10)
        sock.connect(cfg["socket"])
        conn = http.client.HTTPConnection("front", timeout=10)
        conn.sock = sock
        return conn
    return http.client.HTTPConnection("127.0.0.1", cfg["port"], timeout=10)


def send(cfg, req):
    try:
        conn = connect(cfg)
    except OSError as exc:
        return {"status": None, "error": type(exc).__name__}
    try:
        conn.putrequest(req["method"], req["path"], skip_accept_encoding=True)
        body = (req.get("body") or "").encode()
        for name, value in (req.get("headers") or {}).items():
            conn.putheader(name, value)
        if req.get("length") is not None:
            conn.putheader("Content-Length", str(req["length"]))
        elif req.get("body") is not None:
            conn.putheader("Content-Length", str(len(body)))
        conn.endheaders()
        if req.get("body") is not None and req.get("length") is None:
            conn.send(body)
        r = conn.getresponse()
        data = r.read()
        try:
            doc = json.loads(data)
        except ValueError:
            doc = None
        return {"status": r.status, "proxy": r.getheader("X-MDF-Proxy"), "body": doc}
    except (OSError, http.client.HTTPException) as exc:
        return {"status": None, "error": type(exc).__name__}
    finally:
        conn.close()


def wait_ready(cfg, within=20):
    end = time.monotonic() + within
    while time.monotonic() < end:
        got = send(cfg, {"method": "GET", "path": "/health"})
        if got["status"] == 200:
            return True
        time.sleep(0.1)
    return False


def main():
    cfg = json.loads(Path(sys.argv[1]).read_text())
    work = Path(cfg["dir"])
    record = work / "record.jsonl"
    record.touch()
    upstream = free_port()
    fake = [sys.executable, "-I", "-B", str(HERE / "fake_vllm.py"), str(upstream), str(record)]
    args = ["serve", "--hf-home", str(work / "hf"), "--upstream-port", str(upstream)]
    if cfg["mode"] == "local":
        cfg["socket"] = str(work / "gpu" / "front.sock")
        args += ["--socket", cfg["socket"]]
    else:
        cfg["port"] = free_port()
        key = work / "key"
        key.write_text(cfg["key"] + "\n")
        key.chmod(cfg.get("key_mode", 0o600))
        args += ["--listen", "127.0.0.1:%d" % cfg["port"], "--key-file", str(key)]
    out = {"answers": [], "ready": None, "socket_mode": None}
    # a front that refuses to start returns before it installs its handler: the SIGTERM of the checker is then a no-op
    signal.signal(signal.SIGTERM, lambda *a: None)

    def check():
        try:
            out["ready"] = wait_ready(cfg)
            if cfg["mode"] == "local":
                out["socket_mode"] = oct(stat.S_IMODE(os.lstat(cfg["socket"]).st_mode))
            for req in cfg["requests"]:
                out["answers"].append(send(cfg, req))
        finally:
            os.kill(os.getpid(), signal.SIGTERM)
    threading.Thread(target=check, daemon=True).start()
    try:
        out["exit"] = local_front.main(args, vllm=fake)
    except SystemExit as exc:
        out["exit"] = "SystemExit: %s" % exc
    out["reached"] = [json.loads(line) for line in record.read_text().splitlines()]
    out["socket_left"] = cfg["mode"] == "local" and os.path.exists(cfg["socket"])
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
