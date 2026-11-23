"""Runs the proxy of `scripts/servers/watchdog.py` on real loopback sockets, in front of a fake vLLM,
outside the pytest process (whose guard refuses `connect`); prints one JSON object of results.

    python -I -B watchdog_driver.py
"""
import http.server
import json
import os
from pathlib import Path
import signal
import socket
import sys
import threading
import time
import importlib.util
from urllib.parse import parse_qs

import httpx

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location("mdf_watchdog", ROOT / "scripts" / "servers" / "watchdog.py")
wdmod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wdmod)

RELEASE = threading.Event()
SEEN = {}
# vLLM 0.25.1 with VLLM_SERVER_DEV_MODE=1 (`entrypoints/serve/dev/*/api_router.py`, read on 22/11/2026)
DEV_ROUTES = (("POST", "/reset_prefix_cache"), ("POST", "/reset_mm_cache"), ("POST", "/reset_encoder_cache"),
              ("POST", "/pause"), ("POST", "/resume"), ("GET", "/is_paused"), ("POST", "/init_weight_transfer_engine"),
              ("POST", "/start_weight_update"), ("POST", "/update_weights"), ("POST", "/finish_weight_update"),
              ("GET", "/get_world_size"), ("POST", "/collective_rpc"), ("GET", "/server_info"), ("POST", "/sleep"),
              ("POST", "/wake_up"), ("GET", "/is_sleeping"))


class FakeVLLM(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def parse_request(self):
        ok = super().parse_request()
        if ok:                                    # every path that reaches vLLM, by the fake vLLM it reached
            label = getattr(self.server, "label", "main")
            SEEN.setdefault("paths" if label == "main" else label + "_paths", []).append(self.path)
        return ok

    def body(self):
        n = int(self.headers.get("Content-Length") or 0)
        left = n
        while left:
            block = self.rfile.read(min(65536, left))
            if not block:
                break
            left -= len(block)
        return n

    def reply(self, status, data, ctype="application/json", extra=()):
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        for k, v in extra:
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(data)

    def fake(self):
        """The path without the query, and the behaviour that `fake=` asks (the closed list of the proxy
        forwards only its paths: the behaviours of the tests ride on them, MFL045 c1)."""
        base, _, query = self.path.partition("?")
        return base, (parse_qs(query).get("fake") or [""])[0]

    def do_GET(self):
        base, fake = self.fake()
        p = fake and "/" + fake or base
        if p == "/stall_headers":                       # MFL045 P-2: the probe of the supervisor
            RELEASE.wait(3)
            return self.reply(200, b"")
        if p == "/stall_body":
            self.send_response(200)
            self.send_header("Content-Length", "100")
            self.end_headers()
            self.wfile.write(b"x" * 10)
            self.wfile.flush()
            RELEASE.wait(3)
            return
        if p == "/drip":
            self.send_response(200)
            self.send_header("Content-Length", "1000")
            self.end_headers()
            for _ in range(60):
                self.wfile.write(b"x")
                self.wfile.flush()
                if RELEASE.wait(0.05):
                    return
            return
        if p == "/health":
            return self.reply(200, b"")
        if p == "/metrics":
            return self.reply(200, b"vllm:num_requests_running 0\nvllm:generation_tokens_total 5\n", "text/plain")
        if p == "/v1/models":
            return self.reply(200, json.dumps({"data": [{"id": wdmod.MODEL}]}).encode())
        if p == "/version":
            return self.reply(200, b'{"version": "0.25.1"}')
        if p == "/hop":
            SEEN["hop"] = {k.lower(): v for k, v in self.headers.items()}
            return self.reply(200, b"{}", extra=(("Connection", "X-Up"), ("X-Up", "1"), ("Keep-Alive", "timeout=5"),
                                                  ("X-Kept", "yes")))
        if p == "/chunked":
            self.send_response(200)
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            for part in (b"hello ", b"chunked ", b"world"):
                self.wfile.write(b"%x\r\n%s\r\n" % (len(part), part))
            self.wfile.write(b"0\r\n\r\n")
            return
        if p == "/partial":
            self.send_response(200)
            self.send_header("Content-Length", "100000")
            self.end_headers()
            self.wfile.write(b"x" * 1000)
            self.wfile.flush()
            RELEASE.wait(60)
            return
        if p == "/big":
            return self.reply(200, b"y" * 20_000_000, "application/octet-stream")
        if p == "/e500":
            return self.reply(500, b'{"error": "vllm"}')
        if p == "/hang":
            RELEASE.wait(60)
            return self.reply(200, b"{}")
        return self.reply(404, b"{}")

    def do_HEAD(self):
        self.send_response(200)
        self.send_header("Content-Length", "7")
        self.end_headers()

    def do_POST(self):
        n = self.body()
        base, fake = self.fake()
        if base in ("/reset_prefix_cache", "/reset_mm_cache", "/reset_encoder_cache"):
            return self.reply(200, b"")
        if fake == "hang":
            RELEASE.wait(60)
            return self.reply(200, b"{}")
        if base in ("/v1/chat/completions", "/v1/completions"):
            return self.reply(200, json.dumps({"choices": [{"message": {"content": "OK"}}], "bytes": n}).encode())
        return self.reply(404, b"{}")


class FakeProcess:
    def __init__(self):
        self.signals = []

    def signal(self, sig):
        self.signals.append(signal.Signals(sig).name)


def upstream(label="main"):
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), FakeVLLM)
    srv.label = label
    srv.daemon_threads = True
    srv.handle_error = lambda request, address: None
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv.server_address[1]


def proxy(up, req_max=2.0, fault="", ready=True, levers=""):
    cfg = wdmod.config({"MDF_PROVIDER": "runpod", "MDF_UPSTREAM_PORT": str(up), "MDF_FAULT": fault,
                        "MDF_LEVERS": levers})
    cfg["req_max_s"] = req_max
    exits = []
    wd = wdmod.Watchdog(cfg, process=FakeProcess(), on_exit=exits.append)
    wd.ready = ready
    port = wd.serve(port=0, host="127.0.0.1")

    def controller():
        while True:
            wd.control()
            time.sleep(0.2)
    threading.Thread(target=controller, daemon=True).start()
    return wd, "http://127.0.0.1:%d" % port, port, exits


def raw(port, data, read=True, timeout=10):
    s = socket.create_connection(("127.0.0.1", port), timeout=timeout)
    s.sendall(data)
    if not read:
        return s
    out = b""
    try:
        while True:
            block = s.recv(65536)
            if not block:
                break
            out += block
    except socket.timeout:
        pass
    s.close()
    return out


def status_of(data):
    try:
        return int(data.split(b" ", 2)[1])
    except (IndexError, ValueError):
        return None


def main():
    res = {}
    up = upstream()
    wd, url, port, exits = proxy(up, ready=False)
    c = httpx.Client(timeout=10)
    res["health_before"] = c.get(url + "/health").status_code
    r = c.get(url + "/ping")                     # c7.7: the path RunPod's balancer polls, like /health
    res["ping_before"] = [r.status_code, r.headers.get("x-mdf-proxy")]
    wd.ready = True
    h = c.get(url + "/health")
    res["health_after"] = [h.status_code, h.headers.get("x-mdf-proxy")]
    r = c.get(url + "/ping")
    res["ping_after"] = [r.status_code, r.headers.get("x-mdf-proxy")]
    r = c.get(url + "/ping?from=balancer")
    res["ping_query"] = [r.status_code, r.headers.get("x-mdf-proxy")]
    r = c.head(url + "/ping")
    res["ping_head"] = [r.status_code, r.headers.get("x-mdf-proxy")]
    for path in ("/v1/models", "/metrics", "/version"):
        r = c.get(url + path)
        res["get" + path] = [r.status_code, r.headers.get("x-mdf-proxy")]
    r = c.get(url + "/v1/models?token=mdf-query-secret")      # c7.6: the log names the path, never the query
    res["query"] = [r.status_code, r.headers.get("x-mdf-proxy")]
    r = c.post(url + "/v1/chat/completions", json={"model": wdmod.MODEL, "messages": []})
    res["chat"] = [r.status_code, r.headers.get("x-mdf-proxy"), r.headers.get("content-length") is not None]
    r = c.head(url + "/v1/models")
    res["head"] = [r.status_code, r.headers.get("x-mdf-proxy")]
    r = c.get(url + "/version?fake=hop", headers={"Connection": "X-Drop", "X-Drop": "1", "Keep-Alive": "5",
                                     "Proxy-Authorization": "secret", "X-Keep": "ok"})
    res["hop_request_seen"] = sorted(SEEN.get("hop", {}))
    res["hop_response"] = {k: r.headers.get(k) for k in ("x-up", "keep-alive", "x-kept", "x-mdf-proxy", "connection")}
    r = c.get(url + "/version?fake=e500")
    res["e500"] = [r.status_code, r.headers.get("x-mdf-proxy")]
    r = c.get(url + "/version?fake=chunked")
    res["chunked"] = [r.status_code, r.headers.get("content-length"), r.headers.get("transfer-encoding"), r.text]
    # framing
    res["post_no_length"] = status_of(raw(port, b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\n\r\n"))
    res["both_framings"] = status_of(raw(port, b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Length: 2\r\n"
                                              b"Transfer-Encoding: chunked\r\n\r\n2\r\nab\r\n0\r\n\r\n"))
    res["two_lengths"] = status_of(raw(port, b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Length: 2\r\n"
                                             b"Content-Length: 3\r\n\r\nab"))
    res["too_large"] = status_of(raw(port, b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Length: 30000001\r\n\r\n"))
    res["get_body"] = status_of(raw(port, b"GET /v1/models HTTP/1.1\r\nHost: x\r\nContent-Length: 2\r\n\r\nab"))
    res["get_no_length"] = status_of(raw(port, b"GET /v1/models HTTP/1.1\r\nHost: x\r\n\r\n"))
    # timeouts: before the answer (504), a partial answer then a stall, a client that does not read
    t0 = time.monotonic()
    r = c.get(url + "/v1/models?fake=hang")
    res["timeout"] = [r.status_code, r.headers.get("x-mdf-proxy"), r.json().get("error"), round(time.monotonic() - t0, 1)]
    t0 = time.monotonic()
    data = raw(port, b"GET /metrics?fake=partial HTTP/1.1\r\nHost: x\r\n\r\n", timeout=20)
    res["partial"] = [status_of(data), len(data.split(b"\r\n\r\n", 1)[-1]), round(time.monotonic() - t0, 1)]
    s = raw(port, b"GET /metrics?fake=big HTTP/1.1\r\nHost: x\r\n\r\n", read=False)
    t0 = time.monotonic()
    time.sleep(6)
    got, closed = 0, None
    s.settimeout(10)
    try:
        while True:
            block = s.recv(1 << 20)
            if not block:
                break
            got += len(block)
    except OSError:
        pass
    closed = round(time.monotonic() - t0, 1)
    res["no_read"] = [got < 20_000_000, closed]
    s.close()
    # upstream down
    wd2, url2, port2, _ = proxy(1, ready=True)
    r = c.get(url2 + "/v1/models")
    res["upstream_down"] = [r.status_code, r.headers.get("x-mdf-proxy"), r.json().get("error")]
    r = c.get(url2 + "/ping")
    res["ping_upstream_down_ready"] = [r.status_code, r.headers.get("x-mdf-proxy")]
    _, url2b, _, _ = proxy(1, ready=False)       # 19/11: vLLM not listening yet, /ping got 502 from the proxy
    r = c.get(url2b + "/ping")
    res["ping_upstream_down_starting"] = [r.status_code, r.headers.get("x-mdf-proxy")]
    # 41st handler, then 32 x 30,000,000 B with a trip
    wd3, url3, port3, _ = proxy(up, req_max=60.0)
    socks = [raw(port3, b"GET /v1/models?fake=hang HTTP/1.1\r\nHost: x\r\n\r\n", read=False) for _ in range(40)]
    time.sleep(1.0)
    r = c.get(url3 + "/v1/models")
    res["busy_41st"] = [r.status_code, r.headers.get("x-mdf-proxy"), r.json().get("error"), wd3.handlers]
    wd3.trip("W3")
    answers = [status_of(raw_read(s)) for s in socks]
    res["trip_40"] = [answers.count(503), wd3.process.signals]
    r = c.get(url3 + "/v1/models")
    res["inert"] = [r.status_code, r.headers.get("x-mdf-proxy"), r.json().get("error")]
    res["inert_health"] = c.get(url3 + "/health").status_code
    r = c.get(url3 + "/ping")
    res["inert_ping"] = [r.status_code, r.headers.get("x-mdf-proxy")]
    wd4, url4, port4, _ = proxy(up, req_max=60.0)
    results, health = [], []

    def send_big():
        def gen():
            block = b"z" * 65536
            left = 30_000_000
            while left:
                yield block[:min(65536, left)]
                left -= min(65536, left)
        try:
            with httpx.Client(timeout=60) as cc:
                rr = cc.post(url4 + "/v1/completions?fake=hang", content=gen(), headers={"Content-Length": "30000000"})
                results.append([rr.status_code, rr.headers.get("x-mdf-proxy"), rr.json().get("error")])
        except httpx.HTTPError as exc:
            results.append(["error", type(exc).__name__])
    threads = [threading.Thread(target=send_big) for _ in range(32)]
    for t in threads:
        t.start()
    deadline = time.monotonic() + 30
    while len(wd4.pending) < 32 and time.monotonic() < deadline:
        t1 = time.monotonic()
        health.append(round(c.get(url4 + "/health").elapsed.total_seconds(), 3))
        time.sleep(0.2)
    time.sleep(2.0)
    for _ in range(5):
        health.append(round(c.get(url4 + "/health").elapsed.total_seconds(), 3))
    res["big_pending"] = len(wd4.pending)
    t1 = time.monotonic()
    wd4.trip("W2")
    for t in threads:
        t.join(30)
    res["big_trip"] = [sum(1 for r in results if r[:3] == [503, "error", "mdf_tripped"]), len(results),
                       round(time.monotonic() - t1, 1)]
    res["big_health_max_s"] = max(health)
    # faults
    wd5, url5, _, exits5 = proxy(up, fault="hang-after-first")
    c.post(url5 + "/v1/chat/completions", json={})
    c.post(url5 + "/v1/chat/completions", json={})
    t0 = time.monotonic()
    while not wd5.process.signals and time.monotonic() - t0 < 5:
        time.sleep(0.05)
    res["hang_after_first"] = wd5.process.signals
    wd6, url6, _, exits6 = proxy(up, fault="crash-after-first")
    c.get(url6 + "/v1/models")
    res["crash_before"] = list(exits6)
    c.post(url6 + "/v1/chat/completions", json={})
    t0 = time.monotonic()                   # the exit follows the answer: wait for it a little
    while not exits6 and time.monotonic() - t0 < 5:
        time.sleep(0.05)
    res["crash_after_first"] = list(exits6)
    # weights missing
    wd7, url7, _, _ = proxy(up, ready=False)
    wd7.go_inert("mdf_weights_missing")
    r = c.get(url7 + "/v1/models")
    res["weights_missing"] = [r.status_code, r.headers.get("x-mdf-proxy"), r.json().get("error"),
                              c.get(url7 + "/health").status_code, c.get(url7 + "/ping").status_code]
    # MFL045 c1: the closed list of the forwarding, in every mode: vLLM's 16 routes of its dev mode, a path of
    # nothing, a method that the list does not pair with its path; none of them reaches vLLM
    wd8, url8, port8, _ = proxy(up)
    refused = []
    for method, path in DEV_ROUTES + (("GET", "/anything"), ("GET", "/"), ("GET", "/v1/chat/completions"),
                                      ("POST", "/v1/models"), ("DELETE", "/v1/models"), ("POST", "/sleep?level=1"),
                                      ("GET", "/v1/models/../sleep")):
        r = c.request(method, url8 + path, content=b"{}" if method == "POST" else None)
        body = r.json() if r.headers.get("content-type") == "application/json" else None
        refused.append([method, path, r.status_code, r.headers.get("x-mdf-proxy"), (body or {}).get("error")])
    res["not_forwarded"] = refused
    res["forwarded_list"] = [[m, p, c.request(m, url8 + p, json={} if m == "POST" else None).status_code]
                             for m, p in sorted(wdmod.FORWARD)]
    # MFL045 P-2: each operation of a probe at most its timeout of the operation, the whole probe at most its
    # deadline (scaled down here: 0.5 s and 1.5 s for 5 s and 15 s); a drip is cut at the deadline
    probes = {}
    for fake in ("stall_headers", "stall_body", "drip", "none"):
        w = wdmod.Watchdog(dict(wdmod.config({"MDF_PROVIDER": "runpod", "MDF_UPSTREAM_PORT": str(up)})))
        w.probe_op_s, w.probe_s = 0.5, 1.5
        t0 = time.monotonic()
        status, text = w.probe("/metrics?fake=" + fake if fake != "none" else "/metrics")
        probes[fake] = [status, len(text), round(time.monotonic() - t0, 2)]
    res["probes"] = probes
    res["probe_defaults"] = {p: [wdmod.Watchdog(wdmod.config({"MDF_PROVIDER": p})).probe_op_s,
                                 wdmod.Watchdog(wdmod.config({"MDF_PROVIDER": p})).probe_s] for p in ("runpod", "modal")}
    # MFL045 P-9: with the lever flashboot, after the work ends the proxy resets vLLM's three caches on the loopback
    # (its dev mode); the edge still gets 404 for those routes; without the lever no reset
    resets = {}
    for label, levers in (("flashboot", "flashboot"), ("plain", "")):
        upl = upstream(label)
        wdl, urll, _, _ = proxy(upl, levers=levers)
        wdl.start_resetter()
        r = c.post(urll + "/reset_prefix_cache", content=b"{}")
        edge = [r.status_code, r.headers.get("x-mdf-proxy")]
        c.post(urll + "/v1/chat/completions", json={"model": wdmod.MODEL, "messages": []})
        t0 = time.monotonic()
        while time.monotonic() - t0 < 3 and len(SEEN.get(label + "_paths", [])) < 4:
            time.sleep(0.05)
        time.sleep(0.3)
        resets[label] = {"edge": edge, "paths": SEEN.get(label + "_paths", [])}
    res["resets"] = resets
    res["upstream_paths"] = sorted(set(SEEN.get("paths", [])))
    RELEASE.set()
    sys.stdout.write(json.dumps(res) + "\n")
    sys.stdout.flush()
    os._exit(0)                   # daemon threads of the fake upstream: no orderly shutdown needed


def raw_read(s):
    out = b""
    s.settimeout(10)
    try:
        while True:
            block = s.recv(65536)
            if not block:
                break
            out += block
    except OSError:
        pass
    s.close()
    return out


if __name__ == "__main__":
    sys.exit(main())
