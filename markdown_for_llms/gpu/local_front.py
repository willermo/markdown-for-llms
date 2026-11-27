#!/usr/bin/env python3
"""The front of a GPU of the user (MFL051 B; J.7.2 and J.7.4 of the plan of MFL051): the one door to vLLM.

It runs inside the container of the pinned vLLM image (the image of the rented presets,
`vllm/vllm-openai@sha256:f0b9a0dc...`), with the `python3` of that image, next to `watchdog.py`, whose constants it
imports and never changes (its SHA-256 is a pin of the rented endpoints): the model and its revision, the pinned
weight, the closed list of what is forwarded (`watchdog.FORWARD`), the arguments of vLLM (`watchdog.vllm_argv`),
the hop-by-hop headers and the limit of a body. Standard library only.

    python3 local_front.py serve --socket /run/mdf/run/gpu/front.sock --hf-home /models
        the local GPU (`gpu` profile of compose.yaml): it listens on a Unix socket of the volume `mdf-run`, mode
        0600, of the user; the container has no network (`network_mode: none`), so vLLM, which it starts on
        127.0.0.1:8001 in its own process group, is reached by nobody else. The relay of the worker reaches the
        socket as it reaches the local CPU server: no port, no transit, no spend.
    python3 local_front.py serve --listen 0.0.0.0:8000 --key-file /run/mdf/server/key --hf-home /models
        the remote server (`compose.server.yaml`, `make server-up` on the machine of the GPU): it listens on the
        internal network of that project, behind nginx, which holds the TLS; every request but /health needs
        `Authorization: Bearer <key>`, compared in constant time with the key of a private file mounted read-only
        (never an environment variable).
    python3 local_front.py prepare --hf-home /models
        once (`make gpu-prepare`, `make server-prepare`): the pinned revision of the model into HF_HOME
        (`watchdog.download`), then its `model.safetensors` checked against its size and SHA-256.
    python3 local_front.py health (--socket PATH | --listen HOST:PORT)
        the healthcheck of compose: exit 0 when vLLM answers its /v1/models through the front.

What it forwards: the pairs of `watchdog.FORWARD` (GET and HEAD of /v1/models, GET of /version and /metrics, POST
of /v1/chat/completions and /v1/completions); anything else is 404 `mdf_not_forwarded` and never reaches vLLM. A
body needs one Content-Length (411 without it, 413 over `watchdog.BODY_LIMIT`, 400 with an ambiguous framing); the
incoming headers are dropped but the content type; an answer of vLLM over the same limit is cut and closed. At most
`watchdog.MAX_HANDLERS` requests at once (503 `mdf_busy`). Until vLLM answers, a forwarded request gets 503
`mdf_starting`; `/health` is served here: 200 once vLLM listed the model, 503 before. Before vLLM starts the
weights are checked (`watchdog.weights_ok`): missing or different, the front does not start vLLM and exits 3
(`mdf_weights_missing`: run the prepare). When vLLM exits, the front exits with its status, and compose decides
(a bounded `on-failure` restart). Its log is JSON lines on stderr (`watchdog.log`): events, statuses and times,
never a document, a header or a key.
"""
import argparse
import hmac
import http.client
import json
import os
import signal
import socket
import socketserver
import stat
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler
from pathlib import Path


def _watchdog():
    """`watchdog.py` next to this file: from the package when this module is `markdown_for_llms.gpu.local_front`,
    else from the directory of this file (the container of vLLM mounts the two files alone)."""
    import importlib
    if __package__:
        return importlib.import_module(__package__ + ".watchdog")
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    return importlib.import_module("watchdog")


wd = _watchdog()

UPSTREAM_PORT = 8001                    # vLLM on the loopback of the container (`watchdog.vllm_argv`)
UPSTREAM_TIMEOUT_S = 300.0              # a request of Surya waits at most this long for vLLM
CHUNK = 65536
KEY_MIN = 32                            # characters of the bearer key of the remote server
PROXY_HEADER = "X-MDF-Proxy"
READY_POLL_S = 2.0


def error(code, status):
    return status, json.dumps({"error": code}).encode("ascii")


class State:
    """What the handlers share: the port of vLLM, its readiness, the bearer key, the handlers in flight."""

    def __init__(self, upstream_port=UPSTREAM_PORT, key=None, model=None):
        self.upstream_port, self.key, self.model = upstream_port, key, model or wd.MODEL
        self.ready = False
        self.slots = threading.BoundedSemaphore(wd.MAX_HANDLERS)

    def authorized(self, header):
        """The bearer key of the remote mode, compared in constant time; True in the local mode (no key)."""
        if self.key is None:
            return True
        given = header or ""
        if not given.startswith("Bearer "):
            return False
        return hmac.compare_digest(given[len("Bearer "):].encode("utf-8", "replace"), self.key.encode("utf-8"))


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "mdf-front"
    sys_version = ""

    def log_message(self, *args):                    # the access log of http.server would name every path
        pass

    def address_string(self):
        return "front"

    def do_GET(self):
        self.handle_one("GET")

    def do_HEAD(self):
        self.handle_one("HEAD")

    def do_POST(self):
        self.handle_one("POST")

    def do_PUT(self):
        self.handle_one("PUT")

    def do_DELETE(self):
        self.handle_one("DELETE")

    def do_PATCH(self):
        self.handle_one("PATCH")

    def reply(self, status, body=b"", content_type="application/json", mark="error", close=False):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header(PROXY_HEADER, mark)
        if close:
            self.send_header("Connection", "close")
            self.close_connection = True
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def length(self, method):
        """The length of the body: one Content-Length, never with Transfer-Encoding; 0 for a GET or HEAD without
        one; an error tuple otherwise."""
        lengths = self.headers.get_all("Content-Length") or []
        if self.headers.get("Transfer-Encoding") is not None:
            return error("mdf_framing", 400) if lengths else error("mdf_length_required", 411)
        if len(lengths) > 1:
            return error("mdf_framing", 400)
        if not lengths:
            return 0 if method in ("GET", "HEAD") else error("mdf_length_required", 411)
        try:
            n = int(lengths[0])
        except ValueError:
            return error("mdf_framing", 400)
        if n < 0:
            return error("mdf_framing", 400)
        if n > wd.BODY_LIMIT:
            return error("mdf_too_large", 413)
        return n

    def handle_one(self, method):
        state = self.server.state
        path = (self.path or "").split("?", 1)[0]
        if path == "/health" and method in ("GET", "HEAD"):
            status = 200 if state.ready else 503
            self.reply(status, json.dumps({"ready": state.ready}).encode("ascii"), mark="health")
            return
        if not state.authorized(self.headers.get("Authorization")):
            wd.log("mdf_front_request", method=method, path=wd.request_path(path), status=401, code="mdf_unauthorized")
            self.reply(*error("mdf_unauthorized", 401), close=True)
            return
        if not wd.forwarded(method, path):
            wd.log("mdf_front_request", method=method, path="other", status=404, code="mdf_not_forwarded")
            self.reply(*error("mdf_not_forwarded", 404))
            return
        n = self.length(method)
        if isinstance(n, tuple):
            self.reply(*n, close=True)
            return
        if not state.slots.acquire(blocking=False):
            self.reply(*error("mdf_busy", 503), close=True)
            return
        try:
            self.forward(state, method, path, n)
        finally:
            state.slots.release()

    def forward(self, state, method, path, n):
        body = self.rfile.read(n) if n else b""
        headers = {"Content-Length": str(len(body))}
        if self.headers.get("Content-Type"):
            headers["Content-Type"] = self.headers.get("Content-Type")
        conn = http.client.HTTPConnection("127.0.0.1", state.upstream_port, timeout=UPSTREAM_TIMEOUT_S)
        try:
            try:
                conn.request(method, self.path, body=body, headers=headers)
                answer = conn.getresponse()
            except (ConnectionRefusedError, ConnectionResetError, FileNotFoundError):
                wd.log("mdf_front_request", method=method, path=wd.request_path(path), status=503, code="mdf_starting")
                self.reply(*error("mdf_starting", 503))
                return
            except (OSError, http.client.HTTPException):
                wd.log("mdf_front_request", method=method, path=wd.request_path(path), status=502, code="mdf_upstream")
                self.reply(*error("mdf_upstream", 502), close=True)
                return
            self.relay_answer(answer, method, path)
        finally:
            conn.close()

    def relay_answer(self, answer, method, path):
        """The answer of vLLM: its status, its headers but the hop-by-hop ones, its body within the limit."""
        declared = answer.getheader("Content-Length")
        try:
            size = int(declared) if declared is not None else None
        except ValueError:
            size = None
        if size is not None and size > wd.BODY_LIMIT:
            wd.log("mdf_front_request", method=method, path=wd.request_path(path), status=502, code="mdf_too_large")
            self.reply(*error("mdf_too_large", 502), close=True)
            return
        data, total = [], 0
        while True:
            piece = answer.read(CHUNK)
            if not piece:
                break
            total += len(piece)
            if total > wd.BODY_LIMIT:
                wd.log("mdf_front_request", method=method, path=wd.request_path(path), status=502,
                       code="mdf_too_large")
                self.reply(*error("mdf_too_large", 502), close=True)
                return
            data.append(piece)
        body = b"".join(data)
        self.send_response(answer.status)
        for name, value in answer.getheaders():
            if name.lower() not in wd.HOP and name.lower() != PROXY_HEADER.lower():
                self.send_header(name, value)
        self.send_header("Content-Length", str(len(body)))
        self.send_header(PROXY_HEADER, "forwarded")
        self.end_headers()
        if method != "HEAD":
            self.wfile.write(body)
        wd.log("mdf_front_request", method=method, path=wd.request_path(path), status=answer.status, kind="forwarded")


class UnixServer(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = True

    def __init__(self, path, state):
        self.state = state
        super().__init__(path, Handler)

    def server_bind(self):
        old = os.umask(0o177)                         # the socket is 0600: the user alone
        try:
            super().server_bind()
        finally:
            os.umask(old)


class TCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address, state):
        self.state = state
        super().__init__(address, Handler)


def socket_path(path):
    """The socket of the local mode: its folder made 0700 when missing, a socket left by a previous start removed
    (only a socket of this user, never another file)."""
    path = Path(path)
    if len(str(path).encode()) > 107:
        raise SystemExit("front: the socket path is longer than 107 bytes")
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    try:
        info = os.lstat(path)
    except FileNotFoundError:
        return str(path)
    if not stat.S_ISSOCK(info.st_mode) or info.st_uid != os.getuid():
        raise SystemExit("front: %s exists and is not a socket of this user" % path.name)
    os.unlink(path)
    return str(path)


def read_key(path):
    """The bearer key of the remote mode from a private regular file (no link, of this user, not readable by
    others), at least KEY_MIN characters; never printed."""
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    except OSError as exc:
        raise SystemExit("front: the key file cannot be opened (%s)" % type(exc).__name__)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
            raise SystemExit("front: the key file is not a private regular file of this user")
        data = os.read(fd, 4097)
    finally:
        os.close(fd)
    key = data.decode("utf-8", "replace").strip()
    if len(key) < KEY_MIN or len(data) > 4096 or any(c.isspace() for c in key):
        raise SystemExit("front: the key file holds no key of at least %d characters on one line" % KEY_MIN)
    return key


def listen_address(text):
    host, _, port = text.rpartition(":")
    if not host or not port.isdigit():
        raise SystemExit("front: --listen HOST:PORT")
    return host, int(port)


def make_server(args, state):
    if args.socket:
        return UnixServer(socket_path(args.socket), state)
    return TCPServer(listen_address(args.listen), state)


def vllm_command(hf_home):
    """The arguments of vLLM, those of the rented workers (`watchdog.vllm_argv`, no lever), with the MTP choice of
    the verified snapshot (`watchdog.mtp_head`)."""
    mtp, _ = wd.mtp_head(hf_home)
    return wd.vllm_argv({"levers": (), "upstream": UPSTREAM_PORT}, mtp)


def watch_ready(state, stop, poll=READY_POLL_S):
    """`state.ready` once vLLM lists the model on its /v1/models."""
    while not stop.is_set() and not state.ready:
        try:
            conn = http.client.HTTPConnection("127.0.0.1", state.upstream_port, timeout=5)
            conn.request("GET", "/v1/models")
            r = conn.getresponse()
            data = r.read(1 << 20)
            conn.close()
            if r.status == 200 and state.model in [m.get("id") for m in json.loads(data).get("data") or []]:
                state.ready = True
                wd.log("mdf_front_ready")
                return
        except (OSError, ValueError, http.client.HTTPException, AttributeError):
            pass
        stop.wait(poll)


def serve(args, vllm=None):
    """The front and vLLM until vLLM exits (or SIGTERM): the exit status of vLLM, 3 without the weights. `vllm`, the
    command of a fake vLLM in the tests."""
    hf_home = args.hf_home
    if vllm is None and not wd.weights_ok(hf_home):
        wd.log("mdf_weights_missing", hint="run the prepare of the GPU (make gpu-prepare or make server-prepare)")
        return 3
    key = read_key(args.key_file) if args.key_file else None
    if args.listen and key is None:
        raise SystemExit("front: --listen needs --key-file: the remote server always checks a key")
    state = State(args.upstream_port, key)
    server = make_server(args, state)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    argv = vllm or vllm_command(hf_home)
    env = wd.vllm_env(os.environ, HF_HOME=hf_home, HF_HUB_OFFLINE="1")
    proc = wd.VLLM(argv, env=env)
    wd.log("mdf_front_start", mode="local" if args.socket else "remote", vllm_args=[a for a in argv if a.startswith("--")],
           pid=proc.proc.pid)
    stop = threading.Event()

    def ended(signum, frame):
        stop.set()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, ended)
    threading.Thread(target=watch_ready, args=(state, stop), daemon=True).start()
    try:
        while not stop.is_set() and proc.poll() is None:
            stop.wait(0.5)
    finally:
        if proc.poll() is None:
            proc.signal(signal.SIGTERM)
            for _ in range(60):
                if proc.poll() is not None:
                    break
                time.sleep(0.5)
            proc.signal(signal.SIGKILL)
        server.shutdown()
        server.server_close()
        if args.socket:
            try:
                os.unlink(args.socket)
            except OSError:
                pass
    code = proc.proc.wait()
    wd.log("mdf_front_exit", vllm_exit=code, stopped=stop.is_set())
    return 0 if stop.is_set() else (code if code > 0 else 1)


def prepare(args):
    """The pinned revision into HF_HOME once, then the check of its weight: 0 when the weights are verified."""
    if wd.weights_ok(args.hf_home):
        wd.log("mdf_weights_check", phase="prepare", ok=True)
        return 0
    code = wd.download(wd.Standalone(), args.hf_home, args.limit_s)
    ok = wd.weights_ok(args.hf_home)
    wd.log("mdf_weights_check", phase="after_download", ok=ok, download_exit=code)
    return 0 if ok else 4


def health(args):
    """Exit 0 when the front says vLLM is ready (/health 200)."""
    try:
        if args.socket:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.settimeout(5)
            sock.connect(args.socket)
            conn = http.client.HTTPConnection("front", timeout=5)
            conn.sock = sock
        else:
            host, port = listen_address(args.listen)
            conn = http.client.HTTPConnection("127.0.0.1" if host in ("0.0.0.0", "") else host, port, timeout=5)
        conn.request("GET", "/health")
        return 0 if conn.getresponse().status == 200 else 1
    except (OSError, http.client.HTTPException):
        return 1


def parser():
    p = argparse.ArgumentParser(prog="local_front.py", description="the front of a GPU of the user (MFL051)")
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("serve", "health"):
        s = sub.add_parser(name)
        where = s.add_mutually_exclusive_group(required=True)
        where.add_argument("--socket", help="the Unix socket of the local GPU")
        where.add_argument("--listen", help="HOST:PORT of the remote server, behind nginx")
        if name == "serve":
            s.add_argument("--hf-home", required=True)
            s.add_argument("--key-file", help="the private file of the bearer key (remote server)")
            s.add_argument("--upstream-port", type=int, default=UPSTREAM_PORT, help=argparse.SUPPRESS)
    pr = sub.add_parser("prepare")
    pr.add_argument("--hf-home", required=True)
    pr.add_argument("--limit-s", type=int, default=3600)
    return p


def main(argv=None, vllm=None):
    args = parser().parse_args(argv)
    if args.command == "serve":
        return serve(args, vllm)
    if args.command == "prepare":
        return prepare(args)
    return health(args)


if __name__ == "__main__":
    sys.exit(main())
