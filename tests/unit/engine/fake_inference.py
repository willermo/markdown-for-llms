"""A fake OpenAI-compatible inference server for the tests of the `marker-server` route (MFL033).

The suite forbids `socket.connect` in its own process (`tests/conftest.py`), so the fake is an
`httpx.MockTransport` handed to the relay (`Relay(..., transport=fake.transport())`): the relay
runs real `httpx` towards it, without a socket. It frames requests as the proxy in front of vLLM
does (C.2 of plan r4 with the prescription on lengths): a request with a body needs a
`Content-Length` (411 otherwise), a GET without a body has length 0. It answers `GET /health`,
`/v1/models`, `/version`, `/props`, `/metrics` and `POST /v1/chat/completions`, checks the key when
it has one (401 otherwise, `/health` excepted) and records what it saw. Its behaviour can change
while it runs (`fake.models_hang = True`, `fake.fail_next = 3`, ...); `respond(request_json) ->
str` gives the content of an answer (default `<p>ok</p>`); `mark` adds `X-MDF-Proxy` to every
answer (a string, or `callable(path, status) -> str or None`).

A server that repeats the credentials it received (S-B1 of the review of MFL033): with `reflect`
every error answer repeats them in its body and every answer in `X-Echo` (and in its
`Content-Type` with `reflect_type`); `script` holds the statuses of the next chat answers, a 200
among them answering the credentials as its content.

`PairTransport(relay)` is the side of the Marker child for the tests in this process: every
request goes through a `socketpair` (allowed by the guard) whose other end the relay's own server
handles, so the inbound path of the relay is the real one.
"""
import http.client
import json
import socket
import threading
import time

import httpx

MODEL = "datalab-to/surya-ocr-2.5"


class Fake:
    def __init__(self, key=None, model=MODEL, version="0.25.1", mark=None, respond=None, delay=0.0):
        self.key, self.model, self.version, self.mark = key, model, version, mark
        self.respond = respond or (lambda request: "<p>ok</p>")
        self.delay = delay
        self.fail_next = 0                  # the next N chat completions answer `fail_status`
        self.fail_always = False
        self.fail_status = 500
        self.models_status = 200            # the status of GET /v1/models
        self.models_hang = False            # GET /v1/models never answered (until `close`)
        self.chat_hang = False
        self.oversize = 0                   # bytes of an answer over the limit, when > 0
        self.reflect = False                # errors and headers repeat the credentials received
        self.reflect_type = False           # ... and the Content-Type of every answer
        self.script = []                    # statuses of the next chat answers, repeating the credentials
        self.lock = threading.Lock()
        self.seen = []                      # (method, path, Content-Length or None, Transfer-Encoding or None)
        self.auth, self.agents = [], []
        self.chat = self.in_flight = self.max_in_flight = 0
        self.release = threading.Event()    # ends every hang

    def transport(self, limits=None):
        """A transport for the relay; with `limits`, a pool of `max_connections` whose wait past the
        pool timeout of the request is `httpx.PoolTimeout`, as in a real pool."""
        if limits is None or limits.max_connections is None:
            return httpx.MockTransport(self.handle)
        slots = threading.BoundedSemaphore(limits.max_connections)
        fake = self

        class Pooled(httpx.BaseTransport):
            def handle_request(self, request):
                wait = (request.extensions.get("timeout") or {}).get("pool")
                if not slots.acquire(timeout=wait):
                    raise httpx.PoolTimeout("pool", request=request)
                try:
                    response = fake.handle(request)
                    response.read()
                    return response
                finally:
                    slots.release()
        return Pooled()

    def factory(self):
        return lambda limits: self.transport(limits)

    def _send(self, path, status, body, ctype="application/json", creds=""):
        if self.reflect and status >= 400:
            body = {"error": {"message": "invalid credential: " + creds, "type": "auth"}}
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        headers = {"Content-Type": ctype + ("; echo=" + creds if self.reflect_type and creds else "")}
        mark = self.mark(path, status) if callable(self.mark) else self.mark
        if mark:
            headers["X-MDF-Proxy"] = mark
        if self.reflect and creds:
            headers["X-Echo"] = creds
        return httpx.Response(status, headers=headers, content=data)

    def handle(self, request):
        path, method = request.url.path, request.method
        creds = " ".join(v for v in (request.headers.get(h) for h in ("Authorization", "Modal-Key", "Modal-Secret"))
                         if v)

        def send(path, status, body, ctype="application/json"):
            return self._send(path, status, body, ctype, creds)
        with self.lock:
            self.seen.append((method, path, request.headers.get("Content-Length"),
                              request.headers.get("Transfer-Encoding")))
            self.auth.append(request.headers.get("Authorization"))
            self.agents.append(request.headers.get("User-Agent"))
        authorized = self.key is None or request.headers.get("Authorization") == "Bearer " + self.key
        if method == "GET":
            if path == "/health":
                return send(path, 200, {})
            if not authorized:
                return send(path, 401, {"error": "unauthorized"})
            if path == "/v1/models":
                if self.models_hang:
                    self.release.wait(120)
                    raise httpx.ReadTimeout("hang", request=request)
                if self.models_status != 200:
                    return send(path, self.models_status, {"error": "not ready"})
                return send(path, 200, {"object": "list", "data": [{"id": self.model, "object": "model"}]})
            if path == "/version":
                return send(path, 200, {"version": self.version})
            if path == "/props":
                return send(path, 200, {"build_info": self.version})
            if path == "/metrics":
                return send(path, 200, b"vllm:num_requests_running 0\n", "text/plain")
            return send(path, 404, {"error": "not found"})
        if request.headers.get("Transfer-Encoding") or request.headers.get("Content-Length") is None:
            return send(path, 411, {"error": "length required"})
        body = request.read()
        if not authorized:
            return send(path, 401, {"error": "unauthorized"})
        if method != "POST" or path != "/v1/chat/completions":
            return send(path, 404, {"error": "not found"})
        payload = json.loads(body)
        with self.lock:
            self.chat += 1
            self.in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self.in_flight)
            fail = self.fail_always or self.fail_next > 0
            if self.fail_next > 0:
                self.fail_next -= 1
        try:
            if self.chat_hang:
                self.release.wait(120)
                raise httpx.ReadTimeout("hang", request=request)
            if self.delay:
                time.sleep(self.delay)
            with self.lock:
                scripted = self.script.pop(0) if self.script else None
            if scripted is not None and scripted != 200:
                return send(path, scripted, {"error": {"message": "invalid credential: " + creds}})
            if fail:
                return send(path, self.fail_status, {"error": "injected"})
            if self.oversize:
                return send(path, 200, b"x" * self.oversize, "text/plain")
            content = "<p>%s</p>" % creds if scripted == 200 else self.respond(payload)
            return send(path, 200, {
                "id": "fake", "object": "chat.completion", "created": 0, "model": payload.get("model"),
                "choices": [{"index": 0, "message": {"role": "assistant", "content": content},
                             "finish_reason": "stop", "logprobs": None}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}})
        finally:
            with self.lock:
                self.in_flight -= 1

    def paths(self, method=None):
        with self.lock:
            return [p for m, p, _, _ in self.seen if method is None or m == method]

    def close(self):
        self.release.set()


def pair(relay):
    """A connected socket whose other end the relay's server handles in its own thread."""
    ours, theirs = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    relay.server.process_request(theirs, "pair")
    return ours


class PairTransport(httpx.BaseTransport):
    """Requests of the child side, one `socketpair` each, read with `http.client`."""

    def __init__(self, relay, timeout=30):
        self.relay, self.timeout = relay, timeout

    def handle_request(self, request):
        sock = pair(self.relay)
        sock.settimeout(self.timeout)
        conn = http.client.HTTPConnection("relay")
        conn.sock = sock
        try:
            headers = {k: v for k, v in request.headers.items() if k.lower() not in ("host", "content-length",
                                                                                      "transfer-encoding")}
            body = request.read()
            if body:
                headers["Content-Length"] = str(len(body))
            conn.request(request.method, request.url.raw_path.decode("ascii"), body=body or None, headers=headers)
            reply = conn.getresponse()
            data = reply.read()
            return httpx.Response(reply.status, headers=reply.getheaders(), content=data)
        except (OSError, http.client.HTTPException) as exc:
            raise httpx.RemoteProtocolError(type(exc).__name__, request=request) from None
        finally:
            conn.close()


def raw(relay, data, timeout=10):
    """Bytes sent as they are to the relay; returns the status of the answer."""
    with pair(relay) as s:
        s.settimeout(timeout)
        s.sendall(data)
        reply = b""
        while b"\r\n" not in reply:
            chunk = s.recv(4096)
            if not chunk:
                break
            reply += chunk
    return int(reply.split(b" ", 2)[1])
