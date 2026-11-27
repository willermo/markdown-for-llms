"""The relay of the `marker-server` route (MFL033): the only way out of the Marker child.

The Marker child has no network (its container is started without one; MFL051). On this route it reaches the
inference server through a Unix socket of the converting process, `relay/relay.sock` in a private
directory (`sockets.Private`), bound read-only on `/relay` of the child. The relay runs in the
converting process, never in the child, with `ThreadingMixIn` + `UnixStreamServer`:

- inbound: `POST /v1/chat/completions` and the read-only `GET` of `READ_PATHS`; anything else is
  404 and is never forwarded. A request with a body needs one `Content-Length` (411 otherwise,
  `Transfer-Encoding` included) of at most 30,000,000 bytes (413 before reading it); a `GET` or
  `HEAD` without a body has length 0; ambiguous framing (`Content-Length` and
  `Transfer-Encoding` together, or several lengths) is 400. The incoming headers are dropped:
  each request goes out with a fresh set, the headers of the endpoint (its key) and
  `User-Agent: markdown-for-llms/<version>`;
- outbound: one `httpx.Client` (`trust_env=False`, certificates verified, redirects never
  followed, `Limits` on `parallel`): HTTPS, HTTP to a literal loopback address, or the Unix
  socket of the local CPU server or of the front of the local GPU (kind `gpu`, MFL051 B). With the
  `ca_file` of a `user` endpoint (MFL051 B) the two clients, that of the requests and that of the
  probes, verify the certificate of the server and its name against that file alone, never the
  authorities of the system; a file that cannot be read is `server_not_configured` before the start. Every forwarded request asks `Accept-Encoding: identity`
  (MFL046 d) and its answer is read as received: as it is, or gzip or deflate decoded piece by
  piece, each piece and each tail left by the decoder within what is left of the same limit (the
  decoder is never flushed, which is not bounded); over the limit it is 502 and `oversize`;
  another encoding, a stream that does not end where its encoding ends (truncated) or has bytes
  after its end is 502 and `encoding`, with the outcome of an answer over the limit. The checks
  of the credentials read the decoded body, `bytes_out` counts the bytes received. A 3xx is never
  followed and reaches the child as 502;
- what reaches the child (MFL033 S-B1): the status of every answer, the body and `Content-Type`
  of a 2xx only, never another header. Any other status reaches it with a reason of the relay in
  place of the remote body. A 2xx whose body or `Content-Type` repeats a credential of the relay
  (as it is, escaped in JSON, percent-encoded or in base64, or in a string of its JSON) reaches
  it as 502; such an answer is counted in `reflected`, the counters of the statuses stay those of
  the server. The version of the server passes the same filter before `versions` and the detail
  of `server_mismatch`;
- readiness (`ready`): `GET /v1/models` until a 200 that lists the model of the endpoint, within
  `ready_within` (401 or 403: `server_rejected`; another model: `server_mismatch`; never ready:
  `server_unavailable`), then the declared version (`GET /version` of vLLM, `GET /props` of
  llama.cpp) against the pins of the endpoint (`server_mismatch`). The first request of the
  readiness is the wake of the server, `t_wake`;
- health: a client of its own (`Limits(1, 0)`, short timeouts, at most `probe_deadline` seconds
  per probe, pool wait and connection included) sends `GET /v1/models` every `probe_interval`
  seconds while the child runs. Three failures in a row put the relay `down`: new requests get
  503 and `failure` is `server_unavailable`, which the engine reads at every poll of the child;
- every probe (readiness, version, health; MFL033 S-B5) asks `Accept-Encoding: identity` and reads
  its answer as a stream, at most `PROBE_LIMIT` bytes as received (a gzip or deflate answer is
  decoded within the same limit, another encoding is refused), within its absolute term: past the
  limit or the term the answer is closed and the probe fails (`oversize`, `deadline`, `encoding`
  among the statuses of the readiness and in `probes`);
- term: at `t_wake + D - 5 s` the relay closes and `failure` is `server_deadline`.

For an endpoint with a provider (RunPod, Modal) the proxy in front of vLLM marks each of its
answers (`X-MDF-Proxy`): a response read in full is complete with `forwarded`, counted in
`proxy_errors` with `error` (an answer of the proxy itself); it is uncertain without the header
(the edge of the provider), with a 3xx, a timeout, an error of transport, or when the relay
closes under it. The close is clean only after a readiness, never `down`, with nothing in flight,
nothing uncertain and no `error`: the bound of the attempt then uses the short tail (`servers`).
The counters and times go to `engine.json` (`marker_server`).
"""
from http.server import BaseHTTPRequestHandler
import base64
import json
import zlib
import socket
import socketserver
import threading
import time
from urllib.parse import quote

from . import servers, sockets
from .codes import Code, EngineError

BODY_LIMIT = 30_000_000
READ_PATHS = ("/v1/models", "/health", "/metrics", "/version", "/props")
CHAT = "/v1/chat/completions"
PROXY_HEADER = "X-MDF-Proxy"
QUEUE = 64
CLOSE_MARGIN = 5.0
PROBE_LIMIT = 1 << 20                     # bytes of the answer of a probe, as received and decoded


class ProbeFailed(Exception):
    """A probe whose answer was closed unread (S-B5): `kind` is oversize, deadline or encoding."""

    def __init__(self, kind):
        super().__init__(kind)
        self.kind = kind


def _decoded(data, encoding, limit):
    """The body of a probe from its bytes as received: as it is, or gzip/deflate within `limit`."""
    encoding = (encoding or "identity").strip().lower()
    if encoding in ("", "identity"):
        return data
    if encoding not in ("gzip", "x-gzip", "deflate"):
        raise ProbeFailed("encoding")
    try:
        d = zlib.decompressobj(47)                # gzip or zlib, by the header
        out = d.decompress(data, limit + 1)
    except zlib.error:
        raise ProbeFailed("encoding") from None
    if len(out) > limit or d.unconsumed_tail:
        raise ProbeFailed("oversize")
    return out
REFLECTED = b'{"error": {"message": "relay: the answer repeats a credential of the relay", "type": "relay"}}'


class Body:
    """The body of a forwarded answer from its bytes as received (MFL046 d): identity, or gzip or deflate decoded
    piece by piece, each input piece and each `unconsumed_tail` with at most what is left of `limit` plus one byte,
    so that at most `limit + 1` decoded bytes and one piece received are in memory; never `flush()`, whose argument
    is the size of a buffer, not a limit. `feed` and `finish` raise ProbeFailed: `oversize` past the limit,
    `encoding` for another encoding, a stream that is not a whole member (truncated) or has bytes after it."""

    def __init__(self, encoding, limit):
        encoding = (encoding or "identity").strip().lower()
        if encoding in ("", "identity"):
            self.decoder = None
        elif encoding in ("gzip", "x-gzip", "deflate"):
            self.decoder = zlib.decompressobj(47)         # gzip or zlib, by the header
        else:
            raise ProbeFailed("encoding")
        self.limit, self.parts, self.size = limit, [], 0

    def _add(self, data):
        self.size += len(data)
        if self.size > self.limit:
            raise ProbeFailed("oversize")
        self.parts.append(data)

    def feed(self, data):
        if self.decoder is None:
            self._add(data)
            return
        while data:
            if self.decoder.eof:
                raise ProbeFailed("encoding")              # bytes after the end of the member
            try:
                self._add(self.decoder.decompress(data, self.limit - self.size + 1))
            except zlib.error:
                raise ProbeFailed("encoding") from None
            data = self.decoder.unconsumed_tail
        if self.decoder.unused_data:
            raise ProbeFailed("encoding")

    def finish(self):
        if self.decoder is not None and (not self.decoder.eof or self.decoder.unused_data):
            raise ProbeFailed("encoding")                  # truncated, or bytes after the end
        return b"".join(self.parts)


def _error_body(status):
    """What the child reads in place of the body of an answer that is not a 2xx."""
    return json.dumps({"error": {"message": "relay: the server answered %d" % status, "type": "relay"}}).encode()


def credential_forms(values):
    """The byte forms in which an answer may repeat one of the credential `values`: as it is,
    escaped in JSON (also with `\\/`), percent-encoded, in base64 (standard and URL-safe, without
    the padding)."""
    forms = set()
    for value in values:
        if not value:
            continue
        escaped = json.dumps(value)[1:-1]
        for text in (value, escaped, escaped.replace("/", "\\/"), quote(value, safe=""), quote(value)):
            forms.add(text.encode("utf-8", "surrogatepass"))
        raw = value.encode("utf-8", "surrogatepass")
        forms.update((base64.b64encode(raw).rstrip(b"="), base64.urlsafe_b64encode(raw).rstrip(b"=")))
    return tuple(sorted(forms))


def _strings(doc):
    """Every string (key or value) of a decoded JSON document, without recursion."""
    stack = [doc]
    while stack:
        item = stack.pop()
        if isinstance(item, str):
            yield item
        elif isinstance(item, dict):
            stack.extend(item.keys())
            stack.extend(item.values())
        elif isinstance(item, list):
            stack.extend(item)


def _version():
    try:
        from importlib.metadata import version
        return version("markdown-for-llms")
    except Exception:
        return "unknown"


class _Server(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = True
    request_queue_size = QUEUE

    def __init__(self, path, handler, relay):
        self.relay = relay
        super().__init__(str(path), handler)

    def process_request(self, request, client_address):
        self.relay._track(request, True)
        super().process_request(request, client_address)

    def shutdown_request(self, request):
        self.relay._track(request, False)
        super().shutdown_request(request)


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "mdf-relay"
    sys_version = ""

    def address_string(self):
        return "relay"

    def log_message(self, *args):
        pass

    def do_GET(self):
        self.server.relay._inbound(self, "GET")

    def do_POST(self):
        self.server.relay._inbound(self, "POST")

    def do_HEAD(self):
        self.server.relay._inbound(self, "HEAD")

    def do_PUT(self):
        self.server.relay._inbound(self, "PUT")

    def do_DELETE(self):
        self.server.relay._inbound(self, "DELETE")


def authorities(ep):
    """What the clients of the relay verify the server with: True (the authorities of the system), or for a `user`
    endpoint with `ca_file` (MFL051 B) an SSL context of that file alone, which also checks the name of the
    certificate; `server_not_configured` when the file cannot be read or holds no certificate."""
    import ssl
    if getattr(ep, "ca_file", None) is None:
        return True
    try:
        context = ssl.create_default_context(cafile=ep.ca_file)
    except (OSError, ssl.SSLError, ValueError) as exc:
        raise EngineError(Code.SERVER_NOT_CONFIGURED, "ca_file of %s: %s" % (ep.name, type(exc).__name__)) from None
    if not context.get_ca_certs():
        raise EngineError(Code.SERVER_NOT_CONFIGURED, "ca_file of %s: no certificate authority" % ep.name)
    context.check_hostname = True
    context.verify_mode = ssl.CERT_REQUIRED
    return context


class Relay:
    """One relay per conversion: `start`, `ready`, `set_term`, then `close` in a `finally`."""

    def __init__(self, ep, keys, socket_path, upstream_socket=None, clock=time.monotonic, wall=time.time,
                 probe_interval=10.0, probe_deadline=5.0, probe_failures=3, ready_poll=2.0, transport_factory=None):
        import httpx
        self.httpx, self.ep, self.clock, self.wall = httpx, ep, clock, wall
        # the drift is read on a pair of clocks: the real ones, or both given (a test that gives a monotonic clock
        # alone moves it without the wall clock)
        self._drift_on = clock is time.monotonic or wall is not time.time
        self.socket_path = socket_path
        sockets.check(socket_path)
        self._anchor = (wall(), clock())
        self._out = dict(servers.headers(ep, keys), **{"User-Agent": "markdown-for-llms/" + _version()})
        self._plain = tuple(servers.secret_values(ep, keys))
        self._forms = credential_forms(self._plain)
        self.probe_interval, self.probe_deadline, self.probe_failures = probe_interval, probe_deadline, probe_failures
        self.ready_poll = ready_poll
        base = {"cpu": "http://llama", "gpu": "http://gpu"}.get(ep.kind) or servers.canonical_url(ep.url)
        if ep.kind == "gpu":
            upstream_socket = ep.socket                  # the front of the local GPU (MFL051 B)
        if ep.kind == "cpu" and upstream_socket is None and transport_factory is None:
            raise EngineError(Code.SERVER_NOT_CONFIGURED, "the CPU server has no socket")
        verify = authorities(ep)
        if transport_factory is None:
            def transport_factory(limits):
                kw = {"uds": str(upstream_socket)} if upstream_socket is not None else {}
                return httpx.HTTPTransport(verify=verify, limits=limits, retries=0, **kw)

        def make(limits, timeout):
            return httpx.Client(base_url=base, transport=transport_factory(limits), timeout=timeout, trust_env=False,
                                follow_redirects=False, verify=verify)
        n = ep.parallel
        self.client = make(httpx.Limits(max_connections=n, max_keepalive_connections=n),
                           httpx.Timeout(connect=15.0, read=float(ep.read_timeout), write=60.0, pool=float(ep.read_timeout)))
        self.probe_client = make(httpx.Limits(max_connections=1, max_keepalive_connections=0),
                                 httpx.Timeout(connect=2.0, read=3.0, write=1.0, pool=1.0))
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.state = "new"                       # new, open, down, closed
        self.failure = None                      # Code.SERVER_UNAVAILABLE or Code.SERVER_DEADLINE
        self.server = self._thread = self._monitor = None
        self._conns = set()
        self.deadline_at = None
        self.until = None                        # MFL023 B2: the same term on the wall clock (epoch)
        self.ready_at = None
        self.versions = {}
        self.stats = {"by_status": {}, "requests": {"chat": 0, "read": 0}, "refused_paths": 0, "framing_refused": 0,
                      "transport_errors": 0, "timeouts": 0, "oversize": 0, "encoding": 0, "abandoned": 0, "uncertain": 0,
                      "proxy_errors": 0, "unmarked": 0, "redirects": 0, "reflected": 0, "bytes_in": 0, "bytes_out": 0,
                      "seconds": 0.0,
                      "in_flight": 0, "max_in_flight": 0, "in_flight_at_close": 0,
                      "probes": {"sent": 0, "failed": 0, "max_ms": 0.0, "oversize": 0, "deadline": 0, "encoding": 0}}
        self.readiness = {"attempts": 0, "statuses": {}, "wait_seconds": None}
        self.t_wake = self.t_last = self.t_close = None
        self.last_request = None                 # monotonic: the start or end of the last forwarded request

    # --- times -------------------------------------------------------------------------------

    def _epoch(self, mono):
        return None if mono is None else round(self._anchor[0] + (mono - self._anchor[1]), 3)

    def _woke(self):
        with self.lock:
            if self.t_wake is None:
                self.t_wake = self.clock()

    # --- lifecycle ---------------------------------------------------------------------------

    def start(self):
        self.server = _Server(self.socket_path, _Handler, self)
        self.state = "open"
        self._thread = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.2},
                                        name="mdf-relay", daemon=True)
        self._thread.start()
        return self

    def set_term(self, seconds, until=None):
        """The relay closes at `t_wake + seconds - 5 s` (`server_deadline`). The same term holds on the wall clock
        (MFL023 B2, SOL-R2 of the plan r1): `self.until`, the epoch of that moment, at most `until` (the term T
        of the session, epoch seconds), is compared with the wall clock before every new request and by `check`,
        so a suspend of the laptop, which the monotonic clock does not count, never lets a request leave after
        it."""
        self._woke()
        self.deadline_at = self.t_wake + float(seconds) - CLOSE_MARGIN
        wall_deadline = self.wall() - (self.clock() - self.deadline_at)
        self.until = wall_deadline if until is None else min(wall_deadline, float(until))
        if self.clock is time.monotonic:
            # the relay closes at its term by itself, whoever polls `check` (the engine also does)
            timer = threading.Timer(max(0.0, self.deadline_at - self.clock()), self.check)
            timer.daemon = True
            timer.start()
            self._timer = timer

    def activity(self):
        """The monotonic time of the last activity of the child through the relay: now while one of
        its requests is in flight, else the end of the last one (None before any). A child waiting
        on a slow server is not stalled (`isolation.run`)."""
        with self.lock:
            if self.stats["in_flight"]:
                return self.clock()
            return self.last_request

    def check(self):
        """The failure the engine must stop the child with, or None: also the term on the wall clock and an
        interval the relay did not observe (`unobserved`)."""
        if self.failure is None and ((self.deadline_at is not None and self.clock() >= self.deadline_at)
                                     or self._past_wall()):
            self._fail(Code.SERVER_DEADLINE)
        elif self.failure == Code.SERVER_DEADLINE and self.state == "down":
            self.close()                         # a request refused at the term on the wall clock (`_inbound`)
        return self.failure

    def drift(self):
        """The drift of the wall clock on the monotonic clock since the relay started (seconds): a suspend of
        the laptop or a step of its clock."""
        return (self.wall() - self._anchor[0]) - (self.clock() - self._anchor[1])

    def _past_wall(self):
        """MFL023 B2: the term on the wall clock is past, or the wall clock drifted from the monotonic one beyond
        `servers.DRIFT_LIMIT_S` (an interval not observed: the attempt cannot bound what it did not see)."""
        if self.until is not None and self.wall() >= self.until:
            return True
        return self._drift_on and abs(self.drift()) > servers.DRIFT_LIMIT_S

    def _fail(self, code):
        with self.lock:
            if self.failure is None:
                self.failure = code
            if self.state == "open":
                self.state = "down"
        if code == Code.SERVER_DEADLINE:
            self.close()

    def close(self):
        """Idempotent: no new request, the open connections and the clients closed."""
        with self.lock:
            if self.state == "closed":
                return
            self.state = "closed"
            self.t_close = self.clock()
            # what is in flight now is abandoned and uncertain, whenever its thread ends
            self.stats["in_flight_at_close"] = self.stats["abandoned"] = self.stats["in_flight"]
            self.stats["uncertain"] += self.stats["in_flight"]
            conns = list(self._conns)
        self.stop.set()
        if getattr(self, "_timer", None) is not None and self._timer is not threading.current_thread():
            self._timer.cancel()
        if self.server is not None:
            self.server.shutdown()
            self.server.server_close()
        for conn in conns:
            try:
                conn.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        for client in (self.client, self.probe_client):
            try:
                client.close()
            except Exception:
                pass
        if self._monitor is not None and self._monitor is not threading.current_thread():
            self._monitor.join(timeout=self.probe_deadline + 2)

    def _track(self, conn, add):
        with self.lock:
            (self._conns.add if add else self._conns.discard)(conn)

    # --- the credentials in the answers ---------------------------------------------------------

    def reflects(self, data):
        """True when `data` (bytes or text of an answer) repeats a credential of the relay."""
        if not self._forms or not data:
            return False
        blob = data if isinstance(data, bytes) else data.encode("utf-8", "surrogatepass")
        if any(form in blob for form in self._forms):
            return True
        if blob.lstrip()[:1] not in (b"{", b"["):
            return False
        try:
            doc = json.loads(blob)
        except (ValueError, RecursionError):
            return False
        return any(value in text for text in _strings(doc) for value in self._plain)

    def _reflected(self):
        with self.lock:
            self.stats["reflected"] += 1

    # --- readiness and health ----------------------------------------------------------------

    def _probe(self, path, deadline=None):
        """(status, mark, json or None) of one GET with the probe client, within `deadline` seconds in
        all (S-B5): the answer is read as a stream, at most PROBE_LIMIT bytes, and closed past the limit
        or the term (`ProbeFailed`); TimeoutError when the caller stops waiting first, and the thread
        then closes the answer at its next chunk or its read timeout."""
        box, cancel = {}, threading.Event()
        seconds = self.probe_deadline if deadline is None else deadline
        started = self.clock()
        end = started + seconds
        headers = dict(self._out, **{"Accept-Encoding": "identity"})

        def run():
            try:
                with self.probe_client.stream("GET", path, headers=headers) as response:
                    chunks, size = [], 0
                    try:
                        for chunk in response.iter_raw():
                            size += len(chunk)
                            if size > PROBE_LIMIT:
                                raise ProbeFailed("oversize")
                            if cancel.is_set() or self.clock() >= end:
                                raise ProbeFailed("deadline")
                            chunks.append(chunk)
                        data = _decoded(b"".join(chunks), response.headers.get("Content-Encoding"), PROBE_LIMIT)
                    except self.httpx.StreamConsumed:
                        # an answer already in memory (a transport of the tests): its content, same limit
                        data = response.content
                        if len(data) > PROBE_LIMIT:
                            raise ProbeFailed("oversize") from None
                    box["value"] = (response.status_code, response.headers.get(PROXY_HEADER), data)
            except BaseException as exc:        # noqa: BLE001 - reported to the caller
                box["error"] = exc
        worker = threading.Thread(target=run, daemon=True)
        worker.start()
        worker.join(seconds)
        ms = (self.clock() - started) * 1000
        with self.lock:
            self.stats["probes"]["max_ms"] = round(max(self.stats["probes"]["max_ms"], ms), 1)
        if worker.is_alive():
            cancel.set()
            raise TimeoutError(path)
        if "error" in box:
            if isinstance(box["error"], ProbeFailed):
                with self.lock:
                    self.stats["probes"][box["error"].kind] += 1
            raise box["error"]
        status, mark, data = box["value"]
        try:
            body = json.loads(data) if data else None
        except ValueError:
            body = None
        return status, mark, body

    def ready(self, within=None):
        """Wait for the server (see the module docstring); returns the readiness record."""
        within = float(self.ep.ready_within if within is None else within)
        self._woke()
        started = self.clock()
        statuses = self.readiness["statuses"]
        while True:
            if self.stop.is_set():
                raise EngineError(Code.SERVER_UNAVAILABLE, "relay closed during the readiness")
            self.readiness["attempts"] += 1
            try:
                status, mark, body = self._probe("/v1/models")
            except Exception as exc:
                key = exc.kind if isinstance(exc, ProbeFailed) else type(exc).__name__
                statuses[key] = statuses.get(key, 0) + 1
            else:
                statuses[str(status)] = statuses.get(str(status), 0) + 1
                if status in (401, 403):
                    raise EngineError(Code.SERVER_REJECTED, "readiness answered %d" % status)
                if status == 200:
                    ids = [m.get("id") for m in (body or {}).get("data") or [] if isinstance(m, dict)]
                    if self.ep.model not in ids:
                        raise EngineError(Code.SERVER_MISMATCH, "the server does not serve %s" % self.ep.model)
                    self._completed(mark)
                    break
            if self.check() is not None:
                raise EngineError(self.failure, "the term of the attempt during the readiness")
            if self.clock() - started >= within:
                raise EngineError(Code.SERVER_UNAVAILABLE, "not ready within %d s" % within)
            self.stop.wait(self.ready_poll)
        self.ready_at = self.clock()
        self.readiness["wait_seconds"] = round(self.ready_at - started, 3)
        self._versions()
        if self._monitor is None and not self.stop.is_set():
            self._monitor = threading.Thread(target=self._watch, name="mdf-relay-health", daemon=True)
            self._monitor.start()
        return self.readiness

    def _versions(self):
        path, key = ("/version", "version") if self.ep.server == "vllm" else ("/props", "build_info")
        pin = self.ep.pins.get("vllm" if self.ep.server == "vllm" else "llama.cpp")
        try:
            status, mark, body = self._probe(path)
        except Exception:
            status, mark, body = None, None, None
        value = body.get(key) if isinstance(body, dict) and isinstance(body.get(key), str) else None
        if value is not None and self.reflects(value):
            self._reflected()
            value = None                              # never in `versions` nor in the detail below
        self.versions = {self.ep.server: value[:120] if value else None}
        if status == 200:
            self._completed(mark)
        if pin is not None and not (value and (value == pin or value.startswith(pin + "-")
                                               or value.startswith(pin + "+"))):
            raise EngineError(Code.SERVER_MISMATCH, "%s %r, pinned %s" % (self.ep.server, value, pin))

    def _completed(self, mark):
        """A response read in full: it moves `t_last` when it came from the container."""
        with self.lock:
            if not self.ep.costs or mark == "forwarded":
                self.t_last = self.clock()

    def _watch(self):
        failures = 0
        while not self.stop.wait(self.probe_interval):
            with self.lock:
                self.stats["probes"]["sent"] += 1
            try:
                status, mark, _ = self._probe("/v1/models")
                ok = status == 200
                if ok:
                    self._completed(mark)
            except Exception:
                ok = False
            if not ok:
                with self.lock:
                    self.stats["probes"]["failed"] += 1
                    if self.ep.costs:
                        self.stats["uncertain"] += 1
            failures = 0 if ok else failures + 1
            if failures >= self.probe_failures and not self.stop.is_set():
                self._fail(Code.SERVER_UNAVAILABLE)
                return

    # --- inbound -----------------------------------------------------------------------------

    def _reply(self, h, status, data=b"", content_type="application/json", close=False):
        h.send_response(status)
        h.send_header("Content-Type", content_type)
        h.send_header("Content-Length", str(len(data)))
        if close:
            h.send_header("Connection", "close")
            h.close_connection = True
        h.end_headers()
        if h.command != "HEAD":
            h.wfile.write(data)

    def _length(self, h, method):
        """The length of the body of the request, or (status, reason) of its refusal."""
        lengths = h.headers.get_all("Content-Length") or []
        chunked = h.headers.get_all("Transfer-Encoding") or []
        if (lengths and chunked) or len(lengths) > 1:
            return None, (400, "ambiguous framing")
        if lengths:
            text = lengths[0].strip()
            if not text.isdigit():
                return None, (400, "Content-Length")
            n = int(text)
        elif chunked:
            return None, (411 if method == "POST" else 400, "Transfer-Encoding without a length")
        elif method == "POST":
            return None, (411, "Content-Length required")
        else:
            n = 0                                 # a GET or HEAD without a body
        if method in ("GET", "HEAD") and n:
            return None, (400, "a body on a read")
        if n > BODY_LIMIT:
            return None, (413, "request over the limit")
        return n, None

    def _inbound(self, h, method):
        allowed = (method == "POST" and h.path == CHAT) or (method == "GET" and h.path in READ_PATHS)
        n, refusal = self._length(h, method)
        if refusal is not None:
            with self.lock:
                self.stats["framing_refused"] += 1
            return self._reply(h, refusal[0], json.dumps({"error": refusal[1]}).encode(), close=True)
        if not allowed:
            with self.lock:
                self.stats["refused_paths"] += 1
            if n:
                h.rfile.read(n)
            return self._reply(h, 404, b'{"error": "not forwarded"}')
        body = h.rfile.read(n) if n else b""
        if len(body) != n:
            return self._reply(h, 400, b'{"error": "body cut short"}', close=True)
        if self._past_wall():
            # MFL023 B2: the term on the wall clock, or an interval not observed, before this request: it does not
            # leave; the failure is noted here and the engine stops its child at its next `check`
            with self.lock:
                if self.failure is None:
                    self.failure = Code.SERVER_DEADLINE
                if self.state == "open":
                    self.state = "down"
        with self.lock:
            refused = self.state != "open"
        if refused:
            return self._reply(h, 503, b'{"error": "relay closed"}', close=True)
        status, data, content_type = self._forward(method, h.path, body, h.headers.get("Content-Type"))
        self._reply(h, status, data, content_type)

    def _forward(self, method, path, body, content_type):
        httpx = self.httpx
        self._woke()
        kind = "chat" if path == CHAT else "read"
        with self.lock:
            self.last_request = self.clock()
            self.stats["requests"][kind] += 1
            self.stats["in_flight"] += 1
            self.stats["max_in_flight"] = max(self.stats["max_in_flight"], self.stats["in_flight"])
            self.stats["bytes_in"] += len(body)
        started = self.clock()
        headers = dict(self._out, **{"Accept-Encoding": "identity"})
        if body:
            headers["Content-Type"] = "application/json" if not content_type else content_type[:100]
        outcome, status, data, ctype = "uncertain", 502, b'{"error": "relay"}', "application/json"
        try:
            request = self.client.build_request(method, path, content=body or None, headers=headers)
            response = self.client.send(request, stream=True)
            size, refused, payload = 0, None, b""
            try:
                decoded = Body(response.headers.get("Content-Encoding"), BODY_LIMIT)
                try:
                    for chunk in response.iter_raw():
                        size += len(chunk)
                        decoded.feed(chunk)
                    payload = decoded.finish()
                except self.httpx.StreamConsumed:
                    # an answer already in memory (a transport of the tests), as for the probes: its content, same limit
                    payload = response.content
                    size = len(payload)
                    if size > BODY_LIMIT:
                        raise ProbeFailed("oversize") from None
            except ProbeFailed as exc:
                refused = exc.kind
            finally:
                response.close()
            mark = response.headers.get(PROXY_HEADER)
            remote_type = response.headers.get("Content-Type") or ""
            reflected = refused is None and (self.reflects(payload) or self.reflects(remote_type))
            if reflected:
                self._reflected()
            if 300 <= response.status_code < 400:
                with self.lock:
                    self.stats["redirects"] += 1
                status, data = 502, b'{"error": "redirect not followed"}'
            elif refused is not None:
                with self.lock:
                    self.stats[refused] += 1
                status = 502
                data = b'{"error": "response over the limit"}' if refused == "oversize" else \
                    b'{"error": "response encoding"}'
                outcome = "complete" if (not self.ep.costs or mark == "forwarded") else "uncertain"
            else:
                status = response.status_code
                if not 200 <= status < 300:
                    data = _error_body(status)        # never the remote body of an error
                elif reflected:
                    status, data = 502, REFLECTED
                else:
                    data, ctype = payload, (remote_type or "application/json")[:100]
                if not self.ep.costs or mark == "forwarded":
                    outcome = "complete"
                elif mark == "error":
                    outcome = "error"
                else:
                    with self.lock:
                        self.stats["unmarked"] += 1
            with self.lock:
                self.stats["by_status"][str(response.status_code)] = self.stats["by_status"].get(
                    str(response.status_code), 0) + 1
                self.stats["bytes_out"] += size
        except httpx.TimeoutException:
            with self.lock:
                self.stats["timeouts"] += 1
            status, data = 504, b'{"error": "timeout"}'
        except Exception:                               # transport, or the relay closed under it
            with self.lock:
                if self.state != "closed":
                    self.stats["transport_errors"] += 1
            status, data = 502, b'{"error": "transport"}'
        finally:
            with self.lock:
                self.last_request = self.clock()
                self.stats["in_flight"] -= 1
                self.stats["seconds"] = round(self.stats["seconds"] + self.clock() - started, 3)
                if self.state == "closed" and self.t_close is not None and self.clock() >= self.t_close:
                    pass                                # counted by `close` as abandoned and uncertain
                elif outcome == "complete":
                    self.t_last = self.clock()
                elif outcome == "error":
                    self.stats["proxy_errors"] += 1
                else:
                    self.stats["uncertain"] += 1
        return status, data, ctype

    # --- the record --------------------------------------------------------------------------

    def clean(self):
        s = self.stats
        return (self.ready_at is not None and self.failure is None and s["in_flight_at_close"] == 0
                and s["uncertain"] == 0 and s["proxy_errors"] == 0)

    def record(self):
        """`marker_server` of `engine.json`: counters and times (epoch seconds), never a key."""
        with self.lock:
            stats = json.loads(json.dumps(self.stats))
        stats.pop("in_flight", None)
        return {**stats, "readiness": dict(self.readiness), "versions": dict(self.versions),
                "down": self.failure == Code.SERVER_UNAVAILABLE, "deadline_reached": self.failure == Code.SERVER_DEADLINE,
                "t_wake": self._epoch(self.t_wake), "t_last": self._epoch(self.t_last),
                "t_close": self._epoch(self.t_close), "clean": self.clean()}
