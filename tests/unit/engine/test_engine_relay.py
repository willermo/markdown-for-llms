"""The relay of the `marker-server` route (MFL033): paths, framing, keys, readiness, health,
term, outcomes. A fake server on the loopback stands for vLLM behind its proxy; the client side
is real `httpx`. The suite forbids `socket.connect` in its process (`tests/conftest.py`): the
fake is an `httpx.MockTransport` behind the relay, and the side of the child reaches the relay's
own server through a `socketpair` (`fake_inference.PairTransport`)."""
import json
import os
import threading
import time
import unittest

from fake_inference import Fake, MODEL, PairTransport, raw
from markdown_for_llms.engine import relay as relay_mod, servers, sockets
from markdown_for_llms.engine.codes import Code, EngineError
from markdown_for_llms.engine.envfile import Secret

KEY = "sk-test-0123456789"


URL = "http://127.0.0.1:9"                   # never connected: the fake is a transport


def endpoint(url=URL, kind="user", **extra):
    raw = {"kind": kind, "url": url, "key_env": "TEST_SERVER_KEY", "ready_within": 5, **extra}
    if kind in ("runpod", "modal"):
        raw.pop("key_env")
        raw.setdefault("rate_usd_per_s", "0.0002")
        raw.update(ready_within=900)
    return servers.endpoint("t", raw)


def keys_for(ep):
    return {name: Secret(KEY) for name in ep.key_env}


class RelayCase(unittest.TestCase):
    def setUp(self):
        self.private = sockets.Private()
        self.addCleanup(self.private.close)

    def fake(self, **kw):
        fake = Fake(**kw)
        self.addCleanup(fake.close)
        return fake

    def relay(self, ep, fake, keys=None, **kw):
        r = relay_mod.Relay(ep, keys_for(ep) if keys is None else keys, self.private.relay_socket,
                            transport_factory=fake.factory(), **kw)
        self.addCleanup(r.close)
        self.relay_ = r
        return r.start()

    def child(self):
        import httpx
        client = httpx.Client(transport=PairTransport(self.relay_), base_url="http://relay", trust_env=False, timeout=30)
        self.addCleanup(client.close)
        return client


class TestPaths(RelayCase):
    def test_reads_and_the_chat_of_surya_cross_the_relay(self):
        """Prescription 1 of the review of r4: GET without a body is length 0; the probes and the
        POST of Surya reach the server; the key is the relay's, never the child's."""
        from PIL import Image
        from openai import OpenAI
        import httpx
        from surya.inference.backends.openai_client import chat_completions_batch
        from surya.inference.schema import BatchInputItem
        fake = self.fake(key=KEY)
        r = self.relay(endpoint(), fake)
        r.ready()
        c = self.child()
        for path in ("/v1/models", "/health", "/metrics", "/version"):
            self.assertEqual(c.get(path, headers={"Authorization": "Bearer not-the-key"}).status_code, 200, path)
        client = OpenAI(api_key="not-the-key", base_url="http://relay/v1", max_retries=0,
                        http_client=httpx.Client(transport=PairTransport(r), trust_env=False, timeout=30))
        batch = [BatchInputItem(image=Image.new("RGB", (64, 32), "white"), prompt_type="block") for _ in range(6)]
        outs = chat_completions_batch(batch, client=client, model_name=MODEL, max_workers=3, max_retries=0)
        self.assertEqual([o.raw for o in outs], ["<p>ok</p>"] * 6)
        self.assertEqual([o.error for o in outs], [False] * 6)
        for method, path, length, chunked in fake.seen:
            self.assertIsNone(chunked)
            if method == "GET":
                self.assertIsNone(length, path)
            else:
                self.assertGreater(int(length), 0)
        self.assertEqual(set(fake.auth), {"Bearer " + KEY})
        self.assertTrue(all(a.startswith("markdown-for-llms/") for a in fake.agents))
        self.assertEqual(fake.chat, 6)
        rec = r.record()
        self.assertEqual(rec["requests"], {"chat": 6, "read": 4})
        self.assertEqual(rec["uncertain"], 0)
        self.assertEqual(rec["versions"], {"vllm": "0.25.1"})
        self.assertNotIn(KEY, json.dumps(rec))

    def test_framing_and_paths_refused_before_the_server(self):
        fake = self.fake()
        r = self.relay(endpoint(key_env=None), fake)
        r.ready()
        sock = r
        before = len(fake.seen)
        self.assertEqual(raw(sock, b"POST /v1/chat/completions HTTP/1.1\r\nHost: r\r\n\r\n"), 411)
        self.assertEqual(raw(sock, b"POST /v1/chat/completions HTTP/1.1\r\nHost: r\r\nTransfer-Encoding: chunked"
                                           b"\r\n\r\n0\r\n\r\n"), 411)
        self.assertEqual(raw(sock, b"POST /v1/chat/completions HTTP/1.1\r\nHost: r\r\nContent-Length: 2\r\n"
                                           b"Transfer-Encoding: chunked\r\n\r\n{}"), 400)
        self.assertEqual(raw(sock, b"POST /v1/chat/completions HTTP/1.1\r\nHost: r\r\nContent-Length: 2\r\n"
                                           b"Content-Length: 2\r\n\r\n{}"), 400)
        self.assertEqual(raw(sock, b"POST /v1/chat/completions HTTP/1.1\r\nHost: r\r\nContent-Length: 30000001"
                                           b"\r\n\r\n"), 413)
        self.assertEqual(raw(sock, b"GET /v1/models HTTP/1.1\r\nHost: r\r\nContent-Length: 3\r\n\r\nabc"), 400)
        self.assertEqual(raw(sock, b"GET /v1/models HTTP/1.1\r\nHost: r\r\n\r\n"), 200)
        self.assertEqual(raw(sock, b"HEAD /v1/models HTTP/1.1\r\nHost: r\r\n\r\n"), 404)
        self.assertEqual(raw(sock, b"GET /admin HTTP/1.1\r\nHost: r\r\n\r\n"), 404)
        self.assertEqual(raw(sock, b"POST /v1/completions HTTP/1.1\r\nHost: r\r\nContent-Length: 2\r\n\r\n{}"), 404)
        self.assertEqual(fake.paths()[before:], ["/v1/models"])
        rec = r.record()
        self.assertEqual(rec["framing_refused"], 6)
        self.assertEqual(rec["refused_paths"], 3)

    def test_an_answer_over_the_limit_is_502(self):
        fake = self.fake()
        r = self.relay(endpoint(key_env=None), fake)
        r.ready()
        fake.oversize = relay_mod.BODY_LIMIT + 1
        reply = self.child().post("/v1/chat/completions", json={"model": MODEL, "messages": []})
        self.assertEqual(reply.status_code, 502)
        self.assertEqual(r.record()["oversize"], 1)

    def test_a_redirect_is_not_followed(self):
        fake = self.fake()
        r = self.relay(endpoint(key_env=None), fake)
        r.ready()
        fake.fail_next, fake.fail_status = 1, 303
        reply = self.child().post("/v1/chat/completions", json={"model": MODEL, "messages": []})
        self.assertEqual(reply.status_code, 502)
        self.assertEqual(r.record()["redirects"], 1)

    def test_a_socket_path_over_107_bytes_is_refused_before_bind(self):
        ep = endpoint(key_env=None)
        with self.assertRaises(EngineError) as caught:
            relay_mod.Relay(ep, {}, "/tmp/" + "x" * 110 + "/relay.sock")
        self.assertEqual(caught.exception.code, Code.SOCKET_PATH_TOO_LONG)


class TestReadiness(RelayCase):
    def check(self, code, ready_within=3, **fake_kw):
        fake = self.fake(**{k: v for k, v in fake_kw.items() if k in ("key", "model", "version")})
        if "models_status" in fake_kw:
            fake.models_status = fake_kw["models_status"]
        extra = {"pins": fake_kw["pins"]} if "pins" in fake_kw else {}
        if fake_kw.get("wrong_key"):
            r = self.relay(endpoint(ready_within=ready_within), fake, keys={"TEST_SERVER_KEY": Secret("wrong")},
                           ready_poll=0.2)
        else:
            r = self.relay(endpoint(key_env=None, ready_within=ready_within, **extra), fake, ready_poll=0.2)
        with self.assertRaises(EngineError) as caught:
            r.ready()
        self.assertEqual(caught.exception.code, code)
        return r

    def test_rejected_mismatched_unavailable(self):
        self.check(Code.SERVER_REJECTED, key="k", wrong_key=True)

    def test_another_model(self):
        self.check(Code.SERVER_MISMATCH, model="someone/else")

    def test_another_version_than_the_pin(self):
        self.check(Code.SERVER_MISMATCH, version="0.24.0", pins={"vllm": "0.25.1"})

    def test_never_ready(self):
        started = time.monotonic()
        r = self.check(Code.SERVER_UNAVAILABLE, ready_within=1, models_status=503)
        self.assertLess(time.monotonic() - started, 5)
        self.assertGreaterEqual(r.record()["readiness"]["statuses"]["503"], 2)


class TestHealthAndTerm(RelayCase):
    def test_a_server_that_stops_answering_puts_the_relay_down(self):
        fake = self.fake()
        r = self.relay(endpoint(key_env=None), fake, probe_interval=0.2, probe_deadline=0.3)
        r.ready()
        c = self.child()
        self.assertEqual(c.get("/v1/models").status_code, 200)
        fake.models_hang = True
        started = time.monotonic()
        while r.check() is None and time.monotonic() - started < 10:
            time.sleep(0.05)
        self.assertEqual(r.check(), Code.SERVER_UNAVAILABLE)
        self.assertLess(time.monotonic() - started, 3 * (0.2 + 0.3) + 1.5)
        fake.models_hang = False
        self.assertEqual(c.post("/v1/chat/completions", json={"model": MODEL}).status_code, 503)
        self.assertTrue(r.record()["down"])

    def test_probes_have_their_own_pool(self):
        """32 slow inferences fill the pool of the inferences; every probe still answers in time."""
        fake = self.fake(delay=1.5)
        r = self.relay(endpoint(key_env=None, parallel=32), fake, probe_interval=0.2, probe_deadline=1.0)
        r.ready()
        c = self.child()
        threads = [threading.Thread(target=c.post, args=("/v1/chat/completions",), kwargs={"json": {"model": MODEL}})
                   for _ in range(32)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        rec = r.record()
        self.assertGreaterEqual(rec["probes"]["sent"], 3)
        self.assertEqual(rec["probes"]["failed"], 0)
        self.assertEqual(rec["max_in_flight"], 32)
        self.assertIsNone(r.check())

    def test_the_term_closes_the_relay(self):
        now = [1000.0]
        fake = self.fake()
        r = self.relay(endpoint(key_env=None), fake, clock=lambda: now[0])
        r.set_term(30)
        r.ready()
        now[0] += 24.0
        self.assertIsNone(r.check())
        now[0] += 1.0
        self.assertEqual(r.check(), Code.SERVER_DEADLINE)
        self.assertEqual(r.state, "closed")
        rec = r.record()
        self.assertTrue(rec["deadline_reached"])
        self.assertFalse(rec["clean"])
        self.assertLessEqual(rec["t_close"] - rec["t_wake"], 30 - 5 + 0.01)


class TestOutcomes(RelayCase):
    def provider(self, mark):
        fake = self.fake(mark=mark)
        r = self.relay(endpoint(kind="runpod"), fake)
        r.ready()
        return fake, r

    def test_forwarded_answers_are_complete_and_the_close_is_clean(self):
        fake, r = self.provider("forwarded")
        self.assertEqual(self.child().post("/v1/chat/completions", json={"model": MODEL}).status_code, 200)
        r.close()
        rec = r.record()
        self.assertTrue(rec["clean"])
        self.assertEqual((rec["uncertain"], rec["proxy_errors"]), (0, 0))
        self.assertGreaterEqual(rec["t_last"], rec["t_wake"])

    def test_an_error_of_the_proxy_is_not_clean(self):
        fake, r = self.provider(lambda path, status: "error" if status >= 500 else "forwarded")
        fake.fail_next = 1
        self.assertEqual(self.child().post("/v1/chat/completions", json={"model": MODEL}).status_code, 500)
        r.close()
        rec = r.record()
        self.assertEqual(rec["proxy_errors"], 1)
        self.assertFalse(rec["clean"])

    def test_an_answer_of_the_edge_is_uncertain(self):
        fake, r = self.provider(lambda path, status: None if path == "/v1/chat/completions" else "forwarded")
        self.child().post("/v1/chat/completions", json={"model": MODEL})
        r.close()
        rec = r.record()
        self.assertEqual((rec["uncertain"], rec["unmarked"]), (1, 1))
        self.assertFalse(rec["clean"])

    def test_without_a_provider_the_mark_is_not_asked(self):
        fake = self.fake()
        r = self.relay(endpoint(key_env=None), fake)
        r.ready()
        self.child().post("/v1/chat/completions", json={"model": MODEL})
        r.close()
        self.assertEqual(r.record()["uncertain"], 0)
        self.assertTrue(r.record()["clean"])

    def test_a_request_in_flight_at_the_close_is_abandoned(self):
        fake, r = self.provider("forwarded")
        fake.chat_hang = True
        box, client = {}, self.child()

        def send():
            try:
                box["reply"] = client.post("/v1/chat/completions", json={"model": MODEL}).status_code
            except Exception as exc:            # the relay closed the connection of the child
                box["reply"] = type(exc).__name__
        t = threading.Thread(target=send)
        t.start()
        started = time.monotonic()
        while fake.in_flight == 0 and time.monotonic() - started < 5:
            time.sleep(0.02)
        r.close()
        t.join(10)
        self.assertEqual(box["reply"], "RemoteProtocolError")
        rec = r.record()
        self.assertEqual(rec["in_flight_at_close"], 1)
        self.assertEqual(rec["abandoned"], 1)
        self.assertFalse(rec["clean"])


class TestNoKeyInTheRecord(RelayCase):
    def test_repr_and_record(self):
        fake = self.fake(key=KEY)
        ep = endpoint()
        r = self.relay(ep, fake)
        r.ready()
        self.assertNotIn(KEY, repr(keys_for(ep)))
        self.assertNotIn(KEY, json.dumps(r.record()))
        self.assertNotIn(KEY, json.dumps(ep.record()))
        self.assertEqual(os.stat(self.private.relay_socket).st_uid, os.getuid())


if __name__ == "__main__":
    unittest.main()


class TestTermWithoutPolling(RelayCase):
    def test_the_relay_closes_at_its_term_by_itself(self):
        fake = self.fake()
        r = self.relay(endpoint(key_env=None), fake)
        r.set_term(5.6)                                  # closes 0.6 s after the wake
        r.ready()
        time.sleep(1.5)
        self.assertEqual(r.state, "closed")
        self.assertEqual(r.failure, Code.SERVER_DEADLINE)
        rec = r.record()
        self.assertLessEqual(rec["t_close"] - rec["t_wake"], 5.6 - 5 + 0.3)


class _Answers:
    """A server whose chat answers are the given bodies, one each (for the forms a fake does not write)."""

    def __init__(self, bodies):
        self.bodies = list(bodies)

    def factory(self):
        import httpx

        def handle(request):
            if request.url.path == "/v1/models":
                return httpx.Response(200, json={"object": "list", "data": [{"id": MODEL}]})
            if request.url.path == "/version":
                return httpx.Response(200, json={"version": "0.25.1"})
            return httpx.Response(200, content=self.bodies.pop(0), headers={"Content-Type": "application/json"})
        return lambda limits: httpx.MockTransport(handle)

    def close(self):
        pass


class TestReflectedCredentials(RelayCase):
    """S-B1 of the final review of MFL033: no credential of the relay reaches the child, the record or a
    refusal, also when the server repeats it in an answer; the counters of the statuses stay the server's."""

    def assert_clean(self, reply, *secrets):
        shown = reply.content.decode("utf-8", "replace") + " ".join("%s: %s" % kv for kv in reply.headers.items())
        for secret in secrets or (KEY,):
            self.assertNotIn(secret, shown)

    def reflecting(self, **kw):
        fake = self.fake(key=KEY)
        fake.reflect = True
        r = self.relay(endpoint(**kw), fake)
        r.ready()
        return fake, r

    def test_an_error_reaches_the_child_with_its_status_and_a_reason_of_the_relay(self):
        fake, r = self.reflecting()
        c = self.child()
        for status in (401, 500, 503):
            with self.subTest(status=status):
                fake.script = [status]
                reply = c.post("/v1/chat/completions", json={"model": MODEL, "messages": []})
                self.assertEqual(reply.status_code, status)
                self.assertEqual(reply.json()["error"]["message"], "relay: the server answered %d" % status)
                self.assertNotIn("X-Echo", reply.headers)
                self.assert_clean(reply)
        rec = r.record()
        self.assertEqual({k: rec["by_status"][k] for k in ("401", "500", "503")}, {"401": 1, "500": 1, "503": 1})
        self.assertEqual(rec["reflected"], 3)
        self.assertNotIn(KEY, json.dumps(rec))

    def test_a_2xx_that_repeats_the_key_reaches_the_child_as_502(self):
        fake, r = self.reflecting()
        c = self.child()
        fake.script = [200]
        reply = c.post("/v1/chat/completions", json={"model": MODEL, "messages": []})
        self.assertEqual(reply.status_code, 502)
        self.assert_clean(reply)
        reply = c.post("/v1/chat/completions", json={"model": MODEL, "messages": []})      # the next one is clean
        self.assertEqual((reply.status_code, reply.json()["choices"][0]["message"]["content"]), (200, "<p>ok</p>"))
        self.assertNotIn("X-Echo", reply.headers)                      # no remote header but its type
        self.assert_clean(reply)
        rec = r.record()
        self.assertEqual((rec["by_status"], rec["reflected"]), ({"200": 2}, 1))     # the server's statuses
        self.assertNotIn(KEY, json.dumps(rec))

    def test_the_forms_of_a_repeated_key(self):
        import base64
        from urllib.parse import quote
        slashed = KEY + "/+="
        escaped = "".join("\\u%04x" % ord(ch) for ch in slashed)
        forms = {"json escapes": b'{"choices": [{"message": {"content": "%s"}}]}' % escaped.encode(),
                 "a key of the JSON": b'{"%s": 1}' % escaped.encode(),
                 "slash escaped": json.dumps({"c": slashed}).replace("/", "\\/").encode(),
                 "percent": quote(slashed, safe="").encode(),
                 "base64": base64.b64encode(slashed.encode()),
                 "base64 url": base64.urlsafe_b64encode(slashed.encode()).rstrip(b"=")}
        answers = _Answers(forms.values())
        r = relay_mod.Relay(endpoint(), {"TEST_SERVER_KEY": Secret(slashed)}, self.private.relay_socket,
                            transport_factory=answers.factory())
        self.addCleanup(r.close)
        self.relay_ = r.start()
        r.ready()
        c = self.child()
        for name in forms:
            with self.subTest(form=name):
                reply = c.post("/v1/chat/completions", json={"model": MODEL})
                self.assertEqual(reply.status_code, 502)
                self.assert_clean(reply, slashed)
        self.assertEqual(r.record()["reflected"], len(forms))

    def test_a_key_in_the_content_type(self):
        fake, r = self.reflecting()
        fake.reflect_type = True
        c = self.child()
        for path in ("/v1/models", "/v1/chat/completions"):
            with self.subTest(path=path):
                reply = c.get(path) if path != relay_mod.CHAT else c.post(path, json={"model": MODEL})
                self.assertEqual(reply.status_code, 502)
                self.assert_clean(reply)
        self.assertEqual(r.record()["reflected"], 2)

    def test_the_version_of_the_server_passes_the_same_filter(self):
        for pins in ({}, {"vllm": "0.25.1"}):
            with self.subTest(pins=pins):
                self.private = sockets.Private()                 # a socket of its own for each relay
                self.addCleanup(self.private.close)
                fake = self.fake(key=KEY, version="0.25.1-" + KEY)
                r = self.relay(endpoint(pins=pins), fake)
                if pins:
                    with self.assertRaises(EngineError) as caught:
                        r.ready()
                    self.assertEqual(caught.exception.code, Code.SERVER_MISMATCH)
                    self.assertNotIn(KEY, str(caught.exception) + repr(caught.exception.detail))
                else:
                    r.ready()
                self.assertEqual(r.versions, {"vllm": None})
                self.assertNotIn(KEY, json.dumps(r.record()))
                r.close()

    def test_the_two_keys_of_modal(self):
        keys = {"MODAL_PROXY_TOKEN_ID": Secret("wk-reflect-test-id"), "MODAL_PROXY_TOKEN_SECRET": Secret("ws-reflect-test-secret")}
        fake = self.fake(mark="forwarded")
        fake.reflect = True
        r = self.relay(endpoint(kind="modal"), fake, keys=keys)
        r.ready()
        c = self.child()
        fake.script = [500, 200]
        statuses = [c.post("/v1/chat/completions", json={"model": MODEL}) for _ in range(2)]
        self.assertEqual([s.status_code for s in statuses], [500, 502])
        for reply in statuses:
            self.assert_clean(reply, "wk-reflect-test-id", "ws-reflect-test-secret")
        self.assertEqual(r.record()["reflected"], 2)
        self.assertNotIn("reflect-test", json.dumps(r.record()))


PROBE_LIMIT = 1 << 20                      # the limit of a probe of the relay (S-B5)
CHUNK = 64 << 10
SAFETY_STOP = 128 << 20                    # a body "without end" stops here: never reached within the limit


class _Body:
    """A body produced in chunks of 64 KiB, `total` bytes or without end (the test stops it at
    SAFETY_STOP, which a relay within its limit never reaches); counts what was produced."""

    def __init__(self, total=None, delay=0.0):
        self.total, self.delay, self.produced = total, delay, 0

    def stream(self):
        import httpx
        body = self

        class Stream(httpx.SyncByteStream):
            def __iter__(self):
                chunk = b"x" * (64 << 10)
                while (body.total is None or body.produced < body.total) and body.produced < SAFETY_STOP:
                    if body.delay:
                        time.sleep(body.delay)
                    body.produced += len(chunk)
                    yield chunk
        return Stream()


class TestBoundedProbes(RelayCase):
    """S-B5 of the final review of MFL033: readiness, version and health read their answer as a stream,
    within PROBE_LIMIT bytes and their term, and close it past them with a declared outcome."""

    def relay_with(self, models, version=None, **kw):
        import httpx

        def handle(request):
            if request.url.path == "/v1/models":
                return models(request)
            if request.url.path == "/version":
                return version(request) if version else httpx.Response(200, json={"version": "0.25.1"})
            return httpx.Response(404, json={})
        r = relay_mod.Relay(endpoint(key_env=None, **kw.pop("ep", {})), {}, self.private.relay_socket,
                            transport_factory=lambda limits: httpx.MockTransport(handle), ready_poll=0.1, **kw)
        self.addCleanup(r.close)
        self.relay_ = r
        return r.start()

    def ok_models(self, request):
        import httpx
        return httpx.Response(200, json={"object": "list", "data": [{"id": MODEL}]})

    def test_an_endless_and_a_32_mib_answer_of_the_readiness(self):
        import httpx
        for name, total in (("endless", None), ("32 MiB", 32 << 20)):
            with self.subTest(body=name):
                self.private = sockets.Private()
                self.addCleanup(self.private.close)
                bodies = []

                def models(request):
                    bodies.append(_Body(total))
                    return httpx.Response(200, stream=bodies[-1].stream())
                r = self.relay_with(models)
                started = time.monotonic()
                with self.assertRaises(EngineError) as caught:
                    r.ready(within=0.5)
                self.assertEqual(caught.exception.code, Code.SERVER_UNAVAILABLE)
                self.assertLess(time.monotonic() - started, 10)
                self.assertGreaterEqual(len(bodies), 1)
                time.sleep(0.3)                            # a probe thread still reading would go on
                for body in bodies:
                    self.assertLessEqual(body.produced, PROBE_LIMIT + CHUNK)
                rec = r.record()
                self.assertGreaterEqual(rec["readiness"]["statuses"]["oversize"], 1)
                self.assertEqual(rec["probes"]["oversize"], len(bodies))
                r.close()

    def test_an_answer_past_the_term_of_the_probe_is_closed(self):
        import httpx
        body = _Body(None, delay=0.2)
        r = self.relay_with(lambda request: httpx.Response(200, stream=body.stream()), probe_deadline=0.6)
        started = time.monotonic()
        with self.assertRaises((getattr(relay_mod, "ProbeFailed", TimeoutError), TimeoutError)):
            r._probe("/v1/models")
        self.assertLess(time.monotonic() - started, 0.6 + 0.5)
        time.sleep(0.6)                                    # the thread closes the answer at its next chunk
        produced = body.produced
        time.sleep(0.6)
        self.assertEqual(body.produced, produced)
        self.assertLessEqual(produced, 6 * CHUNK)

    def test_the_version_and_the_health_are_bounded_too(self):
        import httpx
        bodies = []

        def endless(request):
            bodies.append(_Body(None))
            return httpx.Response(200, stream=bodies[-1].stream())
        r = self.relay_with(self.ok_models, version=endless, ep={"pins": {"vllm": "0.25.1"}})
        with self.assertRaises(EngineError) as caught:
            r.ready()
        self.assertEqual(caught.exception.code, Code.SERVER_MISMATCH)
        self.assertEqual(r.versions, {"vllm": None})
        self.assertLessEqual(bodies[0].produced, PROBE_LIMIT + CHUNK)
        # the health: readiness passes, then every probe of /v1/models is endless
        self.private = sockets.Private()
        self.addCleanup(self.private.close)
        state = {"endless": False}

        def models(request):
            if state["endless"]:
                return endless(request)
            return self.ok_models(request)
        r = self.relay_with(models, probe_interval=0.1, probe_deadline=1.0)
        r.ready()
        state["endless"] = True
        started = time.monotonic()
        while r.check() is None and time.monotonic() - started < 15:
            time.sleep(0.05)
        self.assertEqual(r.check(), Code.SERVER_UNAVAILABLE)
        self.assertGreaterEqual(r.record()["probes"]["oversize"], 3)
        self.assertTrue(all(b.produced <= PROBE_LIMIT + CHUNK for b in bodies))

    def test_the_limit_of_the_probes(self):
        self.assertEqual(relay_mod.PROBE_LIMIT, PROBE_LIMIT)

    def test_a_compressed_answer_is_decoded_within_the_limit(self):
        import gzip
        import httpx
        def streamed(data, encoding):
            class Stream(httpx.SyncByteStream):
                def __iter__(self):
                    yield data
            return httpx.Response(200, stream=Stream(), headers={"Content-Encoding": encoding})
        small = gzip.compress(json.dumps({"object": "list", "data": [{"id": MODEL}]}).encode())
        bomb = gzip.compress(b"\0" * (32 << 20))             # 32 KiB as received
        self.assertLess(len(bomb), PROBE_LIMIT)
        answers = [streamed(bomb, "gzip"), streamed(b"abc", "br"), streamed(small, "gzip")]
        seen = []

        def models(request):
            seen.append(request.headers.get("Accept-Encoding"))
            return answers.pop(0) if answers else streamed(small, "gzip")
        r = self.relay_with(models)
        r.ready()
        statuses = r.record()["readiness"]["statuses"]
        self.assertEqual((statuses["oversize"], statuses["encoding"], statuses["200"]), (1, 1, 1))
        self.assertEqual(set(seen), {"identity"})
