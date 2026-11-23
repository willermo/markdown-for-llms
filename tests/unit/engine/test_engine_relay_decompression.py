"""The answers the relay forwards to the Marker child (MFL046 d, plan r3 §3.6): asked with `Accept-Encoding: identity`,
read as received and decoded piece by piece within the limit, with no `flush()`; another encoding, a truncated stream or
bytes after its end are 502 (`encoding`), a bomb stops at the limit (`oversize`) with at most the limit plus one byte
and one piece in memory, a key inside a gzip body is seen (502, `reflected`), a normal gzip or zlib body reaches the
child decoded, and `bytes_out` counts the bytes received. The answers are real streams of a MockTransport."""
import gzip
import json
import unittest
import zlib

import httpx

from fake_inference import MODEL, PairTransport
from markdown_for_llms.engine import relay as relay_mod, servers, sockets
from markdown_for_llms.engine.envfile import Secret

KEY = "sk-test-0123456789"
URL = "http://127.0.0.1:9"                   # never connected: the fake is a transport
ANSWER = {"id": "fake", "object": "chat.completion", "created": 0, "model": MODEL,
          "choices": [{"index": 0, "message": {"role": "assistant", "content": "<p>ok</p>"}, "finish_reason": "stop",
                       "logprobs": None}], "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}}


def streamed(pieces, encoding=None, status=200):
    class Stream(httpx.SyncByteStream):
        def __iter__(self):
            yield from pieces
    headers = {"Content-Type": "application/json"}
    if encoding:
        headers["Content-Encoding"] = encoding
    return httpx.Response(status, stream=Stream(), headers=headers)


def chunks(data, size=4096):
    return [data[i:i + size] for i in range(0, len(data), size)]


class RelayDecompressionTests(unittest.TestCase):
    def setUp(self):
        self.private = sockets.Private()
        self.addCleanup(self.private.close)
        self.answers, self.seen = [], []

    def relay(self):
        ep = servers.endpoint("t", {"kind": "user", "url": URL, "key_env": "TEST_SERVER_KEY", "ready_within": 5})

        def handle(request):
            if request.url.path == "/v1/models":
                return httpx.Response(200, json={"object": "list", "data": [{"id": MODEL}]})
            if request.url.path == "/version":
                return httpx.Response(200, json={"version": "0.25.1"})
            self.seen.append(request.headers.get("Accept-Encoding"))
            return self.answers.pop(0)
        r = relay_mod.Relay(ep, {"TEST_SERVER_KEY": Secret(KEY)}, self.private.relay_socket,
                            transport_factory=lambda limits: httpx.MockTransport(handle), ready_poll=0.1)
        self.addCleanup(r.close)
        r.start()
        r.ready()
        client = httpx.Client(transport=PairTransport(r), base_url="http://relay", trust_env=False, timeout=30)
        self.addCleanup(client.close)
        return r, client

    def chat(self, client):
        return client.post("/v1/chat/completions", json={"model": MODEL, "messages": []})

    def test_the_answers_through_the_relay(self):
        body = json.dumps(ANSWER).encode()
        gz = gzip.compress(body)
        deflate = zlib.compress(body)
        raw_deflate = zlib.compressobj(wbits=-15)
        raw_deflate = raw_deflate.compress(body) + raw_deflate.flush()
        leaked = gzip.compress(json.dumps(dict(ANSWER, id=KEY)).encode())
        cases = [
            ("identity", [body], None, 200, None),
            ("gzip one byte at a time", [gz[i:i + 1] for i in range(len(gz))], "gzip", 200, None),
            ("deflate (zlib)", chunks(deflate, 7), "deflate", 200, None),
            ("br", [b"abc"], "br", 502, "encoding"),
            ("raw deflate without its header", [raw_deflate], "deflate", 502, "encoding"),
            ("truncated gzip", [gz[:-1]], "gzip", 502, "encoding"),
            ("empty gzip", [], "gzip", 502, "encoding"),
            ("bytes after the end, same piece", [gz + b"x"], "gzip", 502, "encoding"),
            ("bytes after the end, next piece", [gz, b"x"], "gzip", 502, "encoding"),
            ("two gzip members", [gz + gz], "gzip", 502, "encoding"),
            ("a key inside a gzip body", chunks(leaked, 5), "gzip", 502, "reflected"),
        ]
        r, client = self.relay()
        received = 0
        for what, pieces, encoding, status, counter in cases:
            with self.subTest(case=what):
                before = r.record()
                self.answers.append(streamed(pieces, encoding))
                reply = self.chat(client)
                received += sum(map(len, pieces)) if encoding != "br" else 0     # another encoding: never read
                self.assertEqual(reply.status_code, status, reply.text)
                if status == 200:
                    self.assertEqual(reply.json(), ANSWER)
                    self.assertNotIn("Content-Encoding", reply.headers)
                after = r.record()
                for name in ("encoding", "reflected", "oversize"):
                    self.assertEqual(after[name] - before[name], int(name == counter), name)
                self.assertNotIn(KEY.encode(), reply.content)
        self.assertEqual(set(self.seen), {"identity"})
        self.assertEqual(r.record()["bytes_out"], received)

    def test_a_bomb_stops_at_the_limit(self):
        bomb = gzip.compress(b"\0" * (relay_mod.BODY_LIMIT + (8 << 20)))
        self.assertLess(len(bomb), 1 << 20)
        r, client = self.relay()
        pieces = chunks(bomb)
        self.answers.append(streamed(pieces, "gzip"))
        reply = self.chat(client)
        self.assertEqual(reply.status_code, 502)
        record = r.record()
        self.assertEqual((record["oversize"], record["encoding"]), (1, 0))
        self.assertLess(record["bytes_out"], len(bomb))               # the stream is closed at the limit


class BodyTests(unittest.TestCase):
    """`relay.Body` alone: what it keeps in memory with a small limit."""

    def test_memory_within_the_limit_plus_one_piece(self):
        limit = 1_000_000
        bomb = gzip.compress(b"\0" * 50_000_000)
        body = relay_mod.Body("gzip", limit)
        largest = 0
        with self.assertRaises(relay_mod.ProbeFailed) as ctx:
            for piece in chunks(bomb):
                body.feed(piece)
                largest = max(largest, body.size)
        self.assertEqual(ctx.exception.kind, "oversize")
        self.assertLessEqual(sum(map(len, body.parts)), limit)
        self.assertEqual(body.size, limit + 1)                  # the decoder never gave more than one byte over
        self.assertLessEqual(largest, limit)

    def test_exactly_the_limit(self):
        data = b"a" * 1000
        body = relay_mod.Body("gzip", 1000)
        body.feed(gzip.compress(data))
        self.assertEqual(body.finish(), data)
        over = relay_mod.Body("gzip", 999)
        with self.assertRaises(relay_mod.ProbeFailed):
            over.feed(gzip.compress(data))
        plain = relay_mod.Body(None, 3)
        plain.feed(b"abc")
        with self.assertRaises(relay_mod.ProbeFailed):
            plain.feed(b"d")


if __name__ == "__main__":
    unittest.main()
