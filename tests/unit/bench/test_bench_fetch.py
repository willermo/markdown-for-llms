"""Fetch with a fake transport: verification, quarantine, sessions, Wikisource and U6."""
import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from markdown_for_llms.bench import cli as bench_cli, fetch, manifest
from markdown_for_llms.bench.cache import Cache
from markdown_for_llms.bench.codes import BenchError, Code
from markdown_for_llms.bench.fetch import Response


def sha(data):
    return hashlib.sha256(data).hexdigest()


class FakeTransport:
    """Answers by URL; records (session, url, params) of every GET. No socket is opened."""

    def __init__(self, routes):
        self.routes, self.calls, self.sessions = routes, [], 0

    def new_session(self):
        self.sessions += 1
        return f"session-{self.sessions}"

    def get(self, session, url, params=None, limit=0):
        self.calls.append((session, url, dict(params or {})))
        answer = self.routes[url]
        if callable(answer):
            answer = answer(params or {})
        status, body = answer
        return Response(status, "application/octet-stream", body)


DATA = b"%PDF-1.4 synthetic source\n"
SRC = {"kind": "file", "url": "https://example.org/a.pdf", "bytes": len(DATA), "sha256": sha(DATA)}


class FileSourceTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.cache = Cache(Path(tmp.name) / "cache")

    def quarantined(self):
        return sorted(p.name for p in self.cache.quarantine_dir.glob("*")) if self.cache.quarantine_dir.exists() else []

    def test_download_then_cache_without_network(self):
        t = FakeTransport({SRC["url"]: (200, DATA)})
        path = fetch.fetch_file("a", SRC, self.cache, t)
        self.assertEqual(path.read_bytes(), DATA)
        self.assertEqual(fetch.fetch_file("a", SRC, self.cache, None, offline=True), path)
        self.assertEqual(len(t.calls), 1)

    def test_failures_are_closed_codes_and_mismatches_go_to_quarantine(self):
        cases = [((404, b""), Code.SOURCE_UNREACHABLE, 0),
                 ((200, DATA + b"x"), Code.SOURCE_SIZE_MISMATCH, 1),
                 ((200, DATA.replace(b"1", b"2")), Code.SOURCE_HASH_MISMATCH, 1)]
        for answer, code, quarantined in cases:
            with self.subTest(code=code):
                before = len(self.quarantined())
                with self.assertRaises(BenchError) as ctx:
                    fetch.fetch_file("a", SRC, self.cache, FakeTransport({SRC["url"]: answer}))
                self.assertEqual(ctx.exception.code, code)
                self.assertEqual(len(self.quarantined()) - before, quarantined)
                self.assertFalse(self.cache.source_path(SRC).exists())

    def test_altered_cache_offline_is_unavailable_and_online_is_downloaded_again(self):
        fetch.fetch_file("a", SRC, self.cache, FakeTransport({SRC["url"]: (200, DATA)}))
        self.cache.source_path(SRC).write_bytes(DATA.replace(b"P", b"Q"))
        with self.assertRaises(BenchError) as ctx:
            fetch.fetch_file("a", SRC, self.cache, None, offline=True)
        self.assertEqual(ctx.exception.code, Code.SOURCE_HASH_MISMATCH)
        self.assertEqual(len(self.quarantined()), 1)
        with self.assertRaises(BenchError) as ctx:
            fetch.fetch_file("a", SRC, self.cache, None, offline=True)
        self.assertEqual(ctx.exception.code, Code.SOURCE_UNREACHABLE)
        path = fetch.fetch_file("a", SRC, self.cache, FakeTransport({SRC["url"]: (200, DATA)}))
        self.assertEqual(path.read_bytes(), DATA)

    def test_session_page_first_with_the_same_session(self):
        src = dict(SRC, session_url="https://example.org/act")
        t = FakeTransport({"https://example.org/act": (200, b"<html>"), SRC["url"]: (200, DATA)})
        fetch.fetch_file("a", src, self.cache, t)
        self.assertEqual([(s, u) for s, u, _ in t.calls], [("session-1", src["session_url"]), ("session-1", SRC["url"])])
        t = FakeTransport({"https://example.org/act": (503, b""), SRC["url"]: (200, DATA)})
        with self.assertRaises(BenchError) as ctx:
            fetch.fetch_file("b", dict(src, sha256=sha(b"other"), bytes=5), self.cache, t)
        self.assertEqual(ctx.exception.code, Code.SESSION_FAILED)


def sha1(text):
    return hashlib.sha1(text.encode()).hexdigest()


class WikisourceTests(unittest.TestCase):
    API = "https://it.wikisource.org/w/api.php"

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.cache = Cache(Path(tmp.name) / "cache")
        self.texts = {101: "Pagina uno, α e β.", 102: "Pagina due<ref>nota</ref>."}
        self.src = {"kind": "wikisource", "api": self.API, "index": "X.djvu",
                    "pages": [{"page": 1, "revid": 101, "sha1": sha1(self.texts[101]), "quality": 4},
                              {"page": 2, "revid": 102, "sha1": sha1(self.texts[102]), "quality": 4}]}
        patcher = patch.object(fetch, "WIKI_PAUSE_S", 0)
        patcher.start()
        self.addCleanup(patcher.stop)

    def api(self, texts=None, sha_override=None):
        texts = texts or self.texts

        def answer(params):
            if params["action"] == "query":
                revs = [int(r) for r in params["revids"].split("|")]
                pages = [{"revisions": [{"revid": r, "sha1": (sha_override or sha1(texts[r])),
                                         "slots": {"main": {"content": texts[r]}}}]} for r in revs]
                return 200, json.dumps({"query": {"pages": pages}}).encode()
            return 200, json.dumps({"parse": {"text": f"<p>{texts[int(params['oldid'])]}</p>"}}).encode()
        return FakeTransport({self.API: answer})

    def test_revisions_are_verified_and_cached(self):
        t = self.api()
        pages = fetch.fetch_wikisource("w", self.src, self.src["pages"], self.cache, t)
        self.assertEqual(pages[2]["wikitext"], self.texts[102])
        self.assertIn("<p>", pages[1]["html"])
        self.assertEqual([c[2]["action"] for c in t.calls], ["query", "parse", "parse"])
        again = fetch.fetch_wikisource("w", self.src, self.src["pages"], self.cache, None, offline=True)
        self.assertEqual(again, pages)

    def test_sha1_mismatch_and_offline_and_altered_cache(self):
        altered = {**self.texts, 102: "Pagina due cambiata."}
        with self.assertRaises(BenchError) as ctx:
            fetch.fetch_wikisource("w", self.src, self.src["pages"], self.cache, self.api(altered))
        self.assertEqual(ctx.exception.code, Code.REVISION_MISMATCH)
        with self.assertRaises(BenchError) as ctx:
            fetch.fetch_wikisource("w", self.src, self.src["pages"], self.cache, None, offline=True)
        self.assertEqual(ctx.exception.code, Code.REVISION_UNREACHABLE)
        fetch.fetch_wikisource("w", self.src, self.src["pages"], self.cache, self.api())
        record = self.cache.wiki_dir / "101.json"
        data = json.loads(record.read_text())
        data["wikitext"] += "!"
        record.write_text(json.dumps(data))
        with self.assertRaises(BenchError):
            fetch.fetch_wikisource("w", self.src, self.src["pages"], self.cache, None, offline=True)
        t = self.api()
        fetch.fetch_wikisource("w", self.src, self.src["pages"], self.cache, t)
        self.assertEqual(t.calls[0][2]["revids"], "101")


class U6Tests(unittest.TestCase):
    """A changed source makes its document unavailable; the others go on (exit 2)."""

    def test_fetch_command_with_one_changed_source(self):
        good, changed = b"docx bytes one", b"docx bytes two"
        m = copy.deepcopy(manifest.load().data)
        m["sources"] = {"s1": {"kind": "file", "url": "https://example.org/1", "bytes": len(good), "sha256": sha(good)},
                        "s2": {"kind": "file", "url": "https://example.org/2", "bytes": len(changed), "sha256": sha(b"docx bytes old")}}
        base = {k: v for k, v in m["documents"][9].items()}
        m["documents"] = [dict(base, id="D10", input={"source": "s1"}, twin={"kind": "docx", "source": "s1"}),
                          dict(base, id="D11", input={"source": "s2"}, twin={"kind": "docx", "source": "s2"})]
        transport = FakeTransport({"https://example.org/1": (200, good), "https://example.org/2": (200, changed)})
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.json"
            path.write_text(json.dumps(m))
            out = io.StringIO()
            with patch.object(fetch, "HttpTransport", lambda: transport), contextlib.redirect_stdout(out):
                code = bench_cli.main(["fetch", "--docs", "D10,D11", "--manifest", str(path), "--cache", tmp + "/c"])
        self.assertEqual(code, 2)
        self.assertIn("D10 available", out.getvalue())
        self.assertIn("D11 unavailable source_hash_mismatch", out.getvalue())


class FakeHttpResponse:
    """What `requests` returns after following redirects: the final URL and the hops before it."""

    def __init__(self, url, history=(), body=DATA):
        self.url, self.history, self.body = url, [FakeHttpResponse(h, ()) for h in history], body
        self.status_code, self.headers = 200, {"Content-Type": "application/pdf"}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def iter_content(self, size):
        yield self.body


class FakeSession:
    def __init__(self, response):
        self.response, self.kwargs = response, None

    def get(self, url, **kwargs):
        self.kwargs = kwargs
        return self.response


class HttpTransportRedirectTests(unittest.TestCase):
    """HttpTransport accepts a body only when every URL of the redirect chain is https:// (K5)."""

    def get(self, response):
        session = FakeSession(response)
        return fetch.HttpTransport().get(session, SRC["url"]), session

    def test_https_chain_is_read(self):
        r, session = self.get(FakeHttpResponse("https://example.org/b.pdf", ["https://example.org/a.pdf"]))
        self.assertEqual((r.status, r.body), (200, DATA))
        self.assertTrue(session.kwargs["allow_redirects"])

    def test_plain_http_final_url_is_refused(self):
        r, _ = self.get(FakeHttpResponse("http://example.org/b.pdf", ["https://example.org/a.pdf"]))
        self.assertEqual((r.status, r.body), (0, b""))

    def test_plain_http_intermediate_hop_is_refused(self):
        r, _ = self.get(FakeHttpResponse("https://example.org/c.pdf",
                                         ["https://example.org/a.pdf", "http://example.org/b.pdf"]))
        self.assertEqual((r.status, r.body), (0, b""))
