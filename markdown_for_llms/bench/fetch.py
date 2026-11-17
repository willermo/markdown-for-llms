"""Download and verify the sources of the manifest (U6: a changed source becomes unavailable).

Only `fetch` uses the network; `checks` and `run` read the cache with `offline=True`, so no
measure includes a download. Every cached file is verified at each use: a mismatching one goes
to quarantine and, online, is downloaded again.
"""
from dataclasses import dataclass, field
import hashlib
import json
import time

from . import BENCH_VERSION
from .cache import sha256_file, write_atomic
from .codes import BenchError, Code

USER_AGENT = f"markdown-for-llms-bench/{BENCH_VERSION} (https://github.com/willermo/markdown-for-llms)"
SESSION_PAGE_LIMIT = 8 * 1024 * 1024
API_LIMIT = 32 * 1024 * 1024
WIKI_BATCH = 50
WIKI_PAUSE_S = 0.2


@dataclass
class Response:
    status: int
    content_type: str
    body: bytes


class HttpTransport:
    """GET over HTTPS with `requests`; one session (cookie jar) per source."""

    def __init__(self, timeout=(15, 120)):
        self.timeout = timeout

    def new_session(self):
        import requests
        session = requests.Session()
        session.headers["User-Agent"] = USER_AGENT
        session.trust_env = False      # no proxies or netrc from the environment
        return session

    def get(self, session, url, params=None, limit=API_LIMIT):
        import requests
        try:
            with session.get(url, params=params, timeout=self.timeout, stream=True, allow_redirects=True) as r:
                # every hop of the redirect chain, not only the final URL
                if not all(u.startswith("https://") for u in [*(h.url for h in r.history), r.url]):
                    return Response(0, "", b"")
                body = bytearray()
                for chunk in r.iter_content(1 << 16):
                    body += chunk
                    if len(body) > limit:
                        break
                return Response(r.status_code, r.headers.get("Content-Type", ""), bytes(body))
        except requests.RequestException as exc:
            raise BenchError(Code.SOURCE_UNREACHABLE, type(exc).__name__) from None


def _verified(path, src):
    return path.is_file() and path.stat().st_size == src["bytes"] and sha256_file(path) == src["sha256"]


def fetch_file(source_id, src, cache, transport=None, offline=False):
    """Path of a verified file source in the cache."""
    path = cache.source_path(src)
    if path.exists():
        if _verified(path, src):
            return path
        cache.quarantine(source_id, path=path)
        if offline:
            raise BenchError(Code.SOURCE_HASH_MISMATCH, source_id + " (cached copy altered)")
    if offline or transport is None:
        raise BenchError(Code.SOURCE_UNREACHABLE, source_id + " (not in the cache; run `mdf bench fetch`)")
    session = transport.new_session()
    if "session_url" in src:
        page = transport.get(session, src["session_url"], limit=SESSION_PAGE_LIMIT)
        if page.status != 200:
            raise BenchError(Code.SESSION_FAILED, f"{source_id}: HTTP {page.status}")
    response = transport.get(session, src["url"], limit=src["bytes"])
    if response.status != 200:
        raise BenchError(Code.SOURCE_UNREACHABLE, f"{source_id}: HTTP {response.status}")
    if len(response.body) != src["bytes"]:
        cache.quarantine(source_id, data=response.body)
        raise BenchError(Code.SOURCE_SIZE_MISMATCH, f"{source_id}: {len(response.body)} bytes")
    if hashlib.sha256(response.body).hexdigest() != src["sha256"]:
        cache.quarantine(source_id, data=response.body)
        raise BenchError(Code.SOURCE_HASH_MISMATCH, source_id)
    write_atomic(path, response.body)
    return path


# --- Wikisource -------------------------------------------------------------------------------

def _sha1(text):
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _wiki_json(transport, session, api, params):
    response = transport.get(session, api, params=dict(params, format="json", formatversion="2"), limit=API_LIMIT)
    if response.status != 200:
        raise BenchError(Code.REVISION_UNREACHABLE, f"HTTP {response.status}")
    try:
        data = json.loads(response.body)
    except ValueError:
        raise BenchError(Code.REVISION_UNREACHABLE, "invalid JSON") from None
    if "error" in data:
        raise BenchError(Code.REVISION_UNREACHABLE, str(data["error"].get("code")))
    return data


def _cached_revision(cache, page):
    path = cache.wiki_dir / f"{page['revid']}.json"
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_bytes())
    except ValueError:
        data = None
    if (not isinstance(data, dict) or data.get("revid") != page["revid"] or not isinstance(data.get("wikitext"), str)
            or _sha1(data["wikitext"]) != page["sha1"] or not isinstance(data.get("html"), str)):
        cache.quarantine(f"revision-{page['revid']}", path=path)
        return None
    return data


def selected_pages(src, twin):
    pages = src["pages"] if twin["pages"] == "all" else [p for p in src["pages"] if p["page"] in set(twin["pages"])]
    return pages


def fetch_wikisource(source_id, src, pages, cache, transport=None, offline=False):
    """Revisions of the pages: {page: {"revid", "sha1", "wikitext", "html"}}, sha1 verified."""
    result, missing = {}, []
    for page in pages:
        data = _cached_revision(cache, page)
        if data is None:
            missing.append(page)
        else:
            result[page["page"]] = data
    if missing and (offline or transport is None):
        raise BenchError(Code.REVISION_UNREACHABLE, f"{source_id}: {len(missing)} revisions not in the cache")
    session = transport.new_session() if missing else None
    for start in range(0, len(missing), WIKI_BATCH):
        batch = missing[start:start + WIKI_BATCH]
        data = _wiki_json(transport, session, src["api"], {
            "action": "query", "prop": "revisions", "revids": "|".join(str(p["revid"]) for p in batch),
            "rvprop": "ids|sha1|content", "rvslots": "main"})
        found = {}
        for item in data.get("query", {}).get("pages", []):
            for rev in item.get("revisions", []):
                content = rev.get("slots", {}).get("main", {}).get("content")
                found[rev.get("revid")] = (rev.get("sha1"), content)
        for page in batch:
            sha1, content = found.get(page["revid"], (None, None))
            if not isinstance(content, str):
                raise BenchError(Code.REVISION_UNREACHABLE, f"{source_id}: revision {page['revid']}")
            if sha1 != page["sha1"] or _sha1(content) != page["sha1"]:
                raise BenchError(Code.REVISION_MISMATCH, f"{source_id}: revision {page['revid']}")
            time.sleep(WIKI_PAUSE_S)
            parsed = _wiki_json(transport, session, src["api"], {
                "action": "parse", "oldid": str(page["revid"]), "prop": "text",
                "disablelimitreport": "1", "disableeditsection": "1"})
            html = parsed.get("parse", {}).get("text")
            if not isinstance(html, str):
                raise BenchError(Code.REVISION_UNREACHABLE, f"{source_id}: HTML of {page['revid']}")
            record = {"revid": page["revid"], "sha1": page["sha1"], "page": page["page"], "wikitext": content,
                      "html": html}
            write_atomic(cache.wiki_dir / f"{page['revid']}.json",
                         json.dumps(record, ensure_ascii=False).encode("utf-8"))
            result[page["page"]] = record
    return {p["page"]: result[p["page"]] for p in pages}


# --- documents --------------------------------------------------------------------------------

@dataclass
class DocumentFiles:
    doc_id: str
    input: object                 # Path of the file to convert (derived if the manifest says so)
    twin: dict                    # "file": Path, or "pages": {page: revision}; "inventory": Path
    warnings: list = field(default_factory=list)
    hashes: dict = field(default_factory=dict)


def twin_sources(doc):
    twin = doc["twin"]
    if twin is None:
        return []
    ids = [twin["source"]]
    if "inventory" in twin:
        ids.append(twin["inventory"])
    return ids


def resolve_document(doc, m, cache, transport=None, offline=True):
    """All the files of a document, verified; raises BenchError when something is unavailable."""
    from .derive import derive_input

    src_id = doc["input"]["source"]
    source_path = fetch_file(src_id, m.sources[src_id], cache, transport, offline)
    twin, hashes = {}, {"source": m.sources[src_id]["sha256"], "twin": None}
    t = doc["twin"] or {}
    tsrc = m.sources[t["source"]] if t else None
    if tsrc is None:
        pass                        # an oracle document (`checks: "encoding"`): no twin
    elif tsrc["kind"] == "wikisource":
        pages = selected_pages(tsrc, t)
        twin["pages"] = fetch_wikisource(t["source"], tsrc, pages, cache, transport, offline)
        twin["quality"] = {p["page"]: p["quality"] for p in pages}
        hashes["twin"] = [p["sha1"] for p in pages]
    else:
        twin["file"] = fetch_file(t["source"], tsrc, cache, transport, offline)
        hashes["twin"] = tsrc["sha256"]
    if "inventory" in t:
        twin["inventory"] = fetch_file(t["inventory"], m.sources[t["inventory"]], cache, transport, offline)
        hashes["inventory"] = m.sources[t["inventory"]]["sha256"]
    input_path, warnings, derived = derive_input(doc, source_path, cache)
    if derived:
        hashes["derived"] = derived
    return DocumentFiles(doc["id"], input_path, twin, warnings, hashes)
