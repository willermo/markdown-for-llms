"""The manifest of the reference set: sources (URL, bytes, SHA-256) and documents.

The repository holds only this list, never the documents or their text. A strict validator
rejects unknown keys, so a typo cannot silently drop a hash or a derivation.
"""
from dataclasses import dataclass
import hashlib
from importlib import resources
import json
from pathlib import Path
import re

from .codes import BenchError, Code

SCHEMA = "mdf-bench-manifest/1"
LEVELS = ("quick", "standard", "long")
PROFILES = ("scientific", "scan", "office", "legal")
FORMATS = ("pdf", "html", "xlsx", "pptx", "docx", "epub", "png")
PAGED_FORMATS = ("pdf", "png")          # formats with an exact expected page count
TWIN_KINDS = ("latex", "jats", "xlsx", "xlsx-projection", "pptx", "docx", "odt", "epub", "wikisource",
              "akoma-ntoso")
SOURCE_ID = re.compile(r"[a-z0-9][a-z0-9-]{0,62}")
DOC_ID = re.compile(r"D\d{2}")
SHA256 = re.compile(r"[0-9a-f]{64}")
SHA1 = re.compile(r"[0-9a-f]{40}")
CELL_RANGE = re.compile(r"[A-Z]{1,3}:[A-Z]{1,3}(,[A-Z]{1,3}:[A-Z]{1,3})*")
ROW_RANGE = re.compile(r"\d{1,6}:\d{1,6}")


def _fail(detail):
    raise BenchError(Code.MANIFEST_INVALID, detail)


def _keys(value, required, optional=(), where=""):
    if not isinstance(value, dict):
        _fail(f"{where}: object expected")
    missing = set(required) - set(value)
    unknown = set(value) - set(required) - set(optional)
    if missing or unknown:
        _fail(f"{where}: missing {sorted(missing)}, unknown {sorted(unknown)}")


def _int(value, low, where, high=2**31 - 1):
    if type(value) is not int or not low <= value <= high:
        _fail(f"{where}: integer in [{low}, {high}] expected")


def _text(value, where):
    if not isinstance(value, str) or not value.strip() or "\0" in value:
        _fail(f"{where}: non-empty text expected")


def _https(value, where):
    _text(value, where)
    if not value.startswith("https://") or any(c.isspace() for c in value):
        _fail(f"{where}: https URL expected")


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            _fail(f"duplicate key {key!r}")
        result[key] = value
    return result


def _validate_source(sid, src):
    where = f"sources.{sid}"
    if not SOURCE_ID.fullmatch(sid):
        _fail(f"{where}: invalid id")
    kind = src.get("kind") if isinstance(src, dict) else None
    if kind == "file":
        _keys(src, ("kind", "url", "bytes", "sha256"), ("session_url",), where)
        _https(src["url"], where + ".url")
        _int(src["bytes"], 1, where + ".bytes", 2**34)
        if not isinstance(src["sha256"], str) or not SHA256.fullmatch(src["sha256"]):
            _fail(where + ".sha256: 64 lowercase hex digits expected")
        if "session_url" in src:
            _https(src["session_url"], where + ".session_url")
    elif kind == "wikisource":
        _keys(src, ("kind", "api", "index", "pages"), (), where)
        _https(src["api"], where + ".api")
        _text(src["index"], where + ".index")
        if not isinstance(src["pages"], list) or not src["pages"]:
            _fail(where + ".pages: non-empty list expected")
        previous = 0
        for i, page in enumerate(src["pages"]):
            pw = f"{where}.pages[{i}]"
            _keys(page, ("page", "revid", "sha1", "quality"), (), pw)
            _int(page["page"], previous + 1, pw + ".page")
            previous = page["page"]
            _int(page["revid"], 1, pw + ".revid")
            _int(page["quality"], 0, pw + ".quality", 4)
            if not isinstance(page["sha1"], str) or not SHA1.fullmatch(page["sha1"]):
                _fail(pw + ".sha1: 40 lowercase hex digits expected")
    else:
        _fail(f"{where}: unknown kind")


def _validate_derive(steps, where):
    if not isinstance(steps, list) or not steps:
        _fail(where + ": non-empty list expected")
    for i, step in enumerate(steps):
        sw = f"{where}[{i}]"
        op = step.get("op") if isinstance(step, dict) else None
        if op == "djvu-to-pdf":
            _keys(step, ("op",), (), sw)
        elif op == "pdf-pages":
            _keys(step, ("op", "first", "last"), (), sw)
            _int(step["first"], 1, sw + ".first")
            _int(step["last"], step["first"], sw + ".last")
        elif op == "page-png":
            _keys(step, ("op", "page", "dpi", "color"), (), sw)
            _int(step["page"], 1, sw + ".page")
            _int(step["dpi"], 72, sw + ".dpi", 600)
            if step["color"] != "rgb":   # pdftoppm 24.02 ignores -gray when it writes PNG
                _fail(sw + ".color: rgb expected")
        else:
            _fail(sw + ": unknown op")
    if steps[0]["op"] not in ("djvu-to-pdf", "pdf-pages") or any(s["op"] != "page-png" for s in steps[2:]):
        _fail(where + ": unsupported sequence")


def _validate_twin(twin, sources, where):
    kind = twin.get("kind") if isinstance(twin, dict) else None
    fields = {
        "latex": (("kind", "source", "main", "bbl"), ()),
        "jats": (("kind", "source"), ()),
        "xlsx": (("kind", "source"), ()),
        "xlsx-projection": (("kind", "source", "select", "verify"), ()),
        "pptx": (("kind", "source"), ()),
        "docx": (("kind", "source"), ()),
        "odt": (("kind", "source"), ()),
        "epub": (("kind", "source"), ()),
        "wikisource": (("kind", "source", "pages"), ("isolated_pages",)),
        "akoma-ntoso": (("kind", "source", "inventory"), ("discard", "rules")),
    }
    if kind not in fields:
        _fail(where + ": unknown twin kind")
    _keys(twin, *fields[kind], where)
    source = sources.get(twin["source"])
    if source is None:
        _fail(where + ".source: unknown source")
    if (kind == "wikisource") != (source["kind"] == "wikisource"):
        _fail(where + ".source: wrong source kind")
    if kind == "latex":
        for name in ("main", "bbl"):
            if not isinstance(twin[name], str) or not re.fullmatch(r"[A-Za-z0-9_.-]+\.(tex|bbl)", twin[name]):
                _fail(f"{where}.{name}: plain file name expected")
    elif kind == "wikisource":
        known = {p["page"] for p in source["pages"]}
        pages = twin["pages"]
        if pages != "all" and (not isinstance(pages, list) or not pages or not set(pages) <= known
                               or any(type(p) is not int for p in pages)):
            _fail(where + ".pages: 'all' or pages of the index expected")
        isolated = twin.get("isolated_pages", [])
        if not isinstance(isolated, list) or not set(isolated) <= known:
            _fail(where + ".isolated_pages: pages of the index expected")
    elif kind == "akoma-ntoso":
        inv = sources.get(twin["inventory"])
        if inv is None or inv["kind"] != "file":
            _fail(where + ".inventory: unknown file source")
        # Declared reader and normalization rules exist only for the Normattiva twin (C2, C3).
        if not set(twin.get("discard", [])) <= {"normattiva-preface-notes"}:
            _fail(where + ".discard: unknown rule")
        if not set(twin.get("rules", [])) <= {"it-final-accent"}:
            _fail(where + ".rules: unknown rule")
    elif kind == "xlsx-projection":
        if not isinstance(twin["select"], list) or not twin["select"]:
            _fail(where + ".select: non-empty list expected")
        for i, sel in enumerate(twin["select"]):
            sw = f"{where}.select[{i}]"
            _keys(sel, ("sheet", "rows", "cols", "header_rows"), ("table",), sw)
            if "table" in sel:
                _int(sel["table"], 1, sw + ".table", 100)
            _text(sel["sheet"], sw + ".sheet")
            rows = sel["rows"]
            if not ((isinstance(rows, str) and ROW_RANGE.fullmatch(rows))
                    or (isinstance(rows, list) and rows and all(type(r) is int and r > 0 for r in rows))):
                _fail(sw + ".rows: 'a:b' or a list of rows expected")
            if not isinstance(sel["cols"], str) or not CELL_RANGE.fullmatch(sel["cols"]):
                _fail(sw + ".cols: column ranges expected")
            if not isinstance(sel["header_rows"], list) or not all(type(r) is int and r > 0 for r in sel["header_rows"]):
                _fail(sw + ".header_rows: list of rows expected")
        if twin["verify"] != {"text": "pdftotext-layout", "second": "ghostscript-txtwrite"}:
            _fail(where + ".verify: unsupported verification")


def validate(data):
    _keys(data, ("schema", "set", "sources", "documents"), (), "manifest")
    if data["schema"] != SCHEMA:
        _fail("unsupported schema")
    if data["set"] != "public":
        _fail("only the public set is supported by this version")
    if not isinstance(data["sources"], dict) or not data["sources"]:
        _fail("sources: non-empty object expected")
    for sid, src in data["sources"].items():
        _validate_source(sid, src)
    if not isinstance(data["documents"], list) or not data["documents"]:
        _fail("documents: non-empty list expected")
    seen = set()
    for i, doc in enumerate(data["documents"]):
        where = f"documents[{i}]"
        _keys(doc, ("id", "title", "role", "license", "attribution", "levels", "profile", "format", "pages",
                    "input", "twin"), ("pages_estimate",), where)
        if not isinstance(doc["id"], str) or not DOC_ID.fullmatch(doc["id"]) or doc["id"] in seen:
            _fail(where + ".id: unique Dnn expected")
        seen.add(doc["id"])
        where = doc["id"]
        for name in ("title", "role", "license", "attribution"):
            _text(doc[name], f"{where}.{name}")
        levels = doc["levels"]
        if not isinstance(levels, list) or not levels or len(set(levels)) != len(levels) or not set(levels) <= set(LEVELS):
            _fail(where + ".levels")
        if doc["profile"] not in PROFILES:
            _fail(where + ".profile")
        if doc["format"] not in FORMATS:
            _fail(where + ".format")
        if doc["format"] in PAGED_FORMATS:
            _int(doc["pages"], 1, where + ".pages")
            if "pages_estimate" in doc:
                _fail(where + ".pages_estimate: only for formats without pages")
        else:
            if doc["pages"] is not None:
                _fail(where + ".pages: null expected for a format without pages")
            _int(doc.get("pages_estimate"), 1, where + ".pages_estimate")
        inp = doc["input"]
        _keys(inp, ("source", "derive", "reference") if "derive" in inp else ("source",), (), where + ".input")
        src = data["sources"].get(inp["source"])
        if src is None or src["kind"] != "file":
            _fail(where + ".input.source: unknown file source")
        if "derive" in inp:
            _validate_derive(inp["derive"], where + ".input.derive")
            ref = inp["reference"]
            _keys(ref, ("sha256", "toolchain"), (), where + ".input.reference")
            if not isinstance(ref["sha256"], str) or not SHA256.fullmatch(ref["sha256"]):
                _fail(where + ".input.reference.sha256")
            # the content identity of the reference toolchain that produced the reference (toolchain.json)
            if not isinstance(ref["toolchain"], str) or not SHA256.fullmatch(ref["toolchain"]):
                _fail(where + ".input.reference.toolchain")
        _validate_twin(doc["twin"], data["sources"], where + ".twin")
    return data


@dataclass(frozen=True)
class Manifest:
    data: dict
    sha256: str
    path: str

    @property
    def documents(self):
        return self.data["documents"]

    @property
    def sources(self):
        return self.data["sources"]

    def document(self, doc_id):
        for doc in self.documents:
            if doc["id"] == doc_id:
                return doc
        raise BenchError(Code.UNKNOWN_DOCUMENT, doc_id)

    def select(self, level=None, ids=None):
        """Documents of a level, or the listed ids in manifest order."""
        if ids is not None:
            wanted = list(dict.fromkeys(ids))
            for doc_id in wanted:
                self.document(doc_id)
            return [d for d in self.documents if d["id"] in wanted]
        if level is None:
            return list(self.documents)
        if level not in LEVELS:
            raise BenchError(Code.INVALID_SETTING, "level")
        return [d for d in self.documents if level in d["levels"]]


def parse(raw: bytes, path="<bytes>") -> Manifest:
    try:
        data = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_pairs)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BenchError(Code.MANIFEST_INVALID, type(exc).__name__) from None
    return Manifest(validate(data), hashlib.sha256(raw).hexdigest(), path)


def load(path=None) -> Manifest:
    """The public manifest packaged with the bench, or a manifest file."""
    if path is None:
        raw = resources.files("markdown_for_llms.bench").joinpath("data/reference-set.json").read_bytes()
        return parse(raw, "reference-set.json")
    return parse(Path(path).read_bytes(), str(path))
