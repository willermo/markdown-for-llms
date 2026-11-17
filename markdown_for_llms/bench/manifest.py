"""The manifest of the reference set: sources (URL, bytes, SHA-256) and documents.

The repository holds only this list, never the documents or their text. A strict validator
rejects unknown keys, so a typo cannot silently drop a hash or a derivation.

Schema 2 (plan r2 §6): every format of the engine; derivations as a typed pipeline (each step
reads the kind the previous one wrote, the last one writes the format of the document); the
language (BCP 47) and the scripts (ISO 15924) of each document; per-document limits; for a txt
input its expected encoding and the SHA-256 of its oracle (the UTF-8 text before the recode,
as the engine writes it); a public document without a twin only with `checks: "encoding"`.
A manifest of schema 1 is refused with `manifest_schema_unsupported`.

A personal set (`set: "personal"`, plan r2 §13, delta r3 Δ9) is a local manifest without sources:
documents `Pnn` with the absolute path of their input and, optionally, of a twin of a kind that
needs nothing but its file; no `checks` (no oracle without a twin), no levels (it is selected with
`--docs`). Its documents are completed with the defaults of the fields the bench reads.
"""
from dataclasses import dataclass
import hashlib
from importlib import resources
import json
from pathlib import Path
import re

from .codes import BenchError, Code

SCHEMA = "mdf-bench-manifest/2"
LEVELS = ("quick", "standard", "long")
PROFILES = ("scientific", "scan", "office", "legal")
FORMATS = ("pdf", "png", "jpeg", "tiff", "djvu", "docx", "xlsx", "pptx", "odt", "ods", "epub", "doc", "rtf", "html",
           "txt")                       # the formats of the engine (`engine.formats.FORMATS`)
PAGED_FORMATS = ("pdf", "djvu", "png", "jpeg", "tiff")      # formats with an exact expected page count
TWIN_KINDS = ("latex", "jats", "xlsx", "xlsx-projection", "pptx", "docx", "odt", "epub", "wikisource",
              "akoma-ntoso", "html")
# recode targets -> the canonical name the engine reports (`engine.text`), the `expected_encoding`
ENCODINGS = {"utf-8": "utf_8", "utf-16": "utf_16", "utf-32": "utf_32", "cp1250": "cp1250", "cp1251": "cp1251",
             "cp1252": "cp1252", "cp1253": "cp1253"}
OFFICE_TARGETS = ("doc", "rtf", "ods", "odt")
LIMIT_KEYS = ("max_pages", "memory_mib", "output_mib", "timeout_seconds")
LANGUAGE = re.compile(r"[a-z]{2,3}(-[A-Za-z0-9]{2,8})*")
SCRIPT = re.compile(r"[A-Z][a-z]{3}")
SOURCE_ID = re.compile(r"[a-z0-9][a-z0-9-]{0,62}")
DOC_ID = re.compile(r"D\d{2}")
PERSONAL_ID = re.compile(r"P\d{2}")
PERSONAL_TWINS = ("docx", "odt", "pptx", "xlsx", "epub", "jats")    # twins read from their file alone
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


def step_kinds(step):
    """(kinds the step reads, kind it writes) of a validated derivation step."""
    op = step["op"]
    if op == "djvu-to-pdf":
        return ("djvu",), "pdf"
    if op == "pdf-pages":
        return ("pdf",), "pdf"
    if op == "page-png":
        return ("pdf",), "png"
    if op == "page-image":
        return ("pdf",), step["format"]
    if op == "office-convert":
        return ("docx", "xlsx", "odt"), step["to"]
    if op == "text-export":
        return ("docx", "odt", "doc", "rtf"), "txt"
    return ("txt",), "txt"          # recode, excerpt


def _validate_derive(steps, where, fmt):
    if not isinstance(steps, list) or not steps:
        _fail(where + ": non-empty list expected")
    for i, step in enumerate(steps):
        sw = f"{where}[{i}]"
        op = step.get("op") if isinstance(step, dict) else None
        if op == "djvu-to-pdf" or op == "text-export":
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
        elif op == "page-image":
            _keys(step, ("op", "page", "dpi", "format"), (), sw)
            _int(step["page"], 1, sw + ".page")
            _int(step["dpi"], 72, sw + ".dpi", 600)
            if step["format"] not in ("jpeg", "tiff"):
                _fail(sw + ".format: jpeg or tiff expected")
        elif op == "office-convert":
            _keys(step, ("op", "to"), (), sw)
            if step["to"] not in OFFICE_TARGETS:
                _fail(sw + ".to: unknown target")
        elif op == "recode":
            _keys(step, ("op", "encoding", "bom"), (), sw)
            if step["encoding"] not in ENCODINGS or type(step["bom"]) is not bool:
                _fail(sw + ": known encoding and boolean bom expected")
            # the engine reads UTF-16 and UTF-32 only after a byte order mark; code pages have none
            if step["bom"] != (step["encoding"] in ("utf-16", "utf-32")) and step["encoding"] != "utf-8":
                _fail(sw + ".bom: a mark for UTF-16 and UTF-32 only")
        elif op == "excerpt":
            _keys(step, ("op", "first_line", "last_line"), (), sw)
            _int(step["first_line"], 1, sw + ".first_line")
            _int(step["last_line"], step["first_line"], sw + ".last_line")
        else:
            _fail(sw + ": unknown op")
    for a, b in zip(steps, steps[1:]):
        if step_kinds(a)[1] not in step_kinds(b)[0]:
            _fail(where + ": unsupported sequence")
    if step_kinds(steps[-1])[1] != fmt:
        _fail(where + ": the last step does not write the format of the document")


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
        "html": (("kind", "source"), ()),
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


def _codes(value, pattern, where):
    if not isinstance(value, list) or not value or len(set(value)) != len(value) or \
            not all(isinstance(v, str) and pattern.fullmatch(v) for v in value):
        _fail(where + ": non-empty list of codes expected")


def _validate_document(doc, sources, where):
    _keys(doc, ("id", "title", "role", "license", "attribution", "levels", "profile", "format", "pages",
                "language", "scripts", "input", "twin"),
          ("pages_estimate", "limits", "expected_encoding", "checks"), where)
    for name in ("title", "role", "license", "attribution"):
        _text(doc[name], f"{where}.{name}")
    levels = doc["levels"]
    if not isinstance(levels, list) or not levels or len(set(levels)) != len(levels) or not set(levels) <= set(LEVELS):
        _fail(where + ".levels")
    if doc["profile"] not in PROFILES:
        _fail(where + ".profile")
    fmt = doc["format"]
    if fmt not in FORMATS:
        _fail(where + ".format")
    if fmt in PAGED_FORMATS:
        _int(doc["pages"], 1, where + ".pages")
        if "pages_estimate" in doc:
            _fail(where + ".pages_estimate: only for formats without pages")
    else:
        if doc["pages"] is not None:
            _fail(where + ".pages: null expected for a format without pages")
        _int(doc.get("pages_estimate"), 1, where + ".pages_estimate")
    _codes(doc["language"], LANGUAGE, where + ".language")
    _codes(doc["scripts"], SCRIPT, where + ".scripts")
    if "limits" in doc:
        limits = doc["limits"]
        if not isinstance(limits, dict) or not limits or not set(limits) <= set(LIMIT_KEYS):
            _fail(where + ".limits: known limits expected")
        for name, value in limits.items():
            _int(value, 1, f"{where}.limits.{name}")
    inp = doc["input"]
    _keys(inp, ("source", "derive", "reference") if "derive" in inp else ("source",), (), where + ".input")
    src = sources.get(inp["source"])
    if src is None or src["kind"] != "file":
        _fail(where + ".input.source: unknown file source")
    recode = None
    if "derive" in inp:
        _validate_derive(inp["derive"], where + ".input.derive", fmt)
        recode = next((s for s in reversed(inp["derive"]) if s["op"] == "recode"), None)
        ref = inp["reference"]
        _keys(ref, ("sha256", "toolchain"), ("deterministic", "oracle_sha256"), where + ".input.reference")
        for name in ("sha256", "toolchain") + (("oracle_sha256",) if "oracle_sha256" in ref else ()):
            # `toolchain`: the content identity of the reference toolchain that produced the reference
            if not isinstance(ref[name], str) or not SHA256.fullmatch(ref[name]):
                _fail(f"{where}.input.reference.{name}")
        if "deterministic" in ref and type(ref["deterministic"]) is not bool:
            _fail(where + ".input.reference.deterministic")
        if ("oracle_sha256" in ref) != (recode is not None):
            _fail(where + ".input.reference.oracle_sha256: required with a recode, and only then")
    if fmt == "txt":
        # the encoding the engine must find: the one of the last recode
        if recode is None or doc.get("expected_encoding") != ENCODINGS[recode["encoding"]]:
            _fail(where + ".expected_encoding: the encoding of the last recode expected")
    elif "expected_encoding" in doc:
        _fail(where + ".expected_encoding: only for txt")
    if "checks" in doc and (doc["checks"] != "encoding" or fmt != "txt"):
        _fail(where + ".checks: 'encoding' for a txt document only")
    if doc["twin"] is None:
        if doc.get("checks") != "encoding":
            _fail(where + ".twin: null only with checks 'encoding'")
    else:
        if "checks" in doc:
            _fail(where + ".checks: only without a twin")
        _validate_twin(doc["twin"], sources, where + ".twin")


def _absolute(value, where):
    if not isinstance(value, str) or not value.startswith("/") or "\0" in value or len(value) > 4096:
        _fail(where + ": an absolute path expected")


def _validate_personal(data):
    _keys(data, ("schema", "set", "documents"), (), "manifest")
    if not isinstance(data["documents"], list) or not data["documents"]:
        _fail("documents: non-empty list expected")
    seen = set()
    for i, doc in enumerate(data["documents"]):
        where = f"documents[{i}]"
        if not isinstance(doc, dict) or not isinstance(doc.get("id"), str) or not PERSONAL_ID.fullmatch(doc["id"]) \
                or doc["id"] in seen:
            _fail(where + ".id: unique Pnn expected")
        seen.add(doc["id"])
        _keys(doc, ("id", "format", "input", "twin"), ("profile", "pages", "pages_estimate", "language", "scripts",
                                                      "limits", "title"), where)
        if doc["format"] not in FORMATS:
            _fail(where + ".format")
        _keys(doc["input"], ("path",), (), where + ".input")
        _absolute(doc["input"]["path"], where + ".input.path")
        if doc["twin"] is not None:
            _keys(doc["twin"], ("kind", "path"), (), where + ".twin")
            if doc["twin"]["kind"] not in PERSONAL_TWINS:
                _fail(where + ".twin.kind")
            _absolute(doc["twin"]["path"], where + ".twin.path")
        if "profile" in doc and doc["profile"] not in PROFILES:
            _fail(where + ".profile")
        for name in ("pages", "pages_estimate"):
            if name in doc:
                _int(doc[name], 1, f"{where}.{name}")
        if "language" in doc:
            _codes(doc["language"], LANGUAGE, where + ".language")
        if "scripts" in doc:
            _codes(doc["scripts"], SCRIPT, where + ".scripts")
        if "limits" in doc:
            if not isinstance(doc["limits"], dict) or not doc["limits"] or not set(doc["limits"]) <= set(LIMIT_KEYS):
                _fail(where + ".limits: known limits expected")
            for name, value in doc["limits"].items():
                _int(value, 1, f"{where}.limits.{name}")
        if "title" in doc:
            _text(doc["title"], where + ".title")
    data["documents"] = [{"id": d["id"], "format": d["format"], "profile": d.get("profile", "office"),
                          "levels": [], "pages": d.get("pages") if d["format"] in PAGED_FORMATS else None,
                          "pages_estimate": d.get("pages_estimate", 1),
                          "language": d.get("language", []), "scripts": d.get("scripts", []),
                          "input": dict(d["input"]), "twin": None if d["twin"] is None else dict(d["twin"]),
                          **({"limits": dict(d["limits"])} if "limits" in d else {})}
                         for d in data["documents"]]
    return data


def validate(data):
    if isinstance(data, dict) and data.get("schema") == SCHEMA and data.get("set") == "personal":
        return _validate_personal(data)
    _keys(data, ("schema", "set", "sources", "documents"), (), "manifest")
    if data["schema"] != SCHEMA:
        raise BenchError(Code.MANIFEST_SCHEMA_UNSUPPORTED, str(data["schema"])[:40])
    if data["set"] != "public":
        _fail("set: public or personal")
    if not isinstance(data["sources"], dict) or not data["sources"]:
        _fail("sources: non-empty object expected")
    for sid, src in data["sources"].items():
        _validate_source(sid, src)
    if not isinstance(data["documents"], list) or not data["documents"]:
        _fail("documents: non-empty list expected")
    seen = set()
    for i, doc in enumerate(data["documents"]):
        where = f"documents[{i}]"
        if not isinstance(doc, dict) or not isinstance(doc.get("id"), str) or not DOC_ID.fullmatch(doc["id"]) \
                or doc["id"] in seen:
            _fail(where + ".id: unique Dnn expected")
        seen.add(doc["id"])
        _validate_document(doc, data["sources"], doc["id"])
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
        return self.data.get("sources", {})

    @property
    def personal(self):
        return self.data["set"] == "personal"

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
