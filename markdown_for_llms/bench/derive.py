"""Derivations of the input to convert: DjVu to PDF, PDF pages, a page as PNG, JPEG or TIFF, an
office file converted by LibreOffice (doc, rtf, ods), the text export of Writer, a text recoded
without loss and an excerpt of a text (manifest schema 2).

They run with the reference toolchain (`toolchain.py`), whose identity, not a version read from
a tool, enters the key: a derivation found in the cache needs no container, so a cache prepared
elsewhere and copied works without Docker (U4); a missing or altered one is derived again, which
fails closed (`toolchain_missing`) where there is no engine. The bench records the hash of the
starting file, of the derived file and of the reference in the manifest (`reference.toolchain`
names the toolchain that produced it). A derived file that differs from the reference is a
warning, not an unavailable document: `derived_differs_from_reference`, or
`derived_nondeterministic` for a reference the manifest declares `deterministic: false`
(LibreOffice writes a different .doc or .ods at every run for D23 and D24).

A recode fails closed (`recode_lossy`, with the count of code points only, never the text) when
one character of the text has no code in the target encoding: nothing is replaced. It records
the oracle of the txt input: the SHA-256 of the UTF-8 text before the recode, as the `text`
route of the engine writes it (no byte order mark, line ends LF, a final line end).
"""
import hashlib
import json
from pathlib import Path
import re
import shutil

from ..engine import toolchain, tools
from .cache import canonical, sha256_file, write_atomic
from .codes import BenchError, Code
from .manifest import step_kinds

SUFFIX = {"pdf": ".pdf", "png": ".png", "jpeg": ".jpg", "tiff": ".tif", "doc": ".doc", "rtf": ".rtf", "ods": ".ods",
          "odt": ".odt", "txt": ".txt"}
OFFICE_INPUTS = ("docx", "xlsx", "odt")
WRITER_TEXT = "txt:Text (encoded):UTF8"
BOMS = {"utf-8": b"\xef\xbb\xbf", "utf-16": b"\xff\xfe", "utf-32": b"\xff\xfe\x00\x00"}
CODECS = {"utf-8": "utf-8", "utf-16": "utf-16-le", "utf-32": "utf-32-le"}

PDF_DATE = re.compile(rb"(/(?:CreationDate|ModDate)\s*\()(D:[0-9Z+\-']{4,30})(\))")
FIXED_DATE = b"19700101000000"


def normalize_pdf_dates(data: bytes) -> bytes:
    """Replace the digits of /CreationDate and /ModDate with a fixed date of the same length.

    `ddjvu -format=pdf` writes the current time there (and nowhere else); keeping the length
    keeps every xref offset valid.
    """
    def repl(match):
        out, i = bytearray(match.group(2)), 0
        for pos, byte in enumerate(out):
            if 48 <= byte <= 57:
                out[pos] = FIXED_DATE[i] if i < len(FIXED_DATE) else 48
                i += 1
        return match.group(1) + bytes(out) + match.group(3)
    return PDF_DATE.sub(repl, data)


def pdf_pages(path, work_root):
    text = tools.run("pdfinfo", [tools.In(path)], work_root, timeout=120).decode("utf-8", "replace")
    match = re.search(r"^Pages:\s+(\d+)$", text, re.MULTILINE)
    if not match:
        raise BenchError(Code.TOOL_FAILED, "pdfinfo: no page count")
    return int(match.group(1))


def oracle_text(text):
    """The text as the `text` route of the engine writes it: no mark, LF line ends, a final one."""
    text = text[1:] if text.startswith("\ufeff") else text
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return text if text.endswith("\n") else text + "\n"


def _utf8(path):
    try:
        text = Path(path).read_bytes().decode("utf-8", "strict")
    except UnicodeDecodeError:
        raise BenchError(Code.SOURCE_UNEXPECTED_TYPE, "text to recode is not UTF-8") from None
    return text[1:] if text.startswith("\ufeff") else text


def recode(text, encoding, bom):
    """Bytes of `text` in `encoding`, or `recode_lossy` with the number of code points it cannot hold."""
    codec = CODECS.get(encoding, encoding)
    lossy = 0
    for ch in set(text):
        try:
            ch.encode(codec)
        except UnicodeEncodeError:
            lossy += text.count(ch)
    if lossy:
        raise BenchError(Code.RECODE_LOSSY, f"{lossy} code points outside {encoding}")
    return (BOMS[encoding] if bom else b"") + text.encode(codec)


def _office_input(src):
    from ..engine.formats import signature
    kind = next((k for k in OFFICE_INPUTS if signature(src, k)), None)
    if kind is None:
        raise BenchError(Code.SOURCE_UNEXPECTED_TYPE, "not an office file LibreOffice converts here")
    return "input." + kind


def _run_step(step, src, workdir, work_root):
    """Path of the output of one step (the self-test of the toolchain runs the tool steps)."""
    return _step(step, src, workdir, work_root)[0]


def _step(step, src, workdir, work_root):
    """(path of the step's output, oracle SHA-256 or None)."""
    op = step["op"]
    if op == "djvu-to-pdf":
        out = workdir / "derived.pdf"
        raw = workdir / "ddjvu.pdf"
        tools.run("ddjvu", ["-format=pdf", tools.In(src), tools.Out(raw)], work_root, timeout=1800)
        out.write_bytes(normalize_pdf_dates(raw.read_bytes()))
        raw.unlink()
        return out, None
    if op == "pdf-pages":
        out = workdir / "pages.pdf"
        tools.run("qpdf", ["--deterministic-id", "--empty", "--pages", tools.In(src),
                           f"{step['first']}-{step['last']}", "--", tools.Out(out)], work_root, timeout=600)
        return out, None
    if op in ("page-png", "page-image"):
        fmt = "png" if op == "page-png" else step["format"]
        flag, produced = {"png": ("-png", "page.png"), "jpeg": ("-jpeg", "page.jpg"), "tiff": ("-tiff", "page.tif")}[fmt]
        out = workdir / produced
        # RGB only: pdftoppm (24.02 and 26.11) ignores -gray when it writes PNG.
        args = ["-f", str(step["page"]), "-l", str(step["page"]), "-r", str(step["dpi"]), flag]
        tools.run("pdftoppm", [*args, "-singlefile", tools.In(src), tools.Out(out, arg="page", produced=produced)],
                  work_root, timeout=600)
        return out, None
    if op == "office-convert":
        return tools.soffice_convert(src, _office_input(src), step["to"], workdir, work_root), None
    if op == "text-export":
        return tools.soffice_convert(src, _office_input(src), WRITER_TEXT, workdir, work_root), None
    if op == "excerpt":
        lines = _utf8(src).split("\n")
        if step["last_line"] > len(lines):
            raise BenchError(Code.MANIFEST_INVALID, f"excerpt beyond the {len(lines)} lines of the text")
        out = workdir / "excerpt.txt"
        out.write_bytes(("\n".join(lines[step["first_line"] - 1:step["last_line"]]) + "\n").encode("utf-8"))
        return out, None
    if op == "recode":
        text = _utf8(src)
        out = workdir / "recoded.txt"
        out.write_bytes(recode(text, step["encoding"], step["bom"]))
        return out, hashlib.sha256(oracle_text(text).encode("utf-8")).hexdigest()
    raise BenchError(Code.MANIFEST_INVALID, "derivation " + op)


def derive_input(doc, source_path, cache):
    """(path to convert, warnings, derived record or None) for a document."""
    steps = doc["input"].get("derive")
    if not steps:
        return Path(source_path), [], None
    identity = toolchain.identity()
    source_sha = sha256_file(source_path)
    key = hashlib.sha256(canonical({"input": source_sha, "steps": steps, "toolchain": identity})).hexdigest()
    target_dir = cache.derived_dir / doc["id"] / key[:24]
    meta_path = target_dir / "meta.json"
    kind = step_kinds(steps[-1])[1]
    output = target_dir / ("input" + SUFFIX[kind])
    if meta_path.is_file() and output.is_file():
        try:
            meta = json.loads(meta_path.read_bytes())
        except ValueError:
            meta = None
        if isinstance(meta, dict) and meta.get("sha256") == sha256_file(output) and meta.get("from_sha256") == source_sha \
                and meta.get("steps") == steps and meta.get("toolchain") == identity and meta.get("pages") == doc["pages"]:
            return output, _warnings(doc, meta), meta
    if target_dir.exists():
        shutil.rmtree(target_dir)
    work = cache.work("derive-" + doc["id"])
    try:
        current, oracle = Path(source_path), None
        for i, step in enumerate(steps):
            stepdir = work / f"step{i}"
            stepdir.mkdir()
            current, step_oracle = _step(step, current, stepdir, cache.tool_work_dir)
            oracle = step_oracle or oracle
        # pages counted for a PDF, one for a page image, none for a format without pages
        pages = pdf_pages(current, cache.tool_work_dir) if kind == "pdf" else 1 if kind in ("png", "jpeg", "tiff") else None
        if pages != doc["pages"]:
            raise BenchError(Code.PAGES_MISMATCH, f"{doc['id']}: {pages} pages")
        target_dir.mkdir(parents=True, exist_ok=True)
        write_atomic(output, current.read_bytes())
        meta = {"from_sha256": source_sha, "sha256": sha256_file(output), "steps": steps, "toolchain": identity,
                "pages": pages}
        if oracle:
            meta["oracle_sha256"] = oracle
        write_atomic(meta_path, json.dumps(meta, sort_keys=True).encode())
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return output, _warnings(doc, meta), meta


def _warnings(doc, meta):
    reference = doc["input"].get("reference")
    if not reference:
        return []
    if reference.get("oracle_sha256") not in (None, meta.get("oracle_sha256")):
        return [Code.DERIVED_DIFFERS]           # the oracle comes from the source and code alone
    if reference["sha256"] != meta["sha256"]:
        return [Code.DERIVED_NONDETERMINISTIC if reference.get("deterministic") is False else Code.DERIVED_DIFFERS]
    return []
