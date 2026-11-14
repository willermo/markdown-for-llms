"""Deterministic derivations of the input to convert: DjVu to PDF, PDF pages, a page as PNG.

They run with the reference toolchain (`toolchain.py`), whose identity, not a version read from
a tool, enters the key: a derivation found in the cache needs no container, so a cache prepared
elsewhere and copied works without Docker (U4); a missing or altered one is derived again, which
fails closed (`toolchain_missing`) where there is no engine. The bench records the hash of the
starting file, of the derived file and of the reference in the manifest (`reference.toolchain`
names the toolchain that produced it). A derived file that differs from the reference is a
warning, not an unavailable document.
"""
import hashlib
import json
from pathlib import Path
import re
import shutil

from . import toolchain, tools
from .cache import canonical, sha256_file, write_atomic
from .codes import BenchError, Code

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


def _run_step(step, src, workdir, work_root):
    op = step["op"]
    if op == "djvu-to-pdf":
        out = workdir / "derived.pdf"
        raw = workdir / "ddjvu.pdf"
        tools.run("ddjvu", ["-format=pdf", tools.In(src), tools.Out(raw)], work_root, timeout=1800)
        out.write_bytes(normalize_pdf_dates(raw.read_bytes()))
        raw.unlink()
        return out
    if op == "pdf-pages":
        out = workdir / "pages.pdf"
        tools.run("qpdf", ["--deterministic-id", "--empty", "--pages", tools.In(src),
                           f"{step['first']}-{step['last']}", "--", tools.Out(out)], work_root, timeout=600)
        return out
    if op == "page-png":
        out = workdir / "page.png"
        # RGB only: pdftoppm (24.02 and 26.11) ignores -gray when it writes PNG.
        args = ["-f", str(step["page"]), "-l", str(step["page"]), "-r", str(step["dpi"]), "-png"]
        tools.run("pdftoppm", [*args, "-singlefile", tools.In(src), tools.Out(out, arg="page", produced="page.png")],
                  work_root, timeout=600)
        return out
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
    suffix = ".png" if steps[-1]["op"] == "page-png" else ".pdf"
    output = target_dir / ("input" + suffix)
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
        current = Path(source_path)
        for i, step in enumerate(steps):
            stepdir = work / f"step{i}"
            stepdir.mkdir()
            current = _run_step(step, current, stepdir, cache.tool_work_dir)
        pages = 1 if suffix == ".png" else pdf_pages(current, cache.tool_work_dir)
        if pages != doc["pages"]:
            raise BenchError(Code.PAGES_MISMATCH, f"{doc['id']}: {pages} pages")
        target_dir.mkdir(parents=True, exist_ok=True)
        write_atomic(output, current.read_bytes())
        meta = {"from_sha256": source_sha, "sha256": sha256_file(output), "steps": steps, "toolchain": identity,
                "pages": pages}
        write_atomic(meta_path, json.dumps(meta, sort_keys=True).encode())
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return output, _warnings(doc, meta), meta


def _warnings(doc, meta):
    reference = doc["input"].get("reference")
    if reference and reference["sha256"] != meta["sha256"]:
        return [Code.DERIVED_DIFFERS]
    return []
