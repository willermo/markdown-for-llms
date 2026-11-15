"""Self-test of the reference toolchain, run by `mdf bench prepare --toolchain` on the image just
built, before it is tagged, through the runner of the bench (`tools.run`): what the bench needs of
the tools and of the container, on synthetic files only.

versions        each tool reports the version of `toolchain.json`
isolation       a host file outside the mounts is not visible in the container
sandbox         pandoc does not read a file named by `\\input`, although it is mounted
runtime-cap     pandoc stops at the cap of its runtime: exit 251, `tool_memory_limit`
data-limit      poppler under `--ulimit data`: "Out of memory" with exit 0, `tool_memory_limit`
group-memory    pandoc above the memory of its container: exit 137, `tool_memory_limit`
soffice         LibreOffice converts in its private /work and the output exists
qpdf, ddjvu, pdftoppm   derivations of synthetic files with the SHA-256 of `toolchain.json`
                (ddjvu after the normalization of the PDF dates)
timeout         a call past its timeout is removed by name: no container left
"""
import base64
import hashlib
import os
from pathlib import Path
import re
import shutil

from . import derive
from ..engine import toolchain, tools
from .codes import BenchError, Code

VERSION_COMMANDS = {
    "ddjvu": (["--help"], r"DjVuLibre-([\d.]+)"),
    "qpdf": (["--version"], r"qpdf version ([\d.]+)"),
    "pdftoppm": (["-v"], r"pdftoppm version ([\d.]+)"),
    "pdftotext": (["-v"], r"pdftotext version ([\d.]+)"),
    "pdfinfo": (["-v"], r"pdfinfo version ([\d.]+)"),
    "pandoc": (["--version"], r"pandoc ([\d.]+)"),
    "soffice": (["--version"], r"LibreOffice ([\d.]+)"),
    "gs": (["--version"], r"^([\d.]+)"),
}
# A 64x32 bilevel page (a frame and a checkerboard), encoded once with `cjb2 -dpi 72` of the image.
SYNTHETIC_DJVU = base64.b64decode(
    "QVQmVEZPUk0AAABGREpWVUlORk8AAAAKAEAAIBgASAAWAVNqYnoAAAAogfrfWWzOwnf8jiHljLClXij11//qvzdaVlTfBVsWIshQd8xACUn/Jw==")
# PostScript that never ends: the timeout of the call stops it.
ENDLESS_PS = b"%!PS\n{ 1 pop } loop\n"


def make_pdf(pages, date=b"D:20261114120000+01'00'"):
    """A small valid PDF with one line of Helvetica text per page (offsets computed)."""
    objects = [b"<< /Type /Catalog /Pages 2 0 R >>", None,
               b"<< /Producer (bench test) /CreationDate (" + date + b") /ModDate (" + date + b") >>",
               b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
    kids = []
    for text in pages:
        stream = b"BT /F1 24 Tf 72 720 Td (" + text.encode("latin-1") + b") Tj ET"
        objects.append(b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream")
        content = len(objects)
        objects.append(b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> "
                       b"/Contents " + str(content).encode() + b" 0 R >>")
        kids.append(len(objects))
    objects[1] = (b"<< /Type /Pages /Kids [" + b" ".join(str(k).encode() + b" 0 R" for k in kids)
                  + b"] /Count " + str(len(kids)).encode() + b" >>")
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, 1):
        offsets.append(len(out))
        out += str(number).encode() + b" 0 obj\n" + body + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 " + str(len(objects) + 1).encode() + b"\n0000000000 65535 f \n"
    out += b"".join(b"%010d 00000 n \n" % o for o in offsets)
    out += (b"trailer\n<< /Size " + str(len(objects) + 1).encode() + b" /Root 1 0 R /Info 3 0 R >>\nstartxref\n"
            + str(xref).encode() + b"\n%%EOF\n")
    return bytes(out)


def growing_html(paragraphs):
    """HTML whose Pandoc AST grows with the number of paragraphs (memory caps)."""
    return ("<html><body>" + "".join(f"<p>word{i} <b>bold</b> <i>it</i></p>" for i in range(paragraphs))
            + "</body></html>").encode()


def _fail(step, detail):
    raise BenchError(Code.TOOLCHAIN_MISMATCH, f"self-test {step}: {detail}")


def _expect_code(step, code, call):
    try:
        call()
    except BenchError as exc:
        if exc.code != code:
            _fail(step, f"{exc.code.value} instead of {code.value}")
        return exc.detail
    _fail(step, f"no {code.value}")


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_selftest(cache, out=print):
    """Run every step; raises `toolchain_mismatch` at the first one that fails. Returns {step: result}."""
    root = cache.tool_work_dir
    expected = toolchain.spec()["selftest"]
    work = cache.work("selftest")
    results = {}

    def done(step, result):
        results[step] = result
        out(f"selftest {step} ok {result}")

    try:
        pdf = work / "four.pdf"
        pdf.write_bytes(make_pdf(["alpha", "beta", "gamma", "delta"]))

        found = {}
        for name, (args, pattern) in VERSION_COMMANDS.items():
            _, stdout, stderr = tools.probe(name, args, root)
            m = re.search(pattern, (stdout + stderr).decode("utf-8", "replace"), re.MULTILINE)
            found[name] = m.group(1) if m else None
        if found != toolchain.versions():
            _fail("versions", f"{found}")
        done("versions", " ".join(f"{k} {v}" for k, v in found.items()))

        _expect_code("isolation", Code.TOOL_FAILED, lambda: tools.run("pdfinfo", [str(pdf)], root, timeout=120))
        tools.run("pdfinfo", [tools.In(pdf)], root, timeout=120)
        done("isolation", "host path not visible, the same file mounted is")

        secret = work / "sentinel.tex"
        token = "SENTINEL" + os.urandom(6).hex().upper()
        secret.write_text(token + "\n")
        text = tools.run("pandoc", ["-f", "latex", "-t", "plain", tools.In(secret, name="sentinel.tex")], root,
                         stdin=None).decode()
        if token not in text:
            _fail("sandbox", "the mounted file is not readable from the command line")
        source = b"Before \\input{/in/sentinel.tex} after.\n"
        # the same file mounted at the same path but not an input of the command line (a metadata
        # value), named by \input in the source on stdin: the sandbox must keep pandoc from reading it
        rendered = tools.run("pandoc", ["-f", "latex", "-t", "plain",
                                        tools.In(secret, name="sentinel.tex", prefix="--metadata=unused:")],
                             root, stdin=source)
        if token in rendered.decode() or b"Before" not in rendered:
            _fail("sandbox", "\\input was read")
        done("sandbox", "\\input not read, exit 0")

        detail = _expect_code("runtime-cap", Code.TOOL_MEMORY_LIMIT, lambda: tools.run(
            "pandoc", ["-f", "html", "-t", "json"], root, stdin=growing_html(60000), memory_mb=64))
        if not detail.endswith("exit 251"):
            _fail("runtime-cap", detail)
        done("runtime-cap", detail)

        detail = _expect_code("data-limit", Code.TOOL_MEMORY_LIMIT, lambda: tools.run(
            "pdftoppm", ["-r", "1200", "-png", "-singlefile", tools.In(pdf), tools.Out(work / "big.png", arg="big",
                                                                                        produced="big.png")],
            root, memory_mb=128))
        if not detail.endswith("exit 0"):
            _fail("data-limit", detail)
        done("data-limit", detail)

        detail = _expect_code("group-memory", Code.TOOL_MEMORY_LIMIT, lambda: tools.run(
            "pandoc", ["-f", "html", "-t", "json"], root, stdin=growing_html(300000), memory_mb=4096, group_mb=300))
        if not detail.endswith("exit 137"):
            _fail("group-memory", detail)
        done("group-memory", detail)

        note = work / "note.txt"
        note.write_text("A line converted by LibreOffice.\n")
        converted = tools.soffice_convert(note, "note.txt", "pdf", work, root)
        if not converted.read_bytes().startswith(b"%PDF"):
            _fail("soffice", "not a PDF")
        done("soffice", "PDF written in /work and moved out")

        step = work / "qpdf"
        step.mkdir()
        got = _sha(derive._run_step({"op": "pdf-pages", "first": 2, "last": 3}, pdf, step, root))
        if got != expected["qpdf"]:
            _fail("qpdf", got)
        done("qpdf", got[:16])

        djvu = work / "box.djvu"
        djvu.write_bytes(SYNTHETIC_DJVU)
        step = work / "ddjvu"
        step.mkdir()
        got = _sha(derive._run_step({"op": "djvu-to-pdf"}, djvu, step, root))
        if got != expected["ddjvu"]:
            _fail("ddjvu", got)
        done("ddjvu", got[:16])

        step = work / "pdftoppm"
        step.mkdir()
        got = _sha(derive._run_step({"op": "page-png", "page": 2, "dpi": 72, "color": "rgb"}, pdf, step, root))
        if got != expected["pdftoppm"]:
            _fail("pdftoppm", got)
        done("pdftoppm", got[:16])

        endless = work / "endless.ps"
        endless.write_bytes(ENDLESS_PS)
        detail = _expect_code("timeout", Code.TOOL_FAILED, lambda: tools.run(
            "gs", ["-q", "-dNOPAUSE", "-dBATCH", "-dSAFER", "-sDEVICE=nullpage", tools.In(endless)], root, timeout=5))
        left = toolchain.engine_run(["ps", "-a", "--filter", f"name=mdf-bench-{os.getpid()}-", "--format", "{{.Names}}"], 60)
        if not detail.endswith("timeout") or left.returncode != 0 or left.stdout.strip():
            _fail("timeout", f"{detail}; left: {left.stdout.decode().split()}")
        done("timeout", "container removed by name")
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return results
