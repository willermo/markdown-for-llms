"""The native routes of the engine: `pandoc` (born-digital formats) and `djvutxt` (DjVu text layer).

Both run in the toolchain image with the isolation of `tools.run` (no network, read-only root,
no capabilities, the caller's user, a private /work under the caps of a `tools.Watch`).

`pandoc`: doc and ods are first converted by LibreOffice to docx and xlsx; every ZIP package
(docx, pptx, xlsx, odt, epub) is audited in the light child (`pkgaudit`), which may also write a
copy with standard namespace prefixes and the images of the sheets; then

    pandoc +RTS -M<cap>m -RTS --sandbox -f <reader> -t commonmark_x --wrap=none \\
      --extract-media=assets --lua-filter=/in/contract.lua -o /work/document.md /in/source.<ext>

The filter (`filters/contract.lua`) reads the audit, writes the original extraction and its
report beside the document, and leaves in the mediabag only the media the document references.
Pandoc then reads `document.md` back (`-f commonmark_x -t references.lua`, a custom writer that
lists the destinations of images and links, nested ones included, and the raw HTML, whose img
src and a href the engine reads): a file of `assets/` that no reference names (the writer left
its image out, as an image in the alternative text of an image inside an HTML table) is
removed, and a removed file whose bytes the audit does not account for (an inline image, as
`office:binary-data` or a `data:` URI, or any format without an audit) is `media_lost` itself.
The engine then compares the audit with the files actually written to `assets/` (SHA-1): a
content media file that is not among them is `media_lost`, an object of the content whose
preview image was not written `object_not_converted` (both incomplete); furniture and unused
media are warnings.
The child of the audit gets the memory cap of the conversion, at most `AUDIT_MEMORY_MIB`.

`libreoffice` (xlsx and pptx): the package is audited, then LibreOffice writes HTML (Calc for a
workbook, every sheet; `impress_html_Export` for a presentation, one `h1` per slide and its
text) and pandoc reads it with the same filter. `source.html` must exist and not be empty
(soffice exits 0 when an export fails), otherwise `tool_failed`. The images of Calc are files
beside its HTML, mounted with their list for the filter; the HTML of Impress has none, so the
audit copies the primary images of the pictures of each slide (`p:sldIdLst` order) and the
filter places them after the heading of their slide when the headings are as many as the
slides; otherwise none is placed and each is `media_lost` with `slide_boundaries_unknown`
(`engine.json`: `images: package`, `grouped: slide`; the layout inside a slide is not rebuilt).

`djvutxt`: the text layer of each page, verbatim, under the page separator of Marker
(`{N}` and 48 dashes, N counted from 0); a file without any text is `route_invalid`.
"""
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
from urllib.parse import unquote

from . import contract, isolation, tools
from .codes import Code, EngineError

FILTER = Path(__file__).resolve().parent / "filters" / "contract.lua"
REFERENCES = Path(__file__).resolve().parent / "filters" / "references.lua"
PACKAGES = ("docx", "pptx", "xlsx", "odt", "epub")
LIBREOFFICE_TARGET = {"doc": "docx", "ods": "xlsx"}
LIBREOFFICE_EXPORT = {"xlsx": "html", "pptx": "html:impress_html_Export"}
AUDIT_MODULE = "markdown_for_llms.engine.pkgaudit"
AUDIT_MEMORY_MIB = 2048
PAGE_SEPARATOR = "-" * 48


def filter_sha256():
    return hashlib.sha256(FILTER.read_bytes()).hexdigest()


def libreoffice(ctx, src, fmt):
    """`src` (doc or ods) converted by LibreOffice in its private /work; returns the new file."""
    target = LIBREOFFICE_TARGET[fmt]
    dest = ctx.work / "libreoffice" / ("source." + target)
    ctx.event("libreoffice")
    tools.run("soffice", ["-env:UserInstallation=file:///work/profile", "--headless", "--norestore",
                          "--convert-to", target, "--outdir",
                          tools.Out(dest, arg="out", produced=f"out/source.{target}"),
                          tools.In(src, name="source." + fmt)],
              ctx.tool_root, timeout=ctx.timeout(), memory_mb=ctx.memory_mib, watch=ctx.watch())
    return dest, target


def audit(ctx, path, fmt, flags=()):
    """The package audit in the light child; returns (audit dict, its directory)."""
    scratch = ctx.work / "audit"
    ctx.event("audit")
    try:
        rc, log = isolation.run(scratch, AUDIT_MODULE, [fmt, "/in/source." + fmt, *flags], inputs=[(path, "source." + fmt)],
                                memory_mib=min(AUDIT_MEMORY_MIB, ctx.memory_mib), output_mib=ctx.output_mib, timeout=min(600, ctx.timeout()),
                                steps=ctx.steps, observer=ctx.observer)
    except EngineError as exc:
        ctx.diagnostics["audit.log"] = getattr(exc, "log", b"")
        raise
    ctx.diagnostics["audit.log"] = log
    from .child import MEMORY_EXIT
    from .pkgaudit import REFUSED_EXIT
    if rc == REFUSED_EXIT:
        raise EngineError(Code.PACKAGE_UNSAFE, "package audit refused the package")
    if rc == MEMORY_EXIT:
        raise EngineError(Code.MEMORY_LIMIT, "package audit")
    if rc != 0 or not (scratch / "audit.json").is_file():
        raise EngineError(Code.ENGINE_FAILED, f"package audit exit {rc}")
    return json.loads((scratch / "audit.json").read_text(encoding="utf-8")), scratch


def asset_digests(stage):
    found = {}
    assets = stage / "assets"
    if assets.is_dir():
        for p in sorted(assets.iterdir()):
            if p.is_file() and not p.is_symlink():
                found[hashlib.sha1(p.read_bytes()).hexdigest()] = "assets/" + p.name
    return found


class HtmlTargets(HTMLParser):
    """The src of img and the href of a in raw HTML, with their character references decoded."""
    KEYS = {"img": "src", "a": "href"}

    def __init__(self, found):
        super().__init__(convert_charrefs=True)
        self.found = found

    def handle_starttag(self, tag, attrs):
        for key, value in attrs:
            if key == self.KEYS.get(tag) and value:
                self.found.add(value)

    handle_startendtag = handle_starttag


def references(ctx, stage):
    """The local targets `document.md` references, as pandoc's commonmark_x reader reads it."""
    dest = ctx.work / "references.json"
    ctx.event("references")
    tools.run("pandoc", ["-f", "commonmark_x", "-t", tools.In(REFERENCES, name="references.lua"),
                         "-o", tools.Out(dest, arg="references.json"), tools.In(stage / "document.md", name="document.md")],
              ctx.tool_root, timeout=ctx.timeout(), memory_mb=ctx.memory_mib, watch=ctx.watch())
    data = json.loads(dest.read_text(encoding="utf-8"))
    found = {t for t in data.get("targets") or [] if isinstance(t, str)}     # an empty Lua table is {}
    for chunk in data.get("html") or []:
        if isinstance(chunk, str):
            parser = HtmlTargets(found)
            parser.feed(chunk)
            parser.close()
    return found | {unquote(t.split("#", 1)[0].split("?", 1)[0]).removeprefix("./") for t in found}


def drop_unreferenced(ctx, stage):
    """Remove the files of `assets/` that no reference of `document.md` names; returns {path: SHA-1}."""
    assets = stage / "assets"
    if not assets.is_dir() or assets.is_symlink() or not any(assets.iterdir()):
        return {}
    named = references(ctx, stage)
    dropped = {}
    for p in sorted(assets.iterdir()):
        if p.is_file() and not p.is_symlink() and ("assets/" + p.name) not in named:
            dropped["assets/" + p.name] = hashlib.sha1(p.read_bytes()).hexdigest()
            p.unlink()
    if not any(assets.iterdir()):
        assets.rmdir()
    return dropped


def media_problems(report, extracted):
    """Problems of the loss policy (D9, D11) from the audit and the SHA-1 of the written assets."""
    problems, counts = [], {}
    sha1_of = {m["member"]: m["sha1"] for m in report.get("media", []) if m.get("bytes")}
    for m in report.get("media", []):
        if not m.get("bytes"):
            continue
        written = m["sha1"] in extracted
        role, kind = m["role"], m.get("kind", "image")
        counts[role] = counts.get(role, 0) + 1
        if kind != "image":
            # an object is converted only as its preview image written to assets/
            shown = m.get("preview") in sha1_of and sha1_of[m["preview"]] in extracted
            if role in ("content", "unresolved") and not shown:
                problems.append(contract.problem("object_not_converted", path=m["member"], detail=kind))
            continue
        if written or role == "alternate":
            continue
        if role in ("content", "unresolved"):
            problems.append(contract.problem("media_lost", path=m["member"],
                                             detail=None if role == "content" else "reference not resolved"))
        elif role == "furniture":
            problems.append(contract.problem("media_furniture", path=m["member"]))
        else:
            problems.append(contract.problem("media_unused", path=m["member"]))
    return problems, counts


def pandoc_route(ctx, src, fmt):
    record = {"reader": fmt, "filter_sha256": filter_sha256(),
              "references_sha256": hashlib.sha256(REFERENCES.read_bytes()).hexdigest()}
    path, reader = Path(src), fmt
    if fmt in LIBREOFFICE_TARGET:
        path, reader = libreoffice(ctx, src, fmt)
        record.update(reader=reader, pre_step={"tool": "soffice", "from": fmt, "to": reader})
    package = None
    extra = []
    if reader in PACKAGES:
        package, audit_dir = audit(ctx, path, reader)
        normalized = audit_dir / ("normalized." + reader)
        if package.get("normalized") and normalized.is_file():
            path = normalized
            ctx.corrections.append({"kind": "package_prefixes_normalized", "original": ", ".join(sorted(package["normalized"]))})
        extra = [tools.In(audit_dir / "audit.json", name="package-media.json", argument=False)]
        if reader == "xlsx" and any(s.get("images") for s in package.get("sheets", [])):
            extra.append(tools.In(audit_dir / "media", name="media", directory=True, argument=False))
    run_pandoc(ctx, path, reader, package, extra, record)
    return record


def libreoffice_route(ctx, src, fmt):
    """xlsx and pptx through the HTML of LibreOffice, which pandoc reads with the contract filter."""
    package, audit_dir = audit(ctx, src, fmt, ("slides",) if fmt == "pptx" else ())
    export = ctx.work / "libreoffice-html"
    ctx.event("libreoffice")
    tools.run("soffice", ["-env:UserInstallation=file:///work/profile", "--headless", "--norestore",
                          "--convert-to", LIBREOFFICE_EXPORT[fmt], "--outdir",
                          tools.Out(export, arg="out", directory=True), tools.In(src, name="source." + fmt)],
              ctx.tool_root, timeout=ctx.timeout(), memory_mb=ctx.memory_mib, watch=ctx.watch())
    html = export / "source.html"
    if html.is_symlink() or not html.is_file() or html.stat().st_size == 0:
        raise EngineError(Code.TOOL_FAILED, "soffice wrote no HTML")       # it exits 0 when an export fails
    record = {"export": LIBREOFFICE_EXPORT[fmt], "reader": "html", "filter_sha256": filter_sha256(),
              "references_sha256": hashlib.sha256(REFERENCES.read_bytes()).hexdigest()}
    extra = [tools.In(audit_dir / "audit.json", name="package-media.json", argument=False)]
    if fmt == "xlsx":
        names = sorted(p.name for p in export.iterdir() if p.name != html.name and p.is_file() and not p.is_symlink())
        listing = ctx.work / "lo-media.json"
        listing.write_text(json.dumps(names), encoding="utf-8")
        extra.append(tools.In(listing, name="lo-media.json", argument=False))
        if names:
            extra.append(tools.In(export, name="lo-media", directory=True, argument=False))
        record["images"] = "export"
    else:
        if (audit_dir / "media").is_dir():
            extra.append(tools.In(audit_dir / "media", name="media", directory=True, argument=False))
        record.update(images="package", grouped="slide",
                      slide_images=sum(len(s["images"]) for s in package.get("slides", [])))
    report = run_pandoc(ctx, html, "html", package, extra, record)
    if fmt == "pptx":
        record["slide_boundaries"] = report.get("slide_boundaries") or "unknown"
        if record["slide_boundaries"] != "known":
            # no image is placed: each one is media_lost, and says why
            members = {img["member"] for s in package.get("slides", []) for img in s["images"]}
            for p in ctx.problems:
                if p["code"] == "media_lost" and p.get("path") in members:
                    p["detail"] = "slide_boundaries_unknown"
    return record


def run_pandoc(ctx, path, reader, package, extra, record):
    """pandoc on `path` read as `reader`, with the contract filter and the `extra` inputs; then
    the unreferenced assets and the loss policy of the audit `package` (None: no package). Sets
    `record["media_extracted"]` and returns the report of the filter."""
    stage, diag = ctx.stage, ctx.stage / "diagnostics"
    report_path = ctx.work / "contract-report.json"
    ctx.event("pandoc")
    tools.run("pandoc", ["-f", reader, "-t", "commonmark_x", "--wrap=none", "--extract-media=assets",
                         tools.In(FILTER, name="contract.lua", prefix="--lua-filter="), *extra,
                         "-o", tools.Out(stage / "document.md", arg="document.md", allow_empty=True),
                         tools.Out(stage / "assets", arg="assets", directory=True, optional=True, argument=False),
                         tools.Out(diag / "pre-filter.md", arg="pre-filter.md", allow_empty=True, argument=False),
                         tools.Out(report_path, arg="contract-report.json", argument=False),
                         tools.In(path, name="source." + reader)],
              ctx.tool_root, timeout=ctx.timeout(), memory_mb=ctx.memory_mib, watch=ctx.watch())
    report = json.loads(report_path.read_text(encoding="utf-8"))
    for item in report.get("corrections", []):
        ctx.corrections.append(item)
    ctx.correction_counts.update({k: v for k, v in report.get("counts", {}).items()})
    listed = {}
    for p in report.get("problems", []):
        ctx.problems.append(contract.problem(p["code"], path=p.get("path")))
        listed[p["code"]] = listed.get(p["code"], 0) + 1
    for code, n in report.get("problem_counts", {}).items():       # counted by the filter beyond its list
        contract.problem(code)
        if n > listed.get(code, 0):
            ctx.unlisted_problems[code] = ctx.unlisted_problems.get(code, 0) + n - listed.get(code, 0)
    dropped = drop_unreferenced(ctx, stage)
    if dropped:
        ctx.correction_counts["asset_unreferenced_removed"] = len(dropped)
        # the audit accounts for the bytes of the content images it lists (media_problems below
        # reports them); the others are lost here. A member with the same bytes but another role
        # (unused, alternate, furniture) does not account for an inline image that was dropped.
        kept = asset_digests(stage)
        audited = {m["sha1"] for m in (package or {}).get("media", [])
                   if m.get("bytes") and m.get("kind", "image") == "image" and m["role"] in ("content", "unresolved")}
        original = {m["path"]: m.get("original") for m in report.get("media", [])}
        for path, digest in dropped.items():
            if digest not in kept and digest not in audited:
                ctx.problems.append(contract.problem("media_lost", path=original.get(path) or path,
                                                     detail="not in document.md"))
    record["media_extracted"] = len(report.get("media", [])) - len(dropped)
    if package is not None:
        problems, roles = media_problems(package, asset_digests(stage))
        ctx.problems.extend(problems)
        ctx.package = {"format": package.get("format", reader), "members": package.get("members"), "structure": package.get("structure"),
                       "media_roles": roles, "normalized": package.get("normalized", {}),
                       "not_normalized": package.get("not_normalized", {}),
                       "unresolved_references": package.get("unresolved_references", 0),
                       "sheet_images": sum(len(s.get("images", [])) for s in package.get("sheets", []))}
    return report


ESCAPES = {ord("n"): b"\n", ord("t"): b"\t", ord("r"): b"\r", ord("b"): b"\b", ord("f"): b"\f", ord("v"): b"\v",
           ord("a"): b"\a", ord("\\"): b"\\", ord('"'): b'"'}


def page_texts(raw):
    """The text of each page from `djvutxt -detail=page`: one S-expression per page, `()` for a
    page without text layer, `(page x0 y0 x1 y1 "text")` otherwise (C escapes, UTF-8)."""
    pages, depth, i, n = [], 0, 0, len(raw)
    current = None
    while i < n:
        c = raw[i]
        if c in b" \t\r\n\f":
            i += 1
        elif c == ord("("):
            if depth == 0:
                current = []
            depth += 1
            i += 1
        elif c == ord(")"):
            depth -= 1
            if depth < 0:
                raise EngineError(Code.CONVERTER_FAILED, "djvutxt: unbalanced output")
            if depth == 0:
                pages.append("".join(current))
            i += 1
        elif c == ord('"'):
            i += 1
            buf = bytearray()
            while True:
                if i >= n:
                    raise EngineError(Code.CONVERTER_FAILED, "djvutxt: unterminated string")
                c = raw[i]
                if c == ord('"'):
                    i += 1
                    break
                if c == ord("\\") and i + 1 < n:
                    e = raw[i + 1]
                    if e in ESCAPES:
                        buf += ESCAPES[e]
                        i += 2
                    elif 48 <= e <= 55:
                        j = i + 1
                        while j < n and j < i + 4 and 48 <= raw[j] <= 55:
                            j += 1
                        buf.append(int(raw[i + 1:j], 8) & 0xFF)
                        i = j
                    else:
                        buf.append(e)
                        i += 2
                else:
                    buf.append(c)
                    i += 1
            if depth == 0 or current is None:
                raise EngineError(Code.CONVERTER_FAILED, "djvutxt: string outside a page")
            try:
                current.append(buf.decode("utf-8", "strict"))
            except UnicodeDecodeError:
                raise EngineError(Code.CONVERTER_FAILED, "djvutxt: text is not UTF-8") from None
        else:
            while i < n and raw[i] not in b' \t\r\n\f()"':
                i += 1
            if depth == 0:
                raise EngineError(Code.CONVERTER_FAILED, "djvutxt: atom outside a page")
    if depth != 0:
        raise EngineError(Code.CONVERTER_FAILED, "djvutxt: unbalanced output")
    return pages


def djvutxt_route(ctx, src):
    ctx.event("djvused")
    count = tools.run("djvused", ["-e", "n", tools.In(src, name="source.djvu")], ctx.tool_root,
                      timeout=ctx.timeout(), memory_mb=ctx.memory_mib, watch=ctx.watch())
    try:
        pages = int(count.decode("ascii").strip())
    except ValueError:
        raise EngineError(Code.CONVERTER_FAILED, "djvused page count") from None
    if pages < 1:
        raise EngineError(Code.CONVERTER_FAILED, "no pages")
    if pages > ctx.max_pages:
        raise EngineError(Code.PAGE_LIMIT, f"{pages} pages, limit {ctx.max_pages}")
    ctx.event("djvutxt")
    raw = tools.run("djvutxt", ["-detail=page", tools.In(src, name="source.djvu")], ctx.tool_root,
                    timeout=ctx.timeout(), memory_mb=ctx.memory_mib, watch=ctx.watch())
    texts = page_texts(raw)
    if len(texts) != pages:
        raise EngineError(Code.CONVERTER_FAILED, f"djvutxt gave {len(texts)} pages of {pages}")
    if not any(t.strip() for t in texts):
        raise EngineError(Code.ROUTE_INVALID, "the DjVu file has no text layer")
    parts, records = [], []
    for i, t in enumerate(texts):
        t = t.replace("\r\n", "\n").replace("\r", "\n").strip("\n")
        if "\x00" in t:
            raise EngineError(Code.CONVERTER_FAILED, "NUL in the text layer")
        chars = sum(1 for c in t if not c.isspace())
        records.append({"page": i + 1, "chars": chars})
        if chars == 0:
            ctx.problems.append(contract.problem("page_blank", page=i + 1))
        parts.append("\n\n{" + str(i) + "}" + PAGE_SEPARATOR + "\n\n" + t)
    (ctx.stage / "document.md").write_text("".join(parts).lstrip("\n") + "\n", encoding="utf-8")
    ctx.pages = {"count": pages, "method": "djvutxt text layer", "pages": records}
    return {"pages": pages, "separator": "{N}" + PAGE_SEPARATOR}
