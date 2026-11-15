"""Audit of a ZIP package (docx, pptx, xlsx, odt, epub) before pandoc, run in the light child.

What it answers, for the loss policy of the contract (D9, D11 of MFL032): which media of the
package the content uses, which only the furniture (layouts, masters, headers, footers,
comments, a cover named only by metadata), which are alternates of another representation
(the SVG twin of a raster picture, the preview of an ODF formula, an image after the first non-empty one in
an ODF frame, which holds alternatives of one picture) and which no part references.
The engine compares that list with the media pandoc extracted: a `content` media file that was
not extracted is `media_lost` (incomplete), a `furniture` one `media_furniture` (warning), an
`unused` one `media_unused` (warning). A reference the audit cannot resolve never proves that a
media file is unused (PA3): such files get the role `unresolved` and count as content. A CSS
reference of a content document is never furniture: an inline `style` is content, a `style`
block or a stylesheet of the spine (whose selectors the audit does not match) is unresolved.
Objects of the content are listed as well (OOXML charts and OLE objects, OpenDocument objects
other than formulas, by their member and with the preview image of their frame when there is
one), so that an object pandoc does not convert is declared.

Two recoveries, written to the scratch directory for pandoc and the filter:
- `normalized.<ext>`: pandoc 3.14 matches the element prefixes of the OOXML readers literally,
  so a content part that binds a known namespace to another prefix (C02: `ns2` for
  `wordprocessingDrawing`) loses its pictures. Such a part is rewritten with the standard prefix
  on its elements and attributes; every original declaration stays and the standard one is
  added beside it, so values that name prefixes (`mc:Ignorable`, `Requires`) keep their
  bindings; default namespaces, comments, processing instructions and character data are written
  back as they were parsed (PA6). Other members are copied byte for byte. A part over
  `MAX_NORMALIZE` (16 MiB), or with a standard prefix already bound to another namespace, is
  not rewritten.
- `package-media.json` and `media/`: pandoc 3.14 does not read the drawings of spreadsheet
  sheets; the images anchored by the drawings that a sheet uses are copied out and listed per
  sheet, in row and column order, for the filter to place after the sheet's table.

Limits: at most `MAX_MEMBERS` members, `MAX_PART` decompressed bytes per member, `MAX_TOTAL`
decompressed bytes read in all (counted on the bytes actually decompressed, the copy included),
no DOCTYPE or entity declaration. Over a limit or on an unreadable archive the audit refuses
the package (exit 4, `package_unsafe`). Standard library only.

Usage in the child: pkgaudit FORMAT /in/source.<ext>  ->  /scratch/audit.json [, normalized.<ext>, media/]
"""
import hashlib
from html.parser import HTMLParser
import io
import json
from pathlib import Path, PurePosixPath
import re
import sys
from urllib.parse import unquote, urlsplit
import xml.parsers.expat
import zipfile

MAX_MEMBERS = 10000
MAX_PART = 64 << 20
MAX_TOTAL = 512 << 20
MAX_NORMALIZE = 16 << 20
CHUNK = 1 << 16
REFUSED_EXIT = 4
R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PKG_R = "http://schemas.openxmlformats.org/package/2006/relationships"
O = "urn:schemas-microsoft-com:office:office"
T = R + "/"
IMAGE_TYPES = (T + "image",)
OBJECT_TYPES = (T + "chart", T + "oleObject", T + "package", T + "diagramData")
SVG_BLIP = "http://schemas.microsoft.com/office/drawing/2016/SVG/main svgBlip"
XLINK_HREF = "http://www.w3.org/1999/xlink href"
MATHML = "http://www.w3.org/1998/Math/MathML math"
ODF_CHART = "urn:oasis:names:tc:opendocument:xmlns:office:1.0 chart"
# The prefixes pandoc's OOXML readers look up by name, with their namespaces.
STD = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main", "r": R,
       "wp": "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing",
       "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
       "pic": "http://schemas.openxmlformats.org/drawingml/2006/picture",
       "m": "http://schemas.openxmlformats.org/officeDocument/2006/math",
       "v": "urn:schemas-microsoft-com:vml", "o": O,
       "mc": "http://schemas.openxmlformats.org/markup-compatibility/2006",
       "p": "http://schemas.openxmlformats.org/presentationml/2006/main",
       "xdr": "http://schemas.openxmlformats.org/drawingml/2006/spreadsheetDrawing",
       "w14": "http://schemas.microsoft.com/office/word/2010/wordml",
       "w15": "http://schemas.microsoft.com/office/word/2012/wordml",
       "wps": "http://schemas.microsoft.com/office/word/2010/wordprocessingShape",
       "wpg": "http://schemas.microsoft.com/office/word/2010/wordprocessingGroup",
       "wpc": "http://schemas.microsoft.com/office/word/2010/wordprocessingCanvas",
       "w10": "urn:schemas-microsoft-com:office:word",
       "x": "urn:schemas-microsoft-com:office:excel",
       "c": "http://schemas.openxmlformats.org/drawingml/2006/chart",
       "asvg": "http://schemas.microsoft.com/office/drawing/2016/SVG/main",
       "a14": "http://schemas.microsoft.com/office/drawing/2010/main"}
STD_PREFIX = {uri: prefix for prefix, uri in STD.items()}
MEDIA_EXT = {"png", "jpg", "jpeg", "gif", "bmp", "tif", "tiff", "emf", "wmf", "svg", "webp", "eps"}
ROLE_ORDER = ("content", "unresolved", "furniture", "alternate")
BAD_ESCAPE = re.compile(r"%(?![0-9A-Fa-f]{2})")
CSS_URL = re.compile(r"""url\(\s*(?:"([^"]*)"|'([^']*)'|([^)\s]*))\s*\)""", re.I)


class Refused(Exception):
    pass


class Pkg:
    """A ZIP package read member by member, every decompressed byte counted."""

    def __init__(self, path):
        try:
            self.z = zipfile.ZipFile(path)
            infos = self.z.infolist()
        except (zipfile.BadZipFile, zipfile.LargeZipFile, OSError, ValueError, EOFError) as exc:
            raise Refused("unreadable archive: " + type(exc).__name__) from None
        if len(infos) > MAX_MEMBERS:
            raise Refused("too many members")
        self.infos = {}
        for info in infos:
            if info.filename in self.infos:
                raise Refused("duplicate member")
            self.infos[info.filename] = info
        self.names = set(self.infos)
        self.total = 0

    def stream(self, name, limit=None):
        limit = MAX_PART if limit is None else limit
        info = self.infos[name]
        if info.file_size > limit:
            raise Refused("member over the limit")
        count = 0
        try:
            with self.z.open(info) as member:
                while True:
                    chunk = member.read(CHUNK)
                    if not chunk:
                        break
                    count += len(chunk)
                    self.total += len(chunk)
                    if count > limit or self.total > MAX_TOTAL:
                        raise Refused("decompressed size over the limit")
                    yield chunk
        except (zipfile.BadZipFile, RuntimeError, NotImplementedError, OSError, EOFError, ValueError) as exc:
            raise Refused("unreadable member: " + type(exc).__name__) from None

    def read(self, name, limit=None):
        return b"".join(self.stream(name, limit))


def _refuse_dtd(*_):
    raise Refused("DOCTYPE or entity declaration in package XML")


def parser(ns=True):
    p = xml.parsers.expat.ParserCreate(namespace_separator=" " if ns else None)
    p.StartDoctypeDeclHandler = _refuse_dtd
    p.EntityDeclHandler = _refuse_dtd
    return p


def parse(pkg, name, start=None, end=None, text=None, decl=None):
    """Parse a member as a stream; ExpatError propagates to the caller."""
    p = parser()
    if start:
        p.StartElementHandler = start
    if end:
        p.EndElementHandler = end
    if text:
        p.CharacterDataHandler = text
    if decl:
        p.StartNamespaceDeclHandler = decl
    for chunk in pkg.stream(name):
        p.Parse(chunk, False)
    p.Parse(b"", True)


def resolve_target(base_dir, target):
    parts = []
    for seg in (base_dir + "/" + target).split("/"):
        if seg in ("", "."):
            continue
        if seg == "..":
            if parts:
                parts.pop()
            continue
        parts.append(seg)
    return "/".join(parts)


def rels_of(pkg, part):
    """{rId: (type, target member or None for an external target)} of a part."""
    pp = PurePosixPath(part)
    parent = str(pp.parent) if str(pp.parent) != "." else ""
    rel = (parent + "/" if parent else "") + "_rels/" + pp.name + ".rels"
    out = {}
    if rel not in pkg.names:
        return out

    def start(tag, attrs):
        if tag == PKG_R + " Relationship":
            target = attrs.get("Target", "")
            if attrs.get("TargetMode") == "External":
                resolved = None
            else:
                resolved = resolve_target("" if target.startswith("/") else parent, unquote(target))
            out[attrs.get("Id")] = (attrs.get("Type", ""), resolved)
    parse(pkg, rel, start)
    return out


class PartScan:
    """One streaming pass over a part: used rIds, rIds used only by an SVG twin, and every
    namespace declaration (prefix or None for a default, URI)."""

    def __init__(self, pkg, name):
        self.used, self.alternates, self.declarations = set(), set(), set()
        self.ok = True

        def start(tag, attrs):
            for k, v in attrs.items():
                if k in (R + " embed", R + " link", R + " id", R + " pict", O + " relid"):
                    (self.alternates if tag == SVG_BLIP else self.used).add(v)

        def decl(prefix, uri):
            self.declarations.add((prefix, uri))
        try:
            parse(pkg, name, start, decl=decl)
        except xml.parsers.expat.ExpatError:
            self.ok = False


def sha1(data):
    return hashlib.sha1(data).hexdigest()


# --- OOXML ----------------------------------------------------------------------------------

def ooxml(pkg, fmt):
    """(content parts, media {member: record}, sheets, scans of the content parts)."""
    root_rels = rels_of(pkg, "") if "_rels/.rels" in pkg.names else {}
    main = next((t for ty, t in root_rels.values() if ty == T + "officeDocument"), None)
    if not main or main not in pkg.names:
        return [], None, [], {}
    content, furniture, sheets = [], [], []
    mrels = rels_of(pkg, main)
    if fmt == "docx":
        content = [main] + [t for ty, t in mrels.values() if ty in (T + "footnotes", T + "endnotes")]
        furniture = [t for ty, t in mrels.values() if ty in (T + "header", T + "footer", T + "comments")]
    elif fmt == "pptx":
        slides = [t for ty, t in mrels.values() if ty == T + "slide"]
        content = list(slides)
        for s in slides:
            content += [t for ty, t in rels_of(pkg, s).values() if ty == T + "notesSlide"]
        furniture = [t for ty, t in mrels.values() if ty in (T + "slideMaster", T + "notesMaster", T + "handoutMaster")]
        for s in slides:
            furniture += [t for ty, t in rels_of(pkg, s).values() if ty == T + "slideLayout"]
    elif fmt == "xlsx":
        order = []

        def start(tag, attrs):
            if tag.endswith(" sheet"):
                order.append((attrs.get("name"), attrs.get(R + " id")))
        try:
            parse(pkg, main, start)
        except xml.parsers.expat.ExpatError:
            return [], None, [], {}
        for idx, (name, rid) in enumerate(order, 1):
            _, sheet = mrels.get(rid, (None, None))
            if not sheet or sheet not in pkg.names:
                continue
            content.append(sheet)
            scan = PartScan(pkg, sheet)
            entry = {"index": idx, "name": name, "part": sheet, "drawings": []}
            for rid2, (ty2, t2) in rels_of(pkg, sheet).items():
                if ty2 in (T + "drawing", T + "vmlDrawing") and t2 in pkg.names:
                    if rid2 in scan.used or not scan.ok:
                        entry["drawings"].append(t2)       # PA6: only a drawing the sheet uses
                    else:
                        furniture.append(t2)
            sheets.append(entry)
    seen, media, scans = set(), {}, {}

    def record(target, kind="image"):
        return media.setdefault(target, {"member": target, "kind": kind, "roles": set(), "from": set()})

    def visit(part, role):
        if part in seen or part not in pkg.names:
            return
        seen.add(part)
        scan = PartScan(pkg, part)
        if role == "content":
            scans[part] = scan
        for rid, (ty, target) in rels_of(pkg, part).items():
            if target is None:
                continue
            used = rid in scan.used or not scan.ok        # an unparsable part: every relation may be used
            if ty in IMAGE_TYPES and target in pkg.names:
                m = record(target)
                if used:
                    m["roles"].add(role)
                    m["from"].add(part)
                elif rid in scan.alternates:
                    m["roles"].add("alternate")
                    m["from"].add(part)
            elif ty in OBJECT_TYPES:
                m = record(target, ty.rsplit("/", 1)[-1])
                if used:
                    m["roles"].add(role)
                    m["from"].add(part)
            elif ty in (T + "drawing", T + "vmlDrawing"):
                visit(target, role if used else "furniture")      # PA6: a drawing nobody uses is no content
            elif ty in (T + "slideLayout", T + "slideMaster"):
                visit(target, "furniture")
    for s in sheets:
        for d in s["drawings"]:
            visit(d, "content")
    for p in content:
        visit(p, "content")
    for p in furniture:
        visit(p, "furniture")
    for name in pkg.names:
        if "/media/" in name and name not in media and not name.endswith("/"):
            record(name)
    return content, media, sheets, scans


def drawing_images(pkg, part):
    """Images a sheet drawing anchors: member, column, row, cell, name, description."""
    rels = rels_of(pkg, part)
    images = []
    if part.endswith(".vml"):
        cur = {}

        def start(tag, attrs):
            if tag.endswith(" shape"):
                cur.clear()
                cur["name"] = attrs.get("alt") or attrs.get("id") or ""
            elif tag.endswith(" imagedata"):
                cur["rid"] = attrs.get(O + " relid")
            elif tag.endswith(" Anchor"):
                cur["anchor"] = ""

        def text(t):
            if "anchor" in cur and len(cur["anchor"]) < 200:
                cur["anchor"] += t

        def end(tag):
            if tag.endswith(" shape") and cur.get("rid") in rels and rels[cur["rid"]][1]:
                nums = [int(x) for x in cur.get("anchor", "").replace(" ", "").split(",") if x.strip().isdigit()]
                col, row = (nums[0], nums[2]) if len(nums) >= 4 else (0, 0)
                images.append({"member": rels[cur["rid"]][1], "col": col, "row": row, "cell": col_name(col) + str(row + 1),
                               "name": cur.get("name") or "", "descr": "", "via": "vml"})
        try:
            parse(pkg, part, start, end, text)
        except xml.parsers.expat.ExpatError:
            pass        # VML is often not well-formed XML; its images stay counted by role
        return images
    stack, cur = [], {}

    def start(tag, attrs):
        local = tag.rsplit(" ", 1)[-1]
        stack.append(local)
        if local in ("twoCellAnchor", "oneCellAnchor", "absoluteAnchor"):
            cur.clear()
            cur.update(col=0, row=0)
        elif local == "cNvPr":
            cur["name"], cur["descr"] = attrs.get("name", ""), attrs.get("descr", "")
        elif local == "blip":
            cur["rid"] = attrs.get(R + " embed")

    def text(t):
        if len(stack) >= 2 and stack[-2] == "from" and stack[-1] in ("col", "row") and t.strip().isdigit():
            cur[stack[-1]] = int(t.strip())

    def end(tag):
        local = stack.pop()
        if local in ("twoCellAnchor", "oneCellAnchor", "absoluteAnchor") and cur.get("rid") in rels and rels[cur["rid"]][1]:
            images.append({"member": rels[cur["rid"]][1], "col": cur["col"], "row": cur["row"],
                           "cell": col_name(cur["col"]) + str(cur["row"] + 1),
                           "name": cur.get("name", ""), "descr": cur.get("descr", ""), "via": "drawingml"})
    try:
        parse(pkg, part, start, end, text)
    except xml.parsers.expat.ExpatError:
        pass
    return images


def col_name(c):
    s = ""
    c += 1
    while c:
        c, r = divmod(c - 1, 26)
        s = chr(65 + r) + s
    return s


# --- prefix normalization (PA6) -------------------------------------------------------------

def renames_of(declarations):
    """{prefix or None: standard prefix} for the declarations that bind a standard namespace to
    another prefix, or None when a standard prefix is bound elsewhere to another namespace."""
    renames = {}
    for prefix, uri in declarations:
        std = STD_PREFIX.get(uri)
        if std is not None and std != prefix:
            renames[prefix] = std
    if not renames:
        return {}
    for prefix, uri in declarations:
        if prefix in renames.values() and STD[prefix] != uri:
            return None
    return renames


def _escape(text, attribute=False):
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\r", "&#13;")
    if attribute:
        text = text.replace('"', "&quot;").replace("\t", "&#9;").replace("\n", "&#10;")
    return text


def rewrite(chunks):
    """The part re-serialized from expat events (no namespace processing) with the standard
    prefixes of `STD`: elements and attributes whose prefix resolves to a standard namespace get
    its standard prefix; each element that declares such a namespace keeps its declaration and
    gets the standard one too. Everything else is written as parsed."""
    out = io.StringIO()
    scopes = [{"xml": "http://www.w3.org/XML/1998/namespace"}]
    state = {"open": False, "cdata": False}

    def close():
        if state["open"]:
            out.write(">")
            state["open"] = False

    def qname(q, scope, element):
        if ":" in q:
            prefix, local = q.split(":", 1)
            if prefix in ("xmlns", "xml"):
                return q
            std = STD_PREFIX.get(scope.get(prefix))
            return std + ":" + local if std and std != prefix else q
        if element:
            std = STD_PREFIX.get(scope.get(None))
            if std:
                return std + ":" + q
        return q

    def start(name, attrs):
        close()
        pairs = list(zip(attrs[0::2], attrs[1::2]))
        scope = dict(scopes[-1])
        for k, v in pairs:
            if k == "xmlns":
                scope[None] = v or None
            elif k.startswith("xmlns:"):
                scope[k[6:]] = v
        scopes.append(scope)
        written, added = [], set()
        for k, v in pairs:
            written.append((qname(k, scope, False) if not k.startswith("xmlns") else k, v))
            uri = v if k == "xmlns" or k.startswith("xmlns:") else None
            std = STD_PREFIX.get(uri) if uri else None
            if std and k != "xmlns:" + std and "xmlns:" + std not in dict(pairs):
                added.add(("xmlns:" + std, uri))
        out.write("<" + qname(name, scope, True))
        for k, v in written + sorted(added):
            out.write(" " + k + '="' + _escape(v, True) + '"')
        state["open"] = True

    def end(name):
        scope = scopes.pop()
        if state["open"]:
            out.write("/>")
            state["open"] = False
        else:
            out.write("</" + qname(name, scope, True) + ">")

    def text(data):
        close()
        out.write(data if state["cdata"] else _escape(data))

    def cdata_start():
        close()
        out.write("<![CDATA[")
        state["cdata"] = True

    def cdata_end():
        out.write("]]>")
        state["cdata"] = False

    def comment(data):
        close()
        out.write("<!--" + data + "-->")

    def pi(target, data):
        close()
        out.write("<?" + target + (" " + data if data else "") + "?>")

    def xmldecl(version, encoding, standalone):
        out.write('<?xml version="' + (version or "1.0") + '" encoding="UTF-8"'
                  + ('' if standalone == -1 else ' standalone="' + ("yes" if standalone else "no") + '"') + "?>")
    p = parser(ns=False)
    p.ordered_attributes = True
    p.buffer_text = True
    p.StartElementHandler = start
    p.EndElementHandler = end
    p.CharacterDataHandler = text
    p.StartCdataSectionHandler = cdata_start
    p.EndCdataSectionHandler = cdata_end
    p.CommentHandler = comment
    p.ProcessingInstructionHandler = pi
    p.XmlDeclHandler = xmldecl
    for chunk in chunks:
        p.Parse(chunk, False)
    p.Parse(b"", True)
    return out.getvalue().encode("utf-8")


def normalize(pkg, scans, dst):
    """Write `dst` with the content parts that need standard prefixes rewritten; returns
    {part: ["old->std", ...]} and the parts left as they were, with the reason."""
    changed, skipped = {}, {}
    for part, scan in sorted(scans.items()):
        if not scan.ok or not part.endswith(".xml"):
            continue
        renames = renames_of(scan.declarations)
        if renames is None:
            skipped[part] = "standard prefix bound to another namespace"
            continue
        if not renames:
            continue
        if pkg.infos[part].file_size > MAX_NORMALIZE:
            skipped[part] = "part over the normalization limit"
            continue
        try:
            body = rewrite(pkg.stream(part, MAX_NORMALIZE))
        except xml.parsers.expat.ExpatError:
            skipped[part] = "not well-formed"
            continue
        changed[part] = (body, sorted(("(default)" if k is None else k) + "->" + v for k, v in renames.items()))
    if not changed:
        return {}, skipped
    with zipfile.ZipFile(dst, "x") as zout:
        for name, info in pkg.infos.items():
            copy = zipfile.ZipInfo(name, date_time=info.date_time)
            copy.compress_type, copy.external_attr, copy.create_system = info.compress_type, info.external_attr, info.create_system
            with zout.open(copy, "w") as target:
                if name in changed:
                    target.write(changed[name][0])
                else:
                    for chunk in pkg.stream(name):        # counted like every other read (PA6)
                        target.write(chunk)
    return {k: v[1] for k, v in changed.items()}, skipped


# --- OpenDocument ---------------------------------------------------------------------------

def odt(pkg):
    media = {}
    for part, role in (("content.xml", "content"), ("styles.xml", "furniture")):
        if part not in pkg.names:
            continue
        frames = []

        def start(tag, attrs, role=role, part=part):
            local = tag.rsplit(" ", 1)[-1]
            href = attrs.get(XLINK_HREF)
            if local == "frame":
                frames.append({"formula": False, "objects": [], "images": []})
            elif local in ("object", "object-ole") and frames and href:
                found = odf_object(pkg, href[2:] if href.startswith("./") else href)
                if found and found[1] == "formula":
                    frames[-1]["formula"] = True
                elif found:
                    frames[-1]["objects"].append(found)
            if local == "image" and href and not href.startswith(("http:", "https:", "#")):
                href = unquote(href[2:] if href.startswith("./") else href)
                if href in pkg.names:
                    m = media.setdefault(href, {"member": href, "kind": "image", "roles": set(), "from": set()})
                    # the preview of a formula, or an image after the first of its frame (an alternative)
                    alt = bool(frames) and ((frames[-1]["formula"] and href.startswith("ObjectReplacements/"))
                                            or bool(frames[-1]["images"]))
                    m["roles"].add("alternate" if alt else role)
                    m["from"].add(part)
                    if frames and pkg.infos[href].file_size:     # an empty image is no picture of the frame
                        frames[-1]["images"].append(href)

        def end(tag, role=role, part=part):
            if tag.rsplit(" ", 1)[-1] == "frame" and frames:
                frame = frames.pop()
                for member, kind in frame["objects"]:
                    m = media.setdefault(member, {"member": member, "kind": kind, "roles": set(), "from": set()})
                    m["roles"].add(role)
                    m["from"].add(part)
                    if frame["images"]:
                        m["preview"] = frame["images"][0]
        try:
            parse(pkg, part, start, end)
        except xml.parsers.expat.ExpatError:
            for name in pkg.names:
                if name.startswith("Pictures/"):
                    media.setdefault(name, {"member": name, "kind": "image", "roles": set(), "from": set()})["roles"].add("unresolved")
    for name in pkg.names:
        if (name.startswith("Pictures/") or name.startswith("ObjectReplacements/")) and name not in media and not name.endswith("/"):
            media[name] = {"member": name, "kind": "image", "roles": set(), "from": set()}
    return media


def odf_object(pkg, href):
    """(member, kind) of an embedded object: `formula` (a MathML document), `chart`, `object` (another
    OpenDocument) or `ole` (a binary object); None when the package does not hold it."""
    for obj in dict.fromkeys((href, unquote(href))):
        obj = obj.rstrip("/")
        name = obj + "/content.xml"
        if name in pkg.names:
            tags = []

            def start(tag, attrs):
                if not tags or tag == ODF_CHART:
                    tags.append(tag)
            try:
                parse(pkg, name, start)
            except xml.parsers.expat.ExpatError:
                return name, "object"
            if tags and tags[0] == MATHML:
                return name, "formula"
            return name, "chart" if ODF_CHART in tags else "object"
        if obj in pkg.names:
            return obj, "ole"
    return None


# --- EPUB -----------------------------------------------------------------------------------

def _decoded(value):
    """The percent-decoded path of a local URI reference, None for a remote or absolute one;
    raises ValueError when it cannot be decoded."""
    value = value.strip()
    if not value or value.startswith("#"):
        return None
    if "\\" in value or any(ord(c) < 32 for c in value):
        raise ValueError("unsafe reference")
    uri = urlsplit(value)
    if uri.scheme or uri.netloc or value.startswith("/"):
        return None
    if BAD_ESCAPE.search(uri.path):
        raise ValueError("malformed percent escape")
    path = unquote(uri.path, errors="strict")
    if any(ord(c) < 32 for c in path):
        raise ValueError("unsafe reference")
    return path or None


def srcset_urls(value):
    """The URLs of the candidates of a `srcset` (comma-separated, each URL then descriptors)."""
    urls = []
    for candidate in value.split(","):
        parts = candidate.strip().split()
        if parts:
            urls.append(parts[0])
    return urls


def epub(pkg):
    container = "META-INF/container.xml"
    if container not in pkg.names:
        return None
    roots = []

    def start(tag, attrs):
        if tag.endswith(" rootfile"):
            roots.append(attrs.get("full-path"))
    parse(pkg, container, start)
    if not roots or roots[0] not in pkg.names:
        return None
    opf = roots[0]
    base = str(PurePosixPath(opf).parent)
    base = "" if base == "." else base
    items, spine, cover = {}, [], set()
    unresolved = [0]

    def member(doc_dir, path):
        return resolve_target(doc_dir, path)

    def ostart(tag, attrs):
        if tag.endswith(" item"):
            try:
                href = _decoded(attrs.get("href", ""))
            except ValueError:
                unresolved[0] += 1
                return
            if href:
                items[attrs.get("id")] = (member(base, href), attrs.get("media-type", ""), attrs.get("properties", ""))
        elif tag.endswith(" itemref"):
            spine.append(attrs.get("idref"))
        elif tag.endswith(" meta") and attrs.get("name") == "cover":
            cover.add(attrs.get("content"))
    parse(pkg, opf, ostart)
    media = {}
    for iid, (href, mt, props) in items.items():
        if mt.startswith("image/") and href in pkg.names:
            media[href] = {"member": href, "kind": "image", "roles": set(), "from": set()}
            if iid in cover or "cover-image" in props.split():
                media[href]["roles"].add("furniture")
    css_done = set()

    def css(doc, text, role):
        doc_dir = str(PurePosixPath(doc).parent)
        doc_dir = "" if doc_dir == "." else doc_dir
        for m in CSS_URL.finditer(text):
            value = next(g for g in m.groups() if g is not None)
            try:
                path = _decoded(value)
            except ValueError:
                unresolved[0] += 1
                continue
            target = member(doc_dir, path) if path else None
            if target in media:
                media[target]["roles"].add(role)
                media[target]["from"].add(doc)

    class Spine(HTMLParser):
        RESOURCE_TAGS = ("img", "image", "object", "video", "audio", "input", "source", "embed", "a", "picture", "track")
        RESOURCE_ATTRS = ("src", "href", "xlink:href", "data", "poster")

        def __init__(self, doc):
            super().__init__(convert_charrefs=True)
            self.doc = doc
            d = str(PurePosixPath(doc).parent)
            self.dir = "" if d == "." else d
            self.in_style = False

        def mark(self, value):
            try:
                path = _decoded(value)
            except ValueError:
                unresolved[0] += 1
                return
            target = member(self.dir, path) if path else None
            if target in media:
                media[target]["roles"].add("content")
                media[target]["from"].add(self.doc)

        def handle_starttag(self, tag, attrs):
            for k, v in attrs:
                if v is None:
                    continue
                if k == "style" and "url(" in v.lower():
                    css(self.doc, v, "content")           # the element is in the body: its style is used
                if tag not in self.RESOURCE_TAGS:
                    continue
                if k in self.RESOURCE_ATTRS:
                    self.mark(v)
                elif k == "srcset":                       # PA3: responsive images are references too
                    for url in srcset_urls(v):
                        self.mark(url)
            if tag == "link":
                d = dict(attrs)
                if "stylesheet" in (d.get("rel") or "").lower() and d.get("href"):
                    try:
                        path = _decoded(d["href"])
                    except ValueError:
                        unresolved[0] += 1
                        return
                    sheet = member(self.dir, path) if path else None
                    if sheet in pkg.names and sheet not in css_done:
                        css_done.add(sheet)
                        css(sheet, pkg.read(sheet).decode("utf-8", "replace"), "unresolved")
            self.in_style = tag == "style"

        handle_startendtag = handle_starttag

        def handle_endtag(self, tag):
            if tag == "style":
                self.in_style = False

        def handle_data(self, data):
            if self.in_style and "url(" in data.lower():
                css(self.doc, data, "unresolved")
    for idref in spine:
        href = items.get(idref, (None,))[0]
        if href and href in pkg.names:
            h = Spine(href)
            h.feed(pkg.read(href).decode("utf-8", "replace"))
            h.close()
    if unresolved[0]:
        for m in media.values():
            if not m["roles"]:
                m["roles"].add("unresolved")
    return media, unresolved[0]


# --- main -----------------------------------------------------------------------------------

def audit(fmt, source, out):
    out = Path(out)
    pkg = Pkg(source)
    report = {"format": fmt, "members": len(pkg.names), "normalized": {}, "not_normalized": {},
              "unresolved_references": 0, "structure": "known", "sheets": []}
    media, sheets = None, []
    if fmt in ("docx", "pptx", "xlsx"):
        _, media, sheets, scans = ooxml(pkg, fmt)
        if media is not None:
            report["normalized"], report["not_normalized"] = normalize(pkg, scans, out / ("normalized." + fmt))
    elif fmt == "odt":
        media = odt(pkg)
    elif fmt == "epub":
        found = epub(pkg)
        if found is not None:
            media, report["unresolved_references"] = found
    else:
        raise Refused("format")
    if media is None:
        # a structure the audit does not know (for example strict OOXML): no media is proven unused
        report["structure"] = "unknown"
        media = {n: {"member": n, "kind": "image", "roles": {"unresolved"}, "from": set()}
                 for n in pkg.names if n.rsplit(".", 1)[-1].lower() in MEDIA_EXT}
    rows = []
    for m in sorted(media.values(), key=lambda x: x["member"]):
        data = pkg.read(m["member"]) if m["member"] in pkg.names else b""
        role = next((r for r in ROLE_ORDER if r in m["roles"]), "unused")
        row = {"member": m["member"], "kind": m["kind"], "role": role, "bytes": len(data), "sha1": sha1(data),
               "from": sorted(m["from"])[:3]}
        if m.get("preview"):
            row["preview"] = m["preview"]
        rows.append(row)
    if fmt == "xlsx":
        media_dir = out / "media"
        n = 0
        for s in sheets:
            s["images"] = []
            for d in s["drawings"]:
                for img in drawing_images(pkg, d):
                    ext = img["member"].rsplit(".", 1)[-1].lower()
                    if img["member"] not in pkg.names or ext not in MEDIA_EXT:
                        continue
                    n += 1
                    data = pkg.read(img["member"])
                    media_dir.mkdir(exist_ok=True)
                    name = f"m-{n:04d}.{ext}"
                    (media_dir / name).write_bytes(data)
                    s["images"].append(dict(img, file=name, sha1=sha1(data)))
            s["images"].sort(key=lambda i: (i["row"], i["col"]))
        report["sheets"] = sheets
    report["media"] = rows
    report["bytes_read"] = pkg.total
    (out / "audit.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
    return report


def main(args):
    fmt, source = args
    try:
        report = audit(fmt, source, "/scratch")
    except Refused as exc:
        Path("/scratch/audit.json").write_text(json.dumps({"refused": str(exc)}))
        return REFUSED_EXIT
    print(json.dumps({"media": len(report["media"]), "normalized": sorted(report["normalized"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
