"""EPUB twin reader: the spine's XHTML documents through an (X)HTML walker.

Headings h1-h6, paragraphs with emphasis and MathML, nested lists, tables with spans, figures
with captions, footnotes (`epub:type` footnote/endnote, or a footnotes section). Navigation,
scripts and styles are skipped. Note markers in the text are not part of the paragraph.
Bibliography entries (`class="csl-entry"`, `role="doc-biblioentry"`, `epub:type="biblioentry"`) are
bibliography entries, not paragraphs; an inline `svg` (a cover, a drawing) is a figure.
"""
import posixpath
import re

from ..model import Builder, Cell
from .mathml import mathml_linear
from .htmltree import parse_html
from .safe import SafeZip, local, parse_xml, text_of

OPF = "{http://www.idpf.org/2007/opf}"
CONTAINER = "{urn:oasis:names:tc:opendocument:xmlns:container}"
EPUB_TYPE = "{http://www.idpf.org/2007/ops}type"
NOTE_TYPES = {"footnote", "endnote", "rearnote", "footnotes", "endnotes", "rearnotes"}
SKIP = {"script", "style", "head", "nav", "template", "svg"}
BLOCK = {"p", "div", "section", "article", "aside", "header", "footer", "main", "blockquote", "ul", "ol", "li",
         "table", "figure", "h1", "h2", "h3", "h4", "h5", "h6", "dl", "pre", "hr", "figcaption", "body"}


def _clean(text):
    return re.sub(r"\s+", " ", text).strip()


def _epub_type(el):
    return (el.get(EPUB_TYPE) or el.get("epub:type") or "")


def _is_note(el):
    types = set(_epub_type(el).split()) | set((el.get("role") or "").replace("doc-", "").split())
    classes = set((el.get("class") or "").split())
    return bool(types & NOTE_TYPES) or bool({"footnotes", "endnotes"} & classes)


def _is_noteref(el):
    return local(el.tag) == "a" and ("noteref" in _epub_type(el) or "doc-noteref" in (el.get("role") or "")
                                     or "footnote-ref" in (el.get("class") or ""))


def _inline(el, skip=()):
    math, emph, parts = [], [], []

    def rec(x):
        name = local(x.tag)
        if _is_noteref(x) or name in SKIP or (name in skip and x is not el):
            parts.append(x.tail or "")
            return
        if name == "math":
            math.append(mathml_linear(x))
            parts.append(x.tail or "")
            return
        start = len(parts)
        parts.append(x.text or "")
        for c in x:
            rec(c)
        if name in ("em", "i", "strong", "b") and x is not el:
            t = _clean("".join(parts[start:]))
            if t:
                emph.append(("i" if name in ("em", "i") else "b", t))
        if x is not el:
            parts.append(x.tail or "")
    rec(el)
    return _clean("".join(parts)), tuple(math), tuple(emph)


def _is_bib_entry(el):
    return ("csl-entry" in (el.get("class") or "").split() or "doc-biblioentry" in (el.get("role") or "")
            or "biblioentry" in _epub_type(el).split())


def _has_block_child(el):
    return any(local(c.tag) in BLOCK for c in el)


def walk(el, b, list_depth=0, in_note=False):
    for child in el:
        name = local(child.tag)
        if name == "svg":
            b.add("figure")             # a picture; its text, if any, is not read
            continue
        if name in SKIP:
            continue
        if _is_note(child) and not in_note:
            _notes(child, b)
            continue
        if re.fullmatch(r"h[1-6]", name):
            text = _clean(text_of(child, SKIP))
            if text:
                b.add("heading", text, level=int(name[1]))
        elif _is_bib_entry(child):
            text, _, _ = _inline(child)
            if text:
                b.add("bib_entry", text)
        elif name in ("p", "pre") or (name in ("div", "blockquote") and not _has_block_child(child) and _clean(text_of(child))):
            text, math, emph = _inline(child)
            if not text and len(math) == 1:
                b.add("formula", math[0], display=True, math_kind="mathml")
            elif text or math:
                kind = "list_item" if list_depth else "paragraph"
                b.add(kind, text, level=list_depth, math=math, emphasis=emph, math_kind="mathml")
            for img in (x for x in child.iter() if local(x.tag) in ("img", "svg")):
                b.add("figure")
        elif name in ("ul", "ol"):
            _list(child, b, list_depth + 1)
        elif name == "table":
            _table(child, b)
        elif name == "figure":
            cap = next((x for x in child.iter() if local(x.tag) == "figcaption"), None)
            b.add("figure", _clean(text_of(cap)) if cap is not None else "")
        elif name == "img":
            b.add("figure", _clean(child.get("alt") or ""))
        elif name == "math":
            b.add("formula", mathml_linear(child), display=True, math_kind="mathml")
        else:
            walk(child, b, list_depth, in_note)


def _list(lst, b, depth):
    for li in lst:
        if local(li.tag) != "li":
            continue
        if _has_block_child(li):
            text, math, emph = _inline(li, skip={"ul", "ol", "table", "figure"})
            if text:
                b.add("list_item", text, level=depth, math=math, emphasis=emph, math_kind="mathml")
            for x in li:
                if local(x.tag) in ("ul", "ol"):
                    _list(x, b, depth + 1)
        else:
            text, math, emph = _inline(li)
            if text:
                b.add("list_item", text, level=depth, math=math, emphasis=emph, math_kind="mathml")


def _notes(container, b):
    items = [x for x in container.iter() if local(x.tag) in ("li", "aside") and x is not container] or [container]
    for item in items:
        text, _, _ = _inline(item)
        text = re.sub(r"\s*↩\s*$", "", text).strip()
        if text:
            b.add("footnote", text)


def _table(table, b):
    cells, header_rows, occupied = [], 0, set()
    rows = [tr for tr in table.iter() if local(tr.tag) == "tr"]
    head = {id(tr) for part in table.iter() if local(part.tag) == "thead" for tr in part.iter() if local(tr.tag) == "tr"}
    for r, tr in enumerate(rows):
        if id(tr) in head:
            header_rows = r + 1
        c = 0
        for cell in tr:
            if local(cell.tag) not in ("td", "th"):
                continue
            while (r, c) in occupied:
                c += 1
            rs, cs = int(cell.get("rowspan", "1") or 1), int(cell.get("colspan", "1") or 1)
            for dr in range(rs):
                for dc in range(cs):
                    occupied.add((r + dr, c + dc))
            cells.append(Cell(_clean(_inline(cell)[0]), r, c, rs, cs, id(tr) in head or local(cell.tag) == "th"))
            c += cs
    caption = next((x for x in table if local(x.tag) == "caption"), None)
    if caption is not None and _clean(text_of(caption)):
        b.add("paragraph", _clean(text_of(caption)))
    b.add("table", cells=tuple(cells), level=header_rows)


def spine_documents(path):
    """[(href, XHTML bytes)] of the spine in reading order, as `read_epub` reads it (navigation and
    non-linear items left out): the documents the self-check renders with Writer."""
    z = SafeZip(path)
    container = parse_xml(z.read("META-INF/container.xml"))
    rootfile = next(x for x in container.iter(CONTAINER + "rootfile"))
    opf_path = rootfile.get("full-path")
    opf = parse_xml(z.read(opf_path))
    base = posixpath.dirname(opf_path)
    items = {i.get("id"): i for i in opf.iter(OPF + "item")}
    out = []
    for ref in opf.iter(OPF + "itemref"):
        item = items.get(ref.get("idref"))
        if item is None or ref.get("linear") == "no" or "nav" in (item.get("properties") or "").split():
            continue
        if item.get("media-type") != "application/xhtml+xml":
            continue
        href = posixpath.normpath(posixpath.join(base, item.get("href")))
        out.append((href, z.read(href)))
    return out


def read_epub(path):
    z = SafeZip(path)
    container = parse_xml(z.read("META-INF/container.xml"))
    rootfile = next(x for x in container.iter(CONTAINER + "rootfile"))
    opf_path = rootfile.get("full-path")
    opf = parse_xml(z.read(opf_path))
    base = posixpath.dirname(opf_path)
    items = {i.get("id"): i for i in opf.iter(OPF + "item")}
    b = Builder("epub")
    for ref in opf.iter(OPF + "itemref"):
        item = items.get(ref.get("idref"))
        if item is None or ref.get("linear") == "no" or "nav" in (item.get("properties") or "").split():
            continue
        if item.get("media-type") != "application/xhtml+xml":
            continue
        root = parse_html(z.read(posixpath.normpath(posixpath.join(base, item.get("href")))))
        body = next((x for x in root.iter() if local(x.tag) == "body"), None)
        if body is not None:
            walk(body, b)
    return b.twin
