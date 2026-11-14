"""PPTX twin reader: slides in presentation order; titles as headings, text paragraphs, bullets as
list items, tables, pictures. Speaker notes are excluded (declared limit: the self-check path
through LibreOffice does not print them). Slide-number fields are page furniture.
"""
import posixpath
import re

from ..model import Builder, Cell
from .safe import SafeZip, local, parse_xml

A = "{http://schemas.openxmlformats.org/drawingml/2006/main}"
P = "{http://schemas.openxmlformats.org/presentationml/2006/main}"
R = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
REL = "{http://schemas.openxmlformats.org/package/2006/relationships}"
TITLE_TYPES = {"title", "ctrTitle"}


def paragraph_text(p):
    """Text of an a:p; a:br is a space; slide-number fields are skipped."""
    parts, emph = [], []
    for child in p:
        name = local(child.tag)
        if name == "r":
            t = child.find(A + "t")
            text = t.text or "" if t is not None else ""
            parts.append(text)
            rpr = child.find(A + "rPr")
            if text.strip() and rpr is not None:
                if rpr.get("b") in ("1", "true"):
                    emph.append(("b", text.strip()))
                elif rpr.get("i") in ("1", "true"):
                    emph.append(("i", text.strip()))
        elif name == "br":
            parts.append(" ")
        elif name == "fld" and child.get("type") != "slidenum":
            t = child.find(A + "t")
            parts.append(t.text or "" if t is not None else "")
    return re.sub(r"\s+", " ", "".join(parts)).strip(), tuple(emph)


def _rels(z, part):
    folder, name = posixpath.split(part)
    rel_path = posixpath.join(folder, "_rels", name + ".rels")
    if not z.has(rel_path):
        return {}
    return {r.get("Id"): posixpath.normpath(posixpath.join(folder, r.get("Target")))
            for r in parse_xml(z.read(rel_path)).iter(REL + "Relationship")}


def read_pptx(path):
    z = SafeZip(path)
    pres = parse_xml(z.read("ppt/presentation.xml"))
    rels = _rels(z, "ppt/presentation.xml")
    b = Builder("pptx")
    slides = [rels[s.get(R + "id")] for s in pres.iter(P + "sldId")]
    for number, slide in enumerate(slides, 1):
        b.page = number
        root = parse_xml(z.read(slide))
        tree = root.find(f"{P}cSld/{P}spTree")
        if tree is not None:
            _shapes(tree, b)
    b.page = 0
    b.twin.stats["slides"] = len(slides)
    return b.twin


def _shapes(tree, b):
    for shape in tree:
        name = local(shape.tag)
        if name == "sp":
            ph = shape.find(f"{P}nvSpPr/{P}nvPr/{P}ph")
            kind = ph.get("type", "body") if ph is not None else None
            if kind in ("sldNum", "dt", "ftr"):
                continue
            body = shape.find(P + "txBody")
            if body is None:
                continue
            for p in body.findall(A + "p"):
                text, emph = paragraph_text(p)
                if not text:
                    continue
                ppr = p.find(A + "pPr")
                bullet = ppr is not None and (ppr.find(A + "buChar") is not None or ppr.find(A + "buAutoNum") is not None)
                level = int(ppr.get("lvl", "0")) if ppr is not None else 0
                if kind in TITLE_TYPES:
                    b.add("heading", text, level=1 if kind == "ctrTitle" else 2)
                elif bullet or (kind == "body" and ppr is not None and ppr.find(A + "buNone") is None and level > 0):
                    b.add("list_item", text, level=level + 1, emphasis=emph)
                else:
                    b.add("paragraph", text, emphasis=emph)
        elif name == "graphicFrame":
            tbl = shape.find(f".//{A}tbl")
            if tbl is not None:
                _table(tbl, b)
        elif name == "pic":
            b.add("figure")
        elif name == "grpSp":
            _shapes(shape, b)


def _table(tbl, b):
    props = tbl.find(A + "tblPr")
    header = 1 if props is not None and props.get("firstRow") in ("1", "true") else 0
    cells = []
    for r, tr in enumerate(tbl.findall(A + "tr")):
        for c, tc in enumerate(tr.findall(A + "tc")):
            if tc.get("hMerge") in ("1", "true") or tc.get("vMerge") in ("1", "true"):
                continue
            texts = [paragraph_text(p)[0] for p in tc.iter(A + "p")]
            cells.append(Cell(" ".join(t for t in texts if t), r, c, int(tc.get("rowSpan", "1")),
                              int(tc.get("gridSpan", "1")), r < header))
    b.add("table", cells=tuple(cells), level=header)
