"""DOCX twin reader (`zipfile` + `xml.etree`): headings by style, paragraphs with emphasis and
inline OMML, display formulas, lists by numbering level, tables with merged cells, footnotes.

Inline formulas inside table cells (generator 5, delta r3 Δ4): each non-empty one is a `formula`
with role "table-cell", added after every other element of the twin, so the positions of the
elements of generator 4 do not change. A display formula of a cell keeps its own branch.
"""
import re

from ..model import Builder, Cell
from .safe import SafeZip, local, parse_xml

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
M = "{http://schemas.openxmlformats.org/officeDocument/2006/math}"
HEADING_NAME = re.compile(r"(?i)^(heading|título|titolo|überschrift|titre)\s*(\d)$")
NARY_DEFAULT = "∫"


def omml_linear(el):
    """Linearize OMML into Unicode atoms close to the TeX atoms of `text.math_atoms`."""
    out = []

    def rec(x):
        name = local(x.tag)
        if name == "t" and x.tag == M + "t":
            out.append(x.text or "")
        elif name in ("sSup", "sSub", "sSubSup"):
            for child in x:
                cn = local(child.tag)
                if cn == "e":
                    rec(child)
            for child in x:
                cn = local(child.tag)
                if cn == "sub":
                    out.append("_")
                    rec(child)
                elif cn == "sup":
                    out.append("^")
                    rec(child)
        elif name == "rad":
            out.append("√")
            for child in x:
                if local(child.tag) in ("deg", "e"):
                    rec(child)
        elif name == "nary":
            chr_el = x.find(f"{M}naryPr/{M}chr")
            out.append(chr_el.get(M + "val") if chr_el is not None else NARY_DEFAULT)
            for child in x:
                cn = local(child.tag)
                if cn == "sub":
                    out.append("_")
                elif cn == "sup":
                    out.append("^")
                if cn in ("sub", "sup", "e"):
                    rec(child)
        elif name in ("rPr", "ctrlPr", "naryPr", "fPr", "radPr", "sSupPr", "sSubPr", "dPr", "accPr", "barPr",
                      "funcPr", "mPr", "eqArrPr", "limLowPr", "limUppPr", "groupChrPr", "borderBoxPr", "boxPr"):
            return
        elif name == "d":
            beg = x.find(f"{M}dPr/{M}begChr")
            end = x.find(f"{M}dPr/{M}endChr")
            out.append(beg.get(M + "val") if beg is not None else "(")
            for child in x:
                rec(child)
            out.append(end.get(M + "val") if end is not None else ")")
        else:
            for child in x:
                rec(child)
    rec(el)
    return "".join(out)


def _on(prop):
    return prop is not None and prop.get(W + "val") not in ("0", "false", "none")


class _Para:
    def __init__(self):
        self.parts, self.math, self.runs, self.notes, self.drawings = [], [], [], [], 0

    @property
    def has_drawing(self):
        return self.drawings > 0

    @property
    def emph(self):
        return _merge_emphasis(self.runs)


def _runs(el, para, deleted=False):
    for child in el:
        name, ns = local(child.tag), child.tag.split("}")[0] + "}"
        if ns == M and name == "oMath":
            para.math.append(omml_linear(child))
            continue
        if ns != W:
            continue
        if name in ("del", "moveFrom", "instrText", "delText", "pPr", "rPr", "proofErr", "bookmarkStart", "bookmarkEnd"):
            continue
        if name == "r":
            rpr = child.find(W + "rPr")
            bold = rpr is not None and _on(rpr.find(W + "b"))
            italic = rpr is not None and _on(rpr.find(W + "i"))
            text = []
            for x in child:
                n = local(x.tag)
                if x.tag == W + "t":
                    text.append(x.text or "")
                elif n in ("tab", "br", "cr"):
                    text.append(" ")
                elif n == "noBreakHyphen":
                    text.append("-")
                elif n == "sym":
                    text.append("")
                elif n == "footnoteReference":
                    para.notes.append(x.get(W + "id"))
                elif n in ("drawing", "pict", "object"):
                    para.drawings += 1          # one figure per drawing
                elif n == "AlternateContent":   # the drawing of the first choice; the fallback repeats it
                    choice = next((c for c in x if local(c.tag) == "Choice"), None)
                    if choice is not None and any(local(d.tag) in ("drawing", "pict", "object") for d in choice):
                        para.drawings += 1
            run = "".join(text)
            para.parts.append(run)
            para.runs.append(("b" if bold else "i" if italic else "", run))
        else:   # hyperlink, ins, smartTag, fldSimple, sdt/sdtContent, customXml...
            _runs(child, para)


def _merge_emphasis(runs):
    """Emphasis runs of a paragraph from its text runs, in order: adjacent runs of the same kind
    are one run (their texts concatenated as printed); a run of another kind or of plain text ends
    it, except a plain run of spaces between two runs of the same kind."""
    merged, open_kind, buffer = [], "", []

    def close():
        text = re.sub(r"\s+", " ", "".join(buffer)).strip()
        if open_kind and text:
            merged.append((open_kind, text))
    for kind, text in runs:
        if not text:
            continue
        if not kind and not text.strip() and open_kind:
            buffer.append(text)             # spaces between two runs of the same kind, kept tentatively
            continue
        if kind != open_kind:
            close()
            open_kind, buffer = kind, []
        buffer.append(text)
    close()
    return tuple(merged)


def read_docx(path):
    z = SafeZip(path)
    doc = parse_xml(z.read("word/document.xml"))
    styles = {}
    if z.has("word/styles.xml"):
        for st in parse_xml(z.read("word/styles.xml")).iter(W + "style"):
            name = st.find(W + "name")
            outline = st.find(f"{W}pPr/{W}outlineLvl")
            styles[st.get(W + "styleId")] = (name.get(W + "val") if name is not None else "",
                                            int(outline.get(W + "val")) if outline is not None else None)
    notes = {}
    if z.has("word/footnotes.xml"):
        for fn in parse_xml(z.read("word/footnotes.xml")).iter(W + "footnote"):
            if fn.get(W + "type") in ("separator", "continuationSeparator", "continuationNotice"):
                continue
            texts = []
            for p in fn.iter(W + "p"):
                para = _Para()
                _runs(p, para)
                texts.append("".join(para.parts))
            notes[fn.get(W + "id")] = " ".join(t.strip() for t in texts if t.strip())
    b = Builder("docx")
    pending_notes, table_figures, cell_math = [], [], []

    def heading_level(ppr):
        if ppr is None:
            return 0
        outline = ppr.find(W + "outlineLvl")
        if outline is not None and outline.get(W + "val", "9").isdigit() and int(outline.get(W + "val")) < 9:
            return int(outline.get(W + "val")) + 1
        style = ppr.find(W + "pStyle")
        if style is None:
            return 0
        name, level = styles.get(style.get(W + "val"), ("", None))
        if name.lower() == "title":
            return 1
        match = HEADING_NAME.match(name) or HEADING_NAME.match(style.get(W + "val") or "")
        if match:
            return int(match.group(2))
        return level + 1 if level is not None and level < 9 else 0

    def paragraph(p, in_table=False):
        ppr = p.find(W + "pPr")
        if p.find(f"{M}oMathPara") is not None:
            for omp in p.iter(M + "oMathPara"):
                for om in omp.iter(M + "oMath"):
                    b.add("formula", omml_linear(om), display=True, math_kind="linear")
            return None
        para = _Para()
        _runs(p, para)
        text = re.sub(r"\s+", " ", "".join(para.parts)).strip()
        pending_notes.extend(n for n in para.notes if n in notes)
        if in_table:
            table_figures.extend([""] * para.drawings)
            cell_math.extend(m for m in para.math if m.strip())
            return text
        if not text and not para.math:
            for _ in range(para.drawings):
                b.add("figure")
            return None
        if not text and len(para.math) == 1:
            b.add("formula", para.math[0], display=True, math_kind="linear")
            return None
        level = heading_level(ppr)
        if level:
            b.add("heading", text, level=level)
        elif ppr is not None and ppr.find(W + "numPr") is not None:
            ilvl = ppr.find(f"{W}numPr/{W}ilvl")
            depth = int(ilvl.get(W + "val", "0")) + 1 if ilvl is not None else 1
            b.add("list_item", text, level=depth, math=tuple(para.math), emphasis=para.emph)
        else:
            b.add("paragraph", text, math=tuple(para.math), emphasis=para.emph)
        if text:
            for _ in range(para.drawings):
                b.add("figure")
        flush_notes()
        return None

    def flush_notes():
        while pending_notes:
            nid = pending_notes.pop(0)
            if notes.get(nid):
                b.add("footnote", notes[nid])

    def table(tbl):
        cells, header_rows = [], 0
        open_vmerge = {}            # grid column -> index in cells of the cell that started the merge
        for r, tr in enumerate(tbl.findall(W + "tr")):
            if tr.find(f"{W}trPr/{W}tblHeader") is not None and r == header_rows:
                header_rows += 1
            col = 0
            for tc in tr.findall(W + "tc"):
                tcpr = tc.find(W + "tcPr")
                span = tcpr.find(W + "gridSpan") if tcpr is not None else None
                colspan = int(span.get(W + "val", "1")) if span is not None else 1
                vmerge = tcpr.find(W + "vMerge") if tcpr is not None else None
                if vmerge is not None and vmerge.get(W + "val", "continue") == "continue" and col in open_vmerge:
                    i = open_vmerge[col]
                    c = cells[i]
                    cells[i] = Cell(c.text, c.row, c.col, c.rowspan + 1, c.colspan, c.header)
                    col += colspan
                    continue
                texts = [paragraph(p, in_table=True) for p in tc.iter(W + "p")]
                text = " ".join(t for t in texts if t)
                cells.append(Cell(text, r, col, 1, colspan, r < header_rows))
                if vmerge is not None:
                    open_vmerge[col] = len(cells) - 1
                else:
                    open_vmerge.pop(col, None)
                col += colspan
        b.add("table", cells=tuple(cells), level=header_rows)
        while table_figures:
            b.add("figure", table_figures.pop(0))
        flush_notes()

    def walk(container):
        for el in container:
            name = local(el.tag)
            if el.tag == W + "p":
                paragraph(el)
            elif el.tag == W + "tbl":
                table(el)
            elif name in ("sdt", "sdtContent", "customXml"):
                walk(el)
    body = doc.find(W + "body")
    walk(body if body is not None else doc)
    flush_notes()
    for m in cell_math:
        b.add("formula", m, math_kind="linear", role="table-cell")
    return b.twin
