"""JATS twin reader: title, abstract, sections, paragraphs with inline formulas and emphasis,
lists, display formulas (MathML or TeX), tables with spans, figures, footnotes and references.
Front matter other than title and abstract (authors, affiliations, metadata) is not checked.
"""
from pathlib import Path
import re

from ..model import Builder, Cell
from .mathml import mathml_linear
from .safe import local, parse_xml, text_of

SKIP_INLINE = {"inline-formula", "disp-formula", "fn", "table-wrap", "fig", "list", "def-list", "boxed-text",
               "alternatives", "tex-math", "math"}


def _clean(text):
    return re.sub(r"\s+", " ", text).strip()


def _inline(p):
    """(text, inline formulas, emphasis) of a paragraph-like element."""
    math, emph = [], []
    for f in p.iter():
        if local(f.tag) == "inline-formula":
            m = next((x for x in f.iter() if local(x.tag) == "math"), None)
            tex = next((x for x in f.iter() if local(x.tag) == "tex-math"), None)
            if m is not None:
                math.append(("mathml", mathml_linear(m)))
            elif tex is not None:
                math.append(("tex", text_of(tex)))
        elif local(f.tag) in ("italic", "bold"):
            t = _clean(text_of(f))
            if t:
                emph.append(("i" if local(f.tag) == "italic" else "b", t))
    return _clean(text_of(p, SKIP_INLINE)), tuple(m for _, m in math), tuple(emph), (math[0][0] if math else "tex")


def _citation(ref):
    mixed = next((x for x in ref.iter() if local(x.tag) == "mixed-citation"), None)
    if mixed is not None:   # PubMed ids and DOIs added by PMC are not printed by the journal
        return _clean(text_of(mixed, {"pub-id", "object-id"}))
    element = next((x for x in ref.iter() if local(x.tag) == "element-citation"), None)
    if element is None:
        return ""
    parts = []
    for child in element:
        name = local(child.tag)
        if name == "person-group":
            parts.append(", ".join(_clean(" ".join(x.itertext())) for x in child if local(x.tag) in ("name", "collab", "etal")))
        elif name not in ("pub-id", "comment", "ext-link", "uri", "date-in-citation"):
            parts.append(_clean(text_of(child)))
    return _clean(" ".join(p for p in parts if p))


def read_jats(path):
    root = parse_xml(Path(path).read_bytes())
    b = Builder("jats")
    title = next((x for x in root.iter() if local(x.tag) == "article-title" and _in(root, x, "title-group")), None)
    if title is not None:
        b.add("heading", _clean(text_of(title)), level=1)
    for abstract in (x for x in root.iter() if local(x.tag) == "abstract"):
        if abstract.get("abstract-type") in ("graphical", "teaser"):
            continue
        b.add("heading", "Abstract", level=2)
        _blocks(abstract, b, 2)
    body = next((x for x in root if local(x.tag) == "body"), None)
    if body is not None:
        _blocks(body, b, 1)
    back = next((x for x in root if local(x.tag) == "back"), None)
    if back is not None:
        _blocks(back, b, 1)
    return b.twin


def _in(root, el, ancestor_name):
    parents = {c: p for p in root.iter() for c in p}
    while el in parents:
        el = parents[el]
        if local(el.tag) == ancestor_name:
            return True
    return False


def _blocks(el, b, depth):
    for child in el:
        name = local(child.tag)
        if name in ("sec", "ack", "app", "notes", "glossary", "ref-list", "fn-group", "app-group", "boxed-text"):
            title = next((x for x in child if local(x.tag) == "title"), None)
            if title is not None and _clean(text_of(title)):
                b.add("heading", _clean(text_of(title)), level=depth + 1)
            if name == "ref-list":
                for ref in (x for x in child.iter() if local(x.tag) == "ref"):
                    text = _citation(ref)
                    if text:
                        b.add("bib_entry", text)
            elif name == "fn-group":
                for fn in (x for x in child if local(x.tag) == "fn"):
                    _footnote(fn, b)
            else:
                _blocks(child, b, depth + 1)
        elif name == "p":
            text, math, emph, kind = _inline(child)
            if text or math:
                b.add("paragraph", text, math=math, emphasis=emph, math_kind=kind)
            for nested in child:
                if local(nested.tag) in ("disp-formula", "table-wrap", "fig", "list", "def-list"):
                    _blocks_one(nested, b, depth)
        elif name in ("disp-formula", "table-wrap", "fig", "list", "def-list", "fn"):
            _blocks_one(child, b, depth)
        elif name in ("title", "label", "sec-meta", "object-id"):
            continue
        else:
            _blocks(child, b, depth)


def _blocks_one(el, b, depth, list_depth=0):
    name = local(el.tag)
    if name == "disp-formula":
        m = next((x for x in el.iter() if local(x.tag) == "math"), None)
        tex = next((x for x in el.iter() if local(x.tag) == "tex-math"), None)
        if m is not None:
            b.add("formula", mathml_linear(m), display=True, math_kind="mathml")
        elif tex is not None:
            b.add("formula", text_of(tex), display=True, math_kind="tex")
    elif name == "table-wrap":
        _table(el, b)
    elif name == "fig":
        caption = next((x for x in el if local(x.tag) == "caption"), None)
        label = next((x for x in el if local(x.tag) == "label"), None)
        text = _clean((text_of(label) + " " if label is not None else "") + (text_of(caption) if caption is not None else ""))
        b.add("figure", text)
    elif name in ("list", "def-list"):
        for item in el:
            if local(item.tag) not in ("list-item", "def-item"):
                continue
            parts = [text_of(x, SKIP_INLINE) for x in item if local(x.tag) in ("label", "p", "term", "def")]
            text = _clean(" ".join(parts))
            if text:   # a glossary (def-list) prints as lines of text, not as a list
                b.add("list_item" if name == "list" else "paragraph", text, level=list_depth + 1 if name == "list" else 0)
            for nested in item.iter():
                if nested is not item and local(nested.tag) == "list":
                    _blocks_one(nested, b, depth, list_depth + 1)
    elif name == "fn":
        _footnote(el, b)


def _footnote(fn, b):
    text = _clean(" ".join(text_of(p) for p in fn if local(p.tag) == "p"))
    if text:
        b.add("footnote", text)


def _table(wrap, b):
    caption = next((x for x in wrap if local(x.tag) == "caption"), None)
    label = next((x for x in wrap if local(x.tag) == "label"), None)
    if caption is not None or label is not None:
        b.add("paragraph", _clean((text_of(label) + " " if label is not None else "") +
                                  (text_of(caption) if caption is not None else "")), role="table-caption")
    table = next((x for x in wrap.iter() if local(x.tag) == "table"), None)
    if table is not None:
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
                cells.append(Cell(_clean(text_of(cell, {"fn"})), r, c, rs, cs, id(tr) in head or local(cell.tag) == "th"))
                c += cs
        b.add("table", cells=tuple(cells), level=header_rows)
    foot = next((x for x in wrap if local(x.tag) == "table-wrap-foot"), None)
    if foot is not None:
        for fn in foot.iter():
            if local(fn.tag) == "fn":
                _footnote(fn, b)
        for p in foot:
            if local(p.tag) == "p" and _clean(text_of(p)):
                b.add("footnote", _clean(text_of(p)))
