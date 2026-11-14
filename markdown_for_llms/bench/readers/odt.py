"""ODT twin reader (`content.xml`): headings by outline level, paragraphs with emphasis from the
automatic styles, lists, tables with spans, frames, notes. The table of contents
(`text:table-of-content`) is skipped: its entries carry page numbers of the source.

Frames (calibration M4b): a frame with an embedded formula object (`draw:object` whose
`content.xml` is MathML) is a formula, inline when it is anchored as a character inside text,
display otherwise; its replacement image is not a figure. A frame with an image is a figure;
a text box holding an image is a figure whose caption is the text of the box, never glued to
the paragraph or heading that anchors it; a text box without images is text of its own. Images
inside table cells are figures too.
"""
import re

from ..model import Builder, Cell
from .mathml import mathml_linear
from .safe import SafeZip, local, parse_xml

TEXT = "urn:oasis:names:tc:opendocument:xmlns:text:1.0"
TABLE = "urn:oasis:names:tc:opendocument:xmlns:table:1.0"
STYLE = "urn:oasis:names:tc:opendocument:xmlns:style:1.0"
FO = "urn:oasis:names:tc:opendocument:xmlns:xsl-fo-compatible:1.0"
DRAW = "urn:oasis:names:tc:opendocument:xmlns:drawing:1.0"
OFFICE = "urn:oasis:names:tc:opendocument:xmlns:office:1.0"
SKIP = {"table-of-content", "alphabetical-index", "illustration-index", "bibliography", "tracked-changes",
        "sequence-decls", "variable-decls", "user-field-decls", "note-citation"}
# `text:bookmark-ref` and `text:sequence-ref` are cross-reference fields: their text is printed


def _q(ns, name):
    return "{" + ns + "}" + name


def _styles(root):
    """Automatic text styles: name -> 'b', 'i' or ''."""
    out = {}
    auto = root.find(_q(OFFICE, "automatic-styles"))
    for st in (auto if auto is not None else []):
        props = st.find(_q(STYLE, "text-properties"))
        if props is None:
            continue
        bold = props.get(_q(FO, "font-weight")) in ("bold", "700", "800", "900")
        italic = props.get(_q(FO, "font-style")) == "italic"
        out[st.get(_q(STYLE, "name"))] = "b" if bold else "i" if italic else ""
    return out


XLINK = "http://www.w3.org/1999/xlink"


class _Inline:
    def __init__(self, styles, b, objects=None):
        self.styles, self.b, self.objects = styles, b, objects or (lambda href: None)
        self.parts, self.emph, self.notes, self.math = [], [], [], []
        self.figures = []            # captions ("" without one), in order
        self.boxes = []              # paragraphs of text boxes without images
        self.display = []            # formula objects that stand alone in a frame of their own

    @property
    def images(self):
        return len(self.figures)

    def walk(self, el, style=""):
        name, ns = local(el.tag), el.tag[1:].split("}")[0] if el.tag.startswith("{") else ""
        if name in SKIP:
            return
        if ns == TEXT and name == "note":
            body = el.find(_q(TEXT, "note-body"))
            if body is not None:
                self.notes.append(re.sub(r"\s+", " ", " ".join(" ".join(x.itertext()) for x in body)).strip())
            return
        if ns == DRAW and name == "frame":
            self.frame(el, style)
            return
        if name in ("s",):
            self.parts.append(" " * int(el.get(_q(TEXT, "c"), "1")))
        elif name in ("tab", "line-break"):
            self.parts.append(" ")
        local_style = self.styles.get(el.get(_q(TEXT, "style-name")), "") if name == "span" else ""
        start = len(self.parts)
        if el.text and name not in ("s", "tab", "line-break"):
            self.parts.append(el.text)
        for child in el:
            self.walk(child, local_style or style)
            if child.tail:
                self.parts.append(child.tail)
        kind = local_style
        if kind:
            text = re.sub(r"\s+", " ", "".join(self.parts[start:])).strip()
            if text:
                self.emph.append((kind, text))

    def frame(self, el, style):
        obj = el.find(_q(DRAW, "object"))
        if obj is not None:
            formula = self.objects(obj.get(_q(XLINK, "href"), ""))
            if formula is not None:
                if formula:
                    self.math.append(formula)
                    self.parts.append(" ")
                return
        box = el.find(_q(DRAW, "text-box"))
        if box is not None:
            caption = _Inline(self.styles, self.b, self.objects)
            for child in box:
                if local(child.tag) in ("p", "h"):
                    caption.parts.append(" ")
                    caption.walk(child, style)
            if caption.figures:
                self.figures.append(caption.text())
                self.figures.extend(caption.figures[1:])
            elif caption.text():
                self.boxes.append(caption.text())
            return
        if el.find(_q(DRAW, "image")) is not None or obj is not None:
            self.figures.append("")

    def text(self):
        return re.sub(r"\s+", " ", "".join(self.parts)).strip()


def read_odt(path):
    z = SafeZip(path)
    root = parse_xml(z.read("content.xml"))
    styles = _styles(root)
    b = Builder("odt")
    body = root.find(f"{_q(OFFICE, 'body')}/{_q(OFFICE, 'text')}")

    def objects(href):
        """Linear MathML of a formula object ("" when empty), None for any other object."""
        member = href[2:] if href.startswith("./") else href
        if not member or not z.has(member + "/content.xml"):
            return None
        obj = parse_xml(z.read(member + "/content.xml"))
        if local(obj.tag) != "math":
            return None
        return mathml_linear(obj)

    def block(el, list_depth=0):
        name = local(el.tag)
        if name in SKIP:
            return
        if name in ("h", "p"):
            inline = _Inline(styles, b, objects)
            inline.walk(el)
            text = inline.text()
            math = tuple(m for m in inline.math if m)
            if text:
                if name == "h":
                    b.add("heading", text, level=int(el.get(_q(TEXT, "outline-level"), "1")))
                elif list_depth:
                    b.add("list_item", text, level=list_depth, emphasis=tuple(inline.emph), math=math, math_kind="mathml")
                else:
                    b.add("paragraph", text, emphasis=tuple(inline.emph), math=math, math_kind="mathml")
            elif math:
                for m in math:
                    b.add("formula", m, display=True, math_kind="mathml")
            for caption in inline.figures:
                b.add("figure", caption)
            for box in inline.boxes:
                b.add("paragraph", box)
            for note in inline.notes:
                if note:
                    b.add("footnote", note)
        elif name == "list":
            for item in el:
                for child in item:
                    block(child, list_depth + 1)
        elif name == "table":
            table(el)
        elif name in ("section", "index-body", "list-item", "list-header"):
            for child in el:
                block(child, list_depth)

    def table(tbl):
        cells, r, figures = [], 0, []
        rows = [x for x in tbl.iter(_q(TABLE, "table-row"))]
        header = sum(1 for _ in tbl.iter(_q(TABLE, "table-header-rows")))
        header_rows = len(list(tbl.find(_q(TABLE, "table-header-rows")))) if header else 0
        for r, tr in enumerate(rows):
            c = 0
            for tc in tr:
                if local(tc.tag) == "covered-table-cell":
                    c += 1
                    continue
                if local(tc.tag) != "table-cell":
                    continue
                parts = []
                for p in tc:
                    if local(p.tag) in ("p", "h"):
                        inline = _Inline(styles, b, objects)
                        inline.walk(p)
                        parts.append(inline.text())
                        figures.extend(inline.figures)
                cells.append(Cell(" ".join(x for x in parts if x), r, c,
                                  int(tc.get(_q(TABLE, "number-rows-spanned"), "1")),
                                  int(tc.get(_q(TABLE, "number-columns-spanned"), "1")), r < header_rows))
                c += 1
        b.add("table", cells=tuple(cells), level=header_rows)
        for caption in figures:
            b.add("figure", caption)

    for el in (body if body is not None else []):
        block(el)
    return b.twin
