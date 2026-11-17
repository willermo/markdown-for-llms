"""HTML renderings of a twin read as an evaluable document and as an inventory (plan r2 §5).

One walk of the tree (`readers.htmltree`, standard library, inert: no script runs, no URL is
visited) gives both the `outdoc.OutputDoc` the checks are judged on and the inventory of units
the coverage compares with the generated checks: the inventory comes from the tree, never from
the checks. Dialects:

- `writer`: LibreOffice Writer HTML of a DOCX or ODT twin. Its header and footer, its tables of
  contents and indexes are not body text; notes are the `div#sdfootnoteN` (`sdendnoteN`) bodies,
  their markers in the text are not text; formula objects are images (`img name="Object…"`), not
  figures, nor are horizontal lines: formulas come from the StarMath channel (`starmath`). A
  paragraph that begins with an image and goes on with text is a figure with its caption (a
  Writer frame), and so is a frame (`span#FrameN` with an image) inside a heading or a paragraph,
  whose caption is not text of that heading or paragraph; for an ODT twin, the outline number
  Writer prints before a heading (`5.5Exercise`) is not text; a `pre` block is one paragraph per line (Writer joins
  consecutive preformatted paragraphs).
- `calc`: LibreOffice Calc HTML of a workbook: the "Overview" block of links to the sheets and the
  "Sheet N:" prefix of each sheet heading are not workbook content.
- `latexml`: the LaTeXML HTML of arXiv. Only the article: site chrome, navigation, the authors and
  keywords blocks and the titles LaTeXML gives to the abstract, the keywords and the bibliography
  are not text of the twin; `span.ltx_tag` (section, figure, equation and reference numbers) is
  not text; `span.ltx_note` is a footnote; `math@alttext` is the TeX of a formula, with the
  explicit spacing LaTeXML writes (`hskip 7.1pt`) dropped and its prime (`^{prime}`) written as
  the source writes it (`'`); an equation table (`ltx_eqn_table`, one per equation or group) is
  one display formula, not a table; `li.ltx_bibitem` is a reference; a figure is its images or
  embedded objects with its caption.
- `writer-epub`: Writer HTML of the XHTML documents of an EPUB spine, read in order as one
  document: a list item holding the target of a note link (and its back link) is a note, by
  (file, anchor), also across documents.

Declared limits of these renderings are in `selfcheck.DECLARED_LIMITS`.
"""
from collections import defaultdict
import hashlib
import posixpath
import re

from .codes import BenchError, Code
from .outdoc import Block, _add_block, _finish, _table_block
from .readers.htmltree import parse_html
from .readers.safe import local
from .text import align_key, math_atoms

BLOCK = {"p", "div", "section", "article", "aside", "header", "footer", "main", "blockquote", "ul", "ol", "li",
         "table", "thead", "tbody", "tfoot", "tr", "td", "th", "caption", "figure", "figcaption", "h1", "h2", "h3",
         "h4", "h5", "h6", "dl", "dt", "dd", "pre", "hr", "center", "address", "details", "summary", "body", "html",
         "html-root", "nav", "dialog", "form"}
LEAF = {"p", "pre", "dt", "dd", "address", "summary", "caption", "figcaption"}
SKIP = {"script", "style", "head", "title", "template", "noscript", "svg", "button", "dialog", "form", "nav",
        "iframe", "object", "embed", "select", "textarea", "input", "meta", "link"}
WRITER_INDEX = re.compile(r"(Table of Contents|Table of Figures|Alphabetical Index|Illustration Index|Table Index|Object Index|"
                          r"User-Defined|Bibliography|Index of)\d*$")
WRITER_NOTE = re.compile(r"sd(foot|end)note\d+$")
NOT_FIGURE = re.compile(r"(Object|Horizontal line)")
SHEET_PREFIX = re.compile(r"^\s*Sheet\s+\d+:\s*")
OUTLINE_NUMBER = re.compile(r"^\s*\d+(?:\.\d+)*\s*(?=[^\W\d_])")
LATEXML_TITLES = {"ltx_title_abstract", "ltx_title_keywords", "ltx_title_bibliography"}
LATEXML_SPACING = re.compile(r"\\(?:hskip|kern|mskip|hspace\*?)\s*\{?-?[\d.]+\s*(?:pt|em|ex|mu|bp|cm|mm|in)\}?")
LATEXML_PRIME = re.compile(r"\^\s*\{\s*((?:\\prime\s*)+)\}|\^\s*\\prime(?![A-Za-z])")


def latexml_tex(tex):
    """The TeX of a LaTeXML `alttext` as an author writes it: explicit spacing dropped, primes as `'`."""
    tex = LATEXML_SPACING.sub(" ", tex)
    return LATEXML_PRIME.sub(lambda m: "'" * max(1, (m.group(1) or "").count("prime")), tex)


def _classes(el):
    return (el.get("class") or "").split()


def _fragment(href):
    """(file, anchor) of an internal link, None for an external one."""
    if not href or re.match(r"^[a-z][a-z0-9+.-]*:", href, re.I) or "#" not in href:
        return None
    path, frag = href.split("#", 1)
    return posixpath.basename(path), frag


class _Inline:
    """Text, inline formulas, emphasis runs and counted images of inline content."""

    def __init__(self, reader):
        self.r = reader
        self.parts, self.math, self.runs, self.images = [], [], [], []
        self.frames = []                 # Writer frames with an image: (images, caption text)
        self.lead_image = None           # True when a counted image comes before any text

    def add_text(self, text):
        if text:
            self.parts.append(text)
            if self.lead_image is None and text.strip():
                self.lead_image = False

    def node(self, el):
        """The element and its subtree, not its tail (the caller owns the tail)."""
        r = self.r
        name = local(el.tag)
        if r.skipped(el) or r.is_note_body(el):
            return
        if name == "br":
            self.add_text(" ")
            return
        if name == "img":
            if r.counted_image(el):
                self.images.append(el.get("src") or "")
                if self.lead_image is None:
                    self.lead_image = True
            self.add_text(" ")
            return
        if name == "math":
            tex = el.get("alttext") or ""
            if r.dialect == "latexml":
                tex = latexml_tex(tex)
            if tex.strip():
                if el.get("display") == "block":
                    r.display_formula(tex)
                else:
                    self.math.append(tex)
            self.add_text(" ")
            return
        if r.not_text(el):
            return
        if r.note_inline(el):
            return
        if r.dialect == "writer" and name == "span" and (el.get("id") or "").startswith("Frame") \
                and any(local(x.tag) == "img" and r.counted_image(x) for x in el.iter()):
            frame = _Inline(r)
            frame.add_text(el.text)
            for child in el:
                frame.node(child)
                frame.add_text(child.tail)
            self.frames.append((frame.images, frame.text()))
            self.add_text(" ")
            return
        kind = r.emphasis_kind(el)
        start = len(self.parts)
        if name in BLOCK:
            self.add_text(" ")
        self.add_text(el.text)
        for child in el:
            self.node(child)
            self.add_text(child.tail)
        if name in BLOCK:
            self.add_text(" ")
        if kind:
            self.runs.append((kind, start, len(self.parts)))

    def text(self):
        return re.sub(r"\s+", " ", "".join(self.parts)).strip()

    def emphasis(self):
        """Runs of each kind, merged when only spaces (or nothing but tags) separate two of them:
        Writer splits one italic caption into a run per font change."""
        merged = []
        for kind in ("i", "b"):
            outer = []
            for _, a, b in sorted((r for r in self.runs if r[0] == kind), key=lambda r: (r[1], -r[2])):
                if outer and a < outer[-1][1]:
                    outer[-1] = (outer[-1][0], max(b, outer[-1][1]))      # nested or overlapping
                elif outer and not "".join(self.parts[outer[-1][1]:a]).strip():
                    outer[-1] = (outer[-1][0], max(b, outer[-1][1]))
                else:
                    outer.append((a, b))
            merged += [(kind, a, b) for a, b in outer]
        out = []
        for kind, a, b in sorted(merged, key=lambda r: r[1]):
            text = re.sub(r"\s+", " ", "".join(self.parts[a:b])).strip()
            if text:
                out.append((kind, text))
        return out


class HtmlReader:
    def __init__(self, dialect, outline=False):
        if dialect not in ("writer", "calc", "latexml", "writer-epub"):
            raise ValueError(dialect)
        self.dialect = dialect
        self.outline = outline          # an ODT twin: Writer prints the outline number of its headings
        self.blocks, self.tokens, self.token_block, self.broken = [], [], [], set()
        self.formulas, self.images = [], []
        self.units = defaultdict(list)
        self.contains_block = set()
        self.note_bodies, self.note_targets, self.note_links = set(), set(), set()
        self.file = ""
        self.pending_notes = []

    # --- classification ------------------------------------------------------------------------

    def skipped(self, el):
        name = local(el.tag)
        if name in SKIP:
            return True
        if self.dialect == "writer":
            if name == "div" and el.get("title") in ("header", "footer"):
                return True
            if name in ("div", "section") and WRITER_INDEX.match(el.get("id") or ""):
                return True
        if self.dialect == "latexml":
            cls = _classes(el)
            if {"ltx_authors", "ltx_dates", "ltx_ERROR", "ltx_page_navbar", "ltx_keywords"} & set(cls):
                return True
            if re.fullmatch(r"h[1-6]", name) and LATEXML_TITLES & set(cls):
                return True
        if self.dialect == "calc":
            # the sheet index: its "Overview" heading and its links to the sheets (`#tableN`)
            if name == "h1" and "".join(el.itertext()).strip() == "Overview":
                return True
            if name == "a" and re.fullmatch(r"#table\d+", el.get("href") or ""):
                return True
        return False

    def not_text(self, el):
        """Inline elements whose text is not text of the document."""
        cls = _classes(el)
        if self.dialect == "latexml" and any(c == "ltx_tag" or c.startswith("ltx_tag_") for c in cls):
            return True
        if self.dialect == "writer" and local(el.tag) == "a" and ({"sdfootnotesym", "sdendnotesym",
                                                                   "sdfootnoteanc", "sdendnoteanc"} & set(cls)):
            return True
        if self.dialect == "writer-epub" and local(el.tag) == "a":
            target = _fragment(el.get("href") or "")
            if target and ((target[0] or self.file, target[1]) in self.note_targets
                           or (target[0] or self.file, target[1]) in self.note_links):
                return True                   # a note marker, or the back link of a note
        return False

    def note_inline(self, el):
        """A LaTeXML note inside a paragraph: a footnote of its own, not text of the paragraph."""
        if self.dialect != "latexml" or "ltx_note" not in _classes(el):
            return False
        content = next((x for x in el.iter() if "ltx_note_content" in _classes(x)), None)
        if content is not None:
            inline = _Inline(self)
            for child in content:
                if "ltx_note_mark" in _classes(child):
                    inline.add_text(child.tail)
                    continue
                inline.node(child)
                inline.add_text(child.tail)
            self.pending_notes.append(inline)
        return True

    def emphasis_kind(self, el):
        name, cls = local(el.tag), _classes(el)
        if name in ("i", "em", "cite") or "ltx_font_italic" in cls or "ltx_emph" in cls:
            return "i"
        if name in ("b", "strong") or "ltx_font_bold" in cls:
            return "b"
        return ""

    def counted_image(self, el):
        return not (self.dialect == "writer" and NOT_FIGURE.match(el.get("name") or ""))

    def is_note_body(self, el):
        return id(el) in self.note_bodies

    # --- the walk ------------------------------------------------------------------------------

    def read(self, root, file=""):
        """Walk one tree (after `prepare`)."""
        self.file = file
        start = root
        if self.dialect == "latexml":
            start = next((x for x in root.iter() if local(x.tag) == "article" and "ltx_document" in _classes(x)), root)
        self.container(start, "body", 0)

    def prepare(self, root):
        parent = {c: p for p in root.iter() for c in p}
        for el in root.iter():
            if local(el.tag) in BLOCK:
                x = parent.get(el)
                while x is not None and id(x) not in self.contains_block:
                    self.contains_block.add(id(x))
                    x = parent.get(x)
        if self.dialect == "writer":
            for el in root.iter():
                if local(el.tag) == "div" and WRITER_NOTE.match(el.get("id") or ""):
                    self.note_bodies.add(id(el))
        return parent

    def is_block(self, el):
        return local(el.tag) in BLOCK or id(el) in self.contains_block

    def container(self, el, ctx, depth):
        pending = []

        def flush():
            if pending:
                self.leaf(pending, ctx, depth)
                pending.clear()
        if el.text and el.text.strip():
            pending.append(el.text)
        for child in el:
            if self.skipped(child):
                pass
            elif self.is_note_body(child):
                flush()
                self.note(child)
            elif self.is_block(child):
                flush()
                self.block(child, ctx, depth)
            else:
                pending.append(child)
            if child.tail and child.tail.strip():
                pending.append(child.tail)
        flush()

    def block(self, el, ctx, depth):
        name = local(el.tag)
        cls = _classes(el)
        if self.dialect == "latexml" and name == "li" and "ltx_bibitem" in cls:
            self.bib_entry(el)
        elif self.dialect == "latexml" and name == "table" and "ltx_eqn_table" in cls:
            tex = " ".join(latexml_tex(m.get("alttext")) for m in el.iter()
                           if local(m.tag) == "math" and (m.get("alttext") or "").strip())
            if tex.strip():
                self.display_formula(tex)
        elif self.dialect == "latexml" and name == "figure":
            self.figure(el, ctx, depth)
        elif re.fullmatch(r"h[1-6]", name):
            self.heading(el, int(name[1]), ctx, depth)
        elif name == "pre" and self.dialect == "writer":
            # a new line separates two paragraphs of the twin; a line break inside one is a <br/>
            for line in "".join(self._inline_of([el]).parts).split("\n"):
                if line.strip():
                    self.leaf([line], ctx, depth)
        elif name in LEAF:
            self.leaf([el], ctx, depth)
        elif name in ("ul", "ol"):
            for item in el:
                if self.is_note_body(item):
                    self.note(item)
                elif self.dialect == "latexml" and local(item.tag) == "li" and "ltx_bibitem" in _classes(item):
                    self.bib_entry(item)
                elif local(item.tag) == "li" and not self.skipped(item):
                    self.container(item, "list", depth + 1)
                elif not self.skipped(item):
                    self.block(item, ctx, depth)
        elif name == "table":
            self.table(el)
        elif name == "hr":
            return
        else:
            self.container(el, ctx, depth)

    def _inline_of(self, items):
        inline = _Inline(self)
        for item in items:
            if isinstance(item, str):
                inline.add_text(item)
            elif local(item.tag) in LEAF or self.is_block(item):
                inline.add_text(item.text)
                for child in item:
                    inline.node(child)
                    inline.add_text(child.tail)
            else:
                inline.node(item)
        return inline

    def leaf(self, items, ctx, depth):
        inline = self._inline_of(items)
        text = inline.text()
        self._block_from(inline, text, ctx, depth)

    def _block_from(self, inline, text, ctx, depth, kind=None):
        math = inline.math
        if kind is None:
            kind = "list_item" if ctx == "list" else "paragraph"
        if text or math:
            self._emit(Block(kind, depth if kind == "list_item" else 0, math=list(math), emph=inline.emphasis()), text)
            self.formulas.extend((False, m) for m in math)
            for m in math:
                self._unit("inline_math", " ".join(math_atoms(m)), raw=True)
        captioned = self.dialect == "writer" and inline.lead_image and text and kind == "paragraph"
        for k, src in enumerate(inline.images):
            self.images.append((src, True))
            self._unit("figure", text if captioned and k == 0 else "", empty_ok=True)
        if text and not captioned:
            self._unit(kind, text)
        self._frames(inline)
        self._flush_notes()

    def _emit(self, block, text):
        _add_block(self.blocks, self.tokens, self.token_block, self.broken, block, text)

    def _unit(self, kind, text, raw=False, empty_ok=False):
        key = text if raw else align_key(text)
        if key or empty_ok:
            self.units[kind].append(key)

    def _flush_notes(self):
        notes, self.pending_notes = self.pending_notes, []
        for inline in notes:
            text = inline.text()
            if text:
                self._emit(Block("footnote", math=list(inline.math)), text)
                self.formulas.extend((False, m) for m in inline.math)
                self._unit("footnote", text)

    def heading(self, el, level, ctx, depth):
        inline = self._inline_of([el])
        text = inline.text()
        if self.dialect == "calc":
            text = SHEET_PREFIX.sub("", text)
        if self.dialect == "writer" and self.outline:
            text = OUTLINE_NUMBER.sub("", text).strip()
        if ctx not in ("body", "list"):         # a numbered heading sits in a list item: still a heading
            self._block_from(inline, text, ctx, depth)
            return
        if text:
            self._emit(Block("heading", level, emph=inline.emphasis()), text)
            self._unit("heading", text)
        for src in inline.images:
            self.images.append((src, True))
            self._unit("figure", "", empty_ok=True)
        self._frames(inline)
        self._flush_notes()

    def _frames(self, inline):
        """The frames of a heading or a paragraph: figures with their captions, after it."""
        for images, caption in inline.frames:
            for k, src in enumerate(images):
                self.images.append((src, True))
                self._unit("figure", caption if k == 0 else "", empty_ok=True)
            if caption:
                self._emit(Block("paragraph"), caption)

    def display_formula(self, tex):
        self.formulas.append((True, tex))
        self.blocks.append(Block("math", start=len(self.tokens), end=len(self.tokens), math=[tex]))
        self._unit("formula", " ".join(math_atoms(tex)), raw=True)

    def note(self, el):
        """One note: all its paragraphs as one footnote block and one unit."""
        inline = _Inline(self)
        inline.add_text(el.text)
        for child in el:
            inline.add_text(" ")
            if local(child.tag) in BLOCK or self.is_block(child):
                inline.add_text(child.text)
                for k in child:
                    inline.node(k)
                    inline.add_text(k.tail)
            else:
                inline.node(child)
            inline.add_text(child.tail)
        text = inline.text()
        if text:
            self._emit(Block("footnote", math=list(inline.math), emph=inline.emphasis()), text)
            self.formulas.extend((False, m) for m in inline.math)
            self._unit("footnote", text)
        for src in inline.images:
            self.images.append((src, True))

    def bib_entry(self, el):
        inline = self._inline_of([el])
        text = inline.text()
        if text:
            self._emit(Block("paragraph", emph=inline.emphasis()), text)
            self._unit("bib_entry", text)

    def figure(self, el, ctx, depth):
        """A LaTeXML float: a figure (its images, one unit with the caption), or another float (a
        table) walked as blocks; the caption follows as a paragraph."""
        caption = next((x for x in el if local(x.tag) == "figcaption"), None)
        text = self._inline_of([caption]).text() if caption is not None else ""
        if "ltx_figure" in _classes(el):
            images = [x for x in el.iter() if local(x.tag) in ("img", "object")]
            for k, img in enumerate(images):
                self.images.append((img.get("src") or img.get("data") or "", True))
                self._unit("figure", text if k == 0 else "", empty_ok=True)
            if text:
                self._emit(Block("paragraph"), text)
                if not images:
                    self._unit("paragraph", text)
            return
        for child in el:
            if child is caption or self.skipped(child):
                continue
            if self.is_block(child):
                self.block(child, ctx, depth)
            else:
                self.leaf([child], ctx, depth)
        if caption is not None:
            self.leaf([caption], ctx, depth)

    def table(self, table):
        rows, header_rows, occupied = [], 0, {}
        trs = []

        def collect(x):
            for child in x:
                n = local(child.tag)
                if n == "tr":
                    trs.append((child, False))
                elif n in ("thead", "tbody", "tfoot"):
                    for tr in child:
                        if local(tr.tag) == "tr":
                            trs.append((tr, n == "thead"))
        collect(table)
        cell_units = []
        for r, (tr, in_head) in enumerate(trs):
            cells, c = [], 0
            tds = [td for td in tr if local(td.tag) in ("td", "th")]
            for td in tds:
                while occupied.get((r, c)):
                    cells.append("")
                    c += 1
                inline = self._inline_of([td])
                text = inline.text()
                for src in inline.images:
                    self.images.append((src, True))
                    self._unit("figure", "", empty_ok=True)
                for m in inline.math:
                    self.formulas.append((False, m))
                    self._unit("formula", " ".join(math_atoms(m)), raw=True)
                if text:
                    cell_units.append(text)
                span_c = int(td.get("colspan")) if (td.get("colspan") or "").isdigit() and int(td.get("colspan")) > 0 else 1
                span_r = int(td.get("rowspan")) if (td.get("rowspan") or "").isdigit() and int(td.get("rowspan")) > 0 else 1
                for dr in range(1, span_r):
                    for dc in range(span_c):
                        occupied[(r + dr, c + dc)] = True
                cells += [text] + [""] * (span_c - 1)
                c += span_c
            while occupied.get((r, c)):
                cells.append("")
                c += 1
            if (in_head or (tds and all(local(td.tag) == "th" for td in tds))) and len(rows) == header_rows:
                header_rows += 1
            rows.append(cells)
        if any(any(x for x in row) for row in rows):
            _table_block(self.blocks, self.tokens, self.token_block, self.broken, rows, header_rows)
            self.units["table"].append("table")
            for text in cell_units:
                self._unit("cell", text)
        self._flush_notes()

    def result(self):
        return _finish(self.blocks, self.tokens, self.token_block, self.broken, False, self.formulas, self.images), \
            {k: list(v) for k, v in self.units.items() if v}


def read_html(data, dialect, outline=False):
    """(OutputDoc, inventory {type: [keys]}) of one HTML rendering; `outline`: the Writer HTML of an
    ODT twin, whose headings carry the outline number as text."""
    reader = HtmlReader(dialect, outline)
    root = parse_html(data)
    reader.prepare(root)
    reader.read(root)
    return reader.result()


def read_html_files(files, dialect="writer-epub"):
    """(OutputDoc, inventory) of several HTML documents read in order as one (an EPUB spine through
    Writer). `files` is a list of (original file name, HTML bytes); a note is found by (file, anchor)."""
    reader = HtmlReader(dialect)
    roots = []
    for name, data in files:
        root = parse_html(data)
        reader.prepare(root)
        roots.append((posixpath.basename(name), root))
    anchors = {}                                       # (file, anchor) -> the anchor element
    links = []                                         # (file, link element, target)
    for name, root in roots:
        for el in root.iter():
            if local(el.tag) == "a":
                for key in ("name", "id"):
                    if el.get(key):
                        anchors[(name, el.get(key))] = el
                target = _fragment(el.get("href") or "")
                if target:
                    links.append((name, el, (target[0] or name, target[1])))
    for name, root in roots:
        for li in (x for x in root.iter() if local(x.tag) in ("li", "aside")):
            own = [(name, a.get("name") or a.get("id")) for a in li.iter() if local(a.tag) == "a" and (a.get("name") or a.get("id"))]
            targeted = [k for k in own if any(t == k for _, _, t in links)]
            backs = [(f, t) for f, a, t in links if f == name and any(a is x for x in li.iter())]
            if targeted and backs:
                reader.note_bodies.add(id(li))
                reader.note_targets.update(targeted)
                reader.note_links.update(t for _, t in backs)
    for name, root in roots:
        reader.read(root, name)
    return reader.result()


def strip_data_uris(src, dst, chunk=1 << 20):
    """Copy an HTML file replacing each `data:` URI of an attribute or of a CSS `url(...)` (the byte
    before it is a quote or an opening parenthesis) with `data:omitted;sha256=<16 hex>;bytes=<n>`,
    block by block (Writer writes the backgrounds of D26 as 353 MB of URIs). A URI whose delimiter
    does not come before a `<`, a `>` or the end of the file is `twin_render_invalid`: the copy is
    removed, never left truncated."""
    closing = {ord('"'): b'"', ord("'"): b"'", ord("("): b")"}
    try:
        with open(src, "rb") as fin, open(dst, "wb") as fout:
            _strip(fin, fout, chunk, closing)
    except BaseException:
        import os
        if os.path.exists(dst):
            os.unlink(dst)
        raise


def _strip(fin, fout, chunk, closing):
    carry, prev, delim, digest, size = b"", None, None, None, 0
    while True:
        block = fin.read(chunk)
        data, carry, pos = carry + block, b"", 0
        while pos < len(data):
            if delim is not None:                       # inside a URI
                stops = [i for i in (data.find(delim, pos), data.find(b"<", pos), data.find(b">", pos)) if i >= 0]
                end = min(stops) if stops else -1
                if end < 0:
                    digest.update(data[pos:])
                    size += len(data) - pos
                    pos = len(data)
                    break
                if data[end:end + 1] != delim:
                    raise BenchError(Code.TWIN_RENDER_INVALID, "data URI not closed before a tag")
                digest.update(data[pos:end])
                size += end - pos
                fout.write(b"data:omitted;sha256=" + digest.hexdigest()[:16].encode() + b";bytes=" + str(size).encode())
                delim, prev, pos = None, None, end
                continue
            i = data.find(b"data:", pos)
            if i < 0:
                cut = max(pos, len(data) - 4) if block else len(data)     # "data:" may straddle two blocks
                fout.write(data[pos:cut])
                if cut > pos:
                    prev = data[cut - 1]
                carry, pos = data[cut:], len(data)
                break
            before = data[i - 1] if i > 0 else prev
            fout.write(data[pos:i])
            pos, prev = i + 5, ord(":")
            if before in closing:                       # the URI value of an attribute or of url(...)
                delim, digest, size = closing[before], hashlib.sha256(), 0
            else:
                fout.write(b"data:")                    # "data:" in running text
        if not block:
            break
    if delim is not None:
        raise BenchError(Code.TWIN_RENDER_INVALID, "data URI not closed at the end of the file")
