"""The converted output as an evaluable document.

Markdown is parsed with `markdown-it-py` (CommonMark, GFM tables, dollar math, footnotes) and the
HTML tables inside it with the lenient tree of `readers.htmltree`; plain text (the output of a
third-party tool in the self-check) is read line by line. Declared normalizations: Marker page
separators removed, words broken at a line end joined (and recorded for the form checks),
`text.score_tokens` for the tokens. Math never enters the text tokens.

Images (C R6.2 of plan r2): a Markdown image, also in the cell of a table, and every `img` with a
`src` in an HTML block (`<figure>` with its caption, the `<table>` of Calc, any other) or in an
inline HTML tag, in the order of the document. With `assets_dir` (the `out/` of an attempt) each one
is checked: a local file under it, never a URL, an absolute path or `..`.
"""
from dataclasses import dataclass, field
import os
import re

from .text import dehyphenate, score_tokens

PAGE_SEPARATOR = re.compile(r"^\{\d+\}-{10,}\s*$", re.M)
BROKEN = re.compile(r"(\w+)-\n([a-zà-öø-ÿ]\w*)")
HTML_TAG = re.compile(r"</?[A-Za-z][A-Za-z0-9]*(\s[^<>]*)?/?>")


@dataclass
class Block:
    kind: str                 # heading, paragraph, list_item, table, math, footnote, code, line
    level: int = 0
    text: str = ""
    start: int = 0            # first token index in the document stream
    end: int = 0
    math: list = field(default_factory=list)
    rows: list = field(default_factory=list)      # tables: rows of cell texts
    header_rows: int = 0
    emph: list = field(default_factory=list)      # (kind, text)
    html: bool = False
    images: list = field(default_factory=list)


@dataclass
class OutputDoc:
    blocks: list
    tokens: list
    token_block: list
    broken: set               # token indexes that were joined from a word broken at a line end
    plain: bool               # plain text (no Markdown structure): structure and form not applicable
    formulas: list            # (display, tex)
    images: list              # (src, exists)
    index: dict = field(default_factory=dict)

    def positions(self, token):
        return self.index.get(token, ())


def _add_block(doc_blocks, tokens, token_block, broken, block, text):
    words, joined = _tokens_with_breaks(text)
    block.start = len(tokens)
    for k, w in enumerate(words):
        if k in joined:
            broken.add(len(tokens))
        tokens.append(w)
        token_block.append(len(doc_blocks))
    block.end = len(tokens)
    block.text = re.sub(r"\s+", " ", text).strip()
    doc_blocks.append(block)


def _tokens_with_breaks(text):
    """Tokens of `text` after de-hyphenation, and the indexes of the joined words."""
    joined_words = {(a + b).lower() for a, b in BROKEN.findall(text)}
    words = score_tokens(dehyphenate(text))
    joined = {k for k, w in enumerate(words) if w in joined_words}
    return words, joined


def _finish(blocks, tokens, token_block, broken, plain, formulas, images):
    doc = OutputDoc(blocks, tokens, token_block, broken, plain, formulas, images)
    for i, t in enumerate(tokens):
        doc.index.setdefault(t, []).append(i)
    return doc


def from_text(text):
    """Plain text of a third-party tool (pdftotext, Ghostscript): one block per non-empty line."""
    blocks, tokens, token_block, broken = [], [], [], set()
    # a line ending with a hyphen before a line starting with a lower-case letter: one word
    text = re.sub(r"(\w)-\n\s*(?=[a-zà-öø-ÿ])", r"\1", text)
    for line in text.split("\n"):
        if line.strip():
            _add_block(blocks, tokens, token_block, broken, Block("line"), line)
    return _finish(blocks, tokens, token_block, broken, True, [], [])


def from_markdown(markdown, assets_dir=None):
    from markdown_it import MarkdownIt
    from mdit_py_plugins.dollarmath import dollarmath_plugin
    from mdit_py_plugins.footnote import footnote_plugin

    markdown = PAGE_SEPARATOR.sub("", markdown)
    md = MarkdownIt("commonmark", {"html": True}).enable("table").use(dollarmath_plugin, double_inline=True).use(footnote_plugin)
    # Display math is read by the inline `$$...$$` rule only: the block rule opens on any line that
    # starts with `$$` and swallows the following paragraphs up to a later `$$` line end.
    md.block.ruler.disable([r for r in md.block.ruler.get_all_rules() if "math" in r])
    stream = md.parse(markdown)
    blocks, tokens, token_block, broken, formulas, images = [], [], [], set(), [], []
    list_depth, in_footnote, i = 0, 0, 0
    pending_kind = None
    while i < len(stream):
        tok = stream[i]
        t = tok.type
        if t in ("bullet_list_open", "ordered_list_open"):
            list_depth += 1
        elif t in ("bullet_list_close", "ordered_list_close"):
            list_depth -= 1
        elif t == "footnote_open":
            in_footnote += 1
        elif t == "footnote_close":
            in_footnote -= 1
        elif t == "heading_open":
            pending_kind = ("heading", int(tok.tag[1]))
        elif t == "paragraph_open":
            pending_kind = ("footnote", 0) if in_footnote else ("list_item", list_depth) if list_depth else ("paragraph", 0)
        elif t == "inline" and pending_kind:
            text, math, emph, html, imgs = _inline(tok.children or [])
            block = Block(pending_kind[0], pending_kind[1], math=math, emph=emph, html=html, images=imgs)
            formulas.extend((False, m) for m in math)
            images.extend(imgs)
            _add_block(blocks, tokens, token_block, broken, block, text)
            pending_kind = None
        elif t in ("math_block", "math_block_label"):
            formulas.append((True, tok.content))
            blocks.append(Block("math", start=len(tokens), end=len(tokens), math=[tok.content]))
        elif t == "table_open":
            j, rows, header_rows, row, cell_text = i + 1, [], 0, None, None
            while j < len(stream) and stream[j].type != "table_close":
                s = stream[j]
                if s.type == "tr_open":
                    row = []
                elif s.type == "tr_close":
                    rows.append(row)
                elif s.type == "thead_close":
                    header_rows = len(rows)
                elif s.type == "inline" and row is not None:
                    text, math, emph, html, imgs = _inline(s.children or [])
                    row.append(text)
                    formulas.extend((False, m) for m in math)
                    images.extend(imgs)
                j += 1
            _table_block(blocks, tokens, token_block, broken, rows, header_rows)
            i = j
        elif t == "html_block":
            _html_block(tok.content, blocks, tokens, token_block, broken, images)
        elif t in ("fence", "code_block"):
            _add_block(blocks, tokens, token_block, broken, Block("code"), tok.content)
        i += 1
    if assets_dir is not None:
        images = [(src, _asset_exists(assets_dir, src)) for src, _ in images]
    return _finish(blocks, tokens, token_block, broken, False, formulas, images)


def _asset_exists(assets_dir, src):
    if re.match(r"^[a-z]+:", src) or src.startswith("/") or ".." in src.split("/"):
        return False
    return os.path.isfile(os.path.join(assets_dir, src))


def _table_block(blocks, tokens, token_block, broken, rows, header_rows):
    block = Block("table", rows=rows, header_rows=header_rows)
    _add_block(blocks, tokens, token_block, broken, block, "\n".join(" | ".join(r) for r in rows))


def _img_sources(root):
    """(src, True) of every `img` with a `src` under an HTML tree, in document order."""
    from .readers.safe import local
    return [(el.get("src"), True) for el in root.iter() if local(el.tag) == "img" and el.get("src")]


def _html_block(content, blocks, tokens, token_block, broken, images):
    from .readers.htmltree import parse_html
    from .readers.safe import local
    root = parse_html(content.encode("utf-8"))
    images.extend(_img_sources(root))
    tables = [x for x in root.iter() if local(x.tag) == "table"]
    if tables:
        for table in tables:
            rows, header_rows, occupied = [], 0, {}
            for r, tr in enumerate(x for x in table.iter() if local(x.tag) == "tr"):
                cells, c = [], 0
                tds = [td for td in tr if local(td.tag) in ("td", "th")]
                for td in tds:
                    while occupied.get((r, c)):
                        cells.append("")
                        c += 1
                    text = re.sub(r"\s+", " ", "".join(td.itertext())).strip()
                    span_c = int(td.get("colspan", "1") or 1) if (td.get("colspan") or "1").isdigit() else 1
                    span_r = int(td.get("rowspan", "1") or 1) if (td.get("rowspan") or "1").isdigit() else 1
                    for dr in range(1, span_r):
                        for dc in range(span_c):
                            occupied[(r + dr, c + dc)] = True
                    cells += [text] + [""] * (span_c - 1)   # the grid of the table, spans included
                    c += span_c
                while occupied.get((r, c)):
                    cells.append("")
                    c += 1
                if tds and all(local(td.tag) == "th" for td in tds) and len(rows) == header_rows:
                    header_rows += 1
                rows.append(cells)
            _table_block(blocks, tokens, token_block, broken, rows, header_rows)
        return
    text = re.sub(r"\s+", " ", "".join(root.itertext())).strip()
    if text:
        _add_block(blocks, tokens, token_block, broken, Block("paragraph", html=True), text)


def _inline(children):
    """(text, math, emphasis runs, html residue, images) of an inline token list."""
    parts, math, emph, html, images, stack = [], [], [], False, [], []
    for c in children:
        t = c.type
        if t in ("text", "code_inline"):
            parts.append(c.content)
        elif t == "softbreak":
            parts.append("\n")
        elif t == "hardbreak":
            parts.append(" ")
        elif t in ("math_inline", "math_inline_double"):
            math.append(c.content)
            parts.append(" ")
        elif t in ("em_open", "strong_open"):
            stack.append(("i" if t == "em_open" else "b", len(parts)))
        elif t in ("em_close", "strong_close") and stack:
            kind, start = stack.pop()
            run = re.sub(r"\s+", " ", "".join(parts[start:])).strip()
            if run:
                emph.append((kind, run))
        elif t == "image":
            images.append((c.attrs.get("src", ""), True))
            parts.append(" " + c.content + " ")
        elif t == "html_inline":
            # <sup> and <sub> are the usual form of raised and lowered text where Markdown has none
            if not re.match(r"</?(sup|sub)\b", c.content, re.I):
                html = True
            if re.match(r"<br\s*/?>", c.content, re.I):
                parts.append(" ")
            elif re.match(r"<img\b", c.content, re.I):
                from .readers.htmltree import parse_html
                images.extend(_img_sources(parse_html(c.content.encode("utf-8"))))
        elif t == "footnote_ref":
            parts.append(" ")
    return "".join(parts), math, emph, html, images
