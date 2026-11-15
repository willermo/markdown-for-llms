"""Self-check: the twin rendered by a third-party tool must pass its own checks (plan r2 R2-B).

No tool here uses Marker or the libraries Marker uses. Renderings: Pandoc (LaTeX, JATS, DOCX,
ODT, EPUB, MediaWiki HTML of the same revisions, LibreOffice HTML of XLSX) written as GitHub
Markdown; plain text of poppler (`pdftotext -raw`: PPTX through LibreOffice PDF, Akoma Ntoso
against the PDF of D19) and of Ghostscript `txtwrite` (D06). Title and abstract that Pandoc keeps
in the metadata are promoted to the body, as a converter of the PDF would print them.

Coverage (B1, C4): every unit of an independent inventory (Pandoc AST, the flat ODP of
LibreOffice, the HTML export of Normattiva, the selection of D06) must have a generated check of
the same type with the same alignment key (or, failing that, 90 % of its tokens); the omission
proof drops one unit in ten, at least one per type, and must turn coverage low, except where it is
declared not determinable for a twin (`OMISSION_NOT_DETERMINABLE`, counted in the report).

Alignment of the inventory (calibration M4b, coverage only, never in the scores): a display
formula inside a Pandoc paragraph splits it, as it is a block in print; the alternative text of an
image is not printed text; an inventory unit of at least eight tokens that a tool split out of a
longer paragraph (MediaWiki around a `<ref>`, Pandoc around a JATS `fig`) is covered when 90 % of
its tokens fall in one contiguous window of a generated unit, the window test of the matcher.
"""
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import shutil

from ..engine import tools
from .codes import BenchError, Code
from .readers.htmltree import parse_html
from .readers.safe import local, parse_xml, read_tar
from .text import align_key, math_atoms, score_tokens

THRESHOLDS = {"content": 0.98, "structure": 0.95, "form": 0.90}
COVERAGE_MIN = 0.95
EXCLUSION_CAP = 0.05
# Types a rendering cannot represent, per twin kind: not applicable to the self-check, outside the
# 5 % cap, still in the scores (C5; the "limit" column of R2-B).
DECLARED_LIMITS = {
    "latex": "Pandoc reads \\newblock as a command with an argument and drops the next group (a journal in "
             "\\em), and prints the field names of revtex \\bibinfo and \\BibitemShut: with such a .bbl, references "
             "are judged and inventoried on pdftotext -raw of a PDF input, or on the bibliography list of a LaTeXML "
             "HTML input (arXiv); picture environments are drawings, outside the inventory",
    "jats": "Pandoc keeps JATS references only in the metadata and drops back notes: references and notes are judged "
            "and inventoried on pdftotext of the PDF, in reading order; linked notes are not applicable",
    "pptx": "plain text of the slides: structure and form only as order",
    "akoma-ntoso": "plain text of the Gazzetta PDF in the reading order of poppler: structure and form only as order; "
                   "its layout analysis moves a few blocks (a centred line, the two columns of a page): those order "
                   "checks are self_check_tool_limit exclusions",
    "xlsx-projection": "plain text of Ghostscript: values and rows only",
    "xlsx": "LibreOffice HTML: its sheet index, the Overview heading and the 'Sheet N:' prefix are not workbook "
            "content; drawings are not read by the twin, figures are outside the coverage",
    "wikisource": "MediaWiki HTML: references are an ordered list, not linked notes",
}
NOT_APPLICABLE_BY_KIND = {"wikisource": {"note_linked"}, "jats": {"note_linked"}}
# Omission proofs that cannot fail for a twin, per document and type, counted in the report
# (rule 13: the class resisted the corrections of M4b and of the reference toolchain).
OMISSION_NOT_DETERMINABLE = {
    "D16": {"figure": "the Pandoc inventory sees 164 of the 207 figures of the twin (the others are images in nested "
                      "frames that Pandoc does not read, and it drops 40 captions that LibreOffice prints), so figures "
                      "without a caption pair by number and dropping one generated figure in ten leaves enough"},
}


# Types the primary rendering does not represent, judged instead on `pdftotext -raw` of the input
# PDF (poppler, independent of the PDFium of Marker): Pandoc keeps JATS references only in the
# metadata (citeproc, not the journal's form) and drops back-matter notes; it reads table-wrap,
# so cells and tables are judged on its rendering.
SECONDARY_TYPES = {"jats": {"bib_entry", "footnote"}}


# pdftotext of a PDF: `-raw` (content stream order), except for the PDFs of publishers where
# poppler of the reference toolchain drops word spaces in raw mode (merged words: 65 tokens on D04,
# 12 on D19) and its reading-order mode keeps them; LibreOffice PDFs of PPTX and the arXiv PDF of
# D01 keep `-raw`.
RAW_TEXT_KINDS = {"latex", "pptx"}


def pdf_text(path, kind, root):
    mode = ["-raw"] if kind in RAW_TEXT_KINDS else []
    return tools.run("pdftotext", [*mode, tools.In(path), "-"], root, timeout=300).decode("utf-8", "replace")


def pdf_bib_entries(text):
    """The numbered entries [1], [2], ... of the last reference list of a `pdftotext -raw` text,
    each up to the next number (a page number alone on a line ends the last one)."""
    lines = text.replace("\f", "\n").split("\n")
    run = []
    for i, line in enumerate(lines):
        m = re.match(r"\s*\[(\d+)\]\s", line)
        if not m:
            continue
        n = int(m.group(1))
        if n == 1:
            run = [i]
        elif run and n == len(run) + 1:
            run.append(i)
    entries = []
    for k, i in enumerate(run):
        end = run[k + 1] if k + 1 < len(run) else next((j for j in range(i + 1, len(lines)) if re.fullmatch(r"\s*\d+\s*", lines[j])), len(lines))
        chunk = re.sub(r"(\w)-\n(?=[a-zà-öø-ÿ])", r"\1", "\n".join(lines[i:end]))
        entries.append(align_key(re.sub(r"^\s*\[\d+\]\s*", "", chunk)))
    return [e for e in entries if e]


def _bbl_unreadable(doc, files):
    """True when Pandoc misprints the references of the `.bbl` (`\\newblock`, revtex `\\bibinfo`)."""
    t = doc["twin"]
    bbl = read_tar(files.twin["file"], {t["bbl"]})[t["bbl"]]
    return b"\\newblock" in bbl or b"\\bibinfo" in bbl


def html_bib_entries(data):
    """The entries of the bibliography of a LaTeXML HTML page (arXiv): the text of each
    `li.ltx_bibitem` without its number (`span.ltx_tag_bibitem`)."""
    root = parse_html(data)
    entries = []

    def text(el):
        parts = [] if "ltx_tag_bibitem" in (el.get("class") or "") else [el.text or ""]
        if "ltx_tag_bibitem" not in (el.get("class") or ""):
            for k in el:
                parts.append(text(k))
        parts.append(el.tail or "")
        return "".join(parts)
    for li in root.iter():
        if local(li.tag) == "li" and "ltx_bibitem" in (li.get("class") or "").split():
            body = re.sub(r"\s+", " ", "".join([li.text or ""] + [text(k) for k in li])).strip()
            if body:
                entries.append(body)
    return entries


def _pandoc_json(args, work_root, stdin=None, timeout=600):
    out = tools.run("pandoc", [*args, "-t", "json"], work_root, timeout=timeout, stdin=stdin)
    return json.loads(out)


def _plain_ast(x):
    """Drop attributes Markdown cannot express (links, spans, divs), so gfm needs no raw HTML."""
    if isinstance(x, list):
        out = []
        for v in x:
            if isinstance(v, dict) and v.get("t") in ("RawInline", "RawBlock"):
                continue        # tags kept from an HTML input: the text is in the other nodes
            if isinstance(v, dict) and v.get("t") == "Span":
                out.extend(_plain_ast(v["c"][1]))
            elif isinstance(v, dict) and v.get("t") in ("SmallCaps", "Underline"):
                out.extend(_plain_ast(v["c"]))
            elif isinstance(v, dict) and v.get("t") == "Div":
                out.extend(_plain_ast(v["c"][1]))
            elif isinstance(v, dict) and v.get("t") == "Figure":
                # a figure as GitHub Markdown writes one, its images and then its caption as paragraphs:
                # pandoc writes a Figure block as raw HTML (<figure>, <img>, <figcaption>)
                _, caption, content = v["c"]
                out.extend(_plain_ast([dict(b, t="Para") if b.get("t") == "Plain" else b for b in content + caption[1]]))
            else:
                out.append(_plain_ast(v))
        return out
    if isinstance(x, dict):
        if x.get("t") in ("Link", "Image") and isinstance(x.get("c"), list):
            x["c"][0] = ["", [], []]
        return {k: _plain_ast(v) for k, v in x.items()}
    return x


def _pandoc_gfm(ast, work_root):
    """GitHub Markdown with TeX math between dollars (`$x$`, `$$x$$`), the form the outputs of the
    converters use and `outdoc` reads: without `tex_math_gfm`, which writes `` $`x`$ `` and math
    code blocks. This fixes the format the self-check emits; it does not change the generator."""
    ast = _plain_ast(ast)
    return tools.run("pandoc", ["-f", "json", "-t", "gfm-tex_math_gfm", "--wrap=none"], work_root,
                     stdin=json.dumps(ast).encode()).decode("utf-8", "replace")


def _promote_meta(ast, abstract_heading=False):
    """Title and abstract of the metadata as the first blocks of the body."""
    meta, blocks = ast.get("meta", {}), []
    title = meta.get("title")
    if title and title.get("t") == "MetaInlines":
        blocks.append({"t": "Header", "c": [1, ["", [], []], title["c"]]})
    abstract = meta.get("abstract")
    if abstract and abstract_heading:
        blocks.append({"t": "Header", "c": [2, ["", [], []], [{"t": "Str", "c": "Abstract"}]]})
    if abstract:
        if abstract.get("t") == "MetaBlocks":
            for block in abstract["c"]:
                if abstract_heading and block.get("t") == "Header":
                    block["c"][0] += 2      # sections of the abstract sit under "Abstract"
                blocks.append(block)
        elif abstract.get("t") == "MetaInlines":
            blocks.append({"t": "Para", "c": abstract["c"]})
    ast["blocks"] = blocks + ast["blocks"]
    return ast


def render(doc, files, cache):
    """(kind, text, inventory source) of the self-check rendering of a document."""
    root, kind = cache.tool_work_dir, doc["twin"]["kind"]
    work = cache.work("selfcheck-" + doc["id"])
    try:
        if kind == "latex":
            from .readers.latex import self_check_source
            t = doc["twin"]
            members = read_tar(files.twin["file"], {t["main"], t["bbl"]})
            # the title promoted from the metadata sits above the sections, as in print
            ast = _promote_meta(_pandoc_json(["-f", "latex", "--shift-heading-level-by=1"], root,
                                             stdin=self_check_source(members[t["main"]], members[t["bbl"]]).encode()))
            return "markdown", _pandoc_gfm(ast, root), ("pandoc", ast)
        if kind in ("jats", "docx", "odt", "epub"):
            ast = _promote_meta(_pandoc_json(["-f", kind, tools.In(files.twin["file"])], root), abstract_heading=kind == "jats")
            return "markdown", _pandoc_gfm(ast, root), ("pandoc", ast)
        if kind == "xlsx":
            html = tools.soffice_convert(files.twin["file"], "twin.xlsx", "html", work, root, timeout=600)
            ast = _pandoc_json(["-f", "html", tools.In(html)], root)
            return "markdown", _pandoc_gfm(ast, root), ("pandoc", ast)
        if kind == "xlsx-projection":
            out = work / "gs.txt"
            tools.run("gs", ["-q", "-dNOPAUSE", "-dBATCH", "-dSAFER", "-sDEVICE=txtwrite",
                             tools.Out(out, prefix="-sOutputFile=", allow_empty=True), tools.In(files.input)],
                      root, timeout=600)
            return "plain", out.read_text("utf-8", "replace"), ("selection", None)
        if kind == "pptx":
            pdf = tools.soffice_convert(files.twin["file"], "twin.pptx", "pdf", work, root)
            fodp = tools.soffice_convert(files.twin["file"], "twin.pptx", "fodp", work, root)
            return "plain", pdf_text(pdf, kind, root), ("fodp", Path(fodp).read_bytes())
        if kind == "akoma-ntoso":
            return "plain", pdf_text(files.input, kind, root), ("normattiva", files.twin["inventory"].read_bytes())
        if kind == "wikisource":
            parts = [_page_html(files.twin["pages"][p]["html"]) for p in sorted(files.twin["pages"])
                     if files.twin["quality"].get(p) == 4 and p not in set(doc["twin"].get("isolated_pages", []))]
            html = "\n".join(parts)
            ast = _pandoc_json(["-f", "html"], root, stdin=html.encode())
            return "markdown", _pandoc_gfm(ast, root), ("mediawiki", (ast, html))
    finally:
        shutil.rmtree(work, ignore_errors=True)
    raise BenchError(Code.SELF_CHECK_NOT_RUN, kind)


def _page_html(html):
    """The page text of a MediaWiki rendering, without the quality banner and the header row."""
    start = html.find('<div class="pagetext">')
    body = html[start:] if start >= 0 else html
    body = re.sub(r'<table[^>]*class="ri"[^>]*>.*?</table>', "", body, flags=re.S)
    return body


# --- inventories ------------------------------------------------------------------------------

def _inline_text(xs):
    out = []

    def walk(x):
        if isinstance(x, dict):
            t = x.get("t")
            if t == "Str":
                out.append(x["c"])
            elif t in ("Space", "SoftBreak", "LineBreak"):
                out.append(" ")
            elif t in ("Note", "Math"):
                out.append(" ")
            elif t == "Image":
                out.append(" ")         # the alternative text of an image is not printed
            elif t == "Cite":
                walk(x["c"][1])
            elif t == "Code":
                out.append(x["c"][1])
            elif "c" in x:
                walk(x["c"])
        elif isinstance(x, list):
            for v in x:
                walk(v)
    walk(xs)
    return "".join(out)


PICTURES = {"picture", "tikzpicture", "pspicture"}


def pandoc_inventory(ast, split_display=True):
    """Units per type from a Pandoc AST: heading, paragraph, list_item, cell, table, formula,
    inline_math, footnote, bib_entry, figure. The width paragraph that Pandoc keeps from
    `\\begin{thebibliography}{N}` is discarded (C4)."""
    units = defaultdict(list)

    def inlines(xs):
        for x in xs if isinstance(xs, list) else []:
            if not isinstance(x, dict):
                continue
            t = x.get("t")
            if t == "Math":
                kind = x["c"][0]["t"]
                units["formula" if kind == "DisplayMath" else "inline_math"].append(" ".join(math_atoms(x["c"][1])))
            elif t == "Note":
                text = " ".join(_inline_text(b.get("c", [])) for b in x["c"] if isinstance(b, dict))
                if align_key(text):
                    units["footnote"].append(align_key(text))
                for b in x["c"]:
                    if isinstance(b, dict):
                        inlines(b.get("c", []))
            elif t == "Image":
                units["figure"].append("")      # an image outside a Figure block: no caption (alt text is not printed)
            elif isinstance(x.get("c"), list):
                inlines(x["c"])

    def blocks(bs, ctx):
        for b in bs:
            t = b.get("t")
            c = b.get("c")
            if t == "Header":
                units["heading"].append(align_key(_inline_text(c[2])))
                inlines(c[2])
            elif t in ("Para", "Plain"):
                for segment in ([c] if ctx == "bib" or not split_display else _display_segments(c)):
                    text = align_key(_inline_text(segment))
                    if re.match(r"(fig(ure)?|table|tab)\s*\d", text) and ctx != "bib":
                        continue    # a caption or a flattened table, not a body paragraph
                    if ctx == "bib":
                        if text and not re.fullmatch(r"\d+", text):
                            units["bib_entry"].append(text)
                    elif text and ctx in ("body", "list"):
                        units["list_item" if ctx == "list" else "paragraph"].append(text)
                inlines(c)
            elif t in ("BulletList", "OrderedList"):
                items = c if t == "BulletList" else c[1]
                for item in items:
                    first = next((x for x in item if x.get("t") in ("Plain", "Para")), None)
                    if first is not None:
                        text = align_key(_inline_text(first["c"]))
                        if text:
                            units["list_item"].append(text)
                        inlines(first["c"])
                    blocks([x for x in item if x is not first], "list")
            elif t == "Div" and "table-wrap" in c[0][1]:
                # a JATS table-wrap: caption, table, then the paragraphs of its table-wrap-foot, which
                # print as notes below the table (Pandoc writes them as paragraphs)
                seen_table = False
                for child in c[1]:
                    if seen_table and child.get("t") in ("Para", "Plain"):
                        text = align_key(_inline_text(child["c"]))
                        if text:
                            units["footnote"].append(text)
                        inlines(child["c"])
                    else:
                        seen_table = seen_table or child.get("t") == "Table"
                        blocks([child], ctx)
            elif t == "Div":
                attrs = c[0]
                if any(k in PICTURES for k in attrs[1]):
                    continue        # a drawing: Pandoc prints its coordinates as text
                bib = attrs[0] in ("refs",) or any(k in ("references", "thebibliography", "csl-bib-body") for k in attrs[1])
                blocks(c[1], "bib" if bib or ctx == "bib" else ctx)
            elif t == "BlockQuote":
                blocks(c, ctx)
            elif t == "Table":
                units["table"].append("table")
                for cell_text in _table_cells(c):
                    if align_key(cell_text):
                        units["cell"].append(align_key(cell_text))
            elif t == "Figure":
                caption = c[1][1] if len(c) > 1 and c[1] else []
                units["figure"].append(align_key(" ".join(_inline_text(x.get("c", [])) for x in caption)))
                blocks(_without_images(c[2]), "figure")     # its images are this one figure
            elif t == "DefinitionList":
                for term, defs in c:
                    first = [b for d in defs for b in d[:1] if isinstance(b, dict)]
                    units["list_item"].append(align_key(_inline_text(term) + " " + " ".join(_inline_text(b.get("c", [])) for b in first)))
    blocks(ast["blocks"], "body")
    return units


def _without_images(x):
    if isinstance(x, list):
        return [_without_images(v) for v in x if not (isinstance(v, dict) and v.get("t") == "Image")]
    if isinstance(x, dict):
        return {k: _without_images(v) for k, v in x.items()}
    return x


def _display_segments(xs):
    """The inline list of a paragraph split at its display formulas."""
    segments, current = [], []
    for x in xs:
        if isinstance(x, dict) and x.get("t") == "Math" and x["c"][0].get("t") == "DisplayMath":
            segments.append(current)
            current = []
        else:
            current.append(x)
    segments.append(current)
    return [seg for seg in segments if seg]


def _table_cells(c):
    texts = []

    def walk(x):
        if isinstance(x, dict):
            if x.get("t") in ("Plain", "Para"):
                texts.append(_inline_text(x["c"]))
                return
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            # a Pandoc cell: [attr, alignment, rowspan, colspan, blocks]
            if len(x) == 5 and isinstance(x[4], list) and isinstance(x[2], int) and isinstance(x[3], int):
                texts.append(" ".join(_inline_text(b.get("c", [])) for b in x[4] if isinstance(b, dict)))
                return
            for v in x:
                walk(v)
    walk(c[3:])
    return texts


def fodp_inventory(data):
    """Text paragraphs of the flat ODP of LibreOffice, speaker notes excluded (B1 for PPTX)."""
    root = parse_xml(data)
    units = defaultdict(list)

    def walk(el):
        name = local(el.tag)
        if name == "notes":
            return
        if name in ("p", "h") and el.tag.startswith("{urn:oasis:names:tc:opendocument:xmlns:text"):
            parts = []

            def rec(x):
                n = local(x.tag)
                if n in ("line-break", "tab", "s"):
                    parts.append(" ")
                elif n != "page-number":
                    parts.append(x.text or "")
                    for k in x:
                        rec(k)
                if x is not el:
                    parts.append(x.tail or "")
            rec(el)
            key = align_key("".join(parts))
            if key:
                units["text"].append(key)
            return
        for k in el:
            walk(k)
    for page in (x for x in root.iter() if local(x.tag) == "page"):
        walk(page)
    return units


NORMATTIVA_CLASSES = {"article-num-akn": "article_num", "article-heading-akn": "article_heading",
                      "art-comma-div-akn": "comma", "pointedList-rest-akn": "point",
                      "preamble-citation-akn": "citation"}


def normattiva_inventory(data):
    """Articles, commi, points and citations of the Normattiva HTML export (B1 for Akoma Ntoso)."""
    root = parse_html(data)
    found = defaultdict(list)

    def text_without(el, skip):
        parts = []

        def rec(x):
            if x is not el and (x.get("class") in skip):
                parts.append(x.tail or "")
                return
            parts.append(x.text or "")
            for k in x:
                rec(k)
            if x is not el:
                parts.append(x.tail or "")
        rec(el)
        return "".join(parts)
    for el in root.iter():
        kind = NORMATTIVA_CLASSES.get(el.get("class", ""))
        if kind:
            found[kind].append(align_key(text_without(el, {"pointedList-rest-akn", "comma-num-akn"} if kind == "comma" else set())))
    units = defaultdict(list)
    units["heading"] = [align_key(a + " " + h) for a, h in zip(found["article_num"], found["article_heading"])]
    units["paragraph"] = found["comma"] + found["citation"]
    units["list_item"] = found["point"]
    return units


def mediawiki_inventory(ast, html):
    units = pandoc_inventory(ast)
    root = parse_html(html.encode())
    # one note per child of a reference list: an <li>, or a block before them, which ProofreadPage
    # writes for the continuation of a note begun on the previous page (Pandoc makes it a list item)
    notes = [align_key("".join(child.itertext()).replace("↑", " ")) for ol in root.iter()
             if local(ol.tag) == "ol" and "references" in (ol.get("class") or "") for child in ol]
    units["footnote"] = [n for n in notes if n]
    units["list_item"] = [u for u in units["list_item"] if u not in set(units["footnote"])]
    return units


def _calc_html(ast):
    """The LibreOffice HTML of a workbook without what LibreOffice adds: the "Overview" heading with
    its paragraph of links to the sheets, and the "Sheet N:" prefix of each sheet heading."""
    blocks, out = ast["blocks"], []
    for i, b in enumerate(blocks):
        if b.get("t") == "Header" and b["c"][1][0] == "overview":
            continue
        if b.get("t") == "Para" and all(x.get("t") in ("Link", "LineBreak", "SoftBreak", "Space") for x in b["c"]) and \
                any(x.get("t") == "Link" and x["c"][2][0].startswith("#table") for x in b["c"]):
            continue
        if b.get("t") == "Header":
            inl = b["c"][2]
            if len(inl) >= 3 and inl[0] == {"t": "Str", "c": "Sheet"} and inl[1].get("t") == "Space" and \
                    re.fullmatch(r"\d+:", inl[2].get("c", "")):
                b = {"t": "Header", "c": [b["c"][0], b["c"][1], inl[4:] if len(inl) > 3 and inl[3].get("t") == "Space" else inl[3:]]}
        out.append(b)
    return dict(ast, blocks=out)


def inventory(source, twin_kind=None):
    kind, data = source
    if kind == "pandoc":
        if twin_kind == "xlsx":
            units = pandoc_inventory(_calc_html(data))
            units.pop("figure", None)       # the twin does not read drawings (D07: controls not printed)
            return units
        # Pandoc reads every ODT formula object as display math, inline ones too: no split there
        return pandoc_inventory(data, split_display=twin_kind != "odt")
    if kind == "fodp":
        return fodp_inventory(data)
    if kind == "normattiva":
        return normattiva_inventory(data)
    if kind == "mediawiki":
        return mediawiki_inventory(*data)
    return {}


# --- coverage ---------------------------------------------------------------------------------

GENERATED_TYPES = {  # inventory type -> check types and how a generated unit is keyed
    "heading": ("heading",), "paragraph": ("paragraph",), "list_item": ("paragraph",), "cell": ("cell",),
    "table": ("table",), "formula": ("formula",), "inline_math": ("inline_math",), "footnote": ("footnote",),
    "bib_entry": ("bib_entry",), "figure": ("figure",), "text": ("paragraph", "heading", "cell"),
}
# Pandoc reads every ODT formula object as display math: formulas of the inventory align with the
# display and the inline formulas of the twin
GENERATED_TYPES_BY_KIND = {"odt": {"formula": ("formula", "inline_math")}}


TEXT_TYPES = ("paragraph", "heading", "cell", "footnote", "bib_entry", "figure")


def generated_units(checks):
    """Alignment keys of the generated checks, by check type."""
    units = defaultdict(list)
    for c in checks:
        t = c["type"]
        if t == "inline_math":
            units[t].extend(" ".join(f) for f in c["formulas"])
        elif t == "formula":
            units[t].append(" ".join(c["atoms"]))
        elif t == "table":
            units[t].append("table")
        elif t in TEXT_TYPES:
            units[t].append(align_key(" ".join(c.get("tokens", []))))
    return units


def allowances(checks):
    """Generated key -> printed tokens the twin cannot know (two per gap, as in the matcher)."""
    out = {}
    for c in checks:
        if c.get("gaps") and c["type"] in ("paragraph", "heading"):
            key = align_key(" ".join(c["tokens"]))
            out[key] = max(out.get(key, 0), 2 * c["gaps"])
    return out


def _fuzzy(a, b, allow=0):
    """Shared tokens over the longer unit; the generated unit `b` may lack `allow` tokens of `a`."""
    ta, tb = Counter(a.split()), Counter(b.split())
    la, lb = sum(ta.values()), sum(tb.values())
    la -= min(allow, max(0, la - lb))
    return sum((ta & tb).values()) / max(1, la, lb)


def _window(inner, outer):
    """(start, end) in `outer` of the tightest span of the best window of the matcher's length,
    when it holds 90 % of `inner`; None otherwise."""
    n = len(inner)
    need = Counter(inner)
    length = int(n * 1.1) + 1
    if n == 0 or len(outer) < 0.9 * n:
        return None
    have, covered, best, best_end = Counter(), 0, 0, -1
    for i, tok in enumerate(outer):
        if have[tok] < need.get(tok, 0):
            covered += 1
        have[tok] += 1
        if i >= length:
            old = outer[i - length]
            have[old] -= 1
            if have[old] < need.get(old, 0):
                covered -= 1
        if covered > best:
            best, best_end = covered, i + 1
    if best < 0.9 * n:
        return None
    a = max(0, best_end - length)
    window = Counter(outer[a:best_end])
    while a < best_end and window[outer[a]] > need.get(outer[a], 0):
        window[outer[a]] -= 1         # a token the coverage does not need: drop it
        a += 1
    return a, best_end


def _rare(keys):
    """The three rarest tokens of each key, by the number of keys holding them."""
    df = Counter(t for k in keys for t in set(k.split()))
    return {k: sorted(set(k.split()), key=lambda t: (df[t], t))[:3] for k in keys}


def _assign(inv, pool, allowance=None, others=()):
    """Match inventory units to generated units of `pool`. Returns (matched count, the generated
    keys used, the inventory units left).

    1. equal keys; 2. 90 % of the tokens (the generated unit may lack its declared gaps);
    3. an inventory piece (8 tokens or more) that a tool split out of one generated unit: 90 % of
    it in a window of that unit not yet claimed by another inventory unit;
    4. an inventory unit assembled by a tool from two or more generated units of any text type
    (`others`, for example a paragraph and the caption of the figure it holds): their disjoint
    windows hold 90 % of its tokens, and one of them is an unused unit of `pool`.
    Each part of a generated unit of `pool` is claimed once: the whole unit by an equal key, a
    window by the other steps.
    """
    allowance = allowance or {}
    claimed = defaultdict(list)          # generated key -> claimed windows (start, end) of its tokens
    tokens_of = {}

    def toks(k):
        if k not in tokens_of:
            tokens_of[k] = k.split()
        return tokens_of[k]

    def claim(key, unit):
        w = _window(toks(unit), toks(key))
        claimed[key].append(w or (0, len(toks(key))))

    def free(key, w):
        return all(w[1] <= a or w[0] >= b for a, b in claimed[key])

    available = Counter(pool)
    rest, used = [], []
    for unit in inv:
        if available[unit] > 0:
            available[unit] -= 1
            used.append(unit)
            claimed[unit].append((0, len(toks(unit))))
        else:
            rest.append(unit)
    leftovers = list(available.elements())
    index = defaultdict(set)
    for i, g in enumerate(leftovers):
        for tok in set(g.split()):
            index[tok].add(i)
    taken, left = set(), []
    for unit in rest:
        present = [x for x in set(unit.split()) if index.get(x)]
        candidates = set().union(*(index[x] for x in sorted(present, key=lambda x: (len(index[x]), x))[:3])) - taken if present else set()
        score = lambda i: _fuzzy(unit, leftovers[i], allowance.get(leftovers[i], 0))
        best = max(sorted(candidates), key=score, default=None)
        if best is not None and score(best) >= 0.9:
            taken.add(best)
            used.append(leftovers[best])
            claim(leftovers[best], unit)
        else:
            left.append(unit)
    own = sorted(set(pool))
    own_set = set(own)
    every = sorted(own_set | {k for k in others if len(k.split()) >= 4})
    rare = _rare(every)
    holders = defaultdict(set)
    for k in own:
        for tok in set(toks(k)):
            holders[tok].add(k)
    final = []
    for unit in left:
        words = toks(unit)
        if len(words) < 8:
            final.append(unit)
            continue
        have = set(words)
        # 3. the piece inside a free window of one generated unit
        seeds = sorted((t for t in have if holders.get(t)), key=lambda t: (len(holders[t]), t))[:3]
        hosts = set.intersection(*(holders[t] for t in seeds)) if len(seeds) == 3 else set()
        host = None
        for k in sorted(hosts):
            w = _window(words, toks(k))
            if w and free(k, w):
                host = k
                claimed[k].append(w)
                break
        if host is not None:
            used.append(host)
            continue
        # 4. several generated units inside the inventory unit (own units not claimed yet)
        spans = []
        for k in every:
            if k in own_set and claimed[k]:
                continue
            if len(rare[k]) == 3 and sum(t in have for t in rare[k]) >= 2:     # a prefilter; the window decides
                w = _window(toks(k), words)
                if w:
                    spans.append((len(toks(k)), w, k))
        chosen, covered = [], 0
        for size, (a, b), k in sorted(spans, key=lambda x: (-x[0], x[1], x[2])):
            if all(b <= a2 or a >= b2 for (a2, b2), _ in chosen):
                chosen.append(((a, b), k))
                covered += min(size, b - a)
        if len(chosen) >= 2 and covered >= 0.9 * len(words) and any(k in own_set for _, k in chosen):
            for _, k in chosen:
                if k in own_set:
                    used.append(k)
                    claimed[k].append((0, len(toks(k))))
        else:
            final.append(unit)
    return len(inv) - len(final), used, final


def _assign_figures(inv, pool):
    """Figures: equal captions, then 90 % of the caption tokens, or the caption of the inventory in a
    window of a generated one (a tool that drops the printed label, "Figure 1", that the twin keeps);
    an image without a caption on either side (an empty key) pairs with any figure left, the
    caption being a separate check."""
    available = Counter(pool)
    used, left = [], []
    for unit in inv:
        if unit and available[unit] > 0:
            available[unit] -= 1
            used.append(unit)
        else:
            left.append(unit)
    remaining = sorted(available.elements(), key=lambda g: (g == "", g))
    final = []
    for unit in sorted(left, key=lambda u: (u == "", u)):
        pick = next((g for g in remaining if g and unit and (_fuzzy(unit, g) >= 0.9 or _window(unit.split(), g.split()))), None)
        if pick is None:
            pick = next((g for g in remaining if not g or not unit), None)
        if pick is None:
            final.append(unit)
            continue
        remaining.remove(pick)
        used.append(pick)
    return len(inv) - len(final), used, final


def coverage(inventory_units, generated, drop_every=None, allowance=None, twin_kind=None):
    """{type: (matched, inventory, ratio)}. With `drop_every` (omission proof), the generator
    forgets one in N of the generated keys used by the matching (at least one), duplicates included."""
    result = {}
    others = [k for t in TEXT_TYPES for k in generated.get(t, [])]
    for kind, inv in inventory_units.items():
        if not inv:
            continue
        pool = []
        for t in GENERATED_TYPES_BY_KIND.get(twin_kind, {}).get(kind, GENERATED_TYPES.get(kind, (kind,))):
            pool += generated.get(t, [])
        rest = others
        assign = (lambda i, p, r: _assign_figures(i, p)) if kind == "figure" else \
            (lambda i, p, r: _assign(i, p, allowance, r if kind in TEXT_INVENTORY else ()))
        if drop_every:
            _, used, _ = assign(inv, pool, others)
            keys = list(dict.fromkeys(used))
            dropped = {k for i, k in enumerate(keys) if (i + 1) % drop_every == 0} or set(keys[:1])
            pool = [u for u in pool if u not in dropped]
            rest = [u for u in others if u not in dropped]
        matched, _, _ = assign(inv, pool, rest)
        result[kind] = (matched, len(inv), matched / len(inv))
    return result


TEXT_INVENTORY = {"paragraph", "list_item", "heading", "footnote", "bib_entry", "text"}


# --- judgement --------------------------------------------------------------------------------

def run_self_check(doc, files, cache, checkset, exclusion_entries):
    """Numbers of the self-check of one document (no document text)."""
    from . import exclusions as excl
    from .match import Matcher, evaluate
    from .outdoc import from_markdown, from_text

    mode, text, inv_source = render(doc, files, cache)
    out = from_markdown(text) if mode == "markdown" else from_text(text)
    kind = checkset["twin_kind"]
    results = evaluate(checkset, out)
    secondary = set(SECONDARY_TYPES.get(kind, set()))
    bbl_limit = kind == "latex" and _bbl_unreadable(doc, files)
    pdf_bib, html_bib = bbl_limit and doc["format"] == "pdf", bbl_limit and doc["format"] == "html"
    if bbl_limit and (pdf_bib or html_bib):
        secondary.add("bib_entry")
    printed = None
    if secondary and str(files.input).endswith(".pdf") or (secondary and doc["format"] == "pdf"):
        printed = pdf_text(files.input, kind, cache.tool_work_dir)
    elif secondary and html_bib:
        printed = "\n\n".join(html_bib_entries(Path(files.input).read_bytes()))
    if printed is not None:
        second = evaluate({**checkset, "checks": [c for c in checkset["checks"] if c["type"] in secondary
                                                  or c.get("role") == "table-caption"]},
                          from_text(printed))
        by_id = {r["id"]: r for r in second}
        results = [by_id.get(r["id"], r) for r in results]
    for r in results:
        if r["type"] in NOT_APPLICABLE_BY_KIND.get(kind, set()):
            r["outcome"] = "not_applicable"
    checks = checkset["checks"]
    if inv_source[0] == "selection":
        rows = [c["id"] for c in checks if c["type"] == "table_row"]
        inv = {"row": [align_key(r) for r in rows]}
        gen = {"row": [align_key(r) for r in rows]}
        GENERATED_TYPES.setdefault("row", ("row",))
    else:
        inv = inventory(inv_source, kind)
        if pdf_bib:
            inv["bib_entry"] = pdf_bib_entries(printed)
        elif html_bib:
            inv["bib_entry"] = [k for k in (align_key(e) for e in printed.split("\n\n")) if k]
        gen = generated_units(checks)
    allow = allowances(checks)
    cov = coverage(inv, gen, allowance=allow, twin_kind=kind)
    omission = coverage(inv, gen, drop_every=10, allowance=allow, twin_kind=kind)
    matcher = Matcher(checks, out, checkset.get("rules", ()))
    inv_keys = {k: Counter(v) for k, v in inv.items()}

    def inventory_matched(check):
        key = align_key(" ".join(check.get("tokens", [])))
        kinds = [k for k, types in GENERATED_TYPES.items() if check["type"] in types]
        return any(inv_keys.get(k, {}).get(key) for k in kinds)

    def rendering_ratio(check):
        return matcher.find(check.get("tokens", []))[0] if check.get("tokens") else 0.0

    statuses = excl.validate(exclusion_entries, checkset, results, inventory_matched, rendering_ratio)
    excluded = {e["check"] for e in exclusion_entries if statuses.get(e["check"]) == "ok"}
    classes = {}
    for cls in THRESHOLDS:
        rs = [r for r in results if r["class"] == cls]
        applicable = [r for r in rs if r["outcome"] != "not_applicable"]
        kept = [r for r in applicable if r["id"] not in excluded]
        before = sum(r["outcome"] == "pass" for r in applicable) / len(applicable) if applicable else None
        after = sum(r["outcome"] == "pass" for r in kept) / len(kept) if kept else None
        share = (len(applicable) - len(kept)) / len(rs) if rs else 0.0
        classes[cls] = {"generated": len(rs), "not_applicable": len(rs) - len(applicable), "excluded": len(applicable) - len(kept),
                        "excluded_share": round(share, 4), "before": before, "after": after, "threshold": THRESHOLDS[cls],
                        "meets": after is None or after >= THRESHOLDS[cls], "cap_ok": share <= EXCLUSION_CAP}
    coverage_ok = all(ratio >= COVERAGE_MIN for _, _, ratio in cov.values())
    not_determinable = OMISSION_NOT_DETERMINABLE.get(doc["id"], {})
    omission_ok = all(omission[k][2] < COVERAGE_MIN for k in omission if k not in not_determinable)
    failing = [{"id": r["id"], "class": r["class"], "type": r["type"], "outcome": r["outcome"]}
               for r in results if r["outcome"] not in ("pass", "not_applicable") and r["id"] not in excluded]
    verdict = (all(c["meets"] and c["cap_ok"] for c in classes.values()) and coverage_ok and omission_ok
               and all(s == "ok" for s in statuses.values()))
    return {"document": doc["id"], "rendering": mode, "limit": DECLARED_LIMITS.get(kind),
            "classes": classes, "coverage": {k: {"matched": m, "inventory": n, "ratio": round(r, 4)} for k, (m, n, r) in cov.items()},
            "omission": {k: round(r, 4) for k, (_, _, r) in omission.items()}, "omission_ok": omission_ok,
            "omission_not_determinable": sorted(k for k in not_determinable if k in omission),
            "coverage_ok": coverage_ok, "exclusions": statuses, "failing": failing, "verdict": verdict}
