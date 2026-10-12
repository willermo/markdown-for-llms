#!/usr/bin/env python3
"""Genera contenitori sintetici dalla specifica manuale; renderer già locali.

Due directory nuove danno due generazioni indipendenti. Non esegue Pandoc,
TeX, OCR o modelli. Ogni tool è esplicito, validato, confinato e rendicontato.
"""
import argparse
import base64
import copy
import html as htmlmod
import io
import json
from pathlib import Path
import shutil
import struct
import textwrap
import zipfile
import zlib
import xml.etree.ElementTree as ET

from data import counts, descriptor, digest, digest_bytes, load, write_new
from resources import run
from safety import html, png, regular, svg, xml, zip_members, no_symlink_root

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
M = "http://schemas.openxmlformats.org/officeDocument/2006/math"
R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
REL = "http://schemas.openxmlformats.org/package/2006/relationships"
X = "http://www.w3.org/1999/xhtml"
MM = "http://www.w3.org/1998/Math/MathML"
for prefix, namespace in [("w", W), ("m", M), ("r", R)]:
    ET.register_namespace(prefix, namespace)


def esc(text):
    return htmlmod.escape(text, quote=True)


def put(root, name, content):
    from safety import relative
    relative(name)
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as f:
        f.write(content.encode("utf-8") if isinstance(content, str) else content)
    return path


def figure(language):
    """PNG RGB original: axes, 3 colored regions, 3 square legend swatches."""
    width, height = 240, 140
    colors = [(36, 99, 171), (239, 145, 42), (56, 151, 103)]
    if language == "en":
        colors = [colors[2], colors[0], colors[1]]
    rows = []
    for y in range(height):
        row = bytearray()
        for x in range(width):
            color = (255, 255, 255)
            if (x in (19, 20) and 15 <= y <= 116) or (y in (115, 116) and 19 <= x <= 221):
                color = (20, 20, 20)
            for i, c in enumerate(colors):
                if 40 + i * 55 <= x < 70 + i * 55 and 92 - i * 25 <= y < 115:
                    color = c
                if 30 + i * 70 <= x < 40 + i * 70 and 126 <= y < 136:
                    color = c
            row.extend(color)
        rows.append(b"\0" + row)
    def chunk(name, payload):
        return struct.pack(">I", len(payload)) + name + payload + struct.pack(">I", zlib.crc32(name + payload) & 0xffffffff)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(b"".join(rows), 9)) + chunk(b"IEND", b"")


def archive(root, source_dir, name, first=None):
    members = sorted(p for p in source_dir.rglob("*") if p.is_file())
    if first:
        members.sort(key=lambda p: (p.relative_to(source_dir).as_posix() != first, p.relative_to(source_dir).as_posix()))
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as z:
        for path in members:
            rel = path.relative_to(source_dir).as_posix()
            info = zipfile.ZipInfo(rel, (1980, 1, 1, 0, 0, 0))
            info.external_attr = 0o100644 << 16
            info.compress_type = zipfile.ZIP_STORED  # fixed, no misleading compression ratio
            z.writestr(info, path.read_bytes())
    content = stream.getvalue()
    zip_members(content)
    return put(root, name, content)


def elem(tag, parent=None, **attributes):
    result = ET.Element(tag, attributes) if parent is None else ET.SubElement(parent, tag, attributes)
    return result


def wx(tag):
    return "{" + W + "}" + tag


def mx(tag):
    return "{" + M + "}" + tag


def runtext(parent, value):
    r = elem(wx("r"), parent)
    t = elem(wx("t"), r, **{"{http://www.w3.org/XML/1998/namespace}space": "preserve"})
    t.text = value


def omml(node, parent):
    kind = node["type"]
    if kind == "Seq":
        for item in node["items"]:
            omml(item, parent)
    elif kind == "Arg":
        omml(node["body"], parent)
    elif kind in ("Symbol", "Number", "Operator", "Text"):
        r = elem(mx("r"), parent)
        t = elem(mx("t"), r)
        t.text = node.get("value", node.get("lexeme", {"equals": "=", "plus": "+", "minus": "−"}.get(node.get("kind"))))
    elif kind == "Scripts":
        tag = "sSubSup" if node["sub"] and node["sup"] else "sSub" if node["sub"] else "sSup"
        el = elem(mx(tag), parent)
        omml(node["base"], elem(mx("e"), el))
        for field in ("sub", "sup"):
            if node[field]:
                omml(node[field], elem(mx(field), el))
    elif kind == "Fraction":
        el = elem(mx("f"), parent)
        omml(node["numerator"], elem(mx("num"), el))
        omml(node["denominator"], elem(mx("den"), el))
    else:
        raise ValueError("OMML subset authoring: " + kind)


def xmlbytes(node):
    return ET.tostring(node, encoding="utf-8", xml_declaration=True)


def docx(root, document, assets):
    source = root / "sources" / document["document_id"]
    source.mkdir()
    tree = elem(wx("document"))
    body = elem(wx("body"), tree)
    relationships = elem("Relationships", xmlns=REL)
    elem("Relationship", relationships, Id="rFoot", Type=R + "/footnotes", Target="footnotes.xml")
    elem("Relationship", relationships, Id="rStyles", Type=R + "/styles", Target="styles.xml")
    elem("Relationship", relationships, Id="rImage", Type=R + "/image", Target="media/figure-it.png")
    foot = elem(wx("footnotes"))
    top_index, page, bookmark = 0, 1, 0
    for b in document["blocks"]:
        if b["type"] == "note":
            el = elem(wx("footnote"), foot, **{wx("id"): "1"})
            p = elem(wx("p"), el)
            runtext(p, b["text"])
            b["locator"] = {"kind": "docx", "part": "word/footnotes.xml", "footnote": "1"}
            continue
        if b["page"] != page:
            p = elem(wx("p"), body)
            r = elem(wx("r"), p)
            elem(wx("br"), r, **{wx("type"): "page"})
            top_index += 1
            page = b["page"]
        b["locator"] = {"kind": "docx", "part": "word/document.xml", "body_child": top_index}
        if b["type"] == "table":
            tbl = elem(wx("tbl"), body)
            grid = elem(wx("tblGrid"), tbl)
            for _ in range(3):
                elem(wx("gridCol"), grid, **{wx("w"): "2800"})
            for row in b["cells"]:
                tr = elem(wx("tr"), tbl)
                for value in row:
                    tc = elem(wx("tc"), tr)
                    runtext(elem(wx("p"), tc), value)
            if b.get("caption"):
                caption = elem(wx("p"), body)
                runtext(caption, b["caption"])
                b["locator"]["caption_child"] = top_index + 1
                top_index += 1
        else:
            p = elem(wx("p"), body)
            pr = elem(wx("pPr"), p)
            if b["type"] == "heading":
                elem(wx("pStyle"), pr, **{wx("val"): "Heading1"})
            if b.get("anchor"):
                bookmark += 1
                elem(wx("bookmarkStart"), p, **{wx("id"): str(bookmark), wx("name"): b["anchor"]})
            if b["type"] == "formula":
                omml(b["formulas"][0]["structure_expected"], elem(mx("oMath"), p))
            elif b["type"] == "figure":
                r = elem(wx("r"), p)
                drawing = elem(wx("drawing"), r)
                # Standard inline DrawingML, independent of Pandoc.
                fragment = '<wp:inline xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing" xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture" xmlns:r="' + R + '"><wp:extent cx="2286000" cy="1333500"/><wp:docPr id="1" name="Figura originale" descr="' + esc(b["caption"]) + '"/><a:graphic><a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture"><pic:pic><pic:nvPicPr><pic:cNvPr id="0" name="figure-it.png"/><pic:cNvPicPr/></pic:nvPicPr><pic:blipFill><a:blip r:embed="rImage"/><a:stretch><a:fillRect/></a:stretch></pic:blipFill><pic:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="2286000" cy="1333500"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr></pic:pic></a:graphicData></a:graphic></wp:inline>'
                drawing.append(xml(fragment.encode()))
                caption = elem(wx("p"), body)
                runtext(caption, b["caption"])
                b["locator"]["caption_child"] = top_index + 1
                top_index += 1
            else:
                runtext(p, b["text"])
                for n, ref in enumerate(b.get("references", [])):
                    runtext(p, " ")
                    if ref["kind"] == "note_call":
                        r = elem(wx("r"), p)
                        elem(wx("footnoteReference"), r, **{wx("id"): "1"})
                    else:
                        if ref["target"].startswith("#"):
                            link = elem(wx("hyperlink"), p, **{wx("anchor"): ref["target"][1:]})
                        else:
                            relid = "rLink" + b["block_id"] + str(n)
                            elem("Relationship", relationships, Id=relid, Type=R + "/hyperlink", Target=ref["target"], TargetMode="External")
                            link = elem(wx("hyperlink"), p, **{"{" + R + "}id": relid})
                        runtext(link, ref["text"])
            if b.get("anchor"):
                elem(wx("bookmarkEnd"), p, **{wx("id"): str(bookmark)})
        top_index += 1
    section = elem(wx("sectPr"), body)
    elem(wx("pgSz"), section, **{wx("w"): "11906", wx("h"): "16838"})
    elem(wx("pgMar"), section, **{wx("top"): "900", wx("right"): "900", wx("bottom"): "900", wx("left"): "900"})
    styles = elem(wx("styles"))
    defaults = elem(wx("docDefaults"), styles)
    rpr = elem(wx("rPr"), elem(wx("rPrDefault"), defaults))
    elem(wx("rFonts"), rpr, **{wx("ascii"): "DejaVu Sans", wx("hAnsi"): "DejaVu Sans", wx("cs"): "DejaVu Sans"})
    elem(wx("sz"), rpr, **{wx("val"): "20"})
    s = elem(wx("style"), styles, **{wx("type"): "paragraph", wx("styleId"): "Heading1"})
    elem(wx("name"), s, **{wx("val"): "heading 1"})
    spr = elem(wx("rPr"), s)
    elem(wx("sz"), spr, **{wx("val"): "28"})
    contenttypes = '<?xml version="1.0" encoding="UTF-8"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Default Extension="png" ContentType="image/png"/><Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/><Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/><Override PartName="/word/footnotes.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.footnotes+xml"/></Types>'
    for name, content in [("[Content_Types].xml", contenttypes), ("_rels/.rels", '<Relationships xmlns="' + REL + '"><Relationship Id="rDoc" Type="' + R + '/officeDocument" Target="word/document.xml"/></Relationships>'), ("word/document.xml", xmlbytes(tree)), ("word/styles.xml", xmlbytes(styles)), ("word/footnotes.xml", xmlbytes(foot)), ("word/_rels/document.xml.rels", xmlbytes(relationships)), ("word/media/figure-it.png", assets["figure-it.png"])]:
        put(source, name, content)
    return archive(root, source, "inputs/" + document["document_id"] + ".docx")


def mathml(node):
    kind = node["type"]
    if kind == "Seq":
        return "<mrow>" + "".join(mathml(x) for x in node["items"]) + "</mrow>"
    if kind == "Arg":
        return mathml(node["body"])
    if kind == "Symbol":
        return "<mi>" + esc(node["value"]) + "</mi>"
    if kind == "Number":
        return "<mn>" + esc(node["lexeme"]) + "</mn>"
    if kind == "Operator":
        return "<mo>" + {"equals": "=", "minus": "−", "plus": "+"}[node["kind"]] + "</mo>"
    if kind == "Scripts":
        tag = "msubsup" if node["sub"] and node["sup"] else "msub" if node["sub"] else "msup"
        return "<" + tag + ">" + mathml(node["base"]) + (mathml(node["sub"]) if node["sub"] else "") + (mathml(node["sup"]) if node["sup"] else "") + "</" + tag + ">"
    if kind == "Fraction":
        return "<mfrac>" + mathml(node["numerator"]) + mathml(node["denominator"]) + "</mfrac>"
    raise ValueError("MathML authoring subset")


def epub(root, document, assets):
    source = root / "sources" / document["document_id"]
    source.mkdir()
    put(source, "mimetype", "application/epub+zip")
    put(source, "META-INF/container.xml", '<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container"><rootfiles><rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/></rootfiles></container>')
    put(source, "OEBPS/content.opf", '<package xmlns="http://www.idpf.org/2007/opf" version="3.0" unique-identifier="book"><metadata xmlns:dc="http://purl.org/dc/elements/1.1/"><dc:identifier id="book">urn:synthetic:a002:C03</dc:identifier><dc:title>A portable laboratory</dc:title><dc:language>en</dc:language><meta property="dcterms:modified">2000-01-01T00:00:00Z</meta></metadata><manifest><item id="c1" href="chapter1.xhtml" media-type="application/xhtml+xml" properties="mathml"/><item id="c2" href="chapter2.xhtml" media-type="application/xhtml+xml" properties="mathml"/><item id="nav" href="nav.xhtml" media-type="application/xhtml+xml" properties="nav"/><item id="css" href="style.css" media-type="text/css"/><item id="fig" href="figure-en.png" media-type="image/png"/></manifest><spine><itemref idref="c1"/><itemref idref="c2"/></spine></package>')
    put(source, "OEBPS/style.css", 'body { font-family: "DejaVu Sans"; max-width: 44em; margin: 2em; font-size: 18px; } td,th { border: 1px solid black; padding: 0.5em; } img { width: 240px; height: 140px; } math { font-family: "DejaVu Sans"; font-size: 22px; }')
    put(source, "OEBPS/nav.xhtml", '<html xmlns="' + X + '" xmlns:epub="http://www.idpf.org/2007/ops"><head><title>Navigation</title></head><body><nav epub:type="toc"><ol><li><a href="chapter1.xhtml">Opening</a></li><li><a href="chapter2.xhtml">Results</a></li></ol></nav></body></html>')
    put(source, "OEBPS/figure-en.png", assets["figure-en.png"])
    for chapter in (1, 2):
        parts = ['<html xmlns="' + X + '" xmlns:epub="http://www.idpf.org/2007/ops"><head><title>Chapter ' + str(chapter) + '</title><link rel="stylesheet" href="style.css"/></head><body>']
        for b in (x for x in document["blocks"] if x["chapter"] == chapter):
            ident = b.get("note_id", b.get("anchor", b["block_id"]))
            b["locator"] = {"kind": "epub", "part": f"OEBPS/chapter{chapter}.xhtml", "id": ident}
            if b["type"] == "table":
                piece = '<table id="' + ident + '"><caption>' + esc(b["caption"]) + '</caption>'
                for ri, row in enumerate(b["cells"]):
                    tag = "th" if ri == 0 else "td"
                    piece += "<tr>" + "".join("<" + tag + ">" + esc(c) + "</" + tag + ">" for c in row) + "</tr>"
                piece += "</table>"
            elif b["type"] == "formula":
                piece = '<p id="' + ident + '"><math xmlns="' + MM + '">' + mathml(b["formulas"][0]["structure_expected"]) + '</math></p>'
            elif b["type"] == "figure":
                piece = '<figure id="' + ident + '"><img src="figure-en.png" alt="' + esc(b["caption"]) + '"/><figcaption>' + esc(b["caption"]) + '</figcaption></figure>'
            else:
                tag = "h1" if b["type"] == "heading" else "aside" if b["type"] == "note" else "p"
                piece = '<' + tag + ' id="' + ident + '"' + (' epub:type="footnote"' if b["type"] == "note" else '') + '>' + esc(b["text"])
                for ref in b.get("references", []):
                    attr = ' epub:type="noteref" id="call-n1"' if ref["kind"] == "note_call" else ''
                    piece += ' <a href="' + esc(ref["target"]) + '"' + attr + '>' + esc(ref["text"]) + '</a>'
                piece += '</' + tag + '>'
            parts.append(piece)
        parts.append('</body></html>')
        content = "\n".join(parts)
        html(content)
        put(source, f"OEBPS/chapter{chapter}.xhtml", content)
    return archive(root, source, "inputs/" + document["document_id"] + ".epub", first="mimetype")


def rtftext(text):
    out = []
    for c in text:
        if c in "\\{}":
            out.append("\\" + c)
        elif ord(c) > 127:
            n = ord(c)
            if n > 32767:
                n -= 65536
            out.append("\\u" + str(n) + "?")
        else:
            out.append(c)
    return "".join(out)


def rtf(root, document):
    source = root / "sources" / document["document_id"]
    source.mkdir()
    chunks = [r'{\rtf1\ansi\ansicpg1252\uc1\deff0{\fonttbl{\f0 DejaVu Sans;}}\paperw11906\paperh16838\margl800\margr800\margt800\margb800\f0\fs18' + '\n']
    note = next(b for b in document["blocks"] if b["type"] == "note")
    for b in document["blocks"]:
        if b["type"] == "note":
            continue
        start = len("".join(chunks).encode("ascii"))
        if b["type"] == "table":
            content = ""
            for row in b["cells"]:
                content += r'\pard\trowd\cellx2800\cellx5600\cellx8400 ' + "".join(r'\intbl ' + rtftext(c) + r'\cell ' for c in row) + r'\row' + '\n'
            content += r'\pard ' + rtftext(b.get("caption", "")) + r'\par' + '\n'
        else:
            content = (r'\b\fs26 ' if b["type"] == "heading" else '') + rtftext(b["text"]) + (r'\b0\fs18 ' if b["type"] == "heading" else '')
            for ref in b.get("references", []):
                if ref["kind"] == "note_call":
                    content += r' {\super\chftn}{\footnote\pard\plain\f0\fs16 ' + rtftext(note["text"]) + '}'
                    ns = start + content.index(r'{\footnote')
                    note["locator"] = {"kind": "rtf", "start": ns, "length": len(content[content.index(r'{\footnote'):].encode("ascii"))}
                else:
                    content += r' {\field{\*\fldinst HYPERLINK "' + ref["target"] + r'"}{\fldrslt ' + rtftext(ref["text"]) + '}}'
            content += r'\par' + '\n'
        chunks.append(content)
        b["locator"] = {"kind": "rtf", "start": start, "length": len(content.encode("ascii"))}
    chunks.append('}\n')
    content = "".join(chunks)
    # Intentional group-scoped uc2 witness, preserving the visible accent.
    content = content.replace(r'\u233?', r'{\uc2\u233??}')
    # Recompute locators after the explicitly authored fallback variant.
    # Each block range is selected using its original fragment with same variant.
    actual = content.encode("ascii")
    for b in document["blocks"]:
        old = b["locator"]
        if b["type"] == "note":
            fragment = r'{\footnote\pard\plain\f0\fs16 ' + rtftext(b["text"]) + '}'
        else:
            fragment = "".join(chunks)[old["start"]:old["start"] + old["length"]]
        fragment = fragment.replace(r'\u233?', r'{\uc2\u233??}').encode("ascii")
        b["locator"] = {"kind": "rtf", "start": actual.index(fragment), "length": len(fragment)}
    put(source, "source.rtf", actual)
    return put(root, "inputs/" + document["document_id"] + ".rtf", actual)


def tex(root, document, assets):
    source = root / "sources" / document["document_id"]
    source.mkdir()
    put(source, "assets/figure-en.png", assets["figure-en.png"])
    mainhead = '\\documentclass{article}\n\\usepackage{amsmath}\n\\usepackage{graphicx}\n\\usepackage{hyperref}\n\\begin{document}\n'
    for unit, name in [("main", "main.tex"), ("one", "sections/one.tex"), ("two", "sections/two.tex")]:
        parts = [mainhead] if unit == "main" else []
        for b in (x for x in document["blocks"] if x["unit"] == unit):
            start = len("".join(parts).encode())
            if b["type"] == "heading":
                content = '\\section{' + b["text"] + '}\\label{' + b["anchor"] + '}\n'
            elif b["type"] == "formula":
                f = b["formulas"][0]
                content = ('$' + f["reference_latex"] + '$\n') if f["formula_id"] == "energy" else '\\[\n' + f["reference_latex"] + '\n\\]\n'
            elif b["type"] == "table":
                content = '\\begin{table}\n\\begin{tabular}{lll}\n' + '\\\\\n'.join(' & '.join(c for c in row) for row in b["cells"]) + '\n\\end{tabular}\n\\caption{' + b["caption"] + '}\n\\end{table}\n'
            elif b["type"] == "figure":
                content = '\\begin{figure}\n\\includegraphics{assets/figure-en.png}\n\\caption{' + b["caption"] + '}\\label{fig1}\n\\end{figure}\n'
            elif b["type"] == "bibliography":
                content = '\\begin{thebibliography}{1}\n\\bibitem{bib1}' + b["text"] + '\n\\end{thebibliography}\n'
            else:
                content = b["text"]
                for ref in b.get("references", []):
                    if ref["kind"] == "ref":
                        content += ' \\ref{' + ref["target"] + '}'
                    elif ref["kind"] == "citation":
                        content += ' \\cite{' + ref["target"] + '}'
                    else:
                        # \& escaped within target; source checker decodes one explicit escape.
                        content += ' \\href{' + ref["target"].replace('&', r'\&') + '}{' + ref["text"] + '}'
                content += '\n'
            b["locator"] = {"kind": "tex", "file": name, "start": start, "length": len(content.encode())}
            parts.append(content + '\n')
        if unit == "main":
            parts += ['\\input{sections/one.tex}\n', '\\include{sections/two.tex}\n', '\\end{document}\n']
        put(source, name, ''.join(parts))
    inputroot = root / "inputs" / document["document_id"]
    shutil.copytree(source, inputroot)
    return inputroot / "main.tex"


def text_svg(text, x, y, size=11, width=500, weight=None):
    # Fixed code-point wrap, not metadata stripping; no content removed.
    lines = textwrap.wrap(text, width=max(12, int(width / (size * 0.6))), replace_whitespace=False, drop_whitespace=True) or [""]
    attrs = ' font-weight="bold"' if weight else ''
    result = []
    for line in lines:
        result.append(f'<text x="{x}" y="{y}" font-family="DejaVu Sans Mono" font-size="{size}"{attrs}>{esc(line)}</text>')
        y += size * 1.5
    return ''.join(result), y, lines


def pdf_page(root, document, page, blocks, suffix=""):
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="595pt" height="842pt" viewBox="0 0 595 842"><rect x="0" y="0" width="595" height="842" fill="white"/>']
    columns = document["pdf_class"] == "native-two-columns"
    positions = {"left": 88, "right": 88, "full": 40}
    sourcepath = root / "sources" / document["document_id"]
    sourcepath.mkdir(exist_ok=True)
    for b in blocks:
        column = b.get("column", "left") if columns else "left"
        x, width = (315, 245) if columns and column == "right" else (35, 245) if columns and column == "left" else (35, 520)
        y = positions[column]
        begin = y - 14
        encoded = []
        if b["type"] == "formula":
            f = b["formulas"][0]
            lines = f["visible_lines"]
            if f["formula_id"] == "fraction":
                a, _, _ = text_svg(lines[0], x + 25, y, size=14, width=width)
                parts += [a, f'<line x1="{x+15}" y1="{y+6}" x2="{x+98}" y2="{y+6}" stroke="black" stroke-width="1"/>']
                a, _, _ = text_svg(lines[1], x + 47, y + 25, size=14, width=width)
                parts.append(a)
                y += 46
            else:
                for line in lines:
                    a, y, _ = text_svg(line, x + 16, y, size=13, width=width)
                    parts.append(a)
                if f["formula_id"] == "matrix":
                    # Full matrix delimiters drawn, native cell text preserved.
                    top, bot = begin - 1, y - 8
                    for xx, toward in [(x+9, 5), (x+92, -5)]:
                        parts.append(f'<path d="M {xx+toward} {top} H {xx} V {bot} H {xx+toward}" fill="none" stroke="black"/>')
                y += 8
            encoded.extend(lines)
        elif b["type"] == "table":
            cw = width / 3
            for row in b["cells"]:
                for col, val in enumerate(row):
                    parts.append(f'<rect x="{x+col*cw}" y="{y-13}" width="{cw}" height="25" fill="white" stroke="black"/>')
                    a, _, _ = text_svg(val, x + col * cw + 5, y + 3, size=10, width=cw - 10)
                    parts.append(a)
                    if val:
                        encoded.append(val)
                y += 25
            a, y, lines = text_svg(b["caption"], x, y + 4, size=10, width=width)
            parts.append(a)
            encoded.extend(lines)
        elif b["type"] == "figure":
            asset = b["asset"]
            # Relative source asset verified before Inkscape sees it.
            path = sourcepath / asset
            if not path.exists():
                shutil.copyfile(root / "assets" / asset, path)
            parts.append(f'<image x="{x}" y="{y-10}" width="180" height="105" xlink:href="{asset}"/>')
            y += 105
            a, y, lines = text_svg(b["caption"], x, y, size=10, width=width)
            parts.append(a)
            encoded.extend(lines)
        else:
            value = b["text"]
            # Reference strings are printed in full, not embedded active URI actions.
            for ref in b.get("references", []):
                value += " " + ref["target"]
            a, y, lines = text_svg(value, x, y, size=15 if b["type"] == "heading" else 10, width=width, weight=b["type"] == "heading")
            parts.append(a)
            encoded.extend(lines)
        y += 18
        positions[column] = y
        if y > 806:
            raise ValueError("Layout overflow: " + document["document_id"] + " " + b["block_id"])
        b["locator"] = {"kind": "pdf", "page": page, "bbox": [x - 3, begin - 3, x + width + 3, y - 18], "encoded_lines": encoded,
                        "native": b.get("native", document["pdf_class"] != "image-only")}
        if columns and column == "full":
            positions["left"] = positions["right"] = max(y + 10, 88)
    parts.append('</svg>')
    content = ''.join(parts)
    svg(content.encode())
    return put(sourcepath, f"page-{page}{suffix}.svg", content)


class Tools:
    def __init__(self, root, environment):
        self.root, self.env, self.records = root, environment, []

    def call(self, argv, cwd=None, timeout=120):
        ident = f"tool-{len(self.records)+1:03d}"
        rec = run(argv, cwd or self.root, self.root / "tools" / ident, self.env, timeout)
        self.records.append(rec)
        write_new(self.root / "tools" / (ident + ".json"), rec)
        if rec["adapter_exit"] or rec["stop_reason"]:
            raise RuntimeError("Tool fallito: " + ident + " " + repr(argv))
        return rec


def pdf(root, document, tools):
    paths = []
    for page in range(1, document["pages"] + 1):
        blocks = [b for b in document["blocks"] if b["page"] == page]
        path = pdf_page(root, document, page, blocks)
        native = root / "derivations" / f'{document["document_id"]}-{page}-native.pdf'
        tools.call(["/usr/bin/inkscape", str(path), "--export-type=pdf", "--export-filename=" + str(native)])
        image_only = document["pdf_class"] == "image-only" or (document["document_id"] == "C10-pdf-mixed-it-en" and page == 2)
        if image_only:
            prefix = root / "derivations" / f'{document["document_id"]}-{page}-parent'
            tools.call(["/usr/bin/pdftoppm", "-singlefile", "-r", "180", "-png", str(native), str(prefix)])
            image = prefix.with_suffix(".png")
            png(image.read_bytes())
            script = put(root, "derivations/" + document["document_id"] + f"-{page}-image.txt", f'%%MediaBox 0 0 595 842\n%%Image Im {image}\nq 595 0 0 842 0 0 cm /Im Do Q\n')
            rasterpdf = root / "derivations" / f'{document["document_id"]}-{page}-raster.pdf'
            tools.call(["/usr/bin/mutool", "create", "-o", str(rasterpdf), str(script)])
            paths.append(rasterpdf)
            for b in blocks:
                b["locator"]["native"] = False
                b["locator"]["raster_parent"] = str(image.relative_to(root))
                b["locator"]["raster_parent_sha256"] = digest_bytes(image.read_bytes())
                b["locator"]["progenitor_pdf"] = str(native.relative_to(root))
        elif document["document_id"] == "C10-pdf-mixed-it-en" and page == 3:
            # Generate a raster of actual lower-region content, composite into
            # SVG together with only the upper native blocks. No hidden overlay.
            prefix = root / "derivations" / "C10-p3-full"
            tools.call(["/usr/bin/pdftoppm", "-singlefile", "-r", "180", "-png", str(native), str(prefix)])
            lower = [b for b in blocks if not b["native"]]
            upper = [b for b in blocks if b["native"]]
            boundary = min(b["locator"]["bbox"][1] for b in lower) - 3
            # Use PNG full image clipped to lower bbox with an SVG clipPath.
            # Its parent contains native text but canonical p3 contains only
            # upper native plus this raster. Glyph encodings below are N/A.
            image = prefix.with_suffix(".png")
            parentdata = image.read_bytes()
            imagecopy = root / "sources" / document["document_id"] / "p3-raster.png"
            put(root, str(imagecopy.relative_to(root)), parentdata)
            upper_svg = pdf_page(root, document, page, upper, suffix="-upper")
            content = upper_svg.read_text().removesuffix('</svg>')
            content += f'<defs><clipPath id="lower"><rect x="0" y="{boundary}" width="595" height="{842-boundary}"/></clipPath></defs><image x="0" y="0" width="595" height="842" clip-path="none" href="p3-raster.png"/>'
            # SVG validator forbids arbitrary URL attributes; use explicit
            # viewport nesting via a cropped PNG instead below.
            # Cropping is a deterministic PNG byte operation on synthetic asset,
            # not an AI image edit. The final source embeds just lower pixels.
            from png_codec import crop
            w, h = png(parentdata)
            cropped = crop(parentdata, 0, int(boundary*h/842), w, h)
            put(root, "sources/" + document["document_id"] + "/p3-lower.png", cropped)
            actual_height = png(cropped)[1] * 842 / h
            content = upper_svg.read_text().removesuffix('</svg>') + f'<image x="0" y="{842-actual_height}" width="595" height="{actual_height}" href="p3-lower.png"/></svg>'
            mixedsvg = put(root, "sources/" + document["document_id"] + "/page-3-mixed.svg", content)
            svg(content.encode())
            mixed = root / "derivations" / "C10-3-mixed.pdf"
            tools.call(["/usr/bin/inkscape", str(mixedsvg), "--export-type=pdf", "--export-filename=" + str(mixed)])
            paths.append(mixed)
            for b in lower:
                b["locator"]["raster_parent"] = str(imagecopy.relative_to(root))
                b["locator"]["raster_parent_sha256"] = digest_bytes(parentdata)
                b["locator"]["progenitor_pdf"] = str(native.relative_to(root))
        else:
            paths.append(native)
    output = root / "inputs" / (document["document_id"] + ".pdf")
    tools.call(["/usr/bin/mutool", "merge", "-o", str(output), *map(str, paths)])
    return output


def render(root, document, inputpath, tools):
    docid = document["document_id"]
    renderroot = root / "renders" / docid
    renderroot.mkdir()
    if document["format"] in ("docx", "rtf"):
        profile = root / "lo-profile" / docid
        tools.call(["/usr/bin/libreoffice", "-env:UserInstallation=" + profile.as_uri(), "--headless", "--convert-to", "pdf:writer_pdf_Export", "--outdir", str(renderroot), str(inputpath)])
        pdfpath = renderroot / (inputpath.stem + ".pdf")
    elif document["format"] == "pdf":
        pdfpath = inputpath
    else:
        return
    rec = tools.call(["/usr/bin/pdfinfo", str(pdfpath)])
    text = (root / "tools" / f'tool-{len(tools.records):03d}' / "stdout.bin").read_text()
    actual = int(next(line.split(":", 1)[1].strip() for line in text.splitlines() if line.startswith("Pages:")))
    if actual != document["pages"]:
        raise ValueError(f"Pagine reali {actual}, attese {document['pages']}: {docid}")
    for page in range(1, actual + 1):
        tools.call(["/usr/bin/pdftoppm", "-f", str(page), "-l", str(page), "-singlefile", "-r", "180", "-png", str(pdfpath), str(renderroot / f"page-{page}")])
        png(regular(renderroot, f"page-{page}.png"))
    tools.call(["/usr/bin/pdffonts", str(pdfpath)])
    document["renders"] = [descriptor(renderroot / f"page-{p}.png") for p in range(1, actual + 1)]
    document["render_pdf"] = str(pdfpath.relative_to(root))


def transcript(gold):
    """Complete annotation transcript: caption follows its cells, once."""
    values=[]
    for b in gold['blocks']:
        values.append(f"[{b['block_id']}; pagina={b.get('page','N/A')}; tipo={b['type']}]\n"+b['text'])
        values.extend(f['reference_latex'] for f in b.get('formulas',[]))
        values.extend(' | '.join(row) for row in b.get('cells',[]))
        if b['type']=='table' and b.get('caption') is not None:values.append(b['caption'])
        values.extend(r['kind']+': '+r['text']+' → '+r['target'] for r in b.get('references',[]))
    return '\n\n'.join(values)+'\n'


def generate(spec_path, output, environment):
    output = Path(output).absolute()
    output.mkdir(exist_ok=False)
    for d in ("sources", "inputs", "assets", "gold", "derivations", "tools", "renders"):
        (output / d).mkdir()
    spec = load(spec_path)
    put(output, "sources/documents.json", Path(spec_path).read_bytes())
    assets = {"figure-" + lang + ".png": figure(lang) for lang in ("it", "en")}
    for name, content in assets.items():
        put(output, "assets/" + name, content)
    tools = Tools(output, environment)
    documents = copy.deepcopy(spec["documents"])
    for d in documents:
        kind = d["format"]
        if kind == "txt":
            source = put(output, "sources/" + d["document_id"] + "/source.txt", d["blocks"][0]["text"])
            inputpath = put(output, "inputs/" + d["document_id"] + ".txt", source.read_bytes())
            d["blocks"][0]["locator"] = {"kind": "txt", "start": 0, "length": source.stat().st_size}
        elif kind == "docx":
            inputpath = docx(output, d, assets)
        elif kind == "epub":
            inputpath = epub(output, d, assets)
        elif kind == "rtf":
            inputpath = rtf(output, d)
        elif kind == "tex":
            inputpath = tex(output, d, assets)
        else:
            inputpath = pdf(output, d, tools)
        d["input"] = str(inputpath.relative_to(output))
        d["input_sha256"] = digest_bytes(inputpath.read_bytes())
        render(output, d, inputpath, tools)
        gold = {"schema_version": 1, "document_id": d["document_id"], "canonical_sha256": d["input_sha256"], "authoring_method": spec["authoring_method"], "note_order_policy": "definitions-collected-at-end", "blocks": d["blocks"], "verification": {"status": "PROVISIONAL", "M": "PENDING", "V": "PENDING", "kind": "human_visual", "verifier": None}, "transcription_convention": d.get("transcription_convention")}
        gold["denominators"] = counts(gold)
        for b in gold["blocks"]:
            native = b.get("locator", {}).get("native", True)
            b["properties"] = [{"property_id": b["block_id"] + ":content", "evidence_kind": "encoded" if native else "visible", "m_required": True, "v_required": True, "observable_in_canonical": native, "reason": None if native else "Raster: code points not encoded; M checks image provenance; V visible transcription"}]
        goldpath = output / "gold" / d["document_id"]
        goldpath.mkdir()
        write_new(goldpath / "annotations.json", gold)
        put(output, "gold/" + d["document_id"] + "/transcript.txt", transcript(gold))
        d["gold"] = str((goldpath / "annotations.json").relative_to(output))
        d["gold_sha256"] = digest_bytes((goldpath / "annotations.json").read_bytes())
    write_new(output / "generation.json", {"schema_version": 1, "spec": descriptor(spec_path), "generator": descriptor(__file__), "documents": documents, "tool_calls": tools.records, "environment": environment})
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--spec", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--environment", required=True, type=Path)
    args = p.parse_args()
    env = load(args.environment)
    if "environment" in env:
        env = env["environment"]
    generate(args.spec, args.output, env)
    print("GENERATION_COMPLETE " + str(args.output))


if __name__ == "__main__":
    main()
