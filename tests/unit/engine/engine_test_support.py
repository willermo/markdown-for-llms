"""Helpers of the engine tests. Engine tests never skip (`scripts/bench_junit_guard.py`).

Tests that run converters run them for real in the toolchain image, through the runner of the
engine; `require_toolchain` makes such a test fail, saying how to build the image, when it is
missing. Every input is synthetic, built here in pure Python or derived from the redistributable
fixtures of `tests/fixtures/conversion-corpus/` (provenance in that directory).
"""
import base64
import io
import json
from pathlib import Path
import struct
import tempfile
import zipfile
import zlib

from markdown_for_llms.engine.codes import EngineError

ROOT = Path(__file__).resolve().parents[3]
FIXTURES = ROOT / "tests" / "fixtures" / "conversion-corpus" / "inputs"
# A two-page DjVu with a text layer ("Alpha Beta") on page 1 only: the 64x32 bilevel page of the
# bench self-test (`bench/selftest.py`, SYNTHETIC_DJVU) twice, bundled with `djvm -c` and given
# its text with `djvused set-txt` in the toolchain image (DjVuLibre 3.5.30).
TEXT_DJVU = base64.b64decode(
    "QVQmVEZPUk0AAAEIREpWTURJUk0AAAAmgQACAAAAPgAAAMb//+W/ih+yh5Drxbc8XkJp2/Vt6f6KADroFH9GT1JNAAAAgERKVlVJTkZP"
    "AAAACgBAACAYAEgAFgFTamJ6AAAAKIH631lszsJ3/I4h5YywpV4o9df/6r83WlZU3wVbFiLIUHfMQAlJ/ydUWFR6AAAAMv//qohkQEch"
    "BsQLLrFwlivEELQbkCWubQNas20PaTS/DIchEL7WthWWspsDsh662LgbRk9STQAAAEZESlZVSU5GTwAAAAoAQAAgGABIABYBU2piegAA"
    "ACiB+t9ZbM7Cd/yOIeWMsKVeKPXX/+q/N1pWVN8FWxYiyFB3zEAJSf8n")
R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
CT = "http://schemas.openxmlformats.org/package/2006/content-types"
PR = "http://schemas.openxmlformats.org/package/2006/relationships"
W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
WP = "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"
A = "http://schemas.openxmlformats.org/drawingml/2006/main"
PIC = "http://schemas.openxmlformats.org/drawingml/2006/picture"


def require_toolchain():
    """ID of the verified toolchain image, or a failure (never a skip) that says how to build it."""
    from markdown_for_llms.engine import toolchain
    try:
        return toolchain.image()
    except EngineError as exc:
        raise AssertionError(f"toolchain image: {exc.code.value} ({exc.detail}); build it with "
                             "`mdf prepare --toolchain` at the root of the checkout") from None


def png(width=8, height=6, rgb=(200, 30, 30)):
    """A small valid RGB PNG (pure Python)."""
    raw = b"".join(b"\x00" + bytes(rgb) * width for _ in range(height))

    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


def write_zip(path, members, stored_first=None):
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        if stored_first:
            z.writestr(zipfile.ZipInfo(stored_first[0]), stored_first[1], compress_type=zipfile.ZIP_STORED)
        for name, data in members.items():
            z.writestr(name, data)
    return Path(path)


def drawing(rid, prefixes=("wp", "a", "pic"), name="Figure one"):
    wp, a, pic = prefixes
    return (f'<w:drawing><{wp}:inline><{wp}:extent cx="76200" cy="57150"/><{wp}:docPr id="1" name="{name}" descr="{name}"/>'
            f'<{a}:graphic><{a}:graphicData uri="{PIC}"><{pic}:pic><{pic}:nvPicPr><{pic}:cNvPr id="1" name="{name}"/>'
            f'<{pic}:cNvPicPr/></{pic}:nvPicPr><{pic}:blipFill><{a}:blip r:embed="{rid}"/></{pic}:blipFill>'
            f'<{pic}:spPr/></{pic}:pic></{a}:graphicData></{a}:graphic></{wp}:inline></w:drawing>')


def make_docx(path, body, media=None, rels=None, root_ns=None, extra=None, stored_first=None):
    """A minimal docx: `body` is the inner XML of w:body; `media` {name: bytes} under word/media/,
    `rels` [(rId, type suffix, target)] of the document part; `root_ns` the namespace
    declarations of the root (default: w, r, wp, a, pic); `stored_first` (name, bytes) a member
    written stored before all the others."""
    media = media or {}
    rels = rels if rels is not None else [(f"rId{i}", "image", "media/" + n) for i, n in enumerate(media, 1)]
    decl = root_ns or f'xmlns:w="{W}" xmlns:r="{R}" xmlns:wp="{WP}" xmlns:a="{A}" xmlns:pic="{PIC}"'
    members = {
        "[Content_Types].xml": f'<?xml version="1.0" encoding="UTF-8"?><Types xmlns="{CT}">'
                               '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
                               '<Default Extension="xml" ContentType="application/xml"/><Default Extension="png" ContentType="image/png"/>'
                               '<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/></Types>',
        "_rels/.rels": f'<?xml version="1.0" encoding="UTF-8"?><Relationships xmlns="{PR}"><Relationship Id="rId1" '
                       f'Type="{R}/officeDocument" Target="word/document.xml"/></Relationships>',
        "word/document.xml": f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:document {decl}><w:body>{body}</w:body></w:document>',
        "word/_rels/document.xml.rels": f'<?xml version="1.0" encoding="UTF-8"?><Relationships xmlns="{PR}">'
                                        + "".join(f'<Relationship Id="{i}" Type="{R}/{t}" Target="{g}"/>' for i, t, g in rels)
                                        + "</Relationships>",
    }
    for name, data in media.items():
        members["word/media/" + name] = data
    members.update(extra or {})
    return write_zip(path, members, stored_first=stored_first)


def para(text):
    return f"<w:p><w:r><w:t>{text}</w:t></w:r></w:p>"


SSML = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
XDR = "http://schemas.openxmlformats.org/drawingml/2006/spreadsheetDrawing"


def make_xlsx(path, image=True, drawing_used=True, sheet_ns=None):
    """A one-sheet xlsx with "Hello" in A1 and, with `image`, a picture anchored at A4 by a drawing
    that the sheet uses (`drawing_used`) or only relates to."""
    drawing_ref = '<drawing r:id="rId1"/>' if drawing_used else ""
    ns = sheet_ns or f'xmlns="{SSML}" xmlns:r="{R}"'
    members = {
        "[Content_Types].xml": f'<?xml version="1.0" encoding="UTF-8"?><Types xmlns="{CT}">'
                               '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
                               '<Default Extension="xml" ContentType="application/xml"/><Default Extension="png" ContentType="image/png"/>'
                               '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
                               '<Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
                               '<Override PartName="/xl/drawings/drawing1.xml" ContentType="application/vnd.openxmlformats-officedocument.drawing+xml"/>'
                               '</Types>',
        "_rels/.rels": f'<?xml version="1.0" encoding="UTF-8"?><Relationships xmlns="{PR}"><Relationship Id="rId1" '
                       f'Type="{R}/officeDocument" Target="xl/workbook.xml"/></Relationships>',
        "xl/workbook.xml": f'<?xml version="1.0" encoding="UTF-8"?><workbook xmlns="{SSML}" xmlns:r="{R}"><sheets>'
                           '<sheet name="Synthetic" sheetId="1" r:id="rId1"/></sheets></workbook>',
        "xl/_rels/workbook.xml.rels": f'<?xml version="1.0" encoding="UTF-8"?><Relationships xmlns="{PR}">'
                                      f'<Relationship Id="rId1" Type="{R}/worksheet" Target="worksheets/sheet1.xml"/></Relationships>',
        "xl/worksheets/sheet1.xml": f'<?xml version="1.0" encoding="UTF-8"?><worksheet {ns}><sheetData>'
                                    '<row r="1"><c r="A1" t="inlineStr"><is><t>Hello</t></is></c></row></sheetData>'
                                    f'{drawing_ref}</worksheet>',
    }
    if image:
        members.update({
            "xl/worksheets/_rels/sheet1.xml.rels": f'<?xml version="1.0" encoding="UTF-8"?><Relationships xmlns="{PR}">'
                                                   f'<Relationship Id="rId1" Type="{R}/drawing" Target="../drawings/drawing1.xml"/></Relationships>',
            "xl/drawings/drawing1.xml": f'<?xml version="1.0" encoding="UTF-8"?><xdr:wsDr xmlns:xdr="{XDR}" xmlns:a="{A}" xmlns:r="{R}">'
                                        '<xdr:oneCellAnchor><xdr:from><xdr:col>0</xdr:col><xdr:colOff>0</xdr:colOff><xdr:row>3</xdr:row>'
                                        '<xdr:rowOff>0</xdr:rowOff></xdr:from><xdr:ext cx="76200" cy="57150"/><xdr:pic><xdr:nvPicPr>'
                                        '<xdr:cNvPr id="2" name="Picture 1" descr="Synthetic figure"/><xdr:cNvPicPr/></xdr:nvPicPr>'
                                        '<xdr:blipFill><a:blip r:embed="rId1"/></xdr:blipFill><xdr:spPr/></xdr:pic><xdr:clientData/>'
                                        '</xdr:oneCellAnchor></xdr:wsDr>',
            "xl/drawings/_rels/drawing1.xml.rels": f'<?xml version="1.0" encoding="UTF-8"?><Relationships xmlns="{PR}">'
                                                   f'<Relationship Id="rId1" Type="{R}/image" Target="../media/image1.png"/></Relationships>',
            "xl/media/image1.png": png(),
        })
    return write_zip(path, members)


ODF_NS = ('xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0" '
          'xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0" '
          'xmlns:draw="urn:oasis:names:tc:opendocument:xmlns:drawing:1.0" '
          'xmlns:svg="urn:oasis:names:tc:opendocument:xmlns:svg-compatible:1.0" '
          'xmlns:xlink="http://www.w3.org/1999/xlink" '
          'xmlns:presentation="urn:oasis:names:tc:opendocument:xmlns:presentation:1.0"')


def make_odt(path, text_box=False, fallback=False, empty_first=False):
    """An odt with one paragraph and one picture, directly in a frame or inside a text box; with
    `fallback`, the frame holds a second image after it (`Pictures/fallback.png`), as LibreOffice
    writes a PNG twin after an SVM picture; with `empty_first`, an image of zero bytes
    (`Pictures/zero.svm`) comes before the picture in its frame."""
    twin = '<draw:image xlink:href="Pictures/fallback.png"/>' if fallback else ""
    zero = '<draw:image xlink:href="Pictures/zero.svm"/>' if empty_first else ""
    image = (f'<draw:frame draw:name="F1" svg:width="2cm" svg:height="1.5cm">{zero}'
             f'<draw:image xlink:href="Pictures/fig.png"/>{twin}</draw:frame>')
    if text_box:      # as in D16: the picture frame directly inside the text box of another frame
        image = ('<draw:frame draw:name="T1" text:anchor-type="paragraph" svg:width="4cm"><draw:text-box>'
                 + image.replace('<draw:frame ', '<draw:frame text:anchor-type="frame" ')
                 + '<text:p>Caption in the box.</text:p></draw:text-box></draw:frame>')
    content = (f'<?xml version="1.0" encoding="UTF-8"?><office:document-content {ODF_NS} office:version="1.3">'
               f'<office:body><office:text><text:p>Odt paragraph text.</text:p><text:p>{image}</text:p>'
               '</office:text></office:body></office:document-content>')
    manifest = ('<?xml version="1.0" encoding="UTF-8"?><manifest:manifest xmlns:manifest="urn:oasis:names:tc:opendocument:xmlns:manifest:1.0" manifest:version="1.3">'
                '<manifest:file-entry manifest:full-path="/" manifest:media-type="application/vnd.oasis.opendocument.text"/>'
                '<manifest:file-entry manifest:full-path="content.xml" manifest:media-type="text/xml"/>'
                '<manifest:file-entry manifest:full-path="Pictures/fig.png" manifest:media-type="image/png"/></manifest:manifest>')
    styles = (f'<?xml version="1.0" encoding="UTF-8"?><office:document-styles {ODF_NS} office:version="1.3">'
              '<office:styles/></office:document-styles>')
    members = {"META-INF/manifest.xml": manifest, "content.xml": content, "styles.xml": styles, "Pictures/fig.png": png()}
    if fallback:
        members["Pictures/fallback.png"] = png(8, 6, (30, 200, 30))
    if empty_first:
        members["Pictures/zero.svm"] = b""
    return write_zip(path, members, stored_first=("mimetype", b"application/vnd.oasis.opendocument.text"))


CHART_NS = 'xmlns:chart="urn:oasis:names:tc:opendocument:xmlns:chart:1.0"'


def make_odt_nested(path, html_table, inner="member", after="After the table.", twin=None):
    """An odt whose table cell holds a text box with two picture frames: pandoc's ODT reader puts the
    second image inside the alternative text of the first (as in D16). With `html_table` a cell before
    it holds two paragraphs, so the writer emits the table as HTML, where alternative text is text.
    `inner`: the second image as the member `Pictures/b.png` ("member"), or its bytes inline in the
    content, as `office:binary-data` ("binary") or a `data:` URI ("data"); `after`: the paragraph
    after the table; `twin`: the bytes of the second image also as the member `Pictures/duplicate.png`,
    not referenced ("unused") or the second image of the frame of the first picture ("alternate")."""
    data = base64.b64encode(png(8, 6, (0, 0, 250))).decode()
    second = {"member": '<draw:image xlink:href="Pictures/b.png"/>',
              "binary": f'<draw:image><office:binary-data>{data}</office:binary-data></draw:image>',
              "data": f'<draw:image xlink:href="data:image/png;base64,{data}"/>'}[inner]
    alternate = '<draw:image xlink:href="Pictures/duplicate.png"/>' if twin == "alternate" else ""
    pictures = ('<draw:frame draw:name="A" svg:width="2cm" svg:height="1.5cm"><draw:image xlink:href="Pictures/a.png"/>'
                f'{alternate}'
                f'<svg:title>First picture</svg:title></draw:frame><text:tab/>'
                f'<draw:frame draw:name="B" svg:width="2cm" svg:height="1.5cm">{second}</draw:frame>')
    cells = ('<table:table-cell><text:p>First paragraph of a cell.</text:p><text:p>Second paragraph.</text:p></table:table-cell>'
             if html_table else "")
    cells += ('<table:table-cell><text:p><draw:frame draw:name="Box" text:anchor-type="paragraph" svg:width="8cm">'
              f'<draw:text-box><text:p>{pictures}</text:p></draw:text-box></draw:frame></text:p></table:table-cell>')
    content = (f'<?xml version="1.0" encoding="UTF-8"?><office:document-content {ODF_NS} {TABLE_NS} office:version="1.3">'
               '<office:body><office:text><text:p>Before the table.</text:p><table:table table:name="T1">'
               f'<table:table-column table:number-columns-repeated="{2 if html_table else 1}"/>'
               f'<table:table-row>{cells}</table:table-row></table:table>'
               f'<text:p>{after}</text:p></office:text></office:body></office:document-content>')
    styles = (f'<?xml version="1.0" encoding="UTF-8"?><office:document-styles {ODF_NS} office:version="1.3">'
              '<office:styles/></office:document-styles>')
    manifest = ('<?xml version="1.0" encoding="UTF-8"?><manifest:manifest xmlns:manifest="urn:oasis:names:tc:opendocument:xmlns:manifest:1.0" manifest:version="1.3">'
                '<manifest:file-entry manifest:full-path="/" manifest:media-type="application/vnd.oasis.opendocument.text"/></manifest:manifest>')
    members = {"META-INF/manifest.xml": manifest, "content.xml": content, "styles.xml": styles, "Pictures/a.png": png()}
    if inner == "member":
        members["Pictures/b.png"] = png(8, 6, (0, 0, 250))
    if twin:
        members["Pictures/duplicate.png"] = png(8, 6, (0, 0, 250))
    return write_zip(path, members, stored_first=("mimetype", b"application/vnd.oasis.opendocument.text"))


TABLE_NS = 'xmlns:table="urn:oasis:names:tc:opendocument:xmlns:table:1.0"'


def make_odt_chart(path, preview=None):
    """An odt whose content embeds a chart object (`Object 1`, an OpenDocument chart with its own
    content.xml) and, with `preview` (a member name), its cached preview image in the same frame:
    `ObjectReplacements/Object 1` as LibreOffice writes it, or a picture under `Pictures/`."""
    href = ("./" + preview if preview.startswith("ObjectReplacements/") else preview) if preview else None
    image = f'<draw:image xlink:href="{href}" xlink:type="simple"/>' if preview else ""
    content = (f'<?xml version="1.0" encoding="UTF-8"?><office:document-content {ODF_NS} office:version="1.3">'
               '<office:body><office:text><text:p>Text before the chart.</text:p><text:p>'
               '<draw:frame draw:name="Chart1" svg:width="8cm" svg:height="5cm">'
               f'<draw:object xlink:href="./Object 1" xlink:type="simple" xlink:show="embed"/>{image}</draw:frame></text:p>'
               '<text:p>Text after the chart.</text:p></office:text></office:body></office:document-content>')
    chart = (f'<?xml version="1.0" encoding="UTF-8"?><office:document-content {ODF_NS} {CHART_NS} office:version="1.3">'
             '<office:body><office:chart><chart:chart chart:class="chart:bar"><chart:plot-area/></chart:chart>'
             '</office:chart></office:body></office:document-content>')
    styles = (f'<?xml version="1.0" encoding="UTF-8"?><office:document-styles {ODF_NS} office:version="1.3">'
              '<office:styles/></office:document-styles>')
    manifest = ('<?xml version="1.0" encoding="UTF-8"?><manifest:manifest xmlns:manifest="urn:oasis:names:tc:opendocument:xmlns:manifest:1.0" manifest:version="1.3">'
                '<manifest:file-entry manifest:full-path="/" manifest:media-type="application/vnd.oasis.opendocument.text"/>'
                '<manifest:file-entry manifest:full-path="content.xml" manifest:media-type="text/xml"/>'
                '<manifest:file-entry manifest:full-path="Object 1/" manifest:media-type="application/vnd.oasis.opendocument.chart"/>'
                '<manifest:file-entry manifest:full-path="Object 1/content.xml" manifest:media-type="text/xml"/>'
                + (f'<manifest:file-entry manifest:full-path="{preview}" manifest:media-type="image/png"/>' if preview else "")
                + '</manifest:manifest>')
    members = {"META-INF/manifest.xml": manifest, "content.xml": content, "styles.xml": styles,
               "Object 1/content.xml": chart}
    if preview:
        members[preview] = png(16, 10, (10, 90, 160))
    return write_zip(path, members, stored_first=("mimetype", b"application/vnd.oasis.opendocument.text"))


def make_fodp(path):
    """A flat OpenDocument presentation (one slide, a title and an embedded picture) that
    LibreOffice turns into a pptx."""
    data = base64.b64encode(png(16, 12, (20, 120, 200))).decode()
    body = (f'<?xml version="1.0" encoding="UTF-8"?><office:document {ODF_NS} office:version="1.3" '
            'office:mimetype="application/vnd.oasis.opendocument.presentation"><office:body><office:presentation>'
            '<draw:page draw:name="page1"><draw:frame svg:x="1cm" svg:y="1cm" svg:width="20cm" svg:height="3cm">'
            '<draw:text-box><text:p>Slide title text</text:p></draw:text-box></draw:frame>'
            '<draw:frame svg:x="2cm" svg:y="5cm" svg:width="4cm" svg:height="3cm"><draw:image>'
            f'<office:binary-data>{data}</office:binary-data></draw:image></draw:frame></draw:page>'
            '</office:presentation></office:body></office:document>')
    Path(path).write_text(body, encoding="utf-8")
    return Path(path)


def derive_epub(path, chapter_edit, rename=None, edits=None):
    """C03 (fixture) with `chapter_edit(text) -> text` applied to OEBPS/chapter1.xhtml, `edits`
    {member: edit} to other text members and, with `rename` {old member: new member}, those
    members renamed (their references are changed only by the edits)."""
    src = FIXTURES / "C03-epub-en.epub"
    rename, edits = rename or {}, dict(edits or {})
    edits["OEBPS/chapter1.xhtml"] = chapter_edit
    with zipfile.ZipFile(src) as zin, zipfile.ZipFile(path, "w") as zout:
        for info in zin.infolist():
            data = zin.read(info.filename)
            if info.filename in edits:
                data = edits[info.filename](data.decode("utf-8")).encode("utf-8")
            if info.filename in rename:
                info = zipfile.ZipInfo(rename[info.filename], info.date_time)
                info.compress_type = zipfile.ZIP_DEFLATED
            zout.writestr(info, data)
    return Path(path)


def replace_figure(text, new):
    """The `figure` element of C03's chapter 1 replaced by `new`."""
    start = text.index('<figure id="b05">')
    end = text.index("</figure>", start) + len("</figure>")
    return text[:start] + new + text[end:]


def run_filter(test, text, reader="markdown"):
    """pandoc with the contract filter on `text` read by `reader`: (markdown, report)."""
    from markdown_for_llms.engine import tools
    from markdown_for_llms.engine.native import FILTER
    require_toolchain()
    tmp = tempfile.TemporaryDirectory()
    test.addCleanup(tmp.cleanup)
    root = Path(tmp.name)
    src = root / "source.txt"
    src.write_text(text, encoding="utf-8")
    tools.run("pandoc", ["-f", reader, "-t", "commonmark_x", "--wrap=none", "--extract-media=assets",
                         tools.In(FILTER, name="contract.lua", prefix="--lua-filter="),
                         "-o", tools.Out(root / "document.md", arg="document.md", allow_empty=True),
                         tools.Out(root / "assets", arg="assets", directory=True, optional=True, argument=False),
                         tools.Out(root / "pre-filter.md", arg="pre-filter.md", allow_empty=True, argument=False),
                         tools.Out(root / "report.json", arg="contract-report.json", argument=False),
                         tools.In(src)], root / "tools", timeout=120, memory_mb=1024)
    return (root / "document.md").read_text(encoding="utf-8"), json.loads((root / "report.json").read_text())


def inspect_out(out):
    """`references.inspect` of a published output, as the product reads it."""
    from markdown_for_llms import references
    out = Path(out)
    present = {p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file()}
    return references.inspect((out / "document.md").read_text(encoding="utf-8"), present)


def zip_bytes(members):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in members.items():
            z.writestr(name, data)
    return buf.getvalue()


# --- synthetic PDFs of the pages family (MFL032 tranche B) ---------------------------------

def checker_rgb(w, h, cell=10):
    """A grey checkerboard, RGB bytes."""
    return bytes(v for y in range(h) for x in range(w) for v in ((40,) * 3 if (x // cell + y // cell) % 2 == 0 else (220,) * 3))


def scan_of(text_lines, scale=2.0):
    """(w, h, RGB bytes) of the rendering of a native page with `text_lines`: the image of a scan
    whose text OCR can read."""
    import pypdfium2 as pdfium
    content = b"BT /F1 20 Tf 60 760 Td 26 TL " + b" ".join(b"(" + line.encode("latin-1") + b") '" for line in text_lines) + b" ET"
    doc = pdfium.PdfDocument(make_pdf([{"content": content}]))
    try:
        bitmap = doc[0].render(scale=scale, rev_byteorder=True)
        img = bitmap.to_pil().convert("RGB")
    finally:
        doc.close()
    return img.width, img.height, img.tobytes()


def make_pdf(pages, font="Helvetica"):
    """A PDF of `pages`, each a dict: `content` (page content stream, bytes; /F1 is the font,
    /Im1 the image, /Fm1 the form), `image` ((w, h, RGB bytes) drawn by /Im1), `form` (content
    of a Form XObject /Fm1 over the whole page, which may draw /Im1 and /F1), `images` (more
    images {name: (w, h, RGB bytes)}), `mediabox`,
    `cropbox`, `rotate`. The font is a Type1 font named `font`."""
    objs = []

    def add(body):
        objs.append(body)
        return len(objs)
    f1 = add(b"<< /Type /Font /Subtype /Type1 /BaseFont /" + font.encode() + b" >>")
    kids = []
    pages_id = None
    deferred = []
    for page in pages:
        xobjects = {}
        images = dict(page.get("images") or {})
        if page.get("image"):
            images["Im1"] = page["image"]
        for name, (w, h, rgb) in images.items():
            data = zlib.compress(rgb)
            xobjects[name] = add(b"<< /Type /XObject /Subtype /Image /Width %d /Height %d /ColorSpace /DeviceRGB "
                                 b"/BitsPerComponent 8 /Filter /FlateDecode /Length %d >>\nstream\n" % (w, h, len(data))
                                 + data + b"\nendstream")
        box = page.get("mediabox", (0, 0, 595, 842))
        if page.get("form") is not None:
            fs = page["form"]
            res = b"<< /Font << /F1 %d 0 R >>" % f1 + (b" /XObject << /Im1 %d 0 R >>" % xobjects["Im1"] if "Im1" in xobjects else b"") + b" >>"
            xobjects["Fm1"] = add(b"<< /Type /XObject /Subtype /Form /BBox [%d %d %d %d] /Resources " % tuple(box) + res
                                  + b" /Length %d >>\nstream\n" % len(fs) + fs + b"\nendstream")
        content = page.get("content", b"")
        k = add(b"<< /Length %d >>\nstream\n" % len(content) + content + b"\nendstream")
        xo = b" ".join(b"/%s %d 0 R" % (n.encode(), i) for n, i in xobjects.items())
        extra = b""
        if page.get("cropbox"):
            extra += b" /CropBox [%d %d %d %d]" % tuple(page["cropbox"])
        if page.get("rotate"):
            extra += b" /Rotate %d" % page["rotate"]
        deferred.append((k, xo, box, extra))
    pages_id = len(objs) + len(deferred) + 1
    for k, xo, box, extra in deferred:
        kids.append(add(b"<< /Type /Page /Parent %d 0 R /MediaBox [%d %d %d %d]" % ((pages_id,) + tuple(box)) + extra
                        + b" /Resources << /Font << /F1 %d 0 R >> /XObject << " % f1 + xo + b" >> >> /Contents %d 0 R >>" % k))
    assert add(b"<< /Type /Pages /Kids [" + b" ".join(b"%d 0 R" % i for i in kids) + b"] /Count %d >>" % len(kids)) == pages_id
    catalog = add(b"<< /Type /Catalog /Pages %d 0 R >>" % pages_id)
    buf = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, body in enumerate(objs, 1):
        offsets.append(len(buf))
        buf += b"%d 0 obj\n" % i + body + b"\nendobj\n"
    xref = len(buf)
    buf += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objs) + 1) + b"".join(b"%010d 00000 n \n" % o for o in offsets)
    buf += b"trailer\n<< /Size %d /Root %d 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objs) + 1, catalog, xref)
    return bytes(buf)


NATIVE_TEXT = b"BT /F1 14 Tf 72 760 Td (Native text of the synthetic page, with a figure below.) Tj ET"
FIGURE = b"q 200 0 0 150 72 500 cm /Im1 Do Q"
SCAN = b"q 595 0 0 842 0 0 cm /Im1 Do Q"
INVISIBLE = b"BT 3 Tr /F1 12 Tf 72 700 Td (INVISIBLE OCR LAYER TEXT OF THE SCANNED PAGE) Tj ET"
GLYPHLESS_MODE0 = b"BT 0 Tr /F1 12 Tf 72 700 Td (GLYPHLESS OCR LAYER TEXT OF THE SCANNED PAGE) Tj ET"
