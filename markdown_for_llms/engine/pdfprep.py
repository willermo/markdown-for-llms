"""PDF preparation of the Marker and Datalab routes, run in the light child (`isolation.py`).

Commands (`main(args)`; inputs read-only under /in, results under /scratch):

    classify MAX_PAGES        /in/source.pdf -> /scratch/pdfprep.json
    strip PAGE[,PAGE...]      /in/source.pdf -> /scratch/stripped.pdf and /scratch/strip.json
    image FORMAT MAX_PAGES MAX_PIXELS   /in/source.<ext> -> /scratch/source.pdf

`classify` gives, per page: the class (`native`: visible text; `ocr_layer`: an image over at
least 65 % of the page with only invisible text; `image`: no text), the counts behind it, the
rotation and every raster image object drawn on the page (D14 of MFL032: all of them, before
any threshold), each with its box projected to the page as rendered (top-left origin, in
points of the rendered page: `FPDF_PageToDevice`, so a translated MediaBox or CropBox moves the
box with the page), its size in pixels and the digest of its stored data (identical images
share it).

Text counts only non-space characters whose box meets the page (text outside the CropBox does
not make an empty page `native`). Invisible text is render mode 3, and also the glyphless OCR
layers drawn in mode 0: a font named `GlyphlessFont` or glyphs without a single path segment
(PB6). Objects inside Form XObjects are walked with the matrices of their forms.

`strip` removes from the given pages (the `ocr_layer` ones) the invisible text objects and the
top-level Form XObjects that hold, recursively, only invisible text, leaves forms that mix it
with anything else, saves, classifies the saved copy again and compares the rendering of every
touched page with pdfium (the renderer of Marker and Indra) before and after.

`image` makes a PDF of a png, jpeg or tiff: the JPEG stream embedded as it is, PNG and every
TIFF frame as a lossless bitmap, one page per frame, the page size from the image resolution.
"""
import ctypes
import hashlib
import json
from pathlib import Path
import sys

SCRATCH = Path("/scratch")
SOURCE = Path("/in/source.pdf")
OCR_LAYER_COVER = 0.65
FORM_DEPTH = 8
RASTER_HASH_LIMIT = 64 << 20         # bigger stored images get no digest (never cornice)
DEVICE_SCALE = 8                     # FPDF_PageToDevice works in integers: 1/8 point
DEFAULT_DPI, MIN_DPI, MAX_DPI = 200, 50, 1200


def _raw():
    import pypdfium2 as pdfium
    import pypdfium2.raw as raw
    return pdfium, raw


def walk(page):
    """(object, matrices of the enclosing forms, innermost first, depth) for every object."""
    pdfium, raw = _raw()

    def rec(objs, mats, depth):
        for obj in objs:
            yield obj, mats, depth
            if obj.type == raw.FPDF_PAGEOBJ_FORM and depth < FORM_DEPTH:
                kids = [pdfium.PdfObject(raw.FPDFFormObj_GetObject(obj.raw, i), page=page, pdf=page.pdf, level=depth + 1)
                        for i in range(raw.FPDFFormObj_CountObjects(obj.raw))]
                yield from rec(kids, [obj.get_matrix()] + mats, depth + 1)
    yield from rec(list(page.get_objects(max_depth=1)), [], 0)


def page_bounds(obj, mats):
    """Bounds of `obj` in page space: get_bounds() of an object inside a form is in the form's space."""
    left, bottom, right, top = obj.get_bounds()
    points = [(left, bottom), (left, top), (right, bottom), (right, top)]
    for m in mats:
        points = [m.on_point(x, y) for x, y in points]
    xs, ys = [p[0] for p in points], [p[1] for p in points]
    return min(xs), min(ys), max(xs), max(ys)


def to_device(page, box):
    """A page-space box as [x0, y0, x1, y1] on the rendered page (top-left origin, points)."""
    _, raw = _raw()
    w, h = page.get_size()
    sx, sy = max(1, round(w * DEVICE_SCALE)), max(1, round(h * DEVICE_SCALE))
    xs, ys = [], []
    for px, py in ((box[0], box[1]), (box[0], box[3]), (box[2], box[1]), (box[2], box[3])):
        dx, dy = ctypes.c_int(), ctypes.c_int()
        raw.FPDF_PageToDevice(page.raw, 0, 0, sx, sy, 0, px, py, ctypes.byref(dx), ctypes.byref(dy))
        xs.append(dx.value / DEVICE_SCALE)
        ys.append(dy.value / DEVICE_SCALE)
    return [round(min(xs), 2), round(min(ys), 2), round(max(xs), 2), round(max(ys), 2)]


def _text_of(obj, textpage):
    _, raw = _raw()
    size = raw.FPDFTextObj_GetText(obj.raw, textpage.raw, None, 0)
    if size <= 2:
        return ""
    buf = ctypes.create_string_buffer(size)
    raw.FPDFTextObj_GetText(obj.raw, textpage.raw, ctypes.cast(buf, ctypes.POINTER(ctypes.c_ushort)), size)
    return buf.raw[:size - 2].decode("utf-16-le", "replace")


def text_invisible(obj, textpage):
    """True for render mode 3 and for glyphless text drawn in another mode (PB6)."""
    _, raw = _raw()
    if raw.FPDFTextObj_GetTextRenderMode(obj.raw) == raw.FPDF_TEXTRENDERMODE_INVISIBLE:
        return True
    font = raw.FPDFTextObj_GetFont(obj.raw)
    if not font:
        return False
    size = raw.FPDFFont_GetBaseFontName(font, None, 0)
    if size:
        buf = ctypes.create_string_buffer(size)
        raw.FPDFFont_GetBaseFontName(font, buf, size)
        if b"glyphless" in buf.value.lower():
            return True
    text = _text_of(obj, textpage).strip()
    if not text:
        return False
    fsize = ctypes.c_float()
    raw.FPDFTextObj_GetFontSize(obj.raw, ctypes.byref(fsize))
    path = raw.FPDFFont_GetGlyphPath(font, ord(text[0]), fsize.value or 12.0)
    return not path or raw.FPDFGlyphPath_CountGlyphSegments(path) <= 0


def _visible_chars(textpage, crop):
    """Non-space characters whose box meets the CropBox."""
    _, raw = _raw()
    n = raw.FPDFText_CountChars(textpage.raw)
    left, bottom, right, top = crop
    count = 0
    lb, rb, bb, tb = ctypes.c_double(), ctypes.c_double(), ctypes.c_double(), ctypes.c_double()
    for i in range(max(0, n)):
        cp = raw.FPDFText_GetUnicode(textpage.raw, i)
        if cp == 0 or chr(cp).isspace() or cp in (0xFFFE, 0xFFFF):
            continue
        if not raw.FPDFText_GetCharBox(textpage.raw, i, ctypes.byref(lb), ctypes.byref(rb), ctypes.byref(bb), ctypes.byref(tb)):
            continue
        if rb.value >= left and lb.value <= right and tb.value >= bottom and bb.value <= top:
            count += 1
    return count


def _raster_identity(obj):
    """(digest of the stored data and pixel size, [width, height] in pixels)."""
    _, raw = _raw()
    w, h = ctypes.c_uint(), ctypes.c_uint()
    raw.FPDFImageObj_GetImagePixelSize(obj.raw, ctypes.byref(w), ctypes.byref(h))
    pixels = [w.value, h.value]
    size = raw.FPDFImageObj_GetImageDataRaw(obj.raw, None, 0)
    if not size or size > RASTER_HASH_LIMIT:
        return None, pixels
    buf = ctypes.create_string_buffer(size)
    raw.FPDFImageObj_GetImageDataRaw(obj.raw, buf, size)
    return hashlib.sha1(b"%dx%d:" % (w.value, h.value) + buf.raw).hexdigest(), pixels


def classify_page(page, index):
    _, raw = _raw()
    w, h = page.get_size()
    crop = page.get_cropbox()
    textpage = page.get_textpage()
    try:
        chars = _visible_chars(textpage, crop)
        visible = invisible = 0
        image_area = 0.0
        rasters = []
        for obj, mats, depth in walk(page):
            if obj.type == raw.FPDF_PAGEOBJ_IMAGE:
                box = page_bounds(obj, mats)
                device = to_device(page, box)
                area = max(0.0, device[2] - device[0]) * max(0.0, device[3] - device[1])
                image_area += area
                digest, pixels = _raster_identity(obj)
                rasters.append({"box": device, "area": round(area / max(w * h, 1e-6), 4),
                                "digest": digest, "pixels": pixels, "depth": depth})
            elif obj.type == raw.FPDF_PAGEOBJ_TEXT:
                if text_invisible(obj, textpage):
                    invisible += 1
                else:
                    visible += 1
    finally:
        textpage.close()
    cover = min(image_area / max(w * h, 1e-6), 1.0)
    if chars == 0:
        cls = "image"
    elif visible == 0 and invisible > 0 and cover >= OCR_LAYER_COVER:
        cls = "ocr_layer"
    else:
        cls = "native"
    return {"page": index + 1, "class": cls, "size": [round(w, 2), round(h, 2)], "rotation": page.get_rotation(),
            "chars": chars, "visible_text_objects": visible, "invisible_text_objects": invisible,
            "invisible_chars": chars if cls == "ocr_layer" else 0, "image_cover": round(cover, 3),
            "rasters": rasters}


def classify(path, max_pages):
    pdfium, _ = _raw()
    doc = pdfium.PdfDocument(str(path))
    try:
        n = len(doc)
        if n == 0 or n > max_pages:
            return {"pages": n, "page": []}
        return {"pages": n, "page": [classify_page(doc[i], i) for i in range(n)]}
    finally:
        doc.close()


def _form_kind(obj, textpage):
    """'invisible' when a form holds only invisible text, recursively; 'mixed' when it holds
    invisible text and anything else; None without invisible text."""
    _, raw = _raw()
    inv = other = 0
    for child, _, depth in walk_children(obj):
        if child.type == raw.FPDF_PAGEOBJ_TEXT and text_invisible(child, textpage):
            inv += 1
        elif child.type != raw.FPDF_PAGEOBJ_FORM:
            other += 1
    return None if not inv else "invisible" if not other else "mixed"


def walk_children(form, depth=1):
    pdfium, raw = _raw()
    for i in range(raw.FPDFFormObj_CountObjects(form.raw)):
        child = pdfium.PdfObject(raw.FPDFFormObj_GetObject(form.raw, i), page=form.page, pdf=form.pdf, level=depth)
        yield child, None, depth
        if child.type == raw.FPDF_PAGEOBJ_FORM and depth < FORM_DEPTH:
            yield from walk_children(child, depth + 1)


def render_digest(page):
    bitmap = page.render(scale=100 / 72)
    try:
        return hashlib.sha256(bytes(bitmap.buffer)).hexdigest()
    finally:
        bitmap.close()


def strip(path, out, pages):
    """Remove the invisible text of `pages` (1-based); the report says what was removed, which
    pages are still `ocr_layer` and whether their rendering changed."""
    pdfium, raw = _raw()
    doc = pdfium.PdfDocument(str(path))
    report = {"pages": []}
    try:
        before = {p: render_digest(doc[p - 1]) for p in pages}
        for p in pages:
            page = doc[p - 1]
            textpage = page.get_textpage()
            removed = forms_removed = mixed = 0
            try:
                for obj in list(page.get_objects(max_depth=1)):
                    if obj.type == raw.FPDF_PAGEOBJ_TEXT and text_invisible(obj, textpage):
                        if raw.FPDFPage_RemoveObject(page.raw, obj.raw):
                            raw.FPDFPageObj_Destroy(obj.raw)
                            removed += 1
                    elif obj.type == raw.FPDF_PAGEOBJ_FORM:
                        kind = _form_kind(obj, textpage)
                        if kind == "invisible" and raw.FPDFPage_RemoveObject(page.raw, obj.raw):
                            raw.FPDFPageObj_Destroy(obj.raw)
                            forms_removed += 1
                        elif kind == "mixed":
                            mixed += 1
            finally:
                textpage.close()
            generated = bool(raw.FPDFPage_GenerateContent(page.raw))
            report["pages"].append({"page": p, "text_removed": removed, "forms_removed": forms_removed,
                                    "mixed_forms_left": mixed, "generated": generated})
        doc.save(str(out))
    finally:
        doc.close()
    copy = pdfium.PdfDocument(str(out))
    try:
        after = {p: classify_page(copy[p - 1], p - 1)["class"] for p in pages}
        rendered = {p: render_digest(copy[p - 1]) for p in pages}
    finally:
        copy.close()
    report["still_ocr_layer"] = [p for p in pages if after[p] == "ocr_layer"]
    report["rendering_changed"] = [p for p in pages if rendered[p] != before[p]]
    report["removable"] = not report["still_ocr_layer"] and not report["rendering_changed"]
    return report


def image_to_pdf(source, out, fmt, max_pages, max_pixels):
    """png, jpeg or tiff -> a PDF of one page per frame; (pages, frames) or a ValueError."""
    pdfium, _ = _raw()
    from PIL import Image, ImageSequence
    Image.MAX_IMAGE_PIXELS = max_pixels
    import warnings
    warnings.simplefilter("error", Image.DecompressionBombWarning)
    pdf = pdfium.PdfDocument.new()
    pages = 0
    with Image.open(source) as img:
        frames = 1 if fmt in ("png", "jpeg") else getattr(img, "n_frames", 1)
        if frames > max_pages:
            raise ValueError("page_limit:%d" % frames)
        for frame in ImageSequence.Iterator(img) if fmt == "tiff" else [img]:
            w_px, h_px = frame.size
            if w_px < 1 or h_px < 1 or w_px * h_px > max_pixels:
                raise ValueError("image size")
            dpi = frame.info.get("dpi") or (DEFAULT_DPI, DEFAULT_DPI)
            dx = min(MAX_DPI, max(MIN_DPI, float(dpi[0] or DEFAULT_DPI)))
            dy = min(MAX_DPI, max(MIN_DPI, float(dpi[1] or DEFAULT_DPI)))
            w_pt, h_pt = w_px * 72 / dx, h_px * 72 / dy
            page = pdf.new_page(w_pt, h_pt)
            obj = pdfium.PdfImage.new(pdf)
            if fmt == "jpeg":
                obj.load_jpeg(str(source), inline=False)
            else:
                mode = frame.mode
                pil = frame.convert("RGB" if mode not in ("L", "RGB") else mode)
                bitmap = pdfium.PdfBitmap.from_pil(pil)
                obj.set_bitmap(bitmap)
                bitmap.close()
            obj.set_matrix(pdfium.PdfMatrix().scale(w_pt, h_pt))
            page.insert_obj(obj)
            page.gen_content()
            page.close()
            pages += 1
    pdf.save(str(out))
    pdf.close()
    return pages


def main(args):
    command = args[0]
    if command == "classify":
        result = classify(SOURCE, int(args[1]))
        (SCRATCH / "pdfprep.json").write_text(json.dumps(result))
    elif command == "strip":
        pages = [int(p) for p in args[1].split(",")]
        result = strip(SOURCE, SCRATCH / "stripped.pdf", pages)
        (SCRATCH / "strip.json").write_text(json.dumps(result))
    elif command == "image":
        fmt, max_pages, max_pixels = args[1], int(args[2]), int(args[3])
        ext = {"png": "png", "jpeg": "jpg", "tiff": "tiff"}[fmt]
        try:
            pages = image_to_pdf(Path("/in/source." + ext), SCRATCH / "source.pdf", fmt, max_pages, max_pixels)
            result = {"pages": pages}
        except Exception as exc:        # the caller maps it to a closed code
            result = {"error": str(exc)[:200] if str(exc).startswith("page_limit:") else type(exc).__name__}
        (SCRATCH / "image.json").write_text(json.dumps(result))
    else:
        print("pdfprep: unknown command", file=sys.stderr)
        return 2
    return 0
