"""Formats of the conversion matrix: recognized by extension **and** signature.

`detect(path, name)` reads the file (never the whole of it into memory) and returns the format,
its family and the signature that matched. An extension outside the matrix is
`format_unsupported`; an extension whose signature does not match the bytes is
`format_mismatch`; a file that announces a ZIP package but whose central directory cannot be
read is `input_unreadable` (a damaged file). ZIP packages are recognized from the central
directory of the whole file (K3 of MFL032: the directory is at the end, so the first bytes are
not enough) and, for OpenDocument and EPUB, from the `mimetype` member: at most `MAX_MEMBERS`
members, a `mimetype` of at most 128 bytes, nothing extracted and no XML parsed. A txt is text
when its first 64 KiB hold no NUL byte or, after a UTF-32 or UTF-16 byte order mark (looked for
in this order, the 4-byte ones first), when they decode strictly (incrementally: a code unit or
a surrogate pair cut at the end of the probe is no error) into text without C0 control
characters other than tab, line feed, vertical tab, form feed and carriage return. Standard
library only: the API process imports this module.
"""
import codecs
from pathlib import Path
import re
import zipfile

from .codes import Code, EngineError

NATIVE, PAGES = "native", "pages"
# format -> (family, extensions); the first extension is the canonical one
FORMATS = {
    "pdf": (PAGES, ("pdf",)),
    "png": (PAGES, ("png",)),
    "jpeg": (PAGES, ("jpg", "jpeg")),
    "tiff": (PAGES, ("tif", "tiff")),
    "djvu": (PAGES, ("djvu", "djv")),
    "docx": (NATIVE, ("docx",)),
    "xlsx": (NATIVE, ("xlsx",)),
    "pptx": (NATIVE, ("pptx",)),
    "odt": (NATIVE, ("odt",)),
    "ods": (NATIVE, ("ods",)),
    "epub": (NATIVE, ("epub",)),
    "doc": (NATIVE, ("doc",)),
    "rtf": (NATIVE, ("rtf",)),
    "html": (NATIVE, ("html", "htm")),
    "txt": (NATIVE, ("txt",)),
}
EXTENSIONS = {ext: fmt for fmt, (_, exts) in FORMATS.items() for ext in exts}
# the media type of a source of each format (the download of the archived source, MFL032 C2)
MIME = {"pdf": "application/pdf", "png": "image/png", "jpeg": "image/jpeg", "tiff": "image/tiff",
        "djvu": "image/vnd.djvu",
        "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        "odt": "application/vnd.oasis.opendocument.text", "ods": "application/vnd.oasis.opendocument.spreadsheet",
        "epub": "application/epub+zip", "doc": "application/msword", "rtf": "application/rtf",
        "html": "text/html", "txt": "text/plain"}
MAX_MEMBERS = 10000
MIMETYPE_MAX = 128
TEXT_PROBE = 64 * 1024
OOXML_MAIN = {"docx": "word/document.xml", "xlsx": "xl/workbook.xml", "pptx": "ppt/presentation.xml"}
ODF_MIMETYPE = {"odt": "application/vnd.oasis.opendocument.text",
                "ods": "application/vnd.oasis.opendocument.spreadsheet",
                "epub": "application/epub+zip"}
OLE2 = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"
# byte order marks of a txt, the 4-byte ones first (FF FE 00 00 starts with the UTF-16 LE mark):
# (bytes, canonical name of the encoding, Python codec that does not look for a mark itself)
BOMS = ((b"\xff\xfe\x00\x00", "utf_32", "utf-32-le"), (b"\x00\x00\xfe\xff", "utf_32", "utf-32-be"),
        (b"\xff\xfe", "utf_16", "utf-16-le"), (b"\xfe\xff", "utf_16", "utf-16-be"),
        (b"\xef\xbb\xbf", "utf_8", "utf-8"))
# C0 controls refused in a text after a UTF-16 or UTF-32 mark: all but \t \n \v \f \r
C0_REFUSED = re.compile("[\x00-\x08\x0e-\x1f]")


def family(fmt):
    return FORMATS[fmt][0]


def extension_of(name):
    """The lower-case extension of a file name, without the dot ('' when there is none)."""
    suffix = Path(name).suffix
    return suffix[1:].lower() if suffix else ""


def _zip_kind(path):
    """'docx', 'xlsx', 'pptx', 'odt', 'ods', 'epub' or None for another ZIP; raises
    `input_unreadable` when the central directory cannot be read or is over the limits."""
    try:
        with zipfile.ZipFile(path) as z:
            infos = z.infolist()
            if len(infos) > MAX_MEMBERS:
                raise EngineError(Code.INPUT_UNREADABLE, "ZIP: too many members")
            names = {i.filename for i in infos}
            if "[Content_Types].xml" in names:
                for fmt, main in OOXML_MAIN.items():
                    if main in names:
                        return fmt
                return None
            info = next((i for i in infos if i.filename == "mimetype"), None)
            if info is None or info.file_size > MIMETYPE_MAX or info.compress_size > MIMETYPE_MAX + 64:
                return None
            with z.open(info) as stream:
                value = stream.read(MIMETYPE_MAX + 1)
            if len(value) > MIMETYPE_MAX:
                return None
            text = value.decode("ascii", "replace").strip()
            return next((fmt for fmt, mime in ODF_MIMETYPE.items() if mime == text), None)
    except (zipfile.BadZipFile, zipfile.LargeZipFile, EOFError, NotImplementedError, ValueError) as exc:
        raise EngineError(Code.INPUT_UNREADABLE, f"ZIP: {type(exc).__name__}") from None


def text_bom(head):
    """(canonical name, codec, length) of the byte order mark that starts `head`, or None."""
    for bom, name, codec in BOMS:
        if head.startswith(bom):
            return name, codec, len(bom)
    return None


def _wide_text(head, found):
    """The signature of a txt that starts with a UTF-16 or UTF-32 mark, or None."""
    name, codec, n = found
    try:
        sample = codecs.getincrementaldecoder(codec)("strict").decode(head[n:], final=False)
    except UnicodeDecodeError:
        return None
    if C0_REFUSED.search(sample):
        return None
    return f"text ({name} BOM)"


def signature(path, fmt):
    """The name of the signature of `fmt` found in the file, or None."""
    with open(path, "rb") as stream:
        head = stream.read(TEXT_PROBE)
    if fmt == "pdf":
        return "%PDF-" if head.startswith(b"%PDF-") else None
    if fmt == "png":
        return "PNG" if head.startswith(b"\x89PNG\r\n\x1a\n") else None
    if fmt == "jpeg":
        return "JPEG SOI" if head.startswith(b"\xff\xd8\xff") else None
    if fmt == "tiff":
        return "TIFF" if head[:4] in (b"II*\x00", b"MM\x00*") else None
    if fmt == "djvu":
        return "AT&TFORM" if head[:8] == b"AT&TFORM" and head[12:16] in (b"DJVU", b"DJVM") else None
    if fmt == "doc":
        return "OLE2" if head.startswith(OLE2) else None
    if fmt == "rtf":
        return "{\\rtf" if head.startswith(b"{\\rtf") else None
    if fmt == "html":
        text = head[:1024 + 3]
        if text.startswith(b"\xef\xbb\xbf"):
            text = text[3:]
        text = text.lstrip(b" \t\r\n\f").lower()
        return "HTML" if text.startswith((b"<!doctype html", b"<html")) else None
    if fmt == "txt":
        found = text_bom(head)
        if found is not None and found[0] != "utf_8":
            return _wide_text(head, found)
        return "text" if b"\x00" not in head else None
    if fmt in OOXML_MAIN or fmt in ODF_MIMETYPE:
        if not head.startswith(b"PK\x03\x04"):
            return None
        kind = _zip_kind(path)
        if kind != fmt:
            return None
        return "ZIP " + (OOXML_MAIN[fmt] if fmt in OOXML_MAIN else "mimetype " + ODF_MIMETYPE[fmt])
    raise ValueError("not a format of the matrix: " + fmt)


def detect(path, name=None):
    """{format, family, extension, signature} of a file named `name` (default: its own name)."""
    name = Path(path).name if name is None else name
    ext = extension_of(name)
    fmt = EXTENSIONS.get(ext)
    if fmt is None:
        raise EngineError(Code.FORMAT_UNSUPPORTED, f"extension {ext!r}")
    try:
        found = signature(path, fmt)
    except OSError as exc:
        raise EngineError(Code.INPUT_UNREADABLE, f"{type(exc).__name__}") from None
    if found is None:
        raise EngineError(Code.FORMAT_MISMATCH, f"extension {ext} without the signature of {fmt}")
    return {"format": fmt, "family": family(fmt), "extension": ext, "signature": found}
