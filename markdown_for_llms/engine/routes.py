"""The conversion matrix as data: for each format the default route and its alternatives.

A route is named after its converter:
- `text`: plain text decoded with charset-normalizer in the engine process (txt);
- `pandoc`: pandoc in the engine toolchain image, with the contract filter (born-digital
  formats; doc and ods are first converted by LibreOffice to docx and xlsx in the same image);
- `djvutxt`: the text layer of a DjVu file, page by page, in the same image;
- `marker`: Marker 2.2.0 on the CPU (the pages family, and an alternative for some born-digital
  formats through its own providers);
- `datalab`: the remote Datalab API, only when the user chooses it and it is configured.

`resolve(fmt, route)` validates a requested route (`route_invalid` when the format does not
have it) and says who chose it. Whether a route can run on this machine is `availability` of
the engine, not of this table. Standard library only: the API process imports this module.
"""
from .codes import Code, EngineError
from .formats import FORMATS

TEXT, PANDOC, DJVUTXT, MARKER, DATALAB = "text", "pandoc", "djvutxt", "marker", "datalab"
CONVERTERS = (TEXT, PANDOC, DJVUTXT, MARKER, DATALAB)
REMOTE = frozenset({DATALAB})
OCR_MODES = ("auto", "force")
# format -> (default, alternatives)
MATRIX = {
    "pdf": (MARKER, (DATALAB,)),
    "png": (MARKER, (DATALAB,)),
    "jpeg": (MARKER, (DATALAB,)),
    "tiff": (MARKER, (DATALAB,)),
    "djvu": (MARKER, (DJVUTXT, DATALAB)),
    "docx": (PANDOC, (MARKER,)),
    "xlsx": (PANDOC, (MARKER,)),
    "pptx": (PANDOC, (MARKER,)),
    "epub": (PANDOC, (MARKER,)),
    "html": (PANDOC, (MARKER,)),
    "odt": (PANDOC, ()),
    "ods": (PANDOC, ()),
    "doc": (PANDOC, ()),
    "rtf": (PANDOC, ()),
    "txt": (TEXT, ()),
}
# Which converters accept which request options besides the route.
ACCEPTS_OCR_FORCE = frozenset({MARKER})

assert set(MATRIX) == set(FORMATS), "the matrix covers exactly the formats"


def routes_for(fmt):
    """{default, alternatives} of a format of the matrix."""
    default, alternatives = MATRIX[fmt]
    return {"default": default, "alternatives": list(alternatives)}


def resolve(fmt, route=None):
    """(converter, chosen_by, reason) of the route requested for `fmt` (None: the default)."""
    default, alternatives = MATRIX[fmt]
    if route is None or route == default:
        return default, "matrix", "matrix-default" if route is None else "imposed"
    if route not in CONVERTERS or route not in alternatives:
        raise EngineError(Code.ROUTE_INVALID, f"{fmt} has no route {route!r}")
    return route, "user", "imposed"


def check_options(converter, request):
    """Options a converter does not accept are refused, never ignored."""
    ocr = request.get("ocr", "auto")
    if ocr not in OCR_MODES:
        raise EngineError(Code.ROUTE_OPTION_UNSUPPORTED, "ocr")
    if ocr == "force" and converter not in ACCEPTS_OCR_FORCE:
        raise EngineError(Code.ROUTE_OPTION_UNSUPPORTED, f"ocr force with {converter}")
