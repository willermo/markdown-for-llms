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
import re

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
DATALAB_MODES = ("lite", "fast", "balanced", "accurate")      # the modes priced in datalab.py
# The options of a job of the product (MFL032 C2): validated here by the API and again by the
# worker, so the two processes refuse the same requests.
JOB_OPTIONS = ("route", "ocr", "lang", "datalab_mode", "transit_confirmed")
LANG = re.compile(r"[A-Za-z]{2,8}(-[A-Za-z0-9]{1,8}){0,4}")

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


def job_options(fmt, options=None):
    """The normalized options of a job for a file of format `fmt`: {route (None: the default of
    the matrix), ocr, lang, datalab_mode, transit_confirmed}. A route the format does not have,
    an option its converter does not accept, Datalab without the confirmed transit: a closed code
    (`route_invalid`, `route_option_unsupported`, `transit_not_confirmed`, `invalid_setting`)."""
    options = dict(options or {})
    if set(options) - set(JOB_OPTIONS):
        raise EngineError(Code.INVALID_SETTING, "unknown job option")
    route = options.get("route")
    if route is not None and not isinstance(route, str):
        raise EngineError(Code.ROUTE_INVALID, "route")
    converter = resolve(fmt, route)[0]
    ocr = "auto" if options.get("ocr") is None else options["ocr"]
    check_options(converter, {"ocr": ocr})
    lang = options.get("lang")
    if lang is not None and (not isinstance(lang, str) or not LANG.fullmatch(lang)):
        raise EngineError(Code.INVALID_SETTING, "lang: a language tag such as it or en-GB")
    mode, confirmed = options.get("datalab_mode"), options.get("transit_confirmed", False)
    if type(confirmed) is not bool:
        raise EngineError(Code.INVALID_SETTING, "transit_confirmed: true or false")
    if converter != DATALAB and (mode is not None or confirmed):
        raise EngineError(Code.ROUTE_OPTION_UNSUPPORTED, "Datalab options without the Datalab route")
    if converter == DATALAB:
        if mode is not None and mode not in DATALAB_MODES:
            raise EngineError(Code.ROUTE_OPTION_UNSUPPORTED, "datalab mode")
        if not confirmed:
            raise EngineError(Code.TRANSIT_NOT_CONFIRMED, "the document would leave the machine")
    return {"route": route, "ocr": ocr, "lang": lang, "datalab_mode": mode, "transit_confirmed": confirmed}
