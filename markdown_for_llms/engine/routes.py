"""The conversion matrix as data: for each format the default route and its alternatives.

A route is named after its converter:
- `text`: plain text decoded with charset-normalizer in the engine process (txt);
- `pandoc`: pandoc in the engine toolchain image, with the contract filter (born-digital
  formats; doc and ods are first converted by LibreOffice to docx and xlsx in the same image);
- `libreoffice`: LibreOffice in the same image writes HTML (Calc for xlsx, `impress_html_Export`
  for pptx), which pandoc reads (an alternative for spreadsheets and slides; the images of the
  slides come from the package, after the heading of their slide);
- `djvutxt`: the text layer of a DjVu file, page by page, in the same image;
- `marker`: Marker 2.2.0 on the CPU (the pages family, and an alternative for some born-digital
  formats through its own providers);
- `datalab`: the remote Datalab API, only when the user chooses it and it is configured;
- `marker-server`: Marker 2.2.0 in `balanced` or `accurate` with the inference of Surya on an
  OpenAI-compatible server the user names (MFL033): the local CPU server, a server on the
  loopback, the user's own server, a RunPod or Modal endpoint. In the product (MFL023 B3) only on
  a GPU preset of the instance (`gpu_preset`: a rented GPU, or since MFL051 B the local GPU of the
  `gpu` profile and the remote server of the user), with a server mode, and the transit confirmed
  when the preset sends the page images out of the machine (`leaves_machine`: a rented GPU and the
  remote server; never the local GPU, for which a confirmation is refused as an option it does not
  take); without a GPU preset it stays for the engine, `mdf convert` and `mdf bench`
  (`route_not_in_product`: a local or own server is not in the archive, D4 of MFL023). The
  product does not list it among the alternatives: it is reached through the conversion presets.

`resolve(fmt, route)` validates a requested route (`route_invalid` when the format does not
have it) and says who chose it. Whether a route can run on this machine is `availability` of
the engine, not of this table. `routes_for` is the view of the product (without the routes of
`PRODUCT_EXCLUDED`), `engine_routes_for` the whole matrix. The routes of `EXPLICIT` run only
when asked by name: the bench leaves them out of its default runs. Standard library only: the
API process imports this module.
"""
import re

from .codes import Code, EngineError
from .formats import FORMATS

TEXT, PANDOC, LIBREOFFICE, DJVUTXT, MARKER, DATALAB = "text", "pandoc", "libreoffice", "djvutxt", "marker", "datalab"
MARKER_SERVER = "marker-server"
CONVERTERS = (TEXT, PANDOC, LIBREOFFICE, DJVUTXT, MARKER, DATALAB, MARKER_SERVER)
REMOTE = frozenset({DATALAB})
EXPLICIT = frozenset({DATALAB, MARKER_SERVER})          # never part of a default selection of the bench
PRODUCT_EXCLUDED = frozenset({MARKER_SERVER})           # not listed by the product: reached through its presets
OCR_MODES = ("auto", "force")
SERVER_MODES = ("balanced", "accurate")
# format -> (default, alternatives)
MATRIX = {
    "pdf": (MARKER, (DATALAB, MARKER_SERVER)),
    "png": (MARKER, (DATALAB, MARKER_SERVER)),
    "jpeg": (MARKER, (DATALAB, MARKER_SERVER)),
    "tiff": (MARKER, (DATALAB, MARKER_SERVER)),
    "djvu": (MARKER, (DJVUTXT, DATALAB, MARKER_SERVER)),
    "docx": (PANDOC, (MARKER,)),
    "xlsx": (PANDOC, (MARKER, LIBREOFFICE)),
    "pptx": (PANDOC, (MARKER, LIBREOFFICE)),
    "epub": (PANDOC, (MARKER,)),
    "html": (PANDOC, (MARKER,)),
    "odt": (PANDOC, ()),
    "ods": (PANDOC, ()),
    "doc": (PANDOC, ()),
    "rtf": (PANDOC, ()),
    "txt": (TEXT, ()),
}
# Which converters accept which request options besides the route.
ACCEPTS_OCR_FORCE = frozenset({MARKER, MARKER_SERVER})
DATALAB_MODES = ("lite", "fast", "balanced", "accurate")      # the modes priced in datalab.py
# The options of a job of the product (MFL032 C2): validated here by the API and again by the
# worker, so the two processes refuse the same requests. `server_mode` and `gpu_preset` (MFL023 B3)
# only on the route marker-server, and stored only there: a job of another route keeps the five
# options of schema 3.
JOB_OPTIONS = ("route", "ocr", "lang", "datalab_mode", "transit_confirmed", "server_mode", "gpu_preset")
GPU_OPTIONS = ("server_mode", "gpu_preset")
LANG = re.compile(r"[A-Za-z]{2,8}(-[A-Za-z0-9]{1,8}){0,4}")
GPU_PRESET = re.compile(r"[a-z0-9][a-z0-9-]{0,63}")

assert set(MATRIX) == set(FORMATS), "the matrix covers exactly the formats"


def engine_routes_for(fmt):
    """{default, alternatives} of a format of the matrix (engine, CLI and bench)."""
    default, alternatives = MATRIX[fmt]
    return {"default": default, "alternatives": list(alternatives)}


def routes_for(fmt):
    """{default, alternatives} of a format in the product: the matrix without `PRODUCT_EXCLUDED`."""
    default, alternatives = MATRIX[fmt]
    return {"default": default, "alternatives": [r for r in alternatives if r not in PRODUCT_EXCLUDED]}


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
    server = request.get("server")
    if converter != MARKER_SERVER and server is not None:
        raise EngineError(Code.ROUTE_OPTION_UNSUPPORTED, f"server options with {converter}")
    if converter == MARKER_SERVER:
        if not isinstance(server, dict) or server.get("mode") not in SERVER_MODES:
            raise EngineError(Code.ROUTE_OPTION_UNSUPPORTED, "marker-server needs a server mode: balanced or accurate")
        if request.get("datalab") is not None:
            raise EngineError(Code.ROUTE_OPTION_UNSUPPORTED, "Datalab options with marker-server")


def job_options(fmt, options=None, leaves_machine=True):
    """The normalized options of a job for a file of format `fmt`: {route (None: the default of
    the matrix), ocr, lang, datalab_mode, transit_confirmed}, and on the route marker-server also
    {server_mode, gpu_preset} (MFL023 B3). A route the format does not have, an option its
    converter does not accept, Datalab or a GPU without the confirmed transit, marker-server
    without a GPU preset: a closed code (`route_invalid`, `route_option_unsupported`,
    `transit_not_confirmed`, `invalid_setting`, `route_not_in_product`). Whether the instance
    enables the GPU preset is checked by the caller (`services.job_options`), which also says
    whether its server is out of this machine (`leaves_machine`, MFL051 B, from the provider of
    the preset; true by default, the rule before MFL051): only then is the transit confirmed, and a
    confirmation given to a server of this machine is `route_option_unsupported`, as a transit
    without a remote route."""
    options = dict(options or {})
    if set(options) - set(JOB_OPTIONS):
        raise EngineError(Code.INVALID_SETTING, "unknown job option")
    route = options.get("route")
    if route is not None and not isinstance(route, str):
        raise EngineError(Code.ROUTE_INVALID, "route")
    server_mode, gpu = options.get("server_mode"), options.get("gpu_preset")
    if route == MARKER_SERVER and gpu is None:
        raise EngineError(Code.ROUTE_NOT_IN_PRODUCT, f"{route}: in the archive only on a GPU preset of the instance")
    converter = resolve(fmt, route)[0]
    ocr = "auto" if options.get("ocr") is None else options["ocr"]
    server = {"server": {"mode": server_mode}} if converter == MARKER_SERVER else {}
    check_options(converter, dict({"ocr": ocr}, **server))
    lang = options.get("lang")
    if lang is not None and (not isinstance(lang, str) or not LANG.fullmatch(lang)):
        raise EngineError(Code.INVALID_SETTING, "lang: a language tag such as it or en-GB")
    mode, confirmed = options.get("datalab_mode"), options.get("transit_confirmed", False)
    if type(confirmed) is not bool:
        raise EngineError(Code.INVALID_SETTING, "transit_confirmed: true or false")
    if converter != DATALAB and mode is not None:
        raise EngineError(Code.ROUTE_OPTION_UNSUPPORTED, "Datalab options without the Datalab route")
    if converter not in (DATALAB, MARKER_SERVER) and confirmed:
        raise EngineError(Code.ROUTE_OPTION_UNSUPPORTED, "a transit without a remote route")
    if converter == MARKER_SERVER and confirmed and not leaves_machine:
        raise EngineError(Code.ROUTE_OPTION_UNSUPPORTED, "a transit to a GPU of this machine: nothing leaves it")
    if converter != MARKER_SERVER and (server_mode is not None or gpu is not None):
        raise EngineError(Code.ROUTE_OPTION_UNSUPPORTED, "GPU options without the marker-server route")
    if converter == DATALAB:
        if mode is not None and mode not in DATALAB_MODES:
            raise EngineError(Code.ROUTE_OPTION_UNSUPPORTED, "datalab mode")
        if not confirmed:
            raise EngineError(Code.TRANSIT_NOT_CONFIRMED, "the document would leave the machine")
    out = {"route": route, "ocr": ocr, "lang": lang, "datalab_mode": mode, "transit_confirmed": confirmed}
    if converter == MARKER_SERVER:
        if not isinstance(gpu, str) or not GPU_PRESET.fullmatch(gpu):
            raise EngineError(Code.INVALID_SETTING, "gpu_preset: the name of a GPU preset")
        if server_mode not in SERVER_MODES:
            raise EngineError(Code.ROUTE_OPTION_UNSUPPORTED, "marker-server needs a server mode: balanced or accurate")
        if leaves_machine and not confirmed:
            raise EngineError(Code.TRANSIT_NOT_CONFIRMED, "the page images would leave the machine")
        out.update(server_mode=server_mode, gpu_preset=gpu)
    return out
