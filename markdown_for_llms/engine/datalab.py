"""The Datalab route: the remote conversion API of Datalab, only when the user chooses it.

The document leaves the machine. The route runs only with the transit confirmed by the user
(`request["datalab"]["transit_confirmed"]`) and under a spending cap of the execution
(`cap_usd`, a finite decimal >= 0); without the confirmation it stops before any byte leaves,
with the pages and the estimated cost in `engine.json`. There is no fallback between the local
and the remote routes.

What is sent: a PDF (a PDF as it is; png, jpeg, tiff and djvu made PDF as for Marker), from which
the PDF preparation has removed an invisible OCR layer (P7 of MFL032; Datalab has no option to
redo it): when the layer cannot be removed, or the copy renders differently, the route refuses
(`ocr_layer_not_removable`) and nothing is sent. `--ocr force` is refused (Datalab accepts no
such option); the language is recorded and not passed (the API accepts none).

The exchange (https://documentation.datalab.to, read on 2026-11-15): `POST /api/v1/convert`
(multipart: the file as an upload, never `file_url`; `mode`, `output_format=markdown`,
`paginate=true`), the `request_id` written in the attempt right after the answer; `GET
/api/v1/convert/{request_id}` until `status` is `complete`, the observer called at every poll;
`DELETE /api/v1/results/{request_id}` always, in `finally` (success, error, timeout,
cancellation; a failed delete is the visible warning `datalab_delete_failed`; without it the
result stays one day on Datalab). An answer of the POST that is not a clear acceptance or a
clear refusal is never followed by another POST.

The key is `DATALAB_API_KEY`, read from the file given with `--env-file` (a regular file of the
caller with mode 0600, opened without following links; otherwise refused) or from the
environment of the process; it is never a setting, never written, never passed to a child.

Requests: `requests` with `trust_env=False` (no proxy or netrc of the environment), HTTPS to the
fixed host only, certificate verification on, bounded timeouts.
"""
from decimal import Decimal, InvalidOperation, ROUND_CEILING
import base64
import json
import os
import re
import time

from . import contract, envfile, imagerefs, routes
from .codes import Code, EngineError

API = "https://www.datalab.to/api/v1"
KEY_NAME = "DATALAB_API_KEY"
KEY_FILE_LIMIT = envfile.LIMIT
# Prices of Convert in USD per 1,000 pages, from https://www.datalab.to/pricing, read on
# 2026-11-15. The page states them with the 25 % discount for sharing data for training; the
# estimate uses the list price without the discount (the larger one): discounted / 0.75.
PRICE_SOURCE = "https://www.datalab.to/pricing (read 2026-11-15; listed with the 25% data-sharing discount)"
DISCOUNTED_PER_1000 = {"lite": Decimal("0.375"), "fast": Decimal("1"), "balanced": Decimal("3"),
                       "accurate": Decimal("7.50")}
LIST_PER_1000 = {mode: price / Decimal("0.75") for mode, price in DISCOUNTED_PER_1000.items()}
assert tuple(LIST_PER_1000) == routes.DATALAB_MODES, "the job options of routes.py name the priced modes"
DEFAULT_MODE = "balanced"
MARGIN_CENTS = Decimal(1)
POLL_SECONDS = 2.0
WAIT_LIMIT = 1800                     # seconds of polling at most, within the deadline of the conversion
CONNECT_TIMEOUT, READ_TIMEOUT = 15, 120
RESPONSE_LIMIT = 256 << 20
# Limits of a conversion request (https://documentation.datalab.to/docs/common/limits.md, read on
# 2026-11-17: «200 MB» per file, «7,000» pages): a PDF over them is refused before it is sent.
FILE_LIMIT_BYTES = 200_000_000
PAGE_LIMIT = 7000
IMAGE_TYPES = ((b"\x89PNG\r\n\x1a\n", "png"), (b"\xff\xd8\xff", "jpg"), (b"GIF8", "gif"))


class Key:
    """The API key; its value never appears in a repr, a log or an error."""
    __slots__ = ("_value",)

    def __init__(self, value):
        self._value = value

    def __repr__(self):
        return "Key(***)"

    __str__ = __repr__

    def header(self):
        return {"X-API-Key": self._value}


def _parse_env(data):
    return envfile.parse(data, {KEY_NAME}).get(KEY_NAME)


def read_key(env_file=None, environ=None):
    """The key from `env_file` (checked as the module docstring says) or from the environment."""
    if env_file is not None:
        try:
            value = envfile.read(env_file, [KEY_NAME]).get(KEY_NAME)
        except envfile.Unsafe as exc:
            raise EngineError(Code.DATALAB_KEY_UNSAFE, str(exc)) from None
        if value:
            return Key(value)
    value = (os.environ if environ is None else environ).get(KEY_NAME)
    if not value:
        raise EngineError(Code.DATALAB_KEY_MISSING, KEY_NAME + " in the env file or the environment")
    return Key(value)


def parse_cap(value):
    try:
        cap = Decimal(str(value))
    except (InvalidOperation, ValueError):
        cap = None
    if cap is None or not cap.is_finite() or cap < 0:
        raise EngineError(Code.INVALID_SETTING, "datalab cap: a finite amount in USD, 0 or more")
    return cap


def estimate_cents(pages, mode):
    """Pages at the list price of the mode, rounded up to a hundredth of a cent, plus one cent."""
    cents = Decimal(pages) * LIST_PER_1000[mode] / 10
    return (cents + MARGIN_CENTS).quantize(Decimal("0.01"), rounding=ROUND_CEILING)


def billed_cents(breakdown):
    """The cost billed, in cents, from `cost_breakdown` (its fields are not documented): a field
    named in cents, else None (the estimate then stands as the bound, with a warning)."""
    if not isinstance(breakdown, dict):
        return None
    for key in ("final_cost_cents", "total_cost_cents", "cost_cents"):
        value = breakdown.get(key)
        if type(value) in (int, float) and value >= 0:
            return Decimal(str(value))
    return None


class Transport:
    """HTTPS to the fixed API host with `requests`; (status, JSON body or None) per call."""

    def __init__(self):
        import requests
        self.requests = requests
        self.session = requests.Session()
        self.session.trust_env = False

    def _call(self, method, url, **kw):
        if not url.startswith(API + "/"):
            raise EngineError(Code.DATALAB_FAILED, "URL outside the API")
        response = self.session.request(method, url, timeout=(CONNECT_TIMEOUT, READ_TIMEOUT), allow_redirects=False,
                                        verify=True, stream=True, **kw)
        try:
            body = response.raw.read(RESPONSE_LIMIT + 1, decode_content=True)
        finally:
            response.close()
        if len(body) > RESPONSE_LIMIT:
            raise EngineError(Code.OUTPUT_LIMIT, "Datalab response over the limit")
        try:
            data = json.loads(body) if body else None
        except ValueError:
            data = None
        return response.status_code, data

    def post(self, url, headers, files, data):
        return self._call("POST", url, headers=headers, files=files, data=data)

    def get(self, url, headers):
        return self._call("GET", url, headers=headers)

    def delete(self, url, headers):
        return self._call("DELETE", url, headers=headers)

    def network_errors(self):
        return (self.requests.RequestException, OSError)


def _image_ext(data):
    for magic, ext in IMAGE_TYPES:
        if data.startswith(magic):
            return ext
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    return None


def datalab_route(ctx, src, fmt, request, key=None, transport=None, environ=None):
    from . import pages
    options = request.get("datalab") or {}
    mode = options.get("mode") or DEFAULT_MODE
    if mode not in LIST_PER_1000:
        raise EngineError(Code.ROUTE_OPTION_UNSUPPORTED, "datalab mode " + str(mode))
    cap = parse_cap(0 if options.get("cap_usd") is None else options["cap_usd"])
    key = key if key is not None else read_key(environ=environ)
    record = {"endpoint": API, "mode": mode, "price_source": PRICE_SOURCE,
              "list_price_usd_per_1000": str(LIST_PER_1000[mode].quantize(Decimal("0.0001")))}
    pdf = pages.source_pdf(ctx, src, fmt)
    page_list = pages.classify(ctx, pdf)
    layer = [p["page"] for p in page_list if p["class"] == "ocr_layer"]
    if layer:
        ctx.event("strip")
        scratch = pages._prep(ctx, "strip", ["strip", ",".join(map(str, layer))], [(pdf, "source.pdf")])
        report = json.loads((scratch / "strip.json").read_text())
        record["strip"] = report
        if not report["removable"]:
            raise EngineError(Code.OCR_LAYER_NOT_REMOVABLE, "pages %s" % (report["still_ocr_layer"] or report["rendering_changed"]))
        pdf = scratch / "stripped.pdf"
    n = len(page_list)
    estimate = estimate_cents(n, mode)
    ctx.cost = {"estimated_cents": str(estimate), "billed_cents": None, "cap_usd": str(cap), "pages": n}
    ctx.transit = {"left_machine": False, "service": "datalab", "endpoint": API}
    size = os.stat(pdf).st_size
    if size > FILE_LIMIT_BYTES or n > PAGE_LIMIT:
        # the PDF that would be sent, after the removal of an OCR layer: refused before the
        # event `datalab-send`, so the bench reserves nothing
        raise EngineError(Code.DATALAB_LIMIT, "%d bytes and %d pages; Datalab accepts %d bytes and %d pages"
                          % (size, n, FILE_LIMIT_BYTES, PAGE_LIMIT))
    if estimate > cap * 100:
        raise EngineError(Code.DATALAB_CAP, "estimate %s cents over the cap of %s USD" % (estimate, cap))
    if options.get("transit_confirmed") is not True:
        raise EngineError(Code.TRANSIT_NOT_CONFIRMED, "the document would leave the machine for Datalab")
    transport = transport or Transport()
    errors = transport.network_errors()
    ctx.event("datalab-send")
    ctx.transit["left_machine"] = True                  # from here the bytes may have left
    with open(pdf, "rb") as stream:
        try:
            status, body = transport.post(API + "/convert", key.header(),
                                          files={"file": ("source.pdf", stream, "application/pdf")},
                                          data={"mode": mode, "output_format": "markdown", "paginate": "true",
                                                "disable_image_extraction": "false"})
        except errors as exc:
            # no answer: the request may or may not have been accepted; never posted again
            raise EngineError(Code.DATALAB_UNAVAILABLE, "POST without an answer: %s" % type(exc).__name__) from None
    request_id = body.get("request_id") if isinstance(body, dict) else None
    if not isinstance(request_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", request_id):
        if status in (400, 401, 402, 403, 404, 413, 415, 422, 429):
            raise EngineError(Code.DATALAB_REJECTED, "POST refused with HTTP %d" % status)
        raise EngineError(Code.DATALAB_FAILED, "POST answer not understood (HTTP %d); not posted again" % status)
    # From a valid request id on, the result exists on Datalab: the id goes to the result first,
    # and every later step, local writes included, is inside the try whose finally deletes it
    # (FC3 of the review of B).
    ctx.transit["request_id"] = request_id
    try:
        if status != 200:
            raise EngineError(Code.DATALAB_FAILED, "POST answered HTTP %d with a request id" % status)
        (ctx.work / "datalab-request.json").write_text(json.dumps({"request_id": request_id, "at": time.time()}))
        if body.get("success") is False:
            raise EngineError(Code.DATALAB_REJECTED, "POST answered success=false")
        result = _poll(ctx, transport, key, request_id, errors)
        return _collect(ctx, result, n, mode, record, estimate)
    finally:
        _delete(ctx, transport, key, request_id)


def _poll(ctx, transport, key, request_id, errors):
    deadline = time.monotonic() + min(WAIT_LIMIT, ctx.timeout())
    failures = 0
    while True:
        if ctx.observer is not None and (stop := ctx.observer({"event": "tick", "service": "datalab"})):
            raise EngineError(Code.INTERRUPTED if stop == "interrupted" else Code.CANCELLED, "datalab")
        try:
            status, body = transport.get(API + "/convert/" + request_id, key.header())
            failures = 0
        except errors as exc:
            failures += 1
            if failures >= 5:
                raise EngineError(Code.DATALAB_UNAVAILABLE, "polling: %s" % type(exc).__name__) from None
            status, body = None, None
        if status == 200 and isinstance(body, dict) and body.get("status") == "complete":
            return body
        if status is not None and status != 200:
            raise EngineError(Code.DATALAB_FAILED, "polling answered HTTP %d" % status)
        if time.monotonic() > deadline:
            raise EngineError(Code.TIMEOUT, "Datalab did not complete in time")
        time.sleep(POLL_SECONDS)


def _delete(ctx, transport, key, request_id):
    try:
        status, body = transport.delete(API + "/results/" + request_id, key.header())
        deleted = status == 200 and not (isinstance(body, dict) and body.get("success") is False)
    except Exception:
        deleted = False
    ctx.transit["deleted"] = deleted
    if not deleted:
        ctx.problems.append(contract.problem("datalab_delete_failed", detail="request %s: the result stays on Datalab "
                                             "until its one-day lifecycle removes it" % request_id))


def _collect(ctx, body, n, mode, record, estimate):
    from . import pages
    if body.get("success") is not True:
        raise EngineError(Code.DATALAB_FAILED, "conversion not successful")
    text = body.get("markdown")
    if not isinstance(text, str):
        raise EngineError(Code.OUTPUT_INVALID, "no markdown in the answer")
    raw = text.encode("utf-8")
    text = contract.check_document(raw)
    if not pages.SEPARATOR.sub("", text).strip():
        raise EngineError(Code.OUTPUT_EMPTY, "only page separators")
    if body.get("page_count") is not None and body["page_count"] != n:
        raise EngineError(Code.CONVERTER_FAILED, "Datalab converted %s pages of %d" % (body["page_count"], n))
    (ctx.stage / "diagnostics" / "datalab-original.md").write_bytes(raw)
    images = body.get("images") or {}
    if not isinstance(images, dict):
        raise EngineError(Code.OUTPUT_INVALID, "images of the answer")
    data, exts = {}, {}
    for name, encoded in images.items():
        if not isinstance(name, str) or "/" in name or "\\" in name or name.startswith(".") or not isinstance(encoded, str):
            raise EngineError(Code.OUTPUT_INVALID, "an image name of the answer")
        try:
            blob = base64.b64decode(encoded, validate=True)
        except ValueError:
            raise EngineError(Code.OUTPUT_INVALID, "an image of the answer is not base64") from None
        ext = _image_ext(blob)
        if ext is None:
            raise EngineError(Code.OUTPUT_INVALID, "an image of the answer of an unknown type")
        data[name], exts[name] = blob, ext
    refs = imagerefs.scan(text)             # Markdown images and <img> tags (MFL043)
    used = refs.used()
    unused = [name for name in data if name not in used]
    for name in unused:             # an image of the answer that the document does not use
        ctx.problems.append(contract.problem("media_unused", path=name, detail="image of the Datalab answer"))
        del data[name]
    text, mapping, found = imagerefs.rename(text, list(data), ext_of=exts, refs=refs)
    ctx.problems.extend(found)
    (ctx.stage / "assets").mkdir(exist_ok=True)
    for name, new in mapping.items():
        (ctx.stage / new).write_bytes(data[name])
    (ctx.stage / "document.md").write_text(text, encoding="utf-8")
    billed = billed_cents(body.get("cost_breakdown"))
    ctx.cost.update(billed_cents=None if billed is None else str(billed), breakdown=body.get("cost_breakdown"))
    if billed is None:
        ctx.problems.append(contract.problem("datalab_cost_unknown", detail="no billed cost in cents in the answer; "
                                             "the estimate of %s cents is the bound" % estimate))
    ctx.datalab_versions = body.get("versions")
    return dict(record, pages=n, images=len(mapping), versions=body.get("versions"),
                runtime=body.get("runtime"), parse_quality_score=body.get("parse_quality_score"))
