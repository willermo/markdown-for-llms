"""The Marker child: Marker 2.2.0 in a container of its own started by the launcher (no network, a memory
cap; MFL051), `fast` on the CPU or, on the `marker-server` route (MFL033), `balanced` or `accurate` with the
inference of Surya on a server.

Run by `pages.py` through `isolation.run` (never imported by the engine process):

    to-pdf EXT     /in/source.<EXT> (docx, xlsx, pptx, epub, html) -> /scratch/out/source.pdf,
                   the PDF that Marker's own provider makes (mammoth, WeasyPrint...), kept
                   instead of deleted, so that it goes through the PDF preparation like any
                   PDF (PB3 of MFL032)
    convert        /in/source.pdf and /scratch/request.json -> /scratch/out/document.md,
                   /scratch/out/images/ and /scratch/result.json

The adapter is bound to marker-pdf 2.2.0 and indra-ocr 1.1.2 (`engine_adapter_mismatch`
otherwise). It works at the Indra bridge, the only place where the information still exists:

- provenance (Δ3): `RecordingIndra._page_to_ocr_result` records for each page Indra's
  `source` (text, hybrid, ocr, error), its note and the text blocks Indra returned, before
  Marker turns `source` into its two methods and a failed page into an empty one;
- figures (Δ2, D14): in `fast` mode Marker never runs its layout model and Indra reads native
  pages without pictures, so a figure of a born-digital PDF is lost; the adapter adds a
  `Picture` block for every raster image object that the PDF preparation found on a `native`
  page and the engine did not judge cornice, in reading order (top, then left), before Marker
  builds the page; Marker then crops it from the rendering of the page;
- the gate, before structure, processors and render: a page without provenance is an adapter
  mismatch; `source=error` or block errors fail the document (`page_failed`); a `native` page
  without text blocks fails it even when a figure was added (PB1); an `ocr_layer` page that
  Indra read from its text layer asks the caller for one run with `force_ocr`; a page without
  blocks is otherwise blank (`page_blank`), or `page_text_lost` for an OCR layer with text.

Progress goes to /scratch/events.jsonl, one JSON object per line: models, provider, every page
read by Indra, structure, every processor, render, write; each with the wall clock (`time`) and
the monotonic clock of the child (`mono`), from which the engine measures stages and pages.

Server mode (`request["mode"]` `balanced` or `accurate`, plan r2 §6, r3 §6, r4 §6). The child
still has no network: `RelayManager` sends Surya's requests through the Unix socket of the relay
of the engine (`/relay/relay.sock`, `httpx` with `uds`, `max_retries=0`, `Limits` on `parallel`),
with Surya's own `chat_completions_batch` and its retries; it is the default manager of Surya and
the manager of every predictor, and the spawning paths of Surya and Marker (`attach_or_spawn`,
`create_model_dict`) are replaced by functions that count and fail, so no backend of their own
can start. `balanced` keeps `RecordingIndra` as layout and Indra; `accurate` (Marker's full-page
OCR on every page) has neither. `RecordingOcrBuilder` records the provenance of every page that
the server read (`indra_source = "server"`) and adds the raster figures of a native page through
a view of the OCR result (`pictures_into`). The gate adds: final errors of the server are
`server_failed` (exit 7); the exceptions of the client equal the `Inference error` lines of Surya,
else an adapter mismatch; no spawn; the blocks Marker drops after a failure or a repetition loop
are counted (`ocr_block_dropped`). A connection to the relay socket is not a network call.
"""
import json
from pathlib import Path
import sys
import time
import traceback

SCRATCH = Path("/scratch")
OUT = SCRATCH / "out"
SOURCE = "/in/source.pdf"
VERSIONS = {"marker-pdf": "2.2.0", "indra-ocr": "1.1.2", "surya-ocr": "0.23.0"}
EXIT_PAGE_FAILED, EXIT_RETRY, EXIT_MISMATCH, EXIT_SERVER_FAILED = 4, 5, 6, 7
LOG_LINES = ("falling back", "inference error")
RELAY = "/relay/relay.sock"
SERVER_MODES = ("balanced", "accurate")
# the lines of Marker 2.2.0 on a block it drops (builders/ocr.py:675, :714, :739, builders/layout.py:263,
# processors/table.py:258)
DROPPED = ("OCR failed for", "Dropping OCR block output", "Layout inference failed for page", "Table OCR failed for block")


NETWORK = []
RELAY_CONNECTS = [0]


def _audit(event, args):
    """Every attempt of a network call in the child, counted (r2.1 §6.7 of MFL032); a connection to
    the relay socket of the `marker-server` route is counted apart."""
    if event == "socket.connect" and len(args) > 1 and args[1] == RELAY:
        RELAY_CONNECTS[0] += 1
        return
    if event in ("socket.connect", "socket.getaddrinfo", "socket.sendto", "socket.sendmsg", "urllib.Request",
                 "http.client.connect"):
        NETWORK.append(event)


def emit(event, **fields):
    with open(SCRATCH / "events.jsonl", "a", encoding="utf-8") as stream:
        stream.write(json.dumps({"event": event, "time": round(time.time(), 3), "mono": round(time.monotonic(), 3),
                                 **fields}) + "\n")


class AdapterMismatch(RuntimeError):
    """Marker or indra-ocr no longer behave as the adapter expects."""


class OcrLayerRead(RuntimeError):
    def __init__(self, page, source):
        super().__init__("page %d: ocr_layer read as %s" % (page, source))
        self.page, self.source = page, source


class PageFailed(RuntimeError):
    def __init__(self, page, note):
        super().__init__("page %d: %s" % (page, note))
        self.page, self.note = page, note


class ServerFailed(RuntimeError):
    """Requests of Surya that still failed after its retries (`server_failed`)."""


# Coordinates of the boxes are pixels of the page as Indra renders it (about 2-3 per point):
# one pixel absorbs the 1/8 point rounding of FPDF_PageToDevice and the float composition of
# the form matrices, nothing more.
TOLERANCE = 1.0


def _inside(a, b):
    """`a` lies entirely inside `b`, within TOLERANCE."""
    return (a[0] >= b[0] - TOLERANCE and a[1] >= b[1] - TOLERANCE
            and a[2] <= b[2] + TOLERANCE and a[3] <= b[3] + TOLERANCE)


def _meets(a, b):
    """`a` and `b` share an area wider and taller than TOLERANCE (boxes that only touch do not)."""
    return min(a[2], b[2]) - max(a[0], b[0]) > TOLERANCE and min(a[3], b[3]) - max(a[1], b[1]) > TOLERANCE


def _union(a, b):
    return [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]


def add_pictures(page, boxes):
    """Insert `Picture` blocks into an Indra page. A box entirely inside a kept crop (or inside a
    picture block of Indra) is part of that figure: its pixels are already in the crop. A box
    that only partly overlaps a kept crop widens it to their union, so no pixel of either is cut
    (two photographs or diagrams overlapping on a slide). The crops go, top then left, before the
    first block below their centre that overlaps them horizontally (the rule of
    marker/ocr/indra.py); a picture block of Indra that a union widened moves by the same rule
    (H1 of the review of the fix of B), the other blocks keep their order. Returns (added,
    contained, merged)."""
    if not boxes:
        return 0, 0, 0
    if not page.blocks and page.lines:
        # without blocks Marker builds them from the lines: materialize them first, or a
        # picture alone would hide the text
        page.blocks = [{"label": "Text", "bbox": ln.box, "text": ln.text} for ln in page.lines
                       if ln.text and ln.text.strip()]
    pictures = [b for b in page.blocks if b.get("label") in ("Picture", "Figure")]
    kept, widened, contained, merged = [], [], 0, 0
    for box in sorted(boxes, key=lambda b: -(b[2] - b[0]) * (b[3] - b[1])):
        box = [float(v) for v in box]
        if (box[2] - box[0]) <= 0 or (box[3] - box[1]) <= 0:
            contained += 1                      # no area: nothing of it is drawn
            continue
        if any(_inside(box, k) for k in kept) or any(_inside(box, b["bbox"]) for b in pictures):
            contained += 1
            continue
        block = next((b for b in pictures if _meets(box, b["bbox"])), None)
        if block is not None:                   # widen the picture of Indra instead of cutting
            block["bbox"] = _union(block["bbox"], box)
            if not any(w is block for w in widened):
                widened.append(block)
            merged += 1
            continue
        joined = False
        while True:
            other = next((k for k in kept if _meets(box, k)), None)
            if other is None:
                break
            kept.remove(other)
            box, joined = _union(box, other), True
        merged += joined
        kept.append(box)
    def place(block):
        box = block["bbox"]
        cy = (box[1] + box[3]) / 2
        at = next((j for j, b in enumerate(page.blocks)
                   if (b["bbox"][1] + b["bbox"][3]) / 2 > cy and min(box[2], b["bbox"][2]) > max(box[0], b["bbox"][0])),
                  len(page.blocks))
        page.blocks.insert(at, block)

    for block in sorted(widened, key=lambda b: (round(b["bbox"][1], 1), b["bbox"][0])):
        del page.blocks[next(j for j, b in enumerate(page.blocks) if b is block)]
        place(block)
    for box in sorted(kept, key=lambda b: (round(b[1], 1), b[0])):
        place({"label": "Picture", "bbox": box})
    return len(kept), contained, merged


def pictures_into(result, boxes):
    """`add_pictures` on a page read by the server (r4 §6, Sol C-ACCURATE): a view of the
    `PageOCRResult`, its blocks as dictionaries `{label, bbox, text, _src}` in reading order; on
    return the original blocks are the same objects (`html`, `error`, `skipped` unchanged, the
    polygon widened when a union asks for it), every new figure is a skipped `Picture` block, and
    the reading order is numbered again. Returns (added, contained, merged)."""
    from surya.recognition.schema import BlockOCRResult

    class View:
        lines = ()
    view = View()
    ordered = sorted(result.blocks, key=lambda b: b.reading_order)
    view.blocks = [{"label": b.label, "bbox": list(b.bbox), "text": b.html, "_src": b} for b in ordered]
    added, contained, merged = add_pictures(view, boxes)
    blocks = []
    for i, item in enumerate(view.blocks):
        x0, y0, x1, y1 = (float(v) for v in item["bbox"])
        polygon = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
        src = item.get("_src")
        if src is None:
            src = BlockOCRResult(polygon=polygon, label="Picture", raw_label="Picture", reading_order=i, html="",
                                 skipped=True, error=False)
        else:
            if [x0, y0, x1, y1] != [float(v) for v in src.bbox]:
                src.polygon = polygon
            src.reading_order = i
        blocks.append(src)
    result.blocks = blocks
    return added, contained, merged


def _recording_indra():
    from marker.ocr.indra import IndraRecognitionPredictor

    class RecordingIndra(IndraRecognitionPredictor):
        """IndraRecognitionPredictor that keeps the provenance of each page and adds the
        figures of native pages. `pictures`: {page index: [[x0, y0, x1, y1] in points of the
        rendered page]}; `page_info`: {page index: {"class", "size", "rotation"}}."""

        def __init__(self, *args, pictures=None, page_info=None, **kwargs):
            super().__init__(*args, **kwargs)
            self.pictures = pictures or {}
            self.page_info = page_info or {}
            self.provenance = {}
            self._pending, self._capture = None, []

        def convert_pdf_pages(self, filepath, page_ids, force_ocr=False):
            self._pending, self._capture = list(page_ids), []
            results, sources = super().convert_pdf_pages(filepath, page_ids, force_ocr=force_ocr)
            if not (len(results) == len(sources) == len(self._capture) == len(page_ids)):
                raise AdapterMismatch("convert_pdf_pages: one result, source and capture per page expected")
            for pid, src, cap, res in zip(page_ids, sources, self._capture, results):
                if cap["indra_source"] != src:
                    raise AdapterMismatch("convert_pdf_pages: source differs from the page read")
                self.provenance[pid] = dict(cap, blocks=len(res.blocks),
                                            block_errors=sum(1 for b in res.blocks if b.error))
            self._pending = None
            return results, sources

        def _page_to_ocr_result(self, page, orig_w, orig_h):
            if self._pending is not None and len(self._capture) < len(self._pending):
                pid = self._pending[len(self._capture)]
                info = self.page_info.get(pid, {})
                source = getattr(page, "source", None)
                text_blocks = sum(1 for b in (page.blocks or []) if b.get("label") not in ("Picture", "Figure")
                                  and (b.get("text") or "").strip()) or sum(
                    1 for ln in (page.lines or []) if ln.text and ln.text.strip())
                added = contained = merged = skipped = 0
                boxes = self.pictures.get(pid, [])
                if info.get("class") == "native" and boxes:
                    if getattr(page, "rotation", 0) or info.get("rotation"):
                        skipped = len(boxes)
                    else:
                        w_pt = info["size"][0]
                        scale = orig_w / w_pt if w_pt else 0
                        added, contained, merged = (add_pictures(page, [[v * scale for v in b] for b in boxes])
                                                    if scale else (0, 0, 0))
                self._capture.append({"indra_source": source, "note": (getattr(page, "note", "") or "")[:300],
                                      "text_blocks": text_blocks, "pictures_added": added,
                                      "pictures_contained": contained, "pictures_merged": merged,
                                      "pictures_skipped_rotated": skipped})
                emit("page", page=pid + 1, pages=len(self._pending), source=source)
            return super()._page_to_ocr_result(page, orig_w, orig_h)

    return RecordingIndra


def _recording_ocr_builder():
    from marker.builders.ocr import OcrBuilder

    class RecordingOcrBuilder(OcrBuilder):
        """OcrBuilder that records the provenance of every page the server read in full and adds
        the figures of its native pages before Marker rebuilds it. The class attributes are set by
        `convert_server` before the converter resolves it."""
        pictures, page_info, provenance, emitted = {}, {}, {}, set()

        def replace_page_structure(self, document, pages, images, recognition_results):
            total = len(document.pages)
            for page, image, result in zip(pages, images, recognition_results):
                pid = page.page_id
                info = self.page_info.get(pid, {})
                added = contained = merged = skipped = 0
                boxes = self.pictures.get(pid, [])
                if info.get("class") == "native" and boxes:
                    if info.get("rotation"):
                        skipped = len(boxes)
                    elif info.get("size") and info["size"][0]:
                        scale = image.size[0] / info["size"][0]
                        added, contained, merged = pictures_into(result, [[v * scale for v in b] for b in boxes])
                blocks = result.blocks
                text_blocks = sum(1 for b in blocks if b.label not in ("Picture", "Figure") and not b.skipped
                                  and not b.error and (b.html or "").strip())
                self.provenance[pid] = {"indra_source": "server", "note": "", "blocks": len(blocks),
                                        "text_blocks": text_blocks, "block_errors": sum(1 for b in blocks if b.error),
                                        "pictures_added": added, "pictures_contained": contained,
                                        "pictures_merged": merged, "pictures_skipped_rotated": skipped}
                if pid not in self.emitted:
                    self.emitted.add(pid)
                    emit("page", page=pid + 1, pages=total, source="server")
            return super().replace_page_structure(document, pages, images, recognition_results)

    return RecordingOcrBuilder


class _Counting:
    """The OpenAI client of the manager, with its exceptions counted (each `Inference error` line of
    Surya comes from one of them)."""

    def __init__(self, manager):
        self.chat = self.completions = self
        self.manager = manager

    def create(self, **kwargs):
        with self.manager.lock:
            self.manager.requests += 1
        try:
            return self.manager.openai.chat.completions.create(**kwargs)
        except Exception:
            with self.manager.lock:
                self.manager.exceptions += 1
            raise


class RelayManager:
    """The inference manager of Surya on the relay socket (see the module docstring)."""

    def __init__(self, parallel, model, allow_stop_token_ids, allow_guided_json, timeout):
        import threading
        import httpx
        from openai import OpenAI
        limits = httpx.Limits(max_connections=parallel, max_keepalive_connections=parallel)
        self.http = httpx.Client(transport=httpx.HTTPTransport(uds=RELAY, limits=limits, retries=0),
                                 base_url="http://relay", trust_env=False, timeout=timeout, limits=limits)
        self.openai = OpenAI(api_key="relay", base_url="http://relay/v1", max_retries=0, http_client=self.http)
        self.parallel, self.model, self.timeout = parallel, model, timeout
        self.allow = {"allow_stop_token_ids": allow_stop_token_ids, "allow_guided_json": allow_guided_json}
        self.lock = threading.Lock()
        self.requests = self.exceptions = self.items = self.final_errors = 0
        self.client = _Counting(self)
        self.method = "relay"

    def generate(self, batch):
        from surya.inference.backends.openai_client import chat_completions_batch
        from surya.settings import settings
        outs = chat_completions_batch(batch, client=self.client, model_name=self.model, timeout=self.timeout,
                                      max_workers=self.parallel, request_logprobs_default=settings.SURYA_INFERENCE_LOGPROBS,
                                      **self.allow)
        with self.lock:
            self.items += len(outs)
            self.final_errors += sum(1 for o in outs if o.error)
        return outs

    def capacity(self):
        return self.parallel

    def start(self):
        return None

    def stop(self):
        return None


class _LineCount:
    """A logging handler counting the records that contain one of `needles`."""

    def __init__(self, needles):
        import logging

        class Handler(logging.Handler):
            def emit(handler, record):
                text = record.getMessage()
                if any(n.lower() in text.lower() for n in needles):
                    self.count += 1
                    if len(self.first) < 5:
                        self.first.append(text[:200])
        self.count, self.first = 0, []
        self.handler = Handler(logging.WARNING)

    def attach(self, logger):
        import logging
        if logger.getEffectiveLevel() > logging.WARNING:
            logger.setLevel(logging.WARNING)
        logger.addHandler(self.handler)
        return self


def gate(document, provenance, page_info, server=False):
    """The checks before the render; returns the page problems of a document that passes. On the
    server route a page read by the server counts as read with OCR."""
    problems = []
    for page in document.pages:
        n = page.page_id + 1
        prov = provenance.get(page.page_id)
        if prov is None:
            raise AdapterMismatch("no provenance for page %d" % n)
        if prov["indra_source"] == "error":
            raise PageFailed(n, prov["note"] or "indra error")
        if prov["block_errors"]:
            raise PageFailed(n, "block errors: %d" % prov["block_errors"])
        info = page_info.get(page.page_id, {})
        cls = info.get("class")
        if cls == "ocr_layer" and prov["indra_source"] not in (("ocr", "server") if server else ("ocr",)):
            raise OcrLayerRead(n, prov["indra_source"])
        if prov["text_blocks"] == 0:
            if cls == "native":
                raise PageFailed(n, "no text from a page with visible text")
            if cls == "ocr_layer" and info.get("invisible_chars", 0) >= 20:
                problems.append({"code": "page_text_lost", "page": n})
            elif prov["blocks"] == 0:
                problems.append({"code": "page_blank", "page": n})
            if page.children is None:
                # Marker 2.2.0 renders a page without blocks only with an empty child list
                page.children = []
        if prov["pictures_skipped_rotated"]:
            problems.append({"code": "pictures_skipped_rotated", "page": n,
                             "detail": "%d raster images" % prov["pictures_skipped_rotated"]})
    return problems


def _converter_class(RecordingIndra, mode="fast", RecordingOcrBuilder=None):
    from marker.builders.document import DocumentBuilder
    from marker.builders.line import LineBuilder
    from marker.builders.ocr import OcrBuilder
    from marker.builders.structure import StructureBuilder
    from marker.converters.pdf import PdfConverter
    from marker.providers.pdf import PdfProvider

    class EnginePdfConverter(PdfConverter):
        """build_document of marker/converters/pdf.py with the gate before the processors and
        the render, a PDF provider only, and one progress event per step."""
        page_info = None
        page_problems = None

        def build_document(self, filepath):
            layout_builder = self.resolve_dependencies(self.layout_builder_class)
            line_builder = self.resolve_dependencies(LineBuilder)
            if mode == "fast":
                ocr_builder = self.resolve_dependencies(OcrBuilder)
                if not isinstance(ocr_builder.indra_model, RecordingIndra):
                    raise AdapterMismatch("the OCR builder does not use RecordingIndra")
                provenance = ocr_builder.indra_model.provenance
            else:
                ocr_builder = self.resolve_dependencies(RecordingOcrBuilder)
                if type(ocr_builder) is not RecordingOcrBuilder:
                    raise AdapterMismatch("the OCR builder is not RecordingOcrBuilder")
                if mode == "balanced" and not isinstance(ocr_builder.indra_model, RecordingIndra):
                    raise AdapterMismatch("balanced without RecordingIndra")
                if mode == "accurate" and ocr_builder.indra_model is not None:
                    raise AdapterMismatch("accurate with an Indra model")
                provenance = RecordingOcrBuilder.provenance
            self.ocr_builder = ocr_builder
            with PdfProvider(filepath, self.config) as provider:
                emit("provider", pages=len(provider))
                document = DocumentBuilder(self.config)(provider, layout_builder, line_builder, ocr_builder)
            self.page_problems = gate(document, provenance, self.page_info or {}, server=mode != "fast")
            emit("structure")
            self.resolve_dependencies(StructureBuilder)(document)
            for processor in self.processor_list:
                emit("processor", name=type(processor).__name__)
                processor(document)
            emit("render")
            self.document = document
            return document

    return EnginePdfConverter


def check_versions():
    from importlib.metadata import version
    found = {name: version(name) for name in VERSIONS}
    if found != VERSIONS:
        raise AdapterMismatch("versions %s, adapter bound to %s" % (found, VERSIONS))
    return {name: version(name) for name in ("marker-pdf", "surya-ocr", "indra-ocr", "torch", "pypdfium2", "onnxruntime")}


def to_pdf(ext):
    """Marker's provider for a born-digital format, its temporary PDF kept in /scratch/out."""
    import shutil
    check_versions()
    from marker.providers.registry import provider_from_ext
    source = "/in/source." + ext
    provider_cls = provider_from_ext(source)
    if provider_cls.__name__ == "PdfProvider":
        raise AdapterMismatch("no Marker provider for ." + ext)
    OUT.mkdir(exist_ok=True)
    emit("provider", name=provider_cls.__name__)
    with provider_cls(source, {"disable_tqdm": True}) as provider:
        shutil.copyfile(provider.temp_pdf_path, OUT / "source.pdf")
    (SCRATCH / "result.json").write_text(json.dumps({"provider": provider_cls.__name__}))
    return 0


def _write(rendered, conv, provenance, versions, request, seconds, extra=None):
    """document.md, images/ and result.json of a conversion."""
    emit("write")
    OUT.mkdir(exist_ok=True)
    (OUT / "images").mkdir(exist_ok=True)
    (OUT / "document.md").write_text(rendered.markdown, encoding="utf-8")
    names = []
    for name, image in rendered.images.items():
        if "/" in name or name.startswith(".") or not name.lower().endswith((".png", ".jpg", ".jpeg")):
            raise AdapterMismatch("image name from Marker: %r" % name[:80])
        with open(OUT / "images" / name, "xb") as stream:
            image.save(stream, format="JPEG" if name.lower().endswith((".jpg", ".jpeg")) else "PNG")
        names.append(name)
    methods = {p.page_id: p.text_extraction_method for p in conv.document.pages}
    result = {"versions": versions, "pages": conv.page_count, "images": names,
              "provenance": {str(k + 1): v for k, v in sorted(provenance.items())},
              "methods": {str(k + 1): v for k, v in methods.items()},
              "page_problems": conv.page_problems or [], "force_ocr": bool(request["force_ocr"]),
              "seconds": seconds, "network_calls": len(NETWORK), **(extra or {})}
    (SCRATCH / "result.json").write_text(json.dumps(result))
    return 0


def convert(request):
    versions = check_versions()
    if request.get("mode", "fast") in SERVER_MODES:
        return convert_server(request, versions)
    import torch
    torch.set_num_threads(request["threads"])
    torch.set_num_interop_threads(1)
    page_info = {int(k): v for k, v in request["pages"].items()}
    pictures = {int(k): v for k, v in request["pictures"].items()}
    RecordingIndra = _recording_indra()
    Converter = _converter_class(RecordingIndra)
    started = time.monotonic()
    emit("models")
    pred = RecordingIndra(threads=request["threads"], model_path=request["indra_dir"], pictures=pictures,
                          page_info=page_info)
    pred.engine                                         # load the ONNX models once, before the pages
    models = {"recognition_model": pred, "layout_model": pred, "fast_layout_model": pred,
              "ocr_error_model": None, "indra_model": pred, "inference_manager": None}
    config = {"mode": "fast", "disable_tqdm": True, "paginate_output": True, "pdftext_workers": 1,
              "force_ocr": bool(request["force_ocr"]), "extract_images": True}
    conv = Converter(artifact_dict=models, config=config)
    conv.page_info = page_info
    loaded = time.monotonic()
    rendered = conv(SOURCE)
    seconds = {"models": round(loaded - started, 3), "conversion": round(time.monotonic() - loaded, 3)}
    return _write(rendered, conv, pred.provenance, versions, request, seconds)


def _no_spawn(counter):
    def refuse(*args, **kwargs):
        counter[0] += 1
        raise AdapterMismatch("a backend of Surya or Marker was about to start")
    return refuse


def convert_server(request, versions):
    """`balanced` or `accurate` with the inference on the server behind the relay (module docstring)."""
    import logging
    import torch
    torch.set_num_threads(request["threads"])
    torch.set_num_interop_threads(1)
    mode, srv = request["mode"], request["server"]
    page_info = {int(k): v for k, v in request["pages"].items()}
    pictures = {int(k): v for k, v in request["pictures"].items()}
    spawns = [0]
    import surya.inference as surya_inference
    import surya.inference.backends.llamacpp as surya_llamacpp
    import surya.inference.backends.spawn as surya_spawn
    import surya.inference.backends.vllm as surya_vllm
    import marker.models as marker_models
    for module in (surya_spawn, surya_vllm, surya_llamacpp):
        if hasattr(module, "attach_or_spawn"):
            module.attach_or_spawn = _no_spawn(spawns)
    marker_models.create_model_dict = _no_spawn(spawns)
    manager = RelayManager(srv["parallel"], srv["model"], srv["allow_stop_token_ids"], srv["allow_guided_json"],
                           srv["timeout"])
    surya_inference._default_manager = manager
    from surya.recognition import RecognitionPredictor
    RecordingOcrBuilder = _recording_ocr_builder()
    started = time.monotonic()
    emit("models")
    rec = RecognitionPredictor(manager)
    pred = RecordingIndra = None
    if mode == "balanced":
        RecordingIndra = _recording_indra()
        pred = RecordingIndra(threads=request["threads"], model_path=request["indra_dir"], pictures=pictures,
                              page_info=page_info)
        pred.engine
    RecordingOcrBuilder.pictures, RecordingOcrBuilder.page_info = pictures, page_info
    RecordingOcrBuilder.provenance = pred.provenance if pred is not None else {}
    RecordingOcrBuilder.emitted = set()
    models = {"recognition_model": rec, "layout_model": pred, "fast_layout_model": pred, "ocr_error_model": None,
              "indra_model": pred, "inference_manager": manager}
    config = {"mode": mode, "disable_tqdm": True, "paginate_output": True, "pdftext_workers": 1,
              "force_ocr": bool(request["force_ocr"]) or mode == "accurate", "extract_images": True}
    conv = _converter_class(RecordingIndra, mode, RecordingOcrBuilder)(artifact_dict=models, config=config)
    conv.page_info = page_info
    surya_log, marker_log = logging.getLogger("surya"), logging.getLogger("marker")
    inference = _LineCount(["inference error"]).attach(surya_log)
    dropped = _LineCount(DROPPED).attach(marker_log)
    fallback = _LineCount(["falling back"]).attach(marker_log)
    fallback.attach(surya_log)
    loaded = time.monotonic()
    try:
        rendered = conv(SOURCE)
    except (PageFailed, AdapterMismatch):
        if manager.final_errors:
            raise ServerFailed("%d requests failed after the retries of Surya" % manager.final_errors) from None
        raise
    seconds = {"models": round(loaded - started, 3), "conversion": round(time.monotonic() - loaded, 3)}
    if inference.handler not in surya_log.handlers or dropped.handler not in marker_log.handlers:
        raise AdapterMismatch("the counters of the logs were removed")
    if surya_inference.get_default_manager() is not manager:
        raise AdapterMismatch("the default manager of Surya is not the relay")
    users = [conv.ocr_builder] + [proc for proc in conv.processor_list if hasattr(proc, "recognition_model")]
    for user in users:
        model = getattr(user, "recognition_model", None)
        if model is not None and getattr(model, "manager", manager) is not manager:
            raise AdapterMismatch("%s does not use the relay" % type(user).__name__)
    if spawns[0] or fallback.count:
        raise AdapterMismatch("spawn attempts %d, fallback lines %d" % (spawns[0], fallback.count))
    if manager.final_errors:
        raise ServerFailed("%d requests failed after the retries of Surya" % manager.final_errors)
    if manager.exceptions != inference.count:
        raise AdapterMismatch("%d exceptions of the client, %d inference error lines" % (manager.exceptions, inference.count))
    server = {"mode": mode, "requests": manager.requests, "items": manager.items, "exceptions": manager.exceptions,
              "final_errors": manager.final_errors, "inference_error_lines": inference.count,
              "dropped_blocks": dropped.count, "dropped_examples": dropped.first, "spawn_attempts": spawns[0],
              "relay_connects": RELAY_CONNECTS[0], "checked_users": [type(u).__name__ for u in users]}
    return _write(rendered, conv, RecordingOcrBuilder.provenance, versions, request, seconds, {"server": server})


def main(args):
    sys.addaudithook(_audit)
    try:
        if args[0] == "to-pdf":
            return to_pdf(args[1])
        return convert(json.loads((SCRATCH / "request.json").read_text()))
    except PageFailed as exc:
        (SCRATCH / "result.json").write_text(json.dumps({"error": "page_failed", "page": exc.page, "note": exc.note[:300]}))
        print("page_failed: page %d: %s" % (exc.page, exc.note[:300]), flush=True)
        return EXIT_PAGE_FAILED
    except OcrLayerRead as exc:
        (SCRATCH / "result.json").write_text(json.dumps({"error": "ocr_layer_read", "page": exc.page, "source": exc.source}))
        return EXIT_RETRY
    except ServerFailed as exc:
        (SCRATCH / "result.json").write_text(json.dumps({"error": "server_failed", "detail": str(exc)[:300]}))
        print("server_failed: %s" % exc, flush=True)
        return EXIT_SERVER_FAILED
    except AdapterMismatch as exc:
        (SCRATCH / "result.json").write_text(json.dumps({"error": "engine_adapter_mismatch"}))
        print("engine_adapter_mismatch: %s" % exc, flush=True)
        return EXIT_MISMATCH
    except MemoryError:
        raise
    except Exception:
        traceback.print_exc()
        sys.stdout.flush()
        return 1
