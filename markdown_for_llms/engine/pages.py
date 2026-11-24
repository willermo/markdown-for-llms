"""The pages family on the Marker route: Marker 2.2.0 `fast` on the CPU.

Every input becomes one PDF first, so that it goes through the same preparation and the same
bridge with the page provenance (Δ3 of MFL032): a PDF as it is; png, jpeg and tiff through
`pdfprep image` in the light child; djvu through `ddjvu -format=pdf` in the toolchain image;
docx, xlsx, pptx, epub and html (the Marker alternatives of the born-digital formats, scenario
5) through Marker's own provider in the Marker child, its PDF kept (PB3).

Then `pdfprep classify` gives the class of each page and the raster images of the native
pages, and the plan (D3 of MFL032, with r2.1 and PB6):

- `--ocr force`, or no native page at all: Marker with `force_ocr`;
- only native pages: Marker `fast` without `force_ocr` (Indra still reads a broken text layer
  with OCR and says so: `hybrid`, recorded per page);
- native pages with `ocr_layer` pages: their invisible text is removed (`pdfprep strip`); the
  clean copy goes to Marker when every such page is now `image` and renders the same; when the
  rendering changed the copy is discarded and the original goes with `force_ocr` (PB6,
  `ocr_forced_document`); when a page is still `ocr_layer` the original goes without it, and the
  gate of the child asks for a single run with `force_ocr` if Indra reads such a page from its
  text layer (PB7, the reason `ocr_layer_not_removable`).

Figures (Δ2, D14, PB12): every raster image of a native page becomes a `Picture` block, so an
asset, unless it is a fill of 1x1 pixels or cornice (`furniture`: the repetitions of an image
drawn on many pages, the images of a running header or footer). Cornice is reported once per
group as `media_furniture` (`warning`, as D11); every raster keeps its outcome in `engine.json`.
A raster entirely inside another kept crop is part of that figure; one that partly overlaps it
widens the crop to their union (no pixel cut, FC1 of the review of B).

The child runs in Bubblewrap without network, the model root read-only, under a systemd scope
with the memory cap of the conversion; the weights are verified before each start. Its events
carry a monotonic clock (`mono`): `timings.stages` and `timings.pages` of `engine.json` come
from the events of the child whose result is kept (`marker_timings`).
"""
import collections
import json
import re
import shutil
import statistics
import time

from . import contract, isolation, weights
from .codes import Code, EngineError

PREP = "markdown_for_llms.engine.pdfprep"
CHILD = "markdown_for_llms.engine.marker_child"
FURNITURE_PAGES = 3
FURNITURE_BAND = 0.12
PREP_MEMORY_MIB = 2048
MAX_PIXELS = 100_000_000
LIMITS = ["Marker 2.2.0 fast with indra-ocr v1.1.0 neither extracts nor detects vector figures (also the "
          "vector panels of a composite figure) nor the figures of scanned pages: they are missing from the "
          "output without a problem of the document (D13 of MFL032)."]
# The limits of the marker-server route (MFL033), in words the user reads in engine.json and in the summary:
# no internal reference. The second one (N6): on 21/11 D04 of the reference set in balanced differed in 6 lines
# between two conversions without the cache of torch.compile on the same A10G of Modal (a letter, the case of
# a word, the cells of a table on several lines), and D06 differed in its bytes between RunPod and Modal
# (vLLM 0.25.1, bf16); D18 was the same in 22 conversions.
SERVER_LIMITS = ["The marker-server route runs Marker 2.2.0 balanced or accurate with the OCR of Surya on the "
                 "inference server the user names: the page images go to that server; the quality of the route is "
                 "not certified.",
                 "On a GPU server two conversions of the same document can differ in a few places, also on the same "
                 "GPU with the same settings, because the reading of the server is not exactly repeatable: a letter, "
                 "the case of a word, or the part of a table cell written on several lines that is kept, as measured "
                 "in balanced mode on a scientific article (two conversions on the same GPU) and on a statistical "
                 "report (two GPU servers). Compare two outputs by their content, not by their bytes; the accurate "
                 "mode has not been measured for this."]
# The texts declared by earlier commits of MFL033, still in the engine.json of the outputs made with them: the
# bench report recognizes them (`bench.report._known_limits`); the engine never declares them again.
FORMER_LIMITS = ("The marker-server route (MFL033) runs Marker 2.2.0 balanced or accurate with the OCR of Surya on "
                 "the inference server the user names: the page images go to that server; the quality of the route "
                 "is not certified.",
                 "In balanced mode Marker 2.2.0 rebuilds the tables on this machine from the lines that the server "
                 "reads (reconstruct_table_html): a table cell written on several lines can lose part of its text or "
                 "have it moved to another cell, as measured on the table of a scientific article with the CPU "
                 "server and with the GPU servers. For such tables use the accurate mode, where the server reads "
                 "the whole page; its result on them has not been measured.")
# R10 of MFL033 (plan r2 §6): D04 of the reference set, measured on the local CPU server and on the GPU
# servers of RunPod and Modal; `accurate` has no local rebuild of the cells (Marker's table processor
# returns no rebuilt table in that mode, and the child has no Indra there).
BALANCED_SERVER_LIMITS = ["In balanced mode Marker 2.2.0 rebuilds the tables on this machine from the lines that the "
                          "server reads: a table cell written on several lines can lose part of its text or have it "
                          "moved to another cell, as measured on the table of a scientific article with the CPU server "
                          "and with the GPU servers. For such tables use the accurate mode, where the server reads the "
                          "whole page; its result on them has not been measured."]
SEPARATOR = re.compile(r"^\{\d+\}-{48}$", re.M)
BAD_LOG = re.compile(rb"falling back|inference error", re.I)
EXITS = {4: Code.PAGE_FAILED, 6: Code.ENGINE_ADAPTER_MISMATCH, 3: Code.MEMORY_LIMIT, 7: Code.SERVER_FAILED}
FALLING_BACK = re.compile(rb"falling back", re.I)
TO_PDF = {"docx": "docx", "xlsx": "xlsx", "pptx": "pptx", "epub": "epub", "html": "html"}


def _prep(ctx, name, args, inputs, memory_mib=PREP_MEMORY_MIB):
    scratch = ctx.work / ("pdfprep-" + name)
    shutil.rmtree(scratch, ignore_errors=True)
    rc, log = isolation.run(scratch, PREP, args, inputs=inputs, memory_mib=min(memory_mib, ctx.memory_mib),
                            output_mib=ctx.output_mib, timeout=min(1800, ctx.timeout()), steps=ctx.steps,
                            observer=ctx.observer, poll_seconds=ctx.poll_seconds)
    if rc == 3:
        raise EngineError(Code.MEMORY_LIMIT, "pdfprep " + name)
    if rc != 0:
        err = EngineError(Code.INPUT_UNREADABLE if name == "classify" else Code.CONVERTER_FAILED, "pdfprep %s exit %s" % (name, rc))
        err.log = log
        raise err
    return scratch


def _scope(ctx, n):
    return "mdf-eng-%s-marker-%d" % (ctx.attempt[:16], n)


def child_env(model_root, threads):
    m = weights.manifest()
    return {"HOME": "/scratch/home", "TMPDIR": "/scratch/tmp", "XDG_CACHE_HOME": "/scratch/tmp",
            "TORCH_DEVICE": "cpu", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
            "SURYA_INFERENCE_AUTOSTART": "false", "TOKENIZERS_PARALLELISM": "false",
            "FONT_PATH": "/models/" + weights.font(m), "INDRA_OCR_MODEL_DIR": "/models/" + weights.indra_dir(m),
            "OMP_NUM_THREADS": str(threads), "MKL_NUM_THREADS": str(threads)}


def _relay_stop(check):
    def stop():
        code = check()
        return None if code is None else EngineError(code, "relay")
    return stop


def run_child(ctx, args, inputs, request=None, binds=(), check=None, activity=None):
    """One Marker child; returns (exit status, scratch). Weights verified before the start. On the
    `marker-server` route `binds` adds the relay directory on `/relay` and `check` stops the child
    with the failure of the relay; an `Inference error` line is then a retried request, counted by
    the child, and only `falling back` fails the log."""
    report = weights.verify(ctx.model_root)
    ctx.weights = report
    ctx.child_runs += 1
    ctx.child_events = []                       # the timings come from the child whose result is kept
    scratch = ctx.work / ("marker-%d" % ctx.child_runs)
    for sub in ("home", "tmp", "out"):
        (scratch / sub).mkdir(parents=True, exist_ok=True, mode=0o700)
    if request is not None:
        (scratch / "request.json").write_text(json.dumps(request))
    rc, log = isolation.run(
        scratch, CHILD, args, inputs=inputs, memory_mib=ctx.memory_mib, output_mib=ctx.output_mib,
        timeout=ctx.timeout(), steps=ctx.steps, observer=ctx.observer, poll_seconds=ctx.poll_seconds,
        ro_binds=[(str(ctx.model_root), "/models"), *binds], env=child_env(ctx.model_root, ctx.threads), fonts=True,
        scope=_scope(ctx, ctx.child_runs), process_limit=512, files_limit=1024,
        address_space_mib=ctx.memory_mib * 2 + 8192, cpu_seconds=max(60, ctx.timeout()) * max(1, ctx.threads),
        stall_seconds=ctx.stall_seconds, on_events=ctx.feed_events,
        check=None if check is None else _relay_stop(check), activity=activity)
    ctx.diagnostics["marker-%d.log" % ctx.child_runs] = log[-contract.LOG_TAIL:]
    if rc != 0 and rc not in (4, 5):
        err = EngineError(EXITS.get(rc, Code.CONVERTER_FAILED), "marker child exit %s" % rc)
        err.log = log
        raise err
    server = (request or {}).get("mode") in ("balanced", "accurate")
    if rc == 0 and (FALLING_BACK if server else BAD_LOG).search(log):
        err = EngineError(Code.ENGINE_ADAPTER_MISMATCH if server else Code.CONVERTER_FAILED,
                          "marker reported a fallback" + ("" if server else " or an inference error"))
        err.log = log
        raise err
    return rc, scratch


def source_pdf(ctx, src, fmt):
    """The PDF of the input (see the module docstring)."""
    if fmt == "pdf":
        return src
    if fmt in ("png", "jpeg", "tiff"):
        ext = {"png": "png", "jpeg": "jpg", "tiff": "tiff"}[fmt]
        ctx.event("image-to-pdf")
        scratch = _prep(ctx, "image", ["image", fmt, ctx.max_pages, MAX_PIXELS], [(src, "source." + ext)])
        info = json.loads((scratch / "image.json").read_text())
        if "error" in info:
            if info["error"].startswith("page_limit:"):
                raise EngineError(Code.PAGE_LIMIT, info["error"])
            raise EngineError(Code.INPUT_UNREADABLE, "image: " + info["error"])
        return scratch / "source.pdf"
    if fmt == "djvu":
        from . import tools
        ctx.event("ddjvu")
        dest = ctx.work / "ddjvu" / "source.pdf"
        dest.parent.mkdir(parents=True, exist_ok=True)
        tools.run("ddjvu", ["-format=pdf", tools.In(src, name="source.djvu"), tools.Out(dest)], ctx.tool_root,
                  timeout=ctx.timeout(), memory_mb=ctx.memory_mib, watch=ctx.watch())
        return dest
    if fmt in TO_PDF:
        ctx.event("provider")
        rc, scratch = run_child(ctx, ["to-pdf", TO_PDF[fmt]], [(src, "source." + TO_PDF[fmt])])
        out = scratch / "out" / "source.pdf"
        if rc != 0 or not out.is_file():
            raise EngineError(Code.CONVERTER_FAILED, "Marker provider for " + fmt)
        return out
    raise EngineError(Code.ROUTE_INVALID, fmt)


def _ranges(numbers):
    """[1, 2, 3, 5] -> '1-3, 5'."""
    out, run = [], []
    for n in sorted(set(numbers)):
        if run and n == run[-1] + 1:
            run.append(n)
            continue
        if run:
            out.append(str(run[0]) if len(run) == 1 else "%d-%d" % (run[0], run[-1]))
        run = [n]
    if run:
        out.append(str(run[0]) if len(run) == 1 else "%d-%d" % (run[0], run[-1]))
    return ", ".join(out)


def furniture(pages):
    """The outcome of every raster image of the native pages (D14 and PB12 of MFL032).

    Returns (outcomes, groups): `outcomes` {(page, index): "cornice" | "fill"} for the rasters
    that do not become a figure (all the others do), `groups` [(case, pages, repetitions,
    kept_on)] the cornice groups, one `media_furniture` each. A raster of 1x1 pixels is a fill
    of plain colour drawn as an image, never a figure (`fill`, no problem: it is no loss). The
    others are cornice in three cases:

    1. the same image (stored data and size) on at least FURNITURE_PAGES pages, not always in
       the band: the first occurrence stays a figure, at its place, and only the repetitions
       are cornice (a figure repeated on consecutive slides, as Beamer overlays draw it, is
       kept once; the worst cost is one spurious logo of a side bar or a watermark);
    2. the same image on at least two pages, always inside the header or footer band
       (FURNITURE_BAND of the page height at the top or at the bottom): every use;
    3. an image inside the band at the same place (2 pt) on at least FURNITURE_PAGES pages,
       whatever its content (the per-page codes or stamps of a running footer): every use."""
    by_digest, by_place = collections.defaultdict(list), collections.defaultdict(list)
    outcomes = {}
    for p in pages:
        if p["class"] != "native":
            continue
        h = p["size"][1] or 1
        for i, r in enumerate(p["rasters"]):
            if r.get("pixels") == [1, 1]:
                outcomes[(p["page"], i)] = "fill"
                continue
            band = r["box"][3] <= FURNITURE_BAND * h or r["box"][1] >= (1 - FURNITURE_BAND) * h
            if r["digest"]:
                by_digest[r["digest"]].append((p["page"], i, band))
            if band:
                by_place[tuple(round(v / 2) for v in r["box"])].append((p["page"], i, band))
    groups = []
    for case, table in ((None, by_digest), (3, by_place)):
        for uses in table.values():
            uses = sorted(uses)
            on = sorted({page for page, _, _ in uses})
            kept = None
            if case == 3:
                if len(on) < FURNITURE_PAGES:
                    continue
                marked = uses
            elif len(on) >= 2 and all(b for _, _, b in uses):
                case_here, marked = 2, uses
            elif len(on) >= FURNITURE_PAGES:
                case_here, marked, kept = 1, uses[1:], uses[0][0]
            else:
                continue
            fresh = [(page, i) for page, i, _ in marked if (page, i) not in outcomes]
            if not fresh:
                continue
            for item in fresh:
                outcomes[item] = "cornice"
            groups.append((case if case else case_here, sorted({page for page, _ in fresh}), len(fresh), kept))
    return outcomes, groups


CORNICE_CASES = {1: "the same image repeated", 2: "the same image in the header or footer",
                 3: "an image at the same place of the header or footer"}


def classify(ctx, pdf):
    ctx.event("classify")
    scratch = _prep(ctx, "classify", ["classify", ctx.max_pages], [(pdf, "source.pdf")])
    info = json.loads((scratch / "pdfprep.json").read_text())
    if info["pages"] == 0:
        raise EngineError(Code.INPUT_UNREADABLE, "no pages")
    if info["pages"] > ctx.max_pages:
        raise EngineError(Code.PAGE_LIMIT, "%d pages, limit %d" % (info["pages"], ctx.max_pages))
    return info["page"]


def keep_layer_classes(original, cleaned):
    """The pages of the clean copy with the class and the invisible characters of the original
    (FC2 of the review of B): a page that had an OCR layer stays `ocr_layer` for the gate, which
    declares `page_text_lost` when OCR finds nothing where the layer had text; the class of the
    copy is kept apart in `class_after_strip`."""
    by_page = {p["page"]: p for p in original}
    out = []
    for p in cleaned:
        before = by_page.get(p["page"])
        if before is not None and before["class"] == "ocr_layer":
            p = dict(p, class_after_strip=p["class"], **{"class": "ocr_layer", "invisible_chars": before["invisible_chars"]})
        out.append(p)
    return out


def plan(ctx, pdf, pages, ocr):
    """(pdf for Marker, pages of that pdf, force_ocr, reason, strip report)."""
    classes = [p["class"] for p in pages]
    if ocr == "force":
        return pdf, pages, True, "requested", None
    if "native" not in classes:
        return pdf, pages, True, "no_native_pages", None
    layer = [p["page"] for p in pages if p["class"] == "ocr_layer"]
    if not layer:
        return pdf, pages, False, "native_pages", None
    ctx.event("strip")
    scratch = _prep(ctx, "strip", ["strip", ",".join(map(str, layer))], [(pdf, "source.pdf")])
    report = json.loads((scratch / "strip.json").read_text())
    if report["removable"]:
        clean = scratch / "stripped.pdf"
        return clean, keep_layer_classes(pages, classify(ctx, clean)), False, "ocr_layer_removed", report
    if report["rendering_changed"]:
        ctx.problems.append(contract.problem("ocr_forced_document", detail="the copy without the OCR layer renders "
                                             "differently on pages %s" % report["rendering_changed"]))
        return pdf, pages, True, "ocr_layer_not_removable", report
    return pdf, pages, False, "ocr_layer_kept", report


def rename_images(text, names, ext_of=None, problems=None):
    """Image names of the converter -> assets/asset-NNNN.<ext>, in the Markdown images and the
    `<img>` tags found by the parser (`imagerefs`, MFL043), each replacement counted against the
    parse. `ext_of` {name: ext} (from the bytes); by default the extension of the name, jpg or
    png. The problems of the `<img>` tags without an extracted source go to `problems` when
    given. Returns (text, mapping)."""
    from . import imagerefs
    text, mapping, found = imagerefs.rename(text, names, ext_of=ext_of)
    if problems is not None:
        problems.extend(found)
    return text, mapping


def figures(ctx, pages):
    """The figures of the native pages for the child, the outcome of every raster and the
    `media_furniture` problems: (page_info, pictures, rasters), by page index."""
    outcomes, groups = furniture(pages)
    for case, on, repetitions, kept in groups:
        detail = "raster cornice (%s): %d uses on pages %s" % (CORNICE_CASES[case], repetitions, _ranges(on))
        if kept is not None:
            detail += "; the first use, on page %d, is kept as a figure" % kept
        ctx.problems.append(contract.problem("media_furniture", page=on[0], detail=detail))
    page_info, pictures, rasters = {}, {}, {}
    for p in pages:
        i = p["page"] - 1
        page_info[i] = {"class": p["class"], "size": p["size"], "rotation": p["rotation"],
                        "invisible_chars": p["invisible_chars"]}
        # every raster has an outcome in engine.json (PB2): a figure of the page, cornice, a fill
        # of 1x1 pixels, or the image of a page that is not native (read with OCR)
        if p["class"] == "native":
            ends = [outcomes.get((p["page"], j), "picture") for j in range(len(p["rasters"]))]
            keep = [r["box"] for r, end in zip(p["rasters"], ends) if end == "picture"]
            if keep:
                pictures[i] = keep
        else:
            ends = ["page_image"] * len(p["rasters"])
        rasters[i] = dict(collections.Counter(ends), total=len(ends))
    return page_info, pictures, rasters


# stage -> (event that starts it, events that may end it, the first found)
STAGE_BOUNDS = (("models", "models", ("provider",)), ("pages", "provider", ("structure",)),
                ("structure", "structure", ("processor", "render")), ("processors", "processor", ("render",)),
                ("render", "render", ("write",)))
PAGE_TIMES = {"unit": "s", "clock": "monotonic, Marker child",
              "method": "interval between consecutive page events, the first from the provider event",
              "covers": "the reading of the page by Indra (layout, lines, OCR); not structure, processors and render"}


def marker_timings(events, max_pages):
    """`stages` (seconds of models, pages, structure, processors and render) and `pages` (the
    seconds of each page read by Indra, with median and maximum) from the `mono` clock of the
    events of the Marker child; a stage whose events are missing is left out, and `pages` is
    None without a provider event or without page events. The seconds per page of the whole
    conversion are `total_seconds / pages`, not these values."""
    marks = [(e.get("event"), e["mono"]) for e in events
             if isinstance(e, dict) and type(e.get("mono")) in (int, float)]
    first = {}
    for name, at in marks:
        first.setdefault(name, at)
    stages = {}
    for stage, start, ends in STAGE_BOUNDS:
        end = next((first[e] for e in ends if e in first), None)
        if start in first and end is not None and end >= first[start]:
            stages[stage] = round(end - first[start], 3)
    pages = None
    reads = [at for name, at in marks if name == "page"]
    if "provider" in first and reads:
        values, previous = [], first["provider"]
        for at in reads[:max_pages]:
            values.append(round(max(0.0, at - previous), 3))
            previous = at
        pages = dict(PAGE_TIMES, count=len(values), median=round(statistics.median(values), 3),
                     max=max(values), values=values)
    return {"stages": stages, "pages": pages}


def marker_route(ctx, src, fmt, request):
    ocr = request.get("ocr") or "auto"
    pdf = source_pdf(ctx, src, fmt)
    pages = classify(ctx, pdf)
    pdf, pages, force, reason, strip_report = plan(ctx, pdf, pages, ocr)
    page_info, pictures, rasters = figures(ctx, pages)
    child_request = {"threads": ctx.threads, "indra_dir": "/models/" + weights.indra_dir(), "force_ocr": force,
                     "pages": page_info, "pictures": pictures}
    ctx.event("marker")
    rc, scratch = run_child(ctx, ["convert"], [(pdf, "source.pdf")], child_request)
    retried = None
    if rc == 5:
        first = json.loads((scratch / "result.json").read_text())
        retried = {"page": first.get("page"), "source": first.get("source")}
        ctx.problems.append(contract.problem("ocr_forced_document", page=first.get("page"),
                                             detail="an OCR layer read as %s" % first.get("source")))
        force, reason = True, "ocr_layer_not_removable"
        rc, scratch = run_child(ctx, ["convert"], [(pdf, "source.pdf")], dict(child_request, force_ocr=True))
        if rc == 5:
            raise EngineError(Code.PAGE_FAILED, "an OCR layer page read from its text layer under force_ocr")
    result = json.loads((scratch / "result.json").read_text())
    if rc == 4:
        raise EngineError(Code.PAGE_FAILED, "page %s: %s" % (result.get("page"), result.get("note")))
    return collect(ctx, scratch, result, pages, page_info, rasters, force, reason, strip_report, retried)


def server_limits(mode):
    """The declared limits of the marker-server route in `mode` (a new list): those of the route, then the
    cells on several lines (R10) in `balanced` only."""
    return SERVER_LIMITS + (BALANCED_SERVER_LIMITS if mode == "balanced" else [])


def collect(ctx, scratch, result, pages, page_info, rasters, force, reason, strip_report, retried, limits=LIMITS):
    n = len(pages)
    prov, methods = result["provenance"], result["methods"]
    if result["pages"] != n or len(prov) != n or len(methods) != n:
        raise EngineError(Code.CONVERTER_FAILED, "Marker returned %s pages of %d" % (result["pages"], n))
    records = []
    for p in pages:
        k = str(p["page"])
        pr = prov[k]
        if pr["indra_source"] == "error" or not methods.get(k):
            raise EngineError(Code.PAGE_FAILED, "page %s without a method" % k)
        layer = {"invisible_chars": p["invisible_chars"]} if p["class"] == "ocr_layer" else {}
        if "class_after_strip" in p:
            layer["class_after_strip"] = p["class_after_strip"]
        records.append({"page": p["page"], "class": p["class"], **layer,
                        "indra_source": pr["indra_source"],
                        "marker_method": methods[k], "note": pr["note"], "blocks": pr["blocks"],
                        "text_blocks": pr["text_blocks"], "rasters": rasters[p["page"] - 1],
                        "pictures_added": pr["pictures_added"], "pictures_contained": pr["pictures_contained"],
                        "pictures_merged": pr.get("pictures_merged", 0)})
    for item in result["page_problems"]:
        ctx.problems.append(contract.problem(item["code"], page=item.get("page"), detail=item.get("detail")))
    raw = (scratch / "out" / "document.md").read_bytes()
    text = contract.check_document(raw)
    if not SEPARATOR.sub("", text).strip():
        raise EngineError(Code.OUTPUT_EMPTY, "only page separators")
    ctx.diagnostics["marker-original.md"] = raw
    diag = ctx.stage / "diagnostics"
    (diag / "marker-original.md").write_bytes(raw)
    for name, data in ctx.diagnostics.items():
        if name.endswith(".log"):
            (diag / name).write_bytes(data[-contract.LOG_TAIL:])
    text, mapping = rename_images(text, result["images"], problems=ctx.problems)
    (ctx.stage / "assets").mkdir(exist_ok=True)
    for name, new in mapping.items():
        shutil.copyfile(scratch / "out" / "images" / name, ctx.stage / new)
    (ctx.stage / "document.md").write_text(text, encoding="utf-8")
    added = sum(r["pictures_added"] for r in records)
    if len(mapping) < added:
        # a Picture block that Marker did not render as an image (for example a raster of a few
        # pixels): the image is part of the page but not of the output
        ctx.problems.append(contract.problem("media_lost", detail="%d of %d figures of native pages not rendered by "
                                             "Marker" % (added - len(mapping), added)))
    ctx.pages = {"count": n, "classes": dict(collections.Counter(p["class"] for p in pages)), "pages": records}
    ctx.timings = marker_timings(ctx.child_events, ctx.max_pages)
    ctx.limits.extend(limits)
    ctx.marker_versions = result["versions"]
    if result.get("network_calls"):
        raise EngineError(Code.ENGINE_ADAPTER_MISMATCH, "network calls in the Marker child: %d" % result["network_calls"])
    return {"force_ocr": force, "reason": reason, "retried_with_force_ocr": retried,
            "strip": strip_report, "images": len(mapping), "seconds": result["seconds"],
            "rasters": dict(sum((collections.Counter(r["rasters"]) for r in records), collections.Counter())),
            "network_calls": result.get("network_calls", 0)}


def marker_server_route(ctx, src, fmt, request, ep, keys, transport_factory=None, cpu=None):
    """Marker `balanced` or `accurate` with Surya on the inference server `ep` (MFL033).

    The pages are prepared as on the Marker route (`accurate` reads every page with OCR, without
    the plan of the text layers). Before anything wakes the server: the transit of a server
    outside the loopback (`transit_not_confirmed`), then the term of the attempt and, for an
    endpoint with a provider, its reservation at the event `server-wake`, which the caller turns
    into a row of the ledger or a refusal. Then the private socket directory, the relay, its
    readiness and the child with the relay on `/relay`; the relay closes in a `finally` and its
    counters and times go to `engine.json` (`marker_server`), with the measured bound of the cost
    for a provider (`server_cost_bound`). The `cpu` endpoint starts the local CPU server first
    (`cpuserver`, verified before the wake; `cpu` = {"root", "manifest"} replaces its defaults in
    the tests), in its own sandbox and scope, on the `llama/` socket of the same directory."""
    from . import relay as relay_mod, servers, sockets
    from .codes import Code as C
    opts = request.get("server") or {}
    mode = opts.get("mode")
    cpu = dict(cpu or {})
    if ep.kind == "cpu":
        from . import cpuserver
        cpu["manifest"] = cpu.get("manifest") or cpuserver.manifest()
        cpu["record"] = cpuserver.verify(cpu.get("root"), cpu["manifest"])
    pdf = source_pdf(ctx, src, fmt)
    pages = classify(ctx, pdf)
    if mode == "accurate":
        force, reason, strip_report = True, "server_accurate", None
    else:
        pdf, pages, force, reason, strip_report = plan(ctx, pdf, pages, request.get("ocr") or "auto")
    page_info, pictures, rasters = figures(ctx, pages)
    leaves = ep.leaves_machine()
    ctx.server = {"endpoint": ep.record(), "mode": mode, "versions": None, "leaves_machine": leaves}
    if ep.kind == "cpu":
        ctx.server["cpu"] = dict(cpu["record"])
    ctx.transit = {"left_machine": False, "service": ep.provider or ep.kind, "endpoint": ep.host()}
    if leaves and opts.get("transit_confirmed") is not True:
        raise EngineError(C.TRANSIT_NOT_CONFIRMED, "the page images would leave the machine for %s" % ep.host())
    session = opts.get("session") or {}
    term = servers.term(ep, len(pages), budget_cents=opts.get("budget_cents"), until=session.get("armed_until"),
                        now=time.time())
    ctx.server["term"] = {k: term[k] for k in ("D", "d_doc", "d_min", "d_budget", "d_session")}
    if ep.costs:
        ctx.cost = {"provider": ep.provider, "rate_usd_per_s": ep.rate_usd_per_s, "arm_id": session.get("arm_id"),
                    "budget_cents": term["budget_cents"], "reserve_cents": term["reserve_cents"], "bound_cents": None,
                    "billed_cents": None, "clean": False}
    # MFL023 B2 (C-B2 of the plan r1): the canonical URL of the endpoint, which the reservation compares with the URL
    # of the window it draws from
    ctx.event("server-wake", reserve={"provider": ep.provider, "endpoint": ep.name, "identity": ep.identity_sha256(),
                                      "url": None if ep.kind == "cpu" else servers.canonical_url(ep.url),
                                      "reserve_cents": term["reserve_cents"], "rate": ep.rate_usd_per_s,
                                      "tail_idle_s": ep.tail_idle_s, "D": term["D"]})
    sockets.sweep()
    private = sockets.Private()
    relay = started = None
    done = False
    try:
        if ep.kind == "cpu":
            from . import cpuserver, isolation
            ctx.event("cpu-server")
            started = cpuserver.start(cpu["record"]["root"], private.llama_dir, "mdf-eng-%s-llama" % ctx.attempt[:16],
                                      ctx.work / "llama-server.log", cpu["manifest"])
        relay = relay_mod.Relay(ep, keys, private.relay_socket, transport_factory=transport_factory,
                                upstream_socket=private.llama_socket if ep.kind == "cpu" else None)
        relay.start()
        if leaves:
            ctx.transit["left_machine"] = True            # from the first request on
        relay.set_term(term["D"], until=session.get("armed_until"))
        relay.ready()
        child_request = {"threads": ctx.threads, "indra_dir": "/models/" + weights.indra_dir(), "force_ocr": force,
                         "pages": page_info, "pictures": pictures, "mode": mode,
                         "server": {"parallel": ep.parallel, "model": ep.model, "timeout": ep.read_timeout,
                                    **ep.requests()}}
        binds = [(str(private.relay_dir), "/relay")]
        ctx.event("marker")
        rc, scratch = run_child(ctx, ["convert"], [(pdf, "source.pdf")], child_request, binds, relay.check,
                                relay.activity)
        retried = None
        if rc == 5:
            first = json.loads((scratch / "result.json").read_text())
            retried = {"page": first.get("page"), "source": first.get("source")}
            ctx.problems.append(contract.problem("ocr_forced_document", page=first.get("page"),
                                                 detail="an OCR layer read as %s" % first.get("source")))
            force, reason = True, "ocr_layer_not_removable"
            rc, scratch = run_child(ctx, ["convert"], [(pdf, "source.pdf")], dict(child_request, force_ocr=True), binds,
                                    relay.check, relay.activity)
            if rc == 5:
                raise EngineError(C.PAGE_FAILED, "an OCR layer page read from its text layer under force_ocr")
        result = json.loads((scratch / "result.json").read_text())
        if rc == 4:
            raise EngineError(C.PAGE_FAILED, "page %s: %s" % (result.get("page"), result.get("note")))
        done = True
    except EngineError as exc:
        if relay is not None and relay.failure is not None and exc.code != relay.failure:
            # the child failed because the server went away or the term came: that is the cause
            err = EngineError(relay.failure, "relay: %s" % exc.code.value)
            err.log = getattr(exc, "log", None)
            raise err from None
        raise
    finally:
        if relay is not None:
            relay.close()
            record = relay.record()
            clean = bool(record["clean"] and done)
            record["clean"] = clean
            ctx.marker_server = record
            ctx.server["versions"] = dict(relay.versions)
            if ep.costs:
                ctx.cost["bound_cents"] = servers.bound_cents(ep, relay.ready_at is not None, clean, record["t_wake"],
                                                              record["t_last"], record["t_close"])
                ctx.cost["clean"] = clean
                ctx.problems.append(contract.problem("server_cost_bound", detail="%s: a measured upper bound of %s "
                                                     "cents, not a billed amount" % (ep.provider, ctx.cost["bound_cents"])))
        if started is not None:
            from . import isolation
            step = isolation.stop(started)
            ctx.steps.append(dict(step, tool="bwrap:llama-server", args=[]))
            ctx.server["cpu"]["run"] = step
            try:
                ctx.diagnostics["llama-server.log"] = started.log_path.read_bytes()[-contract.LOG_TAIL:]
            except OSError:
                pass
        private.close()
    out = collect(ctx, scratch, result, pages, page_info, rasters, force, reason, strip_report, retried,
                  limits=server_limits(mode))
    srv = result.get("server") or {}
    if srv.get("exceptions"):
        record = ctx.marker_server or {}
        failed = {k: v for k, v in (record.get("by_status") or {}).items() if not k.startswith("2")}
        ctx.problems.append(contract.problem("server_retried", detail="%d requests failed and succeeded when retried; "
                                             "answers %s, transport errors %d, timeouts %d"
                                             % (srv["exceptions"], failed or "{}", record.get("transport_errors", 0),
                                                record.get("timeouts", 0))))
    if srv.get("dropped_blocks"):
        ctx.problems.append(contract.problem("ocr_block_dropped", detail="%d blocks dropped by Marker: %s"
                                             % (srv["dropped_blocks"], "; ".join(srv.get("dropped_examples") or []))))
    out["server"] = srv
    return out
