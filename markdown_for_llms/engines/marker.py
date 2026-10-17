"""Direct adapter for the pinned Marker CPU profile, run only inside bwrap."""
from importlib.metadata import version
from pathlib import Path
import json
import time

from ..files import identify, relative, write, write_json


def main(args):
    start = time.monotonic()
    config = json.loads(Path("/scratch/request.json").read_text())
    import pypdfium2 as pdfium
    pdf = pdfium.PdfDocument("/source.pdf")
    try:
        pages = len(pdf)
        if pages == 0 or pages > config["max_pages"]:
            raise ValueError("PDF page limit or empty document")
    finally:
        pdf.close()
    import torch
    torch.set_num_threads(config["threads"])
    torch.set_num_interop_threads(1)
    from marker.converters.pdf import PdfConverter
    from marker.models import create_model_dict
    from marker.output import text_from_rendered
    from marker.builders.line import LineBuilder
    from marker.builders.ocr import OcrBuilder
    from marker.builders.document import DocumentBuilder
    from marker.processors.table import TableProcessor
    from marker.providers.pdf import PdfProvider

    opts = config["options"]
    native_options = dict(opts)
    native_options["extract_images"] = not native_options.pop("disable_image_extraction")
    converter = PdfConverter(config=native_options, artifact_dict=create_model_dict(device="cpu"))
    line = converter.resolve_dependencies(LineBuilder)
    ocr = converter.resolve_dependencies(OcrBuilder)
    layout = converter.resolve_dependencies(converter.layout_builder_class)
    doc_builder = DocumentBuilder(opts)
    renderer = converter.resolve_dependencies(converter.renderer)
    provider = PdfProvider("/source.pdf", opts)
    table = next(p for p in converter.processor_list if isinstance(p, TableProcessor))
    effective = dict(use_llm=converter.use_llm, force_ocr=provider.force_ocr,
                     strip_existing_ocr=provider.strip_existing_ocr,
                     pdftext_workers=provider.pdftext_workers,
                     disable_image_extraction=not renderer.extract_images,
                     paginate_output=renderer.paginate_output,
                     lowres_image_dpi=doc_builder.lowres_image_dpi,
                     highres_image_dpi=doc_builder.highres_image_dpi,
                     layout_batch_size=layout.get_batch_size(),
                     recognition_batch_size=ocr.get_recognition_batch_size(),
                     detection_batch_size=line.get_detection_batch_size(),
                     ocr_error_batch_size=line.get_ocr_error_batch_size(),
                     table_rec_batch_size=table.get_table_rec_batch_size())
    if effective != opts:
        raise ValueError("Effective Marker options differ from requested profile")
    loaded = time.monotonic()
    document = converter.build_document("/source.pdf")
    converter.page_count = len(document.pages)
    rendered = renderer(document)
    text, _, images = text_from_rendered(rendered)
    result = Path("/scratch/original")
    result.mkdir()
    write(result, "document.md", text.encode("utf-8", errors="strict"))
    if not text.strip() or len(document.pages) != pages:
        raise ValueError("Empty output or omitted PDF pages")
    names = []
    for name, image in images.items():
        relative(name)
        if name == "document.md" or name in names:
            raise ValueError("Asset name collision")
        (result / name).parent.mkdir(parents=True, exist_ok=True)
        with (result / name).open("xb") as stream:
            image.save(stream, format="JPEG" if name.lower().endswith((".jpg", ".jpeg")) else "PNG")
        names.append(name)
    image_fields = {"lowres_image": True, "highres_image": True}
    excluded = {"pages": {"__all__": dict(image_fields, children={"__all__": image_fields})}}
    write(Path("/scratch"), "native-ir.json", document.model_dump_json(exclude=excluded).encode())
    # Only explicitly selected metadata leave the child. Native IR remains a
    # diagnostic artifact; no settings/environment or credential fields are dumped.
    page_stats = [{"page_id": p.page_id, "text_extraction_method": p.text_extraction_method,
                   "llm_error_count": p.aggregate_block_metadata().llm_error_count} for p in document.pages]
    issues = [{"code": "engine_error", "page_id": p['page_id']} for p in page_stats if p['llm_error_count']]
    metadata = dict(engine="marker", versions={n: version(n) for n in ("marker-pdf", "surya-ocr", "torch")},
                    requested_options=opts, native_options=native_options,
                    resolved_defaults={"device": "cpu", "torch_interop_threads": 1},
                    effective_options=effective, pages=pages, output_pages=len(document.pages),
                    startup_seconds=loaded - start, conversion_seconds=time.monotonic() - loaded,
                    torch_threads=torch.get_num_threads(), torch_interop_threads=torch.get_num_interop_threads(),
                    ir_scope="Native same-inference structure; low/high-resolution image buffers excluded; diagnostic only",
                    inventory=[identify(result, n) for n in ["document.md", *names]],
                    inventory_complete=True, issues=issues, page_stats=page_stats)
    write_json(Path("/scratch"), "engine.json", metadata)
    return 0
