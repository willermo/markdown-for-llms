"""Adapter minimo CPU a003, chiamato solo dopo sonde e preflight dal driver."""
import json
from pathlib import Path
import sys
import time
import importlib.util


def safety_relative(directory):
    # Adding the benchmark directory to sys.path shadows stdlib profile.py,
    # which torch/transformers load via cProfile. Bind only the needed helper.
    spec=importlib.util.spec_from_file_location('a003_pdf_safety',Path(directory)/'safety.py')
    helper=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    return helper.relative


def docling_markdown(document, root):
    from docling_core.types.doc import ImageRefMode
    # A relative artifacts_dir makes native links relative even with an
    # absolute output filename. Keep native serializer defaults for content.
    document.save_as_markdown(root/'document.md',artifacts_dir=Path('document_artifacts'),
                              image_mode=ImageRefMode.REFERENCED)


def main():
    import os
    paths = list(Path('/proc/self/fd').iterdir())
    entry_fds = {p.name: os.readlink(p) for p in paths if p.exists()}
    if any('memfd:a003-consumer-admission' in value for value in entry_fds.values()) or 'A003_ADMISSION_FD' in os.environ:
        raise ValueError('Admission FD leaked into engine')
    cfg=json.loads(Path(sys.argv[1]).read_text())
    root=Path(cfg['output'])
    (root/'engine-entry-fds.json').write_text(json.dumps({'fds':entry_fds,'admission_fd_absent':True})+'\n')
    relative=safety_relative(cfg['benchmark_helpers'])
    if cfg.get('operation')=='docling-ir-serialization':
        from docling_core.types.doc import DoclingDocument
        document=DoclingDocument.model_validate_json(Path(cfg['native_ir']).read_text(encoding='utf-8'))
        docling_markdown(document,root)
        (root/'metadata.json').write_text(json.dumps({'operation':'native Docling serialization of retained same-inference IR',
            'source_ir':cfg['native_ir'],'source_conversion_attempt':cfg['source_conversion_attempt'],
            'new_inference':False,'page_count':len(document.pages),'artifact_reference_policy':'relative'},indent=2)+'\n')
        return 0
    import torch
    torch.set_num_threads(cfg['threads'])
    torch.set_num_interop_threads(1)
    start=time.monotonic()
    if cfg['engine']=='marker':
        from marker.converters.pdf import PdfConverter
        from marker.models import create_model_dict
        from marker.output import text_from_rendered
        from marker.settings import settings
        from surya.settings import settings as surya_settings
        options=dict(cfg['options'])
        if cfg.get('page_range') is not None:
            options['page_range']=cfg['page_range']
        converter=PdfConverter(config=options,artifact_dict=create_model_dict(device='cpu'))
        # Same pipeline as __call__, retaining the document of this inference.
        document=converter.build_document(cfg['input'])
        converter.page_count=len(document.pages)
        rendered=converter.resolve_dependencies(converter.renderer)(document)
        text,_,images=text_from_rendered(rendered)
        (root/'document.md').write_bytes(text.encode('utf-8',errors='strict'))
        # Buffered PIL page/crop images are not JSON values. The structural IR
        # keeps all blocks; the native renderer's extracted assets remain below.
        image_fields={'lowres_image':True,'highres_image':True}
        excluded={'pages':{'__all__':dict(image_fields,children={'__all__':image_fields})}}
        (root/'native-ir.json').write_text(document.model_dump_json(exclude=excluded),encoding='utf-8')
        for name,image in images.items():
            rel=relative(name)
            path=root/rel;path.parent.mkdir(parents=True,exist_ok=True)
            if path.is_symlink() or any(p.is_symlink() for p in path.parents):
                raise ValueError('Image output symlink')
            image.save(path)
        metadata={'settings':settings.model_dump(mode='python'),'surya_settings':surya_settings.model_dump(mode='python'),
                  'rendered_metadata':rendered.metadata,'options':options,'page_count':len(document.pages),
                  'ir_scope':'native structural model; page/block lowres_image/highres_image buffers excluded; renderer assets preserved'}
    elif cfg['engine']=='docling':
        from docling.document_converter import DocumentConverter, PdfFormatOption
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions, TesseractCliOcrOptions
        from docling.datamodel.accelerator_options import AcceleratorOptions, AcceleratorDevice
        options=PdfPipelineOptions(artifacts_path=Path(cfg['artifacts_path']),do_ocr=True,do_table_structure=True,
                                   generate_picture_images=True,generate_page_images=False,do_formula_enrichment=False,
                                   enable_remote_services=False,allow_external_plugins=False,
                                   ocr_options=TesseractCliOcrOptions(lang=['ita','eng'],force_full_page_ocr=False),
                                   accelerator_options=AcceleratorOptions(num_threads=cfg['threads'],device=AcceleratorDevice.CPU))
        converter=DocumentConverter(allowed_formats=[InputFormat.PDF],format_options={InputFormat.PDF:PdfFormatOption(pipeline_options=options)})
        kwargs={'page_range':(min(cfg['page_range'])+1,max(cfg['page_range'])+1)} if cfg.get('page_range') is not None else {}
        result=converter.convert(cfg['input'],raises_on_error=False,**kwargs)
        document=result.document
        (root/'native-ir.json').write_text(document.model_dump_json(),encoding='utf-8')
        # Native serializer owns the bundle; raw never receives cleaning/AI fixes.
        docling_markdown(document,root)
        metadata={'options':options.model_dump(mode='json',serialize_as_any=True),'status':result.status.value,'errors':[str(e) for e in result.errors],
                  'page_count':len(document.pages),'formula_enrichment':'disabled; semantic/typographic observations remain separate'}
        if str(result.status.value).lower()!='success':
            (root/'metadata.json').write_text(json.dumps(metadata,indent=2,default=str)+'\n')
            return 2
    else:
        raise ValueError('Engine not selected')
    metadata.update(elapsed_child_s=time.monotonic()-start,torch_threads=torch.get_num_threads(),torch_interop_threads=torch.get_num_interop_threads())
    (root/'metadata.json').write_text(json.dumps(metadata,indent=2,default=str)+'\n')
    return 0


if __name__=='__main__':raise SystemExit(main())
