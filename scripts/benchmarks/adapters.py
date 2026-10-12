"""Registry locale esplicito; validazione prima dei tool e argv senza shell."""
from pathlib import Path
import re
import shutil
import sys
from data import descriptor, digest, digest_bytes, write_new
from resources import run
from pandoc_guard import check as pandoc_check
from safety import regular, zip_members, relative, Unsafe

REGISTRY={'txt-literal-v1':{'txt'},'pandoc-direct-v1':{'docx','epub','rtf','tex'},'pandoc-legacy-v1':{'docx','epub','rtf'}}
INCLUDES=('sections/one.tex','sections/two.tex')


def tex_validate(content):
    text=content.decode('utf-8',errors='strict')
    # Consume a control symbol (including \\ row separator) as one token:
    # the second backslash must not become a command before the next cell.
    commands={m[1] for m in re.finditer(r'\\(?:\\|[^A-Za-z]|([A-Za-z]+))',text) if m[1]}
    allowed={'documentclass','usepackage','begin','end','section','label','input','include','includegraphics','caption','bibitem','ref','eqref','cite','href','frac','sum','text'}
    if commands-allowed:raise Unsafe('TeX command outside finite source subset: '+','.join(sorted(commands-allowed)))
    for classname in re.findall(r'\\documentclass(?:\[[^]]*\])?\{([^{}]+)\}',text):
        if classname!='article':raise Unsafe('Document class not allowlisted')
    for env in re.findall(r'\\(?:begin|end)\{([^{}]+)\}',text):
        if env not in {'document','figure','table','tabular','aligned','bmatrix','matrix','thebibliography'}:raise Unsafe('TeX environment not allowlisted')
    if commands & {'write','openin','openout','read','catcode','def','gdef','edef','xdef','newcommand','renewcommand','csname','directlua','immediate','special','url','shellescape'}:raise Unsafe('Dynamic TeX forbidden')
    for package in re.findall(r'\\usepackage(?:\[[^]]*\])?\{([^{}]+)\}',text):
        if package not in ('amsmath','graphicx','hyperref'):raise Unsafe('TeX package not allowlisted')
    includes=re.findall(r'\\(?:input|include)\{([^{}]+)\}',text)
    for name in includes:
        relative(name)
        if name not in INCLUDES:raise Unsafe('Include not allowlisted')
    if len(re.findall(r'\\(?:input|include)\b',text))!=len(includes):raise Unsafe('Nonliteral include')
    for path in re.findall(r'\\includegraphics(?:\[[^]]*\])?\{([^{}]+)\}',text):relative(path)
    return includes


def expand_tex(root,main,allowlist=INCLUDES):
    total=0;seen=[]
    def expand(name,active):
        nonlocal total
        relative(name)
        if name in active:raise Unsafe('TeX include cycle')
        payload=regular(root,name,256*1024);total+=len(payload)
        if total>256*1024:raise Unsafe('TeX expanded bytes limit')
        text=payload.decode();tex_validate(payload)
        def replace(match):
            path=match[1]
            if path not in allowlist or path in seen:raise Unsafe('Duplicate/outside include')
            seen.append(path);return expand(path,active+[name])
        return re.sub(r'\\(?:input|include)\{([^{}]+)\}',replace,text)
    result=expand(main,[])
    if tuple(seen)!=tuple(allowlist):raise Unsafe('Expected two includes in literal order')
    return result.encode(),{'method':'literal-include-v1','includes':seen,'expanded_bytes':len(result.encode()),'source_compile':False}


def validate(corpus,document,engine):
    if engine not in REGISTRY or document['format'] not in REGISTRY[engine]:raise Unsafe('Unregistered tool/format')
    payload=regular(corpus,document['input'])
    if digest_bytes(payload)!=document['input_sha256']:raise Unsafe('Source hash STALE')
    format=document['format'];related={}
    if format in ('docx','epub'):
        members=zip_members(payload)
        if format=='epub':
            from safety import xml,local_uri
            for name,data in members.items():
                if data is not None and name.endswith('.opf'):
                    for item in xml(data).iter():
                        if item.tag.endswith('}item') and item.get('href'):local_uri(item.get('href'),False)
    elif format=='tex':
        parent=Path(document['input']).parent
        if tex_validate(payload)!=list(INCLUDES):raise Unsafe('Main includes contract')
        for name in INCLUDES:
            data=regular(corpus,(parent/name).as_posix());tex_validate(data);related[name]=data
        for name,data in [('main.tex',payload),*related.items()]:
            for path in re.findall(rb'\\includegraphics\{([^{}]+)\}',data):
                key=path.decode();relative(key);related[key]=regular(corpus,(parent/key).as_posix())
    elif format=='rtf':
        if not payload.startswith(b'{\\rtf1') or any(x in payload.lower() for x in (b'\\object',b'\\objdata',b'\\filetbl',b'include_picture',b'includepicture')):raise Unsafe('RTF active/unsupported object')
        payload.decode('ascii',errors='strict')
    elif format=='txt':payload.decode('utf-8',errors='strict')
    return payload,related


def convert(corpus,document,engine,config,raw,environment,installation):
    # Called only after validate: INVALID_INPUT does not spawn an engine.
    payload,related=validate(corpus,document,engine)
    expected=config['config_sha256'];copy={k:v for k,v in config.items() if k!='config_sha256'}
    if digest(copy)!=expected or descriptor('/usr/bin/pandoc')['sha256']!=config['pandoc']['sha256']:raise ValueError('Tool/config tampering')
    token_before=pandoc_check(environment,config['pandoc'],config['pandoc_version_required']) if engine!='txt-literal-v1' else None
    raw.mkdir();source=raw/'source';source.mkdir();(raw/'data').mkdir()
    input=source/('main.tex' if document['format']=='tex' else 'input.'+document['format']);input.write_bytes(payload)
    for name,data in related.items():
        p=source/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
    source_hashes=[descriptor(p) for p in source.rglob('*') if p.is_file()]
    expansion=None
    if document['format']=='tex':
        content,expansion=expand_tex(source,'main.tex');input=source/'expanded.tex';input.write_bytes(content);source_hashes.append(descriptor(input))
    legacy=False
    if engine=='txt-literal-v1':argv=[sys.executable,'-I','-B',str(Path(__file__).with_name('literal_child.py')),str(input),str(raw/'document.md')]
    elif engine=='pandoc-direct-v1':argv=['/usr/bin/pandoc',str(input),*config['ordered_options'],'--output=document.md']
    else:
        legacy=True;child={'input':str(input),'options':config['ordered_options'],'output':str(raw/'document.md'),'receipt':str(raw/'legacy-result.json'),'installation':str(installation),'installation_sha256':descriptor(installation)['sha256'],'pandoc':config['pandoc'],'timeout_s':60,'pandoc_resolution_before':token_before,'pandoc_version_required':config['pandoc_version_required']}
        write_new(raw/'legacy-config.json',child);argv=[sys.executable,'-I','-B',str(Path(__file__).with_name('legacy_child.py')),str(raw/'legacy-config.json')]
    resources=run(argv,raw,raw/'process',environment,60)
    token_after=pandoc_check(environment,config['pandoc'],config['pandoc_version_required']) if engine!='txt-literal-v1' else None
    if token_before!=token_after:raise ValueError('Pandoc token changed during attempt')
    after=descriptor('/usr/bin/pandoc')
    if after['sha256']!=config['pandoc']['sha256']:raise ValueError('Pandoc changed during attempt')
    for row in source_hashes:
        if descriptor(row['path'])['sha256']!=row['sha256']:raise ValueError('Source copy changed during conversion')
    return {'resources':resources,'engine_exit':None if legacy else resources['adapter_exit'],'argv_observed':argv,'argv_reconstructed':(['pandoc',str(input),*config['ordered_options'],'-o',str(input.with_suffix('.temp.md'))] if legacy else None),'legacy_internal_argv_observed':False if legacy else None,'raw_text_origin':'returned-text-serialized; original temp removed by API' if legacy else 'engine-output-file','expansion':expansion,'source_copies':source_hashes,'latency_series':'legacy end-to-end child/API' if legacy else 'direct process including launcher for TXT','installation':descriptor(installation),'pandoc_before':config['pandoc'],'pandoc_after':after,'pandoc_resolution_before':token_before,'pandoc_resolution_after':token_after}
