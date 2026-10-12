"""Revisione solo metadati/trascrizioni: mai genera canonici, PDF o render."""
import argparse
from copy import deepcopy
from pathlib import Path
from data import canonical,descriptor,digest_bytes,load,write_new
from generate_corpus import transcript


def policy(format):
    return {'epub':(1,'encoded-th-v1'),'control-markdown':(1,'authored-pipe-header-v1'),'docx':(0,'encoded-no-tblHeader-v1'),'rtf':(0,'encoded-no-trhdr-v1'),'tex':(0,'author-declared-headerless-tabular-v1'),'pdf':(0,'author-declared-headerless-drawn-grid-v1')}[format]


def headers(blocks,format):
    for b in blocks:
        if b['type']=='table':b['header_rows'],b['header_policy']=policy(format)


def revise(corpus,old,receipt):
    corpus=corpus.absolute();old=old.absolute()
    original={p.relative_to(old).as_posix():descriptor(p) for p in old.rglob('*') if p.is_file()}
    current={p.relative_to(corpus).as_posix():descriptor(p) for p in corpus.rglob('*') if p.is_file()}
    # Only README may have an independent new administrative note at entry.
    if set(original)!=set(current) or any(original[k]['sha256']!=current[k]['sha256'] for k in original if k!='README.md'):raise ValueError('Revision entry is not preserved rev4')
    changed=[]
    def save(path,value):
        path.write_bytes(canonical(value)+b'\n');changed.append(path.relative_to(corpus).as_posix())
    spec=load(corpus/'sources/documents.json');spec['corpus_revision']=5
    for d in spec['documents']:headers(d['blocks'],d['format'])
    save(corpus/'sources/documents.json',spec)
    manifest=load(corpus/'manifest.json');manifest['revision']=5
    for d in manifest['documents']:
        g=load(corpus/d['gold']);headers(g['blocks'],d['format'])
        g['metadata_revision']={'revision':5,'previous':descriptor(old/d['gold']),'change':'explicit-source-header-semantics-only; original V inherited only by separate invariant receipt'}
        save(corpus/d['gold'],g);d['blocks']=deepcopy(g['blocks']);d['gold_sha256']=descriptor(corpus/d['gold'])['sha256']
        path=corpus/'gold'/d['document_id']/'transcript.txt';content=transcript(g).encode()
        if path.read_bytes()!=content:path.write_bytes(content);changed.append(path.relative_to(corpus).as_posix())
    cspec=load(corpus/'controls/source.json')
    for d in cspec['controls']:headers(d['blocks'],'control-markdown')
    save(corpus/'controls/source.json',cspec)
    cm=load(corpus/'controls/manifest.json');cm['metadata_revision']=5;cm['source']=descriptor(corpus/'controls/source.json')
    for d in cm['documents']:
        g=load(corpus/'controls'/d['gold']);headers(g['blocks'],'control-markdown');g['metadata_revision']={'revision':5,'previous':descriptor(old/'controls'/d['gold']),'change':'explicit-source-header-semantics-only'}
        save(corpus/'controls'/d['gold'],g);d['gold_sha256']=descriptor(corpus/'controls'/d['gold'])['sha256']
    save(corpus/'controls/manifest.json',cm)
    manifest['provenance']['metadata_revision']={'revision':5,'previous_manifest':descriptor(old/'manifest.json'),'helper':descriptor(__file__),'generator_current':descriptor(Path(__file__).with_name('generate_corpus.py')),'spec':descriptor(corpus/'sources/documents.json'),'source_and_render_regenerated':False,'transcript_change':'11 table captions in nine documents; no other visible annotation changes'}
    save(corpus/'manifest.json',manifest)
    after={p.relative_to(corpus).as_posix():descriptor(p) for p in corpus.rglob('*') if p.is_file()}
    immutable=[k for k in original if k not in changed and k!='README.md']
    if any(original[k]['sha256']!=after[k]['sha256'] for k in immutable):raise ValueError('Unexpected source/asset/render mutation')
    # Independently remove only the technical additions from every gold before
    # comparing all original fields, including source locators and denominators.
    for k in [k for k in changed if k.endswith('annotations.json')]:
        before=load(old/k);after_gold=load(corpus/k);after_gold.pop('metadata_revision')
        for b in after_gold['blocks']:b.pop('header_rows',None);b.pop('header_policy',None)
        if after_gold!=before:raise ValueError('Visible annotation changed: '+k)
    value={'schema_version':1,'status':'PASS','helper':descriptor(__file__),'old_root':str(old),'new_root':str(corpus),'changed':[{'path':k,'before':original[k],'after':after[k]} for k in changed],'unchanged':[{'path':k,'before':original[k],'after':after[k]} for k in immutable],'visible_annotation_invariant':True,'source_render_asset_bytes_invariant':True,'human_new_confirmation':False}
    write_new(receipt,value);print('METADATA_REVISION_5',len(changed),len(immutable))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('corpus','old','receipt'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();revise(a.corpus,a.old,a.receipt)

if __name__=='__main__':main()
