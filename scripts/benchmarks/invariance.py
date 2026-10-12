"""Controllo indipendente del delta tecnico rev4→rev5; nessun serializer/scorer."""
from copy import deepcopy
from pathlib import Path
from data import descriptor,load
from safety import regular

POLICIES={'epub':(1,'encoded-th-v1'),'control-markdown':(1,'authored-pipe-header-v1'),'docx':(0,'encoded-no-tblHeader-v1'),'rtf':(0,'encoded-no-trhdr-v1'),'tex':(0,'author-declared-headerless-tabular-v1'),'pdf':(0,'author-declared-headerless-drawn-grid-v1')}


def gold_invariant(old,new,format):
    projected=deepcopy(new);revision=projected.pop('metadata_revision',None)
    if not revision or revision.get('revision')!=5:raise ValueError('Metadata revision absent')
    delta=[]
    for b in projected['blocks']:
        if b['type']=='table':
            expected=POLICIES[format];actual=(b.pop('header_rows',None),b.pop('header_policy',None))
            if actual!=expected:raise ValueError('Technical header policy mismatch')
            delta.append({'block_id':b['block_id'],'header_rows':actual[0],'header_policy':actual[1],'human_confirmation_new':False})
    if projected!=old:raise ValueError('Visible annotation/source locator/denominator changed')
    return {'status':'PASS','rule':'all-original-gold-fields-equal-except-explicit-header-v1','delta':delta,'invariants':'text, Unicode, numbers, AST/raw formulas, order, cell presence/content, captions, assets, notes, references, source locators, denominators','new_human_confirmation':False}


def corpus_invariant(old,new):
    old=Path(old);new=Path(new);before=load(old/'manifest.json');after=load(new/'manifest.json')
    if before['revision']!=4 or after['revision']!=5:raise ValueError('Revision identity')
    documents=[]
    for control in (False,True):
        a=old/'controls' if control else old;b=new/'controls' if control else new
        ads=load(a/'manifest.json')['documents'];bds=load(b/'manifest.json')['documents']
        if [d['document_id'] for d in ads]!=[d['document_id'] for d in bds]:raise ValueError('Document set/order changed')
        for ad,bd in zip(ads,bds):
            if regular(a,ad['input'])!=regular(b,bd['input']):raise ValueError('Canonical source changed')
            if descriptor(a/ad['gold'])['sha256']!=ad['gold_sha256'] or descriptor(b/bd['gold'])['sha256']!=bd['gold_sha256']:raise ValueError('Gold identity STALE')
            proof=gold_invariant(load(a/ad['gold']),load(b/bd['gold']),bd['format'])
            documents.append({'document_id':bd['document_id'],'control':control,'old_gold':descriptor(a/ad['gold']),'new_gold':descriptor(b/bd['gold']),'canonical_sha256':bd['input_sha256'],**proof})
    # Independently check every original non-metadata file, including ignored
    # PDF, assets and fixed views, not just manifest-listed main sources.
    immutable=[]
    for path in sorted(old.rglob('*')):
        if not path.is_file():continue
        name=path.relative_to(old).as_posix()
        metadata=name in ('manifest.json','sources/documents.json','controls/manifest.json','controls/source.json','README.md') or name.endswith('/annotations.json') or name.endswith('/transcript.txt')
        if not metadata:
            if regular(old,name)!=regular(new,name):raise ValueError('Source/asset/render changed: '+name)
            immutable.append({'path':name,'old':descriptor(path),'new':descriptor(new/name)})
    return {'schema_version':1,'status':'PASS','checker':descriptor(__file__),'old_root':str(old),'new_root':str(new),'documents':documents,'immutable_files':immutable,'new_human_confirmation':False}


def validate_delta(path):
    value=load(path)
    if value.get('status')!='PASS' or value.get('visible_annotation_invariant') is not True:raise ValueError('Metadata delta not PASS')
    old=Path(value['old_root']);new=Path(value['new_root'])
    # The promotion preserves the new-copy bytes; its descriptor is part of
    # the run input. Current corpus is checked again independently at derive.
    for row in value['changed']:
        if descriptor(old/row['path'])!=row['before'] or descriptor(new/row['path'])!=row['after']:raise ValueError('Metadata delta STALE')
    corpus_invariant(old,new)
    return value
