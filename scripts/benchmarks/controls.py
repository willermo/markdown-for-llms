"""Materializza controlli manuali del misuratore; nessun output motore."""
import argparse
from copy import deepcopy
from pathlib import Path
import re
import shutil
from data import canonical, descriptor, digest_bytes, load, provenance, write_new, counts


def build(spec, corpus):
    root=corpus/'controls';root.mkdir(exist_ok=False)
    shutil.copyfile(spec,root/'source.json')
    documents=[]
    for d in load(spec)['controls']:
        directory=root/d['document_id'];directory.mkdir()
        (directory/'assets').mkdir()
        shutil.copyfile(corpus/'assets/figure-it.png',directory/'assets/figure-it.png')
        body=bytearray();blocks=deepcopy(d['blocks'])
        for b in blocks:
            literal=b.pop('literal').encode()
            b['locator']={'kind':'control-literal','start':len(body),'length':len(literal),'literal':literal.decode()}
            body.extend(literal);body.extend(b'\n')
            values=[b['text']]+[c for row in b.get('cells',[]) for c in row]
            b['numbers']=[{'text_index':i,'offset':m.start(),'lexeme':m[0]} for i,v in enumerate(values) for m in re.finditer(r'[−-]?[0-9]+(?:[.,][0-9]+)?',v)]
        (directory/'document.md').write_bytes(body)
        gold={'schema_version':1,'document_id':d['document_id'],'canonical_sha256':digest_bytes(body),'blocks':blocks,'note_order_policy':'definitions-collected-at-end','verification':{'status':'PROVISIONAL','M':'PENDING','V':'PENDING','kind':'human_visual','verifier':None},'authoring_method':d['authoring_method']}
        gold['denominators']=counts(gold);write_new(directory/'annotations.json',gold)
        documents.append({'document_id':d['document_id'],'format':'control-markdown','input':d['document_id']+'/document.md','gold':d['document_id']+'/annotations.json','input_sha256':digest_bytes(body),'gold_sha256':descriptor(directory/'annotations.json')['sha256'],'denominators':counts(gold),'license':'MIT'})
    write_new(root/'manifest.json',{'schema_version':1,'controls':True,'documents':documents,'source':provenance(root/'source.json')})
    print('CONTROLS_CREATED',len(documents))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--spec',type=Path,required=True);p.add_argument('--corpus',type=Path,required=True)
    a=p.parse_args();build(a.spec,a.corpus)

if __name__=='__main__':main()
