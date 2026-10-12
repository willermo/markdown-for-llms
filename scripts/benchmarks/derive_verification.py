"""Receipt derivate: V storica su contenuto invariato + nuovo M tecnico."""
import argparse
from pathlib import Path
from data import descriptor,digest,load,write_new
from document_response import validate_response
from packet import verify_packet
from comparison_check import check_document,check_stylesheet,check_epub_dependency
from invariance import corpus_invariant


def derive(old_corpus,corpus,old_packet,packet,response,old_verification,output):
    old=verify_packet(old_packet);new=verify_packet(packet);received=validate_response(response)
    if received['packet_sha256_received']!=old['packet_sha256'] or received['status']!='ACCEPTED':raise ValueError('Original actual V not ACCEPTED for old packet')
    if [p['document_id'] for p in old['identities']]!=[p['document_id'] for p in new['identities']]:raise ValueError('Pair set changed')
    invariant=corpus_invariant(old_corpus,corpus);check_stylesheet(old_packet,old);check_stylesheet(packet,new)
    output.mkdir(exist_ok=False);write_new(output/'corpus-invariant.json',invariant)
    mf=[load(packet/'technical'/n) for n in ('M-gold.json','M-controls.json')];mrows=[d for m in mf for d in m['documents']];records=[r for m in mf for r in m['records']]
    results=[]
    for pair,prior_pair in zip(new['identities'],old['identities']):
        ident=pair['document_id'];control=pair['control'];root=corpus/'controls' if control else corpus;old_root=old_corpus/'controls' if control else old_corpus
        d=next(d for d in load(root/'manifest.json')['documents'] if d['document_id']==ident);od=next(d for d in load(old_root/'manifest.json')['documents'] if d['document_id']==ident)
        g=load(root/d['gold']);og=load(old_root/od['gold']);source_proof=next(d for d in invariant['documents'] if d['document_id']==ident)
        if source_proof['new_gold']['sha256']!=pair['gold_sha256'] or source_proof['old_gold']['sha256']!=prior_pair['gold_sha256'] or pair['canonical_sha256']!=prior_pair['canonical_sha256']:raise ValueError('Pair/gold invariant identity mismatch')
        if pair['scope']!=prior_pair['scope']:raise ValueError('V scope changed')
        visual=next(v for v in received['documents'] if v['document_id']==ident)
        historic=load(old_verification/(ident+'.json'))
        if historic.get('status')!='VERIFIED' or any(historic.get(k)!=prior_pair[k] or visual.get(k)!=prior_pair[k] for k in ('document_id','canonical_sha256','gold_sha256','pair_sha256','scope')):raise ValueError('Old V qualification STALE')
        mechanical=next(m for m in mrows if m['document_id']==ident)
        required=[r for r in records if r['doc_id']==ident and r.get('required')]
        if mechanical['status']!='PASS' or mechanical['gold_sha256']!=pair['gold_sha256'] or mechanical['canonical_sha256']!=pair['canonical_sha256'] or len(required)!=mechanical['required'] or not required or any(r['status']!='PASS' for r in required):raise ValueError('Current independent M not PASS')
        for table in [b for b in g['blocks'] if b['type']=='table']:
            if not any(r['check_id'].startswith(table['block_id']+':semantic-header') for r in required):raise ValueError('Technical header M absent')
        previous_view=check_document(old_packet,od,og,old_root,control,require_original=True)
        current_view=check_document(packet,d,g,root,control,require_original=True)
        if previous_view['status']!='PASS' or current_view['status']!='PASS':raise ValueError('All approved visible contents not proved invariant')
        if d['format']=='epub':check_epub_dependency(packet,new,pair)
        write_new(output/(ident+'-old-view.json'),previous_view);write_new(output/(ident+'-new-view.json'),current_view)
        value={'schema_version':3,'kind':'derived_reference_content_qualification','granularity':'document-content-inheritance','document_id':ident,'canonical_sha256':pair['canonical_sha256'],'gold_sha256':pair['gold_sha256'],'pair_sha256':pair['pair_sha256'],'corpus_revision':5,'scope':pair['scope'],'status':'VERIFIED','qualification_basis':'real old V, independently proved unchanged content, current M; no new human response','M':mechanical,'technical_header_qualified_by':'current independent M/source policy, not user V','V_inheritance':{'original_response':descriptor(response),'original_received_sha256':received['original_received_sha256'],'original_receipt':received['receipt'],'original_document_decision':visual,'old_verification':descriptor(old_verification/(ident+'.json')),'old_packet':descriptor(old_packet/'packet.json'),'old_pair_sha256':prior_pair['pair_sha256'],'new_human_confirmation':False},'new_packet':descriptor(packet/'packet.json'),'invariance':source_proof,'corpus_invariant':descriptor(output/'corpus-invariant.json'),'old_view_check':descriptor(output/(ident+'-old-view.json')),'new_view_check':descriptor(output/(ident+'-new-view.json')),'formula_output_typography_semantics':'UNAVAILABLE: separate output judgments not assigned','renderer_limit':'Historical EPUB renderer NONPASS preserved; no new render or isolation claim'}
        write_new(output/(ident+'.json'),value);results.append({'document_id':ident,'status':value['status'],'verification':descriptor(output/(ident+'.json'))})
    write_new(output/'summary.json',{'schema_version':3,'kind':'derived_reference_content_qualification','all_verified':True,'documents':results,'original_response':descriptor(response),'new_human_confirmations':0,'invariance':descriptor(output/'corpus-invariant.json'),'checker':descriptor(__file__)})
    print('DERIVED_CONTENT_VERIFIED',len(results),'NEW_HUMAN_CONFIRMATIONS',0)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('old-corpus','corpus','old-packet','packet','response','old-verification','output'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();derive(a.old_corpus,a.corpus,a.old_packet,a.packet,a.response,a.old_verification,a.output)

if __name__=='__main__':main()
