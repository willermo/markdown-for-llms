"""Unisce M e risposta V reale senza modificare le annotazioni manuali."""
import argparse
from pathlib import Path
from data import descriptor,digest,digest_bytes,load,write_new
from packet import verify_packet,import_response
from safety import regular


def combine(identity,mechanical,visual,response_status):
    keys=('document_id','canonical_sha256','gold_sha256')
    if any(identity[k]!=mechanical.get(k) or identity[k]!=visual.get(k) for k in keys):status='STALE'
    elif mechanical.get('status')!='PASS':status='REJECTED' if mechanical.get('status') in ('FAIL','DIVERGENT') else 'PENDING'
    elif response_status=='ACCEPTED' and visual.get('status')=='ACCEPTED':status='VERIFIED'
    else:status=response_status if response_status in ('REJECTED','AMBIGUOUS','STALE') else 'PENDING'
    return {'schema_version':1,**identity,'status':status,'M':mechanical,'V':visual,'V_global_status':response_status,'formula_output_typography_semantics':'UNAVAILABLE: separate judgments not assigned'}


def compile(packet,response,corpus,output):
    p=verify_packet(packet);received=load(response)
    if p.get('schema_version')==2:return compile_documents(packet,[response],corpus,output)
    original=regular(response.parent,'received.tsv')
    if digest_bytes(original)!=received['original_received_sha256']:raise ValueError('Received response copy STALE')
    output.mkdir(exist_ok=False)
    # Reclassify the actual received bytes, never manufacture accepted rows.
    import_response(packet,response.parent/'received.tsv',received['packet_sha256_received'],output/'validated-import')
    reclassified=load(output/'validated-import/response.json')
    for key in ('rows','status','accepted_judgments','documents','packet_sha256_current'):
        if received[key]!=reclassified[key]:raise ValueError('Imported response contract mismatch: '+key)
    mechanical=load(packet/'M-gold.json')['documents']+load(packet/'M-controls.json')['documents']
    manifests=load(corpus/'manifest.json')['documents']+load(corpus/'controls/manifest.json')['documents'];results=[]
    for identity in p['identities']:
        source=next(d for d in manifests if d['document_id']==identity['document_id']);root=corpus/'controls' if identity['control'] else corpus
        current={'document_id':source['document_id'],'canonical_sha256':digest_bytes(regular(root,source['input'])),'gold_sha256':digest_bytes(regular(root,source['gold']))}
        m=next(d for d in mechanical if d['document_id']==identity['document_id']);v=next(d for d in received['documents'] if d['document_id']==identity['document_id'])
        value=combine(current,m,v,received['status'])
        value.update(packet=descriptor(packet/'packet.json'),received_response=descriptor(response),received_original_sha256=received['original_received_sha256'],judgments_as_received=[r for r in received['rows'] if r['document_id']==identity['document_id']])
        path=output/(identity['document_id']+'.json');write_new(path,value);results.append({'document_id':identity['document_id'],'status':value['status'],'verification':descriptor(path)})
    write_new(output/'summary.json',{'schema_version':1,'all_verified':all(d['status']=='VERIFIED' for d in results),'documents':results,'response_status':received['status']})
    return 0 if all(d['status']=='VERIFIED' for d in results) else 5


def combine_document(identity,mechanical,visual):
    keys=('document_id','canonical_sha256','gold_sha256')
    if any(identity[k]!=mechanical.get(k) or identity[k]!=visual.get(k) for k in keys) or identity.get('pair_sha256')!=visual.get('pair_sha256') or identity.get('scope')!=visual.get('scope'):
        state='STALE'
    elif mechanical.get('status')!='PASS':state='REJECTED' if mechanical.get('status') in ('FAIL','DIVERGENT') else 'PENDING'
    elif visual.get('status')=='ACCEPTED' and visual.get('explicit_complete_visible_document') is True:state='VERIFIED'
    else:state=visual.get('status') if visual.get('status') in ('REJECTED','AMBIGUOUS','STALE') else 'PENDING'
    return {'schema_version':2,'granularity':'document',**identity,'status':state,'M':mechanical,'V':visual,'formula_output_typography_semantics':'UNAVAILABLE: separate judgments not assigned'}


def compile_documents(packet,responses,corpus,output):
    from document_response import validate_response
    from comparison_check import check_document,check_epub_dependency,check_stylesheet
    p=verify_packet(packet)
    if p.get('schema_version')!=2:raise ValueError('Document verification needs packet v2')
    check_stylesheet(packet,p)
    received=[validate_response(r) for r in responses]
    output.mkdir(exist_ok=False)
    mfiles=[load(packet/'technical'/name) for name in ('M-gold.json','M-controls.json')]
    mechanical=[d for m in mfiles for d in m['documents']]
    checks=[r for m in mfiles for r in m['records']]
    view=load(packet/'technical/reference-check.json')
    if view.get('status')!='PASS':raise ValueError('Reference view proof absent/non PASS')
    manifests=load(corpus/'manifest.json')['documents']+load(corpus/'controls/manifest.json')['documents'];results=[]
    for pair in p['identities']:
        if digest({k:v for k,v in pair.items() if k not in ('pair_sha256','corpus_revision')})!=pair['pair_sha256']:raise ValueError('Pair metadata STALE')
        d=next(d for d in manifests if d['document_id']==pair['document_id']);root=corpus/'controls' if pair['control'] else corpus
        current={'document_id':d['document_id'],'canonical_sha256':digest_bytes(regular(root,d['input'])),'gold_sha256':digest_bytes(regular(root,d['gold'])),'pair_sha256':pair['pair_sha256'],'corpus_revision':load(corpus/'manifest.json')['revision'],'scope':pair['scope']}
        m=dict(next(m for m in mechanical if m['document_id']==pair['document_id']))
        required=[r for r in checks if r['doc_id']==pair['document_id'] and r.get('required')]
        if len(required)!=m['required'] or not required or any(r['status']!='PASS' for r in required):m['status']='FAIL'
        originals=[(d['input'],'originals/'+d['document_id']+'/'+Path(d['input']).name)]
        originals += [(n,'originals/'+d['document_id']+'/page-'+str(i)+'.png') for i,n in enumerate(d.get('renders',[]),1)]
        originals += [((Path(d['input']).parent/n).as_posix(),'originals/'+d['document_id']+'/'+Path(n).as_posix()) for n in d.get('includes',[])]
        if d['format'] in ('tex','control-markdown'):
            originals += [((Path(d['input']).parent/'assets'/name).as_posix(),'originals/'+d['document_id']+'/assets/'+name) for name in dict.fromkeys(b['asset'] for b in load(root/d['gold'])['blocks'] if b.get('asset'))]
        if d['format']=='epub':check_epub_dependency(packet,p,pair)
        current_source_same=all(regular(root,a)==regular(packet,b) for a,b in originals)
        actual_view=check_document(packet,d,load(root/d['gold']),root,pair['control'],require_original=True)
        write_new(output/(d['document_id']+'-view-check.json'),actual_view)
        if actual_view['status']!='PASS':m['status']='FAIL'
        candidates=[]
        for path,r in zip(responses,received):
            v=next(v for v in r['documents'] if v['document_id']==pair['document_id'])
            if v['response_text']:candidates.append((r['receipt']['received_at'],v,path,r))
        if candidates:
            # Explicit later feedback on this same document supersedes its
            # earlier feedback; absent documents never erase other decisions.
            from datetime import datetime
            candidates.sort(key=lambda item:datetime.fromisoformat(item[0]))
            _,visual,path,r=candidates[-1];visual=dict(visual)
            if len(candidates)>1 and candidates[-1][0]==candidates[-2][0] and candidates[-1][1]!=candidates[-2][1]:visual['status']='AMBIGUOUS'
            receipt={'response':descriptor(path),'original_received_sha256':r['original_received_sha256'],'receipt':r['receipt']}
        else:
            visual={**current,'status':'PENDING','response_text':[],'explicit_complete_visible_document':False};receipt=None
        if not current_source_same:visual={**visual,'status':'STALE'}
        value=combine_document(current,m,visual)
        if not current_source_same:value['status']='STALE'
        value.update(packet=descriptor(packet/'packet.json'),received_response=receipt,received_history=[descriptor(c[2]) for c in candidates],reference_view_check=descriptor(output/(d['document_id']+'-view-check.json')))
        path=output/(d['document_id']+'.json');write_new(path,value);results.append({'document_id':d['document_id'],'status':value['status'],'verification':descriptor(path)})
    summary={'schema_version':2,'granularity':'document','all_verified':all(d['status']=='VERIFIED' for d in results),'documents':results,'responses':[descriptor(r) for r in responses]}
    write_new(output/'summary.json',summary);return 0 if summary['all_verified'] else 5


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('packet','corpus','output'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--response',type=Path,required=True,action='append')
    a=p.parse_args()
    if verify_packet(a.packet).get('schema_version')==2:return compile_documents(a.packet,a.response,a.corpus,a.output)
    if len(a.response)!=1:raise ValueError('Legacy v1 expects exactly one TSV receipt')
    return compile(a.packet,a.response[0],a.corpus,a.output)

if __name__=='__main__':raise SystemExit(main())
