"""V2: prove locali del contratto, nessun giudizio umano o motore simulato."""
from copy import deepcopy
import io
from pathlib import Path
import sys
import zipfile

import pytest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/benchmarks'))
from comparison import reference_document,page,STYLE,math_markup,SCOPE
from comparison_check import check_document,markdown_block
from data import descriptor,digest,digest_bytes,load,write_new
from document_response import classify,decision,import_document_response,validate_response
from packet import verify_packet
from reference_check import check_native,mathml_source,TeXSource,annotated_pdf_lines
from safety import Unsafe,html,xml,tree_inventory
from verification import combine_document

CORPUS=ROOT/'tests/fixtures/conversion-corpus'
DOCS=[(CORPUS,d,False) for d in load(CORPUS/'manifest.json')['documents']]+[(CORPUS/'controls',d,True) for d in load(CORPUS/'controls/manifest.json')['documents']]


def view(tmp_path,source,d,control):
    gold=load(source/d['gold']);assets={}
    for b in gold['blocks']:
        if b.get('asset'):
            name='assets/'+d['document_id']+'/'+b['asset'];target=tmp_path/name;target.parent.mkdir(parents=True,exist_ok=True)
            original=source/(d['document_id']+'/assets/'+b['asset'] if control else 'assets/'+b['asset'])
            target.write_bytes(original.read_bytes());assets[b['asset']]=name
    body,md,ranges=reference_document(gold,assets,control)
    (tmp_path/'comparison.css').write_text(STYLE);(tmp_path/(d['document_id']+'.html')).write_text(page('Test contract only',body))
    (tmp_path/(d['document_id']+'.md')).write_text(md)
    write_new(tmp_path/'technical'/(d['document_id']+'-reference-map.json'),{'blocks':ranges})
    return gold,assets


@pytest.mark.parametrize('source,d,control',DOCS,ids=[d['document_id'] for _,d,_ in DOCS])
def test_complete_document_view_and_download_independent(tmp_path,source,d,control):
    g,_=view(tmp_path,source,d,control);proof=check_document(tmp_path,d,g,source,control)
    assert proof['status']=='PASS',[r for r in proof['records'] if r['status']!='PASS']


@pytest.mark.parametrize('property', ['text','math','cell','caption','figure','note','order','markdown'])
def test_omission_or_mutation_does_not_pass_with_same_gold(tmp_path,property):
    source,d,control=DOCS[1];g,_=view(tmp_path,source,d,control);path=tmp_path/(d['document_id']+'.html')
    raw=path.read_text()
    if property=='text':raw=raw.replace(g['blocks'][0]['text'],'Testo modificato',1)
    elif property=='math':raw=raw.replace('<mn>2</mn>','<mn>3</mn>',1)
    elif property=='cell':raw=raw.replace('<td>'+g['blocks'][3]['cells'][0][0]+'</td>','',1)
    elif property=='caption':raw=raw.replace('<caption>Celle misurate</caption>','',1)
    elif property=='figure':
        import re
        raw=re.sub(r'<figcaption[^>]*>.*?</figcaption>','',raw,count=1)
    elif property in ('note','order'):
        import re
        article=re.search(r'<article[^>]*>(.*?)</article>',raw).group(1)
        sections=re.findall(r'<section .*?</section>',article)
        replacement=''.join(s for s in sections if 'data-block-type="note"' not in s) if property=='note' else ''.join(sections[1:]+sections[:1])
        raw=raw.replace(article,replacement,1)
    else:
        md=tmp_path/(d['document_id']+'.md');md.write_bytes(md.read_bytes().replace(b'Celle misurate',b'Celle omissive'))
    path.write_text(raw)
    assert check_document(tmp_path,d,g,source,control)['status']=='FAIL'


@pytest.mark.parametrize('raw',[r'x_i^2',r'\frac{a}{b}',r'\sum_{i=1}^{n}x_i',r'\begin{bmatrix}0&-1\\2&3\end{bmatrix}',r'\begin{aligned}a&=1\\b&=-2\end{aligned}'])
def test_visible_mathml_is_independently_structural(raw):
    expected=TeXSource(raw).parse()
    actual=xml(('<math xmlns="http://www.w3.org/1998/Math/MathML">'+math_markup(expected)+'</math>').encode())
    assert mathml_source(actual)==expected


def test_docx_caption_absent_fails_when_all_cells_correct(tmp_path):
    # Tiny hand-written canonical XML, no generator or converter.
    ns='http://schemas.openxmlformats.org/wordprocessingml/2006/main'
    block={'block_id':'b01','type':'table','text':'','cells':[['A',''],['−7','0012']],'caption':'Celle misurate','locator':{'part':'word/document.xml','body_child':0,'caption_child':1}}
    doc={'document_id':'test','format':'docx','input':'test.docx','gold_sha256':'unit-only'}
    for caption in (False,True):
        rows=''.join('<w:tr>'+''.join('<w:tc><w:p><w:r><w:t>'+v+'</w:t></w:r></w:p></w:tc>' for v in row)+'</w:tr>' for row in block['cells'])
        body='<w:tbl>'+rows+'</w:tbl>'+('<w:p><w:r><w:t>Celle misurate</w:t></w:r></w:p>' if caption else '')
        with zipfile.ZipFile(tmp_path/'test.docx','w') as archive:archive.writestr('word/document.xml','<w:document xmlns:w="'+ns+'"><w:body>'+body+'</w:body></w:document>')
        records=check_native(tmp_path,doc,{'blocks':[block]})
        assert next(r for r in records if r['check_id'].endswith(':cells'))['status']=='PASS'
        assert next(r for r in records if r['check_id'].endswith(':caption'))['status']==('PASS' if caption else 'FAIL')


@pytest.mark.parametrize('kind',['rtf','tex'])
def test_other_source_table_caption_discriminates(tmp_path,kind):
    caption='Measured cells';cells=[['A','B','C'],['0012','','-7']]
    if kind=='rtf':
        literal=r'\trowd A\cell B\cell C\cell\row '+r'\pard\trowd 0012\cell \cell -7\cell\row '
    else:literal='\\begin{table}\n\\begin{tabular}{lll}\nA & B & C\\\\\n0012 &  & -7\\\\\n\\end{tabular}\n'
    for present in (False,True):
        value=literal+(r'\pard Measured cells\par' if kind=='rtf' else '\\caption{Measured cells}\n') if present else literal
        if kind=='tex':value+='\\end{table}\n'
        name='test.'+kind;payload=(('{\\rtf1 '+value+'}') if kind=='rtf' else value).encode();(tmp_path/name).write_bytes(payload)
        loc={'start':0,'length':len(payload)} if kind=='rtf' else {'file':name,'start':0,'length':len(payload)}
        if kind=='rtf':loc['literal']=payload.decode()
        b={'block_id':'b01','type':'table','text':'','caption':caption,'cells':cells,'locator':loc}
        d={'document_id':'test','format':kind,'input':name,'gold_sha256':'UNIT_ONLY','includes':[]}
        records=check_native(tmp_path,d,{'blocks':[b]})
        assert next(r for r in records if r['check_id'].endswith(':cells'))['status']=='PASS'
        assert next(r for r in records if r['check_id'].endswith(':caption'))['status']==('PASS' if present else 'FAIL')


def test_pdf_manual_caption_and_formula_cannot_hide_in_locator():
    g=load(CORPUS/DOCS[5][1]['gold'])
    for b in g['blocks']:
        if b['type'] not in ('formula','table'):continue
        mutated=deepcopy(b)
        if b['type']=='table':mutated['caption']+=' omessa'
        else:mutated['formulas'][0]['reference_latex']='x+7'
        assert annotated_pdf_lines(mutated)!=annotated_pdf_lines(b)


def test_safe_literal_text_and_remote_assets(tmp_path):
    text='<script>alert("x")</script> & −7'
    g={'blocks':[{'block_id':'b01','type':'paragraph','text':text}]}
    body,md,_=reference_document(g,{})
    html(body);assert '&lt;script&gt;' in body and '<script>' not in body
    assert '\\<script\\>' in md
    with pytest.raises(Unsafe):html('<img src="https://example.invalid/x.png"/>')
    with pytest.raises(Unsafe):html('<p onclick="x">a</p>')
    with pytest.raises(Unsafe):reference_document({'blocks':[{'block_id':'b01','type':'figure','caption':'x','asset':'x'}]},{'x':'../remote.png'})


def packet_value():
    # Explicitly unit-only synthetic receipt metadata, never a real V import.
    pairs=[]
    for name in ('C02-docx-it','C03-epub-en'):
        pairs.append({'document_id':name,'canonical_sha256':'canonical-'+name,'gold_sha256':'gold-'+name,'pair_sha256':'pair-'+name,'corpus_revision':4,'scope':SCOPE})
    return {'schema_version':2,'kind':'document-comparison-v2','identities':pairs,'packet_sha256':'UNIT_ONLY'}


def classify_test(raw,sha='UNIT_ONLY'):
    return classify(packet_value(),raw.encode(),sha,'synthetic-unit-only','2026-10-08T12:00:00+00:00','synthetic unit test, not a human judgment')


@pytest.mark.parametrize('response,state',[('C02: corrisponde','ACCEPTED'),('C02: manca la didascalia','REJECTED'),('C02: non riesco a confrontare la formula','AMBIGUOUS'),('C02: ok','PENDING'),('C02: sembra corrispondere','AMBIGUOUS'),('C02: corrisponde tranne la figura','REJECTED'),('C02: corrisponde\nC02: corrisponde','AMBIGUOUS'),('C02: corrisponde','STALE')])
def test_document_natural_response(response,state):
    result=classify_test(response,'wrong' if state=='STALE' else 'UNIT_ONLY')
    assert result['documents'][0]['status']==state
    assert 'accepted_judgments' not in result and 'rows' not in result
    assert result['documents'][1]['status']=='PENDING'


def test_rejected_document_does_not_invalidate_accepted_other():
    result=classify_test('C02: manca la didascalia\nC03: corrisponde')
    assert result['status']=='REJECTED'
    for v in result['documents']:
        identity={k:v[k] for k in ('document_id','canonical_sha256','gold_sha256','pair_sha256','scope')}
        m={**identity,'status':'PASS'}
        combined=combine_document(identity,m,v)
        assert combined['status']==('REJECTED' if v['document_id'].startswith('C02') else 'VERIFIED')
        assert combined['granularity']=='document'


def test_m_fail_stale_and_incomplete_never_verify():
    v=classify_test('C02: corrisponde')['documents'][0]
    identity={k:v[k] for k in ('document_id','canonical_sha256','gold_sha256','pair_sha256','scope')};m={**identity,'status':'PASS'}
    assert combine_document(identity,{**m,'status':'FAIL'},v)['status']=='REJECTED'
    assert combine_document({**identity,'pair_sha256':'changed'},m,v)['status']=='STALE'
    assert combine_document(identity,m,{**v,'explicit_complete_visible_document':False})['status']=='PENDING'


def test_unknown_identifier_and_missing_do_not_create_signatures():
    result=classify_test('C99: corrisponde\nC02: corrisponde')
    assert result['unattributed_text']==['C99: corrisponde']
    assert result['documents'][0]['status']=='ACCEPTED' and result['documents'][1]['status']=='PENDING'


@pytest.mark.parametrize('raw',[
    'Tutti i documenti corrispondono.',
    'Tutti  idocumenti corrispondono. Ad un esame visivo, valido le conversioni verificate.',
    'Tutti documenti corrispondono!',
    'Tutti i documenti corrispondono.\nAd un esame visivo, valido le conversioni verificate.',
])
def test_collective_complete_reply_keeps_one_statement(raw):
    value=classify_test(raw)
    assert value['status']=='ACCEPTED' and not value['unattributed_text']
    assert len(value['statements'])==1
    statement=value['statements'][0]
    assert statement['text']==statement['original_line']==raw
    assert statement['document_ids']==[d['document_id'] for d in packet_value()['identities']]
    assert 'document_id' not in statement and 'accepted_judgments' not in value
    for row in value['documents']:
        assert row['status']=='ACCEPTED' and row['response_text']==[raw]
        assert row['response_attribution']=='projection_of_single_explicit_all_documents_statement'
        identity={k:row[k] for k in ('document_id','canonical_sha256','gold_sha256','pair_sha256','scope')}
        assert combine_document(identity,{**identity,'status':'PASS'},row)['status']=='VERIFIED'
    stale=classify_test(raw,'wrong')
    assert stale['status']=='STALE' and all(d['status']=='STALE' for d in stale['documents'])


@pytest.mark.parametrize('raw',[
    'Valido le conversioni verificate.',
    'Tutti i documenti sembrano corrispondere.',
    'Tutti i documenti corrispondono tranne C02.',
    'Tutti i documenti corrispondono. Non ho visto le formule.',
    'Tutti i documenti verificati corrispondono.',
    'Non tutti i documenti corrispondono.',
    'Tutti i documenti corrispondono.\nC02: manca la didascalia',
])
def test_collective_vague_partial_or_qualified_reply_never_accepts_all(raw):
    value=classify_test(raw)
    assert value['status']!='ACCEPTED'
    assert all(d['status']!='ACCEPTED' for d in value['documents'])


def test_collective_receipt_reclassifies_original_and_scope(tmp_path):
    root=tmp_path/'packet';root.mkdir();p=packet_value();p.pop('packet_sha256');p['files']=[];p['packet_sha256']=digest(p);write_new(root/'packet.json',p)
    source=tmp_path/'synthetic-collective.txt';source.write_bytes(b'Tutti  idocumenti corrispondono.')
    output=tmp_path/'imported'
    assert import_document_response(root,source,p['packet_sha256'],output,'synthetic-unit-only','2026-10-08T12:00:00+00:00','UNIT_ONLY not actual human')==0
    receipt=output/'response.json';value=validate_response(receipt)
    assert (output/'received.txt').read_bytes()==source.read_bytes()
    value['statements'][0]['document_ids'].pop()
    value['response_sha256']=digest({k:v for k,v in value.items() if k!='response_sha256'})
    receipt.write_bytes(__import__('data').canonical(value))
    with pytest.raises(ValueError,match='contract mismatch'):validate_response(receipt)


def test_original_response_and_classification_integrity(tmp_path):
    root=tmp_path/'packet';root.mkdir();p=packet_value();p.pop('packet_sha256');p['files']=[];p['packet_sha256']=digest(p);write_new(root/'packet.json',p)
    original=tmp_path/'actual-test-input.txt';original.write_bytes('C02: corrisponde\nC03: manca −7\n'.encode())
    imported=tmp_path/'imported'
    assert import_document_response(root,original,p['packet_sha256'],imported,'synthetic-unit-only','2026-10-08T12:00:00+00:00','UNIT_ONLY not actual human')==5
    receipt=imported/'response.json';value=validate_response(receipt)
    assert (imported/'received.txt').read_bytes()==original.read_bytes()
    assert value['original_received_sha256']==digest_bytes(original.read_bytes())
    value['documents'][1]['status']='ACCEPTED';value['response_sha256']=digest({k:v for k,v in value.items() if k!='response_sha256'});receipt.write_bytes(__import__('data').canonical(value))
    with pytest.raises(ValueError,match='contract mismatch'):validate_response(receipt)
    (imported/'received.txt').write_text('C02: manca tutto')
    with pytest.raises(ValueError):validate_response(receipt)


def test_compile_actual_text_partial_history_and_changed_include(tmp_path):
    from comparison import build
    from verification import compile_documents
    from comparison_check import validate
    import shutil
    corpus=tmp_path/'corpus';shutil.copytree(CORPUS,corpus)
    # Required-M stubs solely test receipt combination; they explicitly do
    # not assert fidelity of these canonical files or replace real M runs.
    for control,name in ((False,'M-stub.json'),(True,'controls-M-stub.json')):
        source=corpus/'controls' if control else corpus
        docs=load(source/'manifest.json')['documents']
        rows=[{'document_id':d['document_id'],'canonical_sha256':d['input_sha256'],'gold_sha256':d['gold_sha256'],'required':1,'status':'PASS'} for d in docs]
        records=[{'doc_id':d['document_id'],'required':True,'status':'PASS','method':'UNIT_ONLY required-M stub, not canonical fidelity evidence'} for d in docs]
        write_new(tmp_path/name,{'status':'PASS','documents':rows,'records':records})
    packet=tmp_path/'comparison'
    # Deliberate local view-byte stub for provenance/path contract only;
    # no installed browser, historical temp/ or genuine renderer claim.
    old=tmp_path/'epub-view-byte-stub';(old/'images').mkdir(parents=True)
    epub=next(d for d in load(corpus/'manifest.json')['documents'] if d['format']=='epub')
    for chapter in (1,2):shutil.copyfile(corpus/'assets/figure-en.png',old/'images'/('C03-epub-en-chapter'+str(chapter)+'.png'))
    stub={'schema_version':1,'kind':'UNIT_ONLY image-byte stub, not an actual EPUB render','identities':[{'document_id':epub['document_id'],'canonical_sha256':epub['input_sha256'],'gold_sha256':epub['gold_sha256']}],'files':[f for f in tree_inventory(old) if f['kind']=='regular']}
    stub['packet_sha256']=digest(stub);write_new(old/'packet.json',stub)
    metadata=build(corpus,tmp_path/'M-stub.json',tmp_path/'controls-M-stub.json',old,packet)
    assert validate(packet,corpus)['status']=='PASS'
    paths=[]
    for index,raw in enumerate(('C02: manca la didascalia\nC03: corrisponde\nC05: corrisponde','C01: corrisponde')):
        source=tmp_path/('response-'+str(index)+'.txt');source.write_text(raw)
        destination=tmp_path/('received-'+str(index))
        import_document_response(packet,source,metadata['packet_sha256'],destination,'synthetic-unit-only','2026-10-08T12:0'+str(index)+':00+00:00','UNIT_ONLY workflow contract')
        paths.append(destination/'response.json')
    output=tmp_path/'verification';assert compile_documents(packet,paths,corpus,output)==5
    statuses={d['document_id']:d['status'] for d in load(output/'summary.json')['documents']}
    assert statuses['C01-txt-it']=='VERIFIED' and statuses['C03-epub-en']=='VERIFIED'
    assert statuses['C02-docx-it']=='REJECTED' and statuses['C04-rtf-it']=='PENDING'
    collective_source=tmp_path/'synthetic-complete.txt';collective_source.write_text('Tutti i documenti corrispondono.')
    complete=tmp_path/'complete-received'
    assert import_document_response(packet,collective_source,metadata['packet_sha256'],complete,'synthetic-unit-only','2026-10-08T12:02:00+00:00','UNIT_ONLY complete contract')==0
    complete_output=tmp_path/'verification-complete'
    assert compile_documents(packet,paths+[complete/'response.json'],corpus,complete_output)==0
    assert load(complete_output/'summary.json')['all_verified'] is True
    assert len(load(complete_output/'summary.json')['documents'])==13
    # Main TeX file unchanged; a changed included source still invalidates
    # that pair alone, preserving genuine decisions for other documents.
    include=corpus/'inputs/C05-tex-en/sections/one.tex';include.write_bytes(include.read_bytes()+b'\n% changed\n')
    output2=tmp_path/'verification-stale';assert compile_documents(packet,paths,corpus,output2)==5
    statuses={d['document_id']:d['status'] for d in load(output2/'summary.json')['documents']}
    assert statuses['C05-tex-en']=='STALE' and statuses['C03-epub-en']=='VERIFIED'
    stale_source=tmp_path/'stale-response.txt';stale_source.write_text('C02: corrisponde')
    destination=tmp_path/'stale-received'
    import_document_response(packet,stale_source,'wrong-packet',destination,'synthetic-unit-only','2026-10-08T12:03:00+00:00','UNIT_ONLY stale attribution')
    output3=tmp_path/'verification-stale-response';assert compile_documents(packet,paths+[destination/'response.json'],corpus,output3)==5
    statuses={d['document_id']:d['status'] for d in load(output3/'summary.json')['documents']}
    assert statuses['C02-docx-it']=='STALE' and statuses['C03-epub-en']=='VERIFIED' and statuses['C01-txt-it']=='VERIFIED'
    # Independent completeness checks the original panel too, even when
    # unchanged image files still exist beside the omitted page element.
    page_path=packet/'C02-docx-it.html';original_html=page_path.read_text()
    import re
    page_path.write_text(re.sub(r'<img src="originals/C02-docx-it/page-1.png"[^>]*>', '',original_html,count=1))
    doc=next(d for d in load(corpus/'manifest.json')['documents'] if d['document_id']=='C02-docx-it')
    proof=check_document(packet,doc,load(corpus/doc['gold']),corpus,require_original=True)
    assert proof['status']=='FAIL' and any(r['property']=='original-page-asset-order' and r['status']=='FAIL' for r in proof['records'])
    page_path.write_text(original_html)
    # Packet bytes bind the whole view, independently of receipt decision.
    asset=packet/'assets/C02-docx-it/figure-it.png';asset.write_bytes(asset.read_bytes()+b'x')
    with pytest.raises(ValueError,match='Packet file STALE'):verify_packet(packet)
