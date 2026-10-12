"""Completezza indipendente della vista v2 e del .md, contro gold manuale.

Nessun import del serializer, generatori, observations o scorer. Il MathML è
letto dal verificatore M; il Markdown usa qui un subset finito distinto.
"""
import argparse
from pathlib import Path
import re
import xml.etree.ElementTree as ET

from data import descriptor,digest,digest_bytes,load,write_new
from packet import verify_packet
from reference_check import mathml_source,TeXSource
from safety import css,html,local_uri,png,regular,xml


def text(node):
    return ''.join(node.itertext()) if node is not None else None


def unescape(value):
    return re.sub(r'\\([\\`*_\[\]<>#!|])',r'\1',value)


def markdown_block(raw,block,assets,control):
    """Inspect actual bytes at the declared range, not serializer-generated text."""
    literal=raw.decode('utf-8');kind=block['type'];checks=[]
    def emit(prop,expected,actual):checks.append((prop,expected,actual))
    if kind!='code':
        anchors=re.findall(r'(?<!\\)<a id="([A-Za-z][A-Za-z0-9_.-]*)"></a>',literal)
        expected=([block['anchor']] if block.get('anchor') and kind!='note' else [])+['call-'+r['target'].rsplit('#',1)[-1] for r in block.get('references',[]) if r['kind']=='note_call']
        emit('explicit-anchors',expected,anchors)
        literal=re.sub(r'(?<!\\)<a id="[A-Za-z][A-Za-z0-9_.-]*"></a>\n?','',literal)
    if kind=='formula':
        formulas=re.findall(r'\$\$(.*?)\$\$',literal,re.S)
        emit('formulas',[f['structure_expected'] for f in block['formulas']],[TeXSource(f).parse() for f in formulas])
        emit('formula-only','',re.sub(r'\$\$.*?\$\$','',literal,flags=re.S).strip())
    elif kind=='table':
        lines=literal.splitlines();rows=[]
        for line in lines:
            if line.startswith('|') and line.endswith('|'):
                cells=re.split(r'(?<!\\)\|',line)[1:-1]
                if all(c=='---' for c in cells):continue
                rows.append([unescape(c) for c in cells])
        if block.get('header_rows',1)==0:
            emit('empty-syntactic-header',['']*len(block['cells'][0]),rows[0] if rows else None)
            rows=rows[1:]
        emit('cells',block['cells'],rows)
        captions=[unescape(line[2:]) for line in lines if line.startswith(': ')]
        emit('caption',[block['caption']] if block.get('caption') else [],captions)
        emit('table-only',[],[line for line in lines if line and not line.startswith(('|',': '))])
    elif kind=='figure':
        match=re.fullmatch(r'!\[((?:\\.|[^\]])*)\]\(([^)]+)\)(?: \{#([^}]+)\})?\n',literal)
        emit('figure',[block['caption'],assets[block['asset']],None],
             [unescape(match[1]),match[2],match[3]] if match else None)
    elif kind=='code':
        lines=literal.splitlines(keepends=True);fence=lines[0].rstrip('\n') if lines else ''
        valid=re.fullmatch(r'`{3,}',fence) and len(lines)>=2 and lines[-1]==fence+'\n'
        actual=''.join(lines[1:-1]) if valid else None
        expected=block['text']+('' if block['text'].endswith('\n') else '\n')
        emit('code',expected,actual)
    else:
        refs=block.get('references',[]);links=re.findall(r'(?<!!)\[((?:\\.|[^\]])*)\]\(([^)]+)\)',literal)
        expected_targets=[(r['text'],r['target'] if r['target'].startswith(('https://','http://','mailto:')) else '#'+r['target'].rsplit('#',1)[-1]) for r in refs if r['kind']!='note_call']
        emit('links',expected_targets,[(unescape(a),b) for a,b in links])
        calls=re.findall(r'\[\^([^]]+)\](?!:)',literal)
        emit('note-calls',[r['target'].rsplit('#',1)[-1] for r in refs if r['kind']=='note_call'],calls)
        value=re.sub(r'\[\^([^]]+)\](?!:)','',literal)
        value=re.sub(r'\[((?:\\.|[^\]])*)\]\(([^)]+)\)',lambda m:m[1] if control else '',value)
        if kind=='heading':
            match=re.fullmatch(r'# (.*?)(?: \{#([^}]+)\})?\n',value)
            emit('heading-attribute',None,match[2] if match else None);value=match[1] if match else None
        elif kind=='note':
            match=re.fullmatch(r'\[\^([^]]+)\]: (.*)\n',value)
            emit('note-identity',block.get('note_id',block.get('anchor')),match[1] if match else None);value=match[2] if match else None
        else:value=value.removesuffix('\n')
        # Added reference labels are separate from authored full body text.
        if value is not None and not control:value=value.rstrip(' ')
        emit('text',block['text'],unescape(value) if value is not None else None)
    return checks


def check_document(root,document,gold,corpus,control=False,require_original=False):
    ident=document['document_id'];raw=regular(root,ident+'.html');html(raw)
    # Accept only our fixed doctype, then XML (no DTD, JS, external entities).
    if not raw.startswith(b'<!doctype html>'):raise ValueError('Comparison doctype')
    page=xml(raw[len(b'<!doctype html>'):]);records=[]
    def emit(prop,expected,actual,block=None):
        records.append({'document_id':ident,'block_id':block,'property':prop,'expected':expected,'actual':actual,'status':'PASS' if expected==actual else 'FAIL'})
    if require_original:
        originals=[n for n in page.iter('section') if n.get('id')=='original-document']
        if len(originals)!=1:raise ValueError('Original document view absent/duplicate')
        original=originals[0]
        expected_pages=['originals/'+ident+'/page-'+str(i)+'.png' for i,_ in enumerate(document.get('renders',[]),1)]
        if document['format']=='epub':expected_pages=['originals/'+ident+'/chapter-'+str(i)+'.png' for i in (1,2)]
        expected_assets=['originals/'+ident+'/assets/'+name for name in dict.fromkeys(b['asset'] for b in gold['blocks'] if b.get('asset'))] if document['format'] in ('tex','control-markdown') else []
        emit('original-page-asset-order',expected_pages+expected_assets,[n.get('src') for n in original.iter('img')])
        names=[document['input']]+[(Path(document['input']).parent/n).as_posix() for n in document.get('includes',[])] if document['format'] in ('txt','tex','control-markdown') else []
        emit('original-full-source-units',[regular(corpus,n).decode() for n in names],[text(n) for n in original.iter('pre')])
        canonical='originals/'+ident+'/'+Path(document['input']).name
        emit('original-canonical-download',True,canonical in [n.get('href') for n in original.iter('a')])
    for node in page.iter():
        if any(a in node.attrib for a in ('hidden','aria-hidden')):raise ValueError('Hidden reference content')
        for key in ('src','href'):
            if key in node.attrib:
                target=local_uri(node.attrib[key],allow_external=False)
                if target:regular(root,target)
    articles=[node for node in page.iter('article') if node.get('id')=='reference-document']
    if len(articles)!=1:raise ValueError('Reference document absent/duplicate')
    sections=list(articles[0]);emit('block-order',[b['block_id'] for b in gold['blocks']],[n.get('data-reference-block') for n in sections])
    assets={b['asset']:'assets/'+ident+'/'+b['asset'] for b in gold['blocks'] if b.get('asset')}
    for name,path in assets.items():
        source=ident+'/assets/'+name if control else 'assets/'+name
        actual=regular(root,path);png(actual);emit('asset-bytes:'+name,digest_bytes(regular(corpus,source)),digest_bytes(actual))
    by_id={n.get('data-reference-block'):n for n in sections}
    for b in gold['blocks']:
        node=by_id.get(b['block_id']);kind=b['type'];bid=b['block_id']
        if node is None:continue  # order record already FAIL, keep other evidence.
        emit('typed-block',kind,node.get('data-block-type'),bid)
        body=[x for x in node.iter() if 'body-text' in x.get('class','').split()]
        if kind in ('heading','paragraph','literal_text','code','note','bibliography','figure'):
            emit('full-text',[b['caption'] if kind=='figure' else b['text']],[text(n) for n in body],bid)
        formulas=[x for x in node.iter() if x.tag.rsplit('}',1)[-1]=='math']
        emit('math',[f['structure_expected'] for f in b.get('formulas',[])],[mathml_source(x) for x in formulas],bid)
        tables=list(node.iter('table'));cells=[[[text(cell) for cell in row if cell.tag in ('td','th')] for row in table.iter('tr')] for table in tables]
        emit('cells',[b['cells']] if kind=='table' else [],cells,bid)
        if kind=='table' and 'header_rows' in b:
            header_counts=[sum(bool(list(row)) and all(c.tag=='th' for c in row) for row in table.iter('tr')) for table in tables]
            emit('semantic-header',[b.get('header_rows',1)],header_counts,bid)
        captions=[text(x) for table in tables for x in table.iter('caption')]
        emit('table-caption',[b['caption']] if kind=='table' and b.get('caption') else [],captions,bid)
        figures=list(node.iter('figure'));emit('figure-src',[assets[b['asset']]] if kind=='figure' else [],[i.get('src') for f in figures for i in f.iter('img')],bid)
        refs=[x for x in node.iter() if x.get('data-reference-kind')]
        expected=[(r['kind'],r['target'],r['text'] if r['kind']!='note_call' else None) for r in b.get('references',[])]
        actual=[(r.get('data-reference-kind'),r.get('data-source-target'),text(r) if r.get('data-reference-kind')!='note_call' else None) for r in refs]
        emit('references',expected,actual,bid)
        anchor=b.get('note_id') or b.get('anchor');emit('anchor','ref-'+anchor if anchor else None,node.get('id'),bid)
    all_ids=[n.get('id') for n in page.iter() if n.get('id')];emit('unique-anchors',len(set(all_ids)),len(all_ids))
    for node in articles[0].iter('a'):
        href=node.get('href','')
        if href.startswith('#'):emit('resolved-reference:'+href,True,href[1:] in all_ids)
    md=regular(root,ident+'.md');mapping=load(root/'technical'/(ident+'-reference-map.json'))['blocks']
    emit('markdown-block-order',[b['block_id'] for b in gold['blocks']],[b['block_id'] for b in mapping])
    cursor=0
    for b,m in zip(gold['blocks'],mapping):
        start,length=m['start'],m['length']
        if not isinstance(start,int) or not isinstance(length,int) or start<cursor or length<1 or start+length>len(md):raise ValueError('Markdown reference range')
        emit('markdown-gap',b'\n'.hex() if cursor else '',md[cursor:start].hex(),b['block_id'])
        for prop,expected,actual in markdown_block(md[start:start+length],b,assets,control):emit('markdown-'+prop,expected,actual,b['block_id'])
        cursor=start+length
    emit('markdown-tail',b'\n'.hex(),md[cursor:].hex())
    return {'schema_version':2,'document_id':ident,'status':'PASS' if all(r['status']=='PASS' for r in records) else 'FAIL','view':descriptor(root/(ident+'.html')),'markdown':descriptor(root/(ident+'.md')),'records':records}


def validate(root,corpus):
    packet=verify_packet(root)
    if packet.get('schema_version')!=2 or packet.get('kind')!='document-comparison-v2':raise ValueError('Comparison schema')
    style=check_stylesheet(root,packet)
    documents=load(corpus/'manifest.json')['documents'];controls=load(corpus/'controls/manifest.json')['documents'];results=[]
    expected=[d['document_id'] for d in documents+controls]
    if len(expected)!=13 or [p['document_id'] for p in packet['identities']]!=expected:raise ValueError('Expected thirteen complete pairs')
    for identity in packet['identities']:
        if digest({k:v for k,v in identity.items() if k not in ('pair_sha256','corpus_revision')})!=identity['pair_sha256']:raise ValueError('Pair metadata STALE')
        if digest_bytes(style)!=identity['stylesheet_sha256']:raise ValueError('Pair stylesheet STALE')
        control=identity['control'];source=corpus/'controls' if control else corpus
        d=next(d for d in (controls if control else documents) if d['document_id']==identity['document_id'])
        if digest_bytes(regular(source,d['input']))!=identity['canonical_sha256'] or digest_bytes(regular(source,d['gold']))!=identity['gold_sha256']:raise ValueError('Current canonical/gold STALE')
        for f in identity['original_and_reference_files']:
            if digest_bytes(regular(root,f['path']))!=f['sha256']:raise ValueError('Pair file STALE')
        originals=[(d['input'],'originals/'+d['document_id']+'/'+Path(d['input']).name)]
        originals += [(n,'originals/'+d['document_id']+'/page-'+str(i)+'.png') for i,n in enumerate(d.get('renders',[]),1)]
        originals += [((Path(d['input']).parent/n).as_posix(),'originals/'+d['document_id']+'/'+Path(n).as_posix()) for n in d.get('includes',[])]
        if d['format'] in ('tex','control-markdown'):
            originals += [((Path(d['input']).parent/'assets'/name).as_posix(),'originals/'+d['document_id']+'/assets/'+name) for name in dict.fromkeys(b['asset'] for b in load(source/d['gold'])['blocks'] if b.get('asset'))]
        if d['format']=='epub':check_epub_dependency(root,packet,identity)
        if any(regular(source,a)!=regular(root,b) for a,b in originals):raise ValueError('Original view source STALE')
        results.append(check_document(root,d,load(source/d['gold']),source,control,require_original=True))
    return {'schema_version':2,'method':'independent-full-reference-view-and-markdown-v2','checker':descriptor(__file__),'packet_sha256':packet['packet_sha256'],'status':'PASS' if all(r['status']=='PASS' for r in results) else 'FAIL','documents':results}


def check_epub_dependency(root,packet,identity):
    dependencies=[r for r in packet['renderer_dependencies'] if r['document_id']==identity['document_id']]
    if len(dependencies)!=1 or dependencies[0]['renderer_R_status']!='NON_PASS_NESTED_NAMESPACE':raise ValueError('EPUB renderer provenance/limit absent')
    source=Path(dependencies[0]['source_packet']['path'])
    if descriptor(source)!=dependencies[0]['source_packet']:raise ValueError('EPUB source packet STALE')
    old=verify_packet(source.parent)
    match=[d for d in old['identities'] if all(d.get(k)==identity[k] for k in ('document_id','canonical_sha256'))]
    if len(match)!=1:raise ValueError('EPUB source pair STALE')
    if match[0]['gold_sha256']!=identity['gold_sha256']:
        inherited=dependencies[0].get('metadata_inheritance')
        if not inherited:raise ValueError('EPUB gold delta proof absent')
        from invariance import validate_delta,gold_invariant
        for name in ('delta','old_gold','new_gold'):
            if descriptor(inherited[name]['path'])!=inherited[name]:raise ValueError('EPUB inheritance input STALE')
        validate_delta(Path(inherited['delta']['path']))
        if inherited['old_gold']['sha256']!=match[0]['gold_sha256'] or inherited['new_gold']['sha256']!=identity['gold_sha256']:raise ValueError('EPUB inherited gold identity')
        gold_invariant(load(inherited['old_gold']['path']),load(inherited['new_gold']['path']),'epub')
    for chapter in (1,2):
        if regular(source.parent,'images/'+identity['document_id']+'-chapter'+str(chapter)+'.png')!=regular(root,'originals/'+identity['document_id']+'/chapter-'+str(chapter)+'.png'):raise ValueError('EPUB actual chapter view differs')


def check_stylesheet(root,packet):
    style=regular(root,'comparison.css');css(style)
    if re.search(r'display\s*:\s*none|visibility\s*:\s*hidden|opacity\s*:|font-size\s*:\s*0|color\s*:\s*(?:white|transparent)|overflow\s*:\s*hidden',style.decode(),re.I):raise ValueError('Content hiding CSS')
    if any(digest_bytes(style)!=pair['stylesheet_sha256'] for pair in packet['identities']):raise ValueError('Pair stylesheet STALE')
    return style


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('packet','corpus','output'):parser.add_argument('--'+name,type=Path,required=True)
    a=parser.parse_args();result=validate(a.packet,a.corpus);write_new(a.output,result);print('COMPARISON_'+result['status']);return 0 if result['status']=='PASS' else 3


if __name__=='__main__':raise SystemExit(main())
