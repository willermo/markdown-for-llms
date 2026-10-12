"""Confronto umano documentale v2: gold manuale, nessun output motore letto."""
import argparse
from html import escape
from pathlib import Path
import re

from data import descriptor,digest,digest_bytes,load,write_new
from packet import verify_packet
from safety import css,html,no_symlink_root,png,regular,relative,tree_inventory

MATH_NS='http://www.w3.org/1998/Math/MathML'
SCOPE='complete-visible-document: text,numbers,formulas,cells,figures,captions,notes,references,bibliography,reading-order; raster encoding and TeX compilation not certified'
LABELS={'C01-txt-it':'C01 · Testo italiano','C02-docx-it':'C02 · DOCX italiano, due pagine','C03-epub-en':'C03 · EPUB inglese, due capitoli','C04-rtf-it':'C04 · RTF italiano','C05-tex-en':'C05 · Progetto LaTeX inglese','C06-pdf-text-it':'C06 · PDF italiano, due pagine','C07-pdf-columns-en':'C07 · PDF inglese a colonne','C08-scan-it':'C08 · Scansione italiana','C09-scan-en':'C09 · Scansione inglese','C10-pdf-mixed-it-en':'C10 · PDF misto, tre pagine','K01-full':'K01 · Controllo completo','K02-three-sections':'K02 · Tre sezioni','K03-repeated':'K03 · Testo ripetuto'}
STYLE='''th {font-weight:normal;text-align:left} body {margin:0;color:#17243a;background:#f2f5fa;font:17px/1.6 system-ui,sans-serif} header {padding:16px 28px;background:white;border-bottom:1px solid #d1d9e6} main {max-width:1800px;margin:auto;padding:18px} h1 {font-size:1.65em;line-height:1.3} h2 {font-size:1.3em} a {color:#174aac} .comparison {display:grid;grid-template-columns:minmax(360px,1fr) minmax(360px,1fr);gap:20px} .panel {background:white;border:1px solid #ccd5e3;border-radius:8px;overflow:auto;max-height:78vh;padding:22px} .panel-title {position:sticky;top:-22px;background:white;padding:12px 0;border-bottom:1px solid #d9e0eb} img {max-width:100%;height:auto} .original-page {margin:12px 0 25px} .document {font-family:Georgia,serif;font-size:1.12em} .document section {margin:16px 0} .body-text {white-space:pre-wrap} table {border-collapse:collapse;margin:14px 0;width:100%} th,td {border:1px solid #8191a8;padding:7px 10px;min-width:36px;vertical-align:middle} caption,figcaption {font-size:.95em;padding:8px 0;text-align:left} figure {margin:20px 0} figure img {max-width:360px} pre {white-space:pre-wrap;overflow-wrap:anywhere;background:#f5f7fb;padding:14px;font:14px/1.6 monospace} code {overflow-wrap:anywhere} .formula {font-size:1.45em;padding:12px 0} math {font-family:math;font-size:1.1em} .reference-links {font-size:.9em} .note {border-left:3px solid #8ca2c2;padding-left:14px} footer {padding:16px 28px;background:white} .start {display:block;padding:18px;background:white;border:2px solid #174aac;border-radius:8px;font-size:1.2em;margin:18px 0} .catalog {background:white;padding:18px 30px}'''


def math_markup(node,depth=0):
    if depth>16:raise ValueError('Reference math depth')
    kind=node['type'];child=lambda n:math_markup(n,depth+1)
    if kind=='Seq':return '<mrow>'+''.join(child(c) for c in node['items'])+'</mrow>'
    if kind=='Arg':return child(node['body'])
    if kind in ('Symbol','Number','Text'):
        tag={'Symbol':'mi','Number':'mn','Text':'mtext'}[kind]
        return '<'+tag+'>'+escape(node.get('lexeme',node.get('value','')))+'</'+tag+'>'
    if kind=='Operator':return '<mo>'+{'equals':'=','plus':'+','minus':'−','lt':'&lt;','gt':'&gt;'}[node['kind']]+'</mo>'
    if kind=='Scripts':
        sub,sup=node.get('sub'),node.get('sup')
        if not sub and not sup:raise ValueError('Empty script')
        tag='msubsup' if sub and sup else 'msub' if sub else 'msup'
        return '<'+tag+'>'+child(node['base'])+(child(sub) if sub else '')+(child(sup) if sup else '')+'</'+tag+'>'
    if kind=='Fraction':return '<mfrac>'+child(node['numerator'])+child(node['denominator'])+'</mfrac>'
    if kind=='Sum':return '<munderover><mo>∑</mo>'+child(node['lower'])+child(node['upper'])+'</munderover>'
    if kind in ('Aligned','Matrix'):
        if not node['rows'] or sum(len(r) for r in node['rows'])>256:raise ValueError('Reference math cells')
        body='<mtable>'+''.join('<mtr>'+''.join('<mtd>'+child(c)+'</mtd>' for c in r)+'</mtr>' for r in node['rows'])+'</mtable>'
        if kind=='Matrix':
            if node['delimiter']!='square':raise ValueError('Reference matrix delimiter')
            body='<mrow><mo>[</mo>'+body+'<mo>]</mo></mrow>'
        return body
    raise ValueError('Unsupported reference AST: '+kind)


def md_text(value):
    return ''.join('\\'+c if c in '\\`*_[]<>#!|' else c for c in value)


def fragment(value):
    value=value.rsplit('#',1)[-1]
    if not re.fullmatch(r'[A-Za-z][A-Za-z0-9_.-]*',value):raise ValueError('Reference anchor subset')
    return value


def markdown_target(value):
    return value if value.startswith(('https://','http://','mailto:')) else '#'+fragment(value)


def markdown_note(target):
    ident=fragment(target)
    return '<a id="call-'+ident+'"></a>[^'+ident+']'


def reference_document(gold,asset_paths,control=False):
    """Serialize only authored typed blocks. No parser/scorer output input."""
    content=[];markdown=[];ranges=[];notes=[b for b in gold['blocks'] if b['type']=='note']
    note_numbers={b.get('note_id',b.get('anchor')):str(i+1) for i,b in enumerate(notes)}
    def link(ref):
        target=ref['target'];kind=ref['kind'];text=ref.get('text','')
        if kind=='note_call':
            number=note_numbers.get(fragment(target))
            if number is None:raise ValueError('Reference note absent')
            return '<sup class="note-call" id="ref-call-'+fragment(target)+'"><a data-reference-kind="note_call" data-source-target="'+escape(target,quote=True)+'" href="#ref-'+fragment(target)+'">'+number+'</a></sup>'
        if target.startswith(('https://','http://','mailto:')):
            return '<span data-reference-kind="'+escape(kind)+'" data-source-target="'+escape(target,quote=True)+'">'+escape(text)+'</span>'
        return '<a data-reference-kind="'+escape(kind)+'" data-source-target="'+escape(target,quote=True)+'" href="#ref-'+fragment(target)+'">'+escape(text)+'</a>'
    for b in gold['blocks']:
        kind=b['type'];bid=b['block_id'];anchor=b.get('note_id') or b.get('anchor')
        opening='<section data-reference-block="'+escape(bid)+'" data-block-type="'+escape(kind)+'"'+(' id="ref-'+fragment(anchor)+'"' if anchor else '')+'>'
        refs=b.get('references',[]);body='';md=''
        if kind=='heading':
            body='<h1 class="body-text">'+escape(b['text'])+'</h1>';md='# '+md_text(b['text'])+'\n'
        elif kind=='code':
            fence='`'*max(3,1+max((len(m[0]) for m in re.finditer(r'`+',b['text'])),default=0))
            body='<pre class="body-text"><code>'+escape(b['text'])+'</code></pre>';md=fence+'\n'+b['text']+('' if b['text'].endswith('\n') else '\n')+fence+'\n'
        elif kind=='formula':
            for f in b['formulas']:
                body+='<div class="formula"><math xmlns="'+MATH_NS+'" display="block">'+math_markup(f['structure_expected'])+'</math></div>'
                md+='$$'+f['reference_latex']+'$$\n'
        elif kind=='table':
            rows=b['cells'];caption=b.get('caption')
            body='<table>'+('<caption>'+escape(caption)+'</caption>' if caption else '')+'<tbody>'+''.join('<tr>'+''.join(('<th>'+escape(c)+'</th>' if ri<b.get('header_rows',1) else '<td>'+escape(c)+'</td>') for c in r)+'</tr>' for ri,r in enumerate(rows))+'</tbody></table>'
            md_rows=rows if b.get('header_rows',1) else [['']*len(rows[0]),*rows]
            md='|'+'|'.join(md_text(c) for c in md_rows[0])+'|\n|'+'|'.join('---' for _ in rows[0])+'|\n'+''.join('|'+'|'.join(md_text(c) for c in row)+'|\n' for row in md_rows[1:])
            if caption:md+='\n: '+md_text(caption)+'\n'
        elif kind=='figure':
            path=asset_paths[b['asset']];relative(path)
            body='<figure><img src="'+path+'" alt="'+escape(b['caption'],quote=True)+'"/><figcaption class="body-text">'+escape(b['caption'])+'</figcaption></figure>'
            md='!['+md_text(b['caption'])+']('+path+')\n'
        elif kind in ('paragraph','literal_text','note','bibliography'):
            text=escape(b['text']);md=md_text(b['text'])
            if kind=='note':
                ident=anchor
                if not ident:raise ValueError('Note identity')
                body='<aside class="note"><strong>Nota '+note_numbers[ident]+'</strong><p class="body-text">'+text+'</p>'+''.join(link(r) for r in refs)+'</aside>'
                md='[^'+ident+']: '+md+''.join(' ['+md_text(r['text'])+']('+markdown_target(r['target'])+')' for r in refs)+'\n'
            else:
                if control:
                    # Control authored text already contains link labels.
                    # Replace each unique label at its position; never append it.
                    segments=[];cursor=0
                    for ref in refs:
                        if ref['kind']=='note_call':continue
                        label=ref['text'];index=b['text'].find(label,cursor)
                        if index<0:raise ValueError('Control link label absent')
                        segments+=[escape(b['text'][cursor:index]),link(ref)];cursor=index+len(label)
                    segments.append(escape(b['text'][cursor:]));text=''.join(segments)
                    for ref in refs:
                        if ref['kind']!='note_call':
                            label=md_text(ref['text']);md=md.replace(label,'['+label+']('+markdown_target(ref['target'])+')',1)
                    note_calls=[r for r in refs if r['kind']=='note_call']
                    body='<p><span class="body-text">'+text+'</span>'+''.join(link(r) for r in note_calls)+'</p>'
                    md+=''.join(markdown_note(r['target']) for r in note_calls)
                else:
                    body='<p><span class="body-text">'+text+'</span>'+''.join(' '+link(r) for r in refs)+'</p>'
                    for ref in refs:md+=' '+(markdown_note(ref['target']) if ref['kind']=='note_call' else '['+md_text(ref['text'])+']('+markdown_target(ref['target'])+')')
                external=[r for r in refs if r['target'].startswith(('https://','http://','mailto:'))]
                if external:body+='<p class="reference-links">'+''.join('<code>'+escape(r['target'])+'</code>' for r in external)+'</p>'
                md+='\n'
        else:raise ValueError('Reference block type: '+kind)
        if anchor and kind!='note':md='<a id="'+fragment(anchor)+'"></a>\n'+md
        start=len(''.join(markdown).encode());markdown.append(md+'\n');ranges.append({'block_id':bid,'start':start,'length':len(md.encode())})
        content.append(opening+body+'</section>')
    return '<article class="document" id="reference-document">'+''.join(content)+'</article>',''.join(markdown),ranges


def page(title,body):
    return '<!doctype html><html lang="it"><head><meta charset="utf-8"/><meta name="viewport" content="width=device-width, initial-scale=1"/><title>'+escape(title)+'</title><link rel="stylesheet" href="comparison.css"/></head><body>'+body+'</body></html>'


def build(corpus,mechanical,control_m,epub_packet,output,metadata_delta=None):
    no_symlink_root(output.parent);output.mkdir(exist_ok=False)
    delta=None
    if metadata_delta:
        from invariance import validate_delta
        delta=validate_delta(metadata_delta)
    docs=load(corpus/'manifest.json')['documents'];controls=load(corpus/'controls/manifest.json')['documents']
    checks=load(mechanical)['documents']+load(control_m)['documents'];supplied=verify_packet(epub_packet);pairs=[];dependencies=[]
    for d,is_control in [(d,False) for d in docs]+[(d,True) for d in controls]:
        ident=d['document_id'];relative(ident);root=corpus/'controls' if is_control else corpus
        source=regular(root,d['input']);goldbytes=regular(root,d['gold']);g=load(root/d['gold'])
        if digest_bytes(source)!=d['input_sha256'] or digest_bytes(goldbytes)!=d['gold_sha256']:raise ValueError('Comparison input STALE')
        m=next(x for x in checks if x['document_id']==ident)
        if m['status']!='PASS' or m['canonical_sha256']!=d['input_sha256'] or m['gold_sha256']!=d['gold_sha256']:raise ValueError('Comparison needs current M PASS')
        copied=[]
        def copy(raw,name):
            relative(name);target=output/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
            copied.append({'path':name,'bytes':len(raw),'sha256':digest_bytes(raw)});return name
        source_name=copy(source,'originals/'+ident+'/'+Path(d['input']).name)
        original='<p><a href="'+source_name+'">Apri il documento originale</a></p>'
        for i,name in enumerate(d.get('renders',[]),1):
            raw=regular(root,name);png(raw);target=copy(raw,'originals/'+ident+'/page-'+str(i)+'.png')
            original+='<div class="original-page"><p>Pagina '+str(i)+' · <a href="'+target+'">Apri e ingrandisci</a></p><a href="'+target+'"><img src="'+target+'" alt="Pagina '+str(i)+' dell’originale"/></a></div>'
        if d['format']=='epub':
            identity=next(x for x in supplied['identities'] if x['document_id']==ident)
            if identity['canonical_sha256']!=d['input_sha256']:raise ValueError('EPUB source views STALE')
            inherited=None
            if identity['gold_sha256']!=d['gold_sha256']:
                if not delta:raise ValueError('EPUB metadata inheritance proof absent')
                from invariance import gold_invariant
                old_gold=Path(delta['old_root'])/d['gold']
                if descriptor(old_gold)['sha256']!=identity['gold_sha256']:raise ValueError('EPUB old gold STALE')
                gold_invariant(load(old_gold),g,'epub')
                inherited={'delta':descriptor(metadata_delta),'old_gold':descriptor(old_gold),'new_gold':descriptor(root/d['gold']),'rule':'unchanged canonical renderer bytes; technical header delta only'}
            for i in (1,2):
                raw=regular(epub_packet,'images/'+ident+'-chapter'+str(i)+'.png');png(raw);target=copy(raw,'originals/'+ident+'/chapter-'+str(i)+'.png')
                original+='<div class="original-page"><p>Capitolo '+str(i)+' · <a href="'+target+'">Apri e ingrandisci</a></p><img src="'+target+'" alt="Capitolo '+str(i)+' originale"/></div>'
            original+='<p>Viste dei capitoli prodotte dal renderer locale; l’isolamento del renderer non è stato convalidato.</p>'
            dependencies.append({'document_id':ident,'source_packet':descriptor(epub_packet/'packet.json'),'renderer_R_status':'NON_PASS_NESTED_NAMESPACE','views':'exact actual chapter PNGs, unchanged canonical identity; no new renderer claim','metadata_inheritance':inherited})
        if d['format'] in ('txt','tex','control-markdown'):
            names=[d['input']]
            if d['format']=='tex':
                original+='<p>Progetto LaTeX non compilato: qui sono mostrati i file sorgenti e la figura originale.</p>'
                names += [(Path(d['input']).parent/n).as_posix() for n in d['includes']]
            for name in names:
                raw=regular(root,name)
                if name!=d['input']:copy(raw,'originals/'+ident+'/'+Path(name).relative_to(Path(d['input']).parent).as_posix())
                original+='<h3>'+escape(Path(name).name)+'</h3><pre>'+escape(raw.decode())+'</pre>'
        assets={}
        for b in g['blocks']:
            if b.get('asset') and b['asset'] not in assets:
                raw=regular(root,ident+'/assets/'+b['asset']) if is_control else regular(root,'assets/'+b['asset']);png(raw)
                assets[b['asset']]=copy(raw,'assets/'+ident+'/'+b['asset'])
                if d['format'] in ('tex','control-markdown'):
                    source_asset=(Path(d['input']).parent/'assets'/b['asset']).as_posix()
                    actual_source_asset=regular(root,source_asset)
                    if actual_source_asset!=raw:raise ValueError('Original project asset differs from gold asset')
                    original_asset=copy(actual_source_asset,'originals/'+ident+'/assets/'+b['asset'])
                    original+='<p>Asset del sorgente:</p><img src="'+original_asset+'" alt="Figura originale"/>'
        reference,md,ranges=reference_document(g,assets,is_control)
        copy(md.encode(),ident+'.md');write_new(output/'technical'/ (ident+'-reference-map.json'),{'blocks':ranges,'schema_version':2})
        body='<header><a href="index.html">Tutti i documenti</a><h1>'+escape(LABELS[ident])+'</h1><p>Confronta l’originale con il Markdown di riferimento e segnala ciò che manca o cambia.</p><p>Si controlla il riferimento usato per misurare le conversioni. Impaginazione e colonne possono cambiare; contenuto e ordine devono restare completi.</p></header><main><div class="comparison"><section class="panel" id="original-document"><h2 class="panel-title">Originale</h2>'+original+'</section><section class="panel"><h2 class="panel-title">Markdown di riferimento</h2><p><a href="'+ident+'.md" download="'+ident+'.md">Scarica il file Markdown</a></p>'+reference+'</section></div></main><footer>Rispondi in chat, per esempio: «'+ident.split('-')[0]+': corrisponde», «'+ident.split('-')[0]+': manca …» oppure «'+ident.split('-')[0]+': non riesco a valutarlo».</footer>'
        htmltext=page(LABELS[ident],body);html(htmltext);copy(htmltext.encode(),ident+'.html')
        identity={'document_id':ident,'canonical_sha256':d['input_sha256'],'gold_sha256':d['gold_sha256'],'control':is_control,'corpus_revision':load(corpus/'manifest.json')['revision'],'stylesheet_sha256':digest_bytes(STYLE.encode()),'scope':SCOPE,'original_and_reference_files':copied,'reference_map':{'path':'technical/'+ident+'-reference-map.json','sha256':descriptor(output/'technical'/(ident+'-reference-map.json'))['sha256']}}
        identity['pair_sha256']=digest({k:v for k,v in identity.items() if k!='corpus_revision'});pairs.append(identity)
    css(STYLE.encode());(output/'comparison.css').write_text(STYLE)
    first=next(p for p in pairs if p['document_id']=='C02-docx-it')
    body='<header><h1>Originale e Markdown di riferimento</h1><p>Confronta l’originale con il Markdown di riferimento e segnala ciò che manca o cambia.</p></header><main><a class="start" href="C02-docx-it.html">Inizia da C02 · DOCX italiano</a><p>Un documento per volta. Puoi rispondere direttamente in chat.</p><div class="catalog"><ul>'+''.join('<li><a href="'+p['document_id']+'.html">'+escape(LABELS[p['document_id']])+'</a></li>' for p in pairs if p!=first)+'</ul></div></main>'
    index=page('Confronto dei documenti',body);html(index);(output/'index.html').write_text(index)
    write_new(output/'technical/M-gold.json',load(mechanical));write_new(output/'technical/M-controls.json',load(control_m))
    estimate={'documents':13,'C02_minutes':[5,8],'total_minutes':[60,90],'status':'estimate only, new duration not agreed','method':'complete paired reading; 14 fixed page/chapter views plus TXT/TeX/control source units; source complexity and formulas raise time','old_260_minutes':'historical agreement for old workflow, no actual human work attributed'}
    write_new(output/'technical/estimate.json',estimate)
    # Independent checker reads the actual serialized HTML/MD, not this AST
    # walk. Its receipt has its own provenance, never a human signature.
    from comparison_check import check_document
    view_checks=[]
    for pair in pairs:
        source=corpus/'controls' if pair['control'] else corpus
        doc=next(d for d in (controls if pair['control'] else docs) if d['document_id']==pair['document_id'])
        view_checks.append(check_document(output,doc,load(source/doc['gold']),source,pair['control'],require_original=True))
    if any(check['status']!='PASS' for check in view_checks):
        write_new(output/'technical/reference-check-FAIL.json',view_checks)
        raise ValueError('Independent reference completeness FAIL')
    write_new(output/'technical/reference-check.json',{'schema_version':2,'method':'independent-full-reference-view-and-markdown-v2','checker':descriptor(Path(__file__).with_name('comparison_check.py')),'status':'PASS','documents':view_checks})
    value={'schema_version':2,'kind':'document-comparison-v2','identities':pairs,'documents':len(pairs),'source_manifest':descriptor(corpus/'manifest.json'),'M_gold':descriptor(mechanical),'M_controls':descriptor(control_m),'renderer_dependencies':dependencies,'V_granularity':'document','response_status':'PENDING','human_output_judgments':'UNAVAILABLE not assigned','estimate':estimate,'files':[r for r in tree_inventory(output) if r['kind']=='regular']}
    value['packet_sha256']=digest(value);write_new(output/'packet.json',value);return value


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('corpus','mechanical','control-m','epub-packet','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--metadata-delta',type=Path)
    a=p.parse_args();result=build(a.corpus.absolute(),a.mechanical,a.control_m,a.epub_packet,a.output.absolute(),a.metadata_delta);print('DOCUMENT_COMPARISON_READY',result['documents'],result['packet_sha256']);return 0


if __name__=='__main__':raise SystemExit(main())
