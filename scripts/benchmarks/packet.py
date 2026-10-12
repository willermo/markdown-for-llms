"""Pacchetto V passivo offline e importazione fedele di giudizi reali."""
import argparse
import csv
from datetime import datetime, timezone
from html import escape
import io
from pathlib import Path
import re
import shutil

from data import canonical, descriptor, digest, digest_bytes, load, write_new
from resources import run
from safety import regular, html, zip_members, png, tree_inventory
from png_codec import decode, encode


def critical(text):
    tokens=re.findall(r'\S*[\u0080-\uffff]\S*|[−-]?[0-9]+(?:[.,][0-9]+)?',text)
    return [{'literal':t,'utf8_hex':t.encode().hex(),'codepoints':['U+%04X'%ord(c) for c in t]} for t in tokens]


def build(corpus, mechanical, control_m, output, environment, epub_render_packet=None):
    output.mkdir(exist_ok=False);(output/'images').mkdir();(output/'canonical').mkdir();(output/'sources').mkdir();(output/'tools').mkdir()
    documents=load(corpus/'manifest.json')['documents']
    controls=load(corpus/'controls/manifest.json')['documents']
    m=load(mechanical);cm=load(control_m)
    if m['status']!='PASS' or cm['status']!='PASS':
        raise ValueError('M required non PASS')
    write_new(output/'M-gold.json',m);write_new(output/'M-controls.json',cm)
    judgments=[];identities=[];commands=[];renderer_dependencies=[]
    supplied=verify_packet(epub_render_packet) if epub_render_packet else None
    def tool(argv,cwd=None,env=None):
        helpers=(descriptor('/opt/google/chrome/chrome_crashpad_handler'),) if argv[0]=='/opt/google/chrome/chrome' else ()
        rec=run(argv,cwd or output,output/'tools'/f'tool-{len(commands)+1:03d}',env or environment,120,cleanup_executables=helpers)
        commands.append(rec);write_new(output/'tools'/f'tool-{len(commands):03d}.json',rec)
        if rec['adapter_exit'] or rec['stop_reason']:raise RuntimeError('Packet renderer failed '+repr(argv))
    index=['<h1>Verifica umana V: gold e controlli</h1><p>Pacchetto offline, nessun JavaScript o risorsa remota. Nessun giudizio è precompilato. M attesta proprietà codificate; V confronta contenuto visibile, formule, tabelle, figure e ordine. Per raster la trascrizione Unicode è convenzionale: il glifo non prova NFC/NFD. Gli output dei motori non sono oggetto di questa firma.</p><p>Copia judgments-template.tsv fuori dal pacchetto; compila decision, reason, verifier e verified_at (ISO 8601 con fuso). ACCEPTED, REJECTED, PENDING o AMBIGUOUS; non modificare i canonici. L’importazione richiede il digest in packet.json.</p>']
    for is_control,root,docs,checks in [(False,corpus,documents,m),(True,corpus/'controls',controls,cm)]:
        for d in docs:
            docid=d['document_id'];g=load(root/d['gold'])
            identity={'document_id':docid,'canonical_sha256':d['input_sha256'],'gold_sha256':d['gold_sha256'],'control':is_control}
            identities.append(identity)
            md=[x for x in checks['documents'] if x['document_id']==docid]
            if len(md)!=1 or md[0]['canonical_sha256']!=d['input_sha256'] or md[0]['gold_sha256']!=d['gold_sha256']:raise ValueError('M STALE')
            shutil.copyfile(root/d['input'],output/'canonical'/(docid+Path(d['input']).suffix))
            content=['<h1>'+escape(docid)+'</h1><p>'+('Controllo del misuratore scritto a mano; non conversione reale.' if is_control else 'Canonico sintetico originale.')+'</p><pre>'+escape(str(identity))+'</pre>']
            render_cache={}
            for i,name in enumerate(d.get('renders',[]),1):
                raw=regular(root,name);png(raw);target=f'images/{docid}-p{i}.png';(output/target).write_bytes(raw)
                content.append(f'<h2>Render effettivo pagina {i}</h2><a href="{target}"><img src="{target}" width="740" alt="Pagina {i}"/></a>')
                render_cache[i]=target
            if d['format']=='epub':
                if supplied:
                    identity_match=[x for x in supplied['identities'] if x['document_id']==docid and x['canonical_sha256']==d['input_sha256'] and x['gold_sha256']==d['gold_sha256']]
                    if len(identity_match)!=1:raise ValueError('Supplied EPUB views STALE')
                    parts=zip_members(regular(root,d['input']))
                    for chapter in (1,2):
                        name=f'images/{docid}-chapter{chapter}.png';raw=regular(epub_render_packet,name);png(raw);(output/name).write_bytes(raw)
                        xhtml=parts[f'OEBPS/chapter{chapter}.xhtml'];(output/'sources'/f'{docid}-chapter{chapter}.txt').write_bytes(xhtml)
                        content.append(f'<h2>Capitolo EPUB {chapter}, vista effettiva 1080×1600</h2><p>Render locale osservato: il target browser R precedente è NON PASS per namespace Chrome annidati. Nessuna attestazione R sul renderer. Questi byte sono copiati e verificati nel target R di composizione del packet.</p><img src="{name}" width="740" alt="Capitolo actual"/><pre>'+escape(xhtml.decode())+'</pre>')
                    renderer_dependencies.append({'packet':descriptor(epub_render_packet/'packet.json'),'source_identity':identity_match[0],'renderer_R_status':'NON_PASS_NESTED_NAMESPACE','composition_method':'reuse exact actual PNGs from preserved failed browser target; canonical XHTML/gold identity equal; no renderer executed by this target','views':[descriptor(epub_render_packet/f'images/{docid}-chapter{i}.png') for i in (1,2)]})
                else:
                # Render actual canonical XHTML, CSS and PNG in a separate
                # trusted local workspace. Only screenshots enter the viewer.
                    parts=zip_members(regular(root,d['input']));browser=output.parent/(output.name+'-browser-'+docid);browser.mkdir()
                    shorttmp=Path('/tmp/a002-r001')/('v-'+digest_bytes(str(output).encode())[:10]);shorttmp.mkdir(mode=0o700)
                    chrome_env={**environment,'TMPDIR':str(shorttmp)}
                    for name,payload in parts.items():
                        if payload is None:continue
                        p=browser/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(payload)
                    for chapter in (1,2):
                        source=browser/f'OEBPS/chapter{chapter}.xhtml'
                        if not source.exists():raise ValueError('EPUB chapter path not found')
                        target=output/'images'/f'{docid}-chapter{chapter}.png'
                        argv=['/opt/google/chrome/chrome','--headless=new','--disable-gpu','--disable-background-networking','--disable-component-update','--disable-extensions','--no-first-run','--no-default-browser-check','--user-data-dir='+str(browser/('profile'+str(chapter))),'--window-size=1080,1600','--screenshot='+str(target),source.as_uri()]
                        tool(argv,browser,chrome_env)
                        png(target.read_bytes())
                        content.append(f'<h2>Capitolo EPUB {chapter}, viewport 1080×1600</h2><img src="images/{target.name}" width="740" alt="Capitolo actual"/>')
                        (output/'sources'/f'{docid}-chapter{chapter}.txt').write_bytes(source.read_bytes())
            if d['format'] in ('txt','rtf','tex','control-markdown'):
                names=[d['input']]
                if d['format']=='tex':names += [(Path(d['input']).parent/p).as_posix() for p in d['includes']]
                for k,name in enumerate(names):
                    raw=regular(root,name).decode('utf-8' if d['format']!='rtf' else 'ascii')
                    content += ['<h2>Sorgente canonico '+escape(name)+'</h2><pre>'+escape(raw)+'</pre>']
                    (output/'sources'/f'{docid}-{k}.txt').write_text(raw,encoding='utf-8')
            # Illustrative plate is explicitly source authoring, not compiled
            # TeX or an output judgment. Unicode baseline shapes are secondary.
            if d['format']=='tex':
                content.append('<p>Tavola matematica illustrativa: sorgente LaTeX e AST manuale sotto ogni formula; non compilazione né verifica tipografica TeX.</p>')
                plate=['<svg xmlns="http://www.w3.org/2000/svg" width="680" height="520"><rect width="680" height="520" fill="white"/><text x="20" y="28" font-family="DejaVu Sans" font-size="16">Tavola illustrativa — TeX non compilato</text>']
                y=60
                for block in g['blocks']:
                    for formula in block.get('formulas',[]):
                        plate.append('<text x="20" y="'+str(y)+'" font-family="DejaVu Sans" font-size="13">'+escape(formula['reference_latex'])+'</text>');y+=25
                        for line in formula.get('visible_lines',[]):
                            plate.append('<text x="45" y="'+str(y)+'" font-family="DejaVu Sans Mono" font-size="18">'+escape(line)+'</text>');y+=25
                        y+=12
                plate.append('</svg>')
                from safety import svg
                raw=''.join(plate).encode();svg(raw);(output/'images/tex-illustrative.svg').write_bytes(raw)
                content.append('<img src="images/tex-illustrative.svg" alt="Illustrazione da sorgenti manuali, non compilazione"/>')
            for b in g['blocks']:
                bid=b['block_id'];loc=b['locator'];page=b.get('page')
                content.append('<h2 id="'+escape(bid)+'">'+escape(bid+' · '+b['type'])+'</h2><pre>Locator '+escape(str(loc))+'</pre>')
                if d['format']=='pdf' and page in render_cache:
                    if isinstance(render_cache[page],str):
                        render_cache[page]=(render_cache[page],decode((output/render_cache[page]).read_bytes()))
                    _,(width,height,channels,rows)=render_cache[page]
                    x1,y1,x2,y2=loc['bbox'];left=max(0,int(x1*width/595));right=min(width,int(x2*width/595)+1);top=max(0,int(y1*height/842));bottom=min(height,int(y2*height/842)+1)
                    crop=f'images/{docid}-{bid}.png'
                    (output/crop).write_bytes(encode(right-left,bottom-top,channels,[r[left*channels:right*channels] for r in rows[top:bottom]]))
                    content.append(f'<img src="{crop}" alt="Ritaglio canonico"/>')
                transcript=b['text']
                content.append('<h3>Trascrizione completa attesa</h3><pre>'+escape(transcript)+'</pre>')
                if b.get('cells'):content.append('<pre>Celle con coordinate (stringa vuota ≠ assenza):\n'+escape(str(b['cells']))+'</pre>')
                for f in b.get('formulas',[]):
                    content.append('<pre>Formula LaTeX: '+escape(f['reference_latex'])+'\nAST manuale: '+escape(str(f['structure_expected']))+'</pre>')
                if b.get('caption'):content.append('<p>Caption: '+escape(b['caption'])+'</p>')
                if b.get('asset'):
                    source=root/(docid+'/assets/'+b['asset']) if is_control else root/'assets'/b['asset']
                    target='images/'+docid+'-asset-'+b['asset'];shutil.copyfile(source,output/target)
                    content.append('<p>Asset originale atteso (assi, regioni, legenda):</p><img src="'+target+'" alt="Figura originale"/>')
                if b.get('references'):content.append('<pre>Riferimenti: '+escape(str(b['references']))+'</pre>')
                cp=critical(transcript+' '+b.get('caption','')+' '+' '.join(c for row in b.get('cells',[]) for c in row)+' '+' '.join(f['reference_latex']+' '+' '.join(f.get('visible_lines',[])) for f in b.get('formulas',[])))
                content.append('<h3>Token critici, byte UTF-8 e code point</h3><pre>'+escape('\n'.join(str(v) for v in cp))+'</pre>')
                props=['content']+(['formula'] if b.get('formulas') else [])+(['figure'] if b.get('asset') else [])+(['table'] if b.get('cells') else [])
                for prop in props:
                    jid=docid+':'+bid+':'+prop
                    prompt={'content':'Confrontare testo completo, numeri/segni, nota/link/bibliografia con il canonico; raster: forma visibile e convenzione trascrizione, non encoding.','formula':'Confrontare segni, pedici/apici, argomenti, frazione, righe/celle/delimitatori; nessun giudizio sugli output.','figure':'Confrontare figura, assi, regioni e legenda, associazione con caption.','table':'Confrontare ogni coordinata/cella, inclusa la cella presente vuota.'}[prop]
                    judgments.append({**identity,'judgment_id':jid,'property':prop,'prompt':prompt,'required':True})
                    content.append('<p><strong>'+escape(jid)+'</strong>: '+escape(prompt)+'</p>')
            judgments.append({**identity,'judgment_id':docid+':order','property':'order','prompt':'Confrontare ordine completo, pagine/capitoli/unità e contenuti esclusivi; native/raster distinti.','required':True})
            page='<!doctype html><html lang="it"><head><meta charset="utf-8"/><title>'+escape(docid)+'</title><link rel="stylesheet" href="packet.css"/></head><body>'+''.join(content)+'</body></html>'
            html(page);(output/(docid+'.html')).write_text(page,encoding='utf-8');index.append('<p><a href="'+docid+'.html">'+escape(docid)+'</a></p>')
    (output/'packet.css').write_text('body { font-family: sans-serif; margin: 2em; max-width: 1000px; } pre { white-space: pre-wrap; overflow-wrap: anywhere; border: 1px solid gray; padding: 1em; } img { max-width: 100%; } h2 { border-top: 2px solid gray; padding-top: 1em; }',encoding='utf-8')
    page='<!doctype html><html lang="it"><head><meta charset="utf-8"/><title>Packet V</title><link rel="stylesheet" href="packet.css"/></head><body>'+''.join(index)+'</body></html>';html(page);(output/'index.html').write_text(page,encoding='utf-8')
    write_new(output/'judgments.json',judgments)
    with (output/'judgments-template.tsv').open('x',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['document_id','judgment_id','decision','reason','verifier','verified_at'],delimiter='\t');writer.writeheader()
        writer.writerows({'document_id':r['document_id'],'judgment_id':r['judgment_id']} for r in judgments)
    files=[r for r in tree_inventory(output) if r['kind']=='regular']
    packet={'schema_version':1,'kind':'gold-and-measurer-controls-only','identities':identities,'judgments':len(judgments),'gold_judgments':sum(not j['control'] for j in judgments),'control_judgments':sum(j['control'] for j in judgments),'files':files,'source_manifest':descriptor(corpus/'manifest.json'),'mechanical':descriptor(mechanical),'control_m':descriptor(control_m),'created_utc':datetime.now(timezone.utc).isoformat(),'response':{'status':'PENDING','verifier':None},'outputs_review':'UNAVAILABLE: separate human review not assigned'}
    if supplied:verify_packet(epub_render_packet)
    packet['renderer_dependencies']=renderer_dependencies
    packet['packet_sha256']=digest(packet);write_new(output/'packet.json',packet)
    print('PACKET_READY',packet['packet_sha256'],packet['judgments'])


def verify_packet(root):
    p=load(root/'packet.json');sha=p.pop('packet_sha256')
    if digest(p)!=sha:raise ValueError('Packet metadata STALE')
    for row in p['files']:
        if digest_bytes(regular(root,row['path']))!=row['sha256']:raise ValueError('Packet file STALE: '+row['path'])
    p['packet_sha256']=sha;return p


def import_response(root,response,packet_sha,output,author=None,received_at=None,receipt_method='actual response receipt at import; source message time unavailable'):
    packet=verify_packet(root)
    if packet.get('schema_version')==2:
        from document_response import import_document_response
        if not author:raise ValueError('Actual response author required, no new user designation needed')
        return import_document_response(root,response,packet_sha,output,author,received_at,receipt_method)
    output.mkdir(exist_ok=False)
    raw=regular(response.parent,response.name);(output/'received.tsv').write_bytes(raw)
    expected=load(root/'judgments.json');ids={j['judgment_id']:j for j in expected}
    rows=list(csv.DictReader(io.StringIO(raw.decode('utf-8')),delimiter='\t'));seen=set();accepted=[]
    if packet_sha!=packet['packet_sha256']:status='STALE'
    else:
        ambiguous=False;pending=False;rejected=False
        for row in rows:
            jid=row.get('judgment_id');decision=row.get('decision','')
            if jid not in ids or jid in seen or row.get('document_id')!=ids[jid]['document_id']:ambiguous=True;continue
            seen.add(jid)
            if decision not in ('ACCEPTED','REJECTED','PENDING','AMBIGUOUS'):pending=True;continue
            if not row.get('verifier') or not row.get('verified_at'):pending=True;continue
            try:
                date=datetime.fromisoformat(row['verified_at'])
                if date.tzinfo is None:raise ValueError('timezone')
            except ValueError:ambiguous=True;continue
            if decision=='REJECTED':rejected=True
            elif decision=='AMBIGUOUS':ambiguous=True
            elif decision=='PENDING':pending=True
            else:accepted.append(jid)
        status='REJECTED' if rejected else 'AMBIGUOUS' if ambiguous else 'PENDING' if pending or seen!=set(ids) else 'ACCEPTED'
    result={'schema_version':1,'source':descriptor(response),'original_received_sha256':digest_bytes(raw),'packet_sha256_received':packet_sha,'packet_sha256_current':packet['packet_sha256'],'kind':'human_visual_response_as_received','rows':rows,'status':status,'accepted_judgments':accepted,'imported_utc':datetime.now(timezone.utc).isoformat(),'documents':[{'document_id':i['document_id'],'canonical_sha256':i['canonical_sha256'],'gold_sha256':i['gold_sha256'],'status':'ACCEPTED' if status!='STALE' and all(j['judgment_id'] in accepted for j in expected if j['document_id']==i['document_id']) else status} for i in packet['identities']]}
    write_new(output/'response.json',result);print('V_'+status)
    return 0 if status=='ACCEPTED' else 5


def main():
    p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='command',required=True)
    b=s.add_parser('build');b.add_argument('--corpus',type=Path,required=True);b.add_argument('--mechanical',type=Path,required=True);b.add_argument('--control-m',type=Path,required=True);b.add_argument('--output',type=Path,required=True);b.add_argument('--environment',type=Path,required=True)
    b.add_argument('--epub-render-packet',type=Path)
    i=s.add_parser('import-response');i.add_argument('--packet',type=Path,required=True);i.add_argument('--response',type=Path,required=True);i.add_argument('--packet-sha256',required=True);i.add_argument('--output',type=Path,required=True)
    i.add_argument('--author');i.add_argument('--received-at');i.add_argument('--receipt-method',default='actual response receipt at import; source message time unavailable')
    a=p.parse_args()
    if a.command=='build':
        env=load(a.environment);build(a.corpus.absolute(),a.mechanical,a.control_m,a.output.absolute(),env.get('environment',env),a.epub_render_packet.absolute() if a.epub_render_packet else None);return 0
    return import_response(a.packet,a.response,a.packet_sha256,a.output,a.author,a.received_at,a.receipt_method)

if __name__=='__main__':raise SystemExit(main())
