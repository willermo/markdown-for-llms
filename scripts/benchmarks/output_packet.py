"""Inventario passivo dei giudizi L04 non assegnati sugli output reali."""
import argparse
from html import escape
from pathlib import Path
from data import descriptor,digest,load,write_new
from campaign import manifest,check_file,check_record
from safety import regular,html,tree_inventory,png


def build(path,summary,output):
    m=manifest(path);result=load(summary)
    if result['manifest']!=descriptor(path) or not result['execution_complete'] or len(result['groups'])!=len(m['groups']):raise ValueError('Complete actual campaign required')
    output.mkdir(exist_ok=False);(output/'images').mkdir();questions=[];unique={};identities=[];pages=[]
    for g,group_descriptor in zip(m['groups'],result['groups']):
        group=load(check_file(group_descriptor));sections=['<h1>'+escape(g['group_id'])+'</h1><p>Output reali: contenuto escapato, nessuna esecuzione di Markdown/HTML/LaTeX. Giudizi umani L04 non assegnati. Non è il packet Vgold.</p>']
        expected=[{'block_id':b['block_id'],**f} for b in load(g['gold']['path'])['blocks'] for f in b.get('formulas',[])]
        sections+=['<h2>Riferimento manuale distinto dagli output</h2><pre>'+escape(str(expected))+'</pre>']
        for ident,row_descriptor in zip(g['attempts'],group['records']):
            row=check_record(check_file(row_descriptor),g,ident,m,load(row_descriptor['path'])['evaluate_exit']);obs=load(check_file(row['observations']));rawroot=Path(row['run']['path']).parent/'raw'
            identity={k:row[k] for k in ('experiment_id','document_id','attempt_id','source_sha256','gold_sha256','config_sha256','engine_id')};identities.append(identity)
            sections+=['<h2>'+escape(ident)+'</h2><pre>'+escape(str(identity))+'</pre><h3>Markdown effettivo completo (dato passivo)</h3><pre>'+escape(regular(rawroot,'document.md').decode())+'</pre>']
            metrics=load(row['metrics']['path'])
            for i,a in enumerate(metrics['asset_verifications']):
                if not a.get('content_verified') or not a.get('path','').endswith('.png'):continue
                raw=regular(rawroot,a['path']);png(raw)
                if __import__('data').digest_bytes(raw)!=a['sha256']:raise ValueError('Output asset STALE')
                name='images/'+ident+'-'+str(i)+'.png';(output/name).write_bytes(raw);sections+=['<p>Asset effettivo referenziato e verificato:</p><img src="'+name+'" alt="Asset actual"/>']
            for n in obs['nodes']:
                for index,f in enumerate(n.get('formulas',[])):
                    qid=ident+':'+n['observation_id']+':f'+str(index+1)
                    key=digest({'group':g['group_id'],'source':row['source_sha256'],'config':row['config_sha256'],'raw':f['raw'],'structure':f['math']['structure']})
                    unique.setdefault(key,[]).append(qid)
                    sections+=['<h3>'+escape(qid)+'</h3><pre>Raw LaTeX osservato: '+escape(f['raw'])+'\nStruttura osservata: '+escape(str(f['math']))+'\nIntervallo: '+escape(str(n['offset']))+'</pre>']
                    for kind in ('typography','semantics'):
                        questions.append({**identity,'judgment_id':qid+':'+kind,'kind':kind,'observation_id':n['observation_id'],'formula_index':index,'raw_formula':f['raw'],'observation_sha256':row['observations']['sha256'],'status':'UNAVAILABLE','assigned_verifier':None,'rendered_output_available':False,'reason':'Separate human output judgments not assigned; this viewer shows literal output, not a typography render','equivalent_repetition_group':key})
        target=g['group_id']+'.html';page='<!doctype html><html lang="it"><head><meta charset="utf-8"/><title>'+escape(g['group_id'])+'</title></head><body>'+''.join(sections)+'</body></html>';html(page);(output/target).write_text(page);pages.append(target)
    n=len(unique);estimate={'distinct_formula_cases':n,'actual_formula_occurrences':len(questions)//2,'judgments':len(questions),'prospective_minutes':round(10+2*len(pages)+n*70/60),'method':'10min setup +2min/path context +70s/distinct formula for two judgments; four equivalent repetitions grouped, each original identity retained','typography_limit':'No actual output renderer used; a later typography assignment must supply a compatible visual rendering before accepting those judgments','agreement':'not requested or assigned in this mandate'}
    page='<!doctype html><html lang="it"><head><meta charset="utf-8"/><title>L04 output inventory</title></head><body><h1>Inventario L04 — giudizi output non assegnati</h1><p>Gold V è separato. Nessuna firma o giudizio precompilato. Markdown e LaTeX sono dati passivi; tipografia non verificabile da un render output qui assente.</p><pre>'+escape(str(estimate))+'</pre>'+''.join('<p><a href="'+p+'">'+escape(p)+'</a></p>' for p in pages)+'</body></html>';html(page);(output/'index.html').write_text(page)
    write_new(output/'judgments.json',questions);write_new(output/'estimate.json',estimate)
    value={'schema_version':1,'kind':'actual-output-L04-inventory','status':'UNAVAILABLE','manifest':descriptor(path),'campaign_summary':descriptor(summary),'identities':identities,'judgments':len(questions),'estimate':estimate,'files':tree_inventory(output)};value['packet_sha256']=digest(value);write_new(output/'packet.json',value);return value


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('manifest','summary','output'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();build(a.manifest,a.summary,a.output);return 0

if __name__=='__main__':raise SystemExit(main())
