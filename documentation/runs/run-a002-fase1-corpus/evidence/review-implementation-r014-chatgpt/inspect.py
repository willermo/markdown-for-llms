"""Review statica: identità e confronti JSON ricevuti, nessun import prodotto."""
from pathlib import Path
import json, hashlib, difflib, resource
R=Path.cwd(); B=R/'temp/run-a002-fase1-corpus'; E=B/'evidence/implementation-r014'; O=Path(__file__).parent
def rawjson(p): return json.loads(Path(p).read_text())
common=rawjson(B/'snapshots/implementation-r014-review-r002.json')
original=rawjson(B/'snapshots/implementation-r014.json')
descriptors={str((R/d['path']).absolute()):d for d in original['files']+original['artifacts'] if d['kind']=='file'}
descriptors.update({str((R/d['path']).absolute()):d for d in common['files']+common['artifacts'] if d['kind']=='file'})
used={}; misses=[]
def bind(p):
 p=Path(p).absolute(); data=p.read_bytes(); sha=hashlib.sha256(data).hexdigest(); d=descriptors.get(str(p))
 row={'path':str(p),'bytes':len(data),'sha256':sha,'binding':'direct-common' if any(x.get('path')==str(p.relative_to(R)) for x in common['files']+common['artifacts']) else 'original-inventory'}
 if d is None: row['binding']='live-context-or-not-in-inventory'; misses.append(str(p))
 else: assert sha==d['sha256'],str(p)
 used[str(p)]=row; return data
def read(p): return json.loads(bind(p))
for receipt in O.glob('*.json'):
 d=rawjson(receipt)
 for arg in d.get('argv',[]):
  p=Path(arg)
  if p.is_file() and str(p.absolute()).startswith(str(R)) and not str(p.absolute()).startswith(str(O)):
   bind(p)
print('COMMON',hashlib.sha256((B/'snapshots/implementation-r014-review-r002.json').read_bytes()).hexdigest(),common['worktree_sha256'])
print('ORIGINAL',hashlib.sha256((B/'snapshots/implementation-r014.json').read_bytes()).hexdigest())
for n in ['reception.json','snapshot-receipt-r002.json','transition-r002.json']:
 print('RECEPTION',n,read(B/'evidence/supervisor-implementation-review-r014'/n))
for n in ['freeze-001.json','freeze-002.json','cases-003.json','check-baseline-001.json','check-grammar-001.json','check-after-final.json']:
 d=read(E/n); print('CASES',n,{k:v for k,v in d.items() if k!='rows'})
 if 'rows' in d: print('FAILURES',[r for r in d['rows'] if r.get('failures')])
expectations=read(E/'expectations-001.json')['cases']; after=read(E/'after-final.json')['rows']
cases=[c for n in ['cases-001.json','cases-002.json'] for c in read(E/n)['cases']]
baseline={r['id']:r for n in ['baseline-001.json','grammar-001.json'] for r in read(E/n)['rows']}
print('FINITE',len(cases),len({c['raw_sha256'] for c in cases}),max(len(c['raw'].encode()) for c in cases))
for c in cases:
 if c['id'].startswith('grammar-'): print('GRAMMAR',c['id'],c['raw'],json.loads(baseline[c['id']]['pandoc']['stdout'])['blocks'])
for ident in ['GPT-first-unsafe-invalid-uri','GPT-uri-validator-error','GPT-tilde-info-root','GPT-tilde-info-asset','CLA-K1a-quote-after-para-tilde-script','CLA-K2c-lazy-list-tilde-script','CLA-K2e-figure-line-then-tilde-script','CLA-K4a-span-braces-img-evasive','CLA-K4e-span-braces-img-safe','heading-attached','metadata-crlf']:
 r=next(r for r in after if r['id']==ident); e=next(r for r in expectations if r['id']==ident)
 print('DECISIVE',ident,'expected',e,'actual',{'detail':r['detail'],'error':r['error'],'visits':r['visits'],'events':r['events'],'reset':r['reset'],'audit':{k:r['audit'].get(k) for k in ['status','reason','detail','assets','unknown']}})
for name in ['mutations-inputs.json','mutations-results.json','metadata-preservation.json','delivery-checks-002.json','current-preflight.json','stage-s002/preparation.json','R-final-001/receipt.json','R-final-001/inside.json','H12-summary.json','final-workload/summary.json','cli-final/summary.json']:
 d=read(E/name)
 if name in ['mutations-results.json','metadata-preservation.json','current-preflight.json','H12-summary.json','final-workload/summary.json','cli-final/summary.json']: print('RECEIVED',name,d)
 else: print('RECEIVED',name,str(d)[:3500])
runtime=read(E/'runtime-byte-binding.json'); copies=read(E/'final-001-inputs.json'); copiesafter=read(E/'final-001-inputs-after.json'); assert copies==copiesafter
for d in runtime['files']: assert hashlib.sha256(Path(d['path']).read_bytes()).hexdigest()==d['sha256']
for d in copies:
 data=Path(d['path']).read_bytes(); assert hashlib.sha256(data).hexdigest()==d['sha256'] and data==Path(d['copy']).read_bytes()
print('RUNTIME',len(runtime['files']),len(copies),'MATCH source/copy/after')
summary=read(E/'rescore-001/summary.json'); checks=read(E/'derived-check-001.json'); attribution=read(E/'epub-delta-attribution.json')
fields=['quality','automatic_gates','score_exit','totals','denominators','CER','WER','blocks','order','pages','text_metric_availability','WER_decomposition','conditional_text_counts','text_lower_bound','alignment','failures','unavailable','details','metrics_version']
delta=[]; bounds={}; quality={}
for row,check in zip(summary['records'],checks['checks']):
 assert row['derived_measurement_id']==check['id']
 m=read(row['metrics']['path']); old=read(row['old_metrics']['path']); different={k:{'old':old[k],'new':m[k]} for k in fields if old[k]!=m[k]}
 assert different==check['actual_numeric_gate_delta']; assert set(different)<= {'failures','details'}
 if different: delta.append({'id':check['id'],'fields':list(different)})
 bounds[m['text_lower_bound']['status']]=bounds.get(m['text_lower_bound']['status'],0)+1
 quality[str(m['quality'])]=quality.get(str(m['quality']),0)+1
 for k in ['observations','old_observations','audit','gold','verification','run','previous_derived_final','conversion_final']: read(row[k]['path'])
print('DERIVED',len(summary['records']),delta,bounds,quality,'sets',summary['sets'],'attribution', {k:v for k,v in attribution.items() if k!='records'})
for n in ['compound-id-before.json','compound-id-after.json','hc2-ast.json']:
 print('AST-CORRECTION',n,read(E/n))
g=B/'evidence/review-implementation-r013-chatgpt'
for n in ['probe-results-001.json','probe-results-002.json','own-check-001.json']:
 d=read(g/n); print('HISTORY-GPT',n,'keys',list(d))
 if isinstance(d.get('rows'),list): print('history summary',[(r.get('id'),r.get('failures'),r.get('detail'),r.get('error')) for r in d['rows']])
c=B/'evidence/review-implementation-r013-claude'
for n in ['check-002.json','out-probe-001/summary.json']:
 d=read(c/n); print('HISTORY-CLA',n,'keys',list(d)); print(str(d)[:3200])
for d in original['artifacts']:
 if d['kind']=='file' and '/out-probe-001/' in d['path'] and ('ast' in d['path'].lower() or d['path'].endswith('/result.json')):
  bind(R/d['path'])
for n in ['received-bytes/manifest.json','gate-binding.json','current-S.json','current-B.json','current-I.json','current-origin-002.json','current-archives.json','current-installed.json']:
 d=read(E/n); print('GATE',n,str(d)[:3000])
for f in [R/'documentation/benchmarks/2026-10-11-a002-r014-contratti-sintassi-data.json',B/'snapshots/implementation-r014-delivery-r001.json']:
 d=read(f);print('PERMANENT-DELIVERY',str(f),list(d))
for f in [R/'scripts/benchmarks/observations.py',R/'scripts/benchmarks/safety.py']:
 oldpath=E/'received-bytes'/f.name
 if not oldpath.exists():
  candidates=list((E/'received-bytes').glob('*'+f.name)); print('RECEIVED-BYTE-NAMES',list((E/'received-bytes').iterdir())); continue
 old=bind(oldpath).decode();new=bind(f).decode();diff=''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True)))
 (O/(f.stem+'-independent.diff')).write_text(diff)
print('BINDINGS',len(used),'NOT BOUND',misses)
(O/'static-check.json').write_text(json.dumps({'status':'MATCH used descriptors and JSON comparisons','used':list(used.values()),'not_bound':misses,'metrics_fields':fields,'derived_delta':delta,'bounds':bounds,'quality':quality,'runtime_files':len(runtime['files']),'copies':len(copies),'product_imports':0,'own_runtime_tests':0},ensure_ascii=False,indent=2)+'\n')
