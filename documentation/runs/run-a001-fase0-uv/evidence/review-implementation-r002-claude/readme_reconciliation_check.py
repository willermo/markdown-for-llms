"""Sonda di sola lettura: verifica 74 fence + 9 inline della riconciliazione README."""
import json,hashlib,collections,re,time
from pathlib import Path
t=time.monotonic()
R=Path('/home/davide/workarea/markdown-for-llms')
E=R/'temp/run-a001-fase0-uv/evidence/implementation-r001/fix-review-r001/readme-reconciliation-verified.json'
d=json.loads(E.read_text())
orig=Path(d['original']['path']).read_bytes()
out={'original_sha_ok':hashlib.sha256(orig).hexdigest()==d['original']['sha256']}
lines=orig.decode().split('\n')
# fence reali nel README originale
fences=[];i=0
while i<len(lines):
    m=re.match(r'^(\s*)(```+|~~~+)(.*)$',lines[i])
    if m:
        ind,mark=m.group(1),m.group(2);j=i+1
        while j<len(lines) and not re.match(r'^\s*'+re.escape(mark[0])+'{%d,}\s*$'%len(mark),lines[j]):j+=1
        fences.append((i+1,j+1,m.group(3).strip(),'\n'.join(lines[i+1:j])));i=j+1
    else:i+=1
out['fences_found']=len(fences)
byfirst={f[0]:f for f in fences}
mism=[];disp=collections.Counter();dest_missing=[];noreason=[]
for b in d['blocks']:
    f=byfirst.get(b['first_line'])
    if not f or f[1]!=b['last_line']:mism.append((b['id'],'lines'));continue
    body=f[3]
    cands={hashlib.sha256(x.encode()).hexdigest() for x in (body,body.strip(),body+'\n',body.strip()+'\n')}
    if b['text_sha256'] not in cands:mism.append((b['id'],'sha'))
    disp[b['disposition']]+=1
    if not b.get('reason'):noreason.append(b['id'])
    dst=b.get('destination')
    if dst and not (R/dst.split('#')[0]).exists():dest_missing.append((b['id'],dst))
    if not dst and b['disposition']!='rimosso':dest_missing.append((b['id'],None))
out.update(blocks=len(d['blocks']),ids_unique=len({b['id'] for b in d['blocks']})==74,mismatch=mism,dispositions=dict(disp),dest_missing=dest_missing,no_reason=noreason)
inl=[];idisp=collections.Counter()
for x in d['inline']:
    line=lines[x['line']-1];ok=x['text'] in line
    inl.append((x['line'],ok,hashlib.sha256(x['text'].encode()).hexdigest()==x['text_sha256']));idisp[x['disposition']]+=1
    dst=x.get('destination')
    if dst and not (R/dst).exists():dest_missing.append(('inline',dst))
out.update(inline=inl,inline_dispositions=dict(idisp),planning_vs_actual_differs=sum(1 for b in d['blocks'] if b.get('planning_text_sha256')!=b['text_sha256']),seconds=time.monotonic()-t)
print(json.dumps(out,ensure_ascii=False,indent=1))
