"""F43–F45: finite, deterministic interactions; manual oracles before product.

No random generation, V, engine or expected value obtained from the reader.
"""
from copy import deepcopy
from html.parser import HTMLParser
from pathlib import Path
import shutil,sys
import pytest
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/benchmarks'))
from benchmark import observe_bundle
import audit as auditor
from metrics import score,evaluate_exit,known_text
TAIL='\n\n# Visible heading\n\nAfter words.'
ASSET=ROOT/'tests/fixtures/conversion-corpus/assets/figure-it.png'
HOSTS=['paragraph','quote','list','note','note-tab','cell']
BAD=['<img src="../outside.png">','<script>alert(1)</script>','<img src="p.png" onerror="x()">']
LITERAL=['<img src="../outside.png"/>','<script>x</script>','<b onclick="x">literal</b>']
def transport(value,host):
    if host=='cell':return '| h |\n| --- |\n| '+value+' |'
    return {'paragraph':value,'quote':'> '+value.replace('\n','\n> '),'list':'- '+value.replace('\n','\n    '),'note':'[^n]: '+value.replace('\n','\n    '),'note-tab':'[^n]: '+value.replace('\n','\n\t')}[host]
def gold(text,host='paragraph'):
    if host=='cell':first={'block_id':'t','type':'table','text':'','header_rows':1,'cells':[['h'],[text]],'page':1}
    else:
        first={'block_id':'p','type':'note' if host.startswith('note') else 'paragraph','text':text,'page':1}
        if host.startswith('note'):first.update(note_id='n',anchor='n')
    return {'document_id':'r010-synthetic','blocks':[first,{'block_id':'h','type':'heading','text':'Visible heading','page':1},{'block_id':'a','type':'paragraph','text':'After words.','page':1}]}
CASES=[]
def add(ident,raw,kind,reason,host='paragraph',text='Known words.',**extra):
    CASES.append(dict(id=ident,raw=raw,kind=kind,reason=reason,host=host,gold=gold(text,host),text=text,**extra))
# Exact reviewer discriminants. A9 accepts the explicit unsupported alternative
# authorized by prompt34, preserving the fact that the resource is unrepresented.
for ident,value,host in [
 ('A1','See `x '+BAD[0]+' tail.','paragraph'),('A2','See $x '+BAD[0]+' tail.','paragraph'),
 ('A3','See [a](b c) '+BAD[0]+' tail.','paragraph'),('A4','See `x '+BAD[1]+' tail.','paragraph'),
 ('A5','See `x '+BAD[0]+' tail.','quote'),('A6','`x '+BAD[0],'cell'),
 ('A7','See `x` '+BAD[0]+' tail.','paragraph'),('A8','See '+BAD[0]+' `x tail.','paragraph'),
 ('A10','See $x '+BAD[2]+' tail.','list')]:add(ident,transport(value,host)+'\n'+TAIL+'\n','unsafe','UNSAFE_REFERENCE',host)
add('A9','See `x <img src="p.png"> tail.\n'+TAIL+'\n','safe-fallback','UNSUPPORTED_RESOURCE_SYNTAX')
for ident,fragment,text,host in [
 ('GPT-plain','Known words. `'+LITERAL[0]+'`','Known words. '+LITERAL[0],'paragraph'),
 ('GPT-before','Known `'+LITERAL[0]+'` <span>words.</span>','Known '+LITERAL[0]+' words.','paragraph'),
 ('GPT-after','Known <span>words.</span> `'+LITERAL[0]+'`','Known words. '+LITERAL[0],'paragraph'),
 ('GPT-script','Known <span>words.</span> `'+LITERAL[1]+'`','Known words. '+LITERAL[1],'paragraph'),
 ('GPT-event','Known <span>words.</span> `'+LITERAL[2]+'`','Known words. '+LITERAL[2],'paragraph'),
 ('GPT-quote','Known <span>words.</span> `'+LITERAL[0]+'`','Known words. '+LITERAL[0],'quote')]:add(ident,transport(fragment,host)+TAIL,'literal',None,host,text,spelling='`')
add('GPT-real','Known <span>words.</span> <img src="../outside.png"/>'+TAIL,'unsafe','UNSAFE_REFERENCE')
for ident,value in [('B1','<div <!-- -- ><img src="../outside.png">\n'),('B2','<img <!-- src="../outside.png"> -->\n'),('B3','```{=html}\n<div <!-- -- ><script>x</script>\n```\n'),('B4','<alpha <!--\n\nHidden words.\n<img src="../outside.png">\n-->\n')]:
    add(ident,value+TAIL+'\n','F36','UNSUPPORTED_RESOURCE_SYNTAX',pure_first={'B1':'div','B2':'img','B3':'div','B4':'alpha'}[ident])
PREFIXES=['`x ','``x ','$x ','$$x ','[a](b c) ','[a `'+LITERAL[0]+'`](b c) ']
for host in HOSTS:
    for pi,prefix in enumerate(PREFIXES):
        for bi,bad in enumerate(BAD):add(f'fallback-{host}-{pi}-{bi}',transport('È See '+prefix+bad+' tail.',host)+TAIL,'unsafe','UNSAFE_REFERENCE',host)
    for li,literal in enumerate(LITERAL):
        for ti,ticks in enumerate(('`','``')):
            for order in (0,1):
                html='<span>words.</span>';fragment='È Known '+(html+' '+ticks+literal+ticks if order==0 else ticks+literal+ticks+' '+html)
                text='È Known '+('words. '+literal if order==0 else literal+' words.')
                add(f'literal-{host}-{li}-{ti}-{order}',transport(fragment,host)+TAIL,'literal',None,host,text,spelling=ticks)
    for ei,error in enumerate(('<span class="a" class="b">words.</span>','<span>'*17+'words.'+'</span>'*17)):
        for order in (0,1):
            literal='`'+LITERAL[order]+'`';fragment='È Known '+(error+' '+literal if order==0 else literal+' '+error)
            add(f'structure-literal-{host}-{ei}-{order}',transport(fragment,host)+TAIL,'structure','UNSUPPORTED_RESOURCE_SYNTAX',host)
    for pi,prefix in enumerate(('`x ','$x ','[a](b c) ')):
        add(f'safe-fallback-{host}-{pi}',transport('È See '+prefix+'<img src="p.png"> tail.',host)+TAIL,'safe-fallback','UNSUPPORTED_RESOURCE_SYNTAX',host)
    for order in (0,1):
        # Closed math remains math; the deliberately prose-only gold omits this
        # formula and therefore is a negative score oracle, not a false PASS.
        html='<span>words.</span>';math='$'+LITERAL[0]+'$'
        fragment='È Known '+(html+' '+math if order==0 else math+' '+html)+' end.'
        text='È Known '+('words.  end.' if order==0 else ' words. end.')
        add(f'math-closed-{host}-{order}',transport(fragment,host)+TAIL,'math-closed',None,host,text,formula=LITERAL[0])
    for ki,(opening,close) in enumerate((('<!--','-->'),('<![CDATA[',']]>'))):
        for closed in (False,True):
            token=opening+'\nHidden words.\n<img src="../outside.png">\n$x_i$\n'+(close if closed else '')
            fragment='È See `x '+token
            raw=transport(fragment,host)+(TAIL if closed else '')
            at=raw.index(opening);end=raw.index(close,at)+len(close) if closed else len(raw)
            add(f'token-{host}-{ki}-{closed}',raw,'token','UNSUPPORTED_RESOURCE_SYNTAX',host,token=raw[at:end],closed=closed)
    for prefix,closing in [('`x ','`'),('$x ','$')]:
        add(f'foreign-{host}-{ord(closing)}',transport('È See '+prefix+BAD[0],host)+TAIL+'\n\n'+closing,'unsafe','UNSAFE_REFERENCE',host)
    for order in (0,1):
        first,second=(BAD[0],BAD[1]) if order==0 else (BAD[1],BAD[0])
        fragment='È See `'+LITERAL[2]+'` <span class="a" class="b">x</span> `unclosed '+first+' '+second
        add(f'first-unsafe-{host}-{order}',transport(fragment,host)+TAIL,'unsafe','UNSAFE_REFERENCE',host,detail_contains='Path evasivo' if order==0 else 'script')

def pipeline(path,raw,g):
    path.mkdir(parents=True,exist_ok=True);(path/'document.md').write_bytes(raw.encode());shutil.copyfile(ASSET,path/'p.png')
    o=observe_bundle(path,path/'observations.json');a=auditor.audit(path);m=score(deepcopy(g),o,a,verification=None)
    return o,a,m
def checks(c,o,a,m,opened):
    raw=c['raw'];result={'reason':a['reason']==c['reason'],'status':a['status']==('COMPLETE' if c['reason'] is None else 'INCOMPLETE'),'no-evasive-open':'../outside.png' not in opened,'raw-bytes':all(n['raw']==raw[n['offset']['char_start']:n['offset']['char_end']] and n['offset']['byte_start']==len(raw[:n['offset']['char_start']].encode()) and n['offset']['byte_end']==len(raw[:n['offset']['char_end']].encode()) for n in o['nodes']),'denominator':m['totals']['text_ref_codepoints']==sum(len(b['text']) for b in c['gold']['blocks']),'arithmetic':m['totals']['word_errors']==sum(v['known_word_edits'] for v in m['WER_decomposition']['entries'])}
    if c['kind']=='unsafe':result.update(detail=bool(a.get('detail')),FAIL=m['score_exit']==3,evaluate=evaluate_exit(2,m['score_exit'])==2)
    else:
        result['no-assets']=not a['assets'] and 'p.png' not in opened
        result['no-image-ref']=not any(r['kind']=='image' for n in o['nodes'] for r in n['references'])
    if c.get('detail_contains'):result['first-unsafe']=c['detail_contains'] in a.get('detail','')
    if c['kind']=='literal':
        first=o['nodes'][0];value=first['cells'][1][0] if c['host']=='cell' and first['type']=='table' else first['text']
        result.update(text=value==c['text'],types=[n['type'] for n in o['nodes']]==['table' if c['host']=='cell' else 'note' if c['host'].startswith('note') else 'paragraph','heading','paragraph'],external=len(o['nodes'])>=3 and o['nodes'][-2]['text']=='Visible heading' and o['nodes'][-1]['text']=='After words.',gate=m['automatic_gates']=='PASS',provisional=m['quality']=='INDETERMINATE' and m['score_exit']==5,zero_text=m['totals']['text_errors']==m['totals']['word_errors']==0)
        if c['host']!='cell':
            code=[v for v in first.get('segments',[]) if v['type']=='code'];at=raw.index(c['spelling']);result['code-map']=len(code)==1 and code[0]['source_position']['char']==at and code[0]['source_position']['byte']==len(raw[:at].encode())
    if c['kind'] in ('safe-fallback','structure','token','F36'):
        result.update(FAIL=m['score_exit']==3,unsupported={'metric':'bundle','reason':'UNSUPPORTED_RESOURCE_SYNTAX'} in m['failures'])
    if c['kind'] in ('safe-fallback','structure'):
        result['external']=any(n['type']=='heading' and n['text']=='Visible heading' for n in o['nodes']) and o['nodes'][-1]['text']=='After words.'
    if c['kind']=='token':
        u=[v for n in o['nodes'] for v in n.get('unknown',[]) if v.get('raw')==c['token'] and v.get('profile')=='html-tokens-v1'];at=raw.index(c['token'])
        result['opaque']=len(u)==1 and not any('Hidden' in known_text(n) for n in o['nodes']) and not any(n['formulas'] for n in o['nodes'])
        result['token-map']=bool(u) and u[0]['source_position']['char']==at and u[0]['source_position']['byte']==len(raw[:at].encode())
        if c['closed']:result['external']=o['nodes'][-1]['text']=='After words.'
    if c['kind']=='F36':
        class Pure(HTMLParser):
            def handle_starttag(self,tag,attrs):self.tags.append(tag)
            handle_startendtag=handle_starttag
        p=Pure();p.tags=[];p.feed(raw);p.close();result['pure-divergence']=bool(p.tags) and p.tags[0]==c['pure_first']
    if c['kind']=='math-closed':
        result['formula-inert']=len([v for n in o['nodes'] for v in n['formulas'] if v['raw']==c['formula']])==1
        result['external']=o['nodes'][-1]['text']=='After words.'
    return result
@pytest.mark.parametrize('case',CASES,ids=[c['id'] for c in CASES])
def test_inline_HTML_domain_pipeline(tmp_path,monkeypatch,case):
    opened=[];original=auditor.regular
    def spy(root,name,*args,**kw):opened.append(name);return original(root,name,*args,**kw)
    monkeypatch.setattr(auditor,'regular',spy);o,a,m=pipeline(tmp_path,case['raw'],case['gold']);result=checks(case,o,a,m,opened)
    assert all(result.values()),result

@pytest.mark.parametrize('first,second,detail',[(BAD[0],BAD[1],'Path evasivo'),(BAD[1],BAD[0],'script')])
def test_inline_guard_visits_after_first_unsafe(monkeypatch,first,second,detail):
    import observations as reader
    visited=[];original=reader.PassiveHTML.handle_starttag
    def spy(self,tag,attrs):
        visited.append(tag);return original(self,tag,attrs)
    monkeypatch.setattr(reader.PassiveHTML,'handle_starttag',spy)
    raw='È `'+LITERAL[2]+'` <span class="a" class="b">x</span> `unclosed '+first+' '+second+' '+BAD[2]
    with pytest.raises(reader.Unsafe,match=detail):reader.inline(raw)
    assert visited==['span','img' if first==BAD[0] else 'script','script' if first==BAD[0] else 'img','img']
