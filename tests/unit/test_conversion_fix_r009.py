"""F39–F42: finite interaction classes with hand-authored synthetic oracles.

No gold comes from the reader. Deterministic covering rows, no random fuzzing.
The pipeline checks content/reference/mapping as well as audit and score.
"""
from copy import deepcopy
from html.parser import HTMLParser
from pathlib import Path
import shutil
import sys
import pytest
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/benchmarks'))
from benchmark import observe_bundle
import audit as auditor
from metrics import score, known_text, evaluate_exit
from observations import markdown_lexemes, tag_end

TAIL='\n\n# Visible heading\n\nAfter words.'
URL='https://example.invalid'
ASSET=ROOT/'tests/fixtures/conversion-corpus/assets/figure-it.png'
def transport(value,host):
    return {'paragraph':value,'quote':'> '+value.replace('\n','\n> '),'list':'- '+value.replace('\n','\n    '),'note':'[^n]: '+value.replace('\n','\n    '),'note-tab':'[^n]: '+value.replace('\n','\n\t')}[host]
def html_transport(value,host):
    return {'block':value,'inline':'See '+value,'raw':'```{=html}\n'+value+'\n```','quote':'> '+value,'cell':'| h |\n| --- |\n| '+value+' |','caption':'| h |\n| --- |\n| x |\n\nTable: '+value,'wrapper':'<div>'+value+'</div>'}[host]
def gold(first='Known words.',kind='paragraph',link=False):
    b={'block_id':'p','type':kind,'text':first,'page':1}
    if kind=='note':b.update(note_id='n',anchor='n')
    if link:b['references']=[{'kind':'link','text':first.removeprefix('È See '),'target':URL}]
    return {'document_id':'r009-synthetic','blocks':[b,{'block_id':'h','type':'heading','text':'Visible heading','page':1},{'block_id':'a','type':'paragraph','text':'After words.','page':1}]}
CASES=[]
# Exact reviewer discriminants, including controls and safe D7.
EXACT=[
('<div class="a" class="b">\n<img src="../outside.png">\n</div>\n','D1'),
('<p id="a" id="b">x</p>\n<script src="https://example.invalid/x.js"></script>\n','D2'),
('```{=html}\n<div class="a" class="b"></div>\n<img src="p.png" onerror="x()">\n```\n','D3'),
('<div>\n<img src="../outside.png">\n</div>\n','D4'),
('See <span class="a" class="b">x</span> <img src="../outside.png">\n','D5'),
('<div>'*17+'<img src="../outside.png">'+'</div>'*17+'\n','D6'),
("<p title=it's><img src=\"../outside.png\"> Don't <!-- c --></p>\n",'D7')]
for raw,name in EXACT:CASES.append({'id':name,'class':'unsafe','raw':raw,'reason':'UNSAFE_REFERENCE'})
# Pair coverage: structural error, safety condition, order and transport.
ERRORS=['<span class="a" class="b">x</span>','<span>'*17+'x'+'</span>'*17]
BAD=['<img src="../outside.png">','<script>x</script>','<img src="p.png" onerror="x()">']
HOSTS=['block','inline','raw','quote','cell','caption']
for ei,error in enumerate(ERRORS):
    for bi,bad in enumerate(BAD):
        for order in (0,1):
            for hi,host in enumerate(HOSTS):
                fragment=error+bad if order==0 else bad+error
                CASES.append({'id':f'guard-{ei}-{bi}-{order}-{host}','class':'unsafe','raw':html_transport(fragment,host)+TAIL,'reason':'UNSAFE_REFERENCE'})
# Safe structural failure and inert tokens must not manufacture unsafe attrs.
for error in ERRORS:
    for host in ('block','inline','raw'):
        for opener,closer in [('<!--','-->'),('<![CDATA[',']]>')]:
            token=opener+'<script>x</script><img src="../outside.png" onerror="x()">'+closer
            CASES.append({'id':f'guard-inert-{len(CASES)}','class':'inert-error','raw':html_transport(error+token,host)+TAIL,'reason':'UNSUPPORTED_RESOURCE_SYNTAX'})
# Attribute values: quote state is decided at '=', never inside bare values.
# Every row has a fixed expected title and exact source tag boundary.
ATTRS=[
    ("title=it's", "it's"),('title=it"s','it"s'),
    ('title="it\'s > < <!--"',"it's > < <!--"),
    ("title='it\"s > < <![CDATA['",'it"s > < <![CDATA['),
    ('title=a<b','a<b'),('title=<!--literal','<!--literal'),
    ('title="<?x"','<?x'),("title='<!x'",'<!x'),
    ('title = "ü &amp; ñ"','ü & ñ'),('title=bare','bare')]
TOKENS=[('<!-- c -->','comment'),('<![CDATA[x > y]]>','cdata'),('<?x?>','processing-instruction'),('<!x>','declaration')]
for ai,(attribute,value) in enumerate(ATTRS):
    for hi,host in enumerate(('block','inline','raw','wrapper')):
        token,kind=TOKENS[(ai+hi)%len(TOKENS)]
        fragment='<p '+attribute+'><img src="p.png"> Don\'t '+token+'</p>'
        CASES.append({'id':f'attribute-{ai}-{host}','class':'safe-attribute','raw':html_transport(fragment,host)+TAIL,'reason':'UNSUPPORTED_RESOURCE_SYNTAX','token':token,'tag':'<p '+attribute+'>','title':value})
    CASES.append({'id':f'attribute-unsafe-{ai}','class':'unsafe','raw':'<p '+attribute+'><img src="../outside.png"> Don\'t <!-- c --></p>'+TAIL,'reason':'UNSAFE_REFERENCE'})
# Content-preserving transports: host, link/image, tick width, Unicode,
# literal block markers and inert opener spelling. Cover all pairs, not a
# Cartesian product across every dimension.
MARKERS=['- ','# ','> ','::: ','[^x]: ','1. ']
HOST_MD=['paragraph','quote','list','note','note-tab']
for hi,host in enumerate(HOST_MD):
    for image in (False,True):
        for ti,ticks in enumerate(('`','``')):
            for mi,marker in enumerate(MARKERS):
                # Use literal openers, without closure, exclusively inside code.
                opener=['<!--','<![CDATA[','<?x','<!x'][(hi+mi+ti)%4]
                continuation=('soft ' if host=='paragraph' else '')+marker+opener
                label='Known '+ticks+'primo\n'+continuation+ticks+' words.'
                fragment='È See '+('!' if image else '')+'['+label+']('+('p.png' if image else URL)+')'
                text='È See Known primo '+continuation+' words.'
                CASES.append({'id':f'domain-{host}-{int(image)}-{ti}-{mi}','class':'domain','raw':transport(fragment,host)+TAIL,'host':host,'image':image,'ticks':ticks,'reason':None,'text':text,'gold':gold(text,'note' if host.startswith('note') else 'paragraph',not image)})
# Exact minimum from ChatGPT, and a no-marker control.
for marker in ('- ',''):
    text='See Known first '+marker+'<!-- words.'
    g=gold(text);g['blocks'][0]['references']=[{'kind':'link','text':text.removeprefix('See '),'target':URL}]
    CASES.append({'id':'GPT-exact' if marker else 'GPT-control','class':'domain','raw':'> See [Known `first\n> '+marker+'<!--` words.]('+URL+')'+TAIL,'host':'quote','image':False,'ticks':'`','reason':None,'text':text,'gold':g})
# Never borrow closure from a subsequent producer. Outside code, true tokens
# keep the full closed/EOF body opaque through supported prefix transports.
for hi,host in enumerate(HOST_MD):
    value=transport('È See [Known `first',host)+TAIL+'\n\n` words.]('+URL+')'
    CASES.append({'id':'foreign-close-'+host,'class':'foreign','raw':value,'reason':None})
    for closed in (False,True):
        opener,closer=[('<!--','-->'),('<![CDATA[',']]>')][hi%2]
        body=opener+'\n\nHidden words.\n\n<img src="../outside.png">\n$x_i$\n'+(closer if closed else '')
        raw=transport('È See [Known `first\n- literal` '+body,host)+(TAIL if closed else '')
        at=raw.index(opener);end=raw.index(closer,at)+len(closer) if closed else len(raw)
        CASES.append({'id':f'real-token-{host}-{closed}','class':'token','raw':raw,'token':raw[at:end],'closed':closed,'reason':'UNSUPPORTED_RESOURCE_SYNTAX'})
# F36 incomplete-start, independently from attribute values above.
for host in ('block','inline','raw','quote','cell','caption','wrapper'):
    for closed in (True,False):
        token='<!--\n\nHidden words.\n<img src="../outside.png">\n'+('-->' if closed else '')
        raw=html_transport('<alpha '+token,host)+(TAIL if closed else '')
        at=raw.index('<!--');end=raw.index('-->',at)+3 if closed else len(raw)
        CASES.append({'id':f'prefix-{host}-{closed}','class':'token','raw':raw,'token':raw[at:end],'closed':closed,'reason':'UNSUPPORTED_RESOURCE_SYNTAX'})
# EOF quoted/bare attributes and closure after all inert-token families.
for attribute in ("title=it's",'title=it"s','title="<!-- literal >"',"title='<![CDATA[ literal >'"):
    for token,kind in TOKENS:
        eof=token.split('>')[0] if kind!='cdata' else '<![CDATA[x > y'
        raw='<p '+attribute+'><img src="p.png"> Don\'t '+eof
        CASES.append({'id':f'attribute-EOF-{len(CASES)}','class':'safe-attribute','raw':raw,'reason':'UNSUPPORTED_RESOURCE_SYNTAX','token':eof,'tag':'<p '+attribute+'>','title':None})

def pipeline(path,raw,g=None):
    path.mkdir(parents=True,exist_ok=True);(path/'document.md').write_bytes(raw.encode())
    shutil.copyfile(ASSET,path/'p.png')
    o=observe_bundle(path,path/'observations.json');a=auditor.audit(path)
    return o,a,score(deepcopy(g or gold()),o,a,verification=None)
def check_case(case,o,a,m,opened):
    raw=case['raw'];checks={'reason':a['reason']==case['reason'],'no-evasive-open':'../outside.png' not in opened,'bundle':a['status']==('COMPLETE' if case['reason'] is None else 'INCOMPLETE')}
    checks['raw-and-bytes']=all(n['raw']==raw[n['offset']['char_start']:n['offset']['char_end']] and n['offset']['byte_start']==len(raw[:n['offset']['char_start']].encode()) and n['offset']['byte_end']==len(raw[:n['offset']['char_end']].encode()) for n in o['nodes'])
    checks['denominators']=m['totals']['text_ref_codepoints']==sum(len(b['text']) for b in (case.get('gold') or gold())['blocks'])
    checks['arithmetic']=m['totals']['word_errors']==sum(v['known_word_edits'] for v in m['WER_decomposition']['entries'])
    if case['class']=='unsafe':
        checks.update(detail=bool(a.get('detail')),fail=m['score_exit']==3,evaluate=evaluate_exit(2,m['score_exit'])==2)
    if case['class']=='domain':
        checks.update(text=o['nodes'][0]['text']==case['text'],types=[n['type'] for n in o['nodes']]==['note' if case['host'].startswith('note') else 'paragraph','heading','paragraph'],external=o['nodes'][-2]['text']=='Visible heading' and o['nodes'][-1]['text']=='After words.',automatic=m['automatic_gates']=='PASS',provisional=m['quality']=='INDETERMINATE' and m['score_exit']==5,zero_errors=m['totals']['text_errors']==m['totals']['word_errors']==0,no_token=not markdown_lexemes(raw))
        refs=[r for n in o['nodes'] for r in n['references']]
        checks['reference']=[r['target'] for r in refs]==['p.png' if case['image'] else URL]
        if refs:
            at=raw.index('![' if case['image'] else '[Known');checks['ref-map']=refs[0]['source_position']['char']==at and refs[0]['source_position']['byte']==len(raw[:at].encode())
        code=[s for n in o['nodes'] for s in n.get('segments',[]) if s['type']=='code']
        at=raw.index(case['ticks']);checks['code-map']=len(code)==1 and code[0]['source_position']['char']==at and code[0]['source_position']['byte']==len(raw[:at].encode())
        if case['image']:checks['asset']=len(a['assets'])==1 and a['assets'][0]['sha256']==__import__('hashlib').sha256(ASSET.read_bytes()).hexdigest() and a['assets'][0]['dimensions']==[240,140] and 'p.png' in opened
    if case['class']=='safe-attribute':
        checks['image']=[r['target'] for n in o['nodes'] for r in n['references'] if r['kind']=='image']==['p.png']
        checks['asset']='p.png' in opened and len(a['assets'])==1
        checks['text']=any("Don't " in n['text'] for n in o['nodes'])
    if case['class']=='foreign':
        checks['no-borrow']=not any(r['kind']=='link' for n in o['nodes'] for r in n['references'])
        checks['external']=any(n['type']=='heading' and n['text']=='Visible heading' for n in o['nodes']) and any(n['text']=='After words.' for n in o['nodes'])
    if case['class']=='token':
        values=[u for n in o['nodes'] for u in n.get('unknown',[]) if u.get('raw')==case['token'] and u.get('profile')=='html-tokens-v1']
        checks['token-whole']=len(values)==1
        at=raw.index(case['token']);checks['token-map']=bool(values) and values[0]['source_position']['char']==at and values[0]['source_position']['byte']==len(raw[:at].encode())
        checks['body-opaque']=not any('Hidden' in known_text(n) for n in o['nodes']) and not a['assets'] and not any(n['formulas'] for n in o['nodes'])
        checks['unknown-not-wildcard']='p' in m['blocks']['missing'] and m['score_exit']==3
        if case['closed']:checks['external']=o['nodes'][-1]['text']=='After words.' and any(n['type']=='heading' and n['text']=='Visible heading' for n in o['nodes'])
    return checks
@pytest.mark.parametrize('case',CASES,ids=[c['id'] for c in CASES])
def test_interaction_pipeline(tmp_path,monkeypatch,case):
    opened=[];old=auditor.regular
    def spy(root,name,*a,**kw):opened.append(name);return old(root,name,*a,**kw)
    monkeypatch.setattr(auditor,'regular',spy)
    o,a,m=pipeline(tmp_path,case['raw'],case.get('gold'))
    checks=check_case(case,o,a,m,opened)
    assert all(checks.values()),checks
@pytest.mark.parametrize('attribute,value',ATTRS)
@pytest.mark.parametrize('selfclosing',[False,True])
def test_tag_boundary_matches_pinned_tokenizer(attribute,value,selfclosing):
    opening='<p '+attribute+(' />' if selfclosing else '>')
    raw=opening+" Don't <!-- c --> <img src=\"p.png\">"
    class Oracle(HTMLParser):
        def handle_starttag(self,tag,attrs):self.events.append((tag,attrs,self.get_starttag_text()))
        handle_startendtag=handle_starttag
    oracle=Oracle();oracle.events=[];oracle.feed(raw);oracle.close()
    assert oracle.events[0]==('p',[('title',value)],opening)
    assert tag_end(raw,0,len(raw))==len(opening)
@pytest.mark.parametrize('raw',['<p title="unfinished','<p title=bare','<p title=it\'s'])
def test_real_incomplete_tag_boundary(raw):assert tag_end(raw,0,len(raw)) is None

def test_guard_visits_every_opening_even_after_unsafe_and_structure(tmp_path,monkeypatch):
    from safety import PassiveHTML
    visited=[];original=PassiveHTML.handle_starttag
    def spy(self,tag,attributes):visited.append(tag);return original(self,tag,attributes)
    monkeypatch.setattr(PassiveHTML,'handle_starttag',spy)
    raw='<span class="a" class="b">x</span><script>x</script><img src="../outside.png"><img src="p.png">'
    o,a,m=pipeline(tmp_path,raw)
    assert visited==['span','script','img','img']*2
    assert a['reason']=='UNSAFE_REFERENCE' and 'script' in a['detail']
    assert m['automatic_gates']=='FAIL' and m['score_exit']==3
    assert not a['assets'] and evaluate_exit(2,m['score_exit'])==2

@pytest.mark.parametrize('bad',[False,True])
def test_projection_node_limit_cannot_stop_guard(tmp_path,bad):
    raw='<br>'*20001+('<img src="../outside.png">' if bad else '')
    o,a,m=pipeline(tmp_path,raw)
    assert a['reason']==('UNSAFE_REFERENCE' if bad else 'UNSUPPORTED_RESOURCE_SYNTAX')
    if not bad:assert any(u['reason']=='HTML nodes>20000' for n in o['nodes'] for u in n['unknown'])
    assert not a['assets'] and m['score_exit']==3

@pytest.mark.parametrize('host',HOST_MD)
def test_positive_mapping_after_stripped_continuation(tmp_path,host):
    marker='soft - ' if host=='paragraph' else '- '
    label='Known first '+marker+'<!-- words.'
    fragment='È See [Known `first\n'+marker+'<!--` words.]('+URL+') puis ü [Later]('+URL+'/later) and `ö`.'
    raw=transport(fragment,host)+TAIL
    text='È See '+label+' puis ü Later and ö.'
    g=gold(text,'note' if host.startswith('note') else 'paragraph')
    g['blocks'][0]['references']=[{'kind':'link','text':label,'target':URL},{'kind':'link','text':'Later','target':URL+'/later'}]
    o,a,m=pipeline(tmp_path,raw,g)
    assert a['status']=='COMPLETE' and m['automatic_gates']=='PASS'
    assert m['quality']=='INDETERMINATE' and m['score_exit']==5
    assert m['totals']['text_errors']==m['totals']['word_errors']==0
    assert o['nodes'][0]['text']==text and o['nodes'][-1]['text']=='After words.'
    refs=o['nodes'][0]['references'];code=[s for s in o['nodes'][0]['segments'] if s['type']=='code']
    assert [r['target'] for r in refs]==[URL,URL+'/later'] and len(code)==2
    for leaf,spelling in zip([*refs,*code],['[Known','[Later','`first','`ö`']):
        at=raw.index(spelling)
        assert leaf['source_position']['char']==at
        assert leaf['source_position']['byte']==len(raw[:at].encode())
