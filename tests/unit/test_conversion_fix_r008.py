"""F35–F38: finite domains, strict cursor progress and source-proven literals."""
from copy import deepcopy
from pathlib import Path
import shutil,sys
import pytest
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'scripts/benchmarks'))
from benchmark import observe_bundle
from observations import markdown_lexemes,inline,observe
import audit as auditor
from metrics import score,known_text,evaluate_exit
"""Fixed synthetic cases/expectations independent of reader; no imports product."""
F35_RAW='```{=latex}\n\\begin{center}\n\n\\end{center}\n```\n\n# Visible heading\n\nAfter words.\n'
GOLD={'document_id':'r008-synthetic','blocks':[{'block_id':'p','type':'paragraph','text':'Known words.','page':1},{'block_id':'q','type':'paragraph','text':'Hidden words.','page':2}]}
CASES=[]
for prefix in ['See `x','See $x','See $$','See <alpha']:
 for soft in [False,True]:
  for kind,opening,closing in [('comment','<!--','-->'),('cdata','<![CDATA[',']]>')]:
   for closed in [False,True]:
    for uri in ['p.png','../outside.png']:
     token=opening+'\n\nHidden words.\n\n# hidden heading\n\n![hidden]('+uri+')\n$x_i$\n'+(closing if closed else '')
     first=prefix+('\n' if soft else ' ')+token
     raw='Known words.\n\n'+first+('\n\n# Visible heading\n\nAfter words.' if closed else '')
     CASES.append({'id':'F36-'+str(len(CASES)),'fix':'F36','raw':raw,'token':token,'kind':kind,'closed':closed,'expected_status':'INCOMPLETE','expected_reason':'UNSUPPORTED_RESOURCE_SYNTAX','expected_missing':'q'})
for image in [False,True]:
 for ticks in ['`','``']:
  for marker in ['<!--','<![CDATA[','<?x','<!x']:
   for host in ['paragraph','quote','list','note']:
    label='[Known '+ticks+'first\n'+marker+ticks+' words.]';link=('!' if image else '')+label+'('+('outer.png' if image else 'https://example.invalid')+')'
    value='See '+link
    if host=='quote':value='> '+value.replace('\n','\n> ')
    elif host=='list':value='- '+value.replace('\n','\n    ')
    elif host=='note':value='[^n]: '+value.replace('\n','\n    ')
    raw=value+'\n\n# Visible heading\n\nAfter words.'
    CASES.append({'id':'F37-'+str(len(CASES)),'fix':'F37','raw':raw,'image':image,'marker':marker,'host':host,'first_text':'See Known first '+marker+' words.','expected_status':'COMPLETE','expected_reason':None,'expected_types':['note' if host=='note' else 'paragraph','heading','paragraph']})


def pipeline(tmp_path,raw,gold=GOLD):
    (tmp_path/'document.md').write_text(raw)
    for asset in ('p.png','outer.png'):
        shutil.copyfile(ROOT/'tests/fixtures/conversion-corpus/assets/figure-it.png',tmp_path/asset)
    o=observe_bundle(tmp_path,tmp_path/'observations.json');a=auditor.audit(tmp_path)
    return o,a,score(deepcopy(gold),o,a,verification=None)

def tokens(o):return [u for n in o['nodes'] for u in n['unknown'] if u.get('profile')=='html-tokens-v1']

@pytest.mark.parametrize('case',CASES,ids=[v['id'] for v in CASES])
def test_real_pipeline_domains(tmp_path,monkeypatch,case):
    opened=[];old=auditor.regular
    def spy(root,path,*a,**kw):opened.append(path);return old(root,path,*a,**kw)
    monkeypatch.setattr(auditor,'regular',spy)
    raw=case['raw'];o,a,m=pipeline(tmp_path,raw)
    assert a['status']==case['expected_status'] and a['reason']==case['expected_reason']
    assert 'p.png' not in opened and '../outside.png' not in opened
    for n in o['nodes']:
        lo=n['offset']['char_start'];hi=n['offset']['char_end']
        assert n['raw']==raw[lo:hi] and n['offset']['byte_start']==len(raw[:lo].encode())
    if case['fix']=='F36':
        values=[u for u in tokens(o) if u['raw']==case['token']];assert len(values)==1
        at=raw.index(case['token']);u=values[0]
        assert u['end']-u['offset']==len(case['token'])
        assert u['source_position']['char']==at and u['source_position']['byte']==len(raw[:at].encode())
        assert not a['assets'] and not a['references'] and not any(n['formulas'] for n in o['nodes'])
        assert not any('Hidden' in known_text(n) or n['text']=='hidden heading' for n in o['nodes'])
        assert 'q' in m['blocks']['missing'] and next(p for p in m['pages'] if p['page']==2)['missing_page']
        assert m['score_exit']==3 and evaluate_exit(2,m['score_exit'])==2
        if case['closed']:assert o['nodes'][-1]['text']=='After words.' and any(n['type']=='heading' and n['text']=='Visible heading' for n in o['nodes'])
    else:
        assert [n['type'] for n in o['nodes']]==case['expected_types']
        assert o['nodes'][0]['text']==case['first_text'] and o['nodes'][-1]['text']=='After words.'
        assert not tokens(o) and not markdown_lexemes(raw)
        assert [r['target'] for n in o['nodes'] for r in n['references']]==['outer.png' if case['image'] else 'https://example.invalid']

@pytest.mark.parametrize('between',['\n\n','\n# raw heading\n','\n- raw list\n','\n> raw quote\n','\n[^raw]: note\n','\n::: raw\n'])
@pytest.mark.parametrize('tail',[r'\end{center}','`unfinished','$x','$$'])
@pytest.mark.parametrize('closed',[True,False])
def test_raw_latex_finite_body_advances(tmp_path,between,tail,closed):
    raw='```{=latex}\n'+r'\begin{center}'+between+tail+'\n'+('```\n\n# Visible heading\n\nAfter words.' if closed else '')
    # Termination is also checked in an independently timed R invocation.
    spans=markdown_lexemes(raw);assert not spans
    o,a,m=pipeline(tmp_path,raw)
    assert a['status']==('COMPLETE' if closed else 'INCOMPLETE') and any(n['unknown'] for n in o['nodes'])
    assert any(r'\begin{center}' in u['raw'] for n in o['nodes'] for u in n['unknown'])
    if closed:
        assert o['nodes'][-2]['type']=='heading' and o['nodes'][-2]['text']=='Visible heading'
        assert o['nodes'][-1]['text']=='After words.'

@pytest.mark.parametrize('prefix',['Before `x','Before $x','Before $$','Before <alpha'])
@pytest.mark.parametrize('kind',['comment','cdata'])
@pytest.mark.parametrize('host',['heading','quote','list','note','cell','caption','label'])
def test_incomplete_atoms_do_not_hide_token_in_transports(tmp_path,monkeypatch,prefix,kind,host):
    token=('<!--' if kind=='comment' else '<![CDATA[')+'\n\n![hidden](../outside.png)\nHidden words.\n'+('-->' if kind=='comment' else ']]>')
    payload='È '+prefix+' '+token+' tail.'
    raw={'heading':'# '+payload,'quote':'> '+payload.replace('\n','\n> '),'list':'- '+payload.replace('\n','\n    '),'note':'[^n]: '+payload.replace('\n','\n    '),'cell':'| x |\n| --- |\n| '+payload+' |','caption':'| x |\n| --- |\n| 1 |\n\nTable: '+payload,'label':'['+payload+'](https://example.invalid)'}[host]
    opened=[];old=auditor.regular
    def spy(root,path,*a,**kw):opened.append(path);return old(root,path,*a,**kw)
    monkeypatch.setattr(auditor,'regular',spy);o,a,m=pipeline(tmp_path,raw)
    assert a['status']=='INCOMPLETE' and a['reason']=='UNSUPPORTED_RESOURCE_SYNTAX'
    assert '../outside.png' not in opened and not a['assets']
    assert not any('Hidden' in known_text(n) for n in o['nodes'])
    assert 'q' in m['blocks']['missing'] and m['score_exit']==3
    assert len(tokens(o))==1

@pytest.mark.parametrize('raw',['See `<!--` here.','See $x<y$ here.','See <span title="<!--">words.</span>','It costs $5 today.','a < b and p < 0.05','See [Known ``first\n<!--`` words.](https://example.invalid)'])
def test_closed_literals_and_currency_are_positive(tmp_path,raw):
    o,a,m=pipeline(tmp_path,raw);assert a['status']=='COMPLETE' and not tokens(o) and not markdown_lexemes(raw)

@pytest.mark.parametrize('raw',['Before `x <!--È\n\n![x](p.png)','Before $x <![CDATA[È\n\n![x](p.png)','[Known `first\n<!--È\n\n![x](p.png)'])
def test_EOF_token_preserves_unicode_and_never_opens_body(tmp_path,raw):
    o,a,m=pipeline(tmp_path,raw);assert a['status']=='INCOMPLETE' and not a['assets']
    values=tokens(o);assert len(values)==1;u=values[0];at=raw.index('<')
    assert u['raw']==raw[at:] and u['source_position']['char']==at and u['source_position']['byte']==len(raw[:at].encode())

@pytest.mark.parametrize('uri',['outer.png','absent.png','../outside.png'])
def test_active_resource_outside_literal_keeps_boundary(tmp_path,uri):
    raw='See [Known `first\n<!--` words.](https://example.invalid)\n\n![visible]('+uri+')'
    o,a,m=pipeline(tmp_path,raw)
    assert a['status']==('COMPLETE' if uri=='outer.png' else 'INCOMPLETE')
    assert a['reason']==(None if uri=='outer.png' else 'MISSING_RESOURCE' if uri=='absent.png' else 'UNSAFE_REFERENCE')

@pytest.mark.parametrize('closed',[True,False])
@pytest.mark.parametrize('kind',['comment','cdata'])
def test_raw_latex_EOF_or_closed_preserves_whole_inert_token(tmp_path,monkeypatch,kind,closed):
    token=('<!--' if kind=='comment' else '<![CDATA[')+'\n\n![hidden](../outside.png)\nHidden words.\n```\n'+('-->' if kind=='comment' else ']]>')
    raw='```{=latex}\n'+r'\foo{x}'+'\n\n'+token+'\n'+('```\n\n# Visible heading\n\nAfter words.' if closed else '')
    opened=[];old=auditor.regular
    def spy(root,path,*a,**kw):opened.append(path);return old(root,path,*a,**kw)
    monkeypatch.setattr(auditor,'regular',spy);o,a,m=pipeline(tmp_path,raw)
    assert a['status']=='INCOMPLETE' and a['reason']=='UNSUPPORTED_RESOURCE_SYNTAX'
    assert '../outside.png' not in opened and not a['assets'] and not a['references']
    u=tokens(o);assert len(u)==1 and u[0]['raw']==token
    assert not any('Hidden' in known_text(n) for n in o['nodes'])
    assert 'q' in m['blocks']['missing'] and m['score_exit']==3
    if closed:assert o['nodes'][-1]['text']=='After words.'

@pytest.mark.parametrize('host',['bare','wrapper','cell','caption','raw-html'])
@pytest.mark.parametrize('kind',['comment','cdata'])
def test_HTML_producer_incomplete_prefix_cannot_consume_token(tmp_path,monkeypatch,host,kind):
    token=('<!--' if kind=='comment' else '<![CDATA[')+'\n\n![hidden](../outside.png)\nHidden words.\n'+('-->' if kind=='comment' else ']]>')
    payload='<alpha '+token
    raw={'bare':payload,'wrapper':'<div>See '+payload+'</div>','cell':'<table><tr><td>'+payload+'</td></tr></table>','caption':'<table><caption>'+payload+'</caption><tr><td>x</td></tr></table>','raw-html':'```{=html}\n'+payload+'\n```'}[host]+'\n\nAfter words.'
    opened=[];old=auditor.regular
    def spy(root,path,*a,**kw):opened.append(path);return old(root,path,*a,**kw)
    monkeypatch.setattr(auditor,'regular',spy);o,a,m=pipeline(tmp_path,raw)
    assert a['status']=='INCOMPLETE' and a['reason']=='UNSUPPORTED_RESOURCE_SYNTAX'
    assert '../outside.png' not in opened and not a['assets'] and not a['references']
    assert len([u for u in tokens(o) if u['raw']==token])==1
    assert not any('Hidden' in known_text(n) for n in o['nodes']) and 'q' in m['blocks']['missing']
    assert m['score_exit']==3 and o['nodes'][-1]['text']=='After words.'

@pytest.mark.parametrize('tail',[r'\foo{x}','`unfinished','$x'])
def test_raw_fence_looking_suffix_is_not_a_physical_closure(tmp_path,tail):
    token='<!--\n```\n-->'
    raw='```{=latex}\n'+token+'```\n'+tail+'\n```\n\n# Visible heading\n\nAfter words.'
    o,a,m=pipeline(tmp_path,raw)
    assert a['status']=='INCOMPLETE' and len(tokens(o))==1 and tokens(o)[0]['raw']==token
    assert o['nodes'][-2]['type']=='heading' and o['nodes'][-1]['text']=='After words.'
    assert not a['assets'] and not a['references']
