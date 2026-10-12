"""F22–F26: parser diagnostics, origin mapping and finite HTML hosts."""
from copy import deepcopy
from pathlib import Path
import sys
import pytest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/benchmarks'))
from observations import observe
from metrics import score
from test_conversion_fix_r004 import pipeline,TABLE,G,distance
from test_conversion_fix_r003 import M,U,AST

P={'document_id':'synthetic','blocks':[{'block_id':'p','type':'paragraph','text':'Known words.'}]}
H={'document_id':'synthetic','blocks':[{'block_id':'h','type':'heading','text':'Titolo'}]}
F={'document_id':'synthetic','blocks':[{'block_id':'f','type':'figure','text':'','caption':'caption','asset':'figure-it.png'}]}
FIG='<figure><img src="p.png"/><figcaption>caption</figcaption></figure>'

@pytest.mark.parametrize('tag',['div','section','body','article'])
@pytest.mark.parametrize('body,gold',[('<p>Known words.</p>',P),(TABLE,G),(FIG,F),('',{'document_id':'empty','blocks':[]})])
@pytest.mark.parametrize('fenced',[False,True])
def test_F22_wrapper_diagnostic_survives_projection(tmp_path,tag,body,gold,fenced):
    raw='<'+tag+'>'+body
    if fenced:raw='```{=html}\n'+raw+'\n```\n'
    o,a,s=pipeline(tmp_path,raw,gold,True)
    assert a['status']=='INCOMPLETE' and s['automatic_gates']!='PASS'
    errors=[u for n in o['nodes'] for u in n['unknown'] if u['reason']=='unclosed HTML container']
    assert len(errors)==1 and errors[0]['location']['tag']==tag
    assert errors[0]['raw'].startswith('<'+tag+'>')
    assert errors[0]['source_position']['char']==raw.index('<'+tag+'>')
    if body==FIG:assert len(a['assets'])==1
    if body==TABLE:assert next(n for n in o['nodes'] if n['type']=='table')['cells']==[['h'],['v']]

@pytest.mark.parametrize('tag',['div','section','body','article'])
@pytest.mark.parametrize('body,gold',[('<p>Known words.</p>',P),(TABLE,G),(FIG,F)])
def test_F22_balanced_wrapper_positive(tmp_path,tag,body,gold):
    o,a,s=pipeline(tmp_path,'<'+tag+'>'+body+'</'+tag+'>',gold,True)
    assert a['status']=='COMPLETE' and s['automatic_gates']=='PASS'
    assert not any(n['unknown'] for n in o['nodes'])

@pytest.mark.parametrize('raw,count',[('<section><div><p>Known words.</p>',2),('<div><section><p>Known words.</p></div>',3),('<div></section>',2),('<div><div></div>',1)])
def test_F22_nested_incoherent_diagnoses_once(tmp_path,raw,count):
    o,a,s=pipeline(tmp_path,raw,P)
    errors=[u for n in o['nodes'] for u in n['unknown'] if u.get('resource_syntax')]
    assert len(errors)==count and a['status']=='INCOMPLETE'
    assert len({(u['offset'],u['reason']) for u in errors})==count
    assert [u['source_position']['char'] for u in errors]==sorted(u['source_position']['char'] for u in errors)
    assert s['automatic_gates']!='PASS'

@pytest.mark.parametrize('level',range(1,7))
@pytest.mark.parametrize('host',['{}','<section><div>{}</div></section>','```{=html}\n{}\n```\n'])
def test_F23_HTML_heading_positive_and_difference(tmp_path,level,host):
    raw=host.replace('{}','<h'+str(level)+'>Titolo</h'+str(level)+'>')
    o,a,s=pipeline(tmp_path,raw,H)
    assert o['nodes'][0]['type']=='heading' and a['status']=='COMPLETE'
    assert s['automatic_gates']=='PASS' and s['WER']==0
    (tmp_path/'document.md').unlink();(tmp_path/'observations.json').unlink()
    o,a,s=pipeline(tmp_path,raw.replace('Titolo','Titolo Additional invented content.'),H)
    assert s['score_exit']==3 and s['WER']>0

@pytest.mark.parametrize('host',['<h1>{}</h1>','<div><h6>{}</h6></div>','```{=html}\n<h3>{}</h3>\n```\n','# {}','- {}','> {}','[^n]: {}'])
def test_F23_origin_coordinates_unicode_math_resources(tmp_path,host):
    payload='È <span><a href="https://example.invalid">ref</a></span> '+M+'<img src="p.png"/>'
    raw=host.replace('{}',payload);o,a,s=pipeline(tmp_path,raw,H,True)
    assert a['status']=='COMPLETE'
    n=o['nodes'][0]
    assert len(n['references'])==2 and len(n['formulas'])==1
    for leaf,prefix in [(n['references'][0],'<a '),(n['references'][1],'<img '),(n['formulas'][0],'<math>')]:
        at=raw.index(prefix);assert leaf['source_position']['char']==at
        assert leaf['source_position']['byte']==len(raw[:at].encode())

@pytest.mark.parametrize('host',['h1','h6','code'])
@pytest.mark.parametrize('block',['<table><tr><td>99</td></tr></table>','<figure><img src="p.png"/><figcaption>99</figcaption></figure>'])
@pytest.mark.parametrize('wrapper',['{}','<span><em>{}</em></span>'])
def test_F24_nested_blocks_known_text_cannot_PASS(tmp_path,host,block,wrapper):
    raw='<'+host+'>Titolo '+wrapper.format(block)+'</'+host+'>'
    g=deepcopy(H if host.startswith('h') else P);g['blocks'][0]['text']='Titolo 99'
    o,a,s=pipeline(tmp_path,raw,g,True)
    assert a['status']=='COMPLETE' and s['automatic_gates']!='PASS'
    assert o['nodes'][0]['text']=='Titolo 99' and s['WER'] is None
    errors=[u for u in o['nodes'][0]['unknown'] if u.get('profile')=='html-inline-v1']
    assert errors and any(block in u['raw'] for u in errors)
    if '<img' in block:assert len(a['assets'])==1

@pytest.mark.parametrize('host',['h2','code'])
def test_F24_nested_resources_math_unknown_preserved(tmp_path,host):
    raw='<'+host+'>È <span><table><tr><td>'+M+U+'<img src="absent.png"/></td></tr></table></span></'+host+'>'
    o,a,s=pipeline(tmp_path,raw,H)
    n=o['nodes'][0];assert a['status']=='INCOMPLETE' and a['reason']=='MISSING_RESOURCE'
    assert len(n['references'])==len(n['formulas'])==1 and n['formulas'][0]['math']['structure']==AST
    assert any(u['raw']==U and u['scope']=='math' for u in n['unknown'])
    assert n['formulas'][0]['source_position']['char']==raw.index('<math>')

@pytest.mark.parametrize('host',['h1','code'])
def test_F24_phrasing_positive(tmp_path,host):
    raw='<'+host+'>Known <span><em>words.</em></span></'+host+'>'
    g=deepcopy(H if host=='h1' else P);g['blocks'][0]['text']='Known words.'
    o,a,s=pipeline(tmp_path,raw,g)
    assert a['status']=='COMPLETE' and s['automatic_gates']=='PASS'

@pytest.mark.parametrize('token',['<!-- secret <img src="../outside.png"/> 99 -->','<!bad>','<?opaque?>','<!DOCTYPE html>'])
@pytest.mark.parametrize('context',['wrapper','paragraph','cell','caption','residual','inline'])
@pytest.mark.parametrize('position',['start','middle','end'])
def test_F25_tokens_localized_in_every_projection(tmp_path,token,context,position,monkeypatch):
    text={'start':token+'Known words.','middle':'Known '+token+'words.','end':'Known words.'+token}[position]
    raw={'wrapper':'<div>'+token+'<p>Known words.</p></div>' if position=='start' else '<div><p>Known words.</p>'+token+'</div>',
         'paragraph':'<p>'+text+'</p>','cell':TABLE.replace('<td>v</td>','<td>'+text+'</td>'),
         'caption':TABLE.replace('<table>','<table><caption>'+text+'</caption>'),
         'residual':TABLE.replace('</table>','<span>'+text+'</span></table>'),'inline':text}[context]
    import audit as module
    opened=[];original=module.regular
    def spy(root,path):opened.append(path);return original(root,path)
    monkeypatch.setattr(module,'regular',spy)
    o,a,s=pipeline(tmp_path,raw,P if context in ('wrapper','paragraph','inline') else G)
    tokens=[u for n in o['nodes'] for u in n['unknown'] if u.get('profile')=='html-tokens-v1']
    assert len(tokens)==1 and tokens[0]['raw']==token
    assert tokens[0]['source_position']['char']==raw.index(token)
    assert a['status']=='INCOMPLETE' and s['automatic_gates']!='PASS'
    assert not a['assets'] and not a['references'] and set(opened)=={'document.md'}
    assert all('secret' not in n.get('text','') for n in o['nodes'])

def test_F25_multiple_tokens_and_visible_unicode(tmp_path):
    raw='<div><p>È <!--a--> &amp; <!bad>noto<!--b--></p></div>'
    o,a,s=pipeline(tmp_path,raw,P);n=o['nodes'][0]
    assert n['text']=='È  & noto' and len(n['unknown'])==3
    assert [u['raw'] for u in n['unknown']]==['<!--a-->','<!bad>','<!--b-->']
    for u in n['unknown']:
        at=raw.index(u['raw']);assert u['source_position']=={'char':at,'byte':len(raw[:at].encode()),'basis':'raw-block-relative-inline-v1'}
    assert a['status']=='INCOMPLETE'

@pytest.mark.parametrize('token',['<!-->','<!--->','<!-- body --!>','<!-- body > literal -->'])
@pytest.mark.parametrize('context',['paragraph','cell','inline'])
def test_F25_tokenizer_comment_closures_keep_exact_interval(tmp_path,token,context):
    text='Known '+token+'words.'
    raw='<p>'+text+'</p>' if context=='paragraph' else TABLE.replace('<td>v</td>','<td>'+text+'</td>') if context=='cell' else text
    o,a,s=pipeline(tmp_path,raw,P if context!='cell' else G)
    tokens=[u for n in o['nodes'] for u in n['unknown'] if u.get('token_type')=='comment']
    assert len(tokens)==1 and tokens[0]['raw']==token
    at=raw.index(token);assert tokens[0]['source_position']['char']==at
    assert tokens[0]['end']==at+len(token)
    assert a['status']=='INCOMPLETE' and s['automatic_gates']!='PASS'
    value=o['nodes'][0]['cell_contents'][1][0]['text'] if context=='cell' else o['nodes'][0]['text']
    assert value=='Known words.'

@pytest.mark.parametrize('raw',['`<h1><!-- <img src="../outside.png"/> --></h1>`','```html\n<h1><!-- <img src="../outside.png"/> --></h1>\n```\n'])
def test_F24_F25_Markdown_literal_code_inert(tmp_path,raw):
    o,a,s=pipeline(tmp_path,raw,P)
    assert a['status']=='COMPLETE' and not a['assets'] and not a['references']
    assert not any(n.get('unknown') for n in o['nodes'])

def test_F26_global_distance_can_exceed_structural_measure():
    p1='Alpha beta gamma delta epsilon.';p2='Zeta eta theta iota kappa.'
    g={'document_id':'independent','blocks':[{'block_id':'a','type':'paragraph','text':p1},{'block_id':'b','type':'paragraph','text':p2}]}
    m=score(g,observe((p2+'\n\n'+p1).encode()),{'status':'COMPLETE'})
    assert m['CER']==0 and m['totals']['text_errors']==0
    assert m['text_lower_bound']['char_errors']==distance(p1+'\n'+p2,p2+'\n'+p1)>0
    assert m['order']['inversions']>0 and m['score_exit']==3
    omitted=score(g,observe(p1.encode()),{'status':'COMPLETE'})
    assert omitted['totals']['text_errors']==len(p2)
    assert omitted['text_lower_bound']['char_errors']==len(p2)+1
