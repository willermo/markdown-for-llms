"""F13–F16: independent typed leaves, complete pipeline, no engine/browser."""
from copy import deepcopy
from pathlib import Path
import shutil
import sys
import pytest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/benchmarks'))
from observations import observe
from audit import audit
from metrics import score,evaluate_exit
from benchmark import observe_bundle,score_bundle
from data import load,descriptor
from test_conversion_benchmark import FP

C=ROOT/'tests/fixtures/conversion-corpus'
M='<math><semantics><mi>x</mi><annotation encoding="application/x-tex">x_i</annotation></semantics></math>'
U='<math><mi>x</mi></math>'
AST=FP[0][3]

def wrap(value,container):
    return {'plain':value,'span':'<span>'+value+'</span>','nested':'<span><span>'+value+'</span></span>','anchor':'<a href="https://example.invalid">'+value+'</a>','mixed':'<span>prima '+value+' dopo <code>$literal$</code></span>'}[container]

def document(value,context):
    return {'cell':'|h|\n|---|\n|'+value+'|\n','caption':'|h|\n|---|\n|v|\n\n: '+value+'\n','html-cell':'<table><tr><th>h</th></tr><tr><td>'+value+'</td></tr></table>','html-caption':'<table><caption>'+value+'</caption><tr><th>h</th></tr><tr><td>v</td></tr></table>','paragraph':'prosa '+value+' fine.'}[context]

@pytest.mark.parametrize('context',['cell','caption','html-cell','html-caption','paragraph'])
@pytest.mark.parametrize('container',['plain','span','nested','anchor','mixed'])
@pytest.mark.parametrize('payload',[M,U])
def test_F13_typed_leaf_transport(tmp_path,context,container,payload):
    raw=document(wrap(payload,container),context);(tmp_path/'document.md').write_text(raw)
    o=observe_bundle(tmp_path,tmp_path/'observations.json');a=audit(tmp_path)
    n=o['nodes'][0];field='formulas' if payload==M else 'unknown'
    assert len(n[field])==1,(raw,n)
    leaf=n[field][0];assert leaf['source_position']['char']==raw.index('<math>')
    assert leaf['source_position']['byte']==len(raw[:raw.index('<math>')].encode())
    if context!='paragraph':assert leaf['location']['kind']==('cell' if 'cell' in context else 'caption')
    if payload==M:assert leaf['raw']=='x_i' and leaf['math']['structure']==AST
    else:assert leaf['raw']==U and leaf['scope']=='math'
    if container=='mixed':
        contents=n['caption_content'] if 'caption' in context else n['cell_contents'][1][0] if 'cell' in context else n
        assert any(s['type']=='code' and s['text']=='$literal$' for s in contents['segments'])
    assert a['status']=='COMPLETE'
    g={'document_id':'own','blocks':[{'block_id':'b','type':'paragraph','text':'prosa  fine.'}]}
    if context!='paragraph':g['blocks']=[{'block_id':'b','type':'table','text':'','cells':[['h'],['v' if 'caption' in context else '']],'header_rows':1,'caption':'' if 'caption' in context else None}]
    r=score(g,o,a,verification={'status':'VERIFIED'})
    assert r['automatic_gates']!='PASS',(raw,r)

def test_F13_supported_equivalent_positive_and_duplicates(tmp_path):
    raw='|h|\n|---|\n|<span>'+M+'</span>|\n'
    g={'document_id':'own','blocks':[{'block_id':'b','type':'table','text':'','cells':[['h'],['']],'header_rows':1,'cell_contents':[[{}],[{'formulas':[{'structure_expected':AST}]}]],'formulas':[{'formula_id':'f','reference_latex':'x_{i}','structure_expected':AST}]}]}
    r=score(g,observe(raw.encode()),{'status':'COMPLETE'},verification={'status':'VERIFIED'})
    assert r['automatic_gates']=='PASS' and r['score_exit']==5 and r['WER'] is None
    assert any(u['metric']=='formula_output_typography_and_semantics' for u in r['unavailable'])
    added=raw.replace(M,M+M);assert score(g,observe(added.encode()),{'status':'COMPLETE'})['score_exit']==3
    o=observe(('prosa <span>'+M+'<span>'+M+'</span></span> fine').encode())
    assert len(o['nodes'][0]['formulas'])==2
    assert [f['source_position']['char'] for f in o['nodes'][0]['formulas']]==[12,12+len(M)+6]

@pytest.mark.parametrize('context',['cell','caption','html-cell','html-caption','paragraph'])
@pytest.mark.parametrize('container',['span','nested','anchor','mixed'])
def test_F13_positive_gold_independent_of_transport(context,container):
    visible='prima  dopo $literal$' if container=='mixed' else ''
    refs=[{'kind':'link','text':'','target':'https://example.invalid'}] if container=='anchor' else []
    formula={'formula_id':'f','reference_latex':'x_{i}','structure_expected':AST}
    leaf={'formulas':[{'structure_expected':AST}],'references':refs}
    b={'block_id':'b','type':'paragraph','text':'prosa '+visible+' fine.','formulas':[formula],'references':refs}
    if context!='paragraph':
        b.update(type='table',text='',header_rows=1,cells=[['h'],['v' if 'caption' in context else visible]],cell_contents=[[{}],[{} if 'caption' in context else leaf]],caption=visible if 'caption' in context else None,caption_content=leaf if 'caption' in context else {})
    r=score({'document_id':'own','blocks':[b]},observe(document(wrap(M,container),context).encode()),{'status':'COMPLETE'},verification={'status':'VERIFIED'})
    assert r['automatic_gates']=='PASS',r
    assert r['score_exit']==5 and r['totals']['formulas_exact']==1

@pytest.mark.parametrize('context',['cell','caption','html-cell','html-caption','paragraph'])
@pytest.mark.parametrize('container',['plain','span','nested','anchor'])
def test_F14_image_reaches_audit(tmp_path,context,container):
    raw=document(wrap('<IMG SRC="p.png" ALT="alt"/>',container),context)
    (tmp_path/'document.md').write_text(raw);a=audit(tmp_path)
    assert a['status']=='INCOMPLETE' and a['reason']=='MISSING_RESOURCE',a
    assert evaluate_exit(2,3)==2
    shutil.copyfile(C/'assets/figure-it.png',tmp_path/'p.png');a=audit(tmp_path)
    assert a['status']=='COMPLETE' and len(a['assets'])==1
    assert a['assets'][0]['sha256']==descriptor(C/'assets/figure-it.png')['sha256']

@pytest.mark.parametrize('tag',['figure','aside','ul','nav','main'])
def test_F14_fallback_keeps_resources(tmp_path,tag):
    raw='<'+tag+'><span><img src="p.png" alt="alt"/></span></'+tag+'>'
    (tmp_path/'document.md').write_text(raw);assert audit(tmp_path)['status']=='INCOMPLETE'
    o=observe(raw.encode());assert len(o['nodes'][0]['references'])==1
    shutil.copyfile(C/'assets/figure-it.png',tmp_path/'p.png');assert audit(tmp_path)['status']=='COMPLETE'
    assert o['unknown']>0

def test_F14_figure_positive_independent_caption_asset(tmp_path):
    raw='<figure><img src="p.png" alt="alt"/><figcaption>caption</figcaption></figure>'
    (tmp_path/'document.md').write_text(raw);shutil.copyfile(C/'assets/figure-it.png',tmp_path/'p.png')
    o=observe(raw.encode());a=audit(tmp_path)
    a['assets'][0].update(content_verified=True,expected_asset='figure-it.png')
    g={'document_id':'own','blocks':[{'block_id':'b','type':'figure','text':'','caption':'caption','asset':'figure-it.png'}]}
    assert score(g,o,a,verification={'status':'VERIFIED'})['automatic_gates']=='PASS'

@pytest.mark.parametrize('raw',['Vedi <img src="../outside.png"/> qui.','<span><img src="file:///outside.png"/></span>','<aside><img src="https://example.invalid/a.png"/></aside>'])
def test_F14_unsafe_never_opened(tmp_path,monkeypatch,raw):
    import audit as module
    real=module.regular;opened=[]
    def spy(root,path):
        opened.append(path);assert path=='document.md';return real(root,path)
    monkeypatch.setattr(module,'regular',spy);(tmp_path/'document.md').write_text(raw)
    a=audit(tmp_path);assert a['status']=='INCOMPLETE' and a['reason']=='UNSAFE_REFERENCE'
    assert opened==['document.md']

@pytest.mark.parametrize('raw',['Vedi <span><img src="p.png"/>','<p><span><img src="p.png"/></p>','<img src="p.png" srcset="../outside.png 1x"/>'])
def test_F14_malformed_or_unhandled_resources_noncomplete(tmp_path,raw):
    (tmp_path/'document.md').write_text(raw);shutil.copyfile(C/'assets/figure-it.png',tmp_path/'p.png')
    assert audit(tmp_path)['status']=='INCOMPLETE'

def measure(raw,g=None):
    return score(g or {'document_id':'own','blocks':[{'block_id':'b','type':'paragraph','text':'Known words.'}]},observe(raw.encode()),{'status':'COMPLETE'},verification={'status':'VERIFIED'})

@pytest.mark.parametrize('suffix',[r'\unknown{Additional invented content.}',r'Additional invented content. \foo{x}',r'*Additional invented content. \foo{x}*',r'[Additional invented content. \foo{x}]{.span}','- Additional invented content. \\foo{x}'])
def test_F15_unmatched_text_unknown_unavailable(suffix):
    r=measure('Known words.\n\n'+suffix)
    assert r['WER'] is None and r['CER'] is None
    a=r['text_metric_availability'];assert a['status']=='UNAVAILABLE' and a['rule']=='text-domain-unknown-v1'
    assert any(v.get('observation_id') for v in a['reasons'])
    assert r['totals']['word_ref']==2
    assert r['totals']['word_errors']==(0 if suffix.startswith(r'\unknown') else 3)

def test_F15_known_positive_addition_and_attributed_missing():
    assert measure('Known words.')['WER']==0
    assert measure('Known words.\n\nAdditional invented content.')['WER']==1.5
    g={'document_id':'own','blocks':[{'block_id':'b','type':'paragraph','text':'Specific approved content.'},{'block_id':'b2','type':'paragraph','text':'Missing approved content.','page':2}]}
    r=measure(r'\unknown{Specific approved content.}',g)
    assert r['blocks']['unresolved_attributed']==['b'] and r['blocks']['missing']==['b2']
    assert r['WER'] is None and r['totals']['word_errors']==3 and r['score_exit']==3
    assert r['pages'][0]['missing_page']

def test_F15_scope_math_code_caption():
    # Unknown math outside whole-block prose does not poison that domain.
    r=measure('Known words.\n\n<span>'+U+'</span>')
    assert r['WER']==0 and r['text_metric_availability']['status']=='AVAILABLE'
    g={'document_id':'own','blocks':[{'block_id':'b','type':'code','text':'\\foo{x}\n'}]}
    r=measure('```\n\\foo{x}\n```',g);assert r['WER']==0
    g={'document_id':'own','blocks':[{'block_id':'b','type':'table','text':'','header_rows':1,'cells':[['h'],['v']],'caption':'caption'}]}
    r=measure('|h|\n|---|\n|v|\n\n: caption \\foo{x}',g)
    assert r['text_metric_availability']['status']=='AVAILABLE' and r['automatic_gates']!='PASS'

def test_F16_decomposition_independent_structural_example():
    g={'document_id':'own','blocks':[{'block_id':'b','type':'heading','text':'Three word title'},{'block_id':'t','type':'table','text':'','header_rows':1,'cells':[['h'],['v']],'caption':'Two words'},{'block_id':'p','type':'paragraph','text':'Known words.'}]}
    r=measure('Three word title\n\n|h|\n|---|\n|v|\n\nTwo words\n\nKnown words!',g)
    d=r['WER_decomposition'];assert d['categories']=={'aligned_block_edits':1,'expected_omission':3,'unmatched_observed_insertion':5}
    assert d['known_word_edits']==9 and d['reference_words']==5
    assert any(e.get('role_candidates',[{}])[0].get('role')=='caption' for e in d['entries'] if e.get('role_candidates'))


def test_F14_depth_and_mismatched_close_are_explicit(tmp_path):
    for raw in ['<p><img src="p.png"/></span></p>', 'prosa '+'<span>'*17+'<img src="p.png"/>'+'</span>'*17]:
        (tmp_path/'document.md').write_text(raw)
        assert audit(tmp_path)['status']=='INCOMPLETE'


def test_F15_comparison_limit_unavailable():
    text='a'*2100
    g={'document_id':'own','blocks':[{'block_id':'b','type':'paragraph','text':text}]}
    r=measure(text,g)
    assert r['CER'] is None and r['WER'] is None
    assert r['text_metric_availability']['status']=='UNAVAILABLE'
    assert r['totals']['word_ref']==1 and r['totals']['word_errors']==0

@pytest.mark.parametrize('raw',[
    '# titolo <span>'+M+'</span>\n',
    '- prima\n    dopo <span>'+M+'</span>\n',
    '> prima\n> dopo <span>'+M+'</span>\n',
    '[^n]: prima\n    dopo <span>'+M+'</span>\n',
    '```{=html}\n<table><caption><span>'+M+'</span></caption><tr><td>v</td></tr></table>\n```\n',
    '```{=html}\n<p><span>'+M+'</span></p>\n```\n',
])
def test_F13_offsets_across_prefixes_and_raw_fences(raw):
    o=observe(raw.encode());f=o['nodes'][0]['formulas'][0]
    assert f['source_position']['char']==raw.index('<math>')
    assert f['source_position']['byte']==len(raw[:raw.index('<math>')].encode())

@pytest.mark.parametrize('raw',[
    '```{=other}\n<img src="p.png"/>\n```',
    '`<img src="p.png"/>`{=html}',
    '```{=html}\n'+'<span>'*17+'<img src="p.png"/>'+'</span>'*17+'\n```',
])
def test_F14_raw_fallback_cannot_hide_resource(tmp_path,raw):
    (tmp_path/'document.md').write_text(raw);shutil.copyfile(C/'assets/figure-it.png',tmp_path/'p.png')
    assert audit(tmp_path)['status']=='INCOMPLETE'

def test_F14_real_code_literal_not_a_resource(tmp_path):
    (tmp_path/'document.md').write_text('`<img src="../outside.png"/>`\n\n```\n<img src="../outside.png"/>\n```')
    assert audit(tmp_path)['status']=='COMPLETE'

@pytest.mark.parametrize('raw',[
    'Vedi <img src="missing.png" src="present.png"/> qui.',
    '<p><IMG SRC="missing.png" SRC="present.png"/></p>',
    '<table><caption><img src="missing.png" src="present.png"/></caption><tr><td>x</td></tr></table>',
])
def test_F14_duplicate_attributes_cannot_drop_first_resource(tmp_path,raw):
    (tmp_path/'document.md').write_text(raw);shutil.copyfile(C/'assets/figure-it.png',tmp_path/'present.png')
    a=audit(tmp_path);assert a['status']=='INCOMPLETE' and a['reason']=='UNSUPPORTED_RESOURCE_SYNTAX'
