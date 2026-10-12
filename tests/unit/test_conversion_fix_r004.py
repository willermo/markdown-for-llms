"""F17–F21: gold independent of HTML producer and completion-distance oracle."""
from copy import deepcopy
from pathlib import Path
import shutil
import sys
import pytest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/benchmarks'))
from audit import audit
from benchmark import observe_bundle,score_bundle
from data import descriptor
from metrics import score,evaluate_exit
from observations import observe
from test_conversion_fix_r003 import M,U,AST,document,measure

C=ROOT/'tests/fixtures/conversion-corpus'
TABLE='<table><tr><th>h</th></tr><tr><td>v</td></tr></table>'
G={'document_id':'independent','blocks':[{'block_id':'t','type':'table','text':'','header_rows':1,'cells':[['h'],['v']]}]}

def pipeline(tmp_path,raw,gold=G,asset=False):
    (tmp_path/'document.md').write_text(raw)
    if asset:shutil.copyfile(C/'assets/figure-it.png',tmp_path/'p.png')
    o=observe_bundle(tmp_path,tmp_path/'observations.json');a=audit(tmp_path)
    for v in a['assets']:v.update(content_verified=True,expected_asset='figure-it.png')
    s=score(deepcopy(gold),o,a,verification={'status':'VERIFIED'})
    return o,a,s

@pytest.mark.parametrize('context',['cell','caption','html-cell','html-caption','paragraph'])
@pytest.mark.parametrize('block',['<table><tr><td>99 inventato</td></tr></table>','<figure><img src="p.png"/><figcaption>99 inventato</figcaption></figure>'])
@pytest.mark.parametrize('wrapper',['{}','<span>{}</span>','<em><a href="https://example.invalid">{}</a></em>'])
def test_F17_inline_blocks_conserve_known_leaves(tmp_path,context,block,wrapper):
    raw=document(wrapper.format(block),context);o,a,s=pipeline(tmp_path,raw,asset=True)
    assert s['automatic_gates']!='PASS'
    n=o['nodes'][0];assert n['unknown']
    assert any(u.get('profile')=='html-inline-v1' and '99 inventato' in u['raw'] for u in n['unknown'])
    leaves=n['caption_content'] if 'caption' in context else n['cell_contents'][1][0] if 'cell' in context else n
    assert '99 inventato' in leaves['text']
    if context=='paragraph':assert s['WER'] is None and s['CER'] is None
    if '<img' in block:assert len(a['assets'])==1

@pytest.mark.parametrize('context',['cell','caption','html-cell','html-caption','paragraph'])
def test_F17_inline_block_math_and_missing_asset(tmp_path,context):
    raw=document('<span><table><tr><td>'+M+'<img src="absent.png"/>'+U+'</td></tr></table></span>',context)
    o,a,s=pipeline(tmp_path,raw)
    assert len(o['nodes'][0]['formulas'])==1 and o['nodes'][0]['formulas'][0]['math']['structure']==AST
    assert any(u.get('scope')=='math' and u['raw']==U for u in o['nodes'][0]['unknown'])
    assert len(o['nodes'][0]['references'])==1
    assert a['status']=='INCOMPLETE' and a['reason']=='MISSING_RESOURCE'
    assert evaluate_exit(2,s['score_exit'])==2
    f=o['nodes'][0]['formulas'][0];assert f['source_position']['char']==raw.index('<math>')

@pytest.mark.parametrize('where',['table','tr','wrapper','duplicate-caption'])
@pytest.mark.parametrize('payload',['<img src="p.png"/>',M,U,'Valore inventato 99','<a href="https://example.invalid">Reference</a>'])
def test_F18_all_residual_branches_consumed(tmp_path,where,payload):
    if where=='table':raw=TABLE.replace('</table>',payload+'</table>')
    elif where=='tr':raw=TABLE.replace('</tr>',payload+'</tr>',1)
    elif where=='wrapper':raw=TABLE.replace('</table>','<span><span>'+payload+'</span></span></table>')
    else:raw=TABLE.replace('<table>','<table><caption>'+payload+'</caption><caption>Expected caption</caption>')
    o,a,s=pipeline(tmp_path,raw)
    assert s['automatic_gates']!='PASS'
    n=o['nodes'][0];assert n['residual_contents'] and n['unknown']
    if payload.startswith('<img'):
        assert a['status']=='INCOMPLETE' and a['reason']=='MISSING_RESOURCE'
        assert len(n['references'])==1 and evaluate_exit(2,s['score_exit'])==2
        shutil.copyfile(C/'assets/figure-it.png',tmp_path/'p.png');present=audit(tmp_path)
        assert present['status']=='COMPLETE' and len(present['assets'])==1
    elif payload==M:assert len(n['formulas'])==1 and s['score_exit']==3
    elif payload==U:assert any(u.get('scope')=='math' and u['raw']==U for u in n['unknown'])
    else:assert s['score_exit']==3
    for field in ('formulas','references'):
        for leaf in n[field]:assert leaf['source_position']['char']==raw.index('<math>' if field=='formulas' else '<img' if payload.startswith('<img') else '<a ')

@pytest.mark.parametrize('extra',['Valore inventato 99','<p>Valore inventato 99</p>','<span>'+M+'</span>','<img src="absent.png"/>',U])
def test_F18_figure_residue_consumed(tmp_path,extra):
    raw='<figure><img src="p.png" alt="alt"/>'+extra+'<figcaption>caption</figcaption></figure>'
    gold={'document_id':'figure','blocks':[{'block_id':'f','type':'figure','text':'','caption':'caption','asset':'figure-it.png'}]}
    o,a,s=pipeline(tmp_path,raw,gold,True)
    assert s['automatic_gates']!='PASS'
    if 'absent.png' in extra:assert a['status']=='INCOMPLETE' and len(o['nodes'][0]['references'])==2
    if extra==M or M in extra:assert len(o['nodes'][0]['formulas'])==1 and s['score_exit']==3

@pytest.mark.parametrize('raw',[TABLE,'<table> \n<thead>\n<tr><th>h</th></tr>\n</thead> \n<tbody><tr><td>v</td></tr></tbody>\n</table>'])
def test_F18_positive_table_and_structural_whitespace(tmp_path,raw):
    o,a,s=pipeline(tmp_path,raw)
    assert a['status']=='COMPLETE' and s['automatic_gates']=='PASS'
    assert s['totals']['cells_exact']==2 and not o['nodes'][0]['unknown']

def test_F18_positive_figure_caption_inline_resources(tmp_path):
    raw='<figure> \n<img src="p.png" alt="alt"/> \n<figcaption><em>caption</em> <a href="https://example.invalid">ref</a>'+M+'</figcaption> \n</figure>'
    gold={'document_id':'figure','blocks':[{'block_id':'f','type':'figure','text':'','caption':'caption ref','asset':'figure-it.png','references':[{'kind':'link','text':'ref','target':'https://example.invalid'}],'caption_content':{'references':[{'kind':'link','text':'ref','target':'https://example.invalid'}],'formulas':[{'structure_expected':AST}]},'formulas':[{'formula_id':'f1','reference_latex':'x_i','structure_expected':AST}]}]}
    o,a,s=pipeline(tmp_path,raw,gold,True)
    assert a['status']=='COMPLETE' and s['automatic_gates']=='PASS' and s['score_exit']==5
    assert len(o['nodes'][0]['formulas'])==1 and len(o['nodes'][0]['references'])==2
    assert o['nodes'][0]['formulas'][0]['location']['kind']=='caption'

@pytest.mark.parametrize('where',['table','tr','wrapper','duplicate-caption','figure'])
def test_F18_unsafe_residual_rejected_before_open(tmp_path,monkeypatch,where):
    import audit as module
    payload='<img src="../outside.png"/>';raw=TABLE.replace('</table>',payload+'</table>')
    if where=='tr':raw=TABLE.replace('</tr>',payload+'</tr>',1)
    if where=='wrapper':raw=TABLE.replace('</table>','<span>'+payload+'</span></table>')
    if where=='duplicate-caption':raw=TABLE.replace('<table>','<table><caption>'+payload+'</caption><caption>x</caption>')
    if where=='figure':raw='<figure>'+payload+'<figcaption>x</figcaption></figure>'
    (tmp_path/'document.md').write_text(raw);opened=[];original=module.regular
    def spy(root,path):
        opened.append(path);assert path=='document.md';return original(root,path)
    monkeypatch.setattr(module,'regular',spy)
    a=audit(tmp_path);assert a['reason']=='UNSAFE_REFERENCE' and opened==['document.md']

@pytest.mark.parametrize('expected',[None,'','Expected caption'])
@pytest.mark.parametrize('actual',[None,'','Expected caption','Added text',U,M])
def test_F19_caption_presence_content_all_truth_values(tmp_path,expected,actual):
    g=deepcopy(G)
    if expected is not None:g['blocks'][0]['caption']=expected
    raw=TABLE if actual is None else TABLE.replace('<table>','<table><caption>'+actual+'</caption>')
    o,a,s=pipeline(tmp_path,raw,g)
    if actual==U:assert s['automatic_gates']!='PASS'
    elif actual==M:assert s['score_exit']==3 and len(o['nodes'][0]['formulas'])==1
    elif (expected or '')==(actual or ''):assert s['automatic_gates']=='PASS'
    else:assert s['score_exit']==3 and any(f['metric']=='table_caption' for f in s['failures'])
    assert s['totals']['word_ref']==0 and s['WER'] is None

# Independent edit oracle (full matrix, not scorer's rolling-row implementation).
def distance(a,b):
    d=[[0]*(len(b)+1) for _ in range(len(a)+1)]
    for i in range(len(a)+1):d[i][0]=i
    for j in range(len(b)+1):d[0][j]=j
    for i in range(1,len(a)+1):
        for j in range(1,len(b)+1):d[i][j]=min(d[i-1][j]+1,d[i][j-1]+1,d[i-1][j-1]+(a[i-1]!=b[j-1]))
    return d[-1][-1]

@pytest.mark.parametrize('raw',[r'Known \foo{x}words.',r'- Known \foo{x}words.','Known <span><aside></aside></span>words.',r'\foo{x}Known words.',r'Known words.\foo{x}',r'Known \foo{x} \bar{y}words.'])
@pytest.mark.parametrize('completion',['','extra ','many visible words '])
def test_F20_bound_for_arbitrary_zero_and_word_completions(raw,completion):
    s=measure(raw);bound=s['text_lower_bound']
    assert s['WER'] is None and s['CER'] is None
    assert bound['guaranteed'] and bound['status']=='TRIVIAL_ZERO'
    assert bound['word_errors']==0 and bound['char_errors']==0
    assert s['conditional_text_counts']['guaranteed_bound'] is False
    visible=raw.removeprefix('- ').replace(r'\foo{x}',completion).replace(r'\bar{y}',completion).replace('<span><aside></aside></span>',completion)
    assert bound['word_errors']<=distance('Known words.'.split(),visible.split())
    assert bound['char_errors']<=distance('Known words.',visible)
    assert s['totals']['word_ref']==2 and s['totals']['text_ref_codepoints']==12
    assert all(e['guaranteed_bound'] is False for e in s['WER_decomposition']['entries'])

def test_F20_L1_L1c_and_multiblock_missing_preserved():
    raw=(C/'controls/K01-full/document.md').read_text();gold=__import__('json').loads((C/'controls/K01-full/annotations.json').read_text())
    zero=raw.replace('intenzionale','intenzionali',1);unknown=zero.replace('cambio −7.','cambio −7. \\foo{x}',1)
    actual=score(gold,observe(unknown.encode()),{'status':'COMPLETE'})
    known=score(gold,observe(zero.encode()),{'status':'COMPLETE'})
    assert known['totals']['word_errors']==1 and known['totals']['word_ref']==30
    assert actual['text_lower_bound']['word_errors']<=1
    assert actual['text_lower_bound']['char_errors']<=known['totals']['text_errors']
    b02=next(b['text'] for b in gold['blocks'] if b['block_id']=='b02')
    completed=b02.replace('intenzionale','intenzionali',1)
    assert distance(b02.split(),completed.split())==1
    assert actual['text_lower_bound']['char_errors']<=distance(b02,completed)
    assert actual['score_exit']==3 and 'b02' in actual['blocks']['missing']
    k02=__import__('json').loads((C/'controls/K02-three-sections/annotations.json').read_text())
    text=(C/'controls/K02-three-sections/document.md').read_text()
    # An omitted page remains a known structural FAIL even with unrelated unknown.
    lost='\n\n'.join(text.split('\n\n')[:2])+r'\unknown{x}'
    r=score(k02,observe(lost.encode()),{'status':'COMPLETE'})
    assert r['score_exit']==3 and any(p['missing_page'] for p in r['pages']) and r['WER'] is None
    assert r['text_lower_bound']['word_errors']==0

@pytest.mark.parametrize('suffix',['', '\n\nAdditional invented content.',r'\n\nAdditional invented content. \foo{x}'])
def test_F20_known_counts_distinct_and_full_measure_unavailable(suffix):
    suffix=suffix.replace(r'\n','\n');s=measure('Known words.'+suffix)
    assert s['totals']['word_ref']==2
    assert s['conditional_text_counts']['word_errors']==(3 if suffix else 0)
    assert s['WER_decomposition']['categories']['unmatched_observed_insertion']==(3 if suffix else 0)
    if r'\foo' in suffix:assert s['WER'] is None and s['text_lower_bound']['word_errors']==0
    else:assert s['WER']==(1.5 if suffix else 0) and s['text_lower_bound']['word_errors']==(3 if suffix else 0)

@pytest.mark.parametrize('raw',['Se a < b allora c.','p < 0.05','a < b e c < d','0 < 1','`<img src="absent.png"/>`'])
def test_F21_literal_less_than_is_not_resource(tmp_path,raw):
    expected=raw.strip('`');g={'document_id':'literal','blocks':[{'block_id':'p','type':'paragraph','text':expected}]}
    o,a,s=pipeline(tmp_path,raw,g)
    assert a['status']=='COMPLETE' and a['assets']==[]
    assert s['automatic_gates']=='PASS' and s['WER']==0 and s['CER']==0

@pytest.mark.parametrize('raw',['<!-- comment -->','<!bad>','<img','<img src="p.png" src="present.png"/>','Text <span><img src="p.png"/>'])
def test_F21_actual_markup_boundaries_preserved(tmp_path,raw):
    (tmp_path/'document.md').write_text(raw)
    assert audit(tmp_path)['status']=='INCOMPLETE'
