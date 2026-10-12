"""F27–F30: inert token boundaries through every finite Markdown producer.

Synthetic gold is provisional. Automatic gates, fixture content assertions and
actual lexical coordinates are tested; none is a new human V.
"""
from copy import deepcopy
from pathlib import Path
import shutil,sys
import pytest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/benchmarks'))
from observations import observe,inline,HTMLTree,markdown_units
from benchmark import observe_bundle
import audit as auditor
from metrics import score,evaluate_exit,known_text
from test_conversion_fix_r003 import M,AST

P={'document_id':'synthetic-r006','blocks':[{'block_id':'p','type':'paragraph','text':'È Known words.'}]}
T={'document_id':'synthetic-r006-table','blocks':[{'block_id':'t','type':'table','text':'','header_rows':1,'cells':[['h'],['È Known words.']]}]}
TOKENS=['<!-- secret > <img src="absent.png"/> $x_i$ [r](../outside.png) -->',
        '<![CDATA[secret > <img src="absent.png"/> $x_i$ [r](../outside.png)]]>']
MULTILINE=['<!--\n\n# secret heading\n- secret list\n```\n> secret quote\n[^secret]: secret note\n::: hidden\n![secret](../outside.png)\n<img src="absent.png"/>\n$x_i$\n| hidden | 99 |\n\n-->',
           '<![CDATA[secret >\n\n# secret heading\n![secret](../outside.png)\n<img src="absent.png"/>\n$x_i$\n| hidden | 99 |\n]]>']

def pipeline(tmp_path,raw,gold=P,asset=False):
    (tmp_path/'document.md').write_text(raw)
    if asset:shutil.copyfile(ROOT/'tests/fixtures/conversion-corpus/assets/figure-it.png',tmp_path/'p.png')
    o=observe_bundle(tmp_path,tmp_path/'observations.json');a=auditor.audit(tmp_path)
    s=score(deepcopy(gold),o,a,verification=None)
    return o,a,s


def host(payload,where):
    if where=='inline':return payload,deepcopy(P)
    if where=='paragraph':return '<p>'+payload+'</p>',deepcopy(P)
    if where=='wrapper':return '<div><section><p>'+payload+'</p></section></div>',deepcopy(P)
    if where=='fence':return '```{=html}\n<div><p>'+payload+'</p></div>\n```\n',deepcopy(P)
    if where=='pipe-cell':return '| h |\n| --- |\n| '+payload+' |\n',deepcopy(T)
    if where=='html-cell':return '<table><tr><th>h</th></tr><tr><td>'+payload+'</td></tr></table>',deepcopy(T)
    if where in ('pipe-caption','html-caption'):
        g=deepcopy(T);g['blocks'][0]['caption']='È Known words.'
        return ('| h |\n| --- |\n| È Known words. |\n\nTable: '+payload+'\n' if where=='pipe-caption' else '<table><caption>'+payload+'</caption><tr><th>h</th></tr><tr><td>È Known words.</td></tr></table>'),g
    if where=='residual':return '<table><tr><th>h</th></tr><tr><td>È Known words.</td></tr><span>'+payload+'</span></table>',deepcopy(T)
    if where=='figure':return '![%s](p.png)'%payload,{'document_id':'f','blocks':[{'block_id':'f','type':'figure','text':'','caption':'È Known words.','asset':'figure-it.png'}]}
    if where in ('list','quote','heading','note'):
        prefix={'list':'- ','quote':'> ','heading':'# ','note':'[^n]: '}[where];g=deepcopy(P)
        if where=='heading':g['blocks'][0]['type']='heading'
        if where=='note':g['blocks'][0].update(type='note',note_id='n')
        return prefix+payload+'\n',g
    raise AssertionError(where)


def lexemes(o):return [u for n in o['nodes'] for u in n.get('unknown',[]) if u.get('profile')=='html-tokens-v1']


def assert_exact_token(raw,o,token,kind):
    values=[u for u in lexemes(o) if u.get('token_type')==kind]
    assert len(values)==1
    u=values[0];at=raw.index(token)
    assert u['raw']==token and u['end']-u['offset']==len(token)
    assert u['source_position']['char']==at and u['source_position']['byte']==len(raw[:at].encode())
    assert raw[at:at+len(u['raw'])]==u['raw']


def assert_inert(o,a,s,opened,outer_image=False):
    assert a['status']=='INCOMPLETE' and a['reason']=='UNSUPPORTED_RESOURCE_SYNTAX'
    assert s['automatic_gates']!='PASS' and evaluate_exit(2,s['score_exit'])==2
    assert len(a['assets'])==int(outer_image)
    assert set(opened)==({'document.md','p.png'} if outer_image else {'document.md'})
    refs=[r for n in o['nodes'] for r in n.get('references',[])]
    assert len(refs)==int(outer_image) and all(r['target']=='p.png' for r in refs)
    assert not any(n.get('formulas') for n in o['nodes'])
    for n in o['nodes']:
        leaves=[n,*[c for row in n.get('cell_contents',[]) for c in row],*n.get('residual_contents',[])]
        if n.get('caption_content'):leaves.append(n['caption_content'])
        assert all('secret' not in v.get('text','') and 'hidden' not in known_text(v) for v in leaves)


@pytest.mark.parametrize('where',['inline','paragraph','wrapper','fence','pipe-cell','html-cell','pipe-caption','html-caption','residual','list','quote','heading','note'])
def test_F27_empty_close_diagnostic_known_once(tmp_path,where):
    raw,g=host('È Known </>words.',where);o,a,s=pipeline(tmp_path,raw,g)
    assert_exact_token(raw,o,'</>','empty-close')
    assert a['status']=='INCOMPLETE' and s['automatic_gates']!='PASS'
    assert evaluate_exit(2,s['score_exit'])==2
    assert sum(len(n.get('references',[])) for n in o['nodes'])==0
    n=o['nodes'][0];leaf=n['cell_contents'][1][0] if 'cell' in where else n['caption_content'] if 'caption' in where else n
    if where!='residual':assert leaf['text']=='È Known words.'


@pytest.mark.parametrize('where',['inline','paragraph','wrapper','fence','pipe-cell','html-cell','pipe-caption','html-caption','residual','figure','list','quote','heading','note'])
@pytest.mark.parametrize('token',TOKENS+MULTILINE)
def test_F28_F29_whole_token_inert_every_producer(tmp_path,monkeypatch,where,token):
    raw,g=host('È Known '+token+'words.',where)
    opened=[];original=auditor.regular
    def spy(root,path):opened.append(path);return original(root,path)
    monkeypatch.setattr(auditor,'regular',spy)
    o,a,s=pipeline(tmp_path,raw,g,asset=where=='figure')
    assert_exact_token(raw,o,token,'comment' if token.startswith('<!--') else 'cdata')
    assert_inert(o,a,s,opened,where=='figure')
    n=o['nodes'][0]
    leaf=n['cell_contents'][1][0] if 'cell' in where else n['caption_content'] if 'caption' in where or where=='figure' else n
    if where!='residual':assert leaf['text']=='È Known words.'
    if where not in ('pipe-cell','html-cell','pipe-caption','html-caption','residual','figure'):
        assert s['CER'] is None and s['WER'] is None
    assert n['offset']['line_end']==len(raw.splitlines())


@pytest.mark.parametrize('body',['\n\nHidden prose.\n\n','\n# Hidden heading\n','\n- Hidden list\n','\n```\nHidden code\n```\n','\n> Hidden quote\n','\n[^n]: Hidden note\n','\n::: hidden\n','\n\n![Hidden](p.png)\n<img src="p.png"/>\n$x_i$\n'])
@pytest.mark.parametrize('wrapper',['{}','<div>{}</div>','```{=html}\n{}\n```\n'])
@pytest.mark.parametrize('closure',['-->','--!>'])
def test_F29_blank_and_markers_never_create_nodes(tmp_path,monkeypatch,body,wrapper,closure):
    token='<!--'+body+closure;raw=wrapper.replace('{}',token)+'\n\nÈ Known words.\n'
    opened=[];original=auditor.regular
    def spy(root,path):opened.append(path);return original(root,path)
    monkeypatch.setattr(auditor,'regular',spy)
    o,a,s=pipeline(tmp_path,raw,P,True)
    assert_exact_token(raw,o,token,'comment');assert_inert(o,a,s,opened)
    assert not any(n['type'] in ('heading','figure','table','formula','note','code') for n in o['nodes'])
    assert all('Hidden' not in n.get('text','') for n in o['nodes'])
    assert sum(n.get('text','').count('È Known words.') for n in o['nodes'])==1


@pytest.mark.parametrize('token',['<!--\n\nsecret <img src="../outside.png"/>\n# secret','<![CDATA[secret >\n\n<img src="absent.png"/>\n$x_i$'])
@pytest.mark.parametrize('prefix',['È Known ','<p>È Known ','- È Known ','# È Known ','> È Known ','[^n]: È Known '])
def test_F28_F29_EOF_covers_entire_remaining_input(tmp_path,monkeypatch,token,prefix):
    raw=prefix+token;opened=[];original=auditor.regular
    def spy(root,path):opened.append(path);return original(root,path)
    monkeypatch.setattr(auditor,'regular',spy)
    o,a,s=pipeline(tmp_path,raw)
    assert_exact_token(raw,o,token,'comment' if token.startswith('<!--') else 'cdata');assert_inert(o,a,s,opened)
    assert o['nodes'][0]['text']=='È Known '


@pytest.mark.parametrize('closure',['-->','--!>',''])
def test_F29_commented_gold_never_satisfies_missing_page(tmp_path,closure):
    gold={'document_id':'missing','blocks':[{'block_id':'p','type':'paragraph','text':'È Known words.','page':1},{'block_id':'q','type':'paragraph','text':'Other words here.','page':2}]}
    raw='È Known words.\n\n<!--\n\nOther words here.\n\n'+closure
    o,a,s=pipeline(tmp_path,raw,gold)
    assert s['blocks']['missing']==['q'] and s['blocks']['unresolved_attributed']==[]
    assert s['score_exit']==3 and next(p for p in s['pages'] if p['page']==2)['missing_page']
    assert s['totals']['word_ref']==6 and s['WER'] is None
    assert s['text_lower_bound']['word_errors']==0


@pytest.mark.parametrize('raw',['<p>È Known words.</p>','<div><p>È Known words.</p></div>','È Known words.','# È Known words.','- È Known words.','> È Known words.','p < 0.05','a < b'])
def test_positive_without_tokens_is_measurable(tmp_path,raw):
    g=deepcopy(P)
    if raw.startswith('#'):g['blocks'][0]['type']='heading'
    if raw in ('p < 0.05','a < b'):g['blocks'][0]['text']=raw
    o,a,s=pipeline(tmp_path,raw,g)
    assert a['status']=='COMPLETE' and s['automatic_gates']=='PASS' and s['WER']==0


@pytest.mark.parametrize('literal',['</>','<!--\n\n# Hidden\n![x](absent.png)\n-->','<![CDATA[a>b <img src="absent.png"/>]]>'])
@pytest.mark.parametrize('ticks',['```','~~~'])
def test_literal_fence_does_not_open_HTML_token(tmp_path,literal,ticks):
    raw=ticks+'html\n'+literal+'\n'+ticks+'\n';g={'document_id':'code','blocks':[{'block_id':'c','type':'literal_text','text':literal+'\n'}]}
    o,a,s=pipeline(tmp_path,raw,g)
    assert a['status']=='COMPLETE' and s['automatic_gates']=='PASS'
    assert not lexemes(o) and not a['references'] and not a['assets']


@pytest.mark.parametrize('value',['</>','<!--','<![CDATA[a>b]]>'])
@pytest.mark.parametrize('context',['`{}`','![`{}`](p.png)','Known `{}` words.'])
def test_inline_code_comment_opener_remains_literal(tmp_path,value,context):
    raw=context.replace('{}',value)+'\n\nAfter words.\n';o=observe(raw.encode())
    assert not lexemes(o)
    assert o['nodes'][-1]['text']=='After words.'
    assert not any(r.get('target')!='p.png' for n in o['nodes'] for r in n['references'])


@pytest.mark.parametrize('raw,reason,count',[('<div>','unclosed HTML container',1),('</div>','unmatched HTML close',1),('<div><section>','unclosed HTML container',2),('<div></section>','unmatched HTML close',2)])
def test_F30_neutral_summary_preserves_occurrences(tmp_path,raw,reason,count):
    o,a,s=pipeline(tmp_path,raw,{'document_id':'empty','blocks':[]})
    n=o['nodes'][0];assert n['reason']=='HTML parser diagnostics without projected leaves'
    assert len(n['unknown'])==count and any(u['reason']==reason for u in n['unknown'])
    assert a['status']=='INCOMPLETE' and not n['text']
    assert all(raw[u['source_position']['char']:].startswith(u['raw']) for u in n['unknown'])


def test_multiple_tokens_unicode_and_known_math_reference_once(tmp_path):
    raw='<p>È Known </><span><a href="https://example.invalid">ref</a>'+M+'</span><![CDATA[a>b]]><!--x-->words.</p>'
    o,a,s=pipeline(tmp_path,raw)
    assert [u['token_type'] for u in lexemes(o)]==['empty-close','cdata','comment']
    n=o['nodes'][0];assert n['text']=='È Known refwords.' and len(n['references'])==len(n['formulas'])==1
    assert n['formulas'][0]['math']['structure']==AST
    for token,kind in [('</>','empty-close'),('<![CDATA[a>b]]>','cdata'),('<!--x-->','comment')]:assert_exact_token(raw,o,token,kind)
    assert a['status']=='INCOMPLETE'


@pytest.mark.parametrize('token',['<!-->','<!--->','<!-- --!>','<!-->secret-->','<!--secret--','<!--secret-','<![CDATA[secret >]]>','<![CDATA[secret >'])
def test_shared_intervals_match_actual_tokenizer_callbacks(token):
    p=HTMLTree(token);p.feed(token);p.close();nodes=p.root['children']
    assert len(nodes)==1 and nodes[0]['tag']=='#token'
    assert nodes[0]['start']==0 and nodes[0]['end']==len(token)
    units,_,_=markdown_units(token);assert ''.join(units)==token

@pytest.mark.parametrize('token',['<!--secret * ** <img src="absent.png"/> [x](../outside.png) -->','<![CDATA[secret > * ** <img src="absent.png"/>]]>','<!--secret\n\n* **\n<img src="absent.png"/>\n-->'])
@pytest.mark.parametrize('marker',['*','**','_','__'])
@pytest.mark.parametrize('image_label',[False,True])
def test_F28_F29_emphasis_never_cuts_inert_token(tmp_path,monkeypatch,token,marker,image_label):
    payload=marker+'È Known '+token+'words.'+marker
    raw='!['+payload+'](p.png)' if image_label else payload
    opened=[];original=auditor.regular
    def spy(root,path):opened.append(path);return original(root,path)
    monkeypatch.setattr(auditor,'regular',spy)
    o,a,s=pipeline(tmp_path,raw,P,image_label)
    assert_exact_token(raw,o,token,'comment' if token.startswith('<!--') else 'cdata')
    assert_inert(o,a,s,opened,image_label)
    assert o['nodes'][0]['text']=='È Known words.'


@pytest.mark.parametrize('prefix',['È Known ','# È Known ','- È Known ','> È Known ','[^n]: È Known ','| h |\n| --- |\n| È Known ','| h |\n| --- |\n| È Known words. |\n\nTable: È Known '])
@pytest.mark.parametrize('ending',['\n','\r\n','\n\n',' |',' {untrusted}'])
@pytest.mark.parametrize('opener',['<!--secret','<![CDATA[secret >'])
def test_F28_F29_EOF_padding_and_metadata_inside_token_preserved(tmp_path,monkeypatch,prefix,ending,opener):
    token=opener+'\n\nsecret <img src="absent.png"/>'+ending;raw=prefix+token
    opened=[];original=auditor.regular
    def spy(root,path):opened.append(path);return original(root,path)
    monkeypatch.setattr(auditor,'regular',spy)
    o,a,s=pipeline(tmp_path,raw,T if prefix.startswith('|') else P)
    assert_exact_token(raw,o,token,'comment' if opener.startswith('<!--') else 'cdata')
    assert_inert(o,a,s,opened)


@pytest.mark.parametrize('tail',['','\n```\n','\n~~~\n'])
def test_F29_open_comment_in_raw_fence_reaches_EOF(tmp_path,tail):
    token='<!--secret\n\n# hidden\n![secret](../outside.png)'+tail
    raw='```{=html}\n<p>È Known words.</p>\n'+token
    o,a,s=pipeline(tmp_path,raw)
    assert_exact_token(raw,o,token,'comment')
    assert a['status']=='INCOMPLETE' and not a['references'] and not a['assets']
    assert all('secret' not in n.get('text','') and 'hidden' not in n.get('text','') for n in o['nodes'])
    assert sum(n.get('text','').count('È Known words.') for n in o['nodes'])==1


def test_F29_unclosed_fence_does_not_own_comment_token(tmp_path):
    raw='```html\n<!--\n\n# hidden\n<img src="../outside.png"/>'
    o,a,s=pipeline(tmp_path,raw)
    # F55: EOF is not ordinary code ownership. The existing opaque-comment
    # contract still owns its own tail, independently of that invalid fence.
    assert_exact_token(raw,o,raw[8:],'comment')
    assert not any(n['type']=='code' for n in o['nodes'])
    assert not a['references'] and not a['assets']
    assert a['status']=='INCOMPLETE' and a['reason']=='UNSUPPORTED_RESOURCE_SYNTAX'


def test_F29_HTML_subtree_delimiter_and_attribute_not_markdown(tmp_path):
    raw='*È Known <span title="<!--"><em>*</em><!--secret * <img src="absent.png"/>--></span>words.*'
    o,a,s=pipeline(tmp_path,raw)
    assert_exact_token(raw,o,'<!--secret * <img src="absent.png"/>-->','comment')
    assert len(o['nodes'])==1 and o['nodes'][0]['text']=='È Known *words.'
    assert a['status']=='INCOMPLETE' and not a['references']

@pytest.mark.parametrize('marker',['*','**','_','__'])
@pytest.mark.parametrize('payload,domain,expected',[
    ('$a*b$','formulas','a*b'),
    ('[ref](https://example.invalid/a*b "*title*")','references','https://example.invalid/a*b'),
    ('\\ref{id*ref}','references','id*ref'),
    ('\\foo{opaque * raw}','unknown','\\foo{opaque * raw}'),
])
def test_known_leaf_is_not_split_by_outer_formatting(tmp_path,marker,payload,domain,expected):
    token='<!--secret * <img src="absent.png"/>-->'
    raw=marker+'È Known '+payload+' '+token+'words.'+marker
    o,a,s=pipeline(tmp_path,raw)
    assert_exact_token(raw,o,token,'comment')
    values=[v for n in o['nodes'] for v in n.get(domain,[])]
    assert sum(v.get('target')==expected if domain=='references' else v.get('raw')==expected for v in values)==1
    assert not any(r['kind']=='image' for n in o['nodes'] for r in n['references'])
    assert a['status']=='INCOMPLETE' and a['reason']=='UNSUPPORTED_RESOURCE_SYNTAX'
    assert not any('secret' in n.get('text','') for n in o['nodes'])
    if domain=='formulas':
        f=o['nodes'][0]['formulas'][0];assert f['raw']==expected
        assert f['source_position']['char']==raw.index('$')
    if payload.startswith('[ref]'):
        assert o['nodes'][0]['references'][0]['title']=='*title*'
        assert o['nodes'][0]['references'][0]['source_position']['char']==raw.index('[ref]')


def test_balanced_HTML_sibling_does_not_activate_literal_comment_code(tmp_path):
    raw='Known <span>words.</span> `<!--`\n\nAfter words.'
    o,a,s=pipeline(tmp_path,raw)
    assert not lexemes(o) and a['status']=='COMPLETE'
    assert o['nodes'][-1]['text']=='After words.'
