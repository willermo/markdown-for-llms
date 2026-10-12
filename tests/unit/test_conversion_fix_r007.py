"""F31–F34 finite producer domains; synthetic PROVISIONAL gold, no new V."""
from copy import deepcopy
from pathlib import Path
import shutil,sys
import pytest
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'scripts/benchmarks'))
from benchmark import observe_bundle
from observations import markdown_lexemes,inline,observe
import audit as auditor
from metrics import score,evaluate_exit,known_text
P={'document_id':'synthetic-r007','blocks':[{'block_id':'p','type':'paragraph','text':'Known words.'},{'block_id':'q','type':'paragraph','text':'Hidden words.','page':2}]}

def pipeline(tmp_path,raw,gold=P,asset=False):
    (tmp_path/'document.md').write_text(raw)
    if asset:shutil.copyfile(ROOT/'tests/fixtures/conversion-corpus/assets/figure-it.png',tmp_path/'p.png')
    o=observe_bundle(tmp_path,tmp_path/'observations.json');a=auditor.audit(tmp_path);m=score(deepcopy(gold),o,a,verification=None)
    return o,a,m

def tokens(o):return [u for n in o['nodes'] for u in n.get('unknown',[]) if u.get('profile')=='html-tokens-v1']

PREFIXES=['Before `unclosed','Before $x','Before $$x','It costs $5 today.','Before <alpha','Before `unclosed\nnext soft line']
BODIES=['\n\nHidden words.\n\n![hidden](p.png)\n\n# hidden heading\n$x_i$\n<img src="absent.png"/>\n',
        '\n# hidden heading\n- hidden list\n> hidden quote\n[^hidden]: hidden note\n::: hidden\n```\n![hidden](../outside.png)\n\nHidden words.\n']
@pytest.mark.parametrize('prefix',PREFIXES)
@pytest.mark.parametrize('body',BODIES)
@pytest.mark.parametrize('kind',['comment','cdata'])
@pytest.mark.parametrize('closed',[True,False])
def test_F31_literal_domain_does_not_suspend_next_real_token(tmp_path,monkeypatch,prefix,body,kind,closed):
    opener,closer=('<!--','-->') if kind=='comment' else ('<![CDATA[secret >',']]>')
    token=opener+body+(closer if closed else '')
    suffix='\n\nAfter ` literal $x_i$.\n' if closed else ''
    raw=prefix+'\n\nKnown words.\n\n'+token+suffix
    opened=[];original=auditor.regular
    def spy(root,path,*a,**kw):opened.append(path);return original(root,path,*a,**kw)
    monkeypatch.setattr(auditor,'regular',spy);o,a,m=pipeline(tmp_path,raw,asset=True)
    values=[u for u in tokens(o) if u['token_type']==kind];assert len(values)==1
    u=values[0];at=raw.index(opener);assert u['raw']==token and u['end']-u['offset']==len(token)
    assert u['source_position']['char']==at and u['source_position']['byte']==len(raw[:at].encode())
    assert a['status']=='INCOMPLETE' and a['reason']=='UNSUPPORTED_RESOURCE_SYNTAX'
    assert not a['assets'] and not a['references'] and set(opened)=={'document.md'}
    assert not any('Hidden' in known_text(n) or n['type']=='figure' or n.get('text')=='hidden heading' for n in o['nodes'])
    assert 'q' in m['blocks']['missing'] and next(v for v in m['pages'] if v['page']==2)['missing_page']
    assert m['score_exit']==3 and evaluate_exit(2,m['score_exit'])==2
    assert sum(n['text']=='Known words.' for n in o['nodes'])==1

@pytest.mark.parametrize('marker',['`','$x','$$x','$5','<alpha'])
@pytest.mark.parametrize('suffix',['` literal.','$x$ literal.','~~~\nx=5$ <!-- c\n~~~','Use `<!--` here.'])
def test_F31_delimiter_never_resumes_inside_later_fence_or_literal(tmp_path,marker,suffix):
    raw='Before '+marker+'\n\n'+suffix+'\n\n# Visible heading\n\nAfter words.'
    o,a,m=pipeline(tmp_path,raw)
    assert any(n['type']=='heading' and n['text']=='Visible heading' for n in o['nodes'])
    assert o['nodes'][-1]['text']=='After words.' and not tokens(o)
    if suffix.startswith('~~~'):
        c=next(n for n in o['nodes'] if n['type']=='code');assert c['text']=='x=5$ <!-- c\n'

@pytest.mark.parametrize('uri',['https://example.invalid','http://example.invalid','mailto:x@example.invalid'])
@pytest.mark.parametrize('literal',['<!--','<?php','<![CDATA[','<!x','</>'])
@pytest.mark.parametrize('later',['After words.','Then type `-->` here.\n\nAfter words.'])
def test_F32_autolink_never_installs_HTML_context(tmp_path,uri,literal,later):
    first='See <'+uri+'> and type `'+literal+'` first.';raw=first+'\n\n# Visible heading\n\n'+later
    o,a,m=pipeline(tmp_path,raw);assert a['status']=='COMPLETE' and not tokens(o) and not markdown_lexemes(raw)
    assert [n['type'] for n in o['nodes']]==['paragraph','heading']+['paragraph']*(2 if later.startswith('Then') else 1)
    assert o['nodes'][0]['text']=='See '+uri+' and type '+literal+' first.' and o['nodes'][1]['text']=='Visible heading' and o['nodes'][-1]['text']=='After words.'
    assert [v['target'] for v in o['nodes'][0]['references']]==[uri] and not a['assets']
    for n in o['nodes']:
        assert n['raw']==raw[n['offset']['char_start']:n['offset']['char_end']]
        assert n['offset']['byte_start']==len(raw[:n['offset']['char_start']].encode())

@pytest.mark.parametrize('uri',['https://example.invalid','mailto:x@example.invalid'])
@pytest.mark.parametrize('tail',[r'\<!--', '[ref](https://example.invalid "<!--")', '[ref](https://example.invalid/<!-- "<![CDATA[")', '`<!--`{title="<![CDATA["}'])
def test_F32_following_escape_destination_title_or_attrs_stays_local(tmp_path,uri,tail):
    raw='See <'+uri+'> '+tail+'\n\n# Visible heading\n\nAfter words.'
    o,a,m=pipeline(tmp_path,raw);assert a['status']=='COMPLETE' and not tokens(o) and not markdown_lexemes(raw)
    assert [n['type'] for n in o['nodes']]==['paragraph','heading','paragraph']

@pytest.mark.parametrize('value',['ordinary','<!--','<![CDATA[','<?x','</>'])
@pytest.mark.parametrize('where',['link','span','code','heading'])
def test_F33_Metadata_atom_is_shared_and_does_not_absorb_external_blocks(tmp_path,where,value):
    meta='{#local title="'+value+'"}';first={'link':'È [ref](https://example.invalid)'+meta,'span':'È [ref]'+meta,'code':'È `ref`'+meta,'heading':'# È Known words. '+meta}[where]
    raw=first+'\n\n# Visible heading\n\nAfter words.';o,a,m=pipeline(tmp_path,raw)
    assert a['status']=='COMPLETE' and not tokens(o) and not markdown_lexemes(raw)
    assert [n['type'] for n in o['nodes']]==['heading' if where=='heading' else 'paragraph','heading','paragraph']
    assert o['nodes'][0]['text']==('È Known words.' if where=='heading' else 'È ref') and o['nodes'][1]['text']=='Visible heading' and o['nodes'][-1]['text']=='After words.'
    assert o['nodes'][0]['anchors'].count('local')==1
    metadata=o['nodes'][0]['attrs'] if where=='heading' else next(v['attrs'] for v in o['nodes'][0]['segments'] if v.get('attrs'))
    assert metadata['title']==value
    if where=='link':assert o['nodes'][0]['references'][0]['target']=='https://example.invalid'

@pytest.mark.parametrize('command',['ref','eqref','label','foo'])
@pytest.mark.parametrize('value',['plain','<!--','<![CDATA[','<?x'])
def test_F33_finite_TeX_atom_remains_local_and_preserves_its_semantics(tmp_path,command,value):
    atom='\\'+command+'{'+value+'}';raw='È Known '+atom+'words.\n\n# Visible heading\n\nAfter words.';o,a,m=pipeline(tmp_path,raw)
    assert a['status']=='COMPLETE' and not tokens(o) and not markdown_lexemes(raw)
    assert [n['type'] for n in o['nodes']]==['unknown' if command=='foo' else 'paragraph','heading','paragraph']
    assert o['nodes'][1]['text']=='Visible heading' and o['nodes'][-1]['text']=='After words.'
    n=o['nodes'][0]
    if command=='foo':
        assert n['unknown']==[{'raw':atom,'offset':len('È Known '),'reason':'raw TeX unsupported','source_position':{'char':len('È Known '),'byte':len('È Known '.encode()),'basis':'raw-block-relative-inline-v1'}}]
    elif command=='label':assert n['anchors']==[value]
    else:assert [v['target'] for v in n['references']]==[value]

@pytest.mark.parametrize('host',['paragraph','heading','link','span'])
def test_F33_real_HTML_in_visible_label_prosa_still_reaches_consumers(tmp_path,host):
    token='<!--\n\n![hidden](../outside.png)\n--> ';payload='Known '+token+'words.'
    raw={'paragraph':payload,'heading':'# '+payload,'link':'['+payload+'](https://example.invalid)','code':'`'+payload+'`','span':'['+payload+']{title="ordinary"}'}[host]
    o,a,m=pipeline(tmp_path,raw)
    assert a['status']=='INCOMPLETE' and a['reason']=='UNSUPPORTED_RESOURCE_SYNTAX' and not a['assets']

@pytest.mark.parametrize('raw',['Known [ref](https://example.invalid){onclick="x"}', 'Known `ref`{style="x"}', 'Known [ref](file:///outside)', '<p srcdoc="x">Known</p>'])
def test_F33_actual_active_metadata_and_URI_are_still_rejected(tmp_path,raw):
    (tmp_path/'document.md').write_text(raw)
    assert auditor.audit(tmp_path)['reason']=='UNSAFE_REFERENCE'

@pytest.mark.parametrize('marker',['<!--','<![CDATA['])
@pytest.mark.parametrize('host',['heading','cell','caption','label','note'])
@pytest.mark.parametrize('closed',[True,False])
def test_F31_source_barrier_survives_producer_projection(tmp_path,monkeypatch,marker,host,closed):
    close='-->' if marker=='<!--' else ']]>'
    token=marker+'\nsecret$ marker\n\n![hidden](p.png)\n\n'+(close if closed else '')
    first={'heading':'# È $5 '+token+' words.','cell':'| h |\n| --- |\n| È $5 '+token+' words. |','caption':'| h |\n| --- |\n| known |\n\nTable: È $5 '+token+' words.','label':'![È $5 '+token+' words.](outer.png)','note':'[^n]: È $5 '+token+' words.'}[host]
    raw=first+'\n\n# Visible heading\n\nAfter words.' if closed else first
    if not closed:token=raw[raw.index(marker):]
    shutil.copyfile(ROOT/'tests/fixtures/conversion-corpus/assets/figure-it.png',tmp_path/'outer.png')
    opened=[];original=auditor.regular
    def spy(root,path,*a,**kw):opened.append(path);return original(root,path,*a,**kw)
    monkeypatch.setattr(auditor,'regular',spy);o,a,m=pipeline(tmp_path,raw,asset=True)
    u=[u for u in tokens(o) if u['raw']==token];assert len(u)==1
    assert u[0]['source_position']['char']==raw.index(marker)
    assert u[0]['source_position']['byte']==len(raw[:raw.index(marker)].encode())
    assert a['status']=='INCOMPLETE' and a['reason']=='UNSUPPORTED_RESOURCE_SYNTAX'
    assert 'p.png' not in opened and not any(n['formulas'] for n in o['nodes'])
    assert 'q' in m['blocks']['missing'] and m['score_exit']==3
    if closed:
        assert any(n['type']=='heading' and n['text']=='Visible heading' for n in o['nodes'])
        assert o['nodes'][-1]['text']=='After words.'

@pytest.mark.parametrize('middle',['<!--literal',' # <!-- literal\n> end'])
def test_F31_quote_soft_code_is_a_single_consumer_domain(tmp_path,middle):
    raw='> Known `first\n> '+middle+'` words.\n\n# Visible heading\n\nAfter words.'
    o,a,m=pipeline(tmp_path,raw);assert a['status']=='COMPLETE' and not tokens(o) and not markdown_lexemes(raw)
    assert o['nodes'][0]['block_syntax']=='blockquote' and [n['type'] for n in o['nodes'][1:]]==['heading','paragraph']
    assert o['nodes'][-1]['text']=='After words.'

@pytest.mark.parametrize('cell',[r'\ref{a|b}', '[ref](https://example.invalid/a|b)', '[ref]{title="a|b"}', '`a|b`{title="<!--|"}'])
def test_F33_pipe_delimiter_respects_shared_finite_atom(tmp_path,cell):
    raw='| h | k |\n| --- | --- |\n| '+cell+' | last |\n\n# Visible heading'
    o,a,m=pipeline(tmp_path,raw);assert a['status']=='COMPLETE' and not tokens(o)
    table=o['nodes'][0];assert table['type']=='table' and table['cells'][-1][-1]=='last' and len(table['cells'][-1])==2
    assert o['nodes'][-1]['text']=='Visible heading'
