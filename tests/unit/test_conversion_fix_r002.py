"""F01–F11: attesi autonomi; payload passivi, nessun motore sui negativi."""
from copy import deepcopy
import io
import json
import os
from pathlib import Path
import shutil
import sys
import zipfile
import pytest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/benchmarks'))
from observations import observe,inline
from metrics import score,evaluate_exit
from audit import audit
from data import load,digest_bytes,descriptor
from safety import Unsafe,html,css,svg,zip_members
from adapters import validate
from pandoc_guard import check as pandoc_check
from generate_corpus import transcript
from test_conversion_benchmark import FP,FN,checked_control
CORPUS=ROOT/'tests/fixtures/conversion-corpus'


def gold(blocks):return {'document_id':'autonomous','blocks':blocks}
def block(type='paragraph',text='Known words.',**kwargs):return {'block_id':'b01','type':type,'text':text,**kwargs}
def measure(g,raw):return score(g,observe(raw.encode()),{'status':'COMPLETE'},verification={'status':'VERIFIED'})


@pytest.mark.parametrize('ident,source,alternative,ast',FP,ids=[v[0] for v in FP])
def test_F02_equivalence_complete_pipeline(ident,source,alternative,ast):
    g=gold([block('formula','',formulas=[{'formula_id':'f','reference_latex':source,'structure_expected':ast}])])
    o=observe(('$$'+alternative+'$$\n').encode());r=score(g,o,{'status':'COMPLETE'},verification={'status':'VERIFIED'})
    assert r['automatic_gates']=='PASS',r
    assert r['totals']['formulas_exact']==1 and r['WER'] is None
    assert r['score_exit']==5 and any(v['metric']=='formula_output_typography_and_semantics' for v in r['unavailable'])
    assert o['nodes'][0]['formulas'][0]['raw']==alternative


@pytest.mark.parametrize('ident,mutation,ast',FN,ids=[v[0] for v in FN])
def test_F02_inequivalent_is_not_pass(ident,mutation,ast):
    g=gold([block('formula','',formulas=[{'formula_id':'f','reference_latex':'independent expected AST','structure_expected':ast}])])
    r=measure(g,'$$'+mutation+'$$\n');assert r['score_exit']==3,r


def test_F02_unsupported_duplicates_and_ambiguous_remain_distinct():
    ast=FP[0][3];b=block('formula','',formulas=[{'formula_id':'f','reference_latex':'x_i','structure_expected':ast}])
    assert measure(gold([b]),r'$$\unknown{x}$$')['automatic_gates']=='INDETERMINATE'
    assert measure(gold([b]),'$$x_i$$\n\n$$x_{i}$$')['score_exit']==3
    assert measure(gold([b,{**b,'block_id':'b02'}]),'$$x_i$$\n\n$$x_{i}$$')['automatic_gates']!='PASS'


@pytest.mark.parametrize('added', ['$x_i$', '![](assets/absent.png)', '[r](missing.txt)', r'\unknown{x}'])
def test_F01_empty_cell_cannot_hide_typed_content(tmp_path,added):
    src=CORPUS/'controls/K01-full';shutil.copytree(src,tmp_path/'raw');raw=tmp_path/'raw'
    text=(raw/'document.md').read_text().replace('|0012|−7||','|0012|−7|'+added+'|')
    (raw/'document.md').write_text(text);a=audit(raw);g=load(src/'annotations.json');r=score(g,observe(text.encode()),a)
    assert r['automatic_gates']!='PASS',r
    table=next(n for n in observe(text.encode())['nodes'] if n['type']=='table')
    assert table['cell_contents'][1][2]['raw']==added
    if 'absent' in added:assert a['status']=='INCOMPLETE' and evaluate_exit(2,r['score_exit'])==2 and r['score_exit']==3


@pytest.mark.parametrize('markup',[
    '|h|\n|---|\n|$x_i$ [r](present.txt) ![](p.png)|\n\n: $x_i$ [r](present.txt) ![](p.png)\n',
    '<table><caption><a href="present.txt">r</a><img src="p.png"/><math><semantics><mi>x</mi><annotation encoding="application/x-tex">x_i</annotation></semantics></math></caption><tr><th>h</th></tr><tr><td><a href="present.txt">r</a><img src="p.png"/><math><semantics><mi>x</mi><annotation encoding="application/x-tex">x_i</annotation></semantics></math></td></tr></table>\n'])
def test_F01_cell_caption_assets_all_seen_once(tmp_path,markup):
    (tmp_path/'document.md').write_text(markup);(tmp_path/'present.txt').write_text('present')
    shutil.copyfile(CORPUS/'assets/figure-it.png',tmp_path/'p.png')
    a=audit(tmp_path);o=observe(markup.encode());t=o['nodes'][0]
    assert a['status']=='COMPLETE',a
    assert len(a['assets'])==2 and len(a['references'])==2
    assert len(t['formulas'])==2 and len(t['references'])==4
    assert {v['location']['kind'] for v in t['formulas']}=={'cell','caption'}
    # All visible text and semantic occurrences are authored here independently.
    ast=FP[0][3];cell_text=' r ' if markup.startswith('|') else 'r';caption_text=cell_text
    leaf={'references':[{'kind':'link','target':'present.txt'},{'kind':'image','target':'p.png','asset':'figure-it.png'}],'formulas':[{'structure_expected':ast}]}
    g=gold([block('table','',cells=[['h'],[cell_text]],header_rows=1,caption=caption_text,formulas=[{'formula_id':str(i),'reference_latex':'x_i','structure_expected':ast} for i in range(2)],references=[{'kind':'link','text':'r','target':'present.txt'}]*2,cell_contents=[[{}],[leaf]],caption_content=leaf)])
    for image in a['assets']:image.update(content_verified=image['sha256']==descriptor(CORPUS/'assets/figure-it.png')['sha256'],expected_asset='figure-it.png')
    r=score(g,o,a);assert r['automatic_gates']=='PASS',r
    (tmp_path/'p.png').unlink();missing=audit(tmp_path)
    assert missing['status']=='INCOMPLETE' and score(g,o,missing)['score_exit']==3


def test_F01_typed_cell_positive_score(tmp_path):
    text='|h|\n|---|\n|$x_i$|\n';(tmp_path/'document.md').write_text(text)
    g=gold([block('table','',header_rows=1,cells=[['h'],['']],cell_contents=[[{}],[{'formulas':[{}]}]],formulas=[{'formula_id':'f','reference_latex':'x_i','structure_expected':FP[0][3]}])])
    assert score(g,observe(text.encode()),audit(tmp_path))['automatic_gates']=='PASS'


@pytest.mark.parametrize('rows',[[['Codice','Valore','Nota'],['0012','0',''],['B','−7','123,45']],[['Codice','Valore','Nota'],['0012','0','']],[['Code','Value','Note'],['0012','0',''],['B','−7','123.45']]])
def test_F07_headerless_empty_placeholder_body_exact(rows):
    g=gold([block('table','',cells=rows,header_rows=0,numbers=[{'text_index':4,'offset':0,'lexeme':'0012'},{'text_index':5,'offset':0,'lexeme':'0'}])])
    raw='|||\n|---|---|---|\n' # replace with three genuinely empty cells
    raw='||||\n|---|---|---|\n'+''.join('|'+'|'.join(r)+'|\n' for r in rows)
    r=measure(g,raw);assert r['automatic_gates']=='PASS',r
    assert r['totals']['cells_exact']==sum(map(len,rows)) and r['totals']['numbers_exact']==2
    assert r['details'][0]['body_cells']['exact']==sum(map(len,rows))
    promoted='|'+'|'.join(rows[0])+'|\n|---|---|---|\n'+''.join('|'+'|'.join(v)+'|\n' for v in rows[1:])
    r=measure(g,promoted);assert r['score_exit']==3 and any(v['metric']=='table_header' for v in r['failures'])
    assert r['details'][0]['body_cells']['exact']<sum(map(len,rows))


def test_F07_real_empty_header_not_dropped_and_empty_cell_missing():
    g=gold([block('table','',header_rows=1,cells=[['',''],['x','']])])
    assert measure(g,'|||\n|---|---|\n|x||\n')['automatic_gates']=='PASS'
    assert measure(g,'|||\n|---|---|\n|x|\n')['score_exit']==3
    assert measure(g,'|x||\n|---|---|\n')['score_exit']==3


def test_F05_word_insertions_fixed_denominator():
    g=gold([block()]);assert measure(g,'Known words.')['WER']==0
    r=measure(g,'Known words.\n\nAdditional invented content.')
    assert r['totals']['word_ref']==2 and r['totals']['word_errors']==3 and r['WER']==1.5
    # Missing, substitution and independent insertion: 2+3+1 edits / 4 expected.
    g=gold([block(text='First known words.'),block(text='Missing block.',block_id='b02')])
    r=measure(g,'First changed words.\n\nAdditional invented content.')
    assert r['totals']['word_ref']==5 and r['totals']['word_errors']>0


def test_F08_foreign_unknown_does_not_hide_page_loss():
    src=CORPUS/'controls/K02-three-sections';g=load(src/'annotations.json');text=(src/'document.md').read_text()
    # Copy source literal blocks excluding page2; never modify reviewer/source.
    text='\n'.join(b['locator']['literal'] for b in g['blocks'] if b.get('page')!=2)+r'\unknown{x}'+'\n'
    r=score(g,observe(text.encode()),checked_control('K02-three-sections'))
    assert r['score_exit']==3 and next(p for p in r['pages'] if p['page']==2)['missing_page']
    assert any(v['metric']=='reader' for v in r['unavailable'])


def test_F08_specific_unknown_only_covers_unique_payload():
    g=gold([block(text='Specific approved content.'),block(text='Other missing content.',block_id='b02')])
    r=measure(g,r'\unknown{Specific approved content.}')
    assert r['blocks']['unresolved_attributed']==['b01']
    assert r['blocks']['missing']==['b02'] and r['score_exit']==3
    repeated=gold([block(text='Specific approved content.'),block(text='Specific approved content.',block_id='b02')])
    assert not measure(repeated,r'\unknown{Specific approved content.}')['blocks']['unresolved_attributed']


@pytest.mark.parametrize('title',['"title: value"',"'title: value'",'(title: value)'])
def test_F09_destination_title_and_escaped_parens(tmp_path,title):
    shutil.copyfile(CORPUS/'assets/figure-it.png',tmp_path/'p(1).png')
    raw='![a](p\\(1\\).png '+title+')\n';(tmp_path/'document.md').write_text(raw)
    o=observe(raw.encode())['nodes'][0];assert o['asset_uri']=='p(1).png' and o['references'][0]['title']=='title: value'
    assert audit(tmp_path)['status']=='COMPLETE'
    assert inline('[a](https://example.invalid/a(b) '+title+')')['references'][0]['target']=='https://example.invalid/a(b)'
    (tmp_path/'document.md').write_text('![a](file:///outside.png '+title+')');assert audit(tmp_path)['status']=='INCOMPLETE'


def test_F09_raw_attribute_localized_code_unchanged():
    raw=r'`\cite{bib1}`{=latex}';v=inline(raw)
    assert v['unknown'][0]['raw']==raw and v['unknown'][0]['format']=='latex'
    assert v['segments'][0]['type']=='raw_inline'
    assert inline(r'`\cite{bib1}`')['segments'][0]['type']=='code'


@pytest.mark.parametrize('payload',[b'<p srcset="file:///outside 1x">x</p>',b'<img src="ok.png" srcset="https://example.invalid/x 1x"/>',b'<a href="https://example.invalid" ping="file:///outside">x</a>'])
def test_F04_html_resource_attributes_rejected(payload):
    with pytest.raises(Unsafe):html(payload)


def test_F04_css_and_svg_finite_negative_and_positive():
    for payload in [rb'p {background:u\72l(file:///outside)}',b'p{background:u/**/rl(file:///outside)}']:
        with pytest.raises(Unsafe):css(payload)
    for payload in [b'<svg><script/></svg>',b'<svg><image href="file:///outside"/></svg>']:
        with pytest.raises(Unsafe):svg(payload)
    html(b'<p><a href="https://example.invalid">inert</a><img src="ok.png"/></p>');css(b'p {color:black}');svg(b'<svg><image href="ok.png"/></svg>')


@pytest.mark.parametrize('change',['srcset-file','srcset-remote','css-escape','svg-script','svg-resource','ncx-file','opf-uppercase','xml-base','html-xml-events'])
def test_F04_actual_epub_rejected_before_engine(tmp_path,change,monkeypatch):
    from benchmark import run_attempt
    from profile import config
    original=next(d for d in load(CORPUS/'manifest.json')['documents'] if d['format']=='epub')
    input=CORPUS/original['input'];dest=tmp_path/'mutated.epub'
    with zipfile.ZipFile(input) as z:members={n:z.read(n) for n in z.namelist()}
    if change.startswith('srcset'):members['OEBPS/chapter1.xhtml']=members['OEBPS/chapter1.xhtml'].replace(b'<img ',b'<img srcset="'+(b'file:///outside' if change=='srcset-file' else b'https://example.invalid/x')+b' 1x" ',1)
    elif change=='css-escape':members['OEBPS/style.css']+=rb' p{background:u\72l(file:///outside)}'
    elif change in ('svg-script','svg-resource'):members['OEBPS/new.svg']=b'<svg><script/></svg>' if change=='svg-script' else b'<svg><image href="file:///outside"/></svg>'
    elif change=='ncx-file':members['OEBPS/test.ncx']=b'<ncx><content src="file:///outside"/></ncx>'
    elif change=='opf-uppercase':members['OEBPS/test.OPF']=b'<package><manifest><item href="file:///outside"/></manifest></package>'
    elif change=='xml-base':members['OEBPS/base.opf']=b'<package xml:base="file:///outside/"><manifest><item href="secret.xml"/></manifest></package>'
    else:members['OEBPS/hidden.xml']=b'<html><body><p onclick="bad()">x</p></body></html>'
    outside=tmp_path/'outside-sentinel';outside.write_bytes(b'not opened by validation');outside_marker=tmp_path/'outside-opened'
    members={n:v.replace(b'file:///outside',outside.as_uri().encode()) for n,v in members.items()}
    real_open=os.open
    def guarded_open(path,*a,**k):
        if str(path)==str(outside):outside_marker.write_text('bad');raise AssertionError('outside target must not open')
        return real_open(path,*a,**k)
    monkeypatch.setattr(os,'open',guarded_open)
    with zipfile.ZipFile(dest,'w') as z:
        for n,v in members.items():z.writestr(n,v)
    doc={k:original[k] for k in ('document_id','format')};doc.update(input=dest.name,input_sha256=digest_bytes(dest.read_bytes()))
    marker=tmp_path/'engine-opened';installation=tmp_path/'I.json';installation.write_text(json.dumps({'payload':{'installed':{'canonical_archive':{}}}}))
    def forbidden(*a,**k):marker.write_text('bad');raise AssertionError('engine must not run')
    monkeypatch.setattr('benchmark.convert',forbidden)
    r=run_attempt(tmp_path,doc,'pandoc-direct-v1',config('epub','pandoc-direct-v1'),tmp_path/'attempt','f04',change,{'PATH':'/usr/bin'},installation)
    assert r['run_exit']==4 and not r['engine_started'] and not marker.exists() and not (tmp_path/'attempt/raw').exists() and not outside_marker.exists()


@pytest.mark.parametrize('path',['/tmp/fake:/usr/bin','/home/user/bin:/usr/bin','.:/usr/bin',':/usr/bin','relative:/usr/bin','/usr/bin:'])
def test_F11_bad_path_before_tool(tmp_path,path,monkeypatch):
    marker=[]
    monkeypatch.setattr('pandoc_guard.subprocess.run',lambda *a,**k:marker.append(a))
    with pytest.raises(ValueError):pandoc_check({'PATH':path},descriptor('/usr/bin/pandoc'))
    assert not marker


def test_F11_closed_path_resolution():
    r=pandoc_check({'PATH':'/usr/bin'},descriptor('/usr/bin/pandoc'))
    assert r['which']==r['realpath']=='/usr/bin/pandoc' and r['version']=='pandoc 3.1.3'


def test_F10_transcript_caption_expected_autonomously():
    g=gold([block('table','',cells=[['A',''],['0012','−7']],caption='Caption α'),block(text='After.',block_id='b02')])
    assert transcript(g)=='[b01; pagina=N/A; tipo=table]\n\n\nA | \n\n0012 | −7\n\nCaption α\n\n[b02; pagina=N/A; tipo=paragraph]\nAfter.\n'


def test_F01_html_unknown_caption_and_cell_localized():
    raw='<table><caption><math><mi>x</mi></math></caption><tr><td><math><mi>y</mi></math></td></tr></table>'
    t=observe(raw.encode())['nodes'][0]
    assert len(t['unknown'])==2 and {u['location']['kind'] for u in t['unknown']}=={'cell','caption'}
    assert measure(gold([block('table','',cells=[['']],header_rows=0)]),raw)['automatic_gates']!='PASS'


def test_F11_invalid_environment_actual_run_receipt(tmp_path,monkeypatch):
    from benchmark import run_attempt
    from profile import config
    (tmp_path/'input.rtf').write_bytes(b'{\\rtf1 text}')
    installation=tmp_path/'I.json';installation.write_text(json.dumps({'payload':{'installed':{'canonical_archive':{}}}}))
    doc={'document_id':'path-negative','format':'rtf','input':'input.rtf','input_sha256':digest_bytes((tmp_path/'input.rtf').read_bytes())}
    marker=tmp_path/'spawned'
    def forbidden(*a,**k):marker.write_text('bad');raise AssertionError('must not spawn')
    monkeypatch.setattr('benchmark.convert',forbidden)
    r=run_attempt(tmp_path,doc,'pandoc-legacy-v1',config('rtf','pandoc-legacy-v1'),tmp_path/'attempt','f11','path',{'PATH':'/tmp/fake:/usr/bin'},installation)
    assert r['run_exit']==4 and not r['engine_started'] and not marker.exists()


def test_F11_mutation_between_hash_checks_detected(monkeypatch):
    expected=descriptor('/usr/bin/pandoc');original=Path.read_bytes;calls=[]
    def mutated(p):
        data=original(p)
        if str(p)=='/usr/bin/pandoc':
            calls.append(p)
            if len(calls)>1:return data+b'deliberate unit-only post-check mutation'
        return data
    monkeypatch.setattr(Path,'read_bytes',mutated)
    with pytest.raises(ValueError,match='changed during resolution'):pandoc_check({'PATH':'/usr/bin'},expected)
    assert len(calls)==2


def test_F07_inheritance_rejects_visible_change():
    from invariance import gold_invariant
    old=gold([block('table','',cells=[['0012','']],caption='Approved caption')])
    new=deepcopy(old);new['metadata_revision']={'revision':5};new['blocks'][0].update(header_rows=0,header_policy='encoded-no-tblHeader-v1')
    assert gold_invariant(old,new,'docx')['status']=='PASS'
    new['blocks'][0]['cells'][0][0]='12'
    with pytest.raises(ValueError,match='Visible annotation'):gold_invariant(old,new,'docx')


def test_F01_cell_caption_source_positions():
    raw='Before.\n\n|h|\n|---|\n|$x_i$|\n\n: $c^2$\n'
    t=next(n for n in observe(raw.encode())['nodes'] if n['type']=='table')
    assert [f['source_position']['char'] for f in t['formulas']]==[raw.index('$x_i$'),raw.index('$c^2$')]
    assert t['formulas'][0]['location']=={'kind':'cell','row':1,'col':0}
    assert t['formulas'][1]['location']=={'kind':'caption'}


def test_F04_original_reviewer_epub_payload_remains_rejected():
    source=ROOT/'temp/run-a002-fase1-corpus/work/implementation-r002/review-payload-copies/probe-actual-source.epub'
    if not source.exists():pytest.skip('Run-local original reviewer fixture not present; standalone portable cases above remain normative')
    doc={'document_id':'review-epub','format':'epub','input':source.name,'input_sha256':digest_bytes(source.read_bytes())}
    with pytest.raises(Unsafe):validate(source.parent,doc,'pandoc-direct-v1')


def test_F04_svg_cannot_hide_in_xml_and_css_unknown_functions():
    payload=io.BytesIO()
    with zipfile.ZipFile(payload,'w') as z:z.writestr('hidden.xml','<svg><script/></svg>')
    with pytest.raises(Unsafe):zip_members(payload.getvalue())
    for value in [b'p {background:image("https://example.invalid/a")}',b'p {background:attr(src url)}',b'p {filter:blur(1px)}']:
        with pytest.raises(Unsafe):css(value)
    css(b'p {font-family:"DejaVu Sans"; color:black} .comparison {grid-template-columns:minmax(360px,1fr) minmax(360px,1fr)}')


def test_F12_campaign_attempt_ids_include_new_experiment(tmp_path):
    from campaign import freeze
    c=tmp_path/'corpus';c.mkdir();stage=tmp_path/'stage';stage.mkdir();docs=[]
    from campaign import PATHS
    for ident,format in [('C01-txt-it','txt'),('C02-docx-it','docx'),('C03-epub-en','epub'),('C04-rtf-it','rtf'),('C05-tex-en','tex')]:
        (c/(ident+'.input')).write_bytes(b'unit-only input')
        (c/(ident+'.gold')).write_text('{}')
        docs.append({'document_id':ident,'format':format,'input':ident+'.input','input_sha256':descriptor(c/(ident+'.input'))['sha256'],'gold':ident+'.gold'})
    (c/'manifest.json').write_text(json.dumps({'documents':docs}))
    (stage/'target.json').write_text(json.dumps({'environment':{'PATH':'/usr/bin'}}));(stage/'I.json').write_text('{}');(stage/'benchmark-inputs.json').write_text('{}')
    result=freeze(c,stage,tmp_path/'frozen','new-r002')
    ids=[a for g in result['groups'] for a in g['attempts']]
    assert len(set(ids))==32 and all(a.startswith('new-r002-') for a in ids)


def test_F08_repeated_ambiguity_keeps_definite_extra_occurrence_failure():
    text='Repeated approved sentence.';g=gold([block(text=text),block(text=text,block_id='b02')])
    r=measure(g,'\n\n'.join([text]*3))
    assert r['blocks']['ambiguous'] and r['score_exit']==3
    assert any(f.get('reason')=='UNEXPECTED_AMBIGUOUS_OCCURRENCES' and f['count']==1 for f in r['failures'])


def test_F04_viewer_style_is_finite_passive_positive():
    from comparison import STYLE
    css(STYLE.encode())


@pytest.mark.parametrize('added',['![](p.png)','[r](present.txt)','$x_i$',r'\unknown{x}'])
def test_F01_added_caption_inline_never_hidden_by_text(tmp_path,added):
    shutil.copyfile(CORPUS/'assets/figure-it.png',tmp_path/'p.png');(tmp_path/'present.txt').write_text('present')
    raw='|h|\n|---|\n|v|\n\n: Caption'+added+'\n';(tmp_path/'document.md').write_text(raw)
    g=gold([block('table','',cells=[['h'],['v']],header_rows=1,caption='Caption')])
    r=score(g,observe(raw.encode()),audit(tmp_path));assert r['automatic_gates']!='PASS',r
    if added.startswith('!'):assert r['score_exit']==3 and any(f['metric']=='caption_inline' for f in r['failures'])


def test_F01_html_caption_occurrences_keep_source_order():
    raw='<table><caption><a href="caption.txt">c</a><math><semantics><annotation encoding="application/x-tex">c^2</annotation></semantics></math></caption><tr><td><a href="cell.txt">v</a><math><semantics><annotation encoding="application/x-tex">x_i</annotation></semantics></math></td></tr></table>'
    t=observe(raw.encode())['nodes'][0]
    assert [r['target'] for r in t['references']]==['caption.txt','cell.txt']
    assert [f['raw'] for f in t['formulas']]==['c^2','x_i']
    assert t['formulas'][0]['source_position']['char']<t['formulas'][1]['source_position']['char']
