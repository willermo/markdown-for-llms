"""Contratti a002. Attesi manuali e stub espliciti, non prove di conversione."""
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
HELPERS=ROOT/'scripts/benchmarks'
sys.path.insert(0,str(HELPERS))
from data import canonical,digest,digest_bytes,load,counts,descriptor,write_new
from math_structure import observe as math_observe,differences
from reference_check import TeXSource,rtf_source
from observations import observe,inline
from metrics import score,evaluate_exit,batch
from safety import Unsafe,regular,relative,zip_members,tree_inventory,html,svg
from audit import audit
from adapters import expand_tex,validate
from resources import run
from profile import config,ENABLED
from packet import verify_packet,import_response

CORPUS=ROOT/'tests/fixtures/conversion-corpus'
Q=lambda *items:{'type':'Seq','items':list(items)}
A=lambda *items:{'type':'Arg','body':Q(*items)}
S=lambda value:{'type':'Symbol','value':value}
N=lambda value:{'type':'Number','lexeme':value}
O=lambda kind:{'type':'Operator','kind':kind}
SCR=lambda base,sub=None,sup=None:{'type':'Scripts','base':base,'sub':sub,'sup':sup}
MATRIX={'type':'Matrix','delimiter':'square','rows':[[Q(N('0')),Q(O('minus'),N('1'))],[Q(N('2')),Q(N('3'))]]}
ALIGNED={'type':'Aligned','rows':[[Q(S('a')),Q(O('equals'),N('1'))],[Q(S('b')),Q(O('equals'),O('minus'),N('2'))]]}
ENERGY=Q(S('E'),O('equals'),S('m'),SCR(S('c'),sup=A(N('2'))))
SUM=Q({'type':'Sum','lower':A(S('i'),O('equals'),N('1')),'upper':A(S('n'))},SCR(S('x'),A(S('i'))))
FP=[
 ('FP01',r'x_i',r'x_{i}',Q(SCR(S('x'),A(S('i'))))),
 ('FP02',r'c^2',r'c^{2}',Q(SCR(S('c'),sup=A(N('2'))))),
 ('FP03',r'x_i^2',r'x^{2}_{i}',Q(SCR(S('x'),A(S('i')),A(N('2'))))),
 ('FP04',r'E=mc^{2}',r'E = m c^2',ENERGY),
 ('FP05',r'\frac{a}{b}',r'\frac a b',Q({'type':'Fraction','numerator':A(S('a')),'denominator':A(S('b'))})),
 ('FP06','-007','−007',Q(O('minus'),N('007'))),
 ('FP07','x−7','x - 7',Q(S('x'),O('minus'),N('7'))),
 ('FP08',r'\sum_{i=1}^{n}x_i',r'\sum^{n}_{i = 1} x_{i}',SUM),
 ('FP09',r'\begin{bmatrix}0&-1\\2&3\end{bmatrix}',r'\left[\begin{matrix}0&−1\\2&3\end{matrix}\right]',MATRIX),
 ('FP10',r'\begin{bmatrix}0&-1\\2&3\end{bmatrix}',r'[\begin{matrix}0&-1\\2&3\end{matrix}]',MATRIX),
 ('FP11',r'\begin{aligned}a&=1\\b&=-2\end{aligned}',r'\begin{aligned} a &= 1 \\ b &= −2 \end{aligned}',ALIGNED),
 ('FP12','a+b',r'a\,+\quad b',Q(S('a'),O('plus'),S('b'))),
 ('FP13',r'a+\text{−7  kg}',r'a + \text{−7  kg}',Q(S('a'),O('plus'),{'type':'Text','value':'−7  kg','command':'text'})),
]


@pytest.mark.parametrize('ident,source,alternative,expected',FP,ids=[r[0] for r in FP])
def test_math_positive_and_independent_source_M(ident,source,alternative,expected):
    assert source!=alternative
    # Expected is a hand-authored AST constant. Source M is a separate lexer,
    # and neither implementation creates expected by parsing engine output.
    assert TeXSource(source).parse()==expected
    for raw in (source,alternative):
        observed=math_observe(raw)
        assert observed['fully_supported'],observed
        assert observed['raw']==raw
        assert observed['structure']==expected


FN=[('FN01','007',FP[5][3]),('FN02','x+7',FP[6][3]),('FN03','x_j',FP[0][3]),
 ('FN04','x_i+1',Q(SCR(S('x'),A(S('i'),O('plus'),N('1'))))),
 ('FN05','x_12',Q(SCR(S('x'),A(N('12'))))),
 ('FN06',r'\frac{a}{b+c}',Q({'type':'Fraction','numerator':A(S('a'),O('plus'),S('b')),'denominator':A(S('c'))})),
 ('FN07-upper',r'\sum_{i=1}^{m}x_i',SUM),('FN07-body',r'\sum_{i=1}^{n}x_j',SUM),
 ('FN08',r'\begin{aligned}a&=1\end{aligned}',ALIGNED),
 ('FN09',r'\begin{bmatrix}0&-1\\2\end{bmatrix}',MATRIX),
 ('FN10-swap',r'\begin{bmatrix}0&-1\\3&2\end{bmatrix}',MATRIX),
 ('FN10-empty',r'\begin{bmatrix}&-1\\2&3\end{bmatrix}',MATRIX),
 ('FN11',r'\begin{matrix}0&-1\\2&3\end{matrix}',MATRIX),
 ('FN12-zeros','-7',FP[5][3]),('FN12-decimal','1.20',Q(N('1,20'))),
 ('FN13',r'a+\text{-7 kg}',FP[12][3])]


@pytest.mark.parametrize('ident,mutation,expected',FN,ids=[r[0] for r in FN])
def test_math_negative_localized(ident,mutation,expected):
    observed=math_observe(mutation)
    assert observed['fully_supported'],observed
    diff=differences(expected,observed['structure'])
    assert diff and all(d['path'].startswith('$') for d in diff)
    if ident=='FN05':assert any('.sub.' in d['path'] for d in diff)
    if ident=='FN11':assert any('.delimiter' in d['path'] for d in diff)
    if ident=='FN08':assert any('.rows.length' in d['path'] for d in diff)


@pytest.mark.parametrize('raw',[r'\unknown{x}',r'\verb|−7|','x'+('_i'*2),'{'*17+'x'+'}'*17,'x'*4097])
def test_math_unavailable_keeps_whole_raw(raw):
    result=math_observe(raw);assert not result['fully_supported'];assert result['raw']==raw;assert result['structure'] is None;assert result['offset'] is not None


def test_math_spacing_cannot_join_numeric_lexemes_or_remove_groups():
    assert math_observe(r'1\quad2')['structure']==Q(N('1'),N('2'))
    assert math_observe(r'1\quad2')['structure']!=math_observe('12')['structure']
    assert math_observe('{x}')['structure']!=math_observe('x')['structure']


def gold_control(name):return load(CORPUS/'controls'/name/'annotations.json')
def checked_control(name):
    root=CORPUS/'controls'/name;result=audit(root)
    result=deepcopy(result)
    for image in result['assets']:image.update(content_verified=True,expected_asset='figure-it.png',verification_method='unit test exact canonical hash')
    return result


@pytest.mark.parametrize('name',['K01-full','K02-three-sections','K03-repeated'])
def test_positive_all_automatic_gates_and_manual_pending(name):
    g=gold_control(name);o=observe((CORPUS/'controls'/name/'document.md').read_bytes());r=score(g,o,checked_control(name))
    assert r['automatic_gates']=='PASS',r['failures']
    assert r['quality']=='INDETERMINATE' and r['score_exit']==5
    assert not r['blocks']['missing'] and not r['blocks']['unexpected']
    assert r['blocks']['recall']==1 and r['blocks']['precision']==1 and r['CER']==0
    for key in ('numbers','formulas','cells','figures','references'):assert r['totals'][key+'_exact']==r['totals'][key+'_expected']
    assert r['order']['unresolved_pairs']==0 and r['order']['inversions']==0


@pytest.mark.parametrize('old,new,metric',[('123,45','123,46','numbers'),('−7','7','numbers'),('0012','12','numbers'),('E=mc^2','E=mc^3','formula'),('![Figura del controllo](assets/figure-it.png)','Figura del controllo','blocks'),('https://example.invalid/check?x=0012&y=-7','https://example.invalid/check?x=0012&y=-8','reference'),('|0012|−7||','|0012|||','cells'),('|0012|−7||','|0012|−7|','cells')])
def test_mutations_detected_with_fixed_denominators(old,new,metric):
    g=gold_control('K01-full');raw=(CORPUS/'controls/K01-full/document.md').read_text();observed=observe(raw.replace(old,new,1).encode());r=score(g,observed,checked_control('K01-full'))
    assert r['score_exit']==3 and r['quality']=='FAIL'
    assert r['denominators']==g['denominators']
    assert any(f['metric']==metric for f in r['failures']),r['failures']


def test_softwrap_and_note_renumbering_are_positive():
    g=gold_control('K01-full');raw=(CORPUS/'controls/K01-full/document.md').read_text().replace('spazio intenzionale; serie','spazio intenzionale;\nserie').replace('[^n1]','[^9]')
    r=score(g,observe(raw.encode()),checked_control('K01-full'))
    assert r['automatic_gates']=='PASS',r['failures'];assert r['totals']['raw_text_errors']>0
    assert r['totals']['references_exact']==r['denominators']['references']


def test_page_loss_distinct_from_single_block_and_order_swap():
    g=gold_control('K02-three-sections');raw=(CORPUS/'controls/K02-three-sections/document.md').read_text()
    without_page=raw[:raw.index('# Sezione 2')]+raw[raw.index('# Sezione 3'):]
    lost=score(g,observe(without_page.encode()),checked_control('K02-three-sections'))
    assert lost['score_exit']==3;assert next(p for p in lost['pages'] if p['page']==2)['missing_page'];assert lost['denominators']==counts(g)
    literal=g['blocks'][5]['locator']['literal'];partial=score(g,observe(raw.replace(literal,'',1).encode()),checked_control('K02-three-sections'))
    assert partial['score_exit']==3;assert not next(p for p in partial['pages'] if p['page']==2)['missing_page']
    swapped=raw[raw.index('# Sezione 2'):raw.index('# Sezione 3')]+raw[:raw.index('# Sezione 2')]+raw[raw.index('# Sezione 3'):]
    reorder=score(g,observe(swapped.encode()),checked_control('K02-three-sections'))
    assert reorder['blocks']['recall']==1 and reorder['score_exit']==3 and reorder['order']['inversions']>0
    assert reorder['order']['expected_pairs']==12*11//2


def test_duplicate_and_ambiguous_do_not_reuse_observation():
    name='K03-repeated';raw=(CORPUS/'controls'/name/'document.md').read_text();g=gold_control(name)
    duplicated=score(g,observe((raw+'\nFrase inattesa.\n').encode()),checked_control(name))
    assert duplicated['score_exit']==3 and duplicated['blocks']['unexpected']
    # Remove actual visible landmarks from source and corresponding headings
    # from expected copy; permanent gold remains protected.
    eg=deepcopy(g);eg['blocks']=[b for b in eg['blocks'] if b['type']=='paragraph'];eg['denominators']=counts(eg)
    ambiguous=score(eg,observe(b'Osservazione ripetuta senza ID nel testo.\n\nOsservazione ripetuta senza ID nel testo.\n'),checked_control(name))
    assert ambiguous['score_exit']==5 and ambiguous['blocks']['ambiguous']
    pairs=ambiguous['alignment']['free_pairs'];assert len({o for _,o in pairs})==len(pairs)


def test_gold_blind_digest_and_offsets_keep_unicode():
    raw=b'# Titolo {#t}\n\n'+('Caffè Cafè spazio −7 $x_i$.\n').encode()
    first=observe(raw);before=digest(first);g=gold_control('K03-repeated');score(g,first,{'status':'COMPLETE'})
    alternative=deepcopy(g);alternative['blocks'][1]['text']='Altro testo';score(alternative,first,{'status':'COMPLETE'})
    assert digest(first)==before==digest(observe(raw))
    for node in first['nodes']:
        off=node['offset'];assert raw[off['byte_start']:off['byte_end']].decode()==node['raw']
    assert first['nodes'][1]['text']=='Caffè Cafè spazio −7 .'
    assert first['nodes'][1]['formulas'][0]['math']['structure']==FP[0][3]


@pytest.mark.parametrize('context',['paragraph','code','cell'])
def test_FN14_literal_minus_not_math_equivalence(context):
    g={'document_id':'stub-literal','blocks':[{'block_id':'b1','type':'code' if context=='code' else 'table' if context=='cell' else 'paragraph','text':'Dato −7\n' if context=='code' else 'Dato −7' if context=='paragraph' else '', 'numbers':[{'text_index':1 if context=='cell' else 0,'offset':5,'lexeme':'−7'}]}]}
    if context=='cell':g['blocks'][0]['cells']=[['Dato −7','Vuoto','']]
    raw='```\nDato -7\n```\n' if context=='code' else '|Dato -7|Vuoto||\n|---|---|---|\n' if context=='cell' else 'Dato -7\n'
    r=score(g,observe(raw.encode()),{'status':'COMPLETE'})
    assert r['score_exit']==3 and any(x['metric']=='numbers' for x in r['failures'])
    assert not observe(raw.encode())['nodes'][0].get('formulas')


@pytest.mark.parametrize('case,raw,types',[
 ('P01','"quote" - −7 Caffè Cafè\n',['paragraph']),
 ('P02','|A|B|C|\n|---|---|---|\n|x\\|y|0||\n',['table']),
 ('P04','Costo \\$7; `x_$` e $x_i$.\n',['paragraph']),
 ('P05','$$\\unknown{x}$$\n',['formula']),
 ('P06','# Heading {#id .class}\n',['heading']),
 ('P07','::: {#outer}\n\n[Testo]{#span}\n\n::: \n',['paragraph']),
 ('P08','Nota[^2]\n\n[^2]: Prima riga\n    seconda riga\n',['paragraph','note']),
 ('P09','![Caption](media/fig.png)\n',['figure']),
 ('P10','```` {.python}\n$x_i$ `literal` \\ref{a} −7\n````\n',['code']),
 ('P11','<table><tr><td>0012</td><td></td></tr></table>\n',['table']),
 ('P12','\\label{a} testo \\ref{a}\n',['paragraph']),
 ('P13','```{=html}\n<p>Testo <span id="s">leggibile</span></p>\n```\n',['paragraph']),
])
def test_reader_profile_positive_cases(case,raw,types):
    result=observe(raw.encode());assert [n['type'] for n in result['nodes']]==types,result
    if case=='P02':assert result['nodes'][0]['cells']==[['A','B','C'],['x|y','0','']]
    if case=='P05':assert not result['nodes'][0]['formulas'][0]['math']['fully_supported']
    if case=='P06':assert result['nodes'][0]['anchors']==['id']
    if case=='P07':assert result['nodes'][0]['text']=='Testo' and 'span' in result['nodes'][0]['anchors']
    if case=='P08':assert result['nodes'][1]['text']=='Prima riga\nseconda riga'
    if case=='P10':assert result['nodes'][0]['text']=='$x_i$ `literal` \\ref{a} −7\n' and not result['nodes'][0]['formulas']


@pytest.mark.parametrize('raw',['<table><tr><td colspan="2">x</td></tr></table>\n','```{=other}\nx\n```\n','\\unknown{significant}\n','```\nunclosed\n',('::: {#a}\n'*17)+'x\n'])
def test_P03_P12_P13_P14_unknown_not_silent(raw):
    result=observe(raw.encode());assert result['unknown']>0
    assert any(n['raw'] or n.get('reason') for n in result['nodes'] if n['type']=='unknown')


@pytest.mark.parametrize('value',['../outside','/absolute','x/../y','x%2fy','x%252fy','x\\y','C:drive','x//y'])
def test_unsafe_path_before_open(value):
    with pytest.raises(Unsafe):relative(value)


def test_symlink_hardlink_zip_and_active_markup_are_inert(tmp_path):
    sentinel=tmp_path/'sentinel.txt';sentinel.write_text('synthetic sentinel');root=tmp_path/'raw';root.mkdir();(root/'document.md').write_text('![x](evil.png)\n');(root/'evil.png').symlink_to(sentinel)
    before=sentinel.read_bytes();result=audit(root);assert result['reason']=='UNSAFE_REFERENCE';assert sentinel.read_bytes()==before
    inv=tree_inventory(root);assert next(x for x in inv if x['path']=='evil.png')['kind']=='symlink'
    (root/'evil.png').unlink();os.link(sentinel,root/'evil.png');assert audit(root)['reason']=='UNSAFE_REFERENCE';assert sentinel.read_bytes()==before
    z=io.BytesIO()
    with zipfile.ZipFile(z,'w') as a:a.writestr('../escape','x')
    with pytest.raises(Unsafe):zip_members(z.getvalue())
    with pytest.raises(Unsafe):html('<script>fetch("https://example.invalid")</script>')
    with pytest.raises(Unsafe):html('<img src="https://example.invalid/x.png"/>')
    with pytest.raises(Unsafe):svg(b'<svg><image href="../outside"/></svg>')


def test_missing_media_empty_and_unsafe_bundle(tmp_path):
    root=tmp_path/'raw';root.mkdir();(root/'document.md').write_text('![Caption](media/missing.png)\n');assert audit(root)['reason']=='MISSING_RESOURCE'
    (root/'document.md').write_text(' \n');assert audit(root)['reason']=='EMPTY'
    (root/'document.md').write_text('<p onclick="x()">unsafe</p>\n');assert audit(root)['reason']=='UNSAFE_REFERENCE'


def test_include_positive_and_cycle_escape_are_checked(tmp_path):
    root=tmp_path/'source';(root/'sections').mkdir(parents=True);(root/'main.tex').write_text('A\\input{sections/one.tex}B\\include{sections/two.tex}C');(root/'sections/one.tex').write_text('Uno');(root/'sections/two.tex').write_text('Due')
    expanded,rec=expand_tex(root,'main.tex');assert expanded==b'AUnoBDueC' and rec['includes']==['sections/one.tex','sections/two.tex']
    (root/'sections/one.tex').write_text('\\input{sections/one.tex}')
    with pytest.raises(Unsafe):expand_tex(root,'main.tex')
    (root/'main.tex').write_text('\\input{../outside}')
    with pytest.raises(Unsafe):expand_tex(root,'main.tex')


def test_rtf_scoped_unicode_fallback_independent_M():
    raw=br'{\rtf1\uc1 A\u233? {\uc2\u233??} \u233?}'
    assert rtf_source(raw)=='Aé é é'


def test_exit_precedence_batch_and_coverage_not_summary_success():
    assert evaluate_exit(0,5)==5 and evaluate_exit(0,3)==3 and evaluate_exit(2,3)==2 and evaluate_exit(2,4)==4
    r=batch([{'attempt_id':'one','conversion':'SUCCESS','bundle':'COMPLETE','quality':'INDETERMINATE'},{'attempt_id':'two','conversion':'ERROR','bundle':'UNAVAILABLE','quality':'FAIL'}]);assert r['conversion_batch']=='PARTIAL' and r['errors']==['two'] and r['quality']=='FAIL'


def test_profile_complete_bitset_and_auto_identifiers_off():
    cfg=config('docx','pandoc-direct-v1');assert len(cfg['enabled'])==22 and set(cfg['enabled'])==ENABLED
    assert not set(cfg['enabled'])&set(cfg['disabled']);assert {'smart','multiline_tables','simple_tables','grid_tables','auto_identifiers'}<=set(cfg['disabled'])
    for name in cfg['disabled']:assert '-'+name in cfg['writer']


def test_timeout_partial_and_setsid_child_reaped(tmp_path):
    stub=tmp_path/'stub.py';stub.write_text('import os,time\nfrom pathlib import Path\nPath("partial.md").write_text("partial original")\np=os.fork()\nif p==0:\n os.setsid()\n time.sleep(20)\nelse:\n time.sleep(20)\n')
    out=tmp_path/'receipt';r=run([sys.executable,'-I','-B',str(stub)],tmp_path,out,{'PATH':'/usr/bin'},0.3)
    assert r['timeout'] and r['stop_reason']=='TIMEOUT' and r['children_reaped'];assert (tmp_path/'partial.md').read_text()=='partial original'
    assert r['orphan_children_wait4'] and r['sampled_tree_peak_rss_bytes']>0


def test_crash_raw_preserved_and_normal_child_positive(tmp_path):
    stub=tmp_path/'crash.py';stub.write_text('import sys\nfrom pathlib import Path\nPath("partial.md").write_text("preserved")\nsys.exit(7)\n')
    r=run([sys.executable,'-I','-B',str(stub)],tmp_path,tmp_path/'crash-result',{'PATH':'/usr/bin'},2)
    assert r['adapter_exit']==7 and r['children_reaped'];assert (tmp_path/'partial.md').read_text()=='preserved'
    r=run([sys.executable,'-I','-B','-c','print("positive")'],tmp_path,tmp_path/'positive',{'PATH':'/usr/bin'},2)
    assert r['adapter_exit']==0 and r['stop_reason'] is None and r['children_reaped']


def test_packet_pending_response_and_stale_hash_no_human_simulation(tmp_path):
    # Synthetic protocol fixture, independent of operational temp/. Empty
    # response is a negative stub input, not a human signature.
    packet=tmp_path/'synthetic-packet';packet.mkdir();(packet/'canonical.txt').write_text('Synthetic canonical\n');(packet/'gold.json').write_text('{}\n')
    judgments=[{'document_id':'stub-doc','judgment_id':'stub-doc:b1:content'}];(packet/'judgments.json').write_bytes(canonical(judgments))
    identity={'document_id':'stub-doc','canonical_sha256':digest_bytes((packet/'canonical.txt').read_bytes()),'gold_sha256':digest_bytes((packet/'gold.json').read_bytes())}
    metadata={'schema_version':1,'kind':'explicit unit stub; no operational gold or human signature','identities':[identity],'files':[r for r in tree_inventory(packet) if r['kind']=='regular']};metadata['packet_sha256']=digest(metadata);(packet/'packet.json').write_bytes(canonical(metadata))
    before=verify_packet(packet)['packet_sha256'];response=tmp_path/'response.tsv';response.write_text('document_id\tjudgment_id\tdecision\treason\tverifier\tverified_at\nstub-doc\tstub-doc:b1:content\t\t\t\t\n')
    assert import_response(packet,response,before,tmp_path/'pending')==5;assert load(tmp_path/'pending/response.json')['status']=='PENDING'
    assert import_response(packet,response,'0'*64,tmp_path/'stale')==5;assert load(tmp_path/'stale/response.json')['status']=='STALE'
    assert verify_packet(packet)['packet_sha256']==before
    (packet/'canonical.txt').write_text('Tampered synthetic canonical\n')
    with pytest.raises(ValueError,match='STALE'):verify_packet(packet)


def test_json_contract_guard_duplicate_keys_symlink_and_oversize(tmp_path):
    p=tmp_path/'invalid.json';p.write_text('{"schema_version":1,"schema_version":2}')
    with pytest.raises(ValueError):load(p)
    linked=tmp_path/'linked.json';linked.symlink_to(p)
    with pytest.raises(Unsafe):load(linked)
    # Exercise the bounded reader directly; R itself forbids creating a file
    # above 32MiB (EFBIG), which belongs to the separate H15 contract.
    with pytest.raises(Unsafe):regular(tmp_path,'invalid.json',limit=8)


def test_invalid_output_encoding_has_operational_failure_and_raw_kept(tmp_path):
    from benchmark import observe_bundle
    (tmp_path/'document.md').write_bytes(b'Visible \xff payload\n')
    checked=audit(tmp_path);assert checked['status']=='INCOMPLETE' and checked['reason']=='INVALID_ENCODING'
    observed=observe_bundle(tmp_path,tmp_path/'observations.json')
    assert observed['unknown']==1 and observed['raw_hex']=='56697369626c6520ff207061796c6f61640a'
    assert observed['raw_sha256']==digest_bytes(b'Visible \xff payload\n')


def test_monitor_interrupt_kills_and_reaps_before_return(tmp_path):
    environment={'PATH':'/usr/bin','LANG':'C.UTF-8'}
    stub="import os,signal,time;os.kill(os.getppid(),signal.SIGINT);time.sleep(20)"
    result=run([sys.executable,'-I','-B','-c',stub],tmp_path,tmp_path/'interrupted',environment,5)
    assert result['interrupted'] and result['stop_reason']=='INTERRUPTED' and result['children_reaped']
    assert not result['timeout'] and result['conversion_elapsed_s']<4


def test_base_lists_quotes_link_and_code_attributes_keep_literal_offsets():
    raw='- Prima −7\n- Seconda 0012\n\n> Quota Caffè\n> continua.\n\n[Link](https://example.invalid){#link .class} `−7`{#code}\n'
    result=observe(raw.encode());assert result['unknown']==0
    assert [n['text'] for n in result['nodes']]==['Prima −7','Seconda 0012','Quota Caffè\ncontinua.','Link −7']
    assert [n.get('block_syntax') for n in result['nodes'][:3]]==['list_item','list_item','blockquote']
    assert result['nodes'][-1]['anchors']==['link','code']
    assert result['nodes'][-1]['references'][0]['attrs']=={'id':'link','classes':['class']}
    for n in result['nodes']:assert raw.encode()[n['offset']['byte_start']:n['offset']['byte_end']].decode()==n['raw']


@pytest.mark.parametrize('raw',['$x_i','$$E=mc^2','> > Nested quote'])
def test_ambiguous_math_and_complex_quote_stay_unknown(raw):
    result=observe((raw+'\n').encode());assert result['unknown'] and result['nodes'][0]['raw']==raw+'\n'


def test_resource_post_exit_stream_guard_detects_fast_output(tmp_path):
    receipt=run([sys.executable,'-I','-B','-c','import sys;sys.stdout.write("x"*4096)'],tmp_path,tmp_path/'fast-stream',{'PATH':'/usr/bin'},5,stream_cap=1024)
    assert receipt['stop_reason']=='STREAM_LIMIT' and receipt['children_reaped']
    assert (tmp_path/'fast-stream/stdout.bin').stat().st_size>1024


def test_missing_text_block_keeps_full_word_and_character_denominators():
    control=CORPUS/'controls/K02-three-sections';gold=load(control/'annotations.json');raw=regular(control,'document.md')
    complete=score(gold,observe(raw),{'status':'COMPLETE'})
    # Entire page content removal must not improve the WER reference count.
    nodes=observe(raw)['nodes'];page2=nodes[4:8];start=page2[0]['offset']['byte_start'];end=page2[-1]['offset']['byte_end']
    partial=score(gold,observe(raw[:start]+raw[end:]),{'status':'COMPLETE'})
    assert partial['totals']['word_ref']==complete['totals']['word_ref']
    assert partial['totals']['text_ref_codepoints']==complete['totals']['text_ref_codepoints']
    assert partial['totals']['word_errors']>0 and partial['score_exit']==3


def test_campaign_frozen_manifest_detects_data_and_schema_tampering(tmp_path):
    from campaign import freeze,manifest,PATHS
    corpus=tmp_path/'corpus';stage=tmp_path/'stage';corpus.mkdir();stage.mkdir();documents=[]
    for ident,format in [('C01-txt-it','txt'),('C02-docx-it','docx'),('C03-epub-en','epub'),('C04-rtf-it','rtf'),('C05-tex-en','tex')]:
        (corpus/(ident+'.input')).write_text('Independent administrative fixture '+ident)
        (corpus/(ident+'.gold')).write_text('{"blocks":[]}')
        documents.append({'document_id':ident,'format':format,'input':ident+'.input','input_sha256':descriptor(corpus/(ident+'.input'))['sha256'],'gold':ident+'.gold'})
    write_new(corpus/'manifest.json',{'documents':documents});write_new(stage/'target.json',{'environment':{'PATH':'/usr/bin'}});write_new(stage/'I.json',{'test_only':'no installed product claim'})
    inventory={'schema_version':1,'scope':'benchmark-inputs','files':[]};write_new(stage/'benchmark-inputs.json',dict(inventory,digest=digest(inventory)))
    result=freeze(corpus,stage,tmp_path/'frozen','unit-only');path=tmp_path/'frozen/manifest.json'
    assert len(result['groups'])==8 and sum(len(g['attempts']) for g in result['groups'])==32
    assert manifest(path)['expected']['R_expected_exit']==0
    source=corpus/'C01-txt-it.input';original=source.read_bytes();source.write_bytes(original+b' tampered')
    with pytest.raises(ValueError,match='STALE'):manifest(path)
    source.write_bytes(original)
    bad=tmp_path/'bad.json';write_new(bad,dict(result,schema_version=99))
    with pytest.raises(ValueError,match='schema'):manifest(bad)


def test_R_monitor_post_exit_guard_cannot_accept_fast_oversized_stream(tmp_path):
    import importlib.util
    source=ROOT/'scripts/diagnostics/run-a001-fase0-uv/run_offline.py'
    spec=importlib.util.spec_from_file_location('a002_offline_monitor',source);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    result=module.monitored_command([sys.executable,'-I','-B','-c','import sys;sys.stdout.write("x"*1048577)'],tmp_path,tmp_path,5)
    assert result['timed_out'] and result['guard_reason']=='STREAM_LIMIT'
    assert (tmp_path/'command-stdout.txt').stat().st_size>1048576


def test_TeX_source_validator_consumes_row_separator_as_control_symbol():
    from adapters import tex_validate
    assert tex_validate(r'\begin{aligned}a&=1\\b&=-2\end{aligned}'.encode())==[]
    assert tex_validate(r'\begin{bmatrix}0&-1\\2&3\end{bmatrix}'.encode())==[]
    assert tex_validate(r'\begin{tabular}{ll}A&B\\Cell&−7\end{tabular}'.encode())==[]
    with pytest.raises(Unsafe):tex_validate(r'\write18{outside}'.encode())
    with pytest.raises(Unsafe):tex_validate(r'\def\x{dynamic}'.encode())


def test_note_identity_mapping_uses_note_id_without_requiring_anchor():
    control=CORPUS/'controls/K01-full';gold=load(control/'annotations.json');raw=regular(control,'document.md').replace(b'n1',b'9')
    for block in gold['blocks']:
        if block['type']=='note':block['note_id']=block.pop('anchor')
    checked=audit(control)
    for image in checked['assets']:image.update(content_verified=True,expected_asset='figure-it.png')
    measured=score(gold,observe(raw),checked)
    assert measured['automatic_gates']=='PASS' and measured['score_exit']==5 and not measured['failures']
    duplicated=observe(raw+b'\n[^9]: Another explicit definition.\n')
    after=score(gold,duplicated,checked)
    assert after['score_exit']==3 and any(f.get('reason')=='NON_BIJECTIVE_OR_DUPLICATE_ID' for f in after['failures'])


def test_M_V_combination_never_verifies_pending_stale_or_rejected():
    from verification import combine
    identity={'document_id':'unit-only','canonical_sha256':'a'*64,'gold_sha256':'b'*64};M=dict(identity,status='PASS');V=dict(identity,status='PENDING')
    assert combine(identity,M,V,'PENDING')['status']=='PENDING'
    assert combine(identity,dict(M,gold_sha256='c'*64),V,'PENDING')['status']=='STALE'
    assert combine(identity,dict(M,status='FAIL'),V,'PENDING')['status']=='REJECTED'
    assert combine(identity,M,V,'AMBIGUOUS')['status']=='AMBIGUOUS'


def test_same_source_basename_rerun_id_and_existing_output_cannot_overwrite(tmp_path):
    from benchmark import run_attempt
    sources=tmp_path/'corpus';(sources/'a').mkdir(parents=True);(sources/'b').mkdir()
    for folder,text in [('a','Original A −7\n'),('b','Original B −8\n')]:
        (sources/folder/'identical.txt').write_text(text)
    installation=tmp_path/'unit-installation.json'
    # TXT child uses no product import. This descriptor is explicit unit data,
    # never an I/R receipt or proof of a real installed wheel.
    write_new(installation,{'unit_fixture_only':True,'payload':{'installed':{'canonical_archive':{'path':'UNIT_ONLY_NOT_A_WHEEL','sha256':'0'*64}}}})
    cfg=config('txt','txt-literal-v1');env={'PATH':'/usr/bin','LANG':'C.UTF-8','HOME':str(tmp_path),'TMPDIR':str(tmp_path)}
    a={'document_id':'unit-doc-a','format':'txt','input':'a/identical.txt','input_sha256':descriptor(sources/'a/identical.txt')['sha256']}
    b={'document_id':'unit-doc-b','format':'txt','input':'b/identical.txt','input_sha256':descriptor(sources/'b/identical.txt')['sha256']}
    first=run_attempt(sources,a,'txt-literal-v1',cfg,tmp_path/'attempt-a','unit-experiment','a-r1',env,installation)
    second=run_attempt(sources,b,'txt-literal-v1',cfg,tmp_path/'attempt-b','unit-experiment','b-r1',env,installation)
    rerun=run_attempt(sources,a,'txt-literal-v1',cfg,tmp_path/'attempt-a-r2','unit-experiment','a-r2',env,installation)
    assert first['run_exit']==second['run_exit']==rerun['run_exit']==0
    original=regular(tmp_path/'attempt-a/raw','document.md')
    assert 'Original A −7' in original.decode() and 'Original B −8' in regular(tmp_path/'attempt-b/raw','document.md').decode()
    assert first['source_sha256']!=second['source_sha256'] and first['attempt_id']!=rerun['attempt_id']
    with pytest.raises(FileExistsError):run_attempt(sources,b,'txt-literal-v1',cfg,tmp_path/'attempt-a','unit-experiment','b-r2',env,installation)
    assert regular(tmp_path/'attempt-a/raw','document.md')==original
