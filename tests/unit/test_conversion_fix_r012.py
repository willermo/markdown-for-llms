"""F50–F53: finite independent manual oracles frozen before workloads.
Generator stdlib in run work, seed 12012; no product helper builds expected.
"""
from pathlib import Path
import json,sys,shutil
import pytest
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/benchmarks'))
import observations as reader
import audit as auditor
from benchmark import observe_bundle
from metrics import score
CASES=json.loads((ROOT/'tests/fixtures/conversion-reader-r012.json').read_text())['cases']
# r013 corrects only the 20 EOF expectations contradicted by the pinned AST.
# Keep the 350 original raw/gold fixture bytes as historical evidence.
R013=json.loads((ROOT/'tests/fixtures/conversion-reader-r013-legacy-expectations.json').read_text())['cases']
CASES=[{**c,**R013.get(c['id'],{})} for c in CASES]
ASSET=ROOT/'tests/fixtures/conversion-corpus/assets/figure-it.png'

def pipeline(path,c):
    path.mkdir();(path/'document.md').write_text(c['raw']);shutil.copyfile(ASSET,path/'p.png')
    a=auditor.audit(path);o=observe_bundle(path,path/'observations.json');m=score(c['gold'],o,a,'txt')
    return o,a,m

def checks(c,o,a,m,visits):
    result={'reason':a['reason']==c['reason'],'raw':o['raw_sha256']==a['raw_sha256'],
            'detail':c['detail'] is None or c['detail'] in a.get('detail',''),
            'visits':all(t in visits for t in c['visits']),
            'unsafe_inert':c['reason']=='UNSAFE_REFERENCE' or not any(t=='script' for t in visits)}
    if c['kind']=='formula':
        nodes=o['nodes'];f=[f for n in nodes for f in n['formulas']];refs=[r for n in nodes for r in n['references']]
        result.update(formula=len(f)==1 and f[0]['raw']==c['formula'],no_assets=not a['assets'],
                      blocks=[n['type'] for n in nodes]==['note' if c['host'].startswith('note') else 'paragraph','heading','paragraph'],
                      text=[n['text'] for n in nodes]==[c['text'],'Visible heading','After words.'],
                      formula_position=bool(f) and f[0]['source_position']['char']==c['positions']['formula']['char'] and f[0]['source_position']['byte']==c['positions']['formula']['byte'],
                      link_position=bool(refs) and refs[-1]['source_position']['char']==c['positions']['link']['char'] and refs[-1]['source_position']['byte']==c['positions']['link']['byte'],
                      references=[r['target'] for r in refs]==['https://example.invalid'],
                      lexical_not_semantic=m['score_exit']==5 and m['quality']=='INDETERMINATE')
    if c['kind']=='block' and c['reason']=='UNSUPPORTED_RESOURCE_SYNTAX':
        result['resource_residual']=bool(a.get('unknown')) and not a['assets']
        u=[u for n in o['nodes'] for u in n.get('unknown',[]) if u.get('reason')=='container raw block not projected']
        result['raw_position']=bool(u) and u[0]['source_position']['char']==c['raw'].index(c['logical'].split('\n')[0])
    if c['kind']=='token':result['no_borrowed_formula']=not any(n['formulas'] for n in o['nodes'])
    return result

@pytest.mark.parametrize('c',CASES,ids=lambda c:c['id'])
def test_F50_F53_finite_classes(tmp_path,monkeypatch,c):
    visits=[];original=reader.PassiveHTML.handle_starttag
    def spy(self,tag,attrs):visits.append(tag);return original(self,tag,attrs)
    monkeypatch.setattr(reader.PassiveHTML,'handle_starttag',spy)
    o,a,m=pipeline(tmp_path/'case',c)
    result=checks(c,o,a,m,visits)
    assert all(result.values()),{k:v for k,v in result.items() if not v}

@pytest.mark.parametrize('n',[4,16,64])
def test_F53_monotone_LF_ownership(n):
    raw='È [Known '+('`a\nb` \\\nc '*n)+'$x <script>x</script> y$ words](https://example.invalid)'
    decision=reader.InlineDecision(raw)
    parsed=reader.inline(raw,_decision=decision)
    assert parsed['formulas'][0]['raw']=='x <script>x</script> y'
    assert parsed['references'][0]['target']=='https://example.invalid'
    assert decision.coverage==(0,len(raw))

@pytest.mark.parametrize('host',['quote','list','note-tab'])
@pytest.mark.parametrize('literal',[False,True])
def test_F51_nested_quote_fence_has_owner(tmp_path,host,literal):
    value='> ```'+('' if literal else '{=html}')+'\n> <script>x</script>\n> ```'
    raw=('> '+value.replace('\n','\n> ') if host=='quote' else '- '+value.replace('\n','\n    ') if host=='list' else '[^n]: '+value.replace('\n','\n\t'))
    (tmp_path/'document.md').write_text(raw)
    a=auditor.audit(tmp_path)
    assert a['reason']==(None if literal else 'UNSAFE_REFERENCE')
    if not literal:assert a['detail']=='HTML attivo/sconosciuto: script'


def test_F53_metadata_LF_does_not_expand_label_acceptance():
    assert reader.markdown_link('[`x`{class="a\nb"}](https://example.invalid)',0) is None


def test_F53_shared_decision_preserves_mapping_in_recursion():
    raw='È [Known $x \\\ny$ words](https://example.invalid)'
    mapping=list(range(10,10+len(raw)+1));d=reader.InlineDecision(raw,source_map=mapping)
    link=d.atoms[raw.index('[')]['link'];sub=d.slice(link['label_start'],link['label_end'])
    formula=next(a for a in sub.atoms.values() if a['kind']=='math')
    assert sub.source_map[formula['start']]==10+raw.index('$')
    assert sub.raw[formula['body_start']:formula['body_end']]=='x \\\ny'
