"""F54/F55: source-ordered guard visits and independently frozen ownership."""
from pathlib import Path
import json,sys,shutil
import pytest
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/benchmarks'))
import observations as reader
import audit as auditor
CASES=json.loads((ROOT/'tests/fixtures/conversion-reader-r013.json').read_text())['cases']

@pytest.mark.parametrize('case',CASES,ids=lambda c:c['id'])
def test_guard_sources_and_fence_ownership(tmp_path,monkeypatch,case):
    visits=[];original=reader.PassiveHTML.handle_starttag;raw=case['raw'];observed=None
    def spy(self,tag,attrs):
        event=reader._GUARD.get()['current'];at=event['source_start']
        visits.append(dict(snippet=event['snippet'],char=at,byte=len(raw[:at].encode())))
        # Source identity comes from fixture bytes, not merely the ledger.
        assert raw[at:at+len(event['snippet'])]==event['snippet']
        return original(self,tag,attrs)
    with monkeypatch.context() as m:
        m.setattr(reader.PassiveHTML,'handle_starttag',spy)
        if case['detail']:
            with pytest.raises(reader.Unsafe) as exc:reader.observe(raw.encode())
            assert case['detail'] in str(exc.value)
        else:observed=reader.observe(raw.encode())
    assert visits==case['visits']
    if observed:assert '_guard_coverage' not in json.dumps(observed)
    if case.get('asset'):
        (tmp_path/'document.md').write_text(raw)
        shutil.copyfile(ROOT/'tests/fixtures/conversion-corpus/assets/figure-it.png',tmp_path/'p.png')
        audit=auditor.audit(tmp_path)
        assert any(v.get('uri')=='p.png' for v in audit['assets']) or (
            audit['status']=='INCOMPLETE' and audit['reason']=='UNSUPPORTED_RESOURCE_SYNTAX'
            and any(v.get('resource_syntax') and v.get('raw') and v.get('source_position') for v in audit.get('unknown',[])))

@pytest.mark.parametrize('bad',[b'\xff',b'[a]{onclick="x"}',b'<script>x</script>'])
def test_guard_session_resets_after_error(bad):
    with pytest.raises((UnicodeError,reader.Unsafe)):reader.observe(bad)
    assert reader._GUARD.get() is None
    result=reader.observe(b'# Safe\n\nText.\n')
    assert [n['text'] for n in result['nodes']]==['Safe','Text.']

def test_private_coverage_removed_only_after_consumption():
    result=reader.observe(b'| h |\n| --- |\n| [c](https://example.invalid) |\n\n: caption\n')
    assert '_guard_coverage' not in json.dumps(result)
    cell=result['nodes'][0]['cell_contents'][1][0]
    assert cell['references'][0]['target']=='https://example.invalid'
    assert cell['location']=={'kind':'cell','row':1,'col':0}


@pytest.mark.parametrize('raw',[b'```{=html}',b'~~~{=html}',b'```{=latex}'])
def test_empty_raw_eof_retains_explicit_residual(raw):
    result=reader.observe(raw)
    assert result['unknown'] and any(u.get('resource_syntax') for n in result['nodes'] for u in n.get('unknown',[]))


def test_reader_audit_order_limit_stays_explicit(tmp_path):
    (tmp_path/'document.md').write_text('![a](../first.png)\n\nText <img src="../second.png">')
    result=auditor.audit(tmp_path)
    assert result['reason']=='UNSAFE_REFERENCE' and '../second.png' in result['detail']


@pytest.mark.parametrize('case',json.loads((ROOT/'tests/fixtures/conversion-reader-r013-metadata.json').read_text()),ids=lambda c:c['id'])
def test_multiline_attributes_keep_source_positions(monkeypatch,case):
    events=[];original=reader.drain_guards
    def drain(session):
        events.extend(sorted(session['events'].values(),key=lambda e:e['source_start']))
        return original(session)
    monkeypatch.setattr(reader,'drain_guards',drain)
    with pytest.raises(reader.Unsafe,match=case['detail']):reader.observe(case['raw'].encode())
    metadata=[e for e in events if e['kind']=='markdown-attribute']
    assert metadata==[{'kind':'markdown-attribute','source_start':case['attr_char'],'snippet':case['snippet']}]
    assert len(case['raw'][:metadata[0]['source_start']].encode())==case['attr_byte']
    assert [(e['snippet'],e['source_start']) for e in events if e['kind']=='html']==[('<script>',case['raw'].index('<script>'))]
