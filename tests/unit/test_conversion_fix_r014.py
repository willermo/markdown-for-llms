"""Contratti r014: attesi manuali e premesse Pandoc pinned, nessun gold prodotto."""
from pathlib import Path
import json,shutil,sys
import pytest
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/benchmarks'))
import observations as reader
import audit as auditor
from safety import local_uri,Unsafe
CASES=json.loads((ROOT/'tests/fixtures/conversion-reader-r014.json').read_text())['cases']

@pytest.mark.parametrize('case',CASES,ids=lambda c:c['id'])
def test_syntax_and_source_validation(tmp_path,monkeypatch,case):
    raw=case['raw'];visits=[];events=[];rejections=[]
    validate=reader.PassiveHTML.handle_starttag;drain=reader.drain_guards;report=reader.report_unsafe
    def spy(self,tag,attrs):
        event=reader._GUARD.get()['current'];at=event['source_start']
        assert raw[at:at+len(event['snippet'])]==event['snippet']
        visits.append(dict(snippet=event['snippet'],char=at,byte=len(raw[:at].encode())))
        return validate(self,tag,attrs)
    def collect(session):
        events.extend(sorted(session['events'].values(),key=lambda e:e['source_start']))
        return drain(session)
    def rejected(exc):
        current=reader._GUARD.get().get('current')
        if current:rejections.append(current)
        return report(exc)
    with monkeypatch.context() as m:
        m.setattr(reader.PassiveHTML,'handle_starttag',spy);m.setattr(reader,'drain_guards',collect);m.setattr(reader,'report_unsafe',rejected)
        if case['detail']:
            with pytest.raises(Unsafe) as exc:reader.observe(raw.encode())
            assert case['detail'] in str(exc.value)
        else:
            observation=reader.observe(raw.encode())
            text='\n'.join(n['text'] for n in observation['nodes'])
            for value in case['contains']:assert value in text
            def private(value):
                if isinstance(value,dict):
                    assert not (isinstance(value.get('_guard_coverage'),dict) and value['_guard_coverage'].get('domain')=='producer')
                    for v in value.values():private(v)
                elif isinstance(value,list):
                    for v in value:private(v)
            private(observation)
    assert visits==[{k:e[k] for k in ('snippet','char','byte')} for e in case['events'] if e['kind']=='html']
    if not case['html_events_only']:
        assert [(e['kind'],e['snippet'],e['source_start']) for e in events]==[(e['kind'],e['snippet'],e['char']) for e in case['events']]
        # Metadata events must actually reach rejection, not merely be queued.
        assert [e for e in rejections if e['kind']=='markdown-attribute']==[e for e in events if e['kind']=='markdown-attribute']
    assert reader._GUARD.get() is None
    (tmp_path/'document.md').write_bytes(raw.encode())
    shutil.copyfile(ROOT/'tests/fixtures/conversion-corpus/assets/figure-it.png',tmp_path/'p.png')
    audit=auditor.audit(tmp_path)
    if case['detail']:assert (audit['status'],audit['reason'])==('INCOMPLETE','UNSAFE_REFERENCE')
    else:
        for asset in case['assets']:
            assert any(v.get('uri')==asset for v in audit['assets']) or (audit['status']=='INCOMPLETE' and any(v.get('raw') and v.get('resource_syntax') and v.get('source_position') for v in audit.get('unknown',[])))

def test_uri_parser_refusals_are_contractual():
    # Every ValueError source in pinned urlsplit, including bracket prefix,
    # suffix and IPvFuture. No broad observe exception handler is involved.
    for uri in ('http://[','http://]','http://before[::1]/','http://[::1]tail/','http://[vbad]/','http://[not-ip]/','http://[127.0.0.1]/','http://exa／mple/'):
        for allow in (True,False):
            with pytest.raises(Unsafe,match='URI malformata'):local_uri(uri,allow)

def test_uri_policy_unchanged():
    for uri in ('#section','http://example.invalid/a','https://[::1]/','mailto:a@example.invalid'):
        assert local_uri(uri) is None
    assert local_uri('assets/p.png?x=1#part')=='assets/p.png'
    for uri in ('../p.png','%2e%2e/p.png','a\\b','file:///tmp/a','javascript:x','//example.invalid/a','https://user:pass@example.invalid/a','http%3A//example.invalid/a',''):
        with pytest.raises(Unsafe):local_uri(uri)
    with pytest.raises(Unsafe):reader.PassiveHTML().handle_starttag('img',[('src',None)])
