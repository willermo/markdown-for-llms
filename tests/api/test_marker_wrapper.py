"""Contratto HTTP mock: nessun modello, font o import ML."""
import importlib.metadata
from pathlib import Path
from types import SimpleNamespace
import pytest
from fastapi.testclient import TestClient
import marker_api_server as server


@pytest.fixture
def client(monkeypatch,tmp_path,record_property):
    monkeypatch.setattr(server.tempfile,'tempdir',str(tmp_path))
    observed=[]
    def convert(path):
        observed.append(Path(path));assert Path(path).read_bytes()==b'synthetic input'
        return SimpleNamespace(markdown='# Risultato\n\n123,45 $E=mc^2$ [1]')
    convert.page_count=2
    monkeypatch.setattr(server,'get_converter',lambda:convert)
    for name in ['fastapi','starlette','anyio','httpx']:
        record_property(name,importlib.metadata.version(name))
    with TestClient(server.app) as c:yield c,observed
    assert all(not p.exists() for p in observed)


@pytest.mark.parametrize('endpoint',['/convert','/marker'])
def test_success_aliases_and_content(client,endpoint):
    c,seen=client
    r=c.post(endpoint,files={'file':('test.html',b'synthetic input','text/html')})
    assert r.status_code==200
    d=r.json();assert d['markdown']=='# Risultato\n\n123,45 $E=mc^2$ [1]'
    assert d['success'] and d['page_count']==2 and d['images']=={}
    assert seen[-1].suffix=='.html'


def test_metadata_options_are_reported(client):
    c,_=client
    d=c.post('/marker?use_llm=true&force_ocr=true&paginate=true&max_pages=1',
             files={'file':('x.pdf',b'synthetic input')}).json()
    assert d['metadata']['use_llm'] and d['metadata']['force_ocr'] and d['metadata']['paginate']
    assert d['page_count']==2  # max_pages non è applicato dal wrapper legacy.


@pytest.mark.parametrize('error,detail',[(RuntimeError('synthetic failure'),'Conversion failed: synthetic failure'),
                                       (ImportError('synthetic missing'),'Marker library not properly installed')])
def test_error_and_temporary_cleanup(client,monkeypatch,tmp_path,error,detail):
    c,_=client
    def fail(path):raise error
    monkeypatch.setattr(server,'get_converter',lambda:fail)
    r=c.post('/convert',files={'file':('bad.pdf',b'synthetic input')})
    assert r.status_code==500 and r.json()['detail']==detail
    assert not list(tmp_path.iterdir())


def test_health_is_liveness(client):
    c,_=client
    assert c.get('/health').json()=={'status':'healthy','service':'marker-api'}


def test_startup_failure_does_not_change_health(monkeypatch):
    def fail():raise RuntimeError('synthetic startup failure')
    monkeypatch.setattr(server,'get_converter',fail)
    with TestClient(server.app) as c:assert c.get('/health').status_code==200
