"""Wheel esterna e fedeltà dopo R e S/B/I; nessuna build/sync/installazione."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import pytest

ROOT=Path(__file__).resolve().parents[2]
FIXTURES=ROOT/'tests/fixtures/run-a001-fase0-uv'
MODULES=['config','logging_config','exceptions','unified_converter','master_workflow','clean_markdown',
         'validate_markdown','chunk_markdown','batch_monitor','marker_api_server']


def compare_validation_report(actual, reference):
    """Confronta ogni campo; glob non garantisce l'ordine dei documenti nel report."""
    assert set(actual) == set(reference) == {'summary', 'detailed_results'}
    assert actual['summary'] == reference['summary']
    def documents(report):
        items = report['detailed_results']
        keyed = {item['filename']: item for item in items}
        assert len(keyed) == len(items), 'Documento duplicato nel report'
        return keyed
    assert documents(actual) == documents(reference)


def required(name):
    value=os.environ.get(name)
    assert value, 'Preparazione mancante: '+name
    path=Path(value)
    assert path.is_absolute() and path.exists(), name+': path assoluto esistente richiesto'
    return path


def preflight(python,receipt):
    result=subprocess.run([str(python),'-I','-B',str(ROOT/'scripts/diagnostics/run-a001-fase0-uv/check_preflight.py'),
                           '--repo',str(ROOT),'--source',str(required('RUN_SOURCE_MANIFEST')),
                           '--receipt',str(receipt)],capture_output=True,text=True,timeout=60,close_fds=True)
    assert result.returncode==0,result.stderr


@pytest.fixture
def installed(tmp_path):
    python=required('RUN_WHEEL_PY');receipt=required('RUN_WHEEL_RECEIPT')
    preflight(python,receipt)
    baseline=json.loads(required('RUN_BASELINE').read_text())
    assert baseline['status']=='PASS' and baseline['scope']=='baseline-originals'
    base=required('RUN_BASELINE_WORKSPACES')
    for item in baseline['workspaces']:
        path=base/item['path'];assert not path.is_symlink()
        b=path.read_bytes();assert len(b)==item['bytes'] and hashlib.sha256(b).hexdigest()==item['sha256']
    shutil.copyfile(FIXTURES/'fixture-config.json',tmp_path/'pipeline_config.json')
    for name in ['source_documents','converted_markdown','cleaned_markdown','validated_markdown','chunked_markdown']:
        (tmp_path/name).mkdir()
    yield python,receipt,base,tmp_path
    preflight(python,receipt)


def call(installed,*args,env=None,mode='module',expected=0):
    python,receipt,_,workspace=installed
    preflight(python,receipt)
    environment=dict(os.environ);environment.pop('PYTHONHOME',None);environment.pop('PYTHONPATH',None)
    if env:environment.update(env)
    if mode=='module':argv=[str(python),'-I','-B','-m',*args]
    elif mode=='console':argv=[str(python.parent/args[0]),*args[1:]]
    else:argv=[str(python),'-B',str(ROOT/args[0]),*args[1:]]
    p=subprocess.run(argv,cwd=workspace,env=environment,capture_output=True,text=True,timeout=120,close_fds=True)
    assert p.returncode==expected,p.stdout+p.stderr
    return p


def test_origins_and_transitives(installed):
    python,_,_,ws=installed
    code="import importlib,importlib.util,json;names="+repr(MODULES)+";out={n:importlib.util.find_spec(n).origin for n in names};out.update({n:importlib.import_module(n).__file__ for n in names[:-1]+['requests','dotenv','tqdm','tiktoken']});print(json.dumps(out))"
    r=subprocess.run([str(python),'-I','-B','-c',code],cwd=ws,capture_output=True,text=True,timeout=30,close_fds=True)
    assert r.returncode==0,r.stderr
    origins=json.loads(r.stdout)
    assert len(origins)==14 and all(Path(p).is_relative_to(python.parent.parent) and 'site-packages' in p for p in origins.values())


def test_config_and_help(installed):
    for name in ['markdown-pipeline','markdown-config','markdown-chunk']:
        assert '--help' in call(installed,name,'--help',mode='console').stdout
    call(installed,'config','--create-default')
    expected=json.loads((installed[3]/'pipeline_config.json').read_text())
    assert json.loads(call(installed,'markdown-config','--show',mode='console').stdout)==expected
    assert json.loads(call(installed,'config.py','--show',mode='clone').stdout)==expected


@pytest.mark.parametrize('mode',['module','console','clone'])
def test_clean_validation_and_sentinels(installed,mode):
    _,_,base,ws=installed
    for name in MODULES+['requests','dotenv','tqdm','tiktoken']:
        (ws/(name+'.py')).write_text("from pathlib import Path\nPath('sentinel-"+name+"').write_text('executed')\nraise RuntimeError('sentinel executed')\n")
    for name in ['f1-stable.md','f2-sensitive.md','f5-asset.md','f6-multichunk.md']:
        shutil.copyfile(FIXTURES/name,ws/'converted_markdown'/name)
    (ws/'converted_markdown/assets').mkdir();shutil.copyfile(FIXTURES/'assets/f5.png',ws/'converted_markdown/assets/f5.png')
    target={'module':'master_workflow','console':'markdown-pipeline','clone':'master_workflow.py'}[mode]
    for phase in ['cleaning','validation','chunking']:
        call(installed,target,'--step',phase,'--force','--llm','custom','--chunk-size','1000','--overlap','100',
             mode=mode,env={'PYTHONPATH':str(ws)} if mode=='module' else None)
    for p in sorted((base/'cleaning/cleaned_markdown').glob('*.md')):
        assert (ws/'cleaned_markdown'/p.name).read_bytes()==p.read_bytes()
    for p in sorted((base/'validation-chain/validated_markdown').glob('*.md')):
        assert (ws/'validated_markdown'/p.name).read_bytes()==p.read_bytes()
    compare_validation_report(json.loads((ws/'validation_report.json').read_text()),
                              json.loads((base/'validation-chain/validation_report.json').read_text()))
    assert not (ws/'cleaned_markdown/assets/f5.png').exists()  # Perdita legacy preservata.
    assert not list(ws.glob('sentinel-*'))
    assert hashlib.sha256((ws/'converted_markdown/assets/f5.png').read_bytes()).digest()==hashlib.sha256((FIXTURES/'assets/f5.png').read_bytes()).digest()
    reference=base/'f6-chunk-1000-100/chunked_markdown'
    for p in sorted(reference.glob('f6-multichunk_chunk_*.md')):
        assert (ws/'chunked_markdown'/p.name).read_bytes()==p.read_bytes()


def test_f4_pandoc_content(installed):
    _,_,base,ws=installed
    shutil.copyfile(FIXTURES/'f4-pandoc.html',ws/'source_documents/f4-pandoc.html')
    call(installed,'master_workflow','--step','conversion','--force')
    assert (ws/'converted_markdown/f4-pandoc.md').read_bytes()==(base/'f4-conversion/converted_markdown/f4-pandoc.md').read_bytes()
    summary=json.loads((ws/'converted_markdown/unified_conversion_summary.json').read_text())
    assert summary['conversion_summary']['failed']==0


@pytest.mark.parametrize('size,overlap,shell',[(1200,80,False),(1600,120,True),(1000,100,False)])
def test_chunks_dotenv_and_metadata(installed,size,overlap,shell):
    _,_,base,ws=installed
    shutil.copyfile(FIXTURES/'f6-multichunk.md',ws/'validated_markdown/f6-multichunk.md')
    if size!=1000:shutil.copyfile(FIXTURES/'dotenv-1200-80.txt',ws/'.env')
    environment={'CHUNK_SIZE':str(size),'OVERLAP_SIZE':str(overlap)} if shell else {}
    call(installed,'master_workflow','--step','chunking','--force','--llm','custom','--chunk-size','1000','--overlap','100',env=environment)
    out=ws/'chunked_markdown';reference=base/f'f6-chunk-{size}-{overlap}/chunked_markdown'
    chunks=sorted(out.glob('*_chunk_*.md'));assert len(chunks)>=2
    assert [p.name for p in chunks]==[p.name for p in sorted(reference.glob('*_chunk_*.md'))]
    for p in chunks:assert p.read_bytes()==(reference/p.name).read_bytes()
    for name in ['chunking_metadata.json','chunks_index.json']:
        assert json.loads((out/name).read_text())==json.loads((reference/name).read_text())
    settings=json.loads((ws/'pipeline_logs/pipeline_state.json').read_text())['settings']
    assert (settings['chunk_size'],settings['overlap'])==(size,overlap)


def test_sliding_exact_sequence(installed):
    python,_,base,ws=installed
    code="from pathlib import Path;import json;from chunk_markdown import MarkdownChunker;c=MarkdownChunker(target_llm='custom',chunk_size=1000,overlap=100);print(json.dumps(c.chunk_by_sliding_window(Path("+repr(str(FIXTURES/'f6-multichunk.md'))+").read_text())))"
    r=subprocess.run([str(python),'-I','-B','-c',code],cwd=ws,capture_output=True,text=True,timeout=30,close_fds=True)
    assert r.returncode==0,r.stderr
    old=json.loads((base/'f6-sliding/sliding.json').read_text())
    assert old['nonempty_overlap'] and len(old['chunks'])>=2
    assert json.loads(r.stdout)==old['chunks']


def test_empty_cleaning_error(installed):
    call(installed,'master_workflow','--step','cleaning','--force',expected=1)
    assert not list((installed[3]/'cleaned_markdown').glob('*.md'))
