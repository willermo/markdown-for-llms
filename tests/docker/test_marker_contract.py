"""C-docker separato: driver permanente, daemon locale e nove contratti offline."""
import json,os,re,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]

def test_marker_contract():
    image=os.environ.get('RUN_IMAGE','');assert re.fullmatch(r'sha256:[0-9a-f]{64}',image)
    values={key:Path(os.environ[key]) for key in ['RUN_IMAGE_MANIFEST','RUN_IMAGE_INPUTS','RUN_BACKEND_WHEEL','RUN_IMAGE_CONTEXT_RECEIPT','RUN_TEST_OUTPUT']}
    assert all(p.is_absolute() for p in values.values())
    for key in ['RUN_IMAGE_MANIFEST','RUN_IMAGE_INPUTS','RUN_BACKEND_WHEEL','RUN_IMAGE_CONTEXT_RECEIPT']:assert values[key].is_file()
    out=values['RUN_TEST_OUTPUT'];assert not out.exists()
    argv=[sys.executable,'-I','-B',str(ROOT/'scripts/diagnostics/run-a001-fase0-uv/run_docker_contract.py'),
          '--repo',str(ROOT),'--image',image,'--expected',str(values['RUN_IMAGE_MANIFEST']),
          '--source',os.environ['RUN_SOURCE_MANIFEST'],'--installation',os.environ['RUN_INSTALLATION_RECEIPT'],
          '--mode',os.environ.get('RUN_TEST_MODE','official'),
          '--context-receipt',str(values['RUN_IMAGE_CONTEXT_RECEIPT']),'--inputs-receipt',str(values['RUN_IMAGE_INPUTS']),'--backend-wheel',str(values['RUN_BACKEND_WHEEL']),
          '--output',str(out),'--timeout',os.environ.get('RUN_DOCKER_TIMEOUT','900')]
    env={'PATH':'/usr/bin:/bin','LANG':'C.UTF-8','LC_ALL':'C.UTF-8','PYTHONDONTWRITEBYTECODE':'1'}
    r=subprocess.run(argv,cwd=ROOT,env=env,shell=False,close_fds=True,capture_output=True,text=True,timeout=1080)
    assert (out/'result.json').is_file(),r.stdout+r.stderr
    result=json.loads((out/'result.json').read_text());assert r.returncode==0 and result['status']=='PASS',result
    assert result['owned_container_collected'] and len(result['contract']['subcases'])==9
    assert not result['contract']['network_attempts']
