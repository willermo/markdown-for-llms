"""Sonda stdlib/mock propria: soli test permanenti dei confini del launcher."""
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path('/home/davide/workarea/markdown-for-llms')
HERE=Path(__file__).parent
tmp=HERE/'probe-tmp'
tmp.mkdir(exist_ok=False)
argv=[sys.executable,'-I','-B','-m','unittest','discover','-s',str(ROOT/'tests/unit'),'-p','test_core_diagnostics.py','-v']
env={'PATH':'/usr/bin:/bin','LANG':'C.UTF-8','LC_ALL':'C.UTF-8',
     'TMPDIR':str(tmp),'PYTHONDONTWRITEBYTECODE':'1','PYTHONNOUSERSITE':'1'}
t=time.monotonic()
try:
    p=subprocess.run(argv,cwd=ROOT,env=env,shell=False,close_fds=True,
                     capture_output=True,text=True,timeout=15)
    result=dict(argv=argv,cwd=str(ROOT),environment=env,timeout_seconds=15,
                wall_seconds=time.monotonic()-t,exit=p.returncode,stdout=p.stdout,stderr=p.stderr,
                scope='stdlib/mock only; no real R or application',new_R_PASS=False,
                new_V1_PASS=False,shell=False,close_fds=True)
except subprocess.TimeoutExpired as e:
    result=dict(argv=argv,wall_seconds=time.monotonic()-t,timeout=True,
                stdout=str(e.stdout),stderr=str(e.stderr),scope='stdlib/mock only')
(HERE/'probe.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ['stdout','stderr','environment','argv']},ensure_ascii=False))
raise SystemExit(result.get('exit',2))
