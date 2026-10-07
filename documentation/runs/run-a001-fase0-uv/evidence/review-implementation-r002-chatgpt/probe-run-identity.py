"""Sonda stdlib/mock: nessun R, processo prodotto, rete o snapshot nuovo."""
import importlib.util
import json
from pathlib import Path
import sys
import signal
import tempfile
import time
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path('/home/davide/workarea/markdown-for-llms')
OUT = Path(__file__).parent
start = time.monotonic()
signal.alarm(10)
path = ROOT / 'scripts/diagnostics/run-a001-fase0-uv/run_offline.py'
spec = importlib.util.spec_from_file_location('review_wrapper', path)
tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool)
cases = []
with tempfile.TemporaryDirectory(prefix='probe-', dir=OUT) as directory:
    repo = Path(directory)
    for run_id in ('run-a001-fase0-uv', 'run-b002-fase0-uv'):
        snapshot = repo / 'temp' / run_id / 'snapshots' / 'review-valid.json'
        snapshot.parent.mkdir(parents=True)
        snapshot.write_text(json.dumps(dict(schema=1, run_id=run_id,
                                          label='review-valid', worktree_sha256='0'*64)))
        with patch.object(tool.subprocess, 'run', return_value=SimpleNamespace(
                returncode=0, stdout='MATCH synthetic verifier', stderr='')) as verify:
            try:
                result = tool.snapshot_check(repo, snapshot, sys.executable)
                row = dict(run_id=run_id, accepted=True, result=result)
            except ValueError as exc:
                row = dict(run_id=run_id, accepted=False, error=str(exc))
            row['verifier_called'] = verify.called
            cases.append(row)
assert cases[0]['accepted'] and cases[0]['verifier_called']
assert not cases[1]['accepted'] and not cases[1]['verifier_called']
receipt = dict(scope='MOCK_ONLY_NOT_PRODUCT_PASS', cases=cases,
               seconds=time.monotonic()-start, timeout_seconds=10,
               temporary_files_collected=True, subprocess_calls_real=0)
previous = OUT / 'probe-run-identity.json'
if previous.exists():
    (OUT / 'probe-run-identity-first.json').write_bytes(previous.read_bytes())
previous.write_text(json.dumps(receipt, indent=2)+'\n')
signal.alarm(0)
print(json.dumps(receipt))
