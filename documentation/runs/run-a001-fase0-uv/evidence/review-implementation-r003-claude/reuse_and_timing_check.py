"""Sonda reviewer CLA r003: sola lettura stdlib (hash, JSON, mtime). Nessun import applicativo."""
import hashlib, json, os, sys, time
from pathlib import Path
t0 = time.monotonic()
R = Path('/home/davide/workarea/markdown-for-llms'); RUN = R/'temp/run-a001-fase0-uv'
H = RUN/'evidence/implementation-r001/fix-review-r003'; OLD = H.parent/'fix-review-r002'
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ident(p): p = Path(p); return dict(path=str(p), bytes=p.stat().st_size, sha256=sha(p))
load = lambda p: json.loads(Path(p).read_text())
out = {}
eq = load(OLD/'equivalence-final.json')
ctx = [pair for pair in eq['Docker_r007_context_objects']]
out['docker_context_objects'] = len(ctx)
out['docker_context_all_equal_now'] = all(ident(row['path']) == row for pair in ctx for row in pair.values())
names = sorted(Path(pair['current']['path']).relative_to(R).as_posix() for pair in ctx)
out['docker_context_contains_delta_paths'] = [n for n in names if n in (
    'scripts/diagnostics/run-a001-fase0-uv/run_offline.py', 'tests/unit/test_core_diagnostics.py',
    'docs/how-to/ambiente-uv.md', 'docs/how-to/marker-legacy-docker.md', 'docs/reference/toolchain-legacy.md')]
out['docker_context_names'] = names
out['c_docker_exact_now'] = all(sha(R/row['path']) == row['sha256'] for row in eq['C_docker_driver_probe_and_test_exact'])
out['c_docker_paths'] = [row['path'] for row in eq['C_docker_driver_probe_and_test_exact']]
B = load(OLD/'actual-B-artifacts-s056.json')
out['B56_equal_now'] = all(ident(row['path']) == row for row in B.values())
S_new = load(H/'R-run-a001-fase0-uv/S.json')['payload']; S_old = load(OLD/'S-s060.json')['payload']
out['S_build_fields_equal_s060'] = {k: S_new[k] == S_old[k] for k in ['modules', 'build_inputs', 'backend', 'toolchain']}
S_b = load(H/'R-run-b002-fase0-uv/S.json')['payload']
out['S_b002_build_fields_equal_s060'] = {k: S_b[k] == S_old[k] for k in ['modules', 'build_inputs', 'backend', 'toolchain']}
# Are delta paths among inputs consumed by S (build/modules)?
s_paths = {v['path'] for k in ['modules', 'build_inputs'] for v in S_old[k]}
out['delta_paths_in_S_build_inputs'] = [p for p in out['docker_context_contains_delta_paths']] + [
    p for p in ('scripts/diagnostics/run-a001-fase0-uv/run_offline.py', 'tests/unit/test_core_diagnostics.py',
                'docs/how-to/ambiente-uv.md', 'docs/how-to/marker-legacy-docker.md', 'docs/reference/toolchain-legacy.md') if p in s_paths]
# Timing reconstruction from entry and mtimes.
E = load(H/'entry.json'); base = E['historical_conservative_entry'] + E['initial_reading_charge_seconds']
def at(wall): return base + (wall - E['wall_start'])
marks = {}
for name in ['entry.json', 'admission.json', 'calls.json', 'execution-result.json', 'static.json', 'finalize.py', 'matrix.json',
             'final-checks.json', 'manifest.json', 'close_cost.py', 'finalize-failure.json', 'closing-accounting.json']:
    marks[name] = round(at(os.stat(H/name).st_mtime), 3)
for p in ['scripts/diagnostics/run-a001-fase0-uv/run_offline.py', 'tests/unit/test_core_diagnostics.py', 'docs/how-to/ambiente-uv.md',
          'docs/how-to/marker-legacy-docker.md', 'docs/reference/toolchain-legacy.md']:
    marks[p] = round(at(os.stat(R/p).st_mtime), 3)
marks['codex_turn_checkpoint_ef7680e5'] = round(at(1791373867.769), 3)
out['cumulative_at_mtime'] = dict(sorted(marks.items(), key=lambda kv: kv[1]))
out['cap_crossed_wall'] = E['wall_start'] + (8880 - base)
ca = load(H/'closing-accounting.json')
out['closing_monotonic_delta_check'] = round(base + ca['monotonic'] - E['monotonic_start'], 6)
out['closing_wall_delta_check'] = round(base + ca['wall'] - E['wall_start'], 6)
out['last_R_result_charged_at'] = [load(H/f'R-run-{i}-fase0-uv/result.json')['charged_at'] for i in ('a001', 'b002')]
out['seconds_internal'] = round(time.monotonic() - t0, 4)
json.dump(out, sys.stdout, indent=1)
