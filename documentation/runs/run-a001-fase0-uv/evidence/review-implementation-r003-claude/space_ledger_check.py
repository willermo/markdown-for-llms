"""Sonda reviewer CLA r003: ledger spazio stdlib con radici/esclusioni dello scope R021 (sola lettura)."""
import json, os, stat, sys, time
from pathlib import Path
t0 = time.monotonic(); R = Path('/home/davide/workarea/markdown-for-llms')
scope = json.loads((R/'temp/run-a001-fase0-uv/evidence/supervisor-arbitration-implementation-r002/authorized-scope.json').read_text())['core']
excl = [str(R/p) for p in scope['exclude_new_docker_subtrees']]
seen = set(); out = {'roots': {}}
for root in scope['roots']:
    base = Path(root) if root.startswith('/') else R/root
    lg = al = n = 0
    for dirpath, dirnames, filenames in os.walk(base, followlinks=False):
        if any(dirpath == e or dirpath.startswith(e + '/') for e in excl):
            dirnames[:] = []; continue
        for name in dirnames + filenames:
            try: st = os.lstat(os.path.join(dirpath, name))
            except FileNotFoundError: continue
            key = (st.st_dev, st.st_ino)
            if key in seen: continue
            seen.add(key); n += 1; lg += st.st_size; al += st.st_blocks * 512
    out['roots'][root] = dict(logical=lg, allocated=al, entries=n)
lg = sum(v['logical'] for v in out['roots'].values()); al = sum(v['allocated'] for v in out['roots'].values())
H = max(lg, al); delta = max(0, H - scope['Hentry'])
out.update(logical=lg, allocated=al, H=H, delta=delta, incremental_cap=scope['incremental_bytes'],
           remaining=scope['incremental_bytes'] - delta,
           stop_check=H + max(0, scope['incremental_bytes'] - delta) + scope['external_reserve_bytes'] < scope['stop_bytes'],
           admitted=delta < scope['incremental_bytes'],
           repository_free=os.statvfs(R).f_bavail * os.statvfs(R).f_frsize,
           tmp_free=os.statvfs('/tmp').f_bavail * os.statvfs('/tmp').f_frsize,
           seconds_internal=round(time.monotonic() - t0, 3),
           note='Misura indipendente approssimata: inode contati una volta, directory incluse; non sostituisce il ledger del supervisore.')
json.dump(out, sys.stdout, indent=1)
