"""Sonda reviewer CLA r003: link locali e whitespace dei documenti del delta (stdlib, sola lettura)."""
import json, re, sys, time, urllib.parse
from pathlib import Path
t0 = time.monotonic(); R = Path('/home/davide/workarea/markdown-for-llms')
files = ['docs/how-to/ambiente-uv.md', 'docs/how-to/marker-legacy-docker.md', 'docs/reference/toolchain-legacy.md',
         'documentation/CHANGELOG.md', 'documentation/README.md', 'documentation/roadmap.md',
         'documentation/decisions/README.md', 'documentation/decisions/0006-python-toolchain-uv.md']
out = {'links': 0, 'missing': [], 'trailing_ws_or_tab': [], 'final_newline_missing': []}
for f in files:
    text = (R/f).read_text(encoding='utf-8')
    if not text.endswith('\n'): out['final_newline_missing'].append(f)
    for n, line in enumerate(text.splitlines(), 1):
        if line != line.rstrip() or '\t' in line: out['trailing_ws_or_tab'].append(f'{f}:{n}')
        for target in re.findall(r'(?<!!)\[[^\]]*\]\(([^)\s]+)\)', line):
            if target.startswith(('http:', 'https:', 'mailto:', '#')): continue
            out['links'] += 1
            if not ((R/f).parent/urllib.parse.unquote(target.split('#', 1)[0])).exists(): out['missing'].append(f'{f}:{n}:{target}')
out['seconds_internal'] = round(time.monotonic() - t0, 4)
json.dump(out, sys.stdout, indent=1)
