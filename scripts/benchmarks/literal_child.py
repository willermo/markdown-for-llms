"""TXT letterale; delimitatore più lungo di ogni run di backtick sorgente."""
from pathlib import Path
import re
import sys

if not sys.flags.isolated or not sys.dont_write_bytecode:raise ValueError('Require -I -B')
payload=Path(sys.argv[1]).read_bytes();text=payload.decode('utf-8')
marker='`'*max(3,1+max((len(m[0]) for m in re.finditer(r'`+',text)),default=0))
# The child requires a final LF for literal roundtrip; current authored TXT
# has it. Missing LF remains explicit rather than silently added to payload.
if not text.endswith('\n'):raise ValueError('TXT without final LF outside literal-v1')
with Path(sys.argv[2]).open('xb') as f:f.write((marker+'text\n').encode()+payload+(marker+'\n').encode())
