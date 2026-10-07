#!/usr/bin/env python3
"""Inventario statico di fence anche indentate/lista e inline fuori dalle fence."""
import argparse
import hashlib
import json
from pathlib import Path
import re


def scan(path):
    raw=path.read_bytes();lines=raw.decode().splitlines(keepends=True)
    blocks=[];inline=[];opened=None
    for i,line in enumerate(lines,1):
        normalized=re.sub(r'^\s*(?:>\s*)*(?:(?:[-+*]|\d+[.)])\s+)?','',line)
        delimiter=re.match(r'(`{3,}|~{3,})([^\n]*)',normalized)
        if opened:
            if delimiter and delimiter[1][0]==opened['delimiter'][0] and len(delimiter[1])>=len(opened['delimiter']) and not delimiter[2].strip():
                opened.update(last_line=i,text=''.join(lines[opened['first_line']:i-1]));blocks.append(opened);opened=None
        elif delimiter:
            opened={'first_line':i,'delimiter':delimiter[1],'language':delimiter[2].strip()}
        else:
            for match in re.finditer(r'(?<!`)(`+)(.+?)\1(?!`)',line):
                value=match[2]
                inline.append({'line':i,'text':value,'command':bool(re.match(r'^(?:docker(?: compose)?|git|python(?:3)?|pytest|pip|uv|pandoc|curl|poetry|sudo)(?:\s|$)',value))})
    if opened:raise ValueError('Fence non chiusa: '+str(path))
    return {'path':str(path),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'blocks':blocks,'inline':inline}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('files',nargs='+',type=Path)
    a=p.parse_args();result=[scan(path) for path in a.files]
    with a.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps([{'path':x['path'],'blocks':len(x['blocks']),'inline_commands':sum(i['command'] for i in x['inline'])} for x in result]))


if __name__=='__main__':main()
