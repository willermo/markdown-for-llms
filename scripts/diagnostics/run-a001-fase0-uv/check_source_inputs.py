#!/usr/bin/env python3
"""Confronto corrente/S/snapshot prima della build, senza receipt I ancora futura."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',type=Path,required=True)
    parser.add_argument('--source',type=Path,required=True)
    args=parser.parse_args()
    try:
        path=Path(__file__).with_name('make_source_manifest.py')
        spec=importlib.util.spec_from_file_location('source',path)
        helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
        source=helper.load_json(args.source)
        helper.validate_current(source,args.repo,official=True)
        print(json.dumps({'status':'PASS_CURRENT_SOURCE_ONLY','id':source['id']}));return 0
    except (OSError,ValueError) as e:
        print('FAIL source: '+str(e),file=sys.stderr);return 2


if __name__=='__main__':raise SystemExit(main())
