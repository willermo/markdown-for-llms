#!/usr/bin/env python3
"""Launcher -I: carica solo la directory fidata degli helper benchmark.

Gli hash della directory sono controllati dall'inventario esterno prima/dopo
le prove ufficiali. Non aggiunge la root del prodotto a sys.path.
"""
import argparse
from pathlib import Path
import runpy
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("script", choices=("generate_corpus", "corpus", "reference_check", "packet", "benchmark", "campaign", "inventory", "prepare", "controls", "contracts", "verification", "output_packet", "comparison", "comparison_check", "metadata_revision", "derive_verification"))
    parser.add_argument("args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    root = Path(__file__).absolute().parent
    if not sys.flags.isolated or not sys.dont_write_bytecode:
        parser.error("Richiesto -I -B")
    if any(p.is_symlink() for p in (root, *root.parents)):
        parser.error("Root benchmark con symlink")
    sys.path.insert(0, str(root))
    sys.argv = [str(root / (args.script + ".py")), *args.args]
    runpy.run_path(sys.argv[0], run_name="__main__")


if __name__ == "__main__":
    main()
