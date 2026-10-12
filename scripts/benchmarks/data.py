"""Contratto dati minimo v1. Nessun dominio applicativo o import prodotto."""
import hashlib
import json
import os
from pathlib import Path

PROTOCOL = "conversion-fidelity-v1"


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def digest_bytes(data):
    return hashlib.sha256(data).hexdigest()


def digest(value):
    return digest_bytes(canonical(value))


def unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Chiave duplicata: " + key)
        result[key] = value
    return result


def load(path):
    from safety import regular
    path = Path(path).absolute()
    return json.loads(regular(path.parent, path.name).decode("utf-8"), object_pairs_hook=unique)


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(canonical(value) + b"\n")


def atomic_new(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".partial")
    write_new(temporary, value)
    # link is exclusive; existing final records are never overwritten.
    os.link(temporary, path)
    temporary.unlink()


def descriptor(path):
    path = Path(path)
    content = path.read_bytes()
    return {"path": str(path), "bytes": len(content), "sha256": digest_bytes(content)}


def counts(gold):
    return {"blocks": len(gold["blocks"]),
            "formulas": sum(len(b.get("formulas", [])) for b in gold["blocks"]),
            "cells": sum(len(row) for b in gold["blocks"] for row in b.get("cells", [])),
            "figures": sum(b["type"] == "figure" for b in gold["blocks"]),
            "references": sum(len(b.get("references", [])) for b in gold["blocks"]),
            "numbers": sum(len(b.get("numbers", [])) for b in gold["blocks"])}
