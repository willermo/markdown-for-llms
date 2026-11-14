"""Generator of the checks: one deterministic set per twin, kept in the cache, never in the repo.

Each check has one class (content, structure, form), a type and a position-based id. The cache
key (A1 of the plan) is the SHA-256 of the canonical JSON of the document id, the input and twin
hashes, the projection, the generator and normalizer versions and the identity of the reference
toolchain; exclusions are applied later with their own hash. A formula with author macros the reader cannot expand is counted as not
determinable instead of becoming a wrong check.

A text unit is `ambiguous` when it has fewer than four tokens or its tokens recur elsewhere in the
twin (the window test of the matcher on the twin's own text): its position cannot be determined,
so it has no order check (the order chain links the unique units around it, and each ambiguous
unit is counted as not determinable) and its formulas and list membership are judged on any of
its occurrences (`match`). The caption of a floating table (`role` "table-caption") has no fixed
place in the running text either, like figure captions: it stays out of the order chain.
"""
import hashlib
import json
import re

from . import GENERATOR_VERSION, NORMALIZER_VERSION
from .cache import canonical, write_atomic
from .text import DROPPED, GREEK, SYMBOLS, WORD_COMMANDS, align_key, math_atoms, score_tokens, unicode_atoms

SCHEMA = "mdf-bench-checks/1"
MAX_ORDER_PAIRS = 200
CLASS_OF = {
    "paragraph": "content", "cell": "content", "formula": "content", "inline_math": "content", "figure": "content",
    "footnote": "content", "bib_entry": "content",
    "heading": "structure", "heading_level": "structure", "table": "structure", "table_shape": "structure",
    "table_header": "structure", "table_row": "structure", "list_item": "structure", "note_linked": "structure",
    "order": "structure",
    "emphasis": "form", "hyphenation": "form", "html_residue": "form", "spaced_math": "form", "noise": "form",
}
KNOWN_COMMANDS = set(GREEK) | set(SYMBOLS) | WORD_COMMANDS | DROPPED | {
    "sqrt", "frac", "sum", "prod", "int", "lim", "infty", "partial", "nabla", "cdot", "dots", "ldots", "cdots",
    "left", "right", "text", "operatorname", "mathrm", "label", "nonumber", "notag", "quad", "qquad"}


def cache_key(doc, hashes):
    """The key of a check set; the toolchain identity comes from `toolchain.json`, without a container
    (the projection of D06 is read with poppler of the toolchain)."""
    from . import toolchain
    twin = doc["twin"]
    value = {"document_id": doc["id"], "input_sha256": hashes.get("derived", {}).get("sha256") if isinstance(hashes.get("derived"), dict) else hashes["source"],
             "twin_sha256": hashes["twin"], "inventory_sha256": hashes.get("inventory"),
             "projection": twin.get("select") if twin["kind"] == "xlsx-projection" else None,
             "rules": twin.get("rules", []), "discard": twin.get("discard", []),
             "generator_version": GENERATOR_VERSION, "normalizer_version": NORMALIZER_VERSION,
             "toolchain": toolchain.identity()}
    return hashlib.sha256(canonical(value)).hexdigest()


def formula_atoms(text, kind):
    return unicode_atoms(text) if kind in ("mathml", "linear") else math_atoms(text)


def determinable_tex(tex):
    return all(c in KNOWN_COMMANDS for c in re.findall(r"\\([A-Za-z]+)", tex))


def spaced_words(tex):
    """Multi-letter words of a formula that a converter may spell with spaced letters (R4)."""
    words = set(re.findall(r"\\(" + "|".join(sorted(WORD_COMMANDS)) + r")(?![A-Za-z])", tex))
    for m in re.finditer(r"\\(?:text|mathrm|operatorname|textrm)\s*\{([^}]*)\}", tex):
        words |= {w for w in re.findall(r"[A-Za-z]{3,}", m.group(1))}
    return sorted(w for w in words if len(w) >= 3)


def _twin_stream(twin):
    """Tokens of all the text of the twin in reading order (cells included)."""
    tokens = []
    for e in twin.elements:
        if e.type == "table":
            for c in e.cells:
                tokens += score_tokens(c.text)
        else:
            tokens += score_tokens(e.text)
    return tokens


def generate(doc, twin):
    """The check set of a twin: {"checks": [...], "not_determinable": {class: n}}."""
    from .match import Matcher, Stream
    checks, nd = [], {"content": 0, "structure": 0, "form": 0}
    rules = (doc.get("twin") or {}).get("rules", ())
    in_twin = Matcher([], Stream(_twin_stream(twin)), rules)
    ambiguous, floating = set(), set()

    def add(type_, id_, **payload):
        checks.append(dict({"id": id_, "class": CLASS_OF[type_], "type": type_}, **payload))

    text_units, headings, figure_index = [], [], 0
    for e in twin.elements:
        if e.type in ("paragraph", "list_item"):
            tokens = score_tokens(e.text)
            if not tokens:
                continue
            extra = {"role": e.role} if e.role else {}
            if e.gaps:
                extra["gaps"] = e.gaps
            if len(tokens) < 4 or len(in_twin.find_all(tokens, e.gaps)) > 1:
                extra["ambiguous"] = True
                ambiguous.add(e.pos)
            add("paragraph", e.pos, tokens=tokens, **extra)
            text_units.append(e.pos)
            if e.role == "table-caption":
                floating.add(e.pos)
            if e.type == "list_item":
                add("list_item", e.pos + ":list", ref=e.pos)
            elif len(tokens) >= 8 and not e.role:
                add("hyphenation", e.pos + ":hyphen", ref=e.pos)
                add("html_residue", e.pos + ":html", ref=e.pos)
            if e.math:
                formulas = [formula_atoms(m, e.math_kind) for m in e.math]
                if e.math_kind == "tex" and not all(determinable_tex(m) for m in e.math):
                    nd["content"] += 1
                elif any(formulas):
                    add("inline_math", e.pos + ":math", ref=e.pos, formulas=[f for f in formulas if f])
            for k, (kind, run) in enumerate(e.emphasis, 1):
                run_tokens = score_tokens(run)
                if run_tokens:
                    add("emphasis", f"{e.pos}:emph{k}", ref=e.pos, kind=kind, tokens=run_tokens)
        elif e.type == "heading":
            tokens = score_tokens(e.text)
            if tokens:
                add("heading", e.pos, tokens=tokens, level=e.level, **({"gaps": e.gaps} if e.gaps else {}))
                headings.append(e)
        elif e.type == "formula":
            if e.math_kind == "tex" and not determinable_tex(e.text):
                nd["content"] += 1
                continue
            atoms = formula_atoms(e.text, e.math_kind)
            if atoms:
                add("formula", e.pos, atoms=atoms)
                words = spaced_words(e.text) if e.math_kind == "tex" else []
                if words:
                    add("spaced_math", e.pos + ":spaced", ref=e.pos, words=words)
        elif e.type == "figure":
            figure_index += 1
            add("figure", e.pos, tokens=score_tokens(e.text), index=figure_index)
        elif e.type == "footnote":
            tokens = score_tokens(e.text)
            if tokens:
                add("footnote", e.pos, tokens=tokens)
                add("note_linked", e.pos + ":linked", ref=e.pos)
        elif e.type == "bib_entry":
            tokens = score_tokens(e.text)
            if tokens:
                add("bib_entry", e.pos, tokens=tokens)
        elif e.type == "noise":
            tokens = score_tokens(e.text)
            if tokens:
                add("noise", e.pos + ":noise", key=align_key(e.text), tokens=tokens)
        elif e.type == "table":
            _table_checks(e, twin.kind, add)
    for a, b in zip(headings, headings[1:]):
        relation = (b.level > a.level) - (b.level < a.level)
        add("heading_level", f"{b.pos}:level", a=a.pos, b=b.pos, relation=relation)
    unique = [u for u in text_units if u not in ambiguous and u not in floating]
    nd["structure"] += max(0, len(text_units) - 1) - max(0, len(unique) - 1)
    pairs = list(zip(unique, unique[1:]))
    if len(pairs) > MAX_ORDER_PAIRS:
        step = len(pairs) / MAX_ORDER_PAIRS
        pairs = [pairs[int(k * step)] for k in range(MAX_ORDER_PAIRS)]
    for a, b in pairs:
        add("order", f"{a}:order", a=a, b=b)
    return {"checks": checks, "not_determinable": nd}


def _table_checks(e, kind, add):
    cells = [c for c in e.cells if score_tokens(c.text) or c.value]
    if not cells:
        return
    rows = sorted({c.row for c in cells})          # rows with content (empty rows are not rows)
    width = max(c.col + c.colspan for c in e.cells)
    finder = [score_tokens(c.text) for c in cells]
    for c in cells:
        add("cell", f"{e.pos}/r{c.row + 1}/c{c.col + 1}", table=e.pos, tokens=score_tokens(c.text),
            value=c.value or None, digits=c.digits)
    add("table", e.pos + ":table", table=e.pos, cells=finder)
    if kind == "xlsx-projection":
        by_row = {}
        for c in cells:
            by_row.setdefault(c.row, []).append(c)
        for r, row_cells in sorted(by_row.items()):
            add("table_row", f"{e.pos}/r{r + 1}:row", table=e.pos,
                values=[(c.value, c.digits) for c in sorted(row_cells, key=lambda x: x.col)])
        add("table_shape", e.pos + ":shape", table=e.pos, cells=finder, rows=len(rows), cols=None)
        return
    add("table_shape", e.pos + ":shape", table=e.pos, cells=finder, rows=len(rows), cols=width)
    header = [score_tokens(c.text) for c in cells if c.header]
    if e.level and header:
        add("table_header", e.pos + ":header", table=e.pos, cells=finder, header=header)


def checkset(doc, files, twin):
    key = cache_key(doc, files.hashes)
    generated = generate(doc, twin)
    body = {"schema": SCHEMA, "key": key, "document": doc["id"], "generator": GENERATOR_VERSION,
            "normalizer": NORMALIZER_VERSION, "twin_kind": twin.kind, "rules": doc["twin"].get("rules", []),
            "stats": twin.stats, "not_determinable": generated["not_determinable"], "checks": generated["checks"]}
    body["checks_sha256"] = hashlib.sha256(canonical(body["checks"])).hexdigest()
    return body


def load_or_generate(doc, files, cache, read_twin):
    """Checks of a document from the cache, or generated and written atomically."""
    key = cache_key(doc, files.hashes)
    path = cache.checks_dir / f"{key}.json"
    if path.is_file():
        try:
            body = json.loads(path.read_bytes())
            if body.get("key") == key and body.get("checks_sha256") == hashlib.sha256(canonical(body["checks"])).hexdigest():
                return body
        except ValueError:
            pass
    body = checkset(doc, files, read_twin())
    write_atomic(path, json.dumps(body, ensure_ascii=False, sort_keys=True).encode("utf-8"))
    return body
