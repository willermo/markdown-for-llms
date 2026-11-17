"""HTML twin reader (twin kind `html`): the HTML edition of a Project Gutenberg text (D31-D34).

The page is read by the (X)HTML walker of the EPUB reader. Its Project Gutenberg header and
licence (the elements of class `pg-boilerplate`) are not text of the work: they are removed
before the walk, and a page without them is `twin_boilerplate_unknown`, so that the licence can
never become checks (delta r3 Δ10).
"""
from pathlib import Path

from ..codes import BenchError, Code
from ..model import Builder
from .epub import walk
from .htmltree import parse_html
from .safe import local

BOILERPLATE = "pg-boilerplate"


def without_boilerplate(root):
    """Remove the boilerplate elements from the tree; their number (0: `twin_boilerplate_unknown`)."""
    parent = {c: p for p in root.iter() for c in p}
    found = [el for el in root.iter() if BOILERPLATE in (el.get("class") or "").split()]
    if not found:
        raise BenchError(Code.TWIN_BOILERPLATE_UNKNOWN, "no element of class " + BOILERPLATE)
    removed = 0
    for el in found:
        p = parent.get(el)
        if p is not None and any(c is el for c in p):
            p.remove(el)
            removed += 1
    return removed


def read_html_twin(path):
    root = parse_html(Path(path).read_bytes())
    b = Builder("html")
    b.count("boilerplate_removed", without_boilerplate(root))
    body = next((x for x in root.iter() if local(x.tag) == "body"), root)
    walk(body, b)
    return b.twin
