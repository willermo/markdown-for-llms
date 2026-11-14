"""Neutral model of a twin: typed elements with stable positions.

Every reader produces the same model; one generator turns it into checks. Positions
(`p12`, `t1`, `t1/r3/c2`, `pg3/p2`) identify checks, exclusions and feedback without text.
"""
from dataclasses import dataclass, field

TYPES = ("heading", "paragraph", "list_item", "table", "formula", "figure", "footnote", "bib_entry", "noise")
ABBR = {"heading": "h", "paragraph": "p", "list_item": "li", "table": "t", "formula": "f", "figure": "fig",
        "footnote": "n", "bib_entry": "b", "noise": "z"}


@dataclass(frozen=True)
class Cell:
    text: str
    row: int
    col: int
    rowspan: int = 1
    colspan: int = 1
    header: bool = False
    value: str = ""                # raw number of a numeric cell (XLSX), compared at `digits`
    digits: int = -1               # decimals shown by the twin for a numeric cell


@dataclass(frozen=True)
class Element:
    type: str
    pos: str
    text: str = ""
    level: int = 0                 # heading level, list depth, header rows of a table
    display: bool = False          # display formula
    math: tuple = ()               # inline formulas of a paragraph (TeX or linearized MathML)
    math_kind: str = "tex"         # "tex" or "mathml" for the formula text
    emphasis: tuple = ()           # (("i" | "b"), text) runs of a paragraph
    cells: tuple = ()              # Cell objects of a table
    page: int = 0                  # page of a paged twin (Wikisource), 0 otherwise
    role: str = ""                 # finer role of a paragraph, for example "table-caption"
    gaps: int = 0                  # printed items the reader cannot know (LaTeX references, citation keys)


@dataclass
class Twin:
    kind: str
    elements: list = field(default_factory=list)
    stats: dict = field(default_factory=dict)   # counters declared by the reader (unknown templates, ...)

    def of(self, kind):
        return [e for e in self.elements if e.type == kind]


class Builder:
    """Assigns positions in reading order; a page prefix is set by paged readers."""

    def __init__(self, kind):
        self.twin = Twin(kind)
        self.counters = {}
        self.page = 0

    def add(self, type_, text="", **kw):
        if type_ not in TYPES:
            raise ValueError(type_)
        prefix = f"pg{self.page}/" if self.page else ""
        key = (prefix, type_)
        self.counters[key] = self.counters.get(key, 0) + 1
        element = Element(type_, f"{prefix}{ABBR[type_]}{self.counters[key]}", text, page=self.page, **kw)
        self.twin.elements.append(element)
        return element

    def count(self, name, n=1):
        self.twin.stats[name] = self.twin.stats.get(name, 0) + n
