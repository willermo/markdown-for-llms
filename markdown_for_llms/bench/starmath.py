"""The StarMath channel of the Writer twins (plan r2 §5, delta r3 Δ4).

Writer HTML prints a formula as an image. The formulas of a DOCX or ODT twin are judged instead
on its Flat ODT (`soffice --convert-to fodt`): every formula object keeps its StarMath source as
an annotation (`<annotation encoding="StarMath 5.0">`), which this module turns into atoms with a
keyword table of its own, written from the command reference of LibreOffice Math (operators,
relations, set operations, functions, large operators, attributes, brackets, indexes, Greek and
special symbols), never from the MathML the generator reads. The atoms follow the conventions of
`readers.mathml.mathml_linear`: a subscript or a lower limit is `_`, a superscript or an upper
limit is `^`, a root is `√` followed by its radicand and its index, a fraction is its two
operands, an accent follows its base after `^`; functions are their letters.

The table is frozen (`VERSION`): after the first run on D26 it is not adapted to the cases left;
a documented element that is missing is a correction for a cause (delta r3 Δ4.7). The Flat ODT
walker leaves tables of contents and indexes out, as the ODT reader does.
"""
import re

from .outdoc import Block, _add_block, _finish
from .readers.safe import local
from .text import GREEK as TEX_GREEK, unicode_atoms

VERSION = "1"
TEXT_NS = "urn:oasis:names:tc:opendocument:xmlns:text:1.0"
MATH_NS = "http://www.w3.org/1998/Math/MathML"
INDEXES = {"table-of-content", "alphabetical-index", "illustration-index", "table-index", "object-index",
           "user-index", "bibliography"}

# Unary and binary operators, relations, set operations, arrows, large operators and other symbols.
SYMBOLS = {
    "+-": "±", "-+": "∓", "neg": "¬", "cdot": "·", "times": "×", "div": "÷", "and": "∧", "&": "∧", "or": "∨",
    "circ": "∘", "odivide": "⊘", "odot": "⊙", "ominus": "⊖", "oplus": "⊕", "otimes": "⊗", "bslash": "∖",
    "slash": "/", "wideslash": "/", "widebslash": "∖",
    "<>": "≠", "neq": "≠", "<=": "≤", "le": "≤", "leslant": "≤", ">=": "≥", "ge": "≥", "geslant": "≥", "<<": "≪",
    "ll": "≪", ">>": "≫", "gg": "≫", "approx": "≈", "sim": "∼", "simeq": "≃", "equiv": "≡", "prop": "∝",
    "parallel": "∥", "ortho": "⊥", "divides": "∣", "ndivides": "∤", "toward": "→", "dlarrow": "⇐", "dlrarrow": "⇔",
    "drarrow": "⇒", "def": "≝", "prec": "≺", "succ": "≻", "preccurlyeq": "≼", "succcurlyeq": "≽", "precsim": "≾",
    "succsim": "≿", "nprec": "⊀", "nsucc": "⊁", "transl": "⊷", "transr": "⊶",
    "in": "∈", "notin": "∉", "owns": "∋", "intersection": "∩", "union": "∪", "setminus": "∖", "subset": "⊂",
    "subseteq": "⊆", "supset": "⊃", "supseteq": "⊇", "nsubset": "⊄", "nsubseteq": "⊈", "nsupset": "⊅",
    "nsupseteq": "⊉", "emptyset": "∅", "aleph": "ℵ", "setN": "ℕ", "setZ": "ℤ", "setQ": "ℚ", "setR": "ℝ",
    "setC": "ℂ",
    "sum": "∑", "prod": "∏", "coprod": "∐", "int": "∫", "iint": "∬", "iiint": "∭", "lint": "∮", "llint": "∯",
    "lllint": "∰",
    "infinity": "∞", "infty": "∞", "partial": "∂", "nabla": "∇", "exists": "∃", "notexists": "∄", "forall": "∀",
    "hbar": "ℏ", "lambdabar": "ƛ", "Re": "ℜ", "Im": "ℑ", "wp": "℘", "leftarrow": "←", "rightarrow": "→",
    "uparrow": "↑", "downarrow": "↓", "dotslow": "…", "dotsaxis": "⋯", "dotsvert": "⋮", "dotsup": "⋰",
    "dotsdiag": "⋰", "dotsdown": "⋱", "backepsilon": "∍",
}
# Functions print their letters; `lim` and the like are large operators that print their name.
FUNCTIONS = {"exp", "ln", "log", "sin", "cos", "tan", "cot", "sinh", "cosh", "tanh", "coth", "arcsin", "arccos",
             "arctan", "arccot", "arsinh", "arcosh", "artanh", "arcoth", "lim", "liminf", "limsup"}
# Brackets: printed ones; `{` and `}` only group, `left`, `right` and `none` print nothing themselves.
BRACKETS = {"(": "(", ")": ")", "[": "[", "]": "]", "lbrace": "{", "rbrace": "}", "langle": "⟨", "rangle": "⟩",
            "lceil": "⌈", "rceil": "⌉", "lfloor": "⌊", "rfloor": "⌋", "lline": "|", "rline": "|", "ldline": "‖",
            "rdline": "‖", "lbrack": "[", "rbrack": "]", "ldbracket": "⟦", "rdbracket": "⟧", "mline": "|",
            "\\{": "{", "\\}": "}", "\\(": "(", "\\)": ")", "\\[": "[", "\\]": "]", "\\lbrace": "{",
            "\\rbrace": "}", "\\langle": "⟨", "\\rangle": "⟩", "\\lline": "|", "\\rline": "|"}
# Indexes and limits: the mark, then the operand.
MARKS = {"^": "^", "sup": "^", "rsup": "^", "lsup": "^", "csup": "^", "to": "^", "_": "_", "sub": "_",
         "rsub": "_", "lsub": "_", "csub": "_", "from": "_"}
# Attributes (accents) print after their base: base, "^", accent.
ACCENTS = {"acute": "´", "grave": "`", "breve": "˘", "circle": "˚", "dot": "˙", "ddot": "¨", "dddot": "⃛",
           "bar": "¯", "vec": "⃗", "widevec": "⃗", "tilde": "~", "widetilde": "~", "hat": "^", "widehat": "^",
           "check": "ˇ", "harpoon": "⃑", "wideharpoon": "⃑", "overline": "¯"}
# Formatting that takes a value before its operand (`color red x`, `size 12 x`, `font sans x`).
WITH_VALUE = {"color", "size", "font"}
# Formatting, spacing and layout that print nothing of their own.
SILENT = {"bold", "nbold", "ital", "italic", "nitalic", "underline", "overstrike", "phantom", "newline", "nospace",
          "alignl", "alignc", "alignr", "alignt", "alignm", "alignb", "left", "right", "none", "over", "binom",
          "stack", "matrix", "func", "oper", "boper", "uoper", "#", "##", "~", "`", "evaluate", "overbrace",
          "underbrace", "frac"}
# Special symbols written with % (besides the Greek letters).
SPECIAL = {"and": "∧", "angle": "∢", "element": "∈", "identical": "≡", "infinite": "∞", "noelement": "∉",
           "notequal": "≠", "or": "∨", "perthousand": "‰", "strictlygreaterthan": "≫", "strictlylessthan": "≪",
           "tendto": "→"}
GREEK = {**{k: v for k, v in TEX_GREEK.items() if not k.startswith("var") and k[0].islower()},
         "varepsilon": "ε", "vartheta": "ϑ", "varpi": "ϖ", "varrho": "ϱ", "varsigma": "ς", "varphi": "φ"}
GREEK_UPPER = {"ALPHA": "Α", "BETA": "Β", "GAMMA": "Γ", "DELTA": "Δ", "EPSILON": "Ε", "ZETA": "Ζ", "ETA": "Η",
               "THETA": "Θ", "IOTA": "Ι", "KAPPA": "Κ", "LAMBDA": "Λ", "MU": "Μ", "NU": "Ν", "XI": "Ξ",
               "OMICRON": "Ο", "PI": "Π", "RHO": "Ρ", "SIGMA": "Σ", "TAU": "Τ", "UPSILON": "Υ", "PHI": "Φ",
               "CHI": "Χ", "PSI": "Ψ", "OMEGA": "Ω"}
TOKEN = re.compile(r'"(?:[^"\\]|\\.)*"|%[A-Za-z]+|\\[A-Za-z]+|\\.|<>|<=|>=|<<|>>|\+-|-\+|##|[A-Za-z]+|'
                   r"\d+(?:[.,]\d+)?|\S")


def _greek(name):
    base = name[1:] if name[:1] in ("i", "I") and (name[1:] in GREEK or name[1:] in GREEK_UPPER
                                                    or name[1:].lower() in GREEK) else name
    if base in GREEK:
        return GREEK[base]
    if base in GREEK_UPPER:
        return GREEK_UPPER[base]
    if base in SPECIAL:
        return SPECIAL[base]
    if base.lower() in GREEK and base[:1].isupper():
        return GREEK[base.lower()].upper()
    return None


class _Parser:
    def __init__(self, src):
        self.toks = TOKEN.findall(src)
        self.i = 0

    def peek(self):
        return self.toks[self.i] if self.i < len(self.toks) else None

    def take(self):
        tok = self.peek()
        self.i += 1
        return tok

    def sequence(self, closer=None):
        out = []
        while self.peek() is not None and self.peek() != closer:
            out += self.item()
        if closer is not None and self.peek() == closer:
            self.i += 1
        return out

    def operand(self):
        """The next operand: a group, or one item with the indexes that follow it."""
        if self.peek() is None:
            return []
        out = self.item()
        while self.peek() in MARKS and self.peek() not in ("from", "to"):
            out += [MARKS[self.take()]] + self.operand_core()
        return out

    def operand_core(self):
        return self.item() if self.peek() is not None else []

    def item(self):
        tok = self.take()
        if tok is None:
            return []
        if tok == "{":
            return self.sequence("}")
        if tok == "}":
            return []
        if tok.startswith('"'):
            return unicode_atoms(tok[1:-1].replace('\\"', '"'))
        if tok.startswith("%"):
            sym = _greek(tok[1:])
            return [sym] if sym else unicode_atoms(tok[1:])
        if tok in WITH_VALUE:
            self.take()
            return self.operand()
        if tok in ("left", "right"):
            nxt = self.take()
            return [BRACKETS[nxt]] if nxt in BRACKETS else []
        if tok in SILENT:
            return []
        if tok in MARKS:
            return [MARKS[tok]] + self.operand()
        if tok == "sqrt":
            return ["√"] + self.operand()
        if tok == "nroot":
            index = self.operand()
            return ["√"] + self.operand() + index
        if tok == "abs":
            return ["|"] + self.operand() + ["|"]
        if tok == "fact":
            return self.operand() + ["!"]
        if tok in ACCENTS:
            return self.operand() + ["^", ACCENTS[tok]]
        if tok in FUNCTIONS:
            return list(tok)
        if tok in SYMBOLS:
            return [SYMBOLS[tok]]
        if tok in BRACKETS:
            return [BRACKETS[tok]]
        if tok.startswith("\\") and len(tok) == 2:
            return [tok[1]]
        return unicode_atoms(tok)


def atoms(src):
    """Atoms of a StarMath formula, comparable with `mathml_linear` and `text.math_atoms`."""
    return [a for a in _Parser(src or "").sequence() if a]


def annotation(math_el):
    """The StarMath source of a MathML element, or None."""
    for a in math_el.iter():
        if local(a.tag) == "annotation" and "StarMath" in (a.get("encoding") or ""):
            return a.text or ""
    return None


def fodt_document(root):
    """(OutputDoc, inventory) of a Flat ODT: one paragraph block per `text:p`/`text:h` with the
    StarMath formulas it holds (a paragraph with formulas and no word is display math); the
    inventory is one `formula` unit per formula with a StarMath source (keys: atoms joined by a
    space). Tables of contents and indexes are left out."""
    blocks, tokens, token_block, broken, formulas = [], [], [], set(), []
    units = []
    parent = {c: p for p in root.iter() for c in p}

    def in_index(el):
        x = parent.get(el)
        while x is not None:
            if x.tag.startswith("{%s}" % TEXT_NS) and local(x.tag) in INDEXES:
                return True
            x = parent.get(x)
        return False

    def own(el):
        parts, maths = [], []

        def rec(x, top):
            name, ns = local(x.tag), x.tag[1:].split("}")[0] if x.tag.startswith("{") else ""
            if not top and ns == TEXT_NS and name in ("p", "h"):
                return
            if ns == MATH_NS and name == "math":
                src = annotation(x)
                found = atoms(src) if src is not None else []
                if found:
                    maths.append(" ".join(found))
                parts.append(" ")
                return
            if name in ("line-break", "tab", "s"):
                parts.append(" ")
            if name not in ("binary-data", "note-citation"):
                parts.append(x.text or "")
            for k in x:
                rec(k, False)
                parts.append(k.tail or "")
        rec(el, True)
        return re.sub(r"\s+", " ", "".join(parts)).strip(), maths

    for el in root.iter():
        if el.tag in ("{%s}p" % TEXT_NS, "{%s}h" % TEXT_NS) and not in_index(el):
            text, maths = own(el)
            if not maths and not text:
                continue
            display = bool(maths) and not re.search(r"\w", text)
            formulas.extend((display, m) for m in maths)
            units.extend(maths)
            _add_block(blocks, tokens, token_block, broken, Block("paragraph", math=list(maths)), text)
    return _finish(blocks, tokens, token_block, broken, False, formulas, []), {"formula": units} if units else {}
