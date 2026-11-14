"""Declared normalizations and tokens.

Three normalizations stay separate (C3 of the plan):
- `score_tokens`: twin and converter output, for the scores. NFC, ligatures, quotes, dashes,
  invisible characters, spaces, lower case. Never folds accents.
- `align_key`: alignment of generated units with an independent inventory (coverage only, never
  in the scores). Also folds diacritics, apostrophes and punctuation.
- `selfcheck` normalization is `score_tokens` plus the per-document twin rules of the manifest
  (`it-final-accent`), which only widen what a twin token accepts (see `accepts`).
"""
import difflib
import re
import unicodedata

QUOTES = {"\u2019": "'", "\u2018": "'", "\u201a": "'", "\u2032": "'", "\u02bc": "'", "`": "'",
          "\u201c": '"', "\u201d": '"', "\u201e": '"', "\u00ab": '"', "\u00bb": '"'}
DASHES = {"\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2013": "-", "\u2014": "-", "\u2015": "-", "\u2212": "-"}
LIGATURES = {"\ufb00": "ff", "\ufb01": "fi", "\ufb02": "fl", "\ufb03": "ffi", "\ufb04": "ffl", "\ufb05": "st", "\ufb06": "st"}
INVISIBLE = {"\u00ad": "", "\u200b": "", "\u200c": "", "\u200d": "", "\u2060": "", "\u2061": "", "\u2062": "",
             "\u2063": "", "\u2064": "", "\ufeff": ""}   # U+2061-2064: invisible math operators
SPACES = {"\u00a0": " ", "\u2009": " ", "\u202f": " ", "\u2002": " ", "\u2003": " ", "\u2007": " "}
TABLE = str.maketrans({**QUOTES, **DASHES, **LIGATURES, **INVISIBLE, **SPACES})
TOKEN = re.compile(r"\w+(?:'(?!\w))?")
LINE_HYPHEN = re.compile(r"(\w)-\n(?=[a-zà-öø-ÿ])")


def clean(text):
    text = unicodedata.normalize("NFC", text).translate(TABLE)
    return re.sub(r"[ \t\r\f\v]+", " ", text)


def dehyphenate(text):
    """Join words broken at a line end (`contin-\\nuo`); declared, output side only."""
    return LINE_HYPHEN.sub(r"\1", text)


def score_tokens(text):
    return TOKEN.findall(clean(text).lower())


FINAL_ACCENTS = {"a": ("à",), "e": ("è", "é"), "i": ("ì",), "o": ("ò",), "u": ("ù",)}


def accepts(rules):
    """Function twin_token -> set of output tokens it accepts, for the twin rules of a document.

    `it-final-accent` (D19 only): a twin token ending with vowel + apostrophe (`abusivita'`,
    `e'`) also accepts the same word with that vowel accented (`abusività`, `è` or `é`). The rule
    widens twin tokens only; output tokens are never rewritten, so a converter that drops or
    changes an accent still fails.
    """
    def accept(token):
        alts = {token}
        if "it-final-accent" in rules and len(token) >= 2 and token.endswith("'") and token[-2] in FINAL_ACCENTS:
            alts |= {token[:-2] + v for v in FINAL_ACCENTS[token[-2]]}
        return alts
    return accept


def align_key(text):
    text = unicodedata.normalize("NFKD", clean(text))
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", re.sub(r"[^\w]+", " ", text.lower())).strip()


# --- formulas -----------------------------------------------------------------------------------

GREEK = {"alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ", "epsilon": "ε", "varepsilon": "ε", "zeta": "ζ",
         "eta": "η", "theta": "θ", "vartheta": "θ", "iota": "ι", "kappa": "κ", "lambda": "λ", "mu": "μ", "nu": "ν",
         "xi": "ξ", "pi": "π", "varpi": "π", "rho": "ρ", "varrho": "ρ", "sigma": "σ", "varsigma": "σ", "tau": "τ",
         "upsilon": "υ", "phi": "φ", "varphi": "φ", "chi": "χ", "psi": "ψ", "omega": "ω", "Gamma": "Γ",
         "Delta": "Δ", "Theta": "Θ", "Lambda": "Λ", "Xi": "Ξ", "Pi": "Π", "Sigma": "Σ", "Upsilon": "Υ",
         "Phi": "Φ", "Psi": "Ψ", "Omega": "Ω", "varGamma": "Γ", "varDelta": "Δ", "varTheta": "Θ", "varLambda": "Λ",
         "varXi": "Ξ", "varPi": "Π", "varSigma": "Σ", "varUpsilon": "Υ", "varPhi": "Φ", "varPsi": "Ψ", "varOmega": "Ω",
         "varkappa": "κ", "digamma": "ϝ"}
SYMBOLS = {"le": "≤", "leq": "≤", "ge": "≥", "geq": "≥", "neq": "≠", "ne": "≠", "times": "×", "cdot": "·",
           "pm": "±", "mp": "∓", "infty": "∞", "partial": "∂", "nabla": "∇", "sum": "∑", "prod": "∏",
           "int": "∫", "oint": "∮", "sqrt": "√", "in": "∈", "notin": "∉", "subset": "⊂", "subseteq": "⊆",
           "supset": "⊃", "cup": "∪", "cap": "∩", "to": "→", "rightarrow": "→", "leftarrow": "←",
           "Rightarrow": "⇒", "Leftrightarrow": "⇔", "mapsto": "↦", "approx": "≈", "sim": "∼", "simeq": "≃",
           "equiv": "≡", "propto": "∝", "forall": "∀", "exists": "∃", "emptyset": "∅", "varnothing": "∅",
           "ldots": "…", "cdots": "⋯", "dots": "…", "langle": "⟨", "rangle": "⟩", "hbar": "ℏ", "ell": "ℓ",
           "circ": "∘", "prime": "'", "mid": "|", "vert": "|", "lvert": "|", "rvert": "|", "|": "‖", "Vert": "‖",
           "setminus": "∖", "neg": "¬", "wedge": "∧", "vee": "∨", "oplus": "⊕", "otimes": "⊗", "ast": "*",
           "star": "⋆", "bullet": "•", "deg": "°", "lfloor": "⌊", "rfloor": "⌋", "lceil": "⌈", "rceil": "⌉",
           "{": "{", "}": "}", "%": "%", "#": "#", "&": "&", "_": "_", "$": "$",
           "lbrack": "[", "rbrack": "]", "lbrace": "{", "rbrace": "}", "backslash": "\\", "iint": "∬", "iiint": "∭",
           "iiiint": "⨌", "ll": "≪", "gg": "≫", "lll": "⋘", "ggg": "⋙", "leftrightarrow": "↔", "Leftarrow": "⇐",
           "longrightarrow": "⟶", "longleftarrow": "⟵", "Longrightarrow": "⟹", "Longleftarrow": "⟸",
           "longleftrightarrow": "⟷", "Longleftrightarrow": "⟺", "uparrow": "↑", "downarrow": "↓", "i": "ı", "j": "ȷ"}
WORD_COMMANDS = {"sin", "cos", "tan", "log", "ln", "exp", "lim", "max", "min", "sup", "inf", "det", "dim",
                 "deg", "arg", "ker", "gcd", "Pr", "mod", "bmod", "pmod", "sinh", "cosh", "tanh", "arctan",
                 "arcsin", "arccos", "cot", "sec", "csc", "lg", "limsup", "liminf"}
DROPPED = {"left", "right", "big", "Big", "bigg", "Bigg", "bigl", "bigr", "Bigl", "Bigr", "biggl", "biggr",
           "displaystyle", "textstyle", "scriptstyle", "limits", "nolimits", "quad", "qquad", "label",
           "nonumber", "notag", "tag", "mathrm", "mathbf", "mathit", "mathsf", "mathtt", "mathcal", "mathbb",
           "mathfrak", "boldsymbol", "bm", "operatorname", "text", "textrm", "textit", "textbf", "mbox",
           "hbox", "rm", "bf", "it", "cal", "mathnormal", "frac", "dfrac", "tfrac", "binom", "over",
           "hat", "widehat", "tilde", "widetilde", "bar", "overline", "underline", "vec", "dot", "ddot",
           "begin", "end", "array", "matrix", "pmatrix", "bmatrix", "vmatrix", "cases", "aligned", "align",
           "split", "gathered", "hspace", "vspace", "phantom", "hphantom", "vphantom", "mathop", "mathord",
           "mathrel", "mathbin", "stackrel", "overset", "underset", "substack", "nolinebreak", "allowbreak",
           "ensuremath", "overleftarrow", "overrightarrow", "overleftrightarrow", "underleftarrow",
           "underrightarrow", "overbrace", "underbrace"}
TEX_TOKEN = re.compile(r"\\([A-Za-z]+|.)|([A-Za-z])|(\d)|(\S)", re.S)   # `\<newline>`: a control space
ENV_NAME = re.compile(r"\\(?:begin|end)\{[^}]*\}")


NOT_PRINTED = re.compile(r"\\(?:label|tag\*?|hspace\*?|vspace\*?|mspace|phantom|hphantom|vphantom)\s*\{[^{}]*\}"
                         r"|(?<=\\\\)\s*\[[^\]]*\]|(?<!\\)%[^\n]*")


# One atom per printed glyph, whatever its encoding: the dot operator of MathML and OMML (U+22C5)
# prints as `\\cdot` (here the middle dot), and a degree written as a superscript ring in TeX
# (`^{\\circ}`, how Pandoc writes the MathML degree sign) prints as the degree sign.
SAME_GLYPH = {"\u22c5": "\u00b7"}
TEX_DEGREE = re.compile(r"\^\s*(?:\{\s*\\circ\s*\}|\\circ(?![A-Za-z]))")


def math_atoms(tex):
    """Atoms of a TeX formula: letters, digits, symbols, Greek letters and operators as characters.
    Labels, tags, spacing arguments (`\\[0.25cm]`, `\\mspace{9mu}`) and comments print nothing."""
    atoms = []
    tex = ENV_NAME.sub(" ", NOT_PRINTED.sub(" ", clean(tex)))
    tex = TEX_DEGREE.sub("\u00b0", tex)
    for cmd, letter, digit, other in TEX_TOKEN.findall(tex):
        if cmd:
            if cmd in GREEK:
                atoms.append(GREEK[cmd])
            elif cmd in SYMBOLS:
                atoms.append(SYMBOLS[cmd])
            elif cmd in WORD_COMMANDS:
                atoms.extend(cmd)
            else:   # formatting, spacing, structure and unknown macros carry no atom
                continue
        elif letter or digit:
            atoms.append(letter or digit)
        elif other and other not in "{}&~":
            atoms.append(SAME_GLYPH.get(other, other))
    return atoms


def unicode_atoms(text):
    """Atoms of linearized math written with Unicode characters (MathML text, pdftotext)."""
    out = []
    for ch in clean(text):
        if ch.isspace():
            continue
        out.append(SAME_GLYPH.get(ch, ch))
    return out


def formula_similarity(a, b):
    if not a and not b:
        return 1.0
    return difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()
