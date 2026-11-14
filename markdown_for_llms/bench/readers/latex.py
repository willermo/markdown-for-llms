"""LaTeX twin reader (own tokenizer, standard library only).

Comments, simple `\\newcommand`/`\\def` macros of the preamble, `\\newtheorem` with numbering,
sections, paragraphs (split by blank lines) with inline math, emphasis and footnotes, display
math environments, lists, figures (captions), `tabular`, and the bibliography of the `.bbl`
file, which replaces `\\bibliography{...}` exactly as in the self-check input. Pictures are
skipped. Author macros that this reader does not expand leave a formula `not determinable`
rather than a wrong check (declared limit). References and citations print numbers or names that
exist only after compilation: the reader drops them and counts one `gap` per key, which the
matcher accepts as two printed tokens. Placement options and width arguments of floats and boxes
are not text.
"""
import re
import unicodedata

from ..codes import BenchError, Code
from ..model import Builder, Cell

MATH_ENVS = {"equation", "equation*", "align", "align*", "eqnarray", "eqnarray*", "gather", "gather*", "multline",
             "multline*", "displaymath", "flalign", "flalign*", "alignat", "alignat*", "math"}
LIST_ENVS = {"itemize", "enumerate", "description"}
SKIP_ENVS = {"picture", "tikzpicture", "pspicture", "comment", "titlepage", "thebibliography-skip", "filecontents"}
RECURSE_ENVS = {"center", "flushleft", "flushright", "minipage", "quote", "quotation", "appendices", "small",
                "footnotesize", "document", "sloppypar", "landscape", "raggedright", "spacing", "singlespace"}
SECTIONS = {"part": 1, "chapter": 1, "section": 2, "subsection": 3, "subsubsection": 4, "paragraph": 5, "subparagraph": 6}
DROP_WITH_ARG = {"label", "ref", "eqref", "pageref", "cite", "citep", "citet", "citealt", "citeauthor", "citeyear",
                 "nocite", "cref", "Cref", "autoref", "vspace", "hspace", "vspace*", "hspace*", "setcounter",
                 "addtocounter", "bibliographystyle", "includegraphics", "input", "include", "pagestyle",
                 "thispagestyle", "affiliation", "author", "email", "date", "address", "keywords", "pacs",
                 "counterwithin", "numberwithin", "graphicspath", "hypersetup", "titleformat", "titlespacing",
                 "renewcommand", "newcommand", "BibitemShut", "natexlab-drop", "doibase", "urlprefix", "linespread",
                 "addcontentsline", "markboth", "markright", "color", "textcolor-drop", "bibfield-name"}
KEEP_ARG = {"emph", "textit", "textbf", "textsc", "texttt", "textrm", "textsf", "textnormal", "text", "mbox", "hbox",
            "url", "underline", "textup", "textmd", "uppercase", "MakeUppercase", "enquote", "bibnamefont",
            "bibfnamefont", "natexlab", "selectlanguage-drop", "mathrm", "textsuperscript", "textsubscript", "fbox"}
SYMBOL_COMMANDS = {"ldots": "…", "dots": "…", "S": "§", "P": "¶", "copyright": "©", "ss": "ß", "ae": "æ", "AE": "Æ",
                   "oe": "œ", "OE": "Œ", "o": "ø", "O": "Ø", "l": "ł", "L": "Ł", "i": "ı", "aa": "å", "AA": "Å",
                   "textendash": "–", "textemdash": "—", "textquoteright": "'", "textquoteleft": "'", "&": "&",
                   "%": "%", "$": "$", "#": "#", "_": "_", "{": "{", "}": "}", "textbackslash": "\\", "dag": "†",
                   "ddag": "‡", "textregistered": "®", "LaTeX": "LaTeX", "TeX": "TeX", "newblock": " ", "par": "\n\n",
                   "\\": " ", ",": " ", ";": " ", ":": " ", "!": "", " ": " ", "quad": " ", "qquad": " ",
                   "noindent": "", "indent": "", "maketitle": "", "centering": "", "item": "", "hline": "",
                   "BibitemOpen": "", "bibAnnoteFile-drop": "", "smallskip": "", "medskip": "", "bigskip": "",
                   "newpage": "", "clearpage": "", "FloatBarrier": "", "tableofcontents": "", "appendix": "",
                   "protect": "", "relax": "", "frenchspacing": "", "sloppy": "", "raggedbottom": "",
                   "flushbottom": "", "nobreak": "", "linebreak": " ", "allowbreak": ""}
ACCENTS = {"'": "\u0301", "`": "\u0300", "^": "\u0302", '"': "\u0308", "~": "\u0303", "=": "\u0304", ".": "\u0307",
           "c": "\u0327", "v": "\u030c", "u": "\u0306", "H": "\u030b", "k": "\u0328", "r": "\u030a"}
DROP_TWO_ARGS = {"renewcommand", "newcommand", "providecommand", "setcounter", "addtocounter", "setlength",
                 "addtolength", "newenvironment-drop"}
EMPH = {"emph": "i", "textit": "i", "textbf": "b", "em": "i", "it": "i", "bf": "b", "itshape": "i", "bfseries": "b"}
REF_COMMANDS = {"ref", "eqref", "cref", "Cref", "autoref", "pageref", "cite", "citep", "citet", "citealt", "citeauthor",
                "citeyear"}
# environment -> (optional arguments, mandatory arguments) that precede the content and print nothing
ENV_ARGS = {"figure": (1, 0), "table": (1, 0), "wrapfigure": (1, 2), "wraptable": (1, 2), "minipage": (3, 1),
            "subfigure": (1, 1), "subtable": (1, 1)}


def strip_comments(tex):
    return re.sub(r"(?<!\\)%.*", "", tex)


def group(s, i):
    """(content, index after) of the {...} group starting at s[i] (after spaces)."""
    while i < len(s) and s[i] in " \t\n":
        i += 1
    if i >= len(s) or s[i] != "{":
        if i < len(s) and s[i] == "\\":
            m = re.match(r"\\([A-Za-z@]+|.)", s[i:], re.S)      # `\<newline>` is a control space
            return m.group(0), i + len(m.group(0))
        return (s[i], i + 1) if i < len(s) else ("", i)
    depth, j = 0, i
    while j < len(s):
        ch = s[j]
        if ch == "\\":
            j += 2
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return s[i + 1:j], j + 1
        j += 1
    raise BenchError(Code.TWIN_UNREADABLE, "unbalanced braces")


def optional(s, i):
    j = i
    while j < len(s) and s[j] in " \t":
        j += 1
    if j < len(s) and s[j] == "[":
        depth, k = 0, j
        while k < len(s):
            if s[k] == "[":
                depth += 1
            elif s[k] == "]":
                depth -= 1
                if depth == 0:
                    return s[j + 1:k], k + 1
            elif s[k] == "{":
                _, k = group(s, k)
                continue
            k += 1
    return None, i


class Macros:
    def __init__(self, preamble):
        self.defs = {}
        self.theorems = {}            # env -> (title, counter name)
        for m in re.finditer(r"\\(?:re)?newcommand\*?\s*", preamble):
            i = m.end()
            if i < len(preamble) and preamble[i] == "{":
                name, i = group(preamble, i)
            else:
                nm = re.match(r"\\[A-Za-z]+", preamble[i:])
                if not nm:
                    continue
                name, i = nm.group(0), i + nm.end()
            nargs, i = optional(preamble, i)
            _, i = optional(preamble, i)
            body, _ = group(preamble, i)
            if name.startswith("\\") and name[1:].isalpha():
                self.defs[name[1:]] = (int(nargs) if nargs and nargs.isdigit() else 0, body)
        for m in re.finditer(r"\\def\s*\\([A-Za-z]+)\s*(?=\{)", preamble):
            body, _ = group(preamble, m.end())
            self.defs[m.group(1)] = (0, body)
        for m in re.finditer(r"\\DeclareMathOperator\*?\s*\{\\([A-Za-z]+)\}\s*\{([^}]*)\}", preamble):
            self.defs[m.group(1)] = (0, "\\operatorname{" + m.group(2) + "}")
        for m in re.finditer(r"\\newtheorem\*?\s*\{(\w+)\}\s*(?:\[(\w+)\])?\s*\{([^}]*)\}", preamble):
            env, shared, title = m.group(1), m.group(2), m.group(3)
            self.theorems[env] = (title, self.theorems.get(shared, (None, shared))[1] if shared else env)

    def expand(self, s, depth=0):
        if not self.defs or depth > 8:
            return s
        names = "|".join(sorted(map(re.escape, self.defs), key=len, reverse=True))
        pattern = re.compile(r"\\(" + names + r")(?![A-Za-z])")
        out, i, changed = [], 0, False
        for m in pattern.finditer(s):
            if m.start() < i:
                continue
            nargs, body = self.defs[m.group(1)]
            j, args = m.end(), []
            try:
                for _ in range(nargs):
                    a, j = group(s, j)
                    args.append(a)
            except BenchError:
                continue
            for k, a in enumerate(args, 1):
                body = body.replace("#" + str(k), a)
            out.append(s[i:m.start()])
            out.append(body if nargs or not body[:1].isalpha() else " " + body)
            i, changed = j, True
        out.append(s[i:])
        result = "".join(out)
        return self.expand(result, depth + 1) if changed else result


class Inline:
    """Plain text of LaTeX inline material, with inline math, emphasis runs and footnotes."""

    def __init__(self):
        self.math, self.emph, self.notes = [], [], []
        self.gaps = 0

    def convert(self, s, mode=""):
        out, i = [], 0
        while i < len(s):
            ch = s[i]
            if ch == "\\":
                if s.startswith("\\(", i):
                    j = s.find("\\)", i + 2)
                    j = len(s) if j < 0 else j
                    self.math.append(s[i + 2:j])
                    out.append(" ")
                    i = j + 2
                    continue
                m = re.match(r"\\([A-Za-z@]+\*?|.)", s[i:], re.S)
                name = m.group(1)
                i += len(m.group(0))
                if name in ACCENTS and name not in ("c", "v", "u", "H", "k", "r") or (name in ACCENTS and re.match(r"\s*[{A-Za-z]", s[i:])):
                    arg, i = group(s, i)
                    base = self.convert(arg)
                    out.append(unicodedata.normalize("NFC", base[:1] + ACCENTS[name] + base[1:]) if base else "")
                elif name in SYMBOL_COMMANDS:
                    out.append(SYMBOL_COMMANDS[name])
                elif name in ("footnote", "thanks", "footnotetext"):
                    _, i = optional(s, i)
                    arg, i = group(s, i)
                    sub = Inline()
                    self.notes.append(re.sub(r"\s+", " ", sub.convert(arg)).strip())
                elif name in ("bibfield", "bibinfo"):
                    _, i = group(s, i)
                    arg, i = group(s, i)
                    out.append(self.convert(arg))
                elif name in ("href", "href@noop", "Eprint"):
                    _, i = group(s, i)
                    arg, i = group(s, i)
                    out.append(self.convert(arg))
                elif name in ("textcolor",):
                    _, i = group(s, i)
                    arg, i = group(s, i)
                    out.append(self.convert(arg))
                elif name in DROP_TWO_ARGS:
                    _, i = optional(s, i)
                    _, i = group(s, i)
                    _, i = optional(s, i)
                    _, i = optional(s, i)
                    _, i = group(s, i)
                elif name in DROP_WITH_ARG or name.rstrip("*") in DROP_WITH_ARG:
                    _, i = optional(s, i)
                    _, i = optional(s, i)
                    if i < len(s) and s[i:].lstrip(" ").startswith("{"):
                        keys, i = group(s, i)
                        if name.rstrip("*") in REF_COMMANDS:
                            self.gaps += len([k for k in keys.split(",") if k.strip()])
                elif name in KEEP_ARG or name in EMPH:
                    if name in ("em", "it", "bf", "itshape", "bfseries"):
                        continue
                    arg, i = group(s, i)
                    text = self.convert(arg)
                    if name in EMPH and text.strip():
                        self.emph.append((EMPH[name], re.sub(r"\s+", " ", text).strip()))
                    out.append(text)
                elif name in ("multicolumn",):
                    _, i = group(s, i)
                    _, i = group(s, i)
                    arg, i = group(s, i)
                    out.append(self.convert(arg))
                elif re.fullmatch(r"[A-Za-z@]+\*?", name):
                    j = i
                    while j < len(s) and s[j] == " ":
                        j += 1
                    if j < len(s) and s[j] == "{":
                        arg, i = group(s, j)
                        out.append(self.convert(arg))
                    else:
                        out.append(" ")
                else:
                    out.append(" ")
            elif ch == "$":
                if s.startswith("$$", i):
                    j = s.find("$$", i + 2)
                    j = len(s) if j < 0 else j
                    self.math.append(s[i + 2:j])
                    i = j + 2
                else:
                    j = i + 1
                    while j < len(s) and (s[j] != "$" or s[j - 1] == "\\"):
                        j += 1
                    self.math.append(s[i + 1:j])
                    i = j + 1
                out.append(" ")
            elif ch == "{":
                arg, j = group(s, i)
                m = re.match(r"\s*\\(em|it|bf|itshape|bfseries|sc|tt|rm|sf|small|footnotesize|large|Large)\b", arg)
                text = self.convert(arg[m.end():] if m else arg)
                if m and m.group(1) in EMPH and text.strip():
                    self.emph.append((EMPH[m.group(1)], re.sub(r"\s+", " ", text).strip()))
                out.append(text)
                i = j
            elif ch in "}":
                i += 1
            elif ch == "~":
                out.append(" ")
                i += 1
            elif s.startswith("---", i):
                out.append("—")
                i += 3
            elif s.startswith("--", i):
                out.append("–")
                i += 2
            elif s.startswith("``", i) or s.startswith("''", i):
                out.append('"')
                i += 2
            elif ch == "&":
                out.append(" ")
                i += 1
            else:
                out.append(ch)
                i += 1
        return "".join(out)


def _clean(text):
    return re.sub(r"\s+", " ", text).strip()


class Reader:
    def __init__(self, tex, bbl):
        tex = strip_comments(tex)
        bbl = strip_comments(bbl)
        tex = re.sub(r"\\bibliography\s*\{[^}]*\}", lambda _: "\n" + bbl.replace("\\", "\\") + "\n", tex, count=1)
        pre, sep, body = tex.partition("\\begin{document}")
        if not sep:
            raise BenchError(Code.TWIN_UNREADABLE, "no document environment")
        body = body.split("\\end{document}")[0]
        self.macros = Macros(pre)
        self.preamble = pre
        self.body = self.macros.expand(body)
        self.b = Builder("latex")
        self.counters = {}

    def run(self):
        title_src = re.search(r"\\title\s*(\[[^\]]*\])?\s*\{", self.preamble + self.body)
        if title_src:
            text, _ = group(self.preamble + self.body, title_src.end() - 1)
            inline = Inline()
            title = _clean(inline.convert(self.macros.expand(text)))
            if title:
                self.b.add("heading", title, level=1)
        self.blocks(re.sub(r"\\title\s*(\[[^\]]*\])?\s*\{", r"\\titledrop{", self.body))
        return self.b.twin

    def paragraph(self, text, prefix=""):
        for chunk in re.split(r"\n\s*\n|\\par\b", text):
            inline = Inline()
            plain = _clean(prefix + " " + inline.convert(chunk) if prefix else inline.convert(chunk))
            prefix = ""
            if plain or inline.math:
                if not plain and len(inline.math) == 1:
                    self.b.add("formula", inline.math[0], display=True)
                elif plain:
                    self.b.add("paragraph", plain, math=tuple(inline.math), emphasis=tuple(inline.emph), gaps=inline.gaps)
            for note in inline.notes:
                if note:
                    self.b.add("footnote", note)

    def blocks(self, s, prefix=""):
        i, text_start = 0, 0
        pattern = re.compile(r"\\begin\s*\{([^}]+)\}|\\(part|chapter|section|subsection|subsubsection|paragraph|subparagraph)\*?\s*(?=[\[{])|\\\[|\$\$|\\(caption|titledrop|bibitem)\b|\\item\b")
        while True:
            m = pattern.search(s, i)
            if not m:
                self.paragraph(s[text_start:], prefix)
                return
            self.paragraph(s[text_start:m.start()], prefix)
            prefix = ""
            if m.group(1):
                env = m.group(1).strip()
                end = self._env_end(s, m.end(), env)
                content = s[m.end():end]
                i = text_start = end + len(f"\\end{{{env}}}")
                self.environment(env, content)
            elif m.group(2):
                _, j = optional(s, m.end())
                title, j = group(s, j)
                inline = Inline()
                text = _clean(inline.convert(title))
                if text:
                    self.b.add("heading", text, level=SECTIONS[m.group(2)], gaps=inline.gaps)
                i = text_start = j
            elif m.group(0) in ("\\[", "$$"):
                close = "\\]" if m.group(0) == "\\[" else "$$"
                j = s.find(close, m.end())
                j = len(s) if j < 0 else j
                self.b.add("formula", s[m.end():j], display=True)
                i = text_start = j + len(close)
            elif m.group(3) == "titledrop":
                _, j = group(s, m.end())
                i = text_start = j
            elif m.group(3) == "caption":
                _, j = optional(s, m.end())
                cap, j = group(s, j)
                inline = Inline()
                self.b.add("figure", _clean(inline.convert(cap)))
                i = text_start = j
            elif m.group(3) == "bibitem":
                _, j = optional(s, m.end())
                _, j = group(s, j)
                nxt = re.compile(r"\\bibitem\b|\\end\s*\{thebibliography\}").search(s, j)
                k = nxt.start() if nxt else len(s)
                inline = Inline()
                text = _clean(inline.convert(s[j:k]))
                if text:
                    self.b.add("bib_entry", text)
                i = text_start = k
            else:   # \item outside a list environment: plain text
                i = text_start = m.end()

    def _env_end(self, s, start, env):
        depth, pos = 1, start
        token = re.compile(r"\\(begin|end)\s*\{" + re.escape(env) + r"\}")
        while True:
            m = token.search(s, pos)
            if not m:
                return len(s)
            depth += 1 if m.group(1) == "begin" else -1
            if depth == 0:
                return m.start()
            pos = m.end()

    def environment(self, env, content):
        base = env.rstrip("*")
        if base in ENV_ARGS:
            n_opt, n_args = ENV_ARGS[base]
            j = 0
            for _ in range(n_opt):
                _, j = optional(content, j)
            for _ in range(n_args):
                _, j = group(content, j)
            content = content[j:]
        if env in MATH_ENVS:
            self.b.add("formula", content, display=True)
        elif env in SKIP_ENVS:
            self.b.count("skipped_" + base)
        elif env in LIST_ENVS:
            self.list_items(content, 1)
        elif env == "abstract":
            self.blocks(content)
        elif env == "thebibliography":
            first = re.search(r"\\bibitem\b", content)   # skip the width and any \providecommand
            self.blocks(content[first.start():] if first else "")
        elif base in ("figure", "wrapfigure", "subfigure"):
            self.blocks(content)
        elif base == "table":
            self.blocks(content)
        elif base in ("tabular", "tabularx", "longtable"):
            self.tabular(content)
        elif env in self.macros.theorems:
            title, counter = self.macros.theorems[env]
            self.counters[counter] = self.counters.get(counter, 0) + 1
            opt, j = optional(content, 0)
            label = f"{title} {self.counters[counter]}" + (f" ({_clean(Inline().convert(opt))})" if opt else "") + "."
            self.blocks(content[j:], prefix=label)
        elif base == "proof":
            opt, j = optional(content, 0)
            self.blocks(content[j:], prefix=(_clean(Inline().convert(opt)) if opt else "Proof") + ".")
        elif base in ("verbatim", "lstlisting"):
            self.b.add("paragraph", _clean(content))
        else:   # center, minipage, appendices and any other text environment
            self.blocks(content)

    def list_items(self, content, depth):
        parts, level, start = [], 0, None
        for m in re.finditer(r"\\(begin|end)\s*\{[^}]*\}|\\item\b", content):
            if m.group(1) == "begin":
                level += 1
            elif m.group(1) == "end":
                level -= 1
            elif level == 0:
                if start is not None:
                    parts.append(content[start:m.start()])
                start = m.end()
        if start is not None:
            parts.append(content[start:])
        for part in parts:
            label, j = optional(part, 0)
            nested = re.search(r"\\begin\s*\{(itemize|enumerate|description)\}", part)
            text = part[j:nested.start()] if nested else part[j:]
            inline = Inline()
            plain = _clean((Inline().convert(label) + " " if label else "") + inline.convert(text))
            if plain:
                self.b.add("list_item", plain, level=depth, math=tuple(inline.math), emphasis=tuple(inline.emph),
                           gaps=inline.gaps)
            if nested:
                end = self._env_end(part, nested.end(), nested.group(1))
                self.list_items(part[nested.end():end], depth + 1)

    def tabular(self, content):
        _, j = group(content, 0)
        cells = []
        for r, row in enumerate(x for x in re.split(r"\\\\", content[j:]) if _clean(re.sub(r"\\[hc]line(\{[^}]*\})?", "", x))):
            row = re.sub(r"\\(hline|toprule|midrule|bottomrule|cline\{[^}]*\})", "", row)
            c = 0
            for cell in row.split("&"):
                span = 1
                mm = re.match(r"\s*\\multicolumn\s*\{(\d+)\}", cell)
                if mm:
                    span = int(mm.group(1))
                cells.append(Cell(_clean(Inline().convert(cell)), r, c, 1, span, r == 0))
                c += span
        self.b.add("table", cells=tuple(cells), level=1)


def _source(data):
    """A TeX source as text with LF line ends: a carriage return ends a line in TeX as a line
    feed does (D02: a CRLF `.bbl` broke the argument scan after `\\bibinfo` and `\\href`)."""
    return data.decode("utf-8", "replace").replace("\r\n", "\n").replace("\r", "\n")


def read_latex(tex_bytes, bbl_bytes):
    return Reader(_source(tex_bytes), _source(bbl_bytes)).run()


def self_check_source(tex_bytes, bbl_bytes):
    """The `.tex` with the `.bbl` in place of `\\bibliography{...}`: the input of pandoc."""
    tex = tex_bytes.decode("utf-8", "replace")
    bbl = bbl_bytes.decode("utf-8", "replace")
    return re.sub(r"\\bibliography\s*\{[^}]*\}", lambda _: "\n" + bbl + "\n", tex, count=1)
