"""Wikisource twin reader: the wikitext of each `Pagina:` revision (SAL 100 % pages only).

Headers and footers (`noinclude`, `RigaIntestazione`, `PieDiPagina`) are noise; `<ref>` are
footnotes; `''`/`'''` are emphasis; `<math>` is a formula; wiki tables are tables; `{{FI}}` is a
figure with its caption. Templates follow a closed list; an unknown template drops its text and
is counted (`unknown_templates`). `{{Pt|a|b}}` yields the whole word `b` once, on the page that
carries it. `{{Ec|printed|corrected}}` keeps the printed form, which is what the scan shows.
Block templates (`{{Centrato}}`, `{{Ct}}`, `{{Indent}}`, `{{Smaller block}}`, `{{Rule}}`: a `div` in
the rendering of MediaWiki) end the paragraph around them.
"""
import html
import re

from ..model import Builder, Cell

# template -> how to expand its positional parameters (names are matched case-insensitively)
FIRST = {"sc", "ct", "centrato", "tc", "smaller", "x-smaller", "larger", "a destra", "testoassente", "ec", "sup",
         "sub", "type", "blackletter", "ac2"}
SECOND_OR_FIRST = {"autorecitato", "ac", "wl", "wi", "pg", "testocitato", "§", "indent"}
EMPTY = {"nop", "rule", "ruleleft", "spazi", "cs", "indentinverso", "no rientro", "smaller block", "vi spazio",
         "clear", "-"}
NOISE = {"rigaintestazione", "piedipagina"}
BLOCK = {"centrato", "ct", "indent", "smaller block", "rule", "ruleleft"}
TEMPLATE = re.compile(r"\{\{((?:(?!\{\{|\}\}).)*)\}\}", re.S)
LINK = re.compile(r"\[\[([^\[\]|]*)(?:\|([^\[\]]*))?\]\]")
EXTLINK = re.compile(r"\[(?:https?:)?//\S+\s+([^\]]*)\]")
REF = re.compile(r"<ref(?:\s[^>/]*)?>(.*?)</ref>|<ref\s[^>]*/>", re.S | re.I)
MARK = "\x00"


def _params(body):
    parts = body.split("|")
    name = parts[0].strip()
    positional, named = [], {}
    for part in parts[1:]:
        key, eq, value = part.partition("=")
        if eq and re.fullmatch(r"\s*[\w ]+\s*", key) and not key.strip().isdigit():
            named[key.strip().lower()] = value.strip()
        else:
            positional.append(part.strip())
    return name, positional, named


class _Expander:
    def __init__(self, builder):
        self.b = builder
        self.figures, self.noise, self.notes = [], [], []

    def template(self, match):
        text = self.inline(match)
        name = _params(match.group(1))[0].lower().replace("_", " ")
        return "\n\n" + text + "\n\n" if name in BLOCK else text

    def inline(self, match):
        name, pos, named = _params(match.group(1))
        key = name.lower().replace("_", " ")
        if key in EMPTY:
            return " " if key == "spazi" else ""
        if key in NOISE:
            self.noise.append(" ".join(p for p in pos if p))
            return f"{MARK}Z{len(self.noise) - 1}{MARK}"
        if key == "pt":
            return pos[1] if len(pos) > 1 and pos[1] else ""
        if key == "fi":
            self.figures.append(named.get("caption", ""))
            return f"{MARK}F{len(self.figures) - 1}{MARK}"
        if key == "vi":
            return f"{named.get('titolo', '')} {named.get('pagina', '')}"
        if key in FIRST:
            return pos[0] if pos else ""
        if key in SECOND_OR_FIRST:
            return pos[1] if len(pos) > 1 and pos[1] else (pos[0] if pos else "")
        self.b.count("unknown_templates")
        self.b.count("unknown_template_chars", len(match.group(0)))
        return ""

    def expand(self, text):
        previous = None
        while previous != text:
            previous = text
            text = TEMPLATE.sub(self.template, text)
        return text


def _strip_tags(text):
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    text = re.sub(r"<includeonly>.*?</includeonly>", "", text, flags=re.S | re.I)
    text = re.sub(r"<nowiki>(.*?)</nowiki>", r"\1", text, flags=re.S | re.I)
    text = re.sub(r"<br\s*/?>", " ", text, flags=re.I)
    text = re.sub(r"<(pagequality|section|references)\b[^>]*/?>", "", text, flags=re.I)
    text = re.sub(r"</?(span|div|big|small|sup|sub|center|poem|u|s|font|p|blockquote)\b[^>]*>", "", text, flags=re.I)
    return html.unescape(text)


def _emphasis(text):
    runs = []
    for m in re.finditer(r"'''''(.+?)'''''|'''(.+?)'''|''(.+?)''", text):
        if m.group(1):
            runs.append(("b", m.group(1)))
        elif m.group(2):
            runs.append(("b", m.group(2)))
        else:
            runs.append(("i", m.group(3)))
    plain = re.sub(r"'{2,5}", "", text)
    return plain, tuple((k, re.sub(r"'{2,5}", "", v).strip()) for k, v in runs if re.sub(r"'{2,5}", "", v).strip())


def _clean(text):
    return re.sub(r"\s+", " ", text).strip()


def _table(lines, b):
    cells, row, col = [], -1, 0
    for line in lines[1:]:
        s = line.strip()
        if s.startswith("|}"):
            break
        if s.startswith("|-"):
            row, col = row + 1, 0
            continue
        if s.startswith("|+"):
            continue
        if s.startswith("|") or s.startswith("!"):
            if row < 0:
                row = 0
            header = s.startswith("!")
            for raw in re.split(r"\|\||!!", s[1:]):
                attrs, sep, content = raw.rpartition("|")
                if not sep or "[[" in attrs:
                    attrs, content = "", raw
                span_c = re.search(r'colspan\s*=\s*"?(\d+)', attrs)
                span_r = re.search(r'rowspan\s*=\s*"?(\d+)', attrs)
                cells.append(Cell(_clean(_emphasis(content)[0]), row, col, int(span_r.group(1)) if span_r else 1,
                                  int(span_c.group(1)) if span_c else 1, header))
                col += int(span_c.group(1)) if span_c else 1
        elif cells:
            last = cells[-1]
            cells[-1] = Cell(_clean(last.text + " " + _emphasis(s)[0]), last.row, last.col, last.rowspan, last.colspan, last.header)
    if cells:
        b.add("table", cells=tuple(c for c in cells if True), level=0)


def read_page(wikitext, page, b):
    """Elements of one page; the builder's page prefix gives positions like `pg3/p2`."""
    b.page = page
    exp = _Expander(b)
    head, body, foot = "", wikitext, ""
    m = re.match(r"\s*<noinclude>(.*?)</noinclude>(.*)", body, re.S)
    if m:
        head, body = m.group(1), m.group(2)
    m = re.match(r"(.*)<noinclude>(.*?)</noinclude>\s*$", body, re.S)
    if m:
        body, foot = m.group(1), m.group(2)
    for part in (head, foot):
        part = REF.sub("", part)
        expanded = exp.expand(LINK.sub(lambda x: x.group(2) or x.group(1), part))
        for i, noise in enumerate(exp.noise):
            if _clean(noise):
                b.add("noise", _clean(_strip_tags(noise)))
        exp.noise = []
        _ = expanded
    notes = []

    def ref(match):
        if match.group(1) is not None and match.group(1).strip():
            notes.append(match.group(1))
        return ""
    body = REF.sub(ref, body)
    body = LINK.sub(lambda x: (x.group(2) if x.group(2) is not None else x.group(1).split(":")[-1]), body)
    body = EXTLINK.sub(r"\1", body)
    body = _strip_tags(exp.expand(body))
    blocks, current = [], []
    lines = body.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.strip().startswith("{|"):
            if current:
                blocks.append(("text", "\n".join(current)))
                current = []
            j = i
            while j < len(lines) and not lines[j].strip().startswith("|}"):
                j += 1
            blocks.append(("table", lines[i:j + 1]))
            i = j + 1
            continue
        if not line.strip():
            if current:
                blocks.append(("text", "\n".join(current)))
                current = []
        else:
            current.append(line)
        i += 1
    if current:
        blocks.append(("text", "\n".join(current)))
    for kind, value in blocks:
        if kind == "table":
            _table(value, b)
            continue
        for piece in re.split(f"({MARK}[FZ]\\d+{MARK})", value):
            marker = re.fullmatch(f"{MARK}([FZ])(\\d+){MARK}", piece)
            if marker:
                if marker.group(1) == "F":
                    b.add("figure", _clean(_emphasis(_strip_tags(exp.figures[int(marker.group(2))]))[0]))
                else:
                    b.add("noise", _clean(_strip_tags(exp.noise[int(marker.group(2))])))
                continue
            math = tuple(re.findall(r"<math[^>]*>(.*?)</math>", piece, re.S))
            text = re.sub(r"<math[^>]*>.*?</math>", " ", piece, flags=re.S)
            stripped = re.sub(r"^[:*#;]+", "", text.strip())
            plain, emph = _emphasis(stripped)
            plain = _clean(plain)
            if not plain and len(math) == 1:
                b.add("formula", math[0], display=True)
            elif plain:
                kind_ = "list_item" if re.match(r"^[*#]", text.strip()) else "paragraph"
                b.add(kind_, plain, level=1 if kind_ == "list_item" else 0, math=math, emphasis=emph)
    for note in notes:
        expanded = _strip_tags(exp.expand(note))
        plain, _ = _emphasis(expanded)
        if _clean(plain):
            b.add("footnote", _clean(plain))
    b.page = 0


def read_wikisource(pages, quality, isolated=()):
    """`pages`: {page: revision record}; `quality`: {page: SAL level}; SAL 100 % pages only."""
    b = Builder("wikisource")
    for page in sorted(pages):
        if quality.get(page) != 4 or page in isolated:
            b.count("pages_not_checked")
            continue
        read_page(pages[page]["wikitext"], page, b)
    return b.twin
