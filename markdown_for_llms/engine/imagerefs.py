"""Image references of a converted document: Markdown images and HTML `<img>` tags (MFL043).

A converter writes an image in two forms: `![alt](name)`, and `<img src="name" alt="..." ...>`
(the HTML that Datalab writes for figures, also inside table cells). Both are read with the
parser and the profile of the product's reference check (`references.inspect`: commonmark,
tables, raw HTML, dollarmath), never with a regular expression over the whole text:

- a Markdown image is found by its `](name)` occurrences, whose count must equal the image
  tokens of the parse with that source;
- an `<img>` of an HTML token is found in order inside the source lines of its block (the
  `map` of the token), skipping code spans, inline math and backslash escapes as the parser
  does; every `<img>` of the parse must be found, and its `src` read again from the source as
  `html.parser` reads it must be the same value.

Otherwise the document is `output_invalid`: code is never touched. A rewrite changes only the
value of `src`, its quotes kept (`"` added to an unquoted value); `alt`, `data-bbox` and the rest
of the tag stay byte for byte. An `<img>` whose source is not an extracted image follows the
partition of the contract filter (`filters/contract.lua`): a well-formed http(s) URL stays,
never fetched (`media_external`); any other source is replaced by the alternative text of the
tag, HTML-escaped, with `|` written `&#124;` and line breaks as spaces so that a table row stays
one row (`media_missing`). Assets are numbered by the first occurrence of their name in either
form.
"""
import collections
import html
from html import parser as html_parser
import re
from urllib.parse import unquote, urlsplit

from . import contract
from .codes import Code, EngineError

LINE = re.compile(r"\r\n|\r|\n")
PUNCTUATION = frozenset("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~")


def markdown_parser():
    """The parser of the product's reference profile (`references.inspect`)."""
    from markdown_it import MarkdownIt
    from mdit_py_plugins.dollarmath import dollarmath_plugin
    from ..references import SCOPE
    md = MarkdownIt("commonmark", {"html": True}).enable("table").use(dollarmath_plugin, **SCOPE["math"])
    md.validateLink = lambda value: True        # tokens are never rendered or fetched
    return md


class _Images(html_parser.HTMLParser):
    """The `<img>` start tags of an HTML fragment: (raw tag text, attributes)."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.found = []

    def handle_starttag(self, tag, attrs):
        if tag == "img":
            self.found.append((self.get_starttag_text(), attrs))

    handle_startendtag = handle_starttag


def _images(fragment):
    p = _Images()
    p.feed(fragment)
    p.close()
    return p.found


def _src_span(raw):
    """(start, end) of the value of the first `src` attribute in the raw start tag, quotes
    included, and the value as `html.parser` reads it; (None, None, None) without `src`."""
    m = html_parser.tagfind_tolerant.match(raw, 1)
    k = m.end() if m else len(raw)
    while k < len(raw):
        m = html_parser.attrfind_tolerant.match(raw, k)
        if not m or m.end() == k:
            break
        name, rest, value = m.group(1, 2, 3)
        if name.lower() == "src":
            if not rest:
                return m.end(1), m.end(1), None
            start, end = m.span(3)
            if value[:1] == value[-1:] and value[:1] in ("'", '"') and len(value) >= 2:
                value = value[1:-1]
            return start, end, html.unescape(value) if value else value
        k = m.end()
    return None, None, None


def _backticks(text, i, end):
    n = i
    while n < end and text[n] == "`":
        n += 1
    return n - i


def _escaped(text, i):
    """An odd number of backslashes before position `i` (the test of dollarmath)."""
    n, j = 0, i - 1
    while j >= 0 and text[j] == "\\":
        n, j = n + 1, j - 1
    return n % 2 == 1


def _math_end(text, i, end):
    """The end of the inline math opened by the `$` at `i` (single or double), or None."""
    if _escaped(text, i):
        return None
    double = text.startswith("$$", i)
    pos = i + (2 if double else 1)
    while True:
        close = text.find("$", pos, end)
        if close < 0:
            return None
        if _escaped(text, close):
            pos = close + 1
            continue
        if double and not text.startswith("$", close + 1):
            pos = close + 1
            continue
        if close == i + (2 if double else 1):
            return None                                  # empty math is not math
        return close + (2 if double else 1)


def _inline_positions(text, start, end, expected):
    """The positions of the raw HTML tags `expected` (the html_inline tokens of one inline
    token, in order) in `text[start:end]`, skipping escapes, code spans and inline math."""
    out, i, j = [], start, 0
    while j < len(expected) and i < end:
        c = text[i]
        if c == "\\" and i + 1 < end and text[i + 1] in PUNCTUATION:
            i += 2
        elif c == "<" and text.startswith(expected[j], i) and i + len(expected[j]) <= end:
            out.append(i)
            i += len(expected[j])
            j += 1
        elif c == "`":
            run = _backticks(text, i, end)
            close, k = -1, i + run
            while True:
                k = text.find("`", k, end)
                if k < 0:
                    break
                found = _backticks(text, k, end)
                if found == run:
                    close = k
                    break
                k += found
            i = close + run if close >= 0 else i + run
        elif c == "$":
            stop = _math_end(text, i, end)
            i = stop if stop is not None else i + 1
        else:
            i += 1
    if j < len(expected):
        raise EngineError(Code.OUTPUT_INVALID, "an HTML tag of the parse not found in its block")
    return out


class Refs:
    """The image references of a document: `markdown` {src: count of image tokens}, `html`
    [(start, end, src start, src end, src, alt)] in order, `first` {src: first offset}."""

    def __init__(self, markdown, tags, first):
        self.markdown, self.html, self.first = markdown, tags, first

    def used(self):
        return set(self.markdown) | {t[4] for t in self.html if t[4]}


def scan(text):
    """The image references of `text` (see the module docstring); `output_invalid` when an
    `<img>` of the parse cannot be placed in the source."""
    starts = [0] + [m.end() for m in LINE.finditer(text)]
    starts.append(len(text))

    def region(token_map):
        lo = starts[min(token_map[0], len(starts) - 1)]
        hi = starts[min(token_map[1], len(starts) - 1)]
        return lo, hi

    markdown, tags, cursor = collections.Counter(), [], 0

    def place(raw, attrs, at):
        s, e, src = _src_span(raw)
        reported = next((v for k, v in attrs if k == "src"), None)
        if src != reported:
            raise EngineError(Code.OUTPUT_INVALID, "the src of an <img> read differently from the parse")
        alt = next((v for k, v in attrs if k == "alt"), None) or ""
        tags.append((at, at + len(raw), None if s is None else at + s, None if e is None else at + e, src, alt))

    def children(token, inline):
        for child in token.children or ():
            if child.type == "image":
                markdown[child.attrGet("src")] += 1
            elif child.type == "html_inline":
                inline.append(child.content)
            children(child, inline)

    for token in markdown_parser().parse(text):
        if token.type == "html_block" and token.map:
            found = _images(token.content)
            if not found:
                continue
            lo, hi = region(token.map)
            pos = max(cursor, lo)
            for raw, attrs in found:
                at = text.find(raw, pos, hi)
                if at < 0:
                    raise EngineError(Code.OUTPUT_INVALID, "an <img> of an HTML block not found in its lines")
                place(raw, attrs, at)
                pos = cursor = at + len(raw)
        elif token.type == "inline" and token.map:
            inline = []
            children(token, inline)
            if not any(_images(raw) for raw in inline):
                continue
            lo, hi = region(token.map)
            positions = _inline_positions(text, max(cursor, lo), hi, inline)
            for raw, at in zip(inline, positions):
                for raw_img, attrs in _images(raw):
                    place(raw_img, attrs, at)
            cursor = positions[-1] + len(inline[-1])
    first = {}
    for name in markdown:
        if name:
            at = text.find("](" + name + ")")
            if at >= 0:
                first[name] = at
    for t in tags:
        if t[4] and (t[4] not in first or t[0] < first[t[4]]):
            first[t[4]] = t[0]
    return Refs(markdown, tags, first)


def external(src):
    """A well-formed http(s) URL as the contract filter keeps it (`destination_kind` of
    `filters/contract.lua`, the same checks as the product's reference profile)."""
    if not src or "\\" in src or any(ord(c) < 32 or ord(c) == 127 for c in src):
        return False
    try:
        decoded = unquote(src, errors="strict")
    except UnicodeError:
        return False
    if "\\" in decoded or any(ord(c) < 32 or ord(c) == 127 for c in decoded) or decoded.startswith("//"):
        return False
    m = re.match(r"([A-Za-z][A-Za-z0-9+.-]*)://([^/?#]*)", decoded)
    if not m or m.group(1).lower() not in ("http", "https") or not m.group(2):
        return False
    if re.search(r"[\[\]]", m.group(2)) or any(ord(c) > 127 for c in m.group(2)):
        return False
    try:
        urlsplit(decoded)
    except ValueError:
        return False
    return True


def alt_text(alt):
    """The alternative text that replaces an `<img>` without an extracted source."""
    return html.escape(alt).replace("|", "&#124;").replace("\r\n", " ").replace("\r", " ").replace("\n", " ")


def rename(text, names, ext_of=None, refs=None):
    """The image names of the converter -> `assets/asset-NNNN.<ext>` in both forms, numbered by
    their first occurrence (a name the text does not use comes last); `ext_of` {name: ext} (from
    the bytes), by default the extension of the name, jpg or png. Returns (text, mapping,
    problems): the problems of the `<img>` tags without an extracted source."""
    refs = refs if refs is not None else scan(text)
    order = sorted(names, key=lambda n: (refs.first.get(n, len(text)), n))
    mapping = {}
    for i, name in enumerate(order, 1):
        ext = (ext_of or {}).get(name) or ("jpg" if name.lower().endswith((".jpg", ".jpeg")) else "png")
        mapping[name] = "assets/asset-%04d.%s" % (i, ext)
    edits, problems = [], []
    for name in names:
        needle = "](" + name + ")"
        found, at = 0, text.find(needle)
        while at >= 0:
            edits.append((at + 2, at + 2 + len(name), mapping[name]))
            found += 1
            at = text.find(needle, at + len(needle))
        if found != refs.markdown.get(name, 0):
            raise EngineError(Code.OUTPUT_INVALID, "image reference count of a converter image")
    for start, end, s, e, src, alt in refs.html:
        if src in mapping:
            quote = text[s] if text[s] in "'\"" else '"'
            edits.append((s, e, quote + mapping[src] + quote))
        elif external(src):
            problems.append(contract.problem("media_external", path=src, detail="<img> left as a link, never fetched"))
        else:
            edits.append((start, end, alt_text(alt)))
            problems.append(contract.problem("media_missing", path=src or "", detail="<img> replaced by its alternative text"))
    edits.sort()
    for (a_start, a_end, _), (b_start, _, _) in zip(edits, edits[1:]):
        if b_start < a_end:
            raise EngineError(Code.OUTPUT_INVALID, "overlapping image references")
    parts, at = [], 0
    for start, end, new in edits:
        parts += [text[at:start], new]
        at = end
    parts.append(text[at:])
    return "".join(parts), mapping, problems
