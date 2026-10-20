"""Operational references in the pinned Markdown/HTML profile; never render."""
from html.parser import HTMLParser
import re
from urllib.parse import unquote, urlsplit

from markdown_it import MarkdownIt
from mdit_py_plugins.dollarmath import dollarmath_plugin

from .files import FileSafetyError, relative

SCOPE = {"parser": "markdown-it-py 4.0.0", "preset": "commonmark + table",
         "plugins": "mdit-py-plugins 0.5.0 dollarmath",
         "math": {"allow_labels": True, "allow_space": True, "allow_digits": True,
                  "allow_blank_lines": True, "double_inline": True},
         "html": "html.parser; a[href], img[src]; inert non-resource tags",
         "limits": "No semantic fidelity, PDF asset completeness, anchor existence or general HTML validation"}
LINKS = {'markdown:link', 'html:a[href]'}
# A last path segment with an extension names a file; engine assets always have one.
FILE_NAME = re.compile(r'\.[A-Za-z0-9]{1,10}$')


def inspect(raw, inventory):
    refs, problems = [], []
    present = set(inventory)

    def unknown(reason):
        problems.append({"code": "unchecked", "reason": reason})

    def destination(value, context):
        if value is None:
            unknown("Empty HTML resource attribute")
            return
        try:
            decoded = unquote(value, errors="strict")
            uri = urlsplit(decoded)
            if len(uri.scheme) == 1 or '\\' in decoded or any(ord(c) < 32 for c in decoded):
                raise FileSafetyError('Unsafe resource path')
            if not decoded or decoded.startswith("#"):
                if not decoded and context not in LINKS:
                    unknown('Empty resource destination')
                kind = 'same-document' if not decoded else 'internal-anchor' 
            elif decoded.startswith('//'):
                raise FileSafetyError('Unsafe network path')
            elif uri.scheme or uri.netloc:
                if uri.scheme.lower() == "file" or "\\" in decoded:
                    raise FileSafetyError("Unsafe URI")
                kind = "external-no-fetch"
            else:
                path = uri.path
                if '..' in path.split('/') or path.startswith('/'):
                    raise FileSafetyError('Unsafe local reference')
                while path.startswith('./'):
                    path = path[2:]
                nonfile = path in {'', '.'} or path.endswith('/')
                qualified = path.rstrip('/') if nonfile else path
                if qualified not in {'', '.'}:
                    relative(qualified)
                if uri.query:
                    unknown("Query on local resource")
                if nonfile:
                    unknown('Local non-file reference outside the file profile')
                    kind = 'local-nonfile'
                elif path in present:
                    kind = "local"
                elif context in LINKS and not FILE_NAME.search(path.rsplit('/', 1)[-1]):
                    # A link to a name that is not a file name (for example a
                    # citation misparsed as a link) is a warning, never an asset.
                    kind = 'local-unresolved'
                    problems.append({'code': 'unchecked', 'path': uri.path,
                                     'reason': 'Local link destination is not a file name; not checked as an asset'})
                else:
                    kind = "local-missing"
                if kind == "local-missing":
                    problems.append({"code": "incomplete", "reason": "Missing local reference", "path": uri.path})
            refs.append({"kind": kind, "destination": value, "context": context})
        except (ValueError, UnicodeError):
            refs.append({"kind": "unsafe-local", "destination": value, "context": context})
            problems.append({"code": "invalid", "reason": "Unsafe reference path"})

    class HTML(HTMLParser):
        def handle_starttag(self, tag, attrs):
            if len(dict(attrs)) != len(attrs):
                unknown("Duplicate HTML attributes")
            resource = False
            for key, value in attrs:
                if (tag, key) in {("a", "href"), ("img", "src")}:
                    resource = True
                    destination(value, "html:" + tag + "[" + key + "]")
                elif key in {"src", "href", "srcset", "poster", "data", "background", "action", "formaction", "xlink:href"} or key == "style":
                    resource = True
                    unknown("Unsupported HTML resource context: " + tag + "[" + key + "]")
            if tag in {"script", "style", "iframe", "object", "embed", "svg", "math"}:
                unknown("Unsupported HTML context: " + tag)
            if not resource:
                refs.append({"kind": "inert-html", "context": tag})
        handle_startendtag = handle_starttag

    md = MarkdownIt("commonmark", {"html": True}).enable("table").use(dollarmath_plugin, **SCOPE["math"])
    # Parsing dangerous schemes is safe here: tokens are never rendered or fetched.
    md.validateLink = lambda value: True

    def walk(tokens):
        for token in tokens:
            if token.type in {"fence", "code_block", "code_inline"}:
                refs.append({"kind": "literal-code"})
            elif token.type.startswith("math_"):
                refs.append({"kind": "literal-math"})
            elif token.type == "image":
                destination(token.attrGet("src"), "markdown:image")
            elif token.type == "link_open":
                destination(token.attrGet("href"), "markdown:link")
            elif token.type in {"html_inline", "html_block"}:
                parser = HTML(convert_charrefs=True)
                parser.feed(token.content)
                if parser.rawdata:
                    unknown("Incomplete HTML token")
                parser.close()
            elif token.type == 'text' and '<' in token.content:
                # HTMLParser retains an unfinished tag in rawdata. This only
                # examines text tokens, never code/math or the raw document.
                parser = HTMLParser(convert_charrefs=True)
                parser.feed(token.content)
                if parser.rawdata and parser.rawdata != '<':
                    unknown('Unrecognized/incomplete HTML-like text')
            if token.children and token.type != "image":
                walk(token.children)
    walk(md.parse(raw))
    codes = {p["code"] for p in problems}
    state = next((s for s in ("invalid", "incomplete", "unchecked") if s in codes), "ready")
    return {"bundle": state, "references": refs, "issues": problems, "scope": SCOPE,
            "uncontrolled_forms_count": None if "unchecked" in codes else 0}
