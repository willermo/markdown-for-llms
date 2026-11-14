"""A lenient (X)HTML tree builder on `html.parser`, producing `xml.etree` elements.

Inert: no script runs, no URL is visited, entities are decoded by the standard library. Implied
end tags follow the usual HTML rules for `p`, `li`, `dt`/`dd`, table rows and cells, so pages
that are not well-formed XML (common in EPUB files) still give a usable tree.
"""
from html.parser import HTMLParser
import xml.etree.ElementTree as ET

VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "source", "track", "wbr"}
BLOCK_CLOSES_P = {"p", "div", "ul", "ol", "table", "h1", "h2", "h3", "h4", "h5", "h6", "pre", "blockquote",
                  "section", "aside", "figure", "dl", "hr", "header", "footer", "nav", "article", "li"}
MAX_DEPTH = 512


class _Builder(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.root = ET.Element("html-root")
        self.stack = [self.root]

    def _close_to(self, names, stop=()):
        for k in range(len(self.stack) - 1, 0, -1):
            tag = self.stack[k].tag
            if tag in stop:
                return
            if tag in names:
                del self.stack[k:]
                return

    def handle_starttag(self, tag, attrs):
        tag = tag.lower().split(":")[-1]
        if tag in BLOCK_CLOSES_P:
            self._close_to({"p"}, stop={"div", "li", "td", "th", "blockquote", "section", "aside", "body"})
        if tag == "li":
            self._close_to({"li"}, stop={"ul", "ol"})
        elif tag in ("dt", "dd"):
            self._close_to({"dt", "dd"}, stop={"dl"})
        elif tag in ("td", "th"):
            self._close_to({"td", "th"}, stop={"tr", "table"})
        elif tag == "tr":
            self._close_to({"tr"}, stop={"table", "tbody", "thead", "tfoot"})
        el = ET.SubElement(self.stack[-1], tag, {k.lower(): (v or "") for k, v in attrs})
        if tag not in VOID and len(self.stack) < MAX_DEPTH:
            self.stack.append(el)

    def handle_startendtag(self, tag, attrs):
        tag = tag.lower().split(":")[-1]
        ET.SubElement(self.stack[-1], tag, {k.lower(): (v or "") for k, v in attrs})

    def handle_endtag(self, tag):
        tag = tag.lower().split(":")[-1]
        for k in range(len(self.stack) - 1, 0, -1):
            if self.stack[k].tag == tag:
                del self.stack[k:]
                return

    def handle_data(self, data):
        parent = self.stack[-1]
        if len(parent):
            parent[-1].tail = (parent[-1].tail or "") + data
        else:
            parent.text = (parent.text or "") + data


def parse_html(data: bytes):
    builder = _Builder()
    builder.feed(data.decode("utf-8", "replace"))
    builder.close()
    return builder.root
