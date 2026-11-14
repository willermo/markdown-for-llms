"""Akoma Ntoso twin reader (D19, Normattiva export).

Preface: the act's title as a heading. Preamble: formula, citations and the enacting line as
paragraphs. Body: each article as a heading (number and title), each paragraph (comma) as a
paragraph without its points and notes, each point as a list item, notes of the body as
footnotes. Conclusions as paragraphs.

Declared rule `normattiva-preface-notes` (C2 of the plan): `authorialNote` elements inside the
`preface` are metadata added by Normattiva (for example the entry-into-force line) and are not
text of the Gazzetta Ufficiale; they are discarded, never turned into checks.
"""
from pathlib import Path
import re

from ..model import Builder
from .safe import local, parse_xml, text_of

SKIP_IN_TEXT = {"authorialNote", "num", "point", "list", "intro", "wrapUp", "eol"}


def _clean(text):
    return re.sub(r"\s+", " ", text).strip()


def _first(el, name):
    return next((x for x in el.iter() if local(x.tag) == name), None)


def read_akn(path, discard=()):
    root = parse_xml(Path(path).read_bytes())
    b = Builder("akoma-ntoso")
    act = _first(root, "act")
    act = root if act is None else act
    preface = _first(act, "preface")
    if preface is not None:
        skip = {"authorialNote"} if "normattiva-preface-notes" in discard else set()
        if skip:
            b.count("discarded_preface_notes", sum(1 for x in preface.iter() if local(x.tag) == "authorialNote"))
        for p in (x for x in preface if local(x.tag) == "p"):
            text = _clean(text_of(p, skip))
            if text:
                b.add("heading" if _first(p, "docTitle") is not None else "paragraph", text, level=1)
    preamble = _first(act, "preamble")
    if preamble is not None:
        for el in preamble:
            name = local(el.tag)
            if name == "citations":
                for cit in (x for x in el.iter() if local(x.tag) == "citation"):
                    _paragraph(cit, b)
            elif name in ("formula", "p", "container", "recitals"):
                for p in (x for x in el.iter() if local(x.tag) == "p"):
                    _paragraph(p, b)
    body = _first(act, "body")
    if body is not None:
        for article in (x for x in body.iter() if local(x.tag) == "article"):
            num, heading = _first(article, "num"), _first(article, "heading")
            title = _clean(" ".join(text_of(x) for x in (num, heading) if x is not None))
            if title:
                b.add("heading", title, level=2)
            for child in article:
                if local(child.tag) == "paragraph":
                    _comma(child, b)
                elif local(child.tag) == "section":   # the NOTE of the Gazzetta after the article
                    _notes(child, b)
    conclusions = _first(act, "conclusions")
    if conclusions is not None:
        for p in (x for x in conclusions.iter() if local(x.tag) == "p"):
            _paragraph(p, b)
    return b.twin


def _notes(el, b):
    for note in (x for x in el.iter() if local(x.tag) == "authorialNote"):
        text = _clean(text_of(note))
        if text:
            b.add("footnote", text)


def _paragraph(el, b):
    text = _clean(text_of(el, {"authorialNote", "eol"}))
    if text:
        b.add("paragraph", text)
    _notes(el, b)


def _comma(para, b):
    num = _first(para, "num")
    parts = [text_of(num)] if num is not None else []
    for child in para:
        name = local(child.tag)
        if name == "content":
            parts.append(text_of(child, SKIP_IN_TEXT))
        elif name == "list":
            intro = next((x for x in child if local(x.tag) == "intro"), None)
            if intro is not None:
                parts.append(text_of(intro, {"authorialNote"}))
    text = _clean(" ".join(parts))
    if text:
        b.add("paragraph", text)
    for point in (x for x in para.iter() if local(x.tag) == "point"):
        pnum = _first(point, "num")
        content = [x for x in point if local(x.tag) in ("content", "intro", "p")]
        ptext = _clean(" ".join(([text_of(pnum)] if pnum is not None else []) + [text_of(x, {"authorialNote", "point", "list"}) for x in content]))
        if ptext:
            b.add("list_item", ptext, level=1)
    _notes(para, b)
