#!/usr/bin/env python3
"""Verificatore M indipendente sul canonico; non importa generatori/scorer.

TXT byte/range, OOXML/MathML nodi, RTF scoped, TeX fonte; PDF Poppler+MuPDF
con font/CMap, regioni e immagini. I glifi raster non attestano code point.
"""
import argparse
import io
import json
from pathlib import Path
import re
import zipfile
import xml.etree.ElementTree as ET

from data import descriptor, digest_bytes, load, write_new
from resources import run
from safety import regular, xml, zip_members

NS_W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
NS_M = "http://schemas.openxmlformats.org/officeDocument/2006/math"
NS_R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"


def seq(items):
    return {"type": "Seq", "items": items}


def arg(items):
    return {"type": "Arg", "body": seq(items)}


def source_token(value):
    if value in ("=", "+", "-", "−", "<", ">"):
        return {"type": "Operator", "kind": {"=": "equals", "+": "plus", "-": "minus", "−": "minus", "<": "lt", ">": "gt"}[value]}
    if re.fullmatch(r"[0-9]+(?:[.,][0-9]+)?", value):
        return {"type": "Number", "lexeme": value}
    if re.fullmatch(r"[A-Za-z]", value):
        return {"type": "Symbol", "value": value}
    raise ValueError("Source math token non supportato: " + repr(value))


def omml_source(root):
    """Walk diretto OMML, nessuna traduzione attraverso LaTeX/Pandoc."""
    def children(node):
        items = []
        for child in node:
            tag = child.tag.split("}")[-1]
            if tag in ("r", "sSub", "sSup", "sSubSup", "f"):
                items.append(visit(child))
            elif tag in ("oMath", "oMathPara", "e", "sub", "sup", "num", "den"):
                items.extend(children(child))
            elif tag.endswith("Pr"):
                continue
            else:
                raise ValueError("OMML source unsupported: " + tag)
        return items
    def visit(node):
        tag = node.tag.split("}")[-1]
        if tag == "r":
            literal = "".join(x.text or "" for x in node if x.tag == "{" + NS_M + "}t")
            return source_token(literal)
        if tag == "f":
            return {"type": "Fraction", "numerator": arg(children(node.find("{" + NS_M + "}num"))), "denominator": arg(children(node.find("{" + NS_M + "}den")))}
        base = children(node.find("{" + NS_M + "}e"))
        if len(base) != 1:
            raise ValueError("OMML base non atomico")
        sub = node.find("{" + NS_M + "}sub")
        sup = node.find("{" + NS_M + "}sup")
        return {"type": "Scripts", "base": base[0], "sub": arg(children(sub)) if sub is not None else None, "sup": arg(children(sup)) if sup is not None else None}
    return seq(children(root))


def mathml_source(root):
    """Walk diretto MathML, reader di riferimento separato da math output."""
    def content(node):
        tag = node.tag.split("}")[-1]
        children = list(node)
        if tag=='mrow' and len(children)==3 and children[0].tag.endswith('}mo') and children[0].text=='[' and children[2].text==']' and children[1].tag.endswith('}mtable'):
            matrix=content(children[1]);matrix['type']='Matrix';matrix['delimiter']='square';return matrix
        if tag in ("math", "mrow"):
            return seq([content(c) for c in children])
        if tag in ("mi", "mn", "mo"):
            return source_token(node.text or "")
        if tag in ("msub", "msup", "msubsup"):
            def aa(child):
                item = content(child)
                return arg(item["items"] if item["type"] == "Seq" else [item])
            return {"type": "Scripts", "base": content(children[0]), "sub": aa(children[1]) if tag in ("msub", "msubsup") else None, "sup": aa(children[2] if tag == "msubsup" else children[1]) if tag in ("msup", "msubsup") else None}
        if tag == "mfrac":
            vals = [content(c) for c in children]
            return {"type": "Fraction", "numerator": arg(vals[0]["items"] if vals[0]["type"] == "Seq" else [vals[0]]), "denominator": arg(vals[1]["items"] if vals[1]["type"] == "Seq" else [vals[1]])}
        if tag=='munderover' and len(children)==3 and children[0].text=='∑':
            def argument(child):
                value=content(child);return arg(value['items'] if value['type']=='Seq' else [value])
            return {'type':'Sum','lower':argument(children[1]),'upper':argument(children[2])}
        if tag=='mtable':
            rows=[]
            for row in children:
                if not row.tag.endswith('}mtr'):raise ValueError('MathML row')
                cells=[]
                for cell in row:
                    if not cell.tag.endswith('}mtd') or len(cell)!=1:raise ValueError('MathML cell')
                    value=content(cell[0]);cells.append(value if value['type']=='Seq' else seq([value]))
                rows.append(cells)
            return {'type':'Aligned','rows':rows}
        if tag=='mtext':return {'type':'Text','command':'text','value':node.text or ''}
        raise ValueError("MathML source unsupported: " + tag)
    result = content(root)
    # A math root containing the author's one mrow is an XML wrapper, not a
    # grouping operator. Nested non-wrapper mrow scopes remain in Arg.
    if result["type"] == "Seq" and len(result["items"]) == 1 and result["items"][0]["type"] in ('Seq','Aligned','Matrix'):
        return result["items"][0]
    return result


class TeXSource:
    """Parser fonte indipendente, finito. Non usato per observations/score."""
    def __init__(self, text):
        tokens = re.findall(r"\\text\{(?:\\[{}%$_&#]|[^{}\\])*\}|\\begin\{(?:aligned|bmatrix|matrix)\}|\\end\{(?:aligned|bmatrix|matrix)\}|\\[A-Za-z]+|\\[,;:!]|\\\\|[0-9]+(?:[.,][0-9]+)?|[^ \t\r\n]", text)
        self.tokens = [t for t in tokens if t not in (r'\,',r'\;',r'\:',r'\!',r'\quad',r'\qquad')]
        self.position = 0
    def take(self):
        if self.position >= len(self.tokens):
            raise ValueError("TeX source truncated")
        token = self.tokens[self.position]
        self.position += 1
        return token
    def argument(self):
        if self.tokens[self.position] == "{":
            self.take()
            result = self.sequence({"}"})
            if self.take() != "}":
                raise ValueError("TeX source arg")
            return {"type": "Arg", "body": result}
        token = self.tokens[self.position]
        if token.isdecimal() and len(token) > 1:
            self.tokens[self.position:self.position+1] = list(token)
        return arg([self.atom(False)])
    def atom(self, with_scripts=True):
        token = self.take()
        if token == r"\frac":
            result = {"type": "Fraction", "numerator": self.argument(), "denominator": self.argument()}
        elif token == r"\sum":
            result = {"type": "Sum", "lower": None, "upper": None}
        elif token.startswith(r'\text{'):
            return {'type':'Text','command':'text','value':re.sub(r'\\([{}%$_&#])',r'\1',token[6:-1])}
        elif token in ('{','(','[') or token==r'\left':
            if token==r'\left':
                if self.take()!='[':raise ValueError('Source left delimiter')
                stop=r'\right';closing=']';gkind='square'
            else:stop={'{':'}','(':')','[':']'}[token];closing=None;gkind={'{':'braces','(':'round','[':'square'}[token]
            body=self.sequence({stop})
            if self.take()!=stop:
                raise ValueError("TeX group")
            if closing and self.take()!=closing:raise ValueError('Source right delimiter')
            if gkind=='square' and len(body['items'])==1 and body['items'][0]['type']=='Matrix' and body['items'][0]['delimiter']=='none':
                result=body['items'][0];result['delimiter']='square'
            else:result={'type':'Group','kind':gkind,'body':body}
        elif token.startswith(r"\begin{"):
            env = token[7:-1]
            end = r"\end{" + env + "}"
            rows, row = [], []
            while True:
                row.append(self.sequence({"&", r"\\", end}))
                separator = self.take()
                if separator == "&":
                    continue
                rows.append(row)
                row = []
                if separator == end:
                    break
                if separator != r"\\":
                    raise ValueError("TeX table separator")
            return {"type": "Aligned", "rows": rows} if env == "aligned" else {"type": "Matrix", "delimiter": "square" if env == "bmatrix" else "none", "rows": rows}
        else:
            result = source_token(token)
        fields = {}
        while (with_scripts or result['type']=='Sum') and self.position < len(self.tokens) and self.tokens[self.position] in ("_", "^"):
            key = self.take()
            if key in fields:
                raise ValueError("TeX duplicate script")
            fields[key] = self.argument()
        if result["type"] == "Sum":
            result.update(lower=fields.get("_"), upper=fields.get("^"))
        elif fields:
            result = {"type": "Scripts", "base": result, "sub": fields.get("_"), "sup": fields.get("^")}
        return result
    def sequence(self, stop):
        items = []
        while self.position < len(self.tokens) and self.tokens[self.position] not in stop:
            items.append(self.atom())
        return seq(items)
    def parse(self):
        result = self.sequence(set())
        if len(result["items"]) == 1 and result["items"][0]["type"] in ("Aligned", "Matrix"):
            return result["items"][0]
        return result


def rtf_source(data, include_notes=False):
    """Decoder scoped uc/u/hex e field, con gruppi ignorabili separati."""
    text = data.decode("ascii")
    out, stack, state, pos, fallback = [], [], {"uc": 1, "skip": False, "note": False}, 0, 0
    while pos < len(text):
        c = text[pos]
        if c == "{":
            stack.append(dict(state)); pos += 1
            continue
        if c == "}":
            if not stack:
                raise ValueError("RTF unbalanced")
            state = stack.pop(); pos += 1
            continue
        if c == "\\":
            pos += 1
            if pos >= len(text):
                raise ValueError("RTF truncated")
            char = text[pos]
            if char in "{}\\":
                value, pos = char, pos + 1
            elif char == "'":
                value = bytes([int(text[pos+1:pos+3], 16)]).decode("cp1252"); pos += 3
            elif char == "*":
                state["skip"] = True; pos += 1
                continue
            elif char.isalpha():
                match = re.match(r"([A-Za-z]+)(-?[0-9]+)? ?", text[pos:])
                word, number = match.groups(); pos += len(match[0])
                if word == "uc":
                    state["uc"] = int(number)
                    if not 0 <= state["uc"] <= 8:
                        raise ValueError("RTF uc limit")
                    continue
                if word == "u":
                    n = int(number)
                    if not -32768 <= n <= 32767:
                        raise ValueError("RTF u16")
                    value = chr(n if n >= 0 else n + 65536)
                    if not state["skip"] and (include_notes or not state["note"]):
                        out.append(value)
                    fallback = state["uc"]
                    continue
                if word in ("fonttbl", "colortbl", "stylesheet", "fldinst"):
                    state["skip"] = True
                if word == "footnote":
                    state["note"] = True
                if word == "fldrslt":
                    state["skip"] = False
                value = {"par": "\n", "line": "\n", "cell": "\t", "row": "\n"}.get(word, "")
            else:
                pos += 1; value = {"~": "\u00a0", "-": "\u00ad", "_": "\u2011"}.get(char, "")
        else:
            pos += 1
            if c in "\r\n":
                continue
            value = c
        if fallback and value:
            fallback -= 1
            continue
        if not state["skip"] and (include_notes or not state["note"]):
            out.append(value)
    if stack:
        raise ValueError("RTF unbalanced groups")
    return ''.join(out)


def codepoints(value):
    return [f"U+{ord(c):04X}" for c in value] if isinstance(value, str) else None


def _plain(node, tags):
    return ''.join(x.text or '' for x in node.iter() if x.tag.rsplit('}', 1)[-1] in tags)


def check_native(corpus, document, gold):
    canonical = regular(corpus, document["input"])
    parts = zip_members(canonical) if document["format"] in ("docx", "epub") else None
    results = []
    def emit(block, key, expected, actual, kind="encoded", reason=None):
        results.append({"check_id": block["block_id"] + ":" + key, "doc_id": document["document_id"], "canonical_sha256": digest_bytes(canonical), "gold_sha256": document["gold_sha256"], "locator": block["locator"], "evidence_kind": kind, "expected": expected, "actual": actual, "codepoints": codepoints(actual), "expected_codepoints": codepoints(expected), "method": "independent-canonical-source-v1", "status": "PASS" if actual == expected else "FAIL", "required": True, "reason": reason})
    if document["format"] == "tex":
        includes = re.findall(rb"\\(?:input|include)\{([^}]+)\}", canonical)
        expected = [name.encode() for name in document["includes"]]
        if includes != expected:
            raise ValueError("TeX canonical include order diverges")
    if document["format"] == "epub":
        opf = xml(parts["OEBPS/content.opf"])
        ns = {"o": "http://www.idpf.org/2007/opf"}
        spine = [x.get("idref") for x in opf.findall("o:spine/o:itemref", ns)]
        if spine != ["c1", "c2"]:
            raise ValueError("EPUB spine diverges")
    for b in gold["blocks"]:
        loc = b["locator"]
        kind = document["format"]
        if kind == 'control-markdown':
            literal=canonical[loc['start']:loc['start']+loc['length']].decode('utf-8')
            emit(b,'literal',loc['literal'],literal)
            if b['type']=='formula':
                emit(b,'formula',b['formulas'][0]['structure_expected'],TeXSource(literal.strip().strip('$')).parse())
            if b['type']=='table':
                lines=[s for s in literal.splitlines() if s.startswith('|')]
                actual=[[c.strip(' ') for c in s.strip().removeprefix('|').removesuffix('|').split('|')] for s in [lines[0],*lines[2:]]]
                emit(b,'cells',b['cells'],actual)
                emit(b,'semantic-header',b.get('header_rows',1),1)
                captions=re.findall(r'^(?:Table:|:) (.*)$',literal,re.MULTILINE)
                emit(b,'caption',b.get('caption'),captions[0] if len(captions)==1 else None)
            if b['type']=='figure':
                emit(b,'asset_reference',f'![{b["caption"]}](assets/{b["asset"]})',literal.strip())
                image=regular(corpus,document['document_id']+'/assets/'+b['asset'])
                emit(b,'asset',digest_bytes(regular(corpus.parent,'assets/'+b['asset'])),digest_bytes(image))
            if b['type'] not in ('table','figure','formula'):
                actual=literal.removesuffix('\n')
                if b['type']=='heading':actual=re.sub(r'^#{1,6} ', '', re.sub(r' \{#[^{}]+\}$','',actual))
                elif b['type']=='code':actual='\n'.join(literal.splitlines()[1:-1])+'\n'
                elif b['type']=='note':actual=re.sub(r'^\[\^[^\]]+\]: ', '',actual)
                else:
                    actual=re.sub(r'\[([^\]]+)\]\(([^\n()]*)\)',r'\1',actual)
                    actual=re.sub(r'\[\^[^\]]+\]','',actual)
                emit(b,'content',b['text'],actual)
            if b.get('anchor'):
                values=re.findall(r'\{#([^{}]+)\}',literal) or re.findall(r'^\[\^([^\]]+)\]:',literal)
                emit(b,'anchor',[b['anchor']],values)
            if b.get('references'):
                values=[m[2] if m[1] else m[3] for m in re.finditer(r'\[([^\]]+)\]\(([^\n()]*)\)|\[\^([^\]]+)\]',literal)]
                emit(b,'references',[r['target'] for r in b['references']],values)
        elif kind == "txt":
            actual = canonical[loc["start"]:loc["start"] + loc["length"]].decode("utf-8", errors="strict")
            emit(b, "content", b["text"], actual)
        elif kind == "docx":
            tree = xml(parts[loc["part"]])
            if b["type"] == "note":
                nodes = [x for x in tree if x.get("{" + NS_W + "}id") == loc["footnote"]]
                if len(nodes) != 1:
                    raise ValueError("Footnote locator ambiguous")
                node = nodes[0]
            else:
                node = list(tree.find("{" + NS_W + "}body"))[loc["body_child"]]
            if b["type"] == "formula":
                math = node.find("{" + NS_M + "}oMath")
                emit(b, "formula", b["formulas"][0]["structure_expected"], omml_source(math))
            elif b["type"] == "table":
                rows = [[_plain(cell, {"t"}) for cell in row.findall("{" + NS_W + "}tc")] for row in node.findall("{" + NS_W + "}tr")]
                emit(b, "cells", b["cells"], rows)
                source_rows=node.findall('{'+NS_W+'}tr')
                flags=[row.find('{'+NS_W+'}trPr/{'+NS_W+'}tblHeader') is not None for row in source_rows]
                emit(b,'semantic-header',b.get('header_rows',1),sum(flags))
                caption_index=loc.get('caption_child')
                children=list(tree.find("{"+NS_W+"}body"))
                caption=children[caption_index] if caption_index is not None and caption_index<len(children) else None
                emit(b,'caption',b.get('caption'),_plain(caption,{'t'}) if caption is not None else None)
            elif b["type"] == "figure":
                caption = list(tree.find("{" + NS_W + "}body"))[loc["caption_child"]]
                emit(b, "caption", b["caption"], _plain(caption, {"t"}))
                image = parts["word/media/" + b["asset"]]
                emit(b, "asset", digest_bytes(regular(corpus, "assets/" + b["asset"])), digest_bytes(image))
                blips = [x for x in node.iter() if x.tag.endswith("}blip")]
                emit(b, "asset_relationship", ["rImage"], [x.get("{" + NS_R + "}embed") for x in blips])
                relationships={x.get('Id'):x.get('Target') for x in xml(parts['word/_rels/document.xml.rels'])}
                emit(b,'asset_target',['media/'+b['asset']],[relationships.get(x.get('{'+NS_R+'}embed')) for x in blips])
            else:
                expected = b["text"] + ''.join(" " + r["text"] if r["kind"] != "note_call" else " " for r in b.get("references", []))
                emit(b, "content", expected, _plain(node, {"t"}))
            if b.get("anchor"):
                emit(b, "anchor", [b["anchor"]], [x.get("{" + NS_W + "}name") for x in node.iter() if x.tag.endswith("}bookmarkStart")])
            if b.get("references"):
                rels = {x.get("Id"): x.get("Target") for x in xml(parts["word/_rels/document.xml.rels"])}
                actual = []
                for x in node.iter():
                    if x.tag.endswith("}hyperlink"):
                        actual.append(rels[x.get("{" + NS_R + "}id")] if x.get("{" + NS_R + "}id") else "#" + x.get("{" + NS_W + "}anchor"))
                    if x.tag.endswith("}footnoteReference"):
                        actual.append("n" + x.get("{" + NS_W + "}id"))
                emit(b, "references", [r["target"] for r in b["references"]], actual)
        elif kind == "epub":
            tree = xml(parts[loc["part"]])
            nodes = [x for x in tree.iter() if x.get("id") == loc["id"]]
            if len(nodes) != 1:
                raise ValueError("EPUB locator absent/ambiguous")
            node = nodes[0]
            if b["type"] == "formula":
                math = next(x for x in node.iter() if x.tag.endswith("}math"))
                emit(b, "formula", b["formulas"][0]["structure_expected"], mathml_source(math))
            elif b["type"] == "table":
                rows = [["".join(c.itertext()) for c in row if c.tag.rsplit("}", 1)[-1] in ("td", "th")] for row in node if row.tag.rsplit("}", 1)[-1] == "tr"]
                emit(b, "cells", b["cells"], rows)
                source_rows=[r for r in node if r.tag.rsplit('}',1)[-1]=='tr']
                flags=[bool(list(r)) and all(c.tag.rsplit('}',1)[-1]=='th' for c in r) for r in source_rows]
                emit(b,'semantic-header',b.get('header_rows',1),sum(flags))
                captions=[x for x in node if x.tag.endswith('}caption')]
                emit(b,'caption',b.get('caption'),''.join(captions[0].itertext()) if len(captions)==1 else None)
            elif b["type"] == "figure":
                caption = next(x for x in node if x.tag.endswith("}figcaption"))
                emit(b, "caption", b["caption"], "".join(caption.itertext()))
                image = next(x for x in node if x.tag.endswith("}img"))
                emit(b, "asset", digest_bytes(regular(corpus, "assets/" + b["asset"])), digest_bytes(parts["OEBPS/" + image.get("src")]))
            else:
                expected = b["text"] + ''.join(" " + r["text"] for r in b.get("references", []))
                emit(b, "content", expected, ''.join(node.itertext()))
            if b.get("references"):
                emit(b, "references", [r["target"] for r in b["references"]], [x.get("href") for x in node.iter() if x.tag.endswith("}a")])
            if b.get('anchor') or b.get('note_id'):
                emit(b,'anchor',b.get('anchor') or b.get('note_id'),node.get('id'))
        elif kind == "rtf":
            fragment = canonical[loc["start"]:loc["start"]+loc["length"]]
            actual = rtf_source(fragment, include_notes=b["type"] == "note")
            if b["type"] == "table":
                # Control-word delimiter spaces are consumed by lexer; literal
                # spaces preceding next control word remain actual content.
                rows = [row.split("\t")[:-1] for row in actual.splitlines() if '\t' in row]
                rows = [[c.strip(' ') for c in row] for row in rows]
                emit(b, "cells", b["cells"], rows)
                emit(b,'semantic-header',b.get('header_rows',1),len(re.findall(rb'\\trhdr(?![A-Za-z])',fragment)))
                captions=[row for row in actual.splitlines() if row and '\t' not in row]
                emit(b,'caption',b.get('caption'),captions[0] if len(captions)==1 else None)
            else:
                expected = b["text"] + ''.join(" " + r["text"] if r["kind"] == "link" else " " for r in b.get("references", []))
                emit(b, "content", expected, actual.removesuffix("\n"))
            for i, ref in enumerate(b.get("references", [])):
                actual = re.findall(rb'HYPERLINK "([^"\r\n]+)"', fragment) if ref["kind"] == "link" else ([b"n1"] if b"\\footnote" in fragment else [])
                emit(b, "reference" + str(i), ref["target"], actual[0].decode() if len(actual) == 1 else None)
        elif kind == "tex":
            path = Path(document["input"]).parent / loc["file"]
            source = regular(corpus, path.as_posix())
            literal = source[loc["start"]:loc["start"]+loc["length"]].decode()
            if b["type"] == "formula":
                f = b["formulas"][0]
                raw = literal.strip()
                raw = raw[1:-1] if raw.startswith("$") else raw.removeprefix(r"\[").removesuffix(r"\]").strip()
                emit(b, "formula", f["structure_expected"], TeXSource(raw).parse())
                emit(b, "latex_literal", f["reference_latex"], raw)
            elif b["type"] == "table":
                match=re.search(r'\\begin\{tabular\}\{[^{}]*\}\n(.*?)\\end\{tabular\}',literal,re.DOTALL)
                payload=match[1].strip().removesuffix(r'\\').rstrip() if match else ''
                rows = [[c.strip(' \n') for c in row.split('&')] for row in payload.split(r'\\')]
                emit(b, "cells", b["cells"], rows)
                emit(b,'semantic-header',b.get('header_rows',1),0)
                emit(b,'header-policy','author-declared-headerless-tabular-v1',b.get('header_policy'))
                captions=re.findall(r'\\caption\{([^{}]*)\}',literal)
                emit(b,'caption',b.get('caption'),captions[0] if len(captions)==1 else None)
            else:
                if b['type']=='heading':
                    match=re.fullmatch(r'\\section\{([^{}]*)\}\\label\{[^{}]+\}\n',literal)
                    actual=match[1] if match else None
                elif b['type']=='bibliography':
                    match=re.fullmatch(r'\\begin\{thebibliography\}\{1\}\n\\bibitem\{[^{}]+\}([^\n]*)\n\\end\{thebibliography\}\n',literal)
                    actual=match[1] if match else None
                elif b['type']=='figure':
                    match=re.search(r'\\caption\{([^{}]*)\}',literal)
                    actual=match[1] if match else None
                else:
                    actual=literal.removesuffix('\n').split(' \\',1)[0]
                emit(b, "literal_content", b["text"], actual)
                if b.get("anchor"):
                    anchors = re.findall(r'\\(?:label|bibitem)\{([^}]+)\}', literal)
                    emit(b, "anchor", [b["anchor"]], anchors)
                if b["type"] == "figure":
                    asset = re.findall(r'\\includegraphics\{([^}]+)\}', literal)
                    emit(b, "asset_reference", ['assets/' + b["asset"]], asset)
                    emit(b, "asset", digest_bytes(regular(corpus, 'assets/' + b["asset"])), digest_bytes(regular(corpus, (Path(document["input"]).parent / asset[0]).as_posix())))
                for i, ref in enumerate(b.get("references", [])):
                    if ref["kind"] == "link":
                        values = [t.replace(r'\&', '&') for t in re.findall(r'\\href\{([^}]+)\}', literal)]
                    else:
                        command = 'cite' if ref["kind"] == 'citation' else 'ref'
                        values = re.findall(r'\\' + command + r'\{([^}]+)\}', literal)
                    emit(b, "reference" + str(i), ref["target"], values[0] if len(values) == 1 else None)
    return results


def in_box(x, y, bbox):
    return bbox[0] <= x <= bbox[2] and bbox[1] <= y <= bbox[3]


def annotated_pdf_lines(block):
    """Independent finite glyph convention, never presentation-viewer code.

    Ties manual display fields to locators crosschecked on the actual PDF.
    Flattened glyphs are not geometry or raster Unicode certification.
    """
    def plain(node):
        kind=node['type']
        if kind=='Seq':return ''.join(plain(c) for c in node['items'])
        if kind=='Arg':return plain(node['body'])
        if kind=='Symbol':return node['value']
        if kind=='Number':return node['lexeme']
        if kind=='Operator':return {'equals':'=','plus':'+','minus':'−'}[node['kind']]
        if kind=='Scripts':
            base=plain(node['base'])
            if node['sub']:
                if plain(node['sub'])!='i':raise ValueError('PDF subscript outside corpus convention')
                base+='ᵢ'
            if node['sup']:
                if plain(node['sup'])!='2':raise ValueError('PDF superscript outside corpus convention')
                base+='²'
            return base
        raise ValueError('PDF glyph convention outside corpus')
    if block['type']=='formula':
        if len(block['formulas'])!=1:raise ValueError('PDF formula block cardinality')
        node=TeXSource(block['formulas'][0]['reference_latex']).parse()
        if node['type']=='Seq' and len(node['items'])==1 and node['items'][0]['type']=='Fraction':
            f=node['items'][0];return [plain(f['numerator']),plain(f['denominator'])]
        if node['type']=='Seq' and node['items'] and node['items'][0]['type']=='Sum':
            f=node['items'][0];return [plain(f['upper']),'∑'+''.join(plain(c) for c in node['items'][1:]),plain(f['lower'])]
        if node['type'] in ('Aligned','Matrix'):return [''.join(plain(c) for c in row) for row in node['rows']]
        return [plain(node)]
    if block['type']=='table':return [c for row in block['cells'] for c in row if c]+[block['caption']]
    if block['type']=='figure':return [block['caption']]
    return [block['text']+''.join(' '+r['target'] for r in block.get('references',[]))]


def check_pdf(corpus, document, gold, output, environment):
    canonical = corpus / document["input"]
    output.mkdir()
    commands = []
    def tool(argv):
        result = run(argv, output, output / f"tool-{len(commands)+1:03d}", environment, 120)
        commands.append(result)
        if result["adapter_exit"] or result["stop_reason"]:
            raise ValueError("Reference PDF tool failed")
        return (output / f"tool-{len(commands):03d}/stdout.bin").read_bytes()
    pop = output / "poppler-bbox.xhtml"
    mup = output / "mupdf-stext.xml"
    trace = output / "mupdf-trace.xml"
    tool(["/usr/bin/pdftotext", "-bbox", "-enc", "UTF-8", str(canonical), str(pop)])
    tool(["/usr/bin/mutool", "draw", "-F", "stext", "-o", str(mup), str(canonical)])
    tool(["/usr/bin/mutool", "draw", "-F", "trace", "-o", str(trace), str(canonical)])
    # Poppler emits a fixed XHTML DOCTYPE. Read this tool output in ET with
    # external entity loading absent; canonical XML inputs still reject DTD.
    proot = ET.fromstring(pop.read_bytes())
    mroot = xml(mup.read_bytes())
    troot = xml(trace.read_bytes())
    pp = [p for p in proot.iter() if p.tag.rsplit('}', 1)[-1] == 'page']
    mp = list(mroot)
    tp = list(troot)
    results = []
    extracted_images = output / "images"
    extracted_images.mkdir()
    result = run(["/usr/bin/mutool", "extract", str(canonical)], extracted_images, output / "extract-tool", environment, 120)
    commands.append(result)
    if result["adapter_exit"]:
        raise ValueError("Image extraction reference failed")
    from png_codec import pixel_digest
    images = [(p.name, pixel_digest(regular(extracted_images, p.name))) for p in extracted_images.glob('*.png')]
    raster_pixels_cache = {}
    for b in gold["blocks"]:
        loc, page = b["locator"], b["page"]
        if b['type']=='table':
            results.append({'check_id':b['block_id']+':semantic-header-policy','doc_id':document['document_id'],'canonical_sha256':document['input_sha256'],'gold_sha256':document['gold_sha256'],'required':True,'evidence_kind':'authoring-policy-no-encoded-semantics','method':'synthetic-author-headerless-drawn-grid-v1','expected':{'header_rows':0,'policy':'author-declared-headerless-drawn-grid-v1'},'actual':{'header_rows':b.get('header_rows'),'policy':b.get('header_policy')},'status':'PASS' if b.get('header_rows')==0 and b.get('header_policy')=='author-declared-headerless-drawn-grid-v1' else 'FAIL','reason':'PDF/raster have no encoded semantic header; cell provenance/visible checks remain separate.'})
        expected_annotation=''.join(c for line in annotated_pdf_lines(b) for c in line if c!=' ')
        actual_locator=''.join(c for line in loc['encoded_lines'] for c in line if c!=' ')
        results.append({'check_id':b['block_id']+':manual-fields-to-canonical-locator','doc_id':document['document_id'],'canonical_sha256':descriptor(canonical)['sha256'],'gold_sha256':document['gold_sha256'],'required':True,'evidence_kind':'annotation-crosscheck','method':'independent-finite-PDF-glyph-convention-v2, actual PDF crosscheck separate','expected':expected_annotation,'actual':actual_locator,'status':'PASS' if expected_annotation==actual_locator else 'FAIL','reason':'Raster code point encoding not certified; visible form/provenance V remains required'})
        rec = {"check_id": b["block_id"] + ':content', "doc_id": document["document_id"], "canonical_sha256": descriptor(canonical)["sha256"], "gold_sha256": document["gold_sha256"], "locator": loc, "required": True, "method": "Poppler-bbox+MuPDF-char-v1", "tool_outputs": [descriptor(pop), descriptor(mup)], "evidence_kind": "encoded" if loc["native"] else "provenance_only"}
        words = [w for w in pp[page-1].iter() if w.tag.endswith('word') and in_box(float(w.get('xMin')), float(w.get('yMin')), loc["bbox"])]
        chars = [c for c in mp[page-1].iter('char') if in_box(float(c.get('x')), float(c.get('y')), loc["bbox"])]
        if not loc["native"]:
            raster = corpus / loc['raster_parent']
            raster_name = 'sources/' + document['document_id'] + '/p3-lower.png' if page == 3 else loc['raster_parent']
            if raster_name not in raster_pixels_cache:
                raster_pixels_cache[raster_name] = pixel_digest(regular(corpus, raster_name))
            expectedpixels = raster_pixels_cache[raster_name]
            image_present = any(pixels == expectedpixels for _, pixels in images)
            hidden = bool(words or chars)
            rec.update(expected={"canonical_image_present": True, "encoded_text_in_region": False}, actual={"canonical_image_present": image_present, "encoded_text_in_region": hidden}, status='PASS' if image_present and not hidden else 'FAIL', reason='Raster pixels/provenance verified; Unicode transcription remains conventional and requires V; no code point encoding claim', raster_pixels=expectedpixels)
        else:
            expected = ' '.join(re.split(r'[ \t\r\n]+', ' '.join(loc['encoded_lines'])))
            # Tool word boundaries remove only ordinary layout spaces. MuPDF
            # native glyph stream is recorded separately, including Unicode.
            poptext = ' '.join(w.text or '' for w in words)
            mu_text = ''.join(c.get('c') for c in chars)
            expected_glyphs = ''.join(c for c in expected if c != ' ')
            actual_glyphs = ''.join(c for c in mu_text if c != ' ')
            pop_ok = poptext == expected if b['type'] not in ('formula', 'table') else poptext.replace(' ', '') == expected_glyphs
            rec.update(expected=expected, actual=poptext, mupdf_glyphs=mu_text, codepoints=codepoints(mu_text), status='PASS' if pop_ok and actual_glyphs == expected_glyphs else 'DIVERGENT', reason='Layout spaces are separate from encoded nonspace glyph checks; V certifies shape/reading order')
            if b['type'] == 'figure':
                pixels = pixel_digest(regular(corpus, 'assets/' + b['asset']))
                rec['expected_asset_pixels'] = pixels
                rec['matching_extracted_images'] = [name for name, actual in images if actual == pixels]
                if not rec['matching_extracted_images']:
                    rec['status'] = 'FAIL'
            if b["type"] == 'table':
                cw = (loc['bbox'][2]-loc['bbox'][0]-6)/3
                x = loc['bbox'][0]+3
                top = loc['bbox'][1]+4
                cells = []
                for ri, row in enumerate(b['cells']):
                    for ci, value in enumerate(row):
                        box = [x+ci*cw, top+ri*25, x+(ci+1)*cw, top+(ri+1)*25]
                        actual = ''.join(c.get('c') for c in mp[page-1].iter('char') if in_box(float(c.get('x')), float(c.get('y')), box)).strip(' ')
                        popcell = ' '.join(w.text or '' for w in pp[page-1].iter() if w.tag.endswith('word') and in_box(float(w.get('xMin')), float(w.get('yMin')), box))
                        rectangles = []
                        for path in tp[page-1].iter('stroke_path'):
                            pts = [(float(c.get('x')),float(c.get('y'))) for c in path if c.tag in ('moveto','lineto')]
                            if len(pts) == 4:
                                rect = [min(t[0] for t in pts), min(t[1] for t in pts), max(t[0] for t in pts), max(t[1] for t in pts)]
                                if all(abs(a-z) <= 0.02 for a,z in zip(box,rect)):
                                    rectangles.append(rect)
                        cells.append({'row':ri,'col':ci,'expected':value,'actual':actual,'poppler':popcell,'bbox':box,'canonical_rectangles':rectangles,'status':'PASS' if actual == value and popcell == value and len(rectangles)==1 else 'FAIL'})
                rec['cells'] = cells
                if any(c['status'] != 'PASS' for c in cells):
                    rec['status'] = 'FAIL'
                else:
                    # Global extraction order is not a cell locator. Both tools
                    # must agree independently at every coordinate, including
                    # empty outlined cells; verify caption in its own region.
                    caption_chars = ''.join(c.get('c') for c in chars if float(c.get('y')) > top + 25*len(b['cells']))
                    caption_words = ' '.join(w.text or '' for w in words if float(w.get('yMin')) > top + 25*len(b['cells']))
                    rec['caption_extractors'] = [caption_chars, caption_words]
                    rec['status'] = 'PASS' if caption_chars.replace(' ', '') == b['caption'].replace(' ', '') and caption_words == b['caption'] else 'DIVERGENT'
        results.append(rec)
    # Font/ToUnicode locators retained per page. Their presence does not prove
    # visual shape; cross-extracted glyphs and V remain independent requirements.
    for page in range(1, document['pages']+1):
        fonts = tool(['/usr/bin/mutool','show',str(canonical),f'pages/{page}/Resources/Font']).decode()
        records = []
        for name, number in re.findall(r'/([A-Za-z0-9_-]+)\s+(\d+)\s+0\s+R', fonts):
            font = tool(['/usr/bin/mutool','show',str(canonical),number]).decode()
            cmap = re.search(r'/ToUnicode\s+(\d+)\s+0\s+R',font)
            stream = tool(['/usr/bin/mutool','show',str(canonical),cmap[1]]) if cmap else None
            records.append({'font_resource':name,'object':number,'ToUnicode_object':cmap[1] if cmap else None,'ToUnicode_sha256':digest_bytes(stream) if stream else None})
        native_page = any(b['page']==page and b['locator']['native'] for b in gold['blocks'])
        results.append({'check_id':f'page-{page}:fonts','doc_id':document['document_id'],'canonical_sha256':descriptor(canonical)['sha256'],'gold_sha256':document['gold_sha256'],'locator':{'page':page},'evidence_kind':'encoded' if native_page else 'provenance_only','required':True,'expected':{'native_ToUnicode_maps':native_page},'actual':records,'status':'PASS' if not native_page or (records and all(r['ToUnicode_object'] for r in records)) else 'UNAVAILABLE','reason':'Raster font encoding N/A' if not native_page else None})
    write_new(output / 'tools.json', commands)
    return results


def check_lo_visibility(corpus, document, gold, output, environment):
    """Supplemental presence on actual LO PDF, separate from source encoding.

    No source-to-output correction: canonical locators remain primary. This
    catches truncation of rendered tail; typography/order still require V.
    """
    output.mkdir(exist_ok=False);pdfname=document['render_pdf']
    pdf=corpus/pdfname;regular(corpus,pdfname);records=[]
    def tool(argv,name):
        receipt=run(argv,output,output/name,environment,60)
        write_new(output/(name+'.json'),receipt)
        if receipt['adapter_exit'] or receipt['stop_reason']:raise RuntimeError('LO visibility extraction failed')
    pop=output/'poppler.txt';mu=output/'mupdf.xml'
    tool(['/usr/bin/pdftotext','-layout','-enc','UTF-8',str(pdf),str(pop)],'poppler')
    tool(['/usr/bin/mutool','draw','-F','stext','-o',str(mu),str(pdf)],'mupdf')
    poptext=regular(output,'poppler.txt').decode();tree=xml(regular(output,'mupdf.xml'))
    mutext='\n'.join(''.join(c.get('c','') for c in line.iter() if c.tag=='char') for line in tree.iter() if line.tag=='line')
    normalized_pop=' '.join(poptext.split());normalized_mu=' '.join(mutext.split())
    for b in gold['blocks']:
        if b['type'] not in ('heading','paragraph','note','bibliography','figure','table'):continue
        text=b.get('caption','') if b['type'] in ('figure','table') else b['text']
        if not text:continue
        expected=' '.join(text.split())
        actual={'poppler_literal_present':expected in normalized_pop,'mupdf_literal_present':expected in normalized_mu}
        records.append({'check_id':b['block_id']+':LO-render-visible-content','doc_id':document['document_id'],'canonical_sha256':document['input_sha256'],'gold_sha256':document['gold_sha256'],'locator':{'render_pdf':pdfname,'source_block_locator':b['locator']},'render_pdf':descriptor(pdf),'evidence_kind':'visible-text-crosscheck','method':'independent Poppler+MuPDF actual LO PDF; whitespace-only line wrapping view; no codepoint equivalence claim from glyphs','required':True,'expected':{'poppler_literal_present':True,'mupdf_literal_present':True},'actual':actual,'status':'PASS' if all(actual.values()) else 'DIVERGENT','reason':'Supplemental render completeness check; source Unicode/cells/ref checks remain separate and V required'})
    return records


def verify(corpus, manifest, output, environment):
    output.mkdir(exist_ok=False)
    all_records, documents = [], []
    for d in manifest['documents']:
        goldbytes = regular(corpus, d['gold'])
        if digest_bytes(goldbytes) != d['gold_sha256'] or digest_bytes(regular(corpus,d['input'])) != d['input_sha256']:
            raise ValueError('Canonical/gold identity STALE')
        gold = json.loads(goldbytes)
        records = check_pdf(corpus,d,gold,output/d['document_id'],environment) if d['format']=='pdf' else check_native(corpus,d,gold)
        if d['format'] in ('docx','rtf'):
            records.extend(check_lo_visibility(corpus,d,gold,output/(d['document_id']+'-LO'),environment))
        for block in gold['blocks']:
            for formula in block.get('formulas',[]):
                actual_structure=TeXSource(formula['reference_latex']).parse()
                records.append({'check_id':block['block_id']+':manual-formula-source:'+formula['formula_id'],'doc_id':d['document_id'],'canonical_sha256':d['input_sha256'],'gold_sha256':d['gold_sha256'],'required':True,'evidence_kind':'annotation-crosscheck','method':'independent-TeXSource-on-manual-reference, canonical properties checked separately','expected':formula['structure_expected'],'actual':actual_structure,'status':'PASS' if actual_structure==formula['structure_expected'] else 'FAIL'})
            values=[block['text']]+[c for row in block.get('cells',[]) for c in row]
            actual=[{'text_index':i,'offset':m.start(),'lexeme':m[0]} for i,value in enumerate(values) for m in re.finditer(r'[−+-]?[0-9]+(?:[.,][0-9]+)?(?:[eE][−+-]?[0-9]+)?',value)]
            expected=block.get('numbers',[])
            records.append({'check_id':block['block_id']+':number-annotation-coverage','doc_id':d['document_id'],'canonical_sha256':d['input_sha256'],'gold_sha256':d['gold_sha256'],'required':True,'evidence_kind':'annotation-crosscheck','method':'annotated-literal-occurrences-v1; source content checks above remain independent','expected':expected,'actual':actual,'status':'PASS' if actual==expected else 'FAIL','reason':'Typed text/cell numbers include zero and exponent. Formula Number/Operator nodes have separate structural denominators; reference-generated note ordinals excluded.'})
        status = 'FAIL' if any(r['status']=='FAIL' for r in records) else 'INCOMPLETE' if any(r['status']!='PASS' for r in records if r['required']) else 'PASS'
        documents.append({'document_id':d['document_id'],'status':status,'required':len(records),'canonical_sha256':d['input_sha256'],'gold_sha256':d['gold_sha256']})
        all_records.extend(records)
    result = {'schema_version':1,'method':'canonical-independent-M-v1','verifier':descriptor(__file__),'documents':documents,'records':all_records,'status':'PASS' if all(d['status']=='PASS' for d in documents) else 'FAIL'}
    write_new(output/'M.json',result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--corpus',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--environment',type=Path,required=True)
    a=p.parse_args()
    env=load(a.environment)
    result=verify(a.corpus,load(a.manifest),a.output,env.get('environment',env))
    print('M_'+result['status']+' '+str(a.output/'M.json'))
    return 0 if result['status']=='PASS' else 1


if __name__=='__main__':
    raise SystemExit(main())
