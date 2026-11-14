"""Linearize MathML into Unicode atoms comparable with `text.math_atoms` of TeX."""
from .safe import local


def mathml_linear(el):
    out = []

    def rec(x):
        name = local(x.tag)
        if name in ("annotation", "annotation-xml"):
            return
        if name in ("mi", "mn", "mo", "mtext", "ms"):
            out.append((x.text or "").strip())
            return
        kids = [c for c in x if local(c.tag) not in ("annotation", "annotation-xml")]
        if name in ("msup", "msub", "msubsup", "munder", "mover", "munderover") and kids:
            rec(kids[0])
            marks = {"msup": ["^"], "msub": ["_"], "msubsup": ["_", "^"], "munder": ["_"], "mover": ["^"],
                     "munderover": ["_", "^"]}[name]
            for mark, kid in zip(marks, kids[1:]):
                out.append(mark)
                rec(kid)
        elif name in ("msqrt", "mroot"):
            out.append("√")
            for kid in kids:
                rec(kid)
        elif name == "mfenced":
            out.append(x.get("open", "("))
            for kid in kids:
                rec(kid)
            out.append(x.get("close", ")"))
        else:
            for kid in kids:
                rec(kid)
    rec(el)
    return "".join(out)
