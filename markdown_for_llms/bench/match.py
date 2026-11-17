"""Evaluation of a check set against an output document.

A text unit passes when at least 90 % of its tokens are covered in one contiguous window of the
output (units under four tokens need the exact sequence). The window grows by two tokens for
each printed item the twin cannot know (`gaps`: LaTeX references and citation keys), and the
matched span is the tightest one with the same coverage. Twin rules of the manifest only widen
what a twin token accepts (`text.accepts`); output tokens are never rewritten. Outcomes are
closed codes; on plain text (a third-party tool without structure) the types it cannot
represent are `not_applicable`. The output of a txt input (`txt=True`, plan r2 §9 and delta r3
Δ10) is read as Markdown, but a plain text cannot carry those types either (`TXT_NOT_APPLICABLE`),
and its cells are judged as on plain text: one window of 90 % of the tokens, never `split`.

Alignment (calibration M4b): headings are matched in reading order (a repeated heading text,
like "Methods" in a structured abstract and in the body, takes the first heading after the
previous match); a unit the generator marks `ambiguous` (under four tokens, or recurring in the
twin) is located by any of its occurrences: its formulas are searched among all the formulas of
the output and its list check passes when one occurrence is a list item. Table shapes count
rows with content only, on both sides (GFM writes an empty header row for a table without one).
"""
from collections import Counter
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
import re

from .text import accepts, align_key, formula_similarity, math_atoms, score_tokens

PASS = "pass"
WINDOW_RATIO = 0.90
FORMULA_RATIO = 0.80
TABLE_FOUND = 0.5
PLAIN_NOT_APPLICABLE = {"heading", "heading_level", "table", "table_shape", "table_header", "list_item",
                        "note_linked", "emphasis", "html_residue", "hyphenation", "spaced_math", "figure", "noise",
                        "formula", "inline_math"}
TXT_NOT_APPLICABLE = PLAIN_NOT_APPLICABLE     # the types a plain text input cannot carry
NUMBER = re.compile(r"-?\d[\d.,]*")


class Stream:
    """A bare token stream with an index: the twin's own text, to find units that recur in it."""
    plain = False

    def __init__(self, tokens):
        self.tokens = tokens
        self.index = {}
        for i, t in enumerate(tokens):
            self.index.setdefault(t, []).append(i)

    def positions(self, token):
        return self.index.get(token, ())


class Matcher:
    def __init__(self, checks, doc, rules=(), txt=False):
        self.doc = doc
        self.plain = doc.plain or txt       # plain text, or the output of a txt input
        self.accept = accepts(set(rules))
        self.units = {c["id"]: c for c in checks if c["type"] in ("paragraph",)}
        self.found = {}
        self.freq = Counter(doc.tokens)
        self.heading_cursor = -1

    # --- text windows ------------------------------------------------------------------------

    def alts(self, token):
        return self.accept(token)

    def _candidates(self, tokens, gaps):
        """(need, alternatives, window length, candidate starts) of a window search."""
        n = len(tokens)
        need = Counter(tokens)
        need_alts = {t: self.alts(t) for t in need}
        length = int(n * 1.1) + 1 + 2 * gaps
        seeds = sorted(range(n), key=lambda i: sum(self.freq.get(a, 0) for a in need_alts[tokens[i]]) or 10**9)[:3]
        starts = set()
        for i in seeds:
            for a in need_alts[tokens[i]]:
                occ = self.doc.positions(a)
                for p in (occ if len(occ) <= 400 else occ[:400]):
                    starts.add(max(0, p - i))
                    starts.add(max(0, p - int(i * 1.1) - 1 - 2 * gaps))
        return need, need_alts, length, sorted(starts)

    def _covered(self, need, need_alts, start, length):
        window = Counter(self.doc.tokens[start:start + length])
        return sum(min(need[t], sum(window.get(a, 0) for a in need_alts[t])) for t in need)

    def _tight(self, need, need_alts, start, length):
        """The tightest span inside the window at `start` that keeps the same coverage."""
        stream = self.doc.tokens
        end = min(start + length, len(stream))
        accepted_by = {}
        for t, alts in need_alts.items():
            for a in alts:
                accepted_by.setdefault(a, []).append(t)
        have = Counter(stream[start:end])

        def drop(x):            # remove one x from the window; False (and undone) if coverage falls
            ts = accepted_by.get(x, ())
            before = [min(need[t], sum(have.get(a, 0) for a in need_alts[t])) for t in ts]
            have[x] -= 1
            if any(min(need[t], sum(have.get(a, 0) for a in need_alts[t])) < b for t, b in zip(ts, before)):
                have[x] += 1
                return False
            return True
        a, b = start, end
        while a < b and drop(stream[a]):
            a += 1
        while b > a and drop(stream[b - 1]):
            b -= 1
        return a, b

    def find(self, tokens, gaps=0):
        """(ratio, start, end) of the best contiguous window for `tokens` (the earliest on ties)."""
        n = len(tokens)
        stream = self.doc.tokens
        if n == 0 or not stream:
            return 0.0, -1, -1
        if n < 4 and not gaps:
            alts = [self.alts(t) for t in tokens]
            for first in alts[0]:
                for p in self.doc.positions(first):
                    if all(p + k < len(stream) and stream[p + k] in alts[k] for k in range(n)):
                        return 1.0, p, p + n
            return 0.0, -1, -1
        need, need_alts, length, starts = self._candidates(tokens, gaps)
        best, best_start = 0, -1
        for s in starts:
            covered = self._covered(need, need_alts, s, length)
            if covered > best:
                best, best_start = covered, s
        if best_start < 0:
            return 0.0, -1, -1
        a, b = self._tight(need, need_alts, best_start, length)
        return best / n, a, b

    def find_all(self, tokens, gaps=0):
        """Every non-overlapping occurrence (ratio, start, end) that passes the window test."""
        n = len(tokens)
        stream = self.doc.tokens
        if n == 0 or not stream:
            return []
        if n < 4 and not gaps:
            alts = [self.alts(t) for t in tokens]
            out, last = [], -1
            for p in sorted({p for first in alts[0] for p in self.doc.positions(first)}):
                if p >= last and all(p + k < len(stream) and stream[p + k] in alts[k] for k in range(n)):
                    out.append((1.0, p, p + n))
                    last = p + n
            return out
        need, need_alts, length, starts = self._candidates(tokens, gaps)
        out, last = [], -1
        for s in starts:
            if s < last:
                continue
            covered = self._covered(need, need_alts, s, length)
            if covered / n >= WINDOW_RATIO:
                a, b = self._tight(need, need_alts, s, length)
                if a >= last:
                    out.append((covered / n, a, b))
                    last = b
        return out

    def unit(self, ref):
        if ref not in self.found:
            check = self.units.get(ref)
            self.found[ref] = self.find(check["tokens"], check.get("gaps", 0)) if check else (0.0, -1, -1)
        return self.found[ref]

    def blocks_of(self, start, end):
        if start < 0:
            return []
        ids = sorted(set(self.doc.token_block[max(0, start):min(end, len(self.doc.tokens))]))
        return [self.doc.blocks[i] for i in ids]

    # --- evaluation --------------------------------------------------------------------------

    def evaluate(self, check):
        t = check["type"]
        if self.plain and t in PLAIN_NOT_APPLICABLE:
            return "not_applicable"
        return getattr(self, "_" + t)(check)

    def _paragraph(self, c):
        ratio, _, _ = self.unit(c["id"])
        return PASS if ratio >= WINDOW_RATIO else ("partial" if ratio >= 0.5 else "missing")

    _footnote = _bib_entry = None

    def _text(self, c):
        ratio, _, _ = self.find(c["tokens"])
        return PASS if ratio >= WINDOW_RATIO else ("partial" if ratio >= 0.5 else "missing")

    def _heading(self, c):
        need = Counter(c["tokens"])
        headings = [i for i, b in enumerate(self.doc.blocks) if b.kind == "heading"]
        # reading order: the first matching heading after the previous match, then any earlier one
        for i in [k for k in headings if k > self.heading_cursor] + [k for k in headings if k <= self.heading_cursor]:
            block = self.doc.blocks[i]
            have = Counter(self.doc.tokens[block.start:block.end])
            covered = sum(min(k, sum(have.get(a, 0) for a in self.alts(t))) for t, k in need.items())
            if covered / len(c["tokens"]) >= WINDOW_RATIO:
                self.found["heading:" + c["id"]] = block.level
                self.heading_cursor = max(self.heading_cursor, i)
                return PASS
        ratio, _, _ = self.find(c["tokens"])
        return "not_heading" if ratio >= WINDOW_RATIO else "missing"

    def _heading_level(self, c):
        a, b = self.found.get("heading:" + c["a"]), self.found.get("heading:" + c["b"])
        if a is None or b is None:
            return "unmatched"
        return PASS if ((b > a) - (b < a)) == c["relation"] else "wrong_level"

    def ambiguous(self, ref):
        return bool(self.units.get(ref, {}).get("ambiguous"))

    def _list_item(self, c):
        ratio, start, end = self.unit(c["ref"])
        if ratio < WINDOW_RATIO:
            return "unmatched"
        spans = [(start, end)]
        if self.ambiguous(c["ref"]):
            unit = self.units[c["ref"]]
            spans = [(a, b) for _, a, b in self.find_all(unit["tokens"], unit.get("gaps", 0))] or spans
        return PASS if any(b.kind == "list_item" for a, z in spans for b in self.blocks_of(a, z)) else "not_list"

    def _note_linked(self, c):
        tokens = self.units_tokens(c["ref"])
        ratio, start, end = self.find(tokens)
        if ratio < WINDOW_RATIO:
            return "unmatched"
        return PASS if any(b.kind == "footnote" for b in self.blocks_of(start, end)) else "not_linked"

    def units_tokens(self, ref):
        return self.footnotes.get(ref, [])

    def _order(self, c):
        ra, sa, _ = self.unit(c["a"])
        rb, sb, _ = self.unit(c["b"])
        if ra < WINDOW_RATIO or rb < WINDOW_RATIO:
            return "unmatched"
        return PASS if sa <= sb else "order"

    def _inline_math(self, c):
        ratio, start, end = self.unit(c["ref"])
        if ratio < WINDOW_RATIO:
            return "unmatched"
        if self.ambiguous(c["ref"]):
            out = [math_atoms(m) for _, m in self.doc.formulas]
        else:
            out = [math_atoms(m) for b in self.blocks_of(start, end) for m in b.math]
        good = sum(1 for f in c["formulas"] if out and max(formula_similarity(f, o) for o in out) >= FORMULA_RATIO)
        return PASS if good >= 0.9 * len(c["formulas"]) else "math"

    def _formula(self, c):
        if not self.doc.formulas:
            return "missing"
        best = max(formula_similarity(c["atoms"], math_atoms(tex)) for _, tex in self.doc.formulas)
        return PASS if best >= FORMULA_RATIO else "missing"

    def _spaced_math(self, c):
        for _, tex in self.doc.formulas:
            letters = re.sub(r"\\(?:mathrm|text|mathit|operatorname)\s*\{([^}]*)\}", r"\1", tex)
            for run in re.findall(r"(?:(?<![A-Za-z])[A-Za-z](?![A-Za-z])[\s~,;\\!]*){3,}", letters):
                if "".join(re.findall(r"[A-Za-z]", run)) in c["words"] and re.search(r"[A-Za-z][\s~\\]+[A-Za-z]", run):
                    return "spaced"
        return PASS

    def _figure(self, c):
        images = [img for img in self.doc.images if img[1]]
        if len(images) < min(c["index"], len(self.doc.images) or 1) or not images:
            return "missing"
        if c["tokens"]:
            ratio, _, _ = self.find(c["tokens"])
            if ratio < WINDOW_RATIO:
                return "caption"
        return PASS

    def _emphasis(self, c):
        need = Counter(c["tokens"])
        for block in self.doc.blocks:
            for kind, run in block.emph:
                if kind != c["kind"]:
                    continue
                have = Counter(score_tokens(run))
                covered = sum(min(k, sum(have.get(a, 0) for a in self.alts(t))) for t, k in need.items())
                if covered / len(c["tokens"]) >= WINDOW_RATIO:
                    return PASS
        return "no_emphasis"

    def _hyphenation(self, c):
        ratio, start, end = self.unit(c["ref"])
        if ratio < WINDOW_RATIO:
            return "unmatched"
        return "broken_word" if any(start <= k < end for k in self.doc.broken) else PASS

    def _html_residue(self, c):
        ratio, start, end = self.unit(c["ref"])
        if ratio < WINDOW_RATIO:
            return "unmatched"
        return "html" if any(b.html and b.kind != "table" for b in self.blocks_of(start, end)) else PASS

    def _noise(self, c):
        for block in self.doc.blocks:
            if block.kind != "table" and align_key(block.text) == c["key"]:
                return "repeated"
        return PASS

    # --- tables ------------------------------------------------------------------------------

    def tables(self):
        return [b for b in self.doc.blocks if b.kind == "table"]

    def _cell_match(self, tokens, value, digits, text):
        if value:
            return any(_number_equal(n, value, digits) for n in NUMBER.findall(text))
        cell = score_tokens(text)
        n = len(tokens)
        alts = [self.alts(t) for t in tokens]
        return any(all(cell[p + k] in alts[k] for k in range(n)) for p in range(len(cell) - n + 1))

    def _cell(self, c):
        if self.plain and not c.get("value"):
            ratio, _, _ = self.find(c["tokens"])
            return PASS if ratio >= WINDOW_RATIO else "missing"
        for table in self.tables():
            for row in table.rows:
                for text in row:
                    if self._cell_match(c["tokens"], c.get("value"), c.get("digits", -1), text):
                        return PASS
        for block in self.doc.blocks:
            if block.kind != "table" and self._cell_match(c["tokens"], c.get("value"), c.get("digits", -1), block.text):
                return PASS if not self.tables() or self.plain else "split"
        return "missing"

    def best_table(self, finder):
        best, best_ratio = None, 0.0
        for table in self.tables():
            texts = [x for row in table.rows for x in row]
            found = sum(1 for tokens in finder if any(self._cell_match(tokens, None, -1, x) for x in texts))
            ratio = found / len(finder)
            if ratio > best_ratio:
                best, best_ratio = table, ratio
        return best if best_ratio >= TABLE_FOUND else None

    def _table(self, c):
        return PASS if self.best_table(c["cells"]) else "not_table"

    def _table_shape(self, c):
        table = self.best_table(c["cells"])
        if table is None:
            return "not_table"
        rows = len(_content_rows(table.rows))
        cols = max((len(r) for r in table.rows), default=0)
        ok = rows == c["rows"] and (c["cols"] is None or cols == c["cols"])
        return PASS if ok else "shape"

    def _table_header(self, c):
        table = self.best_table(c["cells"])
        if table is None:
            return "not_table"
        if table.header_rows:
            head_rows = _content_rows(table.rows[:table.header_rows])
        else:
            head_rows = _content_rows(table.rows)[:1]
        head = [x for row in head_rows for x in row]
        found = sum(1 for tokens in c["header"] if any(self._cell_match(tokens, None, -1, x) for x in head))
        return PASS if found >= WINDOW_RATIO * len(c["header"]) else "header"

    def _table_row(self, c):
        rows = [r for t in self.tables() for r in t.rows] if self.tables() else [[b.text] for b in self.doc.blocks]
        for row in rows:
            numbers = [n for text in row for n in NUMBER.findall(text)]
            k = 0
            for n in numbers:
                if k < len(c["values"]) and _number_equal(n, *c["values"][k]):
                    k += 1
            if k == len(c["values"]):
                return PASS
        return "row"


def _content_rows(rows):
    return [r for r in rows if any(x.strip() for x in r)]


Matcher._footnote = Matcher._text
Matcher._bib_entry = Matcher._text


def _number_equal(token, value, digits):
    """A number of the output equals the twin value at the digits the twin shows (either convention)."""
    token = token.strip(".,")
    candidates = {token.replace(".", "").replace(",", "."), token.replace(",", "")}
    for candidate in candidates:
        try:
            got = Decimal(candidate)
            want = Decimal(value)
        except InvalidOperation:
            continue
        if digits is not None and digits >= 0:
            want = want.quantize(Decimal(1).scaleb(-digits), rounding=ROUND_HALF_UP)
            if got == want or got.quantize(Decimal(1).scaleb(-digits), rounding=ROUND_HALF_UP) == want and \
                    -got.as_tuple().exponent >= digits:
                return True
        elif got == want:
            return True
    return False


def evaluate(checkset, doc, txt=False):
    """[{id, class, type, outcome}] for every check, in generation order; `txt`: the output of a
    txt input."""
    matcher = Matcher(checkset["checks"], doc, checkset.get("rules", ()), txt)
    matcher.footnotes = {c["id"]: c["tokens"] for c in checkset["checks"] if c["type"] == "footnote"}
    results = []
    for check in checkset["checks"]:
        results.append({"id": check["id"], "class": check["class"], "type": check["type"],
                        "outcome": matcher.evaluate(check)})
    return results
