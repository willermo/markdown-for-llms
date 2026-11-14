"""XLSX twin reader and the D06 projection (plan r2, R2-A).

Each sheet becomes a table of its non-empty cells, merged ranges as spans, numbers with the digits
the cell format shows: the General format shows at most 15 significant digits (so a stored
`8.6999999999999993` is `8,7`), thousands are grouped only when the format groups them. The projection selects rows of D07 declared in the manifest and keeps a
row only if its rounded values are the numeric tail of exactly one line of `pdftotext -layout`
of D06; other rows are counted as not determinable.
"""
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
import posixpath
import re

from ..codes import BenchError, Code
from ..model import Builder, Cell
from .safe import SafeZip, parse_xml

S = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
R = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
REL = "{http://schemas.openxmlformats.org/package/2006/relationships}"
BUILTIN_FORMATS = {0: "General", 1: "0", 2: "0.00", 3: "#,##0", 4: "#,##0.00", 9: "0%", 10: "0.00%"}
CELL_REF = re.compile(r"([A-Z]{1,3})(\d+)")
# Italian numbers of the PDF: 1.458 / 8,0 / -0,3 (thousands with dots, decimals with a comma)
PDF_NUMBER = re.compile(r"(?<![\w,.])-?\d{1,3}(?:\.\d{3})+(?:,\d+)?(?!\w)|(?<![\w,.])-?\d+(?:,\d+)?(?![\w.,])")


def col_index(letters):
    n = 0
    for ch in letters:
        n = n * 26 + ord(ch) - 64
    return n - 1


def col_letters(index):
    s, index = "", index + 1
    while index:
        index, rem = divmod(index - 1, 26)
        s = chr(65 + rem) + s
    return s


def decimals_of(code, value):
    if code == "General":
        try:
            shown = Decimal(format(float(value), ".15g"))      # the 15 significant digits a spreadsheet shows
        except (InvalidOperation, ValueError):
            return -1
        return 0 if shown == shown.to_integral() else max(0, -shown.normalize().as_tuple().exponent)
    match = re.search(r"0\.(0+)", code.split(";")[0])
    return len(match.group(1)) if match else 0


def groups_thousands(code):
    """Whether a number format groups thousands (`#,##0`), quoted text and [..] sections aside."""
    section = re.sub(r'"[^"]*"|\[[^\]]*\]', "", code.split(";")[0])
    return "," in section.split(".")[0]


def display(value, digits, grouping=True):
    """Italian display of a number with `digits` decimals (the form printed in D06)."""
    q = Decimal(value).quantize(Decimal(1).scaleb(-digits), rounding=ROUND_HALF_UP) if digits >= 0 else Decimal(value)
    whole, _, frac = f"{abs(q):f}".partition(".")
    whole = f"{int(whole):,}".replace(",", ".") if grouping else str(int(whole))
    return ("-" if q < 0 else "") + whole + ("," + frac if frac else "")


class Workbook:
    def __init__(self, path):
        z = SafeZip(path)
        wb = parse_xml(z.read("xl/workbook.xml"))
        rels = {r.get("Id"): r.get("Target") for r in parse_xml(z.read("xl/_rels/workbook.xml.rels")).iter(REL + "Relationship")}
        shared = []
        if z.has("xl/sharedStrings.xml"):
            for si in parse_xml(z.read("xl/sharedStrings.xml")).findall(S + "si"):
                shared.append("".join(t.text or "" for t in si.iter(S + "t")))
        formats, xfs = dict(BUILTIN_FORMATS), []
        if z.has("xl/styles.xml"):
            st = parse_xml(z.read("xl/styles.xml"))
            for nf in st.iter(S + "numFmt"):
                formats[int(nf.get("numFmtId"))] = nf.get("formatCode")
            cell_xfs = st.find(S + "cellXfs")
            xfs = [int(x.get("numFmtId", "0")) for x in cell_xfs] if cell_xfs is not None else []
        self.sheets = []   # (name, {(row, col): (text, raw_number, digits)}, merges [(r0, c0, r1, c1)])
        for sheet in wb.find(S + "sheets"):
            target = rels[sheet.get(R + "id")]
            part = posixpath.normpath(posixpath.join("xl", target)) if not target.startswith("/") else target.lstrip("/")
            root = parse_xml(z.read(part))
            cells = {}
            for c in root.iter(S + "c"):
                ref = CELL_REF.fullmatch(c.get("r", ""))
                if not ref:
                    continue
                t, v = c.get("t"), c.find(S + "v")
                if t == "inlineStr":
                    text, raw = "".join(x.text or "" for x in c.iter(S + "t")), None
                elif v is None or v.text is None:
                    continue
                elif t == "s":
                    text, raw = shared[int(v.text)], None
                elif t in ("str", "e", "b"):
                    text, raw = v.text, None
                else:
                    style = int(c.get("s", "0"))
                    code = formats.get(xfs[style] if style < len(xfs) else 0, "General")
                    digits = decimals_of(code, v.text)
                    text = display(v.text, digits, groups_thousands(code)) if digits >= 0 else v.text
                    raw = (v.text, digits)
                text = re.sub(r"\s+", " ", text).strip()
                if text:
                    cells[(int(ref.group(2)) - 1, col_index(ref.group(1)))] = (text, raw)
            merges = []
            for mc in root.iter(S + "mergeCell"):
                a, _, z2 = mc.get("ref", "").partition(":")
                ra, rb = CELL_REF.fullmatch(a), CELL_REF.fullmatch(z2 or a)
                if ra and rb:
                    merges.append((int(ra.group(2)) - 1, col_index(ra.group(1)), int(rb.group(2)) - 1, col_index(rb.group(1))))
            self.sheets.append((sheet.get("name").strip(), cells, merges))


def read_xlsx(path):
    b = Builder("xlsx")
    for number, (name, cells, merges) in enumerate(Workbook(path).sheets, 1):
        b.page = number
        b.add("heading", name, level=1)       # the sheet name, as LibreOffice and Marker print it
        span = {(r0, c0): (r1 - r0 + 1, c1 - c0 + 1) for r0, c0, r1, c1 in merges}
        out = []
        for (r, c), (text, raw) in sorted(cells.items()):
            rs, cs = span.get((r, c), (1, 1))
            out.append(Cell(text, r, c, rs, cs) if raw is None else Cell(text, r, c, rs, cs, value=raw[0], digits=raw[1]))
        b.add("table", cells=tuple(out), level=0)
    b.page = 0
    return b.twin


def _rows(spec):
    if isinstance(spec, list):
        return spec
    a, b = spec.split(":")
    return list(range(int(a), int(b) + 1))


def _cols(spec):
    cols = []
    for part in spec.split(","):
        a, b = part.split(":")
        cols += list(range(col_index(a), col_index(b) + 1))
    return cols


def _equal(token, raw):
    plain = token.replace(".", "").replace(",", ".")
    digits = len(plain.split(".")[1]) if "." in plain else 0
    try:
        return Decimal(raw).quantize(Decimal(1).scaleb(-digits), rounding=ROUND_HALF_UP) == Decimal(plain)
    except InvalidOperation:
        return False


def project(path, select, layout_text):
    """Twin of D06: one table per selection with the rows found once in the layout text."""
    wb = {name: cells for name, cells, _ in Workbook(path).sheets}
    lines = layout_text.split("\n")
    numbers = [PDF_NUMBER.findall(line) for line in lines]
    b = Builder("xlsx-projection")
    tables = {}                     # table number -> [cells, next output row]
    for sel in select:
        cells = wb.get(sel["sheet"])
        if cells is None:
            raise BenchError(Code.PROJECTION_INVALID, "sheet " + sel["sheet"])
        out, row_out = tables.setdefault(sel.get("table", len(tables) + 1), [[], 0])
        cols = _cols(sel["cols"])
        for row in _rows(sel["rows"]):
            raws = []
            for c in cols:
                value = cells.get((row - 1, c))
                if value is None or value[1] is None:
                    raws = None
                    break
                raws.append(value[1][0])
            if raws is None:
                b.count("projection_missing")
                continue
            hits = [i for i, nums in enumerate(numbers)
                    if len(nums) >= len(raws) and all(_equal(t, v) for t, v in zip(nums[-len(raws):], raws))]
            if len(hits) != 1:
                b.count("projection_not_determinable")
                continue
            shown = numbers[hits[0]][-len(raws):]
            for k, (token, raw) in enumerate(zip(shown, raws)):
                out.append(Cell(token, row_out, k, value=raw, digits=len(token.split(",")[1]) if "," in token else 0))
            row_out += 1
        tables[sel.get("table", len(tables))][1] = row_out
    for number in sorted(tables):
        b.add("table", cells=tuple(tables[number][0]), level=0)
    return b.twin
