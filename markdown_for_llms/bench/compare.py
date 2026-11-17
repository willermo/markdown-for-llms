"""`mdf bench compare A B`: two reports of measured runs, attempt by attempt (plan r2 §10).

Attempts are aligned by document, route, variant (OCR mode, Datalab mode) and repetition. For
each pair: the status and code of both, the difference of the ratio of each class (B − A), the
checks that passed in A and fail in B (only when the document has the same check set in both),
the times and whether the output is the same (SHA-256 of `document.md`). It flags the differences
of identity of the two runs (package and its `RECORD`, toolchain, weights, manifest, exclusions,
versions of the bench) and the documents whose effective input differs (D23 and D24, which
LibreOffice derives differently at every run). A regression is a complete attempt that fails in
B, or a ratio that falls by more than the tolerance; `--fail-on-regression` exits with 1 then.
Numbers and closed codes only.
"""
import json
from pathlib import Path

from .codes import BenchError, Code
from .report import CLASSES, SCHEMA

MAX_REPORT_BYTES = 256 * 1048576


def load_report(path):
    try:
        data = Path(path).read_bytes()
    except OSError:
        raise BenchError(Code.REPORT_INVALID, "report unreadable") from None
    if len(data) > MAX_REPORT_BYTES:
        raise BenchError(Code.REPORT_INVALID, "report too large")
    try:
        report = json.loads(data)
    except ValueError:
        raise BenchError(Code.REPORT_INVALID, "report JSON") from None
    if not isinstance(report, dict) or report.get("schema") != SCHEMA or not isinstance(report.get("documents"), list):
        raise BenchError(Code.REPORT_INVALID, "report schema")
    return report


def attempts_of(report):
    """{(document, route, variant_sha8, repeat): (document entry, attempt)}."""
    out = {}
    for doc in report["documents"]:
        for a in doc.get("attempts", []):
            out[(doc["id"], a["route"], a["variant_sha8"], a["repeat"])] = (doc, a)
    return out


def identity_differences(a, b):
    fields = [("package", "version"), ("package", "record_sha256"), ("toolchain", "content_sha256"),
              ("toolchain", "image_id"), ("weights", "manifest_sha256"), ("manifest", "sha256"),
              ("bench", "generator"), ("bench", "matcher"), ("bench", "normalizer")]
    diff = [f"{x}.{y}" for x, y in fields if (a.get(x) or {}).get(y) != (b.get(x) or {}).get(y)]
    if a.get("exclusions_sha256") != b.get("exclusions_sha256"):
        diff.append("exclusions_sha256")
    return diff


def _ratio(attempt, cls):
    scores = attempt.get("scores")
    return None if not scores else scores[cls]["ratio"]


def compare(a, b, tolerance=0.005):
    pa, pb = attempts_of(a), attempts_of(b)
    pairs, regressions = [], 0
    for key in sorted(set(pa) & set(pb)):
        (da, xa), (db, xb) = pa[key], pb[key]
        delta = {}
        for cls in CLASSES:
            ra, rb = _ratio(xa, cls), _ratio(xb, cls)
            delta[cls] = None if ra is None or rb is None else round(rb - ra, 6)
        same_checks = (da.get("checks") or {}).get("sha256") is not None and \
            (da.get("checks") or {}).get("sha256") == (db.get("checks") or {}).get("sha256")
        newly_failed = []
        if same_checks and isinstance(xa.get("failed"), list) and isinstance(xb.get("failed"), list):
            before = {f["id"] for f in xa["failed"]}
            newly_failed = sorted(f["id"] for f in xb["failed"] if f["id"] not in before)
        status_worse = xa["status"] == "complete" and xb["status"] != "complete"
        fell = [cls for cls, d in delta.items() if d is not None and d < -tolerance]
        regression = status_worse or bool(fell)
        regressions += regression
        ta, tb = xa["timing"]["total_s"], xb["timing"]["total_s"]
        pairs.append({"document": key[0], "route": key[1], "variant_sha8": key[2], "repeat": key[3],
                      "status": [xa["status"], xb["status"]], "error": [xa.get("error"), xb.get("error")],
                      "delta": delta, "fell": fell, "same_checks": same_checks, "newly_failed": newly_failed,
                      "same_output": None if xa.get("markdown_sha256") is None or xb.get("markdown_sha256") is None
                      else xa["markdown_sha256"] == xb["markdown_sha256"],
                      "total_s": [ta, tb], "regression": regression})
    inputs = sorted({d["id"] for d in a["documents"] for e in b["documents"]
                     if d["id"] == e["id"] and d.get("input_sha256") and e.get("input_sha256")
                     and d["input_sha256"] != e["input_sha256"]})
    return {"identity_differences": identity_differences(a, b), "input_differs": inputs,
            "tolerance": tolerance, "pairs": pairs, "regressions": regressions,
            "only_in_a": [list(k) for k in sorted(set(pa) - set(pb))],
            "only_in_b": [list(k) for k in sorted(set(pb) - set(pa))]}


def print_result(result, out=print):
    if result["identity_differences"]:
        out("identity differs: " + ", ".join(result["identity_differences"]))
    if result["input_differs"]:
        out("effective input differs: " + ", ".join(result["input_differs"]))
    for p in result["pairs"]:
        delta = " ".join(f"{c[:4]} {'n/a' if v is None else f'{v:+.3f}'}" for c, v in p["delta"].items())
        same = {True: "same output", False: "other output", None: "-"}[p["same_output"]]
        out(f"{p['document']} {p['route']} {p['variant_sha8']} k{p['repeat']} {p['status'][0]}->{p['status'][1]} "
            f"{delta} newly failed {len(p['newly_failed'])} {same}" + (" REGRESSION" if p["regression"] else ""))
    out(f"{len(result['pairs'])} pairs, {result['regressions']} regressions (tolerance {result['tolerance']}), "
        f"{len(result['only_in_a'])} only in A, {len(result['only_in_b'])} only in B")
