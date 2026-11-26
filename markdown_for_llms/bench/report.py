"""The report of a measured run: `report.json` (`mdf-bench-report/1`) and `report.md` (plan r2 §9, App. E).

Numbers and closed codes only (S20, delta r3 Δ10): of a failed check its id, class, type and
outcome; of the problems of the engine their code and page; never a `detail`, a `path`, a file
name, the arguments of a tool, the diagnostics or the text of a document. Every string read from
an `engine.json` is checked against a closed set (the codes of the engine and of the bench, the
problems of the contract, the declared limitations of the routes) or a short identifier pattern,
and is `unrecognized` otherwise.

Scores of an attempt (§9). A complete output is read with `outdoc.from_markdown`, its assets under
`out/`, and judged on the checks of its document (`match.evaluate`), with the exclusions of the
document (`score.scores`). The output of a txt input carries `format_limit: "txt"`: the types a
plain text cannot carry are not applicable and its cells are judged as plain text (Δ10). A txt
input also has its `encoding` check: the bytes of `document.md` must have the SHA-256 of the oracle
(the UTF-8 text before the recode), compared before the Markdown is read; D35 and D36 have only
this check. A failed attempt (adopted or not) is an outcome with its code and `scores: null`.

Next to the scores: the arbiter of the document and the tools it shares with the route (a route
that shares one is not arbitrated and stays out of the averages), the timings of the engine
(without the per-page values) and seconds per page, the memory peaks of the steps, the problems,
the page methods of Marker, and language and scripts. Summaries (§9): format × route, profile ×
route and script × route, each the mean of the arbitrated documents with their minimum and number,
against the provisional thresholds of `data/thresholds.json` (`threshold` when every document of the
group has the same one, then `mean_meets`; `below_threshold`, the documents under the threshold of
their own profile); format × route also carries the median seconds per page and, for Datalab, the
cost from the ledger; then the routes not arbitrated and the cells of the matrix not measured, with
the reason (D20 is never sent to Datalab).

Datalab (plan r2 §12). Each attempt carries its pages, estimate, billed cost and whether the
result was deleted on Datalab (numbers from `engine.json`); an attempt whose reservation is in the
ledger without a result is `datalab_unresolved`. `cost` sums the ledger lines of the run per
mode (estimates, billed costs, what counts), the summary format × route per cell (`cell_costs`:
billed costs known, estimates, and the uncertain cost, an estimate that stands for a cost not
billed), and `datalab` carries the warning on the transit. `notes` holds fixed remarks on the
routes (`djvutxt_plain_text`).

The run (`run`) says how it ended: `status` (`complete` or `stopped`) and `stop`, the closed code of
a stop. A report written again from the saved outputs (`mdf bench report`, `run.rebuild_report`)
carries `rebuilt`: when, and the package whose scoring it follows; nothing was converted again.
"""
from collections import Counter
from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path
from statistics import median

from . import BENCH_VERSION, GENERATOR_VERSION, MATCHER_VERSION, NORMALIZER_VERSION
from .cache import canonical, write_atomic
from .codes import BenchError, Code

SCHEMA = "mdf-bench-report/1"
UNRECOGNIZED = "unrecognized"
CLASSES = ("content", "structure", "form")
MAX_PAGES_LISTED = 200
# closed sets of the strings an engine.json carries (anything else is `unrecognized`)
PAGE_CLASSES = ("native", "image", "ocr_layer")                    # engine.pdfprep
PAGE_METHODS = ("pdftext", "surya", "gemini")                      # Marker 2.2.0 text_extraction_method
# modules of the Python children: `child:` in a container (MFL051), `bwrap:` in the results written before
ISOLATED = ("child:pkgaudit", "child:pdfprep", "child:marker_child", "bwrap:pkgaudit", "bwrap:pdfprep", "bwrap:marker_child")
PAGE_TIME_KEYS = ("unit", "median", "max", "count")
DATALAB_WARNING_CODE = "datalab_transit_training_retention_us"
DJVUTXT_NOTE = ("djvutxt writes the text layer of a DjVu without structure: the checks of emphasis and linked "
                "notes fail on it, and no local route recovers them from a scan, so crosscheck suspects on those "
                "checks of D20 are expected.")


def _codes():
    return {c.value for c in Code}


def _closed(value, allowed):
    return value if isinstance(value, str) and value in allowed else UNRECOGNIZED


def _tools():
    from ..engine.tools import TOOLS
    return set(TOOLS) | set(ISOLATED)


def _stages():
    from ..engine.pages import STAGE_BOUNDS
    return {b[0] for b in STAGE_BOUNDS}


def _encoding(value):
    """A codec name Python knows (the canonical names the text route reports), else unrecognized."""
    import codecs
    if not isinstance(value, str) or len(value) > 32:
        return UNRECOGNIZED
    try:
        codecs.lookup(value)
    except LookupError:
        return UNRECOGNIZED
    return value


def _num(value):
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _known_limits():
    from ..engine import pages
    return set(pages.LIMITS) | set(pages.SERVER_LIMITS) | set(pages.BALANCED_SERVER_LIMITS) | set(pages.FORMER_LIMITS)


def _dict(value):
    return value if isinstance(value, dict) else {}


def _decimal(value):
    """A non-negative decimal amount as a string (cents of `engine.json`), else None."""
    if isinstance(value, bool) or not isinstance(value, (int, float, str)) or len(str(value)) > 32:
        return None
    try:
        amount = Decimal(str(value))
    except InvalidOperation:
        return None
    return str(amount) if amount.is_finite() and amount >= 0 else None


def _datalab_record(r):
    """Pages, estimate, billed cost and deletion of a Datalab attempt (numbers only)."""
    cost, transit = _dict(r.get("cost")), _dict(r.get("transit"))
    deleted = transit.get("deleted")
    return {"pages": _num(cost.get("pages")), "estimated_cents": _decimal(cost.get("estimated_cents")),
            "billed_cents": _decimal(cost.get("billed_cents")), "cap_usd": _decimal(cost.get("cap_usd")),
            "left_machine": transit.get("left_machine") is True, "deleted": deleted if isinstance(deleted, bool) else None}


def _server_record(r):
    """The server of a marker-server attempt (numbers and closed values only, MFL033): provider,
    kind and GPU of the endpoint, mode, readiness, requests answered 2xx and not, retried and
    uncertain requests, clean close, reservation and measured bound."""
    comp = _dict(_dict(r.get("components")).get("server"))
    ep = _dict(comp.get("endpoint"))
    relay, cost = _dict(r.get("marker_server")), _dict(r.get("cost"))
    route = _dict(r.get("marker_server_route"))
    statuses = {k: _num(v) for k, v in _dict(relay.get("by_status")).items() if isinstance(k, str) and k.isdigit()}
    readiness = _dict(relay.get("readiness"))
    return {"endpoint": ep.get("name") if isinstance(ep.get("name"), str) else None,
            "provider": ep.get("provider") if ep.get("provider") in ("runpod", "modal") else None,
            "kind": ep.get("kind") if ep.get("kind") in ("cpu", "local", "user", "runpod", "modal") else None,
            "gpu": ep.get("gpu") if isinstance(ep.get("gpu"), str) else None,
            "mode": comp.get("mode") if comp.get("mode") in ("balanced", "accurate") else None,
            "readiness_s": _num(readiness.get("wait_seconds")),
            "requests_2xx": sum(v or 0 for k, v in statuses.items() if k.startswith("2")),
            "requests_other": sum(v or 0 for k, v in statuses.items() if not k.startswith("2")),
            "retried": _num(_dict(route.get("server")).get("exceptions")), "uncertain": _num(relay.get("uncertain")),
            "clean": relay.get("clean") if isinstance(relay.get("clean"), bool) else None,
            "reserve_cents": _decimal(cost.get("reserve_cents")), "bound_cents": _decimal(cost.get("bound_cents"))}


def _list(value):
    return value if isinstance(value, list) else []


def arbiter_record(doc, route):
    from .selfcheck import arbiter, shares
    tool = arbiter(doc)
    record = {"tool": tool, "shares": shares(doc, route)}
    record["arbitrated"] = not record["shares"]
    if tool == "writer-html" and doc["twin"]["kind"] == "odt":
        record["formulas"] = "source-coherence"         # D26: StarMath of its own source (Δ4.6)
    return record


def _timing(doc, r):
    t = _dict(r.get("timings"))
    total = _num(t.get("total_seconds"))
    pages = doc["pages"] if doc["pages"] is not None else doc.get("pages_estimate")
    steps = [{"tool": _closed(s.get("tool"), _tools()), "seconds": _num(s.get("seconds")), "cpu_seconds": _num(s.get("cpu_seconds")),
              "memory_peak_bytes": _num(s.get("memory_peak_bytes")), "oom_kill": _num(s.get("oom_kill")),
              "exit": _num(s.get("exit"))}
             for s in _list(_dict(r.get("route")).get("steps")) if isinstance(s, dict)]
    page_times = t.get("pages") if isinstance(t.get("pages"), dict) else None
    return {"total_s": total, "steps": steps,
            "stages": {_closed(k, _stages()): _num(v) for k, v in _dict(t.get("stages")).items()},
            "pages": None if page_times is None else
            {k: (_num(v) if k != "unit" else _closed(v, ("s",))) for k, v in page_times.items() if k in PAGE_TIME_KEYS},
            "s_per_page": round(total / pages, 3) if total is not None and pages else None,
            "pages_basis": "pages" if doc["pages"] is not None else "pages_estimate"}


def _problems(r):
    from ..engine.contract import PROBLEMS
    codes = {(k if k in PROBLEMS else UNRECOGNIZED): _num(v) for k, v in _dict(r.get("problem_codes")).items()}
    pages = {}
    for p in _list(r.get("problems")):
        if isinstance(p, dict) and isinstance(p.get("page"), int) and not isinstance(p.get("page"), bool):
            pages.setdefault(p.get("code") if p.get("code") in PROBLEMS else UNRECOGNIZED, set()).add(p["page"])
    return codes, {k: sorted(v)[:MAX_PAGES_LISTED] for k, v in sorted(pages.items())}


def _pages(r):
    p = _dict(r.get("pages"))
    methods = Counter(_closed(x.get("marker_method"), PAGE_METHODS) for x in _list(p.get("pages")) if isinstance(x, dict))
    return ({_closed(k, PAGE_CLASSES): _num(v) for k, v in _dict(p.get("classes")).items()}, dict(sorted(methods.items())),
            _num(p.get("count")))


def score_attempt(run, doc, a, exclusions):
    """The record of the final generation of a key, and the outcome of its checks (written apart)."""
    from .match import evaluate
    from .outdoc import from_markdown
    from .run import read_result
    from .score import scores
    out = run.dir / "attempts" / a["id"] / "out"
    r = read_result(out) or {}
    status = r.get("status") if r.get("status") in ("complete", "failed") else UNRECOGNIZED
    error = _dict(r.get("error")).get("code")
    codes, problem_pages = _problems(r)
    classes, methods, page_count = _pages(r)
    record = {"attempt_id": a["id"], "route": a["route"], "variant": a["variant"], "variant_sha8": a["v8"],
              "repeat": a["k"], "generation": a["g"], "adopted": a["adopted"], "status": status,
              "error": None if error is None else (error if error in _codes() else UNRECOGNIZED),
              "arbiter": arbiter_record(doc, a["route"]) if doc["twin"] is not None else
              {"tool": "oracle", "shares": [], "arbitrated": True},
              "format_limit": "txt" if doc["format"] == "txt" else None, "encoding": None,
              "timing": _timing(doc, r), "problem_codes": codes, "problem_pages": problem_pages,
              "page_classes": classes, "page_methods": methods, "pages": page_count,
              "limits": [x if x in _known_limits() else UNRECOGNIZED for x in _list(r.get("limits"))],
              "markdown_sha256": None, "inventory_sha256": None, "scores": None, "quality": "unverified", "failed": []}
    peaks = [s["memory_peak_bytes"] for s in record["timing"]["steps"] if s["memory_peak_bytes"] is not None]
    record["memory_peak_bytes"] = max(peaks) if peaks else None
    if a["route"] == "datalab":
        record["datalab"] = _datalab_record(r)
    if a["route"] == "marker-server":
        record["server"] = _server_record(r)
    if a["status"] == "unresolved":
        # the reservation is in the ledger and there is no result: never sent again (§7.4)
        record.update(status="failed", error=Code.DATALAB_UNRESOLVED.value)
    if record["status"] != "complete":
        return record, None
    data = (out / "document.md").read_bytes()
    record["markdown_sha256"] = hashlib.sha256(data).hexdigest()
    record["inventory_sha256"] = hashlib.sha256(canonical(_list(r.get("inventory")))).hexdigest()
    if doc["format"] == "txt":
        oracle = (doc["input"].get("reference") or {}).get("oracle_sha256")     # none in a personal set
        detected = _dict(r.get("text")).get("encoding")
        record["encoding"] = {"expected": doc.get("expected_encoding"), "detected": _encoding(detected),
                              "bom": _dict(r.get("text")).get("bom") is True,
                              "decode_equals_oracle": None if oracle is None else record["markdown_sha256"] == oracle}
    checkset = run.checksets.get(doc["id"])
    if checkset is None:
        return record, None
    results = evaluate(checkset, from_markdown(data.decode("utf-8"), assets_dir=str(out)), txt=doc["format"] == "txt")
    record["scores"] = scores(checkset, results, exclusions, doc["profile"])
    record["failed"] = [{k: x[k] for k in ("id", "class", "type", "outcome")} for x in results
                        if x["outcome"] not in ("pass", "not_applicable")]
    return record, results


def _repeats(attempts):
    """Per route and variant with repetitions: the SHA-256 of each one and the median time."""
    groups = {}
    for a in attempts:
        groups.setdefault((a["route"], a["variant_sha8"]), []).append(a)
    out = []
    for (route, v8), items in groups.items():
        if len(items) < 2:
            continue
        shas = [a["markdown_sha256"] for a in items]
        times = [a["timing"]["total_s"] for a in items if a["timing"]["total_s"] is not None]
        out.append({"route": route, "variant_sha8": v8, "n": len(items), "markdown_sha256": shas,
                    "inventory_sha256": [a["inventory_sha256"] for a in items],
                    "deterministic": None if None in shas else len(set(shas)) == 1,
                    "median_total_s": median(times) if times else None})
    return out


def thresholds_summary():
    """The provisional thresholds of the scores (`data/thresholds.json`), numbers only."""
    from .score import thresholds
    t = thresholds()
    return {"provisional": t["provisional"], "default": t["default"], "profiles": t["profiles"]}


def _aggregate(table):
    """Mean, minimum and number of the arbitrated documents of a group, from {document: {ratios of its
    attempts, threshold of its profile}}, against the provisional thresholds (§9): `threshold` when every
    document has the same one (then `mean_meets`), and `below_threshold`, the documents whose mean ratio
    is under the threshold of their own profile."""
    values = [(sum(v["ratios"]) / len(v["ratios"]), v["threshold"]) for v in table.values()]
    if not values:
        return None
    ratios = [r for r, _ in values]
    limits = {t for _, t in values}
    threshold = next(iter(limits)) if len(limits) == 1 else None
    mean = sum(ratios) / len(ratios)
    return {"mean": round(mean, 4), "min": round(min(ratios), 4), "n": len(ratios), "threshold": threshold,
            "mean_meets": None if threshold is None else mean >= threshold,
            "below_threshold": sum(1 for r, t in values if t is not None and r < t)}


def _empty_cost():
    return {"sends": 0, "estimated_cents": "0", "billed_cents": "0", "billed_known": 0, "uncertain_cents": "0",
            "uncertain": 0, "counted_cents": "0"}


def cell_costs(entries, run_id, attempts, docs):
    """The Datalab spend of the run `run_id` per (format, route, variant_sha8), from the ledger lines
    (every generation that reserved, as in `cost`): `sends` and `estimated_cents`, the reservations and
    their estimates; `billed_cents` and `billed_known`, the costs billed by Datalab (0 for a request
    refused before its acceptance); `uncertain_cents` and `uncertain`, the estimates that stand for a
    cost never billed (no completion, or a completion with `billed_cents: null`); `counted_cents`, what
    the ledger counts (billed + uncertain). `attempts` are those of the state, `docs` the documents by id."""
    billed = {e["attempt_id"]: Decimal(str(e["billed_cents"])) for e in entries
              if e.get("run_id") == run_id and e.get("kind") == "complete" and e.get("billed_cents") is not None}
    out = {}
    for e in entries:
        if e.get("run_id") != run_id or e.get("kind") != "reserve":
            continue
        a = attempts.get(e["attempt_id"]) or {}
        doc = docs.get(a.get("document")) or {}
        key = (doc.get("format", UNRECOGNIZED), a.get("route", UNRECOGNIZED), a.get("v8", UNRECOGNIZED))
        row = out.setdefault(key, {"sends": 0, "estimated_cents": Decimal(0), "billed_cents": Decimal(0),
                                   "billed_known": 0, "uncertain_cents": Decimal(0), "uncertain": 0})
        estimate = Decimal(str(e["estimate_cents"]))
        row["sends"] += 1
        row["estimated_cents"] += estimate
        if e["attempt_id"] in billed:
            row["billed_cents"] += billed[e["attempt_id"]]
            row["billed_known"] += 1
        else:
            row["uncertain_cents"] += estimate
            row["uncertain"] += 1
    return {key: {**{k: (str(v) if isinstance(v, Decimal) else v) for k, v in row.items()},
                  "counted_cents": str(row["billed_cents"] + row["uncertain_cents"])} for key, row in out.items()}


def _summary(docs, records, selection, costs=None):
    """Summaries of the report (§9). `costs`: None for a run without Datalab, else the result of
    `cell_costs` or {"error": code} when the ledger cannot be read."""
    from ..engine.routes import DATALAB, MATRIX
    cells, by_script, by_profile, not_arbitrated = {}, {}, {}, {}
    attempted = set()
    for doc in docs:
        entry = records[doc["id"]]
        for a in entry["attempts"]:
            attempted.add((doc["format"], a["route"]))
            key = (doc["format"], a["route"], a["variant_sha8"])
            cell = cells.setdefault(key, {"format": doc["format"], "route": a["route"], "variant": a["variant"],
                                          "variant_sha8": a["variant_sha8"], "complete": 0, "failed": 0,
                                          "_scores": {}, "_spp": []})
            cell["complete" if a["status"] == "complete" else "failed"] += 1
            if a["timing"]["s_per_page"] is not None and a["status"] == "complete":
                cell["_spp"].append(a["timing"]["s_per_page"])
            if not a["arbiter"]["arbitrated"]:
                na = not_arbitrated.setdefault(key, {"format": doc["format"], "route": a["route"],
                                                     "variant_sha8": a["variant_sha8"], "documents": [],
                                                     "shares": a["arbiter"]["shares"]})
                if doc["id"] not in na["documents"]:
                    na["documents"].append(doc["id"])
                continue
            if a["scores"] is None:
                continue
            groups = [cell["_scores"], by_profile.setdefault((doc.get("profile"), a["route"], a["variant_sha8"]), {})]
            groups += [by_script.setdefault((script, a["route"], a["variant_sha8"]), {}) for script in doc["scripts"]]
            for cls in CLASSES:
                score = a["scores"][cls]
                if score["ratio"] is None:
                    continue
                for table in groups:
                    slot = table.setdefault(cls, {}).setdefault(doc["id"], {"ratios": [], "threshold": score.get("threshold")})
                    slot["ratios"].append(score["ratio"])
    format_route = []
    for key, cell in cells.items():
        spp = cell.pop("_spp")
        table = cell.pop("_scores")
        cell["scores"] = {cls: _aggregate(table.get(cls, {})) for cls in CLASSES}
        cell["s_per_page_median"] = median(spp) if spp else None
        if costs is None or cell["route"] != DATALAB:
            cell["cost"] = None                 # a local route costs nothing
        else:
            cell["cost"] = dict(costs) if "error" in costs else costs.get(key, _empty_cost())
        format_route.append(cell)
    profile_route = [{"profile": p, "route": r, "variant_sha8": v,
                      "scores": {cls: _aggregate(table.get(cls, {})) for cls in CLASSES}}
                     for (p, r, v), table in sorted(by_profile.items(), key=lambda kv: tuple(map(str, kv[0])))]
    script_route = [{"script": s, "route": r, "variant_sha8": v,
                     "scores": {cls: _aggregate(table.get(cls, {})) for cls in CLASSES}}
                    for (s, r, v), table in sorted(by_script.items())]
    not_measured = []
    for fmt in sorted({d["format"] for d in docs}):
        default, alternatives = MATRIX[fmt]
        for route in (default, *alternatives):
            if (fmt, route) in attempted:
                continue
            from .run import DATALAB_EXCLUDED
            same = [d for d in docs if d["format"] == fmt]
            unavailable = all(records[d["id"]]["status"] == "unavailable" for d in same)
            excluded = route == "datalab" and all(d["id"] in DATALAB_EXCLUDED for d in same)
            reason = "document_unavailable" if unavailable else "datalab_excluded" if excluded else "not_selected"
            entry = {"format": fmt, "route": route, "reason": reason}
            if excluded:
                entry["documents"] = {d["id"]: DATALAB_EXCLUDED[d["id"]] for d in same}
            not_measured.append(entry)
    return {"format_route": sorted(format_route, key=lambda c: (c["format"], c["route"], c["variant_sha8"])),
            "profile_route": profile_route, "script_route": script_route,
            "not_arbitrated": list(not_arbitrated.values()), "not_measured": not_measured,
            "thresholds": thresholds_summary()}


def build(run, identity_changed=False, rebuilt=None, out_dir=None):
    """`report.json` of a run; `rebuilt` ({at, package}) when it is written again from the saved
    outputs by `run.rebuild_report`. The outcome of the checks of each attempt goes to
    `<out_dir>/attempts/<id>/results.json`, by default the run directory (MFL044 (d))."""
    from . import exclusions as excl_mod
    from .starmath import VERSION as STARMATH_VERSION
    state = run.state
    personal = run.manifest.personal
    records = {}
    warnings = []
    for doc in run.docs:
        ds = state["documents"].get(doc["id"], {"status": "unavailable", "code": None})
        entry = {"id": doc["id"], "format": doc["format"], "profile": doc["profile"], "pages": doc["pages"],
                 "pages_estimate": doc.get("pages_estimate")}
        if not personal:
            entry.update(language=doc["language"], scripts=doc["scripts"])
        if ds["status"] != "available":
            entry.update(status="unavailable", code=ds.get("code"), attempts=[], repeats=[])
            records[doc["id"]] = entry
            continue
        checkset = run.checksets.get(doc["id"])
        entry.update(status="measured", warnings=[w for w in ds.get("warnings", []) if w in _codes()],
                     checks=None if checkset is None else {
                         "sha256": checkset["checks_sha256"],
                         "counts": {cls: sum(1 for c in checkset["checks"] if c["class"] == cls) for cls in CLASSES},
                         "not_determinable": checkset["not_determinable"],
                         "feedback": checkset.get("feedback", {"entries": 0, "missing": 0})})
        if checkset is not None and entry["checks"]["feedback"]["missing"]:
            entry["warnings"].append(Code.FEEDBACK_POSITION_MISSING.value)
        if not personal:
            entry["input_sha256"] = ds["input_sha256"]
        elif entry["checks"] is not None:
            del entry["checks"]["sha256"]
        exclusions = excl_mod.for_document(run.exclusions, doc["id"])
        attempts = []
        for key in run.keys:
            if key["document"] != doc["id"]:
                continue
            ids = state["keys"].get(f"{doc['id']}|{key['route']}|{key['v8']}|{key['k']}") or []
            if not ids or state["attempts"][ids[-1]]["status"] not in ("done", "unresolved", "stopped"):
                continue
            a = state["attempts"][ids[-1]]
            record, results = score_attempt(run, doc, a, exclusions)
            if results is not None:
                target = Path(out_dir if out_dir is not None else run.dir) / "attempts" / a["id"]
                if out_dir is not None:
                    target.mkdir(parents=True, exist_ok=True, mode=0o700)
                write_atomic(target / "results.json", json.dumps(results, sort_keys=True).encode("utf-8"))
            attempts.append(record)
        entry["repeats"] = _repeats(attempts)
        if personal:            # numbers only (§13): no hash of the input or the output, no check ids
            for record in attempts:
                for name in ("markdown_sha256", "inventory_sha256"):
                    record.pop(name)
                record["failed"] = len(record["failed"])
            for rep in entry["repeats"]:
                rep.pop("markdown_sha256")
                rep.pop("inventory_sha256")
        entry["attempts"] = attempts
        if not attempts:
            entry["status"] = "not_measured"
        records[doc["id"]] = entry
    if identity_changed:
        warnings.append(Code.IDENTITY_CHANGED.value)
    datalab = ledger = costs = None
    if state.get("datalab"):
        warnings.append(DATALAB_WARNING_CODE)
        datalab = {"modes": state["selection"].get("datalab_modes"), "warning": DATALAB_WARNING_CODE,
                   "caps": dict(state["datalab"])}
        ledger = _ledger(run)
        costs = {"error": ledger} if isinstance(ledger, str) else \
            cell_costs(ledger[0], state["run_id"], state["attempts"], {d["id"]: d for d in run.docs})
    routes_seen = {a["route"] for d in records.values() for a in d.get("attempts", [])}
    notes = ["djvutxt_plain_text"] if "djvutxt" in routes_seen else []
    sel = state["selection"]
    executions = state["executions"]
    identity = state["identity"]
    return {"schema": SCHEMA, "valid": not identity_changed,
            "bench": {"version": BENCH_VERSION, "generator": GENERATOR_VERSION, "matcher": MATCHER_VERSION,
                      "normalizer": NORMALIZER_VERSION, "starmath": STARMATH_VERSION},
            "package": identity["package"],
            "run": {"id": state["run_id"], "created_at": state["created_at"], "level": sel["level"],
                    "documents": len(sel["docs"]), "routes": sel["routes"], "ocr": sel["ocr"], "repeat": sel["repeat"],
                    "set": sel["set"], "resumed": len(executions) > 1, "executions": len(executions),
                    "generations": dict(state["generations"]),
                    "bench_peak_rss_bytes": max((e.get("peak_rss_bytes") or 0 for e in executions), default=None),
                    "exit": executions[-1].get("exit") if executions else None, "identity_changed": identity_changed,
                    "status": state.get("status"), "stop": state.get("stop")},
            "manifest": {"sha256": state["manifest"]["sha256"], "documents": len(run.manifest.documents)},
            "exclusions_sha256": state["exclusions"]["sha256"],
            "environment": state["environment"], "toolchain": identity["toolchain"], "weights": identity["weights"],
            "caps": state["caps"], "cost": cost(run, ledger) if datalab else None, "datalab": datalab,
            "server": state.get("server"),
            "documents": [records[d["id"]] for d in run.docs],
            "summary": _summary(run.docs, records, sel, costs), "warnings": warnings, "notes": notes,
            "rebuilt": rebuilt}


def _ledger(run):
    """(entries, per attempt, total) of the ledger, or the closed code of why it cannot be read."""
    from . import spend
    try:
        entries = spend.datalab_rows(run.ledger_entries())
        per_attempt, total = spend.reconcile(entries)
    except BenchError as exc:
        return exc.code.value if exc.code.value in _codes() else UNRECOGNIZED
    except OSError:
        return Code.SPEND_LEDGER_UNWRITABLE.value
    return entries, per_attempt, total


def cost(run, ledger=None):
    """The Datalab spend of the run from the ledger (Δ6): per mode the estimates reserved, the billed
    costs known and what counts (the billed cost when known, the estimate otherwise)."""
    ledger = _ledger(run) if ledger is None else ledger
    if isinstance(ledger, str):
        return {"error": ledger}
    entries, per_attempt, total = ledger
    run_id = run.state["run_id"]
    modes, billed = {}, {}
    for e in entries:
        if e.get("run_id") == run_id and e.get("kind") == "complete" and e.get("billed_cents") is not None:
            billed[e["attempt_id"]] = Decimal(str(e["billed_cents"]))
    for e in entries:
        if e.get("run_id") != run_id or e.get("kind") != "reserve":
            continue
        attempt = run.state["attempts"].get(e["attempt_id"]) or {}
        mode = _dict(attempt.get("variant")).get("datalab_mode")
        row = modes.setdefault(mode if isinstance(mode, str) else UNRECOGNIZED,
                               {"sends": 0, "pages": 0, "estimated_cents": Decimal(0), "billed_cents": Decimal(0),
                                "billed_unknown": 0, "counted_cents": Decimal(0)})
        row["sends"] += 1
        row["pages"] += e.get("pages") if isinstance(e.get("pages"), int) else 0
        row["estimated_cents"] += Decimal(str(e["estimate_cents"]))
        if e["attempt_id"] in billed:
            row["billed_cents"] += billed[e["attempt_id"]]
        else:
            row["billed_unknown"] += 1
        row["counted_cents"] += per_attempt[(run_id, e["attempt_id"])]
    by_mode = {m: {k: (str(v) if isinstance(v, Decimal) else v) for k, v in row.items()} for m, row in sorted(modes.items())}
    run_total = sum((v for (r, _), v in per_attempt.items() if r == run_id), Decimal(0))
    return {"by_mode": by_mode, "run_counted_cents": str(run_total), "ledger_counted_cents": str(total)}


def _fmt(value, digits=3):
    return "—" if value is None else f"{value:.{digits}f}" if isinstance(value, float) else str(value)


def _score_cell(agg):
    """`mean (min m, n k, b below t)` of an aggregate; without `t` when the documents have different thresholds."""
    if agg is None:
        return "—"
    text = f"{agg['mean']:.3f} (min {agg['min']:.3f}, n {agg['n']}, {agg.get('below_threshold', 0)} below"
    return text + (f" {agg['threshold']:g})" if agg.get("threshold") is not None else ")")


def _cost_cell(cost):
    if cost is None:
        return "—"
    if "error" in cost:
        return cost["error"]
    return f"{cost['billed_cents']} billed + {cost['uncertain_cents']} uncertain (estimate {cost['estimated_cents']})"


def _thresholds_line(thresholds, with_cost):
    if not thresholds:
        return []
    def text(t):
        return ", ".join(f"{cls} {t[cls]:g}" for cls in CLASSES)
    values = {json.dumps(v, sort_keys=True) for v in [thresholds["default"], *thresholds["profiles"].values()]}
    which = f"{text(thresholds['default'])}, the same for every profile" if len(values) == 1 else \
        "; ".join(f"{p} {text(t)}" for p, t in sorted(thresholds["profiles"].items()))
    return [f"Thresholds{' (provisional)' if thresholds.get('provisional') else ''}: {which}. Each score is the mean "
            "of the arbitrated documents (min, number n); `below` counts the documents under the threshold of "
            "their profile." + (" Cost in cents from the spend ledger: billed by Datalab, uncertain (an estimate "
                                "that stands for a cost not billed) and the estimates reserved." if with_cost else ""),
            ""]


def markdown(report):
    """`report.md`: the same numbers as tables (no text of the documents)."""
    run = report["run"]
    status = f"status {run.get('status') or '—'}" + (f" ({run['stop']})" if run.get("stop") else "")
    lines = [f"# Bench run {run['id']}", "",
             f"Valid: {'yes' if report['valid'] else 'no (identity changed)'}. Level {run['level'] or '—'}, "
             f"{run['documents']} documents, repeat {run['repeat']}, OCR {run['ocr']}, executions {run['executions']}, "
             f"generations adopted {run['generations']['adopted']} and abandoned {run['generations']['abandoned']}, "
             f"{status}, exit {run['exit']}.", "",
             f"Package {report['package']['version']} (RECORD {str(report['package']['record_sha256'])[:16]}), "
             f"toolchain {str(report['toolchain']['content_sha256'])[:16]}, weights "
             f"{str(report['weights']['manifest_sha256'])[:16]}, manifest {report['manifest']['sha256'][:16]}, "
             f"bench {report['bench']['version']} (generator {report['bench']['generator']}, matcher "
             f"{report['bench']['matcher']}). Scores are provisional: quality stays unverified.", ""]
    rebuilt = report.get("rebuilt")
    if rebuilt:
        lines += [f"Report written again on {rebuilt['at']} from the saved outputs of the run, with the scoring of "
                  f"package {rebuilt['package']['version']} (RECORD {str(rebuilt['package']['record_sha256'])[:16]}): "
                  "nothing was converted again.", ""]
    summary = report["summary"]
    lines += _thresholds_line(summary.get("thresholds"), any(c.get("cost") for c in summary["format_route"]))
    lines += ["## Format × route", "",
              "| Format | Route | Variant | Complete | Failed | Content | Structure | Form | s/page | Cost (cents) |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for c in summary["format_route"]:
        sc = [_score_cell(c["scores"][k]) for k in CLASSES]
        lines.append(f"| {c['format']} | {c['route']} | {c['variant_sha8']} | {c['complete']} | {c['failed']} | "
                     f"{' | '.join(sc)} | {_fmt(c['s_per_page_median'])} | {_cost_cell(c.get('cost'))} |")
    lines += ["", "## Profile × route", "",
              "| Profile | Route | Variant | Content | Structure | Form |",
              "| --- | --- | --- | --- | --- | --- |"]
    lines += [f"| {p['profile']} | {p['route']} | {p['variant_sha8']} | "
              f"{' | '.join(_score_cell(p['scores'][k]) for k in CLASSES)} |" for p in summary.get("profile_route", [])]
    lines += ["", "## Not arbitrated", ""]
    lines += [f"- {x['format']} × {x['route']}: {', '.join(x['documents'])} (shares {', '.join(x['shares'])})"
              for x in report["summary"]["not_arbitrated"]] or ["- none"]
    lines += ["", "## Not measured", ""]
    lines += [f"- {x['format']} × {x['route']}: {x['reason']}" for x in report["summary"]["not_measured"]] or ["- none"]
    lines += ["", "## Attempts", "",
              "| Document | Route | Variant | k | g | Status | Content | Structure | Form | Encoding | Seconds | Problems |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for d in report["documents"]:
        if d["status"] == "unavailable":
            lines.append(f"| {d['id']} | — | — | — | — | unavailable {d.get('code') or ''} | — | — | — | — | — | — |")
            continue
        for a in d["attempts"]:
            sc = ["—" if a["scores"] is None or a["scores"][k]["ratio"] is None else
                  f"{a['scores'][k]['ratio']:.3f}" for k in CLASSES]
            enc = "—" if a["encoding"] is None else \
                {True: "ok", False: "differs", None: "no oracle"}[a["encoding"]["decode_equals_oracle"]] + \
                f" ({a['encoding']['detected']})"
            problems = ", ".join(f"{k} {v}" for k, v in a["problem_codes"].items()) or "—"
            status = a["status"] + (f" {a['error']}" if a["error"] else "")
            lines.append(f"| {d['id']} | {a['route']} | {a['variant_sha8']} | {a['repeat']} | {a['generation']} | "
                         f"{status} | {' | '.join(sc)} | {enc} | {_fmt(a['timing']['total_s'])} | {problems} |")
    if report.get("datalab"):
        from .run import DATALAB_WARNING
        dl = report["datalab"]
        lines += ["", "## Datalab", "", DATALAB_WARNING, "",
                  f"Modes {', '.join(dl['modes'] or [])}; cap of the run {dl['caps']['max_cost_usd']} USD, "
                  f"cap of the ledger {dl['caps']['spend_cap_usd']} USD. Cents: the estimate is reserved before each "
                  "send and replaced by the billed cost when Datalab reports it.", ""]
        c = report.get("cost") or {}
        if "error" in c:
            lines.append(f"Ledger: {c['error']}.")
        else:
            lines += ["| Mode | Sends | Pages | Estimated | Billed | Billed unknown | Counted |",
                      "| --- | --- | --- | --- | --- | --- | --- |"]
            lines += [f"| {m} | {r['sends']} | {r['pages']} | {r['estimated_cents']} | {r['billed_cents']} | "
                      f"{r['billed_unknown']} | {r['counted_cents']} |" for m, r in c.get("by_mode", {}).items()]
            lines += ["", f"Counted for the run: {c.get('run_counted_cents')} cents; for the whole ledger: "
                      f"{c.get('ledger_counted_cents')} cents."]
    if report.get("server"):
        srv = report["server"]
        rec = srv.get("record") or {}
        lines += ["", "## Inference server", "",
                  f"Endpoint {srv.get('name')} ({rec.get('kind')}, provider {rec.get('provider') or 'none'}, "
                  f"GPU {rec.get('gpu') or '—'}), modes {', '.join(srv.get('modes') or [])}; identity "
                  f"{srv.get('identity_sha256', '')[:12]}. Cents: the reservation before the wake and the measured "
                  "upper bound of each attempt, not billed amounts.", "",
                  "| Document | Mode | Readiness s | 2xx | Other | Retried | Uncertain | Clean | s/page | Reserve | Bound |",
                  "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for d in report["documents"]:
            for a in d["attempts"]:
                sv = a.get("server")
                if not sv:
                    continue
                lines.append(f"| {d['id']} | {sv['mode']} | {_fmt(sv['readiness_s'])} | {sv['requests_2xx']} | "
                             f"{sv['requests_other']} | {sv['retried']} | {sv['uncertain']} | {sv['clean']} | "
                             f"{_fmt(a['timing'].get('s_per_page'))} | {sv['reserve_cents'] or '—'} | "
                             f"{sv['bound_cents'] or '—'} |")
    if report.get("notes"):
        lines += ["", "## Notes", ""]
        lines += [f"- {DJVUTXT_NOTE}" for n in report["notes"] if n == "djvutxt_plain_text"]
    if report["warnings"]:
        lines += ["", "Warnings: " + ", ".join(report["warnings"])]
    return "\n".join(lines) + "\n"


def write(run_dir, report):
    write_atomic(Path(run_dir) / "report.json",
                 (json.dumps(report, ensure_ascii=False, indent=1, sort_keys=True) + "\n").encode("utf-8"))
    write_atomic(Path(run_dir) / "report.md", markdown(report).encode("utf-8"))
