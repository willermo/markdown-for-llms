"""`mdf bench crosscheck R…`: checks that every route fails in the same way (plan r2 §10).

For each document, on the attempts of the given reports that share its check set (the same
`checks.sha256`), each path (route and variant) counts once: its first complete repetition. With at
least three paths, a check that every path fails with the same outcome is `suspect`: the defect may
be in the twin or the generator rather than in the routes. The bench never excludes anything by
itself: a suspect is a pointer for the maintainer, who may turn it into an exclusion or a feedback
entry. With `--images`, the page of a suspect whose position names a page (`pg<N>/…`, the twins of
Wikisource) is rendered from the cached input into `OUT/suspects/` (pdftoppm or ddjvu in the
toolchain image, an image input copied); other suspects have no page. Numbers and closed codes
only in `crosscheck.json`.

Variants and families (C-CROSSCHECK of the review of the final diff). The paths are variants: the
OCR modes of Marker, or the modes of Datalab, are paths of their own. Variants of one route share
their library or service and fail together, so they are correlated, not independent evidence
against the twin. Each document and each suspect carries `variants` (its paths) and `families`
(its distinct routes), `correlated` tells a suspect resting on two variants of one route or more,
and `suspects_by_families` counts the suspects by their number of families. The threshold stays
three paths.
"""
import json
from pathlib import Path
import re
import shutil

from .cache import write_atomic
from .codes import BenchError, Code
from .compare import load_report

MIN_PATHS = 3
PAGE = re.compile(r"pg(\d+)/")


def crosscheck(reports, min_paths=MIN_PATHS):
    slots = {}          # document -> checks sha -> (route, variant) -> attempt
    for report in reports:
        for doc in report["documents"]:
            sha = (doc.get("checks") or {}).get("sha256")
            if not sha:
                continue
            for a in doc.get("attempts", []):
                if a["status"] != "complete" or a.get("scores") is None or not isinstance(a.get("failed"), list):
                    continue
                slot = slots.setdefault(doc["id"], {}).setdefault(sha, {})
                path = (a["route"], a["variant_sha8"])
                if path not in slot or a["repeat"] < slot[path]["repeat"]:
                    slot[path] = a
    documents, total, by_families = [], 0, {}
    for doc_id in sorted(slots):
        sha, paths = max(slots[doc_id].items(), key=lambda kv: (len(kv[1]), kv[0]))
        families = len({route for route, _ in paths})
        entry = {"id": doc_id, "checks_sha256": sha, "paths": [list(p) for p in sorted(paths)], "families": families,
                 "suspects": []}
        if len(paths) < min_paths:
            entry["status"] = "too_few_paths"
            documents.append(entry)
            continue
        failed = [{f["id"]: f for f in a["failed"]} for a in paths.values()]
        for check_id in sorted(set.intersection(*(set(f) for f in failed))):
            outcomes = {f[check_id]["outcome"] for f in failed}
            if len(outcomes) == 1:
                first = failed[0][check_id]
                page = PAGE.match(check_id)
                entry["suspects"].append({"id": check_id, "class": first["class"], "type": first["type"],
                                          "outcome": first["outcome"], "page": int(page.group(1)) if page else None,
                                          "variants": len(paths), "families": families,
                                          "correlated": families < len(paths)})
        entry["status"] = "checked"
        total += len(entry["suspects"])
        if entry["suspects"]:
            by_families[str(families)] = by_families.get(str(families), 0) + len(entry["suspects"])
        documents.append(entry)
    return {"min_paths": min_paths, "documents": documents, "suspects": total,
            "suspects_by_families": dict(sorted(by_families.items(), key=lambda kv: int(kv[0])))}


def render_pages(result, manifest, cache, out_dir):
    """Render the page of each suspect that names one; returns the number of images written."""
    from ..engine import tools
    from .fetch import resolve_document, selected_pages
    target = Path(out_dir) / "suspects"
    target.mkdir(parents=True, exist_ok=True, mode=0o700)
    written = 0
    for entry in result["documents"]:
        pages = sorted({s["page"] for s in entry["suspects"] if s["page"] is not None})
        if not pages:
            continue
        doc = manifest.document(entry["id"])
        twin = doc["twin"]
        if not twin or twin["kind"] != "wikisource":
            continue
        order = [p["page"] for p in selected_pages(manifest.sources[twin["source"]], twin)]
        files = resolve_document(doc, manifest, cache, None, offline=True)
        for page in pages:
            if page not in order:
                continue
            index = order.index(page) + 1          # the page of the input
            name = f"{entry['id']}-pg{page}"
            if doc["format"] in ("png", "jpeg", "tiff"):
                shutil.copyfile(files.input, target / f"{name}.{doc['format']}")
            elif doc["format"] == "pdf":
                tools.run("pdftoppm", ["-f", str(index), "-l", str(index), "-r", "100", "-png", "-singlefile",
                                       tools.In(files.input), tools.Out(target / f"{name}.png", arg="page",
                                                                        produced="page.png")],
                          cache.tool_work_dir, timeout=600)
            elif doc["format"] == "djvu":
                tools.run("ddjvu", ["-format=tiff", f"-page={index}", tools.In(files.input),
                                    tools.Out(target / f"{name}.tif")], cache.tool_work_dir, timeout=600)
            else:
                continue
            written += 1
    return written


def run(report_paths, out_dir=None, images=False, manifest=None, cache=None):
    result = crosscheck([load_report(p) for p in report_paths])
    if images:
        if out_dir is None:
            raise BenchError(Code.INVALID_SETTING, "--images needs --out")
        result["images"] = render_pages(result, manifest, cache, out_dir)
    if out_dir is not None:
        write_atomic(Path(out_dir) / "crosscheck.json", json.dumps(result, indent=1, sort_keys=True).encode("utf-8"))
    return result
