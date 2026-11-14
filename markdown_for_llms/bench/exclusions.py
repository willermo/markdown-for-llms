"""Frozen exclusions of the self-check and feedback checks, by position (never by text).

Codes (B3 of the plan): `twin_only` - the unit corresponds in the independent inventory and
scores under 0.50 on the third-party text of the input; it leaves the scores too.
`self_check_tool_limit` - a type in the declared limits of the rendering; it leaves only the
self-check judgement and stays in the scores. Preconditions are re-run at every self-check;
an exclusion whose precondition no longer holds, or whose position is gone, is `stale` (red).
`run` reads this file and never writes it.
"""
from importlib import resources
import json
from pathlib import Path

from . import GENERATOR_VERSION
from .codes import BenchError, Code

SCHEMA = "mdf-bench-exclusions/1"
CODES = ("twin_only", "self_check_tool_limit")
# check types each rendering may legitimately miss because of the tool (the "limit" column)
LIMIT_TYPES = {
    "jats": {"formula", "inline_math", "bib_entry", "figure", "cell", "heading", "heading_level"},
    "docx": {"formula", "inline_math", "figure", "emphasis"},
    "latex": {"formula", "inline_math", "bib_entry", "figure"},
    "odt": {"figure", "emphasis"},
    "epub": {"figure", "emphasis"},
    "xlsx": {"cell"},
    "wikisource": {"figure", "noise", "emphasis"},
    "pptx": set(),
    "akoma-ntoso": {"order"},
    "xlsx-projection": set(),
}


def load(path=None):
    if path is None:
        raw = resources.files("markdown_for_llms.bench").joinpath("data/exclusions.json").read_bytes()
    else:
        raw = Path(path).read_bytes()
    try:
        data = json.loads(raw)
    except ValueError:
        raise BenchError(Code.MANIFEST_INVALID, "exclusions: invalid JSON") from None
    if not isinstance(data, dict) or data.get("schema") != SCHEMA or set(data) != {"schema", "exclusions", "feedback"}:
        raise BenchError(Code.MANIFEST_INVALID, "exclusions: schema")
    for e in data["exclusions"]:
        if set(e) != {"document", "check", "code", "reason", "origin", "generator_version"} or e["code"] not in CODES:
            raise BenchError(Code.MANIFEST_INVALID, "exclusions: entry")
    for f in data["feedback"]:
        if set(f) != {"document", "id", "class", "type", "position", "origin", "generator_version"}:
            raise BenchError(Code.MANIFEST_INVALID, "exclusions: feedback")
    return data


def for_document(data, doc_id):
    return [e for e in data["exclusions"] if e["document"] == doc_id]


def validate(entries, checkset, results, inventory_matched, rendering_ratio):
    """Statuses of the exclusions of a document: {check id: "ok" | "stale" | "revalidate"}.

    `inventory_matched(check)` tells whether the unit corresponds in the independent inventory;
    `rendering_ratio(check)` is the best window ratio of the unit on the third-party text.
    """
    by_id = {c["id"]: c for c in checkset["checks"]}
    outcome = {r["id"]: r["outcome"] for r in results}
    statuses = {}
    for e in entries:
        check = by_id.get(e["check"])
        if check is None:
            statuses[e["check"]] = "stale"
            continue
        if e["generator_version"] != GENERATOR_VERSION:
            statuses[e["check"]] = "revalidate"
            continue
        if e["code"] == "twin_only":
            ok = inventory_matched(check) and rendering_ratio(check) < 0.50
        else:
            ok = check["type"] in LIMIT_TYPES.get(checkset["twin_kind"], set()) and outcome.get(check["id"]) not in ("pass", "not_applicable")
        statuses[e["check"]] = "ok" if ok else "stale"
    return statuses
