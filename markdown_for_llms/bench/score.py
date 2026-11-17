"""Scores: per class, the share of passed checks among the evaluated ones; never a single grade.

Denominators are explicit: generated, not determinable, excluded by the self-check (twin_only
only: `self_check_tool_limit` stays in the scores), excluded by other origins, feedback,
evaluated, passed. A document without evaluable checks is `no_checks`.
"""
from importlib import resources
import json

CLASSES = ("content", "structure", "form")


def thresholds():
    return json.loads(resources.files("markdown_for_llms.bench").joinpath("data/thresholds.json").read_bytes())


def scores(checkset, results, exclusions=(), profile=None):
    excluded = {e["check"]: e for e in exclusions}
    limits = thresholds()
    table = {}
    for cls in CLASSES:
        generated = [r for r in results if r["class"] == cls]
        twin_only = [r for r in generated if excluded.get(r["id"], {}).get("code") == "twin_only"]
        evaluated = [r for r in generated if r["id"] not in {x["id"] for x in twin_only} and r["outcome"] != "not_applicable"]
        passed = sum(1 for r in evaluated if r["outcome"] == "pass")
        threshold = limits["profiles"].get(profile, limits["default"])[cls]
        ratio = passed / len(evaluated) if evaluated else None
        table[cls] = {"generated": len(generated), "not_determinable": checkset["not_determinable"][cls],
                      "excluded_self_check": len(twin_only), "excluded_repo": 0,
                      "feedback": sum(1 for r in generated if r["id"].startswith("fb:")),
                      "evaluated": len(evaluated), "passed": passed, "ratio": ratio, "threshold": threshold,
                      "provisional": limits["provisional"], "meets": None if ratio is None else ratio >= threshold}
    return table
