#!/usr/bin/env python3
"""Guard of the bench and engine tests in the JUnit report of the fast suite (`scripts/full-check.sh`).

The bench tests (`classname` starting with `tests.unit.bench.`) and the engine tests
(`tests.unit.engine.`) never skip: the tests that use the external tools run them in the
toolchain image and fail, saying how to build it, when it is missing
(`bench_test_support.require_toolchain`, `engine_test_support.require_toolchain`). At least one
test of each group must run; the two groups are counted apart.
Usage: bench_junit_guard.py FAST_SUITE_JUNIT.xml; exit 0 green, 1 red, 2 unreadable report.
"""
import sys
import xml.etree.ElementTree as ET

PREFIX = "tests.unit.bench."
GROUPS = (("bench", PREFIX), ("engine", "tests.unit.engine."))


def check(path, prefix=PREFIX, group="bench"):
    """Return (tests of the group run, problems)."""
    root = ET.parse(path).getroot()
    ran, problems = 0, []
    for case in root.iter("testcase"):
        classname = case.get("classname", "")
        if not classname.startswith(prefix):
            continue
        name = classname + "::" + case.get("name", "")
        skip = case.find("skipped")
        if skip is None:
            ran += 1
        else:
            problems.append(f"{group} test skipped ({skip.get('message', '')}): {name}")
    if ran == 0:
        problems.append(f"no {group} test ran")
    return ran, problems


def main(argv):
    if len(argv) != 2:
        print("usage: bench_junit_guard.py FAST_SUITE_JUNIT.xml", file=sys.stderr)
        return 2
    counts, problems = [], []
    try:
        for group, prefix in GROUPS:
            ran, found = check(argv[1], prefix, group)
            counts.append(f"{group} {ran} run")
            problems += found
    except (OSError, ET.ParseError) as exc:
        print(f"bench guard: unreadable report: {type(exc).__name__}")
        return 2
    for line in problems:
        print("problem: " + line)
    print(f"bench guard: {', '.join(counts)}, {len(problems)} problems")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
