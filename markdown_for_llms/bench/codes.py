"""Closed codes of the bench: the only failure reasons and warnings written to reports.

Exceptions carry a code and an optional local detail; reports and terminal summaries show the
code only, details (paths, exception text) go to the local log of a run.

The external tools of the bench run through the conversion engine (`markdown_for_llms.engine`),
which raises `EngineError` with its own codes. So that every handler of the bench sees those
failures (K2 of MFL032): `BenchError` is the same class as `EngineError`, and `Code` holds the
codes of the engine, with the same names and values, followed by the codes of the bench alone.
An `Enum` with members cannot be subclassed, hence the functional form; the members are `str`,
so a code of the engine compares and hashes equal to the bench code of the same value.
"""
from enum import Enum

from ..engine.codes import Code as EngineCode, EngineError

BENCH_ONLY = (
    # configuration and manifest
    ("UNKNOWN_SETTING", "unknown_setting"),
    ("PORT_OUTSIDE_BLOCK", "port_outside_block"),
    ("ENV_FILE_UNAVAILABLE", "env_file_unavailable"),
    ("MANIFEST_INVALID", "manifest_invalid"),
    ("UNKNOWN_DOCUMENT", "unknown_document"),
    # fetch and derivations
    ("SOURCE_UNREACHABLE", "source_unreachable"),
    ("SOURCE_SIZE_MISMATCH", "source_size_mismatch"),
    ("SOURCE_HASH_MISMATCH", "source_hash_mismatch"),
    ("SOURCE_UNEXPECTED_TYPE", "source_unexpected_type"),
    ("SESSION_FAILED", "session_failed"),
    ("REVISION_MISMATCH", "revision_mismatch"),
    ("REVISION_UNREACHABLE", "revision_unreachable"),
    ("TOOL_MISSING", "tool_missing"),
    ("DERIVED_DIFFERS", "derived_differs_from_reference"),
    ("PAGES_MISMATCH", "pages_mismatch"),
    # twins, readers and checks
    ("ARCHIVE_UNSAFE", "archive_unsafe"),
    ("XML_UNSAFE", "xml_unsafe"),
    ("TWIN_UNREADABLE", "twin_unreadable"),
    ("PROJECTION_INVALID", "projection_invalid"),
    ("COVERAGE_LOW", "coverage_low"),
    ("SELF_CHECK_BELOW_THRESHOLD", "self_check_below_threshold"),
    ("SELF_CHECK_NOT_RUN", "self_check_not_run"),
    ("EXCLUSION_STALE", "exclusion_stale"),
    ("EXCLUSION_CAP_EXCEEDED", "exclusion_cap_exceeded"),
    ("EXCLUSION_REVALIDATE", "exclusion_revalidate"),
    ("NO_CHECKS", "no_checks"),
)

Code = Enum("Code", [(m.name, m.value) for m in EngineCode] + list(BENCH_ONLY), type=str,
            module=__name__, qualname="Code")

BenchError = EngineError
