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
    ("MANIFEST_SCHEMA_UNSUPPORTED", "manifest_schema_unsupported"),
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
    ("DERIVED_NONDETERMINISTIC", "derived_nondeterministic"),
    ("RECODE_LOSSY", "recode_lossy"),
    ("PAGES_MISMATCH", "pages_mismatch"),
    # twins, readers and checks
    ("ARCHIVE_UNSAFE", "archive_unsafe"),
    ("XML_UNSAFE", "xml_unsafe"),
    ("TWIN_UNREADABLE", "twin_unreadable"),
    ("TWIN_RENDER_INVALID", "twin_render_invalid"),
    ("TWIN_BOILERPLATE_UNKNOWN", "twin_boilerplate_unknown"),
    ("PROJECTION_INVALID", "projection_invalid"),
    ("COVERAGE_LOW", "coverage_low"),
    ("SELF_CHECK_BELOW_THRESHOLD", "self_check_below_threshold"),
    ("SELF_CHECK_NOT_RUN", "self_check_not_run"),
    ("EXCLUSION_STALE", "exclusion_stale"),
    ("EXCLUSION_CAP_EXCEEDED", "exclusion_cap_exceeded"),
    ("EXCLUSION_REVALIDATE", "exclusion_revalidate"),
    ("NO_CHECKS", "no_checks"),
    # measured runs (`mdf bench run`) and their resumption
    ("RUN_EXISTS", "run_exists"),
    ("RUN_BUSY", "run_busy"),
    ("ATTEMPT_ID_COLLISION", "attempt_id_collision"),
    ("STATE_UNWRITABLE", "state_unwritable"),
    ("RESUME_STATE_MISSING", "resume_state_missing"),
    ("RESUME_STATE_INVALID", "resume_state_invalid"),
    ("RESUME_IDENTITY_CHANGED", "resume_identity_changed"),
    ("RESUME_OPTION_NOT_ALLOWED", "resume_option_not_allowed"),
    ("IDENTITY_CHANGED", "identity_changed"),
    ("VENV_BUSY", "venv_busy"),
    ("VENV_IN_USE", "venv_in_use"),
    ("SPEND_LEDGER_INCONSISTENT", "spend_ledger_inconsistent"),
    # Datalab runs of the bench
    ("DATALAB_REFUSED_PERSONAL", "datalab_refused_personal"),
    ("DATALAB_UNRESOLVED", "datalab_unresolved"),
    ("DATALAB_BILLED_OVER_ESTIMATE", "datalab_billed_over_estimate"),
    ("SPEND_LEDGER_UNWRITABLE", "spend_ledger_unwritable"),
    # feedback, the personal set and the secret scan
    ("FEEDBACK_POSITION_MISSING", "feedback_position_missing"),
    ("SECRET_FOUND", "secret_found"),
    ("SECRET_SCAN_INCOMPLETE", "secret_scan_incomplete"),
    ("REPORT_INVALID", "report_invalid"),
    # the report of a finished run written again from its saved outputs (`mdf bench report`)
    ("REPORT_RUN_NOT_FINISHED", "report_run_not_finished"),
    ("REPORT_IDENTITY_CHANGED", "report_identity_changed"),
    ("REPORT_CHECKS_CHANGED", "report_checks_changed"),
    ("REPORT_RESULT_CHANGED", "report_result_changed"),
)

Code = Enum("Code", [(m.name, m.value) for m in EngineCode] + list(BENCH_ONLY), type=str,
            module=__name__, qualname="Code")

BenchError = EngineError
