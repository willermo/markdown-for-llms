"""Closed codes of the bench: the only failure reasons and warnings written to reports.

Exceptions carry a code and an optional local detail; reports and terminal summaries show the
code only, details (paths, exception text) go to the local log of a run.
"""
from enum import Enum


class Code(str, Enum):
    # configuration and manifest
    UNKNOWN_SETTING = "unknown_setting"
    INVALID_SETTING = "invalid_setting"
    PORT_OUTSIDE_BLOCK = "port_outside_block"
    ENV_FILE_UNAVAILABLE = "env_file_unavailable"
    MANIFEST_INVALID = "manifest_invalid"
    UNKNOWN_DOCUMENT = "unknown_document"
    # fetch and derivations
    SOURCE_UNREACHABLE = "source_unreachable"
    SOURCE_SIZE_MISMATCH = "source_size_mismatch"
    SOURCE_HASH_MISMATCH = "source_hash_mismatch"
    SOURCE_UNEXPECTED_TYPE = "source_unexpected_type"
    SESSION_FAILED = "session_failed"
    REVISION_MISMATCH = "revision_mismatch"
    REVISION_UNREACHABLE = "revision_unreachable"
    TOOL_MISSING = "tool_missing"
    TOOL_FAILED = "tool_failed"
    TOOL_MEMORY_LIMIT = "tool_memory_limit"
    TOOLCHAIN_MISSING = "toolchain_missing"
    TOOLCHAIN_MISMATCH = "toolchain_mismatch"
    ENGINE_FAILED = "engine_failed"
    DERIVED_DIFFERS = "derived_differs_from_reference"
    PAGES_MISMATCH = "pages_mismatch"
    # twins, readers and checks
    ARCHIVE_UNSAFE = "archive_unsafe"
    XML_UNSAFE = "xml_unsafe"
    TWIN_UNREADABLE = "twin_unreadable"
    PROJECTION_INVALID = "projection_invalid"
    COVERAGE_LOW = "coverage_low"
    SELF_CHECK_BELOW_THRESHOLD = "self_check_below_threshold"
    SELF_CHECK_NOT_RUN = "self_check_not_run"
    EXCLUSION_STALE = "exclusion_stale"
    EXCLUSION_CAP_EXCEEDED = "exclusion_cap_exceeded"
    EXCLUSION_REVALIDATE = "exclusion_revalidate"
    NO_CHECKS = "no_checks"


class BenchError(Exception):
    """A bench failure: `code` is public, `detail` stays in local logs. A `fatal` failure stops the
    whole command instead of one document (for example a tool container that outlived its timeout)."""

    def __init__(self, code: Code, detail: str = "", fatal: bool = False):
        super().__init__(f"{code.value}: {detail}" if detail else code.value)
        self.code = code
        self.detail = detail
        self.fatal = fatal
