"""Closed codes of the conversion engine and its one exception, `EngineError`.

The engine reports a failure as one of these codes in `engine.json`; the detail of an error
(paths, exception text, tool output) stays in the local diagnostics and logs. The bench shares
this exception and these values: `bench/codes.py` builds its own code set from this one plus
the codes of the bench alone, and `BenchError` is `EngineError` (K2 of MFL032), so a failure
raised inside the engine reaches every handler of the bench unchanged. The engine imports
nothing of the bench.
"""
from enum import Enum


class Code(str, Enum):
    # input and format
    INPUT_UNREADABLE = "input_unreadable"
    FORMAT_UNSUPPORTED = "format_unsupported"
    FORMAT_MISMATCH = "format_mismatch"
    PACKAGE_UNSAFE = "package_unsafe"
    # route
    ROUTE_INVALID = "route_invalid"
    ROUTE_UNAVAILABLE = "route_unavailable"
    ROUTE_OPTION_UNSUPPORTED = "route_option_unsupported"
    # tools and isolation
    INVALID_SETTING = "invalid_setting"
    TOOLCHAIN_MISSING = "toolchain_missing"
    TOOLCHAIN_MISMATCH = "toolchain_mismatch"
    WEIGHTS_MISSING = "weights_missing"
    WEIGHTS_MISMATCH = "weights_mismatch"
    ISOLATION_UNAVAILABLE = "isolation_unavailable"
    ENGINE_FAILED = "engine_failed"
    ENGINE_ADAPTER_MISMATCH = "engine_adapter_mismatch"
    TOOL_FAILED = "tool_failed"
    TOOL_MEMORY_LIMIT = "tool_memory_limit"
    TOOL_TIMEOUT = "tool_timeout"
    CONVERTER_FAILED = "converter_failed"
    # pages
    PAGE_LIMIT = "page_limit"
    PAGE_FAILED = "page_failed"
    OCR_LAYER_NOT_REMOVABLE = "ocr_layer_not_removable"
    # output and execution
    OUTPUT_EMPTY = "output_empty"
    OUTPUT_INVALID = "output_invalid"
    OUTPUT_LIMIT = "output_limit"
    MEMORY_LIMIT = "memory_limit"
    STALLED = "stalled"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"
    INTERRUPTED = "interrupted"
    # remote service (Datalab, tranche B)
    TRANSIT_NOT_CONFIRMED = "transit_not_confirmed"
    DATALAB_KEY_MISSING = "datalab_key_missing"
    DATALAB_KEY_UNSAFE = "datalab_key_unsafe"
    DATALAB_CAP = "datalab_cap"
    DATALAB_REJECTED = "datalab_rejected"
    DATALAB_UNAVAILABLE = "datalab_unavailable"
    DATALAB_FAILED = "datalab_failed"


class EngineError(Exception):
    """A failure of the engine (and of the bench): `code` is public, `detail` stays in local logs.
    A `fatal` failure stops a whole command of the bench instead of one document (for example a
    tool container that outlived its timeout)."""

    def __init__(self, code, detail: str = "", fatal: bool = False):
        super().__init__(f"{code.value}: {detail}" if detail else code.value)
        self.code = code
        self.detail = detail
        self.fatal = fatal
