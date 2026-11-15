"""`mdf bench`: prepares the public reference set (MFL026).

It builds the reference toolchain image, downloads and verifies the documents and their twins,
derives the inputs, and generates and self-checks the checks. It converts nothing and measures
no conversion path yet. Used only by the `mdf bench` command, never by the API or the worker.
The modules import lazily and need only the standard library, `requests` and `markdown-it-py`;
the external tools run only inside the toolchain image, through the runner of the conversion
engine (`markdown_for_llms.engine.tools` and `.toolchain`, moved there by MFL032).
"""
import sys as _sys

from ..engine import toolchain as _toolchain, tools as _tools

# The runner and the toolchain identity moved to the engine; their former names in the bench
# stay the same module objects, so `markdown_for_llms.bench.tools` and `.toolchain` (and a
# patch of either) reach the engine's runner.
_sys.modules[__name__ + ".toolchain"] = _toolchain
_sys.modules[__name__ + ".tools"] = _tools
toolchain, tools = _toolchain, _tools

BENCH_VERSION = "1.0.0"
# Versions that enter the identity of a generated set of checks (cache key and report).
GENERATOR_VERSION = "4"
NORMALIZER_VERSION = "2"
MATCHER_VERSION = "2"
