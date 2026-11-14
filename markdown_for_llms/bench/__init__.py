"""`mdf bench`: prepares the public reference set (MFL026).

It builds the reference toolchain image, downloads and verifies the documents and their twins,
derives the inputs, and generates and self-checks the checks. It converts nothing and measures
no conversion path yet. Used only by the `mdf bench` command, never by the API or the worker.
The modules import lazily and need only the standard library, `requests` and `markdown-it-py`;
the external tools run only inside the reference toolchain image.
"""

BENCH_VERSION = "1.0.0"
# Versions that enter the identity of a generated set of checks (cache key and report).
GENERATOR_VERSION = "4"
NORMALIZER_VERSION = "2"
MATCHER_VERSION = "2"
