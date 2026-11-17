"""The `text` route: a plain-text file decoded in the engine process (D5 of MFL032).

A byte order mark chooses the encoding: UTF-32 (`FF FE 00 00`, `00 00 FE FF`), then UTF-16
(`FF FE`, `FE FF`), then UTF-8 (`EF BB BF`), the 4-byte marks first. After a UTF-16 or UTF-32
mark the whole file decodes strictly, and a text with a NUL or another C0 control character
than tab, line feed, vertical tab, form feed and carriage return is `format_mismatch` (a binary
file behind the mark, or a log with ANSI escapes, is refused, not converted). Without a mark,
charset-normalizer chooses the encoding, and a NUL byte, a decoding that charset-normalizer does
not find plausible, or text that does not decode strictly with the chosen encoding is
`format_mismatch`. The text is written to `document.md` verbatim, in UTF-8, with the line
endings normalized to LF: no Markdown escaping, so the words and symbols of the source reach
the output unchanged (a `*` or a `#` of the text keeps its Markdown meaning). An empty or blank
file is `output_empty`; a text whose UTF-8 form is over the output cap is `output_limit`, and
nothing is written. The encoding (its canonical name: `utf_32`, `utf_16`, `utf_8` or the name of
charset-normalizer), the mark, the language and the coherence measures go to `engine.json`.
"""
from .codes import Code, EngineError
from .formats import C0_REFUSED, text_bom

MAX_BYTES = 256 * 1024 * 1024
MIB = 1048576


def _wide(data, found):
    """The text after a UTF-16 or UTF-32 byte order mark."""
    name, codec, n = found
    try:
        text = data[n:].decode(codec, "strict")
    except UnicodeDecodeError:
        raise EngineError(Code.FORMAT_MISMATCH, f"text does not decode as {name}") from None
    if C0_REFUSED.search(text):
        raise EngineError(Code.FORMAT_MISMATCH, f"control character in a {name} text")
    if not text.strip():
        raise EngineError(Code.OUTPUT_EMPTY, "empty text file")
    return text


def decode(data):
    """(text, record) of the bytes of a txt file."""
    found = text_bom(data)
    best = None
    if found is not None and found[0] != "utf_8":
        encoding = found[0]
        text = _wide(data, found)
    else:
        if b"\x00" in data:
            raise EngineError(Code.FORMAT_MISMATCH, "NUL byte in a text file")
        if not data.strip():
            raise EngineError(Code.OUTPUT_EMPTY, "empty text file")
        if found is not None:
            encoding, body = "utf_8", data[found[2]:]
        else:
            from charset_normalizer import from_bytes
            best = from_bytes(data).best()
            if best is None:
                raise EngineError(Code.FORMAT_MISMATCH, "no plausible encoding")
            encoding, body = best.encoding, data
        try:
            text = body.decode(encoding, "strict")
        except (UnicodeDecodeError, LookupError):
            raise EngineError(Code.FORMAT_MISMATCH, f"text does not decode as {encoding}") from None
        if "\x00" in text:
            raise EngineError(Code.FORMAT_MISMATCH, "NUL in the decoded text")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if not text.endswith("\n"):
        text += "\n"
    record = {"encoding": encoding, "bom": found is not None,
              "language": None if best is None else best.language,
              "chaos_percent": None if best is None else round(best.percent_chaos, 3),
              "coherence_percent": None if best is None else round(best.percent_coherence, 3),
              "line_endings": "normalized to LF", "markdown": "verbatim"}
    return text, record


def convert(source, stage, output_mib=None):
    """Write `stage/document.md`; returns the record for `engine.json`. `output_mib` caps the
    UTF-8 form of the text (from UTF-16 a character of the BMP above U+07FF grows by half)."""
    with open(source, "rb") as stream:
        data = stream.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise EngineError(Code.OUTPUT_LIMIT, "text file over the limit")
    text, record = decode(data)
    encoded = text.encode("utf-8")
    if output_mib is not None and len(encoded) > output_mib * MIB:
        raise EngineError(Code.OUTPUT_LIMIT, "the UTF-8 text is over the output cap")
    (stage / "document.md").write_bytes(encoded)
    return record
