"""The `text` route: a plain-text file decoded in the engine process (D5 of MFL032).

charset-normalizer chooses the encoding; the text is written to `document.md` verbatim, in
UTF-8, with the line endings normalized to LF: no Markdown escaping, so the words and symbols
of the source reach the output unchanged (a `*` or a `#` of the text keeps its Markdown
meaning). A NUL byte, a decoding that charset-normalizer does not find plausible, or text that
does not decode strictly with the chosen encoding is `format_mismatch`; an empty or blank file
is `output_empty`. The encoding, its language and the coherence measures go to `engine.json`.
"""
from .codes import Code, EngineError

MAX_BYTES = 256 * 1024 * 1024


def decode(data):
    """(text, record) of the bytes of a txt file."""
    if b"\x00" in data:
        raise EngineError(Code.FORMAT_MISMATCH, "NUL byte in a text file")
    if not data.strip():
        raise EngineError(Code.OUTPUT_EMPTY, "empty text file")
    if data.startswith(b"\xef\xbb\xbf"):
        encoding, best = "utf_8", None
        body = data[3:]
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
    record = {"encoding": encoding, "bom": best is None,
              "language": None if best is None else best.language,
              "chaos_percent": None if best is None else round(best.percent_chaos, 3),
              "coherence_percent": None if best is None else round(best.percent_coherence, 3),
              "line_endings": "normalized to LF", "markdown": "verbatim"}
    return text, record


def convert(source, stage):
    """Write `stage/document.md`; returns the record for `engine.json`."""
    with open(source, "rb") as stream:
        data = stream.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise EngineError(Code.OUTPUT_LIMIT, "text file over the limit")
    text, record = decode(data)
    (stage / "document.md").write_text(text, encoding="utf-8")
    return record
