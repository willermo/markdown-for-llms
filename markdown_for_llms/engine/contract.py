"""The output contract of the engine: what `out/` holds after a conversion, and its checks.

The engine works in a private directory (`stage/` of the attempt) and only on success moves
into `out/`:

    document.md        the complete Markdown document, UTF-8 without NUL
    assets/            the images it references (`asset-NNNN.<ext>`)
    diagnostics/       the original extraction before the corrections, bounded tool logs
    engine.json        this result (`status: complete`)

On a failure `out/` holds `engine.json` (`status: failed`, the closed code, the timings) and
`diagnostics/` with bounded logs, never a `document.md`. Every file is a regular file with a
safe relative name (`files.relative`), the total stays under the output cap, and the inventory
lists path, bytes and SHA-256 of every file but `engine.json` itself: the same fields that the
product checks in a stored manifest (`inventory`, `inventory_complete`). The cap counts
`engine.json` too; under `assets/` only `asset-NNNN.<ext>` names are accepted.

Problems are not errors: a complete conversion can carry problems, each with a severity;
`problem_codes` and `problem_counts` count all of them, `problems` lists at most `MAX_PROBLEMS`,
the incomplete ones first.
`incomplete` means content of the source is missing from the output (a referenced image that
was not extracted); `warning` means a difference that does not lose content of the document
(an image of a slide layout, a remote image left as a link).
"""
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat

from ..files import FileSafetyError, relative
from .codes import Code, EngineError

SCHEMA = "mdf-engine-result/1"
DOCUMENT = "document.md"
RESULT = "engine.json"
INCOMPLETE, WARNING = "incomplete", "warning"
PROBLEMS = {
    # content of the source missing from the output
    "media_lost": INCOMPLETE,             # a media file used by the content and not extracted
    "media_missing": INCOMPLETE,          # an image whose source is not part of the output
    "object_not_converted": INCOMPLETE,   # a chart, OLE object, SmartArt or inline drawing
    "page_text_lost": INCOMPLETE,
    # differences that lose no content of the document
    "media_external": WARNING,            # a remote image, left as a link and never fetched
    "media_furniture": WARNING,           # an image of a layout, master, header or footer (D11)
    "media_unused": WARNING,              # a media file the document does not reference
    "page_blank": WARNING,
    "ocr_forced_document": WARNING,
    "pictures_skipped_rotated": WARNING,
    "datalab_delete_failed": WARNING,     # the result stays on Datalab until its one-day lifecycle
    "datalab_cost_unknown": WARNING,      # no billed cost in the answer: the estimate is the bound
}
MAX_PROBLEMS = 2000
MAX_CORRECTIONS = 2000
LOG_TAIL = 64 * 1024
MIB = 1048576
ASSET_NAME = re.compile(r"assets/asset-[0-9]{4,}\.[a-z0-9]{1,8}")


def problem(code, path=None, page=None, detail=None):
    if code not in PROBLEMS:
        raise ValueError("not a problem code: " + code)
    item = {"code": code, "severity": PROBLEMS[code]}
    if path is not None:
        item["path"] = str(path)[:300]
    if page is not None:
        item["page"] = page
    if detail is not None:
        item["detail"] = str(detail)[:300]
    return item


def bounded(items, limit):
    """At most `limit` items and the counts per code (or kind) of all of them."""
    counts = {}
    for item in items:
        key = item.get("code") or item.get("kind")
        counts[key] = counts.get(key, 0) + 1
    return list(items[:limit]), counts


def bounded_problems(problems, limit):
    """At most `limit` problems, the incomplete ones first (each group in its order)."""
    ordered = [p for p in problems if p["severity"] == INCOMPLETE] + [p for p in problems if p["severity"] != INCOMPLETE]
    return ordered[:limit]


def sha256_file(path):
    h = hashlib.sha256()
    size = 0
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
            size += len(chunk)
    return h.hexdigest(), size


def check_document(data):
    """The Markdown document: strict UTF-8, no NUL; empty (blank) is `output_empty`."""
    try:
        text = data.decode("utf-8", "strict")
    except UnicodeDecodeError:
        raise EngineError(Code.OUTPUT_INVALID, "document.md is not UTF-8") from None
    if "\x00" in text:
        raise EngineError(Code.OUTPUT_INVALID, "NUL in document.md")
    if not text.strip():
        raise EngineError(Code.OUTPUT_EMPTY, "document.md is empty")
    return text


def scan(stage, output_mib):
    """[(relative path, absolute path)] of the files of `stage`, checked: regular files only,
    safe names, `document.md` present and valid, the total under `output_mib`."""
    stage = Path(stage)
    found, total = [], 0
    for parent, dirs, files in os.walk(stage, followlinks=False):
        for name in dirs:
            if (Path(parent) / name).is_symlink():
                raise EngineError(Code.OUTPUT_INVALID, "symbolic link in the output")
        for name in files:
            path = Path(parent) / name
            rel = path.relative_to(stage).as_posix()
            info = os.lstat(path)
            if not stat.S_ISREG(info.st_mode):
                raise EngineError(Code.OUTPUT_INVALID, "not a regular file in the output")
            try:
                relative(rel)
            except FileSafetyError:
                raise EngineError(Code.OUTPUT_INVALID, "unsafe file name in the output") from None
            if rel != DOCUMENT and not rel.startswith("diagnostics/") and not ASSET_NAME.fullmatch(rel):
                raise EngineError(Code.OUTPUT_INVALID, "unexpected file in the output")
            total += info.st_size
            found.append((rel, path))
    if total > output_mib * MIB:
        raise EngineError(Code.OUTPUT_LIMIT, f"output {total} bytes")
    if not any(rel == DOCUMENT for rel, _ in found):
        raise EngineError(Code.OUTPUT_EMPTY, "no document.md")
    check_document((stage / DOCUMENT).read_bytes())
    return sorted(found)


def prepare_out(out):
    """`out` must be a new or empty directory; returns its absolute path."""
    out = Path(out).absolute()
    if out.exists() or out.is_symlink():
        if out.is_symlink() or not out.is_dir():
            raise EngineError(Code.OUTPUT_INVALID, "the output is not a directory")
        if any(out.iterdir()):
            raise EngineError(Code.OUTPUT_INVALID, "the output directory is not empty")
    else:
        out.mkdir(parents=True, mode=0o700)
    return out


def encode_result(result):
    return (json.dumps(result, ensure_ascii=False, indent=1, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def write_result(out, result, data=None):
    data = encode_result(result) if data is None else data
    tmp = Path(out) / (".partial-" + RESULT)
    with open(tmp, "xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, Path(out) / RESULT)


def publish(stage, out, result, output_mib):
    """Move the checked files of `stage` into `out`, add the inventory and write `engine.json`.
    Nothing but `engine.json` (and diagnostics) reaches `out` when a check fails, `engine.json`
    included in the output cap."""
    files = scan(stage, output_mib)
    out = Path(out)
    partial = out / ".partial"
    partial.mkdir(mode=0o700)
    inventory = []
    try:
        for rel, path in files:
            target = partial / rel
            target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            shutil.copyfile(path, target, follow_symlinks=False)
            digest, size = sha256_file(target)
            inventory.append({"path": rel, "bytes": size, "sha256": digest})
        result = dict(result, status="complete", inventory=inventory, inventory_complete=True)
        data = encode_result(result)
        total = sum(i["bytes"] for i in inventory) + len(data)
        if total > output_mib * MIB:
            raise EngineError(Code.OUTPUT_LIMIT, f"output {total} bytes with {RESULT}")
        for entry in sorted(partial.iterdir()):
            os.rename(entry, out / entry.name)
        partial.rmdir()
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True)
        raise
    write_result(out, result, data)
    return result


def publish_failure(out, result, error, diagnostics=None):
    """`engine.json` with the closed code and, when given, bounded diagnostics (name -> bytes)."""
    out = Path(out)
    for name in ("document.md", "assets", ".partial"):
        target = out / name
        if target.is_dir() and not target.is_symlink():
            shutil.rmtree(target, ignore_errors=True)
        elif target.exists() or target.is_symlink():
            target.unlink()
    written = []
    for name, data in (diagnostics or {}).items():
        relative("diagnostics/" + name)
        target = out / "diagnostics" / name
        target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        target.write_bytes(data[-LOG_TAIL:])
        written.append("diagnostics/" + name)
    result = dict(result, status="failed", error={"code": error.code.value, "fatal": bool(error.fatal)},
                  diagnostics=sorted(written))
    result.pop("inventory", None)
    result["inventory_complete"] = False
    write_result(out, result)
    return result
