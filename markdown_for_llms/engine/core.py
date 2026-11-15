"""`convert` and `availability`: the entry points of the engine.

    convert(source, out, request=None, limits=None, work=None, observer=None, name=None) -> dict

`request`: `route` (None: the default of the matrix), `ocr` (`auto` | `force`), `lang` (recorded;
no converter of this version accepts a language), `attempt` (an ID for the names of containers
and directories, `ATTEMPT`; random when absent; any other value is `invalid_setting` before any
directory is made). `limits`: `memory_mib` (the cap of each tool and child),
`output_mib` (the output and every /work), `timeout_seconds` (0: none), `max_pages`,
`poll_seconds`. `work`: where the private directories of the attempt go (a temporary directory
when absent); it is never `out`. `observer(event)` receives the steps and the ticks of every
tool call and child and may stop the conversion by returning `cancelled` or `interrupted`.

The result is the content of `out/engine.json`, also on a failure (see `contract.py`). Marker
and Datalab are routes of the matrix that this version does not run yet (tranche B of MFL032):
`route_unavailable`.
"""
from datetime import datetime, timezone
from pathlib import Path
import re
import secrets
import shutil
import tempfile
import time
import traceback

from . import contract, formats, routes
from .codes import Code, EngineError

DEFAULT_LIMITS = {"memory_mib": 4096, "output_mib": 512, "timeout_seconds": 0, "max_pages": 50,
                  "poll_seconds": 0.25}
NOT_YET = {routes.MARKER: "Marker 2.2.0 arrives in tranche B of MFL032",
           routes.DATALAB: "Datalab arrives in tranche B of MFL032"}
TOOLCHAIN_ROUTES = (routes.PANDOC, routes.DJVUTXT)
ATTEMPT = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}")


class Context:
    """The state of one attempt, shared by the routes."""

    def __init__(self, attempt, work, stage, limits, observer):
        self.attempt, self.work, self.stage, self.observer = attempt, work, stage, observer
        self.tool_root = work / "tools"
        self.memory_mib = limits["memory_mib"]
        self.output_mib = limits["output_mib"]
        self.max_pages = limits["max_pages"]
        self.poll_seconds = limits["poll_seconds"]
        timeout = limits["timeout_seconds"]
        self.deadline = time.monotonic() + timeout if timeout else None
        self.steps, self.problems, self.corrections, self.correction_counts = [], [], [], {}
        self.unlisted_problems = {}         # problems counted by a converter but not in its list
        self.diagnostics, self.package, self.pages = {}, None, None

    def timeout(self):
        """Seconds left for the next call (a day when there is no deadline)."""
        if self.deadline is None:
            return 86400
        left = self.deadline - time.monotonic()
        if left <= 0:
            raise EngineError(Code.TIMEOUT, "deadline of the conversion")
        return max(1, int(left))

    def watch(self):
        from .tools import Watch
        return Watch(self.output_mib, observer=self.observer, attempt=self.attempt, deadline=self.deadline,
                     poll_seconds=self.poll_seconds, steps=self.steps)

    def event(self, step):
        if self.observer is not None:
            stop = self.observer({"event": "step", "step": step})
            if stop:
                raise EngineError(Code.INTERRUPTED if stop == "interrupted" else Code.CANCELLED, step)


def _version(dist):
    try:
        from importlib.metadata import version
        return version(dist)
    except Exception:
        return None


def availability():
    """{converter: {"available": bool, "code": closed code or None}} on this machine."""
    out = {routes.TEXT: {"available": True, "code": None}}
    try:
        from . import toolchain
        toolchain.image()
        ok = {"available": True, "code": None}
    except EngineError as exc:
        ok = {"available": False, "code": exc.code.value}
    for converter in TOOLCHAIN_ROUTES:
        out[converter] = dict(ok)
    from . import isolation
    if ok["available"] and not isolation.available():        # the packages of pandoc are audited in the child
        out[routes.PANDOC] = {"available": False, "code": Code.ISOLATION_UNAVAILABLE.value}
    for converter in NOT_YET:
        out[converter] = {"available": False, "code": Code.ROUTE_UNAVAILABLE.value}
    return out


def detect(path, name=None):
    return formats.detect(path, name)


def routes_for(fmt):
    return routes.routes_for(fmt)


def _source_record(path, name, info):
    digest, size = contract.sha256_file(path)
    return {"name": Path(name).name, "bytes": size, "sha256": digest, **info}


def _components(converter, ctx):
    comp = {"package": {"markdown-for-llms": _version("markdown-for-llms")}}
    if converter in TOOLCHAIN_ROUTES:
        from . import toolchain
        r = toolchain.report()
        used = sorted({s["tool"] for s in ctx.steps if not s["tool"].startswith("bwrap:")})
        comp["toolchain"] = {"content_sha256": r["content_sha256"], "image_id": r["image_id"],
                             "versions": {t: r["versions"].get(t) for t in used if t in r["versions"]}}
        from .tools import VERSION_OF
        for t in used:
            if t in VERSION_OF:
                comp["toolchain"]["versions"][t] = r["versions"].get(VERSION_OF[t])
    if converter == routes.TEXT:
        comp["charset-normalizer"] = _version("charset-normalizer")
    return comp


def convert(source, out, request=None, limits=None, work=None, observer=None, name=None):
    request = dict(request or {})
    limits = {**DEFAULT_LIMITS, **(limits or {})}
    attempt = request.get("attempt")
    if attempt is None:                 # absent: random; any other value is checked, "" and 0 included
        attempt = secrets.token_hex(8)
    elif not isinstance(attempt, str) or not ATTEMPT.fullmatch(attempt):
        raise EngineError(Code.INVALID_SETTING, "attempt: letters, digits, '_', '.' and '-', at most 64")
    out = contract.prepare_out(out)
    started, at = time.monotonic(), datetime.now(timezone.utc).isoformat(timespec="seconds")
    own_root = work is None
    root = Path(tempfile.mkdtemp(prefix="mdf-engine-")) if own_root else Path(work).absolute()
    if root == out or root.is_relative_to(out):
        raise EngineError(Code.OUTPUT_INVALID, "the work directory is inside the output")
    attempt_dir = root / f"attempt-{attempt}"
    attempt_dir.mkdir(parents=True, mode=0o700)
    stage = attempt_dir / "stage"
    (stage / "diagnostics").mkdir(parents=True, mode=0o700)
    ctx = Context(attempt, attempt_dir, stage, limits, observer)
    requested = {k: request.get(k) for k in ("route", "ocr", "lang")}
    result = {"schema": contract.SCHEMA, "attempt": attempt, "started_at": at, "source": None,
              "route": None, "options": {"requested": requested, "effective": None}, "components": None,
              "corrections": None, "problems": [], "limits": [], "transit": {"left_machine": False, "service": None},
              "cost": None}
    converter = None
    try:
        src = Path(source)
        try:
            info = formats.detect(src, name)
        except EngineError:
            result["source"] = {"name": Path(name or src.name).name}
            raise
        result["source"] = _source_record(src, name or src.name, info)
        fmt = info["format"]
        converter, chosen_by, reason = routes.resolve(fmt, request.get("route"))
        result["route"] = {"converter": converter, "family": info["family"], "chosen_by": chosen_by,
                           "reason": reason, "steps": ctx.steps}
        routes.check_options(converter, request)
        result["options"]["effective"] = {"ocr": None, "lang": request.get("lang"), "lang_passed_to": []}
        if converter in NOT_YET:
            raise EngineError(Code.ROUTE_UNAVAILABLE, NOT_YET[converter])
        if converter == routes.TEXT:
            ctx.event("text")
            from . import text as text_route
            result["text"] = text_route.convert(src, stage)
        elif converter == routes.PANDOC:
            from . import native
            result["pandoc"] = native.pandoc_route(ctx, src, fmt)
        elif converter == routes.DJVUTXT:
            from . import native
            result["djvutxt"] = native.djvutxt_route(ctx, src)
        else:
            raise EngineError(Code.ROUTE_INVALID, converter)
        _finish(result, ctx, converter, started)
        ctx.event("publish")
        return contract.publish(stage, out, result, ctx.output_mib)
    except EngineError as exc:
        return _failed(out, result, ctx, converter, started, exc)
    except KeyboardInterrupt:
        _failed(out, result, ctx, converter, started, EngineError(Code.INTERRUPTED, "interrupted"))
        raise
    except Exception as exc:        # a defect of the engine: a closed code, the trace in the local diagnostics
        ctx.diagnostics["engine-traceback.txt"] = traceback.format_exc().encode("utf-8", "replace")
        return _failed(out, result, ctx, converter, started, EngineError(Code.ENGINE_FAILED, type(exc).__name__))
    finally:
        shutil.rmtree(root if own_root else attempt_dir, ignore_errors=True)


def _finish(result, ctx, converter, started):
    items, counts = contract.bounded(ctx.corrections, contract.MAX_CORRECTIONS)
    for kind, value in ctx.correction_counts.items():
        counts[kind] = max(counts.get(kind, 0), value)
    result["corrections"] = {"counts": counts, "items": items}
    # the counts of every problem, then at most MAX_PROBLEMS items, the incomplete ones first
    codes = dict(ctx.unlisted_problems)
    for p in ctx.problems:
        codes[p["code"]] = codes.get(p["code"], 0) + 1
    result["problem_codes"] = dict(sorted(codes.items()))
    result["problem_counts"] = {
        severity: sum(n for code, n in codes.items() if contract.PROBLEMS[code] == severity)
        for severity in (contract.INCOMPLETE, contract.WARNING)}
    result["problems"] = contract.bounded_problems(ctx.problems, contract.MAX_PROBLEMS)
    if ctx.package is not None:
        result["package"] = ctx.package
    if ctx.pages is not None:
        result["pages"] = ctx.pages
    result["components"] = _components(converter, ctx) if converter else None
    result["timings"] = {"total_seconds": round(time.monotonic() - started, 3)}


def _failed(out, result, ctx, converter, started, exc):
    try:
        _finish(result, ctx, converter, started)
    except Exception:
        result["timings"] = {"total_seconds": round(time.monotonic() - started, 3)}
    diagnostics = dict(ctx.diagnostics)
    log = getattr(exc, "log", None)
    if log:
        diagnostics["child.log"] = log
    if exc.detail:
        diagnostics["error.txt"] = (exc.code.value + ": " + exc.detail + "\n").encode("utf-8", "replace")
    return contract.publish_failure(out, result, exc, diagnostics)
