"""`convert` and `availability`: the entry points of the engine.

    convert(source, out, request=None, limits=None, work=None, observer=None, name=None,
            model_root=None) -> dict

`request`: `route` (None: the default of the matrix), `ocr` (`auto` | `force`), `lang` (recorded;
no converter of this version accepts a language), `attempt` (an ID for the names of containers
and directories, `ATTEMPT`; random when absent; any other value is `invalid_setting` before any
directory is made). `limits`: `memory_mib` (the cap of each tool and child),
`output_mib` (the output and every /work), `timeout_seconds` (0: none), `max_pages`,
`poll_seconds`, `threads` (of the Marker child), `stall_seconds` (a Marker child that writes no
event and uses no CPU time for that long is stopped: `stalled`). `model_root`: the weights of the
Marker route (`weights.py`; default `weights.default_root()`). `work`: where the private directories of the attempt go (a temporary directory
when absent); it is never `out`. `observer(event)` receives the steps and the ticks of every
tool call and child, and `{"event": "progress", "progress": snapshot}` from the Marker child
(`progress.py`), and may stop the conversion by returning `cancelled` or `interrupted` to a step
or a tick.

The result is the content of `out/engine.json`, also on a failure (see `contract.py`). The
Datalab route (`datalab.py`) needs `request["datalab"]` (`mode`, `transit_confirmed`, `cap_usd`)
and the key: `datalab_key` (a `datalab.Key`, read by the caller) or `DATALAB_API_KEY` in the
environment of the process; the key is never part of the result.
"""
from datetime import datetime, timezone
from pathlib import Path
import os
import re
import secrets
import shutil
import tempfile
import time
import traceback

from . import contract, formats, routes
from .codes import Code, EngineError

DEFAULT_LIMITS = {"memory_mib": 4096, "output_mib": 512, "timeout_seconds": 0, "max_pages": 50,
                  "poll_seconds": 0.25, "threads": 4, "stall_seconds": 900}
NOT_YET = {}
TOOLCHAIN_ROUTES = (routes.PANDOC, routes.DJVUTXT)
ATTEMPT = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}")
PREPARING = ("image-to-pdf", "ddjvu", "provider", "classify", "strip")      # steps before the Marker child


class Context:
    """The state of one attempt, shared by the routes."""

    def __init__(self, attempt, work, stage, limits, observer, model_root=None):
        self.attempt, self.work, self.stage, self.observer = attempt, work, stage, observer
        self.model_root = model_root
        self.threads = limits["threads"]
        self.stall_seconds = limits["stall_seconds"]
        self.limits, self.weights, self.marker_versions, self.child_runs = [], None, None, 0
        self.progress = None
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
        self.cost = self.transit = self.datalab_versions = None

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

    def feed_events(self, events):
        """Events of the Marker child: the progress snapshot goes to the observer."""
        from .progress import Progress
        if self.progress is None:
            self.progress = Progress()
        if any([self.progress.feed(e) for e in events]) and self.observer is not None:
            self.observer({"event": "progress", "progress": self.progress.snapshot()})

    def event(self, step):
        if step in PREPARING:
            self.feed_events([{"event": "prepare"}])
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


def availability(model_root=None, only=None):
    """{converter: {"available": bool, "code": closed code or None}} on this machine. Marker
    needs Bubblewrap, a systemd user scope and the verified weights under `model_root`. `only`:
    the converters to check (the worker asks for Marker alone)."""
    if only is not None:
        out = {}
        if routes.MARKER in only:
            out[routes.MARKER] = _marker_availability(model_root)
        if set(only) - {routes.MARKER}:
            full = availability(model_root)
            out.update({k: full[k] for k in only})
        return out
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
    out[routes.MARKER] = _marker_availability(model_root)
    out[routes.DATALAB] = ({"available": True, "code": None} if os.environ.get("DATALAB_API_KEY")
                           else {"available": False, "code": Code.DATALAB_KEY_MISSING.value})
    return out


def _marker_availability(model_root):
    from . import isolation, weights
    found = {name: _version(name) for name in ("marker-pdf", "indra-ocr")}
    if found != {"marker-pdf": "2.2.0", "indra-ocr": "1.1.2"}:
        return {"available": False, "code": Code.ENGINE_ADAPTER_MISMATCH.value}
    if not isolation.scope_available():
        return {"available": False, "code": Code.ISOLATION_UNAVAILABLE.value}
    try:
        weights.verify(model_root or weights.default_root())
    except EngineError as exc:
        return {"available": False, "code": exc.code.value}
    return {"available": True, "code": None}


def detect(path, name=None):
    return formats.detect(path, name)


def routes_for(fmt):
    return routes.routes_for(fmt)


def _source_record(path, name, info):
    digest, size = contract.sha256_file(path)
    return {"name": Path(name).name, "bytes": size, "sha256": digest, **info}


def _components(converter, ctx):
    comp = {"package": {"markdown-for-llms": _version("markdown-for-llms")}}
    if converter in TOOLCHAIN_ROUTES or any(not s["tool"].startswith("bwrap:") for s in ctx.steps):
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
    if converter == routes.MARKER:
        comp["marker"] = ctx.marker_versions
        comp["weights"] = ctx.weights
    if converter == routes.DATALAB:
        from . import datalab
        comp["datalab"] = {"endpoint": datalab.API, "versions": ctx.datalab_versions}
    return comp


def convert(source, out, request=None, limits=None, work=None, observer=None, name=None, model_root=None,
            datalab_key=None):
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
    if model_root is None:
        from .weights import default_root
        model_root = default_root()
    ctx = Context(attempt, attempt_dir, stage, limits, observer, Path(model_root).absolute())
    requested = {k: request.get(k) for k in ("route", "ocr", "lang")}
    if request.get("datalab") is not None:
        requested["datalab"] = {k: request["datalab"].get(k) for k in ("mode", "transit_confirmed", "cap_usd")}
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
        elif converter == routes.MARKER:
            from . import pages
            result["marker"] = pages.marker_route(ctx, src, fmt, request)
            result["options"]["effective"]["ocr"] = "force" if result["marker"]["force_ocr"] else "auto"
        elif converter == routes.DATALAB:
            from . import datalab
            result["datalab"] = datalab.datalab_route(ctx, src, fmt, request, key=datalab_key)
            result["options"]["effective"]["ocr"] = "auto"
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
    if ctx.cost is not None:
        result["cost"] = ctx.cost
    if ctx.transit is not None:
        result["transit"] = ctx.transit
    result["limits"] = list(ctx.limits)
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
