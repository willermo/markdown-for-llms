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
Marker route (`weights.py`; default: the models volume, `containers.MODEL_ROOT`). `work`: where the private directories of the attempt go (a temporary directory
when absent); it is never `out`. `observer(event)` receives the steps and the ticks of every
tool call and child, and `{"event": "progress", "progress": snapshot}` from the Marker child
(`progress.py`), and may stop the conversion by returning `cancelled` or `interrupted` to a step
or a tick.

The result is the content of `out/engine.json`, also on a failure (see `contract.py`); its
`timings` hold `total_seconds` and, on the Marker route, `stages` and `pages`
(`pages.marker_timings`). The
Datalab route (`datalab.py`) needs `request["datalab"]` (`mode`, `transit_confirmed`, `cap_usd`)
and the key: `datalab_key` (a `datalab.Key`, read by the caller from its keys file; never from the
environment of the process); the key is never part of the result.

The `marker-server` route (MFL033, `pages.marker_server_route`) needs `request["server"]`
(`name`, `mode`, `transit_confirmed`, and for an endpoint with a provider `budget_cents` and the
`session` {`arm_id`, `armed_until`}), the endpoint `server_endpoint` (`servers.Endpoint`) and its
keys `server_keys`, both from the caller and never part of the result; `server_transport` is a
transport factory of the relay and `server_cpu` ({"root", "manifest"}) the local CPU server, both
for the tests. The result adds `marker_server` (the counters and
times of the relay), `components.server` and, for a provider, the bound of the cost in `cost`.
"""
from datetime import datetime, timezone
from pathlib import Path
import re
import secrets
import shutil
import tempfile
import time
import traceback

from . import containers, contract, formats, routes
from .codes import Code, EngineError

DEFAULT_LIMITS = {"memory_mib": 4096, "output_mib": 512, "timeout_seconds": 0, "max_pages": 50,
                  "poll_seconds": 0.25, "threads": 4, "stall_seconds": 900}
NOT_YET = {}
TOOLCHAIN_ROUTES = (routes.PANDOC, routes.LIBREOFFICE, routes.DJVUTXT)
AUDITED_ROUTES = (routes.PANDOC, routes.LIBREOFFICE)      # the package audit runs in a light child
CHILD_STEPS = ("child:", "bwrap:")      # a step of a Python child; `bwrap:` in the results written before MFL051
ATTEMPT = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}")
PREPARING = ("image-to-pdf", "ddjvu", "provider", "classify", "strip")      # steps before the Marker child
MAX_CHILD_EVENTS = 20000            # events of one Marker child kept for the timings (pages, processors)


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
        self.server = self.marker_server = None    # the marker-server route: endpoint record, relay record
        self.child_events, self.timings = [], {}     # events of the current Marker child; stages and pages
        self._private = None                          # the socket directory of the attempt, made when a child needs it

    def private(self):
        """The socket directory of the attempt on the run volume (`sockets.Private`), made once and held until
        `close`: the launcher mounts it in every Marker child and in the local CPU server."""
        if self._private is None:
            from . import sockets
            sockets.sweep()
            self._private = sockets.Private(containers.prefix_of(self.attempt))
        return self._private

    def close(self):
        if self._private is not None:
            self._private.close()
            self._private = None

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
        room = MAX_CHILD_EVENTS - len(self.child_events)
        if room > 0:
            self.child_events.extend(events[:room])
        if any([self.progress.feed(e) for e in events]) and self.observer is not None:
            self.observer({"event": "progress", "progress": self.progress.snapshot()})

    def event(self, step, **fields):
        if step in PREPARING:
            self.feed_events([{"event": "prepare"}])
        if self.observer is not None:
            stop = self.observer({"event": "step", "step": step, **fields})
            if stop:
                raise EngineError(Code.INTERRUPTED if stop == "interrupted" else Code.CANCELLED, step)


def _version(dist):
    try:
        from importlib.metadata import version
        return version(dist)
    except Exception:
        return None


def availability(model_root=None, only=None, keys_file=None):
    """{converter: {"available": bool, "code": closed code or None}} on this machine. The toolchain
    routes need the launcher and the verified toolchain image, Marker the launcher and the verified
    weights under `model_root`, Datalab its key in `keys_file`, read now by name (never from the environment:
    no key is in the environment of a container). `only`: the converters to check (the worker asks for Marker
    alone)."""
    if only is not None:
        out = {}
        if routes.MARKER in only:
            out[routes.MARKER] = _marker_availability(model_root)
        if set(only) - {routes.MARKER}:
            full = availability(model_root, keys_file=keys_file)
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
    if ok["available"] and not isolation.available():        # the packages are audited in the child
        for converter in AUDITED_ROUTES:
            out[converter] = {"available": False, "code": Code.ISOLATION_UNAVAILABLE.value}
    out[routes.MARKER] = _marker_availability(model_root)
    from . import datalab
    state = datalab.key_state(keys_file)
    out[routes.DATALAB] = {"available": state["available"], "code": state["code"]}
    return out


def _marker_availability(model_root):
    from . import isolation, weights
    found = {name: _version(name) for name in ("marker-pdf", "indra-ocr")}
    if found != {"marker-pdf": "2.2.0", "indra-ocr": "1.1.2"}:
        return {"available": False, "code": Code.ENGINE_ADAPTER_MISMATCH.value}
    if not isolation.available():
        return {"available": False, "code": Code.ISOLATION_UNAVAILABLE.value}
    try:
        weights.verify(model_root or containers.MODEL_ROOT)
    except EngineError as exc:
        return {"available": False, "code": exc.code.value}
    return {"available": True, "code": None}


def detect(path, name=None):
    return formats.detect(path, name)


def routes_for(fmt):
    return routes.engine_routes_for(fmt)


def _source_record(path, name, info):
    digest, size = contract.sha256_file(path)
    return {"name": Path(name).name, "bytes": size, "sha256": digest, **info}


def _components(converter, ctx):
    comp = {"package": {"markdown-for-llms": _version("markdown-for-llms")}}
    if converter in TOOLCHAIN_ROUTES or any(not s["tool"].startswith(CHILD_STEPS) for s in ctx.steps):
        from . import toolchain
        r = toolchain.report()
        used = sorted({s["tool"] for s in ctx.steps if not s["tool"].startswith(CHILD_STEPS)})
        comp["toolchain"] = {"content_sha256": r["content_sha256"], "image_id": r["image_id"],
                             "versions": {t: r["versions"].get(t) for t in used if t in r["versions"]}}
        from .tools import VERSION_OF
        for t in used:
            if t in VERSION_OF:
                comp["toolchain"]["versions"][t] = r["versions"].get(VERSION_OF[t])
    if converter == routes.TEXT:
        comp["charset-normalizer"] = _version("charset-normalizer")
    if converter in (routes.MARKER, routes.MARKER_SERVER):
        comp["marker"] = ctx.marker_versions
        comp["weights"] = ctx.weights
    if converter == routes.MARKER_SERVER and ctx.server is not None:
        comp["server"] = ctx.server
    if converter == routes.DATALAB:
        from . import datalab
        comp["datalab"] = {"endpoint": datalab.API, "versions": ctx.datalab_versions}
    return comp


def convert(source, out, request=None, limits=None, work=None, observer=None, name=None, model_root=None,
            datalab_key=None, server_endpoint=None, server_keys=None, server_transport=None, server_cpu=None):
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
        model_root = containers.MODEL_ROOT
    ctx = Context(attempt, attempt_dir, stage, limits, observer, Path(model_root).absolute())
    requested = {k: request.get(k) for k in ("route", "ocr", "lang")}
    if request.get("datalab") is not None:
        requested["datalab"] = {k: request["datalab"].get(k) for k in ("mode", "transit_confirmed", "cap_usd")}
    if isinstance(request.get("server"), dict):
        requested["server"] = {k: request["server"].get(k) for k in ("name", "mode", "transit_confirmed", "budget_cents")}
        requested["server"]["arm_id"] = (request["server"].get("session") or {}).get("arm_id")
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
            result["text"] = text_route.convert(src, stage, ctx.output_mib)
        elif converter == routes.PANDOC:
            from . import native
            result["pandoc"] = native.pandoc_route(ctx, src, fmt)
        elif converter == routes.LIBREOFFICE:
            from . import native
            result["libreoffice"] = native.libreoffice_route(ctx, src, fmt)
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
        elif converter == routes.MARKER_SERVER:
            from . import pages
            if server_endpoint is None or server_endpoint.name != request["server"].get("name"):
                raise EngineError(Code.SERVER_NOT_CONFIGURED, "the endpoint of the request")
            result["marker_server_route"] = pages.marker_server_route(ctx, src, fmt, request, server_endpoint,
                                                                      server_keys or {}, server_transport, server_cpu)
            result["options"]["effective"]["ocr"] = "force" if result["marker_server_route"]["force_ocr"] else "auto"
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
        ctx.close()
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
    if ctx.marker_server is not None:
        result["marker_server"] = ctx.marker_server
    result["limits"] = list(ctx.limits)
    result["components"] = _components(converter, ctx) if converter else None
    result["timings"] = {"total_seconds": round(time.monotonic() - started, 3), **ctx.timings}


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
