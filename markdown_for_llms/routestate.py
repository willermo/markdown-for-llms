"""The state of the routes of this instance (MFL051): what can convert here, what cannot and why, what to do.

A pure module, standard library only: the worker computes the state (at its start, every READY_SECONDS and on
request) from what it alone can check, writes it in its heartbeat, and the API reads it from there and never
computes it again. Nothing here opens a file, a socket or a provider.

Each route has one state, in the order in which they prevail:

    broken           a check fails: something has to be repaired (the launcher does not answer, the toolchain image
                     or the weights differ, the keys file is not safe, a GPU window left open...)
    not_configured   something is missing: `missing` says what, where to write it and of which kind (a key, a
                     setting, an entry of servers.json, a command to run); never a value
    disabled         everything is there and a cap is 0
    available

`code` is the closed code of today that explains the state (`engine/codes.py`, `gpu/codes.py`), null when
available. `transit` is a REQUIREMENT of the route (does a document, or its page images, leave the machine), never
a confirmation: the confirmation is of each job. A key never appears: only its name.

The routes: `text`, `pandoc`, `libreoffice`, `djvutxt`, `marker`, `datalab`, and one `marker-server` entry for each
GPU preset the instance enables, with its provider.
"""

SCHEMA = "mdf-routes/1"
BROKEN, NOT_CONFIGURED, DISABLED, AVAILABLE = "broken", "not_configured", "disabled", "available"
STATES = (BROKEN, NOT_CONFIGURED, DISABLED, AVAILABLE)
LOCAL = ("text", "pandoc", "libreoffice", "djvutxt", "marker")
DATALAB, MARKER_SERVER = "datalab", "marker-server"
ENV_FILE = ".env"
SERVERS_FILE = "state/config/servers.json"
# The closed codes that mean "something is missing", each with what is missing: (name, where, kind). A name of
# None is filled by the caller (the keys of an endpoint). Every other code of a route that cannot run is `broken`,
# but those of DISABLED.
MISSING = {
    "toolchain_missing": ("the toolchain image", "make prepare", "command"),
    "weights_missing": ("the weights of the Marker route", "make prepare", "command"),
    "datalab_key_missing": ("DATALAB_API_KEY", ENV_FILE, "key"),
    "server_key_missing": (None, ENV_FILE, "key"),
    "server_not_configured": ("the endpoint of the GPU preset", SERVERS_FILE, "servers.json"),
    "gpu_preset_not_enabled": ("MDF_GPU_PRESETS", ENV_FILE, "setting"),
}
# The closed codes that mean "a cap is 0 or used up": the settings that hold the caps.
DISABLED_BY = {
    "datalab_cap": ("MDF_DATALAB_CAP_USD",),
    "server_cap": ("MDF_GPU_SESSION_CAP_USD", "MDF_GPU_CAP_USD_MODAL", "MDF_GPU_CAP_USD_RUNPOD"),
}
NO_TRANSIT = {"required": False, "what": None}


def state_of(check):
    """(state, code) of one check {"available", "code"}, as the worker makes it today."""
    if not isinstance(check, dict):
        return BROKEN, "isolation_unavailable"
    if check.get("available") is True:
        return AVAILABLE, None
    code = check.get("code") if isinstance(check.get("code"), str) else "route_unavailable"
    if code in MISSING:
        return NOT_CONFIGURED, code
    if code in DISABLED_BY:
        return DISABLED, code
    return BROKEN, code


def missing_of(code, names=None):
    """The `missing` list of a code: what, where, of which kind. `names`: the names of the keys of an endpoint
    that are not in the keys file, for `server_key_missing`."""
    if code not in MISSING:
        return []
    name, where, kind = MISSING[code]
    if name is None:
        return [{"name": n, "where": where, "kind": kind} for n in (names or ["the keys of the endpoint (key_env)"])]
    return [{"name": name, "where": where, "kind": kind}]


def route(ident, converter, check, *, provider=None, gpu_preset=None, transit=None, spend="none", caps=None, preset=None,
          ledger=None, live=None, detail=None, key_names=None):
    """One entry of the answer. `detail`: what to do about a broken route (`make restart` after the keys file was
    replaced), a fixed text of the product, never a value."""
    state, code = state_of(check)
    return {"id": ident, "converter": converter, "provider": provider, "gpu_preset": gpu_preset, "state": state,
            "code": code, "detail": detail if state == BROKEN else None,
            "missing": missing_of(code, key_names) if state == NOT_CONFIGURED else [],
            "transit": dict(transit or NO_TRANSIT), "spend": spend, "caps": caps, "preset": preset, "ledger": ledger,
            "live": live}


def compute(converters, datalab_cap_usd, gpu=None, providers=None, details=None, key_names=None, now=None):
    """The routes of the instance, in the order of the matrix, from what the worker checked:

    converters     {converter: {"available", "code"}} (`engine.availability`, Datalab with its cap)
    datalab_cap_usd the cap of one Datalab conversion, a string in USD
    gpu            the readiness of the GPU sessions (`gpu.lifecycle.Manager.readiness`): `presets`, `ledgers`
    providers      {GPU preset: its provider}
    details        {route id: what to do}, for the broken ones
    key_names      {GPU preset: the names of its keys that the keys file does not set}
    now            the time of the computation, seconds since the epoch
    """
    converters, details, key_names = converters or {}, details or {}, key_names or {}
    routes = [route(name, name, converters.get(name)) for name in LOCAL]
    routes.append(route(DATALAB, DATALAB, converters.get(DATALAB), provider="datalab",
                        transit={"required": True, "what": "document"}, spend="capped",
                        caps={"conversion_usd": str(datalab_cap_usd)}, detail=details.get(DATALAB)))
    gpu = gpu if isinstance(gpu, dict) else {}
    ledgers = gpu.get("ledgers") if isinstance(gpu.get("ledgers"), dict) else {}
    for name, item in sorted((gpu.get("presets") or {}).items()):
        provider = (providers or {}).get(name)
        book = ledgers.get(provider) if isinstance(ledgers.get(provider), dict) else None
        ledger = None if book is None else dict(book, read_at=now)
        ident = "%s:%s" % (MARKER_SERVER, name)
        routes.append(route(
            ident, MARKER_SERVER, item, provider=provider, gpu_preset=name,
            transit={"required": True, "what": "page_images"}, spend="capped",
            caps=None if book is None or "cap_cents" not in book else {"ledger_cents": book["cap_cents"]},
            preset={k: item.get(k) for k in ("minimum_reserve_cents", "a_cents", "w_s")}, ledger=ledger,
            live={"open_session": item.get("open_session") is True, "read_at": now},
            detail=details.get(ident), key_names=key_names.get(name)))
    return {"schema": SCHEMA, "computed_at": now, "routes": routes}


def blocked(code, datalab_cap_usd, gpu_presets=(), providers=None, detail=None, now=None):
    """The routes of a worker that takes no job at all (its children would not run its code, or the end of a
    previous conversion is not attested): every route is broken with the code of the worker."""
    check = {"available": False, "code": code}
    gpu = {"presets": {name: dict(check) for name in gpu_presets}, "ledgers": {}}
    computed = compute({name: check for name in LOCAL + (DATALAB,)}, datalab_cap_usd, gpu, providers, now=now)
    for item in computed["routes"]:
        item.update(state=BROKEN, code=code, detail=detail, missing=[])
    return computed


def answer(recorded, alive):
    """The body of `GET /api/v1/routes` from what the worker last recorded (`recorded`: the `routes` object of its
    readiness, or None) and whether its heartbeat is still fresh. Without a heartbeat ever: no routes. With an old
    one: the last routes, marked stale."""
    valid = (isinstance(recorded, dict) and recorded.get("schema") == SCHEMA and isinstance(recorded.get("routes"), list))
    return {"schema": SCHEMA, "worker": "alive" if alive else "absent",
            "computed_at": recorded.get("computed_at") if valid else None,
            "stale": not (alive and valid), "routes": recorded["routes"] if valid else []}


def summary(body):
    """One line per route for a terminal (`mdf routes`, `make status`): the state, the code, what is missing."""
    lines = ["worker %s%s" % (body.get("worker"), " (stale)" if body.get("stale") else "")]
    for item in body.get("routes") or []:
        text = "%-34s %-15s" % (item.get("id"), item.get("state"))
        if item.get("code"):
            text += " " + item["code"]
        for gap in item.get("missing") or []:
            text += "; missing %s (%s: %s)" % (gap.get("name"), gap.get("kind"), gap.get("where"))
        if item.get("detail"):
            text += "; " + item["detail"]
        lines.append(text.rstrip())
    return "\n".join(lines)
