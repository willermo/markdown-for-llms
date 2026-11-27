"""Inference servers of the `marker-server` route (MFL033): endpoints, keys, terms and bounds.

An endpoint is a named entry of a JSON file without secrets, `servers.json`
(`${XDG_CONFIG_HOME:-~/.config}/markdown-for-llms/servers.json`, or the file given with
`--servers`), whose `default` names the endpoint used when a job names none (`--server NAME`):

    {"default": "cpu", "servers": {"NAME": {"kind": ..., ...}, ...}}

Fields of an endpoint (a closed set; anything else is `server_not_configured`):

- `kind`: `cpu` (the local CPU server of `mdf prepare --cpu-server`, on a Unix socket), `local`
  (a server on the loopback, only the literal `http://127.0.0.1:P` or `http://[::1]:P`, else
  `server_not_loopback`), `user` (the user's own server), `runpod`, `modal`, `gpu` (MFL051 B: the
  local GPU of the `gpu` profile of compose, on the Unix socket of its front, `socket`; no url, no
  key, the page images never leave the machine);
- `provider`: `runpod`, `modal` or absent. An endpoint with a provider costs, also used as
  `user`: it needs a spending cap, the ledger and a live session (`server_not_armed`);
- `url`: https, or http to a literal loopback IP only; never userinfo, query or fragment;
- `socket` (kind `gpu` only, and required there): the absolute path of the Unix socket of the front;
- `ca_file` (kind `user` only, MFL051 B): the absolute path of a PEM file of the certificate
  authorities that sign the certificate of the server, in place of those of the system; with it the
  URL is https, and the relay verifies the certificate and its name against it on its two clients;
- `server`: `vllm` or `llamacpp` (what Surya may ask: `stop_token_ids` and `json_schema` with
  vLLM; neither with llama.cpp, run with `--special`); `model`, `revision`, `pins` (declared
  versions, compared with what the server says at readiness), `parallel` (in-flight requests);
- `gpu`, `rate_usd_per_s` (required with a provider), `rate_source`, `rate_date`;
- seconds: `ready_within`, `read_timeout`, `page_seconds`, `cold_start_s`, `w1_s`,
  `tail_idle_s`, `tail_fault_s`, defaults by kind and provider below;
- `key_env`: the names of the variables of its keys (defaults: `RUNPOD_API_KEY`;
  `MODAL_PROXY_TOKEN_ID` and `MODAL_PROXY_TOKEN_SECRET`; none for `local` and `user` unless named),
  never an `MDF_` name: the env file also holds the settings, which refuse unknown `MDF_` names;
  each name ends with a suffix of `envfile.KEY_SUFFIXES` (`_KEY`, `_SECRET`, `_TOKEN`, `_TOKEN_ID`,
  `_BEARER`; MFL046 c), so that `mdf bench secret-scan` searches it.

An endpoint of RunPod or Modal may name a GPU preset (MFL023 A3, `markdown_for_llms/gpu/presets.py`):
`{"kind": ..., "preset": NAME, "resource": ..., "url": ..., "key_env": [...]}` and nothing else. The
preset fixes every other value (server, model, revision, pins, parallel, gpu, rate with its source and
date, the seconds), so a value written by hand next to `preset` is `server_not_configured` (D3 of
MFL023), as are a preset of another provider, a preset that cannot be used (superseded, a global
volume) and two endpoints of one file with the same preset. `resource` is the name of the endpoint or
app at the provider (`mdf-surya`). A preset of a GPU of the user (MFL051 B) is served by the kind its
entry of the table names (`gpu` for the local GPU, with `socket` in place of `url`; `user` for the
remote server, with its `url`, its `key_env` and `ca_file`), without `resource`. The engine depends on nothing of the GPU package: `load` and
`endpoint` receive the table of the presets (`gpu.presets.endpoint_table()`) as a plain dict, and
without it an endpoint with a preset is `server_not_configured`. The identity of the endpoint is
computed on the expanded values.

Keys are read from `--env-file` (a private regular file, `envfile.read`) and, for `mdf convert`
only, from the environment; they are wrapped in `envfile.Secret` and go only into the headers of
the relay: `Authorization: Bearer` (RunPod, `user`, `local`), `Modal-Key` and `Modal-Secret`
(Modal), none for `cpu` and `gpu`. The identity of an endpoint (`identity_sha256`) is the SHA-256 of the
canonical JSON of its URL, kind, provider, server, model, revision, parallel, request extensions
and pins: a name given to another host changes it.

Terms and bounds (plan r2 §11 as changed by r3 and r4, in `Decimal`): for an endpoint with a
provider the term of an attempt is `D = min(D_doc, D_budget, T - now - 60 s)`, with
`D_doc = ready_within + pages x page_seconds + 300 s`, `D_budget = budget / rate - tail_fault_s`
truncated to the second, and the floor `D_min = max(forecast, ready_within)`, `forecast =
cold_start_s + pages x 9.6 s + 30 s`: `server_cap` when `D_budget < D_min`, `server_not_armed`
when there is no session term `T` or `T - now - 60 s < D_min`, both before the server is woken.
The reservation is `(D + tail_fault_s) x rate`, never over the budget. The bound of a closed
attempt is `(t_last - t_wake + tail_idle_s) x rate` for a clean close, `(t_close - t_wake +
tail_fault_s) x rate` otherwise, `(max(t_close - t_wake, w1_s) + tail_fault_s) x rate` without
readiness; each stays within the reservation because the relay closes by `t_wake + D - 5 s`.
Standard library only.
"""
from decimal import Decimal, InvalidOperation, ROUND_CEILING, ROUND_FLOOR
import hashlib
import ipaddress
import json
import os
from pathlib import Path
import re
from urllib.parse import urlsplit

from . import envfile
from .codes import Code, EngineError

KINDS = ("cpu", "local", "user", "runpod", "modal", "gpu")
PROVIDERS = ("runpod", "modal")
SERVERS = ("vllm", "llamacpp")
MODEL = "datalab-to/surya-ocr-2.5"
NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}")
ENV_NAME = re.compile(r"[A-Z_][A-Z0-9_]{0,63}")
FIELDS = {"kind", "provider", "url", "server", "model", "revision", "pins", "parallel", "gpu", "rate_usd_per_s",
          "rate_source", "rate_date", "ready_within", "read_timeout", "page_seconds", "cold_start_s", "w1_s",
          "tail_idle_s", "tail_fault_s", "key_env", "socket", "ca_file"}
# The fields of an endpoint that names a GPU preset (MFL023 A3): the preset fixes all the others.
PRESET_FIELDS = {"kind", "preset", "resource", "url", "key_env", "socket", "ca_file"}
# The paths of `socket` and `ca_file`: absolute, plain, short enough for a Unix socket (108 bytes with its NUL)
SOCKET_MAX, PATH_MAX = 107, 1024
SECONDS = ("ready_within", "read_timeout", "page_seconds", "cold_start_s", "w1_s", "tail_idle_s", "tail_fault_s")
# Defaults (plan r2 §4, §11; r4 §11.4 and C.3): RunPod reads within 5.5 min of its balancer, Modal
# within the 150 s of a web request; the tails are req_max + tick + provider idle + stop margin.
DEFAULTS = {
    "cpu": {"server": "llamacpp", "parallel": 2, "ready_within": 120, "read_timeout": 600, "page_seconds": 900},
    "local": {"server": "vllm", "parallel": 32, "ready_within": 300, "read_timeout": 300, "page_seconds": 30},
    "user": {"server": "vllm", "parallel": 32, "ready_within": 300, "read_timeout": 300, "page_seconds": 30},
    "runpod": {"server": "vllm", "parallel": 32, "ready_within": 900, "read_timeout": 300, "page_seconds": 30},
    "modal": {"server": "vllm", "parallel": 32, "ready_within": 900, "read_timeout": 140, "page_seconds": 30},
    "gpu": {"server": "vllm", "parallel": 32, "ready_within": 900, "read_timeout": 300, "page_seconds": 30},
}
PROVIDER_DEFAULTS = {
    "runpod": {"cold_start_s": 480, "w1_s": 900, "tail_idle_s": 90, "tail_fault_s": 385},
    "modal": {"cold_start_s": 480, "w1_s": 900, "tail_idle_s": 90, "tail_fault_s": 225},
}
# MFL023 B7 (MFL046 (b), point 6 of the plan r1): Modal has `ready_within` 900 (the floor `w1_s`) and `cold_start_s` 480,
# the values of its GPU preset; with 1200 and 1100 `mdf convert` refused a reserve under about 0.53 USD.
KEY_ENV = {"runpod": ["RUNPOD_API_KEY"], "modal": ["MODAL_PROXY_TOKEN_ID", "MODAL_PROXY_TOKEN_SECRET"]}
PAGE_FORECAST_S, FORECAST_GAP_S, DOC_OVERHEAD_S = Decimal("9.6"), Decimal(30), Decimal(300)
CLOCK_MARGIN_S, CLOSE_MARGIN_S = Decimal(60), Decimal(5)
# MFL023 B2 (SOL2-B1 of the plan r2): the largest drift of the wall clock on the monotonic clock that a window or an
# attempt accepts as observed. Beyond it the laptop was suspended or its clock stepped (CLOCK_MONOTONIC does not count
# a suspend on Linux): the relay sends no request more, and a session cannot certify the close of its window
# (`bench.serverspend.closing`).
DRIFT_LIMIT_S = 5
CENT = Decimal("0.01")
MAX_CONFIG_BYTES = 1 << 20


def default_path(environ=None):
    environ = os.environ if environ is None else environ
    base = environ.get("XDG_CONFIG_HOME") or os.path.join(os.path.expanduser("~"), ".config")
    return Path(base) / "markdown-for-llms" / "servers.json"


def _bad(detail, code=Code.SERVER_NOT_CONFIGURED):
    return EngineError(code, detail)


def loopback_host(host):
    """True for a literal loopback IP address (never a name)."""
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def canonical_url(url):
    """scheme://host[:port] plus the path without its final slash; refused with userinfo, a
    query or a fragment."""
    if not isinstance(url, str) or len(url) > 2048 or any(c.isspace() for c in url):
        raise _bad("url")
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise _bad("url: http or https with a host")
    if parts.username is not None or parts.password is not None or parts.query or parts.fragment or "@" in parts.netloc:
        raise _bad("url: no userinfo, query or fragment")
    try:
        port = parts.port
    except ValueError:
        raise _bad("url: port") from None
    host = parts.hostname.lower()
    shown = "[%s]" % host if ":" in host else host
    netloc = shown + (":%d" % port if port is not None else "")
    return "%s://%s%s" % (parts.scheme, netloc, parts.path.rstrip("/"))


class Endpoint:
    """A validated endpoint; `fields` holds every value, defaults included."""

    def __init__(self, name, fields):
        self.name = name
        self.fields = fields
        for key, value in fields.items():
            setattr(self, key, value)

    @property
    def rate(self):
        return Decimal(self.rate_usd_per_s) if self.rate_usd_per_s is not None else None

    @property
    def costs(self):
        return self.provider is not None

    def leaves_machine(self):
        """The page images leave the machine: any endpoint but `cpu`, `gpu` and a literal loopback IP."""
        if self.kind in ("cpu", "gpu"):
            return False
        return not loopback_host(urlsplit(self.url).hostname or "")

    def host(self):
        if self.kind in ("cpu", "gpu"):
            return "local CPU server" if self.kind == "cpu" else "local GPU"
        return urlsplit(self.url).hostname

    def requests(self):
        """The extensions of the requests Surya may send to this server."""
        vllm = self.server == "vllm"
        return {"allow_stop_token_ids": vllm, "allow_guided_json": vllm}

    def identity(self):
        url = {"cpu": "unix:llama", "gpu": "unix:gpu"}.get(self.kind) or canonical_url(self.url)
        return {"url": url, "kind": self.kind,
                "provider": self.provider, "server": self.server, "model": self.model, "revision": self.revision,
                "parallel": self.parallel, "requests": self.requests(), "pins": self.pins}

    def identity_sha256(self):
        data = json.dumps(self.identity(), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        return hashlib.sha256(data.encode("ascii")).hexdigest()

    def record(self):
        """What `engine.json` and the bench record of the endpoint: no key, no key name; the GPU preset and the
        resource only for an endpoint that names a preset."""
        out = {"name": self.name, "kind": self.kind, "provider": self.provider, "host": self.host(),
               "server": self.server, "model": self.model, "revision": self.revision, "parallel": self.parallel,
               "pins": self.pins, "gpu": self.gpu, "rate_usd_per_s": self.rate_usd_per_s,
               "rate_source": self.rate_source, "rate_date": self.rate_date,
               "identity_sha256": self.identity_sha256(),
               "seconds": {k: getattr(self, k) for k in SECONDS if getattr(self, k) is not None}}
        if self.preset is not None:
            out.update(preset=self.preset, resource=self.resource)
        if self.ca_file is not None:
            out["ca_file"] = True                   # pinned authorities: whether, never the path
        return out


def _path(value, what, limit):
    """An absolute, plain path (no `..`, no NUL or line break), at most `limit` bytes."""
    if (not isinstance(value, str) or not value.startswith("/") or len(value.encode("utf-8", "replace")) > limit
            or any(c in value for c in "\0\r\n") or ".." in value.split("/")):
        raise _bad(what + ": an absolute path of at most %d bytes, without .." % limit)
    return value


def _seconds(value, what):
    if isinstance(value, bool) or not isinstance(value, int) or not 0 < value <= 86400:
        raise _bad(what + ": whole seconds from 1 to 86400")
    return value


def _expand(name, raw, presets):
    """The fields of the endpoint `name` that names a GPU preset, the values of the preset included (MFL023 A3)."""
    extra = set(raw) - PRESET_FIELDS
    if extra:
        raise _bad("%s: fields fixed by its preset %s" % (name, sorted(extra)))
    if presets is None:
        raise _bad(name + ": a GPU preset without the table of the presets")
    preset = raw["preset"]
    entry = presets.get(preset) if isinstance(preset, str) else None
    if entry is None:
        raise _bad(name + ": unknown GPU preset")
    if entry["refusal"] is not None:
        raise _bad("%s: the GPU preset %s cannot be used (%s)" % (name, preset, entry["refusal"]))
    kind = entry.get("kind", entry["provider"])
    if raw.get("kind") != kind:
        raise _bad("%s: the GPU preset %s is of %s: kind %s" % (name, preset, entry["provider"], kind))
    rented = entry["provider"] in PROVIDERS
    resource = raw.get("resource")
    if rented and (not isinstance(resource, str) or not NAME.fullmatch(resource)):
        raise _bad(name + ": resource, the name of the endpoint or app at the provider")
    if not rented and resource is not None:
        raise _bad(name + ": resource only for a rented GPU")
    where = "socket" if kind == "gpu" else "url"
    if where not in raw:
        raise _bad("%s: %s" % (name, where))
    fields = dict(entry["fields"], kind=raw["kind"], **{where: raw[where]})
    for key in ("ca_file", "socket", "url"):
        if key != where and key in raw:
            fields[key] = raw[key]
    expected = entry.get("key_env", KEY_ENV.get(entry["provider"], []))
    if "key_env" in raw:
        # MFL023 B10 (C-DB-NB5, a reversible choice of the supervision): next to a preset, only the names the provider of
        # the preset reads, in their order (the worker, its children, the follower and the relay read those)
        given = [raw["key_env"]] if isinstance(raw["key_env"], str) else raw["key_env"]
        if given != expected:
            raise _bad("%s: key_env next to a GPU preset is absent or %s" % (name, ", ".join(expected) or "empty"))
        fields["key_env"] = raw["key_env"]
    elif not rented:
        fields["key_env"] = list(expected)          # the remote server: its key, named by the preset
    return fields, preset, resource


def endpoint(name, raw, presets=None):
    """A validated `Endpoint` from the JSON object `raw` of the entry `name`; `presets`, the table of the GPU presets
    (`gpu.presets.endpoint_table()`), for an entry that names one."""
    if not isinstance(name, str) or not NAME.fullmatch(name):
        raise _bad("endpoint name")
    if not isinstance(raw, dict):
        raise _bad(name + ": an object")
    preset = resource = None
    if "preset" in raw:
        raw, preset, resource = _expand(name, raw, presets)
    unknown = set(raw) - FIELDS
    if unknown:
        raise _bad("%s: unknown fields %s" % (name, sorted(unknown)))
    kind = raw.get("kind")
    if kind not in KINDS:
        raise _bad(name + ": kind")
    provider = raw.get("provider", kind if kind in PROVIDERS else None)
    if provider is not None and provider not in PROVIDERS:
        raise _bad(name + ": provider")
    if kind in PROVIDERS and provider != kind:
        raise _bad(name + ": the provider of a %s endpoint is %s" % (kind, kind))
    if kind in ("cpu", "local", "gpu") and provider is not None:
        raise _bad(name + ": a %s endpoint has no provider" % kind)
    fields = dict(DEFAULTS[kind], **(PROVIDER_DEFAULTS.get(provider) or {}))
    fields.update({k: v for k, v in raw.items() if k != "kind"})
    fields.update(kind=kind, provider=provider)
    for key in ("url", "revision", "gpu", "rate_usd_per_s", "rate_source", "rate_date", "cold_start_s", "w1_s",
                "tail_idle_s", "tail_fault_s", "key_env", "socket", "ca_file"):
        fields.setdefault(key, None)
    fields.setdefault("model", MODEL)
    fields.setdefault("pins", {})
    if kind == "gpu":
        fields["socket"] = _path(fields["socket"], name + ": socket", SOCKET_MAX)
    elif fields["socket"] is not None:
        raise _bad(name + ": socket only for a gpu endpoint")
    if fields["ca_file"] is not None:
        if kind != "user":
            raise _bad(name + ": ca_file only for a user endpoint")
        fields["ca_file"] = _path(fields["ca_file"], name + ": ca_file", PATH_MAX)
    if kind in ("cpu", "gpu"):
        if fields["url"] is not None:
            raise _bad(name + ": a %s endpoint has no url (Unix socket)" % kind)
    else:
        url = canonical_url(fields["url"])
        scheme, host = urlsplit(url).scheme, urlsplit(url).hostname
        if kind == "local" and not (scheme == "http" and loopback_host(host) and urlsplit(url).port):
            raise _bad(name + ": local is http://127.0.0.1:PORT or http://[::1]:PORT", Code.SERVER_NOT_LOOPBACK)
        if scheme == "http" and not loopback_host(host):
            raise _bad(name + ": http only to a literal loopback address; https otherwise")
        if fields["ca_file"] is not None and scheme != "https":
            raise _bad(name + ": ca_file pins the authorities of an https server")
    if fields["server"] not in SERVERS:
        raise _bad(name + ": server vllm or llamacpp")
    if not isinstance(fields["model"], str) or not fields["model"] or len(fields["model"]) > 200:
        raise _bad(name + ": model")
    for key in ("revision", "gpu", "rate_source", "rate_date"):
        if fields[key] is not None and (not isinstance(fields[key], str) or len(fields[key]) > 300):
            raise _bad("%s: %s" % (name, key))
    pins = fields["pins"]
    if not isinstance(pins, dict) or len(pins) > 20 or any(
            not isinstance(k, str) or not isinstance(v, str) or len(k) > 64 or len(v) > 200 for k, v in pins.items()):
        raise _bad(name + ": pins")
    parallel = fields["parallel"]
    if isinstance(parallel, bool) or not isinstance(parallel, int) or not 1 <= parallel <= 64:
        raise _bad(name + ": parallel from 1 to 64")
    for key in SECONDS:
        if fields[key] is not None:
            _seconds(fields[key], "%s: %s" % (name, key))
    if provider is not None:
        try:
            rate = Decimal(str(fields["rate_usd_per_s"]))
        except (InvalidOperation, ValueError):
            rate = None
        if fields["rate_usd_per_s"] is None or rate is None or not rate.is_finite() or not 0 < rate < 1:
            raise _bad(name + ": rate_usd_per_s, a decimal in USD per second")
        fields["rate_usd_per_s"] = str(fields["rate_usd_per_s"])
        if fields["ready_within"] < fields["w1_s"]:
            raise _bad(name + ": ready_within is at least w1_s")
    elif fields["rate_usd_per_s"] is not None:
        raise _bad(name + ": a rate without a provider")
    names = fields["key_env"]
    if names is None:
        names = KEY_ENV.get(provider if kind in PROVIDERS else "", [])
    elif isinstance(names, str):
        names = [names]
    if not isinstance(names, list) or len(names) > 2 or any(not isinstance(n, str) or not ENV_NAME.fullmatch(n)
                                                             or n.startswith("MDF_") or not n.endswith(envfile.KEY_SUFFIXES)
                                                             for n in names):
        raise _bad(name + ": key_env: names of variables ending in " + ", ".join(envfile.KEY_SUFFIXES)
                   + ", not MDF_ settings")
    if kind in ("cpu", "gpu") and names:
        raise _bad(name + ": a %s endpoint has no key" % kind)
    if kind == "modal" and len(names) != 2 or kind != "modal" and len(names) > 1:
        raise _bad(name + ": key_env: two names for modal, at most one otherwise")
    fields["key_env"] = names
    fields.update(preset=preset, resource=resource)
    return Endpoint(name, fields)


def load(path=None, name=None, environ=None, presets=None, host_loopback=True):
    """The endpoint `name` (default: the file's `default`) of the file `path` (default:
    `default_path()`); `server_not_configured` when the file, the entry or a field is missing or
    invalid, or when two entries name the same GPU preset. `presets`: the table of the GPU presets
    (`gpu.presets.endpoint_table()`), for an entry that names one.

    `host_loopback` (MFL051): whether the caller can reach a server on the loopback of the host. A caller in a
    container cannot: it passes False, and an endpoint of kind `local` is refused (`server_not_configured`, with
    the reason) in place of a conversion that waits for a server it will never reach. The rule is of the caller,
    as a policy of the product: this module holds no path and no fact of a container."""
    path = Path(path) if path is not None else default_path(environ)
    try:
        fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NONBLOCK)
    except OSError as exc:
        raise _bad("servers file: %s" % type(exc).__name__) from None
    try:
        data = os.read(fd, MAX_CONFIG_BYTES + 1)
    finally:
        os.close(fd)
    if len(data) > MAX_CONFIG_BYTES:
        raise _bad("servers file too large")
    try:
        doc = json.loads(data)
    except ValueError:
        raise _bad("servers file: not JSON") from None
    if not isinstance(doc, dict) or set(doc) - {"default", "servers"} or not isinstance(doc.get("servers"), dict):
        raise _bad("servers file: {\"default\": NAME, \"servers\": {NAME: {...}}}")
    named = [e.get("preset") for e in doc["servers"].values() if isinstance(e, dict) and "preset" in e]
    if len(named) != len({json.dumps(p, sort_keys=True) for p in named}):
        raise _bad("servers file: two endpoints with the same GPU preset")
    name = name if name is not None else doc.get("default")
    if not isinstance(name, str) or name not in doc["servers"]:
        raise _bad("no endpoint %r in the servers file" % (name,))
    ep = endpoint(name, doc["servers"][name], presets)
    if ep.kind == "local" and not host_loopback:
        raise _bad("endpoint %r is of kind `local`, a server on the loopback of the host: a container of the "
                   "product cannot reach it" % (name,))
    return ep


def read_keys(ep, env_file=None, environ=None):
    """{variable name: envfile.Secret} of the keys of `ep`: from `env_file`, then from `environ`
    when given (only `mdf convert` passes the environment); `server_key_missing`,
    `server_key_unsafe`."""
    found = {}
    if not ep.key_env:
        return found
    if env_file is not None:
        try:
            found.update(envfile.read(env_file, ep.key_env))
        except envfile.Unsafe as exc:
            raise EngineError(Code.SERVER_KEY_UNSAFE, str(exc)) from None
    for key in ep.key_env:
        if key not in found and environ is not None and environ.get(key):
            found[key] = environ[key]
    missing = [k for k in ep.key_env if not found.get(k)]
    if missing:
        raise EngineError(Code.SERVER_KEY_MISSING, ", ".join(missing) + " in the env file"
                          + (" or the environment" if environ is not None else ""))
    if any(c in v for v in found.values() for c in "\r\n\0"):
        raise EngineError(Code.SERVER_KEY_UNSAFE, "a key with a line break")
    return {k: envfile.Secret(v) for k, v in found.items()}


def headers(ep, keys):
    """The headers the relay adds to every request towards `ep` (values revealed only here)."""
    if ep.kind == "modal":
        a, b = ep.key_env
        return {"Modal-Key": keys[a].reveal(), "Modal-Secret": keys[b].reveal()}
    if ep.key_env:
        return {"Authorization": "Bearer " + keys[ep.key_env[0]].reveal()}
    return {}


def secret_values(ep, keys):
    """The values of the credentials `headers` sends to `ep`, for the relay to recognize them when an
    answer repeats one (never recorded)."""
    names = ep.key_env if ep.kind == "modal" else ep.key_env[:1]
    return [keys[n].reveal() for n in names if keys.get(n) is not None and keys[n].reveal()]


# --- terms, reservations and bounds ---------------------------------------------------------------

def _cents_up(usd):
    return (usd * 100).quantize(CENT, rounding=ROUND_CEILING)


def forecast_s(ep, pages):
    """Seconds an attempt is expected to take at most (a cold start, 9.6 s per page, a gap)."""
    return Decimal(ep.cold_start_s or 0) + Decimal(pages) * PAGE_FORECAST_S + FORECAST_GAP_S


def d_doc(ep, pages):
    return Decimal(ep.ready_within) + Decimal(pages) * ep.page_seconds + DOC_OVERHEAD_S


def term(ep, pages, budget_cents=None, until=None, now=None):
    """The term of an attempt before the server is woken: {"D", "d_min", "d_doc", "d_budget",
    "d_session", "reserve_cents", "budget_cents"}; for an endpoint without a provider, D = D_doc
    and nothing is reserved. `budget_cents`: what the caps leave to the attempt; `until`: the
    term `T` of the session (epoch seconds); `now`: epoch seconds."""
    doc = d_doc(ep, pages)
    if not ep.costs:
        return {"D": int(doc), "d_doc": int(doc), "d_min": None, "d_budget": None, "d_session": None,
                "reserve_cents": None, "budget_cents": None}
    rate, tail = ep.rate, Decimal(ep.tail_fault_s)
    d_min = max(forecast_s(ep, pages), Decimal(ep.ready_within))
    budget = Decimal(str(budget_cents if budget_cents is not None else 0)).quantize(CENT, rounding=ROUND_FLOOR)
    d_budget = (budget / 100 / rate - tail).to_integral_value(rounding=ROUND_FLOOR)
    if d_budget < d_min:
        raise EngineError(Code.SERVER_CAP, "budget %s cents: %s s against a floor of %s s" % (budget, d_budget, d_min))
    if until is None or now is None:
        raise EngineError(Code.SERVER_NOT_ARMED, "no live session for an endpoint with a provider")
    d_session = (Decimal(str(until)) - Decimal(str(now)) - CLOCK_MARGIN_S).to_integral_value(rounding=ROUND_FLOOR)
    if d_session < d_min:
        raise EngineError(Code.SERVER_NOT_ARMED, "the session ends in %s s: under the floor of %s s"
                          % (d_session + CLOCK_MARGIN_S, d_min))
    d = min(doc, d_budget, d_session)
    reserve = _cents_up((d + tail) * rate)
    if reserve > budget:                         # cannot happen with the truncations above
        raise EngineError(Code.SERVER_CAP, "reservation over the budget")
    return {"D": int(d), "d_doc": int(doc), "d_min": str(d_min), "d_budget": int(d_budget),
            "d_session": int(d_session), "reserve_cents": str(reserve), "budget_cents": str(budget)}


def bound_cents(ep, ready, clean, t_wake, t_last, t_close):
    """The measured upper bound of an attempt in cents (None without a provider or a wake)."""
    if not ep.costs or t_wake is None:
        return None
    rate = ep.rate
    wake, close = Decimal(str(t_wake)), Decimal(str(t_close))
    if not ready:
        seconds = max(close - wake, Decimal(ep.w1_s)) + ep.tail_fault_s
    elif clean:
        seconds = Decimal(str(t_last if t_last is not None else t_close)) - wake + ep.tail_idle_s
    else:
        seconds = close - wake + ep.tail_fault_s
    return str(_cents_up(max(seconds, Decimal(0)) * rate))
