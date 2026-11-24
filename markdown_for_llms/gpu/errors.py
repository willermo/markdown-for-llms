"""The closed errors of the GPU sessions of the product (MFL023 B1; plan r2 §4.6, r3 §4.6).

An error of a provider reaches the product only as closed fields, never as the text of the provider nor as
str() of an exception:

- `providers.closed(exc)`: `class`, and for a `ProviderError` its `op`, its `http` and `detail`, the text of
  this code (`local`). The diary of a session of the worker (`<ledger>/worker-logs/<arm_id>/session.json`)
  keeps these, bounded by `bounded`;
- `public(fields)`: the same without `detail`, for the API, the Web UI, `runs.error`, `runs.gpu` and
  `gpu_sessions`.

`bounded` also closes again what a child process (`opchild`) answers: only the four fields, each of its type
and length. `secret-scan` stays a check more, never the guarantee.
"""
import re

PUBLIC = ("class", "op", "http")
NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_.\-]{0,79}")
DETAIL_LIMIT = 300


def _name(value):
    return value if isinstance(value, str) and NAME.fullmatch(value) else None


def bounded(fields):
    """The closed fields of `fields` (a dict, maybe from a child): `class` and `op` closed names (letters,
    digits, `_`, `.`, `-`; at most 80), `http` an integer or None, `detail` a text of at most 300 characters
    or None; anything else is dropped. A missing or malformed class is `Error`."""
    fields = fields if isinstance(fields, dict) else {}
    http = fields.get("http")
    detail = fields.get("detail")
    return {"class": _name(fields.get("class")) or "Error", "op": _name(fields.get("op")),
            "http": http if type(http) is int else None,
            "detail": detail[:DETAIL_LIMIT] if isinstance(detail, str) else None}


def public(fields):
    """The fields of `bounded` that the API, the Web UI and the database keep: no `detail`."""
    b = bounded(fields)
    return {k: b[k] for k in PUBLIC}
