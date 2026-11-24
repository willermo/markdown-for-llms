"""The ledger of the rented inference servers (MFL033): windows, attempts, activities, reconciliations.

The rows share `spend.jsonl` with Datalab and carry `provider` (`runpod` or `modal`); a row without
it is Datalab's (`spend.py`), so the ledger of MFL026 stays readable and unchanged. Totals and caps
are per provider. Amounts are cents, read as decimals; times are epoch seconds.

Rows (plan r2 §11, r3 §11.5, §11.7, r4 §11.10 and App. G; prescription 2 of the review of r4):

- `arm` (written by the session before it arms the provider): `arm_id`, `endpoint`, `url`
  (canonical), `t_arm`, `a_cents` (the cap of the window, `A`), `until` (the term `T` of the
  session), `spend_cap_cents` (the cap of the provider), `rate`, `fault`, optional `tail_s`;
- `disarm` (after the zero): `arm_id`, `t_zero`, `clean` (the view of the session: no fault, exit
  0, attempts of the engine only, zero within `Z`), `orphan` (a disarm of a window whose session
  died: `session.py disarm`), optional `t_stop` and `double_s`; since MFL023 also `uncertain` (with
  `orphan: true`): the reason why a live session could not certify the close of its own window
  (`closing`), informative only;
- `reserve` (at the wake of a server, before its first request): `run_id`, `attempt_id`, `arm_id`,
  `reserve_cents`, `rate`, `tail_idle_s`, `t_reserve`, `identity`, `follows` (the previous attempt
  of the window, closed cleanly, whose short tail this attempt cuts); the canonical URL of the
  endpoint must be the URL of the window (MFL023 B2);
- `complete`: `bound_cents` (the measured upper bound; `billed_cents` stays null), `clean`,
  `t_wake`, `t_last`, `t_close`;
- `activity` (a reservation outside the engine without a window: a disarmed check, a CPU
  activity): `activity_id`, `reserve_cents`; it holds its reservation until a reconciliation;
- `reconcile`: `key` (the windows and activities it covers, the hours), `cents`, `manual`,
  `orphan`.

The amount of an attempt is its bound when complete, else its reservation; with `follows`, the
tail of the previous one ends at `min(t_last + tail_idle_s, t_reserve)` of the next, and the next
counts from its `t_reserve`. A window is:

- **open**, owned by a live session (its lock `<ledger>.arm-<id>.lock` held): `A`, once; its
  attempts draw from it;
- **orphan** (an `arm` whose lock is free and has no `disarm`, or a `disarm` with `orphan: true`):
  the whole residual of the cap of the provider, until a manual reconciliation; `orphan: true` also
  marks the close of a live session that cannot be certified within `A` (`uncertain`, `closing`);
- **clean** (disarmed, `clean`, at least one attempt, every attempt complete and clean): the sum
  of its attempts;
- **unclean** (any other disarmed window, faults and drivers included): `max(sum, (t_end -
  t_arm + double_s) x rate)`, where `t_end` is the later of `t_zero` and the `t_stop` of the
  `disarm` row, plus the `tail_s` of the `arm` row (MFL033 c7.19: on RunPod the billing runs until
  the container stops, then a tail of 30 s; `double_s` counts again the seconds when two workers
  were RUNNING at once). A row without these fields counts `t_zero - t_arm`, as before.

A reconciliation replaces the amounts of the windows and activities it covers with the figure of
the provider: automatic (`manual: false`) only for clean windows, manual for any (uncertain
amounts are lowered only by hand). The last row of a key prevails; a figure over the amounts of
the ledger is `server_billed_over_bound`, a persistent stop of the provider.
"""
from decimal import Decimal, InvalidOperation, ROUND_CEILING
import fcntl
import json
import os
from pathlib import Path
import re
import secrets
import time

from .codes import BenchError, Code
from . import spend
from ..engine.servers import DRIFT_LIMIT_S

PROVIDERS = ("runpod", "modal")
KINDS = ("arm", "disarm", "reserve", "complete", "activity", "reconcile")
CENT = Decimal("0.01")
ARM_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}")
SESSION_ENV = "MDF_SESSION_ARM"


def _bad(detail):
    return BenchError(Code.SPEND_LEDGER_INCONSISTENT, detail)


def _cents(value, what):
    return spend._cents(str(value) if isinstance(value, Decimal) else value, what)


def _up(cents):
    return Decimal(cents).quantize(CENT, rounding=ROUND_CEILING)


def _num(value, what):
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise _bad(what)
    try:
        number = Decimal(str(value))
    except Exception:
        raise _bad(what) from None
    if not number.is_finite():
        raise _bad(what)
    return number


def _seconds(row, field):
    """An optional count of seconds of a row (`tail_s`, `double_s`, MFL033 c7.19): 0 when absent,
    else a finite number not below 0."""
    if row.get(field) is None:
        return Decimal(0)
    value = _num(row[field], field)
    if value < 0:
        raise _bad(field)
    return value


def lock_path(ledger_path, arm_id):
    ledger_path = Path(ledger_path)
    return ledger_path.with_name(ledger_path.name + ".arm-" + arm_id + ".lock")


def alive(ledger_path, arm_id):
    """True when a session holds the lock of the window `arm_id`."""
    try:
        fd = os.open(lock_path(ledger_path, arm_id), os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK)
    except OSError:
        return False
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(fd, fcntl.LOCK_UN)
        return False
    finally:
        os.close(fd)


def server_rows(entries):
    return [e for e in entries if isinstance(e, dict) and "provider" in e]


class Provider:
    """The server rows of one provider, checked and indexed in the order of the ledger."""

    def __init__(self, provider):
        self.provider = provider
        self.windows = {}             # arm_id -> {"arm", "disarm", "attempts": [key], "order"}
        self.attempts = {}            # (run_id, attempt_id) -> {"reserve", "complete"}
        self.activities = {}          # activity_id -> row
        self.reconciles = {}          # canonical key -> last row

    def add(self, e):
        kind = e.get("kind")
        if kind not in KINDS:
            raise _bad("unknown kind of a server row")
        if kind == "arm":
            arm_id = e.get("arm_id")
            if not isinstance(arm_id, str) or not ARM_ID.fullmatch(arm_id) or arm_id in self.windows:
                raise _bad("arm")
            for field in ("a_cents", "spend_cap_cents"):
                _cents(e.get(field), field)
            for field in ("t_arm", "until", "rate"):
                _num(e.get(field), field)
            _seconds(e, "tail_s")
            self.windows[arm_id] = {"arm": e, "disarm": None, "attempts": [], "order": len(self.windows)}
        elif kind == "disarm":
            w = self.windows.get(e.get("arm_id"))
            if w is None or w["disarm"] is not None:
                raise _bad("disarm without its arm, or twice")
            _num(e.get("t_zero"), "t_zero")
            if e.get("t_stop") is not None:
                _num(e["t_stop"], "t_stop")
            _seconds(e, "double_s")
            w["disarm"] = e
        elif kind == "reserve":
            key = (e.get("run_id"), e.get("attempt_id"))
            if not all(isinstance(k, str) for k in key) or key in self.attempts:
                raise _bad("reserve")
            w = self.windows.get(e.get("arm_id"))
            if w is None:
                raise _bad("reserve outside a window")
            _cents(e.get("reserve_cents"), "reserve_cents")
            for field in ("rate", "t_reserve", "tail_idle_s"):
                _num(e.get(field), field)
            self.attempts[key] = {"reserve": e, "complete": None}
            w["attempts"].append(key)
        elif kind == "complete":
            a = self.attempts.get((e.get("run_id"), e.get("attempt_id")))
            if a is None or a["complete"] is not None:
                raise _bad("completion without its reservation, or twice")
            if e.get("bound_cents") is not None:
                _cents(e["bound_cents"], "bound_cents")
            a["complete"] = e
        elif kind == "activity":
            aid = e.get("activity_id")
            if not isinstance(aid, str) or aid in self.activities:
                raise _bad("activity")
            _cents(e.get("reserve_cents"), "reserve_cents")
            self.activities[aid] = e
        else:
            key = e.get("key")
            if not isinstance(key, dict):
                raise _bad("reconcile")
            _cents(e.get("cents"), "cents")
            for arm_id in key.get("arms") or []:
                if arm_id not in self.windows:
                    raise _bad("reconcile of an unknown window")
            for aid in key.get("activities") or []:
                if aid not in self.activities:
                    raise _bad("reconcile of an unknown activity")
            self.reconciles[json.dumps(key, sort_keys=True)] = e

    # --- amounts ---------------------------------------------------------------------------

    def attempt_amount(self, key, following=None):
        a = self.attempts[key]
        r, c = a["reserve"], a["complete"]
        if c is None or c.get("bound_cents") is None:
            return _cents(r["reserve_cents"], "reserve_cents")
        amount = _cents(c["bound_cents"], "bound_cents")
        rate = _num(r["rate"], "rate")
        if r.get("follows") and c.get("t_wake") is not None:
            # this attempt counts from its reservation, inside the tail of the previous one
            amount += max(Decimal(0), _num(c["t_wake"], "t_wake") - _num(r["t_reserve"], "t_reserve")) * rate * 100
        if following is not None and c.get("clean") and c.get("t_last") is not None:
            nxt = self.attempts[following]["reserve"]
            end = min(_num(c["t_last"], "t_last") + _num(r["tail_idle_s"], "tail_idle_s"),
                      _num(nxt["t_reserve"], "t_reserve"))
            amount = min(amount, max(Decimal(0), end - _num(c["t_wake"], "t_wake")) * rate * 100)
        return _up(amount)

    def window_attempts(self, arm_id):
        """{attempt key: amount} of a window, with the cuts of `follows`."""
        keys = self.windows[arm_id]["attempts"]
        followed = {}
        for key in keys:
            f = self.attempts[key]["reserve"].get("follows")
            if isinstance(f, dict):
                prev = (f.get("run_id"), f.get("attempt_id"))
                if prev in self.attempts and prev in keys:
                    followed[prev] = key
        return {key: self.attempt_amount(key, followed.get(key)) for key in keys}

    def window_status(self, arm_id, is_alive):
        w = self.windows[arm_id]
        d = w["disarm"]
        if d is None:
            return "open" if is_alive(arm_id) else "orphan"
        if d.get("orphan"):
            return "orphan"
        attempts = [self.attempts[k] for k in w["attempts"]]
        if d.get("clean") is True and attempts and all(a["complete"] is not None and a["complete"].get("clean") is True
                                                        for a in attempts):
            return "clean"
        return "unclean"

    def window_base(self, arm_id, status):
        """The amount of a window before reconciliations and the orphan rule."""
        w = self.windows[arm_id]
        total = sum(self.window_attempts(arm_id).values(), Decimal(0))
        a_cents = _cents(w["arm"]["a_cents"], "a_cents")
        if status == "open":
            return max(a_cents, total)
        if status == "orphan":
            return max(a_cents, total)
        if status == "clean":
            return total
        return max(total, unclean_cents(w["arm"], w["disarm"]))

    def summary(self, is_alive):
        """{"total_cents", "windows": [...], "activities": [...], "orphans": [...], "over_bound": [...]}"""
        status = {arm_id: self.window_status(arm_id, is_alive) for arm_id in self.windows}
        base = {arm_id: self.window_base(arm_id, status[arm_id]) for arm_id in self.windows}
        act_base = {aid: _cents(a["reserve_cents"], "reserve_cents") for aid, a in self.activities.items()}
        covered_w, covered_a, over, groups = {}, {}, [], []
        for key_text, row in self.reconciles.items():
            key = json.loads(key_text)
            arms, acts = key.get("arms") or [], key.get("activities") or []
            if not row.get("manual") and any(status[a] != "clean" for a in arms):
                continue                               # automatic: clean windows only
            figure = _cents(row["cents"], "cents")
            registry = sum((base[a] for a in arms), Decimal(0)) + sum((act_base[a] for a in acts), Decimal(0))
            if row.get("orphan"):
                registry = max(registry, self._cap_cents())
            if figure > registry:
                over.append(key)
            for a in arms:
                covered_w[a] = key_text
            for a in acts:
                covered_a[a] = key_text
            groups.append((key_text, figure))
        total = sum((figure for _, figure in groups), Decimal(0))
        windows = []
        for arm_id, w in self.windows.items():
            reconciled = arm_id in covered_w
            if not reconciled:
                total += base[arm_id]
            windows.append({"arm_id": arm_id, "endpoint": w["arm"].get("endpoint"), "status": status[arm_id],
                            "amount_cents": str(base[arm_id]), "reconciled": reconciled,
                            "attempts": len(w["attempts"])})
        for aid in self.activities:
            if aid not in covered_a:
                total += act_base[aid]
        orphans = [w["arm_id"] for w in windows if w["status"] == "orphan" and not w["reconciled"]]
        if orphans:                                    # the provider is at its cap until reconciled
            total = max(total, self._cap_cents())
        return {"provider": self.provider, "total_cents": str(_up(total)), "windows": windows,
                "activities": [{"activity_id": a, "amount_cents": str(act_base[a]), "reconciled": a in covered_a}
                               for a in self.activities],
                "orphans": orphans, "over_bound": over}

    def _cap_cents(self):
        caps = [_cents(w["arm"]["spend_cap_cents"], "spend_cap_cents") for w in self.windows.values()]
        return max(caps) if caps else Decimal(0)


def unclean_cents(arm, dis):
    """The bound of an unclean window in cents: `(t_end - t_arm + double_s) x rate`, `t_end` the later of the
    `t_zero` and the `t_stop` of the `disarm` row `dis`, plus the `tail_s` of the `arm` row (MFL033 c7.19); the
    calculation of `window_base`, moved here by MFL023 B2 so that `closing` uses the same one."""
    end = _num(dis["t_zero"], "t_zero")
    if dis.get("t_stop") is not None:
        end = max(end, _num(dis["t_stop"], "t_stop"))
    end += _seconds(arm, "tail_s")
    span = max(Decimal(0), end - _num(arm["t_arm"], "t_arm")) + _seconds(dis, "double_s")
    return _up(span * _num(arm["rate"], "rate") * 100)


def closing(arm, t_zero, t_stop=None, double_s=0, drift_s=0):
    """MFL023 B2 (SOL2-B1 of the plan r2): the rule of a live session that closes its own window, shared by the
    worker and `session.py run`. None when the close is certifiable: the session observed the whole window (its
    wall clock moved with its monotonic clock: the largest drift observed, `drift_s`, within
    `engine.servers.DRIFT_LIMIT_S`, so no suspend and no step of the clock) and the unclean bound up to the zero
    observed (`unclean_cents`, with `t_stop` and `double_s`) stays within `A`. Otherwise the reason:

    - `unobserved_interval`: the drift beyond the limit; in that interval the session measured neither the
      overlap of two workers nor the bounds of its attempts, so the bound of the window is not proven, even
      when the interval ended before `T`;
    - `over_window`: the bound over `A` (for a window not shortened, a zero observed after `T + 60 s + Z`).

    A close that is not certifiable writes its `disarm` row with `orphan: true` and the reason as `uncertain`:
    the provider stays at its cap until a manual reconciliation with the invoice (`mdf bench spend reconcile
    --orphan --arm ID --manual-cents N`); the window is never cut at `A`."""
    if abs(Decimal(str(drift_s))) > DRIFT_LIMIT_S:
        return "unobserved_interval"
    bound = unclean_cents(arm, {"t_zero": t_zero, "t_stop": t_stop, "double_s": double_s})
    if bound > _cents(arm["a_cents"], "a_cents"):
        return "over_window"
    return None


def parse(entries):
    """{provider: Provider} of the server rows of `entries`."""
    out = {}
    for e in server_rows(entries):
        provider = e.get("provider")
        if provider not in PROVIDERS:
            raise _bad("provider")
        out.setdefault(provider, Provider(provider)).add(e)
    return out


def summary(entries, provider, ledger_path):
    p = parse(entries).get(provider) or Provider(provider)
    return p.summary(lambda arm_id: alive(ledger_path, arm_id))


def _stop_if_over(s):
    if s["over_bound"]:
        raise BenchError(Code.SERVER_BILLED_OVER_BOUND, "a reconciliation of %s over the ledger" % s["provider"])


# --- the session seen from the engine's callers ---------------------------------------------------

def check_rate(arm_row, rate):
    """MFL033 S-B6: an attempt reserves and bounds its cost at the rate of its endpoint; it must be the
    rate that the arm recorded (for Modal the dearest GPU of the app), else `server_not_configured`
    before any request: a stale `rate_usd_per_s` in servers.json would release spend still due."""
    armed = _num(arm_row.get("rate"), "rate")
    try:
        mine = Decimal(str(rate))
    except (InvalidOperation, ValueError):
        mine = None
    if mine is None or not mine.is_finite() or mine != armed:
        raise BenchError(Code.SERVER_NOT_CONFIGURED, "rate_usd_per_s of the endpoint (%s USD/s) is not the rate of its "
                         "session %s (%s USD/s): set it in servers.json" % (rate, arm_row.get("arm_id"), armed))


def session(ledger_path, provider, url, environ=None, entries=None, rate=None):
    """The live session of `provider` for the canonical `url`: {"arm_id", "armed_until",
    "window_left_cents"}. A readable ledger with an orphan window of the provider is
    `session_orphaned`, which comes before `server_not_armed` (no `MDF_SESSION_ARM`, another
    endpoint, a disarmed window, a free lock); with `rate`, the rate of the endpoint must be the rate
    of the arm (`check_rate`)."""
    environ = os.environ if environ is None else environ
    if entries is None:
        entries = spend.read_only(ledger_path)
    p = parse(entries).get(provider) or Provider(provider)
    s = p.summary(lambda a: alive(ledger_path, a))
    if s["orphans"]:
        raise BenchError(Code.SESSION_ORPHANED, "orphan windows of %s: %s; disarm and reconcile them"
                         % (provider, ", ".join(s["orphans"])))
    _stop_if_over(s)
    arm_id = environ.get(SESSION_ENV)
    w = p.windows.get(arm_id) if isinstance(arm_id, str) else None
    if w is None:
        raise BenchError(Code.SERVER_NOT_ARMED, "no session (scripts/servers/session.py run) for this endpoint")
    if w["arm"].get("url") != url or w["disarm"] is not None or not alive(ledger_path, arm_id):
        raise BenchError(Code.SERVER_NOT_ARMED, "the session %s does not arm this endpoint" % arm_id)
    if rate is not None:
        check_rate(w["arm"], rate)
    left = _cents(w["arm"]["a_cents"], "a_cents") - sum(p.window_attempts(arm_id).values(), Decimal(0))
    return {"arm_id": arm_id, "armed_until": float(_num(w["arm"]["until"], "until")),
            "window_left_cents": str(max(Decimal(0), left).quantize(CENT, rounding="ROUND_FLOOR"))}


def reserve(locked, ledger_path, provider, arm_id, run_id, attempt_id, info, command_left_cents, clock=time.time):
    """Under the lock of the ledger, at the wake of a server: the checks of the session again,
    the budget, `follows`, then the `reserve` row. `info`: {"reserve_cents", "rate", "tail_idle_s",
    "identity", "endpoint", "url", "D"}, `url` the canonical URL of the endpoint, which must be the URL of
    the window (MFL023 B2, C-B2 of the plan r1): an attempt never draws from the window of another
    endpoint. Raises `session_orphaned`, `server_not_armed`, `server_cap`, `server_billed_over_bound`;
    returns the row."""
    entries = locked.entries()
    p = parse(entries).get(provider) or Provider(provider)
    s = p.summary(lambda a: alive(ledger_path, a))
    if s["orphans"]:
        raise BenchError(Code.SESSION_ORPHANED, "orphan windows of " + provider)
    _stop_if_over(s)
    w = p.windows.get(arm_id)
    if w is None or w["disarm"] is not None or not alive(ledger_path, arm_id):
        raise BenchError(Code.SERVER_NOT_ARMED, "the session %s is not live" % arm_id)
    if (run_id, attempt_id) in p.attempts:
        raise BenchError(Code.SPEND_LEDGER_INCONSISTENT, "two reservations of one attempt")
    if info.get("url") != w["arm"].get("url"):            # MFL023 B2: the window of this endpoint only
        raise BenchError(Code.SERVER_NOT_ARMED, "the session %s does not arm the endpoint of this attempt" % arm_id)
    check_rate(w["arm"], info.get("rate"))                # S-B6, again under the lock
    left = _cents(w["arm"]["a_cents"], "a_cents") - sum(p.window_attempts(arm_id).values(), Decimal(0))
    budget = min(left, Decimal(str(command_left_cents)))
    cents = _cents(info["reserve_cents"], "reserve_cents")
    if cents > budget:
        raise BenchError(Code.SERVER_CAP, "reservation %s cents over %s left" % (cents, budget))
    t_reserve = round(clock(), 3)
    follows = None
    if w["attempts"]:
        prev_key = w["attempts"][-1]
        prev = p.attempts[prev_key]
        c = prev["complete"]
        if (c is not None and c.get("clean") is True and c.get("t_last") is not None
                and prev["reserve"].get("identity") == info.get("identity")
                and Decimal(str(t_reserve)) < _num(c["t_last"], "t_last") + _num(prev["reserve"]["tail_idle_s"], "tail")):
            follows = {"run_id": prev_key[0], "attempt_id": prev_key[1]}
    row = {"kind": "reserve", "provider": provider, "run_id": run_id, "attempt_id": attempt_id, "arm_id": arm_id,
           "endpoint": info.get("endpoint"), "identity": info.get("identity"), "reserve_cents": str(cents),
           "rate": str(info["rate"]), "tail_idle_s": info["tail_idle_s"], "d_seconds": info.get("D"),
           "t_reserve": t_reserve, "follows": follows, "at": spend.now()}
    locked.append(row)
    return row


def complete(locked, provider, run_id, attempt_id, result):
    """The `complete` row of an attempt from its `engine.json` (never twice)."""
    p = parse(locked.entries()).get(provider) or Provider(provider)
    a = p.attempts.get((run_id, attempt_id))
    if a is None or a["complete"] is not None:
        return None
    result = result if isinstance(result, dict) else {}
    cost = result.get("cost") if isinstance(result.get("cost"), dict) else {}
    ms = result.get("marker_server") if isinstance(result.get("marker_server"), dict) else {}
    error = result.get("error") if isinstance(result.get("error"), dict) else {}
    row = {"kind": "complete", "provider": provider, "run_id": run_id, "attempt_id": attempt_id,
           "bound_cents": cost.get("bound_cents"), "billed_cents": None, "clean": cost.get("clean") is True,
           "t_wake": ms.get("t_wake"), "t_last": ms.get("t_last"), "t_close": ms.get("t_close"),
           "status": result.get("status"), "code": error.get("code"), "at": spend.now()}
    locked.append(row)
    return row


# --- the rows of the session and of the reconciliations (session.py, the bench CLI) ---------------

def new_arm_id():
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-" + secrets.token_hex(4)


def arm(locked, ledger_path, provider, endpoint, url, a_cents, until, spend_cap_cents, rate, t_arm, fault=None,
        arm_id=None, extra=None):
    """The `arm` row of a session, after the checks: no orphan window, the total of the provider
    plus `A` within its cap (`server_cap`). Returns the row."""
    p = parse(locked.entries()).get(provider) or Provider(provider)
    s = p.summary(lambda a: alive(ledger_path, a))
    if s["orphans"]:
        raise BenchError(Code.SESSION_ORPHANED, "orphan windows of " + provider)
    _stop_if_over(s)
    a_cents, cap = _cents(a_cents, "a_cents"), _cents(spend_cap_cents, "spend_cap_cents")
    if Decimal(s["total_cents"]) + a_cents > cap:
        raise BenchError(Code.SERVER_CAP, "%s: %s cents spent, %s asked, cap %s" % (provider, s["total_cents"], a_cents, cap))
    row = {"kind": "arm", "provider": provider, "arm_id": arm_id or new_arm_id(), "endpoint": endpoint, "url": url,
           "a_cents": str(a_cents), "until": until, "spend_cap_cents": str(cap), "rate": str(rate), "t_arm": t_arm,
           "fault": fault, "at": spend.now(), **(extra or {})}
    locked.append(row)
    return row


def disarm(locked, provider, arm_id, t_zero, clean, orphan=False, extra=None):
    p = parse(locked.entries()).get(provider) or Provider(provider)
    w = p.windows.get(arm_id)
    if w is None or w["disarm"] is not None:
        raise BenchError(Code.SPEND_LEDGER_INCONSISTENT, "disarm of an unknown or disarmed window")
    row = {"kind": "disarm", "provider": provider, "arm_id": arm_id, "t_zero": t_zero, "clean": bool(clean) and not orphan,
           "orphan": bool(orphan), "at": spend.now(), **(extra or {})}
    locked.append(row)
    return row


def activity(locked, provider, name, reserve_cents, spend_cap_cents, rate=None, activity_id=None):
    p = parse(locked.entries()).get(provider) or Provider(provider)
    s = p.summary(lambda a: False)
    cents, cap = _cents(reserve_cents, "reserve_cents"), _cents(spend_cap_cents, "spend_cap_cents")
    if Decimal(s["total_cents"]) + cents > cap:
        raise BenchError(Code.SERVER_CAP, "%s: activity over the cap" % provider)
    row = {"kind": "activity", "provider": provider, "activity_id": activity_id or new_arm_id(), "name": name,
           "reserve_cents": str(cents), "rate": None if rate is None else str(rate), "at": spend.now()}
    locked.append(row)
    return row


def reconcile(locked, ledger_path, provider, cents, manual, arms=None, activities=None, orphan=False, hours=None):
    """A `reconcile` row (idempotent: the same key with the same figure adds nothing). With
    `orphan`, every window must have a `disarm` with `orphan: true` (prescription 2); an automatic
    reconciliation covers clean windows only. Returns (row or None, summary after it); a figure
    over the ledger is written, then `server_billed_over_bound`."""
    entries = locked.entries()
    p = parse(entries).get(provider) or Provider(provider)
    is_alive = (lambda a: alive(ledger_path, a))
    arms, activities = list(arms or []), list(activities or [])
    if not arms and not activities:
        raise BenchError(Code.INVALID_SETTING, "a reconciliation covers windows or activities")
    for arm_id in arms:
        w = p.windows.get(arm_id)
        if w is None:
            raise BenchError(Code.INVALID_SETTING, "no window " + arm_id)
        if orphan and not (w["disarm"] is not None and w["disarm"].get("orphan") is True):
            raise BenchError(Code.INVALID_SETTING, "window %s: no disarm with orphan: true (session.py disarm)" % arm_id)
        if not orphan and p.window_status(arm_id, is_alive) in ("open", "orphan"):
            raise BenchError(Code.INVALID_SETTING, "window %s is open or orphan" % arm_id)
        if not manual and p.window_status(arm_id, is_alive) != "clean":
            raise BenchError(Code.INVALID_SETTING, "an automatic reconciliation covers clean windows only")
    for aid in activities:
        if aid not in p.activities:
            raise BenchError(Code.INVALID_SETTING, "no activity " + aid)
    key = {"provider": provider, "arms": arms, "activities": activities, "hours": hours}
    cents = _cents(cents, "cents")
    last = p.reconciles.get(json.dumps(key, sort_keys=True))
    row = None
    if last is None or _cents(last["cents"], "cents") != cents or bool(last.get("manual")) != bool(manual):
        row = {"kind": "reconcile", "provider": provider, "key": key, "cents": str(cents), "manual": bool(manual),
               "orphan": bool(orphan), "at": spend.now()}
        locked.append(row)
        entries = entries + [row]
    s = summary(entries, provider, ledger_path)
    _stop_if_over(s)
    return row, s


def run_total(entries, provider, run_id):
    """The amounts of the attempts of the run `run_id` on `provider` (each its bound, or its
    reservation without a completion; without the cuts of `follows`, so never less)."""
    p = parse(entries).get(provider) or Provider(provider)
    return sum((p.attempt_amount(key) for key in p.attempts if key[0] == run_id), Decimal(0))


def orphan_windows(entries, provider, ledger_path):
    """The arm ids of the orphan windows of `provider` that wait for `--orphan` reconciliation:
    disarmed with `orphan: true` and not reconciled."""
    p = parse(entries).get(provider) or Provider(provider)
    s = p.summary(lambda a: alive(ledger_path, a))
    return [w["arm_id"] for w in s["windows"] if w["status"] == "orphan" and not w["reconciled"]
            and p.windows[w["arm_id"]]["disarm"] is not None]


class Reservation:
    """The observer of an attempt on a rented endpoint, shared by `mdf convert` and `mdf bench run`:
    at the event `server-wake` of the engine (before the first request) the reservation goes into
    the ledger under its lock, or the attempt stops with a closed code of the engine (`server_cap`,
    `session_orphaned`, `server_not_armed`; a ledger that cannot be read or written stops as
    `server_cap`, its own code kept in `failure`). `complete` writes the completion from
    `engine.json`."""

    def __init__(self, ledger, provider, arm_id, run_id, attempt_id, command_left_cents, clock=time.time):
        self.ledger, self.provider, self.arm_id = ledger, provider, arm_id
        self.run_id, self.attempt_id, self.command_left = run_id, attempt_id, command_left_cents
        self.clock = clock
        self.row = self.failure = None

    def __call__(self, event):
        if event.get("event") == "step" and event.get("step") == "server-wake":
            from ..engine.codes import Code as EngineCode
            info = event.get("reserve") or {}
            try:
                with self.ledger.locked() as locked:
                    self.row = reserve(locked, self.ledger.path, self.provider, self.arm_id, self.run_id,
                                       self.attempt_id, info, self.command_left, clock=self.clock)
            except BenchError as exc:
                self.failure = exc.code
                engine = exc.code.value in {c.value for c in EngineCode}
                raise BenchError(exc.code if engine else Code.SERVER_CAP, exc.detail) from None
            except OSError:
                self.failure = Code.SPEND_LEDGER_UNWRITABLE
                raise BenchError(Code.SERVER_CAP, "spend ledger not writable") from None
        return None

    def complete(self, result):
        if self.row is None:
            return None
        with self.ledger.locked() as locked:
            return complete(locked, self.provider, self.run_id, self.attempt_id, result)
