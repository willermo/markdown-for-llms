"""One call of the product to a rented provider, in a child process with its own process group (MFL023 B1;
plan r2 §4.3, r3 §4.3).

The worker never calls a provider in its own process. Every reading, arm, disarm, zero and restore of
FlashBoot of a GPU session of the product, and the readings of the panels (MFL051 B: `read`, the state of the
provider as a `Reading` of `adapters`, and `balance`, with the key of the worker alone), runs in

    python -I -B -m markdown_for_llms.gpu.opchild

started by `call` with `start_new_session=True` (its own process group, which also holds `uv` and the CLI of
Modal), its request as one JSON object on stdin, its answer as one JSON object on stdout, its stderr on
/dev/null. Each call has the limits of its caller:

- `until`, an epoch of the wall clock: the calls of an arm, whose limit is the term T of the session;
- `budget_s`, seconds of the monotonic clock: the readings before an arm (60 s) and the calls of the cleanup
  (a suspend of the laptop does not spend them);
- `stop`, a callable polled with them (the stop of the worker, a request to close the session).

At a limit the caller kills the whole group (SIGKILL), whatever the child is doing (an answer still arriving
drop by drop, a `modal deploy` with its own children), and the answer is a closed failure: `TermReached`,
`BudgetSpent` or `Stopped`; a child that ends without a valid answer is `ChildFailed`. The child sets
`PR_SET_PDEATHSIG` (SIGKILL) and exits at once when its parent is not the pid of the request. The signal
follows the thread that started the child, so the caller waits for its child in that thread: `call` is
synchronous. A child already started by a child (`uv`, the CLI of Modal) may outlive a SIGKILL of the worker
itself (risk R8 of the plan r2): the recovery of the next start disarms what it armed.

The answer holds closed fields only: `{"ok": true, "result": {...}}` with the facts read from the JSON of the
provider (identifiers, states, counters, times, canonical URLs), or `{"ok": false, "error":
providers.closed(exc)}`; never the text of the provider, never str() of an error. The parent closes it again
(`errors.bounded`).

Request: {"op", "provider", "resource", "env_file", "parent", "args"}; the keys are read by name from
`env_file` in the child (`providers.make`), never passed in its environment nor in its request.
"""
import ctypes
import json
import os
import signal
import subprocess
import sys
import time

from . import errors, providers

STEP_S = 0.25                       # the polling of the limits by the caller
GRACE_S = 5.0                       # the wait for the group after its SIGKILL
REQUEST_LIMIT = 1 << 20
ANSWER_LIMIT = 4 << 20
PR_SET_PDEATHSIG = 1
READ_BUDGET_S = 60                  # the readings before an arm (plan r3 §4.2, step 5)
OPS = ("state", "rest", "arm", "disarm", "zero", "restore_flashboot", "waiting", "read", "balance")


# --- the child -------------------------------------------------------------------------------------

def _canonical(url):
    from ..engine.servers import canonical_url
    return canonical_url(url)


def op_state(prov, args, until):
    return {"state": prov.state()}


def op_rest(prov, args, until):
    """The reading before an arm (plan r3 §4.2, step 5): the state; on RunPod also the invariant pins with the
    GPU preset of `args` (`providers.pin_checks`, MFL023 A3) and the canonical URL of the endpoint."""
    out = {"state": prov.state()}
    if prov.provider == "runpod":
        ep = prov.endpoint()
        checks = providers.pin_checks(ep, *prov.pin_facts(), preset=args.get("preset"))
        out.update(pins_failed=providers.invariant_failures(checks), url=_canonical(prov.url),
                   endpoint_id=ep.get("id"))                   # the follower of the worker logs reads by it
    return out


def op_arm(prov, args, until):
    """The arm, every call of it within `until` (the term T): RunPod with the pins of the preset checked again
    on its own reading; Modal the deploy, whose URL comes back canonical."""
    fault, levers = args.get("fault") or "", args.get("levers") or ""
    if prov.provider == "runpod":
        info = prov.arm(fault, False, levers, deadline=until, preset=args.get("preset"))
        return {"info": info, "url": _canonical(prov.url)}
    info = prov.arm(fault, False, levers, deadline=until)
    return {"info": {"volume": info.get("volume"), "proxy_token": info.get("proxy_token")},
            "url": _canonical(info["url"])}


def op_disarm(prov, args, until):
    prov.disarm()
    return {}


def op_zero(prov, args, until):
    zero, state = prov.zero()
    return {"zero": bool(zero), "state": state}


def op_restore_flashboot(prov, args, until):
    prov.restore_flashboot()
    return {}


def op_waiting(prov, args, until):
    """Modal (MFL033 c7.14, `worker_logs.Scheduler`): the time of the first "waiting to be scheduled" system line of
    the app since `args["since"]` (epoch, or the moment it was read when its time cannot be read) and the number of
    the containers of the app. Closed facts only: never the text of a line."""
    from . import worker_logs
    first = None
    for raw in prov.system_logs(args["since"]):
        m = worker_logs.MODAL_FETCHED.match(raw)
        ts, text = (m.group(1), m.group(4)) if m else (None, raw)
        if worker_logs.WAITING.search(text or ""):
            at = worker_logs.epoch(ts)
            first = round(at if at is not None else time.time(), 3)
            break
    return {"waiting_at": first, "containers": len(prov.app_containers())}


def op_read(prov, args, until):
    """MFL051 B: the state of the provider for a panel, a `Reading` (`adapters`): the management API only, never the
    URL of the service; `running` for the panel, never for the zero."""
    from . import adapters
    return {"reading": adapters.of_provider(prov).read()}


def op_balance(prov, args, until):
    """MFL051 B: the balance with the key of the worker (`adapters`: RunPod `GET /v2/billing/balance`, None when the key
    does not read it; Modal None, without a call)."""
    from . import adapters
    return {"balance": adapters.of_provider(prov).balance()}


def make(request):
    """The provider of the request, with its keys read by name from its env file (`providers.make`)."""
    return providers.make(request["provider"], request["resource"], request["env_file"])


def watch_parent(parent):
    """SIGKILL when the parent dies; False when the parent is already another process."""
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        if libc.prctl(PR_SET_PDEATHSIG, signal.SIGKILL, 0, 0, 0) != 0:
            return False
    except (OSError, AttributeError):
        return False
    return os.getppid() == parent


def _plain(value):
    from decimal import Decimal
    if isinstance(value, Decimal):
        return str(value)
    raise TypeError(type(value).__name__)


def answer(stdout, obj):
    try:
        text = json.dumps(obj, sort_keys=True, default=_plain, allow_nan=False)
    except (TypeError, ValueError) as exc:
        text = json.dumps({"ok": False, "error": {"class": "ResultInvalid", "op": None, "http": None,
                                                  "detail": "the result is not closed JSON: %s" % type(exc).__name__}})
    stdout.write(text + "\n")
    stdout.flush()


def main(stdin=None, stdout=None, factory=None, watch=None):
    """The child: one request, one answer. `factory(request)` replaces `make` and `watch(parent)` replaces
    `watch_parent` in the tests."""
    stdin = sys.stdin.buffer if stdin is None else stdin
    stdout = sys.stdout if stdout is None else stdout
    try:
        raw = stdin.read(REQUEST_LIMIT + 1)
        request = json.loads(raw) if len(raw) <= REQUEST_LIMIT else None
    except ValueError:
        request = None
    if (not isinstance(request, dict) or request.get("op") not in OPS or type(request.get("parent")) is not int
            or request.get("provider") not in ("runpod", "modal") or not isinstance(request.get("resource"), str)
            or not isinstance(request.get("args") or {}, dict)):
        answer(stdout, {"ok": False, "error": {"class": "RequestInvalid", "op": None, "http": None,
                                               "detail": "a request of opchild"}})
        return 2
    if not (watch or watch_parent)(request["parent"]):
        return 125                                   # the parent is gone: nothing is called
    until = request.get("until")
    try:
        prov = (factory or make)(request)
        result = globals()["op_" + request["op"]](prov, request.get("args") or {}, until)
        answer(stdout, {"ok": True, "result": result})
        return 0
    except Exception as exc:                          # every error leaves as its closed fields
        answer(stdout, {"ok": False, "error": providers.closed(exc)})
        return 1


# --- the caller ------------------------------------------------------------------------------------

def command():
    return [sys.executable, "-I", "-B", "-m", "markdown_for_llms.gpu.opchild"]


def _failure(cls, op, detail):
    return {"class": cls, "op": "opchild." + op, "http": None, "detail": detail}


def kill_group(proc):
    """SIGKILL to the process group of `proc` (its pid: `start_new_session`)."""
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


def call(op, provider, resource, env_file, args=None, *, until=None, budget_s=None, stop=None, argv=None,
         environ=None, wall=time.time, mono=time.monotonic, step=STEP_S, on_start=None, also=()):
    """One call `op` in a child (module docstring). Returns {"ok", "result", "error", "child", "seconds"}:
    `result` the closed result of the child, `error` closed fields (`errors.bounded`), `child` {"pid", "exit",
    "killed"} (None when no child started: a limit already reached). `on_start(pid)` is called once the child
    runs (the tests, the diary). `also`: names that leave the environment of the child besides those of the keys
    (`providers.child_env`, the `key_env` of the endpoint)."""
    started = mono()
    request = {"op": op, "provider": provider, "resource": resource, "env_file": env_file, "args": args or {},
               "parent": os.getpid(), "until": until}

    def reached():
        if until is not None and wall() >= until:
            return "TermReached", "the term of the session came during %s" % op
        if budget_s is not None and mono() - started >= budget_s:
            return "BudgetSpent", "%s took its %s s" % (op, budget_s)
        if stop is not None and stop():
            return "Stopped", "%s stopped by its caller" % op
        return None

    def done(ok, result=None, error=None, child=None):
        return {"ok": ok, "result": result, "error": None if error is None else errors.bounded(error),
                "child": child, "seconds": round(mono() - started, 3)}
    first = reached()
    if first is not None:
        return done(False, error=_failure(first[0], op, first[1]))
    env = providers.child_env(os.environ if environ is None else environ, also)
    proc = subprocess.Popen(argv or command(), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.DEVNULL, start_new_session=True, close_fds=True, env=env)
    if on_start is not None:
        on_start(proc.pid)
    data, out, limit = json.dumps(request).encode("utf-8"), None, None
    try:
        while True:
            try:
                out, _ = proc.communicate(data, timeout=step)
                break
            except subprocess.TimeoutExpired:
                data = None
            limit = reached()
            if limit is not None:
                kill_group(proc)
                break
    finally:
        if out is None:                               # killed, or an error of the caller: the group goes
            kill_group(proc)
            try:
                proc.communicate(timeout=GRACE_S)
            except (subprocess.TimeoutExpired, ValueError, OSError):
                for pipe in (proc.stdin, proc.stdout):
                    try:
                        pipe.close()
                    except (OSError, AttributeError):
                        pass
                try:
                    proc.wait(GRACE_S)
                except subprocess.TimeoutExpired:
                    pass
    child = {"pid": proc.pid, "exit": proc.returncode, "killed": limit is not None}
    if limit is not None:
        return done(False, error=_failure(limit[0], op, limit[1]), child=child)
    try:
        doc = json.loads(out[:ANSWER_LIMIT]) if out and len(out) <= ANSWER_LIMIT else None
    except ValueError:
        doc = None
    if not isinstance(doc, dict) or type(doc.get("ok")) is not bool:
        return done(False, error=_failure("ChildFailed", op, "no answer, exit %s" % proc.returncode), child=child)
    if doc["ok"]:
        result = doc.get("result")
        if not isinstance(result, dict):
            return done(False, error=_failure("ChildFailed", op, "a result that is not an object"), child=child)
        return done(True, result=result, child=child)
    return done(False, error=doc.get("error"), child=child)


if __name__ == "__main__":
    sys.exit(main())
